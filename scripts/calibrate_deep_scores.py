"""Post-hoc development calibration for deep anomaly scores.

This script is intentionally diagnostic: it uses prediction labels to inspect whether
model scores have the right direction and whether a label-aware weighted fusion can
increase AUC. Do not present its output as a frozen-test result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import scripts.run_deep_ensemble as deep
import scripts.run_gpu_tune as gpu
from faultdiagnose.data import load_events
from faultdiagnose.evaluation.anomaly import compute_auc
from faultdiagnose.models import LSTMAE, VAE, TransformerAE

RESULTS = Path(__file__).resolve().parents[1] / "results"
CHECKPOINTS = RESULTS / "checkpoints"
MODEL_NAMES = ("vae", "lstm", "transformer")


def load_result(stamp: str) -> dict:
    """Load a deep ensemble result by timestamp stamp."""
    path = RESULTS / f"{stamp}_deep_ensemble_result.json"
    if not path.exists():
        raise SystemExit(f"result not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_models(stamp: str, result: dict, device: str) -> dict[str, VAE | LSTMAE | TransformerAE]:
    """Load trained VAE/LSTM/Transformer checkpoints from one deep run."""
    norm = np.load(CHECKPOINTS / f"{stamp}_deep_ensemble_norm.npz")
    in_dim = int(len(norm["mean"]))
    hidden = int(result["hidden"])
    latent = int(result["latent"])
    seq_len = int(result["seq_len"])
    beta = float(result["vae_beta"])
    nhead = 4 if hidden <= 128 else 8
    models: dict[str, VAE | LSTMAE | TransformerAE] = {
        "vae": VAE(in_dim, latent=latent, hidden=hidden, beta=beta),
        "lstm": LSTMAE(in_dim, seq_len, latent=latent, hidden=hidden, num_layers=2),
        "transformer": TransformerAE(
            in_dim, seq_len, latent=latent, d_model=hidden, nhead=nhead, num_layers=2
        ),
    }
    for name, model in models.items():
        state = torch.load(
            CHECKPOINTS / f"{stamp}_deep_ensemble_{name}.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.to(device)
        model.eval()
    return models


def robust_standardize(scores: np.ndarray, median: float, iqr: float) -> np.ndarray:
    """Apply stored validation median/IQR normalization."""
    scale = iqr if iqr > 1e-12 else 1.0
    return (scores - median) / scale


def labels_for_records(records) -> tuple[np.ndarray, np.ndarray]:
    """Return score/label arrays using the frozen event-window labels."""
    events = load_events(gpu.OUT)
    ev_map = {(row.farm, str(row.event_id)): row for row in events.itertuples()}
    return gpu._records_to_auc_arrays(records, ev_map)


def separation(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Summarize separation after finite filtering."""
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    normal = scores[labels == 0]
    anomaly = scores[labels == 1]
    return {
        "normal_median": float(np.median(normal)),
        "anomaly_median": float(np.median(anomaly)),
        "median_gap": float(np.median(anomaly) - np.median(normal)),
    }


def normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    """Normalize non-negative weights with equal fallback."""
    total = sum(raw.values())
    if total <= 0:
        return {name: 1.0 / len(raw) for name in raw}
    return {name: value / total for name, value in raw.items()}


def fisher_separation(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return standardized mean gap between anomaly and normal scores."""
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    normal = scores[labels == 0]
    anomaly = scores[labels == 1]
    if len(normal) == 0 or len(anomaly) == 0:
        return 0.0
    pooled = np.sqrt(0.5 * (np.var(normal) + np.var(anomaly)))
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(anomaly) - np.mean(normal)) / pooled)


def simplex_grid(step: float = 0.02):
    """Yield non-negative 3-model weights that sum to 1."""
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            yield {
                "vae": i / n,
                "lstm": j / n,
                "transformer": k / n,
            }


def weighted_sum(standardized: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Combine standardized score arrays with supplied weights."""
    return sum(weights[name] * standardized[name] for name in MODEL_NAMES)


def optimize_weights(
    standardized: dict[str, np.ndarray],
    labels: np.ndarray,
    objective: str,
) -> dict:
    """Grid-search ensemble weights for AUC or score-separation diagnostics."""
    best_weights: dict[str, float] | None = None
    best_value = -np.inf
    best_auc = -np.inf
    best_sep = -np.inf
    for weights in simplex_grid():
        scores = weighted_sum(standardized, weights)
        auc = compute_auc(scores, labels)
        sep = fisher_separation(scores, labels)
        value = auc if objective == "auc" else sep
        if value > best_value:
            best_value = float(value)
            best_weights = weights
            best_auc = float(auc)
            best_sep = float(sep)
    if best_weights is None:
        raise RuntimeError("weight optimization failed")
    return {
        "objective": objective,
        "weights": best_weights,
        "auc": best_auc,
        "fisher_separation": best_sep,
        "distribution": separation(weighted_sum(standardized, best_weights), labels),
    }


def main() -> None:
    """Run label-aware score-direction diagnostics for a saved deep run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", required=True, help="timestamp prefix, e.g. 20260717_2206")
    parser.add_argument("--device", default=gpu.DEVICE)
    args = parser.parse_args()

    gpu.DEVICE = args.device
    result = load_result(args.stamp)
    models = load_models(args.stamp, result, args.device)
    norm = np.load(CHECKPOINTS / f"{args.stamp}_deep_ensemble_norm.npz")
    mean, std = norm["mean"], norm["std"]
    farms = result["farms"]
    feature_columns = deep.feature_columns_by_farm(result.get("feature_set", "all"))
    seq_len = int(result["seq_len"])
    use_fft = bool(result["use_fft"])
    seq_batch = 128

    records = {
        "vae": gpu.evaluate_vae_records(
            models["vae"], farms, result["window"], mean, std, use_fft, feature_columns
        ),
        "lstm": gpu.evaluate_seq_records(
            models["lstm"],
            farms,
            result["window"],
            seq_len,
            mean,
            std,
            use_fft,
            batch=seq_batch,
            feature_columns_by_farm=feature_columns,
        ),
        "transformer": gpu.evaluate_seq_records(
            models["transformer"],
            farms,
            result["window"],
            seq_len,
            mean,
            std,
            use_fft,
            batch=seq_batch,
            feature_columns_by_farm=feature_columns,
        ),
    }

    base = deep.ensure_same_timeline(records)
    normalizations = result["ensemble_results"][0]["normalization"]
    standardized: dict[str, np.ndarray] = {}
    diagnostics = {}
    raw_weights = {}
    for name in MODEL_NAMES:
        scores, labels = labels_for_records(records[name])
        auc = compute_auc(scores, labels)
        direction = 1.0 if auc >= 0.5 else -1.0
        raw_weights[name] = abs(float(auc) - 0.5)
        params = normalizations[name]
        standardized[name] = direction * robust_standardize(
            records[name]["score"].to_numpy(float),
            float(params["median"]),
            float(params["iqr"]),
        )
        directed_scores = direction * scores
        diagnostics[name] = {
            "auc": float(auc),
            "directed_auc": float(compute_auc(directed_scores, labels)),
            "direction": direction,
            "separation": separation(directed_scores, labels),
        }

    weights = normalize_weights(raw_weights)
    equal_directed = sum(standardized[name] for name in MODEL_NAMES) / len(MODEL_NAMES)
    weighted_directed = sum(weights[name] * standardized[name] for name in MODEL_NAMES)
    base["score"] = weighted_directed
    optimization_scores, labels_for_optimization = labels_for_records(base)
    finite_for_optimization = np.isfinite(optimization_scores)
    filtered_labels = labels_for_optimization[finite_for_optimization]
    filtered_standardized = {
        name: values[np.isfinite(weighted_directed)][finite_for_optimization]
        for name, values in standardized.items()
    }
    optimized = {
        "auc": optimize_weights(filtered_standardized, filtered_labels, "auc"),
        "separation": optimize_weights(filtered_standardized, filtered_labels, "separation"),
    }
    outputs = {}
    for mode, score in {
        "equal_directed": equal_directed,
        "weighted_directed": weighted_directed,
    }.items():
        frame = base.copy()
        frame["score"] = score
        scores, labels = labels_for_records(frame)
        outputs[mode] = {
            "auc": float(compute_auc(scores, labels)),
            "fisher_separation": fisher_separation(scores, labels),
            "separation": separation(scores, labels),
        }

    out = {
        "source_result": str(RESULTS / f"{args.stamp}_deep_ensemble_result.json"),
        "diagnostic_only": True,
        "weights_from_abs_auc_excess": weights,
        "individual": diagnostics,
        "ensemble": outputs,
        "optimized_weights": optimized,
    }
    path = RESULTS / f"{args.stamp}_deep_score_calibration.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
