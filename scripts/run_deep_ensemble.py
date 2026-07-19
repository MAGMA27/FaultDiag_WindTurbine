"""Strict CARE deep ensemble runner for VAE + LSTM-AE + Transformer-AE.

The legacy ensemble script predates the frozen protocol. This runner reuses the
prediction-only/status-filtered helpers from run_gpu_tune.py, robust-standardizes
each model score using normal validation scores, then evaluates weighted ensembles
with CARE artifacts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import scripts.run_gpu_tune as gpu
from faultdiagnose.data import load_care, load_events
from faultdiagnose.evaluation.anomaly import compute_auc
from faultdiagnose.evaluation.care import evaluate_care, write_care_artifacts
from faultdiagnose.evaluation.ensemble import (
    adaptive_threshold,
    combine_standardized,
    flag,
    robust_location_scale,
    validation_stability_weights,
)
from faultdiagnose.models import LSTMAE, VAE, SeqWindowsDataset, TransformerAE

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS_DIR = RESULTS / "checkpoints"
MODEL_NAMES = ("vae", "lstm", "transformer")
FARM_B_PAPER_TOP_BASES = (
    "sensor_4",
    "sensor_8",
    "sensor_10",
    "sensor_12",
    "sensor_20",
    "sensor_21",
    "sensor_22",
    "sensor_25",
    "sensor_31",
    "sensor_32",
    "sensor_33",
    "sensor_34",
    "sensor_35",
    "sensor_36",
    "sensor_37",
    "sensor_38",
    "sensor_39",
    "sensor_47",
    "sensor_51",
    "sensor_52",
    "sensor_54",
    "sensor_55",
    "sensor_56",
    "power_58",
    "power_62",
    "wind_speed_59",
    "wind_speed_60",
    "wind_speed_61",
)


def feature_columns_by_farm(feature_set: str) -> dict[str, list[str]] | None:
    """Return configured deep-model input columns for a named feature set."""
    if feature_set == "all":
        return None
    if feature_set == "avg_only":
        return {"B": ["__avg_only__"]}
    if feature_set == "paper_top":
        return {"B": [f"{name}_avg" for name in FARM_B_PAPER_TOP_BASES]}
    raise ValueError(f"unknown feature set: {feature_set}")


def score_seq_matrix(
    model: LSTMAE | TransformerAE,
    matrix: np.ndarray,
    seq_len: int,
    batch_size: int,
) -> np.ndarray:
    """Score one validation matrix and align sequence-window errors to row timestamps."""
    if len(matrix) < seq_len:
        return np.full(len(matrix), np.nan, dtype=np.float32)
    dataset = SeqWindowsDataset([matrix], seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    window_errors = np.empty(len(dataset), dtype=np.float32)
    pos = 0
    with torch.inference_mode():
        for xb in loader:
            errors = model.reconstruction_error(xb.to(gpu.DEVICE, non_blocking=True))
            window_errors[pos : pos + len(errors)] = errors.cpu().numpy()
            pos += len(errors)
    scores = np.empty(len(matrix), dtype=np.float32)
    scores[: seq_len - 1] = window_errors[0]
    scores[seq_len - 1 :] = window_errors
    return scores


def validation_score_table(
    models: dict[str, VAE | LSTMAE | TransformerAE],
    validation_mats: list[np.ndarray],
    seq_len: int,
    batch_size: int,
    vae_score_reduction: str,
    vae_include_kld: bool,
) -> dict[str, np.ndarray]:
    """Return aligned normal-validation scores for all ensemble members."""
    scores = {name: [] for name in MODEL_NAMES}
    for matrix in validation_mats:
        loader = DataLoader(torch.from_numpy(matrix).float(), batch_size=batch_size, shuffle=False)
        vae_scores = []
        with torch.inference_mode():
            for xb in loader:
                vae_scores.append(
                    models["vae"]
                    .reconstruction_error(
                        xb.to(gpu.DEVICE, non_blocking=True),
                        reduction=vae_score_reduction,
                        include_kld=vae_include_kld,
                    )
                    .cpu()
                    .numpy()
                )
        scores["vae"].append(np.concatenate(vae_scores))
        scores["lstm"].append(score_seq_matrix(models["lstm"], matrix, seq_len, batch_size))
        scores["transformer"].append(
            score_seq_matrix(models["transformer"], matrix, seq_len, batch_size)
        )
    return {name: np.concatenate(parts) for name, parts in scores.items()}


def robust_norms(validation_scores: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """Compute per-model robust standardization parameters from validation only."""
    return {name: robust_location_scale(scores) for name, scores in validation_scores.items()}


def equal_weights() -> dict[str, float]:
    """Return equal weights for the three paper ensemble members."""
    weight = 1.0 / len(MODEL_NAMES)
    return {name: weight for name in MODEL_NAMES}


def ensure_same_timeline(records: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assert all model records share the same prediction timeline and return a base copy."""
    base = records[MODEL_NAMES[0]].copy()
    key_columns = ["farm", "dataset_id", "time_stamp", "status_type_id", "train_test"]
    for name in MODEL_NAMES[1:]:
        other = records[name]
        if len(base) != len(other) or not base[key_columns].equals(other[key_columns]):
            raise RuntimeError(f"{name} prediction timeline does not align with VAE records")
    return base


def evaluate_weighted_ensemble(
    model_records: dict[str, pd.DataFrame],
    validation_scores: dict[str, np.ndarray],
    weights: dict[str, float],
    weight_mode: str,
    threshold_percentile: float,
    artifact_stamp: str,
) -> dict:
    """Combine model scores, threshold on normal validation, and evaluate prediction CARE."""
    norms = robust_norms(validation_scores)
    validation_ensemble = combine_standardized(validation_scores, weights, norms)
    finite_validation = validation_ensemble[np.isfinite(validation_ensemble)]
    threshold = adaptive_threshold(finite_validation, threshold_percentile)

    base = ensure_same_timeline(model_records)
    prediction_scores = {
        name: frame["score"].to_numpy(dtype=float) for name, frame in model_records.items()
    }
    ensemble_scores = combine_standardized(prediction_scores, weights, norms)
    finite_scores = np.where(np.isfinite(ensemble_scores), ensemble_scores, -np.inf)
    base["score"] = ensemble_scores
    base["is_alarm"] = flag(finite_scores, threshold).astype(bool)

    events = load_events(OUT)
    ev_map = {(row.farm, str(row.event_id)): row for row in events.itertuples()}
    scores, labels = gpu._records_to_auc_arrays(base, ev_map)
    auc = compute_auc(scores, labels)
    care = evaluate_care(base, events)
    care_paths = write_care_artifacts(
        care,
        RESULTS,
        f"{artifact_stamp}_deep_ensemble_{weight_mode}_care",
    )
    return {
        "weight_mode": weight_mode,
        "weights": weights,
        "normalization": {
            name: {"median": float(median), "iqr": float(scale)}
            for name, (median, scale) in norms.items()
        },
        "threshold": float(threshold),
        "threshold_percentile": threshold_percentile,
        "threshold_source": "normal_validation_robust_standardized_ensemble_scores",
        "auc_roc": float(auc),
        "n_test": int(len(scores)),
        "n_positives": int(labels.sum()),
        "care": care.metrics(),
        "care_artifacts": {key: str(path) for key, path in care_paths.items()},
    }


def distribution_report(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """Summarize normal/anomaly score separation for portfolio diagnostics."""
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    normal = scores[labels == 0]
    anomaly = scores[labels == 1]
    if len(normal) == 0 or len(anomaly) == 0:
        return {
            "normal_median": None,
            "normal_p95": None,
            "anomaly_median": None,
            "anomaly_p05": None,
            "median_gap": None,
        }
    return {
        "normal_median": float(np.median(normal)),
        "normal_p95": float(np.percentile(normal, 95)),
        "anomaly_median": float(np.median(anomaly)),
        "anomaly_p05": float(np.percentile(anomaly, 5)),
        "median_gap": float(np.median(anomaly) - np.median(normal)),
    }


def evaluate_individual_model(
    name: str,
    records: pd.DataFrame,
    validation_scores: np.ndarray,
    threshold_percentile: float,
    artifact_stamp: str,
) -> dict:
    """Evaluate one ensemble member under the same CARE protocol."""
    finite_validation = validation_scores[np.isfinite(validation_scores)]
    threshold = adaptive_threshold(finite_validation, threshold_percentile)
    model_records = records.copy()
    raw_scores = model_records["score"].to_numpy(float)
    finite_scores = np.where(np.isfinite(raw_scores), raw_scores, -np.inf)
    model_records["is_alarm"] = flag(finite_scores, threshold).astype(bool)
    events = load_events(OUT)
    ev_map = {(row.farm, str(row.event_id)): row for row in events.itertuples()}
    scores, labels = gpu._records_to_auc_arrays(model_records, ev_map)
    auc = compute_auc(scores, labels)
    care = evaluate_care(model_records, events)
    care_paths = write_care_artifacts(
        care,
        RESULTS,
        f"{artifact_stamp}_deep_{name}_care",
    )
    return {
        "name": name,
        "threshold": float(threshold),
        "auc_roc": float(auc),
        "n_test": int(len(scores)),
        "n_positives": int(labels.sum()),
        "separation": distribution_report(scores, labels),
        "care": care.metrics(),
        "care_artifacts": {key: str(path) for key, path in care_paths.items()},
    }


def train_models(args: argparse.Namespace) -> tuple[
    dict[str, VAE | LSTMAE | TransformerAE],
    dict[str, dict[str, float | int]],
    tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray, np.ndarray],
]:
    """Collect shared data and train the three paper ensemble members."""
    farms = [farm.strip() for farm in args.farms.split(",")]
    configured_columns = feature_columns_by_farm(args.feature_set)
    shared = gpu.collect_all(
        farms,
        args.window,
        args.cap_train,
        args.validation_fraction,
        args.use_fft,
        configured_columns,
    )
    vae_train, train_mats, vae_val, validation_mats, mean, std = shared
    in_dim = vae_train.shape[1]
    models: dict[str, VAE | LSTMAE | TransformerAE] = {}
    train_info: dict[str, dict[str, float | int]] = {}

    models["vae"] = VAE(in_dim, latent=args.latent, hidden=args.hidden, beta=args.vae_beta)
    _, best_epoch, best_val_loss = gpu.train_vae_gpu(
        models["vae"],
        vae_train,
        vae_val,
        args.epochs,
        args.vae_batch,
        args.patience,
        args.min_delta,
        lr=args.vae_lr,
        name="deep_ensemble_vae",
    )
    train_info["vae"] = {"best_epoch": best_epoch, "best_val_loss": float(best_val_loss)}

    models["lstm"] = LSTMAE(
        in_dim,
        args.seq_len,
        latent=args.latent,
        hidden=args.hidden,
        num_layers=2,
    )
    _, best_epoch, best_val_loss = gpu.train_seq_gpu(
        models["lstm"],
        train_mats,
        validation_mats,
        args.seq_len,
        args.epochs,
        args.seq_batch,
        args.patience,
        args.min_delta,
        name="deep_ensemble_lstm",
    )
    train_info["lstm"] = {"best_epoch": best_epoch, "best_val_loss": float(best_val_loss)}

    nhead = 4 if args.hidden <= 128 else 8
    models["transformer"] = TransformerAE(
        in_dim,
        args.seq_len,
        latent=args.latent,
        d_model=args.hidden,
        nhead=nhead,
        num_layers=2,
    )
    _, best_epoch, best_val_loss = gpu.train_seq_gpu(
        models["transformer"],
        train_mats,
        validation_mats,
        args.seq_len,
        args.epochs,
        args.seq_batch,
        args.patience,
        args.min_delta,
        name="deep_ensemble_transformer",
    )
    train_info["transformer"] = {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
    }
    return models, train_info, shared


def main() -> None:
    """Run the strict deep ensemble experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default="B")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--cap-train", type=int, default=60000)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--vae-beta", type=float, default=0.1)
    parser.add_argument("--vae-lr", type=float, default=3e-4)
    parser.add_argument("--vae-score-reduction", choices=["sum", "mean"], default="sum")
    parser.add_argument("--vae-no-kld-score", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--vae-batch", type=int, default=1024)
    parser.add_argument("--seq-batch", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default=gpu.DEVICE)
    parser.add_argument("--feature-set", choices=["all", "avg_only", "paper_top"], default="all")
    parser.add_argument("--no-fft", action="store_true")
    args = parser.parse_args()

    if not OUT.exists():
        raise SystemExit("processed data missing; run convert_care_to_parquet first")
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be between 0 and 1")
    if not 95 <= args.threshold_percentile <= 99:
        raise SystemExit("--threshold-percentile must be between 95 and 99")

    gpu.DEVICE = args.device
    gpu.set_seed(args.seed)
    args.use_fft = not args.no_fft
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    start = time.time()

    print(f"Device: {gpu.DEVICE}")
    print(
        f"Deep ensemble: farms={args.farms} window={args.window} seq_len={args.seq_len} "
        f"hidden={args.hidden} latent={args.latent}"
    )
    models, train_info, shared = train_models(args)
    _, _, _, validation_mats, mean, std = shared
    configured_columns = feature_columns_by_farm(args.feature_set)

    print("Scoring normal validation for ensemble normalization/weights...")
    validation_scores = validation_score_table(
        models,
        validation_mats,
        args.seq_len,
        args.seq_batch,
        args.vae_score_reduction,
        not args.vae_no_kld_score,
    )
    weights_by_mode = {
        "equal": equal_weights(),
        "stability": validation_stability_weights(validation_scores),
    }
    print(f"weights: {weights_by_mode}")

    farms = [farm.strip() for farm in args.farms.split(",")]
    print("Scoring prediction timelines for each model...")
    model_records = {
        "vae": gpu.evaluate_vae_records(
            models["vae"],
            farms,
            args.window,
            mean,
            std,
            args.use_fft,
            configured_columns,
            score_reduction=args.vae_score_reduction,
            include_kld=not args.vae_no_kld_score,
        ),
        "lstm": gpu.evaluate_seq_records(
            models["lstm"],
            farms,
            args.window,
            args.seq_len,
            mean,
            std,
            args.use_fft,
            batch=args.seq_batch,
            feature_columns_by_farm=configured_columns,
        ),
        "transformer": gpu.evaluate_seq_records(
            models["transformer"],
            farms,
            args.window,
            args.seq_len,
            mean,
            std,
            args.use_fft,
            batch=args.seq_batch,
            feature_columns_by_farm=configured_columns,
        ),
    }

    individual_results = [
        evaluate_individual_model(
            name,
            model_records[name],
            validation_scores[name],
            args.threshold_percentile,
            stamp,
        )
        for name in MODEL_NAMES
    ]

    ensemble_results = [
        evaluate_weighted_ensemble(
            model_records,
            validation_scores,
            weights,
            mode,
            args.threshold_percentile,
            stamp,
        )
        for mode, weights in weights_by_mode.items()
    ]

    in_dim = validation_mats[0].shape[1]
    result = {
        "model": "deep_ensemble",
        "farms": farms,
        "window": args.window,
        "seq_len": args.seq_len,
        "in_dim": int(in_dim),
        "hidden": args.hidden,
        "latent": args.latent,
        "vae_beta": args.vae_beta,
        "vae_lr": args.vae_lr,
        "vae_score_reduction": args.vae_score_reduction,
        "vae_include_kld_score": not args.vae_no_kld_score,
        "epochs": args.epochs,
        "cap_train": args.cap_train,
        "feature_set": args.feature_set,
        "feature_columns_by_farm": configured_columns,
        "validation_fraction": args.validation_fraction,
        "threshold_percentile": args.threshold_percentile,
        "seed": args.seed,
        "device": gpu.DEVICE,
        "use_fft": args.use_fft,
        "train_info": train_info,
        "individual_results": individual_results,
        "ensemble_results": ensemble_results,
        "elapsed_s": round(time.time() - start, 1),
    }
    path = RESULTS / f"{stamp}_deep_ensemble_result.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, model in models.items():
        torch.save(model.state_dict(), MODELS_DIR / f"{stamp}_deep_ensemble_{name}.pt")
    np.savez(MODELS_DIR / f"{stamp}_deep_ensemble_norm.npz", mean=mean, std=std)

    for item in ensemble_results:
        care = item["care"]["care"]
        auc = item["auc_roc"]
        print(f"{item['weight_mode']}: CARE={care:.4f} AUC={auc:.4f}")
    for item in individual_results:
        print(f"{item['name']}: CARE={item['care']['care']:.4f} AUC={item['auc_roc']:.4f}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
