"""Evaluate an equal-weight VAE + Transformer ensemble from saved experiment artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_gpu_tune as gpu  # noqa: E402
from faultdiagnose.data import load_care, load_events  # noqa: E402
from faultdiagnose.evaluation.anomaly import compute_auc  # noqa: E402
from faultdiagnose.evaluation.care import evaluate_care, write_care_artifacts  # noqa: E402
from faultdiagnose.evaluation.ensemble import (  # noqa: E402
    adaptive_threshold,
    empirical_percentile_rank,
    flag,
)
from faultdiagnose.models import VAE, SeqWindowsDataset, TransformerAE  # noqa: E402

OUT = load_care.OUT_DEFAULT
RESULTS = PROJECT_ROOT / "results"


def load_result(path: Path) -> dict:
    """Load one model's saved experiment metadata."""
    if not path.exists():
        raise FileNotFoundError(f"result not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def vae_scores(model: VAE, matrix: np.ndarray, batch_size: int, result: dict) -> np.ndarray:
    """Score one normalized feature matrix with the saved VAE scoring convention."""
    loader = DataLoader(torch.from_numpy(matrix).float(), batch_size=batch_size, shuffle=False)
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            parts.append(
                model.reconstruction_error(
                    batch.to(gpu.DEVICE, non_blocking=True),
                    reduction=result["score_reduction"],
                    include_kld=result["include_kld_score"],
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(parts)


def transformer_scores(
    model: TransformerAE, matrix: np.ndarray, seq_len: int, batch_size: int
) -> np.ndarray:
    """Return one Transformer score per row, filling initial sequence positions consistently."""
    dataset = SeqWindowsDataset([matrix], seq_len)
    if len(dataset) == 0:
        raise ValueError("validation segment is shorter than Transformer sequence length")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            errors = model.reconstruction_error(batch.to(gpu.DEVICE, non_blocking=True))
            parts.append(errors.cpu().numpy())
    window_scores = np.concatenate(parts)
    scores = np.empty(len(matrix), dtype=np.float32)
    scores[: seq_len - 1] = window_scores[0]
    scores[seq_len - 1 :] = window_scores
    return scores


def validation_score_pairs(
    vae: VAE,
    transformer: TransformerAE,
    vae_result: dict,
    transformer_result: dict,
    batch_size: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recreate aligned normal validation rows for percentile-rank calibration."""
    farms = vae_result["farms"]
    vae_shared = gpu.collect_all(
        farms,
        vae_result["window"],
        vae_result["n_train"] + vae_result["n_validation"],
        validation_fraction,
        vae_result["use_fft"],
        vae_result["feature_columns_by_farm"],
    )
    transformer_shared = gpu.collect_all(
        farms,
        transformer_result["window"],
        transformer_result["n_train"] + transformer_result["n_validation"],
        validation_fraction,
        transformer_result["use_fft"],
        transformer_result["feature_columns_by_farm"],
        feature_profile=transformer_result.get("feature_profile", "full"),
    )
    _, _, _, vae_val_mats, _, _ = vae_shared
    _, _, _, transformer_val_mats, _, _ = transformer_shared
    if len(vae_val_mats) != len(transformer_val_mats):
        raise RuntimeError("VAE and Transformer validation segments do not align")
    vae_parts: list[np.ndarray] = []
    transformer_parts: list[np.ndarray] = []
    for vae_matrix, transformer_matrix in zip(vae_val_mats, transformer_val_mats, strict=True):
        if len(vae_matrix) != len(transformer_matrix):
            raise RuntimeError("VAE and Transformer validation rows do not align")
        if len(transformer_matrix) < transformer_result["seq_len"]:
            continue
        vae_parts.append(vae_scores(vae, vae_matrix, batch_size, vae_result))
        transformer_parts.append(
            transformer_scores(
                transformer,
                transformer_matrix,
                transformer_result["seq_len"],
                batch_size,
            )
        )
    if not vae_parts:
        raise ValueError("No aligned validation segments are long enough for the Transformer")
    return np.concatenate(vae_parts), np.concatenate(transformer_parts)


def main() -> None:
    """Load checkpoints, calibrate the two-model ensemble, and report strict CARE metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vae-result",
        type=Path,
        default=RESULTS / "20260717_2357_vae_optimized_result.json",
    )
    parser.add_argument(
        "--transformer-result",
        type=Path,
        default=RESULTS / "20260727_203619_689562_transformer_optimized_result.json",
    )
    parser.add_argument("--vae-weight", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument(
        "--threshold-grid",
        default="",
        help="Optional comma-separated validation percentiles to diagnose in one scoring pass.",
    )
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default=gpu.DEVICE)
    args = parser.parse_args()
    if not OUT.exists():
        raise SystemExit("processed data missing; run convert_care_to_parquet first")
    if not 0.0 <= args.vae_weight <= 1.0:
        raise SystemExit("--vae-weight must be between 0 and 1")
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be between 0 and 1")
    if not 95 <= args.threshold_percentile <= 99:
        raise SystemExit("--threshold-percentile must be between 95 and 99")
    threshold_grid = (
        tuple(float(item) for item in args.threshold_grid.split(",") if item.strip())
        if args.threshold_grid
        else ()
    )
    if any(value < 95 or value > 99 for value in threshold_grid):
        raise SystemExit("--threshold-grid values must be between 95 and 99")

    vae_result = load_result(args.vae_result)
    transformer_result = load_result(args.transformer_result)
    if vae_result["farms"] != transformer_result["farms"]:
        raise SystemExit("VAE and Transformer experiments must use the same farms")
    gpu.DEVICE = args.device

    vae = VAE(
        vae_result["in_dim"],
        latent=vae_result["latent"],
        hidden=vae_result["hidden"],
        beta=vae_result["beta"],
    ).to(gpu.DEVICE)
    transformer = TransformerAE(
        transformer_result["in_dim"],
        transformer_result["seq_len"],
        latent=transformer_result["latent"],
        d_model=transformer_result["d_model"],
        nhead=transformer_result["nhead"],
        num_layers=transformer_result["num_layers"],
        dropout=transformer_result["dropout"],
        architecture=transformer_result["architecture"],
    ).to(gpu.DEVICE)
    vae_state = torch.load(vae_result["checkpoint"], map_location=gpu.DEVICE, weights_only=True)
    vae.load_state_dict(vae_state)
    transformer.load_state_dict(
        torch.load(transformer_result["checkpoint"], map_location=gpu.DEVICE, weights_only=True)
    )
    vae.eval()
    transformer.eval()

    print("Calibrating VAE + Transformer ensemble on aligned normal validation rows...", flush=True)
    vae_validation, transformer_validation = validation_score_pairs(
        vae, transformer, vae_result, transformer_result, args.batch, args.validation_fraction
    )
    vae_ranks = empirical_percentile_rank(vae_validation, vae_validation)
    transformer_ranks = empirical_percentile_rank(transformer_validation, transformer_validation)
    transformer_weight = 1.0 - args.vae_weight
    validation_ensemble = args.vae_weight * vae_ranks + transformer_weight * transformer_ranks
    threshold = adaptive_threshold(validation_ensemble, args.threshold_percentile)

    vae_norm = np.load(vae_result["normalization"])
    transformer_norm = np.load(transformer_result["normalization"])
    vae_records = gpu.evaluate_vae_records(
        vae,
        vae_result["farms"],
        vae_result["window"],
        vae_norm["mean"],
        vae_norm["std"],
        vae_result["use_fft"],
        vae_result["feature_columns_by_farm"],
        score_reduction=vae_result["score_reduction"],
        include_kld=vae_result["include_kld_score"],
    )
    transformer_records = gpu.evaluate_seq_records(
        transformer,
        transformer_result["farms"],
        transformer_result["window"],
        transformer_result["seq_len"],
        transformer_norm["mean"],
        transformer_norm["std"],
        transformer_result["use_fft"],
        batch=args.batch,
        feature_columns_by_farm=transformer_result["feature_columns_by_farm"],
        feature_profile=transformer_result.get("feature_profile", "full"),
    )
    key_columns = ["farm", "dataset_id", "time_stamp", "status_type_id", "train_test"]
    if not vae_records[key_columns].equals(transformer_records[key_columns]):
        raise RuntimeError("VAE and Transformer prediction timelines do not align")

    vae_prediction_ranks = empirical_percentile_rank(
        vae_records["score"].to_numpy(), vae_validation
    )
    transformer_prediction_ranks = empirical_percentile_rank(
        transformer_records["score"].to_numpy(), transformer_validation
    )
    ensemble_scores = (
        args.vae_weight * vae_prediction_ranks + transformer_weight * transformer_prediction_ranks
    )
    records = vae_records.copy()
    records["score"] = ensemble_scores
    records["is_alarm"] = flag(np.nan_to_num(ensemble_scores, nan=-np.inf), threshold).astype(bool)
    events = load_events(OUT)
    event_map = {(row.farm, str(row.event_id)): row for row in events.itertuples()}
    scores, labels = gpu._records_to_auc_arrays(records, event_map)
    auc = compute_auc(scores, labels)
    care = evaluate_care(records, events)
    threshold_diagnostics: list[dict[str, float]] = []
    for percentile in threshold_grid:
        candidate_threshold = adaptive_threshold(validation_ensemble, percentile)
        candidate = records.copy()
        candidate["is_alarm"] = flag(
            np.nan_to_num(ensemble_scores, nan=-np.inf), candidate_threshold
        ).astype(bool)
        candidate_care = evaluate_care(candidate, events)
        threshold_diagnostics.append(
            {
                "percentile": percentile,
                "threshold": float(candidate_threshold),
                "care": float(candidate_care.care),
                "coverage_f0_5": float(candidate_care.coverage_f0_5),
                "accuracy": float(candidate_care.accuracy),
                "reliability_event_f0_5": float(candidate_care.reliability_event_f0_5),
                "earliness": float(candidate_care.earliness),
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    care_paths = write_care_artifacts(care, RESULTS, f"{stamp}_vae_transformer_ensemble_care")
    result = {
        "model": "vae_transformer_percentile_ensemble",
        "vae_result": str(args.vae_result),
        "transformer_result": str(args.transformer_result),
        "weights": {"vae": args.vae_weight, "transformer": transformer_weight},
        "validation_fraction": args.validation_fraction,
        "normalization": "validation_empirical_percentile_rank",
        "threshold": float(threshold),
        "threshold_percentile": args.threshold_percentile,
        "threshold_diagnostics": threshold_diagnostics,
        "auc_roc": float(auc),
        "n_test": int(len(scores)),
        "n_positives": int(labels.sum()),
        "care": care.metrics(),
        "care_artifacts": {key: str(path) for key, path in care_paths.items()},
    }
    result_path = RESULTS / f"{stamp}_vae_transformer_ensemble_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"VAE+Transformer ensemble: AUC={auc:.4f} CARE={care.care:.4f} "
        f"threshold={threshold:.4f}",
        flush=True,
    )
    print(f"saved -> {result_path}")


if __name__ == "__main__":
    main()
