"""Reproduce a CARE-style dense autoencoder baseline on Farm B.

This runner deliberately separates the benchmark-style deterministic AE from the
portfolio VAE experiments: it uses a random normal validation split, optional
Gaussian denoising, and reports a fixed grid of normal-validation thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow direct execution (`python scripts/run_care_dense_ae.py`) from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_deep_ensemble as deep  # noqa: E402
import scripts.run_gpu_tune as gpu  # noqa: E402
from faultdiagnose.data import load_care, load_events  # noqa: E402
from faultdiagnose.evaluation.anomaly import compute_auc  # noqa: E402
from faultdiagnose.evaluation.care import evaluate_care, write_care_artifacts  # noqa: E402
from faultdiagnose.evaluation.ensemble import adaptive_threshold, flag  # noqa: E402
from faultdiagnose.models import DenseAutoencoder  # noqa: E402

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"
CHECKPOINTS = RESULTS / "checkpoints"


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    """Parse a comma-separated hidden-layer specification."""
    widths = tuple(int(item) for item in value.split(",") if item.strip())
    if not widths or min(widths) <= 0:
        raise argparse.ArgumentTypeError("--hidden-dims must be comma-separated positive integers")
    return widths


def random_validation_split(
    train_x: np.ndarray, validation_x: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create the random normal validation split used for this benchmark comparison."""
    all_normal = np.concatenate((train_x, validation_x), axis=0)
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(all_normal))
    split = max(1, min(len(all_normal) - 1, round(len(all_normal) * (1.0 - fraction))))
    return all_normal[order[:split]], all_normal[order[split:]]


def main() -> None:
    """Train one dense AE and select a CARE score from a fixed threshold grid."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default="B")
    parser.add_argument("--feature-set", choices=["all", "avg_only", "paper_top"], default="all")
    parser.add_argument("--feature-profile", choices=["full", "stat_aware"], default="full")
    parser.add_argument("--window", type=int, default=576)
    parser.add_argument("--cap-train", type=int, default=60000)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(512, 256, 128))
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--thresholds", default="95,97,98,99")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default=gpu.DEVICE)
    parser.add_argument("--no-fft", action="store_true")
    args = parser.parse_args()

    if not OUT.exists():
        raise SystemExit("processed data missing; run convert_care_to_parquet first")
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be between 0 and 1")
    threshold_grid = tuple(float(item) for item in args.thresholds.split(","))
    if not threshold_grid or any(value < 95 or value > 99 for value in threshold_grid):
        raise SystemExit("--thresholds must be comma-separated values in [95, 99]")
    if not 95 <= args.threshold_percentile <= 99:
        raise SystemExit("--threshold-percentile must be in [95, 99]")

    gpu.DEVICE = args.device
    gpu.set_seed(args.seed)
    farms = [farm.strip() for farm in args.farms.split(",")]
    use_fft = not args.no_fft
    configured_columns = deep.feature_columns_by_farm(args.feature_set)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    started = time.time()

    print(
        f"CARE dense AE: farms={args.farms} feature_set={args.feature_set} "
        f"profile={args.feature_profile} window={args.window} noise={args.noise_std}",
        flush=True,
    )
    collected = gpu.collect_all(
        farms,
        args.window,
        args.cap_train,
        args.validation_fraction,
        use_fft,
        configured_columns,
        feature_profile=args.feature_profile,
    )
    train_x, _, validation_x, _, mean, std = collected
    train_x, validation_x = random_validation_split(
        train_x, validation_x, args.validation_fraction, args.seed
    )
    model = DenseAutoencoder(train_x.shape[1], hidden_dims=args.hidden_dims)
    loss_path = RESULTS / f"{stamp}_care_dense_ae_loss.txt"
    with loss_path.open("w", encoding="utf-8") as loss_file:
        losses, best_epoch, best_validation_loss = gpu.train_dense_ae_gpu(
            model,
            train_x,
            validation_x,
            args.epochs,
            args.batch,
            args.patience,
            args.min_delta,
            lr=args.lr,
            noise_std=args.noise_std,
            loss_f=loss_file,
        )

    validation_scores = gpu.validation_scores_vae(
        model, validation_x, args.batch, score_reduction="mean", include_kld=False
    )
    records = gpu.evaluate_vae_records(
        model,
        farms,
        args.window,
        mean,
        std,
        use_fft,
        configured_columns,
        score_reduction="mean",
        include_kld=False,
        feature_profile=args.feature_profile,
    )
    events = load_events(OUT)
    event_map = {(row.farm, str(row.event_id)): row for row in events.itertuples()}
    scores, labels = gpu._records_to_auc_arrays(records, event_map)
    auc = compute_auc(scores, labels)
    threshold_results: list[dict[str, float]] = []
    for percentile in threshold_grid:
        threshold = adaptive_threshold(validation_scores, percentile)
        candidate = records.copy()
        candidate["is_alarm"] = flag(candidate["score"].fillna(-np.inf), threshold).astype(bool)
        care = evaluate_care(candidate, events)
        threshold_results.append(
            {"percentile": percentile, "threshold": float(threshold), "care": float(care.care)}
        )
    chosen = {
        "percentile": args.threshold_percentile,
        "threshold": float(adaptive_threshold(validation_scores, args.threshold_percentile)),
    }
    records["is_alarm"] = flag(records["score"].fillna(-np.inf), chosen["threshold"]).astype(bool)
    care = evaluate_care(records, events)
    artifact_paths = write_care_artifacts(care, RESULTS, f"{stamp}_care_dense_ae_care")
    checkpoint_path = CHECKPOINTS / f"{stamp}_care_dense_ae.pt"
    normalization_path = CHECKPOINTS / f"{stamp}_care_dense_ae_norm.npz"
    torch.save(model.state_dict(), checkpoint_path)
    np.savez(normalization_path, mean=mean, std=std)
    result = {
        "model": "care_dense_ae",
        "farms": farms,
        "feature_set": args.feature_set,
        "feature_profile": args.feature_profile,
        "window": args.window,
        "use_fft": use_fft,
        "in_dim": int(train_x.shape[1]),
        "hidden_dims": args.hidden_dims,
        "noise_std": args.noise_std,
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_validation_loss),
        "loss_tail": [float(value) for value in losses[-10:]],
        "auc_roc": float(auc),
        "threshold_grid": threshold_results,
        "chosen_threshold": chosen,
        "care": care.metrics(),
        "care_artifacts": {key: str(path) for key, path in artifact_paths.items()},
        "checkpoint": str(checkpoint_path),
        "normalization": str(normalization_path),
        "elapsed_s": round(time.time() - started, 1),
    }
    result_path = RESULTS / f"{stamp}_care_dense_ae_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"CARE dense AE: AUC={auc:.4f} CARE={care.care:.4f} "
        f"threshold=P{chosen['percentile']:.0f} best_epoch={best_epoch}",
        flush=True,
    )
    print(f"saved -> {result_path}", flush=True)


if __name__ == "__main__":
    main()
