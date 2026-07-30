"""Focused VAE optimization runner under the frozen CARE protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow direct execution (`python scripts/run_vae_optimized.py`) from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_deep_ensemble as deep  # noqa: E402
import scripts.run_gpu_tune as gpu  # noqa: E402
from faultdiagnose.data import load_care, load_events  # noqa: E402
from faultdiagnose.evaluation.anomaly import compute_auc  # noqa: E402
from faultdiagnose.evaluation.care import evaluate_care, write_care_artifacts  # noqa: E402
from faultdiagnose.evaluation.ensemble import adaptive_threshold, flag  # noqa: E402
from faultdiagnose.models import VAE  # noqa: E402
from faultdiagnose.training import predicted_normal_scores, train_adaptive_threshold  # noqa: E402

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"
CHECKPOINTS = RESULTS / "checkpoints"


def distribution_report(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """Summarize normal/anomaly score distribution after finite filtering."""
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    normal = scores[labels == 0]
    anomaly = scores[labels == 1]
    if len(normal) == 0 or len(anomaly) == 0:
        return {
            "normal_mean": None,
            "normal_std": None,
            "anomaly_mean": None,
            "anomaly_std": None,
            "mean_gap": None,
            "normal_median": None,
            "anomaly_median": None,
            "median_gap": None,
        }
    return {
        "normal_mean": float(np.mean(normal)),
        "normal_std": float(np.std(normal)),
        "anomaly_mean": float(np.mean(anomaly)),
        "anomaly_std": float(np.std(anomaly)),
        "mean_gap": float(np.mean(anomaly) - np.mean(normal)),
        "normal_median": float(np.median(normal)),
        "anomaly_median": float(np.median(anomaly)),
        "median_gap": float(np.median(anomaly) - np.median(normal)),
    }


def main() -> None:
    """Train and evaluate one VAE configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default="B")
    parser.add_argument(
        "--feature-set", choices=["all", "avg_only", "paper_top"], default="avg_only"
    )
    parser.add_argument(
        "--input-profile",
        choices=["windowed", "care_raw"],
        default="windowed",
        help="Window-engineered vectors or raw CARE-AE-compatible SCADA rows.",
    )
    parser.add_argument(
        "--normalization",
        choices=["global", "per_asset"],
        default="global",
        help="One global Z-score or a normal-history Z-score for each turbine asset.",
    )
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--cap-train", type=int, default=60000)
    parser.add_argument(
        "--normal-sampling",
        choices=["sequential", "balanced"],
        default="sequential",
        help="Sequential cap or balanced contiguous blocks across normal datasets.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument(
        "--threshold-mode",
        choices=["percentile", "adaptive_nn"],
        default="percentile",
        help="Fixed normal-score percentile or CARE-style input-conditioned threshold.",
    )
    parser.add_argument("--threshold-hidden", type=int, default=128)
    parser.add_argument("--threshold-lr", type=float, default=1e-3)
    parser.add_argument("--threshold-epochs", type=int, default=300)
    parser.add_argument("--threshold-batch", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--scheduler", choices=["none", "warmup_cosine", "plateau"], default="warmup_cosine"
    )
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument(
        "--kl-anneal-epochs",
        type=int,
        default=0,
        help="Linearly warm beta from 0 to --beta over this many epochs; 0 disables annealing.",
    )
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--score-reduction", choices=["sum", "mean", "l2"], default="mean")
    parser.add_argument("--no-kld-score", action="store_true")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default=gpu.DEVICE)
    parser.add_argument("--no-fft", action="store_true")
    args = parser.parse_args()

    if not OUT.exists():
        raise SystemExit("processed data missing; run convert_care_to_parquet first")
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be between 0 and 1")
    if not 95 <= args.threshold_percentile <= 99:
        raise SystemExit("--threshold-percentile must be between 95 and 99")
    if args.input_profile == "care_raw" and args.feature_set != "all":
        raise SystemExit("--input-profile care_raw requires --feature-set all")
    if args.normalization == "per_asset" and args.input_profile != "care_raw":
        raise SystemExit("--normalization per_asset currently requires --input-profile care_raw")

    gpu.DEVICE = args.device
    gpu.set_seed(args.seed)
    use_fft = not args.no_fft
    farms = [farm.strip() for farm in args.farms.split(",")]
    configured_columns = deep.feature_columns_by_farm(args.feature_set)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    start = time.time()

    print(
        f"VAE optimize: farms={args.farms} feature_set={args.feature_set} "
        f"profile={args.input_profile}/{args.normalization} window={args.window} "
        f"epochs={args.epochs} "
        f"scheduler={args.scheduler}",
        flush=True,
    )
    asset_standardizers = None
    if args.input_profile == "care_raw":
        if args.normalization == "per_asset":
            *shared, asset_standardizers = gpu.collect_raw_normal_per_asset(
                farms,
                args.cap_train,
                args.validation_fraction,
                args.seed,
                configured_columns,
            )
        else:
            shared = gpu.collect_raw_normal(
                farms,
                args.cap_train,
                args.validation_fraction,
                args.seed,
                configured_columns,
            )
    else:
        shared = gpu.collect_all(
            farms,
            args.window,
            args.cap_train,
            args.validation_fraction,
            use_fft,
            configured_columns,
            normal_sampling=args.normal_sampling,
        )
    train_x, _, validation_x, _, mean, std = shared
    model = VAE(train_x.shape[1], latent=args.latent, hidden=args.hidden, beta=args.beta)
    loss_path = RESULTS / f"{stamp}_vae_optimized_loss.txt"
    with loss_path.open("w", encoding="utf-8") as loss_file:
        losses, best_epoch, best_val_loss = gpu.train_vae_gpu(
            model,
            train_x,
            validation_x,
            args.epochs,
            args.batch,
            args.patience,
            args.min_delta,
            lr=args.lr,
            scheduler=args.scheduler,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
            kl_anneal_epochs=args.kl_anneal_epochs,
            name="vae_optimized",
            loss_f=loss_file,
        )

    include_kld = not args.no_kld_score
    validation_scores = gpu.validation_scores_vae(
        model,
        validation_x,
        args.batch,
        args.score_reduction,
        include_kld,
    )
    threshold_model = None
    threshold_checkpoint = None
    threshold = adaptive_threshold(validation_scores, args.threshold_percentile)
    if args.threshold_mode == "adaptive_nn":
        train_scores = gpu.validation_scores_vae(
            model, train_x, args.batch, args.score_reduction, include_kld
        )
        threshold_model = train_adaptive_threshold(
            train_x,
            train_scores,
            hidden=args.threshold_hidden,
            learning_rate=args.threshold_lr,
            epochs=args.threshold_epochs,
            batch_size=args.threshold_batch,
            device=gpu.DEVICE,
        )
        validation_expected = predicted_normal_scores(
            threshold_model, validation_x, batch_size=args.threshold_batch, device=gpu.DEVICE
        )
        threshold = adaptive_threshold(
            validation_scores - validation_expected, args.threshold_percentile
        )
    records = gpu.evaluate_vae_records(
        model,
        farms,
        args.window,
        mean,
        std,
        use_fft,
        configured_columns,
        score_reduction=args.score_reduction,
        include_kld=include_kld,
        threshold_model=threshold_model,
        threshold_batch=args.threshold_batch,
        raw_input=args.input_profile == "care_raw",
        asset_standardizers=asset_standardizers,
    )
    records["is_alarm"] = flag(records["score"].fillna(-np.inf).to_numpy(), threshold).astype(bool)

    events = load_events(OUT)
    ev_map = {(row.farm, str(row.event_id)): row for row in events.itertuples()}
    scores, labels = gpu._records_to_auc_arrays(records, ev_map)
    auc = compute_auc(scores, labels)
    care = evaluate_care(records, events)
    care_paths = write_care_artifacts(care, RESULTS, f"{stamp}_vae_optimized_care")
    checkpoint_path = CHECKPOINTS / f"{stamp}_vae_optimized.pt"
    norm_path = CHECKPOINTS / f"{stamp}_vae_optimized_norm.npz"
    torch.save(model.state_dict(), checkpoint_path)
    if threshold_model is not None:
        threshold_checkpoint = CHECKPOINTS / f"{stamp}_vae_optimized_adaptive_threshold.pt"
        torch.save(threshold_model.state_dict(), threshold_checkpoint)
    normalization_payload = {"mean": mean, "std": std}
    if asset_standardizers is not None:
        for asset, (asset_mean, asset_std) in asset_standardizers.items():
            normalization_payload[f"asset_{asset}_mean"] = asset_mean
            normalization_payload[f"asset_{asset}_std"] = asset_std
    np.savez(norm_path, **normalization_payload)

    result = {
        "model": "vae_optimized",
        "farms": farms,
        "feature_set": args.feature_set,
        "input_profile": args.input_profile,
        "normalization_mode": args.normalization,
        "feature_columns_by_farm": configured_columns,
        "window": args.window,
        "normal_sampling": args.normal_sampling,
        "use_fft": use_fft,
        "in_dim": int(train_x.shape[1]),
        "hidden": args.hidden,
        "latent": args.latent,
        "beta": args.beta,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "kl_anneal_epochs": args.kl_anneal_epochs,
        "patience": args.patience,
        "score_reduction": args.score_reduction,
        "include_kld_score": include_kld,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "loss_tail": [float(value) for value in losses[-10:]],
        "threshold": float(threshold),
        "threshold_percentile": args.threshold_percentile,
        "threshold_mode": args.threshold_mode,
        "threshold_hidden": args.threshold_hidden if threshold_model is not None else None,
        "threshold_checkpoint": str(threshold_checkpoint) if threshold_checkpoint else None,
        "auc_roc": float(auc),
        "n_train": int(len(train_x)),
        "n_validation": int(len(validation_x)),
        "n_test": int(len(scores)),
        "n_positives": int(labels.sum()),
        "separation": distribution_report(scores, labels),
        "care": care.metrics(),
        "care_artifacts": {key: str(path) for key, path in care_paths.items()},
        "loss_log": str(loss_path),
        "checkpoint": str(checkpoint_path),
        "normalization": str(norm_path),
        "elapsed_s": round(time.time() - start, 1),
    }
    result_path = RESULTS / f"{stamp}_vae_optimized_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"VAE optimized: AUC={auc:.4f} CARE={care.care:.4f} "
        f"best_epoch={best_epoch} val={best_val_loss:.4f}",
        flush=True,
    )
    print(f"saved -> {result_path}", flush=True)


if __name__ == "__main__":
    main()
