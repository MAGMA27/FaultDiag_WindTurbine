"""Focused Transformer-AE runner under the frozen CARE protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow direct execution (`python scripts/run_transformer_optimized.py`) from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_deep_ensemble as deep  # noqa: E402
import scripts.run_gpu_tune as gpu  # noqa: E402
from faultdiagnose.data import load_care, load_events  # noqa: E402
from faultdiagnose.evaluation.anomaly import compute_auc  # noqa: E402
from faultdiagnose.evaluation.care import evaluate_care, write_care_artifacts  # noqa: E402
from faultdiagnose.evaluation.ensemble import adaptive_threshold, flag  # noqa: E402
from faultdiagnose.models import TransformerAE  # noqa: E402
from faultdiagnose.training import predicted_normal_scores, train_adaptive_threshold  # noqa: E402

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"
CHECKPOINTS = RESULTS / "checkpoints"


def main() -> None:
    """Train and evaluate one Transformer-AE configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default="B")
    parser.add_argument(
        "--feature-set", choices=["all", "avg_only", "paper_top"], default="avg_only"
    )
    parser.add_argument(
        "--feature-profile",
        choices=["full", "stat_aware", "raw_stat_compact"],
        default="full",
        help="How original 10-minute avg/min/max/std statistics are expanded.",
    )
    parser.add_argument(
        "--feature-list",
        type=Path,
        help="JSON list emitted by audit_feature_redundancy.py; applied after feature engineering.",
    )
    parser.add_argument("--window", type=int, default=576)
    parser.add_argument("--seq-len", type=int, default=96)
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
    parser.add_argument("--threshold-hidden", type=int, default=256)
    parser.add_argument("--threshold-lr", type=float, default=1e-3)
    parser.add_argument("--threshold-epochs", type=int, default=300)
    parser.add_argument("--threshold-batch", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--latent", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--architecture", choices=["broadcast", "cross_attention"], default="cross_attention"
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler", choices=["none", "warmup_cosine", "plateau"], default="warmup_cosine"
    )
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--device", default=gpu.DEVICE)
    parser.add_argument("--no-fft", action="store_true")
    parser.add_argument(
        "--window-cache-dir",
        default=str(RESULTS / "window_cache"),
        help="Local cache for materialized train/validation sequence windows.",
    )
    parser.add_argument("--no-window-cache", action="store_true")
    parser.add_argument(
        "--ram-window-cache",
        action="store_true",
        help="Materialize windows in RAM only; does not write a persistent cache file.",
    )
    parser.add_argument(
        "--max-window-cache-gb",
        type=float,
        default=1.0,
        help="Skip materialized window cache above this estimated CPU-memory size.",
    )
    parser.add_argument(
        "--window-cache-dtype",
        choices=["float16", "float32"],
        default="float32",
        help="Storage dtype for cached windows; float16 halves CPU-memory use.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if not OUT.exists():
        raise SystemExit("processed data missing; run convert_care_to_parquet first")
    gpu.DEVICE = args.device
    gpu.set_seed(args.seed)
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be between 0 and 1")
    if not 95 <= args.threshold_percentile <= 99:
        raise SystemExit("--threshold-percentile must be between 95 and 99")
    if args.d_model % args.nhead:
        raise SystemExit("--d-model must be divisible by --nhead")
    if args.feature_list is not None and args.feature_profile != "full":
        raise SystemExit("--feature-list and a non-full --feature-profile cannot be combined")
    if args.no_window_cache and args.ram_window_cache:
        raise SystemExit("--no-window-cache and --ram-window-cache cannot be combined")

    use_fft = not args.no_fft
    farms = [farm.strip() for farm in args.farms.split(",")]
    columns = deep.feature_columns_by_farm(args.feature_set)
    engineered_columns: dict[str, list[str]] | None = None
    if args.feature_list is not None:
        if not args.feature_list.exists():
            raise SystemExit(f"feature list not found: {args.feature_list}")
        selected = json.loads(args.feature_list.read_text(encoding="utf-8"))
        if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
            raise SystemExit("--feature-list must contain a JSON list of feature names")
        if not selected:
            raise SystemExit("--feature-list must not be empty")
        columns = None
        engineered_columns = {farm: selected for farm in farms}
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    start = time.time()
    print(
        f"Transformer optimize: farms={args.farms} feature_set={args.feature_set} "
        f"profile={args.feature_profile} window={args.window} seq_len={args.seq_len} "
        f"architecture={args.architecture}",
        flush=True,
    )
    shared = gpu.collect_all(
        farms,
        args.window,
        args.cap_train,
        args.validation_fraction,
        use_fft,
        columns,
        engineered_columns,
        args.feature_profile,
        args.normal_sampling,
    )
    train_x, train_mats, validation_x, validation_mats, mean, std = shared
    model = TransformerAE(
        train_x.shape[1],
        args.seq_len,
        latent=args.latent,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        architecture=args.architecture,
    )
    loss_path = RESULTS / f"{stamp}_transformer_optimized_loss.txt"
    with loss_path.open("w", encoding="utf-8") as loss_file:
        losses, best_epoch, best_val_loss = gpu.train_seq_gpu(
            model,
            train_mats,
            validation_mats,
            args.seq_len,
            args.epochs,
            args.batch,
            args.patience,
            args.min_delta,
            lr=args.lr,
            scheduler=args.scheduler,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
            grad_clip=args.grad_clip,
            name="transformer_optimized",
            loss_f=loss_file,
            cache_dir=(
                None if args.no_window_cache or args.ram_window_cache else args.window_cache_dir
            ),
            max_cache_gb=args.max_window_cache_gb,
            cache_dtype=args.window_cache_dtype,
            num_workers=args.num_workers,
            ram_window_cache=args.ram_window_cache,
        )
    validation_scores = gpu.validation_scores_seq(model, validation_mats, args.seq_len, args.batch)
    threshold = adaptive_threshold(validation_scores, args.threshold_percentile)
    threshold_model = None
    threshold_checkpoint = None
    if args.threshold_mode == "adaptive_nn":
        train_inputs, train_scores = gpu.sequence_inputs_and_scores(
            model, train_mats, args.seq_len, args.batch
        )
        validation_inputs, validation_scores = gpu.sequence_inputs_and_scores(
            model, validation_mats, args.seq_len, args.batch
        )
        threshold_model = train_adaptive_threshold(
            train_inputs,
            train_scores,
            hidden=args.threshold_hidden,
            learning_rate=args.threshold_lr,
            epochs=args.threshold_epochs,
            batch_size=args.threshold_batch,
            device=gpu.DEVICE,
        )
        validation_expected = predicted_normal_scores(
            threshold_model,
            validation_inputs,
            batch_size=args.threshold_batch,
            device=gpu.DEVICE,
        )
        threshold = adaptive_threshold(
            validation_scores - validation_expected, args.threshold_percentile
        )
    records = gpu.evaluate_seq_records(
        model,
        farms,
        args.window,
        args.seq_len,
        mean,
        std,
        use_fft,
        batch=args.batch,
        feature_columns_by_farm=columns,
        engineered_feature_columns_by_farm=engineered_columns,
        feature_profile=args.feature_profile,
        threshold_model=threshold_model,
        threshold_batch=args.threshold_batch,
    )
    records["is_alarm"] = flag(records["score"].fillna(-np.inf).to_numpy(), threshold).astype(bool)
    events = load_events(OUT)
    ev_map = {(row.farm, str(row.event_id)): row for row in events.itertuples()}
    scores, labels = gpu._records_to_auc_arrays(records, ev_map)
    auc = compute_auc(scores, labels)
    care = evaluate_care(records, events)
    care_paths = write_care_artifacts(care, RESULTS, f"{stamp}_transformer_optimized_care")
    checkpoint = CHECKPOINTS / f"{stamp}_transformer_optimized.pt"
    norm_path = CHECKPOINTS / f"{stamp}_transformer_optimized_norm.npz"
    torch.save(model.state_dict(), checkpoint)
    if threshold_model is not None:
        threshold_checkpoint = CHECKPOINTS / f"{stamp}_transformer_optimized_adaptive_threshold.pt"
        torch.save(threshold_model.state_dict(), threshold_checkpoint)
    np.savez(norm_path, mean=mean, std=std)
    result = {
        "model": "transformer_optimized",
        "farms": farms,
        "feature_set": args.feature_set,
        "feature_profile": args.feature_profile,
        "feature_columns_by_farm": columns,
        "engineered_feature_list": str(args.feature_list) if args.feature_list else None,
        "window": args.window,
        "normal_sampling": args.normal_sampling,
        "seq_len": args.seq_len,
        "use_fft": use_fft,
        "in_dim": int(train_x.shape[1]),
        "d_model": args.d_model,
        "latent": args.latent,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "architecture": args.architecture,
        "epochs": args.epochs,
        "batch": args.batch,
        "window_cache_dir": (
            None if args.no_window_cache or args.ram_window_cache else args.window_cache_dir
        ),
        "window_cache_dtype": args.window_cache_dtype,
        "ram_window_cache": args.ram_window_cache,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "grad_clip": args.grad_clip,
        "patience": args.patience,
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
        "care": care.metrics(),
        "care_artifacts": {key: str(path) for key, path in care_paths.items()},
        "loss_log": str(loss_path),
        "checkpoint": str(checkpoint),
        "normalization": str(norm_path),
        "elapsed_s": round(time.time() - start, 1),
    }
    result_path = RESULTS / f"{stamp}_transformer_optimized_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Transformer optimized: AUC={auc:.4f} CARE={care.care:.4f} "
        f"best_epoch={best_epoch} val={best_val_loss:.4f}",
        flush=True,
    )
    print(f"saved -> {result_path}", flush=True)


if __name__ == "__main__":
    main()
