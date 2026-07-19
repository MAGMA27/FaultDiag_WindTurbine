"""Focused LSTM-AE optimization runner under the frozen CARE protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import scripts.run_deep_ensemble as deep
import scripts.run_gpu_tune as gpu
from faultdiagnose.data import load_care, load_events
from faultdiagnose.evaluation.anomaly import compute_auc
from faultdiagnose.evaluation.care import evaluate_care, write_care_artifacts
from faultdiagnose.evaluation.ensemble import adaptive_threshold, flag
from faultdiagnose.models import LSTMAE

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
    """Train and evaluate one LSTM-AE configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default="B")
    parser.add_argument(
        "--feature-set", choices=["all", "avg_only", "paper_top"], default="avg_only"
    )
    parser.add_argument("--window", type=int, default=576)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--cap-train", type=int, default=60000)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--latent", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-loss", choices=["mse", "mae"], default="mse")
    parser.add_argument(
        "--architecture", choices=["symmetric", "paper", "direct"], default="symmetric"
    )
    parser.add_argument(
        "--decoder-init",
        choices=["zero", "state"],
        default="state",
        help="initialize decoder hidden/cell state from z, or use zero state",
    )
    parser.add_argument(
        "--decoder-positional",
        choices=["none", "learned"],
        default="learned",
        help="add learned time-position vectors to decoder inputs",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--scheduler", choices=["none", "warmup_cosine", "plateau"], default="warmup_cosine"
    )
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--device", default=gpu.DEVICE)
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
    use_fft = not args.no_fft
    farms = [farm.strip() for farm in args.farms.split(",")]
    configured_columns = deep.feature_columns_by_farm(args.feature_set)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    # Include microseconds so concurrent experiments cannot overwrite result artifacts.
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    start = time.time()

    print(
        f"LSTM optimize: farms={args.farms} feature_set={args.feature_set} "
        f"window={args.window} seq_len={args.seq_len} epochs={args.epochs} "
        f"scheduler={args.scheduler}",
        flush=True,
    )
    shared = gpu.collect_all(
        farms,
        args.window,
        args.cap_train,
        args.validation_fraction,
        use_fft,
        configured_columns,
    )
    train_x, train_mats, validation_x, validation_mats, mean, std = shared
    model = LSTMAE(
        train_x.shape[1],
        args.seq_len,
        latent=args.latent,
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        decoder_init=args.decoder_init,
        decoder_positional=args.decoder_positional,
        loss_type=args.train_loss,
        architecture=args.architecture,
    )
    loss_path = RESULTS / f"{stamp}_lstm_optimized_loss.txt"
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
            name="lstm_optimized",
            loss_f=loss_file,
        )

    validation_scores = gpu.validation_scores_seq(
        model, validation_mats, args.seq_len, args.batch
    )
    threshold = adaptive_threshold(validation_scores, args.threshold_percentile)
    records = gpu.evaluate_seq_records(
        model,
        farms,
        args.window,
        args.seq_len,
        mean,
        std,
        use_fft,
        batch=args.batch,
        feature_columns_by_farm=configured_columns,
    )
    records["is_alarm"] = flag(records["score"].fillna(-np.inf).to_numpy(), threshold).astype(bool)

    events = load_events(OUT)
    ev_map = {(row.farm, str(row.event_id)): row for row in events.itertuples()}
    scores, labels = gpu._records_to_auc_arrays(records, ev_map)
    auc = compute_auc(scores, labels)
    care = evaluate_care(records, events)
    care_paths = write_care_artifacts(care, RESULTS, f"{stamp}_lstm_optimized_care")
    checkpoint_path = CHECKPOINTS / f"{stamp}_lstm_optimized.pt"
    norm_path = CHECKPOINTS / f"{stamp}_lstm_optimized_norm.npz"
    torch.save(model.state_dict(), checkpoint_path)
    np.savez(norm_path, mean=mean, std=std)

    result = {
        "model": "lstm_optimized",
        "farms": farms,
        "feature_set": args.feature_set,
        "feature_columns_by_farm": configured_columns,
        "window": args.window,
        "seq_len": args.seq_len,
        "use_fft": use_fft,
        "in_dim": int(train_x.shape[1]),
        "hidden": args.hidden,
        "latent": args.latent,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "train_loss": args.train_loss,
        "architecture": args.architecture,
        "decoder_init": args.decoder_init,
        "decoder_positional": args.decoder_positional,
        "epochs": args.epochs,
        "batch": args.batch,
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
        "auc_roc": float(auc),
        "n_train": int(len(train_x)),
        "n_train_sequences": int(sum(max(0, len(mat) - args.seq_len + 1) for mat in train_mats)),
        "n_validation": int(len(validation_x)),
        "n_validation_sequences": int(
            sum(max(0, len(mat) - args.seq_len + 1) for mat in validation_mats)
        ),
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
    result_path = RESULTS / f"{stamp}_lstm_optimized_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"LSTM optimized: AUC={auc:.4f} CARE={care.care:.4f} "
        f"best_epoch={best_epoch} val={best_val_loss:.4f}",
        flush=True,
    )
    print(f"saved -> {result_path}", flush=True)


if __name__ == "__main__":
    main()
