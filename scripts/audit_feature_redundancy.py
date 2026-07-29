"""Build a training-only correlation audit and ``all_pruned`` feature list for CARE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_gpu_tune as gpu  # noqa: E402
from faultdiagnose.data import load_care, load_dataset, load_events, operating_mask  # noqa: E402
from faultdiagnose.features.audit import audit_feature_matrix  # noqa: E402
from faultdiagnose.features.engineer import engineer_features  # noqa: E402

OUT = load_care.OUT_DEFAULT
RESULTS = PROJECT_ROOT / "results"


def collect_normal_training_features(
    farm: str,
    window: int,
    cap_rows: int,
    use_fft: bool,
) -> pd.DataFrame:
    """Engineer all features from normal operating training rows only."""
    events = load_events(OUT)
    normal_ids = [
        str(row.event_id)
        for row in events.itertuples()
        if row.farm == farm and row.event_label == "normal"
    ]
    frames: list[pd.DataFrame] = []
    total_rows = 0
    for dataset_id in normal_ids:
        frame = load_dataset(OUT, farm, dataset_id)
        frame = frame[(frame["train_test"] == "train") & operating_mask(frame)]
        columns = gpu._feature_columns(frame, farm)
        for segment in gpu.contiguous_segments(frame):
            remaining = cap_rows - total_rows
            segment = segment.iloc[: remaining + window]
            engineered = engineer_features(
                segment,
                window=window,
                columns=columns,
                use_fft=use_fft,
                fillna="ffill",
                clip_sigma=5.0,
            ).dropna()
            if engineered.empty:
                continue
            frames.append(engineered.iloc[:remaining])
            total_rows += min(len(engineered), remaining)
            if total_rows >= cap_rows:
                return pd.concat(frames, ignore_index=True)
    if not frames:
        raise ValueError("No normal training features were collected")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """Write feature statistics, correlation clusters, and selected list."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farm", default="B")
    parser.add_argument("--window", type=int, default=576)
    parser.add_argument("--cap-rows", type=int, default=20000)
    parser.add_argument("--corr-threshold", type=float, default=0.995)
    parser.add_argument("--variance-epsilon", type=float, default=1e-10)
    parser.add_argument(
        "--drop-near-constant",
        action="store_true",
        help="Experimental: remove normal-period near-constant features (off by default).",
    )
    parser.add_argument("--no-fft", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "feature_audit")
    args = parser.parse_args()
    if not OUT.exists():
        raise SystemExit("processed data missing; run convert_care_to_parquet first")

    features = collect_normal_training_features(
        args.farm, args.window, args.cap_rows, use_fft=not args.no_fft
    )
    stats, clusters, selected = audit_feature_matrix(
        features,
        args.corr_threshold,
        args.variance_epsilon,
        drop_near_constant=args.drop_near_constant,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"farm_{args.farm}_window{args.window}_corr{args.corr_threshold:.4f}"
    stats_path = args.output_dir / f"{stem}_feature_stats.csv"
    clusters_path = args.output_dir / f"{stem}_correlation_clusters.csv"
    selected_path = args.output_dir / f"{stem}_all_pruned.json"
    summary_path = args.output_dir / f"{stem}_summary.json"
    stats.to_csv(stats_path, index=False)
    clusters.to_csv(clusters_path, index=False)
    selected_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    summary = {
        "farm": args.farm,
        "window": args.window,
        "rows": len(features),
        "input_features": features.shape[1],
        "selected_features": len(selected),
        "correlation_threshold": args.corr_threshold,
        "variance_epsilon": args.variance_epsilon,
        "drop_near_constant": args.drop_near_constant,
        "use_fft": not args.no_fft,
        "paths": {
            "feature_stats": str(stats_path),
            "correlation_clusters": str(clusters_path),
            "selected_features": str(selected_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Feature audit: rows={len(features)} input={features.shape[1]} "
        f"selected={len(selected)} threshold={args.corr_threshold}"
    )
    print(f"saved -> {summary_path}")


if __name__ == "__main__":
    main()
