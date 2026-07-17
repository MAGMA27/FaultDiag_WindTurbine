"""Audit CARE anomaly events that remain pointwise evaluable after status filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from faultdiagnose.data import list_datasets, load_care, load_dataset, load_events
from faultdiagnose.evaluation.care import build_event_eligibility_report

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"


def load_prediction_statuses(farms: list[str]) -> pd.DataFrame:
    """Load only fields required to audit prediction-time event eligibility."""
    frames: list[pd.DataFrame] = []
    for farm in farms:
        for dataset_id in list_datasets(OUT, farm):
            dataset = load_dataset(OUT, farm, dataset_id)
            frames.append(
                dataset[["farm", "dataset_id", "time_stamp", "status_type_id", "train_test"]]
            )
    return pd.concat(frames, ignore_index=True)


def summarize(report: pd.DataFrame) -> pd.DataFrame:
    """Summarize pointwise eligibility by farm."""
    summary = report.groupby("farm", as_index=False).agg(
        anomaly_events=("dataset_id", "size"),
        pointwise_eligible_events=("pointwise_eligible", "sum"),
        event_window_prediction_rows=("event_window_prediction_rows", "sum"),
        operating_event_rows=("operating_event_rows", "sum"),
    )
    summary["eligible_event_fraction"] = (
        summary["pointwise_eligible_events"] / summary["anomaly_events"]
    )
    summary["operating_point_fraction"] = (
        summary["operating_event_rows"] / summary["event_window_prediction_rows"]
    )
    return summary


def main() -> None:
    """Write per-event and per-farm eligibility reports for selected CARE farms."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default="A,B,C")
    args = parser.parse_args()
    if not OUT.exists():
        raise SystemExit("processed data missing; run convert_care_to_parquet first")

    farms = [farm.strip() for farm in args.farms.split(",")]
    events = load_events(OUT)
    events = events[events["farm"].isin(farms)]
    report = build_event_eligibility_report(load_prediction_statuses(farms), events)
    summary = summarize(report)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    report_path = RESULTS / f"{stamp}_event_eligibility.csv"
    summary_path = RESULTS / f"{stamp}_event_eligibility_summary.json"
    report.to_csv(report_path, index=False)
    summary_path.write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"saved -> {report_path}\nsaved -> {summary_path}")


if __name__ == "__main__":
    main()
