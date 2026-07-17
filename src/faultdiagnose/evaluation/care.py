"""CARE benchmark scoring and event-level operational reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

NORMAL_STATUS_IDS = frozenset({0, 2})
FBETA_BETA = 0.5
CRITICALITY_THRESHOLD = 72


@dataclass(frozen=True)
class CareEvaluation:
    """Aggregate CARE metrics together with audit-friendly event tables."""

    care: float
    coverage: float
    accuracy: float
    reliability: float
    earliness: float
    event_report: pd.DataFrame
    event_eligibility_report: pd.DataFrame
    monthly_false_alarms: pd.DataFrame

    def metrics(self) -> dict[str, float]:
        """Return scalar metrics suitable for JSON or CSV experiment summaries."""
        return {
            "care": self.care,
            "coverage_f0.5": self.coverage,
            "accuracy": self.accuracy,
            "reliability_event_f0.5": self.reliability,
            "earliness": self.earliness,
        }


def write_care_artifacts(
    evaluation: CareEvaluation, output_dir: Path, stem: str = "care"
) -> dict[str, Path]:
    """Write CARE scalar metrics and its operational reports to CSV/JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_dir / f"{stem}_metrics.json",
        "events": output_dir / f"{stem}_events.csv",
        "event_eligibility": output_dir / f"{stem}_event_eligibility.csv",
        "monthly_false_alarms": output_dir / f"{stem}_monthly_false_alarms.csv",
    }
    paths["metrics"].write_text(
        json.dumps(evaluation.metrics(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    evaluation.event_report.to_csv(paths["events"], index=False)
    evaluation.event_eligibility_report.to_csv(paths["event_eligibility"], index=False)
    evaluation.monthly_false_alarms.to_csv(paths["monthly_false_alarms"], index=False)
    return paths


def criticality(
    alarms: Iterable[bool] | np.ndarray, normal_status: Iterable[bool] | np.ndarray
) -> np.ndarray:
    """Compute CARE's counter-like criticality sequence."""
    alarm_array = np.asarray(alarms, dtype=bool)
    normal_array = np.asarray(normal_status, dtype=bool)
    if alarm_array.shape != normal_array.shape:
        raise ValueError("alarms and normal_status must have identical shape")

    values = np.zeros(len(alarm_array), dtype=np.int64)
    current = 0
    for index, (alarm, is_normal) in enumerate(zip(alarm_array, normal_array, strict=True)):
        if is_normal:
            current = current + 1 if alarm else max(current - 1, 0)
        values[index] = current
    return values


def f_beta(labels: Iterable[bool] | np.ndarray, alarms: Iterable[bool] | np.ndarray) -> float:
    """Return the precision-weighted F_0.5 score used by CARE."""
    label_array = np.asarray(labels, dtype=bool)
    alarm_array = np.asarray(alarms, dtype=bool)
    if label_array.shape != alarm_array.shape:
        raise ValueError("labels and alarms must have identical shape")

    true_positive = int(np.count_nonzero(label_array & alarm_array))
    false_negative = int(np.count_nonzero(label_array & ~alarm_array))
    false_positive = int(np.count_nonzero(~label_array & alarm_array))
    beta_squared = FBETA_BETA**2
    denominator = (
        (1 + beta_squared) * true_positive + beta_squared * false_negative + false_positive
    )
    return 0.0 if denominator == 0 else (1 + beta_squared) * true_positive / denominator


def earliness_weights(size: int) -> np.ndarray:
    """Return CARE's event-relative weights: first half 1, then linear to 0."""
    if size < 0:
        raise ValueError("size must be non-negative")
    if size == 0:
        return np.array([], dtype=float)
    if size == 1:
        return np.array([1.0])
    position = np.linspace(0.0, 1.0, num=size)
    return np.where(position <= 0.5, 1.0, 2.0 * (1.0 - position))


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _prediction_partition(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep the held-out partition and reject ambiguous non-prediction input."""
    if "train_test" not in predictions.columns:
        raise ValueError(
            "predictions must include train_test to enforce prediction-only CARE scoring"
        )
    partition = predictions[predictions["train_test"] == "prediction"].copy()
    if partition.empty:
        raise ValueError("predictions contains no train_test == 'prediction' rows")
    return partition


def build_event_eligibility_report(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    normal_status_ids: frozenset[int] = NORMAL_STATUS_IDS,
) -> pd.DataFrame:
    """Audit anomaly events that retain point-level labels after CARE status filtering.

    A pointwise-positive timestamp must fall within ``event_start`` through
    ``event_end`` and have a CARE-normal status (0 or 2). Events with zero
    eligible timestamps remain valid for event-level reliability, but cannot
    support pointwise AUC, Coverage, or Earliness.
    """
    _require_columns(
        predictions,
        {"farm", "dataset_id", "time_stamp", "status_type_id", "train_test"},
        "predictions",
    )
    _require_columns(
        events, {"farm", "event_id", "event_label", "event_start", "event_end"}, "events"
    )
    partition = _prediction_partition(predictions)
    event_frame = events.copy()
    event_frame["event_id"] = event_frame["event_id"].astype(str)
    rows: list[dict[str, object]] = []
    for event in event_frame[event_frame["event_label"] == "anomaly"].itertuples():
        frame = partition[
            (partition["farm"] == event.farm)
            & (partition["dataset_id"].astype(str) == str(event.event_id))
        ]
        timestamps = pd.to_datetime(frame["time_stamp"])
        in_event = (timestamps >= pd.Timestamp(event.event_start)) & (
            timestamps <= pd.Timestamp(event.event_end)
        )
        eligible = in_event & frame["status_type_id"].isin(normal_status_ids)
        event_rows = int(in_event.sum())
        eligible_rows = int(eligible.sum())
        rows.append(
            {
                "farm": event.farm,
                "dataset_id": str(event.event_id),
                "event_description": getattr(event, "event_description", None),
                "event_start": pd.Timestamp(event.event_start),
                "event_end": pd.Timestamp(event.event_end),
                "event_window_prediction_rows": event_rows,
                "operating_event_rows": eligible_rows,
                "operating_event_fraction": eligible_rows / event_rows if event_rows else 0.0,
                "pointwise_eligible": eligible_rows > 0,
            }
        )
    columns = [
        "farm",
        "dataset_id",
        "event_description",
        "event_start",
        "event_end",
        "event_window_prediction_rows",
        "operating_event_rows",
        "operating_event_fraction",
        "pointwise_eligible",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["farm", "dataset_id"])
        .reset_index(drop=True)
    )


def _event_labels(frame: pd.DataFrame, event: pd.Series, normal_status: np.ndarray) -> np.ndarray:
    if event.event_label != "anomaly":
        return np.zeros(len(frame), dtype=bool)
    start = pd.Timestamp(event.event_start)
    end = pd.Timestamp(event.event_end)
    timestamps = pd.to_datetime(frame["time_stamp"])
    return ((timestamps >= start) & (timestamps <= end) & normal_status).to_numpy(dtype=bool)


def _first_qualified_alarm_time(
    frame: pd.DataFrame, values: np.ndarray, threshold: int
) -> pd.Timestamp | pd.NaT:
    qualifying = np.flatnonzero(values >= threshold)
    if len(qualifying) == 0:
        return pd.NaT
    return pd.Timestamp(frame.iloc[qualifying[0]]["time_stamp"])


def _monthly_false_alarm_report(normal_predictions: list[pd.DataFrame]) -> pd.DataFrame:
    """Count contiguous false-alarm episodes on normal datasets by calendar month."""
    episodes: list[pd.DataFrame] = []
    for frame in normal_predictions:
        active = frame["is_alarm"].to_numpy(dtype=bool) & frame["is_normal_status"].to_numpy(
            dtype=bool
        )
        starts = active & ~np.concatenate(([False], active[:-1]))
        if not starts.any():
            continue
        rows = frame.loc[starts, ["time_stamp"]].copy()
        rows["month"] = pd.to_datetime(rows["time_stamp"]).dt.to_period("M").astype(str)
        episodes.append(rows)

    if not episodes:
        return pd.DataFrame(columns=["month", "false_alarm_episodes", "false_alarm_points"])

    episode_frame = pd.concat(episodes, ignore_index=True)
    all_normal = pd.concat(normal_predictions, ignore_index=True)
    points = all_normal.loc[
        all_normal["is_alarm"] & all_normal["is_normal_status"], ["time_stamp"]
    ].copy()
    points["month"] = pd.to_datetime(points["time_stamp"]).dt.to_period("M").astype(str)
    report = episode_frame.groupby("month").size().rename("false_alarm_episodes").to_frame()
    report["false_alarm_points"] = points.groupby("month").size()
    return (
        report.fillna(0)
        .astype({"false_alarm_episodes": int, "false_alarm_points": int})
        .reset_index()
    )


def evaluate_care(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    criticality_threshold: int = CRITICALITY_THRESHOLD,
    normal_status_ids: frozenset[int] = NORMAL_STATUS_IDS,
) -> CareEvaluation:
    """Evaluate binary prediction alarms with the CARE benchmark protocol.

    ``predictions`` must contain one row per timestamp with ``farm``,
    ``dataset_id``, ``time_stamp``, ``status_type_id``, ``train_test`` and
    ``is_alarm``. Rows in the training partition are excluded internally.
    """
    if criticality_threshold <= 0:
        raise ValueError("criticality_threshold must be positive")
    _require_columns(
        predictions,
        {"farm", "dataset_id", "time_stamp", "status_type_id", "train_test", "is_alarm"},
        "predictions",
    )
    _require_columns(
        events, {"farm", "event_id", "event_label", "event_start", "event_end"}, "events"
    )

    prediction_frame = _prediction_partition(predictions)
    event_frame = events.copy()
    event_frame["event_id"] = event_frame["event_id"].astype(str)
    eligibility = build_event_eligibility_report(
        prediction_frame, event_frame, normal_status_ids=normal_status_ids
    )
    grouped = prediction_frame.groupby(["farm", "dataset_id"], sort=True)

    coverage_scores: list[float] = []
    earliness_scores: list[float] = []
    accuracy_scores: list[float] = []
    event_labels: list[bool] = []
    event_alarms: list[bool] = []
    event_rows: list[dict[str, object]] = []
    normal_predictions: list[pd.DataFrame] = []

    for (farm, dataset_id), raw_frame in grouped:
        matches = event_frame[
            (event_frame["farm"] == farm) & (event_frame["event_id"] == str(dataset_id))
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one event row for farm={farm!r}, dataset_id={dataset_id!r}")
        event = matches.iloc[0]
        frame = raw_frame.sort_values("time_stamp").copy()
        frame["is_alarm"] = frame["is_alarm"].astype(bool)
        frame["is_normal_status"] = frame["status_type_id"].isin(normal_status_ids)
        normal_status = frame["is_normal_status"].to_numpy(dtype=bool)
        alarms = frame["is_alarm"].to_numpy(dtype=bool)
        labels = _event_labels(frame, event, normal_status)
        values = criticality(alarms, normal_status)
        detected = bool(values.max(initial=0) >= criticality_threshold)

        event_labels.append(event.event_label == "anomaly")
        event_alarms.append(detected)
        if event.event_label == "normal":
            normal_predictions.append(frame)
            normal_alarms = alarms[normal_status]
            accuracy_scores.append(float((~normal_alarms).mean()) if len(normal_alarms) else 0.0)
            continue

        coverage_scores.append(f_beta(labels, alarms & normal_status))
        weights = earliness_weights(int(labels.sum()))
        earliness_scores.append(
            float(np.average(alarms[labels], weights=weights)) if len(weights) else 0.0
        )
        window_indices = np.flatnonzero(labels & alarms)
        first_alarm = (
            pd.Timestamp(frame.iloc[window_indices[0]]["time_stamp"])
            if len(window_indices)
            else pd.NaT
        )
        qualified_alarm = _first_qualified_alarm_time(frame, values, criticality_threshold)
        end = pd.Timestamp(event.event_end)
        lead_time = (
            (end - first_alarm).total_seconds() / 3600 if not pd.isna(first_alarm) else np.nan
        )
        event_rows.append(
            {
                "farm": farm,
                "dataset_id": str(dataset_id),
                "event_description": event.get("event_description", None),
                "event_start": pd.Timestamp(event.event_start),
                "event_end": end,
                "first_alarm_time": first_alarm,
                "criticality_alarm_time": qualified_alarm,
                "max_criticality": int(values.max(initial=0)),
                "event_detected": detected,
                "lead_time_hours": lead_time,
                "covered_24h": bool(not pd.isna(first_alarm) and lead_time >= 24),
                "covered_48h": bool(not pd.isna(first_alarm) and lead_time >= 48),
            }
        )

    coverage = float(np.mean(coverage_scores)) if coverage_scores else 0.0
    earliness = float(np.mean(earliness_scores)) if earliness_scores else 0.0
    accuracy = float(np.mean(accuracy_scores)) if accuracy_scores else 0.0
    reliability = f_beta(np.array(event_labels), np.array(event_alarms))
    any_alarm = bool(prediction_frame["is_alarm"].astype(bool).any())
    care = (
        0.0
        if not any_alarm
        else accuracy
        if accuracy < 0.5
        else (coverage + earliness + reliability + 2 * accuracy) / 5
    )

    event_report = pd.DataFrame(event_rows)
    if not event_report.empty:
        event_report = event_report.sort_values(["farm", "dataset_id"]).reset_index(drop=True)
    return CareEvaluation(
        care=care,
        coverage=coverage,
        accuracy=accuracy,
        reliability=reliability,
        earliness=earliness,
        event_report=event_report,
        event_eligibility_report=eligibility,
        monthly_false_alarms=_monthly_false_alarm_report(normal_predictions),
    )
