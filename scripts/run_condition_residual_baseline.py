"""Run a LightGBM condition-residual baseline under the frozen CARE protocol."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from faultdiagnose.data import list_datasets, load_care, load_dataset, load_events, operating_mask
from faultdiagnose.evaluation.anomaly import compute_auc
from faultdiagnose.evaluation.care import evaluate_care, write_care_artifacts
from faultdiagnose.evaluation.ensemble import adaptive_threshold, flag

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"
BREAK_MINUTES = 60.0


@dataclass(frozen=True)
class FarmResidualConfig:
    """Semantic sensor configuration for one wind farm."""

    inputs: tuple[str, ...]
    targets: tuple[str, ...]


FARM_CONFIGS = {
    "B": FarmResidualConfig(
        inputs=(
            "sensor_4",  # absolute wind direction
            "sensor_8",  # outside temperature
            "sensor_10",  # pitch angle
            "sensor_12",  # estimated torque
            "sensor_20",  # gearbox rotational speed
            "sensor_21",  # nacelle direction
            "sensor_25",  # rotor speed
            "power_58",  # available power
            "power_62",  # active power
            "wind_speed_59",
            "wind_speed_60",
            "wind_speed_61",
        ),
        targets=(
            "sensor_22",  # hub temperature; closest available nacelle-area temperature
            "sensor_31",  # internal transformer consumption temperature
            "sensor_32",  # generator bearing temperature 1
            "sensor_33",  # generator bearing temperature 2
            "sensor_34",  # gearbox bearing temperature 1
            "sensor_35",  # gearbox bearing temperature 2
            "sensor_36",  # gearbox bearing temperature 3
            "sensor_37",  # gearbox bearing temperature 4
            "sensor_38",  # gearbox oil inlet temperature
            "sensor_39",  # gearbox oil tank temperature
            "sensor_47",  # transformer cell temperature
            "sensor_51",  # rotor bearing temperature 1
            "sensor_52",  # rotor bearing temperature 2
            "sensor_54",  # drive train vibration axis Z
            "sensor_55",  # tower vibration axis X
            "sensor_56",  # tower vibration axis Y
        ),
    )
}


def average_columns(sensor_names: tuple[str, ...]) -> list[str]:
    """Return the ten-minute average SCADA column for each configured sensor."""
    return [f"{sensor}_avg" for sensor in sensor_names]


def contiguous_segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a frame at long time gaps so validation stays chronological."""
    timestamps = pd.to_datetime(df["time_stamp"])
    gap_minutes = timestamps.diff().dt.total_seconds() / 60.0
    return [segment for _, segment in df.groupby((gap_minutes > BREAK_MINUTES).cumsum())]


def split_tail(segment: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the chronological tail of a continuous segment for validation."""
    validation_size = max(1, int(np.ceil(len(segment) * fraction)))
    if validation_size >= len(segment):
        return segment.iloc[0:0], segment.iloc[0:0]
    return segment.iloc[:-validation_size], segment.iloc[-validation_size:]


def replace_long_zero_runs(
    frame: pd.DataFrame, input_columns: list[str], minimum_run: int = 6
) -> pd.DataFrame:
    """Treat only long Farm B/C zero runs as missing; short physical zero values remain."""
    if minimum_run <= 0:
        raise ValueError("minimum_run must be positive")
    result = frame.copy()
    for column in input_columns:
        zero = result[column].eq(0) & result[column].notna()
        run_id = zero.ne(zero.shift(fill_value=False)).cumsum()
        long_zero = zero & zero.groupby(run_id).transform("sum").ge(minimum_run)
        result.loc[long_zero, column] = np.nan
    return result


def require_columns(frame: pd.DataFrame, columns: list[str], farm: str) -> None:
    """Reject an invalid farm configuration instead of silently selecting other sensors."""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Farm {farm} is missing configured SCADA columns: {missing}")


def collect_normal_frames(
    farm: str,
    input_columns: list[str],
    target_columns: list[str],
    cap: int,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect normal train/validation rows without reading any prediction partition."""
    events = load_events(OUT)
    normal_ids = events.loc[
        (events["farm"] == farm) & (events["event_label"] == "normal"), "event_id"
    ].astype(str)
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    total = 0
    columns = ["time_stamp", *input_columns, *target_columns]
    for dataset_id in normal_ids:
        raw = load_dataset(OUT, farm, dataset_id)
        normal = raw[(raw["train_test"] == "train") & operating_mask(raw)]
        require_columns(normal, columns, farm)
        for segment in contiguous_segments(normal.loc[:, columns]):
            remaining = cap - total
            if remaining <= 0:
                break
            selected = segment.iloc[:remaining]
            train, validation = split_tail(selected, validation_fraction)
            if not train.empty and not validation.empty:
                train_parts.append(train)
                validation_parts.append(validation)
                total += len(selected)
        if total >= cap:
            break
    if not train_parts or not validation_parts:
        raise ValueError(
            "normal training data did not contain usable train and validation segments"
        )
    return pd.concat(train_parts, ignore_index=True), pd.concat(validation_parts, ignore_index=True)


def fit_target_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    input_columns: list[str],
    target_column: str,
    seed: int,
) -> tuple[lgb.LGBMRegressor, float]:
    """Fit one normal-behaviour model and return its validation residual scale."""
    clean_train = replace_long_zero_runs(train, input_columns)
    clean_validation = replace_long_zero_runs(validation, input_columns)
    train_valid_target = clean_train[target_column].notna() & clean_train[target_column].ne(0)
    validation_valid_target = clean_validation[target_column].notna() & clean_validation[
        target_column
    ].ne(0)
    if not train_valid_target.any() or not validation_valid_target.any():
        raise ValueError(f"target {target_column} has no non-zero normal samples")
    model = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        clean_train.loc[train_valid_target, input_columns],
        clean_train.loc[train_valid_target, target_column],
    )
    residuals = np.abs(
        clean_validation.loc[validation_valid_target, target_column].to_numpy(dtype=float)
        - model.predict(clean_validation.loc[validation_valid_target, input_columns])
    )
    scale = float(np.std(residuals))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"target {target_column} has invalid validation residual scale")
    return model, scale


def score_frame(
    frame: pd.DataFrame,
    models: dict[str, lgb.LGBMRegressor],
    scales: dict[str, float],
    input_columns: list[str],
) -> np.ndarray:
    """Return the maximum standardized residual across configured targets."""
    clean = replace_long_zero_runs(frame, input_columns)
    scores = np.full(len(clean), np.nan, dtype=float)
    target_scores: list[np.ndarray] = []
    for target_column, model in models.items():
        target = clean[target_column]
        valid = target.notna() & target.ne(0)
        values = np.full(len(clean), np.nan, dtype=float)
        if valid.any():
            predicted = model.predict(clean.loc[valid, input_columns])
            residuals = np.abs(target.loc[valid].to_numpy(dtype=float) - predicted)
            values[valid.to_numpy()] = residuals / scales[target_column]
        target_scores.append(values)
    if target_scores:
        stacked = np.vstack(target_scores)
        present = np.isfinite(stacked).any(axis=0)
        scores[present] = np.nanmax(stacked[:, present], axis=0)
    return scores


def score_prediction_records(
    farm: str,
    dataset_ids: list[str],
    models: dict[str, lgb.LGBMRegressor],
    scales: dict[str, float],
    input_columns: list[str],
    target_columns: list[str],
) -> pd.DataFrame:
    """Score operating prediction rows while preserving the full CARE status timeline."""
    records: list[pd.DataFrame] = []
    required = [*input_columns, *target_columns]
    for dataset_id in dataset_ids:
        print(f"  scoring {farm}/{dataset_id}", flush=True)
        raw = load_dataset(OUT, farm, dataset_id)
        prediction = raw[raw["train_test"] == "prediction"]
        require_columns(prediction, required, farm)
        result = prediction[["time_stamp", "status_type_id", "train_test"]].copy()
        result.insert(0, "dataset_id", str(dataset_id))
        result.insert(0, "farm", farm)
        result["score"] = np.nan
        operating = prediction[operating_mask(prediction)]
        result.loc[operating.index, "score"] = score_frame(operating, models, scales, input_columns)
        records.append(result)
    return pd.concat(records, ignore_index=True)


def auc_arrays(records: pd.DataFrame, events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build optional point-level AUC arrays from scored operating prediction rows."""
    scored = records[np.isfinite(records["score"])].copy()
    labels = np.zeros(len(scored), dtype=int)
    event_map = {(row.farm, str(row.event_id)): row for row in events.itertuples()}
    for (farm, dataset_id), positions in scored.groupby(["farm", "dataset_id"]).groups.items():
        event = event_map[(farm, str(dataset_id))]
        if event.event_label == "anomaly":
            timestamps = pd.to_datetime(scored.loc[positions, "time_stamp"])
            labels[scored.index.get_indexer(positions)] = (
                (timestamps >= pd.Timestamp(event.event_start))
                & (timestamps <= pd.Timestamp(event.event_end))
            ).astype(int)
    return scored["score"].to_numpy(dtype=float), labels


def main() -> None:
    """Fit Farm B condition models, calibrate normal-only threshold, and evaluate CARE."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farm", choices=sorted(FARM_CONFIGS), default="B")
    parser.add_argument("--datasets", help="comma-separated prediction dataset IDs; debug only")
    parser.add_argument("--cap-train", type=int, default=60000)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if not OUT.exists():
        raise SystemExit("processed data missing; run convert_care_to_parquet first")
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be between 0 and 1")
    if not 95 <= args.threshold_percentile <= 99:
        raise SystemExit("--threshold-percentile must be between 95 and 99")
    config = FARM_CONFIGS[args.farm]
    input_columns = average_columns(config.inputs)
    target_columns = average_columns(config.targets)
    train, validation = collect_normal_frames(
        args.farm, input_columns, target_columns, args.cap_train, args.validation_fraction
    )
    print(f"Fitting LightGBM normal-behaviour models on {len(train)} rows...")
    models: dict[str, lgb.LGBMRegressor] = {}
    scales: dict[str, float] = {}
    for target in target_columns:
        models[target], scales[target] = fit_target_model(
            train, validation, input_columns, target, args.seed
        )
    validation_scores = score_frame(validation, models, scales, input_columns)
    threshold = adaptive_threshold(
        validation_scores[np.isfinite(validation_scores)], args.threshold_percentile
    )
    dataset_ids = list_datasets(OUT, args.farm)
    if args.datasets:
        requested = {dataset_id.strip() for dataset_id in args.datasets.split(",")}
        dataset_ids = [dataset_id for dataset_id in dataset_ids if dataset_id in requested]
    if not dataset_ids:
        raise SystemExit("no prediction datasets selected")
    events = load_events(OUT)
    events = events[
        (events["farm"] == args.farm) & (events["event_id"].astype(str).isin(dataset_ids))
    ]
    print("Scoring prediction timelines...")
    records = score_prediction_records(
        args.farm, dataset_ids, models, scales, input_columns, target_columns
    )
    records["is_alarm"] = flag(records["score"].fillna(-np.inf), threshold).astype(bool)
    scores, labels = auc_arrays(records, events)
    auc = float(compute_auc(scores, labels)) if len(np.unique(labels)) == 2 else None
    care = evaluate_care(records, events)
    selected_columns = [*input_columns, *target_columns]
    zero_audit = pd.DataFrame(
        {
            "column": selected_columns,
            "role": ["input"] * len(input_columns) + ["target"] * len(target_columns),
            "normal_train_zero_fraction": [
                float(train[column].eq(0).mean()) for column in selected_columns
            ],
            "normal_validation_zero_fraction": [
                float(validation[column].eq(0).mean()) for column in selected_columns
            ],
        }
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    artifact_paths = write_care_artifacts(care, RESULTS, f"{stamp}_condition_residual_care")
    zero_path = RESULTS / f"{stamp}_condition_residual_zero_audit.csv"
    zero_audit.to_csv(zero_path, index=False)
    result = {
        "model": "lightgbm_condition_residual",
        "farm": args.farm,
        "datasets": dataset_ids,
        "inputs": input_columns,
        "targets": target_columns,
        "cap_train": args.cap_train,
        "n_train": len(train),
        "n_validation": len(validation),
        "validation_residual_scales": scales,
        "threshold": float(threshold),
        "threshold_percentile": args.threshold_percentile,
        "threshold_source": "normal_validation_standardized_residuals",
        "auc_roc": auc,
        "care": care.metrics(),
        "care_artifacts": {key: str(path) for key, path in artifact_paths.items()},
        "zero_value_audit": str(zero_path),
    }
    result_path = RESULTS / f"{stamp}_condition_residual_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"CARE={care.care:.4f} AUC={auc if auc is not None else 'not-defined'} "
        f"threshold={threshold:.6f}\nsaved -> {result_path}"
    )


if __name__ == "__main__":
    main()
