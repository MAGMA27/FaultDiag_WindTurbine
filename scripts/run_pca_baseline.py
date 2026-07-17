"""Run a PCA reconstruction-error baseline under the frozen CARE protocol."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from faultdiagnose.data import list_datasets, load_care, load_dataset, load_events, operating_mask
from faultdiagnose.evaluation.anomaly import compute_auc
from faultdiagnose.evaluation.care import evaluate_care, write_care_artifacts
from faultdiagnose.evaluation.ensemble import adaptive_threshold, flag
from faultdiagnose.features import (
    apply_standardizer,
    engineer_features,
    fit_standardizer,
    select_feature_columns,
)

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"
BREAK_MINUTES = 60.0


def contiguous_segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split time series before feature extraction so windows cannot cross long gaps."""
    timestamps = pd.to_datetime(df["time_stamp"])
    gap_minutes = timestamps.diff().dt.total_seconds() / 60.0
    segment_ids = (gap_minutes > BREAK_MINUTES).cumsum()
    return [segment for _, segment in df.groupby(segment_ids)]


def prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the held-out CARE prediction partition."""
    if "train_test" not in df.columns:
        raise ValueError("CARE datasets must provide train_test for prediction-only evaluation")
    return df[df["train_test"] == "prediction"]


def split_train_validation(
    matrices: list[np.ndarray], fraction: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Reserve the chronological tail of every normal segment for validation."""
    train_matrices: list[np.ndarray] = []
    validation_matrices: list[np.ndarray] = []
    for matrix in matrices:
        validation_size = max(1, int(np.ceil(len(matrix) * fraction)))
        if validation_size >= len(matrix):
            continue
        split_at = len(matrix) - validation_size
        train_matrices.append(matrix[:split_at])
        validation_matrices.append(matrix[split_at:])
    if not train_matrices or not validation_matrices:
        raise ValueError(
            "normal training data did not contain usable train and validation segments"
        )
    return train_matrices, validation_matrices


def split_chunked_matrices(
    chunks: Iterator[np.ndarray], row_limit: int, validation_fraction: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split one continuous segment at one tail boundary while streaming its chunks."""
    if row_limit <= 0:
        return [], []
    validation_size = max(1, int(np.ceil(row_limit * validation_fraction)))
    if validation_size >= row_limit:
        train_matrices: list[np.ndarray] = []
        seen = 0
        for chunk in chunks:
            remaining = row_limit - seen
            if remaining <= 0:
                break
            current = chunk[:remaining]
            train_matrices.append(current)
            seen += len(current)
        return train_matrices, []

    split_at = row_limit - validation_size
    train_matrices: list[np.ndarray] = []
    validation_matrices: list[np.ndarray] = []
    seen = 0
    for chunk in chunks:
        remaining = row_limit - seen
        if remaining <= 0:
            break
        current = chunk[:remaining]
        train_end = max(0, min(len(current), split_at - seen))
        if train_end:
            train_matrices.append(current[:train_end])
        if train_end < len(current):
            validation_matrices.append(current[train_end:])
        seen += len(current)
    return train_matrices, validation_matrices


def feature_chunks(
    segment: pd.DataFrame,
    *,
    window: int,
    columns: list[str],
    use_fft: bool,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    """Engineer overlapping chunks so wide Farm B/C frames do not exhaust memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    lookback = max(window - 1, 2)
    for start in range(0, len(segment), chunk_size):
        end = min(start + chunk_size, len(segment))
        context_start = max(0, start - lookback)
        features = engineer_features(
            segment.iloc[context_start:end],
            window=window,
            columns=columns,
            use_fft=use_fft,
            fillna="ffill",
            clip_sigma=5.0,
        ).dropna()
        retained_index = segment.index[start:end]
        features = features.loc[features.index.intersection(retained_index)]
        if not features.empty:
            yield features


def collect_normal_features(
    farms: list[str],
    window: int,
    cap: int,
    validation_fraction: float,
    use_fft: bool,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build standardized normal training/validation matrices without using prediction rows."""
    events = load_events(OUT)
    train_matrices: list[np.ndarray] = []
    validation_matrices: list[np.ndarray] = []
    total = 0
    for farm in farms:
        normal_ids = events.loc[
            (events["farm"] == farm) & (events["event_label"] == "normal"), "event_id"
        ].astype(str)
        for dataset_id in normal_ids:
            df = load_dataset(OUT, farm, dataset_id)
            train_df = df[df["train_test"] == "train"]
            operating_df = train_df[operating_mask(train_df)]
            columns = select_feature_columns(operating_df)
            for segment in contiguous_segments(operating_df):
                remaining = cap - total
                if remaining <= 0:
                    break
                feature_count = sum(
                    len(features)
                    for features in feature_chunks(
                        segment,
                        window=window,
                        columns=columns,
                        use_fft=use_fft,
                        chunk_size=chunk_size,
                    )
                )
                selected_rows = min(feature_count, remaining)
                if selected_rows:
                    chunks = (
                        features.to_numpy(dtype=np.float32)
                        for features in feature_chunks(
                            segment,
                            window=window,
                            columns=columns,
                            use_fft=use_fft,
                            chunk_size=chunk_size,
                        )
                    )
                    segment_train, segment_validation = split_chunked_matrices(
                        chunks, selected_rows, validation_fraction
                    )
                    train_matrices.extend(segment_train)
                    validation_matrices.extend(segment_validation)
                    total += selected_rows
                if total >= cap:
                    break
            if total >= cap:
                break
        if total >= cap:
            break

    if not train_matrices or not validation_matrices:
        raise ValueError(
            "normal training data did not contain usable train and validation segments"
        )
    train_raw = np.concatenate(train_matrices)
    mean, std = fit_standardizer(train_raw)
    train = apply_standardizer(train_raw, mean, std).astype(np.float32)
    validation_raw = np.concatenate(validation_matrices)
    validation = apply_standardizer(validation_raw, mean, std).astype(np.float32)
    return train, validation, mean, std


def reconstruction_error(model: PCA, matrix: np.ndarray) -> np.ndarray:
    """Return per-row mean squared PCA reconstruction error."""
    reconstruction = model.inverse_transform(model.transform(matrix))
    return np.mean(np.square(matrix - reconstruction), axis=1)


def score_one_prediction_dataset(
    model: PCA,
    farm: str,
    dataset_id: str,
    window: int,
    mean: np.ndarray,
    std: np.ndarray,
    use_fft: bool,
    chunk_size: int,
) -> pd.DataFrame:
    """Score one dataset while retaining its complete prediction status timeline."""
    print(f"  scoring {farm}/{dataset_id}", flush=True)
    raw_df = load_dataset(OUT, farm, dataset_id)
    records = prediction_rows(raw_df)[["time_stamp", "status_type_id", "train_test"]].copy()
    records.insert(0, "dataset_id", str(dataset_id))
    records.insert(0, "farm", farm)
    records["score"] = np.nan

    prediction_df = prediction_rows(raw_df)
    operating_df = prediction_df[operating_mask(prediction_df)]
    columns = select_feature_columns(operating_df)
    for segment in contiguous_segments(operating_df):
        for features in feature_chunks(
            segment,
            window=window,
            columns=columns,
            use_fft=use_fft,
            chunk_size=chunk_size,
        ):
            matrix = apply_standardizer(features, mean, std).astype(np.float32)
            records.loc[features.index, "score"] = reconstruction_error(model, matrix)
    del raw_df, prediction_df, operating_df, columns
    gc.collect()
    return records


def _score_worker(args: tuple) -> pd.DataFrame:
    """Unpack process-pool arguments for a memory-isolated dataset scoring task."""
    return score_one_prediction_dataset(*args)


def score_prediction_records(
    model: PCA,
    farms: list[str],
    dataset_ids: dict[str, list[str]],
    window: int,
    mean: np.ndarray,
    std: np.ndarray,
    use_fft: bool,
    chunk_size: int,
    isolate_datasets: bool,
) -> pd.DataFrame:
    """Score prediction timelines, optionally releasing wide dataset memory per task."""
    tasks = [
        (model, farm, dataset_id, window, mean, std, use_fft, chunk_size)
        for farm in farms
        for dataset_id in dataset_ids[farm]
    ]
    if isolate_datasets:
        with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
            records = list(executor.map(_score_worker, tasks))
    else:
        records = [_score_worker(task) for task in tasks]
    return pd.concat(records, ignore_index=True)


def auc_arrays(records: pd.DataFrame, events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build the optional point-level AUC diagnostic from scored operating rows."""
    scored = records[np.isfinite(records["score"])].copy()
    labels = np.zeros(len(scored), dtype=int)
    event_map = {(row.farm, str(row.event_id)): row for row in events.itertuples()}
    for (farm, dataset_id), indices in scored.groupby(["farm", "dataset_id"]).groups.items():
        event = event_map[(farm, str(dataset_id))]
        if event.event_label != "anomaly":
            continue
        timestamps = pd.to_datetime(scored.loc[indices, "time_stamp"])
        labels[scored.index.get_indexer(indices)] = (
            (timestamps >= pd.Timestamp(event.event_start))
            & (timestamps <= pd.Timestamp(event.event_end))
        ).astype(int)
    return scored["score"].to_numpy(dtype=float), labels


def optional_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Return AUC only when the selected debug subset contains both point labels."""
    if len(np.unique(labels)) < 2:
        return None
    return float(compute_auc(scores, labels))


def main() -> None:
    """Fit PCA, calibrate a normal-only threshold, and write CARE artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default="A")
    parser.add_argument("--datasets", help="comma-separated prediction dataset IDs; debug only")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--cap-train", type=int, default=30000)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--variance", type=float, default=0.99, help="PCA explained-variance target"
    )
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument(
        "--feature-chunk-size",
        type=int,
        default=2000,
        help="maximum rows engineered at once for memory-safe wide-farm scoring",
    )
    parser.add_argument(
        "--svd-solver",
        choices=["covariance_eigh", "full", "randomized"],
        default="covariance_eigh",
        help="PCA solver; covariance_eigh reduces peak memory for tall feature matrices",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        help="fixed component count; required with randomized PCA for wide farms",
    )
    parser.add_argument("--no-fft", action="store_true")
    parser.add_argument(
        "--isolate-datasets",
        action="store_true",
        help="score each dataset in a fresh process to cap peak memory on wide farms",
    )
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    if not OUT.exists():
        raise SystemExit("processed data missing; run convert_care_to_parquet first")
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be between 0 and 1")
    if not 0 < args.variance <= 1:
        raise SystemExit("--variance must be in (0, 1]")
    if not 95 <= args.threshold_percentile <= 99:
        raise SystemExit("--threshold-percentile must be between 95 and 99")
    if args.feature_chunk_size <= 0:
        raise SystemExit("--feature-chunk-size must be positive")
    if args.svd_solver == "randomized" and args.n_components is None:
        raise SystemExit("--n-components is required when --svd-solver randomized")

    farms = [farm.strip() for farm in args.farms.split(",")]
    dataset_ids = {farm: list_datasets(OUT, farm) for farm in farms}
    if args.datasets:
        requested = {dataset_id.strip() for dataset_id in args.datasets.split(",")}
        dataset_ids = {
            farm: [dataset_id for dataset_id in ids if dataset_id in requested]
            for farm, ids in dataset_ids.items()
        }
    if not all(dataset_ids.values()):
        raise SystemExit("no prediction datasets selected")

    use_fft = not args.no_fft
    train, validation, mean, std = collect_normal_features(
        farms,
        args.window,
        args.cap_train,
        args.validation_fraction,
        use_fft,
        args.feature_chunk_size,
    )
    print(f"Fitting PCA on {len(train)} normal feature rows with {train.shape[1]} columns...")
    n_components: float | int = args.variance if args.n_components is None else args.n_components
    if isinstance(n_components, int) and n_components >= min(train.shape):
        raise SystemExit(
            "--n-components must be smaller than both training rows and feature columns"
        )
    model = PCA(n_components=n_components, svd_solver=args.svd_solver, random_state=args.seed)
    model.fit(train)
    print("Calibrating normal-validation threshold...")
    threshold = adaptive_threshold(
        reconstruction_error(model, validation), args.threshold_percentile
    )

    events = load_events(OUT)
    events = events[events["farm"].isin(farms)]
    if args.datasets:
        events = events[events["event_id"].astype(str).isin(requested)]
    print("Scoring prediction timelines...")
    records = score_prediction_records(
        model,
        farms,
        dataset_ids,
        args.window,
        mean,
        std,
        use_fft,
        args.feature_chunk_size,
        args.isolate_datasets,
    )
    records["is_alarm"] = flag(records["score"].fillna(-np.inf), threshold).astype(bool)
    scores, labels = auc_arrays(records, events)
    care = evaluate_care(records, events)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    artifact_paths = write_care_artifacts(care, RESULTS, f"{stamp}_pca_care")
    np.savez(
        RESULTS / f"{stamp}_pca_model.npz",
        components=model.components_,
        pca_mean=model.mean_,
        feature_mean=mean,
        feature_std=std,
    )
    auc = optional_auc(scores, labels)
    result = {
        "model": "pca_reconstruction_error",
        "farms": farms,
        "datasets": dataset_ids,
        "window": args.window,
        "use_fft": use_fft,
        "cap_train": args.cap_train,
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_components": int(model.n_components_),
        "n_components_requested": args.n_components,
        "explained_variance_ratio": float(model.explained_variance_ratio_.sum()),
        "threshold": float(threshold),
        "threshold_percentile": args.threshold_percentile,
        "threshold_source": "normal_validation_reconstruction_errors",
        "feature_chunk_size": args.feature_chunk_size,
        "svd_solver": args.svd_solver,
        "isolate_datasets": args.isolate_datasets,
        "auc_roc": auc,
        "care": care.metrics(),
        "care_artifacts": {key: str(path) for key, path in artifact_paths.items()},
    }
    result_path = RESULTS / f"{stamp}_pca_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"PCA components={model.n_components_} variance={result['explained_variance_ratio']:.4f}\n"
        f"CARE={care.care:.4f} AUC={auc if auc is not None else 'not-defined'} "
        f"threshold={threshold:.6f}\n"
        f"saved -> {result_path}"
    )


if __name__ == "__main__":
    main()
