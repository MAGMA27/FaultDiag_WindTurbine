"""GPU-optimized tuning script for all 3 AE variants on CARE Farm A.

Key departures from the CPU scripts:
- Larger epochs (50-100), batch sizes (512-2048), hidden dims (128-512)
- More training data (cap=150k; CARE Farm A has millions of rows)
- Per-epoch AUC recording for learning-curve analysis
- Saves model checkpoints for ensemble reuse
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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
from faultdiagnose.models import LSTMAE, VAE, SeqWindowsDataset, TransformerAE

OUT = load_care.OUT_DEFAULT
RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS_DIR = RESULTS / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BREAK_MINUTES = 60.0


def set_seed(seed: int) -> None:
    """Seed random sources while favoring cuDNN throughput over exact determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_experiment_config(path: str | None) -> dict:
    """Load an optional JSON experiment configuration."""
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise SystemExit(f"config file not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit("config root must be a JSON object")
    return config


def contiguous_segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a time series at operating-state gaps too large for a rolling window."""
    df = df.copy()
    timestamps = pd.to_datetime(df["time_stamp"])
    gap_minutes = timestamps.diff().dt.total_seconds() / 60.0
    segment_id = (gap_minutes > BREAK_MINUTES).cumsum()
    return [segment for _, segment in df.groupby(segment_id)]


def prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the held-out prediction partition; never silently score train rows."""
    if "train_test" not in df.columns:
        raise ValueError("CARE datasets must provide train_test for prediction-only evaluation")
    return df[df["train_test"] == "prediction"]


def split_train_validation(
    mats: list[np.ndarray], validation_fraction: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Reserve the last portion of every contiguous normal segment for validation."""
    train_mats: list[np.ndarray] = []
    val_mats: list[np.ndarray] = []
    for mat in mats:
        validation_size = max(1, int(np.ceil(len(mat) * validation_fraction)))
        if validation_size >= len(mat):
            train_mats.append(mat)
            continue
        split_at = len(mat) - validation_size
        train_mats.append(mat[:split_at])
        val_mats.append(mat[split_at:])
    if not val_mats:
        raise ValueError(
            "No validation vectors available; lower --validation-fraction or add data."
        )
    return train_mats, val_mats


def _feature_columns(
    frame: pd.DataFrame,
    farm: str,
    feature_columns_by_farm: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return configured feature columns for a farm, or fall back to all sensor columns."""
    if feature_columns_by_farm is None or farm not in feature_columns_by_farm:
        return select_feature_columns(frame)
    columns = feature_columns_by_farm[farm]
    if columns == ["__avg_only__"]:
        return [
            column
            for column in select_feature_columns(frame)
            if column.endswith("_avg")
        ]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Farm {farm} is missing configured feature columns: {missing}")
    return columns


def _select_engineered_features(
    frame: pd.DataFrame,
    farm: str,
    engineered_feature_columns_by_farm: dict[str, list[str]] | None,
) -> pd.DataFrame:
    """Restrict post-engineering features, for a train-only feature-audit selection."""
    if engineered_feature_columns_by_farm is None or farm not in engineered_feature_columns_by_farm:
        return frame
    columns = engineered_feature_columns_by_farm[farm]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Farm {farm} is missing engineered features: {missing[:10]}")
    return frame.loc[:, columns]


def _balanced_quotas(total: int, count: int) -> list[int]:
    """Split a sample budget as evenly as possible over normal sequences."""
    if total <= 0 or count <= 0:
        return [0] * count
    base, remainder = divmod(total, count)
    return [base + int(index < remainder) for index in range(count)]


def _contiguous_balanced_chunks(
    segments: list[np.ndarray],
    quota: int,
    block_size: int = 4096,
    min_chunk_length: int = 1,
) -> list[np.ndarray]:
    """Take evenly spaced contiguous blocks without creating artificial sequences."""
    if min_chunk_length <= 0:
        raise ValueError("min_chunk_length must be positive")
    available = sum(len(segment) for segment in segments)
    take = min(quota, available)
    if take <= 0:
        return []
    lengths = np.asarray([len(segment) for segment in segments], dtype=np.float64)
    raw = lengths / lengths.sum() * take
    allocation = np.floor(raw).astype(int)
    allocation = np.minimum(allocation, lengths.astype(int))
    remaining = take - int(allocation.sum())
    for index in np.argsort(-(raw - allocation)):
        if remaining == 0:
            break
        if allocation[index] < lengths[index]:
            allocation[index] += 1
            remaining -= 1

    # Sequence models cannot use tiny fragments.  Move their budget to longer
    # segments instead of emitting chunks that will later be discarded.
    too_small = (allocation > 0) & (allocation < min_chunk_length)
    remaining = int(allocation[too_small].sum())
    allocation[too_small] = 0
    for index in np.argsort(-lengths):
        if remaining == 0:
            break
        capacity = int(lengths[index]) - int(allocation[index])
        minimum = min_chunk_length if allocation[index] == 0 else 1
        if capacity < minimum or remaining < minimum:
            continue
        addition = min(remaining, capacity)
        allocation[index] += addition
        remaining -= addition
    if remaining:
        raise ValueError("quota cannot be represented with the requested minimum chunk length")

    chunks: list[np.ndarray] = []
    for segment, segment_take in zip(segments, allocation, strict=True):
        if segment_take == 0:
            continue
        blocks = int(np.ceil(segment_take / block_size))
        base_length, extra = divmod(int(segment_take), blocks)
        block_lengths = [base_length + int(index < extra) for index in range(blocks)]
        max_start = max(0, len(segment) - max(block_lengths))
        starts = np.linspace(0, max_start, num=blocks, dtype=int)
        for start, length in zip(starts, block_lengths, strict=True):
            chunks.append(segment[start : start + length])
    return chunks


def collect_all(
    farms,
    window,
    cap,
    validation_fraction,
    use_fft=False,
    feature_columns_by_farm: dict[str, list[str]] | None = None,
    engineered_feature_columns_by_farm: dict[str, list[str]] | None = None,
    feature_profile: str = "full",
    normal_sampling: str = "sequential",
):
    """Collect both VAE (flat) mats and LSTM (sequence) mats from normal data."""
    if normal_sampling not in {"sequential", "balanced"}:
        raise ValueError("normal_sampling must be 'sequential' or 'balanced'")
    events = load_events(OUT)
    normal_ids: dict[str, list[str]] = {}
    for r in events.itertuples():
        if r.event_label == "normal":
            normal_ids.setdefault(r.farm, []).append(str(r.event_id))

    normal_sequences = [
        (farm, dataset_id)
        for farm in farms
        for dataset_id in normal_ids.get(farm, [])
    ]
    balanced_sampling = normal_sampling == "balanced" and cap > 0
    quotas = _balanced_quotas(cap, len(normal_sequences)) if balanced_sampling else []
    flat: list[np.ndarray] = []
    total = 0
    for sequence_index, (farm, did) in enumerate(normal_sequences):
        df = load_dataset(OUT, farm, did)
        if "train_test" in df.columns:
            df = df[df["train_test"] == "train"]
        df = df[operating_mask(df)]
        cols = _feature_columns(df, farm, feature_columns_by_farm)
        sequence_segments: list[np.ndarray] = []
        for seg in contiguous_segments(df):
            feats = engineer_features(
                seg,
                window=window,
                columns=cols,
                use_fft=use_fft,
                fillna="ffill",
                clip_sigma=5.0,
                feature_profile=feature_profile,
            )
            feats = _select_engineered_features(
                feats, farm, engineered_feature_columns_by_farm
            ).dropna()
            if len(feats) == 0:
                continue
            X = feats.values.astype(np.float32)
            if balanced_sampling:
                sequence_segments.append(X)
                continue
            X = X[: cap - total] if cap else X
            flat.append(X)
            total += len(X)
            if cap and total >= cap:
                break
        if balanced_sampling:
            selected = _contiguous_balanced_chunks(sequence_segments, quotas[sequence_index])
            flat.extend(selected)
            total += sum(len(chunk) for chunk in selected)
        if cap and total >= cap and normal_sampling == "sequential":
            break
    if not flat:
        raise ValueError("No usable normal training features were collected")
    if balanced_sampling and total < cap:
        print(f"  balanced sampling collected {total}/{cap} rows; some normal sequences were short")

    train_mats, val_mats = split_train_validation(flat, validation_fraction)
    X_all = np.concatenate(train_mats, axis=0)
    mean, std = fit_standardizer(X_all)
    zmats = [apply_standardizer(m, mean, std).astype(np.float32) for m in train_mats]
    zval_mats = [apply_standardizer(m, mean, std).astype(np.float32) for m in val_mats]
    vae_X = np.concatenate(zmats, axis=0)
    vae_val = np.concatenate(zval_mats, axis=0)
    return vae_X, zmats, vae_val, zval_mats, mean, std


def collect_raw_normal(
    farms: list[str],
    cap: int,
    validation_fraction: float,
    seed: int,
    feature_columns_by_farm: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Collect CARE-AE-style raw rows with random normal train/validation split."""
    if len(farms) != 1:
        raise ValueError("care_raw input currently requires exactly one farm")
    farm = farms[0]
    events = load_events(OUT)
    normal_ids = events[(events["farm"] == farm) & (events["event_label"] == "normal")][
        "event_id"
    ].astype(str)
    matrices: list[np.ndarray] = []
    for dataset_id in normal_ids:
        frame = load_dataset(OUT, farm, dataset_id)
        frame = frame[(frame["train_test"] == "train") & operating_mask(frame)]
        columns = _feature_columns(frame, farm, feature_columns_by_farm)
        values = frame.loc[:, columns].replace([np.inf, -np.inf], np.nan)
        matrices.append(values.fillna(values.median()).to_numpy(dtype=np.float32))
    if not matrices:
        raise ValueError(f"no normal operating training rows available for Farm {farm}")
    raw = np.concatenate(matrices, axis=0)
    if cap and cap < len(raw):
        indices = np.random.default_rng(seed).choice(len(raw), size=cap, replace=False)
        raw = raw[indices]
    permutation = np.random.default_rng(seed).permutation(len(raw))
    split = round(len(raw) * (1.0 - validation_fraction))
    train_raw, validation_raw = raw[permutation[:split]], raw[permutation[split:]]
    mean, std = fit_standardizer(train_raw)
    train_x = apply_standardizer(train_raw, mean, std).astype(np.float32)
    validation_x = apply_standardizer(validation_raw, mean, std).astype(np.float32)
    return train_x, [train_x], validation_x, [validation_x], mean, std


def collect_raw_normal_per_asset(
    farms: list[str],
    cap: int,
    validation_fraction: float,
    seed: int,
    feature_columns_by_farm: dict[str, list[str]] | None = None,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    """Build a shared raw VAE dataset after fitting Z-scores per normal asset."""
    if len(farms) != 1:
        raise ValueError("per-asset raw input currently requires exactly one farm")
    farm = farms[0]
    events = load_events(OUT)
    normal_ids = events[(events["farm"] == farm) & (events["event_label"] == "normal")][
        "event_id"
    ].astype(str)
    feature_rows: list[np.ndarray] = []
    asset_rows: list[np.ndarray] = []
    for dataset_id in normal_ids:
        frame = load_dataset(OUT, farm, dataset_id)
        frame = frame[(frame["train_test"] == "train") & operating_mask(frame)]
        columns = _feature_columns(frame, farm, feature_columns_by_farm)
        values = frame.loc[:, columns].replace([np.inf, -np.inf], np.nan)
        feature_rows.append(values.fillna(values.median()).to_numpy(dtype=np.float32))
        asset_rows.append(frame["asset_id"].astype(str).to_numpy())
    if not feature_rows:
        raise ValueError(f"no normal operating training rows available for Farm {farm}")
    raw = np.concatenate(feature_rows, axis=0)
    assets = np.concatenate(asset_rows)
    # A sensor can be entirely missing for one turbine even though it exists on
    # other turbines.  Per-dataset median filling leaves such a column NaN;
    # use the all-normal asset median as a neutral fallback before fitting each
    # asset's own Z-score (it then becomes a constant zero for that asset).
    global_fill = np.nanmedian(raw, axis=0)
    global_fill = np.nan_to_num(global_fill, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.where(np.isfinite(raw), raw, global_fill).astype(np.float32)
    if cap and cap < len(raw):
        indices = np.random.default_rng(seed).choice(len(raw), size=cap, replace=False)
        raw, assets = raw[indices], assets[indices]

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for asset in sorted(np.unique(assets)):
        asset_raw = raw[assets == asset]
        permutation = rng.permutation(len(asset_raw))
        split = round(len(asset_raw) * (1.0 - validation_fraction))
        train_raw = asset_raw[permutation[:split]]
        validation_raw = asset_raw[permutation[split:]]
        if len(train_raw) == 0 or len(validation_raw) == 0:
            raise ValueError(f"asset {asset} has insufficient rows for per-asset split")
        mean, std = fit_standardizer(train_raw)
        normalizers[asset] = (mean, std)
        train_parts.append(apply_standardizer(train_raw, mean, std).astype(np.float32))
        validation_parts.append(apply_standardizer(validation_raw, mean, std).astype(np.float32))
    train_x = np.concatenate(train_parts, axis=0)
    validation_x = np.concatenate(validation_parts, axis=0)
    if not np.isfinite(train_x).all() or not np.isfinite(validation_x).all():
        raise ValueError("per-asset normalization produced non-finite values")
    global_mean, global_std = fit_standardizer(raw)
    return train_x, [train_x], validation_x, [validation_x], global_mean, global_std, normalizers


def collect_raw_normal_sequences(
    farms: list[str],
    cap: int,
    validation_fraction: float,
    feature_columns_by_farm: dict[str, list[str]] | None = None,
    normal_sampling: str = "balanced",
    min_segment_length: int = 1,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Collect raw contiguous normal SCADA blocks for CARE-aligned sequence models."""
    if len(farms) != 1:
        raise ValueError("care_raw sequence input currently requires exactly one farm")
    if normal_sampling not in {"sequential", "balanced"}:
        raise ValueError("normal_sampling must be 'sequential' or 'balanced'")
    if min_segment_length <= 0:
        raise ValueError("min_segment_length must be positive")
    farm = farms[0]
    events = load_events(OUT)
    normal_ids = events[(events["farm"] == farm) & (events["event_label"] == "normal")][
        "event_id"
    ].astype(str).tolist()
    balanced_sampling = normal_sampling == "balanced" and cap > 0
    quotas = _balanced_quotas(cap, len(normal_ids)) if balanced_sampling else []
    raw_mats: list[np.ndarray] = []
    total = 0
    for sequence_index, dataset_id in enumerate(normal_ids):
        frame = load_dataset(OUT, farm, dataset_id)
        frame = frame[(frame["train_test"] == "train") & operating_mask(frame)].copy()
        columns = _feature_columns(frame, farm, feature_columns_by_farm)
        values = frame.loc[:, columns].replace([np.inf, -np.inf], np.nan)
        frame.loc[:, columns] = values.fillna(values.median())
        sequence_segments = [
            segment.loc[:, columns].to_numpy(dtype=np.float32)
            for segment in contiguous_segments(frame)
            if len(segment) >= min_segment_length
        ]
        if balanced_sampling:
            selected = _contiguous_balanced_chunks(
                sequence_segments,
                quotas[sequence_index],
                min_chunk_length=min_segment_length,
            )
            raw_mats.extend(selected)
            total += sum(len(chunk) for chunk in selected)
            continue
        for segment in sequence_segments:
            selected = segment[: cap - total] if cap else segment
            raw_mats.append(selected)
            total += len(selected)
            if cap and total >= cap:
                break
        if cap and total >= cap:
            break
    if not raw_mats:
        raise ValueError("No usable normal raw sequence rows were collected")
    if balanced_sampling and total < cap:
        print(f"  balanced raw sequence sampling collected {total}/{cap} rows")
    train_mats, val_mats = split_train_validation(raw_mats, validation_fraction)
    train_raw = np.concatenate(train_mats, axis=0)
    mean, std = fit_standardizer(train_raw)
    zmats = [apply_standardizer(matrix, mean, std).astype(np.float32) for matrix in train_mats]
    zval_mats = [apply_standardizer(matrix, mean, std).astype(np.float32) for matrix in val_mats]
    return np.concatenate(zmats), zmats, np.concatenate(zval_mats), zval_mats, mean, std


def _empty_prediction_records(df: pd.DataFrame, farm: str, dataset_id: str) -> pd.DataFrame:
    """Create a full status timeline; only operating timestamps receive scores."""
    records = prediction_rows(df)[["time_stamp", "status_type_id", "train_test"]].copy()
    records.insert(0, "dataset_id", str(dataset_id))
    records.insert(0, "farm", farm)
    records["score"] = np.nan
    return records


def _predicted_normal_scores(threshold_model, inputs: np.ndarray, batch_size: int) -> np.ndarray:
    """Predict expected normal score from standardized model inputs."""
    loader = DataLoader(torch.from_numpy(inputs).float(), batch_size=batch_size, shuffle=False)
    values: list[np.ndarray] = []
    threshold_model.eval()
    with torch.inference_mode():
        for xb in loader:
            values.append(threshold_model(xb.to(DEVICE, non_blocking=True)).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def evaluate_vae_records(
    model,
    farms,
    window,
    mean,
    std,
    use_fft=False,
    feature_columns_by_farm: dict[str, list[str]] | None = None,
    score_reduction: str = "sum",
    include_kld: bool = True,
    feature_profile: str = "full",
    threshold_model=None,
    threshold_batch: int = 4096,
    raw_input: bool = False,
    asset_standardizers: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
):
    """Score operating timestamps while retaining every prediction status row for CARE."""
    records_list = []
    for farm in farms:
        for did in list_datasets(OUT, farm):
            raw_df = load_dataset(OUT, farm, did)
            records = _empty_prediction_records(raw_df, farm, did)
            score_df = prediction_rows(raw_df)
            score_df = score_df[operating_mask(score_df)]
            cols = _feature_columns(score_df, farm, feature_columns_by_farm)
            segments = [score_df] if raw_input else contiguous_segments(score_df)
            for segment in segments:
                if raw_input:
                    feats = segment.loc[:, cols].replace([np.inf, -np.inf], np.nan)
                    if asset_standardizers is None:
                        feats = feats.fillna(pd.Series(mean, index=cols))
                        Z = apply_standardizer(feats, mean, std).astype(np.float32)
                    else:
                        asset_ids = segment.loc[feats.index, "asset_id"].astype(str).to_numpy()
                        Z = np.empty(feats.shape, dtype=np.float32)
                        for asset in np.unique(asset_ids):
                            if asset not in asset_standardizers:
                                raise ValueError(
                                    f"no normal standardizer available for asset {asset}"
                                )
                            asset_mean, asset_std = asset_standardizers[asset]
                            rows = asset_ids == asset
                            asset_feats = feats.loc[rows].fillna(pd.Series(asset_mean, index=cols))
                            Z[rows] = apply_standardizer(asset_feats, asset_mean, asset_std)
                else:
                    feats = engineer_features(
                        segment,
                        window=window,
                        columns=cols,
                        use_fft=use_fft,
                        fillna="ffill",
                        clip_sigma=5.0,
                        feature_profile=feature_profile,
                    ).dropna()
                if len(feats) == 0:
                    continue
                if not raw_input:
                    Z = apply_standardizer(feats, mean, std).astype(np.float32)
                with torch.inference_mode():
                    scores = (
                        model.reconstruction_error(
                            torch.from_numpy(Z).to(DEVICE),
                            reduction=score_reduction,
                            include_kld=include_kld,
                        )
                        .cpu()
                        .numpy()
                    )
                if threshold_model is not None:
                    scores = scores - _predicted_normal_scores(threshold_model, Z, threshold_batch)
                records.loc[feats.index, "score"] = scores
            records_list.append(records)
    return pd.concat(records_list, ignore_index=True)


def evaluate_seq_records(
    model,
    farms,
    window,
    seq_len,
    mean,
    std,
    use_fft=False,
    batch=4096,
    feature_columns_by_farm: dict[str, list[str]] | None = None,
    engineered_feature_columns_by_farm: dict[str, list[str]] | None = None,
    feature_profile: str = "full",
    threshold_model=None,
    threshold_batch: int = 4096,
    raw_input: bool = False,
):
    """Score sequence windows and retain the complete prediction status timeline."""
    records_list = []
    for farm in farms:
        for did in list_datasets(OUT, farm):
            raw_df = load_dataset(OUT, farm, did)
            records = _empty_prediction_records(raw_df, farm, did)
            score_df = prediction_rows(raw_df)
            score_df = score_df[operating_mask(score_df)]
            cols = _feature_columns(score_df, farm, feature_columns_by_farm)
            for segment in contiguous_segments(score_df):
                if raw_input:
                    feats = segment.loc[:, cols].replace([np.inf, -np.inf], np.nan)
                    feats = feats.fillna(pd.Series(mean, index=cols))
                else:
                    feats = engineer_features(
                        segment,
                        window=window,
                        columns=cols,
                        use_fft=use_fft,
                        fillna="ffill",
                        clip_sigma=5.0,
                        feature_profile=feature_profile,
                    )
                    feats = _select_engineered_features(
                        feats, farm, engineered_feature_columns_by_farm
                    ).dropna()
                if len(feats) < seq_len:
                    continue
                Z = apply_standardizer(feats, mean, std).astype(np.float32)
                ds = SeqWindowsDataset([Z], seq_len)
                dl = DataLoader(ds, batch_size=batch, shuffle=False, pin_memory=True)
                win_err = np.empty(len(ds), dtype=np.float32)
                pos = 0
                with torch.inference_mode():
                    for xb in dl:
                        e = model.reconstruction_error(xb.to(DEVICE, non_blocking=True))
                        win_err[pos : pos + len(e)] = e.cpu().numpy()
                        pos += len(e)
                if threshold_model is not None:
                    expected = _predicted_normal_scores(
                        threshold_model, Z[seq_len - 1 :], threshold_batch
                    )
                    win_err = win_err - expected
                scores = np.empty(len(feats), dtype=np.float32)
                scores[: seq_len - 1] = win_err[0]
                scores[seq_len - 1 :] = win_err
                records.loc[feats.index, "score"] = scores
            records_list.append(records)
    return pd.concat(records_list, ignore_index=True)


def validation_scores_vae(
    model,
    X_val: np.ndarray,
    batch_size: int,
    score_reduction: str = "sum",
    include_kld: bool = True,
) -> np.ndarray:
    """Return validation reconstruction errors used to calibrate an unsupervised threshold."""
    loader = DataLoader(torch.from_numpy(X_val).float(), batch_size=batch_size, shuffle=False)
    errors = []
    with torch.inference_mode():
        for xb in loader:
            errors.append(
                model.reconstruction_error(
                    xb.to(DEVICE, non_blocking=True),
                    reduction=score_reduction,
                    include_kld=include_kld,
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(errors)


def validation_scores_seq(
    model, val_mats: list[np.ndarray], seq_len: int, batch_size: int
) -> np.ndarray:
    """Return normal validation window errors used to calibrate an unsupervised threshold."""
    dataset = SeqWindowsDataset(val_mats, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    errors = []
    with torch.inference_mode():
        for xb in loader:
            errors.append(
                model.reconstruction_error(xb.to(DEVICE, non_blocking=True)).cpu().numpy()
            )
    return np.concatenate(errors)


def sequence_inputs_and_scores(
    model, matrices: list[np.ndarray], seq_len: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return last-step inputs aligned with sequence reconstruction scores."""
    dataset = SeqWindowsDataset(matrices, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    inputs: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    with torch.inference_mode():
        for xb in loader:
            # ``numpy()`` alone would be a view into the full [B, T, D] batch.
            # Keeping those views for every batch retains all sequence windows
            # (tens of GB for long Transformer runs) during threshold fitting.
            inputs.append(xb[:, -1, :].clone().numpy())
            errors.append(
                model.reconstruction_error(xb.to(DEVICE, non_blocking=True)).cpu().numpy()
            )
    if not inputs:
        raise ValueError("sequence matrices are shorter than seq_len")
    return np.concatenate(inputs), np.concatenate(errors)


def _records_to_auc_arrays(records: pd.DataFrame, ev_map: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build point-level AUC inputs from the same prediction-only CARE timeline."""
    scored = records[np.isfinite(records["score"])].copy()
    labels = np.zeros(len(scored), dtype=int)
    for (farm, dataset_id), index in scored.groupby(["farm", "dataset_id"]).groups.items():
        event = ev_map.get((farm, str(dataset_id)))
        if event is None or event.event_label != "anomaly":
            continue
        start, end = pd.Timestamp(event.event_start), pd.Timestamp(event.event_end)
        timestamps = pd.to_datetime(scored.loc[index, "time_stamp"])
        labels[scored.index.get_indexer(index)] = (
            (timestamps >= start) & (timestamps <= end)
        ).astype(int)
    return scored["score"].to_numpy(dtype=float), labels


def train_vae_gpu(
    model,
    X,
    X_val,
    epochs,
    batch_size,
    patience,
    min_delta,
    lr=1e-3,
    scheduler: str = "none",
    warmup_epochs: int = 0,
    min_lr: float = 1e-5,
    kl_anneal_epochs: int = 0,
    name="",
    loss_f=None,
):
    model.to(DEVICE)
    target_beta = float(getattr(model, "beta", 1.0))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    plateau = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=min_lr,
        )
        if scheduler == "plateau"
        else None
    )
    dl = DataLoader(
        torch.from_numpy(X).float(), batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_dl = DataLoader(
        torch.from_numpy(X_val).float(), batch_size=batch_size, shuffle=False, pin_memory=True
    )
    losses = []
    best_state, best_val_loss, best_epoch, stale_epochs = None, float("inf"), 0, 0
    loss_ema: float | None = None
    for ep in range(epochs):
        if kl_anneal_epochs > 0:
            model.beta = target_beta * min(1.0, float(ep + 1) / float(kl_anneal_epochs))
        else:
            model.beta = target_beta
        if scheduler == "warmup_cosine":
            if warmup_epochs > 0 and ep < warmup_epochs:
                current_lr = lr * float(ep + 1) / float(warmup_epochs)
            else:
                denom = max(1, epochs - max(0, warmup_epochs))
                progress = (ep - max(0, warmup_epochs)) / denom
                cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))
                current_lr = min_lr + (lr - min_lr) * cosine
            for group in opt.param_groups:
                group["lr"] = current_lr
        model.train()
        tot, n, skipped = 0.0, 0, 0
        for xb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, _, _ = model.loss(xb, recon, mu, logvar)
            batch_loss = float(loss.detach().item())
            # A finite but astronomically large reconstruction loss can still produce a
            # clipped update and hide an unstable decoder step.  Normal VAE losses here
            # are O(10--10^3); skip only clear numerical outliers, not hard examples.
            outlier_limit = max(1e4, 100.0 * loss_ema) if loss_ema is not None else 1e4
            if not torch.isfinite(loss) or batch_loss > outlier_limit:
                skipped += 1
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=False
            )
            if not torch.isfinite(grad_norm):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
            opt.step()
            loss_ema = batch_loss if loss_ema is None else 0.99 * loss_ema + 0.01 * batch_loss
            tot += batch_loss
            n += 1
        if n == 0:
            print(f"  stop at ep {ep + 1}; all VAE batches had non-finite loss/grad")
            break
        losses.append(tot / n)
        model.eval()
        val_total, val_batches = 0.0, 0
        with torch.inference_mode():
            for xb in val_dl:
                xb = xb.to(DEVICE, non_blocking=True)
                recon, mu, logvar = model(xb)
                val_loss, _, _ = model.loss(xb, recon, mu, logvar)
                if not torch.isfinite(val_loss):
                    val_total = float("nan")
                    val_batches = 1
                    break
                val_total += val_loss.item()
                val_batches += 1
        current_val_loss = val_total / val_batches
        if not np.isfinite(current_val_loss):
            print(f"  stop at ep {ep + 1}; validation loss became non-finite")
            break
        if current_val_loss < best_val_loss - min_delta:
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_val_loss, best_epoch, stale_epochs = current_val_loss, ep + 1, 0
        else:
            stale_epochs += 1
        if plateau is not None:
            plateau.step(current_val_loss)
        current_lr = float(opt.param_groups[0]["lr"])
        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"  ep {ep + 1}/{epochs} loss={losses[-1]:.4f} "
                f"val={current_val_loss:.4f} lr={current_lr:.2e} "
                f"beta={model.beta:.4g} skipped={skipped}"
            )
        if loss_f is not None:
            loss_f.write(
                f"name={name} ep={ep + 1} loss={losses[-1]:.6f} "
                f"val={current_val_loss:.6f} lr={current_lr:.8g} "
                f"beta={model.beta:.8g} skipped={skipped}\n"
            )
            loss_f.flush()
        if stale_epochs >= patience:
            print(f"  early stop at ep {ep + 1}; best validation loss at ep {best_epoch}")
            break
    if best_state is None:
        raise RuntimeError("VAE training failed before producing a finite validation checkpoint")
    model.load_state_dict(best_state)
    return losses, best_epoch, best_val_loss


def train_dense_ae_gpu(
    model,
    X: np.ndarray,
    X_val: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
    min_delta: float,
    lr: float = 1e-3,
    noise_std: float = 0.0,
    name: str = "dense_ae",
    loss_f=None,
):
    """Train a deterministic denoising autoencoder and restore its best checkpoint."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(
        torch.from_numpy(X).float(), batch_size=batch_size, shuffle=True, pin_memory=True
    )
    validation_loader = DataLoader(
        torch.from_numpy(X_val).float(), batch_size=batch_size, shuffle=False, pin_memory=True
    )
    losses: list[float] = []
    best_state, best_val_loss, best_epoch, stale_epochs = None, float("inf"), 0, 0
    for epoch in range(epochs):
        model.train()
        total, batches = 0.0, 0
        for clean_x in train_loader:
            clean_x = clean_x.to(DEVICE, non_blocking=True)
            noisy_x = clean_x + noise_std * torch.randn_like(clean_x) if noise_std else clean_x
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(noisy_x), clean_x)
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        train_loss = total / batches
        losses.append(train_loss)
        model.eval()
        validation_total, validation_batches = 0.0, 0
        with torch.inference_mode():
            for clean_x in validation_loader:
                clean_x = clean_x.to(DEVICE, non_blocking=True)
                validation_total += torch.nn.functional.mse_loss(model(clean_x), clean_x).item()
                validation_batches += 1
        validation_loss = validation_total / validation_batches
        if validation_loss < best_val_loss - min_delta:
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_val_loss, best_epoch, stale_epochs = validation_loss, epoch + 1, 0
        else:
            stale_epochs += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  ep {epoch + 1}/{epochs} loss={train_loss:.4f} "
                f"val={validation_loss:.4f} noise={noise_std:.3g}",
                flush=True,
            )
        if loss_f is not None:
            loss_f.write(
                f"name={name} ep={epoch + 1} loss={train_loss:.6f} "
                f"val={validation_loss:.6f} noise={noise_std:.6g}\n"
            )
            loss_f.flush()
        if stale_epochs >= patience:
            print(f"  early stop at ep {epoch + 1}; best validation loss at ep {best_epoch}")
            break
    if best_state is None:
        raise RuntimeError("dense AE training failed before producing a checkpoint")
    model.load_state_dict(best_state)
    return losses, best_epoch, best_val_loss


def train_seq_gpu(
    model,
    zmats,
    val_mats,
    seq_len,
    epochs,
    batch_size,
    patience,
    min_delta,
    lr=1e-3,
    scheduler: str = "none",
    warmup_epochs: int = 0,
    min_lr: float = 1e-5,
    grad_clip: float = 1.0,
    name="",
    loss_f=None,
    cache_dir: str | Path | None = None,
    max_cache_gb: float = 1.0,
    cache_dtype: str = "float32",
    num_workers: int = 0,
    ram_window_cache: bool = False,
):
    model.to(DEVICE)
    cache_torch_dtype = {"float16": torch.float16, "float32": torch.float32}.get(cache_dtype)
    if cache_torch_dtype is None:
        raise ValueError("cache_dtype must be 'float16' or 'float32'")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    cache_base = Path(cache_dir) if cache_dir is not None else None
    cache_item_bytes = torch.empty((), dtype=cache_torch_dtype).element_size()
    cache_matrices = (*zmats, *val_mats) if cache_base is not None else tuple(zmats)
    estimated_cache_bytes = sum(
        max(0, len(matrix) - seq_len + 1) * seq_len * matrix.shape[1] * cache_item_bytes
        for matrix in cache_matrices
    )
    wants_window_cache = cache_base is not None or ram_window_cache
    if wants_window_cache and estimated_cache_bytes > max_cache_gb * 1024**3:
        print(
            f"  window cache skipped: estimated {estimated_cache_bytes / 1024**3:.2f} GB "
            f"> limit {max_cache_gb:.2f} GB; using lazy windows"
        )
        cache_base = None
        ram_window_cache = False
    cache_token = ""
    if cache_base is not None:
        digest = hashlib.sha1()
        digest.update(str(seq_len).encode())
        for matrix in (*zmats, *val_mats):
            array = np.asarray(matrix)
            digest.update(str(array.shape).encode())
            sample_step = max(1, len(array) // 1024)
            digest.update(array[::sample_step].tobytes())
        cache_token = digest.hexdigest()[:16]
    train_cache = (
        cache_base / f"{name}_train_seq{seq_len}_{cache_token}.pt"
        if cache_base is not None
        else None
    )
    val_cache = (
        cache_base / f"{name}_val_seq{seq_len}_{cache_token}.pt"
        if cache_base is not None
        else None
    )
    loader_options = {
        "batch_size": batch_size,
        "pin_memory": True,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 2})
    ds = SeqWindowsDataset(
        zmats,
        seq_len,
        cache_path=train_cache,
        cache_dtype=cache_torch_dtype,
        materialize_windows=ram_window_cache,
    )
    dl = DataLoader(ds, shuffle=True, **loader_options)
    val_ds = SeqWindowsDataset(
        val_mats, seq_len, cache_path=val_cache, cache_dtype=cache_torch_dtype
        # Keep validation lazy for RAM-only caching: it is a small once-per-epoch
        # pass and materializing it needlessly raises the host-memory peak.
        , materialize_windows=False
    )
    if len(val_ds) == 0:
        raise ValueError("Validation segments are shorter than seq_len.")
    val_dl = DataLoader(val_ds, shuffle=False, **loader_options)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    plateau = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=min_lr,
        )
        if scheduler == "plateau"
        else None
    )
    losses = []
    best_state, best_val_loss, best_epoch, stale_epochs = None, float("inf"), 0, 0
    for ep in range(epochs):
        if scheduler == "warmup_cosine":
            if warmup_epochs > 0 and ep < warmup_epochs:
                current_lr = lr * float(ep + 1) / float(warmup_epochs)
            else:
                denom = max(1, epochs - max(0, warmup_epochs))
                progress = (ep - max(0, warmup_epochs)) / denom
                cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))
                current_lr = min_lr + (lr - min_lr) * cosine
            for group in opt.param_groups:
                group["lr"] = current_lr
        model.train()
        tot, n, skipped = 0.0, 0, 0
        for xb in dl:
            xb = xb.to(DEVICE, dtype=torch.float32, non_blocking=True)
            opt.zero_grad()
            recon, _ = model(xb)
            loss = model.loss(xb, recon)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=False
            )
            if not torch.isfinite(grad_norm):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
            opt.step()
            tot += loss.item()
            n += 1
        if n == 0:
            print(f"  stop at ep {ep + 1}; all sequence batches had non-finite loss/grad")
            break
        losses.append(tot / n)
        model.eval()
        val_total, val_batches = 0.0, 0
        with torch.inference_mode():
            for xb in val_dl:
                xb = xb.to(DEVICE, dtype=torch.float32, non_blocking=True)
                recon, _ = model(xb)
                val_loss = model.loss(xb, recon)
                if not torch.isfinite(val_loss):
                    val_total = float("nan")
                    val_batches = 1
                    break
                val_total += val_loss.item()
                val_batches += 1
        current_val_loss = val_total / val_batches
        if not np.isfinite(current_val_loss):
            print(f"  stop at ep {ep + 1}; validation loss became non-finite")
            break
        if current_val_loss < best_val_loss - min_delta:
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_val_loss, best_epoch, stale_epochs = current_val_loss, ep + 1, 0
        else:
            stale_epochs += 1
        if plateau is not None:
            plateau.step(current_val_loss)
        current_lr = float(opt.param_groups[0]["lr"])
        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"  ep {ep + 1}/{epochs} loss={losses[-1]:.4f} "
                f"val={current_val_loss:.4f} lr={current_lr:.2e} skipped={skipped}"
            )
        if loss_f is not None:
            loss_f.write(
                f"name={name} ep={ep + 1} loss={losses[-1]:.6f} "
                f"val={current_val_loss:.6f} lr={current_lr:.8g} skipped={skipped}\n"
            )
            loss_f.flush()
        if stale_epochs >= patience:
            print(f"  early stop at ep {ep + 1}; best validation loss at ep {best_epoch}")
            break
    if best_state is None:
        raise RuntimeError("Sequence training failed before producing a finite checkpoint")
    model.load_state_dict(best_state)
    # A RAM window cache can occupy most of a 32 GB host.  Persistent DataLoader
    # workers otherwise keep the dataset alive into post-training scoring, where
    # threshold fitting and full prediction evaluation add another memory peak.
    for loader in (dl, val_dl):
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    del dl, val_dl, ds, val_ds
    gc.collect()
    return losses, best_epoch, best_val_loss


CONFIGS = [
    {
        "name": "vae_h256_l32_b0.1",
        "model": "vae",
        "hidden": 256,
        "latent": 32,
        "beta": 0.1,
        "epochs": 80,
        "batch": 1024,
    },
    {
        "name": "vae_h512_l64_b0.1",
        "model": "vae",
        "hidden": 512,
        "latent": 64,
        "beta": 0.1,
        "epochs": 80,
        "batch": 1024,
    },
    {
        "name": "vae_h256_l64_b1.0",
        "model": "vae",
        "hidden": 256,
        "latent": 64,
        "beta": 1.0,
        "epochs": 80,
        "batch": 1024,
    },
    {
        "name": "vae_h256_l64_b0.1",
        "model": "vae",
        "hidden": 256,
        "latent": 64,
        "beta": 0.1,
        "epochs": 80,
        "batch": 1024,
    },
    {
        "name": "vae_h512_l64_b1.0",
        "model": "vae",
        "hidden": 512,
        "latent": 64,
        "beta": 1.0,
        "epochs": 80,
        "batch": 1024,
    },
    {
        "name": "vae_h512_l128_b1.0",
        "model": "vae",
        "hidden": 512,
        "latent": 128,
        "beta": 1.0,
        "epochs": 80,
        "batch": 1024,
    },
    {
        "name": "lstm_h128_l64",
        "model": "lstm",
        "hidden": 128,
        "latent": 64,
        "epochs": 80,
        "batch": 512,
        "seq_len": 48,
    },
    {
        "name": "lstm_h256_l64",
        "model": "lstm",
        "hidden": 256,
        "latent": 64,
        "epochs": 80,
        "batch": 512,
        "seq_len": 48,
    },
    {
        "name": "lstm_h128_l128",
        "model": "lstm",
        "hidden": 128,
        "latent": 128,
        "epochs": 80,
        "batch": 512,
        "seq_len": 48,
    },
    {
        "name": "lstm_h256_l128",
        "model": "lstm",
        "hidden": 256,
        "latent": 128,
        "epochs": 80,
        "batch": 512,
        "seq_len": 48,
    },
    {
        "name": "tf_h128_l64",
        "model": "transformer",
        "hidden": 128,
        "latent": 64,
        "epochs": 80,
        "batch": 256,
        "seq_len": 48,
    },
    {
        "name": "tf_h256_l64",
        "model": "transformer",
        "hidden": 256,
        "latent": 64,
        "epochs": 80,
        "batch": 256,
        "seq_len": 48,
    },
    {
        "name": "tf_h128_l128",
        "model": "transformer",
        "hidden": 128,
        "latent": 128,
        "epochs": 80,
        "batch": 256,
        "seq_len": 48,
    },
]


def run_config(
    cfg, farms, window, cap, use_fft, shared, label_mode="operating", loss_f=None, args=None
):
    name = cfg["name"]
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")

    seq_len = cfg.get("seq_len", 24)
    epochs = args.epochs if args is not None and getattr(args, "epochs", None) else cfg["epochs"]
    vae_X, zmats, vae_val, zval_mats, mean, std = shared
    in_dim = vae_X.shape[1]
    n_train = len(vae_X)
    n_win = sum(len(m) - seq_len + 1 for m in zmats)
    print(f"  in_dim={in_dim}, train vecs={n_train}, train windows={n_win}")

    events = load_events(OUT)
    ev_map = {(r.farm, str(r.event_id)): r for r in events.itertuples()}

    t0 = time.time()

    if cfg["model"] == "vae":
        model = VAE(in_dim, latent=cfg["latent"], hidden=cfg["hidden"], beta=cfg["beta"])
        _, best_epoch, best_val_loss = train_vae_gpu(
            model,
            vae_X,
            vae_val,
            epochs,
            cfg["batch"],
            args.patience,
            args.min_delta,
            lr=cfg.get("lr", 1e-3),
            name=name,
            loss_f=loss_f,
        )
        score_reduction = cfg.get("score_reduction", "sum")
        include_kld = cfg.get("include_kld", True)
        validation_scores = validation_scores_vae(
            model, vae_val, cfg["batch"], score_reduction, include_kld
        )
        records = evaluate_vae_records(
            model,
            farms,
            window,
            mean,
            std,
            use_fft,
            score_reduction=score_reduction,
            include_kld=include_kld,
        )
    elif cfg["model"] == "lstm":
        model = LSTMAE(in_dim, seq_len, latent=cfg["latent"], hidden=cfg["hidden"], num_layers=2)
        _, best_epoch, best_val_loss = train_seq_gpu(
            model,
            zmats,
            zval_mats,
            seq_len,
            epochs,
            cfg["batch"],
            args.patience,
            args.min_delta,
            name=name,
            loss_f=loss_f,
        )
        validation_scores = validation_scores_seq(model, zval_mats, seq_len, cfg["batch"])
        records = evaluate_seq_records(
            model, farms, window, seq_len, mean, std, use_fft, batch=cfg["batch"]
        )
    else:  # transformer
        nhead = 4 if cfg["hidden"] <= 128 else 8
        model = TransformerAE(
            in_dim, seq_len, latent=cfg["latent"], d_model=cfg["hidden"], nhead=nhead, num_layers=2
        )
        _, best_epoch, best_val_loss = train_seq_gpu(
            model,
            zmats,
            zval_mats,
            seq_len,
            epochs,
            cfg["batch"],
            args.patience,
            args.min_delta,
            name=name,
            loss_f=loss_f,
        )
        validation_scores = validation_scores_seq(model, zval_mats, seq_len, cfg["batch"])
        records = evaluate_seq_records(
            model, farms, window, seq_len, mean, std, use_fft, batch=cfg["batch"]
        )

    elapsed = time.time() - t0
    threshold = adaptive_threshold(validation_scores, args.threshold_percentile)
    records["is_alarm"] = flag(records["score"].fillna(-np.inf).to_numpy(), threshold).astype(bool)
    scores, labels = _records_to_auc_arrays(records, ev_map)
    auc = compute_auc(scores, labels)
    care_evaluation = evaluate_care(records, events)
    artifact_stem = f"{args.artifact_stamp}_{name}_care"
    care_paths = write_care_artifacts(care_evaluation, RESULTS, artifact_stem)
    effective_config = {
        "run": {
            "farms": args.farms,
            "window": window,
            "cap_train": cap,
            "use_fft": use_fft,
            "test_split": "prediction",
            "label_mode": label_mode,
            "validation_fraction": args.validation_fraction,
            "threshold_percentile": args.threshold_percentile,
            "threshold_source": "normal_validation_reconstruction_errors",
            "patience": args.patience,
            "min_delta": args.min_delta,
            "seed": args.seed,
            "device": DEVICE,
        },
        "model": {**cfg, "epochs": epochs},
    }

    result = {
        **cfg,
        "in_dim": in_dim,
        "n_train": n_train,
        "n_test": int(len(scores)),
        "n_positives": int(labels.sum()),
        "auc_roc": float(auc),
        "threshold": float(threshold),
        "care": care_evaluation.metrics(),
        "care_artifacts": {key: str(path) for key, path in care_paths.items()},
        "elapsed_s": round(elapsed, 1),
        "device": DEVICE,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "seed": args.seed,
        "config_source": args.config,
        "effective_config": effective_config,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    print(
        f"  >> CARE={care_evaluation.care:.4f} AUC={auc:.4f} "
        f"threshold={threshold:.4f} ({elapsed:.0f}s)"
    )

    # Save checkpoint + its frozen normalization stats (self-contained artifact)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
    np.savez(MODELS_DIR / f"{name}_norm.npz", mean=mean, std=std)

    return result


def main():
    global DEVICE
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args()
    experiment_config = load_experiment_config(config_args.config)
    run_defaults = experiment_config.get("run", {})
    model_config = experiment_config.get("model")
    if not isinstance(run_defaults, dict):
        raise SystemExit("config.run must be a JSON object")
    if model_config is not None and not isinstance(model_config, dict):
        raise SystemExit("config.model must be a JSON object")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="JSON experiment configuration")
    ap.add_argument("--farms", default=run_defaults.get("farms", "A"))
    ap.add_argument("--window", type=int, default=run_defaults.get("window", 24))
    ap.add_argument("--cap-train", type=int, default=run_defaults.get("cap_train", 150000))
    ap.add_argument(
        "--no-fft",
        action="store_true",
        default=not run_defaults.get("use_fft", True),
        help="disable FFT features (enabled by default)",
    )
    ap.add_argument(
        "--label-mode",
        choices=["operating"],
        default=run_defaults.get("label_mode", "operating"),
        help="score only status 0/2 rows; required by the frozen CARE protocol",
    )
    ap.add_argument("--cfg", help="run single config by name, e.g. vae_h256_l32_b0.1")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument(
        "--validation-fraction", type=float, default=run_defaults.get("validation_fraction", 0.15)
    )
    ap.add_argument("--patience", type=int, default=run_defaults.get("patience", 20))
    ap.add_argument("--min-delta", type=float, default=run_defaults.get("min_delta", 1e-4))
    ap.add_argument(
        "--threshold-percentile",
        type=float,
        default=run_defaults.get("threshold_percentile", 99.0),
        help="percentile of normal validation reconstruction errors used as the alarm threshold",
    )
    ap.add_argument("--seed", type=int, default=run_defaults.get("seed", 20260717))
    ap.add_argument("--device", default=run_defaults.get("device", DEVICE), help="override device")
    args = ap.parse_args()
    DEVICE = args.device
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be between 0 and 1")
    if not 95 <= args.threshold_percentile <= 99:
        raise SystemExit("--threshold-percentile must be between 95 and 99")
    set_seed(args.seed)

    if not OUT.exists():
        raise SystemExit("processed data missing; run convert_care_to_parquet first")
    print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})")
    print(f"Seed: {args.seed}; validation={args.validation_fraction:.0%}; patience={args.patience}")
    print("cuDNN benchmark: enabled (throughput prioritized over bit-exact repeatability)")

    farms = [f.strip() for f in args.farms.split(",")]
    use_fft = not args.no_fft
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    args.artifact_stamp = stamp
    loss_path = RESULTS / f"gpu_tune_loss_{stamp}.txt"
    loss_f = open(loss_path, "w", encoding="utf-8")
    print(f"  loss log -> {loss_path}")
    if model_config is not None:
        required = {"name", "model", "hidden", "latent", "epochs", "batch"}
        missing = required.difference(model_config)
        if missing:
            raise SystemExit(f"config.model missing required keys: {sorted(missing)}")
        configs = [model_config]
    else:
        configs = CONFIGS if not args.cfg else [c for c in CONFIGS if c["name"] == args.cfg]
    if not configs:
        raise SystemExit(f"config '{args.cfg}' not found")

    # ponytail: engineer features once (config-independent) instead of 13x inside run_config
    shared = collect_all(farms, args.window, args.cap_train, args.validation_fraction, use_fft)
    results = []
    for cfg in configs:
        try:
            r = run_config(
                cfg,
                farms,
                args.window,
                args.cap_train,
                use_fft,
                shared,
                args.label_mode,
                loss_f,
                args,
            )
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({**cfg, "error": str(e)})

    loss_f.close()

    # Save all results
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    path = RESULTS / f"gpu_tune_{stamp}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    print(f"\n{'=' * 60}\n  SUMMARY\n{'=' * 60}")
    for r in results:
        if "auc_roc" in r:
            print(
                f"  {r['name']:30s}  CARE={r['care']['care']:.4f} "
                f"AUC={r['auc_roc']:.4f}  ({r['elapsed_s']:.0f}s)"
            )
        else:
            print(f"  {r['name']:30s}  FAILED: {r.get('error', '?')}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
