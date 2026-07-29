from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COLS = ("farm", "dataset_id", "asset_id", "time_stamp", "id", "status_type_id", "train_test")
DEFAULT_WIN = 24  # feature window = 4h (24 x 10-min steps); configurable


# ponytail: dominant non-DC spectral magnitude over a sliding window, vectorized via
# sliding_window_view + rfft (replaces the old rolling.apply callback, which was the perf ceiling).
def _fft_dominant_feature(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    feat = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return feat
    sw = np.lib.stride_tricks.sliding_window_view(
        np.asarray(values, dtype=np.float64), window
    )
    sw = sw - sw.mean(axis=1, keepdims=True)  # remove DC before FFT
    spec = np.abs(np.fft.rfft(sw, axis=1))[:, 1:]  # drop DC bin
    feat[window - 1 :] = spec.max(axis=1)
    return feat


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in KEY_COLS]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def _statistic_family(column: str) -> str:
    """Classify the original 10-minute SCADA statistic encoded in a column name."""
    for suffix, family in (
        ("_avg", "avg"),
        ("_min", "min"),
        ("_max", "max"),
        ("_std", "std"),
        ("_std_dev", "std"),
    ):
        if column.endswith(suffix):
            return family
    return "other"


def engineer_features(
    df: pd.DataFrame,
    window: int = DEFAULT_WIN,
    columns: list[str] | None = None,
    include_raw: bool = False,
    use_fft: bool = False,
    fillna: str | None = "ffill",
    clip_sigma: float | None = 5.0,
    feature_profile: str = "full",
) -> pd.DataFrame:
    """Sliding-window temporal/statistical/frequency features (paper Section II.A).

    ``full`` emits _mean, _std, _skew, _kurt, _deriv, _deriv2 [+ _fft].
    ``stat_aware`` preserves each original 10-minute statistic as _raw; it gives
    ``*_avg`` the full dynamic feature set, while ``*_min/_max/_std`` only receive
    rolling mean and first derivative. ``raw_stat_compact`` keeps all original
    10-minute statistics as _raw, but computes dynamic features only for ``*_avg``.
    Leading window rows are NaN (min_periods=window); dropna before sequencing.

    fillna: gap-fill strategy on raw sensors before feature extraction
           ("ffill"/"bfill"/"median"/None). Keeps sliding windows aligned.
    clip_sigma: clip raw sensors to +/-clip_sigma robust z after fill, to kill
           sensor spikes that survive as finite but huge values. None = off.
    """
    if feature_profile not in {"full", "stat_aware", "raw_stat_compact"}:
        raise ValueError(
            "feature_profile must be 'full', 'stat_aware', or 'raw_stat_compact'"
        )
    if columns is None:
        columns = select_feature_columns(df)
    out: dict[str, pd.Series] = {}
    # ponytail: fill gaps first (keep window alignment), then clip spikes;
    # robust mean/std for clipping computed per-column on the filled series.
    filled_cache: dict[str, pd.Series] = {}
    for col in columns:
        s = df[col].astype(float)
        if fillna is not None:
            if fillna == "median":
                s = s.fillna(s.median())
            s = s.ffill().bfill()
        if clip_sigma is not None:
            mu, sd = s.mean(), s.std()
            if sd and np.isfinite(sd) and sd > 0:
                s = s.clip(mu - clip_sigma * sd, mu + clip_sigma * sd)
        filled_cache[col] = s
        s = filled_cache[col]
        if include_raw or feature_profile in {"stat_aware", "raw_stat_compact"}:
            out[f"{col}_raw"] = s
        statistic_family = _statistic_family(col)
        add_dynamic = feature_profile != "raw_stat_compact" or statistic_family == "avg"
        if add_dynamic:
            out[f"{col}_mean"] = s.rolling(window, min_periods=window).mean()
            out[f"{col}_deriv"] = s.diff()
        if feature_profile == "full" or statistic_family == "avg":
            out[f"{col}_std"] = s.rolling(window, min_periods=window).std()
            out[f"{col}_skew"] = s.rolling(window, min_periods=window).skew()
            out[f"{col}_kurt"] = s.rolling(window, min_periods=window).kurt()
            out[f"{col}_deriv2"] = s.diff().diff()  # x_t - 2x_{t-1} + x_{t-2}
            if use_fft:
                out[f"{col}_fft"] = _fft_dominant_feature(s.to_numpy(), window)
    # ponytail: build from dict in one shot to avoid per-column DataFrame fragmentation
    return pd.DataFrame(out, index=df.index)


def build_sequences(features: pd.DataFrame, seq_len: int) -> np.ndarray:
    """Stack every window into (n_seq, seq_len, n_feat). Small/test use only;
    full data should use iter_sequences to avoid holding everything in memory."""
    f = features.dropna().to_numpy(dtype=np.float32)
    if len(f) < seq_len:
        return np.empty((0, seq_len, f.shape[1]), dtype=np.float32)
    n = len(f) - seq_len + 1
    # ponytail: simple stride; fine at moderate scale, switch to lazy torch Dataset at scale
    return np.stack([f[i : i + seq_len] for i in range(n)])


def iter_sequences(features: pd.DataFrame, seq_len: int):
    """Lazily yield each (seq_len, n_feat) window; memory-safe for training."""
    f = features.dropna().to_numpy(dtype=np.float32)
    for i in range(len(f) - seq_len + 1):
        yield f[i : i + seq_len]


def fit_standardizer(features) -> tuple[np.ndarray, np.ndarray]:
    """z-score stats (mean, std) per column, positionally. Works on DataFrame or ndarray."""
    arr = features.values if hasattr(features, "values") else np.asarray(features)
    # ponytail: dead/constant sensors (std~0 or non-finite) poison z-score;
    # substitute 1.0 so x-mean stays finite.
    return arr.mean(axis=0), np.nan_to_num(arr.std(axis=0), nan=1.0, posinf=1.0, neginf=1.0)


def apply_standardizer(features, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = features.values if hasattr(features, "values") else np.asarray(features)
    std = np.where(np.isfinite(std) & (std != 0), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return (arr - mean) / std
