from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COLS = ("farm", "dataset_id", "asset_id", "time_stamp", "id", "status_type_id", "train_test")
DEFAULT_WIN = 24  # feature window = 4h (24 x 10-min steps); configurable


# ponytail: dominant non-DC spectral magnitude over a window; cheap rfft, called per window.
# NOTE: only used when use_fft=True; rolling.apply is a Python callback per window and is the
# perf ceiling at scale. Replace with a stride-trick sliding FFT if FFT is needed on full data.
def _fft_dominant_magnitude(x: np.ndarray) -> float:
    x = x - x.mean()
    mags = np.abs(np.fft.rfft(x))[1:]
    return float(mags.max()) if mags.size else 0.0


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in KEY_COLS]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def engineer_features(
    df: pd.DataFrame,
    window: int = DEFAULT_WIN,
    columns: list[str] | None = None,
    include_raw: bool = False,
    use_fft: bool = False,
) -> pd.DataFrame:
    """Sliding-window temporal/statistical/frequency features (paper Section II.A).

    Per input column emits: _mean, _std, _skew, _kurt, _deriv [+ _fft if use_fft].
    Leading window rows are NaN (min_periods=window); dropna before sequencing.
    """
    if columns is None:
        columns = select_feature_columns(df)
    out: dict[str, pd.Series] = {}
    for col in columns:
        s = df[col].astype(float)
        out[f"{col}_mean"] = s.rolling(window, min_periods=window).mean()
        out[f"{col}_std"] = s.rolling(window, min_periods=window).std()
        out[f"{col}_skew"] = s.rolling(window, min_periods=window).skew()
        out[f"{col}_kurt"] = s.rolling(window, min_periods=window).kurt()
        out[f"{col}_deriv"] = s.diff()
        if use_fft:
            out[f"{col}_fft"] = s.rolling(window, min_periods=window).apply(
                _fft_dominant_magnitude, raw=True
            )
        if include_raw:
            out[f"{col}_raw"] = s
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
    return arr.mean(axis=0), arr.std(axis=0)


def apply_standardizer(features, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = features.values if hasattr(features, "values") else np.asarray(features)
    std = np.where(std == 0, 1.0, std)
    return (arr - mean) / std
