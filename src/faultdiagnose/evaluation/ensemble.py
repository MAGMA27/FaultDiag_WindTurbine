from __future__ import annotations

import numpy as np
import pandas as pd


def normalize(scores: np.ndarray, ref_min: float, ref_max: float) -> np.ndarray:
    """Min-max scale scores into [0, 1] using a reference range (typically validation)."""
    if ref_max == ref_min:
        return np.zeros_like(scores, dtype=float)
    return np.clip((np.asarray(scores, dtype=float) - ref_min) / (ref_max - ref_min), 0.0, 1.0)


def empirical_percentile_rank(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map scores to percentile ranks using only a finite reference distribution."""
    reference = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    if len(reference) == 0:
        raise ValueError("percentile-rank reference has no finite scores")
    array = np.asarray(scores, dtype=float)
    ranks = np.full(array.shape, np.nan, dtype=float)
    finite = np.isfinite(array)
    ranks[finite] = np.searchsorted(reference, array[finite], side="right") / len(reference)
    return ranks


def robust_location_scale(scores: np.ndarray) -> tuple[float, float]:
    """Return median/IQR parameters for validation-only robust score standardization."""
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0, 1.0
    q25, median, q75 = np.percentile(finite, [25, 50, 75])
    scale = float(q75 - q25)
    if scale <= 1e-12:
        scale = float(np.std(finite))
    if scale <= 1e-12:
        scale = 1.0
    return float(median), scale


def robust_standardize(scores: np.ndarray, median: float, scale: float) -> np.ndarray:
    """Standardize anomaly scores using validation-only robust location/scale."""
    safe_scale = scale if scale > 1e-12 else 1.0
    return (np.asarray(scores, dtype=float) - median) / safe_scale


def validation_weights(val_aucs: dict[str, float]) -> dict[str, float]:
    """Ensemble weights from per-model validation AUC (paper: weights learned on validation).

    A model at/below random (AUC <= 0.5) contributes nothing; weights are the
    normalized excess over 0.5. Falls back to equal weights if none beat random.
    """
    raw = {k: max(0.0, v - 0.5) for k, v in val_aucs.items()}
    tot = sum(raw.values())
    if tot <= 0:
        n = len(val_aucs)
        return {k: 1.0 / n for k in val_aucs}
    return {k: v / tot for k, v in raw.items()}


def validation_stability_weights(per_model_validation: dict[str, np.ndarray]) -> dict[str, float]:
    """Validation-only weights from finite, stable normal-score behavior.

    The Nair & Babu paper says ensemble weights are learned from validation performance
    but does not specify the learner. In the frozen unsupervised protocol validation is
    normal-only, so this helper uses only normal validation stability: finite coverage
    divided by robust tail spread. It avoids touching prediction labels.
    """
    raw: dict[str, float] = {}
    for name, scores in per_model_validation.items():
        arr = np.asarray(scores, dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(arr) == 0 or len(finite) == 0:
            raw[name] = 0.0
            continue
        _, scale = robust_location_scale(finite)
        q95, q99 = np.percentile(finite, [95, 99])
        tail_spread = max(float(q99 - q95), 1e-12)
        finite_rate = len(finite) / len(arr)
        raw[name] = finite_rate / (scale + tail_spread)
    total = sum(raw.values())
    if total <= 0:
        n = len(per_model_validation)
        return {name: 1.0 / n for name in per_model_validation}
    return {name: value / total for name, value in raw.items()}


def combine(per_model: dict[str, np.ndarray], weights: dict[str, float],
            norm: dict[str, tuple[float, float]]) -> np.ndarray:
    """Weighted ensemble of per-model (already normalized) scores."""
    out = np.zeros_like(next(iter(per_model.values())), dtype=float)
    for name, s in per_model.items():
        out += weights.get(name, 0.0) * normalize(s, *norm[name])
    return out


def combine_standardized(
    per_model: dict[str, np.ndarray],
    weights: dict[str, float],
    norm: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Weighted average of robust-standardized scores."""
    first = next(iter(per_model.values()))
    out = np.zeros_like(np.asarray(first, dtype=float), dtype=float)
    for name, scores in per_model.items():
        out += weights.get(name, 0.0) * robust_standardize(scores, *norm[name])
    return out


def adaptive_threshold(scores: np.ndarray, p: float = 99.0) -> float:
    """Adaptive threshold = p-th percentile of (normal) scores (paper Algorithm 1, p in [95,99])."""
    return float(np.percentile(np.asarray(scores, dtype=float), p))


def flag(scores: np.ndarray, tau: float) -> np.ndarray:
    return (np.asarray(scores, dtype=float) > tau).astype(int)


def lead_time_hours(scores: np.ndarray, times, event_start, tau: float, step_minutes: float = 10.0):
    """Hours before event_start that the score first crosses tau. None if no early crossing."""
    scores = np.asarray(scores, dtype=float)
    times = pd.to_datetime(times).to_numpy()
    es = pd.Timestamp(event_start)
    before = times < es
    if not before.any():
        return None
    crossed = (scores > tau) & before
    if not crossed.any():
        return None
    first_time = times[int(np.argmax(crossed))]
    return float((es - first_time).total_seconds() / 3600.0)


def early_detection_report(lead_times: list[float | None]) -> dict:
    vals = [h for h in lead_times if h is not None]
    if not vals:
        return {"n_events": 0, "detected": 0, "mean_hours": None,
                "median_hours": None, "rate_24h": 0.0, "rate_48h": 0.0}
    arr = np.array(vals, dtype=float)
    return {
        "n_events": int(len(arr)),
        "mean_hours": float(arr.mean()),
        "median_hours": float(np.median(arr)),
        "rate_24h": float((arr >= 24).mean()),
        "rate_48h": float((arr >= 48).mean()),
    }

def onset_to_detection(
    scores: np.ndarray, times, event_start, tau: float, step_minutes: float = 10.0
):
    """Hours from fault onset (event_start) until the score first crosses tau. None if never."""
    scores = np.asarray(scores, dtype=float)
    times = pd.to_datetime(times).to_numpy()
    es = pd.Timestamp(event_start)
    after = times >= es
    if not after.any():
        return None
    crossed = (scores > tau) & after
    if not crossed.any():
        return None
    first_time = times[int(np.argmax(crossed))]
    return float((first_time - es).total_seconds() / 3600.0)
