from __future__ import annotations

import numpy as np
import pandas as pd


def normalize(scores: np.ndarray, ref_min: float, ref_max: float) -> np.ndarray:
    """Min-max scale scores into [0, 1] using a reference range (typically validation)."""
    if ref_max == ref_min:
        return np.zeros_like(scores, dtype=float)
    return np.clip((np.asarray(scores, dtype=float) - ref_min) / (ref_max - ref_min), 0.0, 1.0)


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


def combine(per_model: dict[str, np.ndarray], weights: dict[str, float],
            norm: dict[str, tuple[float, float]]) -> np.ndarray:
    """Weighted ensemble of per-model (already normalized) scores."""
    out = np.zeros_like(next(iter(per_model.values())), dtype=float)
    for name, s in per_model.items():
        out += weights.get(name, 0.0) * normalize(s, *norm[name])
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
