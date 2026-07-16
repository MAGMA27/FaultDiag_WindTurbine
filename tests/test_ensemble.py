import numpy as np
import pandas as pd

from faultdiagnose.evaluation.ensemble import (
    adaptive_threshold,
    combine,
    early_detection_report,
    flag,
    lead_time_hours,
    normalize,
    validation_weights,
)


def test_normalize_clips():
    s = np.array([0.0, 5.0, 10.0, 100.0])
    n = normalize(s, 0.0, 10.0)
    assert np.allclose(n, [0.0, 0.5, 1.0, 1.0])
    assert normalize(s, 5.0, 5.0).sum() == 0.0


def test_validation_weights_excess_over_half():
    w = validation_weights({"a": 0.9, "b": 0.6, "c": 0.4})
    assert w["c"] == 0.0
    assert abs(w["a"] - 0.8) < 1e-9 and abs(w["b"] - 0.2) < 1e-9


def test_validation_weights_fallback_equal():
    w = validation_weights({"a": 0.4, "b": 0.3})
    assert abs(w["a"] - 0.5) < 1e-9 and abs(w["b"] - 0.5) < 1e-9


def test_combine_monotonic():
    per = {"a": np.array([0.0, 1.0]), "b": np.array([1.0, 0.0])}
    norm = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
    w = {"a": 1.0, "b": 0.0}
    out = combine(per, w, norm)
    assert np.allclose(out, [0.0, 1.0])


def test_adaptive_threshold_percentile():
    s = np.arange(101, dtype=float)
    assert adaptive_threshold(s, 99) == 99.0


def test_flag_binary():
    assert list(flag(np.array([1.0, 5.0, 9.0]), 5.0)) == [0, 0, 1]


def test_lead_time_and_report():
    times = pd.date_range("2023-01-01", periods=10, freq="10min")
    scores = np.array([0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    lt = lead_time_hours(scores, times, times[9], tau=5.0)
    assert abs(lt - (7 * 10 / 60)) < 1e-6
    rep = early_detection_report([lt, None, 30.0])
    assert rep["n_events"] == 2
    assert rep["rate_24h"] == 0.5
    assert rep["rate_48h"] == 0.0
