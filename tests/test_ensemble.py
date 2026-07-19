import numpy as np
import pandas as pd

from faultdiagnose.evaluation.ensemble import (
    adaptive_threshold,
    combine,
    combine_standardized,
    early_detection_report,
    flag,
    lead_time_hours,
    normalize,
    robust_location_scale,
    robust_standardize,
    validation_stability_weights,
    validation_weights,
)


def test_normalize_clips():
    s = np.array([0.0, 5.0, 10.0, 100.0])
    n = normalize(s, 0.0, 10.0)
    assert np.allclose(n, [0.0, 0.5, 1.0, 1.0])
    assert normalize(s, 5.0, 5.0).sum() == 0.0


def test_robust_standardize_uses_validation_iqr():
    scores = np.array([1.0, 2.0, 3.0, 100.0])
    median, scale = robust_location_scale(scores)

    standardized = robust_standardize(np.array([median, median + scale]), median, scale)

    assert scale > 0
    assert np.allclose(standardized, [0.0, 1.0])


def test_validation_stability_weights_ignore_nonfinite_and_normalize():
    weights = validation_stability_weights(
        {
            "stable": np.array([0.9, 1.0, 1.1, 1.2]),
            "wide": np.array([0.0, 10.0, 20.0, 30.0]),
            "bad": np.array([np.nan, np.inf]),
        }
    )

    assert weights["stable"] > weights["wide"]
    assert weights["bad"] == 0.0
    assert abs(sum(weights.values()) - 1.0) < 1e-9


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


def test_combine_standardized_weighted_average():
    per = {"a": np.array([1.0, 2.0]), "b": np.array([10.0, 20.0])}
    norm = {"a": (1.0, 1.0), "b": (10.0, 10.0)}
    weights = {"a": 0.5, "b": 0.5}

    out = combine_standardized(per, weights, norm)

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
