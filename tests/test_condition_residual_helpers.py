"""Regression tests for the LightGBM condition-residual baseline helpers."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_condition_residual_baseline.py"
SPEC = importlib.util.spec_from_file_location("condition_residual_baseline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BASELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASELINE
SPEC.loader.exec_module(BASELINE)


def test_average_columns_selects_ten_minute_averages():
    assert BASELINE.average_columns(("sensor_8", "wind_speed_61")) == [
        "sensor_8_avg",
        "wind_speed_61_avg",
    ]


def test_split_tail_reserves_chronological_segment_end():
    segment = pd.DataFrame({"value": np.arange(10)})

    train, validation = BASELINE.split_tail(segment, 0.2)

    assert train["value"].tolist() == list(range(8))
    assert validation["value"].tolist() == [8, 9]


def test_replace_long_zero_runs_keeps_short_physical_zeros():
    frame = pd.DataFrame({"signal": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]})

    cleaned = BASELINE.replace_long_zero_runs(frame, ["signal"], minimum_run=6)

    assert cleaned.loc[:1, "signal"].tolist() == [0.0, 0.0]
    assert cleaned.loc[3:8, "signal"].isna().all()
