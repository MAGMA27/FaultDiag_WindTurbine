"""Regression tests for the GPU-tuning data split helpers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_gpu_tune.py"
FORMAL_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "farm_a_lstm_h256_l64_w48_full.json"
)
SPEC = importlib.util.spec_from_file_location("run_gpu_tune", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
GPU_TUNE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GPU_TUNE)


def test_contiguous_segments_split_at_operating_gap():
    df = pd.DataFrame(
        {
            "time_stamp": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:10", "2024-01-02 00:10"]
            ),
            "value": [1.0, 2.0, 3.0],
        }
    )

    segments = GPU_TUNE.contiguous_segments(df)

    assert [len(segment) for segment in segments] == [2, 1]


def test_prediction_rows_excludes_training_partition():
    df = pd.DataFrame({"train_test": ["train", "prediction", "prediction"], "value": [1, 2, 3]})

    prediction = GPU_TUNE.prediction_rows(df)

    assert prediction["value"].tolist() == [2, 3]


def test_prediction_rows_requires_explicit_partition():
    with pytest.raises(ValueError, match="train_test"):
        GPU_TUNE.prediction_rows(pd.DataFrame({"value": [1, 2, 3]}))


def test_auc_records_use_only_scored_operating_rows():
    records = pd.DataFrame(
        {
            "farm": ["A", "A", "A"],
            "dataset_id": ["1", "1", "1"],
            "time_stamp": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:10", "2024-01-01 00:20"]
            ),
            "score": [0.1, np.nan, 0.9],
        }
    )
    event = SimpleNamespace(
        event_label="anomaly",
        event_start=pd.Timestamp("2024-01-01 00:20"),
        event_end=pd.Timestamp("2024-01-01 00:20"),
    )

    scores, labels = GPU_TUNE._records_to_auc_arrays(records, {("A", "1"): event})

    assert scores.tolist() == [0.1, 0.9]
    assert labels.tolist() == [0, 1]


def test_split_train_validation_preserves_time_order():
    train_mats, val_mats = GPU_TUNE.split_train_validation(
        [np.arange(10, dtype=np.float32).reshape(-1, 1)], validation_fraction=0.2
    )

    assert train_mats[0].ravel().tolist() == list(range(8))
    assert val_mats[0].ravel().tolist() == [8.0, 9.0]


def test_formal_config_loads_model_and_run_parameters():
    config = GPU_TUNE.load_experiment_config(str(FORMAL_CONFIG_PATH))

    assert config["run"]["window"] == 48
    assert config["model"]["name"] == "lstm_h256_l64_w48_full"
