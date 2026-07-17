"""Regression tests for PCA baseline evaluation helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_pca_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_pca_baseline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PCA_BASELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PCA_BASELINE)


def test_reconstruction_error_is_zero_for_full_rank_pca():
    matrix = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    model = PCA(n_components=2, svd_solver="full").fit(matrix)

    errors = PCA_BASELINE.reconstruction_error(model, matrix)

    assert np.allclose(errors, 0.0, atol=1e-12)


def test_auc_arrays_ignore_unscored_timestamps():
    records = pd.DataFrame(
        {
            "farm": ["A", "A", "A"],
            "dataset_id": ["0", "0", "0"],
            "time_stamp": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:10", "2024-01-01 00:20"]
            ),
            "score": [0.1, np.nan, 0.9],
        }
    )
    events = pd.DataFrame(
        {
            "farm": ["A"],
            "event_id": ["0"],
            "event_label": ["anomaly"],
            "event_start": [pd.Timestamp("2024-01-01 00:20")],
            "event_end": [pd.Timestamp("2024-01-01 00:20")],
        }
    )

    scores, labels = PCA_BASELINE.auc_arrays(records, events)

    assert scores.tolist() == [0.1, 0.9]
    assert labels.tolist() == [0, 1]


def test_optional_auc_requires_both_classes():
    assert PCA_BASELINE.optional_auc(np.array([0.1, 0.2]), np.array([0, 0])) is None
    assert PCA_BASELINE.optional_auc(np.array([0.1, 0.9]), np.array([0, 1])) == 1.0


def test_feature_chunks_preserve_whole_segment_feature_rows():
    frame = pd.DataFrame(
        {
            "time_stamp": pd.date_range("2024-01-01", periods=12, freq="10min"),
            "sensor": np.arange(12, dtype=float),
        }
    )
    whole = PCA_BASELINE.engineer_features(
        frame, window=4, columns=["sensor"], use_fft=False
    ).dropna()
    chunks = list(
        PCA_BASELINE.feature_chunks(
            frame, window=4, columns=["sensor"], use_fft=False, chunk_size=3
        )
    )
    chunked = pd.concat(chunks)

    assert chunked.index.tolist() == whole.index.tolist()
    assert np.allclose(chunked.to_numpy(), whole.to_numpy())


def test_split_chunked_matrices_reserves_one_segment_tail_for_validation():
    chunks = iter(
        [
            np.array([[0.0], [1.0], [2.0], [3.0]]),
            np.array([[4.0], [5.0], [6.0], [7.0]]),
            np.array([[8.0], [9.0]]),
        ]
    )

    train, validation = PCA_BASELINE.split_chunked_matrices(chunks, 10, 0.2)

    assert np.concatenate(train).ravel().tolist() == list(range(8))
    assert np.concatenate(validation).ravel().tolist() == [8.0, 9.0]
