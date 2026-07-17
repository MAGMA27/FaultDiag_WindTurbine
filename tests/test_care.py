"""Unit tests for the CARE benchmark metric and operational event report."""

import pandas as pd
import pytest

from faultdiagnose.evaluation.care import (
    build_event_eligibility_report,
    criticality,
    earliness_weights,
    evaluate_care,
    write_care_artifacts,
)


def _predictions() -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=5, freq="10min")
    anomaly = pd.DataFrame(
        {
            "farm": "A",
            "dataset_id": "1",
            "time_stamp": times,
            "status_type_id": [0, 0, 0, 3, 0],
            "train_test": "prediction",
            "is_alarm": [True, True, True, True, False],
        }
    )
    normal = pd.DataFrame(
        {
            "farm": "A",
            "dataset_id": "2",
            "time_stamp": times,
            "status_type_id": [0, 0, 0, 0, 0],
            "train_test": "prediction",
            "is_alarm": [False, True, True, False, True],
        }
    )
    training = anomaly.iloc[[0]].assign(train_test="train", is_alarm=False)
    return pd.concat([anomaly, normal, training], ignore_index=True)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "farm": ["A", "A"],
            "event_id": ["1", "2"],
            "event_label": ["anomaly", "normal"],
            "event_start": [pd.Timestamp("2024-01-01"), pd.NaT],
            "event_end": [pd.Timestamp("2024-01-01 00:40"), pd.NaT],
            "event_description": ["test fault", None],
        }
    )


def test_criticality_holds_during_abnormal_status():
    values = criticality([True, True, False, False], [True, True, False, True])
    assert values.tolist() == [1, 2, 2, 1]


def test_earliness_weights_match_care_shape():
    assert earliness_weights(5).tolist() == pytest.approx([1.0, 1.0, 1.0, 0.5, 0.0])


def test_care_report_filters_training_and_reports_event_and_month():
    result = evaluate_care(_predictions(), _events(), criticality_threshold=3)

    assert result.coverage == pytest.approx(0.9375)
    assert result.accuracy == pytest.approx(0.4)
    assert result.reliability == pytest.approx(1.0)
    assert result.earliness == pytest.approx(1.0)
    assert result.care == pytest.approx(0.4)
    report = result.event_report.iloc[0]
    assert report.dataset_id == "1"
    assert report.event_detected
    assert report.first_alarm_time == pd.Timestamp("2024-01-01")
    assert result.monthly_false_alarms.iloc[0].false_alarm_episodes == 2
    assert result.monthly_false_alarms.iloc[0].false_alarm_points == 3


def test_care_artifacts_are_written(tmp_path):
    result = evaluate_care(_predictions(), _events(), criticality_threshold=3)
    paths = write_care_artifacts(result, tmp_path, "example")

    assert paths["metrics"].is_file()
    assert paths["events"].is_file()
    assert paths["event_eligibility"].is_file()
    assert paths["monthly_false_alarms"].is_file()


def test_event_eligibility_counts_only_operating_event_rows():
    report = build_event_eligibility_report(_predictions(), _events())

    event = report.iloc[0]
    assert event.dataset_id == "1"
    assert event.event_window_prediction_rows == 5
    assert event.operating_event_rows == 4
    assert event.pointwise_eligible


def test_care_requires_explicit_prediction_partition():
    predictions = _predictions().drop(columns="train_test")
    with pytest.raises(ValueError, match="train_test"):
        evaluate_care(predictions, _events())
