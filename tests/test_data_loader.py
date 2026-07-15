import json

import pandas as pd
import pytest

from faultdiagnose.data import load_care

OUT = load_care.OUT_DEFAULT
pytestmark = pytest.mark.skipif(
    not OUT.exists(), reason="run convert_care_to_parquet first"
)


def test_manifest():
    m = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert m["total_datasets"] == 95
    assert m["total_rows"] == 5242948


def test_events_count_and_columns():
    ev = load_care.load_events(OUT)
    assert len(ev) == 95
    assert {"farm", "event_id", "event_label"}.issubset(ev.columns)


def test_load_dataset_keys_and_types():
    ds_id = load_care.list_datasets(OUT, "A")[0]
    df = load_care.load_dataset(OUT, "A", ds_id)
    for col in ("farm", "dataset_id", "asset_id", "time_stamp", "status_type_id", "train_test"):
        assert col in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["time_stamp"])
    assert df["dataset_id"].iloc[0] == ds_id


def test_iter_farm_yields_all_datasets():
    seen = [did for did, _ in load_care.iter_farm(OUT, "A")]
    assert len(seen) == 22
