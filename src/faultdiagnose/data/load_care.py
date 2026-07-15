from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DEFAULT = REPO_ROOT / "CARE_To_Compare"
OUT_DEFAULT = REPO_ROOT / "data" / "processed"

FARMS = {"A": "Wind Farm A", "B": "Wind Farm B", "C": "Wind Farm C"}
KEY_COLS = ("farm", "dataset_id", "asset_id", "time_stamp", "id", "status_type_id", "train_test")

# ponytail: 匿名 CSV 里把 ° 编码成了乱码，轻量修一下单位列
_DEG_FIXES = [("锟紺", "°"), ("锟?", "°")]


def _clean_text(series: pd.Series) -> pd.Series:
    for bad, good in _DEG_FIXES:
        series = series.str.replace(bad, good, regex=False)
    return series


def _read_dataset_csv(path: Path, farm_code: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    keys = pd.DataFrame({"farm": farm_code, "dataset_id": path.stem}, index=df.index)
    return pd.concat([keys, df], axis=1)


def convert_care_to_parquet(raw_root: Path = RAW_DEFAULT, out_root: Path = OUT_DEFAULT) -> dict:
    """把 CARE 的 ; 分隔 CSV 转成按数据集拆分的 Parquet(内存安全：一次读一个文件)。

    输出布局:
      <out>/farm_<X>/<dataset_id>.parquet   每个 WT 时间序列一个文件(宽表)
      <out>/events.parquet                  合并的事件标注(含 farm 列)
      <out>/features.parquet                合并的字段字典(含 farm 列)
      <out>/manifest.json                   各风场行数/特征数/字段列表
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"farms": {}}

    for farm_code, farm_dir in FARMS.items():
        farm_raw = raw_root / farm_dir
        farm_out = out_root / f"farm_{farm_code}"
        farm_out.mkdir(parents=True, exist_ok=True)

        feat = pd.read_csv(farm_raw / "feature_description.csv", sep=";")
        for c in feat.columns:
            if pd.api.types.is_string_dtype(feat[c]):
                feat[c] = _clean_text(feat[c])
        feat.insert(0, "farm", farm_code)

        ev = pd.read_csv(farm_raw / "event_info.csv", sep=";")
        ev.insert(0, "farm", farm_code)

        n_rows = 0
        n_datasets = 0
        feature_cols: list[str] | None = None
        for csv_path in sorted(farm_raw.glob("datasets/*.csv")):
            df = _read_dataset_csv(csv_path, farm_code)
            df.to_parquet(farm_out / f"{csv_path.stem}.parquet", index=False)
            n_rows += len(df)
            n_datasets += 1
            if feature_cols is None:
                feature_cols = [c for c in df.columns if c not in KEY_COLS]

        feat.to_parquet(out_root / f"features_farm_{farm_code}.parquet", index=False)
        ev.to_parquet(out_root / f"events_farm_{farm_code}.parquet", index=False)
        manifest["farms"][farm_code] = {
            "n_datasets": n_datasets,
            "n_rows": n_rows,
            "n_features": len(feature_cols or []),
            "feature_columns": feature_cols or [],
        }

    events = pd.concat(
        [pd.read_parquet(out_root / f"events_farm_{f}.parquet") for f in FARMS],
        ignore_index=True,
    )
    events.to_parquet(out_root / "events.parquet", index=False)
    features = pd.concat(
        [pd.read_parquet(out_root / f"features_farm_{f}.parquet") for f in FARMS],
        ignore_index=True,
    )
    features.to_parquet(out_root / "features.parquet", index=False)

    manifest["total_datasets"] = sum(m["n_datasets"] for m in manifest["farms"].values())
    manifest["total_rows"] = sum(m["n_rows"] for m in manifest["farms"].values())
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def load_dataset(out_root: Path, farm: str, dataset_id: str) -> pd.DataFrame:
    return pd.read_parquet(Path(out_root) / f"farm_{farm}" / f"{dataset_id}.parquet")


def load_events(out_root: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(out_root) / "events.parquet")


def load_features(out_root: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(out_root) / "features.parquet")


def list_datasets(out_root: Path, farm: str | None = None) -> list[str]:
    if farm:
        return [p.stem for p in sorted((Path(out_root) / f"farm_{farm}").glob("*.parquet"))]
    return [f"farm_{m}" for m in FARMS]


def iter_farm(out_root: Path, farm: str):
    """逐个 dataset 产出 (dataset_id, DataFrame)，避免一次性把整风场读进内存。"""
    for p in sorted((Path(out_root) / f"farm_{farm}").glob("*.parquet")):
        yield p.stem, pd.read_parquet(p)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    m = convert_care_to_parquet(args.raw, args.out)
    print("Conversion done.")
    for f, info in m["farms"].items():
        print(f"  farm {f}: {info['n_datasets']} datasets, {info['n_rows']} rows, {info['n_features']} features")
    print(f"TOTAL: {m['total_datasets']} datasets, {m['total_rows']} rows")
