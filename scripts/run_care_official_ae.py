"""Reproduce the published CARE adaptive-AE baseline for Farms A and B.

The CARE paper specifies per-farm AE and adaptive-threshold parameters but
does not publish the wind-speed/power cut-offs used by its additional data
filter.  That filter is therefore optional and explicit here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faultdiagnose.data import (  # noqa: E402
    OUT_DEFAULT,
    list_datasets,
    load_dataset,
    load_events,
    operating_mask,
)
from faultdiagnose.evaluation.anomaly import compute_auc  # noqa: E402
from faultdiagnose.evaluation.care import evaluate_care, write_care_artifacts  # noqa: E402
from faultdiagnose.features.engineer import (  # noqa: E402
    apply_standardizer,
    fit_standardizer,
    select_feature_columns,
)
from faultdiagnose.models import DenseAutoencoder, ReconstructionErrorRegressor  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class PublishedFarmConfig:
    """Values reported in CARE paper Table 4.2.2."""

    hidden_dims: tuple[int, ...]
    learning_rate: float
    noise_std: float
    threshold_hidden: int
    gamma: float


PUBLISHED_CONFIGS = {
    "A": PublishedFarmConfig((44, 25, 4, 25, 44), 0.0018, 0.06, 23, 0.344),
    "B": PublishedFarmConfig((40, 15, 40), 0.003, 0.0, 36, 0.234),
}


def set_seed(seed: int) -> None:
    """Make sampling and PyTorch initialization repeatable."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_errors(model: DenseAutoencoder, matrix: np.ndarray, batch_size: int) -> np.ndarray:
    """Return the per-row L2 reconstruction errors used by the published baseline."""
    loader = DataLoader(torch.from_numpy(matrix).float(), batch_size=batch_size, shuffle=False)
    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(DEVICE, non_blocking=True)
            values.append(torch.linalg.vector_norm(model(batch) - batch, dim=1).cpu().numpy())
    return np.concatenate(values)


def train_autoencoder(
    model: DenseAutoencoder,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    config: PublishedFarmConfig,
    epochs: int,
    batch_size: int,
) -> tuple[int, float]:
    """Train the AE with the paper's L2-validation early-stopping rule."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = DataLoader(torch.from_numpy(train_x).float(), batch_size=batch_size, shuffle=True)
    best_state, best_error, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(epochs):
        model.train()
        for clean_x in loader:
            clean_x = clean_x.to(DEVICE, non_blocking=True)
            noisy_x = clean_x + config.noise_std * torch.randn_like(clean_x)
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(model(noisy_x), clean_x).backward()
            optimizer.step()
        validation_error = float(l2_errors(model, validation_x, batch_size).mean())
        if validation_error < best_error:
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_error, best_epoch, stale = validation_error, epoch + 1, 0
        else:
            stale += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  AE ep {epoch + 1}/{epochs} validation_l2={validation_error:.5f}", flush=True)
        if stale >= 3:
            print(f"  AE early stop at ep {epoch + 1}; best epoch={best_epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("AE did not produce a checkpoint")
    model.load_state_dict(best_state)
    return best_epoch, best_error


def train_error_regressor(
    model: ReconstructionErrorRegressor,
    features: np.ndarray,
    errors: np.ndarray,
    learning_rate: float,
    epochs: int,
    batch_size: int,
) -> ReconstructionErrorRegressor:
    """Fit the published input-to-expected-error adaptive threshold model."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(errors).float()),
        batch_size=batch_size,
        shuffle=True,
    )
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_error in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(
                model(batch_x.to(DEVICE, non_blocking=True)),
                batch_error.to(DEVICE, non_blocking=True),
            )
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  threshold NN ep {epoch + 1}/{epochs}", flush=True)
    model.eval()
    return model


def collect_normal_training(
    farm: str, cap: int, seed: int
) -> tuple[np.ndarray, list[str], int]:
    """Load raw operating rows from normal training datasets and randomly cap them."""
    events = load_events(OUT_DEFAULT)
    normal_ids = events[(events["farm"] == farm) & (events["event_label"] == "normal")][
        "event_id"
    ].astype(str)
    matrices: list[np.ndarray] = []
    columns: list[str] | None = None
    for dataset_id in normal_ids:
        frame = load_dataset(OUT_DEFAULT, farm, dataset_id)
        frame = frame[(frame["train_test"] == "train") & operating_mask(frame)]
        if columns is None:
            columns = select_feature_columns(frame)
        values = frame.loc[:, columns].replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.median()).to_numpy(dtype=np.float32)
        matrices.append(values)
    if columns is None:
        raise RuntimeError(f"no normal training data found for Farm {farm}")
    all_rows = np.concatenate(matrices, axis=0)
    available = len(all_rows)
    if cap and cap < available:
        indices = np.random.default_rng(seed).choice(available, size=cap, replace=False)
        all_rows = all_rows[indices]
    return all_rows, columns, available


def prediction_records(
    farm: str,
    columns: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    ae: DenseAutoencoder,
    threshold_nn: ReconstructionErrorRegressor,
    gamma: float,
    batch_size: int,
) -> pd.DataFrame:
    """Score every held-out row with the published adaptive threshold rule."""
    records: list[pd.DataFrame] = []
    for dataset_id in list_datasets(OUT_DEFAULT, farm):
        frame = load_dataset(OUT_DEFAULT, farm, dataset_id)
        frame = frame[frame["train_test"] == "prediction"].copy()
        raw = frame.loc[:, columns].replace([np.inf, -np.inf], np.nan)
        raw = raw.fillna(pd.Series(mean, index=columns))
        normalized = apply_standardizer(raw.to_numpy(dtype=np.float32), mean, std).astype(
            np.float32
        )
        errors = l2_errors(ae, normalized, batch_size)
        loader = DataLoader(
            torch.from_numpy(normalized).float(), batch_size=batch_size, shuffle=False
        )
        expected: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                expected.append(threshold_nn(batch.to(DEVICE, non_blocking=True)).cpu().numpy())
        result = frame[["time_stamp", "status_type_id", "train_test"]].copy()
        result.insert(0, "dataset_id", str(dataset_id))
        result.insert(0, "farm", farm)
        result["score"] = errors - np.concatenate(expected)
        result["is_alarm"] = result["score"] > gamma
        records.append(result)
    return pd.concat(records, ignore_index=True)


def main() -> None:
    """Run the published Farm A/B CARE adaptive-AE configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default="B", help="Comma-separated subset of A,B.")
    parser.add_argument("--cap-train", type=int, default=60000, help="0 keeps all normal rows.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--threshold-epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    farms = [farm.strip().upper() for farm in args.farms.split(",")]
    unsupported = sorted(set(farms).difference(PUBLISHED_CONFIGS))
    if unsupported:
        raise SystemExit(
            "official adaptive configuration is currently available only for A/B: "
            f"{unsupported}"
        )
    set_seed(args.seed)
    started = time.time()
    all_records: list[pd.DataFrame] = []
    run_details: dict[str, dict[str, object]] = {}
    for farm in farms:
        config = PUBLISHED_CONFIGS[farm]
        raw_x, columns, available_rows = collect_normal_training(farm, args.cap_train, args.seed)
        permutation = np.random.default_rng(args.seed).permutation(len(raw_x))
        split = round(len(raw_x) * 0.75)
        train_raw, validation_raw = raw_x[permutation[:split]], raw_x[permutation[split:]]
        mean, std = fit_standardizer(train_raw)
        train_x = apply_standardizer(train_raw, mean, std).astype(np.float32)
        validation_x = apply_standardizer(validation_raw, mean, std).astype(np.float32)
        print(
            f"Official CARE AE Farm {farm}: rows={len(raw_x)}/{available_rows} "
            f"dims={len(columns)}"
        )
        ae = DenseAutoencoder(train_x.shape[1], config.hidden_dims)
        best_epoch, validation_l2 = train_autoencoder(
            ae, train_x, validation_x, config, args.epochs, args.batch
        )
        validation_errors = l2_errors(ae, validation_x, args.batch)
        threshold_nn = train_error_regressor(
            ReconstructionErrorRegressor(train_x.shape[1], config.threshold_hidden),
            validation_x,
            validation_errors,
            config.learning_rate,
            args.threshold_epochs,
            args.batch,
        )
        all_records.append(
            prediction_records(
                farm, columns, mean, std, ae, threshold_nn, config.gamma, args.batch
            )
        )
        run_details[farm] = {
            "published_config": asdict(config),
            "available_normal_train_rows": available_rows,
            "used_normal_train_rows": len(raw_x),
            "input_dimensions": len(columns),
            "best_ae_epoch": best_epoch,
            "validation_l2": validation_l2,
        }
    records = pd.concat(all_records, ignore_index=True)
    events = load_events(OUT_DEFAULT)
    event_map = {(row.farm, str(row.event_id)): row for row in events.itertuples()}
    scored = records[np.isfinite(records["score"])].copy()
    labels = np.zeros(len(scored), dtype=int)
    for (farm, dataset_id), index in scored.groupby(["farm", "dataset_id"]).groups.items():
        event = event_map[(farm, str(dataset_id))]
        if event.event_label == "anomaly":
            timestamps = pd.to_datetime(scored.loc[index, "time_stamp"])
            labels[scored.index.get_indexer(index)] = (
                (timestamps >= event.event_start) & (timestamps <= event.event_end)
            ).astype(int)
    care = evaluate_care(records, events[events["farm"].isin(farms)])
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    paths = write_care_artifacts(care, RESULTS, f"{stamp}_official_care_ae")
    result = {
        "model": "official_care_adaptive_ae",
        "farms": farms,
        "paper": "Gück et al. CARE to Compare, Table 4.2.2",
        "protocol_notes": {
            "raw_features": True,
            "normal_split": "random 75/25",
            "score": "L2 reconstruction norm",
            "threshold": "predicted normal L2 error + published gamma",
            "wind_power_filter": "not enabled: paper does not publish its cut-offs",
        },
        "details": run_details,
        "auc_roc": float(compute_auc(scored["score"].to_numpy(), labels)),
        "care": care.metrics(),
        "care_artifacts": {key: str(value) for key, value in paths.items()},
        "elapsed_s": round(time.time() - started, 1),
    }
    output = RESULTS / f"{stamp}_official_care_ae_result.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Official CARE AE: AUC={result['auc_roc']:.4f} CARE={care.care:.4f} -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
