"""Normal-only adaptive threshold fitting for reconstruction-error models."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from faultdiagnose.models import ReconstructionErrorRegressor


def train_adaptive_threshold(
    features: np.ndarray,
    scores: np.ndarray,
    *,
    hidden: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    device: str,
    verbose: bool = True,
) -> ReconstructionErrorRegressor:
    """Learn expected normal reconstruction score conditional on model inputs."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(scores, dtype=np.float32).reshape(-1)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("features must be a 2D matrix aligned with scores")
    if len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("adaptive threshold training data must be finite and non-empty")
    model = ReconstructionErrorRegressor(x.shape[1], hidden).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x.to(device, non_blocking=True))
            loss = torch.nn.functional.mse_loss(prediction, batch_y.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  adaptive threshold NN ep {epoch + 1}/{epochs}", flush=True)
    model.eval()
    return model


def predicted_normal_scores(
    model: ReconstructionErrorRegressor,
    features: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Predict expected normal reconstruction scores in batches."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False, pin_memory=True)
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch_x in loader:
            output.append(model(batch_x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)
