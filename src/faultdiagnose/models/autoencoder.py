"""Deterministic dense autoencoder used for the CARE benchmark comparison."""

from __future__ import annotations

import torch
import torch.nn as nn


class DenseAutoencoder(nn.Module):
    """Reconstruct one engineered SCADA feature vector without a latent prior."""

    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...] = (512, 256, 128)) -> None:
        """Build a symmetric ReLU autoencoder."""
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        encoder_layers: list[nn.Module] = []
        previous = in_dim
        for width in hidden_dims:
            encoder_layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        decoder_layers: list[nn.Module] = []
        for width in reversed(hidden_dims[:-1]):
            decoder_layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        decoder_layers.append(nn.Linear(previous, in_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the reconstruction for ``x``."""
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = "mean",
        include_kld: bool = False,
    ) -> torch.Tensor:
        """Return per-row reconstruction MSE compatible with VAE scoring helpers."""
        del include_kld
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")
        squared_error = (self(x) - x).square()
        return squared_error.mean(dim=1) if reduction == "mean" else squared_error.sum(dim=1)


class ReconstructionErrorRegressor(nn.Module):
    """Predict the expected normal reconstruction-error norm from SCADA inputs."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        """Build the three-linear-layer threshold network used in the CARE baseline."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one expected reconstruction-error norm per input row."""
        return self.network(x).squeeze(1)
