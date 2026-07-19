from __future__ import annotations

import torch
import torch.nn as nn


class VAE(nn.Module):
    """Multivariate VAE over a single time-step feature vector (paper Section II.B.1).

    Captures cross-sensor correlations; LSTM/Transformer variants handle temporal later.
    Loss = alpha * L_rec + beta * D_KL, with L_rec = sum over features of (x - x_hat)^2.
    """

    def __init__(
        self,
        in_dim: int,
        latent: int = 32,
        hidden: int = 256,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar, sample: bool = True):
        # ponytail: clamp both mu and logvar. logvar upper bound was too loose
        # (10 -> std=exp(5)~148 -> z up to ~454 -> decoder overflow -> loss spike).
        # Tightened to logvar[-5,3] -> std[0.082,4.48] -> z ~ [-10,~23];
        # mu clamp +-10 keeps posterior mean bounded. Full window=96 + 300ep stable.
        logvar = torch.clamp(logvar, -5.0, 3.0)
        mu = torch.clamp(mu, -10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std if sample else mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, sample=self.training)
        return self.decode(z), mu, logvar

    def loss(self, x, recon, mu, logvar):
        recon_loss = ((recon - x) ** 2).sum(dim=1).mean()
        kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return self.alpha * recon_loss + self.beta * kld, recon_loss, kld

    @torch.no_grad()
    def reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = "sum",
        include_kld: bool = True,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Return per-sample anomaly scores.

        `sum` preserves the original Eq. 11 implementation. `mean` matches the
        LSTM/Transformer MSE scale and avoids feature-count dominated scores.
        """
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.eval()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, sample=not deterministic)
        recon = self.decode(z)
        squared_error = (recon - x) ** 2
        per_sample_recon = (
            squared_error.mean(dim=1) if reduction == "mean" else squared_error.sum(dim=1)
        )
        if not include_kld:
            return self.alpha * per_sample_recon
        per_sample_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return self.alpha * per_sample_recon + self.beta * per_sample_kld
