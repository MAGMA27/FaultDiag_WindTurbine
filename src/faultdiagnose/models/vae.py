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
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, recon, mu, logvar):
        recon_loss = ((recon - x) ** 2).sum(dim=1).mean()
        kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return self.alpha * recon_loss + self.beta * kld, recon_loss, kld

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        recon, _, _ = self(x)
        return ((recon - x) ** 2).sum(dim=1)
