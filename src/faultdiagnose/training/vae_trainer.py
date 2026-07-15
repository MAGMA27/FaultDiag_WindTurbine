from __future__ import annotations

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from faultdiagnose.models import VAE


def train_vae(model: VAE, X, epochs: int = 30, batch_size: int = 256, lr: float = 1e-3,
              device: str = "cpu", verbose: bool = True) -> list[float]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=True)
    losses: list[float] = []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        n = 0
        for xb in dl:
            xb = xb.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, _, _ = model.loss(xb, recon, mu, logvar)
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            n += 1
        losses.append(tot / n)
        if verbose:
            print(f"  epoch {ep + 1}/{epochs} loss={losses[-1]:.4f}")
    return losses
