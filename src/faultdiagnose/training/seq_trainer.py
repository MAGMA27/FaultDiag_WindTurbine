from __future__ import annotations

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from faultdiagnose.models.sequence import SeqWindowsDataset


def train_seq_ae(
    model,
    zmats,
    seq_len: int,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> list[float]:
    """Train a sequence autoencoder on standardized per-dataset 2D matrices (memory-safe)."""
    ds = SeqWindowsDataset(zmats, seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        n = 0
        for xb in dl:
            xb = xb.to(device)
            opt.zero_grad()
            recon, _ = model(xb)
            loss = model.loss(xb, recon)
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            n += 1
        losses.append(tot / n)
        if verbose:
            print(f"  epoch {ep + 1}/{epochs} loss={losses[-1]:.4f}")
    return losses
