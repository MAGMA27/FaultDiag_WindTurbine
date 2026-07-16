from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SeqWindowsDataset(Dataset):
    """Lazily slice sliding windows from a list of 2D feature matrices (memory-safe).

    Each matrix is [N_k, D]; windows come from consecutive rows. No large 3D tensor is
    ever materialized, so this scales to the full CARE test set on CPU.
    """

    def __init__(self, mats: Sequence[torch.Tensor | "np.ndarray"], seq_len: int):
        self.seq_len = seq_len
        self.offsets: list[tuple[torch.Tensor, int, int]] = []
        total = 0
        for m in mats:
            if not isinstance(m, torch.Tensor):
                m = torch.from_numpy(m)
            cnt = len(m) - seq_len + 1
            if cnt > 0:
                self.offsets.append((m, total, cnt))
                total += cnt
        self.n = total

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        for m, base, cnt in self.offsets:
            if idx < base + cnt:
                i = idx - base
                return m[i : i + self.seq_len]
        raise IndexError(idx)


class LSTMAE(nn.Module):
    """Sequence autoencoder over a window of consecutive feature-vectors (paper Section II.B.2).

    The encoder LSTM compresses the window into a latent code; the decoder LSTM unfolds it
    back to the full window. Anomaly score = mean squared reconstruction error per window.
    """

    def __init__(
        self,
        in_dim: int,
        seq_len: int,
        latent: int = 64,
        hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(in_dim, hidden, num_layers, batch_first=True, dropout=dropout)
        self.to_latent = nn.Linear(hidden, latent)
        self.dec_rnn = nn.LSTM(latent, hidden, num_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x):
        # x: [B, T, D]
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])  # [B, latent]
        dec_in = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # [B, T, latent]
        d, _ = self.dec_rnn(dec_in)
        return self.out(d), z

    def loss(self, x, recon):
        return ((recon - x) ** 2).mean()

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        recon, _ = self(x)
        return ((recon - x) ** 2).mean(dim=(1, 2))  # [B]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerAE(nn.Module):
    """Transformer autoencoder: self-attention over the feature-window (paper Section II.B.3).

    Input is projected to d_model, positional-encoded, passed through an encoder whose
    sequence mean pools to a latent code; the code is broadcast and decoded back via a
    second Transformer block.
    """

    def __init__(
        self,
        in_dim: int,
        seq_len: int,
        latent: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.pool = nn.Linear(d_model, latent)
        self.unpool = nn.Linear(latent, d_model)
        dec_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers)
        self.out = nn.Linear(d_model, in_dim)

    def forward(self, x):
        z = self.pool(self.encoder(self.pos(self.in_proj(x))).mean(dim=1))  # [B, latent]
        dec_in = self.unpool(z).unsqueeze(1).repeat(1, self.seq_len, 1)  # [B, T, d_model]
        return self.out(self.decoder(self.pos(dec_in))), z

    def loss(self, x, recon):
        return ((recon - x) ** 2).mean()

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        recon, _ = self(x)
        return ((recon - x) ** 2).mean(dim=(1, 2))  # [B]


if __name__ == "__main__":
    # self-check: shapes and finite errors for both sequence variants
    for cls in (LSTMAE, TransformerAE):
        m = cls(in_dim=12, seq_len=8, latent=8, hidden=16)
        x = torch.randn(4, 8, 12)
        recon, z = m(x)
        assert recon.shape == x.shape and z.shape == (4, 8)
        e = m.reconstruction_error(x)
        assert e.shape == (4,) and torch.isfinite(e).all()
    print("sequence self-check OK")
