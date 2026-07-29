from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SeqWindowsDataset(Dataset):
    """Lazily slice sliding windows from a list of 2D feature matrices (memory-safe).

    Each matrix is [N_k, D]; windows come from consecutive rows. No large 3D tensor is
    ever materialized, so this scales to the full CARE test set on CPU.
    """

    def __init__(
        self,
        mats: Sequence[torch.Tensor | "np.ndarray"],
        seq_len: int,
        cache_path: str | Path | None = None,
    ):
        self.seq_len = seq_len
        self.windows: torch.Tensor | None = None
        if cache_path is not None and Path(cache_path).exists():
            payload = torch.load(Path(cache_path), map_location="cpu", weights_only=False)
            if payload.get("seq_len") == seq_len:
                self.windows = payload["windows"]
                self.n = len(self.windows)
                return
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
        if cache_path is not None:
            if self.n == 0:
                raise ValueError("No sequence windows available for cache")
            chunks = [
                m.unfold(0, seq_len, 1).permute(0, 2, 1).contiguous()
                for m, _, _ in self.offsets
            ]
            windows = torch.cat(chunks, dim=0).to(dtype=torch.float32)
            target = Path(cache_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            torch.save({"seq_len": seq_len, "windows": windows}, tmp)
            tmp.replace(target)
            self.windows = windows

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.windows is not None:
            return self.windows[idx]
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
        decoder_init: str = "state",
        decoder_positional: str = "learned",
        loss_type: str = "mse",
        architecture: str = "symmetric",
    ):
        super().__init__()
        if decoder_init not in {"zero", "state"}:
            raise ValueError("decoder_init must be 'zero' or 'state'")
        if decoder_positional not in {"none", "learned"}:
            raise ValueError("decoder_positional must be 'none' or 'learned'")
        if loss_type not in {"mse", "mae"}:
            raise ValueError("loss_type must be 'mse' or 'mae'")
        if architecture not in {"symmetric", "paper", "direct"}:
            raise ValueError("architecture must be 'symmetric', 'paper', or 'direct'")
        if architecture == "paper" and num_layers != 2:
            raise ValueError("paper architecture requires num_layers=2")
        if architecture == "direct" and latent != hidden:
            raise ValueError("direct architecture requires latent=hidden")
        self.seq_len = seq_len
        self.decoder_init = decoder_init
        self.decoder_positional = decoder_positional
        self.loss_type = loss_type
        self.architecture = architecture
        if architecture in {"symmetric", "direct"}:
            self.encoder = nn.LSTM(in_dim, hidden, num_layers, batch_first=True, dropout=dropout)
            self.to_latent = (
                nn.Identity() if architecture == "direct" else nn.Linear(hidden, latent)
            )
            self.dec_rnn = nn.LSTM(latent, hidden, num_layers, batch_first=True, dropout=dropout)
            if decoder_init == "state":
                self.latent_to_h = nn.Linear(latent, num_layers * hidden)
                self.latent_to_c = nn.Linear(latent, num_layers * hidden)
        else:
            # Reference architecture: encoder [hidden -> latent], decoder [latent -> hidden].
            self.enc_first = nn.LSTM(in_dim, hidden, 1, batch_first=True)
            self.enc_second = nn.LSTM(hidden, latent, 1, batch_first=True)
            self.to_latent = nn.Identity()
            self.dec_first = nn.LSTM(latent, latent, 1, batch_first=True)
            self.dec_second = nn.LSTM(latent, hidden, 1, batch_first=True)
            if decoder_init == "state":
                self.latent_to_enc_h = nn.Linear(latent, latent)
                self.latent_to_enc_c = nn.Linear(latent, latent)
                self.latent_to_dec1_h = nn.Linear(latent, latent)
                self.latent_to_dec1_c = nn.Linear(latent, latent)
                self.latent_to_dec2_h = nn.Linear(latent, hidden)
                self.latent_to_dec2_c = nn.Linear(latent, hidden)
        if decoder_positional == "learned":
            self.decoder_pos = nn.Parameter(torch.zeros(1, seq_len, latent))
            nn.init.normal_(self.decoder_pos, mean=0.0, std=0.02)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x):
        # x: [B, T, D]
        if self.architecture in {"symmetric", "direct"}:
            _, (h, _) = self.encoder(x)
            z = self.to_latent(h[-1])  # [B, latent]
        else:
            encoded, _ = self.enc_first(x)
            _, (h, _) = self.enc_second(encoded)
            z = self.to_latent(h[-1])  # [B, latent]
        dec_in = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # [B, T, latent]
        if self.decoder_positional == "learned":
            dec_in = dec_in + self.decoder_pos
        if self.architecture in {"symmetric", "direct"}:
            if self.decoder_init == "state":
                batch_size = z.shape[0]
                h0 = self.latent_to_h(z).view(
                    batch_size, self.dec_rnn.num_layers, -1
                ).transpose(0, 1)
                c0 = self.latent_to_c(z).view(
                    batch_size, self.dec_rnn.num_layers, -1
                ).transpose(0, 1)
                d, _ = self.dec_rnn(dec_in, (h0.contiguous(), c0.contiguous()))
            else:
                d, _ = self.dec_rnn(dec_in)
        else:
            if self.decoder_init == "state":
                h1 = self.latent_to_dec1_h(z).unsqueeze(0)
                c1 = self.latent_to_dec1_c(z).unsqueeze(0)
                d1, _ = self.dec_first(dec_in, (h1, c1))
                h2 = self.latent_to_dec2_h(z).unsqueeze(0)
                c2 = self.latent_to_dec2_c(z).unsqueeze(0)
                d, _ = self.dec_second(d1, (h2, c2))
            else:
                d1, _ = self.dec_first(dec_in)
                d, _ = self.dec_second(d1)
        return self.out(d), z

    def loss(self, x, recon):
        error = recon - x
        if self.loss_type == "mae":
            return error.abs().mean()
        return error.square().mean()

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
        architecture: str = "broadcast",
    ):
        super().__init__()
        if architecture not in {"broadcast", "cross_attention"}:
            raise ValueError("architecture must be 'broadcast' or 'cross_attention'")
        self.seq_len = seq_len
        self.architecture = architecture
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.pool = nn.Linear(d_model, latent)
        self.unpool = nn.Linear(latent, d_model)
        if architecture == "broadcast":
            dec_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
            )
            self.decoder = nn.TransformerEncoder(dec_layer, num_layers)
        else:
            dec_layer = nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
            )
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.out = nn.Linear(d_model, in_dim)

    def forward(self, x):
        memory = self.encoder(self.pos(self.in_proj(x)))
        z = self.pool(memory.mean(dim=1))  # [B, latent]
        dec_in = self.unpool(z).unsqueeze(1).repeat(1, self.seq_len, 1)  # [B, T, d_model]
        target = self.pos(dec_in)
        if self.architecture == "broadcast":
            decoded = self.decoder(target)
        else:
            decoded = self.decoder(target, memory)
        return self.out(decoded), z

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
