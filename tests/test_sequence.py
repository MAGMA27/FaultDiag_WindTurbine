import numpy as np
import torch

from faultdiagnose.models import LSTMAE, SeqWindowsDataset, TransformerAE


def test_seq_windows_dataset_slices():
    mats = [torch.randn(30, 5), torch.randn(12, 5)]
    ds = SeqWindowsDataset(mats, seq_len=8)
    assert len(ds) == (30 - 8 + 1) + (12 - 8 + 1)
    x = ds[0]
    assert x.shape == (8, 5)


def test_lstm_ae_forward_and_error():
    m = LSTMAE(in_dim=10, seq_len=12, latent=8, hidden=16)
    x = torch.randn(8, 12, 10)
    recon, z = m(x)
    assert recon.shape == x.shape and z.shape == (8, 8)
    e = m.reconstruction_error(x)
    assert e.shape == (8,) and torch.isfinite(e).all()


def test_transformer_ae_forward_and_error():
    m = TransformerAE(in_dim=10, seq_len=12, latent=8, d_model=16)
    x = torch.randn(8, 12, 10)
    recon, z = m(x)
    assert recon.shape == x.shape and z.shape == (8, 8)
    e = m.reconstruction_error(x)
    assert e.shape == (8,) and torch.isfinite(e).all()


def test_seq_train_runs():
    torch.manual_seed(0)
    mats = [np.random.randn(100, 6).astype("float32"), np.random.randn(80, 6).astype("float32")]
    m = LSTMAE(in_dim=6, seq_len=10, latent=4, hidden=16)
    from faultdiagnose.training.seq_trainer import train_seq_ae

    losses = train_seq_ae(m, mats, seq_len=10, epochs=2, batch_size=32, verbose=False)
    assert len(losses) == 2 and all(np.isfinite(losses))
