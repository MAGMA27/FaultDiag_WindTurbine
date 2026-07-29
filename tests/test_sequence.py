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


def test_lstm_ae_supports_mae_training_loss():
    x = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
    recon = torch.zeros_like(x)
    mse = LSTMAE(in_dim=2, seq_len=2, latent=2, hidden=4, decoder_positional="none")
    mae = LSTMAE(
        in_dim=2,
        seq_len=2,
        latent=2,
        hidden=4,
        decoder_positional="none",
        loss_type="mae",
    )
    assert torch.isclose(mse.loss(x, recon), torch.tensor(7.5))
    assert torch.isclose(mae.loss(x, recon), torch.tensor(2.5))


def test_lstm_ae_paper_architecture_forward():
    model = LSTMAE(
        in_dim=10,
        seq_len=12,
        latent=8,
        hidden=16,
        num_layers=2,
        decoder_init="state",
        decoder_positional="none",
        architecture="paper",
    )
    recon, z = model(torch.randn(4, 12, 10))
    assert recon.shape == (4, 12, 10) and z.shape == (4, 8)


def test_lstm_ae_direct_latent_forward():
    model = LSTMAE(
        in_dim=10,
        seq_len=12,
        latent=16,
        hidden=16,
        num_layers=2,
        decoder_init="state",
        decoder_positional="none",
        architecture="direct",
    )
    recon, z = model(torch.randn(4, 12, 10))
    assert recon.shape == (4, 12, 10) and z.shape == (4, 16)


def test_transformer_ae_forward_and_error():
    m = TransformerAE(in_dim=10, seq_len=12, latent=8, d_model=16)
    x = torch.randn(8, 12, 10)
    recon, z = m(x)
    assert recon.shape == x.shape and z.shape == (8, 8)
    e = m.reconstruction_error(x)
    assert e.shape == (8,) and torch.isfinite(e).all()


def test_transformer_cross_attention_forward():
    m = TransformerAE(
        in_dim=10,
        seq_len=12,
        latent=8,
        d_model=16,
        nhead=4,
        architecture="cross_attention",
    )
    x = torch.randn(4, 12, 10)
    recon, z = m(x)
    assert recon.shape == x.shape and z.shape == (4, 8)


def test_seq_train_runs():
    torch.manual_seed(0)
    mats = [np.random.randn(100, 6).astype("float32"), np.random.randn(80, 6).astype("float32")]
    m = LSTMAE(in_dim=6, seq_len=10, latent=4, hidden=16)
    from faultdiagnose.training.seq_trainer import train_seq_ae

    losses = train_seq_ae(m, mats, seq_len=10, epochs=2, batch_size=32, verbose=False)
    assert len(losses) == 2 and all(np.isfinite(losses))


def test_sequence_window_cache_round_trip(tmp_path):
    mats = [torch.randn(9, 3), torch.randn(7, 3)]
    cache = tmp_path / "windows.pt"
    first = SeqWindowsDataset(mats, seq_len=4, cache_path=cache)
    second = SeqWindowsDataset(mats, seq_len=4, cache_path=cache)
    assert len(first) == len(second) == 10
    assert torch.equal(first[3], second[3])
