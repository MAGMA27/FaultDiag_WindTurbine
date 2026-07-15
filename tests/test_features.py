import numpy as np
import pytest
import torch

from faultdiagnose.data import load_care
from faultdiagnose.evaluation.anomaly import compute_auc
from faultdiagnose.features import (
    apply_standardizer,
    engineer_features,
    fit_standardizer,
    select_feature_columns,
)
from faultdiagnose.models import VAE
from faultdiagnose.training.vae_trainer import train_vae

OUT = load_care.OUT_DEFAULT
pytestmark = pytest.mark.skipif(not OUT.exists(), reason="run convert first")


def _sample():
    ds_id = load_care.list_datasets(OUT, "A")[0]
    df = load_care.load_dataset(OUT, "A", ds_id).head(1500)
    cols = select_feature_columns(df)[:12]
    return df, cols


def test_engineer_shapes_and_finite():
    df, cols = _sample()
    feats = engineer_features(df, window=24, columns=cols, use_fft=True)
    assert feats.shape[1] == len(cols) * 6
    clean = feats.dropna()
    assert len(clean) > 1000
    assert np.isfinite(clean.to_numpy()).all()


def test_vae_forward_and_loss():
    m = VAE(in_dim=10, latent=4, hidden=16)
    x = torch.randn(8, 10)
    recon, mu, logvar = m(x)
    assert recon.shape == x.shape
    loss, rl, kl = m.loss(x, recon, mu, logvar)
    assert torch.isfinite(loss)


def test_vae_train_runs():
    torch.manual_seed(0)
    X = np.random.randn(200, 10).astype("float32")
    m = VAE(in_dim=10, latent=4, hidden=16)
    losses = train_vae(m, X, epochs=2, batch_size=32, verbose=False)
    assert len(losses) == 2 and all(np.isfinite(losses))


def test_standardizer_zero_mean():
    df, cols = _sample()
    feats = engineer_features(df, window=24, columns=cols, use_fft=True).dropna()
    mean, std = fit_standardizer(feats)
    z = apply_standardizer(feats, mean, std)
    assert np.allclose(z.mean(), 0.0, atol=1e-5)


def test_compute_auc_perfect():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 0, 1])
    assert compute_auc(scores, labels) == 1.0
