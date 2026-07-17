import numpy as np
import torch

from faultdiagnose.evaluation.anomaly import compute_auc
from faultdiagnose.models import VAE
from faultdiagnose.training.vae_trainer import train_vae


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


def test_vae_evaluation_scores_are_deterministic():
    torch.manual_seed(0)
    model = VAE(in_dim=10, latent=4, hidden=16).eval()
    x = torch.randn(8, 10)
    assert torch.equal(model.reconstruction_error(x), model.reconstruction_error(x))


def test_compute_auc_perfect():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 0, 1])
    assert compute_auc(scores, labels) == 1.0
