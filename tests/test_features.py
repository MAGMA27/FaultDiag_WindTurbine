import numpy as np
import pandas as pd
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
from faultdiagnose.features.audit import audit_feature_matrix
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
    assert feats.shape[1] == len(cols) * 7  # mean/std/skew/kurt/deriv/deriv2 + fft
    clean = feats.dropna()
    assert len(clean) > 1000
    assert np.isfinite(clean.to_numpy()).all()


def test_stat_aware_profile_respects_statistic_granularity():
    frame = pd.DataFrame(
        {
            "sensor_1_avg": np.arange(30, dtype=float),
            "sensor_1_std": np.linspace(0.0, 1.0, 30),
        }
    )
    features = engineer_features(
        frame,
        window=6,
        columns=list(frame.columns),
        use_fft=True,
        feature_profile="stat_aware",
    )
    assert features.shape[1] == 11  # avg: raw + 7 dynamic; std: raw + mean + derivative
    assert "sensor_1_avg_fft" in features
    assert "sensor_1_std_raw" in features
    assert "sensor_1_std_std" not in features


def test_raw_stat_compact_keeps_raw_statistics_without_secondary_features():
    frame = pd.DataFrame(
        {
            "sensor_1_avg": np.arange(30, dtype=float),
            "sensor_1_std": np.linspace(0.0, 1.0, 30),
        }
    )
    features = engineer_features(
        frame,
        window=6,
        columns=list(frame.columns),
        use_fft=True,
        feature_profile="raw_stat_compact",
    )
    assert features.shape[1] == 9  # avg: raw + 7 dynamic; std: raw only
    assert "sensor_1_avg_fft" in features
    assert "sensor_1_std_raw" in features
    assert "sensor_1_std_mean" not in features
    assert "sensor_1_std_deriv" not in features


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


def test_feature_audit_drops_duplicate_and_clusters_correlations():
    base = np.arange(20, dtype=float)
    features = pd.DataFrame(
        {
            "sensor_1_avg_mean": base,
            "sensor_2_mean": base * 2,
            "sensor_3_std": base.copy(),
            "constant_mean": np.ones_like(base),
        }
    )
    stats, clusters, selected = audit_feature_matrix(
        features, correlation_threshold=0.999, drop_near_constant=True
    )
    assert "constant_mean" not in selected
    assert selected == ["sensor_1_avg_mean"]
    assert len(clusters) == 2
    duplicate_reason = stats.loc[stats["feature"] == "sensor_3_std", "drop_reason"].item()
    assert duplicate_reason.startswith("duplicate_of:")


def test_feature_audit_protects_near_constant_features_by_default():
    features = pd.DataFrame({"constant_mean": np.ones(20), "varying_mean": np.arange(20)})
    _, _, selected = audit_feature_matrix(features)
    assert set(selected) == {"constant_mean", "varying_mean"}
