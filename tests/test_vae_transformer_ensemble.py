import numpy as np

from faultdiagnose.evaluation.ensemble import empirical_percentile_rank


def test_percentile_rank_uses_only_reference_distribution():
    scores = np.array([0.0, 1.5, np.nan, 3.0])
    ranks = empirical_percentile_rank(scores, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(ranks[[0, 1, 3]], [0.0, 1.0 / 3.0, 1.0])
    assert np.isnan(ranks[2])
