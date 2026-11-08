import numpy as np

from src.correction import (estimate_propensities, estimator_dcgs,
                            make_propensities, topk_lists)


def test_propensities_gamma_zero_is_uniform():
    exposures = np.array([10.0, 100.0, 1000.0])
    p, c = make_propensities(exposures, gamma=0.0, target_mean=0.163)
    assert np.allclose(p, 0.163)
    p1, _ = make_propensities(exposures, gamma=1.0, target_mean=0.163)
    assert p1.min() >= 0.01
    assert p1.max() <= 1.0
    assert p1[2] > p1[0]


def test_estimate_propensities_clips():
    p = estimate_propensities(np.array([0.0, 50.0, 200.0]), n_users=100)
    assert p[0] == 0.01
    assert p[1] == 0.5
    assert p[2] == 1.0


def test_topk_lists():
    scores = np.array([[0.1, 0.9, 0.5, 0.3]])
    top = topk_lists(scores, k=2)
    assert top.tolist() == [[1, 2]]


def test_ips_is_unbiased_on_synthetic_truth():
    # Pack estimator unit test (section 4.8): on a synthetic matrix with
    # known truth, the mean IPS estimate over many draws lies within 3
    # standard errors of the truth.
    rng = np.random.default_rng(0)
    n_users, n_items = 40, 60
    rel = (rng.random((n_users, n_items)) < 0.2).astype(np.float64)
    scores = rel + rng.random((n_users, n_items)) * 0.5
    top = topk_lists(scores, k=10)
    propensity = np.full(n_items, 0.3)
    estimates = []
    truth = None
    for draw in range(1000):
        exposed = rng.random((n_users, n_items)) < propensity[None, :]
        out = estimator_dcgs(top, rel, exposed, propensity)
        estimates.append(out["ips"].mean())
        truth = out["truth"].mean()
    estimates = np.asarray(estimates)
    se = estimates.std(ddof=1) / np.sqrt(estimates.size)
    assert abs(estimates.mean() - truth) < 3 * se + 1e-9
    # The naive estimator must be biased downward on the same draws.
    exposed = rng.random((n_users, n_items)) < propensity[None, :]
    out = estimator_dcgs(top, rel, exposed, propensity)
    assert out["naive"].mean() < truth
