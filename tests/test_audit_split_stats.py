import numpy as np

from src.audit import cold_start_mask, gini, leakage_audit, popularity_report
from src.split import first_time_mask, random_masks, temporal_cutoffs, temporal_masks
from src.stats import holm_adjust, kendall_tau_b, paired_bootstrap


def test_temporal_masks_partition_and_order():
    ts = np.arange(100, dtype=float)
    t1, t2 = temporal_cutoffs(ts, 0.8, 0.9)
    tr, va, te = temporal_masks(ts, t1, t2)
    assert tr.sum() + va.sum() + te.sum() == 100
    assert ts[tr].max() < ts[te].min()


def test_random_masks_partition():
    tr, va, te = random_masks(1000, seed=0)
    assert tr.sum() + va.sum() + te.sum() == 1000
    assert not (tr & te).any()


def test_first_time_mask():
    seen = first_time_mask(np.array([0, 0]), np.array([1, 2]),
                           np.array([0, 0, 1]), np.array([1, 3, 1]), n_items=5)
    assert seen.tolist() == [False, True, True]


def test_leakage_audit_passes_clean_and_flags_planted():
    n_items = 10
    tr_u = np.array([0, 0, 1])
    tr_i = np.array([1, 2, 3])
    tr_t = np.array([1.0, 2.0, 3.0])
    te_u = np.array([0, 1])
    te_i = np.array([4, 5])
    te_t = np.array([10.0, 11.0])
    clean = leakage_audit(tr_u, tr_i, te_u, te_i, tr_t, te_t, n_items)
    assert clean["pair_overlap_ok"] and clean["time_order_ok"]
    # Plant a leak: a test pair copied into train, with a late timestamp.
    tr_u2 = np.append(tr_u, 0)
    tr_i2 = np.append(tr_i, 4)
    tr_t2 = np.append(tr_t, 12.0)
    dirty = leakage_audit(tr_u2, tr_i2, te_u, te_i, tr_t2, te_t, n_items)
    assert not dirty["pair_overlap_ok"]
    assert not dirty["time_order_ok"]


def test_gini_bounds():
    assert gini(np.ones(10)) == 0.0
    assert gini(np.array([0.0] * 9 + [10.0])) > 0.8


def test_popularity_report_shares():
    top = np.array([[0, 1, 2], [0, 1, 3]])
    exposure = np.array([100.0, 90.0, 10.0, 5.0, 1.0])
    rep = popularity_report(top, exposure)
    # Items 0 and 1 are the top-20% (1 of 5 rounds to 1 -> top1 = item 0).
    assert rep["n_recommendations"] == 6
    assert 0.0 <= rep["top10pct_share"] <= 1.0
    assert rep["coverage"] == 4 / 5


def test_cold_start_mask():
    mask = cold_start_mask(np.array([0, 1]), np.array([7, 8]),
                           np.array([10, 2]), np.array([0, 0, 0, 0, 0, 0, 0, 3, 9]),
                           threshold=5)
    assert mask["item_cold"].tolist() == [True, False]
    assert mask["user_cold"].tolist() == [False, True]


def test_kendall_tau_b_values():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert kendall_tau_b(a, a) == 1.0
    assert kendall_tau_b(a, a[::-1]) == -1.0
    # Pack worked example: 5 rankers, one flipped pair -> tau = 0.8.
    b = np.array([1.0, 2.0, 3.0, 5.0, 4.0])
    assert abs(kendall_tau_b(a, b) - 0.8) < 1e-12


def test_holm_adjust_known_example():
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert adj == [0.03, 0.06, 0.06]


def test_paired_bootstrap_detects_positive_difference():
    rng = np.random.default_rng(0)
    diff = rng.normal(0.2, 1.0, size=500)
    out = paired_bootstrap(diff, n_boot=500, seed=0)
    assert out["one_sided_lo"] > 0
    assert out["p_two_sided"] < 0.05
