"""The coverage gate (PREREGISTRATION.md section 5).

The centrepiece fixture reproduces v1's Arm F construction in
miniature: a training pool drawn from other users only, so every
scored user has an empty history. v1 published verdicts on exactly
that construction because nothing checked it. The gate must refuse.
"""
import numpy as np
import pytest

from src.coverage import (GateStop, check_gate, check_score_matrix,
                          compute_coverage, nonzero_row_fraction)


def _v1_defect_fixture():
    """Miniature of v1 Arm F: scored users 0..2, items 0..4. The pool
    holds many events on the scored items, all by other users (3..5),
    exactly as the big-matrix pool held 10,272,525 events on small
    items, none by the 1,411 scored users."""
    pool_user = np.array([3, 3, 4, 4, 5, 5, 3, 4, 5, 3])
    pool_item = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    scored_users = np.array([0, 1, 2])
    scored_items = np.array([0, 1, 2, 3, 4])
    return pool_user, pool_item, scored_users, scored_items


def test_gate_refuses_the_v1_construction():
    pu, pi, su, si = _v1_defect_fixture()
    rec = compute_coverage(pu, pi, su, si, n_users=6, n_items=5)
    # The v1 signature: item coverage is perfect, user coverage is 0.
    assert rec["item_coverage"] == 1.0
    assert rec["user_coverage"] == 0.0
    with pytest.raises(GateStop):
        check_gate(rec, "arm_f_fixture", user_min=0.50, item_min=0.10)
    with pytest.raises(GateStop):
        check_gate(rec, "arm_f_fixture", user_min=0.95, item_min=0.10)


def test_gate_passes_a_covered_construction():
    # Same scored sets, but the pool now contains the scored users' own
    # events (the v2 construction: pool and truth from one matrix).
    pu = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    pi = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    su = np.array([0, 1, 2])
    si = np.array([0, 1, 2, 3, 4])
    rec = compute_coverage(pu, pi, su, si, n_users=6, n_items=5)
    assert rec["user_coverage"] == 1.0
    assert rec["item_coverage"] == 1.0
    out = check_gate(rec, "arm_f_fixture", user_min=0.95, item_min=0.10)
    assert out["gate_passed"]


def test_item_floor_catches_a_wrong_item_space():
    # Pool events land on items 5..9 while the truth lives on 0..4:
    # a mis-mapped item space. User coverage is fine; the item floor
    # is what fires.
    pu = np.array([0, 1, 2, 0, 1, 2])
    pi = np.array([5, 6, 7, 8, 9, 5])
    rec = compute_coverage(pu, pi, np.array([0, 1, 2]), np.array([0, 1, 2, 3, 4]),
                           n_users=3, n_items=10)
    assert rec["user_coverage"] == 1.0
    assert rec["item_coverage"] == 0.0
    with pytest.raises(GateStop):
        check_gate(rec, "arm_f_fixture", user_min=0.50, item_min=0.10)


def test_zero_score_matrix_fails_the_backstop():
    with pytest.raises(GateStop):
        check_score_matrix(np.zeros((4, 7), dtype=np.float32), "arm_f", "ItemKNN")
    with pytest.raises(GateStop):
        check_score_matrix(np.full((4, 7), -np.inf, dtype=np.float32), "arm_f", "EASE")
    ok = np.zeros((4, 7), dtype=np.float32)
    ok[0, 3] = 0.5
    assert check_score_matrix(ok, "arm_f", "ItemKNN") == 1


def test_nonzero_row_fraction():
    s = np.zeros((4, 5), dtype=np.float32)
    s[0, 1] = 1.0
    s[2, 4] = -2.0
    s[3, :] = -np.inf
    assert nonzero_row_fraction(s, np.array([0, 1, 2, 3])) == 0.5
    assert nonzero_row_fraction(s, np.array([0, 2])) == 1.0
