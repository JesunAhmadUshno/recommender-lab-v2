"""The coverage gate (PREREGISTRATION.md section 5; new in v2).

v1's Arm F scored 1,411 users whose training histories were all empty,
because its training pool was drawn from the wrong side of a
pair-disjoint dataset split. No stage checked that the scored users
had any training history at all. This module is that check, made
mandatory and frozen:

- compute_coverage describes a split construction: of the users and
  items that carry truth (the scored sets), what fraction appear in
  the training pool actually fed to the rankers.
- check_gate applies the frozen thresholds and raises GateStop when
  one fails. A stopped run publishes the coverage table and nothing
  else from that arm.
- check_score_matrix is the post-scoring backstop: a score matrix
  with zero nonzero finite entries is the exact signature of the v1
  defect and fails the run.
"""
from __future__ import annotations

import numpy as np


class GateStop(Exception):
    """Raised when a frozen coverage threshold fails. The run stops;
    the coverage table is the output."""


def compute_coverage(train_user: np.ndarray, train_item: np.ndarray,
                     scored_users: np.ndarray, scored_items: np.ndarray,
                     n_users: int, n_items: int) -> dict:
    """Coverage of the scored sets by the training pool.

    train_user / train_item are the pool's event arrays, in the id
    space being scored. scored_users / scored_items are the users and
    items that carry at least one truth positive.
    """
    train_user = np.asarray(train_user)
    train_item = np.asarray(train_item)
    scored_users = np.asarray(scored_users)
    scored_items = np.asarray(scored_items)
    has_user = np.zeros(n_users, dtype=bool)
    if train_user.size:
        has_user[np.unique(train_user)] = True
    has_item = np.zeros(n_items, dtype=bool)
    if train_item.size:
        has_item[np.unique(train_item)] = True
    return {
        "n_scored_users": int(scored_users.size),
        "n_scored_items": int(scored_items.size),
        "scored_users_with_history": int(has_user[scored_users].sum()) if scored_users.size else 0,
        "scored_items_in_training": int(has_item[scored_items].sum()) if scored_items.size else 0,
        "user_coverage": float(has_user[scored_users].mean()) if scored_users.size else 0.0,
        "item_coverage": float(has_item[scored_items].mean()) if scored_items.size else 0.0,
    }


def check_gate(record: dict, arm: str, user_min: float, item_min: float) -> dict:
    """Apply the frozen thresholds to one coverage record. Returns the
    record with its gate outcome; raises GateStop on a failure."""
    out = dict(record)
    out["arm"] = arm
    out["user_min"] = float(user_min)
    out["item_min"] = float(item_min)
    out["gate_passed"] = bool(record["user_coverage"] >= user_min
                              and record["item_coverage"] >= item_min)
    if not out["gate_passed"]:
        raise GateStop(
            f"coverage gate stop on {arm}: user coverage "
            f"{record['user_coverage']:.4f} (min {user_min}), item coverage "
            f"{record['item_coverage']:.4f} (min {item_min})"
        )
    return out


def check_score_matrix(scores: np.ndarray, arm: str, ranker: str) -> int:
    """Post-scoring backstop: count nonzero finite entries; zero fails
    the run (the v1 signature). Returns the count."""
    s = np.asarray(scores)
    finite = np.isfinite(s)
    count = int((finite & (s != 0)).sum())
    if count == 0:
        raise GateStop(
            f"zero score matrix on {arm} for {ranker}: no nonzero finite "
            f"entries; this is the v1 defect signature and fails the run"
        )
    return count


def nonzero_row_fraction(scores: np.ndarray, rows: np.ndarray) -> float:
    """Fraction of the given rows holding at least one nonzero finite
    entry."""
    s = np.asarray(scores)[np.asarray(rows)]
    if s.shape[0] == 0:
        return 0.0
    ok = (np.isfinite(s) & (s != 0)).any(axis=1)
    return float(ok.mean())
