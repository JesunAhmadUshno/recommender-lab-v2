"""Leakage audit, popularity-bias report, cold-start slicing."""
from __future__ import annotations

import numpy as np

from .dataio import pair_keys


def leakage_audit(train_user: np.ndarray, train_item: np.ndarray,
                  test_user: np.ndarray, test_item: np.ndarray,
                  train_ts: np.ndarray, test_ts: np.ndarray,
                  n_items: int) -> dict:
    """Checks for the temporal arm. pair_overlap counts test pairs that also
    appear in train (the temporal arm uses first-time test positives, so the
    frozen protocol requires zero). time_order requires the last training
    event to precede the first test event."""
    train_keys = np.unique(pair_keys(train_user, train_item, n_items))
    test_keys = pair_keys(test_user, test_item, n_items)
    overlap = int(np.isin(test_keys, train_keys).sum())
    return {
        "test_pairs": int(test_keys.size),
        "pair_overlap": overlap,
        "pair_overlap_ok": bool(overlap == 0),
        "max_train_ts": float(train_ts.max()) if train_ts.size else None,
        "min_test_ts": float(test_ts.min()) if test_ts.size else None,
        "time_order_ok": bool(train_ts.size and test_ts.size and train_ts.max() < test_ts.min()),
    }


def gini(counts: np.ndarray) -> float:
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def popularity_report(top_lists: np.ndarray, item_train_exposure: np.ndarray) -> dict:
    """Beyond-accuracy report for one ranker's top-10 lists (design 4.4)."""
    flat = top_lists.ravel()
    total = flat.size
    counts = np.bincount(flat, minlength=item_train_exposure.size).astype(np.float64)
    order = np.argsort(-item_train_exposure, kind="stable")
    n_items = item_train_exposure.size
    top1 = order[: max(1, int(round(0.01 * n_items)))]
    top10 = order[: max(1, int(round(0.10 * n_items)))]
    bottom80 = order[int(0.20 * n_items):]
    rec_exposure = item_train_exposure[flat]
    return {
        "top1pct_share": float(np.isin(flat, top1).sum() / total),
        "top10pct_share": float(np.isin(flat, top10).sum() / total),
        "gini": gini(counts),
        "coverage": float((counts > 0).mean()),
        "mean_recommended_exposure": float(rec_exposure.mean()),
        "long_tail_share": float(np.isin(flat, bottom80).sum() / total),
        "n_recommendations": int(total),
    }


def cold_start_mask(test_user: np.ndarray, test_item: np.ndarray,
                    train_user_counts: np.ndarray, train_item_counts: np.ndarray,
                    threshold: int = 5) -> dict:
    """Cold-start slices (filed pack): items or users with fewer than
    `threshold` training interactions."""
    item_cold = train_item_counts[test_item] < threshold
    user_cold = train_user_counts[test_user] < threshold
    return {"item_cold": item_cold, "user_cold": user_cold,
            "either_cold": item_cold | user_cold}
