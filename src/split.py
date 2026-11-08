"""Split protocols.

Arm T: one global timeline. Cut points at the 80th and 90th percentiles of
the event timestamps. Train t < q80, validation q80 <= t < q90, test
t >= q90. Test positives are first-time positives: a (user, item) pair that
is positive in the test window and does not appear in the training window.

Arm F (v2, corrected): the identical temporal protocol executed inside
the KuaiRec small matrix, so the training histories and the truth set
come from the same matrix (see PREREGISTRATION.md section 4). v1 trained
Arm F on big-matrix events restricted to small-matrix items; that pool
contains zero events by the scored users, because the two matrices are
pair-disjoint, and every learned ranker's Arm F scores were the all-zero
matrix. The functions below are unchanged from v1; only the data fed to
them differs.

Random arm: seeded 80/10/10 split of events, the leakage comparison.
"""
from __future__ import annotations

import numpy as np

from .dataio import pair_keys


def temporal_cutoffs(timestamps: np.ndarray, q1: float = 0.80, q2: float = 0.90) -> tuple[float, float]:
    # nanquantile, not quantile: the small matrix carries 181,992 rows
    # with empty timestamps, and a NaN must not poison the cutoffs of
    # the corrected Arm F. On NaN-free data (both Arm T datasets) the
    # values are identical to np.quantile, so no v1 number changes.
    return float(np.nanquantile(timestamps, q1)), float(np.nanquantile(timestamps, q2))


def temporal_masks(timestamps: np.ndarray, t1: float, t2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = timestamps < t1
    val = (timestamps >= t1) & (timestamps < t2)
    test = timestamps >= t2
    return train, val, test


def first_time_mask(train_user: np.ndarray, train_item: np.ndarray,
                    cand_user: np.ndarray, cand_item: np.ndarray,
                    n_items: int) -> np.ndarray:
    """True where a candidate (user, item) pair does not appear in train."""
    train_keys = np.unique(pair_keys(train_user, train_item, n_items))
    cand_keys = pair_keys(cand_user, cand_item, n_items)
    return ~np.isin(cand_keys, train_keys)


def random_masks(n_events: int, seed: int, fracs: tuple[float, float] = (0.80, 0.10)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_events)
    n1 = int(fracs[0] * n_events)
    n2 = n1 + int(fracs[1] * n_events)
    train = np.zeros(n_events, dtype=bool)
    val = np.zeros(n_events, dtype=bool)
    test = np.zeros(n_events, dtype=bool)
    train[perm[:n1]] = True
    val[perm[n1:n2]] = True
    test[perm[n2:]] = True
    return train, val, test
