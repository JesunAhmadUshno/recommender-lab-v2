"""Statistics helpers: paired bootstrap intervals, Holm correction, Kendall tau-b.

Implemented with numpy only, so the repo has a single dependency.
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap(diff: np.ndarray, n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Paired bootstrap over users for a per-user difference array.

    Returns the mean difference, a two-sided percentile interval, a one-sided
    lower bound (1 - alpha quantile of the bootstrap means), and a two-sided
    bootstrap p-value for the null of zero mean difference.
    """
    diff = np.asarray(diff, dtype=np.float64)
    n = diff.size
    if n == 0:
        raise ValueError("paired_bootstrap needs at least one observation")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    one_sided_lo = float(np.quantile(means, alpha))
    one_sided_hi = float(np.quantile(means, 1 - alpha))
    p_low = float((means <= 0).mean())
    p_high = float((means >= 0).mean())
    p_two = float(min(1.0, 2.0 * min(p_low, p_high)))
    return {
        "mean": float(diff.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "one_sided_lo": one_sided_lo,
        "one_sided_hi": one_sided_hi,
        "p_two_sided": p_two,
        "n": int(n),
        "n_boot": int(n_boot),
    }


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj.tolist()


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b between two equal-length score vectors."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n != y.size or n < 2:
        raise ValueError("kendall_tau_b needs two vectors of length >= 2")
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    n_pairs = n * (n - 1) / 2
    denom = np.sqrt((n_pairs - ties_x) * (n_pairs - ties_y))
    if denom == 0:
        return 0.0
    return float((concordant - discordant) / denom)
