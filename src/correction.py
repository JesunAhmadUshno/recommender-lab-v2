"""Off-policy estimation on the semi-synthetic exposure arm.

The small matrix is fully observed, so true relevance is known. A logging
policy is simulated: item propensity p_i = max(floor, c * (n_i / n_max) ^
gamma), with n_i the item exposure count in the training window and c set
so the mean propensity equals the big-matrix density (design pack 4.4).
Exposure draws are Bernoulli(p_i). Every estimator below sees only the
observed cells; the truth uses all cells. This arm is semi-synthetic by
construction and is labelled that way in every claim.
"""
from __future__ import annotations

import numpy as np

DISCOUNTS10 = 1.0 / np.log2(np.arange(2, 12))


def make_propensities(item_exposures: np.ndarray, gamma: float,
                      target_mean: float = 0.163, floor: float = 0.01) -> tuple[np.ndarray, float]:
    n = np.asarray(item_exposures, dtype=np.float64)
    base = (n / max(n.max(), 1.0)) ** gamma
    c = target_mean / max(base.mean(), 1e-12)
    p = np.maximum(floor, c * base)
    return np.minimum(p, 1.0), float(c)


def estimate_propensities(observed_counts: np.ndarray, n_users: int,
                          floor: float = 0.01) -> np.ndarray:
    """MLE propensity from one exposure draw: exposures per item / users."""
    p = np.asarray(observed_counts, dtype=np.float64) / max(n_users, 1)
    return np.clip(p, floor, 1.0)


def topk_lists(scores: np.ndarray, k: int = 10) -> np.ndarray:
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, part], axis=1, kind="stable")
    return part[rows, order]


def estimator_dcgs(top_lists: np.ndarray, rel: np.ndarray, exposed: np.ndarray,
                   propensity: np.ndarray, imputed: np.ndarray | None = None) -> dict:
    """Per-user DCG@10 estimates from observed cells only.

    top_lists: (U, 10) item ids per ranker. rel: full truth relevance.
    exposed: boolean matrix the same shape as rel, the simulated log.
    propensity: per-item propensity used by the estimator (true or
    estimated). imputed: optional direct-model relevance predictions with
    the same shape as rel, used by DR.
    """
    U, k = top_lists.shape
    rows = np.arange(U)[:, None]
    items = top_lists
    rel_top = rel[rows, items]
    exp_top = exposed[rows, items]
    p_top = np.maximum(propensity[items], 1e-9)
    d = DISCOUNTS10[:k]
    w = exp_top / p_top
    truth = (rel_top * d).sum(axis=1)
    naive = (exp_top * rel_top * d).sum(axis=1)
    ips = (w * rel_top * d).sum(axis=1)
    wsum = (w * d).sum(axis=1)
    snips_total_num = ips.sum()
    snips_total_den = wsum.sum()
    d_total = d.sum()
    snips_mean = snips_total_num / snips_total_den * d_total if snips_total_den > 0 else 0.0
    out = {"truth": truth, "naive": naive, "ips": ips,
           "snips_mean_dcg": float(snips_mean)}
    if imputed is not None:
        m_top = imputed[rows, items]
        dr = (d * (m_top + w * (rel_top - m_top))).sum(axis=1)
        out["dr"] = dr
    return out


class LogisticImputer:
    """Direct model for DR: P(positive | user, item) from training-window
    features only: log item exposure, item positive rate, user positive
    rate. Plain gradient descent, deterministic."""

    def __init__(self, iters: int = 300, lr: float = 0.5, l2: float = 1e-4):
        self.iters = iters
        self.lr = lr
        self.l2 = l2

    @staticmethod
    def _features(user_pos_rate: np.ndarray, item_exposure: np.ndarray,
                  item_pos_rate: np.ndarray, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.column_stack([
            np.ones(users.size),
            np.log1p(item_exposure[items]),
            item_pos_rate[items],
            user_pos_rate[users],
        ])

    def fit(self, users: np.ndarray, items: np.ndarray, labels: np.ndarray,
            user_pos_rate: np.ndarray, item_exposure: np.ndarray,
            item_pos_rate: np.ndarray):
        Xf = self._features(user_pos_rate, item_exposure, item_pos_rate, users, items)
        self.mu = Xf[:, 1:].mean(axis=0)
        self.sd = Xf[:, 1:].std(axis=0) + 1e-9
        Xf[:, 1:] = (Xf[:, 1:] - self.mu) / self.sd
        y = labels.astype(np.float64)
        w = np.zeros(Xf.shape[1])
        n = Xf.shape[0]
        for _ in range(self.iters):
            z = Xf @ w
            p = 1.0 / (1.0 + np.exp(-z))
            grad = Xf.T @ (p - y) / n + self.l2 * w
            w -= self.lr * grad
        self.w = w
        self.user_pos_rate = user_pos_rate
        self.item_exposure = item_exposure
        self.item_pos_rate = item_pos_rate
        return self

    def predict_matrix(self, n_users: int, n_items: int) -> np.ndarray:
        users = np.repeat(np.arange(n_users), n_items)
        items = np.tile(np.arange(n_items), n_users)
        Xf = self._features(self.user_pos_rate, self.item_exposure,
                            self.item_pos_rate, users, items)
        Xf[:, 1:] = (Xf[:, 1:] - self.mu) / self.sd
        z = Xf @ self.w
        return (1.0 / (1.0 + np.exp(-z))).reshape(n_users, n_items).astype(np.float32)
