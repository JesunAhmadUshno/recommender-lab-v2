"""Exact and sampled ranking metrics.

Exact metrics rank the full candidate set. Sampled metrics follow the
Krichene and Rendle protocol: the candidate set is the user's test positives
plus m uniformly sampled negatives, and the metric is computed inside that
set. The correction maps a sampled rank back to an estimated full-set rank
with r_hat = 1 + (r_tilde - 1) * (n - 1) / m and computes the metric on the
corrected ranks (design pack section 4.4).
"""
from __future__ import annotations

import numpy as np

K = 10
DISCOUNTS = 1.0 / np.log2(np.arange(2, K + 2))  # position 1..K discounts


def _topk_order(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k scores per row, descending. Ties broken by index."""
    if scores.shape[1] <= k:
        return np.argsort(-scores, axis=1, kind="stable")
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    part_scores = scores[rows, part]
    order = np.argsort(-part_scores, axis=1, kind="stable")
    return part[rows, order]


def exact_per_user(scores: np.ndarray, rel: np.ndarray, k: int = K) -> dict:
    """Per-user exact NDCG, DCG (unnormalised), Recall, MRR, HR at k.

    rel is a binary relevance matrix with the same shape as scores. Users
    with zero relevant items get NaN for NDCG/Recall/MRR and are excluded by
    callers through the returned mask.

    Rows are processed in blocks of 512 when the matrix is large, so the
    float64 working copies stay small on wide catalogues. Every per-user
    quantity is row-independent, so blocking changes no value.
    """
    scores = np.asarray(scores)
    rel = np.asarray(rel)
    if scores.shape[0] > 512:
        parts = [_exact_block(scores[s:s + 512], rel[s:s + 512], k)
                 for s in range(0, scores.shape[0], 512)]
        return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    return _exact_block(scores, rel, k)


def _exact_block(scores: np.ndarray, rel: np.ndarray, k: int = K) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    rel = np.asarray(rel, dtype=np.float64)
    n_users = scores.shape[0]
    top = _topk_order(scores, k)
    rows = np.arange(n_users)[:, None]
    gains = rel[rows, top]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = gains @ discounts
    n_pos = rel.sum(axis=1)
    ideal = np.minimum(n_pos, k)
    idcg = np.array([discounts[: int(i)].sum() if i > 0 else 0.0 for i in ideal])
    ndcg = np.where(n_pos > 0, dcg / np.maximum(idcg, 1e-12), np.nan)
    recall = np.where(n_pos > 0, gains.sum(axis=1) / np.maximum(n_pos, 1), np.nan)
    hr = (gains.sum(axis=1) > 0).astype(np.float64)
    # MRR at k: reciprocal rank of the first relevant item inside the top-k.
    has_hit = gains.sum(axis=1) > 0
    first_idx = np.argmax(gains > 0, axis=1)
    mrr = np.where(has_hit, 1.0 / (first_idx + 1.0), 0.0)
    mrr = np.where(n_pos > 0, mrr, np.nan)
    return {"ndcg": ndcg, "dcg": dcg, "recall": recall, "mrr": mrr, "hr": hr,
            "n_pos": n_pos}


def sampled_per_user(scores: np.ndarray, rel: np.ndarray, m: int = 100,
                     seed: int = 0, k: int = K,
                     exclude: np.ndarray | None = None) -> dict:
    """Per-user sampled NDCG at k and the sampled ranks of the positives.

    Candidate set per user: all test positives plus m negatives sampled
    uniformly without replacement from the user's non-relevant items. The
    pack states both "100 negatives per test positive" and, in its worked
    example and correction formula, m = 100 negatives in total; the build
    follows the worked example (m negatives per user), recorded in
    DECISIONS.md. Returns sampled
    NDCG, plus for the correction: the sampled rank of every positive and
    the candidate negative count actually used per user.
    """
    rng = np.random.default_rng(seed)
    # Only single rows of scores are read below, so the matrix is kept in
    # its stored dtype: no full-size float64 copy is materialised. Row
    # values are unchanged, so sampled ranks are identical.
    scores = np.asarray(scores)
    rel = np.asarray(rel)
    n_users, n_items = scores.shape
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    sampled_ndcg = np.full(n_users, np.nan)
    pos_sampled_ranks: list[np.ndarray] = []
    neg_used = np.zeros(n_users, dtype=int)
    for u in range(n_users):
        pos = np.flatnonzero(rel[u] > 0)
        if pos.size == 0:
            pos_sampled_ranks.append(np.empty(0, dtype=int))
            continue
        if exclude is not None:
            neg_pool = np.flatnonzero((rel[u] == 0) & (~exclude[u]))
        else:
            neg_pool = np.flatnonzero(rel[u] == 0)
        want = min(m, neg_pool.size)
        negs = rng.choice(neg_pool, size=want, replace=False) if want > 0 else neg_pool[:0]
        neg_used[u] = want
        cand = np.concatenate([pos, negs])
        cand_scores = scores[u, cand]
        order = np.argsort(-cand_scores, kind="stable")
        ranked = cand[order]
        ranks = np.empty(cand.size, dtype=int)
        ranks[order] = np.arange(1, cand.size + 1)
        pos_ranks = ranks[: pos.size]
        pos_sampled_ranks.append(pos_ranks.copy())
        gains = (pos_ranks <= k).astype(np.float64)
        dcg = float((gains / np.log2(pos_ranks + 1.0)).sum())
        idcg = float(discounts[: min(pos.size, k)].sum())
        sampled_ndcg[u] = dcg / idcg if idcg > 0 else np.nan
    return {"ndcg": sampled_ndcg, "pos_ranks": pos_sampled_ranks,
            "neg_used": neg_used}


def corrected_ndcg(pos_ranks_list: list[np.ndarray], n_items: int, m: int,
                   n_pos_per_user: np.ndarray, k: int = K) -> np.ndarray:
    """Rank-mapping corrected NDCG at k from sampled ranks.

    r_hat = 1 + (r_tilde - 1) * (n - 1) / m, with n the full item count and
    m the number of sampled negatives per user in the protocol. A
    positive contributes its discount at its corrected rank when the
    corrected rank is at most k; the ideal DCG uses all test positives.
    """
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    out = np.full(len(pos_ranks_list), np.nan)
    scale = (n_items - 1) / m
    for u, ranks in enumerate(pos_ranks_list):
        n_pos = int(n_pos_per_user[u])
        if n_pos == 0 or ranks.size == 0:
            continue
        r_hat = 1.0 + (ranks - 1) * scale
        contrib = np.where(r_hat <= k, 1.0 / np.log2(r_hat + 1.0), 0.0)
        idcg = float(discounts[: min(n_pos, k)].sum())
        out[u] = float(contrib.sum()) / idcg if idcg > 0 else np.nan
    return out
