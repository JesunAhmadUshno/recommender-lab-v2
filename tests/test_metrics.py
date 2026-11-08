import numpy as np

from src.metrics import corrected_ndcg, exact_per_user, sampled_per_user


def test_exact_ndcg_handcomputed():
    # One user, 5 items. Scores rank items [0,1,2,3,4]; relevance on 1 and 3.
    scores = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    rel = np.array([[0.0, 1.0, 0.0, 1.0, 0.0]])
    out = exact_per_user(scores, rel, k=3)
    # top-3 = items 0,1,2 -> only item 1 relevant at position 2.
    dcg = 1.0 / np.log2(3)
    idcg = 1.0 / np.log2(2) + 1.0 / np.log2(3)
    assert abs(out["dcg"][0] - dcg) < 1e-9
    assert abs(out["ndcg"][0] - dcg / idcg) < 1e-9
    assert out["recall"][0] == 0.5
    assert out["hr"][0] == 1.0


def test_exact_excludes_users_without_positives():
    scores = np.array([[1.0, 2.0], [2.0, 1.0]])
    rel = np.array([[0.0, 0.0], [1.0, 0.0]])
    out = exact_per_user(scores, rel, k=2)
    assert np.isnan(out["ndcg"][0])
    assert out["ndcg"][1] == 1.0


def _rank_setup(n_items=3327, true_rank=300, seed=0):
    """One user whose single positive sits at the given true rank."""
    rng = np.random.default_rng(seed)
    scores = rng.random((1, n_items))
    # Force the ordering: item 0 is the positive; give it the score that
    # places exactly true_rank - 1 negatives above it.
    order = np.argsort(-scores[0])
    target = np.sort(scores[0])[::-1][true_rank - 1]
    scores[0, 0] = target + 1e-12
    # Remove accidental ties by nudging the item that held that score.
    rel = np.zeros((1, n_items))
    rel[0, 0] = 1.0
    return scores, rel


def test_sampled_rank_expectation_matches_theory():
    # Pack worked example (E8): an item at true rank 300 among n = 3,327
    # items, m = 100 sampled negatives, has expected sampled rank about
    # 1 + 100 * 299 / 3326 = 9.99.
    scores, rel = _rank_setup()
    ranks = []
    for seed in range(60):
        out = sampled_per_user(scores, rel, m=100, seed=seed)
        ranks.append(out["pos_ranks"][0][0])
    ranks = np.asarray(ranks, dtype=float)
    expected = 1 + 100 * 299 / 3326
    se = ranks.std(ddof=1) / np.sqrt(ranks.size)
    assert abs(ranks.mean() - expected) < 3 * se + 0.35


def test_corrected_rank_recovers_true_rank_region():
    # Over many sampling seeds, the corrected rank must average back to the
    # true rank 300, and whenever the sampled rank is 2 or worse the
    # corrected NDCG contribution is 0 (corrected rank is then 34 or more),
    # even though the sampled NDCG is positive in a large share of seeds.
    scores, rel = _rank_setup()
    r_hats = []
    sampled_positive = 0
    for seed in range(60):
        out = sampled_per_user(scores, rel, m=100, seed=seed)
        ranks = out["pos_ranks"]
        r_hats.append(1.0 + (ranks[0][0] - 1) * (3327 - 1) / 100)
        if out["ndcg"][0] > 0.0:
            sampled_positive += 1
        if ranks[0][0] >= 2:
            corr = corrected_ndcg(ranks, n_items=3327, m=100,
                                  n_pos_per_user=np.array([1]))
            assert corr[0] == 0.0
    r_hats = np.asarray(r_hats)
    se = r_hats.std(ddof=1) / np.sqrt(r_hats.size)
    assert abs(r_hats.mean() - 300) < 3 * se + 15
    assert sampled_positive >= 10
