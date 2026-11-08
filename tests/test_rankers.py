import numpy as np
import pytest

from src.metrics import exact_per_user
from src.rankers import EASE, IALS, ItemKNN, Popularity, RandomRanker, SASRec


def cluster_data(seed=0, n_users=240, n_items=60, n_clusters=6):
    rng = np.random.default_rng(seed)
    user_cluster = rng.integers(0, n_clusters, size=n_users)
    X = np.zeros((n_users, n_items), dtype=np.float32)
    items_per = n_items // n_clusters
    for u in range(n_users):
        c = user_cluster[u]
        pool = np.arange(c * items_per, (c + 1) * items_per)
        take = rng.choice(pool, size=min(6, items_per), replace=False)
        X[u, take] = 1.0
    return X


def holdout_split(X, seed=1):
    """Hold out one positive per user as test relevance."""
    rng = np.random.default_rng(seed)
    train = X.copy()
    rel = np.zeros_like(X)
    for u in range(X.shape[0]):
        pos = np.flatnonzero(X[u] > 0)
        held = rng.choice(pos)
        train[u, held] = 0.0
        rel[u, held] = 1.0
    return train, rel


def test_popularity_scores_are_counts():
    X = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.float32)
    model = Popularity().fit(X)
    s = model.scores(np.array([0, 1]))
    assert s[0].tolist() == [2.0, 1.0, 1.0]


def test_itemknn_and_ease_beat_random_on_clusters():
    X = cluster_data()
    train, rel = holdout_split(X)
    users = np.arange(X.shape[0])
    for model in (ItemKNN(k=20, shrink=10.0), EASE(lam=50.0)):
        if isinstance(model, EASE):
            model.fit(train)
            scores = model.scores(users, X=train)
        else:
            model.fit(train)
            scores = model.scores(users)
        from src.rankers import mask_seen
        scores = mask_seen(scores, train)
        ndcg = np.nanmean(exact_per_user(scores, rel)["ndcg"])
        assert ndcg > 0.5, (model.name, ndcg)
    rnd = RandomRanker(seed=0).fit(train)
    scores = rnd.scores(users)
    ndcg_r = np.nanmean(exact_per_user(scores, rel)["ndcg"])
    assert ndcg_r < 0.35


def test_ials_fits_and_scores():
    X = cluster_data(n_users=60, n_items=30)
    model = IALS(factors=8, reg=0.1, alpha=10.0, iterations=3, seed=0).fit(X)
    s = model.scores(np.arange(60))
    assert s.shape == (60, 30)
    assert np.isfinite(s).all()


def test_random_ranker_is_deterministic():
    X = np.ones((3, 7), dtype=np.float32)
    a = RandomRanker(seed=5).fit(X).scores(np.arange(3))
    b = RandomRanker(seed=5).fit(X).scores(np.arange(3))
    assert np.array_equal(a, b)


def test_sasrec_is_excluded_by_design():
    with pytest.raises(NotImplementedError):
        SASRec().fit(np.ones((2, 3), dtype=np.float32))
