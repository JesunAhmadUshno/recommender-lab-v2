"""Full pipeline driver (v2, corrected pass). Stages:

  audit   descriptive data audit -> results/data_audit.json
  arm_t   KuaiRec big matrix, global-time split (Arm T)
  arm_f   KuaiRec small matrix, temporal split inside the small matrix
          (Arm F v2) + exposure arm
  ml1m    MovieLens-1M, same temporal protocol (replication)
  random  random-split arm (leakage comparison) for KuaiRec big + ML-1M
  verdicts  frozen-rule verdicts from the committed tables -> verdicts.json,
          plus the replication gate against v1 -> replication.json

Imported from v1 (recommender-evaluation-lab) with the changes named in
PREREGISTRATION.md section 10 and DECISIONS.md. The Arm T code path is
unchanged so the replication gate is meaningful. New in v2: the
coverage gate (PREREGISTRATION.md section 5) runs before any fitting on
every arm and can stop the run with the coverage table as its output,
and Arm F is rebuilt inside the small matrix.

Every number in RESULTS.md traces to a CSV written here.
"""
from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import audit as audit_mod  # noqa: E402
from src import correction, metrics, split as split_mod  # noqa: E402
from src.coverage import (GateStop, check_gate, check_score_matrix,  # noqa: E402
                          compute_coverage, nonzero_row_fraction)
from src.dataio import binary_matrix, load_kuairec, load_ml1m, pair_keys  # noqa: E402
from src.rankers import EASE, IALS, ItemKNN, Popularity, RandomRanker, mask_seen  # noqa: E402
from src.stats import holm_adjust, kendall_tau_b, paired_bootstrap  # noqa: E402

RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
BUNDLES = RESULTS / "bundles"
CACHE = RAW / "cache"

RANKER_ORDER = ["Random", "Popularity", "ItemKNN", "EASE", "iALS"]
SAMPLING_SEEDS = list(range(10))
EXPOSURE_SEEDS = list(range(10))
IALS_SEEDS = list(range(5))
SPLIT_SEEDS = list(range(5))
BOOT = 2000

# Frozen grids (PREREG section 3). Validation window only.
GRIDS = {
    "ItemKNN": [{"k": k, "shrink": s} for k in (50, 100) for s in (0.0, 50.0)],
    "EASE": [{"lam": v} for v in (10.0, 50.0, 100.0, 250.0, 500.0)],
    "iALS": ([{"factors": f, "reg": r, "alpha": 10.0, "iterations": 10}
              for f in (32, 64, 128) for r in (0.1, 1.0)]
             + [{"factors": 64, "reg": 0.1, "alpha": a, "iterations": 10} for a in (1.0, 40.0)]),
}
FINAL_IALS_ITERATIONS = 15

# Coverage gate thresholds (PREREGISTRATION.md section 5, frozen).
GATE_USER_MIN = 0.50
GATE_ITEM_MIN = 0.10
GATE_ARMF_USER_MIN = 0.95
GATE_ARMF_NONZERO_ROW_MIN = 0.95

COVERAGE_RECORDS: list[dict] = []


def record_coverage(arm_label: str, record: dict, user_min: float,
                    item_min: float) -> dict:
    """Run one coverage record through the gate, persist the coverage
    table immediately (a gate stop leaves the table as the run's
    output), and re-raise GateStop on a failure."""
    try:
        out = check_gate(record, arm_label, user_min, item_min)
    except GateStop:
        failed = dict(record)
        failed.update({"arm": arm_label, "user_min": float(user_min),
                       "item_min": float(item_min), "gate_passed": False})
        COVERAGE_RECORDS.append(failed)
        _write_coverage()
        raise
    COVERAGE_RECORDS.append(out)
    _write_coverage()
    log(f"coverage gate passed: {arm_label} user "
        f"{out['user_coverage']:.4f} (min {user_min}), item "
        f"{out['item_coverage']:.4f} (min {item_min})")
    return out


def _write_coverage() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "coverage.json").write_text(json.dumps(COVERAGE_RECORDS, indent=2))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def write_csv(name: str, rows: list[dict]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    path = TABLES / name
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {path} ({len(rows)} rows)")


def load_cached(name: str, loader, path: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{name}.npz"
    if cache_file.exists():
        z = np.load(cache_file)
        d = {k: z[k] for k in z.files}
        d["n_users"] = int(d["n_users"])
        d["n_items"] = int(d["n_items"])
        return d
    d = loader(path)
    np.savez(cache_file, **{k: v for k, v in d.items()})
    return d


def load_big() -> dict:
    return load_cached("kuairec_big", load_kuairec, str(RAW / "big_matrix.csv"))


def load_small() -> dict:
    return load_cached("kuairec_small", load_kuairec, str(RAW / "small_matrix.csv"))


def load_ml() -> dict:
    return load_cached("ml1m", load_ml1m, str(RAW / "ml-1m" / "ratings.dat"))


def load_all() -> tuple[dict, dict, dict]:
    return load_big(), load_small(), load_ml()


def make_ranker(name: str, config: dict, seed: int = 0):
    if name == "Random":
        return RandomRanker(seed=seed)
    if name == "Popularity":
        return Popularity()
    if name == "ItemKNN":
        return ItemKNN(**config)
    if name == "EASE":
        return EASE(**config)
    if name == "iALS":
        return IALS(seed=seed, **config)
    raise KeyError(name)


def score_ranker(model, name: str, users: np.ndarray, X_hist: np.ndarray) -> np.ndarray:
    if name == "EASE":
        return model.scores(users, X=X_hist)
    return model.scores(users)


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    v = values[~np.isnan(values)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    se = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
    return float(v.mean()), float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)


class TemporalArm:
    """Shared Arm T protocol for KuaiRec big and MovieLens-1M."""

    def __init__(self, arm: str, d: dict, pos: np.ndarray):
        self.arm = arm
        self.d = d
        self.pos = pos
        ts = d["timestamp"]
        self.t1, self.t2 = split_mod.temporal_cutoffs(ts, 0.80, 0.90)
        self.tr_m, self.va_m, self.te_m = split_mod.temporal_masks(ts, self.t1, self.t2)
        self.X_train = binary_matrix(d["user"][self.tr_m & pos], d["item"][self.tr_m & pos],
                                     np.ones(int((self.tr_m & pos).sum())),
                                     d["n_users"], d["n_items"])
        self.exposure_train = np.bincount(d["item"][self.tr_m], minlength=d["n_items"]).astype(np.float64)
        self.user_train_counts = np.bincount(d["user"][self.tr_m], minlength=d["n_users"]).astype(np.float64)
        # Validation and test positives, first-time against all train events.
        self.val_rel, self.val_users = self._rel(self.va_m)
        self.test_rel, self.test_users = self._rel(self.te_m)

    def _rel(self, window_mask: np.ndarray):
        d = self.d
        cand = window_mask & self.pos
        first = split_mod.first_time_mask(d["user"][self.tr_m], d["item"][self.tr_m],
                                          d["user"][cand], d["item"][cand], d["n_items"])
        u = d["user"][cand][first]
        i = d["item"][cand][first]
        rel = binary_matrix(u, i, np.ones(u.size), d["n_users"], d["n_items"])
        users = np.unique(u)
        return rel, users

    def coverage_record(self) -> dict:
        """Split-construction coverage for the gate (section 5 of the
        pre-registration): of the scored users and items, the fraction
        present in this arm's training window. Must be called while the
        raw event arrays are still on the dataset dict."""
        d = self.d
        scored_items = np.flatnonzero((self.test_rel > 0).sum(axis=0))
        return compute_coverage(d["user"][self.tr_m], d["item"][self.tr_m],
                                self.test_users, scored_items,
                                d["n_users"], d["n_items"])

    def tune(self, done: dict | None = None, on_new_row=None) -> tuple[dict, list[dict]]:
        """Tune each ranker on the validation window. `done` maps
        (ranker, config-json-sorted) to a previously computed tuning row
        (the VM reboots roughly every two hours; tuning is resumable so
        a reboot loses at most the config in flight). on_new_row(row)
        is called after each newly computed row so the caller can
        persist it immediately."""
        rows: list[dict] = []
        chosen: dict[str, dict] = {"Random": {}, "Popularity": {}}
        done = done or {}
        for name, grid in GRIDS.items():
            best, best_val = None, -np.inf
            for cfg in grid:
                key = (name, json.dumps(cfg, sort_keys=True))
                if key in done:
                    prev = done[key]
                    ndcg = float(prev["val_ndcg"])
                    row = {"arm": self.arm, "ranker": name, "config": json.dumps(cfg),
                           "val_ndcg": ndcg,
                           "fit_seconds": float(prev.get("fit_seconds", 0.0))}
                    log(f"{self.arm} tune {name} {cfg} -> val NDCG {ndcg:.4f} (resumed)")
                else:
                    t0 = time.perf_counter()
                    model = make_ranker(name, cfg)
                    val_rel = self.val_rel
                    self.val_rel = None  # the fit never reads it; park it (D11)
                    model.fit(self.X_train, self.exposure_train)
                    self.val_rel = val_rel
                    ndcg = self._val_ndcg(model, name)
                    row = {"arm": self.arm, "ranker": name, "config": json.dumps(cfg),
                           "val_ndcg": float(ndcg), "fit_seconds": time.perf_counter() - t0}
                    log(f"{self.arm} tune {name} {cfg} -> val NDCG {ndcg:.4f}")
                    if on_new_row is not None:
                        on_new_row(row)
                rows.append(row)
                if ndcg > best_val:
                    best, best_val = cfg, float(ndcg)
            chosen[name] = best
        return chosen, rows

    def _val_ndcg(self, model, name: str) -> float:
        """Mean validation NDCG@10, evaluated in blocks of 512 users so no
        full-width score or relevance copy is materialised. Per-user NDCG
        is row-independent, so the blocked mean is the same number as the
        whole-matrix mean (up to float summation order in the final mean).
        """
        parts = []
        for s in range(0, self.val_users.size, 512):
            bu = self.val_users[s:s + 512]
            sc = score_ranker(model, name, bu, self.X_train)
            sc = mask_seen(sc, self.X_train[bu])
            parts.append(metrics.exact_per_user(sc, self.val_rel[bu])["ndcg"])
        return float(np.nanmean(np.concatenate(parts)))

    def evaluate(self, chosen: dict):
        """Fit final models, score test users once, return (scores, per-user
        exact metrics, timing rows). iALS and Random metrics are averaged
        over their model seeds; their stored score matrix is the seed-mean
        (iALS) or seed 0 (Random) matrix used by the sampled protocol."""
        test_rel = self.test_rel[self.test_users]
        X_test = self.X_train[self.test_users]
        scores: dict[str, np.ndarray] = {}
        per_user: dict[str, dict] = {}
        timing_rows: list[dict] = []
        for name in RANKER_ORDER:
            if name == "Random":
                acc = []
                keep = None
                for s in range(5):
                    model = RandomRanker(seed=s).fit(self.X_train)
                    sc = mask_seen(model.scores(self.test_users), X_test)
                    if keep is None:
                        keep = sc
                    acc.append(metrics.exact_per_user(sc, test_rel))
                per_user[name] = {k: np.nanmean(np.vstack([a[k] for a in acc]), axis=0)
                                  for k in acc[0]}
                scores[name] = keep
                continue
            if name == "iALS":
                cfg = dict(chosen[name])
                cfg["iterations"] = FINAL_IALS_ITERATIONS
                acc = []
                sc_acc = None
                t_fit = 0.0
                for s in IALS_SEEDS:
                    t0 = time.perf_counter()
                    model = make_ranker(name, cfg, seed=s)
                    model.fit(self.X_train, self.exposure_train)
                    t_fit += time.perf_counter() - t0
                    sc = score_ranker(model, name, self.test_users, self.X_train)
                    sc_acc = sc if sc_acc is None else sc_acc + sc
                    acc.append(metrics.exact_per_user(mask_seen(sc, X_test), test_rel))
                per_user[name] = {k: np.nanmean(np.vstack([a[k] for a in acc]), axis=0)
                                  for k in acc[0]}
                scores[name] = mask_seen(sc_acc / len(IALS_SEEDS), X_test)
                timing_rows.append({"arm": self.arm, "ranker": name, "stage": "fit_x5seeds",
                                    "seconds": t_fit, "peak_mb": peak_mb()})
                continue
            cfg = chosen.get(name, {})
            t0 = time.perf_counter()
            model = make_ranker(name, cfg)
            model.fit(self.X_train, self.exposure_train)
            t_fit = time.perf_counter() - t0
            t0 = time.perf_counter()
            sc = mask_seen(score_ranker(model, name, self.test_users, self.X_train), X_test)
            t_score = time.perf_counter() - t0
            scores[name] = sc
            per_user[name] = metrics.exact_per_user(sc, test_rel)
            timing_rows.append({"arm": self.arm, "ranker": name, "stage": "fit",
                                "seconds": t_fit, "peak_mb": peak_mb()})
            timing_rows.append({"arm": self.arm, "ranker": name, "stage": "score_test",
                                "seconds": t_score, "peak_mb": peak_mb()})
        return scores, per_user, timing_rows

    def fit_ranker(self, name: str, chosen: dict):
        """Fit one ranker (or the seed set for Random / iALS) with its
        chosen config and return (model_or_models, timing_rows). Split
        from measurement so the caller can hold no relevance or score
        matrix while the fit's transient memory (the EASE Gram inverse
        above all) is live (DECISIONS.md D11). Mirrors the fit halves of
        evaluate() branch for branch."""
        timing_rows: list[dict] = []
        if name == "Random":
            models = [RandomRanker(seed=s).fit(self.X_train) for s in range(5)]
            return models, timing_rows
        if name == "iALS":
            cfg = dict(chosen[name])
            cfg["iterations"] = FINAL_IALS_ITERATIONS
            models = []
            t_fit = 0.0
            for s in IALS_SEEDS:
                t0 = time.perf_counter()
                model = make_ranker(name, cfg, seed=s)
                model.fit(self.X_train, self.exposure_train)
                t_fit += time.perf_counter() - t0
                models.append(model)
            timing_rows.append({"arm": self.arm, "ranker": name, "stage": "fit_x5seeds",
                                "seconds": t_fit, "peak_mb": peak_mb()})
            return models, timing_rows
        cfg = chosen.get(name, {})
        t0 = time.perf_counter()
        model = make_ranker(name, cfg)
        model.fit(self.X_train, self.exposure_train)
        timing_rows.append({"arm": self.arm, "ranker": name, "stage": "fit",
                            "seconds": time.perf_counter() - t0, "peak_mb": peak_mb()})
        return model, timing_rows

    def measure_ranker(self, name: str, models, test_rel: np.ndarray):
        """Score the test users for a fitted ranker and return (scores,
        per-user exact metrics, timing rows). Mirrors the measurement
        halves of evaluate(). ItemKNN, EASE, and iALS scores are
        row-independent, so their score matrices are assembled in blocks
        of 512 users and masked block by block: identical values, no
        full-width mask copy. The Random ranker's stream depends on the
        first requested user, so it is scored in one call, as before."""
        timing_rows: list[dict] = []
        if name == "Random":
            X_test = self.X_train[self.test_users]
            acc = []
            keep = None
            for model in models:
                sc = mask_seen(model.scores(self.test_users), X_test)
                if keep is None:
                    keep = sc
                acc.append(metrics.exact_per_user(sc, test_rel))
            per_user = {k: np.nanmean(np.vstack([a[k] for a in acc]), axis=0)
                        for k in acc[0]}
            return keep, per_user, timing_rows
        if name == "iALS":
            n = self.test_users.size
            exact_parts: list[list[dict]] = [[] for _ in models]
            acc_blocks = []
            for s in range(0, n, 512):
                bu = self.test_users[s:s + 512]
                rel_b = test_rel[s:s + 512]
                sc_acc = None
                for mi, model in enumerate(models):
                    sc = score_ranker(model, name, bu, self.X_train)
                    exact_parts[mi].append(metrics.exact_per_user(
                        mask_seen(sc, self.X_train[bu]), rel_b))
                    sc_acc = sc if sc_acc is None else sc_acc + sc
                acc_blocks.append(mask_seen(sc_acc / len(models), self.X_train[bu]))
            scores = np.vstack(acc_blocks)
            per_user = {}
            for k in exact_parts[0][0]:
                per_seed = [np.concatenate([blk[k] for blk in parts])
                            for parts in exact_parts]
                per_user[k] = np.nanmean(np.vstack(per_seed), axis=0)
            return scores, per_user, timing_rows
        model = models
        t0 = time.perf_counter()
        blocks = []
        for s in range(0, self.test_users.size, 512):
            bu = self.test_users[s:s + 512]
            sc = score_ranker(model, name, bu, self.X_train)
            blocks.append(mask_seen(sc, self.X_train[bu]))
        scores = np.vstack(blocks)
        t_score = time.perf_counter() - t0
        per_user = metrics.exact_per_user(scores, test_rel)
        timing_rows.append({"arm": self.arm, "ranker": name, "stage": "score_test",
                            "seconds": t_score, "peak_mb": peak_mb()})
        return scores, per_user, timing_rows


def arm_metric_rows(arm: str, per_user: dict[str, dict]) -> list[dict]:
    rows = []
    for name in RANKER_ORDER:
        pu = per_user[name]
        row = {"arm": arm, "ranker": name}
        for key in ("ndcg", "dcg", "recall", "mrr", "hr"):
            mean, lo, hi = mean_ci(pu[key])
            row[key] = mean
            if key == "ndcg":
                row["ndcg_ci_lo"], row["ndcg_ci_hi"] = lo, hi
        row["n_users"] = int((~np.isnan(pu["ndcg"])).sum())
        rows.append(row)
    return rows


def pairwise_rows(arm: str, per_user: dict[str, dict]) -> list[dict]:
    names = ["Popularity", "ItemKNN", "EASE", "iALS"]
    rows = []
    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            a, b = names[a_i], names[b_i]
            x, y = per_user[a]["ndcg"], per_user[b]["ndcg"]
            ok = ~np.isnan(x) & ~np.isnan(y)
            res = paired_bootstrap(x[ok] - y[ok], n_boot=BOOT, seed=0)
            rows.append({"arm": arm, "ranker_a": a, "ranker_b": b,
                         "mean_diff_a_minus_b": res["mean"], "ci_lo": res["ci_lo"],
                         "ci_hi": res["ci_hi"], "one_sided_lo": res["one_sided_lo"],
                         "one_sided_hi": res["one_sided_hi"],
                         "p_two_sided": res["p_two_sided"], "n_users": int(ok.sum())})
    adj = holm_adjust([r["p_two_sided"] for r in rows])
    for r, p in zip(rows, adj):
        r["p_holm"] = p
    return rows


def sampled_rows(arm: str, scores: dict[str, np.ndarray], rel: np.ndarray,
                 n_items: int, exclude: np.ndarray | None = None) -> list[dict]:
    rows = []
    n_pos = (rel > 0).sum(axis=1)
    for name in RANKER_ORDER:
        for seed in SAMPLING_SEEDS:
            out = metrics.sampled_per_user(scores[name], rel, m=100, seed=seed,
                                           exclude=exclude)
            corr = metrics.corrected_ndcg(out["pos_ranks"], n_items=n_items, m=100,
                                          n_pos_per_user=n_pos)
            rows.append({"arm": arm, "ranker": name, "seed": seed,
                         "sampled_ndcg": float(np.nanmean(out["ndcg"])),
                         "corrected_ndcg": float(np.nanmean(corr))})
        log(f"{arm} sampled done: {name}")
    return rows


def popularity_rows(arm: str, scores: dict[str, np.ndarray],
                    exposure: np.ndarray) -> list[dict]:
    rows = []
    pooled = []
    for name in RANKER_ORDER:
        top = correction.topk_lists(scores[name], k=10)
        pooled.append(top)
        rep = audit_mod.popularity_report(top, exposure)
        rows.append({"arm": arm, "ranker": name, **rep})
    all_top = np.vstack(pooled)
    rep = audit_mod.popularity_report(all_top, exposure)
    rows.append({"arm": arm, "ranker": "POOLED", **rep})
    return rows


def cold_start_rows(arm: TemporalArm, scores: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    test_rel = arm.test_rel[arm.test_users]
    item_counts = arm.exposure_train
    cold_items = item_counts < 5
    rel_slice = test_rel * cold_items[None, :]
    user_counts = arm.user_train_counts[arm.test_users]
    cold_users = user_counts < 5
    for slice_name, rel_s, user_sel in (
        ("item_cold", rel_slice, None),
        ("user_cold", test_rel, cold_users),
    ):
        for name in RANKER_ORDER:
            sc = scores[name]
            rel_use = rel_s
            if user_sel is not None:
                if user_sel.sum() == 0:
                    rows.append({"arm": arm.arm, "slice": slice_name, "ranker": name,
                                 "n_users": 0, "ndcg": None, "recall": None, "mrr": None})
                    continue
                sc = sc[user_sel]
                rel_use = rel_s[user_sel]
            pu = metrics.exact_per_user(sc, rel_use)
            rows.append({"arm": arm.arm, "slice": slice_name, "ranker": name,
                         "n_users": int((~np.isnan(pu["ndcg"])).sum()),
                         "ndcg": float(np.nanmean(pu["ndcg"])) if (~np.isnan(pu["ndcg"])).any() else None,
                         "recall": float(np.nanmean(pu["recall"])) if (~np.isnan(pu["recall"])).any() else None,
                         "mrr": float(np.nanmean(pu["mrr"])) if (~np.isnan(pu["mrr"])).any() else None})
    return rows


def _metric_row(arm_name: str, name: str, pu: dict) -> dict:
    """One ranker's row of arm_metrics.csv; same construction as
    arm_metric_rows for a full per-user dict."""
    row = {"arm": arm_name, "ranker": name}
    for key in ("ndcg", "dcg", "recall", "mrr", "hr"):
        mean, lo, hi = mean_ci(pu[key])
        row[key] = mean
        if key == "ndcg":
            row["ndcg_ci_lo"], row["ndcg_ci_hi"] = lo, hi
    row["n_users"] = int((~np.isnan(pu["ndcg"])).sum())
    return row


def _sampled_rows_for(arm_name: str, name: str, scores: np.ndarray,
                      rel: np.ndarray, n_items: int,
                      exclude: np.ndarray | None) -> list[dict]:
    """One ranker's sampled-metrics rows; same computation as
    sampled_rows restricted to a single ranker."""
    rows = []
    n_pos = (rel > 0).sum(axis=1)
    for seed in SAMPLING_SEEDS:
        out = metrics.sampled_per_user(scores, rel, m=100, seed=seed,
                                       exclude=exclude)
        corr = metrics.corrected_ndcg(out["pos_ranks"], n_items=n_items, m=100,
                                      n_pos_per_user=n_pos)
        rows.append({"arm": arm_name, "ranker": name, "seed": seed,
                     "sampled_ndcg": float(np.nanmean(out["ndcg"])),
                     "corrected_ndcg": float(np.nanmean(corr))})
    log(f"{arm_name} sampled done: {name}")
    return rows


def _cold_rows_for(arm: TemporalArm, name: str, scores: np.ndarray,
                   test_rel_rows: np.ndarray, rel_slice: np.ndarray,
                   cold_users: np.ndarray) -> list[dict]:
    """One ranker's cold-start rows; same slices as cold_start_rows."""
    rows = []
    for slice_name, rel_s, user_sel in (
        ("item_cold", rel_slice, None),
        ("user_cold", test_rel_rows, cold_users),
    ):
        sc = scores
        rel_use = rel_s
        if user_sel is not None:
            if user_sel.sum() == 0:
                rows.append({"arm": arm.arm, "slice": slice_name, "ranker": name,
                             "n_users": 0, "ndcg": None, "recall": None, "mrr": None})
                continue
            sc = sc[user_sel]
            rel_use = rel_s[user_sel]
        pu = metrics.exact_per_user(sc, rel_use)
        rows.append({"arm": arm.arm, "slice": slice_name, "ranker": name,
                     "n_users": int((~np.isnan(pu["ndcg"])).sum()),
                     "ndcg": float(np.nanmean(pu["ndcg"])) if (~np.isnan(pu["ndcg"])).any() else None,
                     "recall": float(np.nanmean(pu["recall"])) if (~np.isnan(pu["recall"])).any() else None,
                     "mrr": float(np.nanmean(pu["mrr"])) if (~np.isnan(pu["mrr"])).any() else None})
    return rows


def stage_arm_t(arm_name: str, d: dict, pos: np.ndarray, chosen_out: dict) -> None:
    arm = TemporalArm(arm_name, d, pos)
    log(f"{arm_name}: cutoffs t1={arm.t1:.1f} t2={arm.t2:.1f}; "
        f"train {int(arm.tr_m.sum())} val {int(arm.va_m.sum())} test {int(arm.te_m.sum())}; "
        f"test users {arm.test_users.size}")
    # The coverage gate runs before any fitting (PREREGISTRATION.md
    # section 5), while the raw event arrays are still on the arm.
    record_coverage(arm_name, arm.coverage_record(), GATE_USER_MIN, GATE_ITEM_MIN)
    # Memory discipline on the shared VM (DECISIONS.md D11): the raw
    # event arrays are not needed again until the leakage check, and the
    # test relevance matrix is not needed during tuning. Both are parked
    # outside the arm while the heavy fits run and restored afterwards.
    saved = {k: d[k] for k in ("user", "item", "timestamp") if k in d}
    for k in saved:
        d[k] = np.empty(0, dtype=saved[k].dtype)
    test_rel, test_users = arm.test_rel, arm.test_users
    arm.test_rel = None
    arm.test_users = None
    # Resumable tuning: rows already in the tuning CSV are reused and
    # each new row is persisted the moment it is computed (D12).
    tuning_rows_saved: list[dict] = []
    tuning_path = TABLES / f"tuning_{arm_name}.csv"
    if tuning_path.exists():
        with open(tuning_path, newline="") as fh:
            tuning_rows_saved = list(csv.DictReader(fh))
    done = {(r["ranker"], json.dumps(json.loads(r["config"]), sort_keys=True)): r
            for r in tuning_rows_saved}

    def _persist_tuning_row(row: dict) -> None:
        tuning_rows_saved.append(row)
        write_csv(f"tuning_{arm_name}.csv", tuning_rows_saved)

    chosen, tuning = arm.tune(done=done, on_new_row=_persist_tuning_row)
    arm.test_rel, arm.test_users = test_rel, test_users
    arm.val_rel = None  # validation matrix is dead weight from here on
    chosen_out[arm_name] = chosen
    write_csv(f"tuning_{arm_name}.csv", tuning)
    # Ranker by ranker from here: each score matrix is freed before the
    # next fit. Every downstream table is computed from the same scores
    # the all-at-once path used; only the lifetime of the matrices
    # differs (DECISIONS.md D11).
    test_rel_rows = arm.test_rel[arm.test_users]
    arm.test_rel = None  # the full matrix is dead weight from here (D11)
    exclude = arm.X_train[arm.test_users] > 0
    rel_slice = test_rel_rows * (arm.exposure_train < 5)[None, :]
    cold_users = arm.user_train_counts[arm.test_users] < 5
    per_user_all: dict[str, dict] = {}
    metric_rows: list[dict] = []
    timing_rows: list[dict] = []
    sampled_all: list[dict] = []
    pop_rows: list[dict] = []
    cold_by_slice: dict[str, list[dict]] = {"item_cold": [], "user_cold": []}
    top_lists = []
    for name in RANKER_ORDER:
        # Per-ranker bundle: if a previous (killed) run already finished
        # this ranker under the same chosen config, reload its rows and
        # per-user arrays instead of refitting (D12).
        BUNDLES.mkdir(parents=True, exist_ok=True)
        bundle_json = BUNDLES / f"{arm_name}__{name}.json"
        bundle_npz = BUNDLES / f"{arm_name}__{name}.npz"
        if bundle_json.exists() and bundle_npz.exists():
            bundle = json.loads(bundle_json.read_text())
            if bundle.get("config") == chosen.get(name, {}):
                z = np.load(bundle_npz)
                per_user = {k: z[k] for k in z.files if k != "top"}
                top = z["top"]
                per_user_all[name] = per_user
                timing_rows.extend(bundle["timing_rows"])
                metric_rows.append(bundle["metric_row"])
                sampled_all.extend(bundle["sampled_rows"])
                top_lists.append(top)
                pop_rows.append(bundle["pop_row"])
                for r in bundle["cold_rows"]:
                    cold_by_slice[r["slice"]].append(r)
                log(f"{arm_name} {name}: resumed from bundle")
                continue
        models, fit_trows = arm.fit_ranker(name, chosen)
        scores, per_user, meas_trows = arm.measure_ranker(name, models, test_rel_rows)
        del models
        # Post-scoring backstop (section 5): an all-zero score matrix
        # is the v1 defect signature and fails the run.
        check_score_matrix(scores, arm_name, name)
        ranker_trows = fit_trows + meas_trows
        timing_rows.extend(ranker_trows)
        per_user_all[name] = per_user
        mrow = _metric_row(arm_name, name, per_user)
        metric_rows.append(mrow)
        srows = _sampled_rows_for(arm_name, name, scores,
                                  test_rel_rows, d["n_items"], exclude)
        sampled_all.extend(srows)
        top = correction.topk_lists(scores, k=10)
        top_lists.append(top)
        prow = {"arm": arm_name, "ranker": name,
                **audit_mod.popularity_report(top, arm.exposure_train)}
        pop_rows.append(prow)
        crows = _cold_rows_for(arm, name, scores, test_rel_rows, rel_slice, cold_users)
        for r in crows:
            cold_by_slice[r["slice"]].append(r)
        del scores
        np.savez(bundle_npz, top=top, **per_user)
        bundle_json.write_text(json.dumps(
            {"config": chosen.get(name, {}),
             "timing_rows": ranker_trows,
             "metric_row": mrow, "sampled_rows": srows, "pop_row": prow,
             "cold_rows": crows}, indent=2))
    pooled = np.vstack(top_lists)
    pop_rows.append({"arm": arm_name, "ranker": "POOLED",
                     **audit_mod.popularity_report(pooled, arm.exposure_train)})
    cold_rows = cold_by_slice["item_cold"] + cold_by_slice["user_cold"]
    write_csv("arm_metrics.csv", _merge_existing("arm_metrics.csv", metric_rows, "arm", arm_name))
    write_csv("pairwise.csv", _merge_existing("pairwise.csv", pairwise_rows(arm_name, per_user_all), "arm", arm_name))
    write_csv("timings.csv", _merge_existing("timings.csv", timing_rows, "arm", arm_name))
    write_csv("sampled_metrics.csv", _merge_existing("sampled_metrics.csv", sampled_all, "arm", arm_name))
    write_csv("popularity_report.csv", _merge_existing("popularity_report.csv", pop_rows, "arm", arm_name))
    write_csv("cold_start.csv", _merge_existing("cold_start.csv", cold_rows, "arm", arm_name))
    d.update(saved)  # restore the raw event arrays for the leakage check
    if arm_name == "arm_t_kuairec":
        _leakage_check(arm)


def _merge_existing(name: str, new_rows: list[dict], key: str, value: str) -> list[dict]:
    path = TABLES / name
    old: list[dict] = []
    if path.exists():
        with open(path, newline="") as fh:
            old = [r for r in csv.DictReader(fh) if r.get(key) != value]
    return old + new_rows


def _leakage_check(arm: TemporalArm) -> None:
    d = arm.d
    tr = arm.tr_m
    te_pos = arm.te_m & arm.pos
    first = split_mod.first_time_mask(d["user"][tr], d["item"][tr],
                                      d["user"][te_pos], d["item"][te_pos], d["n_items"])
    tu, ti, tt = d["user"][te_pos][first], d["item"][te_pos][first], d["timestamp"][te_pos][first]
    clean = audit_mod.leakage_audit(d["user"][tr], d["item"][tr], tu, ti,
                                    d["timestamp"][tr], tt, d["n_items"])
    # Planted leak: copy one test pair into train with a post-test timestamp.
    pu = np.append(d["user"][tr], tu[0])
    pi = np.append(d["item"][tr], ti[0])
    pt = np.append(d["timestamp"][tr], tt[0] + 1000.0)
    planted = audit_mod.leakage_audit(pu, pi, tu, ti, pt, tt, d["n_items"])
    out = {"clean": clean, "planted": planted,
           "audit_flags_planted": bool(not planted["pair_overlap_ok"] or not planted["time_order_ok"])}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "leakage_check.json").write_text(json.dumps(out, indent=2))
    log(f"leakage check: clean ok={clean['pair_overlap_ok'] and clean['time_order_ok']}, "
        f"planted flagged={out['audit_flags_planted']}")


def _fit_score_small(name: str, cfg: dict, X_pool: np.ndarray, exposure: np.ndarray,
                     seed: int = 0, iters: int | None = None) -> np.ndarray:
    """Fit one ranker on an Arm F pool and score all small-matrix users.

    v2: the pool and the scored users live in the same (small-matrix)
    id space, so no cross-matrix user mapping exists. v1 fitted on a
    big-matrix pool and scored the small users through a raw-id map;
    their rows in that pool were empty, which is the defect this pass
    corrects.
    """
    n_users = X_pool.shape[0]
    if name == "Random":
        model = RandomRanker(seed=seed).fit(X_pool)
        return model.scores(np.arange(n_users))
    if name == "iALS" and iters is not None:
        cfg = dict(cfg)
        cfg["iterations"] = iters
    model = make_ranker(name, cfg, seed=seed)
    model.fit(X_pool, exposure)
    if name == "EASE":
        return model.scores(np.arange(n_users), X=X_pool)
    return model.scores(np.arange(n_users))


def stage_arm_f(small: dict, chosen: dict) -> None:
    """Arm F v2 (corrected): Arm T's temporal protocol inside the small
    matrix. Training histories and the truth set come from the same
    matrix, so no scored user can have an empty history by
    construction; the coverage gate verifies that before any fit."""
    cfg_map = chosen["arm_t_kuairec"]
    pos = small["watch_ratio"] > 2.0
    arm = TemporalArm("arm_f", small, pos)
    log(f"arm_f: cutoffs t1={arm.t1:.1f} t2={arm.t2:.1f}; "
        f"train {int(arm.tr_m.sum())} val {int(arm.va_m.sum())} test {int(arm.te_m.sum())}; "
        f"scored users {arm.test_users.size}")
    # The coverage gate, at the Arm F threshold (section 5).
    record_coverage("arm_f", arm.coverage_record(),
                    GATE_ARMF_USER_MIN, GATE_ITEM_MIN)
    X_pool = arm.X_train
    exposure_pool = arm.exposure_train
    rel = arm.test_rel
    scored_users = arm.test_users
    (RESULTS / "arm_f_meta.json").write_text(json.dumps(
        {"construction": "temporal split inside the small matrix (Arm T protocol)",
         "t1": arm.t1, "t2": arm.t2,
         "train_events": int(arm.tr_m.sum()),
         "pool_positive_cells": int((X_pool > 0).sum()),
         "scored_users": int(scored_users.size),
         "truth_positives": int((rel > 0).sum())}, indent=2))
    scores: dict[str, np.ndarray] = {}
    per_user: dict[str, dict] = {}
    timing: list[dict] = []
    BUNDLES.mkdir(parents=True, exist_ok=True)
    for name in RANKER_ORDER:
        cfg = cfg_map.get(name, {})
        bundle_json = BUNDLES / f"arm_f__{name}.json"
        bundle_npz = BUNDLES / f"arm_f__{name}.npz"
        if bundle_json.exists() and bundle_npz.exists():
            bundle = json.loads(bundle_json.read_text())
            if bundle.get("config") == cfg:
                z = np.load(bundle_npz)
                scores[name] = z["scores"]
                per_user[name] = {k: z[k] for k in z.files if k != "scores"}
                timing.extend(bundle["timing_rows"])
                log(f"arm_f {name}: truth NDCG@10 = "
                    f"{np.nanmean(per_user[name]['ndcg']):.4f} (resumed)")
                continue
        t_before = len(timing)
        # Training positives are masked before measurement, exactly as
        # Arm T masks them, per seed and per fit.
        if name == "iALS":
            acc, sc_acc = [], None
            t0 = time.perf_counter()
            for s in IALS_SEEDS:
                sc = mask_seen(_fit_score_small(name, cfg, X_pool, exposure_pool,
                                                seed=s, iters=FINAL_IALS_ITERATIONS),
                               X_pool)
                sc_acc = sc if sc_acc is None else sc_acc + sc
                acc.append(metrics.exact_per_user(sc, rel))
            per_user[name] = {k: np.nanmean(np.vstack([a[k] for a in acc]), axis=0) for k in acc[0]}
            scores[name] = sc_acc / len(IALS_SEEDS)
            timing.append({"arm": "arm_f", "ranker": name, "stage": "fit_x5seeds",
                           "seconds": time.perf_counter() - t0, "peak_mb": peak_mb()})
        elif name == "Random":
            acc = []
            keep = None
            for s in range(5):
                sc = mask_seen(_fit_score_small(name, cfg, X_pool, exposure_pool, seed=s),
                               X_pool)
                if keep is None:
                    keep = sc
                acc.append(metrics.exact_per_user(sc, rel))
            per_user[name] = {k: np.nanmean(np.vstack([a[k] for a in acc]), axis=0) for k in acc[0]}
            scores[name] = keep
        else:
            t0 = time.perf_counter()
            sc = mask_seen(_fit_score_small(name, cfg, X_pool, exposure_pool), X_pool)
            timing.append({"arm": "arm_f", "ranker": name, "stage": "fit_and_score",
                           "seconds": time.perf_counter() - t0, "peak_mb": peak_mb()})
            scores[name] = sc
            per_user[name] = metrics.exact_per_user(sc, rel)
        # The frozen post-scoring backstops (section 5).
        check_score_matrix(scores[name], "arm_f", name)
        if name in ("ItemKNN", "EASE", "iALS"):
            frac = nonzero_row_fraction(scores[name], scored_users)
            log(f"arm_f {name}: nonzero-row fraction over scored users {frac:.4f}")
            if frac < GATE_ARMF_NONZERO_ROW_MIN:
                raise GateStop(
                    f"arm_f {name}: nonzero-row fraction {frac:.4f} below the "
                    f"frozen {GATE_ARMF_NONZERO_ROW_MIN} (section 5 backstop)")
        np.savez(bundle_npz, scores=scores[name], **per_user[name])
        bundle_json.write_text(json.dumps(
            {"config": cfg, "timing_rows": timing[t_before:]}, indent=2))
        log(f"arm_f {name}: truth NDCG@10 = {np.nanmean(per_user[name]['ndcg']):.4f}")
    write_csv("arm_metrics.csv", _merge_existing("arm_metrics.csv", arm_metric_rows("arm_f", per_user), "arm", "arm_f"))
    write_csv("pairwise.csv", _merge_existing("pairwise.csv", pairwise_rows("arm_f", per_user), "arm", "arm_f"))
    write_csv("timings.csv", _merge_existing("timings.csv", timing, "arm", "arm_f"))
    # Sampled negatives exclude training positives (the arm's scores
    # are masked there); on this arm every sampled negative is a known
    # negative, because the matrix is fully observed.
    srows = sampled_rows("arm_f", scores, rel, small["n_items"], exclude=X_pool > 0)
    write_csv("sampled_metrics.csv", _merge_existing("sampled_metrics.csv", srows, "arm", "arm_f"))
    prows = popularity_rows("arm_f", scores, exposure_pool)
    write_csv("popularity_report.csv", _merge_existing("popularity_report.csv", prows, "arm", "arm_f"))
    # Sensitivity: truth labels at watch_ratio > 1.0, same scores. The
    # truth set is rebuilt under the same temporal discipline (first-
    # time positives at the 1.0 threshold), not the raw full matrix.
    arm1 = TemporalArm("arm_f", small, small["watch_ratio"] > 1.0)
    sens_rows = []
    for name in RANKER_ORDER:
        pu = metrics.exact_per_user(scores[name], arm1.test_rel)
        sens_rows.append({"arm": "arm_f", "ranker": name, "label_threshold": "watch_ratio>1.0",
                          "ndcg": float(np.nanmean(pu["ndcg"])),
                          "n_users": int((~np.isnan(pu["ndcg"])).sum())})
    write_csv("arm_f_label_sensitivity.csv", sens_rows)
    # Sensitivity pool: train on small-matrix events before the small
    # matrix median timestamp, same truth set, gated like any pool.
    if len(_read_table("arm_f_pool_sensitivity.csv")) == len(RANKER_ORDER):
        log("arm_f sensitivity pool: resumed from saved table")
    else:
        ts = small["timestamp"]
        tr_sens = ts < float(np.nanmedian(ts))
        scored_items = np.flatnonzero((rel > 0).sum(axis=0))
        rec = compute_coverage(small["user"][tr_sens], small["item"][tr_sens],
                               scored_users, scored_items,
                               small["n_users"], small["n_items"])
        record_coverage("arm_f_sensitivity_pool", rec, GATE_USER_MIN, GATE_ITEM_MIN)
        X_sens = binary_matrix(small["user"][tr_sens & pos], small["item"][tr_sens & pos],
                               np.ones(int((tr_sens & pos).sum())),
                               small["n_users"], small["n_items"])
        exposure_sens = np.bincount(small["item"][tr_sens],
                                    minlength=small["n_items"]).astype(np.float64)
        pool_rows = []
        for name in RANKER_ORDER:
            cfg = cfg_map.get(name, {})
            iters = FINAL_IALS_ITERATIONS if name == "iALS" else None
            sc = _fit_score_small(name, cfg, X_sens, exposure_sens, seed=0, iters=iters)
            sc = mask_seen(sc, X_sens)
            check_score_matrix(sc, "arm_f_sensitivity_pool", name)
            pu = metrics.exact_per_user(sc, rel)
            pool_rows.append({"ranker": name, "ndcg": float(np.nanmean(pu["ndcg"])),
                              "pool_events": int(tr_sens.sum())})
            log(f"arm_f sensitivity pool {name}: NDCG {np.nanmean(pu['ndcg']):.4f}")
        write_csv("arm_f_pool_sensitivity.csv", pool_rows)
    write_csv("popularity_slices.csv", _popularity_slices(scores, rel, exposure_pool,
                                                         exclude=X_pool > 0))
    if _read_table("exposure_estimates.csv") and _read_table("exposure_tau.csv"):
        log("exposure arm: resumed from saved tables")
    else:
        tr = arm.tr_m
        pool_events = (small["user"][tr], small["item"][tr], small["watch_ratio"][tr])
        _exposure_arm(scores, rel, scored_users, exposure_pool, pool_events, small)



def _popularity_slices(scores: dict[str, np.ndarray], rel: np.ndarray,
                       exposure_pool: np.ndarray,
                       exclude: np.ndarray | None = None) -> list[dict]:
    """H5: ranker orderings inside item-popularity slices, truth vs the
    naive sampled evaluator (seed-averaged)."""
    order = np.argsort(-exposure_pool, kind="stable")
    n_items = exposure_pool.size
    slices = {
        "head_top1pct": order[: max(1, int(round(0.01 * n_items)))],
        "torso_next9pct": order[int(round(0.01 * n_items)): int(round(0.10 * n_items))],
        "tail_rest": order[int(round(0.10 * n_items)):],
    }
    rows = []
    for slice_name, items in slices.items():
        mask = np.zeros(n_items, dtype=bool)
        mask[items] = True
        rel_s = rel * mask[None, :]
        for name in RANKER_ORDER:
            truth = np.nanmean(metrics.exact_per_user(scores[name], rel_s)["ndcg"])
            sampled = []
            for seed in SAMPLING_SEEDS:
                out = metrics.sampled_per_user(scores[name], rel_s, m=100, seed=seed,
                                               exclude=exclude)
                sampled.append(float(np.nanmean(out["ndcg"])))
            rows.append({"slice": slice_name, "ranker": name,
                         "truth_ndcg": float(truth),
                         "sampled_ndcg_mean": float(np.mean(sampled)),
                         "n_users": int((rel_s.sum(axis=1) > 0).sum())})
        log(f"slice {slice_name} done")
    return rows


def _exposure_arm(scores: dict[str, np.ndarray], rel: np.ndarray,
                  scored_users: np.ndarray, exposure_pool: np.ndarray,
                  pool_events, small: dict) -> None:
    """Off-policy study on the corrected Arm F (PREREGISTRATION.md
    section 7). Differences from v1, all frozen there: the pool is the
    within-small train window (pool_events are its (user, item, ratio)
    arrays in small-matrix id space); exposure draws cover the full
    small-matrix grid and propensities are estimated from the full-grid
    counts over 1,411 users, as in v1; the estimand population is the
    scored users, so the estimator inputs are row-subset to them."""
    from src.correction import LogisticImputer, estimator_dcgs, estimate_propensities, make_propensities, topk_lists
    pu, pi, pratio = pool_events
    labels = (pratio > 2.0).astype(np.float64)
    n_users = small["n_users"]
    user_counts = np.bincount(pu, minlength=n_users)
    user_pos = np.bincount(pu, weights=labels, minlength=n_users)
    user_pos_rate = np.divide(user_pos, np.maximum(user_counts, 1))
    item_pos = np.bincount(pi, weights=labels, minlength=small["n_items"])
    item_pos_rate = np.divide(item_pos, np.maximum(exposure_pool, 1))
    imputer = LogisticImputer().fit(pu, pi, labels, user_pos_rate, exposure_pool, item_pos_rate)
    imputed = imputer.predict_matrix(n_users, small["n_items"])
    su = np.asarray(scored_users)
    rel_s = rel[su]
    imputed_s = imputed[su]
    top_lists = {name: topk_lists(scores[name], k=10)[su] for name in RANKER_ORDER}
    est_rows: list[dict] = []
    tau_rows: list[dict] = []
    for gamma in (0.0, 0.5, 1.0):
        p, c = make_propensities(exposure_pool, gamma=gamma, target_mean=0.163)
        for seed in EXPOSURE_SEEDS:
            rng = np.random.default_rng(20_000 + int(gamma * 10) * 100 + seed)
            exposed_full = rng.random(rel.shape) < p[None, :]
            counts = exposed_full.sum(axis=0)
            p_hat = estimate_propensities(counts, rel.shape[0])
            exposed = exposed_full[su]
            for condition, p_use in (("true", p), ("estimated", p_hat)):
                means: dict[str, dict[str, float]] = {}
                for name in RANKER_ORDER:
                    out = estimator_dcgs(top_lists[name], rel_s, exposed, p_use, imputed=imputed_s)
                    means[name] = {
                        "truth": float(out["truth"].mean()),
                        "naive": float(out["naive"].mean()),
                        "ips": float(out["ips"].mean()),
                        "dr": float(out["dr"].mean()),
                        "snips": float(out["snips_mean_dcg"]),
                    }
                    est_rows.append({"gamma": gamma, "condition": condition, "seed": seed,
                                     "ranker": name, **means[name]})
                truth_order = np.array([means[n]["truth"] for n in RANKER_ORDER])
                tau_row = {"gamma": gamma, "condition": condition, "seed": seed}
                for est in ("naive", "ips", "snips", "dr"):
                    est_vals = np.array([means[n][est] for n in RANKER_ORDER])
                    tau_row[f"tau_{est}"] = kendall_tau_b(est_vals, truth_order)
                tau_rows.append(tau_row)
        log(f"exposure arm gamma={gamma} done (c={c:.4f})")
    write_csv("exposure_estimates.csv", est_rows)
    write_csv("exposure_tau.csv", tau_rows)


def stage_random(big: dict, ml: dict, chosen: dict) -> None:
    # Resumable per (dataset, seed): rows from a previous run are kept
    # only for seeds that completed all five rankers; each finished
    # seed is persisted immediately (D12).
    from collections import Counter
    existing = _read_table("random_split.csv")
    counts = Counter((r["dataset"], int(r["seed"])) for r in existing)
    rows: list[dict] = [r for r in existing
                        if counts[(r["dataset"], int(r["seed"]))] == len(RANKER_ORDER)]
    done_keys = {(r["dataset"], int(r["seed"])) for r in rows}
    if rows:
        log(f"random arm: resuming with {len(done_keys)} completed seeds")
    for ds_name, d, pos_label, cfg_key in (
        ("kuairec_big", big, big["watch_ratio"] > 2.0, "arm_t_kuairec"),
        ("ml1m", ml, ml["rating"] >= 4.0, "arm_t_ml1m"),
    ):
        cfg_map = chosen[cfg_key]
        n = d["user"].size
        for seed in SPLIT_SEEDS:
            if (ds_name, seed) in done_keys:
                log(f"random arm {ds_name} seed {seed} done (resumed)")
                continue
            tr_m, _va_m, te_m = split_mod.random_masks(n, seed)
            X_train = binary_matrix(d["user"][tr_m & pos_label], d["item"][tr_m & pos_label],
                                    np.ones(int((tr_m & pos_label).sum())), d["n_users"], d["n_items"])
            exposure = np.bincount(d["item"][tr_m], minlength=d["n_items"]).astype(np.float64)
            cand = te_m & pos_label
            first = split_mod.first_time_mask(d["user"][tr_m], d["item"][tr_m],
                                              d["user"][cand], d["item"][cand], d["n_items"])
            rel = binary_matrix(d["user"][cand][first], d["item"][cand][first],
                                np.ones(int(first.sum())), d["n_users"], d["n_items"])
            users = np.unique(d["user"][cand][first])
            rel_u = rel[users]
            X_u = X_train[users]
            # Coverage gate for this split seed, before any fit.
            rec = compute_coverage(d["user"][tr_m], d["item"][tr_m], users,
                                   np.unique(d["item"][cand][first]),
                                   d["n_users"], d["n_items"])
            record_coverage(f"random_{ds_name}_seed{seed}", rec,
                            GATE_USER_MIN, GATE_ITEM_MIN)
            for name in RANKER_ORDER:
                cfg = cfg_map.get(name, {})
                if name == "iALS":
                    cfg = dict(cfg)
                    cfg["iterations"] = FINAL_IALS_ITERATIONS
                model = make_ranker(name, cfg, seed=seed if name == "iALS" else 0)
                model.fit(X_train, exposure)
                sc = score_ranker(model, name, users, X_train)
                sc = mask_seen(sc, X_u)
                check_score_matrix(sc, f"random_{ds_name}_seed{seed}", name)
                ndcg = float(np.nanmean(metrics.exact_per_user(sc, rel_u)["ndcg"]))
                rows.append({"dataset": ds_name, "ranker": name, "seed": seed,
                             "ndcg_random": ndcg, "n_users": int(users.size)})
            log(f"random arm {ds_name} seed {seed} done")
            write_csv("random_split.csv", rows)
    write_csv("random_split.csv", rows)


def _read_table(name: str) -> list[dict]:
    path = TABLES / name
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _order_of(values: dict[str, float]) -> list[str]:
    return sorted(values, key=lambda k: -values[k])


def _flip_count(vals_a: dict[str, float], vals_b: dict[str, float]) -> int:
    names = list(vals_a)
    flips = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            da = vals_a[names[i]] - vals_a[names[j]]
            db = vals_b[names[i]] - vals_b[names[j]]
            if da * db < 0:
                flips += 1
    return flips


def _pair_flip_counts(exact: dict[str, float], per_seed: list[dict[str, float]]) -> dict:
    names = list(exact)
    counts: dict[str, list[int]] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            key = f"{a}|{b}"
            counts[key] = [0, len(per_seed)]
            for vals in per_seed:
                if (vals[a] - vals[b]) * (exact[a] - exact[b]) < 0:
                    counts[key][0] += 1
    return counts


# v1's standing record (results of record: recommender-evaluation-lab
# commit a9c0464, RESULTS.md "Verdicts that stand on valid evidence").
# Verdicts are stored in v2 vocabulary (PREREGISTRATION.md section 8:
# v1 SUPPORTED = CONFIRMED). These are the replication gate's targets.
V1_RECORD = {
    "H1a": {"verdict": "CONFIRMED",
            "numbers": {"mean_diff": 0.000870, "one_sided_95_lower_bound": 0.000618,
                        "p_holm": 0.0, "n_users": 6638}},
    "H1b": {"verdict": "CONFIRMED",
            "numbers": {"mean_diff": 0.000500, "one_sided_95_lower_bound": 0.000139,
                        "p_holm": 0.021, "n_users": 6638}},
    "H3a": {"verdict": "CONFIRMED",
            "numbers": {"kuairec_rankers_random_exceeds": 4,
                        "ml1m_rankers_random_exceeds": 3}},
    "H3b": {"verdict": "CONFIRMED",
            "numbers": {"clean_pair_overlap": 0, "planted_pair_overlap": 2}},
    "H7": {"verdict": "CONFIRMED",
           "numbers": {"gap_cold": 0.0000106, "gap_full": 0.000870, "n_users": 6535}},
}


def stage_verdicts() -> dict:
    arm_metrics = {(r["arm"], r["ranker"]): r for r in _read_table("arm_metrics.csv")}
    sampled = _read_table("sampled_metrics.csv")
    pairwise = {(r["arm"], r["ranker_a"], r["ranker_b"]): r for r in _read_table("pairwise.csv")}
    random_rows = _read_table("random_split.csv")
    tau_rows = _read_table("exposure_tau.csv")
    est_rows = _read_table("exposure_estimates.csv")
    pop_rows = _read_table("popularity_report.csv")
    cold_rows = _read_table("cold_start.csv")
    slice_rows = _read_table("popularity_slices.csv")
    leakage = {}
    if (RESULTS / "leakage_check.json").exists():
        leakage = json.loads((RESULTS / "leakage_check.json").read_text())

    def exact(arm: str, ranker: str) -> float:
        return _f(arm_metrics[(arm, ranker)]["ndcg"])

    verdicts: dict = {}

    def guarded(key: str, fn) -> None:
        """A hypothesis whose inputs are missing or unreadable is
        INCONCLUSIVE with the reason recorded (section 8), never a
        crash and never a silent pass."""
        try:
            verdicts[key] = fn()
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            verdicts[key] = {"verdict": "INCONCLUSIVE",
                             "reason": f"{type(exc).__name__}: {exc}"}

    # H1
    def h1() -> dict:
        h1_arms = {}
        for arm in ("arm_t_kuairec", "arm_f"):
            h1_arms[arm] = {"popularity": exact(arm, "Popularity"), "random": exact(arm, "Random")}
        return {"verdict": "CONFIRMED" if all(v["popularity"] > v["random"] for v in h1_arms.values()) else "REFUTED",
                "evidence": h1_arms}
    guarded("H1", h1)

    # H1a / H1b from the Arm T pairwise table (one-sided bounds, reversed pairs).
    def h1a() -> dict:
        row = pairwise[("arm_t_kuairec", "Popularity", "ItemKNN")]
        lo = -_f(row["one_sided_hi"])
        return {"verdict": "CONFIRMED" if lo > 0 else "REFUTED",
                "evidence": {"itemknn_minus_popularity_mean": -_f(row["mean_diff_a_minus_b"]),
                             "one_sided_95_lower_bound": lo,
                             "p_holm": _f(row["p_holm"]), "n_users": int(_f(row["n_users"]))}}
    guarded("H1a", h1a)

    def h1b() -> dict:
        row = pairwise[("arm_t_kuairec", "ItemKNN", "EASE")]
        lo = -_f(row["one_sided_hi"])
        return {"verdict": "CONFIRMED" if lo > 0 else "REFUTED",
                "evidence": {"ease_minus_itemknn_mean": -_f(row["mean_diff_a_minus_b"]),
                             "one_sided_95_lower_bound": lo,
                             "p_holm": _f(row["p_holm"]), "n_users": int(_f(row["n_users"]))}}
    guarded("H1b", h1b)

    # H2a / H2b on Arm F (replications reported for the other arms).
    def h2_block(arm: str) -> dict:
        exact_map = {n: exact(arm, n) for n in RANKER_ORDER}
        per_seed_s = {n: [] for n in RANKER_ORDER}
        per_seed_c = {n: [] for n in RANKER_ORDER}
        seed_rows: dict[int, dict[str, float]] = {}
        seed_rows_c: dict[int, dict[str, float]] = {}
        for r in sampled:
            if r["arm"] != arm:
                continue
            per_seed_s[r["ranker"]].append(_f(r["sampled_ndcg"]))
            per_seed_c[r["ranker"]].append(_f(r["corrected_ndcg"]))
            seed_rows.setdefault(int(r["seed"]), {})[r["ranker"]] = _f(r["sampled_ndcg"])
            seed_rows_c.setdefault(int(r["seed"]), {})[r["ranker"]] = _f(r["corrected_ndcg"])
        s_mean = {n: float(np.mean(v)) for n, v in per_seed_s.items()}
        c_mean = {n: float(np.mean(v)) for n, v in per_seed_c.items()}
        err_s = float(np.mean([abs(s_mean[n] - exact_map[n]) for n in RANKER_ORDER]))
        err_c = float(np.mean([abs(c_mean[n] - exact_map[n]) for n in RANKER_ORDER]))
        flips_s = _pair_flip_counts(exact_map, [seed_rows[s] for s in sorted(seed_rows)])
        flips_c = _pair_flip_counts(exact_map, [seed_rows_c[s] for s in sorted(seed_rows_c)])
        return {"exact": exact_map, "sampled_mean": s_mean, "corrected_mean": c_mean,
                "mean_abs_err_sampled": err_s, "mean_abs_err_corrected": err_c,
                "order_sampled": _order_of(s_mean), "order_exact": _order_of(exact_map),
                "pair_flips_sampled": flips_s, "pair_flips_corrected": flips_c}

    def h2() -> None:
        h2f = h2_block("arm_f")
        h2a_ok = all(h2f["sampled_mean"][n] > h2f["exact"][n] for n in RANKER_ORDER) and \
            h2f["order_sampled"] == h2f["order_exact"]
        verdicts["H2a"] = {"verdict": "CONFIRMED" if h2a_ok else "REFUTED", "evidence": h2f}
        cond1 = h2f["mean_abs_err_corrected"] < h2f["mean_abs_err_sampled"]
        cond2 = any(v[0] >= 8 and h2f["pair_flips_corrected"][k][0] <= 2
                    for k, v in h2f["pair_flips_sampled"].items())
        verdicts["H2b"] = {"verdict": "CONFIRMED" if (cond1 and cond2) else "REFUTED",
                           "evidence": {"err_sampled": h2f["mean_abs_err_sampled"],
                                        "err_corrected": h2f["mean_abs_err_corrected"],
                                        "condition_error_reduction": bool(cond1),
                                        "condition_flip_pattern": bool(cond2),
                                        "pair_flips_sampled": h2f["pair_flips_sampled"],
                                        "pair_flips_corrected": h2f["pair_flips_corrected"]}}
        for arm in ("arm_t_kuairec", "arm_t_ml1m"):
            verdicts[f"H2_replication_{arm}"] = h2_block(arm)
    try:
        h2()
    except Exception as exc:  # noqa: BLE001
        for key in ("H2a", "H2b"):
            verdicts.setdefault(key, {"verdict": "INCONCLUSIVE",
                                      "reason": f"{type(exc).__name__}: {exc}"})

    # H3a
    def h3a() -> dict:
        def h3_block(ds: str, arm: str) -> dict:
            per_ranker = {}
            for name in ("Popularity", "ItemKNN", "EASE", "iALS"):
                vals = [_f(r["ndcg_random"]) for r in random_rows
                        if r["dataset"] == ds and r["ranker"] == name]
                per_ranker[name] = {"random_mean": float(np.mean(vals)) if vals else None,
                                    "temporal": exact(arm, name),
                                    "random_exceeds": bool(vals and np.mean(vals) > exact(arm, name))}
            return per_ranker
        h3k = h3_block("kuairec_big", "arm_t_kuairec")
        h3m = h3_block("ml1m", "arm_t_ml1m")
        count_k = sum(1 for v in h3k.values() if v["random_exceeds"])
        count_m = sum(1 for v in h3m.values() if v["random_exceeds"])
        return {"verdict": "CONFIRMED" if count_k >= 3 else "REFUTED",
                "evidence": {"kuairec_rankers_random_exceeds": count_k, "kuairec": h3k,
                             "ml1m_rankers_random_exceeds": count_m, "ml1m": h3m}}
    guarded("H3a", h3a)

    # H3b
    def h3b() -> dict:
        clean_ok = bool(leakage.get("clean", {}).get("pair_overlap_ok")
                        and leakage.get("clean", {}).get("time_order_ok"))
        return {"verdict": "CONFIRMED" if (clean_ok and leakage.get("audit_flags_planted")) else "REFUTED",
                "evidence": leakage}
    guarded("H3b", h3b)

    # H4a / H4b / S1 from the exposure arm.
    def tau_diff(gamma: float, condition: str, est_a: str, est_b: str = "naive") -> dict:
        rows = [r for r in tau_rows if _f(r["gamma"]) == gamma and r["condition"] == condition]
        diffs = np.array([_f(r[f"tau_{est_a}"]) - _f(r[f"tau_{est_b}"]) for r in rows])
        res = paired_bootstrap(diffs, n_boot=BOOT, seed=0)
        return {"mean_tau_diff": float(diffs.mean()), "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
                "n_seeds": int(diffs.size)}

    def h4a() -> dict:
        ips = tau_diff(1.0, "true", "ips")
        snips = tau_diff(1.0, "true", "snips")
        ok = (ips["mean_tau_diff"] > 0 and ips["ci_lo"] > 0
              and snips["mean_tau_diff"] > 0 and snips["ci_lo"] > 0)
        return {"verdict": "CONFIRMED" if ok else "REFUTED",
                "rule": "substituted rule, DECISIONS.md D6",
                "evidence": {"ips_vs_naive": ips, "snips_vs_naive": snips}}
    guarded("H4a", h4a)

    def h4b() -> dict:
        h = tau_diff(1.0, "estimated", "snips")
        return {"verdict": "CONFIRMED" if (h["mean_tau_diff"] > 0 and h["ci_lo"] > 0) else "REFUTED",
                "evidence": h}
    guarded("H4b", h4b)

    def s1() -> dict:
        bias_rows = [r for r in est_rows if _f(r["gamma"]) == 1.0 and r["condition"] == "true"]
        seeds = sorted({int(r["seed"]) for r in bias_rows})
        diffs = []
        for s in seeds:
            pop = [r for r in bias_rows if int(r["seed"]) == s and r["ranker"] == "Popularity"][0]
            ease = [r for r in bias_rows if int(r["seed"]) == s and r["ranker"] == "EASE"][0]
            diffs.append((_f(pop["naive"]) - _f(pop["truth"])) - (_f(ease["naive"]) - _f(ease["truth"])))
        res = paired_bootstrap(np.array(diffs), n_boot=BOOT, seed=0)
        return {"verdict": "CONFIRMED" if (np.mean(diffs) > 0 and res["ci_lo"] > 0) else "REFUTED",
                "evidence": {"mean_bias_difference_popularity_minus_ease": float(np.mean(diffs)),
                             "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"], "n_seeds": len(seeds)}}
    guarded("S1", s1)

    # H5
    def h5() -> dict:
        slice_orders = {}
        diff = False
        for slice_name in ("head_top1pct", "torso_next9pct", "tail_rest"):
            rows = [r for r in slice_rows if r["slice"] == slice_name]
            t_order = _order_of({r["ranker"]: _f(r["truth_ndcg"]) for r in rows})
            s_order = _order_of({r["ranker"]: _f(r["sampled_ndcg_mean"]) for r in rows})
            slice_orders[slice_name] = {"truth_order": t_order, "sampled_order": s_order}
            if t_order != s_order:
                diff = True
        return {"verdict": "CONFIRMED" if diff else "REFUTED", "evidence": slice_orders}
    guarded("H5", h5)

    # H6
    def h6() -> dict:
        pooled = [r for r in pop_rows if r["arm"] == "arm_f" and r["ranker"] == "POOLED"][0]
        share = _f(pooled["top10pct_share"])
        n_rec = int(_f(pooled["n_recommendations"]))
        z = (share - 0.10) / np.sqrt(0.1 * 0.9 / n_rec)
        return {"verdict": "CONFIRMED" if z > 1.645 else "REFUTED",
                "evidence": {"top10pct_share": share, "top1pct_share": _f(pooled["top1pct_share"]),
                             "n_recommendations": n_rec, "z": float(z), "gini_pooled": _f(pooled["gini"])}}
    guarded("H6", h6)

    # H7
    def h7() -> dict:
        def cold_gap(arm: str):
            rows = {(r["slice"], r["ranker"]): r for r in cold_rows if r["arm"] == arm}
            knn = rows.get(("item_cold", "ItemKNN"))
            pop = rows.get(("item_cold", "Popularity"))
            if not knn or not pop or knn["ndcg"] in ("", "None") or pop["ndcg"] in ("", "None"):
                return None
            return {"n_users": int(_f(knn["n_users"])),
                    "gap_cold": _f(knn["ndcg"]) - _f(pop["ndcg"]),
                    "gap_full": exact(arm, "ItemKNN") - exact(arm, "Popularity")}
        arm_used, h = "arm_t_kuairec", cold_gap("arm_t_kuairec")
        if h is None or h["n_users"] < 30:
            arm_used, h = "arm_t_ml1m", cold_gap("arm_t_ml1m")
        return {"verdict": "CONFIRMED" if (h and h["gap_cold"] < h["gap_full"]) else "REFUTED",
                "evidence": {"arm_used": arm_used, **(h or {})}}
    guarded("H7", h7)

    # Replication gate (section 9): v2's temporal arm against v1's
    # five standing verdicts and their decisive numbers.
    def v2_numbers(key: str) -> dict:
        ev = verdicts.get(key, {}).get("evidence", {})
        if key == "H1a":
            return {"mean_diff": ev.get("itemknn_minus_popularity_mean"),
                    "one_sided_95_lower_bound": ev.get("one_sided_95_lower_bound"),
                    "p_holm": ev.get("p_holm"), "n_users": ev.get("n_users")}
        if key == "H1b":
            return {"mean_diff": ev.get("ease_minus_itemknn_mean"),
                    "one_sided_95_lower_bound": ev.get("one_sided_95_lower_bound"),
                    "p_holm": ev.get("p_holm"), "n_users": ev.get("n_users")}
        if key == "H3a":
            return {"kuairec_rankers_random_exceeds": ev.get("kuairec_rankers_random_exceeds"),
                    "ml1m_rankers_random_exceeds": ev.get("ml1m_rankers_random_exceeds")}
        if key == "H3b":
            return {"clean_pair_overlap": ev.get("clean", {}).get("pair_overlap"),
                    "planted_pair_overlap": ev.get("planted", {}).get("pair_overlap")}
        if key == "H7":
            return {"gap_cold": ev.get("gap_cold"), "gap_full": ev.get("gap_full"),
                    "n_users": ev.get("n_users")}
        return {}

    replication = []
    for key, target in V1_RECORD.items():
        v2n = v2_numbers(key)
        diffs = {}
        for name, v1v in target["numbers"].items():
            v2v = v2n.get(name)
            if isinstance(v1v, (int, float)) and isinstance(v2v, (int, float)):
                diffs[name] = abs(float(v2v) - float(v1v))
        v2_verdict = verdicts.get(key, {}).get("verdict")
        replication.append({
            "hypothesis": key,
            "v1_verdict": target["verdict"],
            "v2_verdict": v2_verdict,
            "verdict_reproduced": bool(v2_verdict == target["verdict"]),
            "v1_numbers": target["numbers"],
            "v2_numbers": v2n,
            "abs_diff": diffs,
        })
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "replication.json").write_text(json.dumps(replication, indent=2))
    verdicts["replication_gate"] = {
        "verdict": "REPRODUCED" if all(r["verdict_reproduced"] for r in replication) else "DIFFERS",
        "hypotheses": [r["hypothesis"] for r in replication],
        "detail_file": "results/replication.json"}

    # Track-level GO conditions (v1 pack section 1.2 / 1.3, carried).
    import subprocess
    prereg_hash = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", "PREREGISTRATION.md"],
        cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    coverage_records = list(COVERAGE_RECORDS)
    cov_path = RESULTS / "coverage.json"
    if cov_path.exists():
        on_disk = json.loads(cov_path.read_text())
        if len(on_disk) > len(coverage_records):
            coverage_records = on_disk
    conditions = {
        "second_dataset_added": any(r["arm"] == "arm_t_ml1m" for r in arm_metrics.values()),
        "causal_estimator_implemented": len(est_rows) > 0,
        "exact_metric_calibration_and_correction": len(sampled) > 0,
        "cold_start_and_popularity_slices_logged": len(cold_rows) > 0 and len(pop_rows) > 0,
        "coverage_gate_passed_all_arms": bool(coverage_records)
        and all(r.get("gate_passed") for r in coverage_records),
        "preregistration_pinned": bool(prereg_hash),
        "preregistration_commit": prereg_hash[0] if prereg_hash else None,
    }
    verdicts["track"] = {
        "verdict": "GO" if all(v for k, v in conditions.items() if k != "preregistration_commit") else "CONDITIONAL GO",
        "conditions": conditions}
    (RESULTS / "verdicts.json").write_text(json.dumps(verdicts, indent=2))
    log("verdicts written")
    return verdicts


def _save_chosen(chosen: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "chosen_configs.json"
    current = {}
    if path.exists():
        current = json.loads(path.read_text())
    current.update(chosen)
    path.write_text(json.dumps(current, indent=2))


def _load_chosen() -> dict:
    path = RESULTS / "chosen_configs.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["audit", "arm_t", "ml1m", "arm_f", "random", "verdicts", "all"])
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    if args.stage in ("audit", "all"):
        import subprocess
        out = subprocess.run([sys.executable, str(ROOT / "experiments" / "audit_data.py")],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stderr[-3000:])
            return 1
        (RESULTS / "data_audit.json").write_text(out.stdout)
        log("audit written to results/data_audit.json")
        if args.stage == "audit":
            return 0

    chosen = _load_chosen()
    # Load only the datasets the requested stage needs: the raw event
    # arrays are several hundred MB and the machine is shared (D10).
    big = small = ml = None
    if args.stage == "arm_t":
        big = load_big()
    elif args.stage == "ml1m":
        ml = load_ml()
    elif args.stage == "arm_f":
        small = load_small()
    elif args.stage == "random":
        big, ml = load_big(), load_ml()
    elif args.stage == "all":
        big, small, ml = load_all()

    try:
        if args.stage in ("arm_t", "all"):
            holder: dict = {}
            pos = big["watch_ratio"] > 2.0
            if args.stage == "arm_t":
                del big["watch_ratio"]  # labels are frozen in pos; free the raw array
            stage_arm_t("arm_t_kuairec", big, pos, holder)
            chosen.update(holder)
            _save_chosen(chosen)
        if args.stage in ("ml1m", "all"):
            holder = {}
            stage_arm_t("arm_t_ml1m", ml, ml["rating"] >= 4.0, holder)
            chosen.update(holder)
            _save_chosen(chosen)
        if args.stage in ("arm_f", "all"):
            if "arm_t_kuairec" not in chosen:
                raise SystemExit("arm_f needs chosen configs from arm_t; run arm_t first")
            stage_arm_f(small, chosen)
        if args.stage in ("random", "all"):
            stage_random(big, ml, chosen)
        if args.stage in ("verdicts", "all"):
            v = stage_verdicts()
            print(json.dumps({k: (val.get("verdict") if isinstance(val, dict) else val)
                              for k, val in v.items()}, indent=2))
    except GateStop as stop:
        # A gate stop ends the run; results/coverage.json (already
        # written by record_coverage) is the output (stop rule S5).
        log(f"GATE STOP: {stop}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
