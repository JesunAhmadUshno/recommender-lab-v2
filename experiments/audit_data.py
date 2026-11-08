"""Data audit (design checks A1 to A4, plus the v2 split-construction
coverage descriptives). Descriptive statistics only: row counts,
timestamp ranges and quantiles, id overlap, label rates, and the
coverage of each frozen split construction. No model is fitted and no
evaluation metric is computed here. The output pins the split cutoffs
that PREREGISTRATION.md freezes and records the coverage the gate
will re-check at run time.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import split as split_mod  # noqa: E402
from src.coverage import compute_coverage  # noqa: E402
from src.dataio import load_kuairec, load_ml1m, pair_keys  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(name: str, d: dict, label: np.ndarray) -> dict:
    ts = d["timestamp"]
    return {
        "dataset": name,
        "rows": int(d["user"].size),
        "n_users": d["n_users"],
        "n_items": d["n_items"],
        "ts_missing": int(np.isnan(ts).sum()),
        "ts_min": float(np.nanmin(ts)),
        "ts_max": float(np.nanmax(ts)),
        "ts_q80": float(np.nanquantile(ts, 0.80)),
        "ts_q90": float(np.nanquantile(ts, 0.90)),
        "positive_rate": float(label.mean()),
        "users_with_zero_positives": int((np.bincount(d["user"][label], minlength=d["n_users"]) == 0).sum()),
    }


def temporal_coverage(d: dict, pos: np.ndarray) -> dict:
    """Split-construction coverage for one temporal arm (descriptive)."""
    ts = d["timestamp"]
    t1 = float(np.nanquantile(ts, 0.80))
    t2 = float(np.nanquantile(ts, 0.90))
    tr, _va, te = split_mod.temporal_masks(ts, t1, t2)
    cand = te & pos
    first = split_mod.first_time_mask(d["user"][tr], d["item"][tr],
                                      d["user"][cand], d["item"][cand], d["n_items"])
    users = np.unique(d["user"][cand][first])
    items = np.unique(d["item"][cand][first])
    rec = compute_coverage(d["user"][tr], d["item"][tr], users, items,
                           d["n_users"], d["n_items"])
    return {"t1": t1, "t2": t2, "train_events": int(tr.sum()),
            "test_positives": int(first.sum()), **rec}


def main() -> int:
    out: dict = {"files": {}}
    for fname in ("KuaiRec.zip", "ml-1m.zip"):
        p = RAW / fname
        out["files"][fname] = {"sha256": sha256(p), "bytes": p.stat().st_size}

    big = load_kuairec(str(RAW / "big_matrix.csv"))
    small = load_kuairec(str(RAW / "small_matrix.csv"))
    big_pos2 = big["watch_ratio"] > 2.0
    big_pos1 = big["watch_ratio"] > 1.0
    small_pos2 = small["watch_ratio"] > 2.0
    small_pos1 = small["watch_ratio"] > 1.0

    out["kuairec_big"] = describe("kuairec_big", big, big_pos2)
    out["kuairec_big"]["positive_rate_gt1"] = float(big_pos1.mean())
    out["kuairec_small"] = describe("kuairec_small", small, small_pos2)
    out["kuairec_small"]["positive_rate_gt1"] = float(small_pos1.mean())

    # A2/A3: timeline and id overlap between the matrices (the v1 defect
    # measures, kept so this audit re-verifies them on the same bytes).
    t_s = float(np.nanmin(small["timestamp"]))
    out["arm_f"] = {
        "small_ts_min": t_s,
        "small_ts_median": float(np.nanmedian(small["timestamp"])),
        "big_frac_before_small_min": float((big["timestamp"] < t_s).mean()),
        "small_users_subset_of_big": bool(np.isin(small["user_ids"], big["user_ids"]).all()),
        "small_items_subset_of_big": bool(np.isin(small["item_ids"], big["item_ids"]).all()),
        "small_user_count_in_big_ids": int(np.isin(small["user_ids"], big["user_ids"]).sum()),
        "small_item_count_in_big_ids": int(np.isin(small["item_ids"], big["item_ids"]).sum()),
    }
    big_keys = np.unique(pair_keys(big["user"], big["item"], big["n_items"]))
    big_user_index = {int(v): i for i, v in enumerate(big["user_ids"])}
    big_item_index = {int(v): i for i, v in enumerate(big["item_ids"])}
    su = np.array([big_user_index[int(v)] for v in small["user_ids"]])[small["user"]]
    si = np.array([big_item_index[int(v)] for v in small["item_ids"]])[small["item"]]
    small_keys = pair_keys(su, si, big["n_items"])
    out["arm_f"]["small_pairs_present_in_big"] = int(np.isin(small_keys, big_keys).sum())
    out["arm_f"]["small_pairs_total"] = int(small_keys.size)
    on_small_items = np.isin(big["item"], np.array([big_item_index[int(v)] for v in small["item_ids"]]))
    out["arm_f"]["big_events_on_small_items"] = int(on_small_items.sum())
    # The defect in one number: big events by small users on small items.
    on_small_users = np.isin(big["user"], np.array([big_user_index[int(v)] for v in small["user_ids"]]))
    out["arm_f"]["big_events_by_small_users_on_small_items"] = int((on_small_users & on_small_items).sum())
    out["arm_f"]["big_events_by_small_users_total"] = int(on_small_users.sum())

    # v2 split-construction coverage (descriptive; the gate re-checks
    # these at run time before any fitting).
    out["coverage_audit"] = {
        "arm_t_kuairec": temporal_coverage(big, big_pos2),
        "arm_f_v2": temporal_coverage(small, small_pos2),
    }
    # Arm F sensitivity pool: train before the small median, same truth.
    cov_f = out["coverage_audit"]["arm_f_v2"]
    ts = small["timestamp"]
    tr_med = ts < float(np.nanmedian(ts))
    _tr, _va, te = split_mod.temporal_masks(ts, cov_f["t1"], cov_f["t2"])
    cand = te & small_pos2
    first = split_mod.first_time_mask(small["user"][_tr], small["item"][_tr],
                                      small["user"][cand], small["item"][cand],
                                      small["n_items"])
    rec = compute_coverage(small["user"][tr_med], small["item"][tr_med],
                           np.unique(small["user"][cand][first]),
                           np.unique(small["item"][cand][first]),
                           small["n_users"], small["n_items"])
    out["coverage_audit"]["arm_f_v2_sensitivity_pool"] = {
        "train_events": int(tr_med.sum()), **rec}

    ml = load_ml1m(str(RAW / "ml-1m" / "ratings.dat"))
    ml_pos = ml["rating"] >= 4.0
    out["ml1m"] = describe("ml1m", ml, ml_pos)
    out["ml1m"]["rating_counts"] = {str(int(r)): int((ml["rating"] == r).sum()) for r in (1, 2, 3, 4, 5)}
    out["coverage_audit"]["arm_t_ml1m"] = temporal_coverage(ml, ml_pos)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
