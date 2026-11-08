"""Dataset loading. Raw files live under data/raw/ and are never committed.

KuaiRec matrices are CSV with header:
user_id, video_id, play_duration, video_duration, time, date, timestamp,
watch_ratio. Only user_id, video_id, timestamp and watch_ratio are loaded.
MovieLens-1M ratings.dat rows are UserID::MovieID::Rating::Timestamp.
"""
from __future__ import annotations

import csv

import numpy as np


def _remap(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, inv = np.unique(ids, return_inverse=True)
    return inv.astype(np.int64), uniq


def load_kuairec(path: str) -> dict:
    users: list[int] = []
    items: list[int] = []
    stamps: list[float] = []
    ratios: list[float] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        iu, ii, it, ir = col["user_id"], col["video_id"], col["timestamp"], col["watch_ratio"]
        for row in reader:
            users.append(int(row[iu]))
            items.append(int(row[ii]))
            # 181,992 small-matrix rows ship with empty time fields (audit
            # A2). Their watch_ratio is present, so Arm F can use the label;
            # the timestamp is stored as NaN and excluded from time rules.
            stamps.append(float(row[it]) if row[it] else float("nan"))
            ratios.append(float(row[ir]))
    u, u_uniq = _remap(np.asarray(users))
    i, i_uniq = _remap(np.asarray(items))
    return {
        "user": u, "item": i,
        "timestamp": np.asarray(stamps, dtype=np.float64),
        "watch_ratio": np.asarray(ratios, dtype=np.float64),
        "n_users": int(u_uniq.size), "n_items": int(i_uniq.size),
        "user_ids": u_uniq, "item_ids": i_uniq,
    }


def load_ml1m(path: str) -> dict:
    users: list[int] = []
    items: list[int] = []
    stamps: list[float] = []
    ratings: list[float] = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 4:
                continue
            users.append(int(parts[0]))
            items.append(int(parts[1]))
            ratings.append(float(parts[2]))
            stamps.append(float(parts[3]))
    u, u_uniq = _remap(np.asarray(users))
    i, i_uniq = _remap(np.asarray(items))
    return {
        "user": u, "item": i,
        "timestamp": np.asarray(stamps, dtype=np.float64),
        "rating": np.asarray(ratings, dtype=np.float64),
        "n_users": int(u_uniq.size), "n_items": int(i_uniq.size),
        "user_ids": u_uniq, "item_ids": i_uniq,
    }


def pair_keys(user: np.ndarray, item: np.ndarray, n_items: int) -> np.ndarray:
    return user.astype(np.int64) * np.int64(n_items) + item.astype(np.int64)


def binary_matrix(user: np.ndarray, item: np.ndarray, values: np.ndarray,
                  n_users: int, n_items: int, dtype=np.float32) -> np.ndarray:
    """Dense matrix with the maximum value kept per (user, item) pair."""
    mat = np.zeros((n_users, n_items), dtype=dtype)
    order = np.argsort(values)
    mat[user[order], item[order]] = values[order].astype(dtype)
    return mat
