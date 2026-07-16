"""Deterministic per-sentence train/test membership.

A sentence's split is a stable hash of its ``(pmcid, sentence_idx)`` key, so the
Approach B search (train only) and the final evaluation (test only) are
guaranteed disjoint without persisting any split file. The same membership is
used for Approach A, so the A-vs-B comparison is on identical held-out data.
"""

from __future__ import annotations

import hashlib

import numpy as np

TEST_FRAC = 0.3


def _bucket(key: str, seed: int) -> float:
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return (int(h[:8], 16) % 10_000) / 10_000.0


def is_test(pmcid: str, sent: int, *, test_frac: float = TEST_FRAC, seed: int = 42) -> bool:
    return _bucket(f"{pmcid}\t{sent}", seed) < test_frac


def subset_mask(pmcid: np.ndarray, sent: np.ndarray, subset: str,
                *, test_frac: float = TEST_FRAC, seed: int = 42) -> np.ndarray:
    """Boolean mask selecting rows in ``subset`` ('train' | 'test' | 'all')."""
    if subset == "all":
        return np.ones(len(pmcid), dtype=bool)
    buckets = np.array([_bucket(f"{p}\t{int(s)}", seed) for p, s in zip(pmcid, sent)])
    test = buckets < test_frac
    return ~test if subset == "train" else test
