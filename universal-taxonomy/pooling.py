"""Pool per-sentence vectors across models into one labelled sample.

Loading every model's full vectors at once is ~8 GB fp16 per variant, so for
clustering + metrics we draw a fixed-seed, per-model-proportional sample. The
full vectors remain on disk as the deliverable; the taxonomy is derived from a
representative sample (per-model sizes stay in the thousands, ample for coverage
statistics).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class Pool:
    X: np.ndarray          # (n, D) float32
    model: np.ndarray      # (n,) object - model id
    pmcid: np.ndarray      # (n,) object
    sent: np.ndarray       # (n,) int32


def sample_pool(
    cfg: Config,
    variant: str,
    *,
    normalized: bool = True,
    n_total: int = 150_000,
    models: List[str] | None = None,
    subset: str = "all",
    seed: int = 42,
) -> Pool:
    import splits
    models = models or cfg.models
    # per-model indices in the requested split
    allowed: Dict[str, np.ndarray] = {}
    for m in models:
        d = np.load(cfg.sentence_vec_npz(variant, m, normalized=normalized), allow_pickle=True)
        mask = splits.subset_mask(d["pmcid"], d["sentence_idx"], subset, seed=seed)
        allowed[m] = np.where(mask)[0]
    total = sum(len(v) for v in allowed.values())
    rng = np.random.default_rng(seed)

    Xs, mods, pms, sts = [], [], [], []
    for m in models:
        d = np.load(cfg.sentence_vec_npz(variant, m, normalized=normalized), allow_pickle=True)
        pool_idx = allowed[m]
        n = len(pool_idx)
        take = min(n, max(1, round(n_total * n / total)))
        sel = np.sort(rng.choice(pool_idx, size=take, replace=False))
        Xs.append(d["vecs"][sel].astype(np.float32))
        mods.append(np.full(take, m, dtype=object))
        pms.append(d["pmcid"][sel])
        sts.append(d["sentence_idx"][sel].astype(np.int32))
        logger.info("  %s: sampled %d / %d (%s)", m, take, n, subset)

    pool = Pool(
        X=np.concatenate(Xs),
        model=np.concatenate(mods),
        pmcid=np.concatenate(pms),
        sent=np.concatenate(sts),
    )
    logger.info("pooled sample: %s", pool.X.shape)
    return pool


def split_pool(pool: Pool, *, test_frac: float = 0.3, seed: int = 42):
    """Deterministic per-row train/test split of a pool."""
    n = len(pool.X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])

    def take(ix):
        return Pool(X=pool.X[ix], model=pool.model[ix], pmcid=pool.pmcid[ix], sent=pool.sent[ix])

    return take(train_idx), take(test_idx)
