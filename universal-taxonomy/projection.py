"""Project per-sentence SAE latents into shared title-embedding space.

The whole map is linear: for a model, stack every latent's title embedding into
``E`` (rows in canonical (layer, k, idx) order) and every sentence's latent
values into ``V``; then ``sentence_vectors = V @ E``.

``build_values`` assembles ``V`` from the per-layer ``latents.npz`` sidecars,
optionally per-SAE L2-normalizing each ``(N, K)`` block so no single layer / K
dominates the sum. Rows are aligned across a model's layers by
``(pmcid, sentence_idx)`` (a strict key-join, since e.g. gemma layer 18 is a few
CoTs short of its siblings).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import Config

logger = logging.getLogger(__name__)


def _row_keys(cfg: Config, variant: str, model: str, layer: int) -> np.ndarray:
    idx = np.load(cfg.index_npz(variant, model, layer), allow_pickle=True)
    pmcid = idx["pmcid"].astype(str)
    sent = idx["sentence_idx"].astype(np.int64)
    # structured key array for fast intersection
    return np.array([f"{p}\t{s}" for p, s in zip(pmcid, sent)], dtype=object)


def common_keys(cfg: Config, variant: str, model: str, layers: List[int]) -> np.ndarray:
    """Ordered array of (pmcid\\tsentence_idx) keys present in *all* layers."""
    keysets = [set(_row_keys(cfg, variant, model, l).tolist()) for l in layers]
    common = set.intersection(*keysets)
    # deterministic order: use layer[0]'s order filtered to the intersection
    first = _row_keys(cfg, variant, model, layers[0])
    ordered = [k for k in first.tolist() if k in common]
    return np.array(ordered, dtype=object)


def _l2norm_blocks(block: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize a single (N, K) SAE block (zeros stay zero)."""
    n = np.linalg.norm(block, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return block / n


def build_values(
    cfg: Config,
    variant: str,
    model: str,
    *,
    normalize: bool,
    layers: Optional[List[int]] = None,
    ks: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    """Assemble ``V`` for a model.

    Returns ``(V, keys, col_meta)`` where ``V`` is ``(N, L)`` in canonical
    (layer, k, idx) column order, ``keys`` is the ``(N,)`` row key array, and
    ``col_meta`` lists ``(layer, k, idx)`` per column.

    ``layers`` / ``ks`` restrict which SAEs contribute (Approach B selects a
    single (layer, k)); default is all.
    """
    layers = layers or cfg.discover_layers(variant, model)
    ks = ks or cfg.n_clusters

    keys = common_keys(cfg, variant, model, layers)
    key_pos: Dict[str, int] = {k: i for i, k in enumerate(keys.tolist())}
    n = len(keys)

    cols: List[np.ndarray] = []
    col_meta: List[Tuple[int, int, int]] = []
    for layer in layers:
        data = np.load(cfg.latents_npz(variant, model, layer))
        lkeys = _row_keys(cfg, variant, model, layer)
        # map this layer's rows onto the common order
        sel = np.full(n, -1, dtype=np.int64)
        for row_i, kk in enumerate(lkeys.tolist()):
            p = key_pos.get(kk)
            if p is not None:
                sel[p] = row_i
        assert (sel >= 0).all(), f"missing rows for {model} layer{layer}"
        for k in ks:
            block = data[f"k{k}"][sel]  # (N, K)
            if normalize:
                block = _l2norm_blocks(block)
            cols.append(block.astype(np.float32))
            col_meta.extend((layer, k, i) for i in range(k))
    V = np.concatenate(cols, axis=1)
    return V, keys, col_meta


def load_embeddings(
    cfg: Config,
    variant: str,
    model: str,
    col_meta: List[Tuple[int, int, int]],
) -> np.ndarray:
    """Load the geometric latent basis ``E`` and reorder rows to match ``col_meta``."""
    d = np.load(cfg.exemplar_emb_npz(variant, model))
    emb = d["emb"]
    lookup: Dict[Tuple[int, int, int], int] = {
        (int(l), int(k), int(i)): r
        for r, (l, k, i) in enumerate(zip(d["layer"], d["k"], d["idx"]))
    }
    rows = [lookup[m] for m in col_meta]
    return emb[rows]


def project(
    cfg: Config,
    variant: str,
    model: str,
    *,
    normalize: bool,
    layers: Optional[List[int]] = None,
    ks: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(sentence_vectors (N, D), keys (N,))`` for a model."""
    V, keys, col_meta = build_values(
        cfg, variant, model, normalize=normalize, layers=layers, ks=ks
    )
    E = load_embeddings(cfg, variant, model, col_meta)
    vecs = (V @ E).astype(np.float32)
    return vecs, keys
