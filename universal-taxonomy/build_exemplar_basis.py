"""Build the geometric latent basis E via activation×embedding cross-covariance.

Each SAE latent is represented by the direction in sentence-embedding space that
co-varies with its activation:

    E_i = Σ_n (A_ni − mean_i A)(s_n − s̄)   →   E = Ã^T S̃   (per (layer, k))

where S are OpenAI sentence embeddings and A the per-sentence latent activations.
Because a latent fires on one reasoning function across many medical topics, the
co-varying direction is the shared *reasoning* direction and topic (uncorrelated
with the latent) cancels — no LLM anywhere in the representation.

Output matches the E-matrix format `projection` expects (keys `emb, layer, k,
idx, dead`, canonical (layer asc, k in N_CLUSTERS, idx) order). Unique sentence
embeddings are cached per model so this never re-hits the API on a rebuild.

Run: `python build_exemplar_basis.py --variant nndedup`
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, Tuple

import numpy as np

import sentence_text
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _row_texts(cfg: Config, variant: str, model: str, layer: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (texts, in_training) aligned to this layer's latents rows."""
    idx = np.load(cfg.index_npz(variant, model, layer), allow_pickle=True)
    pmcid = idx["pmcid"].astype(str)
    sent = idx["sentence_idx"].astype(int)
    tmap = sentence_text.load_sentences(cfg, variant, model)
    texts = np.array([tmap.get((p, int(s)), "") for p, s in zip(pmcid, sent)], dtype=object)
    in_train = idx["in_training"] if "in_training" in idx.files else np.ones(len(texts), bool)
    return texts, in_train


def _unique_embeddings(cfg: Config, variant: str, model: str,
                       layer_texts: dict) -> Tuple[Dict[str, int], np.ndarray]:
    """Unique in-training sentence embeddings, cached to avoid re-hitting the API."""
    cache = cfg.sentence_emb_cache_npz(variant, model)
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {t: i for i, t in enumerate(d["texts"])}, d["emb"].astype(np.float32)

    from oai import embed
    uniq: Dict[str, int] = {}
    for texts, in_train in layer_texts.values():
        for t in texts[in_train]:
            if t and t not in uniq:
                uniq[t] = len(uniq)
    uniq_list = list(uniq)
    logger.info("%s/%s: embedding %d unique sentences", variant, model, len(uniq_list))
    U = embed(uniq_list, model=cfg.embedding_model, dim=cfg.embedding_dim)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, texts=np.array(uniq_list, object), emb=U.astype(np.float16))
    return uniq, U


def _normalize_rows(E: np.ndarray, dead_frac: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """L2-normalize E rows; flag near-zero rows (degenerate latents) as dead."""
    norms = np.linalg.norm(E, axis=1)
    live = norms > 0
    thr = dead_frac * np.median(norms[live]) if live.any() else 0.0
    dead = norms <= thr
    out = E / np.where(norms[:, None] > 0, norms[:, None], 1.0)
    out[dead] = 0.0
    return out.astype(np.float32), dead


def _cross_covariance(cfg, variant, model, layers, layer_texts, uniq, U):
    """E_i = cov(latent i activation, sentence embedding) = Ã^T S̃, per (layer, k)."""
    rows, meta = [], []
    for layer in layers:
        data = np.load(cfg.latents_npz(variant, model, layer))
        texts, in_train = layer_texts[layer]
        ridx = np.where(in_train)[0]
        emb_idx = np.array([uniq.get(texts[i], -1) for i in ridx])
        ridx, emb_idx = ridx[emb_idx >= 0], emb_idx[emb_idx >= 0]
        S = U[emb_idx]
        S = S - S.mean(axis=0, keepdims=True)               # center S over this layer's rows
        for k in cfg.n_clusters:
            A = data[f"k{k}"][ridx].astype(np.float64)
            A = A - A.mean(axis=0, keepdims=True)            # per-latent center
            Eb = A.T @ S                                      # (K, D) cross-covariance
            rows.extend(Eb)
            meta.extend((layer, k, c) for c in range(k))
    return np.stack(rows), meta


def build_model(cfg: Config, variant: str, model: str, *, force: bool) -> None:
    out = cfg.exemplar_emb_npz(variant, model)
    if out.exists() and not force:
        logger.info("skip %s/%s (exists)", variant, model)
        return

    layers = cfg.discover_layers(variant, model)
    layer_texts = {l: _row_texts(cfg, variant, model, l) for l in layers}
    uniq, U = _unique_embeddings(cfg, variant, model, layer_texts)
    E, meta = _cross_covariance(cfg, variant, model, layers, layer_texts, uniq, U)

    emb, dead = _normalize_rows(E)
    meta_arr = np.array(meta, dtype=np.int32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, emb=emb, layer=meta_arr[:, 0], k=meta_arr[:, 1],
                        idx=meta_arr[:, 2], dead=dead)
    logger.info("wrote %s  emb=%s  dead=%d", out, emb.shape, int(dead.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["nndedup", "dedup", "plain"], required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    for model in args.models or cfg.models:
        build_model(cfg, args.variant, model, force=args.force)


if __name__ == "__main__":
    main()
