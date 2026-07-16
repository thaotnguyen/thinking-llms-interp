"""Approach A: aggregate all 36 SAEs into one per-sentence vector per model.

Projects every sentence through the full 540-latent title-embedding matrix and
writes fp16 sentence vectors keyed by (pmcid, sentence_idx).

Run::

    python approach_a.py --variant nndedup                 # all models, normalized
    python approach_a.py --variant dedup --raw             # raw-weight ablation
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

import projection
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_model(cfg: Config, variant: str, model: str, *, normalize: bool, force: bool) -> None:
    out = cfg.sentence_vec_npz(variant, model, normalized=normalize)
    if out.exists() and not force:
        logger.info("skip %s/%s (%s) exists", variant, model, "norm" if normalize else "raw")
        return
    vecs, keys = projection.project(cfg, variant, model, normalize=normalize)
    pmcid = np.array([k.split("\t")[0] for k in keys], dtype=object)
    sent = np.array([int(k.split("\t")[1]) for k in keys], dtype=np.int32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        vecs=vecs.astype(np.float16),
        pmcid=pmcid,
        sentence_idx=sent,
    )
    logger.info("wrote %s  vecs=%s (%s)", out, vecs.shape, "norm" if normalize else "raw")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["nndedup", "dedup", "plain"], required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--raw", action="store_true", help="raw weights (default: per-SAE L2-normalized)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    models = args.models or cfg.models
    for model in models:
        run_model(cfg, args.variant, model, normalize=not args.raw, force=args.force)


if __name__ == "__main__":
    main()
