"""Embed every CoT sentence to build S (commit item 3).

For each model, collect all unique sentence texts and embed them once with OpenAI
``text-embedding-3-large`` (3072-d), caching to ``results/plain/sentence_emb_cache/{model}.npz``
so a rebuild never re-hits the API. The cache stores ``texts`` and ``emb`` (float16); the
latent cache later truncates to the first 256 Matryoshka dims.

Run: ``OPENAI_API_KEY=... python embed_sentences.py``
"""
from __future__ import annotations

import numpy as np

import sentence_text
from cache_latents import MODELS, VARIANT
from config import Config
from oai import embed


def build_model_embeddings(cfg: Config, model: str) -> None:
    dest = cfg.sentence_emb_cache_npz(VARIANT, model)
    if dest.exists():
        print(f"  {model}: cache exists, skipping", flush=True)
        return
    texts = sentence_text.load_sentences(cfg, VARIANT, model)          # (pmcid, idx) -> text
    uniq = sorted({t for t in texts.values() if t})
    print(f"  {model}: embedding {len(uniq)} unique sentences", flush=True)
    U = embed(uniq, model=cfg.embedding_model, dim=cfg.embedding_dim)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, texts=np.array(uniq, object), emb=U.astype(np.float16))


def main() -> None:
    cfg = Config()
    for model in MODELS:
        build_model_embeddings(cfg, model)
    print("EMBED_DONE", flush=True)


if __name__ == "__main__":
    main()
