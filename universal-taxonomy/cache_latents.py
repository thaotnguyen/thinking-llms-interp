"""Cache L and E = Lᵀ S per SAE block so the Optuna search is fast (commit item 4).

For each model and each SAE block (layer, sae_k):
  - **L**: per-block centred (per-latent mean 0 over all sentences) + Frobenius-normalised
    latent activations.
  - **E = Lᵀ S**: latent embeddings ``(sae_k, 256)`` -- S is the raw first-256 Matryoshka
    dims of the sentence embedding (no centring, no normalisation).
  - **argmax**: each sentence's dominant latent index within the block, for routing.

Written once to ``results/plain/latent_cache/{model}.npz``; the search just stacks the
selected blocks. Selection is done on the full dataset (no train/test split).

Run: ``python cache_latents.py``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import projection
import sentence_text
from config import Config

VARIANT = "plain"
MODELS = [
    "deepseek-r1-distill-llama-8b", "deepseek-r1-distill-qwen-14b",
    "deepseek-r1-distill-qwen-1.5b", "gpt-oss-20b",
    "huatuogpt-o1-8b", "ministral-3-14b-reasoning-2512", "qwen3.6-27b", "qwq-32b",
]
S_DIMS = 256


def cache_npz(cfg: Config, model: str) -> Path:
    return cfg.out_root / VARIANT / "latent_cache" / f"{model}.npz"


def build_model_cache(cfg: Config, model: str) -> None:
    layers = sorted(cfg.discover_layers(VARIANT, model))
    V, keys, col_meta = projection.build_values(cfg, VARIANT, model, normalize=False, layers=layers)

    blocks: dict[tuple[int, int], list[int]] = {}
    for i, (layer, sae_k, _) in enumerate(col_meta):
        blocks.setdefault((layer, sae_k), []).append(i)
    Vn = np.empty_like(V, dtype=np.float32)
    for cols in blocks.values():                          # per-block centre + Frobenius -> L
        idx = np.array(cols); block = V[:, idx].astype(np.float32)
        block -= block.mean(0, keepdims=True)
        Vn[:, idx] = block / (np.linalg.norm(block) + 1e-8)

    emb = np.load(cfg.sentence_emb_cache_npz(VARIANT, model), allow_pickle=True)
    text_to_row = {t: i for i, t in enumerate(emb["texts"])}
    cached = emb["emb"].astype(np.float32)
    sent_lookup = sentence_text.load_sentences(cfg, VARIANT, model)
    keep, S, pmcids, sent_idx, texts = [], [], [], [], []
    for row, key in enumerate(keys.tolist()):
        pmcid, sidx = key.split("\t")
        text = sent_lookup.get((pmcid, int(sidx)), "")
        j = text_to_row.get(text, -1)
        if j < 0 or not text:
            continue
        S.append(cached[j]); keep.append(row)
        pmcids.append(pmcid); sent_idx.append(int(sidx)); texts.append(text)
    S = np.asarray(S, np.float32)[:, :S_DIMS]
    Vn = Vn[np.array(keep)]

    out: dict[str, np.ndarray] = {
        "pmcid": np.array(pmcids, object), "sent_idx": np.array(sent_idx, np.int32),
        "text": np.array(texts, object), "S256": S,
        "layers": np.array(sorted({l for l, _ in blocks}), np.int32),
    }
    for (layer, sae_k), cols in blocks.items():
        idx = np.array(cols)
        out[f"E__{layer}__{sae_k}"] = (Vn[:, idx].T @ S).astype(np.float32)     # (sae_k, 256)
        out[f"am__{layer}__{sae_k}"] = Vn[:, idx].argmax(1).astype(np.int16)    # routing
    dest = cache_npz(cfg, model)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print(f"  cached {model}: {S.shape[0]} sents, {len(blocks)} SAEs -> {dest.name}", flush=True)


def main() -> None:
    cfg = Config()
    for model in MODELS:
        build_model_cache(cfg, model)
    print("CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
