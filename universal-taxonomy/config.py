"""Configuration and path resolution for the universal-taxonomy pipeline.

All filesystem layout knowledge lives here so the rest of the package never
hard-codes a path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# The 10 reasoning models that have a full 36-SAE annotation set.
MODELS: List[str] = [
    "deepseek-r1-distill-llama-8b",
    "deepseek-r1-distill-qwen-14b",
    "deepseek-r1-distill-qwen-1.5b",
    "gemma-4-31b-it",
    "glm-4.7-flash",
    "gpt-oss-20b",
    "huatuogpt-o1-8b",
    "ministral-3-14b-reasoning-2512",
    "qwen3.6-27b",
    "qwq-32b",
]

N_CLUSTERS: List[int] = [10, 12, 14, 16, 18, 20]

VARIANTS: List[str] = ["dedup", "plain"]

# variant -> directory name under universal-taxonomy/results/vars/latents*
_LATENTS_DIRNAME = {
    "plain": "latents",
    "dedup": "latents_dedup",
}

def _find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until the repo root (contains ``train-saes`` + ``universal-taxonomy``)."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "train-saes").is_dir() and (parent / "universal-taxonomy").is_dir():
            return parent
    raise FileNotFoundError("Could not locate repo root (train-saes/ + universal-taxonomy/).")


@dataclass(frozen=True)
class Config:
    """Immutable run configuration."""

    repo_root: Path = field(default_factory=_find_repo_root)

    models: List[str] = field(default_factory=lambda: list(MODELS))
    n_clusters: List[int] = field(default_factory=lambda: list(N_CLUSTERS))

    # OpenAI models: embeddings (sentences), and the single final-titling LLM.
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    naming_model: str = "gpt-4o-mini"

    seed: int = 42

    @property
    def out_root(self) -> Path:
        return self.repo_root / "universal-taxonomy" / "results"

    # ---- source data ----------------------------------------------------
    def latents_dir(self, variant: str, model: str) -> Path:
        return self.out_root / "vars" / _LATENTS_DIRNAME[variant] / model

    def latents_npz(self, variant: str, model: str, layer: int) -> Path:
        return self.latents_dir(variant, model) / f"layer{layer}" / "latents.npz"

    def index_npz(self, variant: str, model: str, layer: int) -> Path:
        return self.latents_dir(variant, model) / f"layer{layer}" / "index.npz"

    def discover_layers(self, variant: str, model: str) -> List[int]:
        """Layers present for a (variant, model), read from the latents dir."""
        d = self.latents_dir(variant, model)
        layers = sorted(
            int(p.name.removeprefix("layer"))
            for p in d.glob("layer*")
            if p.is_dir() and p.name.removeprefix("layer").isdigit()
        )
        if not layers:
            raise FileNotFoundError(f"No layer dirs under {d}")
        return layers

    # ---- outputs --------------------------------------------------------
    def exemplar_emb_npz(self, variant: str, model: str) -> Path:
        """Geometric latent basis E (cross-covariance) for a model."""
        return self.out_root / variant / "exemplar_embeddings" / f"{model}.npz"

    def sentence_emb_cache_npz(self, variant: str, model: str) -> Path:
        """Cache of per-model unique-sentence embeddings, so rebuilding E never
        re-hits the embedding API."""
        return self.out_root / variant / "sentence_emb_cache" / f"{model}.npz"

    def sentence_vec_npz(self, variant: str, model: str, *, normalized: bool) -> Path:
        tag = "norm" if normalized else "raw"
        return self.out_root / variant / "sentence_vectors" / f"{model}_{tag}.npz"
