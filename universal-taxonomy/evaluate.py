"""Evaluation metrics for a taxonomy.

Search-time objectives (purely geometric, NO LLM calls) — maximized as a
multi-objective Pareto front by Approach B:
  * ch            = Calinski-Harabasz index of the clustering
  * orthogonality = mean pairwise squared cosine distance ``(1 - cos)^2`` between
                    cluster centers (higher = more distinct; opposite directions
                    are treated as maximally different, unlike ``1 - |cos|``)
  * universality  = mean over clusters of min/max per-model coverage (rewards
                    "every model expresses every category")

End-only metrics (reported, never optimized):
  * title_embedding_orthogonality = ``(1 - cos)^2`` between OpenAI embeddings of
    the final category titles/descriptions (semantic distinctness, no judge)
  * llm_accuracy = relabel held-out sentences from titles/descriptions alone
    (the single LLM-as-judge number, computed only at the very end)
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import calinski_harabasz_score

from oai import chat, map_threaded

logger = logging.getLogger(__name__)


# ---- geometric (search objectives) --------------------------------------
def orthogonality(centers: np.ndarray) -> float:
    """Mean pairwise squared cosine distance ``(1 - cos)^2`` between centers."""
    if len(centers) < 2:
        return 0.0
    n = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    cos = n @ n.T
    iu = np.triu_indices(len(centers), k=1)
    return float(((1.0 - cos[iu]) ** 2).mean())


def calinski_harabasz(Z: np.ndarray, labels: np.ndarray) -> float:
    mask = labels != -1
    lab = labels[mask]
    if len(set(lab.tolist())) < 2:
        return 0.0
    return float(calinski_harabasz_score(Z[mask], lab))


def universality(model_ids: np.ndarray, labels: np.ndarray) -> float:
    """Mean over clusters of (min / max) per-model coverage. 1 = every model
    represented in every cluster with equal within-model share."""
    cov = coverage_matrix(model_ids, labels)
    models = sorted(cov)
    clusters = sorted({c for m in cov for c in cov[m]})
    if not clusters or len(models) < 2:
        return 0.0
    scores = []
    for c in clusters:
        vals = np.array([cov[m][c] for m in models])
        mx = vals.max()
        scores.append(vals.min() / mx if mx > 0 else 0.0)
    return float(np.mean(scores))


def geometric_metrics(Z, labels, centers, model_ids) -> Dict[str, float]:
    return {
        "ch": calinski_harabasz(Z, labels),
        "orthogonality": orthogonality(centers),
        "universality": universality(model_ids, labels),
    }


# ---- end-only: title/description embedding orthogonality -----------------
def title_embedding_orthogonality(titles: Dict[int, str], descs: Dict[int, str],
                                  *, embed_model: str, dim: int) -> float:
    """(1 - cos)^2 between embeddings of the final category labels (no judge)."""
    from oai import embed
    cids = sorted(titles)
    texts = [f"{titles[c]}: {descs[c]}" for c in cids]
    E = embed(texts, model=embed_model, dim=dim)
    return orthogonality(E)


# ---- coverage ------------------------------------------------------------
def coverage_matrix(model_ids: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[int, float]]:
    """Per model: fraction of its sentences in each cluster (excludes noise)."""
    out: Dict[str, Dict[int, float]] = {}
    clusters = sorted(c for c in set(labels.tolist()) if c != -1)
    for m in sorted(set(model_ids.tolist())):
        mask = (model_ids == m) & (labels != -1)
        tot = int(mask.sum())
        lab = labels[mask]
        out[m] = {c: (float((lab == c).sum()) / tot if tot else 0.0) for c in clusters}
    return out


# ---- LLM categorization accuracy ----------------------------------------
def _catalog(titles: Dict[int, str], descs: Dict[int, str]) -> str:
    lines = [f"{cid}. {titles[cid]}: {descs[cid]}" for cid in sorted(titles)]
    return "\n".join(lines)


def _parse_choice(text: str, valid: Sequence[int]) -> int:
    m = re.search(r"-?\d+", text)
    if not m:
        return -999
    v = int(m.group())
    return v if v in valid else -999


def llm_accuracy(
    sentences_by_cluster: Dict[int, List[str]],
    titles: Dict[int, str],
    descs: Dict[int, str],
    *,
    model: str = "gpt-4o-mini",
    n_per_cluster: int = 10,
    repeats: int = 3,
    seed: int = 42,
) -> Dict[str, object]:
    """Accuracy of relabelling sampled sentences from titles/descriptions alone."""
    clusters = sorted(titles)
    catalog = _catalog(titles, descs)
    chance = 1.0 / len(clusters)
    def _classify(task):
        true_cid, sent = task
        prompt = (
            "Categories:\n" + catalog +
            "\n\nAssign the sentence to exactly one category. "
            "Reply with only the category number.\n\nSentence: " + str(sent)
        )
        ans = chat(prompt, model, seed=seed)
        return _parse_choice(ans, clusters) == true_cid

    per_repeat: List[float] = []
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        tasks = []
        for cid in clusters:
            pool = sentences_by_cluster.get(cid, [])
            if not pool:
                continue
            picks = rng.choice(np.asarray(pool, dtype=object),
                               size=min(n_per_cluster, len(pool)), replace=False)
            tasks.extend((cid, s) for s in picks)
        hits = map_threaded(_classify, tasks, workers=8)
        acc = float(np.mean(hits)) if hits else 0.0
        per_repeat.append(acc)
        logger.info("  llm-acc repeat %d: %.3f (chance %.3f)", rep, acc, chance)
    arr = np.array(per_repeat)
    return {
        "accuracy_mean": float(arr.mean()),
        "accuracy_std": float(arr.std()),
        "chance": chance,
        "lift_over_chance": float(arr.mean() - chance),
        "per_repeat": per_repeat,
    }
