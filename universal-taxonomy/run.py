"""End-to-end driver for one experiment: fit on train, evaluate on held-out test.

Protocol (identical for Approach A and B, so the comparison is fair):
  1. Build a train pool and a disjoint test pool (hash-based split, :mod:`splits`).
  2. Standardize on train, L2-normalize (cosine geometry), cluster the train pool.
  3. Assign the test pool to the fitted centroids.
  4. Report all metrics on the TEST pool: geometric (CH, (1-cos)^2 orthogonality,
     universality) + end-only (title-embedding orthogonality, LLM accuracy).

Approach B receives the pre-searched per-model SAE ``selection``; the search ran
on the train split only, so the test pool here is genuinely held out.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import StandardScaler, normalize

import clustering
import evaluate
import naming
import pooling
import sentence_text
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def pool_texts(cfg: Config, variant: str, pool: pooling.Pool) -> np.ndarray:
    out: List[str] = []
    for m, p, s in zip(pool.model, pool.pmcid, pool.sent):
        d = sentence_text.load_sentences(cfg, variant, str(m))
        out.append(d.get((str(p), int(s)), ""))
    return np.array(out, dtype=object)


def build_pools(labels, centers, Z, texts, *, pool_size=60):
    """examples = nearest-centroid sentences per cluster; members = all texts."""
    cids = sorted(c for c in set(labels.tolist()) if c != -1)
    examples: Dict[int, List[str]] = {}
    members: Dict[int, List[str]] = {}
    for c in cids:
        idx = np.where(labels == c)[0]
        sims = Z[idx] @ centers[c]
        order = idx[np.argsort(-sims)]
        examples[c] = [texts[i] for i in order if texts[i]][:pool_size]
        members[c] = [texts[i] for i in idx if texts[i]]
    return examples, members


def run_experiment(
    cfg: Config,
    *,
    variant: str,
    approach: str,
    method: str,
    k: int | None,
    k_sweep: List[int] | None,
    normalized: bool,
    n_total: int,
    seed: int,
    do_name: bool,
    do_llm: bool,
    n_test: int | None = None,
    llm_n: int = 10,
    llm_repeats: int = 3,
    cluster_params: Dict | None = None,
    name_cap: int = 40,
    selection: Dict[str, tuple] | None = None,
) -> dict:
    n_test = n_test or n_total // 2
    if approach == "a":
        train = pooling.sample_pool(cfg, variant, normalized=normalized, n_total=n_total, subset="train", seed=seed)
        test = pooling.sample_pool(cfg, variant, normalized=normalized, n_total=n_test, subset="test", seed=seed)
    else:
        import approach_b
        train = approach_b.sample_pool_selected(cfg, variant, selection, n_total=n_total, subset="train", seed=seed)
        test = approach_b.sample_pool_selected(cfg, variant, selection, n_total=n_test, subset="test", seed=seed)

    scaler = StandardScaler().fit(train.X)
    Z_train = normalize(scaler.transform(train.X).astype(np.float32))
    Z_test = normalize(scaler.transform(test.X).astype(np.float32))

    res = clustering.cluster(Z_train, method, k=k, k_sweep=k_sweep, cosine=False,
                             standardize=False, seed=seed, **(cluster_params or {}))
    centers = res.centers
    labels_test = clustering.assign(Z_test, centers)
    logger.info("clustered: method=%s n_clusters=%d (eval on %d test pts)",
                method, res.n_clusters, len(Z_test))

    metrics = evaluate.geometric_metrics(Z_test, labels_test, centers, test.model)
    coverage = evaluate.coverage_matrix(test.model, labels_test)

    result = {
        "variant": variant, "approach": approach, "method": method,
        "n_clusters": res.n_clusters, "n_train": int(len(train.X)), "n_test": int(len(test.X)),
        "metrics": metrics, "coverage": coverage, "selection": selection,
    }

    do_name = do_name and (1 < res.n_clusters <= name_cap)
    if do_name:
        texts = pool_texts(cfg, variant, test)
        examples, members = build_pools(labels_test, centers, Z_test, texts)
        named = naming.name_clusters(examples, model=cfg.naming_model, seed=seed)
        titles = {c: named[c].title for c in named}
        descs = {c: named[c].description for c in named}
        result["titles"] = titles
        result["descriptions"] = descs
        # end-only semantic distinctness via title/description embeddings (no judge)
        result["title_embedding_orthogonality"] = evaluate.title_embedding_orthogonality(
            titles, descs, embed_model=cfg.embedding_model, dim=cfg.embedding_dim)
        if do_llm:
            result["llm_accuracy"] = evaluate.llm_accuracy(
                members, titles, descs, model=cfg.categorize_model,
                n_per_cluster=llm_n, repeats=llm_repeats, seed=seed)

    return result


def save(cfg: Config, result: dict) -> Path:
    tag = f"{result['approach']}_{result['method']}"
    if result.get("n_clusters"):
        tag += f"_k{result['n_clusters']}"
    out = cfg.out_root / result["variant"] / "runs" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("wrote %s", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["nndedup", "dedup", "plain"], required=True)
    ap.add_argument("--approach", choices=["a", "b"], default="a")
    ap.add_argument("--method", default="minibatch_kmeans")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--k-sweep", type=int, nargs="*", default=None)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--n-total", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-name", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--llm-n", type=int, default=10)
    ap.add_argument("--llm-repeats", type=int, default=3)
    args = ap.parse_args()

    cfg = Config()
    selection = None
    if args.approach == "b":
        sel_path = cfg.out_root / args.variant / "runs" / "approach_b_selection.json"
        selection = {m: tuple(c) for m, c in json.load(open(sel_path))["selection"].items()}

    result = run_experiment(
        cfg, variant=args.variant, approach=args.approach, method=args.method,
        k=args.k, k_sweep=args.k_sweep, normalized=not args.raw,
        n_total=args.n_total, seed=args.seed,
        do_name=not args.no_name, do_llm=not args.no_llm,
        llm_n=args.llm_n, llm_repeats=args.llm_repeats, selection=selection,
    )
    save(cfg, result)


if __name__ == "__main__":
    main()
