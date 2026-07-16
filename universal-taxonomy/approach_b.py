"""Approach B: learn one (layer, n_cluster) SAE per model via joint search.

An Optuna NSGA-II study searches the 9-dimensional categorical space (one SAE
choice per model, 36 options each) as a multi-objective Pareto problem over the
three geometric metrics (CH, (1-cos)^2 orthogonality, universality) of the pooled
universal clustering. To make each trial cheap we cache, for a fixed per-model
search subsample, every layer's per-SAE-normalized latent blocks + geometric E in
memory; a trial is then a slice + matmul + one k-means fit.

The winning selection is re-projected on the full data by :mod:`run` (which calls
``sample_pool_selected``).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler, normalize

import clustering
import evaluate
import projection
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

Choice = Tuple[int, int]  # (layer, k)


@dataclass
class Cache:
    keys: Dict[str, np.ndarray]                # model -> (sub,) key strings
    blocks: Dict[Tuple[str, int], np.ndarray]  # (model, layer) -> (sub, 90) normalized values
    E: Dict[Tuple[str, int, int], np.ndarray]  # (model, layer, k) -> (k, D) geometric latent basis
    layers: Dict[str, List[int]]


def build_cache(cfg: Config, variant: str, models: List[str], *, sub: int, seed: int,
                subset: str = "train") -> Cache:
    import splits
    rng = np.random.default_rng(seed)
    keys: Dict[str, np.ndarray] = {}
    blocks: Dict[Tuple[str, int], np.ndarray] = {}
    E: Dict[Tuple[str, int, int], np.ndarray] = {}
    layers_by_model: Dict[str, List[int]] = {}

    for m in models:
        layers = cfg.discover_layers(variant, m)
        layers_by_model[m] = layers
        common = projection.common_keys(cfg, variant, m, layers)
        # restrict to the requested split (search uses train only)
        pmc = np.array([k.split("\t")[0] for k in common.tolist()], dtype=object)
        snt = np.array([int(k.split("\t")[1]) for k in common.tolist()])
        common = common[splits.subset_mask(pmc, snt, subset, seed=seed)]
        take = min(sub, len(common))
        subkeys = common[np.sort(rng.choice(len(common), size=take, replace=False))]
        keys[m] = subkeys

        d = np.load(cfg.exemplar_emb_npz(variant, m))
        elut = {(int(a), int(b), int(c)): r
                for r, (a, b, c) in enumerate(zip(d["layer"], d["k"], d["idx"]))}

        for layer in layers:
            V, kk_keys, _ = projection.build_values(
                cfg, variant, m, normalize=True, layers=[layer], ks=cfg.n_clusters)
            pos = {k: i for i, k in enumerate(kk_keys.tolist())}
            ridx = np.array([pos[k] for k in subkeys.tolist()], dtype=np.int64)
            blocks[(m, layer)] = V[ridx]
            for k in cfg.n_clusters:
                E[(m, layer, k)] = d["emb"][[elut[(layer, k, i)] for i in range(k)]]
        logger.info("cache built for %s (%d layers, sub=%d)", m, len(layers), take)

    return Cache(keys=keys, blocks=blocks, E=E, layers=layers_by_model)


def _k_cols(cfg: Config, k: int) -> Tuple[int, int]:
    start = 0
    for kk in cfg.n_clusters:
        if kk == k:
            return start, start + kk
        start += kk
    raise KeyError(k)


def project_selection(cfg: Config, cache: Cache, selection: Dict[str, Choice]):
    Xs, mods, keys = [], [], []
    for m, (layer, k) in selection.items():
        a, b = _k_cols(cfg, k)
        block = cache.blocks[(m, layer)][:, a:b]          # (sub, k) normalized
        vecs = block @ cache.E[(m, layer, k)]              # (sub, D)
        Xs.append(vecs.astype(np.float32))
        mods.append(np.full(len(vecs), m, dtype=object))
        keys.append(cache.keys[m])
    X = np.concatenate(Xs)
    model = np.concatenate(mods)
    return X, model


# the three purely-geometric objectives, all maximized
OBJECTIVES = ("ch", "orthogonality", "universality")


def score_selection(cfg: Config, cache: Cache, selection: Dict[str, Choice],
                    *, method: str, k: int, seed: int) -> Tuple[float, float, float]:
    X, model = project_selection(cfg, cache, selection)
    Z = normalize(StandardScaler().fit_transform(X).astype(np.float32))
    res = clustering.cluster(Z, method, k=k, cosine=False, standardize=False, seed=seed)
    m = evaluate.geometric_metrics(Z, res.labels, res.centers, model)
    return tuple(m[o] for o in OBJECTIVES)


def _pareto_pick(trials) -> int:
    """Index of the Pareto-optimal trial whose min-max-normalized objectives sum highest."""
    vals = np.array([t.values for t in trials], dtype=float)
    lo, hi = vals.min(0), vals.max(0)
    norm = (vals - lo) / np.where(hi > lo, hi - lo, 1.0)
    return int(norm.sum(1).argmax())


def search(cfg: Config, variant: str, *, n_trials: int = 120, method: str = "minibatch_kmeans",
           k: int = 12, sub: int = 4000, seed: int = 42):
    import optuna

    models = cfg.models
    cache = build_cache(cfg, variant, models, sub=sub, seed=seed, subset="train")
    combos: Dict[str, List[Choice]] = {
        m: [(l, kk) for l in cache.layers[m] for kk in cfg.n_clusters] for m in models
    }

    def objective(trial):
        selection = {
            m: combos[m][trial.suggest_categorical(m, list(range(len(combos[m]))))]
            for m in models
        }
        return score_selection(cfg, cache, selection, method=method, k=k, seed=seed)

    study = optuna.create_study(directions=["maximize"] * len(OBJECTIVES),
                                sampler=optuna.samplers.NSGAIISampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    front = study.best_trials
    chosen = front[_pareto_pick(front)]
    best = {m: combos[m][chosen.params[m]] for m in models}
    logger.info("Pareto front size=%d; picked %s=%s", len(front), OBJECTIVES,
                tuple(round(v, 3) for v in chosen.values))
    for m, c in best.items():
        logger.info("  %s -> layer%d k%d", m, c[0], c[1])
    return best, study


def sample_pool_selected(cfg: Config, variant: str, selection: Dict[str, Choice],
                         *, n_total: int, subset: str = "all", seed: int = 42):
    """Pooled projection of the chosen per-model SAEs, within a split (for run.py)."""
    import pooling
    import splits
    rng = np.random.default_rng(seed)
    Xs, mods, pms, sts = [], [], [], []
    for m, (layer, k) in selection.items():
        vecs, keys = projection.project(cfg, variant, m, normalize=True, layers=[layer], ks=[k])
        pmc = np.array([kk.split("\t")[0] for kk in keys.tolist()], dtype=object)
        snt = np.array([int(kk.split("\t")[1]) for kk in keys.tolist()], dtype=np.int32)
        allowed = np.where(splits.subset_mask(pmc, snt, subset, seed=seed))[0]
        take = min(len(allowed), max(1, round(n_total / len(selection))))
        sel = np.sort(rng.choice(allowed, size=take, replace=False))
        Xs.append(vecs[sel].astype(np.float32))
        mods.append(np.full(take, m, dtype=object))
        pms.append(pmc[sel])
        sts.append(snt[sel])
    return pooling.Pool(X=np.concatenate(Xs), model=np.concatenate(mods),
                        pmcid=np.concatenate(pms), sent=np.concatenate(sts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["nndedup", "dedup", "plain"], required=True)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--sub", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config()
    best, _study = search(cfg, args.variant, n_trials=args.n_trials, k=args.k,
                          sub=args.sub, seed=args.seed)
    out = cfg.out_root / args.variant / "runs" / "approach_b_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"variant": args.variant,
                   "selection": {m: list(c) for m, c in best.items()}}, f, indent=2)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
