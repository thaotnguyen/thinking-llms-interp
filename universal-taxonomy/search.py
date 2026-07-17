"""Optuna search for the universal taxonomy (commit item 5).

Per model choose (layer, sae_k); plus one global final_k. Base clusterer = exact fair
clustering of the LIVE latents (every model gets >= 1 latent per category). NSGA-II Pareto
over two maximised objectives:
  - **harmonic mean** of per-model TRACE coverage (fraction of a model's traces whose
    sentences cover all K categories) -- collapses toward 0 if any model is poorly
    covered, so the worst model is what gets optimised;
  - **CH** (Calinski-Harabasz) cluster quality.
Winner = min-max-normalised (HM + CH) from the Pareto front. Writes the winner and the
per-sentence category assignment to ``results/plain/``.

Run: ``N_TRIALS=1500 python search.py``
"""
from __future__ import annotations

import json
import os

import numpy as np
import optuna
from sklearn.metrics import calinski_harabasz_score

from cache_latents import MODELS, VARIANT, cache_npz
from config import Config
from fair_cluster import fair_cluster

optuna.logging.set_verbosity(optuna.logging.WARNING)
SAE_KS = [10, 12, 14, 16, 18, 20]
FINAL_KS = [5, 6, 7, 8, 9, 10]
N_TRIALS = int(os.environ.get("N_TRIALS", 1500))
SEED = 42


def hmean(vals) -> float:
    v = np.asarray(list(vals), float)
    return float(len(v) / np.sum(1.0 / (v + 1e-6)))


def load(cfg: Config) -> dict:
    """Per model: live-latent embeddings per SAE block, routing argmax, trace ids."""
    data = {}
    for m in MODELS:
        z = np.load(cache_npz(cfg, m), allow_pickle=True)
        layers = z["layers"].tolist(); blocks = {}
        for l in layers:
            for k in SAE_KS:
                am = z[f"am__{l}__{k}"].astype(np.int64); live = np.unique(am)
                blocks[(l, k)] = (z[f"E__{l}__{k}"][live].astype(np.float32), live, am)
        _, tid = np.unique(z["pmcid"], return_inverse=True)
        data[m] = {"layers": layers, "blocks": blocks, "tid": tid.astype(np.int64),
                   "ntr": int(tid.max() + 1)}
    return data


def cluster_and_cover(data: dict, sel: dict, k: int):
    """Fair-cluster pooled live latents; return (per_model_trace_cov, labels, E, offsets)."""
    parts, groups, offs, off = [], [], {}, 0
    for m in MODELS:
        E, live, am = data[m]["blocks"][sel[m]]
        parts.append(E); groups += [m] * E.shape[0]; offs[m] = off; off += E.shape[0]
    E = np.vstack(parts).astype(np.float64)
    E = E - E.mean(0, keepdims=True); E = (E / (np.linalg.norm(E) + 1e-8)).astype(np.float32)
    labels, feasible = fair_cluster(E, np.array(groups), k, MODELS, iters=6)
    if not feasible:
        return None
    per_model = {}
    for m in MODELS:
        _, live, am = data[m]["blocks"][sel[m]]
        sent_c = labels[offs[m] + np.searchsorted(live, am)]
        cov = np.zeros((data[m]["ntr"], k), bool); cov[data[m]["tid"], sent_c] = True
        per_model[m] = int((cov.sum(1) == k).sum()) / data[m]["ntr"]
    return per_model, labels, E, offs


def search(cfg: Config) -> dict:
    data = load(cfg)
    live = {m: {s: data[m]["blocks"][s][0].shape[0] for s in data[m]["blocks"]} for m in MODELS}

    def objective(trial):
        sel = {m: (trial.suggest_categorical(f"{m}_layer", data[m]["layers"]),
                   trial.suggest_categorical(f"{m}_saek", SAE_KS)) for m in MODELS}
        k = trial.suggest_categorical("final_k", FINAL_KS)
        if min(live[m][sel[m]] for m in MODELS) < k:
            return 0.0, 0.0
        out = cluster_and_cover(data, sel, k)
        if out is None:
            return 0.0, 0.0
        per_model, labels, E, _ = out
        ch = float(calinski_harabasz_score(E, labels)) if len(set(labels.tolist())) > 1 else 0.0
        return hmean(per_model.values()), ch

    study = optuna.create_study(directions=["maximize", "maximize"],
                                sampler=optuna.samplers.NSGAIISampler(population_size=50, seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)

    front = study.best_trials                             # pick by min-max normalised (HM + CH)
    hm = np.array([t.values[0] for t in front]); ch = np.array([t.values[1] for t in front])
    hn = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9); cn = (ch - ch.min()) / (ch.max() - ch.min() + 1e-9)
    best = front[int(np.argmax(hn + cn))]
    sel = {m: (best.params[f"{m}_layer"], best.params[f"{m}_saek"]) for m in MODELS}
    k = best.params["final_k"]
    per_model, labels, E, offs = cluster_and_cover(data, sel, k)

    # save winner + per-sentence category assignment (aligned to each model's cache order)
    out_dir = cfg.out_root / VARIANT
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        _, live_lat, am = data[m]["blocks"][sel[m]]
        np.save(out_dir / "labels" / f"{m}.npy",
                labels[offs[m] + np.searchsorted(live_lat, am)].astype(np.int16))
    winner = {"final_k": k, "hm_coverage": round(hmean(per_model.values()), 4),
              "CH": round(float(calinski_harabasz_score(E, labels)), 1),
              "per_model_trace_cov": {m: round(per_model[m], 3) for m in MODELS},
              "sel": {m: [int(sel[m][0]), int(sel[m][1])] for m in MODELS}}
    json.dump(winner, open(out_dir / "winner.json", "w"), indent=2)
    print(f"WINNER final_k={k} HM-cov={winner['hm_coverage']*100:.1f}% CH={winner['CH']} "
          f"front={len(front)}", flush=True)
    return winner


if __name__ == "__main__":
    search(Config())
