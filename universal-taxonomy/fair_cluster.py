"""Bera-2019-style (alpha,beta)-fair clustering, specialised to small N so the fair
assignment is solved exactly (totally-unimodular LP -> integral optimum, no rounding).

Clusters the pooled latent embeddings (rows of E, one per (model, latent)) into K groups
such that every cluster contains >= 1 latent from every model. Lloyd loop:
  1. spherical k-means centres on row-normalised E,
  2. fair assignment: minimise sum cosine-distance(point, its centre) s.t. each
     (cluster, model) receives >= 1 point -- an LP whose constraint matrix is bipartite,
     so ``scipy.linprog(highs)`` returns an integral optimum,
  3. recompute centres; iterate.
Feasible iff min per-model latent count >= K. Returns (labels, feasible)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans


def _unit(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _fair_assign(cost: np.ndarray, groups: np.ndarray, k: int, models: Sequence[str]) -> Optional[np.ndarray]:
    """Assign each point to a cluster minimising total cost s.t. every (cluster, model)
    gets >= 1 point. cost: (n, k). Returns labels (n,) or None if infeasible."""
    n = cost.shape[0]
    n_vars = n * k                                       # x[p, c] -> index p*k + c
    obj = cost.reshape(-1).astype(np.float64)
    rows, cols = [], []
    for p in range(n):                                   # each point assigned exactly once
        for c in range(k):
            rows.append(p); cols.append(p * k + c)
    a_eq = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n_vars)); b_eq = np.ones(n)
    rows, cols, r = [], [], 0
    for c in range(k):                                   # each (cluster, model) gets >= 1
        for m in models:
            for p in np.where(groups == m)[0]:
                rows.append(r); cols.append(p * k + c)
            r += 1
    a_ub = coo_matrix((-np.ones(len(rows)), (rows, cols)), shape=(k * len(models), n_vars))
    b_ub = -np.ones(k * len(models))
    res = linprog(obj, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=(0, 1), method="highs")
    return res.x.reshape(n, k).argmax(1) if res.success else None


def fair_cluster(E: np.ndarray, groups: Sequence[str], k: int, models: Sequence[str],
                 iters: int = 6, seed: int = 0) -> Tuple[Optional[np.ndarray], bool]:
    """E: (n, d) latent embeddings. groups: (n,) model label per latent."""
    groups = np.asarray(groups)
    if min((groups == m).sum() for m in models) < k:     # feasibility ceiling
        return None, False
    Er = _unit(E.astype(np.float64))
    centres = _unit(KMeans(k, n_init=5, random_state=seed).fit(Er).cluster_centers_)
    labels = None
    for _ in range(iters):
        new = _fair_assign(1.0 - Er @ centres.T, groups, k, models)
        if new is None:
            return None, False
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        centres = _unit(np.stack([Er[labels == c].mean(0) if (labels == c).any() else centres[c]
                                  for c in range(k)]))
    return labels, True
