"""Clustering for the pooled cross-model sentence vectors.

Two methods behind one interface: k-means (with a Calinski-Harabasz k-sweep) and
HDBSCAN. Vectors are standardized + L2-normalized (cosine geometry) upstream.
HDBSCAN may emit a noise label ``-1``; centers are the mean of each non-noise
cluster's members.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    labels: np.ndarray                 # (N,), -1 = noise (HDBSCAN only)
    centers: np.ndarray                # (n_clusters, D) in the clustering space
    n_clusters: int
    method: str
    params: dict = field(default_factory=dict)


def _preprocess(X: np.ndarray, *, cosine: bool, standardize: bool) -> np.ndarray:
    Z = X.astype(np.float32)
    if standardize:
        Z = StandardScaler().fit_transform(Z).astype(np.float32)
    if cosine:
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        n[n == 0] = 1.0
        Z = Z / n
    return Z


def compute_centers(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    ids = sorted(l for l in set(labels.tolist()) if l != -1)
    return np.stack([X[labels == i].mean(axis=0) for i in ids]) if ids else np.zeros((0, X.shape[1]))


def assign(Z: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each row of Z to the nearest center by cosine similarity.

    ``centers`` are indexed by sorted cluster id (as returned by
    :func:`compute_centers`), so the returned labels use those same ids.
    """
    if len(centers) == 0:
        return np.full(len(Z), -1, dtype=int)
    cn = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return (zn @ cn.T).argmax(axis=1).astype(int)


def _ch(X: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabasz index (higher = better; fast, low-variance)."""
    mask = labels != -1
    lab = labels[mask]
    if len(set(lab.tolist())) < 2:
        return 0.0
    return float(calinski_harabasz_score(X[mask], lab))


def fit_kmeans(X, k, seed=42, minibatch=False):
    km = (MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096)
          if minibatch else KMeans(n_clusters=k, random_state=seed, n_init=10))
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_


def fit_hdbscan(X, min_cluster_size=500, min_samples=None, pca_dim=50):
    """HDBSCAN on a randomized-PCA reduction (brute-force HDBSCAN is infeasible in
    3072-dim); cluster labels come from the reduced space, centers from original X."""
    import hdbscan
    from sklearn.decomposition import PCA
    Xr = (PCA(n_components=pca_dim, svd_solver="randomized", random_state=0).fit_transform(X)
          if X.shape[1] > pca_dim else X)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(Xr)
    return labels, compute_centers(X, labels)


def sweep_k(X, ks, seed=42, minibatch=True):
    """Pick k maximizing Calinski-Harabasz; returns (best_k, labels, centers)."""
    best = None
    for k in ks:
        labels, centers = fit_kmeans(X, k, seed=seed, minibatch=minibatch)
        s = _ch(X, labels)
        logger.info("  k=%d calinski_harabasz=%.1f", k, s)
        if best is None or s > best[0]:
            best = (s, k, labels, centers)
    _, k, labels, centers = best
    return k, labels, centers


def cluster(
    X: np.ndarray,
    method: str,
    *,
    k: Optional[int] = None,
    k_sweep: Optional[List[int]] = None,
    cosine: bool = True,
    standardize: bool = False,
    seed: int = 42,
    **params,
) -> ClusterResult:
    Z = _preprocess(X, cosine=cosine, standardize=standardize)

    if method in ("kmeans", "minibatch_kmeans"):
        mb = method == "minibatch_kmeans"
        if k is None and k_sweep:
            k, labels, centers = sweep_k(Z, k_sweep, seed=seed, minibatch=mb)
        else:
            labels, centers = fit_kmeans(Z, k, seed=seed, minibatch=mb)
    elif method == "hdbscan":
        labels, centers = fit_hdbscan(Z, **params)
    else:
        raise ValueError(f"unknown method {method!r}")

    n = len([l for l in set(labels.tolist()) if l != -1])
    return ClusterResult(labels=labels, centers=centers, n_clusters=n,
                         method=method, params={"k": k, **params})
