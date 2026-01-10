#!/usr/bin/env python3
"""
Plot a 2D UMAP (or PCA fallback) of SAE activations for MedPRM steps, coloring
positive (+) and negative (-) steps differently.

Inputs can be:
- --activations: .npy (shape [N, D]) or .csv with numeric feature columns.
- --labels (optional): .csv or .jsonl containing a column with labels (+/-/1/0/true/false or score).
  If omitted and --activations is a CSV, the script will try to auto-detect a label column.

Basic usage examples:
  python scripts/plot_umap_medprm_steps.py \
    --activations path/to/activations.npy \
    --labels path/to/labels.csv \
    --label-column label \
    --out analysis_outputs/umap_medprm_steps.png

  python scripts/plot_umap_medprm_steps.py \
    --activations path/to/activations.csv \
    --label-column prm_label \
    --out analysis_outputs/umap_medprm_steps.png

If umap-learn is not available, the script falls back to PCA.
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import umap  # type: ignore
    HAVE_UMAP = True
except Exception:
    from sklearn.decomposition import PCA  # type: ignore
    HAVE_UMAP = False


LIKELY_LABEL_COLUMNS = [
    "label", "step_label", "prm_label", "posneg", "sign",
    "is_positive", "positive", "polarity", "score", "logit"
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--activations", required=True, help="Path to activations .npy or .csv")
    p.add_argument("--labels", help="Optional path to labels .csv or .jsonl (if activations is .npy)")
    p.add_argument("--label-column", default=None, help="Column in labels/data to use for +/- labels")
    p.add_argument("--index-column", default=None, help="Optional key present in both activations CSV and labels file to merge on")
    p.add_argument("--out", default="umap_medprm_steps.png", help="Output image path (.png)")
    p.add_argument("--sample", type=int, default=0, help="Optional subsample size for speed (0 means use all)")
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--standardize", action="store_true", help="Z-score standardize features before projection")
    return p.parse_args()


def _auto_detect_label_column(df: pd.DataFrame) -> Optional[str]:
    for c in LIKELY_LABEL_COLUMNS:
        if c in df.columns:
            return c
    # Also try case-insensitive match
    lower = {c.lower(): c for c in df.columns}
    for c in LIKELY_LABEL_COLUMNS:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _coerce_label_to_sign(x) -> Optional[int]:
    """Map a variety of label encodings to +1 (positive) or -1 (negative).
    Returns None if cannot be determined.
    """
    if x is None:
        return None
    # Strings like '+', '-', 'pos', 'neg', 'positive', 'negative'
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"+", "+1", "pos", "positive", "true", "t", "yes", "y"}:
            return 1
        if s in {"-", "-1", "neg", "negative", "false", "f", "no", "n"}:
            return -1
        # Try floaty strings
        try:
            v = float(s)
            if v > 0:
                return 1
            if v < 0:
                return -1
            # exactly 0 is ambiguous -> None
            return None
        except Exception:
            return None
    # Booleans
    if isinstance(x, (bool, np.bool_)):
        return 1 if bool(x) else -1
    # Numbers
    try:
        v = float(x)
        if v > 0:
            return 1
        if v < 0:
            return -1
        return None
    except Exception:
        return None


def _load_labels(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in {".jsonl", ".json"}:
        # Stream read jsonl; for a plain json array fall back to pandas
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":
                data = json.load(f)
                rows = data if isinstance(data, list) else []
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported labels file type: {ext}")


def _ensure_2d_array(x) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _standardize(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std


def main() -> int:
    args = parse_args()

    # Load activations
    act_ext = os.path.splitext(args.activations)[1].lower()
    labels_df: Optional[pd.DataFrame] = None
    label_col: Optional[str] = args.label_column
    idx_col: Optional[str] = args.index_column

    if act_ext == ".npy":
        X = np.load(args.activations)
        X = _ensure_2d_array(X)
        if args.labels is None:
            print("--labels is required when --activations is .npy", file=sys.stderr)
            return 2
        labels_df = _load_labels(args.labels)
        if label_col is None:
            label_col = _auto_detect_label_column(labels_df) or "label"
        if idx_col is not None and idx_col in labels_df.columns:
            # Ensure sorted by idx_col just in case
            labels_df = labels_df.sort_values(by=idx_col).reset_index(drop=True)
        labels_raw = labels_df[label_col].tolist()
    elif act_ext == ".csv":
        act_df = pd.read_csv(args.activations)
        # Auto-detect label column if present in the same file
        if label_col is None:
            label_col = _auto_detect_label_column(act_df)

        if args.labels:
            labels_df = _load_labels(args.labels)
            if label_col is None:
                label_col = _auto_detect_label_column(labels_df) or "label"
            if idx_col and (idx_col in act_df.columns) and (idx_col in labels_df.columns):
                merged = pd.merge(act_df, labels_df[[idx_col, label_col]], on=idx_col, how="inner")
                act_df = merged
            else:
                # Assume aligned order
                act_df[label_col] = labels_df[label_col].values

        # Extract labels if we have a column for it
        if label_col and label_col in act_df.columns:
            labels_raw = act_df[label_col].tolist()
            feature_df = act_df.drop(columns=[c for c in act_df.columns if c == label_col or act_df[c].dtype == "O"])  # drop label and non-numeric
        else:
            labels_raw = [None] * len(act_df)
            feature_df = act_df.select_dtypes(include=[np.number])

        X = feature_df.to_numpy()
    else:
        print(f"Unsupported activations file type: {act_ext}", file=sys.stderr)
        return 2

    # Subsample if requested
    N = X.shape[0]
    if args.sample and args.sample > 0 and args.sample < N:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=args.sample, replace=False)
        X = X[idx]
        labels_raw = [labels_raw[i] for i in idx]

    # Standardize if requested
    if args.standardize:
        X = _standardize(X)

    # Coerce labels to +/-
    signs = np.array([_coerce_label_to_sign(v) for v in labels_raw])
    # If all None, warn and mark all as neutral (plot as gray)
    if np.all(pd.isna(signs)):
        print("Warning: Could not determine any +/- labels; plotting all in gray.", file=sys.stderr)
        signs = np.zeros(len(labels_raw), dtype=int)
    else:
        # Replace None/np.nan with 0
        signs = np.where(pd.isna(signs), 0, signs).astype(int)

    # Project to 2D
    if HAVE_UMAP:
        reducer = umap.UMAP(
            n_neighbors=args.__dict__["umap_n_neighbors"],
            min_dist=args.__dict__["umap_min_dist"],
            random_state=42,
            metric="cosine",
        )
        emb = reducer.fit_transform(X)
        method = "UMAP"
    else:
        pca = PCA(n_components=2, random_state=42)
        emb = pca.fit_transform(X)
        method = "PCA"

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = np.where(signs > 0, "#E4572E", np.where(signs < 0, "#4C78A8", "#A0A0A0"))
    labels_legend = []
    for val, name, color in [(1, "+ step", "#E4572E"), (-1, "- step", "#4C78A8"), (0, "unknown", "#A0A0A0")]:
        mask = signs == val
        if mask.any():
            ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.7, label=name, c=color, edgecolors="none")
            labels_legend.append(name)

    ax.set_title(f"{method} of SAE Activations (MedPRM steps)")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", alpha=0.3)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
