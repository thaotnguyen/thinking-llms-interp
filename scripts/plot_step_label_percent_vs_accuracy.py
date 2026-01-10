#!/usr/bin/env python3
"""
Plot a stacked bar chart where the two bars correspond to final answer correctness
(Correct vs Incorrect), and the y-axis shows the percentage split of step labels
that are positive (+) vs negative (-).

Input file can be CSV or JSONL with at least two columns:
- label column: encodes step label as + / - / 1 / 0 / true / false or score (threshold at 0)
- correctness column: encodes final answer correctness per step's example as
  True/False or 1/0 (the step should be associated with a final answer)

Examples:
  python scripts/plot_step_label_percent_vs_accuracy.py \
    --input path/to/steps.csv \
    --label-column prm_label \
    --correct-column final_correct \
    --out analysis_outputs/step_label_percent_vs_accuracy.png

  python scripts/plot_step_label_percent_vs_accuracy.py \
    --input path/to/steps.jsonl \
    --label-column label \
    --correct-column is_correct
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LIKELY_LABEL_COLUMNS = [
    "label", "step_label", "prm_label", "posneg", "sign",
    "is_positive", "positive", "polarity", "score", "logit"
]
LIKELY_CORRECT_COLUMNS = [
    "final_correct", "correct", "is_correct", "answer_correct",
    "was_correct", "gt_is_correct"
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="CSV or JSONL file with step labels and final correctness")
    p.add_argument("--label-column", default=None, help="Name of the column with +/- labels (auto-detect if omitted)")
    p.add_argument("--correct-column", default=None, help="Name of the column with final correctness (auto-detect if omitted)")
    p.add_argument("--out", default="step_label_percent_vs_accuracy.png", help="Path to output PNG file")
    p.add_argument("--filter-col", default=None, help="Optional boolean filter column to keep rows where this is True")
    return p.parse_args()


def _auto_detect(colnames, candidates):
    for c in candidates:
        if c in colnames:
            return c
    lower = {c.lower(): c for c in colnames}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _load_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in {".jsonl", ".json"}:
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
        raise ValueError(f"Unsupported file type: {ext}")


def _coerce_label_to_sign(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"+", "+1", "pos", "positive", "true", "t", "yes", "y"}:
            return 1
        if s in {"-", "-1", "neg", "negative", "false", "f", "no", "n"}:
            return -1
        try:
            v = float(s)
            if v > 0:
                return 1
            if v < 0:
                return -1
            return None
        except Exception:
            return None
    if isinstance(x, (bool, np.bool_)):
        return 1 if bool(x) else -1
    try:
        v = float(x)
        if v > 0:
            return 1
        if v < 0:
            return -1
        return None
    except Exception:
        return None


def _coerce_to_bool(x) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "1", "yes", "y", "+"}:
            return True
        if s in {"false", "f", "0", "no", "n", "-"}:
            return False
        try:
            v = float(s)
            return v != 0.0
        except Exception:
            return None
    try:
        v = float(x)
        return v != 0.0
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    df = _load_df(args.input)

    if args.filter_col and args.filter_col in df.columns:
        df = df[df[args.filter_col].astype(bool)]

    label_col = args.label_column or _auto_detect(df.columns, LIKELY_LABEL_COLUMNS)
    corr_col = args.correct_column or _auto_detect(df.columns, LIKELY_CORRECT_COLUMNS)

    if not label_col or label_col not in df.columns:
        raise SystemExit(f"Could not find label column. Tried: {LIKELY_LABEL_COLUMNS}. Available: {list(df.columns)}")
    if not corr_col or corr_col not in df.columns:
        raise SystemExit(f"Could not find correctness column. Tried: {LIKELY_CORRECT_COLUMNS}. Available: {list(df.columns)}")

    signs = df[label_col].map(_coerce_label_to_sign)
    # Drop rows with unknown sign
    mask_known = signs.notna()
    df = df.loc[mask_known].copy()
    signs = signs.loc[mask_known].astype(int)

    corr_vals = df[corr_col].map(_coerce_to_bool)
    mask_corr_known = corr_vals.notna()
    df = df.loc[mask_corr_known].copy()
    signs = signs.loc[mask_corr_known]
    corr_vals = corr_vals.loc[mask_corr_known].astype(bool)

    # Group and compute percentages
    data = []  # rows: (group_label, pct_pos, pct_neg, n)
    for group_name, group_mask in [("Correct", corr_vals == True), ("Incorrect", corr_vals == False)]:
        group_signs = signs[group_mask]
        n = len(group_signs)
        if n == 0:
            pct_pos = pct_neg = 0.0
        else:
            n_pos = int((group_signs > 0).sum())
            n_neg = int((group_signs < 0).sum())
            # Normalize over known (+ or -)
            denom = max(n_pos + n_neg, 1)
            pct_pos = 100.0 * n_pos / denom
            pct_neg = 100.0 * n_neg / denom
        data.append((group_name, pct_pos, pct_neg, n))

    # Plot stacked bars
    labels = [r[0] for r in data]
    pct_pos = [r[1] for r in data]
    pct_neg = [r[2] for r in data]
    counts = [r[3] for r in data]

    x = np.arange(len(labels))
    width = 0.6
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    bar_neg = ax.bar(x, pct_neg, width, label='- steps', color="#4C78A8")
    bar_pos = ax.bar(x, pct_pos, width, bottom=pct_neg, label='+ steps', color="#E4572E")

    ax.set_ylabel('Percentage of steps (%)')
    ax.set_title('Step label distribution by final answer correctness')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)

    # Annotate percentages and counts
    for i, (neg, pos, n) in enumerate(zip(pct_neg, pct_pos, counts)):
        ax.text(i, neg / 2, f"{neg:.1f}%", ha='center', va='center', color='white', fontsize=10)
        ax.text(i, neg + pos / 2, f"{pos:.1f}%", ha='center', va='center', color='white', fontsize=10)
        ax.text(i, 102, f"n={n}", ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
