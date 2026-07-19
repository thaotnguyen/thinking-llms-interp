#!/usr/bin/env python3
"""Feature generation for the correct-vs-incorrect trace classifier.

Each trace (one model answering one case) is represented by its universal-taxonomy
sentence sequence -- ordered categories 0..4 produced by `universal-taxonomy/run.sh`.
This module turns every trace into 81 length-normalized features and attaches its
label (rec-decoding re-grade), question source, and model.

Inputs (must exist before running):
  - taxonomy:  universal-taxonomy/results_recommended/plain/{labels/{model}.npy,
               latent_cache/{model}.npz}      (run.sh output; 6 recommended-decoding models)
  - labels:    trace-classifier/data/graded/responses_{model}.rec_graded.json
               (rec responses graded by the repo's grade_responses.py; is_correct)
"""
from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[1]
GRADED = Path(__file__).resolve().parent / "data" / "graded"

MODELS = [
    "deepseek-r1-distill-qwen-1.5b", "deepseek-r1-distill-llama-8b",
    "deepseek-r1-distill-qwen-14b", "huatuogpt-o1-8b", "gpt-oss-20b", "qwq-32b",
]
CAT = ["PresentingData", "GeneratingHypoth", "FormulatingDx", "ExplainingMech", "StructuringReason"]
SPLIT_TO_SOURCE = {"nejm": "nejm", "medqa": "medqa", "train": "medmcqa"}
CANON_RANK = {0: 0.0, 1: 1.0, 3: 2.0, 2: 3.0}          # data -> hypothesis -> mechanism -> diagnosis
GOOD_TRANS = [(0, 1), (1, 3), (3, 2), (1, 2), (0, 3)]
BAD_TRANS = [(2, 1), (2, 0), (2, 3)]
TRIGRAMS = [((0, 1, 2), "data_hyp_dx"), ((1, 3, 2), "hyp_mech_dx"), ((2, 1, 2), "dx_hyp_dx"),
            ((2, 0, 2), "dx_data_dx"), ((4, 4, 4), "struct_run"), ((1, 2, 1), "hyp_dx_hyp")]


def _tax_dir() -> Path:
    """Resolve the taxonomy output dir holding all 6 recommended-decoding models."""
    for d in [WT / "universal-taxonomy" / "results_recommended" / "plain",
              WT / "universal-taxonomy" / "results" / "plain"]:
        if all((d / "labels" / f"{m}.npy").exists() for m in MODELS):
            return d
    raise FileNotFoundError("No taxonomy dir contains labels for all 6 models; run universal-taxonomy/run.sh")


def _entropy(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def extract(seq: np.ndarray) -> dict:
    """81 length-normalized features from one 0..4 category sequence."""
    seq = np.asarray(seq, np.int64)
    n = len(seq)
    counts = np.bincount(seq, minlength=5).astype(float)
    f = {f"prop_{CAT[i]}": counts[i] / n for i in range(5)}

    trans = np.zeros((5, 5))
    if n >= 2:
        np.add.at(trans, (seq[:-1], seq[1:]), 1.0)
        trans /= (n - 1)
    for i in range(5):
        for j in range(5):
            f[f"tr_{i}{j}"] = trans[i, j]

    uniq = int((counts > 0).sum())
    f["num_unique"] = float(uniq)
    f["state_diversity"] = uniq / 5
    f["entropy"] = _entropy(counts)
    f["switch_rate"] = float((seq[1:] != seq[:-1]).sum()) / max(n - 1, 1) if n >= 2 else 0.0
    runs = np.diff(np.flatnonzero(np.r_[True, seq[1:] != seq[:-1], True]))
    f["max_run_frac"] = (runs.max() / n) if len(runs) else 1.0 / n

    for i in range(5):
        pos = np.flatnonzero(seq == i)
        f[f"present_{CAT[i]}"] = 1.0 if len(pos) else 0.0
        f[f"first_{CAT[i]}"] = (pos[0] / n) if len(pos) else 1.0
        f[f"last_{CAT[i]}"] = (pos[-1] / n) if len(pos) else 0.0
    ranks = np.array([CANON_RANK.get(int(c), np.nan) for c in seq])
    keep = ~np.isnan(ranks)
    if keep.sum() >= 3 and np.unique(ranks[keep]).size >= 2:
        pc = np.corrcoef(np.arange(n)[keep], ranks[keep])[0, 1]
        f["arc_coherence"] = float(pc) if np.isfinite(pc) else 0.0
    else:
        f["arc_coherence"] = 0.0
    dx = np.flatnonzero(seq == 2)
    f["first_dx_pos"] = (dx[0] / n) if len(dx) else 1.0
    f["frac_after_first_dx"] = (1.0 - dx[0] / n) if len(dx) else 0.0

    if n >= 2:
        f["repetition_rate"] = sum(1 for c in counts if c > 1) / max(uniq, 1)
        f["cycle_rate"] = sum(1 for i in range(n - 2) if seq[i] == seq[i + 2] and seq[i] != seq[i + 1]) / (n - 2) if n > 2 else 0.0
        seen, bt = set(), 0
        for c in seq:
            bt += c in seen
            seen.add(c)
        f["backtracking"] = bt / n
    else:
        f["repetition_rate"] = f["cycle_rate"] = f["backtracking"] = 0.0

    good = sum(trans[a, b] for a, b in GOOD_TRANS)
    bad = sum(trans[a, b] for a, b in BAD_TRANS)
    f["good_flow_rate"], f["bad_flow_rate"], f["flow_balance"] = float(good), float(bad), float(good - bad)

    thirds = np.array_split(seq, 3) if n >= 3 else [seq, seq[:0], seq[:0]]
    for ti, seg in enumerate(thirds[:3]):
        sc = np.bincount(seg, minlength=5).astype(float)
        for i in range(5):
            f[f"seg{ti}_{CAT[i]}"] = sc[i] / max(len(seg), 1)
    if n >= 4:
        f["entropy_drop"] = _entropy(np.bincount(seq[:n // 2], minlength=5).astype(float)) \
            - _entropy(np.bincount(seq[n // 2:], minlength=5).astype(float))
    else:
        f["entropy_drop"] = 0.0
    for (a, b, c), nm in TRIGRAMS:
        f[f"tri_{nm}"] = sum(1 for t in range(n - 2) if seq[t] == a and seq[t + 1] == b and seq[t + 2] == c) / max(n - 2, 1)
    return f


FEATURE_NAMES = list(extract(np.array([0, 1, 3, 2, 4, 2, 1, 0])).keys())

Data = namedtuple("Data", "X y pmcid source model feature_names")


def _sequences(tax: Path, model: str) -> dict:
    labels = np.load(tax / "labels" / f"{model}.npy").astype(np.int64)
    cache = np.load(tax / "latent_cache" / f"{model}.npz", allow_pickle=True)
    pmcid = np.asarray(cache["pmcid"]).astype(str)
    sidx = np.asarray(cache["sent_idx"]).astype(np.int64)
    order = np.lexsort((sidx, pmcid))
    pmcid, labels = pmcid[order], labels[order]
    bnd = np.flatnonzero(pmcid[1:] != pmcid[:-1]) + 1
    return {p[0]: s for p, s in zip(np.split(pmcid, bnd), np.split(labels, bnd))}


def _grades(model: str) -> dict:
    out = {}
    for r in json.load(open(GRADED / f"responses_{model}.rec_graded.json")):
        if r:
            out[str(r["pmcid"])] = (str(r.get("is_correct")).lower() == "true", SPLIT_TO_SOURCE.get(r.get("dataset_split"), "other"))
    return out


def load() -> Data:
    """Build the feature matrix and labels for all traces with both a sequence and a grade."""
    tax = _tax_dir()
    rows, y, pmc, src, mdl = [], [], [], [], []
    for m in MODELS:
        seqs, grades = _sequences(tax, m), _grades(m)
        for p, seq in seqs.items():
            if p not in grades or len(seq) == 0:
                continue
            correct, source = grades[p]
            d = extract(seq)
            rows.append([d[k] for k in FEATURE_NAMES])
            y.append(int(correct)); pmc.append(p); src.append(source); mdl.append(m)
    return Data(np.array(rows, float), np.array(y), np.array(pmc), np.array(src), np.array(mdl), FEATURE_NAMES)


if __name__ == "__main__":
    d = load()
    print(f"{d.X.shape[0]} traces x {d.X.shape[1]} features | correct rate {d.y.mean():.3f}")
    for s in np.unique(d.source):
        print(f"  source {s:9s} n={int((d.source==s).sum())}")
