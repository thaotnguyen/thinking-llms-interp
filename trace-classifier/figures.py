#!/usr/bin/env python3
"""Figures from results/metrics.json (plotly):
  - accuracy.png/html : classifier accuracy per group under each protocol, with the
    always-guess-correct baseline marked (bar above the marker = positive lift).
  - importance.png/html : XGBoost feature importance.
Run after evaluate.py: `python figures.py`.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import features as F
from evaluate import xgb, _spw

RES = Path(__file__).resolve().parent / "results"
FIG = RES / "figures"
BAR, BASE = "#636EFA", "black"   # default plotly blue bars, black baseline diamonds

# --- human-readable feature labels: one template per feature category ---
CAT_NICE = {
    "PresentingData": "Presenting Data", "GeneratingHypoth": "Generating Hypotheses",
    "FormulatingDx": "Formulating Diagnosis", "ExplainingMech": "Explaining Mechanisms",
    "StructuringReason": "Structuring Reasoning",
}
NICE_BY_IDX = [CAT_NICE[c] for c in F.CAT]
SEG = {"0": "first", "1": "middle", "2": "last"}
SINGLE = {
    "num_unique": "how many of the 5 reasoning categories the trace uses",
    "state_diversity": "reasoning-category diversity (distinct categories ÷ 5)",
    "entropy": "how evenly reasoning is spread across the 5 categories (entropy)",
    "switch_rate": "how often the reasoning category changes sentence-to-sentence",
    "max_run_frac": "longest unbroken run of one category (as a share of the trace)",
    "arc_coherence": "how monotonically reasoning follows data → hypotheses → mechanism → diagnosis",
    "first_dx_pos": "how far into the trace the model first commits to a diagnosis",
    "frac_after_first_dx": "share of the trace after the model first commits to a diagnosis",
    "repetition_rate": "share of categories that the trace revisits more than once",
    "cycle_rate": "rate of A→B→A category cycles (leave a category and come back)",
    "backtracking": "share of sentences that return to an already-visited category (thrashing)",
    "good_flow_rate": "rate of forward-arc steps (data → hypotheses → mechanism → diagnosis)",
    "bad_flow_rate": "rate of steps that leave a stated diagnosis back to an earlier stage",
    "flow_balance": "forward-arc minus backward-arc transition rate",
    "entropy_drop": "how much the category mix narrows from the first half to the second half",
}
TRI = {
    "tri_data_hyp_dx": "3-step motif: presenting data → generating hypotheses → stating a diagnosis",
    "tri_hyp_mech_dx": "3-step motif: hypotheses → explaining mechanism → stating a diagnosis",
    "tri_dx_hyp_dx": "3-step motif: diagnosis → back to hypotheses → diagnosis (waffling)",
    "tri_dx_data_dx": "3-step motif: diagnosis → back to the data → diagnosis (re-checking)",
    "tri_struct_run": "3-step motif: three planning/structuring sentences in a row",
    "tri_hyp_dx_hyp": "3-step motif: hypotheses → diagnosis → back to hypotheses (indecision)",
}
# plain-language gerund gloss for each of the 5 reasoning categories
GLOSS = {
    "PresentingData": "restating the case data",
    "GeneratingHypoth": "generating candidate diagnoses",
    "FormulatingDx": "committing to a diagnosis",
    "ExplainingMech": "explaining the disease mechanism",
    "StructuringReason": "planning/structuring the reasoning",
}


def _wrap(s: str, maxlines: int = 2) -> str:
    """Balance a label onto ~maxlines lines using <br> so bars aren't super-wide."""
    width = max(28, -(-len(s) // maxlines))   # ceil(len/maxlines) → break near the middle
    return "<br>".join(textwrap.wrap(s, width=width))


def pretty(name: str) -> str:
    """Map a raw feature name to a self-explanatory, jargon-free label."""
    if name in SINGLE:
        return SINGLE[name]
    if name in TRI:
        return TRI[name]
    if name.startswith("prop_"):
        return f"share of sentences {GLOSS[name[5:]]}"
    if name.startswith("present_"):
        return f"whether the trace ever has a sentence {GLOSS[name[8:]]}"
    if name.startswith("first_"):
        return f"how early the FIRST sentence {GLOSS[name[6:]]} appears"
    if name.startswith("last_"):
        return f"how late the LAST sentence {GLOSS[name[5:]]} appears"
    if name.startswith("tr_"):
        return f"rate of going from {GLOSS[F.CAT[int(name[3])]]} to {GLOSS[F.CAT[int(name[4])]]}"
    if name.startswith("seg"):
        return f"share of the {SEG[name[3]]} third spent {GLOSS[name[5:]]}"
    return name


def _short(g: str) -> str:
    return g.replace("deepseek-r1-distill-", "")


def _panel(fig, col, rows: dict, showlegend: bool) -> None:
    groups = [_short(g) for g in rows]
    acc = [rows[g]["acc"] for g in rows]
    base = [rows[g]["baseline_acc"] for g in rows]
    lift = [rows[g]["acc_lift"] for g in rows]
    fig.add_trace(go.Bar(
        x=groups, y=acc, marker_color=BAR, name="classifier accuracy",
        showlegend=showlegend, legendgroup="acc",
        customdata=np.array(lift), hovertemplate="%{x}<br>acc=%{y:.3f}<br>lift=%{customdata:+.3f}<extra></extra>",
    ), row=1, col=col)
    fig.add_trace(go.Scatter(
        x=groups, y=base, mode="markers", name="always guess correct or always guess incorrect",
        marker=dict(symbol="diamond", size=10, color=BASE, line=dict(width=0)),
        showlegend=showlegend, legendgroup="base",
        hovertemplate="%{x}<br>majority-class acc=%{y:.3f}<extra></extra>",
    ), row=1, col=col)


def fig_accuracy(results: dict) -> None:
    cv_rows = {"overall": results["cv"]}
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("10-fold CV", "Leave-one-dataset-out", "Leave-one-model-out"))
    _panel(fig, 1, cv_rows, showlegend=True)
    _panel(fig, 2, results["dataset"], showlegend=False)
    _panel(fig, 3, results["model"], showlegend=False)
    fig.update_yaxes(range=[0, 1], title_text="accuracy", row=1, col=1)
    fig.update_layout(
        title="Correctness classifier — accuracy vs. always guess correct or always guess incorrect",
        font=dict(size=13), bargap=0.35, height=540, width=1450,
        legend=dict(orientation="h", yanchor="top", y=-0.32, x=0.5, xanchor="center"),
        margin=dict(t=70, b=170))
    fig.update_xaxes(tickangle=-40)
    fig.write_image(FIG / "accuracy.png", scale=2)
    fig.write_html(FIG / "accuracy.html", include_plotlyjs="cdn")


CORRECT_C, WRONG_C = "#2CA02C", "#EF553B"   # green = higher in correct, red = higher in incorrect


def fig_importance() -> None:
    d = F.load()
    m = xgb(_spw(d.y)).fit(d.X, d.y)
    imp = m.feature_importances_
    # direction: sign of each feature's correlation with the correctness label
    sd = d.X.std(0)
    corr = np.array([np.corrcoef(d.X[:, i], d.y)[0, 1] if sd[i] > 0 else 0.0
                     for i in range(d.X.shape[1])])
    idx = np.argsort(imp)[-15:]
    colors = [CORRECT_C if corr[i] >= 0 else WRONG_C for i in idx]
    fig = go.Figure()
    fig.add_bar(x=imp[idx], y=[_wrap(pretty(d.feature_names[i])) for i in idx], orientation="h",
                marker_color=colors, showlegend=False, customdata=corr[idx],
                hovertemplate="gain=%{x:.3f}<br>corr with correct=%{customdata:+.2f}<extra></extra>")
    # legend proxies for the two directions
    fig.add_bar(x=[None], y=[None], marker_color=CORRECT_C, name="higher in correct traces")
    fig.add_bar(x=[None], y=[None], marker_color=WRONG_C, name="higher in incorrect traces")
    fig.update_layout(title="XGBoost feature importance (top 15)", xaxis_title="gain importance",
                      font=dict(size=12), height=1000, width=1080, bargap=0.45,
                      margin=dict(l=340, t=60, b=80),
                      legend=dict(orientation="h", yanchor="top", y=-0.07, x=0.5, xanchor="center"))
    fig.write_image(FIG / "importance.png", scale=2)
    fig.write_html(FIG / "importance.html", include_plotlyjs="cdn")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    results = json.load(open(RES / "metrics.json"))
    fig_accuracy(results)
    fig_importance()
    print(f"wrote {FIG}/accuracy.png and {FIG}/importance.png")


if __name__ == "__main__":
    main()
