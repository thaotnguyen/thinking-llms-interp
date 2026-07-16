"""Assemble the comparison table across completed experiment runs.

Scans ``results/{variant}/runs/*.json`` and ranks every (approach, variant,
method) by the held-out metrics. LLM accuracy (lift over chance) is the primary
arbiter; the geometric objectives and the title-embedding orthogonality follow.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import List

import numpy as np

from config import Config, VARIANTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _rows(cfg: Config) -> List[dict]:
    rows = []
    for variant in VARIANTS:
        rdir = cfg.out_root / variant / "runs"
        if not rdir.exists():
            continue
        for f in sorted(rdir.glob("*.json")):
            if f.name.endswith("_selection.json"):
                continue
            d = json.load(open(f))
            if "metrics" not in d:
                continue
            acc = d.get("llm_accuracy", {})
            rows.append({
                "variant": d["variant"], "approach": d["approach"], "method": d["method"],
                "n_clusters": d["n_clusters"], **d["metrics"],
                "label_orth": d.get("title_embedding_orthogonality"),
                "llm_acc": acc.get("accuracy_mean"), "lift": acc.get("lift_over_chance"),
            })
    return rows


def markdown_table(rows: List[dict]) -> str:
    rows = sorted(rows, key=lambda r: (-(r["lift"] if r["lift"] is not None else -1)))
    head = ("| variant | app | method | k | CH | orth | univ | label_orth | llm_acc | lift |\n"
            "|---|---|---|--:|--:|--:|--:|--:|--:|--:|")
    lines = [head]
    for r in rows:
        def fmt(v, p=3):
            return f"{v:.{p}f}" if isinstance(v, (int, float)) else "-"
        lines.append(
            f"| {r['variant']} | {r['approach'].upper()} | {r['method']} | {r['n_clusters']} "
            f"| {fmt(r['ch'], 0)} | {fmt(r['orthogonality'])} | {fmt(r['universality'])} "
            f"| {fmt(r['label_orth'])} | {fmt(r['llm_acc'])} | {fmt(r['lift'])} |"
        )
    return "\n".join(lines)


def best_taxonomy_md(cfg: Config, rows: List[dict]) -> str:
    """The named taxonomy + per-model coverage of the highest-lift run."""
    ranked = [r for r in rows if r.get("lift") is not None]
    if not ranked:
        return ""
    b = max(ranked, key=lambda r: r["lift"])
    path = cfg.out_root / b["variant"] / "runs" / f"{b['approach']}_{b['method']}_k{b['n_clusters']}.json"
    d = json.load(open(path))
    titles, descs, cov = d.get("titles", {}), d.get("descriptions", {}), d.get("coverage", {})
    if not titles:
        return ""
    models = sorted(cov)
    lines = [f"\n## Best taxonomy — {b['variant']} / Approach {b['approach'].upper()} / "
             f"{b['method']} (k={b['n_clusters']}, LLM-acc lift {b['lift']:+.3f})\n",
             "| id | category | mean cov | min-model cov | models >1% |",
             "|--:|---|--:|--:|--:|"]
    for cid in sorted(titles, key=int):
        vals = [cov[m].get(cid, 0.0) for m in models]
        lines.append(f"| {cid} | {titles[cid]} | {np.mean(vals):.3f} | {min(vals):.3f} "
                     f"| {sum(v > 0.01 for v in vals)}/{len(models)} |")
    lines.append("\n**Descriptions:**")
    for cid in sorted(titles, key=int):
        lines.append(f"- **{titles[cid]}**: {descs.get(cid, '')}")
    return "\n".join(lines)


def main() -> None:
    argparse.ArgumentParser().parse_args()
    cfg = Config()
    rows = _rows(cfg)
    body = markdown_table(rows) + "\n" + best_taxonomy_md(cfg, rows)
    print(markdown_table(rows))
    out = cfg.out_root / "comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Universal-taxonomy experiment comparison\n\n" + body + "\n")
    logger.info("wrote %s (%d runs)", out, len(rows))


if __name__ == "__main__":
    main()
