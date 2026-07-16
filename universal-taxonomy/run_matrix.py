"""Run the focused experiment matrix and assemble the comparison report.

Focused set: {nndedup, dedup} x {Approach A, Approach B} x {minibatch_kmeans
(k-sweep), hdbscan}, each with full naming + LLM-accuracy. Approach B runs one
Optuna search per variant to pick the per-model SAEs, then clusters the winner.
"""

from __future__ import annotations

import argparse
import json
import logging

import approach_a
import approach_b
import report
import run
from config import Config, VARIANTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

K_SWEEP = [8, 10, 12, 14, 16, 18, 20]
METHODS = [
    ("minibatch_kmeans", {"k_sweep": K_SWEEP, "cluster_params": {}}),
    ("hdbscan", {"k_sweep": None, "cluster_params": {"min_cluster_size": 1000}}),
]

N_TOTAL = 120_000
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=VARIANTS, choices=VARIANTS)
    variants = ap.parse_args().variants
    cfg = Config()
    for variant in variants:
      try:
        # Approach A needs per-sentence vectors on disk
        for m in cfg.models:
            approach_a.run_model(cfg, variant, m, normalize=True, force=False)

        # Approach B: one joint (multi-objective) search per variant, train split only
        best_sel, _study = approach_b.search(cfg, variant, n_trials=120, k=12, sub=4000, seed=SEED)
        sel_path = cfg.out_root / variant / "runs" / "approach_b_selection.json"
        sel_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"variant": variant, "selection": {m: list(c) for m, c in best_sel.items()}},
                  open(sel_path, "w"), indent=2)

        for approach in ("a", "b"):
            selection = best_sel if approach == "b" else None
            for method, opts in METHODS:
                logger.info("=== %s | approach %s | %s ===", variant, approach.upper(), method)
                try:
                    result = run.run_experiment(
                        cfg, variant=variant, approach=approach, method=method,
                        k=None, k_sweep=opts["k_sweep"], normalized=True,
                        n_total=N_TOTAL, seed=SEED, do_name=True, do_llm=True,
                        llm_n=8, llm_repeats=2,
                        cluster_params=opts["cluster_params"], selection=selection,
                    )
                    run.save(cfg, result)
                except Exception as e:  # noqa: BLE001 - keep the matrix going
                    logger.exception("run failed (%s/%s/%s): %s", variant, approach, method, e)
      except Exception as e:  # noqa: BLE001 - one variant failing must not kill the rest
        logger.exception("variant %s failed entirely: %s", variant, e)

    rows = report._rows(cfg)
    body = report.markdown_table(rows) + "\n" + report.best_taxonomy_md(cfg, rows)
    out = cfg.out_root / "comparison.md"
    out.write_text("# Universal-taxonomy experiment comparison\n\n" + body + "\n")
    print(report.markdown_table(rows))
    logger.info("wrote %s (%d runs)", out, len(rows))


if __name__ == "__main__":
    main()
