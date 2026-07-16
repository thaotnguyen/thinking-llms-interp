"""Name final universal-taxonomy clusters with the Venhoff-style scheme.

This reuses the repo's original single-shot cluster-description prompt
(``utils/autograder_prompts.build_cluster_description_prompt`` — "identify the
precise cognitive function these sentences serve", with a few-shot set of example
reasoning categories and ``Title:``/``Description:`` output). No multi-round
LABELLER/JUDGE loop; one LLM call per cluster, plain-text parse.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from oai import chat, map_threaded

logger = logging.getLogger(__name__)

# Reuse the original prompt builder + few-shot categories from the repo.
_UTILS = str((Path(__file__).resolve().parent.parent / "utils"))
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)
from autograder_prompts import build_cluster_description_prompt  # noqa: E402

_TITLE_RE = re.compile(r"Title:\s*(.+)")
_DESC_RE = re.compile(r"Description:\s*(.+)", re.DOTALL)


@dataclass
class Label:
    title: str
    description: str


def _parse(text: str) -> Label:
    tm = _TITLE_RE.search(text)
    dm = _DESC_RE.search(text)
    title = tm.group(1).strip() if tm else ""
    desc = dm.group(1).strip() if dm else ""
    # keep description from bleeding into a trailing "Title:" if order flipped
    if "Title:" in desc:
        desc = desc.split("Title:")[0].strip()
    return Label(title=title or "Untitled", description=desc)


def name_cluster(
    examples: Sequence[str],
    *,
    model: str = "gpt-4o-mini",
    n_examples: int = 40,
    n_categories_examples: int = 5,
    seed: int = 42,
) -> Label:
    prompt = build_cluster_description_prompt(
        list(examples)[:n_examples],
        trace_examples_text="",
        n_categories_examples=n_categories_examples,
    )
    return _parse(chat(prompt, model, seed=seed))


def name_clusters(
    cluster_examples: Dict[int, List[str]],
    *,
    model: str = "gpt-4o-mini",
    n_examples: int = 40,
    n_categories_examples: int = 5,
    seed: int = 42,
) -> Dict[int, Label]:
    cids = sorted(cluster_examples)

    def _one(cid: int) -> Label:
        return name_cluster(cluster_examples[cid], model=model, n_examples=n_examples,
                            n_categories_examples=n_categories_examples, seed=seed + cid)

    out = dict(zip(cids, map_threaded(_one, cids)))
    for cid in cids:
        logger.info("c%d: %s", cid, out[cid].title)
    return out
