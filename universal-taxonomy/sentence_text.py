"""Map ``(pmcid, sentence_idx) -> sentence text`` for a (variant, model).

The rows of ``index.npz`` are in the same order as the flattened sentences of
the annotated-responses file (verified exact), so parsing the K=10 annotated
file of the first layer recovers every sentence's text. Text is identical across
layers/K for a model, so one file suffices.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Tuple

from config import Config

logger = logging.getLogger(__name__)

# Split the annotated_thinking field into ordered sentences (annotation output format).
SECTION_RE = re.compile(r'\["([^"\]]+)"\](.*?)\["end-section"\]', re.DOTALL)

Key = Tuple[str, int]

_CACHE: Dict[Tuple[str, str], Dict[Key, str]] = {}


def _annotated_path(cfg: Config, variant: str, model: str) -> Path:
    layer = cfg.discover_layers(variant, model)[0]
    suffix = "" if variant == "plain" else f"_{variant}"  # plain files carry no suffix
    return (
        cfg.repo_root / "generate-responses" / "results" / "vars"
        / f"annotated_responses_{model}_layer{layer}_clusters10{suffix}.json"
    )


def load_sentences(cfg: Config, variant: str, model: str) -> Dict[Key, str]:
    ck = (variant, model)
    if ck in _CACHE:
        return _CACHE[ck]
    path = _annotated_path(cfg, variant, model)
    recs = json.load(open(path))
    out: Dict[Key, str] = {}
    for r in recs:
        pmcid = r["pmcid"]
        for i, (_tag, sent) in enumerate(SECTION_RE.findall(r["annotated_thinking"])):
            out[(pmcid, i)] = sent
    logger.info("loaded %d sentence texts for %s/%s", len(out), variant, model)
    _CACHE[ck] = out
    return out
