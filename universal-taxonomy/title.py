"""Title the whole taxonomy at once.

Instead of naming each category independently, sample N sentences from every category and
present them all in a single prompt so the LLM assigns mutually-distinct titles; repeat R
times with fresh samples, then synthesise the R candidate sets into one final taxonomy
(one title + description per category). Reads the search assignment from ``results/plain/``
and writes ``results/plain/taxonomy.json``.
"""
from __future__ import annotations

import json
import re

import numpy as np

from cache_latents import MODELS, VARIANT, cache_npz
from config import Config
from oai import chat, map_threaded

N_PER_CAT = 15
ROUNDS = 10
SEED = 42


def _parse_json(s: str) -> dict:
    m = re.search(r"\{.*\}", s or "", re.S)
    try:
        return json.loads(m.group()) if m else {}
    except Exception:
        return {}


def title_taxonomy(cfg: Config, members: dict[int, list[str]], model_name: str) -> dict[int, tuple[str, str]]:
    cats = sorted(members)

    def one_round(r: int) -> dict[int, tuple[str, str]]:
        rng = np.random.default_rng(SEED + r); blocks = []
        for c in cats:
            pool = members[c]
            pick = rng.choice(len(pool), min(N_PER_CAT, len(pool)), replace=False) if pool else []
            ex = "\n".join(f"- {pool[i]}" for i in pick) or "(empty)"
            blocks.append(f"Category {c}:\n{ex}")
        prompt = ("You are labeling a taxonomy of REASONING-STEP categories from clinical diagnostic "
                  "reasoning traces. Each category below has example sentences. Give each a short (2-5 "
                  "word) title naming the reasoning FUNCTION (e.g. 'Weighing Evidence'), NOT the medical "
                  "topic, plus a one-sentence description of that function. Titles MUST be mutually "
                  "distinct.\n\n" + "\n\n".join(blocks) +
                  '\n\nRespond with ONLY JSON: {"0":{"title":"...","description":"..."}, ...}')
        d = _parse_json(chat(prompt, model_name, seed=SEED + r))
        out = {}
        for c in cats:
            e = d.get(str(c), {})
            out[c] = (str(e.get("title", "")).strip(), str(e.get("description", "")).strip())
        return out

    rounds = map_threaded(one_round, list(range(ROUNDS)), workers=10)
    cand = {c: [rd[c] for rd in rounds if rd.get(c) and rd[c][0]] for c in cats}   # (title, desc) per round
    blocks = []
    for c in cats:
        opts = "; ".join(f"{t} — {d}" for t, d in cand[c])
        blocks.append(f"Category {c} candidates: {opts}")
    prompt = ("Below are candidate titles+descriptions for each category of a clinical-reasoning-step "
              "taxonomy, from 10 independent samples. Synthesise the FINAL taxonomy: for each category "
              "give ONE concise 2-5 word title and a one-sentence description. Titles MUST be mutually "
              "distinct and name the reasoning function.\n\n" + "\n".join(blocks) +
              '\n\nRespond with ONLY JSON: {"0":{"title":"...","description":"..."}, ...}')
    d = _parse_json(chat(prompt, model_name, seed=SEED))
    out = {}
    for c in cats:
        e = d.get(str(c), {})
        fallback = cand[c][0] if cand[c] else (f"Category {c}", "")
        out[c] = (e.get("title") or fallback[0], e.get("description") or fallback[1])
    return out


def main() -> None:
    import os
    cfg = Config(); base = cfg.out_root / VARIANT
    winner = json.load(open(base / "winner.json")); K = winner["final_k"]
    members: dict[int, list[str]] = {c: [] for c in range(K)}
    for m in MODELS:
        text = np.load(cache_npz(cfg, m), allow_pickle=True)["text"]
        lab = np.load(base / "labels" / f"{m}.npy")
        for c in range(K):
            members[c].extend(text[lab == c].tolist()[:600])
    named = title_taxonomy(cfg, members, cfg.naming_model)
    palette = ["#9467bd", "#ff7f0e", "#1f77b4", "#e377c2", "#2ca02c", "#d62728",
               "#8c564b", "#17becf", "#bcbd22", "#7f7f7f"]
    states = [{"label": named[c][0], "short_name": named[c][0][:28], "description": named[c][1],
               "color": palette[c % len(palette)]} for c in range(K)]
    json.dump({"pipeline": VARIANT, "states": states, "models": MODELS},
              open(base / "taxonomy.json", "w"), ensure_ascii=False, indent=2)
    print("TITLES:", [s["label"] for s in states], flush=True)


if __name__ == "__main__":
    main()
