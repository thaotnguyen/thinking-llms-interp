#!/usr/bin/env python3
"""
Show what each SAE's training corpus actually looks like, old vs dedup.

For every model/layer/K it reports, per latent:
  - how many rows land on it (argmax)
  - its top-activating example sentences
  - a "loop score": what fraction of its rows are duplicate sentences, and the
    single most-repeated sentence on that latent

That is the evidence behind the repetition-loop finding and the dedup fix, and it
is regenerated straight from the artefacts on disk -- nothing is hard-coded.

Usage:
  python3 annotate-saes/inspect_corpus.py --model qwq-32b --layer 9 --k 10
  python3 annotate-saes/inspect_corpus.py --summary          # the whole 9-model table
  python3 annotate-saes/inspect_corpus.py --summary --json out.json
"""
import argparse
import importlib.util
import json
import os
from collections import Counter

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDECAR = {
    "old": os.path.join(REPO, "annotate-saes/results/vars/latents"),
    "dedup": os.path.join(REPO, "annotate-saes/results/vars/latents_dedup"),
}
MODELS = [
    "deepseek-r1-distill-llama-8b", "deepseek-r1-distill-qwen-14b",
    "huatuogpt-o1-8b", "gpt-oss-20b", "qwq-32b", "qwen3.6-27b",
    "gemma-4-31b-it", "ministral-3-14b-reasoning-2512", "glm-4.7-flash",
]

_gt = None


def gt():
    """Reuse generate_titles' verified, row-alignment-checked text loader."""
    global _gt
    if _gt is None:
        spec = importlib.util.spec_from_file_location(
            "gt", os.path.join(REPO, "annotate-saes/generate_titles.py"))
        _gt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_gt)
    return _gt


def load(sae_set, model, layer, k):
    d = os.path.join(SIDECAR[sae_set], model, f"layer{layer}")
    z = np.load(os.path.join(d, "latents.npz"))
    vals, am = z[f"k{k}"], z[f"argmax_k{k}"]
    texts = gt().texts_for(model, layer, d)
    if sae_set == "dedup":
        idx = np.load(os.path.join(d, "index.npz"), allow_pickle=True)
        rows = np.where(idx["in_training"].astype(bool))[0]
        vals, am = vals[rows], am[rows]
    else:
        rows = np.arange(len(am))
    return vals, am, texts, rows


def latent_report(sae_set, model, layer, k, n_examples=5):
    vals, am, texts, rows = load(sae_set, model, layer, k)
    out = []
    for c in range(k):
        sel = np.where(am == c)[0]
        if len(sel) == 0:
            out.append({"latent": c, "n_rows": 0, "dead": True})
            continue
        sent = [texts[rows[i]] for i in sel]
        cnt = Counter(sent)
        top_sent, top_n = cnt.most_common(1)[0]
        n_uniq = len(cnt)
        dup_frac = 1.0 - n_uniq / len(sent)
        order = sel[np.argsort(-vals[sel, c])][:n_examples]
        out.append({
            "latent": c,
            "n_rows": int(len(sel)),
            "n_unique": n_uniq,
            "dup_frac": round(dup_frac, 4),
            "most_repeated": top_sent[:100],
            "most_repeated_n": int(top_n),
            # a latent is "loop-dominated" if one single sentence owns >=25% of it
            "loop_latent": bool(top_n / len(sel) >= 0.25),
            "top_examples": [texts[rows[i]][:120] for i in order],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sae-set", choices=["old", "dedup", "both"], default="both")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()

    if args.summary:
        table, rowsout = [], {}
        for m in MODELS:
            layer = sorted(int(d[5:]) for d in os.listdir(
                os.path.join(SIDECAR["dedup"], m)) if d.startswith("layer"))[0]
            rec = {}
            for s in ("old", "dedup"):
                r = latent_report(s, m, layer, 10, n_examples=0)
                live = [x for x in r if not x.get("dead")]
                rec[s] = {
                    "layer": layer,
                    "n_rows": sum(x["n_rows"] for x in live),
                    "latents_used": len(live),
                    "loop_latents": sum(1 for x in live if x["loop_latent"]),
                    "corpus_dup_frac": round(
                        1 - sum(x["n_unique"] for x in live)
                        / max(1, sum(x["n_rows"] for x in live)), 4),
                    "worst_latent": max(live, key=lambda x: x["most_repeated_n"]),
                }
            rowsout[m] = rec
            o, d = rec["old"], rec["dedup"]
            table.append(
                f"{m:31s} L{o['layer']:<3d} "
                f"rows {o['n_rows']:>7,}->{d['n_rows']:>7,}  "
                f"dup {o['corpus_dup_frac']*100:5.1f}%->{d['corpus_dup_frac']*100:5.1f}%  "
                f"loop {o['loop_latents']}/10->{d['loop_latents']}/10  "
                f"used {o['latents_used']}->{d['latents_used']}")
        print("SAE TRAINING CORPUS, first layer of each model, k=10  (OLD -> DEDUP)\n")
        print("\n".join(table))
        print("\nWorst single repeated sentence per model (old corpus):")
        for m, rec in rowsout.items():
            w = rec["old"]["worst_latent"]
            wd = rec["dedup"]["worst_latent"]
            print(f"  {m:31s} {w['most_repeated']!r} x{w['most_repeated_n']:,}"
                  f"   -> after dedup, worst is x{wd['most_repeated_n']:,}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rowsout, f, indent=2)
            print(f"\nwrote {args.json}")
        return

    for s in (["old", "dedup"] if args.sae_set == "both" else [args.sae_set]):
        print(f"\n{'='*78}\n{s.upper()}  {args.model} layer{args.layer} k={args.k}\n{'='*78}")
        for r in latent_report(s, args.model, args.layer, args.k):
            if r.get("dead"):
                print(f"  latent {r['latent']:2d}: DEAD (0 rows)")
                continue
            flag = "  <-- LOOP LATENT" if r["loop_latent"] else ""
            print(f"  latent {r['latent']:2d}: {r['n_rows']:>7,} rows, "
                  f"{r['dup_frac']*100:5.1f}% dup{flag}")
            print(f"       most repeated: {r['most_repeated']!r} x{r['most_repeated_n']:,}")
            for e in r["top_examples"][:3]:
                print(f"       top: {e!r}")


if __name__ == "__main__":
    main()
