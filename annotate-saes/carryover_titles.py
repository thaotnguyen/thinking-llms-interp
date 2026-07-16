#!/usr/bin/env python3
"""Decide which near-dup latents need fresh titles, and carry the rest over from the
string-dedup titles we already generated.

For each (model, layer, K, latent), compare the near-dup latent's top-N example
sentences to the SAME latent in the string-dedup SAE. If the example sets overlap
heavily (Jaccard >= --overlap), the two latents describe the same behavior, so the
string-dedup title is reused verbatim. Only the latents that genuinely shifted go to
the LLM. For models where near-dup ~= string-dedup this carries over most titles and
cuts the titling time from hours to minutes.

Writes annotate-saes/results/vars/titles/nndedup/titles_{model}_layer{L}_clusters{K}.json
in the SAME schema generate_titles.py produces, so merge_titles.py works on it too.
Latents flagged for re-titling get an empty title here; run generate_titles.py
--retry-failed against titles/nndedup afterwards to fill them in with the LLM.

Usage:
  python3 annotate-saes/carryover_titles.py --overlap 0.6
"""
import argparse
import importlib.util
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NN = os.path.join(REPO, "annotate-saes/results/vars/latents_nndedup")
STR_TITLES = os.path.join(REPO, "annotate-saes/results/vars/titles/dedup")
OUT = os.path.join(REPO, "annotate-saes/results/vars/titles/nndedup")
KS = [10, 12, 14, 16, 18, 20]

spec = importlib.util.spec_from_file_location(
    "gt", os.path.join(REPO, "annotate-saes/generate_titles.py"))
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)


def top_examples(sae_set_dir, model, layer, k, n=25, cap=600):
    """Top-n example sentences per LIVE latent for the given sidecar set."""
    d = os.path.join(sae_set_dir, model, f"layer{layer}")
    z = np.load(os.path.join(d, "latents.npz"))
    idx = np.load(os.path.join(d, "index.npz"), allow_pickle=True)
    mask = idx["in_training"].astype(bool)
    rows = np.where(mask)[0]
    vals, am = z[f"k{k}"][rows], z[f"argmax_k{k}"][rows]
    texts = gt.texts_for(model, layer, d)
    out = {}
    for c in range(k):
        sel = np.where(am == c)[0]
        if len(sel) == 0:
            continue
        order = sel[np.argsort(-vals[sel, c])][:n]
        out[c] = set(texts[rows[i]][:cap] for i in order)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", type=float, default=0.6,
                    help="Jaccard >= this between near-dup and string-dedup top examples "
                         "=> reuse the string-dedup title")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--topn", type=int, default=25)
    args = ap.parse_args()

    models = args.models or gt.MODELS
    os.makedirs(OUT, exist_ok=True)
    carried = fresh = dead = 0

    for m in models:
        layers = gt.layers_for("dedup", m)  # nndedup has the same layers
        for L in layers:
            if not os.path.isdir(os.path.join(NN, m, f"layer{L}")):
                continue
            nn_ex = {k: top_examples(NN, m, L, k, args.topn) for k in KS}
            str_ex = {k: top_examples(gt.SIDECAR["dedup"], m, L, k, args.topn) for k in KS}

            for k in KS:
                stpath = os.path.join(STR_TITLES, f"titles_{m}_layer{L}_clusters{k}.json")
                st = json.load(open(stpath)) if os.path.exists(stpath) else {"categories": []}
                st_title = {c[0]: (c[1], c[2]) for c in st.get("categories", [])}

                cats, raw = [], {}
                live, deadl = [], []
                for c in range(k):
                    if c not in nn_ex[k]:
                        deadl.append(c); dead += 1
                        continue
                    live.append(c)
                    a = nn_ex[k][c]
                    # The two SAEs are trained independently -> latent INDICES are not
                    # aligned. Match this near-dup latent to the string-dedup latent with
                    # the highest example overlap (many-to-one is fine), and carry its
                    # title over only if that best overlap is high enough.
                    best_j, best_c = 0.0, None
                    for sc, b in str_ex[k].items():
                        j = len(a & b) / max(1, len(a | b))
                        if j > best_j:
                            best_j, best_c = j, sc
                    ttl = st_title.get(str(best_c)) if best_c is not None else None
                    if best_j >= args.overlap and ttl and \
                       ttl[0] not in ("Unnamed Cluster", "", "No description available"):
                        cats.append([str(c), ttl[0], ttl[1]])
                        raw[str(c)] = "__CARRIED__ from str-latent %s (jaccard %.2f)" % (best_c, best_j)
                        carried += 1
                    else:
                        cats.append([str(c), "", ""]); raw[str(c)] = "__NEEDS_LLM__"
                        fresh += 1

                gt.write_atomic(os.path.join(OUT, f"titles_{m}_layer{L}_clusters{k}.json"), {
                    "model_id": m, "layer": L, "n_clusters": k, "sae_set": "nndedup",
                    "annotator": "openai/gpt-oss-120b (llama.cpp, native MXFP4)",
                    "description_examples": 200, "n_trace_examples": 3,
                    "n_categories_examples": 5, "seed": gt.stable_seed("nndedup", m, L, k),
                    "n_rows_ranked": int(sum(1 for _ in nn_ex[k])),
                    "live_latents": live, "dead_latents": deadl, "n_errors": 0,
                    "carryover_overlap": args.overlap, "categories": cats,
                    "raw_responses": raw,
                })
            print(f"{m} L{L}: done", flush=True)

    print(f"\ncarried over from string-dedup: {carried}")
    print(f"need fresh LLM titles         : {fresh}")
    print(f"dead (no title)               : {dead}")
    print(f"\nNext: generate_titles.py --retry-failed --sae-set nndedup fills the {fresh} fresh ones.")


if __name__ == "__main__":
    main()
