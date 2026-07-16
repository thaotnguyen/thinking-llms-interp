#!/usr/bin/env python3
"""
Collect the per-taxonomy title files into artefacts the rest of the codebase can read.

Writes TWO things, both under annotate-saes/results/vars/titles/ -- nothing outside
that directory is created or modified. In particular the files under
train-saes/results/vars/ (sourced from matthewshu/medcasereasoning-cot-activations)
are the gold standard and are NEVER touched.

  1. sae_topk_results_{model}_layer{L}_dedup.json
     Same shape as the repo's existing sae_topk_results_*.json, so existing readers
     work unchanged:
         results_by_cluster_size["{K}"]["all_results"][0]["categories"]
             -> [[cluster_id, title, description], ...]
     Plus a "dead_latents" list per K, which the original schema had no way to express
     (the reference simply omitted dead latents, which is why several of the original
     taxonomies silently have fewer than K entries).

  2. universal_taxonomy_input.json
     One flat record per (model, layer, K, latent) -- the join key for the cross-model
     alignment step:
         {model, layer, n_clusters, latent, title, description, n_rows, dead}
     This is the table that pairs with the dense latent values in
     annotate-saes/results/vars/latents_dedup/{model}/layer{L}/latents.npz, where
     column `latent` of `k{K}` is that latent's raw encoder value per sentence.

Usage:  python3 annotate-saes/merge_titles.py
"""
import glob
import json
import os
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TITLES = os.path.join(REPO, "annotate-saes/results/vars/titles/dedup")
OUT = os.path.join(REPO, "annotate-saes/results/vars/titles")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-set", choices=["dedup", "nndedup"], default="dedup")
    args = ap.parse_args()
    titles_dir = os.path.join(os.path.dirname(TITLES), args.sae_set)
    suffix = "" if args.sae_set == "dedup" else "_" + args.sae_set
    files = sorted(glob.glob(os.path.join(titles_dir, "titles_*.json")))
    print(f"{len(files)} taxonomy title files ({args.sae_set})")

    by_model_layer = defaultdict(dict)
    flat = []
    n_titles = n_dead = n_err = 0

    for f in files:
        d = json.load(open(f))
        m, L, k = d["model_id"], d["layer"], d["n_clusters"]
        n_err += d.get("n_errors", 0)

        by_model_layer[(m, L)][str(k)] = {
            "all_results": [{"categories": d["categories"]}],
            "dead_latents": d["dead_latents"],
            "n_rows_ranked": d["n_rows_ranked"],
        }

        desc_by_id = {c[0]: (c[1], c[2]) for c in d["categories"]}
        for c in range(k):
            dead = c in d["dead_latents"]
            title, desc = desc_by_id.get(str(c), ("", ""))
            flat.append({
                "model": m, "layer": L, "n_clusters": k, "latent": c,
                "title": title, "description": desc, "dead": dead,
            })
            n_titles += (not dead)
            n_dead += dead

    os.makedirs(OUT, exist_ok=True)
    dtag = "dedup" if args.sae_set == "dedup" else args.sae_set
    for (m, L), by_k in sorted(by_model_layer.items()):
        p = os.path.join(OUT, f"sae_topk_results_{m}_layer{L}_{dtag}.json")
        with open(p, "w") as fh:
            json.dump({
                "clustering_method": "sae_topk",
                "model_id": m,
                "layer": L,
                "sae_set": args.sae_set,
                "annotator": "openai/gpt-oss-120b (llama.cpp, native MXFP4)",
                "results_by_cluster_size": dict(sorted(by_k.items(), key=lambda x: int(x[0]))),
            }, fh, indent=2)

    p = os.path.join(OUT, f"universal_taxonomy_input{suffix}.json")
    with open(p, "w") as fh:
        json.dump(flat, fh, indent=2)

    print(f"wrote {len(by_model_layer)} sae_topk_results_*_{dtag}.json "
          f"(one per model/layer) + universal_taxonomy_input{suffix}.json")
    print(f"  {n_titles:,} titled latents, {n_dead} dead latents, {n_err} failed LLM calls")
    if n_err:
        print("  NOTE: failed calls left a placeholder title; re-run generate_titles.py "
              "with --force on those taxonomies, or re-parse raw_responses.")


if __name__ == "__main__":
    main()
