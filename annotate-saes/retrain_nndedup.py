#!/usr/bin/env python3
"""Retrain all 324 TopK SAEs on NEAR-DUPLICATE-deduplicated corpora, then re-annotate.

Same idea as retrain_dedup.py, but the dedup criterion is activation NEAR-duplication
rather than exact-string match.

  DEFINITION OF "NEAR" (validated empirically, see the WORKLOG measurement):
    - space:     the centered + L2-normalized activation space -- the exact space the
                 SAE consumes; there cosine similarity = dot product. (Centered with the
                 all-rows mean, which is well-defined before the dedup mask is known.)
    - criterion: per trace (per pmcid), walk sentences in sentence_idx order; KEEP a
                 sentence iff its cosine to EVERY already-kept sentence in that same trace
                 is < THRESH; else drop it as a near-duplicate. String-agnostic.
    - THRESH = 0.99. Justified by measurement on qwq L9 'Hmm.': within-loop repeats have
                 median cosine 1.000 (93% >= 0.99), across-trace same-string median 0.85,
                 distinct steps far lower -- so 0.99 isolates near-identical loop states
                 and leaves genuinely distinct steps alone.

Why near-dup rather than exact-string:
  - It removes only activation-near-identical loop states, and KEEPS same-string repeats
    whose activations are genuinely distinct (which string-dedup over-removes: 5,795 such
    rows in qwq L9, 5,184 in qwen3.6 L9).
  - It also catches loops whose surface strings differ slightly ('Hmm.' vs 'Hmm').
  Note (measured): for these models near-dup lands within ~2-3% of string-dedup, and it
  does NOT fix qwen3.6, whose contamination is cross-CASE single occurrences that no
  within-trace method touches. It is the more principled method, not a different taxonomy.

Everything else matches retrain_dedup.py exactly: same SAE, same training law, mean
recomputed on the KEPT rows, annotation over ALL rows. Outputs use a _nndedup suffix and
separate directories, so the string-dedup and matthewshu/HF artifacts are untouched.

Run with /venv/main/bin/python (needs torch + numpy).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encode_latents import (CLUSTERS, MODELS, RESP_DIR, fmt_row,  # noqa: E402
                            json_atomic, savez_atomic, find_act, model_layers)
# reuse the exact SAE + training + encoding from the string-dedup pipeline
from retrain_dedup import SAE, train_sae, encode_all, raw_path, TOPK  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAE_OUT = os.path.join(REPO, "annotate-saes/results/vars/saes_nndedup")
TEXT_OUT = os.path.join(REPO, "generate-responses/results/vars")
SIDECAR_OUT = os.path.join(REPO, "annotate-saes/results/vars/latents_nndedup")
THRESH = 0.99


def nndedup_mask_gpu(Xg, keys, thresh, device):
    """Greedy per-trace near-dup mask on the GPU. Xg rows are unit-norm, so X @ x = cosine.

    Per trace, keep a row iff its max cosine to the already-kept rows of that trace is
    < thresh. Loop rows collapse the kept set to a handful, so this is fast in practice.
    Gathers each trace's rows once and dots against a growing prefix buffer (no per-row
    re-gather), so a diverse 500-row trace is ~one 500x500 GPU matmul, not 500 gathers.
    """
    by_pm = defaultdict(list)
    for i, (pm, si) in enumerate(keys):
        by_pm[pm].append((si, i))
    d = Xg.shape[1]
    keep = np.zeros(len(keys), dtype=bool)
    for pm, rows in by_pm.items():
        rows.sort()
        idxs = [i for _si, i in rows]
        buf = Xg[idxs]                                   # (n_trace, d), one gather
        Kbuf = torch.empty((len(idxs), d), device=device, dtype=buf.dtype)
        nk = 0
        for j in range(len(idxs)):
            x = buf[j]
            if nk == 0 or float(torch.mv(Kbuf[:nk], x).max()) < thresh:
                keep[idxs[j]] = True
                Kbuf[nk] = x
                nk += 1
    return keep


def run(model, layer, gpu):
    t0 = time.time()
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    rp = raw_path(model, layer)
    if rp is None:
        return {"model": model, "layer": layer, "status": "missing_raw"}

    import pickle
    with open(rp, "rb") as f:
        raw, texts, keys, _old_mean = pickle.load(f)

    # Reorder raw into the canonical centered-pkl key order (raw/centered came from separate
    # unseeded-shuffle runs), so near-dup outputs stay row-aligned with the string-dedup ones.
    cen = find_act(model, layer)
    if cen:
        with open(cen, "rb") as f:
            _cx, cen_texts, cen_keys, _cm = pickle.load(f)
        del _cx
        pos = {k: i for i, k in enumerate(keys)}
        order = [pos[k] for k in cen_keys if k in pos]
        if order != list(range(len(keys))):
            raw = raw[order]
            texts = [texts[i] for i in order]
            keys = [keys[i] for i in order]
        ct = {k: t for k, t in zip(cen_keys, cen_texts)}
        assert all(ct[k] == t for k, t in zip(keys, texts)), \
            f"{model} L{layer}: raw and centered disagree on sentence text for the same key"

    N, d = raw.shape
    raw = raw.astype(np.float32, copy=False)

    # --- provisional centering (all-rows mean) to define the cosine space for the mask ---
    mean_prov = raw.mean(axis=0, keepdims=True).astype(np.float32)
    Xp = raw - mean_prov
    Xp /= (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-8)
    Xpg = torch.from_numpy(Xp).to(device)
    keep = nndedup_mask_gpu(Xpg, keys, THRESH, device)
    del Xp, Xpg
    torch.cuda.empty_cache()
    n_train = int(keep.sum())

    # --- final centering: recompute the mean on the KEPT rows, then center + normalize all ---
    mean_new = raw[keep].mean(axis=0, keepdims=True).astype(np.float32)
    X = raw - mean_new
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    del raw

    Xg = torch.from_numpy(X).to(device)
    Xtr = Xg[torch.from_numpy(keep).to(device)]

    os.makedirs(SAE_OUT, exist_ok=True)
    ldir = os.path.join(SIDECAR_OUT, model, f"layer{layer}")
    os.makedirs(ldir, exist_ok=True)

    latents, argmaxes, params, losses = {}, {}, {}, {}
    for k in CLUSTERS:
        sae, loss = train_sae(Xtr, k, device)
        labels = encode_all(sae, Xtr, device).argmax(axis=1)
        Y = encode_all(sae, Xg, device)
        centers = sae.W_dec.data.cpu().numpy()
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        ck = {
            "encoder_weight": sae.encoder.weight.data.cpu(),
            "encoder_bias": sae.encoder.bias.data.cpu(),
            "decoder_weight": sae.W_dec.data.cpu(),
            "b_dec": sae.b_dec.data.cpu(),
            "input_dim": d, "num_latents": k, "topk": TOPK, "loss": loss,
            "cluster_labels": labels, "cluster_centers": centers,
            "mean_vector": mean_new,
            "n_train": n_train, "n_all": N,
            "dedup": f"within_cot_near_dup_cos{THRESH}",
        }
        torch.save(ck, os.path.join(
            SAE_OUT, f"sae_{model}_layer{layer}_clusters{k}_nndedup.pt"))
        latents[k], argmaxes[k], params[k], losses[k] = Y, Y.argmax(axis=1), ck, loss
        del sae
        torch.cuda.empty_cache()

    write_outputs(model, layer, texts, keys, keep, mean_new, latents, argmaxes, params,
                  losses, N, d, n_train, time.time() - t0)
    del Xg, Xtr
    torch.cuda.empty_cache()
    return {"model": model, "layer": layer, "status": "ok", "N": N, "n_train": n_train,
            "dropped": N - n_train, "secs": round(time.time() - t0, 1),
            "loss": {k: round(losses[k], 4) for k in CLUSTERS}}


def write_outputs(model, layer, texts, keys, keep, mean_new, latents, argmaxes, params,
                  losses, N, d, n_train, secs):
    ldir = os.path.join(SIDECAR_OUT, model, f"layer{layer}")
    pmcids = np.array([k[0] for k in keys], dtype="S32")
    sidx = np.array([k[1] for k in keys], dtype=np.int32)

    savez_atomic(os.path.join(ldir, "latents.npz"),
                 **{f"k{k}": latents[k] for k in CLUSTERS},
                 **{f"argmax_k{k}": argmaxes[k].astype(np.int8) for k in CLUSTERS})
    savez_atomic(os.path.join(ldir, "index.npz"), pmcid=pmcids, sentence_idx=sidx,
                 in_training=keep)
    savez_atomic(os.path.join(ldir, "sae_params.npz"), **{
        f"{fld}_k{k}": np.asarray(params[k][fld]) for k in CLUSTERS
        for fld in ("encoder_weight", "encoder_bias", "b_dec", "decoder_weight",
                    "cluster_centers", "mean_vector")})

    uniq, inv = np.unique(pmcids, return_inverse=True)
    prof = {"pmcids": uniq}
    for k in CLUSTERS:
        cnt = np.bincount(inv, minlength=len(uniq)).astype(np.float64)[:, None]
        means = np.zeros((len(uniq), k))
        np.add.at(means, inv, latents[k].astype(np.float64))
        prof[f"mean_k{k}"] = (means / cnt).astype(np.float32)
        hist = np.zeros((len(uniq), k), dtype=np.int32)
        np.add.at(hist, (inv, argmaxes[k]), 1)
        prof[f"argmax_hist_k{k}"] = hist
    savez_atomic(os.path.join(ldir, "case_profiles.npz"), **prof)

    resp_meta = {}
    rp = os.path.join(RESP_DIR, f"responses_{model}.json")
    if os.path.exists(rp):
        for r in json.load(open(rp)):
            resp_meta[r["pmcid"]] = r

    order, by_case = [], defaultdict(list)
    for row, (pm, si) in enumerate(keys):
        if pm not in by_case:
            order.append(pm)
        by_case[pm].append((si, row))
    for pm in order:
        by_case[pm].sort()

    for k in CLUSTERS:
        Y = latents[k]
        recs = []
        for pm in order:
            parts = ['["%s"]%s["end-section"]' % (fmt_row(Y[row]), texts[row])
                     for _si, row in by_case[pm]]
            src = resp_meta.get(pm, {})
            recs.append({"pmcid": pm, "question_id": src.get("question_id"),
                         "category": src.get("category", "diagnosis"),
                         "dataset_name": src.get("dataset_name"),
                         "annotated_thinking": "".join(parts).strip()})
        json_atomic(os.path.join(
            TEXT_OUT,
            f"annotated_responses_{model}_layer{layer}_clusters{k}_nndedup.json"), recs)

    meta = {"model": model, "layer": layer, "N": int(N), "n_train": int(n_train),
            "n_dropped_near_dup": int(N - n_train), "d": int(d),
            "dedup": f"within_cot_near_dup_cos{THRESH}", "secs": round(secs, 1),
            "mean_vector_sha": None}
    json_atomic(os.path.join(ldir, "meta.json"), meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models = args.models or MODELS
    jobs = []
    for m in models:
        for L in model_layers(m):
            if args.layers and L not in args.layers:
                continue
            done = os.path.join(SIDECAR_OUT, m, f"layer{L}", "meta.json")
            if not args.force and os.path.exists(done):
                continue
            jobs.append((m, L))
    print(f"{len(jobs)} (model,layer) near-dup jobs on GPUs {args.gpus}", flush=True)

    gpus = args.gpus
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=len(gpus)) as ex:
        futs = {ex.submit(run, m, L, gpus[i % len(gpus)]): (m, L)
                for i, (m, L) in enumerate(jobs)}
        for fu in as_completed(futs):
            m, L = futs[fu]
            try:
                r = fu.result()
                done += 1
                if r["status"] == "ok":
                    el = time.time() - t0
                    print(f"[{done}/{len(jobs)}] {m} L{L}: N={r['N']:,} "
                          f"kept={r['n_train']:,} dropped={r['dropped']:,} "
                          f"({r['secs']}s) | {el/60:.1f}m elapsed", flush=True)
                else:
                    print(f"[{done}/{len(jobs)}] {m} L{L}: {r['status']}", flush=True)
            except Exception as e:
                done += 1
                print(f"[{done}/{len(jobs)}] {m} L{L}: EXCEPTION {type(e).__name__}: {e}",
                      flush=True)
    print(f"\nDONE {done} jobs in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
