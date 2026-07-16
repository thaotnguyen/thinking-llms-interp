#!/usr/bin/env python3
"""Retrain all 324 TopK SAEs on de-duplicated corpora, then re-annotate every trace with them.

Why: 28-62% of the rows each SAE was originally trained on are EXACT DUPLICATE sentences,
produced by repetition loops in the generations. That handed 1-4 of every 10 taxonomy items
to "loop clusters" (e.g. qwq-32b layer 9 has two latents that are literally the word "Hmm.",
covering 95k of its 261k rows). The contamination ranges from 12% to 58% across models, which
makes the per-model taxonomies non-comparable -- fatal for the universal-taxonomy alignment.

Fix: dedup WITHIN each CoT (keep the first occurrence of a sentence, drop later repeats), and
recompute the centering mean on the deduped rows -- the existing mean_vector is itself skewed,
since a third to a half of the rows it averaged are loop spam. 1-word sentences like "Hmm." are
KEPT: they are a real reasoning behaviour, just massively over-represented by the loops.

Training reproduces utils/clustering_methods.py::clustering_sae_topk exactly (same LR law, loss,
batch size, epochs, patience, decoder renorm) -- only the corpus and the mean change.

Annotation covers ALL rows (including the dropped duplicate ones), so every trace is complete;
only SAE *training* sees the deduped subset.

Run with /venv/main/bin/python (needs torch + numpy).
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encode_latents import (CLUSTERS, MODELS, SNAPSHOT, RESP_DIR, fmt_row,  # noqa: E402
                            json_atomic, savez_atomic, model_layers)

RAW_DIR = os.path.join(SNAPSHOT, "activations/raw")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAE_OUT = os.path.join(REPO, "annotate-saes/results/vars/saes_dedup")
TEXT_OUT = os.path.join(REPO, "generate-responses/results/vars")
SIDECAR_OUT = os.path.join(REPO, "annotate-saes/results/vars/latents_dedup")
TOPK = 3


class SAE(torch.nn.Module):
    """Verbatim from utils/sae.py -- reproduced so this script has no import-path dependency."""

    def __init__(self, d_in, num_latents, k=1):
        super().__init__()
        self.encoder = torch.nn.Linear(d_in, num_latents, bias=True)
        self.encoder.bias.data.zero_()
        self.W_dec = torch.nn.Parameter(self.encoder.weight.data.clone())
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))
        self.k = k
        self.set_decoder_norm_to_unit_norm()

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + 1e-5

    def encode(self, x):
        forward = self.encoder(x - self.b_dec)
        return forward.topk(self.k, dim=-1)

    def decode(self, top_acts, top_indices):
        bs = top_indices.shape[0]
        offsets = torch.arange(0, bs, device=top_indices.device) * self.k
        res = torch.nn.functional.embedding_bag(
            top_indices.view(-1), self.W_dec, offsets=offsets,
            per_sample_weights=top_acts.view(-1), mode="sum")
        return res + self.b_dec

    def forward(self, x):
        return self.decode(*self.encode(x))


LOCAL_RAW = os.path.join(REPO, "generate-responses/generate-responses/results/vars/raw")


def raw_path(model, layer):
    """Prefer a locally re-extracted *.regen.pkl. gemma L18's copy on HF is the incomplete
    139,460-row one; the regenerated file has all 140,800 rows, aligned with the other layers."""
    for d in (LOCAL_RAW, RAW_DIR):
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if re.fullmatch(rf"activations_{re.escape(model)}_\d+_{layer}\.regen\.pkl", fn):
                return os.path.join(d, fn)
    for fn in os.listdir(RAW_DIR):
        if re.fullmatch(rf"activations_{re.escape(model)}_\d+_{layer}\.pkl", fn):
            return os.path.join(RAW_DIR, fn)
    return None


def dedup_mask(texts, keys):
    """Keep the FIRST occurrence of each sentence within a CoT; drop later exact repeats.

    Scoped per-pmcid, not globally: a sentence two different cases both happen to produce is
    legitimate, whereas the same sentence 852 times inside one CoT is a loop.
    """
    seen = defaultdict(set)
    keep = np.zeros(len(texts), dtype=bool)
    for i, (t, (pm, _si)) in enumerate(zip(texts, keys)):
        if t not in seen[pm]:
            seen[pm].add(t)
            keep[i] = True
    return keep


def train_sae(X_train, k, device, seed=0):
    """Reproduces clustering_sae_topk: same LR law, loss, batching, patience, decoder renorm.

    Deviations, both deliberate and neither changes the objective:
      - the dataset lives on the GPU (the original shipped every minibatch across PCIe);
      - the shuffle is seeded, so a retrain is reproducible (the original was not).
    """
    torch.manual_seed(seed)
    n, d = X_train.shape
    sae = SAE(d, k, k=TOPK).to(device)
    lr = 2e-4 / (k / (2 ** 14)) ** 0.5
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    batch = min(512, n)
    best, patience, bad, best_state = float("inf"), 10, 0, None
    g = torch.Generator(device=device).manual_seed(seed)

    for epoch in range(300):
        idx = torch.randperm(n, device=device, generator=g)
        sae.train()
        total, seen = 0.0, 0
        for i in range(0, n, batch):
            bx = X_train[idx[i:i + batch]]
            # The loss is a variance-normalised MSE, and variance is UNDEFINED for a single
            # sample: bx - bx.mean(0) == 0, so the denominator is 0 and the loss is NaN. This
            # bites whenever n_train % batch == 1 (it killed all 6 llama-8b layers, n=58,369).
            # The same latent bug is in utils/clustering_methods.py; it just never hit a corpus
            # of that size. Dropping one row out of tens of thousands is numerically irrelevant.
            if bx.shape[0] < 2:
                continue
            pred = sae(bx)
            err = pred - bx
            loss = (err ** 2).sum() / ((bx - bx.mean(dim=0, keepdim=True)) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.set_decoder_norm_to_unit_norm()
            total += loss.item() * bx.shape[0]
            seen += bx.shape[0]
        avg = total / seen
        if not np.isfinite(avg):
            raise ValueError(f"non-finite loss at epoch {epoch}")
        if avg < best:
            best, bad = avg, 0
            best_state = {n_: p.detach().clone() for n_, p in
                          (("encoder_weight", sae.encoder.weight.data),
                           ("encoder_bias", sae.encoder.bias.data),
                           ("decoder_weight", sae.W_dec.data),
                           ("b_dec", sae.b_dec.data))}
        else:
            bad += 1
            if bad >= patience:
                break

    sae.encoder.weight.data = best_state["encoder_weight"]
    sae.encoder.bias.data = best_state["encoder_bias"]
    sae.W_dec.data = best_state["decoder_weight"]
    sae.b_dec.data = best_state["b_dec"]
    return sae, best


@torch.no_grad()
def encode_all(sae, X, device, chunk=65536):
    """Raw dense pre-activations encoder(x - b_dec): no ReLU, no TopK. fp64 accumulate."""
    W = sae.encoder.weight.data.double()
    be = sae.encoder.bias.data.double()
    bd = sae.b_dec.data.double()
    out = torch.empty((X.shape[0], W.shape[0]), dtype=torch.float32, device="cpu")
    for s in range(0, X.shape[0], chunk):
        xc = X[s:s + chunk].double() - bd
        out[s:s + chunk] = (xc @ W.T + be).float().cpu()
    return out.numpy()


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

    # The raw and centered pkls were produced by SEPARATE extraction runs, and the shuffle is
    # unseeded -- so for the 5 legacy models the raw pkl is a PERMUTATION of the centered one
    # (same keys, different order; gpt-oss's raw is also a strict subset, missing 3 cases).
    # Reorder raw into the canonical centered-pkl key order, so these de-duped outputs stay
    # row-aligned with the phase-1 outputs and old/new taxonomies can be compared positionally.
    from encode_latents import find_act
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
        # the same key must carry the same sentence in both files, or they are not the same run
        ct = {k: t for k, t in zip(cen_keys, cen_texts)}
        assert all(ct[k] == t for k, t in zip(keys, texts)), \
            f"{model} L{layer}: raw and centered disagree on sentence text for the same key"

    N, d = raw.shape
    keep = dedup_mask(texts, keys)
    n_train = int(keep.sum())

    # Recompute the mean on the DEDUPED rows, then center + L2-normalise every row with it.
    mean_new = raw[keep].mean(axis=0, keepdims=True).astype(np.float32)
    X = raw.astype(np.float32, copy=True)
    X -= mean_new
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
        labels = encode_all(sae, Xtr, device).argmax(axis=1)   # over the TRAINING rows
        Y = encode_all(sae, Xg, device)                        # every row, for annotation
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
            "n_train": n_train, "n_all": N, "dedup": "within_cot_first_occurrence",
        }
        torch.save(ck, os.path.join(
            SAE_OUT, f"sae_{model}_layer{layer}_clusters{k}_dedup.pt"))
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
                 in_training=keep)          # False = a duplicate row, excluded from SAE training
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
            f"annotated_responses_{model}_layer{layer}_clusters{k}_dedup.json"), recs)

    # how much of each taxonomy item is still duplicate rows, for comparison with the old SAEs
    cnt = defaultdict(int)
    for t in texts:
        cnt[t] += 1
    isdup = np.array([cnt[t] > 1 for t in texts])
    loopiness = {str(k): [round(float(isdup[argmaxes[k] == j].mean()), 4)
                          if (argmaxes[k] == j).sum() else None for j in range(k)]
                 for k in CLUSTERS}

    json_atomic(os.path.join(ldir, "meta.json"), {
        "model": model, "layer": layer, "status": "ok", "N": int(N), "d": int(d),
        "n_train": int(n_train), "n_dropped_duplicates": int(N - n_train),
        "dedup": "within_cot_first_occurrence (1-word sentences KEPT)",
        "mean_vector": "recomputed on deduped rows",
        "clusters": CLUSTERS, "loss": {str(k): losses[k] for k in CLUSTERS},
        "dup_frac_per_latent": loopiness, "secs": round(secs, 1),
        "latent_semantics": "raw SAE pre-activation encoder(x - b_dec); no ReLU, no TopK, signed",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = []
    for m in (args.models or MODELS):
        for L in model_layers(m):
            if args.layers and L not in args.layers:
                continue
            if raw_path(m, L) is None:
                continue
            if os.path.exists(os.path.join(SIDECAR_OUT, m, f"layer{L}", "meta.json")):
                continue
            jobs.append((m, L))
    print(f"{len(jobs)} (model,layer) jobs to retrain\n", flush=True)

    ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run, m, L, i % torch.cuda.device_count()): (m, L)
                for i, (m, L) in enumerate(jobs)}
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except Exception as e:  # one bad layer must not abort the other 53
                m, L = futs[fu]
                print(f"  [FAIL] {m} L{L}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                continue
            if r["status"] == "ok":
                ok += 1
                print(f"  [ok] {r['model']} L{r['layer']}  N={r['N']:,} "
                      f"train={r['n_train']:,} dropped={r['dropped']:,} "
                      f"({r['dropped']/r['N']*100:.0f}%)  {r['secs']}s", flush=True)
            else:
                print(f"  [FAIL] {r['model']} L{r['layer']}: {r['status']}", flush=True)
    print(f"\n{ok}/{len(jobs)} ok")


if __name__ == "__main__":
    main()
