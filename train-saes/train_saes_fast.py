#!/usr/bin/env python3
"""Fast, multi-GPU replacement for ``train_clustering.py --clustering_methods sae_topk``.

The algorithm is byte-for-byte the same TopK-SAE recipe as
``utils.clustering_methods.clustering_sae_topk`` (same init, LR law, Adam, loss,
decoder renorm, early stopping, checkpoint schema). Only the *plumbing* changes:

* the activation matrix lives on the GPU for the whole job instead of being
  copied batch-by-batch from CPU,
* ``torch.cuda.empty_cache()`` is not called inside the training loop (it forced a
  device sync on every one of the ~36k steps),
* the per-epoch loss is accumulated on-device (no ``.item()`` sync per step),
* no LLM is loaded (the original script loads the full 32B model only to read a
  cached pkl),
* (model, layer) jobs are sharded across every visible GPU, several workers deep.

Layout matches the original exactly:
``results/vars/saes/sae_{model_id}_layer{layer}_clusters{k}.pt``

Usage
-----
    python train_saes_fast.py --act-dir <dir with activations_*.pkl> \
        --out-dir results/vars/saes --gpus 0 1 2 --workers-per-gpu 2
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
import traceback
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn as nn

CLUSTERS = [10, 12, 14, 16, 18, 20]
TOPK = 3
MAX_EPOCHS = 300
BATCH_SIZE = 512
PATIENCE = 10


# --------------------------------------------------------------------------- SAE (verbatim)

class SAE(nn.Module):
    """Identical to ``utils.sae.SAE``; duplicated so this script has no repo imports."""

    def __init__(self, d_in: int, num_latents: int, k: int = 1):
        super().__init__()
        self.encoder = nn.Linear(d_in, num_latents, bias=True)
        self.encoder.bias.data.zero_()
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.k = k
        self.set_decoder_norm_to_unit_norm()

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self) -> None:
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + 1e-5

    def encode(self, x):
        forward = self.encoder(x - self.b_dec)
        return forward.topk(self.k, dim=-1)

    def decode(self, top_acts, top_indices):
        batch_size = top_indices.shape[0]
        offsets = torch.arange(0, batch_size, device=top_indices.device) * self.k
        res = nn.functional.embedding_bag(
            top_indices.view(-1), self.W_dec, offsets=offsets,
            per_sample_weights=top_acts.view(-1), mode="sum")
        return res + self.b_dec

    def forward(self, x):
        return self.decode(*self.encode(x))


# --------------------------------------------------------------------------- training

def train_one_sae(X: torch.Tensor, n_clusters: int, device: torch.device, seed: int, log=print):
    """Train a single TopK SAE on device-resident ``X`` (N, d). Returns (sae, best_loss, labels)."""
    n_samples, input_dim = X.shape
    # Same seeding contract as utils.clustering_methods.clustering_sae_topk.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sae = SAE(input_dim, n_clusters, k=TOPK).to(device)
    lr = 2e-4 / (n_clusters / (2 ** 14)) ** 0.5
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    batch_size = min(BATCH_SIZE, n_samples)

    best_loss, patience_counter, best_state = float("inf"), 0, None
    for epoch in range(MAX_EPOCHS):
        indices = torch.randperm(n_samples, device=device)  # seeded above
        sae.train()
        total_loss = torch.zeros((), device=device)
        for i in range(0, n_samples, batch_size):
            bidx = indices[i:i + batch_size]
            batch_X = X[bidx]
            predicted = sae(batch_X)
            error = predicted - batch_X
            loss = (error ** 2).sum()
            loss = loss / ((batch_X - batch_X.mean(dim=0, keepdim=True)) ** 2).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sae.set_decoder_norm_to_unit_norm()
            total_loss += loss.detach() * bidx.numel()

        avg_loss = float(total_loss.item()) / n_samples
        if not np.isfinite(avg_loss):
            raise ValueError(f"Total loss is nan at epoch {epoch}")
        if (epoch + 1) % 50 == 0:
            log(f"      k={n_clusters} epoch {epoch + 1}/{MAX_EPOCHS} loss {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {
                "encoder_weight": sae.encoder.weight.data.clone(),
                "encoder_bias": sae.encoder.bias.data.clone(),
                "decoder_weight": sae.W_dec.data.clone(),
                "b_dec": sae.b_dec.data.clone(),
            }
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log(f"      k={n_clusters} early stop at epoch {epoch + 1}")
                break

    if best_state:
        sae.encoder.weight.data = best_state["encoder_weight"]
        sae.encoder.bias.data = best_state["encoder_bias"]
        sae.W_dec.data = best_state["decoder_weight"]
        sae.b_dec.data = best_state["b_dec"]

    sae.eval()
    with torch.no_grad():
        labels = torch.empty(n_samples, dtype=torch.long, device=device)
        for i in range(0, n_samples, 65536):
            j = min(i + 65536, n_samples)
            labels[i:j] = sae.encoder(X[i:j] - sae.b_dec).argmax(dim=1)
    return sae, best_loss, labels.cpu().numpy()


# --------------------------------------------------------------------------- job

def center_and_normalize(X: np.ndarray, mean_vector: np.ndarray) -> np.ndarray:
    """Same math as ``utils.utils.center_and_normalize_activations`` (vectorised)."""
    out = X - mean_vector.reshape(1, -1)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def run_job(act_path: str, model_id: str, layer: int, out_dir: str, device: torch.device,
            clusters, seed: int = 42, log=print) -> dict:
    t0 = time.time()
    with open(act_path, "rb") as f:
        X, _texts, _keys, mean_vector = pickle.load(f)
    X = np.asarray(X, dtype=np.float32)

    idx = np.random.RandomState(0).choice(X.shape[0], size=min(1000, X.shape[0]), replace=False)
    median_norm = float(np.median(np.linalg.norm(X[idx].astype(np.float64), axis=1)))
    if abs(median_norm - 1.0) >= 1e-2:
        X = center_and_normalize(X, np.asarray(mean_vector, dtype=np.float32))
    log(f"    {model_id} L{layer}: N={X.shape[0]} d={X.shape[1]} (raw median L2={median_norm:.2f})")

    Xg = torch.from_numpy(X).to(device)
    del X

    for k in clusters:
        dest = os.path.join(out_dir, f"sae_{model_id}_layer{layer}_clusters{k}.pt")
        if os.path.exists(dest):
            log(f"    {model_id} L{layer} k={k}: exists, skip")
            continue
        t1 = time.time()
        sae, best_loss, labels = train_one_sae(Xg, k, device, seed + k, log=log)
        centers = sae.W_dec.data.detach().cpu().numpy()
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        tmp = dest + f".tmp{os.getpid()}"
        torch.save({
            "encoder_weight": sae.encoder.weight.data.clone().cpu(),
            "encoder_bias": sae.encoder.bias.data.clone().cpu(),
            "decoder_weight": sae.W_dec.data.clone().cpu(),
            "b_dec": sae.b_dec.data.clone().cpu(),
            "input_dim": int(Xg.shape[1]),
            "num_latents": k,
            "topk": TOPK,
            "loss": best_loss,
            "cluster_labels": labels,
            "cluster_centers": centers,
            "mean_vector": np.asarray(mean_vector, dtype=np.float32),
            "seed": seed + k,
        }, tmp)
        os.replace(tmp, dest)
        log(f"    {model_id} L{layer} k={k}: loss={best_loss:.6f} ({time.time() - t1:.0f}s)")
        del sae
    del Xg
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"model": model_id, "layer": layer, "secs": round(time.time() - t0, 1)}


def _worker(rank: int, gpu, jobs, out_dir: str, clusters, seed: int, log_dir: str) -> None:
    """``gpu`` is a CUDA device index, or None to run this worker on CPU."""
    if gpu is None:
        device = torch.device("cpu")
        tag = "cpu"
    else:
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
        tag = f"gpu{gpu}"
    logf = open(os.path.join(log_dir, f"worker{rank}_{tag}.log"), "a", buffering=1)

    def log(msg):
        print(f"[w{rank}/{tag}] {msg}", file=logf, flush=True)

    for act_path, model_id, layer in jobs:
        try:
            r = run_job(act_path, model_id, layer, out_dir, device, clusters, seed=seed, log=log)
            log(f"  DONE {r}")
        except Exception:
            log(f"  FAIL {model_id} L{layer}\n{traceback.format_exc()}")
    logf.close()


# --------------------------------------------------------------------------- main

def discover(act_dir: str, models=None):
    jobs = []
    for fn in sorted(os.listdir(act_dir)):
        m = re.fullmatch(r"activations_(.+)_(\d+)_(\d+)\.pkl", fn)
        if not m:
            continue
        model_id, layer = m.group(1), int(m.group(3))
        if models and model_id not in models:
            continue
        jobs.append((os.path.join(act_dir, fn), model_id, layer))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir", required=True)
    ap.add_argument("--out-dir", default="results/vars/saes")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--clusters", type=int, nargs="*", default=CLUSTERS)
    ap.add_argument("--gpus", type=int, nargs="*", default=None,
                    help="CUDA device indices to use. Default: every visible GPU. "
                         "Pass --gpus with no values, or run on a machine with no CUDA, "
                         "to train on CPU instead.")
    ap.add_argument("--workers-per-gpu", type=int, default=2,
                    help="Concurrent training processes per device. Lower this if a device "
                         "runs out of memory (each worker holds one activation matrix).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base RNG seed; the SAE for cluster size k uses seed + k")
    ap.add_argument("--log-dir", default="results/logs/sae_train")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Adapt to whatever the machine has: an explicit --gpus list, else every visible GPU,
    # else CPU. `None` in the device list means "this worker runs on CPU".
    available = list(range(torch.cuda.device_count()))
    if args.gpus is None:
        devices = available or [None]
    else:
        bad = [g for g in args.gpus if g not in available]
        if bad:
            print(f"requested GPU(s) {bad} not visible (found {available or 'none'})",
                  file=sys.stderr)
            sys.exit(1)
        devices = args.gpus or [None]
    if devices == [None]:
        print("no CUDA device in use — training on CPU (much slower)", file=sys.stderr)

    jobs = discover(args.act_dir, args.models)
    todo = [j for j in jobs
            if not all(os.path.exists(os.path.join(
                args.out_dir, f"sae_{j[1]}_layer{j[2]}_clusters{k}.pt")) for k in args.clusters)]
    print(f"{len(jobs)} (model,layer) jobs found, {len(todo)} incomplete")
    if not todo:
        return

    n_workers = min(len(devices) * args.workers_per_gpu, len(todo))
    shards = [todo[i::n_workers] for i in range(n_workers)]
    ctx = get_context("spawn")
    procs = []
    for r in range(n_workers):
        if not shards[r]:
            continue
        p = ctx.Process(target=_worker,
                        args=(r, devices[r % len(devices)], shards[r], args.out_dir,
                              args.clusters, args.seed, args.log_dir))
        p.start()
        procs.append(p)
    where = "CPU" if devices == [None] else f"GPUs {devices}"
    print(f"launched {len(procs)} workers over {where}; logs in {args.log_dir}")
    for p in procs:
        p.join()

    done = sum(1 for j in jobs
               if all(os.path.exists(os.path.join(
                   args.out_dir, f"sae_{j[1]}_layer{j[2]}_clusters{k}.pt")) for k in args.clusters))
    print(f"complete: {done}/{len(jobs)} (model,layer) jobs")


if __name__ == "__main__":
    main()
