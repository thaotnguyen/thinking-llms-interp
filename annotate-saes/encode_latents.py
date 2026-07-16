#!/usr/bin/env python3
"""Encode every CoT sentence with all 36 of a model's TopK SAEs, keeping every latent value.

Unlike generate-responses/annotate_thinking.py, this does NOT run the LLM. The activation pkls
already hold the per-sentence mean-pooled, mean-centered, L2-normalised vectors -- which is exactly
the space the SAE consumes -- so encoding is just one (N,d)@(d,K) GEMM per SAE.

Latent values written are the RAW SAE outputs: encoder(x - b_dec), i.e. dense pre-activations.
The SAE has no ReLU, so they are signed, and no TopK sparsification is applied.

Run with /usr/bin/python3 (has numpy; torch is not needed and is not installed).
"""
import argparse
import io
import json
import os
import pickle
import re
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT = os.environ.get(
    "SAE_SNAPSHOT",
    "/workspace/.hf_home/hub/datasets--matthewshu--medcasereasoning-cot-activations/"
    "snapshots/620eeda9485ca37bfbb91ee620f6c4931df7f451",
)
# Real activation pkls live under a doubled path (rsync artifact); HF ones land in the snapshot.
ACT_DIRS = [
    os.path.join(REPO, "generate-responses/generate-responses/results/vars"),
    os.path.join(SNAPSHOT, "activations/centered_normalised"),
]
SAE_DIR = os.path.join(SNAPSHOT, "saes")
RESP_DIR = os.path.join(SNAPSHOT, "responses")
TEXT_OUT = os.path.join(REPO, "generate-responses/results/vars")
SIDECAR_OUT = os.path.join(REPO, "annotate-saes/results/vars/latents")

CLUSTERS = [10, 12, 14, 16, 18, 20]
MODELS = [
    "deepseek-r1-distill-llama-8b", "deepseek-r1-distill-qwen-14b", "huatuogpt-o1-8b",
    "gpt-oss-20b", "qwq-32b", "qwen3.6-27b", "gemma-4-31b-it",
    "ministral-3-14b-reasoning-2512", "glm-4.7-flash",
]

# Consumers of the new annotation format.
SECTION_RE = re.compile(r'\["([^"\]]+)"\](.*?)\["end-section"\]', re.DOTALL)
PAIR_RE = re.compile(r"(-?[\d.]+(?:[eE][-+]?\d+)?):idx(\d+)")


# --------------------------------------------------------------------------- checkpoint loading

# torch storage class name -> numpy dtype. The .pt payloads are plain little-endian buffers.
_DTYPE_BY_STORAGE = {
    "FloatStorage": np.dtype("<f4"), "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"), "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"), "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("i1"), "ByteStorage": np.dtype("u1"),
    "BoolStorage": np.dtype("?"),
}


class _StorageType:
    """Stand-in for torch.FloatStorage etc., so persistent_load can recover the dtype."""

    def __init__(self, name):
        self.name = name


def _torch_stub(*_a, **_k):
    return None


class _CkptUnpickler(pickle.Unpickler):
    """Load a torch .pt's data.pkl without torch: rebuild tensors straight from the zip buffers."""

    def __init__(self, buf, zf, prefix):
        super().__init__(buf)
        self.zf, self.prefix, self.cache = zf, prefix, {}

    def find_class(self, module, name):
        if module.startswith("torch") and name.endswith("Storage"):
            return _StorageType(name)
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            def rebuild(storage, offset, size, stride, *_rest):
                dtype, key = storage
                if key not in self.cache:
                    self.cache[key] = np.frombuffer(
                        self.zf.read(f"{self.prefix}data/{key}"), dtype=dtype)
                flat = self.cache[key]
                n = int(np.prod(size)) if len(size) else 1
                return flat[offset:offset + n].reshape(tuple(size)).copy()
            return rebuild
        if module.startswith("torch"):
            return _torch_stub
        try:
            return getattr(__import__(module, fromlist=[name]), name)
        except Exception:
            return _torch_stub

    def persistent_load(self, pid):
        # pid = ('storage', <StorageType>, key, location, numel)
        _tag, stype, key, _loc, _numel = pid
        dtype = _DTYPE_BY_STORAGE.get(getattr(stype, "name", ""), np.dtype("<f4"))
        return (dtype, key)


def _load_ckpt(path):
    z = zipfile.ZipFile(path)
    pkl_name = next(n for n in z.namelist() if n.endswith("data.pkl"))
    prefix = pkl_name[: -len("data.pkl")]
    return _CkptUnpickler(io.BytesIO(z.read(pkl_name)), z, prefix).load()


# --------------------------------------------------------------------------- io helpers

def sae_path(model, layer, k):
    return os.path.join(SAE_DIR, f"sae_{model}_layer{layer}_clusters{k}.pt")


def find_act(model, layer, allow_regen=True):
    """Prefer a *.regen.pkl if one exists (gemma layer 18 was re-extracted; see README)."""
    base = None
    for d in ACT_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if allow_regen and re.fullmatch(
                    rf"activations_{re.escape(model)}_\d+_{layer}\.regen\.pkl", fn):
                return os.path.join(d, fn)
            if base is None and re.fullmatch(
                    rf"activations_{re.escape(model)}_\d+_{layer}\.pkl", fn):
                base = os.path.join(d, fn)
    return base


def model_layers(model):
    """Layers for a model = whatever layers its SAEs exist for."""
    ls = set()
    for fn in os.listdir(SAE_DIR):
        m = re.fullmatch(rf"sae_{re.escape(model)}_layer(\d+)_clusters(\d+)\.pt", fn)
        if m and int(m.group(2)) in CLUSTERS:
            ls.add(int(m.group(1)))
    return sorted(ls)


def atomic_write(path, write_fn):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def savez_atomic(path, **arrays):
    """np.savez appends '.npz' to any path that lacks it, which would defeat the temp-file
    rename. Hand it an open handle instead so the name is used verbatim."""
    def _w(tmp):
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
    atomic_write(path, _w)


def json_atomic(path, obj):
    def _w(tmp):
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=2)
    atomic_write(path, _w)


def fmt_row(vals):
    """K raw latent values -> 'v0:idx0,v1:idx1,...'. %.9g round-trips float32 exactly."""
    return ",".join("%.9g:idx%d" % (v, i) for i, v in enumerate(vals))


# --------------------------------------------------------------------------- core

def encode(X, W, b_enc, b_dec, chunk=32768):
    """Reference math, verbatim: encoder(X - b_dec). fp64 accumulate, no b_dec folding.

    Folding b_dec into the bias (Wx + (b_enc - W@b_dec)) is algebraically identical but NOT
    float-identical, and the difference can flip a near-tie argmax. Keep the op order.
    """
    N = X.shape[0]
    out = np.empty((N, W.shape[0]), dtype=np.float32)
    W64, be64, bd64 = W.astype(np.float64), b_enc.astype(np.float64), b_dec.astype(np.float64)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        xc = X[s:e].astype(np.float64) - bd64
        out[s:e] = (xc @ W64.T + be64).astype(np.float32)
    return out


def run_job(model, layer, force=False):
    t0 = time.time()
    apath = find_act(model, layer)
    if apath is None:
        return {"model": model, "layer": layer, "status": "missing_pkl"}
    missing = [k for k in CLUSTERS if not os.path.exists(sae_path(model, layer, k))]
    if missing:
        return {"model": model, "layer": layer, "status": "missing_saes", "missing_k": missing}

    meta_path = os.path.join(SIDECAR_OUT, model, f"layer{layer}", "meta.json")
    src_stamp = {"pkl": os.path.getsize(apath), "pkl_mtime": int(os.path.getmtime(apath))}
    if not force and os.path.exists(meta_path):
        try:
            old = json.load(open(meta_path))
            if old.get("src_stamp") == src_stamp and old.get("status") == "ok":
                return {"model": model, "layer": layer, "status": "cached"}
        except Exception:
            pass

    with open(apath, "rb") as f:
        X, texts, keys, meanvec = pickle.load(f)
    N, d = X.shape

    # A re-extracted layer (gemma L18) is a strict SUPERSET of the rows its SAE was trained on:
    # the original extraction dropped 6 whole CoTs to a transient OOM. The SAE stays a valid
    # encoder, so we still annotate every row -- but cluster_labels no longer lines up with the
    # pkl, so G3/G4 are checked on the OVERLAP, joined by (pmcid, sentence_idx). Row ORDER would
    # not have survived a re-extraction in general (the shuffle is unseeded), so never join
    # positionally here.
    overlap = None
    if apath.endswith(".regen.pkl"):
        base = find_act(model, layer, allow_regen=False)
        if base:
            with open(base, "rb") as f:
                _bX, _bt, base_keys, _bm = pickle.load(f)
            pos = {k: i for i, k in enumerate(keys)}
            rows = [(pos[k], j) for j, k in enumerate(base_keys) if k in pos]
            overlap = (np.array([r for r, _ in rows]), np.array([j for _, j in rows]))

    norms = np.linalg.norm(X[:: max(1, N // 2000)].astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        return {"model": model, "layer": layer, "status": "not_unit_norm",
                "median_norm": float(np.median(norms))}

    gates, latents, argmaxes, params = {}, {}, {}, {}
    for k in CLUSTERS:
        ck = _load_ckpt(sae_path(model, layer, k))
        g = {}
        # G1 shape
        g["dim_ok"] = int(ck["input_dim"]) == d and int(ck["num_latents"]) == k
        if not g["dim_ok"]:
            return {"model": model, "layer": layer, "status": "shape_mismatch", "k": k,
                    "input_dim": int(ck["input_dim"]), "d": d}
        # G2 provenance: bitwise-identical mean vector proves SAE and pkl share an extraction run
        mv = ck.get("mean_vector")
        g["meanvec_match"] = bool(mv is not None and np.array_equal(
            np.asarray(mv, dtype=np.float32).reshape(1, -1), meanvec.astype(np.float32)))
        # G3 row count
        cl = np.asarray(ck["cluster_labels"]).ravel()
        g["n_labels"] = int(cl.shape[0])
        g["rowcount_match"] = g["n_labels"] == N
        if not g["rowcount_match"] and overlap is not None and g["n_labels"] == len(overlap[0]):
            g["rowcount_match"] = True          # superset pkl; validated on the overlap below
            g["validated_on"] = "overlap"

        Y = encode(X, ck["encoder_weight"], ck["encoder_bias"], ck["b_dec"])
        am = Y.argmax(axis=1)

        # G4 argmax agreement vs the checkpoint's own training-time labels.
        #
        # A handful of rows legitimately disagree: the reference labels were produced by an fp32
        # (likely TF32) GPU matmul, so it breaks near-ties differently than we do. Those are real
        # ties, not errors -- fp32 and fp64 recompute agree with each other on every row here.
        # So the tie tolerance is scaled to the activation magnitude rather than absolute: a
        # disagreement is acceptable only where the top two latents are within 0.1% of the typical
        # top-1 activation. A genuinely mispaired SAE/pkl gives agreement near 1/K with margins of
        # order the full activation scale, which this still rejects decisively.
        if g["rowcount_match"]:
            if g.get("validated_on") == "overlap":
                # Re-extracted layer: compare only the rows the SAE actually saw. The bar is
                # lower (0.99, not 0.999) because a re-extraction is not bit-reproducible --
                # GPU kernel non-determinism flips a fraction of a percent of near-tie argmaxes.
                # A WRONG space would still land near chance (1/K), which this rejects.
                rows, lab_idx = overlap
                am_ov, cl_ov = Y[rows].argmax(axis=1), cl[lab_idx]
                g["agreement"] = float((am_ov == cl_ov).mean())
                g["n_overlap"] = int(len(rows))
                g["max_tie_margin"] = None
                g["n_disagree"] = int((am_ov != cl_ov).sum())
                g["g4_ok"] = g["agreement"] >= 0.99
                gates[k] = g
                latents[k], argmaxes[k], params[k] = Y, am, ck
                continue
            agree = am == cl
            g["agreement"] = float(agree.mean())
            part_all = np.partition(Y, -2, axis=1)
            scale = float(np.median(part_all[:, -1]))
            tol = max(1e-3 * abs(scale), 1e-5)
            g["tie_tolerance"] = tol
            bad = ~agree
            if bad.any():
                p = np.partition(Y[bad], -2, axis=1)
                g["max_tie_margin"] = float((p[:, -1] - p[:, -2]).max())
                g["n_disagree"] = int(bad.sum())
            else:
                g["max_tie_margin"] = 0.0
                g["n_disagree"] = 0
            g["g4_ok"] = g["agreement"] >= 0.999 and g["max_tie_margin"] < tol
        else:
            g["agreement"] = None
            g["max_tie_margin"] = None
            g["g4_ok"] = None  # gemma L18 post-regen: validated on the overlap separately

        gates[k] = g
        latents[k] = Y
        argmaxes[k] = am
        params[k] = ck

    hard_fail = [k for k in CLUSTERS if gates[k]["g4_ok"] is False]
    if hard_fail:
        return {"model": model, "layer": layer, "status": "gate_failed",
                "failed_k": hard_fail, "gates": gates}

    # fp32-vs-fp64 precision probe on the first SAE
    ck0 = params[CLUSTERS[0]]
    probe = X[: min(8192, N)]
    y32 = (probe - ck0["b_dec"]) @ ck0["encoder_weight"].T + ck0["encoder_bias"]
    y64 = encode(probe, ck0["encoder_weight"], ck0["encoder_bias"], ck0["b_dec"])
    fp_diff = float(np.abs(y64.astype(np.float64) - y32.astype(np.float64)).max())

    write_outputs(model, layer, X, texts, keys, latents, argmaxes, params, gates, src_stamp,
                  fp_diff, time.time() - t0)
    return {"model": model, "layer": layer, "status": "ok", "N": N, "d": d,
            "agreement": {k: gates[k]["agreement"] for k in CLUSTERS},
            "secs": round(time.time() - t0, 1)}


def write_outputs(model, layer, X, texts, keys, latents, argmaxes, params, gates, src_stamp,
                  fp_diff, secs):
    N = X.shape[0]
    # --- sidecar (binary, exact) ---
    ldir = os.path.join(SIDECAR_OUT, model, f"layer{layer}")
    os.makedirs(ldir, exist_ok=True)

    pmcids = np.array([k[0] for k in keys], dtype="S32")
    sidx = np.array([k[1] for k in keys], dtype=np.int32)

    savez_atomic(os.path.join(ldir, "latents.npz"),
                 **{f"k{k}": latents[k] for k in CLUSTERS},
                 **{f"argmax_k{k}": argmaxes[k].astype(np.int8) for k in CLUSTERS})
    savez_atomic(os.path.join(ldir, "index.npz"), pmcid=pmcids, sentence_idx=sidx)
    savez_atomic(os.path.join(ldir, "sae_params.npz"), **{
        f"{fld}_k{k}": np.asarray(params[k][fld])
        for k in CLUSTERS
        for fld in ("encoder_weight", "encoder_bias", "b_dec", "decoder_weight",
                    "cluster_centers", "mean_vector")})

    # per-case profiles: sentences differ across models but cases (pmcid) are shared
    uniq, inv = np.unique(pmcids, return_inverse=True)
    prof = {"pmcids": uniq}
    for k in CLUSTERS:
        cnt = np.bincount(inv, minlength=len(uniq)).astype(np.float64)[:, None]
        means = np.zeros((len(uniq), k), dtype=np.float64)
        np.add.at(means, inv, latents[k].astype(np.float64))
        prof[f"mean_k{k}"] = (means / cnt).astype(np.float32)
        hist = np.zeros((len(uniq), k), dtype=np.int32)
        np.add.at(hist, (inv, argmaxes[k]), 1)
        prof[f"argmax_hist_k{k}"] = hist
    savez_atomic(os.path.join(ldir, "case_profiles.npz"), **prof)

    # --- primary: one annotated text file per (model, layer, K) ---
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
        records = []
        for pm in order:
            parts = []
            for _si, row in by_case[pm]:
                parts.append('["%s"]%s["end-section"]' % (fmt_row(Y[row]), texts[row]))
            src = resp_meta.get(pm, {})
            records.append({
                "pmcid": pm,
                "question_id": src.get("question_id"),
                "category": src.get("category", "diagnosis"),
                "dataset_name": src.get("dataset_name"),
                "annotated_thinking": "".join(parts).strip(),
            })
        out = os.path.join(TEXT_OUT, f"annotated_responses_{model}_layer{layer}_clusters{k}.json")
        json_atomic(out, records)

    meta = {
        "model": model, "layer": layer, "status": "ok", "N": int(N), "d": int(X.shape[1]),
        "clusters": CLUSTERS, "src_stamp": src_stamp, "fp64_vs_fp32_max_diff": fp_diff,
        "n_cases": len(order), "secs": round(secs, 1),
        "gates": {str(k): gates[k] for k in CLUSTERS},
        "latent_semantics": "raw SAE pre-activation encoder(x - b_dec); no ReLU, no TopK, signed",
    }
    json_atomic(os.path.join(ldir, "meta.json"), meta)


# --------------------------------------------------------------------------- selftest / scan

def selftest():
    rng = np.random.default_rng(0)
    N, d, k = 5000, 256, 10
    X = rng.standard_normal((N, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    W = rng.standard_normal((k, d)).astype(np.float32)
    be = rng.standard_normal(k).astype(np.float32)
    bd = rng.standard_normal(d).astype(np.float32) * 0.01

    ref = ((X - bd) @ W.T + be).astype(np.float32)          # fp32 reference (as trained)
    got = encode(X, W, be, bd)                               # fp64 accumulate
    assert np.abs(got - ref).max() < 1e-4, np.abs(got - ref).max()
    assert (got.argmax(1) == ref.argmax(1)).mean() >= 0.999
    print("  fp64 path agrees with fp32 reference: OK (max diff %.2e)" % np.abs(got - ref).max())

    folded = (X @ W.T + (be - W @ bd)).astype(np.float32)     # the trap
    assert not np.array_equal(folded, ref), "folding should NOT be bit-identical"
    print("  b_dec folding differs bitwise from reference: OK (max diff %.2e)"
          % np.abs(folded - ref).max())

    vals = rng.standard_normal(20).astype(np.float32)
    back = np.array([float(p.split(":")[0]) for p in fmt_row(vals).split(",")], dtype=np.float32)
    assert np.array_equal(back, vals), "%.9g must round-trip float32 exactly"
    print("  %.9g float32 round-trip is exact: OK")

    tag = fmt_row(vals)
    pairs = PAIR_RE.findall(tag)
    assert [int(i) for _, i in pairs] == list(range(20))
    assert np.array_equal(np.array([float(v) for v, _ in pairs], dtype=np.float32), vals)
    print("  annotation tag parses back exactly: OK")
    print("selftest PASSED")


def scan():
    rows = []
    for m in MODELS:
        for L in model_layers(m):
            a = find_act(m, L)
            miss = [k for k in CLUSTERS if not os.path.exists(sae_path(m, L, k))]
            st = "ready" if (a and not miss) else ("missing_pkl" if not a else "missing_saes")
            rows.append({"model": m, "layer": L, "status": st, "pkl": a, "missing_k": miss})
    ready = [r for r in rows if r["status"] == "ready"]
    by_model = defaultdict(lambda: [0, 0])
    for r in rows:
        by_model[r["model"]][1] += 1
        if r["status"] == "ready":
            by_model[r["model"]][0] += 1
    for m in MODELS:
        got, tot = by_model[m]
        print(f"  {m:34s} {got}/{tot} layers ready")
    print(f"TOTAL ready (model,layer) jobs: {len(ready)}/{len(rows)}")
    return ready


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.scan:
        scan()
        return

    jobs = [(r["model"], r["layer"]) for r in scan()
            if (args.models is None or r["model"] in args.models)
            and (args.layers is None or r["layer"] in args.layers)]
    if not jobs:
        print("no ready jobs matched")
        return
    print(f"\nrunning {len(jobs)} jobs on {args.workers} workers\n")

    ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, m, L, args.force): (m, L) for m, L in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            if r["status"] in ("ok", "cached"):
                ok += 1
                ag = r.get("agreement")
                extra = ""
                if ag:
                    extra = "  agreement=%s" % ("/".join("%.4f" % ag[k] for k in CLUSTERS))
                print(f"  [{r['status']:6s}] {r['model']} L{r['layer']}{extra}"
                      f"{'  %ss' % r['secs'] if 'secs' in r else ''}")
            else:
                print(f"  [FAIL  ] {r['model']} L{r['layer']}: {r['status']} "
                      f"{ {k: v for k, v in r.items() if k not in ('model','layer','status','gates')} }",
                      file=sys.stderr)
    print(f"\n{ok}/{len(jobs)} jobs ok")


if __name__ == "__main__":
    main()
