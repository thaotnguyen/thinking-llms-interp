#!/usr/bin/env python3
"""
Generate taxonomy titles + descriptions for every SAE latent, using a LOCAL
gpt-oss-120b served by vLLM (OpenAI-compatible endpoint).

Replaces train-saes/generate_titles_trained_clustering.py for this purpose. Two
substantive differences, both of which make it strictly cheaper and safer:

  1. NO subject-model forward passes. The reference `direct` path calls
     utils.load_model() + process_saved_responses() purely to recompute the
     activations it needs to rank examples. We already have every latent value
     in the phase-1/phase-3 sidecars, so ranking is a numpy argsort.

  2. FIXED response parser. utils/clustering.py:573 and
     utils/clustering_batched.py:409 use

         re.search(r"Title:\\s*(.*?)(?:\\n|$)", response)

     which matches the *inner* "Title:" of a markdown "**Title:**", so `\\s*`
     consumes nothing and the trailing "**" lands in the capture. In the 3,604
     existing category records this left 85% of titles prefixed with "**" and
     48% of descriptions empty (literally the string "**"). We parse markdown
     tolerantly and let descriptions span multiple lines.

Everything else is reproduced faithfully from the reference:
  - example ranking: rows whose argmax == latent, ordered by the latent's raw
    encoder value `encoder(x - b_dec)[latent]` descending
    (utils/clustering.py:1118, the sae_topk branch)
  - example sampling: top `description_examples//2`, plus that many sampled at
    random from the remainder, then shuffled
    (generate_titles_trained_clustering.py:120)
  - empty (dead) latents are SKIPPED and get no title, exactly as the reference
    does -- this is why several taxonomies legitimately have fewer than K titles
  - prompt: utils.autograder_prompts.build_cluster_description_prompt, imported
    verbatim, with n_trace_examples=3 full reasoning traces and
    n_categories_examples=5 guidance categories

Deliberate, documented deviations:
  - Sampling is SEEDED (per model/layer/K/set) so a re-run reproduces itself.
    The reference uses the unseeded global `random`.
  - The 3 reasoning traces are sampled once PER MODEL and reused across that
    model's 36 taxonomies, rather than resampled per call. This makes the long
    shared prefix cacheable by vLLM (a large speedup) and keeps every taxonomy
    of a model conditioned on identical context. Each trace is truncated to
    --max-trace-chars to survive the degenerate 166k-char repetition-loop
    responses that exist in this corpus.
  - For the _dedup SAEs, examples are drawn only from rows the dedup SAE was
    actually trained on (index.npz['in_training']), so within-CoT repetition
    loops cannot dominate a latent's example list. The old SAEs were trained on
    every row, so for them we rank over every row (faithful to the reference).

Raw model responses are stored alongside the parsed output, so the parser can be
changed later without re-running a single LLM call.
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import pickle
import random
import re
import sys
import time

import numpy as np
from collections import defaultdict


def stable_seed(*parts):
    """Reproducible across processes. Python's hash() is salted per-process."""
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDECAR = {
    "old": os.path.join(REPO, "annotate-saes/results/vars/latents"),
    "dedup": os.path.join(REPO, "annotate-saes/results/vars/latents_dedup"),
    "nndedup": os.path.join(REPO, "annotate-saes/results/vars/latents_nndedup"),
}
OUT_ROOT = os.path.join(REPO, "annotate-saes/results/vars/titles")

MODELS = [
    "deepseek-r1-distill-llama-8b",
    "deepseek-r1-distill-qwen-14b",
    "huatuogpt-o1-8b",
    "gpt-oss-20b",
    "qwq-32b",
    "qwen3.6-27b",
    "gemma-4-31b-it",
    "ministral-3-14b-reasoning-2512",
    "glm-4.7-flash",
]
KS = [10, 12, 14, 16, 18, 20]


# ---------------------------------------------------------------- prompt import
def _load_prompt_builder():
    """Import utils/autograder_prompts.py by path.

    A plain `from utils.autograder_prompts import ...` triggers utils/__init__.py,
    which imports chat_limiter (absent here and not needed for prompt building).
    """
    path = os.path.join(REPO, "utils/autograder_prompts.py")
    spec = importlib.util.spec_from_file_location("_autograder_prompts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_cluster_description_prompt


build_cluster_description_prompt = _load_prompt_builder()


# ---------------------------------------------------------------- response parse
_TITLE_RE = re.compile(
    r"^[^\S\n]*\**[^\S\n]*Title[^\S\n]*:?[^\S\n]*\**[^\S\n]*(.*?)[^\S\n]*$",
    re.MULTILINE | re.IGNORECASE,
)
_DESC_RE = re.compile(
    r"\**[^\S\n]*Description[^\S\n]*:?[^\S\n]*\**[^\S\n]*(.*)\Z",
    re.DOTALL | re.IGNORECASE,
)


def _tidy(s):
    """Strip markdown bold and the prompt's own [bracket] placeholders."""
    s = s.strip()
    for _ in range(3):
        t = s.strip().strip("*").strip()
        if t.startswith("[") and t.endswith("]"):
            t = t[1:-1]
        if t == s:
            break
        s = t
    return s.strip()


_TITLE_NEXTLINE_RE = re.compile(
    r"\**[^\S\n]*Title[^\S\n]*:?[^\S\n]*\**[^\S\n]*\n+([^\n]+)", re.IGNORECASE
)


def parse_title_desc(response):
    """-> (title, description). Tolerant of markdown; description may be multiline."""
    if not response:
        return "Unnamed Cluster", "No description available"
    m = _TITLE_RE.search(response)
    title = _tidy(m.group(1)) if m else ""
    if not title:
        # model put the title on the line *after* the header
        m = _TITLE_NEXTLINE_RE.search(response)
        title = _tidy(m.group(1)) if m else ""
    m = _DESC_RE.search(response)
    desc = _tidy(m.group(1)) if m else ""
    # A description that swallowed a following "Title:" heading is malformed; cut it.
    desc = re.split(r"\n\s*\**\s*Title\s*:", desc, flags=re.IGNORECASE)[0].strip()
    return (title or "Unnamed Cluster"), (desc or "No description available")


# ---------------------------------------------------------------- data loading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encode_latents import find_act  # noqa: E402  (numpy-only; same resolver phase 1 used)


def _read_pkl_texts(path):
    with open(path, "rb") as f:
        acts, texts, keys, _mean = pickle.load(f)
    del acts  # we only need the texts; the (N,d) array is 1-6 GB
    pm = np.array([k[0].encode() if isinstance(k[0], str) else k[0] for k in keys],
                  dtype="S32")
    si = np.array([int(k[1]) for k in keys], dtype=np.int32)
    return list(texts), pm, si


_TEXT_CACHE = {}


def texts_for(model, layer, sidecar_dir):
    """texts[i] = the sentence for sidecar row i, joined BY KEY (pmcid, sentence_idx).

    A positional join is not safe here. The sidecars are mostly row-identical to
    their pkl, but gpt-oss-20b's dedup sidecar has 126,425 rows against a 127,497-row
    pkl (its raw activations are an order-preserving subsequence of the centered
    ones), and gemma layer 18 is served from a *.regen.pkl. Joining on the key is
    correct under reordering AND subsetting; we assert every sidecar key is found,
    so a silent mis-attachment is impossible.
    """
    idx = np.load(os.path.join(sidecar_dir, "index.npz"), allow_pickle=True)
    pm, si = idx["pmcid"], idx["sentence_idx"].astype(np.int32)

    key2text = _TEXT_CACHE.get((model, layer)) or _TEXT_CACHE.get(model)
    if key2text is None:
        path = find_act(model, layer)
        if path is None:
            raise FileNotFoundError(f"no activation pkl for {model} layer {layer}")
        texts, tpm, tsi = _read_pkl_texts(path)
        key2text = {(a, int(b)): t for a, b, t in zip(tpm, tsi, texts)}
        _TEXT_CACHE[model] = key2text

    try:
        out = [key2text[(pm[i], int(si[i]))] for i in range(len(pm))]
    except KeyError:
        # this layer's rows aren't all in the cached pkl -> load its own
        path = find_act(model, layer)
        texts, tpm, tsi = _read_pkl_texts(path)
        key2text = {(a, int(b)): t for a, b, t in zip(tpm, tsi, texts)}
        _TEXT_CACHE[(model, layer)] = key2text
        missing = [i for i in range(len(pm)) if (pm[i], int(si[i])) not in key2text]
        if missing:
            raise AssertionError(
                f"{model} layer{layer}: {len(missing)} sidecar rows have no sentence "
                f"in {os.path.basename(path)}. Refusing to guess."
            )
        out = [key2text[(pm[i], int(si[i]))] for i in range(len(pm))]
    return out


def load_traces(model, n, max_chars, seed):
    """n reasoning traces, sampled once per model, each truncated to max_chars."""
    spec = importlib.util.spec_from_file_location(
        "_responses", os.path.join(REPO, "utils/responses.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    extract_thinking_process = mod.extract_thinking_process

    for base in (
        os.path.join(REPO, "generate-responses/generate-responses/results/vars"),
        os.path.join(REPO, "generate-responses/results/vars"),
    ):
        p = os.path.join(base, f"responses_{model}.json")
        if os.path.exists(p):
            break
    else:
        print(f"  [warn] no responses_{model}.json; proceeding with no trace examples")
        return ""

    with open(p) as f:
        data = json.load(f)

    rng = random.Random(seed)
    thinks = []
    # Prefer mid-length traces: the corpus contains degenerate repetition-loop
    # responses up to 166k chars which would blow the context for no benefit.
    pool = list(data)
    rng.shuffle(pool)
    for sample in pool:
        t = extract_thinking_process(sample.get("full_response", ""))
        if t and 500 < len(t) <= max_chars:
            thinks.append(t)
        if len(thinks) == n:
            break
    if len(thinks) < n:  # fall back to truncation if the corpus is short on them
        for sample in pool:
            t = extract_thinking_process(sample.get("full_response", ""))
            if t and len(t) > max_chars:
                thinks.append(t[:max_chars])
            if len(thinks) == n:
                break
    if not thinks:
        return ""

    out = "Here are some full reasoning traces to help understand the context:\n'''\n"
    for i, t in enumerate(thinks):
        out += f"TRACE {i+1}:\n{t}\n\n"
    out += "'''"
    return out


def representative_examples(vals, argmax, k, texts, row_ids, n_examples, seed,
                            max_example_chars=600, uniq_examples=False):
    """Reproduce generate_representative_examples (sae_topk branch) +
    _prepare_cluster_examples. Returns {latent: [sentence, ...]} for LIVE latents.

    max_example_chars guards the context window: a "sentence" here is really a
    segment, and the corpus contains segments up to 36,450 chars. Capping at 600
    truncates 1.4% of segments and preserves 95% of all text.
    """
    rng = random.Random(seed)
    out = {}
    for c in range(k):
        sel = np.where(argmax == c)[0]
        if len(sel) == 0:
            continue  # dead latent -> no title, exactly as the reference does
        # rank by the latent's own raw encoder value, descending
        order = sel[np.argsort(-vals[sel, c], kind="stable")]
        ranked = [texts[row_ids[i]] for i in order]
        if max_example_chars:
            ranked = [t[:max_example_chars] for t in ranked]
        if uniq_examples:
            # A loop latent's examples are a wall of one repeated fragment; the
            # annotator then echoes it and never answers. Collapse to unique
            # strings (ranking preserved) so it sees variety, not a loop.
            seen_ex, uniq = set(), []
            for t in ranked:
                if t not in seen_ex:
                    seen_ex.add(t); uniq.append(t)
            ranked = uniq

        if len(ranked) <= n_examples:
            ex = list(ranked)
        else:
            n_top = n_examples // 2
            n_rand = n_examples - n_top
            top = ranked[:n_top]
            rest = ranked[n_top:]
            rand = rest if len(rest) < n_rand else rng.sample(rest, n_rand)
            ex = top + rand
        rng.shuffle(ex)
        out[c] = ex
    return out


# ---------------------------------------------------------------- vLLM client
async def call_one(client, url, model_name, prompt, effort, max_tokens, sem, retries=4):
    import httpx  # noqa: F401  (imported lazily so --dry-run needs no deps)

    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        # llama.cpp: reuse the KV of the shared prefix (the per-model reasoning-trace
        # block is identical across every prompt for that model). Ignored by vLLM,
        # which does prefix caching automatically.
        "cache_prompt": True,
    }
    if effort:
        body["reasoning_effort"] = effort

    async with sem:
        for attempt in range(retries):
            try:
                r = await client.post(url, json=body, timeout=1200.0)
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                if content:
                    return content
                # gpt-oss may put everything in the reasoning channel if it ran
                # out of budget before emitting a final message.
                return (msg.get("reasoning_content") or "").strip()
            except Exception as e:
                if attempt == retries - 1:
                    return f"__ERROR__ {type(e).__name__}: {e}"
                await asyncio.sleep(2 * (attempt + 1))
    return "__ERROR__ unreachable"


# ---------------------------------------------------------------- job planning
def out_path(sae_set, model, layer, k):
    return os.path.join(
        OUT_ROOT, sae_set, f"titles_{model}_layer{layer}_clusters{k}.json"
    )


def layers_for(sae_set, model):
    root = os.path.join(SIDECAR[sae_set], model)
    if not os.path.isdir(root):
        return []
    return sorted(
        int(d[5:]) for d in os.listdir(root) if d.startswith("layer") and d[5:].isdigit()
    )


def write_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


async def run(args):
    import httpx

    sets = ["old", "dedup"] if args.sae_set == "both" else [args.sae_set]
    models = args.models or MODELS
    ks = args.ks or KS
    url = args.base_url.rstrip("/") + "/chat/completions"

    # ---- plan
    jobs = []  # (sae_set, model, layer, k)
    for s in sets:
        for m in models:
            for L in layers_for(s, m):
                if args.layers and L not in args.layers:
                    continue
                for k in ks:
                    if not args.force and os.path.exists(out_path(s, m, L, k)):
                        continue
                    jobs.append((s, m, L, k))

    print(f"{len(jobs)} taxonomies to title "
          f"(sets={sets}, models={len(models)}, ks={ks})")
    if args.dry_run:
        for s, m, L, k in jobs[:20]:
            print("  ", s, m, L, k)
        if len(jobs) > 20:
            print(f"   ... and {len(jobs)-20} more")
        return

    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency + 8)
    t0 = time.time()
    done = 0
    n_err = 0

    async with httpx.AsyncClient(limits=limits) as client:
        # group by model so texts/traces load once per model
        for m in models:
            mjobs = [j for j in jobs if j[1] == m]
            if not mjobs:
                continue
            print(f"\n=== {m}: {len(mjobs)} taxonomies ===", flush=True)
            traces = load_traces(m, args.n_trace_examples, args.max_trace_chars,
                                 seed=stable_seed(m))
            print(f"  trace_prefix_chars={len(traces):,}", flush=True)

            for (s, _m, L, k) in mjobs:
                d = os.path.join(SIDECAR[s], m, f"layer{L}")
                texts = texts_for(m, L, d)  # verified row-aligned to this sidecar
                z = np.load(os.path.join(d, "latents.npz"))
                vals = z[f"k{k}"]
                argmax = z[f"argmax_k{k}"]

                if s in ("dedup", "nndedup"):
                    idx = np.load(os.path.join(d, "index.npz"), allow_pickle=True)
                    mask = idx["in_training"].astype(bool)
                    row_ids = np.where(mask)[0]
                    vals, argmax = vals[row_ids], argmax[row_ids]
                else:
                    row_ids = np.arange(len(argmax))

                seed = stable_seed(s, m, L, k)
                reps = representative_examples(
                    vals, argmax, k, texts, row_ids, args.description_examples, seed,
                    args.max_example_chars, args.uniq_examples
                )
                dead = [c for c in range(k) if c not in reps]

                cids = sorted(reps)
                prompts = [
                    build_cluster_description_prompt(
                        reps[c], traces, args.n_categories_examples
                    )
                    for c in cids
                ]
                res = await asyncio.gather(*[
                    call_one(client, url, args.model_name, p, args.reasoning_effort,
                             args.max_tokens, sem)
                    for p in prompts
                ])

                cats, raw = [], {}
                errs = 0
                for c, r in zip(cids, res):
                    raw[str(c)] = r
                    if r.startswith("__ERROR__"):
                        errs += 1
                    t, dsc = parse_title_desc(r)
                    cats.append([str(c), t, dsc])
                n_err += errs

                write_atomic(out_path(s, m, L, k), {
                    "model_id": m, "layer": L, "n_clusters": k, "sae_set": s,
                    "annotator": args.model_name,
                    "reasoning_effort": args.reasoning_effort,
                    "description_examples": args.description_examples,
                    "n_trace_examples": args.n_trace_examples,
                    "n_categories_examples": args.n_categories_examples,
                    "seed": seed,
                    "n_rows_ranked": int(len(argmax)),
                    "live_latents": [int(c) for c in cids],
                    "dead_latents": [int(c) for c in dead],
                    "n_errors": errs,
                    "categories": cats,
                    "raw_responses": raw,
                })

                done += 1
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(jobs) - done) / rate if rate else 0
                flag = f"  ERRORS={errs}" if errs else ""
                print(f"  [{done}/{len(jobs)}] {s} {m} L{L} k{k}: "
                      f"{len(cids)} titles, {len(dead)} dead"
                      f"{flag}  | {el/60:.1f}m elapsed, ETA {eta/60:.1f}m", flush=True)

    print(f"\nDONE  {done} taxonomies, {n_err} failed calls, "
          f"{(time.time()-t0)/60:.1f} min")


def _is_bad(title):
    return title in ("Unnamed Cluster", "", "No description available")


async def retry_failed(args):
    """Re-call ONLY the latents whose title failed to parse or errored, in existing
    output files, and merge them back. Leaves every good title untouched.

    Two known causes, both handled here:
      - prompt over the server's per-slot context -> 400. --max-example-chars is
        lowered here so even the biggest latent fits (glm's long segments).
      - gpt-oss spent its whole budget reasoning and never emitted a final answer
        -> unparseable. --max-tokens is raised and --reasoning-effort lowered.
    """
    import httpx

    url = args.base_url.rstrip("/") + "/chat/completions"
    sub = "dedup" if args.sae_set in ("dedup", "both") else args.sae_set
    files = sorted(glob_titles(sub))
    todo = []  # (path, data, [bad_latent_ints])
    for f in files:
        d = json.load(open(f))
        bad = [int(c[0]) for c in d["categories"] if _is_bad(c[1])]
        if bad:
            todo.append((f, d, bad))
    n_bad = sum(len(b) for _, _, b in todo)
    print(f"retry: {len(todo)} files, {n_bad} failed latents "
          f"(max_example_chars={args.max_example_chars}, max_tokens={args.max_tokens}, "
          f"effort={args.reasoning_effort!r})")
    if args.dry_run or not todo:
        for f, _, b in todo:
            print("  ", os.path.basename(f), "latents", b)
        return

    sem = asyncio.Semaphore(args.concurrency)
    fixed = still = 0
    # group by model so texts/traces load once
    by_model = defaultdict(list)
    for f, d, bad in todo:
        by_model[d["model_id"]].append((f, d, bad))

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.concurrency + 8)) as client:
        for m, items in by_model.items():
            traces = load_traces(m, args.n_trace_examples, args.max_trace_chars, stable_seed(m))
            for f, d, bad in items:
                s, L, k = d["sae_set"], d["layer"], d["n_clusters"]
                dd = os.path.join(SIDECAR[s], m, f"layer{L}")
                texts = texts_for(m, L, dd)
                z = np.load(os.path.join(dd, "latents.npz"))
                vals, argmax = z[f"k{k}"], z[f"argmax_k{k}"]
                if s in ("dedup", "nndedup"):
                    idx = np.load(os.path.join(dd, "index.npz"), allow_pickle=True)
                    row_ids = np.where(idx["in_training"].astype(bool))[0]
                    vals, argmax = vals[row_ids], argmax[row_ids]
                else:
                    row_ids = np.arange(len(argmax))
                reps = representative_examples(
                    vals, argmax, k, texts, row_ids, args.description_examples,
                    stable_seed(s, m, L, k), args.max_example_chars, args.uniq_examples)

                cats = {c[0]: c for c in d["categories"]}
                raw = d.get("raw_responses", {})
                for c in bad:
                    if c not in reps:      # genuinely dead -> nothing to title
                        continue
                    p = build_cluster_description_prompt(reps[c], traces, args.n_categories_examples)
                    r = await call_one(client, url, args.model_name, p,
                                       args.reasoning_effort, args.max_tokens, sem)
                    raw[str(c)] = r
                    t, dsc = parse_title_desc(r)
                    cats[str(c)] = [str(c), t, dsc]
                    if _is_bad(t):
                        still += 1
                    else:
                        fixed += 1
                        print(f"  fixed {os.path.basename(f)} latent {c}: {t}", flush=True)

                d["categories"] = [cats[str(c)] for c in range(k) if str(c) in cats]
                d["raw_responses"] = raw
                d["n_errors"] = sum(1 for v in raw.values() if v.startswith("__ERROR__"))
                d["retry"] = {"max_example_chars": args.max_example_chars,
                              "max_tokens": args.max_tokens,
                              "reasoning_effort": args.reasoning_effort}
                write_atomic(f, d)

    print(f"\nRETRY DONE  fixed {fixed}, still failing {still}")


def glob_titles(sub="dedup"):
    import glob
    return glob.glob(os.path.join(OUT_ROOT, sub, "titles_*.json"))


def main():
    ap = argparse.ArgumentParser()
    # Default is dedup-only, by explicit instruction: the old SAEs and every file
    # sourced from matthewshu/medcasereasoning-cot-activations are the gold
    # standard and must not be edited or regenerated.
    ap.add_argument("--sae-set", choices=["old", "dedup", "nndedup", "both"], default="dedup")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--ks", nargs="+", type=int, default=None)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model-name", default="openai/gpt-oss-120b")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--reasoning-effort", default="medium",
                    choices=["low", "medium", "high", ""])
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--description-examples", type=int, default=200)
    ap.add_argument("--n-trace-examples", type=int, default=3)
    ap.add_argument("--n-categories-examples", type=int, default=5)
    ap.add_argument("--max-trace-chars", type=int, default=8000)
    ap.add_argument("--max-example-chars", type=int, default=600)
    ap.add_argument("--uniq-examples", action="store_true",
                    help="collapse a latent's examples to unique strings before "
                         "prompting (needed for loop latents whose examples are a "
                         "wall of one repeated fragment)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-call only unnamed/errored latents in existing files and "
                         "merge back (good titles untouched)")
    args = ap.parse_args()
    if args.retry_failed:
        asyncio.run(retry_failed(args))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
