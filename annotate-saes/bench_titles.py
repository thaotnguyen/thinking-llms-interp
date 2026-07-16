#!/usr/bin/env python3
"""
Measure real titling throughput against the running server, then project the full run.

Sends N *real* prompts (same builder, same 200 examples, same trace prefix as the
actual job) at the same concurrency the real run will use, and reports observed
prompt-tokens/s and end-to-end tokens/s -> ETA for all 4,801 calls.

  python3 annotate-saes/bench_titles.py --base-url http://localhost:8080/v1 --n 8
"""
import argparse
import asyncio
import importlib.util
import os
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "gt", os.path.join(REPO, "annotate-saes/generate_titles.py"))
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)

TOTAL_CALLS = 4801  # total live latents across the 324 dedup taxonomies


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model-name", default="openai/gpt-oss-120b")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--description-examples", type=int, default=200)
    ap.add_argument("--max-example-chars", type=int, default=600)
    args = ap.parse_args()

    import httpx

    m, L, k = "glm-4.7-flash", 8, 20
    d = os.path.join(REPO, f"annotate-saes/results/vars/latents_dedup/{m}/layer{L}")
    texts = gt.texts_for(m, L, d)
    z = np.load(os.path.join(d, "latents.npz"))
    mask = np.load(os.path.join(d, "index.npz"), allow_pickle=True)["in_training"].astype(bool)
    rid = np.where(mask)[0]
    vals, am = z[f"k{k}"][rid], z[f"argmax_k{k}"][rid]
    traces = gt.load_traces(m, 3, 8000, gt.stable_seed(m))
    reps = gt.representative_examples(vals, am, k, texts, rid,
                                      args.description_examples, gt.stable_seed("b", m, L, k),
                                      args.max_example_chars)
    cids = sorted(reps)[:args.n]
    prompts = [gt.build_cluster_description_prompt(reps[c], traces, 5) for c in cids]
    approx_tok = sum(len(p) for p in prompts) / 4 / len(prompts)
    print(f"{len(prompts)} real prompts, ~{approx_tok:,.0f} prompt tokens each "
          f"(trace prefix ~{len(traces)//4:,} tok, shared)")

    url = args.base_url.rstrip("/") + "/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.concurrency + 4)) as c:
        res = await asyncio.gather(*[
            gt.call_one(c, url, args.model_name, p, args.reasoning_effort,
                        args.max_tokens, sem) for p in prompts])
    el = time.time() - t0

    ok = [r for r in res if not r.startswith("__ERROR__")]
    print(f"\n{len(ok)}/{len(res)} succeeded in {el:.1f}s "
          f"({el/max(1,len(res)):.1f}s per call at concurrency {args.concurrency})")
    if not ok:
        print("ALL FAILED:", res[0][:300]); return

    pt = approx_tok * len(res)
    print(f"observed prompt throughput : ~{pt/el:,.0f} tok/s")
    print(f"per-call wall time          : {el/len(res)*args.concurrency:.1f}s (serial-equivalent)")
    eta_h = (TOTAL_CALLS / (len(res) / el)) / 3600
    print(f"\nPROJECTED FULL RUN ({TOTAL_CALLS:,} calls): {eta_h:.1f} hours")
    for ne in (100, 64):
        print(f"   if --description-examples {ne}: ~{eta_h * (ne/args.description_examples):.1f} h "
              f"(prompt shrinks roughly proportionally)")
    print("\n--- sample title ---")
    t, dsc = gt.parse_title_desc(ok[0])
    print(f"  title: {t}")
    print(f"  desc : {dsc[:220]}")


if __name__ == "__main__":
    asyncio.run(main())
