#!/usr/bin/env python3
"""Fetch the raw (un-centered) activation pkls for the 9 models.

The Xet CAS backend intermittently 401s on signed URLs, so every file gets retried with
backoff. Run with /venv/main/bin/python (has huggingface_hub).
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download

REPO = "matthewshu/medcasereasoning-cot-activations"
MODELS = [
    "deepseek-r1-distill-llama-8b", "deepseek-r1-distill-qwen-14b", "huatuogpt-o1-8b",
    "gpt-oss-20b", "qwq-32b", "qwen3.6-27b", "gemma-4-31b-it",
    "ministral-3-14b-reasoning-2512", "glm-4.7-flash",
]
MAX_TRIES = 12


def want(f):
    return f.startswith("activations/raw/") and any(f"activations_{m}_" in f for m in MODELS)


def fetch(f):
    for attempt in range(1, MAX_TRIES + 1):
        try:
            p = hf_hub_download(REPO, f, repo_type="dataset")
            return f, os.path.getsize(p), attempt, None
        except Exception as e:  # transient 401 from the Xet CAS; back off and retry
            if attempt == MAX_TRIES:
                return f, 0, attempt, str(e)[:80]
            time.sleep(min(2 ** attempt, 30))
    return f, 0, MAX_TRIES, "exhausted"


def main():
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if want(f)]
    print(f"{len(files)} raw pkls to fetch", flush=True)
    done = tot = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for f, size, tries, err in [fu.result() for fu in
                                    as_completed([ex.submit(fetch, f) for f in files])]:
            done += 1
            if err:
                print(f"  [{done}/{len(files)}] FAIL {os.path.basename(f)}: {err}", flush=True)
            else:
                tot += size
                print(f"  [{done}/{len(files)}] ok   {os.path.basename(f):58s} "
                      f"{size/1e9:5.2f} GB  (tries={tries})", flush=True)
    print(f"\nfetched {tot/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
