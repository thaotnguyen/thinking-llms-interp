#!/usr/bin/env python3
"""Analyze model errors by sampling cases across accuracy buckets and auditing failures.

This script expects an eval output CSV from eval.py run with multiple samples per case
(e.g., 10). It will:

- Compute per-case accuracy from results.csv
- Select up to N=50 cases for each accuracy bucket in {0.0, 0.1, ..., 0.9, 1.0}
- For each selected case with at least one incorrect sample, pick the first incorrect row
- Fetch clinician-validated diagnostic_reasoning (and case_prompt if needed) from
  the Hugging Face dataset by matching on pmcid
- Run an error evaluation LLM prompt 5 times per incorrect case
- Save all 5 full responses and extracted <answer> label to a CSV

Environment:
- Requires `DEEPSEEK_API_KEY` set in the environment
- Requires `pip install datasets openai tqdm`

Example:
    ./analyze_error_types.py \
        --input results.csv \
        --output error_evals.csv \
        --dataset zou-lab/MedCaseReasoning \
        --per_bucket 50 \
        --runs 5 \
        --workers 8
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset  # type: ignore
from openai import OpenAI  # type: ignore
from tqdm.auto import tqdm  # type: ignore


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


ERROR_PROMPT_TEMPLATE = (
    "You are an expert clinician–auditor. The model’s diagnosis is wrong. Write a single, cohesive narrative that explains why the model’s reasoning_trace failed, grounded only in the supplied inputs. End with a one-phrase error label inside <answer> … </answer>.\n\n"
    "Inputs\n\n"
    "case_prompt: {case_prompt}\n\n"
    "reasoning_trace (model): {reasoning_trace}\n\n"
    "predicted_diagnosis (model): {predicted_diagnosis}\n\n"
    "true_diagnosis (gold): {true_diagnosis}\n\n"
    "diagnostic_reasoning (clinician-validated): {diagnostic_reasoning}\n\n"
    "Rules (no exceptions)\n\n"
    "Source fidelity. Use only these inputs. Do not invent facts. Support claims with short verbatim snippets in quotes.\n\n"
    "Be specific. Pinpoint the exact failure(s). Explicitly contrast with clinician diagnostic_reasoning using quotes from both.\n\n"
    "Primary failure only. Multiple problems may exist; choose one primary failure that best explains the wrong answer.\n\n"
    "Final line (taxonomy label)\n"
    "Output a very short primary error label in <answer>: 1–3 words, lowercase preferred; hyphenate if helpful.\n"
    "Critically: the <answer> must be a general, case-agnostic term only — no case- or patient-specific details (no names, exact ages, lab values, quoted snippets, or diagnoses).\n"
    "Keep case-specific details in the analysis body, not inside <answer>.\n\n"
    "Now produce the analysis, and end with the error label in <answer>…</answer>. (See <attachments> above for file contents. You may not need to search or read the file again.)"
)


THREAD_LOCAL = threading.local()
CLIENT_SETTINGS: Dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit model errors across accuracy buckets")
    parser.add_argument("--input", required=True, help="Path to results.csv produced by eval.py")
    parser.add_argument("--output", default="error_evals.csv", help="Output CSV for detailed error evaluations")
    parser.add_argument("--dataset", default="zou-lab/MedCaseReasoning", help="HF dataset id for ground-truth diagnostic_reasoning")
    parser.add_argument("--per_bucket", type=int, default=50, help="Max cases to sample per accuracy bucket (0.0..1.0)")
    parser.add_argument("--runs", type=int, default=5, help="Number of LLM eval runs per incorrect case")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent worker threads for LLM calls")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between LLM calls (per task)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip evaluation rows already present in output CSV")
    return parser.parse_args()


def get_client() -> OpenAI:
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        if not CLIENT_SETTINGS:
            raise RuntimeError("Client settings not initialised")
        client = OpenAI(**CLIENT_SETTINGS)
        THREAD_LOCAL.client = client
    return client


def extract_answer_label(text: str) -> Optional[str]:
    match = ANSWER_PATTERN.search(text or "")
    return match.group(1).strip() if match else None


def read_results(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


@dataclass
class CaseStats:
    pmcid: str
    total: int
    n_correct: int
    accuracy: float
    first_incorrect_row: Optional[Dict[str, Any]]
    first_seen_index: int  # for stable ordering


def normalize_bool(value: Any) -> bool:
    v = str(value).strip().lower()
    return v in {"true", "1", "yes", "y"}


def compute_per_case_stats(rows: List[Dict[str, Any]]) -> Dict[str, CaseStats]:
    per_case: Dict[str, CaseStats] = {}
    seen_order: Dict[str, int] = {}
    for idx, row in enumerate(rows):
        pmcid = (row.get("pmcid") or row.get("id") or "").strip()
        if not pmcid:
            # Skip rows without identifier
            continue
        if pmcid not in seen_order:
            seen_order[pmcid] = idx
        cs = per_case.get(pmcid)
        if cs is None:
            cs = CaseStats(pmcid=pmcid, total=0, n_correct=0, accuracy=0.0, first_incorrect_row=None, first_seen_index=seen_order[pmcid])
            per_case[pmcid] = cs
        cs.total += 1
        if normalize_bool(row.get("verified_correct")):
            cs.n_correct += 1
        else:
            if cs.first_incorrect_row is None:
                cs.first_incorrect_row = row
    # finalize accuracy
    for cs in per_case.values():
        cs.accuracy = (cs.n_correct / cs.total) if cs.total else 0.0
    return per_case


def bucketize_cases(per_case: Dict[str, CaseStats], per_bucket: int) -> Dict[float, List[CaseStats]]:
    buckets: Dict[float, List[CaseStats]] = {round(x * 0.1, 1): [] for x in range(0, 11)}
    # Order by first_seen_index for determinism
    ordered = sorted(per_case.values(), key=lambda c: c.first_seen_index)
    for cs in ordered:
        # Round to 1 decimal place; also bound to [0.0, 1.0]
        b = min(1.0, max(0.0, round(cs.accuracy, 1)))
        if len(buckets[b]) < per_bucket:
            buckets[b].append(cs)
    return buckets


def load_dataset_lookup(dataset_name: str, split: str = "train") -> Dict[str, Dict[str, Any]]:
    ds = load_dataset(dataset_name, split=split)
    lookup: Dict[str, Dict[str, Any]] = {}
    for ex in ds:
        pmcid = ex.get("pmcid")
        if pmcid is None:
            continue
        lookup[str(pmcid)] = ex
    return lookup


def build_error_prompt(row: Dict[str, Any], gold: Dict[str, Any]) -> str:
    case_prompt = (row.get("case_prompt") or gold.get("case_prompt") or gold.get("case_presentation") or "").strip()
    reasoning_trace = (row.get("reasoning_trace") or "").strip()
    predicted = (row.get("predicted_diagnosis") or "").strip()
    true_dx = (row.get("true_diagnosis") or gold.get("final_diagnosis") or "").strip()
    diagnostic_reasoning = (gold.get("diagnostic_reasoning") or "").strip()
    return ERROR_PROMPT_TEMPLATE.format(
        case_prompt=case_prompt,
        reasoning_trace=reasoning_trace,
        predicted_diagnosis=predicted,
        true_diagnosis=true_dx,
        diagnostic_reasoning=diagnostic_reasoning,
    )


def call_deepseek_chat(prompt: str, retries: int = 3) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ]
    client = get_client()
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as exc:  # pylint: disable=broad-exception-caught
            wait_time = min(2 ** attempt, 30)
            print(f"Chat error eval failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(wait_time)
    raise RuntimeError("Unreachable retry loop exit")


def read_existing_output(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY environment variable is not set")

    global CLIENT_SETTINGS  # noqa: PLW0603
    CLIENT_SETTINGS = {"api_key": api_key, "base_url": "https://api.deepseek.com"}

    # Read inputs and compute per-case stats
    rows = read_results(args.input)
    per_case = compute_per_case_stats(rows)
    buckets = bucketize_cases(per_case, args.per_bucket)

    # Build quick index of rows per pmcid in input order for stable selection
    rows_by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r.get("pmcid") or r.get("id") or "").strip()
        if key:
            rows_by_case[key].append(r)

    # Prepare dataset lookup for diagnostic_reasoning
    print("Loading dataset for diagnostic_reasoning lookup...", file=sys.stderr)
    gold_lookup = load_dataset_lookup(args.dataset, split="train")

    # Prepare output CSV
    fieldnames = [
        "pmcid",
        "bucket_accuracy",
        "total_samples",
        "n_correct",
        "incorrect_sample_index",
        "predicted_diagnosis",
        "true_diagnosis",
        "reasoning_trace",
        "case_prompt",
        "diagnostic_reasoning",
        "run_index",
        "evaluation_text",
        "error_label",
    ]

    # Existing output handling for skip logic
    existing_rows = read_existing_output(args.output) if args.skip_existing else []
    existing_keys = set()
    for er in existing_rows:
        # Unique key per (pmcid, incorrect_sample_index, run_index)
        ek = (
            (er.get("pmcid") or "").strip(),
            str(er.get("incorrect_sample_index") or "").strip(),
            str(er.get("run_index") or "").strip(),
        )
        existing_keys.add(ek)

    # Open file for appending; write header if new
    file_exists = os.path.isfile(args.output)
    needs_header = (not file_exists) or os.path.getsize(args.output) == 0
    outfile = open(args.output, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, extrasaction="ignore")
    if needs_header:
        writer.writeheader()
        outfile.flush()

    # Build tasks: one task per run for each incorrect selected case
    tasks: List[Tuple[str, int, int]] = []  # (pmcid, incorrect_sample_index, run_index)
    task_meta: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for bucket_value in [round(x * 0.1, 1) for x in range(0, 11)]:
        selected_cases = buckets.get(bucket_value, [])
        for cs in selected_cases:
            pmcid = cs.pmcid
            if cs.first_incorrect_row is None:
                # Accuracy 1.0 or no incorrect sample; no eval tasks created
                continue
            first_bad = cs.first_incorrect_row
            try:
                bad_idx = int(first_bad.get("sample_index") or 0)
            except Exception:
                bad_idx = 0

            # Prepare prompt inputs
            gold = gold_lookup.get(pmcid, {})
            prompt = build_error_prompt(first_bad, gold)

            # Persist meta to attach to each run record without rebuilding strings repeatedly
            meta = {
                "pmcid": pmcid,
                "bucket_accuracy": f"{bucket_value:.1f}",
                "total_samples": cs.total,
                "n_correct": cs.n_correct,
                "incorrect_sample_index": bad_idx,
                "predicted_diagnosis": (first_bad.get("predicted_diagnosis") or "").strip(),
                "true_diagnosis": (first_bad.get("true_diagnosis") or gold.get("final_diagnosis") or "").strip(),
                "reasoning_trace": (first_bad.get("reasoning_trace") or "").strip(),
                "case_prompt": (first_bad.get("case_prompt") or gold.get("case_prompt") or gold.get("case_presentation") or "").strip(),
                "diagnostic_reasoning": (gold.get("diagnostic_reasoning") or "").strip(),
                "prompt": prompt,
            }
            task_meta[(pmcid, bad_idx)] = meta

            for run_idx in range(1, args.runs + 1):
                key = (pmcid, str(bad_idx), str(run_idx))
                if args.skip_existing and key in existing_keys:
                    continue
                tasks.append((pmcid, bad_idx, run_idx))

    if not tasks:
        print("No error evaluation tasks to run (nothing selected or all already present).")
        outfile.close()
        return

    # Define worker
    def run_task(pmcid: str, sample_idx: int, run_idx: int) -> Dict[str, Any]:
        meta = task_meta[(pmcid, sample_idx)]
        text = call_deepseek_chat(meta["prompt"])
        label = extract_answer_label(text) or ""
        if args.sleep > 0:
            time.sleep(args.sleep)
        return {
            **{k: meta[k] for k in (
                "pmcid",
                "bucket_accuracy",
                "total_samples",
                "n_correct",
                "incorrect_sample_index",
                "predicted_diagnosis",
                "true_diagnosis",
                "reasoning_trace",
                "case_prompt",
                "diagnostic_reasoning",
            )},
            "run_index": run_idx,
            "evaluation_text": text,
            "error_label": label,
        }

    # Execute with progress bar
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor, tqdm(total=len(tasks), desc="Error evals", unit="task") as progress:
        futures = [executor.submit(run_task, pmcid, sidx, ridx) for (pmcid, sidx, ridx) in tasks]
        for fut in as_completed(futures):
            rec = fut.result()
            writer.writerow(rec)
            outfile.flush()
            completed += 1
            progress.update(1)

    outfile.close()
    print(f"Completed {completed} error evaluations across {len(tasks)} tasks. Output -> {args.output}")


if __name__ == "__main__":
    main()
