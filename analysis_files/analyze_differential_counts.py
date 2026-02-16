#!/usr/bin/env python3
"""Analyze differential breadth from reasoning traces and correlate with accuracy.

This script:
- Reads results.csv produced by eval.py (one row per trace/sample).
- Uses DeepSeek Chat (deepseek-chat) to extract the unique items considered on the differential
  from the reasoning_trace and/or posthoc_reasoning_trace for each row.
- Writes per-trace outputs with the unique item list, count, and whether the true diagnosis
  appears among the differential items.
- Merges per-case difficulty from results.per_case.csv (difficulty = round((1-accuracy)*10)).
- Computes correlations of differential count with accuracy:
  - Unadjusted (Pearson/Spearman with correctness 0/1)
  - Adjusted for difficulty via logistic regression (correct ~ count + difficulty)
- Summarizes accuracy conditional on whether the true diagnosis is in the extracted differential.
 - Uses LLM-based presence verification: asks DeepSeek whether the true diagnosis is represented
     among the extracted items; if the original model's prediction was correct, also asks if the
     predicted diagnosis is represented. Either being present counts as correct_in_diff.

Environment:
- Requires DEEPSEEK_API_KEY if --provider deepseek is used (default).

Outputs:
- results.csv: updated in place with added per-trace columns (diff_items, diff_count, correct_in_diff, trace_correct)
- results.per_case.csv: updated in place (if present) with added aggregated columns per pmcid (diff_count_mean, diff_count_median, correct_in_diff_rate)
- analysis_outputs/differential_count_correlation.txt: textual summary of correlations
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import threading


# Optional imports guarded for runtime usage only (API client, sklearn)
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    LogisticRegression = None  # type: ignore
    roc_auc_score = None  # type: ignore


DIFF_CACHE_DEFAULT = "analysis_outputs/differential_cache.jsonl"
PER_TRACE_OUT_CSV = "analysis_outputs/differential_counts.csv"
SUMMARY_TXT = "analysis_outputs/differential_count_correlation.txt"

# Serialize cache writes to avoid too many concurrent file opens
CACHE_LOCK = threading.Lock()


def _norm_text(s: str) -> str:
    s = s.strip().lower()
    # Normalize common punctuation, whitespace, unicode quotes/dashes
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_diagnosis_name(name: str) -> str:
    name = _norm_text(name)
    # Remove leading/trailing <answer> tags etc.
    name = re.sub(r"^<answer>\s*|\s*</answer>$", "", name)
    # Strip punctuation and non-alphanumeric except spaces and hyphens
    name = re.sub(r"[^a-z0-9\-/\s]+", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def load_cache(path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not path or not os.path.isfile(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = (str(obj.get("pmcid", "")), int(obj.get("sample_index", 0)))
                cache[key] = obj
            except Exception:
                continue
    return cache


def append_cache(path: str, record: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with CACHE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_client(provider: str = "deepseek") -> Any:
    if provider != "deepseek":  # Only deepseek is supported per request
        raise ValueError("Only provider 'deepseek' is supported in this script.")
    if OpenAI is None:
        raise ImportError("openai package is required. Please install per requirements.txt")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


EXTRACTION_SYSTEM_PROMPT = (
    "You are a meticulous clinical NLP assistant. "
    "Given a model's reasoning text, extract all medical conditions that were at any point "
    "considered as possible diagnoses (i.e., on the differential) during the reasoning. "
    "Normalize synonyms to a concise canonical form (e.g., 'myocardial infarction' ~ 'mi' -> 'myocardial infarction'). "
    "Return strictly valid JSON with the shape: {\"items\": [\"diagnosis1\", \"diagnosis2\", ...]}. "
    "Do not include explanations or additional keys. Provide a deduplicated list with 1-5 words per item, "
    "ordered by the sequence in which they were first considered in the reasoning text (chronological order of consideration). "
    "Only include vague umbrella categories (e.g., 'retroperitoneal process') as separate items if there are no more specific diagnoses under that umbrella present in the list."
)

FEW_SHOT_REASONING = (
    "First, I need to read the case presentation carefully and identify key symptoms and findings. "
    "The patient is a 6-year-old boy with: - Speech regression - Poor social interaction - Aggressive behavior "
    "- Delayed cry at birth - Hospitalized for diarrhea at 9 months - Lost speech at age 2.5 years "
    "- Treated for tuberculosis at age 4 - Suspected syndromic autism spectrum disorder - EEG: intractable epileptiform activity "
    "from left occipital hemisphere with spread and diffuse right-hemisphere damage - Brain MRI: mild thinning of posterior body of corpus callosum, "
    "no cerebral or cerebellar atrophy - Examination: nonverbal, restless, ichthyosis over legs, back, and abdomen; no facial dysmorphism "
    "- Karyotype and Fragile-X testing normal - Leukocyte enzyme assays: low arylsulfatase-A and low arylsulfatase-B activity, but normal beta-galactosidase "
    "Now, the critical clue is the enzyme assays. Low arylsulfatase-A and arylsulfatase-B activity. Arylsulfatases are involved in lysosomal storage diseases. "
    "Arylsulfatase-A deficiency is associated with metachromatic leukodystrophy (MLD), which can cause neurological regression, behavioral issues, and seizures. "
    "But MLD typically has normal arylsulfatase-B. Arylsulfatase-B deficiency is associated with Maroteaux-Lamy syndrome (Mucopolysaccharidosis type VI), "
    "which involves skeletal abnormalities, coarse facial features, and other issues, but not typically the neurological symptoms described here. "
    "The patient has low levels of both arylsulfatase-A and arylsulfatase-B. That suggests a condition where multiple sulfatases are deficient. "
    "There's a disorder called multiple sulfatase deficiency (MSD), also known as Austin disease. This is a rare lysosomal storage disease where there's a deficiency in multiple sulfatase enzymes, "
    "including arylsulfatase A, B, C, etc. It combines features of MLD and mucopolysaccharidoses. Symptoms of multiple sulfatase deficiency include: "
    "- Neurological deterioration (like speech regression, behavioral changes) - Seizures (epileptiform activity on EEG) - Ichthyosis (which is present in this case) - Skeletal abnormalities (not mentioned, but might be absent or mild) "
    "- Often, consanguinity (as in this case, second-degree consanguineous parents) increases risk for autosomal recessive disorders. "
    "In multiple sulfatase deficiency, patients can have features of both leukodystrophy and mucopolysaccharidosis. The ichthyosis is a key feature, as it's common in MSD due to steroid sulfatase deficiency. Other points: "
    "- Delayed cry at birth might indicate neurological issues early on. - Diarrhea at 9 months could be related to gastrointestinal involvement. - Tuberculosis treatment might be coincidental or due to immune compromise, but not directly related. "
    "- EEG shows epileptiform activity, which fits with neurological involvement. - MRI shows thinning of corpus callosum, which can be seen in various neurodegenerative disorders. "
    "- No cerebral or cerebellar atrophy, which might be consistent with early or specific forms. - Normal karyotype and Fragile-X rule out common genetic causes of autism-like symptoms. "
    "The enzyme assays show low arylsulfatase-A and B, but normal beta-galactosidase. Beta-galactosidase is involved in other disorders like GM1 gangliosidosis, "
    "but it's normal here, so that's not it. In multiple sulfatase deficiency, multiple sulfatases are deficient, including arylsulfatases A, B, and others. "
    "The low levels of both A and B strongly point to MSD. Ichthyosis is a hallmark of MSD due to deficiency in steroid sulfatase. So, the most likely diagnosis is multiple sulfatase deficiency."
)

FEW_SHOT_ITEMS = [
    "metachromatic leukodystrophy",
    "maroteaux-lamy syndrome",
    "multiple sulfatase deficiency",
    "gm1 gangliosidosis",
]

# Second few-shot example (acute pancreatitis with umbrella category mentioned)
FEW_SHOT2_REASONING = (
    "First, I need to read the case presentation carefully and identify the key points. The patient is a 44-year-old man with acute onset of severe scrotal pain. "
    "He denies urinary or bowel symptoms, groin lumps, fever, nausea, vomiting, or trauma. He has a history of alcoholic pancreatitis 10 years ago and currently drinks about 60 units of alcohol per week, which is significant. "
    "The scrotal ultrasound was normal, and a urinary tract ultrasound showed fatty liver but was otherwise unremarkable. On examination: afebrile, hemodynamically stable. Abdomen is soft, non-distended, no tenderness, guarding, or rebound. "
    "No hernias. Scrotum is normal and non-tender; testicles are normal. Digital rectal exam normal. Lab studies: - Serum amylase is elevated at 636 U/L (normal 20-105) - C-reactive protein (CRP) is elevated at 46 mg/L (normal <10) - White-cell count is elevated at 13.6 × 10^9/L (normal 3.8-11.0) "
    "Arterial blood gas shows mild respiratory alkalosis, and chest X-ray is normal. The main symptom is scrotal pain, but the scrotal exam and ultrasound are normal. This suggests that the pain might not be originating from the scrotum itself. "
    "Instead, it could be referred pain from another source. Given the history of alcoholic pancreatitis and current heavy alcohol use, along with elevated amylase, which is a marker for pancreatitis, this points towards pancreatitis. "
    "Amylase is specifically elevated in pancreatitis. The patient has no abdominal tenderness on exam, which is unusual for acute pancreatitis, but it's not always present. Sometimes pancreatitis can present with atypical pain, including referred pain to the scrotum or other areas. "
    "Referred pain from pancreatitis can occur due to irritation of the diaphragm or other structures. The genitofemoral nerve or other nerves might be involved, causing scrotal pain. The elevated CRP and white-cell count indicate inflammation, which is consistent with pancreatitis. "
    "The fatty liver on ultrasound is also consistent with chronic alcohol use. The normal scrotal ultrasound rules out testicular torsion, epididymitis, or other local scrotal issues. So, the most likely diagnosis is acute pancreatitis, possibly exacerbated by his alcohol consumption. "
    "The scrotal pain could be referred pain from the pancreatitis. I recall that pancreatitis can sometimes present with extra-abdominal symptoms, including scrotal pain, though it's rare. Another possibility could be something like a retroperitoneal process, but pancreatitis fits well. "
    "The mild respiratory alkalosis might be due to pain or early compensation for metabolic acidosis, but in pancreatitis, it can be associated with complications. Overall, the evidence strongly points to acute pancreatitis. Therefore, my internal reasoning leads me to diagnose acute pancreatitis."
)

FEW_SHOT2_ITEMS = [
    "acute pancreatitis",
    "testicular torsion",
    "epididymitis",
    "retroperitoneal process",
]


def build_user_prompt(reasoning_text: str) -> str:
    return (
        "Extract all medical conditions that were at any point considered as possible diagnoses "
        "(i.e., on the differential) from the following reasoning text.\n\n"
        "--- REASONING TEXT START ---\n"
        f"{reasoning_text}\n"
        "--- REASONING TEXT END ---\n\n"
        "Respond with JSON only: {\"items\": [<strings>]}. Order items by the order they were first considered/mentioned in the reasoning."
    )


def call_deepseek_extract(client: Any, reasoning_text: str, model: str = "deepseek-chat", retries: int = 3, sleep: float = 0.0) -> List[str]:
    # Few-shot example to anchor extraction behavior
    few_shot_messages = [
        {"role": "user", "content": build_user_prompt(FEW_SHOT_REASONING)},
        {"role": "assistant", "content": json.dumps({"items": FEW_SHOT_ITEMS})},
        {"role": "user", "content": build_user_prompt(FEW_SHOT2_REASONING)},
        {"role": "assistant", "content": json.dumps({"items": FEW_SHOT2_ITEMS})},
    ]
    messages = (
        [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}]
        + few_shot_messages
        + [{"role": "user", "content": build_user_prompt(reasoning_text)}]
    )
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, stream=False)
            content = resp.choices[0].message.content or ""
            # Find first JSON object in response
            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                content = m.group(0)
            data = json.loads(content)
            items = data.get("items", [])
            if not isinstance(items, list):
                items = []
            # Normalize and dedupe
            norm_items = []
            seen = set()
            for it in items:
                if not isinstance(it, str):
                    continue
                norm = _norm_diagnosis_name(it)
                if norm and norm not in seen:
                    seen.add(norm)
                    norm_items.append(norm)
            # Apply post-filters to enforce umbrella-category rule
            norm_items = apply_post_filters(norm_items)
            if sleep > 0:
                time.sleep(sleep)
            return norm_items
        except Exception as exc:
            last_exc = exc
            wait = min(2 ** attempt, 30)
            print(f"DeepSeek extract failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
            if attempt < retries:
                time.sleep(wait)
    # Fallback: simple heuristic if model call repeatedly fails
    print(f"Falling back to heuristic extraction due to error: {last_exc}", file=sys.stderr)
    return heuristic_extract(reasoning_text)


def heuristic_extract(reasoning_text: str) -> List[str]:
    # Very conservative: look for comma/semicolon/line separated lists after cue words
    text = _norm_text(reasoning_text)
    cands: List[str] = []
    # Capture phrases following keywords like 'differential includes', 'consider', 'possible', 'vs'
    patterns = [
        r"differential (?:diagnosis )?(?:includes|include|is|are)[:\-]\s*(.+)",
        r"consider(?:ing)?\s+(.+)",
        r"possibilit(?:y|ies)\s+include[:\-]\s*(.+)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            segment = m.group(1)
            for token in re.split(r"[,;\n]|\s+vs\.?\s+", segment):
                token = token.strip()
                token = re.sub(r"\b(such as|like|including)\b", "", token).strip()
                norm = _norm_diagnosis_name(token)
                if norm:
                    cands.append(norm)
    # Deduplicate while preserving order
    out: List[str] = []
    seen = set()
    for it in cands:
        if it not in seen:
            seen.add(it)
            out.append(it)
    out = out[:50]
    return apply_post_filters(out)


def apply_post_filters(items: List[str]) -> List[str]:
    """Apply simple post-processing rules on extracted items.

    - Remove vague umbrella categories when more specific diagnoses are present.
      For now, we implement a conservative rule for 'retroperitoneal process':
      drop it if there are any other items present.
    """
    cleaned = [it for it in items if it]
    # Remove exact duplicates while preserving order
    seen = set()
    deduped: List[str] = []
    for it in cleaned:
        if it not in seen:
            seen.add(it)
            deduped.append(it)
    # Umbrella: retroperitoneal process
    if "retroperitoneal process" in seen and len(deduped) > 1:
        deduped = [it for it in deduped if it != "retroperitoneal process"]
    return deduped


def order_items_by_first_occurrence(reasoning_text: str, items: List[str]) -> List[str]:
    """Order extracted items by the index of their first occurrence in the reasoning text.

    We normalize both text and items similarly and search using simple substring indices.
    Items not found keep their relative order at the end.
    """
    if not items:
        return items
    text_norm = _norm_text(reasoning_text)
    indexed: List[Tuple[int, str]] = []
    not_found: List[str] = []
    for it in items:
        it_norm = _norm_diagnosis_name(it)
        idx = text_norm.find(it_norm)
        if idx >= 0:
            indexed.append((idx, it))
        else:
            not_found.append(it)
    indexed.sort(key=lambda x: x[0])
    ordered = [it for _, it in indexed] + not_found
    # Preserve deduped nature
    seen: set = set()
    result: List[str] = []
    for it in ordered:
        if it not in seen:
            seen.add(it)
            result.append(it)
    return result


@dataclass
class ExtractResult:
    pmcid: str
    sample_index: int
    items: List[str]
    count: int
    correct_in_diff: Optional[bool]


PRESENCE_SYSTEM_PROMPT = (
    "You are a careful clinical NLP verifier. Given a target diagnosis and a list of extracted differential diagnoses, "
    "determine whether the target is represented by any item in the list, accounting for synonyms, abbreviations, plural/singular, and umbrella/specific relations. "
    "Return strictly valid JSON of the shape: {\"present\": true|false}. No explanations."
)


def build_presence_prompt(target: str, items: List[str]) -> str:
    target_norm = _norm_diagnosis_name(target)
    items_norm = [_norm_diagnosis_name(x) for x in items if isinstance(x, str)]
    items_text = "\n".join(f"- {x}" for x in items_norm)
    return (
        "Target diagnosis: "
        + target_norm
        + "\nDifferential items (deduplicated):\n"
        + items_text
        + "\n\nQuestion: Is the target diagnosis represented by any of the listed items? Respond JSON only as {\"present\": true|false}."
    )


def llm_presence_check(client: Any, target: str, items: List[str], model: str = "deepseek-chat", retries: int = 2, sleep: float = 0.0) -> Optional[bool]:
    """Ask DeepSeek if target diagnosis is present in items list (synonyms/umbrella aware).

    Returns True/False if successful, or None on failure (caller can fallback).
    """
    if not target or not items:
        return None
    messages = [
        {"role": "system", "content": PRESENCE_SYSTEM_PROMPT},
        {"role": "user", "content": build_presence_prompt(target, items)},
    ]
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, stream=False)
            content = resp.choices[0].message.content or ""
            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                content = m.group(0)
            data = json.loads(content)
            present = data.get("present", None)
            if isinstance(present, bool):
                if sleep > 0:
                    time.sleep(sleep)
                return present
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            wait = min(2 ** attempt, 15)
            if attempt < retries:
                time.sleep(wait)
    # failed
    if last_exc:
        print(f"DeepSeek presence check failed: {last_exc}", file=sys.stderr)
    return None


def process_row(row: pd.Series, client: Any, cache: Dict[Tuple[str, int], Dict[str, Any]], cache_path: str, retries: int, sleep: float, use_api: bool) -> ExtractResult:
    pmcid = str(row.get("pmcid", ""))
    try:
        sample_index = int(row.get("sample_index", 0))
    except Exception:
        sample_index = 0

    key = (pmcid, sample_index)
    # Use cache when available
    if key in cache:
        obj = cache[key]
        items = obj.get("items", []) or []
        if not isinstance(items, list):
            items = []
        items = [_norm_diagnosis_name(str(x)) for x in items if isinstance(x, (str,))]
    else:
        reasoning = str(row.get("reasoning_trace") or "")
        posthoc = str(row.get("posthoc_reasoning_trace") or "")
        text = (reasoning + "\n\n" + posthoc).strip()
        if not text:
            items = []
        else:
            if use_api and client is not None:
                items = call_deepseek_extract(client, text, retries=retries, sleep=sleep)
            else:
                items = heuristic_extract(text)
        # append to cache
        record = {
            "pmcid": pmcid,
            "sample_index": sample_index,
            "items": items,
        }
        append_cache(cache_path, record)
        cache[key] = record

    # Order items by first occurrence in the combined text
    reasoning = str(row.get("reasoning_trace") or "")
    posthoc = str(row.get("posthoc_reasoning_trace") or "")
    text = (reasoning + "\n\n" + posthoc).strip()
    if text and items:
        items = order_items_by_first_occurrence(text, items)

    count = len(items)

    # Determine if correct answer is in differential using LLM presence check when available.
    true_dx = row.get("true_diagnosis")
    predicted_dx = row.get("predicted_diagnosis")
    verified_correct = bool(row.get("verified_correct", False))
    correct_in_diff: Optional[bool] = None
    # Prefer LLM presence check if API available
    if isinstance(true_dx, str) and true_dx.strip():
        present_true: Optional[bool] = None
        if use_api and client is not None and items:
            present_true = llm_presence_check(client, true_dx, items)
        # If verified correct and predicted available, also check predicted via LLM
        present_pred: Optional[bool] = None
        if verified_correct and isinstance(predicted_dx, str) and predicted_dx.strip():
            if use_api and client is not None and items:
                present_pred = llm_presence_check(client, predicted_dx, items)
        # Combine
        if present_true is not None or present_pred is not None:
            correct_in_diff = bool((present_true is True) or (present_pred is True))
        else:
            # Fallback heuristic containment
            norm_true = _norm_diagnosis_name(true_dx)
            in_list = False
            if norm_true in items:
                in_list = True
            else:
                for it in items:
                    if norm_true and (norm_true in it or it in norm_true):
                        in_list = True
                        break
            # If verified correct and predicted exists, also allow predicted heuristic containment
            if not in_list and verified_correct and isinstance(predicted_dx, str) and predicted_dx.strip():
                norm_pred = _norm_diagnosis_name(predicted_dx)
                if norm_pred in items:
                    in_list = True
                else:
                    for it in items:
                        if norm_pred and (norm_pred in it or it in norm_pred):
                            in_list = True
                            break
            correct_in_diff = in_list
    else:
        correct_in_diff = None

    return ExtractResult(pmcid=pmcid, sample_index=sample_index, items=items, count=count, correct_in_diff=correct_in_diff)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract differential breadth from reasoning traces and correlate with accuracy")
    parser.add_argument("--input", default="results.csv", help="Path to results.csv from eval.py")
    parser.add_argument("--per_case", default="results.per_case.csv", help="Path to per-case CSV for difficulty/accuracy")
    parser.add_argument("--out_csv", default="", help="Optional separate per-trace output CSV path (default: update input in place)")
    parser.add_argument("--summary", default=SUMMARY_TXT, help="Summary text output path")
    parser.add_argument("--cache", default=DIFF_CACHE_DEFAULT, help="Path to JSONL cache of extracted differentials")
    parser.add_argument("--out_dir", default="", help="Optional base output directory; overrides default summary/cache locations if provided")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek"], help="LLM provider (deepseek only)")
    parser.add_argument("--model", default="deepseek-chat", help="Model name for extraction")
    parser.add_argument("--workers", type=int, default=100, help="Parallel workers for API calls")
    parser.add_argument("--retries", type=int, default=3, help="Retries for API calls")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between API calls (per call)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows for quick run (0 = all)")
    parser.add_argument("--dry_run", action="store_true", help="Skip API calls and use heuristic extraction only")
    args = parser.parse_args()

    # If out_dir provided, override summary/cache defaults when user left them as defaults
    if args.out_dir:
        # Only override when user didn't explicitly set custom paths (still default constants)
        if args.summary == SUMMARY_TXT:
            args.summary = os.path.join(args.out_dir, "analysis_outputs", "differential_count_correlation.txt")
        if args.cache == DIFF_CACHE_DEFAULT:
            args.cache = os.path.join(args.out_dir, "analysis_outputs", "differential_cache.jsonl")

    # Ensure analysis_outputs exists for summary and cache
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)

    # Load inputs
    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(args.input)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    # client and cache
    client = None
    use_api = not args.dry_run
    if use_api:
        try:
            client = make_client(args.provider)
        except Exception as exc:
            print(f"API client init failed ({exc}); falling back to heuristic extraction only.", file=sys.stderr)
            use_api = False
    cache = load_cache(args.cache)

    # Extract per row
    tasks: List[Tuple[int, pd.Series]] = list(df.iterrows())

    results: List[ExtractResult] = []
    if args.workers and args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut_to_idx = {
                ex.submit(process_row, row, client, cache, args.cache, args.retries, args.sleep, use_api): idx
                for idx, row in tasks
            }
            for fut in as_completed(fut_to_idx):
                try:
                    results.append(fut.result())
                except Exception as exc:  # pragma: no cover
                    print(f"Row processing failed: {exc}", file=sys.stderr)
    else:
        for _, row in tasks:
            try:
                results.append(process_row(row, client, cache, args.cache, args.retries, args.sleep, use_api))
            except Exception as exc:  # pragma: no cover
                print(f"Row processing failed: {exc}", file=sys.stderr)

    # Build per-trace dataframe
    out_rows: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, int], ExtractResult] = {}
    for r in results:
        by_key[(r.pmcid, r.sample_index)] = r

    # Attach original fields including correctness
    for _, row in df.iterrows():
        pmcid = str(row.get("pmcid", ""))
        try:
            sample_index = int(row.get("sample_index", 0))
        except Exception:
            sample_index = 0
        key = (pmcid, sample_index)
        extr = by_key.get(key)
        if not extr:
            continue
        verified_correct = bool(row.get("verified_correct", False))
        true_dx = row.get("true_diagnosis")
        predicted_dx = row.get("predicted_diagnosis")
        # Compute reasoning length on the fly (characters of reasoning + posthoc)
        reasoning = str(row.get("reasoning_trace") or "")
        posthoc = str(row.get("posthoc_reasoning_trace") or "")
        combined_reasoning = (reasoning + "\n\n" + posthoc).strip()
        reasoning_length = len(combined_reasoning)
        # Per-differential ratio; use NaN when diff_count == 0
        reason_len_per_diff = (float(reasoning_length) / extr.count) if extr.count > 0 else float("nan")
        out_rows.append({
            "pmcid": pmcid,
            "sample_index": sample_index,
            "true_diagnosis": true_dx,
            "predicted_diagnosis": predicted_dx,
            "verified_correct": int(1 if verified_correct else 0),
            "trace_correct": int(1 if verified_correct else 0),
            "diff_items": json.dumps(extr.items, ensure_ascii=False),
            "diff_count": extr.count,
            "correct_in_diff": (None if extr.correct_in_diff is None else int(1 if extr.correct_in_diff else 0)),
            "reasoning_length": reasoning_length,
            "reason_len_per_diff": reason_len_per_diff,
        })

    df_out = pd.DataFrame(out_rows)

    # Merge per-case difficulty
    difficulty_cols = []
    if os.path.isfile(args.per_case):
        df_case = pd.read_csv(args.per_case)
        # Compute difficulty as round((1-accuracy)*10) with bounds [0,10]
        if "accuracy" in df_case.columns:
            diff = (1.0 - df_case["accuracy"].astype(float)) * 10.0
            df_case["difficulty"] = diff.round().astype(int).clip(lower=0, upper=10)
            difficulty_cols = ["difficulty", "accuracy", "n_correct", "total_samples"]
        else:
            # Fallback if only n_correct and total_samples
            if {"n_correct", "total_samples"}.issubset(df_case.columns):
                acc = df_case["n_correct"].astype(float) / df_case["total_samples"].astype(float)
                diff = (1.0 - acc) * 10.0
                df_case["accuracy"] = acc
                df_case["difficulty"] = diff.round().astype(int).clip(lower=0, upper=10)
                difficulty_cols = ["difficulty", "accuracy", "n_correct", "total_samples"]
        df_case_small = df_case[["pmcid"] + difficulty_cols].copy()
        df_out = df_out.merge(df_case_small, on="pmcid", how="left")
    else:
        print(f"Per-case file not found: {args.per_case}. Skipping difficulty merge.", file=sys.stderr)

    # Update original results.csv in place unless a separate out_csv is specified
    # Merge by (pmcid, sample_index)
    merge_keys = ["pmcid", "sample_index"]
    merge_cols = merge_keys + ["trace_correct", "diff_items", "diff_count", "correct_in_diff"]

    # Clean up any prior analysis columns (including leftover suffixed variants) on the left df
    analysis_cols = ["trace_correct", "diff_items", "diff_count", "correct_in_diff"]
    suffixes = ["_x", "_y"]
    drop_cols: List[str] = []
    for col in list(df.columns):
        if col in analysis_cols:
            drop_cols.append(col)
            continue
        for base in analysis_cols:
            if col == f"{base}_x" or col == f"{base}_y":
                drop_cols.append(col)
                break
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Also ensure we don't have duplicate rows per (pmcid, sample_index) on the right
    right_merge_df = df_out[merge_cols].drop_duplicates(subset=merge_keys, keep="last")

    df_updated = df.merge(right_merge_df, on=merge_keys, how="left")

    # Write per-trace results
    if args.out_csv:
        # Optional separate export
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df_out.to_csv(args.out_csv, index=False)
    else:
        # In-place update of results.csv
        df_updated.to_csv(args.input, index=False)

    # Correlation analyses
    lines: List[str] = []
    lines.append("Differential count vs accuracy analyses\n")
    lines.append(f"Rows: {len(df_out)}\n")

    # Unadjusted correlations
    try:
        y = df_out["verified_correct"].astype(float).values
        x = df_out["diff_count"].astype(float).values
        if len(x) > 1 and np.nanstd(x) > 0:
            pearson = np.corrcoef(x, y)[0, 1]
        else:
            pearson = float("nan")
        # Spearman
        xr = pd.Series(x).rank(method="average").values
        yr = pd.Series(y).rank(method="average").values
        if np.nanstd(xr) > 0 and np.nanstd(yr) > 0:
            spearman = np.corrcoef(xr, yr)[0, 1]
        else:
            spearman = float("nan")
        lines.append(f"Unadjusted Pearson r(count, correct): {pearson:.4f}")
        lines.append(f"Unadjusted Spearman rho(count, correct): {spearman:.4f}\n")

        # Reasoning length vs accuracy (unadjusted)
        x_len = df_out["reasoning_length"].astype(float).values
        if len(x_len) > 1 and np.nanstd(x_len) > 0:
            pearson_len = np.corrcoef(x_len, y)[0, 1]
        else:
            pearson_len = float("nan")
        xr_len = pd.Series(x_len).rank(method="average").values
        if np.nanstd(xr_len) > 0 and np.nanstd(yr) > 0:
            spearman_len = np.corrcoef(xr_len, yr)[0, 1]
        else:
            spearman_len = float("nan")
        lines.append(f"Unadjusted Pearson r(reasoning_length, correct): {pearson_len:.4f}")
        lines.append(f"Unadjusted Spearman rho(reasoning_length, correct): {spearman_len:.4f}\n")

        # Reasoning length per differential vs accuracy (unadjusted)
        x_len_per = df_out["reason_len_per_diff"].astype(float).values
        x_len_per_valid = x_len_per[~np.isnan(x_len_per)]
        y_len_per_valid = y[~np.isnan(x_len_per)]
        if len(x_len_per_valid) > 1 and np.nanstd(x_len_per_valid) > 0:
            pearson_len_per = np.corrcoef(x_len_per_valid, y_len_per_valid)[0, 1]
        else:
            pearson_len_per = float("nan")
        xr_len_per = pd.Series(x_len_per_valid).rank(method="average").values if len(x_len_per_valid) else np.array([])
        yr_len_per = pd.Series(y_len_per_valid).rank(method="average").values if len(y_len_per_valid) else np.array([])
        if len(xr_len_per) > 1 and np.nanstd(xr_len_per) > 0 and np.nanstd(yr_len_per) > 0:
            spearman_len_per = np.corrcoef(xr_len_per, yr_len_per)[0, 1]
        else:
            spearman_len_per = float("nan")
        lines.append(f"Unadjusted Pearson r(reasoning_length_per_diff, correct): {pearson_len_per:.4f}")
        lines.append(f"Unadjusted Spearman rho(reasoning_length_per_diff, correct): {spearman_len_per:.4f}\n")
    except Exception as exc:  # pragma: no cover
        lines.append(f"Unadjusted correlations failed: {exc}\n")

    # Adjusted via logistic regression on rows with difficulty available
    if LogisticRegression is not None and "difficulty" in df_out.columns:
        try:
            df_lr = df_out.dropna(subset=["diff_count", "verified_correct", "difficulty"]).copy()
            if len(df_lr) > 5:
                X = df_lr[["diff_count", "difficulty"]].astype(float).values
                yy = df_lr["verified_correct"].astype(int).values
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X, yy)
                coef_count = clf.coef_[0][0]
                coef_difficulty = clf.coef_[0][1]
                intercept = clf.intercept_[0]
                # pseudo ROC AUC
                prob = clf.predict_proba(X)[:, 1]
                auc = float("nan")
                if roc_auc_score is not None:
                    try:
                        auc = roc_auc_score(yy, prob)
                    except Exception:
                        pass
                lines.append("Logistic regression (correct ~ diff_count + difficulty):")
                lines.append(f"  intercept={intercept:.4f}, coef(diff_count)={coef_count:.4f}, coef(difficulty)={coef_difficulty:.4f}")
                lines.append(f"  ROC AUC (in-sample, descriptive): {auc:.4f}\n")
            else:
                lines.append("Not enough rows for logistic regression.\n")
        except Exception as exc:  # pragma: no cover
            lines.append(f"Logistic regression failed: {exc}\n")
    else:
        lines.append("sklearn not available or difficulty missing; skipping adjusted model.\n")

    # Adjusted models for reasoning length metrics
    if LogisticRegression is not None and "difficulty" in df_out.columns:
        # correct ~ reasoning_length + difficulty
        try:
            df_lr2 = df_out.dropna(subset=["reasoning_length", "verified_correct", "difficulty"]).copy()
            if len(df_lr2) > 5:
                X2 = df_lr2[["reasoning_length", "difficulty"]].astype(float).values
                yy2 = df_lr2["verified_correct"].astype(int).values
                clf2 = LogisticRegression(max_iter=1000)
                clf2.fit(X2, yy2)
                coef_len = clf2.coef_[0][0]
                coef_diff = clf2.coef_[0][1]
                intercept2 = clf2.intercept_[0]
                prob2 = clf2.predict_proba(X2)[:, 1]
                auc2 = float("nan")
                if roc_auc_score is not None:
                    try:
                        auc2 = roc_auc_score(yy2, prob2)
                    except Exception:
                        pass
                lines.append("Logistic regression (correct ~ reasoning_length + difficulty):")
                lines.append(f"  intercept={intercept2:.4f}, coef(reasoning_length)={coef_len:.4f}, coef(difficulty)={coef_diff:.4f}")
                lines.append(f"  ROC AUC (in-sample, descriptive): {auc2:.4f}\n")
            else:
                lines.append("Not enough rows for logistic regression (reasoning_length).\n")
        except Exception as exc:
            lines.append(f"Logistic regression (reasoning_length) failed: {exc}\n")

        # correct ~ reasoning_length_per_diff + difficulty
        try:
            df_lr3 = df_out.dropna(subset=["reason_len_per_diff", "verified_correct", "difficulty"]).copy()
            if len(df_lr3) > 5:
                X3 = df_lr3[["reason_len_per_diff", "difficulty"]].astype(float).values
                yy3 = df_lr3["verified_correct"].astype(int).values
                clf3 = LogisticRegression(max_iter=1000)
                clf3.fit(X3, yy3)
                coef_len_per = clf3.coef_[0][0]
                coef_diff3 = clf3.coef_[0][1]
                intercept3 = clf3.intercept_[0]
                prob3 = clf3.predict_proba(X3)[:, 1]
                auc3 = float("nan")
                if roc_auc_score is not None:
                    try:
                        auc3 = roc_auc_score(yy3, prob3)
                    except Exception:
                        pass
                lines.append("Logistic regression (correct ~ reasoning_length_per_diff + difficulty):")
                lines.append(f"  intercept={intercept3:.4f}, coef(reasoning_length_per_diff)={coef_len_per:.4f}, coef(difficulty)={coef_diff3:.4f}")
                lines.append(f"  ROC AUC (in-sample, descriptive): {auc3:.4f}\n")
            else:
                lines.append("Not enough rows for logistic regression (reasoning_length_per_diff).\n")
        except Exception as exc:
            lines.append(f"Logistic regression (reasoning_length_per_diff) failed: {exc}\n")

    # Correlation between difficulty and diff_count (descriptive)
    if "difficulty" in df_out.columns:
        try:
            dd = df_out.dropna(subset=["difficulty", "diff_count"]).copy()
            n_pairs = len(dd)
            if n_pairs > 1:
                a = dd["diff_count"].astype(float).values
                b = dd["difficulty"].astype(float).values
                if np.nanstd(a) > 0 and np.nanstd(b) > 0:
                    pearson_dc = np.corrcoef(a, b)[0, 1]
                else:
                    pearson_dc = float("nan")
                ar = pd.Series(a).rank(method="average").values
                br = pd.Series(b).rank(method="average").values
                if np.nanstd(ar) > 0 and np.nanstd(br) > 0:
                    spearman_dc = np.corrcoef(ar, br)[0, 1]
                else:
                    spearman_dc = float("nan")
                lines.append(f"Correlation between diff_count and difficulty (n={n_pairs}):")
                lines.append(f"  Pearson r: {pearson_dc:.4f}")
                lines.append(f"  Spearman rho: {spearman_dc:.4f}\n")
            else:
                lines.append("Correlation between diff_count and difficulty: not enough paired rows.\n")
        except Exception as exc:  # pragma: no cover
            lines.append(f"Difficulty-count correlation failed: {exc}\n")

    # Accuracy conditional on presence of true diagnosis in differential
    if "correct_in_diff" in df_out.columns:
        sub = df_out.dropna(subset=["correct_in_diff"]).copy()
        if not sub.empty:
            grp = sub.groupby("correct_in_diff")
            for key, g in grp:
                acc = g["verified_correct"].mean() if len(g) else float("nan")
                n = len(g)
                lines.append(f"Accuracy when correct_in_diff={int(key)}: n={n}, acc={acc:.3f}")
            lines.append("")

    # If per-case file exists, augment with aggregates and save back
    if os.path.isfile(args.per_case):
        try:
            # Compute per-case aggregates from df_out
            agg = df_out.groupby("pmcid").agg(
                diff_count_mean=("diff_count", "mean"),
                diff_count_median=("diff_count", "median"),
                correct_in_diff_rate=("correct_in_diff", lambda s: np.nan if s.isna().all() else s.dropna().mean()),
            ).reset_index()

            df_case = pd.read_csv(args.per_case)
            df_case_updated = df_case.merge(agg, on="pmcid", how="left")
            df_case_updated.to_csv(args.per_case, index=False)
        except Exception as exc:  # pragma: no cover
            print(f"Per-case augmentation failed: {exc}", file=sys.stderr)

    # Save summary
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also print concise summary to stdout
    print("\n".join(lines))


if __name__ == "__main__":
    main()
