#!/usr/bin/env python3
"""Grade responses produced by generate_responses.py using the same 3-step
verification method used in eval.py.

Output: same JSON list as input but each item gains these fields:
  - extracted_answer: str (text inside the last <answer>...</answer> in full_response)
  - similarity_score: float (0-10)
  - is_correct: bool (similarity_score >= 8.0)

Requires: OPENAI_API_KEY in environment and the openai Python SDK (the project
already uses OpenAI() in `eval.py`).

Usage:
    python grade_responses.py --input results/vars/responses.json --output results/vars/responses.graded.json

Options:
    --limit N    Limit processing to the first N items (0 = all)

"""
import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
try:  # progress bar support
    from tqdm import tqdm  # type: ignore
except Exception:  # fallback dummy
    def tqdm(iterable=None, total=None, desc=None, unit=None):
        class _NoOp:
            def update(self, n=1):
                pass
            def close(self):
                pass
        return _NoOp()

try:
    # follow project pattern (eval.py uses OpenAI from openai)
    from openai import OpenAI  # type: ignore
except Exception as exc:
    print("Failed to import OpenAI SDK. Make sure it's installed and available.", file=sys.stderr)
    raise


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
CASE_BLOCK_RE = re.compile(
    r"-{40}\nCASE PRESENTATION\n-{40}\n(.*?)\n-{40}\nOUTPUT TEMPLATE\n-{40}",
    re.DOTALL,
)


def extract_answer(full_response: str) -> str:
    """Return the text of the last <answer>...</answer> block or empty string."""
    if not full_response:
        return ""
    matches = ANSWER_RE.findall(full_response)
    if not matches:
        return ""
    return matches[-1].strip().replace("...the name of the disease/entity...", "").strip()


def extract_case(original_message_content: str) -> str:
    """Extract the case presentation text between the CASE PRESENTATION and OUTPUT TEMPLATE markers."""
    if not original_message_content:
        return ""
    m = CASE_BLOCK_RE.search(original_message_content)
    if not m:
        # fallback: try to find the two marker lines simply
        start_marker = "----------------------------------------\nCASE PRESENTATION\n----------------------------------------\n"
        end_marker = "\n----------------------------------------\nOUTPUT TEMPLATE\n----------------------------------------"
        try:
            start = original_message_content.index(start_marker) + len(start_marker)
            end = original_message_content.index(end_marker, start)
            return original_message_content[start:end].strip()
        except Exception:
            return ""
    return m.group(1).strip()


def _parse_similarity_score(text: str) -> float:
    """Parse a 0–10 numeric score from model output. Conservative fallback to 0.0."""
    try:
        m = re.search(r"(?<!\d)(?:10(?:\.0+)?|\d(?:\.\d+)?)(?!\d)", text.strip())
        if not m:
            return 0.0
        val = float(m.group(0))
        if val < 0:
            return 0.0
        if val > 10:
            return 10.0
        return val
    except Exception:
        return 0.0


def call_openai_chat(client: Any, prompt: str, model: str = "deepseek-chat", retries: int = 3) -> str:
    """Call OpenAI chat client.chat.completions.create like eval.py does, with simple retries."""
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], stream=False)
            # mirror eval.py behavior: return content directly
            return resp.choices[0].message.content
        except Exception as exc:  # broad but fine for retries
            wait = min(2 ** attempt, 30)
            print(f"OpenAI call failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(wait)
    raise RuntimeError("Unreachable")


VERIFY_DESCRIBE_TRUE_TEMPLATE = (
    "Here is a case presentation and the diagnosis. Describe the diagnosis.\n\n"
    "----------------------------------------\n"
    "CASE\n"
    "----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\n"
    "DIAGNOSIS\n"
    "----------------------------------------\n"
    "{actual_diagnosis}"
)


VERIFY_DESCRIBE_PREDICTED_TEMPLATE = (
    "Here is a case presentation and the diagnosis. Describe the diagnosis.\n\n"
    "----------------------------------------\n"
    "CASE\n"
    "----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\n"
    "DIAGNOSIS\n"
    "----------------------------------------\n"
    "{predicted_diagnosis}"
)


VERIFY_COMPARE_TEMPLATE = (
    "Here is a case, a predicted diagnosis, and the true diagnosis.\n"
    "How similar are the two diagnoses? Answer only a number from 0-10.\n\n"
    "----------------------------------------\n"
    "CASE\n"
    "----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\n\n"
    "PREDICTED DIAGNOSIS\n"
    "----------------------------------------\n"
    "{predicted_description}\n\n"
    "----------------------------------------\n"
    "TRUE DIAGNOSIS\n"
    "----------------------------------------\n"
    "{true_description}"
)


def grade_item(client: Any, item: Dict[str, Any], model: str, sleep_between: float = 0.0) -> Dict[str, Any]:
    original = item.get("original_message", {})
    full_response = item.get("full_response", "")

    extracted = extract_answer(full_response)
    if not extracted:
        # No answer extracted; return with 0 score
        item_out = dict(item)
        item_out["extracted_answer"] = ""
        item_out["similarity_score"] = 0.0
        item_out["is_correct"] = False
        return item_out
    case_text = extract_case(original.get("content", ""))
    true_diag = item.get("gold_answer", "") or item.get("gold", "") or ""

    # Step 1: describe true diagnosis
    true_prompt = VERIFY_DESCRIBE_TRUE_TEMPLATE.format(case_prompt=case_text, actual_diagnosis=true_diag) + "\n\nRespond in 2-3 short sentences."
    true_description = call_openai_chat(client, true_prompt, model=model)
    if sleep_between:
        time.sleep(sleep_between)

    # Step 2: describe predicted diagnosis
    pred_prompt = VERIFY_DESCRIBE_PREDICTED_TEMPLATE.format(case_prompt=case_text, predicted_diagnosis=extracted or "Unknown") + "\n\nRespond in 2-3 short sentences."
    predicted_description = call_openai_chat(client, pred_prompt, model=model)
    if sleep_between:
        time.sleep(sleep_between)

    # Step 3: numeric similarity
    compare_prompt = VERIFY_COMPARE_TEMPLATE.format(
        case_prompt=case_text,
        predicted_description=predicted_description,
        true_description=true_description,
    )
    rating_raw = call_openai_chat(client, compare_prompt, model=model)
    score = _parse_similarity_score(rating_raw)
    is_correct = score >= 8.0

    # Attach fields
    item_out = dict(item)
    item_out["extracted_answer"] = extracted
    item_out["similarity_score"] = score
    item_out["is_correct"] = bool(is_correct)
    # optional debug fields
    item_out["_rating_raw"] = rating_raw
    item_out["_true_description"] = true_description
    item_out["_predicted_description"] = predicted_description
    return item_out


THREAD_LOCAL = threading.local()


def get_thread_client() -> Any:
    """Return a thread-local OpenAI client, creating one per worker thread."""
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAI()
        THREAD_LOCAL.client = client
    return client


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Grade generated responses using OpenAI verification (3-step)")
    p.add_argument("--input", required=True, help="Path to JSON file produced by generate_responses (list of items)")
    p.add_argument("--output", required=True, help="Path to write augmented JSON")
    p.add_argument("--model", default="deepseek-chat", help="Verifier model name")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between verifier calls to avoid rate limits")
    p.add_argument("--limit", type=int, default=0, help="Limit number of items to grade (0 = all)")
    p.add_argument("--workers", type=int, default=8, help="Number of concurrent worker threads to use")
    args = p.parse_args(argv)

    in_path = args.input
    out_path = args.output
    model = args.model
    limit = int(args.limit or 0)

    if not os.path.isfile(in_path):
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    with open(in_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Input JSON must be a list of response objects.", file=sys.stderr)
        sys.exit(2)

    # Use a thread-local OpenAI client per worker and parallelize grading
    workers = max(1, int(getattr(args, "workers", 8)))

    if limit > 0:
        data = data[:limit]

    out_list: List[Optional[Dict[str, Any]]] = [None] * len(data)
    total = len(data)
    print(f"Grading {total} items using verifier model {model}... (limit={limit}) workers={workers}")

    def _worker(index: int, item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        client = get_thread_client()
        graded = grade_item(client, item, model=model, sleep_between=args.sleep)
        return index, graded

    # Submit tasks
    # Progress bar over completed futures
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, i, item): i for i, item in enumerate(data)}
        pbar = tqdm(total=total, desc="Grading", unit="item")
        for fut in as_completed(futures):
            i_sub = futures[fut]
            try:
                idx, graded = fut.result()
            except Exception as exc:
                # Keep a single concise error line; progress bar will still advance
                print(f"Failed grading item {i_sub+1}/{total}: {exc}", file=sys.stderr)
                failed = dict(data[i_sub])
                failed["extracted_answer"] = extract_answer(data[i_sub].get("full_response", ""))
                failed["similarity_score"] = 0.0
                failed["is_correct"] = False
                failed["_error"] = str(exc)
                out_list[i_sub] = failed
            else:
                out_list[idx] = graded
            finally:
                pbar.update(1)
        pbar.close()

    # Write output JSON
    with open(out_path, "w") as f:
        json.dump(out_list, f, indent=2)

    print(f"Wrote {len(out_list)} graded items to {out_path}")
    
    # Calculate and display overall accuracy
    correct_count = sum(1 for item in out_list if item and item.get("is_correct", False))
    total_count = len(out_list)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
    print(f"\n{'='*50}")
    print(f"Overall Accuracy: {correct_count}/{total_count} ({accuracy:.2f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
