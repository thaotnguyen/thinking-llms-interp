import argparse
import asyncio
import concurrent.futures
import json
import os
import re
from typing import Dict, List, Tuple

from utils.utils import chat_batch


try:
    from tqdm.auto import tqdm
except ImportError as exc:
    raise ImportError("tqdm is required for progress reporting") from exc


MODEL_SPECS: List[Tuple[str, str]] = [
    ("thinking", "Thinking Model"),
    ("base", "Base Model"),
    ("hybrid", "Hybrid Model"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate rolling results with LLM judges")
    parser.add_argument(
        "--prefix",
        required=False,
        help="Rolling file prefix. Accepts absolute path or name inside the rolling directory.",
    )
    parser.add_argument(
        "--rolling-dir",
        type=str,
        default=None,
        help="Directory containing rolling outputs (defaults to results/rolling next to this script).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-5",
        help="Judge model to query via chat_batch.",
    )
    parser.add_argument(
        "--max-judge-tokens",
        type=int,
        default=100,
        help="Maximum tokens returned by the judge model.",
    )
    return parser.parse_known_args()[0]


def _default_rolling_dir() -> str:
    here = os.path.dirname(__file__)
    return os.path.join(here, "results", "rolling")


def _resolve_prefix(raw_prefix: str, rolling_dir: str) -> str:
    if raw_prefix is None:
        raise ValueError("raw_prefix must not be None")
    if os.path.isabs(raw_prefix):
        prefix = raw_prefix
    else:
        prefix = os.path.join(rolling_dir, raw_prefix)
    if prefix.endswith(".jsonl"):
        prefix = prefix[:-6]
    return prefix


def _list_ordered_files(prefix: str) -> List[str]:
    directory = os.path.dirname(prefix) or "."
    base = os.path.basename(prefix)
    assert base, "Prefix must include a filename component"

    files: List[str] = []
    legacy = os.path.join(directory, f"{base}.jsonl")
    if os.path.exists(legacy):
        files.append(legacy)

    part_pattern = re.compile(rf"^{re.escape(base)}_(\d+)\.jsonl$")
    part_paths: List[str] = []
    for name in os.listdir(directory):
        match = part_pattern.match(name)
        if match:
            part_paths.append(os.path.join(directory, name))

    part_paths.sort(key=lambda path: int(part_pattern.match(os.path.basename(path)).group(1)))
    files.extend(part_paths)
    assert files, f"No rolling files found for prefix {prefix}"
    return files


def _list_all_rollings(rolling_dir: str) -> List[str]:
    files: Dict[str, List[str]] = {}
    for name in os.listdir(rolling_dir):
        if not name.endswith(".jsonl"):
            continue
        if not name.startswith("rolling_"):
            continue
        full_path = os.path.join(rolling_dir, name)
        prefix, part = _split_prefix_parts(full_path)
        files.setdefault(prefix, []).append(full_path if part is None else (part, full_path))

    grouped: Dict[str, List[str]] = {}
    for prefix, entries in files.items():
        sorted_paths: List[str] = []
        parts = [e for e in entries if isinstance(e, tuple)]
        legacy = [e for e in entries if isinstance(e, str)]
        if legacy:
            sorted_paths.extend(sorted(legacy))
        for _, path in sorted(parts, key=lambda x: x[0]):
            sorted_paths.append(path)
        grouped[prefix] = sorted_paths

    assert grouped, f"No rolling files found in {rolling_dir}"
    return sorted(grouped.items(), key=lambda kv: kv[0])


def clean_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_chat_batch(prompts, model_name: str, max_tokens: int, **kwargs):
    async def _run():
        return await chat_batch(prompts, model=model_name, max_tokens=max_tokens, **kwargs)

    try:
        asyncio.get_running_loop()

        def _thread_runner():
            return asyncio.run(_run())

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_thread_runner)
            return future.result()
    except RuntimeError:
        return asyncio.run(_run())


def _build_judge_prompt(question: str, correct_answer: str, model_answer: str) -> str:
    return (
        "Please evaluate whether the following answer to a math problem is correct.\n\n"
        f"Question: {question}\n\n"
        f"Correct answer: {correct_answer}\n\n"
        f"Model's answer: {model_answer}\n\n"
        "First, extract the final numerical answer from both the correct answer and model's answer.\n"
        "Then determine if the model's final numerical answer is equivalent to the correct final numerical answer.\n"
        "Just answer YES if the model's answer is correct, or NO if it's incorrect. Nothing else.\n"
    )


def _process_file(path: str, *, judge_model: str, max_tokens: int) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    temp_path = f"{path}.tmp"
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as src:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))

    prompts: Dict[str, List[str]] = {key: [] for key, _ in MODEL_SPECS}
    record_indices: Dict[str, List[int]] = {key: [] for key, _ in MODEL_SPECS}
    original_flags: Dict[str, List[object]] = {key: [] for key, _ in MODEL_SPECS}

    for idx, record in enumerate(tqdm(records, desc=f"Prep {os.path.basename(path)}", unit="rec", leave=False)):
        assert "question" in record, "Record missing question"
        assert "gold_answer" in record, "Record missing gold_answer"
        assert "answers" in record, "Record missing answers"
        question = str(record["question"])
        gold = str(record["gold_answer"])
        answers = record["answers"]
        assert isinstance(answers, dict), "answers must be a dict"
        record.setdefault("judges", {})

        existing_judges = record.get("judges", {})
        assert isinstance(existing_judges, dict), "judges must be a dict"

        for key, _ in MODEL_SPECS:
            assert key in answers, f"Missing answer for {key}"
            answer_text = clean_answer(str(answers[key]))
            prompts[key].append(_build_judge_prompt(question, gold, answer_text))
            record_indices[key].append(idx)
            prev_entry = existing_judges.get(key)
            prev_correct = None
            if isinstance(prev_entry, dict):
                val = prev_entry.get("correct")
                if isinstance(val, bool):
                    prev_correct = val
            original_flags[key].append(prev_correct)

    changed_counts: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}
    final_correct_counts: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}

    for key, label in tqdm(MODEL_SPECS, desc="Judging", unit="model", leave=False):
        if not prompts[key]:
            continue
        responses = safe_chat_batch(prompts[key], model_name=judge_model, max_tokens=max_tokens)
        assert isinstance(responses, (list, tuple)) and len(responses) == len(prompts[key]), "Judge API returned invalid response"
        for response_index, raw in enumerate(responses):
            assert isinstance(raw, str), "Judge response must be string"
            is_correct = "yes" in raw.lower()
            rec_idx = record_indices[key][response_index]
            print(f"{label} [{rec_idx}] evaluated as: {raw}")
            record = records[rec_idx]
            judges = record.setdefault("judges", {})
            judges[key] = {"correct": bool(is_correct), "raw": raw}
            final_correct_counts[key] += int(bool(is_correct))
            prev = original_flags[key][response_index]
            if prev is None or bool(is_correct) != bool(prev):
                changed_counts[key] += 1

    with open(temp_path, "w", encoding="utf-8") as dst:
        for record in records:
            dst.write(json.dumps(record) + "\n")
    os.replace(temp_path, path)
    print(f"Updated {len(records)} records in {path}")
    return len(records), changed_counts, final_correct_counts


def main() -> None:
    args = parse_args()
    rolling_dir = args.rolling_dir or _default_rolling_dir()
    assert os.path.isdir(rolling_dir), f"Rolling directory not found: {rolling_dir}"
    if args.prefix:
        prefix = _resolve_prefix(args.prefix, rolling_dir)
        files = {prefix: _list_ordered_files(prefix)}
        print(f"Found {len(files[prefix])} files for prefix {prefix}")
    else:
        files = _list_all_rollings(rolling_dir)
        print(f"No prefix provided; processing all {len(files)} rolling groups in {rolling_dir}")

    total_records = 0
    aggregate_changed: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}
    aggregate_correct: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}
    per_prefix_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

    for prefix, paths in tqdm(files.items(), desc="Prefixes", unit="prefix"):
        prefix_total = 0
        prefix_changed: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}
        prefix_correct: Dict[str, int] = {key: 0 for key, _ in MODEL_SPECS}
        for path in tqdm(paths, desc=f"Files[{os.path.basename(prefix)}]", unit="file", leave=False):
            updated, changed_counts, correct_counts = _process_file(
                path,
                judge_model=args.judge_model,
                max_tokens=int(args.max_judge_tokens),
            )
            prefix_total += updated
            for key in aggregate_changed:
                delta_changed = changed_counts.get(key, 0)
                delta_correct = correct_counts.get(key, 0)
                aggregate_changed[key] += delta_changed
                aggregate_correct[key] += delta_correct
                prefix_changed[key] += delta_changed
                prefix_correct[key] += delta_correct
        per_prefix_stats[prefix] = {
            "total": prefix_total,
            "changed": prefix_changed,
            "correct": prefix_correct,
        }
        total_records += prefix_total

    print(f"Re-evaluated {total_records} records across {sum(len(paths) for paths in files.values())} files.")
    if total_records > 0:
        print("\n==== Summary by Prefix ====")
        for prefix, stats in per_prefix_stats.items():
            prefix_total = stats["total"]
            if prefix_total == 0:
                continue
            print(f"-- {os.path.basename(prefix)} ({prefix_total} records) --")
            for key, label in MODEL_SPECS:
                changed = stats["changed"].get(key, 0)
                accuracy = stats["correct"].get(key, 0) / prefix_total * 100.0
                changed_pct = (changed / prefix_total * 100.0)
                print(f"  {label}: changed_correct={changed} ({changed_pct:.1f}%), accuracy={accuracy:.1f}%")

        print("\n==== Aggregate Summary ====")
        for key, label in MODEL_SPECS:
            changed = aggregate_changed.get(key, 0)
            changed_pct = (changed / total_records * 100.0)
            accuracy = aggregate_correct.get(key, 0) / total_records * 100.0
            print(f"{label}: changed_correct={changed} ({changed_pct:.1f}%), accuracy={accuracy:.1f}%")
    else:
        print("No records processed; skipping summary.")


if __name__ == "__main__":
    main()

