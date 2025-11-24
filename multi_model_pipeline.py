#!/usr/bin/env python3
"""Full multi-model pipeline orchestration.

Round 1:
  - For each model, generate 10 samples per case (MedCaseReasoning filtered dataset)
  - Grade responses (gpt-5-nano) -> responses_<model>.graded.json
  - Convert to results_<model>.csv compatible with label_traces.py
  - Label traces -> per-model labeled CSV + sentences CSV
  - Run transition/correlation analysis
After all four models are processed, build a combined clustering plot with
cluster_traces_multi_model.py.

Round 2:
  - Determine per-case correctness counts for each model (from its 10 samples)
    - Select cases with 4..6 correct out of 10 (inclusive) for EACH model separately
  - Regenerate 100 samples ONLY for the selected cases per model
  - Repeat grading, conversion, labeling, analysis
  - Produce another combined clustering plot for round2.

Environment requirements:
  OPENAI_API_KEY for grading (gpt-5-nano)
  DEEPSEEK_API_KEY for labeling (DeepSeek Chat)

Usage:
  python multi_model_pipeline.py --out_root analysis_runs --split train
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import load_dataset  # type: ignore
from tqdm import tqdm

# Ensure we can import local modules from subfolders
repo_root = os.path.dirname(__file__)
tl_root = os.path.join(repo_root, "thinking-llms-interp")
if tl_root not in sys.path:
    sys.path.append(tl_root)

# Reuse helpers from repo components (lazy-import some heavy deps later)
import grade_responses as grader  # type: ignore

# Defer heavy model imports (transformers, nnsight, etc.) until actually needed inside generate().
load_model = None  # will be resolved lazily

import re
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
PROMPT_TEMPLATE = (
    "Read the following case presentation and give the most likely diagnosis.\n"
    "First, provide your internal reasoning for the diagnosis within the tags <think> ... </think>.\n"
    "Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>.\n\n"
    "----------------------------------------\nCASE PRESENTATION\n----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\nOUTPUT TEMPLATE\n----------------------------------------\n"
    "<think>\n...your internal reasoning for the diagnosis...\n</think><answer>\n...the name of the disease/entity...\n</answer>"
)
############################################################
# Helper utilities
############################################################

def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)

def exists_nonempty(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0

def short_name(model_id: str) -> str:
    model_id = (model_id or "").strip().replace("/", "_").replace(" ", "_")
    return model_id[-60:] if len(model_id) > 60 else model_id

def build_prompt(case_prompt: str) -> str:
    return PROMPT_TEMPLATE.format(case_prompt=(case_prompt or "").strip())

def extract_answer(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    return (m.group(1).strip() if m else "")

def extract_think(text: str) -> str:
    m = THINK_RE.search(text or "")
    return (m.group(1).strip() if m else "")

MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "FreedomIntelligence/HuatuoGPT-o1-8B",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "qwen/qwq-32b",
    "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-r1",
]

# Models that require OpenRouter (too large for local inference)
OPENROUTER_REQUIRED_MODELS = {
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "qwen/qwq-32b",
    "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-r1",
    # Models previously run locally, now converted to OpenRouter
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}

# Mapping from HuggingFace model names to OpenRouter model names
OPENROUTER_MODEL_MAPPING = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "deepseek/deepseek-r1-distill-qwen-1.5b",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek/deepseek-r1-distill-llama-8b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "deepseek/deepseek-r1-distill-qwen-14b",
    # Keep identity mappings for models already in OpenRouter format
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "qwen/qwen3-32b": "qwen/qwen3-32b",
    "qwen/qwq-32b": "qwen/qwq-32b",
    "deepseek/deepseek-r1-distill-llama-70b": "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-r1": "deepseek/deepseek-r1",
}


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def exists_nonempty(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def short_name(model: str) -> str:
    return model.split("/")[-1].lower()


def build_prompt(case_prompt: str) -> str:
    return PROMPT_TEMPLATE.format(case_prompt=case_prompt)


def extract_answer(text: str) -> str:
    """Extract the LAST <answer>...</answer> block.
    Many models echo the prompt template first; the actual answer is typically the last block.
    """
    if not text:
        return ""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return ""
    ans = matches[-1].strip()
    # Drop template placeholder if present
    placeholder = "...the name of the disease/entity..."
    if placeholder.lower() in ans.lower():
        return ""
    return ans


def extract_think(text: str) -> str:
    """Extract the LAST <think>...</think> block (skip prompt-echoed template).
    """
    if not text:
        return ""
    matches = THINK_RE.findall(text)
    if not matches:
        return ""
    think = matches[-1].strip()
    # If the last block is still the template placeholder, treat as empty
    placeholder = "...your internal reasoning for the diagnosis..."
    if placeholder.lower() in think.lower():
        return ""
    return think


def generate(
    model: str,
    dataset: str,
    split: str,
    samples: int,
    out_json: str,
    *,
    selected_pmcs: Optional[Sequence[str]] = None,
    batch_size: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
    limit_cases: Optional[int] = None,
    mock_generation: bool = False,
    use_vllm: bool = False,
    use_openrouter: bool = False,
    vllm_max_model_len: int = 8192,
    vllm_gpu_memory_utilization: float = 0.95,
    openrouter_workers: int = 10,
) -> None:
    # Automatically enable OpenRouter for models that require it
    if model in OPENROUTER_REQUIRED_MODELS and not use_openrouter:
        use_openrouter = True
        print(f"Model {model} requires OpenRouter; enabling automatically")
    
    rows = list(load_dataset(dataset)[split])
    if selected_pmcs:
        sel_set = {str(x) for x in selected_pmcs}
        rows = [r for r in rows if str(r.get("pmcid")) in sel_set]
    if limit_cases is not None and limit_cases > 0:
        rows = rows[:limit_cases]
    print(f"Model {model}: generating {samples} samples per {len(rows)} cases" + (" [MOCK]" if mock_generation else ""))

    # Mock path to avoid heavyweight model deps for smoke tests
    if mock_generation:
        all_items: List[Dict[str, Any]] = []
        total = len(rows) * samples
        desc = f"Generate (mock) {short_name(model)}"
        with tqdm(total=total, desc=desc, unit="sample") as pbar:
            for r in rows:
                pmcid = str(r.get("pmcid"))
                case_prompt = r.get("case_prompt") or r.get("case_presentation") or ""
                gold = r.get("final_diagnosis", "")
                question = build_prompt(case_prompt)
                for rep in range(samples):
                    qid = f"{pmcid}_{rep}"
                    # Alternate between correct and placeholder incorrect answers for coverage
                    pred = gold if (rep % 2 == 0) else "Unknown condition"
                    mock_response = (
                        f"{question}\n<think>Mock internal reasoning for PMC{pmcid} rep{rep}.\n"
                        f"I will consider differential diagnoses and converge on an answer.</think>"
                        f"<answer>\n{pred}\n</answer>"
                    )
                    all_items.append({
                        "original_message": {"role": "user", "content": question},
                        "full_response": mock_response,
                        "question_id": qid,
                        "question": question,
                        "gold_answer": gold,
                        "dataset_name": dataset,
                        "dataset_split": split,
                        "pmcid": pmcid,
                        "sample_index": rep,
                    })
                    pbar.update(1)
        ensure_dir(os.path.dirname(out_json) or ".")
        with open(out_json, "w") as f:
            json.dump(all_items, f, indent=2)
        print(f"Saved MOCK generation -> {out_json} ({len(all_items)} items)")
        return

    # OpenRouter API path (uses OpenAI Python SDK with base_url override)
    if use_openrouter:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment for --use_openrouter")
        http_referer = os.environ.get("OPENROUTER_HTTP_REFERER") or os.environ.get("HTTP_REFERER")
        x_title = os.environ.get("OPENROUTER_X_TITLE") or os.environ.get("X_TITLE")
        try:
            import openai  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("openai package not installed; add openai>=1.40.0 to requirements") from e
        # Configure client
        openai_client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={k: v for k, v in {
                "HTTP-Referer": http_referer,
                "X-Title": x_title,
            }.items() if v}
        )
        all_items: List[Dict[str, Any]] = []
        total_samples = len(rows) * samples
        pbar = tqdm(total=total_samples, desc=f"Generate (openrouter) {short_name(model)}", unit="sample")

        # Map model name to OpenRouter format if needed
        openrouter_model = OPENROUTER_MODEL_MAPPING.get(model, model)
        
        # Simple retry wrapper
        import time as _time
        def _send_with_retry(messages: List[Dict[str, str]], model_id: str) -> Any:
            backoff = 1.0
            for attempt in range(3):
                try:
                    # Build request parameters
                    request_params = {
                        "model": model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_new_tokens,
                    }
                    # For reasoning models, ensure we get the full reasoning trace
                    # Some models may support a reasoning parameter, but for most R1 models,
                    # the reasoning is included in the content by default
                    return openai_client.chat.completions.create(**request_params)
                except Exception as exc:
                    if attempt == 2:
                        raise
                    _time.sleep(backoff)
                    backoff *= 2

        # Build all tasks (pmcid, rep, question, gold) for parallel processing
        tasks: List[Tuple[str, int, str, str, str]] = []  # (pmcid, rep, qid, question, gold)
        for r in rows:
            pmcid = str(r.get("pmcid"))
            case_prompt = r.get("case_prompt") or r.get("case_presentation") or ""
            gold = r.get("final_diagnosis", "")
            question = build_prompt(case_prompt)
            for rep in range(samples):
                qid = f"{pmcid}_{rep}"
                tasks.append((pmcid, rep, qid, question, gold))

        # Process function for a single task
        def process_task(task: Tuple[str, int, str, str, str]) -> Dict[str, Any]:
            pmcid, rep, qid, question, gold = task
            messages = [{"role": "user", "content": question}]
            completion = _send_with_retry(messages, openrouter_model)
            # OpenAI SDK returns completion.choices list
            text = ""
            reasoning_trace = ""
            try:
                if completion and completion.choices:
                    choice = completion.choices[0]
                    # New SDK: choice.message.content
                    if hasattr(choice, "message") and getattr(choice.message, "content", None):
                        text = choice.message.content
                    elif hasattr(choice, "text"):
                        text = choice.text  # fallback older attr
                    
                    # Check for additional reasoning fields in the response
                    # Some OpenRouter models may return reasoning in separate fields
                    if hasattr(choice, "message"):
                        msg = choice.message
                        # Check for reasoning-related fields
                        if hasattr(msg, "reasoning") and msg.reasoning:
                            reasoning_trace = str(msg.reasoning)
                        elif hasattr(msg, "reasoning_trace") and msg.reasoning_trace:
                            reasoning_trace = str(msg.reasoning_trace)
                    
                    # Also check completion-level fields
                    if hasattr(completion, "reasoning") and completion.reasoning:
                        reasoning_trace = str(completion.reasoning)
                    elif hasattr(completion, "reasoning_trace") and completion.reasoning_trace:
                        reasoning_trace = str(completion.reasoning_trace)
                    
                    # Combine reasoning trace with content if found separately
                    if reasoning_trace and reasoning_trace not in text:
                        text = reasoning_trace + "\n\n" + text
            except Exception:
                text = ""
            full_response = question + "\n" + text
            return {
                "original_message": {"role": "user", "content": question},
                "full_response": full_response,
                "question_id": qid,
                "question": question,
                "gold_answer": gold,
                "dataset_name": dataset,
                "dataset_split": split,
                "pmcid": pmcid,
                "sample_index": rep,
            }

        # Process tasks in parallel with ThreadPoolExecutor
        workers = max(1, int(openrouter_workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(process_task, task): task for task in tasks}
            # Process completed tasks as they finish
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    all_items.append(result)
                    pbar.update(1)
                except Exception as exc:
                    task = future_to_task[future]
                    print(f"Task {task[2]} failed: {exc}", file=sys.stderr)
                    # Create error entry
                    pmcid, rep, qid, question, gold = task
                    all_items.append({
                        "original_message": {"role": "user", "content": question},
                        "full_response": question + "\n[ERROR: " + str(exc) + "]",
                        "question_id": qid,
                        "question": question,
                        "gold_answer": gold,
                        "dataset_name": dataset,
                        "dataset_split": split,
                        "pmcid": pmcid,
                        "sample_index": rep,
                    })
                    pbar.update(1)
        pbar.close()
        ensure_dir(os.path.dirname(out_json) or ".")
        with open(out_json, "w") as f:
            json.dump(all_items, f, indent=2)
        print(f"Saved generation (openrouter) -> {out_json} ({len(all_items)} items)")
        return

    # vLLM fast path
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError("vLLM not available; install vllm to use --use_vllm") from e

        # Initialize vLLM engine and tokenizer (tokenizer for chat template only)
        llm = LLM(
            model=model,
            max_model_len=int(vllm_max_model_len),
            gpu_memory_utilization=float(vllm_gpu_memory_utilization),
            trust_remote_code=True,
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

        all_items: List[Dict[str, Any]] = []
        total_samples = len(rows) * samples
        desc = f"Generate (vLLM) {short_name(model)}"
        pbar = tqdm(total=total_samples, desc=desc, unit="sample")
        sampling = SamplingParams(temperature=max(1e-6, float(temperature)), top_p=float(top_p), max_tokens=int(max_new_tokens))
        for start in range(0, len(rows), batch_size):
            sub = rows[start:start+batch_size]
            prompts: List[str] = []
            meta: List[Tuple[str, int, str, str, str]] = []  # pmcid, rep, qid, question, gold
            for r in sub:
                pmcid = str(r.get("pmcid"))
                case_prompt = r.get("case_prompt") or r.get("case_presentation") or ""
                gold = r.get("final_diagnosis", "")
                question = build_prompt(case_prompt)
                for rep in range(samples):
                    qid = f"{pmcid}_{rep}"
                    prompt = hf_tokenizer.apply_chat_template([
                        {"role": "user", "content": question}
                    ], tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)
                    meta.append((pmcid, rep, qid, question, gold))
            results = llm.generate(prompts, sampling)
            # vLLM may return results in a different order; map by prompt index
            # results is a list where each item has .prompt and .outputs[0].text
            for i, out in enumerate(results):
                pmcid, rep, qid, question, gold = meta[i]
                completion = out.outputs[0].text if out.outputs else ""
                full_response = prompts[i] + completion
                all_items.append({
                    "original_message": {"role": "user", "content": question},
                    "full_response": full_response,
                    "question_id": qid,
                    "question": question,
                    "gold_answer": gold,
                    "dataset_name": dataset,
                    "dataset_split": split,
                    "pmcid": pmcid,
                    "sample_index": rep,
                })
                pbar.update(1)
        pbar.close()
        ensure_dir(os.path.dirname(out_json) or ".")
        with open(out_json, "w") as f:
            json.dump(all_items, f, indent=2)
        print(f"Saved generation (vLLM) -> {out_json} ({len(all_items)} items)")
        return

    # Default HF/nnsight path
    global load_model
    if load_model is None:
        try:
            from utils.utils import load_model as _lm  # type: ignore
            load_model = _lm
        except Exception:
            import importlib.util as _ilu
            _utils_path = os.path.join(tl_root, "utils", "utils.py")
            spec = _ilu.spec_from_file_location("tl_utils", _utils_path)
            if spec and spec.loader:
                tl_utils = _ilu.module_from_spec(spec)
                spec.loader.exec_module(tl_utils)  # type: ignore[attr-defined]
                load_model = getattr(tl_utils, "load_model")  # type: ignore[assignment]
            else:
                raise RuntimeError("Failed to import load_model from utils.utils")

    model_obj, tokenizer = load_model(model_name=model)

    all_items: List[Dict[str, Any]] = []
    total_samples = len(rows) * samples
    desc = f"Generate {short_name(model)}"
    pbar = tqdm(total=total_samples, desc=desc, unit="sample")
    for start in range(0, len(rows), batch_size):
        sub = rows[start:start+batch_size]
        prompts: List[str] = []
        meta: List[Tuple[str, int, str, str, str]] = []  # pmcid, rep, qid, question, gold
        for r in sub:
            pmcid = str(r.get("pmcid"))
            case_prompt = r.get("case_prompt") or r.get("case_presentation") or ""
            gold = r.get("final_diagnosis", "")
            question = build_prompt(case_prompt)
            for rep in range(samples):
                qid = f"{pmcid}_{rep}"
                prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True))
                meta.append((pmcid, rep, qid, question, gold))
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        input_ids = encoded["input_ids"].to(model_obj.device)
        attention_mask = encoded["attention_mask"].to(model_obj.device)
        with model_obj.generate({"input_ids": input_ids, "attention_mask": attention_mask}, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, temperature=temperature + 1e-12, top_p=top_p) as gen:
            outputs = model_obj.generator.output.save()
        prompt_lengths = attention_mask.sum(dim=1)
        for i, (pmcid, rep, qid, question, gold) in enumerate(meta):
            pl = int(prompt_lengths[i].item())
            gen_tokens = outputs[i][pl:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            full_response = prompts[i] + gen_text
            all_items.append({
                "original_message": {"role": "user", "content": question},
                "full_response": full_response,
                "question_id": qid,
                "question": question,
                "gold_answer": gold,
                "dataset_name": dataset,
                "dataset_split": split,
                "pmcid": pmcid,
                "sample_index": rep,
            })
            pbar.update(1)
    pbar.close()
    ensure_dir(os.path.dirname(out_json) or ".")
    with open(out_json, "w") as f:
        json.dump(all_items, f, indent=2)
    print(f"Saved generation -> {out_json} ({len(all_items)} items)")


def grade(input_json: str, out_json: str) -> None:
    argv = ["--input", input_json, "--output", out_json, "--model", "gpt-5-nano", "--workers", "1000"]
    grader.main(argv)


def convert_to_results(graded_json: str, out_csv: str) -> None:
    with open(graded_json, "r") as f:
        data = json.load(f)
    rows: List[Dict[str, Any]] = []
    total = len(data) if isinstance(data, list) else 0
    pbar = tqdm(total=total or None, desc="Convert to results.csv", unit="resp")
    def heuristic_reasoning(full_response: str, question: str) -> str:
        # Try <think> extraction first
        think = extract_think(full_response)
        if think:
            return think
        # Remove original prompt if echoed
        resp = full_response
        if question and question in resp:
            resp = resp.replace(question, "")
        # Strip answer tag if present
        resp_no_answer = re.sub(r"<answer>.*?</answer>", "", resp, flags=re.IGNORECASE|re.DOTALL).strip()
        # If still empty, fallback to full_response truncated
        if not resp_no_answer:
            return full_response[-1000:]
        return resp_no_answer[:4000]
    def heuristic_predicted(full_response: str) -> str:
        ans = extract_answer(full_response)
        # Ignore template placeholder if present
        if ans and "name of the disease" not in ans.lower():
            return ans
        # Try last line
        lines = [l.strip() for l in full_response.splitlines() if l.strip()]
        if lines:
            return lines[-1][:256]
        return "Unknown"
    for item in data:
        if not item:
            continue
        orig = item.get("original_message", {}) or {}
        case_prompt = grader.extract_case(orig.get("content", ""))
        pmcid = str(item.get("pmcid", ""))
        sample_index = int(item.get("sample_index") or 0)
        true_diag = item.get("gold_answer", "")
        full_resp = item.get("full_response", "")
        predicted = item.get("extracted_answer") or heuristic_predicted(full_resp)
        think = heuristic_reasoning(full_resp, orig.get("content", ""))
        verified = bool(item.get("is_correct", False))
        td = item.get("_true_description")
        pd = item.get("_predicted_description")
        rr = item.get("_rating_raw")
        verification = ""
        if td or pd or rr:
            verification = (
                (f"True diagnosis description:\n{td}\n\n" if td else "") +
                (f"Predicted diagnosis description:\n{pd}\n\n" if pd else "") +
                (f"Similarity rating (0-10): {rr}" if rr else "")
            )
        rows.append({
            "pmcid": pmcid,
            "sample_index": sample_index,
            "prompt_edit": "",
            "prompt_insert": "",
            "case_prompt": case_prompt,
            "diagnostic_reasoning": None,
            "true_diagnosis": true_diag,
            "predicted_diagnosis": predicted,
            "reasoning_trace": think,
            "posthoc_reasoning_trace": "",
            "verification_response": verification,
            "verification_similarity": "",
            "verified_correct": verified,
        })
        pbar.update(1)
    pbar.close()
    fieldnames = ["pmcid", "sample_index", "prompt_edit", "prompt_insert", "case_prompt", "diagnostic_reasoning", "true_diagnosis", "predicted_diagnosis", "reasoning_trace", "posthoc_reasoning_trace", "verification_response", "verification_similarity", "verified_correct"]
    ensure_dir(os.path.dirname(out_csv) or ".")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved results CSV -> {out_csv} ({len(rows)} rows)")


def label_and_analyze(results_csv: str, out_dir: str) -> Tuple[str, str]:
    ensure_dir(out_dir)
    labeled_csv = os.path.join(out_dir, "results.labeled.csv")
    labeled_html = os.path.join(out_dir, "results.labeled.html")
    # Pre-scan rows to detect empty reasoning
    import pandas as _pd
    df_tmp = _pd.read_csv(results_csv)
    n_rows = len(df_tmp)
    empty_reasoning = df_tmp['reasoning_trace'].fillna('').str.strip().eq('')
    if n_rows == 0:
        print(f"No rows in {results_csv}; skipping labeling and analysis.")
        return results_csv, results_csv.replace('.csv', '.sentences.csv')
    if empty_reasoning.all():
        print(f"All reasoning traces empty in {results_csv}; skipping labeling and analysis.")
        return results_csv, results_csv.replace('.csv', '.sentences.csv')
    # Lazy import labeler and analysis modules to avoid importing heavy/optional deps if skipped elsewhere
    import label_traces as labeler  # type: ignore
    import analyze_traces_state_transition as transition_analysis  # type: ignore

    with tqdm(total=2, desc="Label + analyze", unit="step") as stepbar:
        labeler.label_csv(
            csv_path=results_csv,
            out_jsonl=os.path.join(out_dir, "labeled_traces.jsonl"),
            out_csv=labeled_csv,
            out_html=labeled_html,
            cache_dir=os.path.join(out_dir, ".cache", "deepseek_labels"),
            resume=True,
            workers=1000,
            max_tokens=8000,
            model="deepseek-chat",
        )
        stepbar.update(1)
        transition_analysis.run_analysis(labeled_csv=labeled_csv, out_dir=os.path.join(out_dir, "analysis_outputs"))
        stepbar.update(1)
    return labeled_csv, labeled_csv.replace(".csv", ".sentences.csv")


def per_case_counts(results_csv: str) -> Dict[str, Tuple[int, int]]:
    acc: Dict[str, Dict[str, int]] = {}
    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmcid = str(row.get("pmcid", "")).strip()
            if not pmcid:
                continue
            bucket = acc.setdefault(pmcid, {"total": 0, "correct": 0})
            bucket["total"] += 1
            v = str(row.get("verified_correct", "")).strip().lower()
            if v in {"true", "1", "yes", "y"}:
                bucket["correct"] += 1
    return {k: (v["total"], v["correct"]) for k, v in acc.items()}


def run_round(
    models: List[str],
    dataset: str,
    split: str,
    samples: int,
    round_tag: str,
    out_root: str,
    *,
    limit_cases: Optional[int] = None,
    batch_size: int = 8,
    skip_labeling: bool = False,
    skip_grading: bool = False,
    mock_generation: bool = False,
    skip_existing: bool = False,
    use_vllm: bool = False,
    use_openrouter: bool = False,
    vllm_max_model_len: int = 8192,
    vllm_gpu_memory_utilization: float = 0.95,
    openrouter_workers: int = 10,
) -> Dict[str, Dict[str, str]]:
    """Run one round and return mapping model_short -> paths dict."""
    outputs: Dict[str, Dict[str, str]] = {}
    for m in tqdm(models, desc=f"{round_tag}: models", unit="model"):
        ms = short_name(m)
        base_dir = os.path.join(out_root, ms, round_tag)
        ensure_dir(base_dir)
        gen_json = os.path.join(base_dir, f"responses_{ms}.json")
        graded_json = os.path.join(base_dir, f"responses_{ms}.graded.json")
        results_csv = os.path.join(base_dir, f"results_{ms}.csv")
        per_case_csv = os.path.join(base_dir, f"results_{ms}.per_case.csv")
        diff_summary = os.path.join(base_dir, "analysis_outputs", "differential_count_correlation.txt")
        diff_cache = os.path.join(base_dir, "analysis_outputs", "differential_cache.jsonl")
        total_steps = 1 + (0 if skip_grading else 2) + (0 if skip_labeling else 1)
        with tqdm(total=total_steps, desc=f"{ms} steps", unit="step", leave=False) as spbar:
            if skip_existing and exists_nonempty(gen_json):
                print(f"{ms}: skip generate (exists) -> {gen_json}")
            else:
                generate(
                    m,
                    dataset,
                    split,
                    samples,
                    gen_json,
                    limit_cases=limit_cases,
                    batch_size=batch_size,
                    mock_generation=mock_generation,
                    use_vllm=use_vllm,
                    use_openrouter=use_openrouter,
                    vllm_max_model_len=vllm_max_model_len,
                    vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
                    openrouter_workers=openrouter_workers,
                )
            spbar.update(1)
            if skip_grading:
                outputs[ms] = {"generated_json": gen_json}
            else:
                if skip_existing and exists_nonempty(graded_json):
                    print(f"{ms}: skip grade (exists) -> {graded_json}")
                else:
                    grade(gen_json, graded_json)
                spbar.update(1)
                if skip_existing and exists_nonempty(results_csv):
                    print(f"{ms}: skip convert_to_results (exists) -> {results_csv}")
                else:
                    convert_to_results(graded_json, results_csv)
                spbar.update(1)
                # Build per-case CSV for difficulty computation if missing
                try:
                    import pandas as _pd
                    df_r = _pd.read_csv(results_csv)
                    if not os.path.isfile(per_case_csv):
                        grp = df_r.groupby("pmcid").agg(
                            total_samples=("verified_correct", "count"),
                            n_correct=("verified_correct", lambda s: int((_pd.Series(s).astype(str).str.lower().isin(["true","1","yes","y"]).sum())))
                        ).reset_index()
                        grp["accuracy"] = grp.apply(lambda r: (r.n_correct / r.total_samples) if r.total_samples else 0.0, axis=1)
                        grp.to_csv(per_case_csv, index=False)
                except Exception as exc:
                    print(f"Per-case CSV build failed for {ms}: {exc}")
                # Differential count analysis (heuristic if mock_generation)
                try:
                    from analyze_differential_counts import main as diff_main  # type: ignore
                    argv_backup = sys.argv[:]
                    if skip_existing and exists_nonempty(diff_summary):
                        print(f"{ms}: skip differential analysis (exists) -> {diff_summary}")
                    else:
                        diff_args = [
                            "analyze_differential_counts.py",
                            "--input", results_csv,
                            "--per_case", per_case_csv,
                            "--out_dir", base_dir,
                            "--summary", diff_summary,
                            "--cache", diff_cache,
                        ]
                        if mock_generation:
                            diff_args.append("--dry_run")
                        sys.argv = diff_args
                        diff_main()
                except Exception as exc:
                    print(f"Differential analysis failed for {ms}: {exc}")
                finally:
                    sys.argv = argv_backup
                spbar.update(1)
                if skip_labeling:
                    outputs[ms] = {"results_csv": results_csv}
                else:
                    expected_labeled = os.path.join(base_dir, "results.labeled.csv")
                    expected_sentences = expected_labeled.replace(".csv", ".sentences.csv")
                    if skip_existing and exists_nonempty(expected_labeled) and exists_nonempty(expected_sentences):
                        print(f"{ms}: skip labeling (exists) -> {expected_labeled}")
                        labeled_csv, sentences_csv = expected_labeled, expected_sentences
                    else:
                        labeled_csv, sentences_csv = label_and_analyze(results_csv, base_dir)
                    spbar.update(1)
                    outputs[ms] = {"results_csv": results_csv, "labeled_csv": labeled_csv, "sentences_csv": sentences_csv}
    return outputs


def aggregate_all_models_analysis(
    all_round_outputs: Dict[str, Dict[str, Dict[str, str]]],
    out_root: str,
    *,
    skip_existing: bool = False,
    skip_labeling: bool = False,
    skip_grading: bool = False,
) -> None:
    """Aggregate labeled and graded responses from all models and rounds, then run combined analyses.
    
    Args:
        all_round_outputs: Dict mapping round_tag -> Dict mapping model_short -> paths dict
        out_root: Root output directory
        skip_existing: Skip if output files already exist
        skip_labeling: Skip if labeling was skipped (no labeled CSVs available)
        skip_grading: Skip if grading was skipped (no graded CSVs available)
    """
    if skip_labeling or skip_grading:
        print("Skipping aggregate analysis (labeling or grading was skipped)")
        return
    
    combined_dir = os.path.join(out_root, "combined", "all_models_analysis")
    ensure_dir(combined_dir)
    
    # Collect all labeled CSVs and results CSVs from all rounds
    all_labeled_csvs: List[Tuple[str, str]] = []  # (model_name, path)
    all_results_csvs: List[Tuple[str, str]] = []  # (model_name, path)
    all_per_case_csvs: List[Tuple[str, str]] = []  # (model_name, path)
    
    for round_tag, round_outputs in all_round_outputs.items():
        for model_short, paths in round_outputs.items():
            if "labeled_csv" in paths:
                all_labeled_csvs.append((f"{model_short}_{round_tag}", paths["labeled_csv"]))
            if "results_csv" in paths:
                all_results_csvs.append((f"{model_short}_{round_tag}", paths["results_csv"]))
                # Try to find per_case CSV in the same directory
                # Per_case CSV naming: results_<model>.csv -> results_<model>.per_case.csv
                results_dir = os.path.dirname(paths["results_csv"])
                results_basename = os.path.basename(paths["results_csv"])
                per_case_basename = results_basename.replace(".csv", ".per_case.csv")
                per_case_path = os.path.join(results_dir, per_case_basename)
                if os.path.isfile(per_case_path):
                    all_per_case_csvs.append((f"{model_short}_{round_tag}", per_case_path))
    
    if not all_labeled_csvs:
        print("No labeled CSVs found; skipping aggregate analysis")
        return
    
    if not all_results_csvs:
        print("No results CSVs found; skipping aggregate analysis")
        return
    
    print(f"Aggregating data from {len(all_labeled_csvs)} model/round combinations...")
    
    # Aggregate labeled CSVs
    combined_labeled_csv = os.path.join(combined_dir, "results.labeled.combined.csv")
    if skip_existing and exists_nonempty(combined_labeled_csv):
        print(f"Skip aggregate labeled CSV (exists) -> {combined_labeled_csv}")
    else:
        try:
            import pandas as pd
            dfs_labeled = []
            for model_name, csv_path in all_labeled_csvs:
                if not os.path.isfile(csv_path):
                    print(f"Warning: labeled CSV not found: {csv_path}")
                    continue
                df = pd.read_csv(csv_path)
                df["model"] = model_name
                dfs_labeled.append(df)
            if dfs_labeled:
                df_combined = pd.concat(dfs_labeled, ignore_index=True)
                df_combined.to_csv(combined_labeled_csv, index=False)
                print(f"Saved combined labeled CSV -> {combined_labeled_csv} ({len(df_combined)} rows)")
            else:
                print("No labeled data to aggregate")
                return
        except Exception as exc:
            print(f"Failed to aggregate labeled CSVs: {exc}")
            return
    
    # Aggregate results CSVs
    combined_results_csv = os.path.join(combined_dir, "results.combined.csv")
    if skip_existing and exists_nonempty(combined_results_csv):
        print(f"Skip aggregate results CSV (exists) -> {combined_results_csv}")
    else:
        try:
            import pandas as pd
            dfs_results = []
            for model_name, csv_path in all_results_csvs:
                if not os.path.isfile(csv_path):
                    print(f"Warning: results CSV not found: {csv_path}")
                    continue
                df = pd.read_csv(csv_path)
                df["model"] = model_name
                dfs_results.append(df)
            if dfs_results:
                df_combined = pd.concat(dfs_results, ignore_index=True)
                df_combined.to_csv(combined_results_csv, index=False)
                print(f"Saved combined results CSV -> {combined_results_csv} ({len(df_combined)} rows)")
            else:
                print("No results data to aggregate")
        except Exception as exc:
            print(f"Failed to aggregate results CSVs: {exc}")
    
    # Aggregate per_case CSVs if available
    combined_per_case_csv = os.path.join(combined_dir, "results.per_case.combined.csv")
    if all_per_case_csvs:
        if skip_existing and exists_nonempty(combined_per_case_csv):
            print(f"Skip aggregate per_case CSV (exists) -> {combined_per_case_csv}")
        else:
            try:
                import pandas as pd
                dfs_per_case = []
                for model_name, csv_path in all_per_case_csvs:
                    if not os.path.isfile(csv_path):
                        continue
                    df = pd.read_csv(csv_path)
                    df["model"] = model_name
                    dfs_per_case.append(df)
                if dfs_per_case:
                    df_combined = pd.concat(dfs_per_case, ignore_index=True)
                    df_combined.to_csv(combined_per_case_csv, index=False)
                    print(f"Saved combined per_case CSV -> {combined_per_case_csv} ({len(df_combined)} rows)")
            except Exception as exc:
                print(f"Failed to aggregate per_case CSVs: {exc}")
    
    # Run transition analysis on combined data
    analysis_out_dir = os.path.join(combined_dir, "analysis_outputs")
    ensure_dir(analysis_out_dir)
    try:
        import analyze_traces_state_transition as transition_analysis  # type: ignore
        transition_summary = os.path.join(analysis_out_dir, "transition_analysis_summary.txt")
        if skip_existing and exists_nonempty(transition_summary):
            print(f"Skip combined transition analysis (exists) -> {transition_summary}")
        else:
            print("Running combined transition analysis...")
            transition_analysis.run_analysis(
                labeled_csv=combined_labeled_csv,
                out_dir=analysis_out_dir,
            )
            print(f"Completed combined transition analysis -> {analysis_out_dir}")
        
        # Verify transition matrices for correct/incorrect exist for classify_traces_transition.py
        # The analysis should create transition_probs_correct.csv and transition_probs_incorrect.csv
        correct_probs = os.path.join(analysis_out_dir, "transition_probs_correct.csv")
        incorrect_probs = os.path.join(analysis_out_dir, "transition_probs_incorrect.csv")
        
        if os.path.isfile(correct_probs) and os.path.isfile(incorrect_probs):
            print(f"✓ Transition matrices verified: {correct_probs}, {incorrect_probs}")
        else:
            print(f"⚠ Warning: Transition matrices not found:")
            if not os.path.isfile(correct_probs):
                print(f"  Missing: {correct_probs}")
            if not os.path.isfile(incorrect_probs):
                print(f"  Missing: {incorrect_probs}")
            print("  These files are required for classify_traces_transition.py")
    except Exception as exc:
        print(f"Combined transition analysis failed: {exc}")
        import traceback
        traceback.print_exc()
    
    # Run differential count analysis on combined data
    try:
        from analyze_differential_counts import main as diff_main  # type: ignore
        diff_summary = os.path.join(analysis_out_dir, "differential_count_correlation.txt")
        diff_cache = os.path.join(analysis_out_dir, "differential_cache.jsonl")
        if skip_existing and exists_nonempty(diff_summary):
            print(f"Skip combined differential analysis (exists) -> {diff_summary}")
        else:
            print("Running combined differential count analysis...")
            argv_backup = sys.argv[:]
            try:
                diff_args = [
                    "analyze_differential_counts.py",
                    "--input", combined_results_csv,
                    "--out_dir", combined_dir,
                    "--summary", diff_summary,
                    "--cache", diff_cache,
                ]
                if all_per_case_csvs and os.path.isfile(combined_per_case_csv):
                    diff_args.extend(["--per_case", combined_per_case_csv])
                sys.argv = diff_args
                diff_main()
                print(f"Completed combined differential analysis -> {diff_summary}")
            finally:
                sys.argv = argv_backup
    except Exception as exc:
        print(f"Combined differential analysis failed: {exc}")
    
    print(f"Combined analysis complete -> {combined_dir}")


def combined_cluster(
    round_outputs: Dict[str, Dict[str, str]],
    out_root: str,
    round_tag: str,
    *,
    skip_existing: bool = False,
    # Combined clustering configuration via args (no envs)
    combined_method: str = "msm_landmarks",
    combined_neighbors: int = 30,
    combined_layout: str = "umap",
    combined_max_timepoints: Optional[int] = None,
    combined_use_elastic: bool = False,
    combined_msm_c: Optional[float] = None,
    combined_msm_jobs: Optional[int] = None,
    combined_msm_backend: Optional[str] = None,
    combined_landmarks_k: Optional[int] = None,
    combined_landmarks_layout: Optional[str] = None,
) -> None:
    from cluster_traces_multi_model import main as combo_main  # type: ignore
    csv_list = []
    name_list = []
    for ms, paths in round_outputs.items():
        if 'sentences_csv' not in paths:
            continue
        csv_list.append(paths["sentences_csv"])
        name_list.append(ms)
    out_dir = os.path.join(out_root, "combined", round_tag)
    ensure_dir(out_dir)
    if len(csv_list) < 2:
        print(f"Combined clustering skipped (need >=2 sentence CSVs, got {len(csv_list)}).")
        return
    argv_backup = sys.argv[:]
    try:
        with tqdm(total=1, desc=f"Combined clustering {round_tag}", unit="job"):
            # If output exists, skip
            out_html = os.path.join(out_dir, "combined_mds.html")
            if skip_existing and exists_nonempty(out_html):
                print(f"Skip combined clustering (exists) -> {out_html}")
            else:
                # Prefer scalable msm_landmarks by default (args-driven)
                sys.argv = [
                    "cluster_traces_multi_model.py",
                    "--sentences_csvs", ",".join(csv_list),
                    "--model_names", ",".join(name_list),
                    "--out_dir", out_dir,
                    "--method", str(combined_method),
                    "--n_neighbors", str(int(combined_neighbors)),
                    "--layout", str(combined_layout),
                ]
                # Optional caps and algorithmic toggles
                if combined_max_timepoints and int(combined_max_timepoints) > 0:
                    sys.argv.extend(["--max_timepoints", str(int(combined_max_timepoints))])
                if combined_use_elastic:
                    sys.argv.append("--use_elastic")
                if combined_msm_c is not None:
                    sys.argv.extend(["--msm_c", str(float(combined_msm_c))])
                if combined_msm_jobs is not None:
                    sys.argv.extend(["--msm_jobs", str(int(combined_msm_jobs))])
                if combined_msm_backend:
                    sys.argv.extend(["--msm_backend", str(combined_msm_backend)])
                if combined_landmarks_k is not None:
                    sys.argv.extend(["--landmarks_k", str(int(combined_landmarks_k))])
                if combined_landmarks_layout:
                    sys.argv.extend(["--landmarks_layout", str(combined_landmarks_layout)])
                combo_main()
    finally:
        sys.argv = argv_backup


def main():
    ap = argparse.ArgumentParser(description="Multi-model MedCase pipeline")
    ap.add_argument("--dataset", default="tmknguyen/MedCaseReasoning-filtered")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_root", default="analysis_runs")
    ap.add_argument("--skip_round2", action="store_true")
    ap.add_argument("--models", default=",".join(MODELS), help="Comma-separated model IDs to run")
    ap.add_argument("--limit_cases", type=int, default=0, help="Limit number of cases for a quick run")
    ap.add_argument("--round1_samples", type=int, default=10, help="Samples per case in round1")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--skip_labeling", action="store_true", help="Skip labeling+analysis steps (for smoke tests)")
    ap.add_argument("--skip_grading", action="store_true", help="Skip grading (generate only; for import/dtype smoke tests)")
    ap.add_argument("--mock_generation", action="store_true", help="Mock generation outputs to bypass model loading (smoke test)")
    ap.add_argument("--skip_existing", action="store_true", help="Skip each step if its output file already exists (resume mode)")
    ap.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for faster generation")
    ap.add_argument("--use_openrouter", action="store_true", help="Use OpenRouter API (remote inference via unified model interface)")
    ap.add_argument("--openrouter_model", default=None, help="Override model ID when using --use_openrouter (defaults to given model list entry)")
    ap.add_argument("--openrouter_workers", type=int, default=10, help="Number of parallel workers for OpenRouter API calls")
    ap.add_argument("--vllm_max_model_len", type=int, default=8192, help="vLLM: max model length (tokens) to size KV cache; lower to fit memory")
    ap.add_argument("--vllm_gpu_mem_util", type=float, default=0.95, help="vLLM: GPU memory utilization fraction for KV cache")
    # Combined clustering configuration (args-based; replaces env usage)
    ap.add_argument("--combined_method", default="msm_landmarks", help="Combined embedding method: msm_landmarks | msm_mds | ann_umap")
    ap.add_argument("--combined_neighbors", type=int, default=30, help="k for neighborhood graphs/UMAP in combined embedding")
    ap.add_argument("--combined_layout", default="umap", help="Layout algorithm for ann_umap/msm_landmarks: umap | pca | spring")
    ap.add_argument("--combined_max_timepoints", type=int, default=0, help="Cap max timepoints per trace during feature build (0=none)")
    ap.add_argument("--combined_use_elastic", action="store_true", help="Use elastic MSM distance path where applicable")
    ap.add_argument("--combined_msm_c", type=float, default=None, help="MSM cost parameter c")
    ap.add_argument("--combined_msm_jobs", type=int, default=None, help="Parallel jobs for MSM distance (rows or pairs)")
    ap.add_argument("--combined_msm_backend", choices=["threads","processes"], default=None, help="joblib backend for MSM parallelism")
    ap.add_argument("--combined_landmarks_k", type=int, default=None, help="Number of landmarks for msm_landmarks method")
    ap.add_argument("--combined_landmarks_layout", default=None, help="Embedding layout for msm_landmarks: umap | pca")
    args = ap.parse_args()

    ensure_dir(args.out_root)

    # Parse models
    models = [m.strip() for m in str(args.models).split(",") if m.strip()]

    # Round 1
    round1 = run_round(
        models,
        args.dataset,
        args.split,
        args.round1_samples,
        "round1_10",
        args.out_root,
        limit_cases=(args.limit_cases or None),
        batch_size=args.batch_size,
        skip_labeling=args.skip_labeling,
        skip_grading=args.skip_grading,
        mock_generation=args.mock_generation,
        skip_existing=args.skip_existing,
        use_vllm=args.use_vllm,
        use_openrouter=args.use_openrouter,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_gpu_memory_utilization=args.vllm_gpu_mem_util,
        openrouter_workers=args.openrouter_workers,
    )
    if not args.skip_labeling and not args.skip_grading:
        combined_cluster(
            round1,
            args.out_root,
            "round1_10",
            skip_existing=args.skip_existing,
            combined_method=args.combined_method,
            combined_neighbors=args.combined_neighbors,
            combined_layout=args.combined_layout,
            combined_max_timepoints=(args.combined_max_timepoints if args.combined_max_timepoints > 0 else None),
            combined_use_elastic=args.combined_use_elastic,
            combined_msm_c=args.combined_msm_c,
            combined_msm_jobs=args.combined_msm_jobs,
            combined_msm_backend=args.combined_msm_backend,
            combined_landmarks_k=args.combined_landmarks_k,
            combined_landmarks_layout=args.combined_landmarks_layout,
        )

    if args.skip_round2:
        return

    # For each model, select cases 4..6 correct out of 10 and run 100-sample round
    round2_outputs: Dict[str, Dict[str, str]] = {}
    for m in tqdm(models, desc="round2_100: models", unit="model"):
        ms = short_name(m)
        r1_paths = round1[ms]
        if args.skip_grading:
            print(f"Model {ms}: grading skipped; cannot select 4..6/10 subset. Skipping round2 for this model.")
            continue
        counts = per_case_counts(r1_paths["results_csv"])
        selected = [pmcid for pmcid, (tot, ok) in counts.items() if tot >= 10 and 4 <= ok <= 6]
        if not selected:
            print(f"Model {ms}: no cases with 4..6/10 correct; skipping round2")
            continue
        base_dir = os.path.join(args.out_root, ms, "round2_100")
        ensure_dir(base_dir)
        gen_json = os.path.join(base_dir, f"responses_{ms}.json")
        graded_json = os.path.join(base_dir, f"responses_{ms}.graded.json")
        results_csv = os.path.join(base_dir, f"results_{ms}.csv")
        per_case_csv = os.path.join(base_dir, f"results_{ms}.per_case.csv")
        diff_summary = os.path.join(base_dir, "analysis_outputs", "differential_count_correlation.txt")
        diff_cache = os.path.join(base_dir, "analysis_outputs", "differential_cache.jsonl")
        total_steps = 1 + (0 if args.skip_grading else 3) + (0 if args.skip_labeling else 1)
        with tqdm(total=total_steps, desc=f"{ms} steps (round2)", unit="step", leave=False) as spbar:
            if args.skip_existing and exists_nonempty(gen_json):
                print(f"{ms} (round2): skip generate (exists) -> {gen_json}")
            else:
                generate(
                    m,
                    args.dataset,
                    args.split,
                    100,
                    gen_json,
                    selected_pmcs=selected,
                    batch_size=args.batch_size,
                    use_vllm=args.use_vllm,
                    use_openrouter=args.use_openrouter,
                    vllm_max_model_len=args.vllm_max_model_len,
                    vllm_gpu_memory_utilization=args.vllm_gpu_mem_util,
                    openrouter_workers=args.openrouter_workers,
                )
            spbar.update(1)
            if args.skip_grading:
                round2_outputs[ms] = {"generated_json": gen_json}
            else:
                if args.skip_existing and exists_nonempty(graded_json):
                    print(f"{ms} (round2): skip grade (exists) -> {graded_json}")
                else:
                    grade(gen_json, graded_json)
                spbar.update(1)
                if args.skip_existing and exists_nonempty(results_csv):
                    print(f"{ms} (round2): skip convert_to_results (exists) -> {results_csv}")
                else:
                    convert_to_results(graded_json, results_csv)
                spbar.update(1)
                # Build per-case CSV
                try:
                    import pandas as _pd
                    df_r = _pd.read_csv(results_csv)
                    if not os.path.isfile(per_case_csv):
                        grp = df_r.groupby("pmcid").agg(
                            total_samples=("verified_correct", "count"),
                            n_correct=("verified_correct", lambda s: int((_pd.Series(s).astype(str).str.lower().isin(["true","1","yes","y"]).sum())))
                        ).reset_index()
                        grp["accuracy"] = grp.apply(lambda r: (r.n_correct / r.total_samples) if r.total_samples else 0.0, axis=1)
                        grp.to_csv(per_case_csv, index=False)
                except Exception as exc:
                    print(f"Per-case CSV build failed for {ms} (round2): {exc}")
                # Differential analysis
                try:
                    from analyze_differential_counts import main as diff_main  # type: ignore
                    argv_backup = sys.argv[:]
                    if args.skip_existing and exists_nonempty(diff_summary):
                        print(f"{ms} (round2): skip differential analysis (exists) -> {diff_summary}")
                    else:
                        diff_args = [
                            "analyze_differential_counts.py",
                            "--input", results_csv,
                            "--per_case", per_case_csv,
                            "--out_dir", base_dir,
                            "--summary", diff_summary,
                            "--cache", diff_cache,
                        ]
                        if args.mock_generation:
                            diff_args.append("--dry_run")
                        sys.argv = diff_args
                        diff_main()
                except Exception as exc:
                    print(f"Differential analysis failed for {ms} (round2): {exc}")
                finally:
                    sys.argv = argv_backup
                spbar.update(1)
                if args.skip_labeling:
                    round2_outputs[ms] = {"results_csv": results_csv}
                else:
                    expected_labeled = os.path.join(base_dir, "results.labeled.csv")
                    expected_sentences = expected_labeled.replace(".csv", ".sentences.csv")
                    if args.skip_existing and exists_nonempty(expected_labeled) and exists_nonempty(expected_sentences):
                        print(f"{ms} (round2): skip labeling (exists) -> {expected_labeled}")
                        labeled_csv, sentences_csv = expected_labeled, expected_sentences
                    else:
                        labeled_csv, sentences_csv = label_and_analyze(results_csv, base_dir)
                    spbar.update(1)
                    round2_outputs[ms] = {"results_csv": results_csv, "labeled_csv": labeled_csv, "sentences_csv": sentences_csv}

    if round2_outputs and not args.skip_labeling and not args.skip_grading:
        combined_cluster(
            round2_outputs,
            args.out_root,
            "round2_100",
            skip_existing=args.skip_existing,
            combined_method=args.combined_method,
            combined_neighbors=args.combined_neighbors,
            combined_layout=args.combined_layout,
            combined_max_timepoints=(args.combined_max_timepoints if args.combined_max_timepoints > 0 else None),
            combined_use_elastic=args.combined_use_elastic,
            combined_msm_c=args.combined_msm_c,
            combined_msm_jobs=args.combined_msm_jobs,
            combined_msm_backend=args.combined_msm_backend,
            combined_landmarks_k=args.combined_landmarks_k,
            combined_landmarks_layout=args.combined_landmarks_layout,
        )
    
    # Aggregate all models and rounds for final combined analysis
    if not args.skip_labeling and not args.skip_grading:
        print("\n" + "="*80)
        print("Aggregating all models and rounds for combined analysis...")
        print("="*80)
        all_round_outputs: Dict[str, Dict[str, Dict[str, str]]] = {}
        if round1:
            all_round_outputs["round1_10"] = round1
        if round2_outputs:
            all_round_outputs["round2_100"] = round2_outputs
        
        if all_round_outputs:
            aggregate_all_models_analysis(
                all_round_outputs,
                args.out_root,
                skip_existing=args.skip_existing,
                skip_labeling=args.skip_labeling,
                skip_grading=args.skip_grading,
            )
        else:
            print("No round outputs available for aggregate analysis")


if __name__ == "__main__":
    main()
