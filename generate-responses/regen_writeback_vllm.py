"""vLLM-batched regeneration of bmj responses with the native chat template + uniform
skip_special_tokens=True decode. Continuous batching → 20-50x faster than HF sequential.
Writes back into canonical responses_<model>.json (atomic, timestamped backup), marks
touched entries `_regenerated: true`. Resumable: chunked, skips already-regenerated rows.

Run gpt-oss with VLLM_USE_FLASHINFER_SAMPLER=0 (greedy needs no flashinfer sampler;
avoids a JIT-compile that needs matching nvcc). MXFP4 uses the prebuilt Triton MoE backend.

USAGE
  VLLM_USE_FLASHINFER_SAMPLER=0 MED_INTERP_ROOT=/home/ubuntu/Documents/med-interp \\
  .venv-vllm/bin/python /tmp/regen_writeback_vllm.py \\
      --model-key gpt-oss-20b --hf-id openai/gpt-oss-20b --target all

TARGET MODES
  raw : entries whose full_response.lstrip().startswith("Read the following case")
        -> llama-8b 302 / qwen-14b 210 (raw-prefix / medmechinterp-reconstructed)
  all : every entry (gpt-oss-20b & qwq-32b - 100% medmechinterp)
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
from datetime import datetime

from vllm import LLM, SamplingParams

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
RAW_PREFIX = "Read the following case"
# Paths overridable via MED_INTERP_ROOT env var.
_ROOT = os.environ.get("MED_INTERP_ROOT", "/home/ubuntu/Documents/med-interp")
RESP_DIR = f"{_ROOT}/bmj_artifacts/activations/activations"
BACKUP_ROOT = f"{_ROOT}/bmj_artifacts/.backups"
RESPONSES_PY = f"{_ROOT}/thinking-llms-interp/utils/responses.py"


def load_extract():
    spec = importlib.util.spec_from_file_location("resp", RESPONSES_PY)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.extract_thinking_process


def is_raw(fr):
    return fr.lstrip().startswith(RAW_PREFIX)


def select_targets(rows, mode):
    if mode == "all":
        return list(range(len(rows)))
    if mode == "raw":
        return [i for i, r in enumerate(rows) if is_raw(str(r.get("full_response", "")))]
    if mode == "noanswer":
        # rows the existing parser left junk (empty / '...' / template placeholder) — re-run w/ higher cap.
        def _junk(a):
            a = str(a).strip()
            return (not a) or a.strip(". ") == "" or "the name of the disease/entity" in a
        return [i for i, r in enumerate(rows) if _junk(r.get("extracted_answer", ""))]
    raise ValueError(f"Unknown target mode: {mode}")


def atomic_write_json(rows, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp, path)


def backup_original(canonical_path, backup_dir):
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, os.path.basename(canonical_path))
    if not os.path.exists(backup_path):
        shutil.copy2(canonical_path, backup_path)
    return backup_path


def derive_answer(full_response):
    """Last <answer>...</answer> block, EXCLUDING the prompt OUTPUT TEMPLATE placeholder.
    When generation hits the token cap before emitting <answer>, return empty rather than
    the placeholder text in the prompt."""
    matches = ANSWER_RE.findall(full_response)
    matches = [m for m in matches if "the name of the disease/entity" not in m]
    return matches[-1].strip() if matches else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--hf-id", required=True)
    parser.add_argument("--target", choices=["raw", "all", "noanswer"], default="all")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Rows per generate()+writeback checkpoint (resumability granularity).")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--dry-run", type=int, default=0,
                        help="If >0, regen only the first N targets and DO NOT writeback.")
    parser.add_argument("--backup-timestamp", default=None)
    args = parser.parse_args()

    responses_path = os.path.join(RESP_DIR, f"responses_{args.model_key}.json")
    if not os.path.exists(responses_path):
        sys.exit(f"FATAL: responses file not found: {responses_path}")

    ts = args.backup_timestamp or datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_dir = os.path.join(BACKUP_ROOT, ts)

    print(f"[regen-vllm] model_key={args.model_key}  hf_id={args.hf_id}  target={args.target}")
    print(f"[regen-vllm] responses_path={responses_path}")
    print(f"[regen-vllm] backup_dir={backup_dir}")
    print(f"[regen-vllm] max_new_tokens={args.max_new_tokens}  chunk_size={args.chunk_size}  gpu_mem_util={args.gpu_mem_util}")

    with open(responses_path) as f:
        rows = json.load(f)
    print(f"[regen-vllm] {len(rows)} total rows loaded")

    target_idx = select_targets(rows, args.target)
    print(f"[regen-vllm] {len(target_idx)} targets selected (mode={args.target})")

    if args.target == "noanswer":
        # these are already _regenerated:true; we deliberately re-run them with a higher cap.
        pending = list(target_idx)
        print(f"[regen-vllm] noanswer mode: re-running {len(pending)} no-answer rows (ignoring _regenerated flag)")
    else:
        pending = [i for i in target_idx if not rows[i].get("_regenerated")]
        skipped = len(target_idx) - len(pending)
        if skipped:
            print(f"[regen-vllm] resume: {skipped} already _regenerated:true; {len(pending)} to do")

    if args.dry_run:
        pending = pending[: args.dry_run]
        print(f"[regen-vllm] DRY-RUN: regenerating {len(pending)} cases; NO writeback")

    if not pending:
        print("[regen-vllm] nothing to do. Exiting.")
        return

    if not args.dry_run:
        bp = backup_original(responses_path, backup_dir)
        print(f"[regen-vllm] backup ready: {bp}")

    print(f"[regen-vllm] loading vLLM engine {args.hf_id} ...")
    llm = LLM(
        model=args.hf_id,
        dtype="auto",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_new_tokens + 4096,  # headroom for the prompt
    )
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, skip_special_tokens=True)
    print(f"[regen-vllm] engine ready")

    extract = load_extract() if args.dry_run else None

    def build_prompt(content):
        return tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    total = len(pending)
    done = 0
    for c0 in range(0, total, args.chunk_size):
        chunk = pending[c0 : c0 + args.chunk_size]
        prompts = [build_prompt(rows[i]["original_message"]["content"]) for i in chunk]
        outs = llm.generate(prompts, sp)  # continuous-batched across the whole chunk

        for i, prompt, out in zip(chunk, prompts, outs):
            gen = out.outputs[0].text
            new_full = prompt + gen
            new_answer = derive_answer(new_full)

            if args.dry_run:
                extracted = extract(new_full) if extract else ""
                leaks = [t for t in ["<|", "</think>", "<answer>", "assistantfinal", "analysis"] if t in extracted]
                print(f"\n--- dry-run pmcid={rows[i].get('pmcid')} ---")
                print(f"  gen tokens: {len(out.outputs[0].token_ids)}")
                print(f"  gen head  : {gen[:120]!r}")
                print(f"  NEW answer: {new_answer!r}")
                print(f"  extract head: {extracted[:120]!r}")
                print(f"  leaks: {leaks if leaks else 'NONE'}")
                continue

            rows[i]["full_response"] = new_full
            rows[i]["extracted_answer"] = new_answer
            rows[i]["_regenerated"] = True
            rows[i]["_regenerated_at"] = ts

        done += len(chunk)
        if not args.dry_run:
            atomic_write_json(rows, responses_path)
            print(f"[regen-vllm] checkpoint: wrote {responses_path}  ({done}/{total} done)")

    if args.dry_run:
        print(f"\n[regen-vllm] DRY-RUN complete. No writeback performed.")
    else:
        print(f"\n[regen-vllm] FINAL: {done}/{total} regenerated and written to {responses_path}")


if __name__ == "__main__":
    main()
