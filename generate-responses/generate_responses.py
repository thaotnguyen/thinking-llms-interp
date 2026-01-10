import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
import os
from datasets import load_dataset
import random
import json
import gc
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils
from tqdm import tqdm
from nnsight import CONFIG

CONFIG.API.APIKEY = os.getenv("NDIF_API_KEY", "")
machine_epsilon = sys.float_info.epsilon

MAX_TOKENS_IN_INPUT = 5000

# Prompt used to build questions from the MedCaseReasoning dataset
PROMPT_TEMPLATE = (
    "Read the following case presentation and give the most likely diagnosis.\nFirst, provide your internal reasoning for the diagnosis within the tags <think> ... </think>.\n"
    "Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>.\n\n"
    "----------------------------------------\n"
    "CASE PRESENTATION\n"
    "----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\n"
    "OUTPUT TEMPLATE\n"
    "----------------------------------------\n"
    "<think>\n"
    "...your internal reasoning for the diagnosis...\n"
    "</think>"
    "<answer>\n"
    "...the name of the disease/entity...\n"
    "</answer>"
)

# Parse arguments
parser = argparse.ArgumentParser(description="Generate responses from models without steering vectors")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to generate responses from")
parser.add_argument("--dataset", type=str, default="tmknguyen/MedCaseReasoning-filtered",
                    help="Dataset in HuggingFace to generate responses from", choices=["zou-lab/MedCaseReasoning", "tmknguyen/MedCaseReasoning-filtered"])
parser.add_argument("--dataset_split", type=str, default="nejm",
                    help="Split of dataset to generate responses from")
parser.add_argument("--max_tokens", type=int, default=8192,
                    help="Maximum number of tokens to generate")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--is_base_model", action="store_true", default=False,
                    help="Whether the model is a base model")
parser.add_argument("--tensor_parallel_size", type=int, default=-1,
                    help="Number of GPUs to use for tensor parallelism")
parser.add_argument("--dtype", type=str, default="auto",
                    help="Data type for model")
parser.add_argument("--temperature", type=float, default=0.0,
                    help="Temperature for sampling")
parser.add_argument("--top_p", type=float, default=1.0,
                    help="Top-p for sampling")
parser.add_argument("--engine", type=str, default="nnsight", choices=["nnsight", "vllm"],
                    help="Generation engine to use")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit (nnsight engine only)")
parser.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for processing (applies to both nnsight and vllm engines). Lower values use less memory.")
parser.add_argument("--save_every", type=int, default=128,
                    help="Save progress to the responses JSON after this many new items. Use 1 to save after every response.")
parser.add_argument("--flash_attn", action="store_true", default=False,
                    help="Enable FlashAttention-2 where supported (nnsight engine only)")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                    help="Fraction of GPU memory to use for vLLM (lower = more headroom, default 0.85)")
parser.add_argument("--max_num_batched_tokens", type=int, default=None,
                    help="Maximum number of batched tokens for vLLM (lower = less memory, default: auto)")
parser.add_argument("--max_num_seqs", type=int, default=None,
                    help="Maximum number of sequences to process concurrently in vLLM (lower = less memory, default: auto)")
parser.add_argument("--enable_chunked_prefill", action="store_true", default=False,
                    help="Enable chunked prefill in vLLM to reduce memory spikes")
parser.add_argument("--swap_space", type=int, default=4,
                    help="CPU swap space size in GB for vLLM KV cache overflow (0 to disable, default: 4)")
parser.add_argument("--quantization", type=str, default=None, choices=["awq", "gptq", "squeezellm"],
                    help="Quantization method for vLLM (awq/gptq/squeezellm). Model must be quantized with this method.")
parser.add_argument("--enforce_eager", action="store_true", default=False,
                    help="Disable CUDA graph in vLLM (uses more memory but can help with OOM)")
args, _ = parser.parse_known_args()

def get_valid_tensor_parallel_size(model_name: str, requested_tp_size: int, num_gpus: int) -> int:
    """Get a valid tensor parallel size that divides evenly into the model's attention heads.
    
    vLLM requires that num_attention_heads % tensor_parallel_size == 0.
    If the requested size doesn't work, we find the largest valid divisor <= num_gpus.
    """
    from transformers import AutoConfig
    
    # Load model config to get number of attention heads
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        num_heads = getattr(config, 'num_attention_heads', None)
        if num_heads is None:
            # Some models use num_key_value_heads or other attributes
            num_heads = getattr(config, 'num_key_value_heads', None)
        if num_heads is None:
            print(f"Warning: Could not determine number of attention heads for {model_name}. Using requested TP size {requested_tp_size}.")
            return min(requested_tp_size, num_gpus)
    except Exception as e:
        print(f"Warning: Could not load config for {model_name}: {e}. Using requested TP size {requested_tp_size}.")
        return min(requested_tp_size, num_gpus)
    
    if num_heads is None:
        return min(requested_tp_size, num_gpus)
    
    # Check if requested size is valid
    if requested_tp_size > 0 and num_heads % requested_tp_size == 0:
        return requested_tp_size
    
    # Find the largest valid divisor <= num_gpus
    valid_sizes = [tp for tp in range(1, num_gpus + 1) if num_heads % tp == 0]
    if not valid_sizes:
        # Fallback: use 1 if nothing works (shouldn't happen, but safety check)
        print(f"Warning: No valid tensor parallel size found for {num_heads} heads. Using TP size 1.")
        return 1
    
    best_size = max(valid_sizes)
    if best_size != requested_tp_size:
        print(f"Warning: Requested tensor_parallel_size={requested_tp_size} is invalid for model with {num_heads} attention heads.")
        print(f"         Using tensor_parallel_size={best_size} instead (valid divisors: {valid_sizes})")
    return best_size

if args.tensor_parallel_size == -1:
    args.tensor_parallel_size = torch.cuda.device_count()

def get_prompts(tokenizer, messages_list):
    if args.is_base_model:
        prompts = [msg["content"] for msg in messages_list]
    else:
        prompts = [tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages_list]
    
    # Heuristic: 1 token ~= 4 chars. The check was comparing chars to token limit.
    prompts_above_max_tokens = [prompt for prompt in prompts if len(prompt) > MAX_TOKENS_IN_INPUT * 4]
    if len(prompts_above_max_tokens) > 0:
        print(f"There are {len(prompts_above_max_tokens)} prompts above MAX_TOKENS_IN_INPUT ({MAX_TOKENS_IN_INPUT} tokens ~= {MAX_TOKENS_IN_INPUT*4} chars)")
        for prompt in prompts_above_max_tokens:
            print(f"Length: {len(prompt)}")
        raise ValueError(f"There are {len(prompts_above_max_tokens)} prompts above MAX_TOKENS_IN_INPUT")
    return prompts

def get_messages_from_dataset(dataset_name, rows) -> dict[str, dict[str, str]]:
    """Build per-question message dicts from a HF dataset split.

    For tmknguyen/MedCaseReasoning-filtered, compose the question using PROMPT_TEMPLATE
    with the row's case_prompt, and store the gold answer from final_diagnosis.
    The question_id will be a stable composite of pmcid and row index.
    """
    if dataset_name != "tmknguyen/MedCaseReasoning-filtered":
        raise ValueError(f"Dataset {dataset_name} not supported")

    messages_by_question_id: dict[str, dict[str, str]] = {}
    for i, row in enumerate(rows):
        case_prompt = row["case_prompt"]
        question_text = PROMPT_TEMPLATE.format(case_prompt=case_prompt)

        # Prefer pmcid if available for traceability; fall back to row index
        pmcid = row.get("pmcid", None)
        question_id = f"{pmcid}_{i}" if pmcid is not None else str(i)

        messages_by_question_id[question_id] = {
            "role": "user",
            "content": question_text,
            # Lightweight category marker
            "category": "diagnosis",
            # Store the natural-language question text for downstream assertions
            "question": question_text,
            # Include gold answer for evaluation
            "gold_answer": row.get("final_diagnosis", ""),
        }
    return messages_by_question_id

def process_model_output_batch_vllm(messages_batch, tokenizer, model):
    """Get model output for a batch of messages using vLLM"""
    # Lazy import to avoid requiring vLLM when using nnsight
    from vllm import SamplingParams
    prompts = get_prompts(tokenizer, messages_batch)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    request_outputs = model.generate(prompts, sampling_params)
    full_responses = [request_output.prompt + request_output.outputs[0].text for request_output in request_outputs]

    # Assert the questions are in the responses
    for message, response in zip(messages_batch, full_responses):
        assert message["question"] in response, f"Question {message['question']} not in response {response}"
    
    return full_responses


def process_model_output_batch_nnsight(messages_batch, tokenizer, model):
    """Get model output for a batch of messages using nnsight/HF generate.

    Returns full responses including the original prompt text, to mirror vLLM behavior
    and keep downstream assertions consistent.
    """
    prompts = get_prompts(tokenizer, messages_batch)

    # Tokenize with left padding as set in utils.load_model
    # Truncate long inputs to avoid excessive KV cache and OOM
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    # Shape assertions per research guidelines
    assert input_ids.dim() == 2, f"Expected 2D input_ids, got {input_ids.shape}"
    assert attention_mask.shape == input_ids.shape, f"attention_mask shape {attention_mask.shape} must match input_ids {input_ids.shape}"

    with model.generate({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, max_new_tokens=args.max_tokens, pad_token_id=tokenizer.pad_token_id, temperature=args.temperature + machine_epsilon, top_p=args.top_p) as gen:
        outputs = model.generator.output.save()

    assert outputs.shape[0] == input_ids.shape[0], "Batch size mismatch between outputs and inputs"

    # Compute per-sample prompt lengths (number of non-pad tokens)
    prompt_lengths = attention_mask.sum(dim=1)
    full_responses = []
    for i in range(outputs.shape[0]):
        prompt_len = int(prompt_lengths[i].item())
        assert prompt_len > 0, f"Empty prompt length for sample {i}"
        new_tokens = outputs[i][prompt_len:]
        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        full_responses.append(prompts[i] + gen_text)

    # Assert the questions are embedded in the full responses
    for message, response in zip(messages_batch, full_responses):
        assert message["question"] in response, f"Question not echoed in response"

    return full_responses


def process_messages(dataset_name, question_ids, messages_by_question_id, tokenizer, model, engine: str,
                    existing_responses: list | None = None, save_every: int | None = None,
                    responses_json_path: str | None = None, dataset_split: str | None = None):
    """Process a batch of messages and return response data"""
    if engine == "vllm":
        assert args.batch_size >= 1, f"batch_size must be >= 1, got {args.batch_size}"

        # Start from any preloaded responses to support resume
        all_data = list(existing_responses or [])
        last_save_len = len(all_data)
        pbar = tqdm(total=len(question_ids), desc=f"{engine} generation", unit="sample")
        for start in range(0, len(question_ids), args.batch_size):
            sub_ids = question_ids[start:start + args.batch_size]
            messages_batch = [messages_by_question_id[qid] for qid in sub_ids]
            responses = process_model_output_batch_vllm(messages_batch, tokenizer, model)

            for message, response, question_id in zip(messages_batch, responses, sub_ids):
                all_data.append({
                    "original_message": {"role": message["role"], "content": message["content"]},
                    "full_response": response,
                    "question_id": question_id,
                    "category": message["category"],
                    "question": message["question"],
                    "gold_answer": message.get("gold_answer", ""),
                    "dataset_name": dataset_name,
                    "dataset_split": dataset_split,
                })

            # Proactively release memory between micro-batches
            torch.cuda.empty_cache()
            gc.collect()

            # Periodically persist progress to disk
            if save_every and responses_json_path and (len(all_data) - last_save_len) >= save_every:
                save_responses(all_data, responses_json_path)
                last_save_len = len(all_data)
            pbar.update(len(sub_ids))
        pbar.close()

        return all_data
    elif engine == "nnsight":
        assert args.batch_size >= 1, f"batch_size must be >= 1, got {args.batch_size}"

        # Start from any preloaded responses to support resume
        all_data = list(existing_responses or [])
        last_save_len = len(all_data)
        pbar = tqdm(total=len(question_ids), desc=f"{engine} generation", unit="sample")
        for start in range(0, len(question_ids), args.batch_size):
            sub_ids = question_ids[start:start + args.batch_size]
            messages_batch = [messages_by_question_id[qid] for qid in sub_ids]
            responses = process_model_output_batch_nnsight(messages_batch, tokenizer, model)

            for message, response, question_id in zip(messages_batch, responses, sub_ids):
                all_data.append({
                    "original_message": {"role": message["role"], "content": message["content"]},
                    "full_response": response,
                    "question_id": question_id,
                    "category": message["category"],
                    "question": message["question"],
                    "gold_answer": message.get("gold_answer", ""),
                    "dataset_name": dataset_name,
                    "dataset_split": dataset_split,
                })

            # Proactively release memory between micro-batches
            torch.cuda.empty_cache()
            gc.collect()

            # Periodically persist progress to disk
            if save_every and responses_json_path and (len(all_data) - last_save_len) >= save_every:
                save_responses(all_data, responses_json_path)
                last_save_len = len(all_data)
            pbar.update(len(sub_ids))
        pbar.close()

        return all_data
    else:
        raise ValueError(f"Engine {engine} not supported")


def save_responses(responses_data, responses_json_path):
    """Atomically save responses to JSON, guarding against partial writes."""
    tmp_path = responses_json_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(responses_data, f, indent=2)
    os.replace(tmp_path, responses_json_path)
    print(f"Saved {len(responses_data)} responses to {responses_json_path}") 


def load_existing_responses(responses_json_path: str, dataset_name: str, dataset_split: str | None = None):
    """Load existing responses if present and return (list, processed_ids_set).

    Only counts entries that match the provided dataset_name and (if available)
    dataset_split to avoid accidental cross-dataset resumes.
    """
    if not os.path.exists(responses_json_path):
        return [], set()

    try:
        with open(responses_json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        # If the file is corrupt, back it up and start fresh
        corrupt_bak = responses_json_path + ".corrupt"
        try:
            os.replace(responses_json_path, corrupt_bak)
            print(f"Warning: Existing responses file was unreadable and was moved to {corrupt_bak}. Starting fresh.")
        except Exception:
            print("Warning: Existing responses file was unreadable. Starting fresh.")
        return [], set()

    processed = set()
    for item in data:
        same_dataset = item.get("dataset_name") == dataset_name
        same_split = (
            dataset_split is None
            or item.get("dataset_split") == dataset_split
            or item.get("dataset_split") is None  # treat legacy entries as matching any split
        )
        if same_dataset and same_split:
            qid = item.get("question_id")
            if qid is not None:
                processed.add(qid)
    print(f"Loaded {len(data)} existing responses; {len(processed)} match current dataset context and will be skipped.")
    return data, processed


# Main execution
if __name__ == "__main__":
    model_name = args.model

    print(f"Loading model {model_name} using engine '{args.engine}'...")
    if args.engine == "vllm":
        # Lazy import to avoid vllm requirement for nnsight
        from vllm import LLM
        from transformers import AutoTokenizer
        
        # Validate and adjust tensor parallel size based on model's attention heads
        num_gpus = torch.cuda.device_count()
        requested_tp_size = args.tensor_parallel_size if args.tensor_parallel_size != -1 else num_gpus
        validated_tp_size = get_valid_tensor_parallel_size(model_name, requested_tp_size, num_gpus)
        
        # Build vLLM kwargs with memory optimizations
        vllm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": validated_tp_size,
            "dtype": args.dtype,
            "seed": args.seed,
            "max_model_len": args.max_tokens + MAX_TOKENS_IN_INPUT,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "swap_space": args.swap_space,
        }
        
        # Add optional memory optimization parameters
        if args.max_num_batched_tokens is not None:
            vllm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
        if args.max_num_seqs is not None:
            vllm_kwargs["max_num_seqs"] = args.max_num_seqs
        if args.enable_chunked_prefill:
            vllm_kwargs["enable_chunked_prefill"] = True
        if args.quantization:
            vllm_kwargs["quantization"] = args.quantization
        if args.enforce_eager:
            vllm_kwargs["enforce_eager"] = True
        
        # Auto-adjust for large models (32B+)
        model_lower = model_name.lower()
        if "32b" in model_lower or "30b" in model_lower or "33b" in model_lower:
            print("Large model detected (30B+), applying aggressive memory optimizations...")
            # Aggressively reduce GPU memory utilization for large models
            if args.gpu_memory_utilization >= 0.70:
                vllm_kwargs["gpu_memory_utilization"] = 0.65
                print(f"  Reduced gpu_memory_utilization to 0.65 for large model (was {args.gpu_memory_utilization})")
            # Severely limit concurrent sequences to reduce memory pressure
            if args.max_num_seqs is None:
                vllm_kwargs["max_num_seqs"] = 8
                print(f"  Set max_num_seqs to 8 for large model")
            # Severely limit batched tokens to reduce peak memory
            if args.max_num_batched_tokens is None:
                vllm_kwargs["max_num_batched_tokens"] = 512
                print(f"  Set max_num_batched_tokens to 512 for large model")
            # Enable chunked prefill by default for large models
            if not args.enable_chunked_prefill:
                vllm_kwargs["enable_chunked_prefill"] = True
                print(f"  Enabled chunked_prefill for large model")
            # Increase swap space for large models
            if args.swap_space < 8:
                vllm_kwargs["swap_space"] = 8
                print(f"  Increased swap_space to 8GB for large model")
            # Disable CUDA graph to save memory (enforce_eager)
            if not args.enforce_eager:
                vllm_kwargs["enforce_eager"] = True
                print(f"  Enabled enforce_eager (disables CUDA graph) to save memory")
            # Suggest reducing batch size if it's too high
            if args.batch_size > 8:
                print(f"  Warning: batch_size={args.batch_size} may be too high for 32B model. Consider using --batch_size 4 or lower if OOM occurs.")
        
        print(f"vLLM initialization parameters: {', '.join(f'{k}={v}' for k, v in vllm_kwargs.items() if k != 'model')}")
        model = LLM(**vllm_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        # Use prior nnsight-based loading
        if args.flash_attn:
            print("FlashAttention requested (nnsight). Ensure flash-attn is installed and compatible with your GPU.")
        num_gpus = torch.cuda.device_count()
        print(f"Loading model with nnsight engine (device_map='auto' will distribute across {num_gpus} GPU(s))...")
        model, tokenizer = utils.load_model(model_name=model_name, load_in_8bit=args.load_in_8bit, enable_flash_attn=args.flash_attn)
        
        # Verify and report device distribution
        if hasattr(model, 'model') and hasattr(model.model, 'hf_device_map'):
            device_map = model.model.hf_device_map
            devices_used = set(device_map.values()) if device_map else set()
            print(f"Model distributed across {len(devices_used)} device(s): {sorted(devices_used)}")
        elif hasattr(model, 'model'):
            # Fallback: check device of model parameters
            devices_used = set()
            for param in model.model.parameters():
                if param.device.type == 'cuda':
                    devices_used.add(param.device)
            if devices_used:
                print(f"Model parameters found on {len(devices_used)} GPU(s): {sorted([str(d) for d in devices_used])}")
            else:
                print("Warning: Could not determine device distribution. Model may be on CPU or device detection failed.")

    # Create directories
    os.makedirs('results/vars', exist_ok=True)

    model_prefix = "base_" if args.is_base_model else ""
    responses_json_path = f"results/vars/{model_prefix}responses_{model_name.split('/')[-1].lower()}.json"

    random.seed(args.seed)

    if args.dataset == "tmknguyen/MedCaseReasoning-filtered" and args.dataset_split == "nejm":
        print(f"Loading local dataset from nejm_complete_graded_scraped.csv instead of huggingface...")
        import csv
        rows = []
        # Use absolute path as requested
        csv_path = "/home/ttn/Development/bmj/nejm_complete_graded_scraped.csv"
        
        if not os.path.exists(csv_path):
            # Path relative to this script: ../../nejm_complete_graded_scraped.csv
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, '../../nejm_complete_graded_scraped.csv')
        
        if not os.path.exists(csv_path):
            # Fallback to checking current working directory
            csv_path = "nejm_complete_graded_scraped.csv"
            
        if not os.path.exists(csv_path):
             raise FileNotFoundError(f"Could not find nejm_complete_graded_scraped.csv")

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    elif args.dataset == "tmknguyen/MedCaseReasoning-filtered" and args.dataset_split == "medqa":
        print(f"Loading local dataset from MedQA_complete_graded_data.csv instead of huggingface...")
        import csv
        rows = []
        # Path relative to this script: ../../MedQA_complete_graded_data.csv
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, '../../MedQA_complete_graded_data.csv')
        
        if not os.path.exists(csv_path):
            # Fallback to checking current working directory
            csv_path = "MedQA_complete_graded_data.csv"
            
        if not os.path.exists(csv_path):
             raise FileNotFoundError(f"Could not find MedQA_complete_graded_data.csv")

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Map columns
                case_prompt = row.get("question", "").strip()
                final_diagnosis = row.get("answer", "").strip()
                
                if not case_prompt:
                    print(f"Warning: Row {idx} missing question, skipping")
                    continue

                # Use index column if available, otherwise row index
                pmcid = row.get("index", "").strip()
                if not pmcid:
                    pmcid = f"medqa_{idx}"
                else:
                    pmcid = f"medqa_{pmcid}"

                rows.append({
                    "pmcid": pmcid,
                    "case_prompt": case_prompt,
                    "final_diagnosis": final_diagnosis,
                })
    else:
        ds = load_dataset(args.dataset)
        rows = ds[args.dataset_split]

    messages_by_question_id = get_messages_from_dataset(args.dataset, rows)
    question_ids = list(messages_by_question_id.keys())

    # Attempt to load existing progress and resume
    existing_responses, already_processed = load_existing_responses(responses_json_path, args.dataset, args.dataset_split)
    question_ids = [qid for qid in question_ids if qid not in already_processed]

    total_questions = len(messages_by_question_id)
    processed_count = len(already_processed)
    remaining_count = len(question_ids)
    print(f"Processing {remaining_count} questions in {args.dataset_split} split of {args.dataset} (resuming: {processed_count} already done out of {total_questions})")
    
    # Auto-reduce batch size for large models to prevent OOM
    if args.engine == "vllm":
        model_lower = model_name.lower()
        if ("32b" in model_lower or "30b" in model_lower or "33b" in model_lower) and args.batch_size > 4:
            original_batch_size = args.batch_size
            args.batch_size = min(args.batch_size, 4)
            if original_batch_size != args.batch_size:
                print(f"Automatically reduced batch_size from {original_batch_size} to {args.batch_size} for large model to prevent OOM")
    
    random.shuffle(question_ids)
    
    responses_data = process_messages(
        args.dataset,
        question_ids,
        messages_by_question_id,
        tokenizer,
        model,
        args.engine,
        existing_responses=existing_responses,
        save_every=args.save_every,
        responses_json_path=responses_json_path,
        dataset_split=args.dataset_split,
    )
        
    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()

    # Save final results
    save_responses(responses_data, responses_json_path)