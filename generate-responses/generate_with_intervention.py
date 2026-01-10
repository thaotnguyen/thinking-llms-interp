#!/usr/bin/env python3
"""
Generate responses with toxic phrase intervention.

If model generates "think likely", "alternatively", "wait", or "maybe",
truncate the response at that point and regenerate from there.

Compare accuracy with and without this intervention.
"""

import argparse
import sys
import json
import os
import re
import torch
from tqdm import tqdm
import dotenv
from datasets import load_dataset

dotenv.load_dotenv("../.env")

# Load model directly with transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Toxic phrases that predict failure
TOXIC_PHRASES = [
    "think likely",
    "alternatively",
    "wait",
    "maybe"
]

PROMPT_TEMPLATE = (
    "Read the following case presentation and give the most likely diagnosis.\n"
    "First, provide your internal reasoning for the diagnosis within the tags <think> ... </think>.\n"
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
    "</think><answer>\n"
    "...the name of the disease/entity...\n"
    "</answer>"
)

parser = argparse.ArgumentParser(description="Generate responses with toxic phrase intervention")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="Model to use")
parser.add_argument("--max_tokens", type=int, default=1024,
                    help="Maximum new tokens to generate (match pipeline default 1024)")
parser.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature (match pipeline default 0.7)")
parser.add_argument("--limit", type=int, default=100,
                    help="Number of questions to process")
parser.add_argument("--split", type=str, default="train",
                    help="Dataset split to use (train|test|validation), default train to match pipeline")
parser.add_argument("--samples", type=int, default=10,
                    help="Samples per case (match pipeline round1 default 10)")
parser.add_argument("--load_in_8bit", action="store_true",
                    help="Load model in 8-bit")
parser.add_argument("--max_retries", type=int, default=3,
                    help="Maximum number of truncation/retry attempts")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--use_vllm", action="store_true",
                    help="Use vLLM for faster generation (recommended for batch processing)")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Batch size for generation (only used with vLLM)")
parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.95,
                    help="GPU memory utilization for vLLM (0.0-1.0)")
args = parser.parse_args()


def find_toxic_phrase(text):
    """Find first occurrence of any toxic phrase in text.
    Returns (phrase, position) or (None, -1) if not found."""
    earliest_pos = len(text)
    earliest_phrase = None
    
    for phrase in TOXIC_PHRASES:
        # Case-insensitive search
        match = re.search(r'\b' + re.escape(phrase) + r'\b', text, re.IGNORECASE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
            earliest_phrase = phrase
    
    if earliest_phrase:
        return earliest_phrase, earliest_pos
    return None, -1


def generate_baseline_hf(model, tokenizer, prompt, max_tokens, temperature):
    """Generate without intervention - baseline using HuggingFace transformers."""
    # Use same format as multi_model_pipeline.py
    messages = [{"role": "user", "content": prompt}]
    try:
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        # Fallback if chat template not available
        prompt_text = prompt
    
    # Explicitly create attention_mask to avoid warning when pad_token == eos_token
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096, padding=False)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(model.device)
    
    with torch.no_grad():
        # Match multi_model_pipeline.py: temperature + machine_epsilon, top_p=0.95
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=max(temperature + 1e-12, 1e-6) if temperature > 0 else 0.0,
            do_sample=(temperature > 0),
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Match multi_model_pipeline.py format: decode only new tokens, then concatenate
    prompt_length = input_ids.shape[1]
    gen_tokens = outputs[0][prompt_length:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    
    # Return full_response = prompt + generated (matching multi_model_pipeline.py)
    full_response = prompt_text + gen_text
    
    return full_response


def generate_baseline_vllm(llm, tokenizer, prompt, max_tokens, temperature):
    """Generate without intervention - baseline using vLLM."""
    from vllm import SamplingParams
    
    messages = [{"role": "user", "content": prompt}]
    try:
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        prompt_text = prompt
    
    sampling_params = SamplingParams(
        temperature=max(temperature + 1e-12, 1e-6) if temperature > 0 else 0.0,
        top_p=0.95 if temperature > 0 else 1.0,
        max_tokens=max_tokens,
    )
    
    outputs = llm.generate([prompt_text], sampling_params)
    completion = outputs[0].outputs[0].text if outputs[0].outputs else ""
    full_response = prompt_text + completion
    
    return full_response


def generate_baseline(model, tokenizer, prompt, max_tokens, temperature, use_vllm=False, llm=None):
    """Generate without intervention - baseline (wrapper for HF or vLLM)."""
    if use_vllm and llm is not None:
        return generate_baseline_vllm(llm, tokenizer, prompt, max_tokens, temperature)
    else:
        return generate_baseline_hf(model, tokenizer, prompt, max_tokens, temperature)


def generate_with_intervention_hf(model, tokenizer, prompt, max_tokens, temperature, max_retries):
    """Generate with toxic phrase intervention - truncate and retry (HuggingFace)."""
    # Use same format as multi_model_pipeline.py
    messages = [{"role": "user", "content": prompt}]
    try:
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        # Fallback if chat template not available
        prompt_text = prompt
    
    current_text = prompt_text
    total_generated_tokens = 0
    interventions = []
    
    for retry_num in range(max_retries):
        # Encode current text with explicit attention_mask
        inputs = tokenizer(current_text, return_tensors="pt", truncation=True, max_length=4096, padding=False)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(model.device)
        
        # Calculate remaining tokens to generate
        remaining_tokens = max_tokens - total_generated_tokens
        if remaining_tokens <= 0:
            break
        
        # Generate
        with torch.no_grad():
            # Match multi_model_pipeline.py: temperature + machine_epsilon, top_p=0.95
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(remaining_tokens, max_tokens // max_retries),
                temperature=max(temperature + 1e-12, 1e-6) if temperature > 0 else 0.0,
                do_sample=(temperature > 0),
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Match multi_model_pipeline.py format: decode only new tokens
        prompt_length = input_ids.shape[1]
        gen_tokens = outputs[0][prompt_length:]
        new_generation = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        
        # Check for toxic phrases in new generation
        toxic_phrase, toxic_pos = find_toxic_phrase(new_generation)
        
        if toxic_phrase is None:
            # No toxic phrase found, we're done
            current_text = current_text + new_generation
            total_generated_tokens += len(tokenizer.encode(new_generation, add_special_tokens=False))
            break
        else:
            # Toxic phrase found - truncate at that point
            truncated_new_gen = new_generation[:toxic_pos]
            current_text = current_text + truncated_new_gen
            
            interventions.append({
                'retry': retry_num,
                'toxic_phrase': toxic_phrase,
                'position': toxic_pos,
                'truncated_at': len(current_text)
            })
            
            total_generated_tokens += len(tokenizer.encode(truncated_new_gen, add_special_tokens=False))
            
            # Check if we hit EOS before toxic phrase
            if tokenizer.eos_token in truncated_new_gen:
                break
            
            # Continue generating from truncated point
            continue
    
    return current_text, interventions


def generate_with_intervention_vllm(llm, tokenizer, prompt, max_tokens, temperature, max_retries):
    """Generate with toxic phrase intervention - truncate and retry (vLLM)."""
    from vllm import SamplingParams
    
    messages = [{"role": "user", "content": prompt}]
    try:
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        prompt_text = prompt
    
    current_text = prompt_text
    total_generated_tokens = 0
    interventions = []
    
    for retry_num in range(max_retries):
        # Calculate remaining tokens
        remaining_tokens = max_tokens - total_generated_tokens
        if remaining_tokens <= 0:
            break
        
        sampling_params = SamplingParams(
            temperature=max(temperature + 1e-12, 1e-6) if temperature > 0 else 0.0,
            top_p=0.95 if temperature > 0 else 1.0,
            max_tokens=min(remaining_tokens, max_tokens // max_retries),
        )
        
        outputs = llm.generate([current_text], sampling_params)
        new_generation = outputs[0].outputs[0].text if outputs[0].outputs else ""
        
        # Check for toxic phrases
        toxic_phrase, toxic_pos = find_toxic_phrase(new_generation)
        
        if toxic_phrase is None:
            # No toxic phrase found, we're done
            current_text = current_text + new_generation
            total_generated_tokens += len(tokenizer.encode(new_generation, add_special_tokens=False))
            break
        else:
            # Toxic phrase found - truncate at that point
            truncated_new_gen = new_generation[:toxic_pos]
            current_text = current_text + truncated_new_gen
            
            interventions.append({
                'retry': retry_num,
                'toxic_phrase': toxic_phrase,
                'position': toxic_pos,
                'truncated_at': len(current_text)
            })
            
            total_generated_tokens += len(tokenizer.encode(truncated_new_gen, add_special_tokens=False))
            
            # Check if we hit EOS
            if tokenizer.eos_token and tokenizer.eos_token in truncated_new_gen:
                break
            
            # Continue generating from truncated point
            continue
    
    return current_text, interventions


def generate_with_intervention(model, tokenizer, prompt, max_tokens, temperature, max_retries, use_vllm=False, llm=None):
    """Generate with toxic phrase intervention - truncate and retry (wrapper)."""
    if use_vllm and llm is not None:
        return generate_with_intervention_vllm(llm, tokenizer, prompt, max_tokens, temperature, max_retries)
    else:
        return generate_with_intervention_hf(model, tokenizer, prompt, max_tokens, temperature, max_retries)


def get_messages_from_dataset(dataset_name, rows):
    """Extract messages from dataset - matching multi_model_pipeline.py format."""
    messages_by_id = {}
    
    for idx, row in enumerate(rows):
        if dataset_name == "tmknguyen/MedCaseReasoning-filtered":
            # Match multi_model_pipeline.py field names
            pmcid = str(row.get("pmcid", ""))
            case_prompt = row.get("case_prompt") or row.get("case_report") or row.get("case_presentation") or ""
            gold_answer = row.get("final_diagnosis") or row.get("answer") or ""
            question_id = f"{pmcid}_{idx}"  # Use index for test set
            category = row.get("category", "diagnosis")
        else:
            question_id = row.get("id", str(len(messages_by_id)))
            case_prompt = row.get("case_report", row.get("question", ""))
            gold_answer = row.get("answer", row.get("gold", ""))
            category = "diagnosis"
        
        # Use same prompt building as multi_model_pipeline.py
        prompt = PROMPT_TEMPLATE.format(case_prompt=case_prompt)
        
        messages_by_id[question_id] = {
            "question": prompt,
            "gold_answer": gold_answer,
            "category": category,
            "case_prompt": case_prompt,
            "pmcid": pmcid if dataset_name == "tmknguyen/MedCaseReasoning-filtered" else None
        }
    
    return messages_by_id


def main():
    torch.manual_seed(args.seed)
    
    # Get model ID
    model_name = args.model
    model_id = model_name.split('/')[-1].lower()
    
    # Load tokenizer (needed for both HF and vLLM)
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Set pad token to eos token (common practice when no pad token exists)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Ensure padding is on the right side (standard for generation)
    tokenizer.padding_side = "right"
    
    # Load model (HF or vLLM)
    model = None
    llm = None
    
    if args.use_vllm:
        print(f"Loading model {model_name} with vLLM...")
        try:
            from vllm import LLM
            llm = LLM(
                model=model_name,
                max_model_len=8192,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                trust_remote_code=True,
            )
            print("  ✓ Loaded with vLLM")
        except ImportError:
            print("  ⚠️  vLLM not available, falling back to HuggingFace")
            args.use_vllm = False
        except Exception as e:
            print(f"  ⚠️  vLLM failed: {e}, falling back to HuggingFace")
            args.use_vllm = False
    
    if not args.use_vllm:
        print(f"Loading model {model_name} with HuggingFace...")
        # Load model with proper 8-bit support
        if args.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("  ✓ Loaded in 8-bit mode")
            except ImportError:
                print("  ⚠️  bitsandbytes not available, falling back to regular loading")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
        print("  ✓ Loaded with HuggingFace")
    
    # Load test dataset
    print(f"Loading dataset split '{args.split}'...")
    ds = load_dataset("tmknguyen/MedCaseReasoning-filtered")
    split_name = args.split if args.split in ds else "train"
    rows = ds[split_name]
    
    messages_by_id = get_messages_from_dataset("tmknguyen/MedCaseReasoning-filtered", rows)
    question_ids = list(messages_by_id.keys())[:args.limit]
    
    print(f"Generating responses for {len(question_ids)} questions...")
    print(f"Engine: {'vLLM' if args.use_vllm else 'HuggingFace'}")
    if args.use_vllm:
        print(f"Batch size: {args.batch_size}")
    print(f"Baseline: standard generation")
    print(f"Intervention: truncate and retry on toxic phrases: {', '.join(TOXIC_PHRASES)}")
    print(f"Max retries: {args.max_retries}")
    print(f"Samples per case: {args.samples}")
    
    # Prepare output
    os.makedirs("results/intervention_experiment", exist_ok=True)
    baseline_responses = []
    intervention_responses = []
    
    # Process in batches if using vLLM
    if args.use_vllm and args.batch_size > 1:
        # Batch processing for vLLM (baseline only - intervention needs per-item logic)
        from vllm import SamplingParams
        
        print(f"\nGenerating baseline in batches of {args.batch_size} with {args.samples} samples/case...")
        for start_idx in tqdm(range(0, len(question_ids), args.batch_size), desc="Baseline batches"):
            batch_ids = question_ids[start_idx:start_idx + args.batch_size]
            batch_prompts = []
            batch_meta = []  # (qid, message, prompt, sample_index, prompt_text)

            for qid in batch_ids:
                message = messages_by_id[qid]
                prompt = message["question"]
                messages_tpl = [{"role": "user", "content": prompt}]
                try:
                    prompt_text_base = tokenizer.apply_chat_template(messages_tpl, add_generation_prompt=True, tokenize=False)
                except Exception:
                    prompt_text_base = prompt

                for sample_index in range(args.samples):
                    batch_prompts.append(prompt_text_base)
                    batch_meta.append((qid, message, prompt, sample_index, prompt_text_base))

            sampling_params = SamplingParams(
                temperature=max(args.temperature + 1e-12, 1e-6) if args.temperature > 0 else 0.0,
                top_p=0.95 if args.temperature > 0 else 1.0,
                max_tokens=args.max_tokens,
            )

            outputs = llm.generate(batch_prompts, sampling_params)

            for (qid, message, prompt, sample_index, prompt_text), output in zip(batch_meta, outputs):
                completion = output.outputs[0].text if output.outputs else ""
                baseline_full = prompt_text + completion

                baseline_responses.append({
                    'question_id': qid,
                    'category': message.get('category', 'diagnosis'),
                    'dataset_name': "tmknguyen/MedCaseReasoning-filtered",
                    'dataset_split': split_name,
                    'original_message': {"role": "user", "content": prompt},
                    'full_response': baseline_full,
                    'gold_answer': message['gold_answer'],
                    'question': prompt,
                    'pmcid': message.get('pmcid'),
                    'method': 'baseline',
                    'sample_index': sample_index,
                })
        
        # Intervention still needs to be per-item (can't batch due to truncation logic)
        print(f"\nGenerating intervention (per-item due to truncation logic)...")
        for question_id in tqdm(question_ids, desc="Intervention"):
            message = messages_by_id[question_id]
            prompt = message["question"]
            for sample_index in range(args.samples):
                intervention_full, interventions = generate_with_intervention(
                    model, tokenizer, prompt,
                    args.max_tokens, args.temperature, args.max_retries,
                    use_vllm=True, llm=llm
                )

                intervention_responses.append({
                    'question_id': question_id,
                    'category': message.get('category', 'diagnosis'),
                    'dataset_name': "tmknguyen/MedCaseReasoning-filtered",
                    'dataset_split': split_name,
                    'original_message': {"role": "user", "content": prompt},
                    'full_response': intervention_full,
                    'gold_answer': message['gold_answer'],
                    'question': prompt,
                    'pmcid': message.get('pmcid'),
                    'method': 'intervention',
                    'interventions': interventions,
                    'num_interventions': len(interventions),
                    'sample_index': sample_index,
                })
    else:
        # Per-item processing (HF or vLLM single-item)
        for question_id in tqdm(question_ids, desc="Generating"):
            message = messages_by_id[question_id]
            prompt = message["question"]

            for sample_index in range(args.samples):
                # Optionally vary seed slightly per sample for HF
                try:
                    torch.manual_seed(int(args.seed) + int(sample_index))
                except Exception:
                    pass

                # Generate baseline (no intervention)
                baseline_full = generate_baseline(
                    model, tokenizer, prompt,
                    args.max_tokens, args.temperature,
                    use_vllm=args.use_vllm, llm=llm
                )

                baseline_responses.append({
                    'question_id': question_id,
                    'category': message.get('category', 'diagnosis'),
                    'dataset_name': "tmknguyen/MedCaseReasoning-filtered",
                    'dataset_split': split_name,
                    'original_message': {"role": "user", "content": prompt},
                    'full_response': baseline_full,
                    'gold_answer': message['gold_answer'],
                    'question': prompt,
                    'pmcid': message.get('pmcid'),
                    'method': 'baseline',
                    'sample_index': sample_index,
                })

                # Generate with intervention
                intervention_full, interventions = generate_with_intervention(
                    model, tokenizer, prompt,
                    args.max_tokens, args.temperature, args.max_retries,
                    use_vllm=args.use_vllm, llm=llm
                )

                intervention_responses.append({
                    'question_id': question_id,
                    'category': message.get('category', 'diagnosis'),
                    'dataset_name': "tmknguyen/MedCaseReasoning-filtered",
                    'dataset_split': split_name,
                    'original_message': {"role": "user", "content": prompt},
                    'full_response': intervention_full,
                    'gold_answer': message['gold_answer'],
                    'question': prompt,
                    'pmcid': message.get('pmcid'),
                    'method': 'intervention',
                    'interventions': interventions,
                    'num_interventions': len(interventions),
                    'sample_index': sample_index,
                })
    
    # Save results
    baseline_file = f"results/intervention_experiment/responses_{model_id}_baseline.json"
    intervention_file = f"results/intervention_experiment/responses_{model_id}_intervention.json"
    
    with open(baseline_file, 'w') as f:
        json.dump(baseline_responses, f, indent=2)
    
    with open(intervention_file, 'w') as f:
        json.dump(intervention_responses, f, indent=2)
    
    # Diagnostic: Check answer extraction
    import re
    ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
    
    def extract_answer_diagnostic(full_response):
        matches = ANSWER_RE.findall(full_response or "")
        if not matches:
            return ""
        ans = matches[-1].strip()
        placeholder = "...the name of the disease/entity..."
        if placeholder.lower() in ans.lower():
            return ""
        return ans
    
    baseline_extracted = [extract_answer_diagnostic(r['full_response']) for r in baseline_responses]
    intervention_extracted = [extract_answer_diagnostic(r['full_response']) for r in intervention_responses]
    
    baseline_with_answers = sum(1 for a in baseline_extracted if a)
    intervention_with_answers = sum(1 for a in intervention_extracted if a)
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Baseline responses: {baseline_file}")
    print(f"Intervention responses: {intervention_file}")
    print(f"\n📊 DIAGNOSTIC:")
    print(f"  Baseline: {baseline_with_answers}/{len(baseline_responses)} responses have extractable answers ({baseline_with_answers/len(baseline_responses)*100:.1f}%)")
    print(f"  Intervention: {intervention_with_answers}/{len(intervention_responses)} responses have extractable answers ({intervention_with_answers/len(intervention_responses)*100:.1f}%)")
    
    if baseline_with_answers == 0:
        print(f"\n⚠️  WARNING: No answers extracted from baseline responses!")
        print(f"  Sample response (first 500 chars):")
        if baseline_responses:
            print(f"  {baseline_responses[0]['full_response'][:500]}...")
        print(f"  Check if model is generating <answer> tags correctly.")
    
    # Calculate intervention statistics
    total_interventions = sum(r['num_interventions'] for r in intervention_responses)
    cases_with_interventions = sum(1 for r in intervention_responses if r['num_interventions'] > 0)
    
    print(f"\nIntervention statistics:")
    print(f"  Cases with toxic phrases: {cases_with_interventions}/{len(intervention_responses)} ({cases_with_interventions/len(intervention_responses)*100:.1f}%)")
    print(f"  Total interventions: {total_interventions}")
    print(f"  Average interventions per case with intervention: {total_interventions/max(1, cases_with_interventions):.2f}")
    
    print(f"\nNext steps:")
    print(f"1. Grade baseline:")
    print(f"   python ../grade_responses.py --input {baseline_file} --output {baseline_file.replace('.json', '.graded.json')} --model gpt-5-nano --workers 1000")
    print(f"2. Grade intervention:")
    print(f"   python ../grade_responses.py --input {intervention_file} --output {intervention_file.replace('.json', '.graded.json')} --model gpt-5-nano --workers 1000")
    print(f"3. Compare accuracies")


if __name__ == "__main__":
    main()

