#!/usr/bin/env python3
"""
Generate responses with SAE feature steering using nnsight.

This script loads a model, applies steering towards a specific SAE feature
using OPTIMIZED steering vectors from train-vectors, and generates responses.
"""
import argparse
import sys
import json
import os
# Mitigate third-party frameworks pre-allocating GPU memory
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import torch
from tqdm import tqdm
import dotenv
from datasets import load_dataset

dotenv.load_dotenv("../.env")

# Add parent directory to path for imports
sys.path.append('..')
from utils.utils import load_model
from utils.sae import load_sae
from generate_responses import get_messages_from_dataset

parser = argparse.ArgumentParser(description="Generate responses with SAE feature steering")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to use for generation")
parser.add_argument("--layer", type=int, required=True,
                    help="Layer to apply steering at")
parser.add_argument("--n_clusters", type=int, required=True,
                    help="Number of clusters in the SAE")
parser.add_argument("--feature_idx", type=int, required=True,
                    help="Feature index to steer towards (e.g., 0 for idx0)")
parser.add_argument("--coefficient", type=float, default=1.0,
                    help="Steering coefficient (strength of intervention)")
parser.add_argument("--use_raw_sae", action="store_true",
                    help="Use raw SAE decoder vectors instead of optimized steering vectors")
parser.add_argument("--dataset", type=str, default="jhu-clsp/medcon-qa",
                    help="Dataset to generate responses for")
parser.add_argument("--dataset_split", type=str, default="valid",
                    help="Dataset split to use")
parser.add_argument("--max_tokens", type=int, default=2048,
                    help="Maximum tokens to generate per response")
parser.add_argument("--temperature", type=float, default=0.8,
                    help="Sampling temperature")
parser.add_argument("--limit", type=int, default=None,
                    help="Limit number of questions to process (None = all)")
parser.add_argument("--load_in_8bit", action="store_true",
                    help="Load model in 8-bit precision")
parser.add_argument("--engine", type=str, default="nnsight", choices=["nnsight", "hf", "vllm"],
                    help="Generation engine: nnsight (token loop), hf (forward hook), vllm (fast baseline; steering fallback to hf)")
parser.add_argument("--diagnose_memory", action="store_true",
                    help="Print CUDA memory stats before and after generation")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
args = parser.parse_args()


def generate_with_sae_steering_nnsight(model, tokenizer, steering_vector, input_ids, attention_mask,
                                       layer, coefficient, max_new_tokens,
                                       temperature, pad_token_id):
    """
    Generate text token-by-token while applying steering vector.
    
    Args:
        model: The nnsight-wrapped model
        tokenizer: Tokenizer
        steering_vector: Pre-computed steering vector [hidden_dim] (optimized or raw SAE)
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask
        layer: Layer to intervene at
        coefficient: Steering strength (positive = towards, negative = away)
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        pad_token_id: Padding token ID
        
    Returns:
        Generated token IDs (full sequence including prompt)
        
    Note:
        This function generates token-by-token using .trace() following the
        pattern from train-vectors/apply_prm_dom_vector.py
    """
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    steering_vector = steering_vector.to(device).to(dtype)
    
    # Generate token-by-token
    current_ids = input_ids.clone()
    generated_tokens = 0
    
    while generated_tokens < max_new_tokens:
        with torch.no_grad():
            with model.trace({
                "input_ids": current_ids,
                "attention_mask": torch.ones_like(current_ids),
            }, scan=False, validate=False):
                # Apply steering to the layer output
                # Handle both 2D (seq, hidden) and 3D (batch, seq, hidden) shapes
                if coefficient != 0.0:
                    layer_output = model.model.layers[layer].output[0]
                    
                    # Apply steering to the last token position
                    if layer_output.dim() == 2:
                        # 2D: (seq, hidden) - apply to last token
                        layer_output[-1, :] += coefficient * steering_vector
                    elif layer_output.dim() == 3:
                        # 3D: (batch, seq, hidden) - apply to last token
                        layer_output[:, -1, :] += coefficient * steering_vector
                    else:
                        raise ValueError(f"Unexpected layer output shape: {layer_output.shape}")
                
                # Get logits for next token
                logits = model.lm_head.output.save()
        
        # Sample next token
        if temperature > 0:
            # Sample from distribution
            probs = torch.softmax(logits[0, -1, :] / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
        else:
            # Greedy decoding
            next_token_id = logits[0, -1, :].argmax(dim=-1).item()
        
        # Check for EOS
        if next_token_id == tokenizer.eos_token_id or next_token_id == pad_token_id:
            break
        
        # Append to sequence
        next_token_tensor = torch.tensor([[next_token_id]], device=device)
        current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
        generated_tokens += 1
    
    return current_ids


def generate_with_sae_steering_hf(model, tokenizer, steering_vector, prompt_text,
                                  layer, coefficient, max_new_tokens, temperature):
    """Generate using HF generate with a forward hook to apply steering each forward pass.

    For linear steering we add coef * vector to hidden states of target layer.
    For adaptive/resid_lora styles we mimic the math used in training scripts.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Normalize steering vector formats
    if isinstance(steering_vector, torch.Tensor):
        vec = steering_vector.to(device=device, dtype=dtype)
    elif isinstance(steering_vector, dict):
        vec = {k: v.to(device=device, dtype=dtype) for k, v in steering_vector.items()}
    else:
        raise ValueError("Unsupported steering vector format")

    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

    if coefficient == 0.0:
        with torch.no_grad():
            return model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
            )

    def hook_fn(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if isinstance(vec, torch.Tensor):
            hidden = hidden + coefficient * vec
        elif 'A' in vec and 'B' in vec:  # resid_lora
            b, s, h = hidden.shape
            flat = hidden.reshape(-1, h).T  # [h, b*s]
            lo = vec['A'] @ flat
            lout = vec['B'] @ lo
            delta = vec['alpha'] * lout.T.reshape(b, s, h)
            hidden = hidden + delta * coefficient
        elif 'vector' in vec and 'W1' in vec:  # adaptive linear
            b, s, h = hidden.shape
            gate_in = hidden @ vec['W1'].T + vec['b1'].unsqueeze(0).unsqueeze(0)
            gate = torch.relu(gate_in) @ vec['W2'].T + vec['b2'].unsqueeze(0).unsqueeze(0)
            gate = torch.sigmoid(gate)
            hidden = hidden + coefficient * gate * vec['vector'].unsqueeze(0).unsqueeze(0)
        return (hidden,) if isinstance(output, tuple) else hidden

    handle = model.model.layers[args.layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
            )
        return gen_ids
    finally:
        handle.remove()


def generate_with_sae_steering_vllm(model_name, tokenizer, steering_vector, prompt_text,
                                    layer, coefficient, max_new_tokens, temperature):
    """Best-effort vLLM path.

    NOTE: vLLM does not expose per-layer hidden states for arbitrary intervention without
    custom model patching. For coefficient==0 we do fast baseline generation. For non-zero
    coefficients we fall back to HF forward-hook path (slower) to ensure correctness.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise RuntimeError("vLLM not installed. pip install vllm to use --engine vllm")

    if coefficient == 0.0:
        llm = LLM(model_name, dtype="bfloat16")
        sampling = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
        outputs = llm.generate([prompt_text], sampling)
        full_text = outputs[0].outputs[0].text
        # Reconstruct full response (vLLM returns only completion by default)
        return prompt_text + full_text
    else:
        print("[vLLM] Steering not directly supported; falling back to HF forward-hook path.")
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        return hf_model, generate_with_sae_steering_hf(hf_model, tokenizer, steering_vector, prompt_text,
                                                       layer, coefficient, max_new_tokens, temperature)


def load_steering_vector(model_id, layer, feature_idx, use_raw_sae, n_clusters=None):
    """
    Load steering vector for the specified feature.
    
    Args:
        model_id: Model identifier (short form)
        layer: Layer number
        feature_idx: Feature index (0 to n_clusters-1)
        use_raw_sae: If True, use raw SAE decoder; if False, use optimized vector
        n_clusters: Number of clusters (required if use_raw_sae=True)
    
    Returns:
        steering_vector tensor [hidden_dim]
    """
    if use_raw_sae:
        # Use raw SAE decoder vectors
        if n_clusters is None:
            raise ValueError("n_clusters required when using raw SAE vectors")
        
        print(f"Loading raw SAE decoder vector...")
        sae, _ = load_sae(model_id, layer, n_clusters)
        steering_vector = sae.W_dec[feature_idx, :].detach().clone()
        print(f"  Using raw SAE W_dec[{feature_idx}, :] (shape: {steering_vector.shape})")
        return steering_vector
    else:
        # Load optimized steering vector from train-vectors
        category_name = f"idx{feature_idx}"
        vector_path = f"../train-vectors/results/vars/optimized_vectors/{model_id}_{category_name}.pt"
        
        if not os.path.exists(vector_path):
            raise FileNotFoundError(
                f"Optimized steering vector not found: {vector_path}\n"
                f"Please run: python train-vectors/optimize_steering_vectors.py "
                f"--model <model> --layer {layer} --steering_vector_idx {feature_idx} --max_iters 50\n"
                f"Or use --use_raw_sae flag to use raw SAE decoder vectors instead."
            )
        
        print(f"Loading optimized steering vector from {vector_path}...")
        obj = torch.load(vector_path, map_location='cpu')
        
        # Handle different save formats
        if isinstance(obj, dict) and category_name in obj:
            steering_vector = obj[category_name]
        elif isinstance(obj, torch.Tensor):
            steering_vector = obj
        else:
            raise ValueError(f"Unexpected vector format in {vector_path}: {type(obj)}")
        
        print(f"  Loaded optimized vector for {category_name} (shape: {steering_vector.shape})")
        return steering_vector


def main():
    torch.manual_seed(args.seed)
    
    # Get model ID
    model_name = args.model
    model_id = model_name.split('/')[-1].lower()
    
    if args.diagnose_memory:
        print("[Memory] Before model load:")
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary(device=None, abbreviated=True))

    print(f"Loading model {model_name} (engine={args.engine})...")
    model = None
    tokenizer = None
    if args.engine in ("nnsight", "hf"):
        model, tokenizer = load_model(model_name=model_name, load_in_8bit=args.load_in_8bit)
    elif args.engine == "vllm":
        # Only need tokenizer for prompt construction; reuse HF tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f"Unknown engine: {args.engine}")

    if args.diagnose_memory and torch.cuda.is_available():
        print("[Memory] After model load:")
        print(torch.cuda.memory_summary(device=None, abbreviated=True))
    
    # Verify feature index is valid
    if args.n_clusters and args.feature_idx >= args.n_clusters:
        raise ValueError(f"Feature index {args.feature_idx} >= n_clusters {args.n_clusters}")
    
    # Load steering vector (optimized or raw SAE)
    steering_vector = load_steering_vector(
        model_id, args.layer, args.feature_idx, 
        args.use_raw_sae, args.n_clusters
    )
    
    print(f"\nSteering configuration:")
    print(f"  Model: {model_name}")
    print(f"  Layer: {args.layer}")
    print(f"  Feature: idx{args.feature_idx}")
    print(f"  Coefficient: {args.coefficient}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Vector type: {'Raw SAE decoder' if args.use_raw_sae else 'Optimized steering vector'}")
    print(f"  Vector norm: {steering_vector.norm().item():.4f}")
    
    # Load dataset
    print(f"Loading dataset {args.dataset} split {args.dataset_split}...")
    ds = load_dataset(args.dataset)
    rows = ds[args.dataset_split]
    
    # Get messages
    messages_by_question_id = get_messages_from_dataset(args.dataset, rows)
    question_ids = list(messages_by_question_id.keys())
    
    if args.limit:
        question_ids = question_ids[:args.limit]
    
    print(f"Generating responses for {len(question_ids)} questions...")
    
    # Prepare output filename
    output_file = f"results/vars/responses_{model_id}_layer{args.layer}_idx{args.feature_idx}_coef{args.coefficient}.json"
    os.makedirs("results/vars", exist_ok=True)
    
    responses = []
    
    for question_id in tqdm(question_ids, desc="Generating"):
        message = messages_by_question_id[question_id]
        
        # Prepare prompt
        messages = [{"role": "user", "content": message["question"]}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Tokenize
        if args.engine in ("nnsight", "hf"):
            if args.engine == "nnsight":
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded["attention_mask"].to(model.device)
                gen_ids = generate_with_sae_steering_nnsight(
                    model, tokenizer, steering_vector,
                    input_ids, attention_mask,
                    args.layer, args.coefficient,
                    args.max_tokens, args.temperature,
                    tokenizer.pad_token_id
                )
                full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                if full_text.startswith(prompt):
                    generated_text = full_text[len(prompt):]
                else:
                    pt_len = input_ids.shape[1]
                    generated_text = tokenizer.decode(gen_ids[0][pt_len:], skip_special_tokens=True)
            else:  # hf engine
                gen_ids = generate_with_sae_steering_hf(
                    model, tokenizer, steering_vector, prompt,
                    args.layer, args.coefficient, args.max_tokens, args.temperature
                )
                full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                generated_text = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
        else:  # vllm
            if args.coefficient == 0.0:
                # Baseline fast path
                try:
                    from vllm import LLM, SamplingParams
                except ImportError:
                    raise RuntimeError("vLLM not installed. pip install vllm")
                llm = LLM(model_name, dtype="bfloat16")
                sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
                outputs = llm.generate([prompt], sampling)
                completion = outputs[0].outputs[0].text
                full_text = prompt + completion
                generated_text = completion
            else:
                # Fallback to HF for steering
                from transformers import AutoModelForCausalLM
                hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
                gen_ids = generate_with_sae_steering_hf(
                    hf_model, tokenizer, steering_vector, prompt,
                    args.layer, args.coefficient, args.max_tokens, args.temperature
                )
                full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                generated_text = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            
            # Store response
            response_item = {
                'question_id': question_id,
                'category': message.get('category', 'unknown'),
                'dataset_name': args.dataset,
                'dataset_split': args.dataset_split,
                'original_message': {"role": "user", "content": message["question"]},
                'full_response': prompt + generated_text,
                'generated_text': generated_text,
                'steering': {
                    'layer': args.layer,
                    'feature_idx': args.feature_idx,
                    'coefficient': args.coefficient,
                    'n_clusters': args.n_clusters,
                }
            }
            
            # Add gold answer if available
            if 'gold_answer' in message:
                response_item['gold_answer'] = message['gold_answer']
            
            responses.append(response_item)
            
            # Save incrementally every 10 responses
            if len(responses) % 10 == 0:
                with open(output_file, 'w') as f:
                    json.dump(responses, f, indent=2)
                    
        # (No try/except now; propagate errors for easier debugging)
    
    # Final save
    print(f"\nSaving {len(responses)} responses to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=2)
    
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Generated: {len(responses)}/{len(question_ids)} responses (engine={args.engine})")
    print(f"Output: {output_file}")
    print(f"Steering: layer {args.layer}, idx{args.feature_idx}, coef={args.coefficient}")


if __name__ == "__main__":
    main()

