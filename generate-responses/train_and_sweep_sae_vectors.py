#!/usr/bin/env python3
"""
Train steering vectors from SAE taxonomy and run coefficient sweep with uniform token-level steering.

This script:
1. Trains a steering vector for a chosen SAE latent (taxonomy item)
2. Performs a coefficient sweep, steering uniformly on all tokens
3. Grades the responses
4. Plots accuracy vs coefficient
"""
import argparse
import os
import sys
import json
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import utils
from utils.clustering import get_latent_descriptions
from utils import steering_opt
from utils.responses import extract_thinking_process

parser = argparse.ArgumentParser(description="Train SAE-based steering vector and run coefficient sweep")

# Model and layer configuration
parser.add_argument("--model", type=str, required=True,
                    help="Model to use for generation (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)")
parser.add_argument("--thinking_model", type=str, default=None,
                    help="Model to train vectors on (defaults to same as --model)")
parser.add_argument("--layer", type=int, required=True,
                    help="Layer to apply steering at")
parser.add_argument("--sae_layer", type=int, default=None,
                    help="SAE layer to load taxonomy from (defaults to same as --layer)")

# SAE taxonomy configuration
parser.add_argument("--n_clusters", type=int, required=True,
                    help="Number of clusters in SAE taxonomy")
parser.add_argument("--feature_idx", type=int, required=True,
                    help="Feature index (taxonomy item) to train vector for")
parser.add_argument("--clustering_method", type=str, default="sae_topk",
                    help="Clustering method used for taxonomy")

# Vector training configuration
parser.add_argument("--n_training_examples", type=int, default=8,
                    help="Number of training examples per category for vector optimization")
parser.add_argument("--batch_size", type=int, default=6,
                    help="Batch size for training and generation (reduce for less memory usage)")
parser.add_argument("--max_iters", type=int, default=1000,
                    help="Maximum optimization iterations for vector training")
parser.add_argument("--lr", type=str, default="1e-1",
                    help="Learning rate(s) for vector optimization (comma-separated)")
parser.add_argument("--steering_type", type=str, choices=["linear", "adaptive_linear", "resid_lora"], 
                    default="linear", help="Type of steering to optimize")
parser.add_argument("--context_sentences", type=int, default=0,
                    help="Number of additional sentences to include after target completion")

# Coefficient sweep configuration
parser.add_argument("--coefficients", type=float, nargs="+", 
                    default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0],
                    help="List of coefficients to test")

# Dataset and generation configuration
parser.add_argument("--dataset", type=str, default="tmknguyen/MedCaseReasoning-filtered",
                    help="Dataset to use for evaluation")
parser.add_argument("--dataset_split", type=str, default="train",
                    help="Dataset split")
parser.add_argument("--max_tokens", type=int, default=2048,
                    help="Max tokens per response")
parser.add_argument("--temperature", type=float, default=0.8,
                    help="Sampling temperature")
parser.add_argument("--limit", type=int, default=100,
                    help="Number of questions to test")

# Evaluation configuration
parser.add_argument("--judge_model", type=str, default="gpt-5-nano",
                    help="Model to use for grading")

# Output configuration
parser.add_argument("--output_dir", type=str, default="results/sae_vector_sweep",
                    help="Directory to save results")
parser.add_argument("--load_in_8bit", action="store_true",
                    help="Load model in 8-bit")
parser.add_argument("--skip_training", action="store_true",
                    help="Skip vector training and use existing vector")
parser.add_argument("--skip_generation", action="store_true",
                    help="Skip generation if files exist, only grade and plot")
parser.add_argument("--skip_grading", action="store_true",
                    help="Skip grading if graded files exist, only plot")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")

args = parser.parse_args()


def get_label_positions(annotated_thinking, response_text, tokenizer, context_sentences=0):
    """Parse SAE annotations and find token positions for each label"""
    import re
    label_positions = {}
    
    ANNOTATION_PATTERN = re.compile(r'\["([\d.]+):(\S+?)"\](.*?)\["end-section"\]', re.DOTALL)
    matches = list(ANNOTATION_PATTERN.finditer(annotated_thinking))
    
    char_to_token = utils.get_char_to_token_map(response_text, tokenizer)
    sentences = utils.split_into_sentences(response_text, min_words=0)
    
    for match in matches[:-1]:
        activation_str = match.group(1).strip()
        label = match.group(2).strip()
        text = match.group(3).strip()
        
        try:
            activation = float(activation_str)
        except ValueError:
            continue
            
        if not text:
            continue
            
        pattern = r'(?:[.?!;\n]|\n\n)\s*(' + re.escape(text) + ')'
        text_match = re.search(pattern, response_text)
        text_pos = text_match.start(1) if text_match else -1
        
        if text_pos >= 0:
            token_start = char_to_token.get(text_pos, None)
            token_end = char_to_token.get(text_pos + len(text) - 1, None)
            
            if token_end is not None:
                token_end += 1

            if token_start is None or token_end is None or token_start >= token_end:
                continue
            
            target_sentence_idx = -1
            for i, sentence in enumerate(sentences):
                if text in sentence:
                    target_sentence_idx = i
                    break
            
            if target_sentence_idx == -1:
                continue
                
            additional_context = ""
            if context_sentences > 0 and target_sentence_idx < len(sentences) - 1:
                end_idx = min(target_sentence_idx + context_sentences + 1, len(sentences))
                additional_sentences = sentences[target_sentence_idx + 1:end_idx]
                
                if additional_sentences:
                    text_end_pos = text_pos + len(text)
                    next_sentence_start = response_text.find(additional_sentences[0], text_end_pos)
                    if next_sentence_start > text_end_pos:
                        original_whitespace = response_text[text_end_pos:next_sentence_start]
                        additional_context = original_whitespace + original_whitespace.join(additional_sentences)
                    else:
                        additional_context = " " + " ".join(additional_sentences)
                
                if additional_context:
                    context_end_pos = text_pos + len(text) + len(additional_context)
                    context_token_end = char_to_token.get(context_end_pos - 1, None)
                    if context_token_end is not None:
                        token_end = context_token_end + 1
            
            if label not in label_positions:
                label_positions[label] = []
            label_positions[label].append((token_start, token_end, text + additional_context, activation, text_pos))
    
    return label_positions


def extract_examples_for_category(responses_data, category_name, tokenizer, n_training_examples, model_name_short):
    """Extract training examples for a specific category from annotated responses"""
    examples_for_category = []
    
    for resp in tqdm(responses_data, desc=f"Extracting examples for {category_name}"):
        if not resp.get('annotated_thinking'):
            continue

        thinking_process = extract_thinking_process(resp["full_response"])
        full_text = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{resp['original_message']['content']}\n\nStep by step answer:\n{thinking_process}"
        
        if category_name not in resp['annotated_thinking']:
            continue
            
        label_positions = get_label_positions(resp['annotated_thinking'], full_text, tokenizer, args.context_sentences)
        
        if category_name in label_positions:
            for start, end, text, activation, text_pos in label_positions[category_name]:
                context = full_text[:text_pos]

                if context[-1] not in ['.', '?', '!', ';', '\n'] and context[-2] not in ['.', '?', '!', ';', '\n'] and context.strip()[-1] not in ['.', '?', '!', ';', '\n']:
                    continue
                
                word_count = len(text.strip().split())
                if word_count < 7:
                    continue
                
                examples_for_category.append({
                    'prompt': context,
                    'target_completion': text,
                    'original_question': resp['original_message']['content'],
                    'full_thinking': extract_thinking_process(resp["full_response"]),
                    'activation': activation
                })
    
    if not examples_for_category:
        return []

    # Random sampling
    pool = examples_for_category.copy()
    random.shuffle(pool)
    training_examples = pool[:min(n_training_examples, len(pool))]
    
    print(f"Found {len(examples_for_category)} examples, selected {len(training_examples)} for training")
    return training_examples


def train_steering_vector(model, tokenizer, category_name, training_examples, model_name_short):
    """Train a steering vector for the given category"""
    print(f"\n{'='*80}")
    print(f"Training steering vector for category: {category_name}")
    print(f"{'='*80}\n")
    
    # Parse learning rates
    learning_rates = [float(lr.strip()) for lr in args.lr.split(',')]
    
    # Extract prompts and target completions
    train_prompts = [ex['prompt'] for ex in training_examples]
    train_target_completions = [ex['target_completion'] for ex in training_examples]
    
    all_results = {}
    
    for lr in tqdm(learning_rates, desc="Training with learning rates"):
        print(f"\nOptimizing with learning rate: {lr}")
        
        try:
            params, loss_info = steering_opt.optimize_vector_simple(
                model,
                tokenizer,
                train_prompts,
                train_target_completions,
                args.layer,
                lr=lr,
                max_iters=args.max_iters,
                optim_minibatch_size=args.batch_size,
                base_gen_minibatch_size=args.batch_size,
                warmup_steps=0,
                min_lr=0,
                starting_norm=1,
                max_norm=None,
                grad_clip=None,
                early_stopping_patience=10,
                early_stopping_min_delta=0.001,
                return_info=True,
                return_loss_history=True,
                steering_token_window=None,
                include_base_objective=False,
                wandb_run=None,
                static_vectors=None,
                steering_type=args.steering_type,
                rank=1,
                adaptive_hidden=128,
                eval_prompts=None,
                eval_target_completions=None
            )

            if args.steering_type == "linear":
                vect_to_store = params.detach().cpu()
            elif args.steering_type == "adaptive_linear":
                vect_to_store = {
                    'vector': params['vector'].detach().cpu(),
                    'W1': params['W1'].detach().cpu(),
                    'b1': params['b1'].detach().cpu(),
                    'W2': params['W2'].detach().cpu(),
                    'b2': params['b2'].detach().cpu(),
                    'hidden': 128,
                }
            else:  # resid_lora
                vect_to_store = {
                    'A': params['A'].detach().cpu(),
                    'B': params['B'].detach().cpu(),
                    'alpha': params['alpha'].detach().cpu(),
                    'rank': 1,
                }
            
            final_loss_val = loss_info['final_loss']
            
            if lr not in all_results:
                all_results[lr] = []
            
            all_results[lr].append({
                'vector': vect_to_store,
                'loss_info': loss_info,
                'final_loss': final_loss_val
            })
            
        except Exception as e:
            print(f"Error optimizing vector with lr {lr}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        raise RuntimeError("No successful optimization runs. Exiting.")
    
    # Find best learning rate based on final loss
    best_lr = None
    best_result = None
    best_loss = float('inf')
    
    for lr, results in all_results.items():
        for result in results:
            final_loss = result['final_loss']
            if final_loss < best_loss:
                best_loss = final_loss
                best_lr = lr
                best_result = result
    
    print(f"\nBest learning rate: {best_lr} (final loss: {best_result['final_loss']:.4f})")
    
    return best_result['vector'], best_lr, best_loss


def generate_responses_with_steering(model, tokenizer, dataset_data, steering_vector, coefficient, output_file):
    """Generate responses with uniform token-level steering using nnsight"""
    print(f"\n{'='*80}")
    print(f"Generating responses with coefficient={coefficient}")
    print(f"{'='*80}\n")
    
    if os.path.exists(output_file) and args.skip_generation:
        print(f"Skipping generation: {output_file} already exists")
        return output_file
    
    results = []
    device = model.device
    
    # Move steering vector to model device and dtype
    if args.steering_type == "linear":
        steer_vec = steering_vector.to(device=device, dtype=model.dtype)
    elif args.steering_type == "adaptive_linear":
        steer_vec = {
            'vector': steering_vector['vector'].to(device=device, dtype=model.dtype),
            'W1': steering_vector['W1'].to(device=device, dtype=model.dtype),
            'b1': steering_vector['b1'].to(device=device, dtype=model.dtype),
            'W2': steering_vector['W2'].to(device=device, dtype=model.dtype),
            'b2': steering_vector['b2'].to(device=device, dtype=model.dtype),
        }
    else:  # resid_lora
        steer_vec = {
            'A': steering_vector['A'].to(device=device, dtype=model.dtype),
            'B': steering_vector['B'].to(device=device, dtype=model.dtype),
            'alpha': steering_vector['alpha'].to(device=device, dtype=model.dtype),
        }
    
    for i, item in enumerate(tqdm(dataset_data, desc="Generating")):
        if i >= args.limit:
            break
            
        question = item['question']
        answer = item['answer']
        
        # Format prompt
        prompt = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{question}\n\nStep by step answer:\n"
        
        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Generate with intervention
        if coefficient == 0.0:
            # Baseline: no steering
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    do_sample=(args.temperature > 0),
                    pad_token_id=tokenizer.eos_token_id
                )
        else:
            # With steering - use hooks for intervention
            from functools import partial
            
            def steering_hook(module, input, output, coef, vec, steering_type):
                """Hook to apply steering to layer output"""
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Apply steering uniformly to all tokens
                if steering_type == "linear":
                    # Simple additive steering
                    steered_hidden = hidden_states + coef * vec
                elif steering_type == "adaptive_linear":
                    # Adaptive steering with MLP gate
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    gate_input = hidden_states @ vec['W1'].T + vec['b1'].unsqueeze(0).unsqueeze(0)
                    gate_output = torch.relu(gate_input)
                    gate_scalar = gate_output @ vec['W2'].T + vec['b2'].unsqueeze(0).unsqueeze(0)
                    gate_scalar = torch.sigmoid(gate_scalar)
                    steered_hidden = hidden_states + coef * gate_scalar * vec['vector'].unsqueeze(0).unsqueeze(0)
                else:  # resid_lora
                    # LoRA steering: h' = h + alpha * B @ A @ h
                    # hidden_states: [batch, seq, hidden_dim]
                    # A: [rank, hidden_dim]
                    # B: [hidden_dim, rank]
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    hidden_reshaped = hidden_states.reshape(-1, hidden_dim).T  # [hidden_dim, batch*seq]
                    lora_intermediate = vec['A'] @ hidden_reshaped  # [rank, batch*seq]
                    lora_output = vec['B'] @ lora_intermediate  # [hidden_dim, batch*seq]
                    lora_delta = vec['alpha'] * lora_output.T.reshape(batch_size, seq_len, hidden_dim)
                    steered_hidden = hidden_states + coef * lora_delta
                
                if isinstance(output, tuple):
                    return (steered_hidden,) + output[1:]
                else:
                    return steered_hidden
            
            # Register hook
            hook_fn = partial(steering_hook, coef=coefficient, vec=steer_vec, steering_type=args.steering_type)
            hook_handle = model.model.layers[args.layer].register_forward_hook(hook_fn)
            
            try:
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        do_sample=(args.temperature > 0),
                        pad_token_id=tokenizer.eos_token_id
                    )
            finally:
                # Remove hook
                hook_handle.remove()
        
        # Decode response
        full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_part = full_response[len(prompt):]
        
        results.append({
            'question_id': i,
            'dataset_name': args.dataset,
            'question': question,
            'answer': answer,
            'full_response': full_response,
            'generated_response': generated_part,
            'prompt': prompt,
            'coefficient': coefficient,
        })
        
        # Clean up
        del input_ids, output_ids
        torch.cuda.empty_cache()
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved responses to {output_file}")
    return output_file


def grade_responses(responses_file):
    """Grade responses using the judge model"""
    graded_file = responses_file.replace('.json', '.graded.json')
    
    if os.path.exists(graded_file) and args.skip_grading:
        print(f"Skipping grading: {graded_file} already exists")
        return graded_file
    
    print(f"\n{'='*80}")
    print(f"Grading responses: {os.path.basename(responses_file)}")
    print(f"{'='*80}\n")
    
    # Load responses
    with open(responses_file, 'r') as f:
        responses = json.load(f)
    
    # Import grading utility
    from utils.utils import chat_batch
    import asyncio
    
    # Prepare grading prompts
    prompts = []
    for resp in responses:
        prompt = f"""Please evaluate whether the following answer to a question is correct.

Question: {resp['question']}

Correct answer: {resp['answer']}

Model's answer: {resp['generated_response']}

First, extract the final answer from both. Then determine if the model's final answer is equivalent to the correct answer.
Just answer YES if the model's answer is correct, or NO if it's incorrect. Nothing else."""
        prompts.append(prompt)
    
    # Grade in batch
    print(f"Grading {len(prompts)} responses...")
    try:
        import concurrent.futures
        def run_async():
            return asyncio.run(chat_batch(prompts, model=args.judge_model, max_tokens=100))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async)
            judge_responses = future.result()
    except RuntimeError:
        judge_responses = asyncio.run(chat_batch(prompts, model=args.judge_model, max_tokens=100))
    
    # Process results
    for resp, judge_resp in zip(responses, judge_responses):
        is_correct = 'yes' in judge_resp.lower()
        resp['is_correct'] = is_correct
        resp['judge_response'] = judge_resp
    
    # Save graded results
    with open(graded_file, 'w') as f:
        json.dump(responses, f, indent=2)
    
    print(f"Saved graded responses to {graded_file}")
    return graded_file


def extract_accuracy(graded_file):
    """Extract accuracy from graded responses"""
    if not os.path.exists(graded_file):
        print(f"Warning: Graded file not found: {graded_file}")
        return None
    
    try:
        with open(graded_file, 'r') as f:
            data = json.load(f)
        
        total = len(data)
        correct = sum(1 for item in data if item.get('is_correct', False))
        accuracy = correct / total if total > 0 else 0.0
        
        print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
        return accuracy, correct, total
    except Exception as e:
        print(f"Error extracting accuracy from {graded_file}: {e}")
        return None


def plot_results(results_df, output_dir, model_id, layer, feature_idx, category_name):
    """Create visualization of accuracy vs coefficient"""
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(results_df['coefficient'], results_df['accuracy'], 
            marker='o', linewidth=2, markersize=8, label='Accuracy')
    
    if 0.0 in results_df['coefficient'].values:
        baseline_acc = results_df[results_df['coefficient'] == 0.0]['accuracy'].values[0]
        ax.axhline(y=baseline_acc, color='red', linestyle='--', 
                   label=f'Baseline (no steering): {baseline_acc:.1%}', alpha=0.7)
    
    ax.set_xlabel('Steering Coefficient', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'Accuracy vs Steering Coefficient\n'
                 f'Model: {model_id}, Layer: {layer}, Feature: {category_name} (idx{feature_idx})',
                 fontsize=16, fontweight='bold')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3)
    
    for idx, row in results_df.iterrows():
        ax.annotate(f"{row['accuracy']:.1%}\n({row['correct']}/{row['total']})",
                   (row['coefficient'], row['accuracy']),
                   textcoords="offset points",
                   xytext=(0, 10),
                   ha='center',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax.legend(fontsize=12)
    
    plot_file = os.path.join(output_dir, 
                             f"accuracy_vs_coefficient_{model_id}_layer{layer}_idx{feature_idx}.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    pdf_file = plot_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"PDF saved to: {pdf_file}")
    
    plt.close()
    
    return plot_file


def main():
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*80)
    print("SAE VECTOR TRAINING AND COEFFICIENT SWEEP")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"SAE n_clusters: {args.n_clusters}")
    print(f"Feature index: {args.feature_idx}")
    print(f"Coefficients: {args.coefficients}")
    print(f"Questions: {args.limit}")
    print()
    
    # Model names
    model_id = args.model.split('/')[-1].lower()
    thinking_model = args.thinking_model if args.thinking_model else args.model
    thinking_model_id = thinking_model.split('/')[-1].lower()
    sae_layer = args.sae_layer if args.sae_layer is not None else args.layer
    
    # Load SAE taxonomy to get category name
    print(f"Loading SAE taxonomy from {thinking_model_id} layer {sae_layer}...")
    latent_descriptions = get_latent_descriptions(
        thinking_model_id, 
        sae_layer, 
        args.n_clusters, 
        clustering_method=args.clustering_method
    )
    
    if args.feature_idx not in latent_descriptions:
        raise ValueError(f"Feature index {args.feature_idx} not found in taxonomy. Available: {list(latent_descriptions.keys())}")
    
    category_info = latent_descriptions[args.feature_idx]
    category_name = category_info['key']  # e.g., "idx7"
    category_title = category_info['title']
    
    print(f"Selected category: {category_name}")
    print(f"Title: {category_title}")
    print(f"Description: {category_info.get('description', 'N/A')}")
    print()
    
    # Vector file path
    vector_file = os.path.join(args.output_dir, "vectors", 
                               f"{model_id}_layer{args.layer}_{category_name}_{args.steering_type}.pt")
    
    # Train or load steering vector
    if args.skip_training and os.path.exists(vector_file):
        print(f"Loading existing vector from {vector_file}...")
        steering_vector = torch.load(vector_file, map_location='cpu')
    else:
        # Load model for training
        print(f"Loading model {thinking_model} for vector training...")
        tokenizer = AutoTokenizer.from_pretrained(thinking_model)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        model = AutoModelForCausalLM.from_pretrained(
            thinking_model,
            device_map="auto",
            load_in_8bit=args.load_in_8bit,
            torch_dtype=torch.bfloat16
        )
        
        for param in model.parameters():
            param.requires_grad = False
        
        # Load annotated responses
        responses_json_path = f"results/vars/responses_{thinking_model_id}.json"
        annotated_responses_json_path = f"results/vars/annotated_responses_{thinking_model_id}.json"
        
        if not os.path.exists(responses_json_path) or not os.path.exists(annotated_responses_json_path):
            raise FileNotFoundError(f"Annotated responses not found. Please run annotation first.")
        
        with open(responses_json_path, 'r') as f:
            responses_data = json.load(f)
        with open(annotated_responses_json_path, 'r') as f:
            annotated_responses_data = json.load(f)
        
        # Merge annotations
        valid_responses = []
        for i, resp in enumerate(responses_data):
            if i < len(annotated_responses_data):
                annotated_resp = annotated_responses_data[i]
                if (resp['question_id'] == annotated_resp['question_id'] and 
                    resp['dataset_name'] == annotated_resp['dataset_name'] and
                    annotated_resp.get('annotated_thinking')):
                    merged_resp = resp.copy()
                    merged_resp['annotated_thinking'] = annotated_resp['annotated_thinking']
                    valid_responses.append(merged_resp)
        
        print(f"Found {len(valid_responses)} annotated responses")
        
        # Extract training examples
        training_examples = extract_examples_for_category(
            valid_responses, 
            category_name, 
            tokenizer,
            args.n_training_examples,
            model_id
        )
        
        if not training_examples:
            raise RuntimeError(f"No training examples found for category {category_name}")
        
        # Train vector
        steering_vector, best_lr, best_loss = train_steering_vector(
            model, tokenizer, category_name, training_examples, model_id
        )
        
        # Save vector
        os.makedirs(os.path.dirname(vector_file), exist_ok=True)
        torch.save(steering_vector, vector_file)
        print(f"\nSaved steering vector to {vector_file}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Load model for generation
    print(f"\nLoading model {args.model} for generation...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
        torch_dtype=torch.bfloat16
    )
    
    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    
    # Run coefficient sweep
    results = []
    
    for coef in tqdm(args.coefficients, desc="Processing coefficients"):
        print(f"\n{'='*80}")
        print(f"COEFFICIENT: {coef}")
        print(f"{'='*80}")
        
        # Generate responses
        if coef == 0.0:
            output_file = f"{args.output_dir}/responses/responses_{model_id}_baseline.json"
        else:
            output_file = f"{args.output_dir}/responses/responses_{model_id}_layer{args.layer}_idx{args.feature_idx}_coef{coef}.json"
        
        if not args.skip_generation:
            generate_responses_with_steering(
                model, tokenizer, dataset, steering_vector, coef, output_file
            )
        elif not os.path.exists(output_file):
            print(f"Warning: {output_file} not found and --skip_generation is set")
            continue
        
        # Grade responses
        graded_file = grade_responses(output_file)
        
        # Extract accuracy
        result = extract_accuracy(graded_file)
        if result is not None:
            accuracy, correct, total = result
            results.append({
                'coefficient': coef,
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'responses_file': output_file,
                'graded_file': graded_file,
            })
    
    if not results:
        print("\nError: No results were generated. Exiting.")
        return
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('coefficient')
    
    # Save results to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_file = os.path.join(args.output_dir, 
                            f"results_{model_id}_layer{args.layer}_idx{args.feature_idx}.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"\nResults saved to: {csv_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Coefficient':<15} {'Accuracy':<15} {'Correct/Total':<20}")
    print("-"*50)
    for _, row in results_df.iterrows():
        print(f"{row['coefficient']:<15.2f} {row['accuracy']:<15.1%} {row['correct']}/{row['total']}")
    
    # Find best coefficient
    best_row = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"\nBest coefficient: {best_row['coefficient']:.2f} (accuracy: {best_row['accuracy']:.1%})")
    
    # Compare to baseline
    if 0.0 in results_df['coefficient'].values:
        baseline_row = results_df[results_df['coefficient'] == 0.0].iloc[0]
        print(f"Baseline (coef=0): {baseline_row['accuracy']:.1%}")
        delta = best_row['accuracy'] - baseline_row['accuracy']
        print(f"Best improvement: {delta:+.1%}")
    
    # Plot results
    plot_file = plot_results(results_df, args.output_dir, model_id, args.layer, 
                             args.feature_idx, category_title)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results CSV: {csv_file}")
    print(f"Plot: {plot_file}")
    print(f"Vector: {vector_file}")


if __name__ == "__main__":
    main()

