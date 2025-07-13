# %%
import sys
import dotenv
import torch
import json
from utils.sae import load_sae
from utils.utils import load_model
from utils.clustering import get_latent_descriptions
from utils.utils import chat
import os
import gc
import colorsys
import math
import matplotlib.pyplot as plt
import re
import argparse
from collections import Counter
# Add parent directory to path for imports
sys.path.append('..')
dotenv.load_dotenv("../.env")

from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate hybrid model on datasets')
    parser.add_argument('--dataset', type=str, choices=['gsm8k', 'math500', "aime"], default='gsm8k',
                      help='Dataset to evaluate on (gsm8k or math500)')
    parser.add_argument('--thinking_model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
                      help='Model for thinking/perplexity')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.1-8B',
                      help='Model for base generation')
    parser.add_argument('--thinking_layer', type=int, default=6,
                      help='Layer to extract from thinking model')
    parser.add_argument('--n_clusters', type=int, default=19,
                      help='Number of clusters for SAE')
    parser.add_argument('--lookahead', action='store_true', default=False,
                      help='Enable lookahead functionality in hybrid generation')
    parser.add_argument('--use_perplexity_selection', action='store_true', default=False,
                      help='Use perplexity-based selection between steered and unsteered generation')
    parser.add_argument('--n_tasks', type=int, default=500,
                      help='Number of tasks to evaluate')
    parser.add_argument('--max_new_tokens', type=int, default=1500,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--eval_start_idx', type=int, default=0,
                      help='Starting index in the dataset')
    parser.add_argument('--cold_start_tokens', type=int, default=1,
                      help='Number of initial tokens to use from thinking model')
    parser.add_argument('--temperature', type=float, default=0.3,
                      help='Temperature for sampling')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                      help='Repetition penalty (1.0 means no penalty, >1.0 discourages repetition)')
    parser.add_argument('--repetition_window', type=int, default=0,
                      help='Window size for repetition detection')
    parser.add_argument('--coefficient', type=float, default=1,
                      help='Steering coefficient')
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--example_idx', type=int, default=13,
                      help='Index of example to run')
    return parser.parse_known_args()[0]

def get_next_token(logits, temperature, model, input_ids=None, repetition_penalty=1.0, repetition_window=128):
    """Get next token from logits using temperature sampling or greedy decoding"""
    # Apply repetition penalty if enabled and input_ids are provided
    if repetition_penalty > 1.0 and input_ids is not None:
        # Only consider tokens in the recent window
        window_start = max(0, input_ids.shape[1] - repetition_window)
        recent_tokens = input_ids[0, window_start:].tolist()
        
        # Get unique tokens in the recent window
        unique_tokens = set(recent_tokens)
        
        # Apply penalty to those tokens
        for token in unique_tokens:
            if token < logits.shape[-1]:  # Ensure token is within vocabulary
                logits[token] /= repetition_penalty
    
    if temperature > 0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
    else:
        return torch.argmax(logits).item()

def get_token_and_string(logits, temperature, tokenizer, input_ids=None, repetition_penalty=1.0, repetition_window=128):
    """Get token ID and string from logits"""
    token = get_next_token(logits, temperature, tokenizer, input_ids, repetition_penalty, repetition_window)
    token_string = tokenizer.decode(token)
    return token, token_string

def get_perplexity(token_string, logits, model):
    """Calculate perplexity of a token string under the given logits"""
    token_id = model.tokenizer.encode(token_string, return_tensors="pt", add_special_tokens=False).to(model.device).to(torch.long)
    if token_id.shape[1] > 0 and token_id[0, 0].item() < logits.shape[-1]:
        log_prob = torch.log_softmax(logits[0, -1], dim=-1)[token_id[0, 0]].item()
        return math.exp(-log_prob)
    else:
        return float('inf')

def hybrid_generate(thinking_model, base_model, base_tokenizer, thinking_input_ids, base_input_ids, 
                   max_new_tokens, thinking_layer, sae, steering_vectors, latent_descriptions,
                   coefficient=1, temperature=1.0, repetition_penalty=1.0, repetition_window=128,
                   verbose=False, lookahead=True, use_perplexity_selection=True):
    base_output_ids = base_input_ids.clone()
    thinking_output_ids = thinking_input_ids.clone()
    token_latent_info = []
    thinking_log_probs = []
    per_token_perplexity = []
    token_position = []
    steering_selection = []
    
    # Detect and break repetition loops
    repetition_detection_counter = 0
    last_sequence = ""
    
    for step in range(max_new_tokens):
        with torch.no_grad():
            if lookahead:
                with thinking_model.trace(thinking_output_ids) as tracer:
                    current_thinking_logits = thinking_model.lm_head.output.save()
                
                thinking_token = get_next_token(current_thinking_logits[0, -1], temperature, thinking_model, 
                                               thinking_output_ids, repetition_penalty, repetition_window)
                thinking_token_string = thinking_model.tokenizer.decode(thinking_token)
                
                temp_thinking_token_id = thinking_model.tokenizer.encode(
                    thinking_token_string, return_tensors="pt", add_special_tokens=False
                ).to(thinking_model.device).to(torch.long)
                temp_thinking_output_ids = torch.cat([thinking_output_ids, temp_thinking_token_id], dim=1)
                
                with thinking_model.trace(temp_thinking_output_ids) as tracer:
                    thinking_model.model.layers.all()
                    future_activations = thinking_model.model.layers[thinking_layer].output[0][0, -1, :].save()
                
                future_latent_activations = sae.encoder(future_activations.to(torch.float32) - sae.b_dec)
                future_latent_id = torch.argmax(future_latent_activations).item()
                future_latent_title = latent_descriptions[future_latent_id]["title"]
                future_activation_value = future_latent_activations[future_latent_id].item()
                
                steering_vector = steering_vectors[future_latent_title.lower().replace(" ", "-")].to(base_model.device)
                
                if verbose:
                    print(f"Look-ahead: '{thinking_token_string}', Latent: {future_latent_title}")
            else:
                with thinking_model.trace(thinking_output_ids) as tracer:
                    thinking_model.model.layers.all()
                    current_activations = thinking_model.model.layers[thinking_layer].output[0][0, -1, :].save()
                    current_thinking_logits = thinking_model.lm_head.output.save()
                
                current_latent_activations = sae.encoder(current_activations.to(torch.float32) - sae.b_dec)
                current_latent_id = torch.argmax(current_latent_activations).item()
                current_latent_title = latent_descriptions[current_latent_id]["title"]
                current_activation_value = current_latent_activations[current_latent_id].item()
                
                steering_vector = steering_vectors[current_latent_title.lower().replace(" ", "-")].to(base_model.device)
                
                if verbose:
                    print(f"Current position latent: {current_latent_title}")
            
            with base_model.trace(base_output_ids) as tracer:
                base_model.model.layers[thinking_layer].input[:, max(1, base_output_ids.shape[1] - 30):, :] += coefficient * steering_vector
                logits_steered = base_model.lm_head.output.save()
            steered_token, steered_string = get_token_and_string(logits_steered[0, -1], temperature, base_tokenizer, 
                                                               base_output_ids, repetition_penalty, repetition_window)
            
            if use_perplexity_selection:
                with base_model.trace(base_output_ids) as tracer:
                    logits_unsteered = base_model.lm_head.output.save()
                unsteered_token, unsteered_string = get_token_and_string(logits_unsteered[0, -1], temperature, base_tokenizer, 
                                                                     base_output_ids, repetition_penalty, repetition_window)
                
                with thinking_model.trace(thinking_output_ids) as tracer:
                    current_thinking_logits = thinking_model.lm_head.output.save()
                
                steered_perplexity = get_perplexity(steered_string, current_thinking_logits, thinking_model)
                unsteered_perplexity = get_perplexity(unsteered_string, current_thinking_logits, thinking_model)
                    
                if steered_perplexity <= unsteered_perplexity:
                    next_token = steered_token
                    next_token_string = steered_string
                    token_perplexity = steered_perplexity
                    log_prob = -math.log(steered_perplexity)
                    selected_option = "steered"
                    if verbose:
                        print(f"Chose steered: '{next_token_string}' ({steered_perplexity:.2f} vs {unsteered_perplexity:.2f})")
                else:
                    next_token = unsteered_token
                    next_token_string = unsteered_string
                    token_perplexity = unsteered_perplexity
                    log_prob = -math.log(unsteered_perplexity)
                    selected_option = "unsteered"
                    if verbose:
                        print(f"Chose unsteered: '{next_token_string}' ({unsteered_perplexity:.2f} vs {steered_perplexity:.2f})")
            else:
                next_token = steered_token
                next_token_string = steered_string
                token_perplexity = get_perplexity(steered_string, current_thinking_logits, thinking_model)
                log_prob = -math.log(token_perplexity)
                selected_option = "steered"
                if verbose:
                    print(f"Using steered: '{next_token_string}'")
            
            # Check for repetition loops
            current_output = base_tokenizer.decode(base_output_ids[0][-64:])
            potential_repetition = current_output + next_token_string
            
            # Check for repeated patterns of at least 8 characters
            for pattern_length in range(8, min(40, len(potential_repetition) // 2)):
                pattern = potential_repetition[-pattern_length:]
                prev_chunk = potential_repetition[-2*pattern_length:-pattern_length]
                
                if pattern == prev_chunk:
                    repetition_detection_counter += 1
                    if repetition_detection_counter >= 3:
                        # Detected repetition, increase temperature temporarily to break out
                        if verbose:
                            print(f"Repetition detected! Increasing temperature temporarily.")
                        
                        # Try again with higher temperature
                        with base_model.trace(base_output_ids) as tracer:
                            logits_break_rep = base_model.lm_head.output.save()
                        
                        # Use higher temperature and stronger repetition penalty to break the loop
                        next_token, next_token_string = get_token_and_string(
                            logits_break_rep[0, -1], temperature * 1.5, base_tokenizer, 
                            base_output_ids, repetition_penalty * 1.5, repetition_window
                        )
                        
                        # Reset counter
                        repetition_detection_counter = 0
                        break
                else:
                    repetition_detection_counter = 0
        
            base_next_token_id = base_tokenizer.encode(next_token_string, return_tensors="pt", add_special_tokens=False).to(base_model.device).to(torch.long)
            thinking_next_token_id = thinking_model.tokenizer.encode(next_token_string, return_tensors="pt", add_special_tokens=False).to(thinking_model.device).to(torch.long)
            base_output_ids = torch.cat([base_output_ids, base_next_token_id], dim=1)
            thinking_output_ids = torch.cat([thinking_output_ids, thinking_next_token_id], dim=1)
            
            steering_selection.append(selected_option)
            thinking_log_probs.append(log_prob)
            per_token_perplexity.append(token_perplexity)
            token_position.append(len(token_latent_info))
            token_latent_info.append({
                "token": next_token_string,
                "latent_id": (future_latent_id if lookahead else current_latent_id) if selected_option == "steered" else None,
                "latent_title": (future_latent_title if lookahead else current_latent_title) if selected_option == "steered" else "No Steering",
                "activation_value": (future_activation_value if lookahead else current_activation_value) if selected_option == "steered" else 0.0,
                "perplexity": token_perplexity,
                "future_token": thinking_token_string if lookahead else None
            })
                        
            if next_token == thinking_model.tokenizer.eos_token_id or next_token == base_tokenizer.eos_token_id:
                if verbose:
                    print("End of sequence reached")
                break

        torch.cuda.empty_cache()

    gc.collect()
    return base_output_ids, token_latent_info, per_token_perplexity, token_position, steering_selection

def load_steering_vectors(model_id):
    """Load steering vectors from train_vectors output"""
    vectors_path = f"../train-vectors/results/vars/optimized_vectors_{model_id}.pt"
    if not os.path.exists(vectors_path):
        return {}
    return torch.load(vectors_path)

def generate_latent_colors(latent_descriptions):
    colors = {}
    
    unique_latents = set([desc["title"] for desc in latent_descriptions.values()])
    num_colors = len(unique_latents)
    
    for i, latent_title in enumerate(unique_latents):
        hue = i / num_colors
        saturation = 0.7
        value = 0.9
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
        colors[latent_title] = hex_color
    
    colors["Cold Start"] = "#808080"
    colors["Fallback"] = "#CCCCCC"
    colors["No Steering"] = "#111111"
    
    return colors

def visualize_generation_results(token_latent_info, steering_selection, per_token_perplexity, token_position, latent_colors):
    tokens = [info["token"] for info in token_latent_info]
    
    fig = plt.figure(figsize=(14, 10))
    
    plt.subplot(4, 1, 1)
    plt.axis('off')
    
    for i, info in enumerate(token_latent_info):
        token = info["token"].replace('$', '\\$')
        latent_title = info["latent_title"]
        color = latent_colors[latent_title]
        plt.text(i, 0, token, color=color, fontsize=10, ha='center')
        
    plt.xlim(-1, len(tokens))
    plt.title("Generated Text (Colored by Latent)")
    
    plt.subplot(4, 1, 2)
    plt.plot(token_position, per_token_perplexity, marker='o', linestyle='-', color='blue', alpha=0.7)
    plt.yscale('log')
    plt.title("Perplexity by Token Position")
    
    plt.subplot(4, 1, 3)
    choices = []
    for choice in steering_selection:
        if choice == "steered":
            choices.append(1)
        elif choice == "unsteered":
            choices.append(0)
        else:
            choices.append(-1)
    
    plt.imshow([choices], cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    escaped_tokens = [t.replace('$', '\\$') for t in tokens]
    plt.xticks(range(len(tokens)), escaped_tokens, rotation=90, fontsize=8)
    plt.yticks([])
    cbar = plt.colorbar(orientation="horizontal", pad=0.1, ticks=[-1, 0, 1])
    cbar.set_ticklabels(["None", "No Steering", "Steering Used"])
    plt.title("Steering Selection")
    
    plt.subplot(4, 1, 4)
    latent_titles = [info.get("latent_title", "None") for info in token_latent_info]
    unique_latents = sorted(set(latent_titles))
    latent_map = {latent: i for i, latent in enumerate(unique_latents)}
    latent_values = [latent_map[latent] for latent in latent_titles]
    
    plt.imshow([latent_values], cmap="tab20", aspect="auto")
    plt.xticks(range(len(tokens)), escaped_tokens, rotation=90, fontsize=8)
    plt.yticks([])
    
    handles = []
    for latent in unique_latents:
        color = latent_colors.get(latent, "#000000")
        patch = plt.Rectangle((0, 0), 1, 1, fc=color)
        handles.append(patch)
    
    plt.legend(handles, unique_latents, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=min(5, len(unique_latents)), frameon=False)
    
    plt.title("Latent Features")
    plt.tight_layout()
    plt.show()

def load_models_and_sae(args):
    thinking_model_id = args.thinking_model.split('/')[-1].lower()
    base_model_id = args.base_model.split('/')[-1].lower()
    
    print(f"Loading models {args.thinking_model} and {args.base_model}...")
    thinking_model, thinking_tokenizer = load_model(model_name=args.thinking_model)
    thinking_model.tokenizer = thinking_tokenizer
    if args.temperature > 0:
        thinking_model.generation_config.do_sample = True

    base_model, base_tokenizer = load_model(model_name=args.base_model)
    if args.temperature > 0:
        base_model.generation_config.do_sample = True

    print(f"Loading SAE for model {thinking_model_id}, layer {args.thinking_layer}...")
    sae, _ = load_sae(thinking_model_id, args.thinking_layer, args.n_clusters)
    sae = sae.to(thinking_model.device)

    print(f"Loading steering vectors and layer effects...")
    steering_vectors = load_steering_vectors(base_model_id)
    descriptions = get_latent_descriptions(thinking_model_id, args.thinking_layer, args.n_clusters)

    return thinking_model, thinking_tokenizer, base_model, base_tokenizer, sae, steering_vectors, descriptions, thinking_model_id, base_model_id

def run_example(thinking_model, thinking_tokenizer, base_model, base_tokenizer, 
               sae, steering_vectors, descriptions, args, dataset):
    sample_idx = args.example_idx
    for i, item in enumerate(dataset):
        if i == sample_idx:
            if args.dataset == "gsm8k":
                example = {
                    "question": item["question"],
                    "answer": item["answer"]
                }
            elif args.dataset == "math500":
                example = {
                    "question": item["problem"],
                    "answer": item["answer"]
                }
            elif args.dataset == "aime":
                example = {
                    "question": item["problem"],
                    "answer": item["answer"]
                }
            
            break

    question = example["question"]
    answer = example["answer"]

    print("\n===== Example =====")
    print(f"Question: {question}")

    thinking_input_ids = thinking_tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(thinking_model.device).to(torch.long)

    base_prompt = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{question}\n\nStep by step answer:\n"
    base_input_ids = base_tokenizer.encode(
        base_prompt,
        return_tensors="pt"
    ).to(base_model.device).to(torch.long)
    
    # Generate with thinking model
    print("\n===== Generating with Thinking Model =====")
    with thinking_model.generate(thinking_input_ids, max_new_tokens=100, temperature=args.temperature, pad_token_id=thinking_tokenizer.eos_token_id) as gen:
        thinking_outputs = thinking_model.generator.output.save()
    thinking_response = thinking_tokenizer.decode(thinking_outputs[0][len(thinking_input_ids[0]):], skip_special_tokens=True)
    print(thinking_response)
    
    # Extract cold start tokens
    thinking_first_tokens = thinking_tokenizer.encode(thinking_response[:100], add_special_tokens=False)[:args.cold_start_tokens]
    cold_start_text = thinking_tokenizer.decode(thinking_first_tokens)
    print(f"\nUsing first {args.cold_start_tokens} tokens as cold start: '{cold_start_text}'")
    
    # Add cold start tokens to base input
    cold_start_base_tokens = base_tokenizer.encode(cold_start_text, add_special_tokens=False, return_tensors="pt").to(base_model.device).to(torch.long)
    base_input_with_cold_start = torch.cat([base_input_ids, cold_start_base_tokens], dim=1)
    
    # Add cold start tokens to thinking input for hybrid generation
    thinking_input_with_cold_start = torch.cat([
        thinking_input_ids, 
        thinking_tokenizer.encode(cold_start_text, add_special_tokens=False, return_tensors="pt").to(thinking_model.device).to(torch.long)
    ], dim=1)
    
    # Generate with base model
    print("\n===== Generating with Base Model =====")
    with base_model.generate(base_input_with_cold_start, max_new_tokens=100, temperature=args.temperature, pad_token_id=base_tokenizer.eos_token_id) as gen:
        base_outputs = base_model.generator.output.save()
    base_response = f"{cold_start_text}{base_tokenizer.decode(base_outputs[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
    print(base_response)
    
    # Generate with hybrid approach
    print("\n===== Generating with Hybrid Approach =====")
    hybrid_output_ids, token_latent_info, per_token_perplexity, token_position, steering_selection = hybrid_generate(
        thinking_model=thinking_model,
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        thinking_input_ids=thinking_input_with_cold_start,
        base_input_ids=base_input_with_cold_start,
        max_new_tokens=args.max_new_tokens,
        thinking_layer=args.thinking_layer,
        sae=sae,
        steering_vectors=steering_vectors,
        latent_descriptions=descriptions,
        temperature=args.temperature,
        coefficient=args.coefficient,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        lookahead=args.lookahead,
        use_perplexity_selection=args.use_perplexity_selection,
        verbose=False
    )
    hybrid_response = f"{cold_start_text}{base_tokenizer.decode(hybrid_output_ids[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
    print(hybrid_response)
    
    # Print correct answer for reference
    print("\n===== Correct Answer =====")
    print(answer)
    
    # Visualize results
    latent_colors = generate_latent_colors(descriptions)
    visualize_generation_results(token_latent_info, steering_selection, per_token_perplexity, token_position, latent_colors)
    
    return thinking_response, base_response, hybrid_response, token_latent_info, steering_selection

def clean_answer(text):
    return re.sub(r'\s+', ' ', text).strip()

def evaluate_answer(model_answer, correct_answer, question, model_name):
    prompt = f"""Please evaluate whether the following answer to a math problem is correct.

Question: {question}

Correct answer: {correct_answer}

Model's answer: {model_answer}

First, extract the final numerical answer from both the correct answer and model's answer. 
Then determine if the model's final numerical answer is equivalent to the correct final numerical answer.
Just answer YES if the model's answer is correct, or NO if it's incorrect. Nothing else.
"""
    
    response = chat(prompt, model="gpt-4.1", max_tokens=100)
    is_correct = "yes" in response.lower()
    print(f"{model_name} evaluated as: {response}")
    return is_correct

def analyze_hybrid_stats(token_latent_info, steering_selection):
    # Calculate no-steering fraction
    steered_count = steering_selection.count("steered")
    unsteered_count = steering_selection.count("unsteered")
    total = steered_count + unsteered_count
    no_steering_fraction = unsteered_count / total if total > 0 else 0
    
    # Analyze latent usage
    latent_counts = Counter()
    for info in token_latent_info:
        if info["latent_title"] != "No Steering":
            latent_counts[info["latent_title"]] += 1
    
    # Calculate percentages
    latent_percentages = {}
    if steered_count > 0:
        for latent, count in latent_counts.items():
            latent_percentages[latent] = (count / steered_count) * 100
    
    # Track steering decisions
    steering_stats = {
        "steered_count": steered_count,
        "unsteered_count": unsteered_count,
        "total_tokens": total,
        "steering_fraction": steered_count / total if total > 0 else 0,
        "no_steering_fraction": no_steering_fraction
    }
    
    return no_steering_fraction, latent_counts, latent_percentages, steering_stats

def save_detailed_results(results, args, thinking_model_id, base_model_id):
    # Create results directory if it doesn't exist
    os.makedirs(f"{args.results_dir}/detailed", exist_ok=True)
    
    # Format filename
    lookahead_str = "" if args.lookahead else "_no_lookahead"
    perplexity_str = "" if args.use_perplexity_selection else "_no_perplexity"
    filename = f"{args.results_dir}/detailed/hybrid_stats_{base_model_id}_{args.dataset}{lookahead_str}{perplexity_str}.json"
    
    # Calculate average steering statistics
    avg_steering_stats = {
        "steered_count": sum(stat["steered_count"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "unsteered_count": sum(stat["unsteered_count"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "total_tokens": sum(stat["total_tokens"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "steering_fraction": sum(stat["steering_fraction"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "no_steering_fraction": sum(stat["no_steering_fraction"] for stat in results["steering_stats"]) / len(results["steering_stats"])
    }
    
    # Prepare data for saving
    detailed_data = {
        "metadata": {
            "base_model": args.base_model,
            "thinking_model": args.thinking_model,
            "dataset": args.dataset,
            "lookahead": args.lookahead,
            "use_perplexity_selection": args.use_perplexity_selection,
            "temperature": args.temperature,
            "coefficient": args.coefficient,
            "n_tasks": len(results["questions"])
        },
        "answer_lengths": {
            "base_model": results["base_lengths"],
            "thinking_model": results["thinking_lengths"],
            "hybrid_model": results["hybrid_lengths"],
            "avg_base": sum(results["base_lengths"]) / len(results["base_lengths"]) if results["base_lengths"] else 0,
            "avg_thinking": sum(results["thinking_lengths"]) / len(results["thinking_lengths"]) if results["thinking_lengths"] else 0,
            "avg_hybrid": sum(results["hybrid_lengths"]) / len(results["hybrid_lengths"]) if results["hybrid_lengths"] else 0
        },
        "steering_stats": {
            "no_steering_fractions": results["no_steering_fractions"],
            "avg_no_steering": sum(results["no_steering_fractions"]) / len(results["no_steering_fractions"]) if results["no_steering_fractions"] else 0,
            "detailed_stats": results["steering_stats"],
            "average_stats": avg_steering_stats
        },
        "latent_usage": results["latent_usage"],
        "accuracies": {
            "base_model": results["base_correct"] / len(results["questions"]) * 100,
            "thinking_model": results["thinking_correct"] / len(results["questions"]) * 100,
            "hybrid_model": results["hybrid_correct"] / len(results["questions"]) * 100
        }
    }
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    
    print(f"Detailed results saved to {filename}")
    
    return detailed_data

def run_evaluation(thinking_model, thinking_tokenizer, base_model, base_tokenizer, 
                  sae, steering_vectors, descriptions, args, dataset, thinking_model_id, base_model_id):
    results = {
        "base_correct": 0,
        "thinking_correct": 0,
        "hybrid_correct": 0,
        "base_answers": [],
        "thinking_answers": [],
        "hybrid_answers": [],
        "questions": [],
        "correct_answers": [],
        "base_lengths": [],
        "thinking_lengths": [],
        "hybrid_lengths": [],
        "no_steering_fractions": [],
        "latent_usage": [],
        "steering_stats": [],  # New field to store detailed steering statistics
        "token_latent_info": [],  # New field to store token latent info for each example
        "steering_selection": []  # New field to store steering selection for each example
    }
    
    task_counter = 0
    for i, item in enumerate(dataset):
        if i < args.eval_start_idx:
            continue
        
        if task_counter >= args.n_tasks:
            break
            
        task_counter += 1
        print(f"\n===== Processing Task {task_counter}/{args.n_tasks} =====")
        
        # Extract question and answer
        if args.dataset == "gsm8k":
            question = item["question"]
            correct_answer = item["answer"]
        elif args.dataset == "aime":
            question = item["problem"]
            correct_answer = item["answer"]
        elif args.dataset == "math500":
            question = item["problem"]
            correct_answer = item["answer"]
        
        print(f"Question: {question[:100]}...")
        results["questions"].append(question)
        results["correct_answers"].append(correct_answer)
        
        # Format input for models
        thinking_input_ids = thinking_tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(thinking_model.device).to(torch.long)

        base_prompt = f"Question: {question}\n\nAnswer:"
        base_input_ids = base_tokenizer.encode(
            base_prompt,
            return_tensors="pt"
        ).to(base_model.device).to(torch.long)
        
        # Generate with thinking model
        print("Generating with Thinking Model...")
        with thinking_model.generate(thinking_input_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, pad_token_id=thinking_tokenizer.eos_token_id) as gen:
            thinking_outputs = thinking_model.generator.output.save()
        thinking_response = thinking_tokenizer.decode(thinking_outputs[0][len(thinking_input_ids[0]):], skip_special_tokens=True)
        results["thinking_answers"].append(thinking_response)
        results["thinking_lengths"].append(len(thinking_response.split()))
        
        # Extract cold start tokens from thinking response
        thinking_first_tokens = thinking_tokenizer.encode(thinking_response[:100], add_special_tokens=False)[:args.cold_start_tokens]
        cold_start_text = thinking_tokenizer.decode(thinking_first_tokens)
        
        # Add cold start tokens to base input
        cold_start_base_tokens = base_tokenizer.encode(cold_start_text, add_special_tokens=False, return_tensors="pt").to(base_model.device).to(torch.long)
        base_input_with_cold_start = torch.cat([base_input_ids, cold_start_base_tokens], dim=1)
        
        # Add cold start tokens to thinking input for hybrid generation
        thinking_input_with_cold_start = torch.cat([
            thinking_input_ids, 
            thinking_tokenizer.encode(cold_start_text, add_special_tokens=False, return_tensors="pt").to(thinking_model.device).to(torch.long)
        ], dim=1)
        
        # Generate with base model
        print("Generating with Base Model...")
        with base_model.generate(base_input_with_cold_start, max_new_tokens=args.max_new_tokens, temperature=args.temperature, pad_token_id=base_tokenizer.eos_token_id) as gen:
            base_outputs = base_model.generator.output.save()
        base_response = f"{cold_start_text}{base_tokenizer.decode(base_outputs[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
        results["base_answers"].append(base_response)
        results["base_lengths"].append(len(base_response.split()))
        
        # Generate with hybrid approach
        print("Generating with Hybrid Approach...")
        hybrid_output_ids, token_latent_info, per_token_perplexity, token_position, steering_selection = hybrid_generate(
            thinking_model=thinking_model,
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            thinking_input_ids=thinking_input_with_cold_start,
            base_input_ids=base_input_with_cold_start,
            max_new_tokens=args.max_new_tokens,
            thinking_layer=args.thinking_layer,
            sae=sae,
            steering_vectors=steering_vectors,
            latent_descriptions=descriptions,
            temperature=args.temperature,
            coefficient=args.coefficient,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            lookahead=args.lookahead,
            use_perplexity_selection=args.use_perplexity_selection,
            verbose=False
        )
        hybrid_response = f"{cold_start_text}{base_tokenizer.decode(hybrid_output_ids[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
        print(hybrid_response)
        results["hybrid_answers"].append(hybrid_response)
        results["hybrid_lengths"].append(len(hybrid_response.split()))
        
        # Store token latent info and steering selection
        results["token_latent_info"].append(token_latent_info)
        results["steering_selection"].append(steering_selection)
        
        # Analyze and store steering statistics
        no_steering_fraction, latent_counts, latent_percentages, steering_stats = analyze_hybrid_stats(token_latent_info, steering_selection)
        results["no_steering_fractions"].append(no_steering_fraction)
        results["latent_usage"].append(latent_percentages)
        results["steering_stats"].append(steering_stats)
        
        # Clean and evaluate answers
        clean_thinking_answer = clean_answer(thinking_response)
        clean_base_answer = clean_answer(base_response)
        clean_hybrid_answer = clean_answer(hybrid_response)
        
        # Evaluate answers
        print("\nEvaluating answers...")
        thinking_correct = evaluate_answer(clean_thinking_answer, correct_answer, question, "Thinking Model")
        base_correct = evaluate_answer(clean_base_answer, correct_answer, question, "Base Model")
        hybrid_correct = evaluate_answer(clean_hybrid_answer, correct_answer, question, "Hybrid Model")
        
        if thinking_correct:
            results["thinking_correct"] += 1
        if base_correct:
            results["base_correct"] += 1
        if hybrid_correct:
            results["hybrid_correct"] += 1
        
        # Print current results
        print(f"\nCurrent Results after {task_counter} tasks:")
        print(f"Thinking Model: {results['thinking_correct']}/{task_counter} correct ({results['thinking_correct']/task_counter*100:.1f}%)")
        print(f"Base Model: {results['base_correct']}/{task_counter} correct ({results['base_correct']/task_counter*100:.1f}%)")
        print(f"Hybrid Model: {results['hybrid_correct']}/{task_counter} correct ({results['hybrid_correct']/task_counter*100:.1f}%)")
        
        # Clean up to prevent memory leaks
        torch.cuda.empty_cache()
        gc.collect()

    # Calculate final accuracies
    thinking_accuracy = results["thinking_correct"] / task_counter * 100
    base_accuracy = results["base_correct"] / task_counter * 100
    hybrid_accuracy = results["hybrid_correct"] / task_counter * 100

    # Print final results
    print("\n===== Final Results =====")
    print(f"Thinking Model: {results['thinking_correct']}/{task_counter} correct ({thinking_accuracy:.1f}%)")
    print(f"Base Model: {results['base_correct']}/{task_counter} correct ({base_accuracy:.1f}%)")
    print(f"Hybrid Model: {results['hybrid_correct']}/{task_counter} correct ({hybrid_accuracy:.1f}%)")

    # Create bar chart
    plt.figure(figsize=(10, 6))
    model_names = ["Base", "Thinking", "Hybrid"]
    accuracies = [base_accuracy, thinking_accuracy, hybrid_accuracy]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    plt.bar(model_names, accuracies, color=colors)
    plt.title(f"Model Accuracy on {task_counter} {args.dataset} Tasks")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)

    for i, accuracy in enumerate(accuracies):
        plt.text(i, accuracy + 2, f"{accuracy:.1f}%", ha='center')

    plt.tight_layout()
    lookahead_str = "lookahead" if args.lookahead else "no_lookahead"
    plt.savefig(f"{args.results_dir}/accuracy_{base_model_id}_{args.dataset}_{lookahead_str}.png")
    plt.show()

    # Format the data for JSON saving
    benchmark_data = {
        "metadata": {
            "base_model": args.base_model,
            "thinking_model": args.thinking_model,
            "n_tasks": task_counter,
            "cold_start_tokens": args.cold_start_tokens,
        },
        "results": {
            "accuracy": {
                "base_model": base_accuracy,
                "thinking_model": thinking_accuracy,
                "hybrid_model": hybrid_accuracy
            },
            "correct_count": {
                "base_model": results["base_correct"],
                "thinking_model": results["thinking_correct"],
                "hybrid_model": results["hybrid_correct"]
            }
        },
        "tasks": []
    }

    # Add detailed results for each task
    for i in range(task_counter):
        task_data = {
            "question": results["questions"][i],
            "correct_answer": results["correct_answers"][i],
            "model_answers": {
                "base_model": results["base_answers"][i],
                "thinking_model": results["thinking_answers"][i],
                "hybrid_model": results["hybrid_answers"][i]
            }
        }
        benchmark_data["tasks"].append(task_data)

    # Save to JSON file
    lookahead_str = "" if args.lookahead else "_no_lookahead"
    use_perplexity_selection_str = "" if args.use_perplexity_selection else "_no_perplexity_selection"
    json_path = f"{args.results_dir}/benchmark_results_{base_model_id}_{args.dataset}{lookahead_str}{use_perplexity_selection_str}.json"
    with open(json_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"Benchmark results saved to {json_path}")

    return results

# Get command line arguments
args = parse_args()

# Create results directory if it doesn't exist
os.makedirs(args.results_dir, exist_ok=True)
os.makedirs(f"{args.results_dir}/vars", exist_ok=True)

# %% Load dataset
print(f"Loading {args.dataset} dataset...")
if args.dataset == 'gsm8k':
    dataset = load_dataset("openai/gsm8k", "main")["test"]  # type: ignore
elif args.dataset == "aime":
    dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]  # type: ignore
elif args.dataset == "math500":
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]  # type: ignore

# %% Load models and SAE
thinking_model, thinking_tokenizer, base_model, base_tokenizer, sae, steering_vectors, descriptions, thinking_model_id, base_model_id = load_models_and_sae(args)

# %% Run an example
print("\n===== Running Example =====")
thinking_response, base_response, hybrid_response, token_latent_info, steering_selection = run_example(
    thinking_model, thinking_tokenizer, base_model, base_tokenizer, 
    sae, steering_vectors, descriptions, args, dataset
)

# Analyze example stats
no_steering_fraction, latent_counts, latent_percentages, steering_stats = analyze_hybrid_stats(token_latent_info, steering_selection)

print("\n===== Example Statistics =====")
print(f"No-steering fraction: {no_steering_fraction:.2f}")
print("Latent usage (top 5):")
for latent, count in latent_counts.most_common(5):
    print(f"  {latent}: {count} tokens ({latent_percentages[latent]:.1f}%)")

# %% Run evaluation
print("\n===== Running Evaluation =====")
results = run_evaluation(
    thinking_model, thinking_tokenizer, base_model, base_tokenizer,
    sae, steering_vectors, descriptions, args, dataset, thinking_model_id, base_model_id
)

# Save detailed results
detailed_data = save_detailed_results(results, args, thinking_model_id, base_model_id)

# Plot additional statistics
if results["no_steering_fractions"]:
    plt.figure(figsize=(10, 6))
    plt.hist(results["no_steering_fractions"], bins=10, color="#2ecc71", alpha=0.7)
    plt.title("Distribution of No-Steering Fraction")
    plt.xlabel("Fraction of Tokens Using No Steering")
    plt.ylabel("Number of Tasks")
    plt.axvline(x=detailed_data["steering_stats"]["avg_no_steering"], color='red', linestyle='--', 
                label=f"Average: {detailed_data['steering_stats']['avg_no_steering']:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/no_steering_distribution_{base_model_id}_{args.dataset}.png")
    plt.show()

# Clean up
print("Evaluation complete. Cleaning up...")
del thinking_model, base_model, sae
torch.cuda.empty_cache()
gc.collect()

# %%
