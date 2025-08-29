# %%
import dotenv
dotenv.load_dotenv("../.env")

import sys
import torch
import json
from utils.sae import load_sae
from utils.utils import load_model
from utils.clustering import get_latent_descriptions
from utils.utils import chat, chat_batch
from utils.utils import load_steering_vectors as _load_all_steering_vectors
import os
import time
import gc
import colorsys
import math
import matplotlib.pyplot as plt
import re
import argparse
from typing import List, Optional
from collections import Counter
from matplotlib.patches import Rectangle


from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate hybrid model on datasets (token-level steering)')
    parser.add_argument('--dataset', type=str, choices=['gsm8k', 'math500', "aime"], default='math500',
                      help='Dataset to evaluate on (gsm8k or math500)')
    parser.add_argument('--thinking_model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
                      help='Model for thinking/perplexity')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.1-8B',
                      help='Model for base generation')
    parser.add_argument('--steering_layer', type=int, default=12,
                      help='Layer to steer in the base model')
    parser.add_argument('--sae_layer', type=int, default=6,
                      help='Layer to read from in the thinking model for SAE projection')
    parser.add_argument('--n_clusters', type=int, default=15,
                      help='Number of clusters for SAE')
    parser.add_argument('--n_tasks', type=int, default=500,
                      help='Number of tasks to evaluate')
    parser.add_argument('--max_new_tokens', type=int, default=600,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--max_thinking_tokens', type=int, default=1200,
                      help='Maximum number of tokens for the thinking model only')
    parser.add_argument('--eval_start_idx', type=int, default=18,
                      help='Starting index in the dataset')
    parser.add_argument('--temperature', type=float, default=0.0,
                      help='Temperature for sampling')
    # Multi‑coefficient support
    # parser.add_argument('--coefficients', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #                     help='List of steering coefficients to evaluate per-token under the guardrail')
    # parser.add_argument('--token_windows', type=int, nargs='+', default=[-15, -50, -100],
    #                     help='List of token windows (negative = last N tokens) to apply steering to; e.g., -1 applies only to the last token')
    parser.add_argument('--coefficients', type=float, nargs='+', default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                      help='List of steering coefficients to evaluate per-token under the guardrail')
    parser.add_argument('--token_windows', type=int, nargs='+', default=[-1, -15, -50, -100, -200],
                      help='List of token windows (negative = last N tokens) to apply steering to; e.g., -1 applies only to the last token')
    parser.add_argument('--n_cold_start_tokens', type=int, default=0,
                      help='Number of initial tokens from the thinking model to prepend as a cold-start prefix')
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--example_idx', type=int, default=0,
                      help='Index of example to run')
    parser.add_argument('--use_perplexity_guardrail', action='store_true', default=True,
                      help='If set, select among steered candidates based on thinking-model perplexity')
    args = parser.parse_known_args()[0]
    return args

def get_next_token(logits, temperature, model, input_ids=None):
    """Get next token from logits using temperature sampling or greedy decoding (repetition penalty removed)"""
    if temperature > 0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).item()
        del logits, probs
        return token
    else:
        token = torch.argmax(logits).item()
        del logits
        return token

def get_token_and_string(logits, temperature, tokenizer, input_ids=None):
    """Get token ID and string from logits"""
    token = get_next_token(logits, temperature, tokenizer, input_ids)
    token_string = tokenizer.decode(token)
    return token, token_string

def get_perplexity(token_string, logits, model):
    """Calculate perplexity of a token string under the given logits"""
    token_id = model.tokenizer.encode(token_string, return_tensors="pt", add_special_tokens=False).to(model.device).to(torch.long)
    if token_id.shape[1] > 0 and token_id[0, 0].item() < logits.shape[-1]:
        log_prob = torch.log_softmax(logits[0, -1], dim=-1)[token_id[0, 0]].item()
        perplexity = math.exp(-log_prob)
        del token_id, log_prob
        return perplexity
    else:
        del token_id
        return float('inf')

# NEW: Helper to identify sentence boundaries (unused but handy for analysis)
def is_sentence_end(token_str: str) -> bool:
    stripped = token_str.strip()
    return bool(re.search(r'[.!?]$', stripped)) or stripped == "\n"

# NEW: Prepare cold-start inputs using the already generated thinking tokens
def prepare_cold_start(
    thinking_outputs: torch.Tensor,
    thinking_input_ids: torch.Tensor,
    base_input_ids: torch.Tensor,
    *,
    thinking_tokenizer,
    base_tokenizer,
    n_cold_start_tokens: int,
):
    if n_cold_start_tokens <= 0:
        return base_input_ids, thinking_input_ids, ""

    gen_slice = thinking_outputs[
        :,
        thinking_input_ids.shape[1] : thinking_input_ids.shape[1] + n_cold_start_tokens,
    ]

    cold_start_text: str = thinking_tokenizer.decode(gen_slice[0], skip_special_tokens=True)

    thinking_with_cold = torch.cat([thinking_input_ids, gen_slice.to(torch.long)], dim=1)

    base_cold_ids = (
        base_tokenizer.encode(cold_start_text, return_tensors="pt")
        .to(base_input_ids.device)
        .to(torch.long)
    )
    base_with_cold = torch.cat([base_input_ids, base_cold_ids], dim=1)

    del gen_slice, base_cold_ids
    torch.cuda.empty_cache()

    return base_with_cold, thinking_with_cold, cold_start_text

# ---------------------------------------------------------------------------------
# Token-level hybrid generation
# ---------------------------------------------------------------------------------

def hybrid_generate_token(
    thinking_model,
    base_model,
    base_tokenizer,
    thinking_input_ids,
    base_input_ids,
    max_new_tokens,
    steering_layer,
    sae_layer,
    sae,
    steering_vectors,
    latent_descriptions,
    *,
    coefficient: float = 1.0,
    temperature: float = 1.0,
    verbose: bool = False,
    use_perplexity_guardrail: bool = False,
    coefficients: Optional[List[float]] = None,
    token_windows: Optional[List[int]] = None,
):
    """Per-token variant of hybrid generation.

    For each output token:
      1) Use thinking model's last-token hidden state at `sae_layer` to choose the dominant latent and its steering vector.
      2) For each (coefficient, token_window) candidate, compute steered logits on base model.
      3) Select the next token by minimum perplexity under the thinking model (guardrail). If guardrail disabled, use the first provided candidate.
    """

    # Clone inputs so we do not modify the originals in-place
    base_output_ids = base_input_ids.clone()
    thinking_output_ids = thinking_input_ids.clone()
    del base_input_ids, thinking_input_ids
    torch.cuda.empty_cache()

    token_latent_info = []
    per_token_perplexity = []
    token_position = []
    steering_selection = []

    generated_tokens = 0

    # Access bias vector if present
    bias_vector = steering_vectors.get("bias", None)

    while generated_tokens < max_new_tokens:
        # 1) THINKING MODEL — derive steering vector from current position
        with torch.inference_mode():
            with thinking_model.trace(thinking_output_ids) as tracer:
                activation_curr = thinking_model.model.layers[sae_layer].output[0][0, -1, :].save()

        activation_curr = activation_curr.detach().clone()
        latent_acts = sae.encoder(activation_curr.to(torch.float32) - sae.b_dec)
        del activation_curr
        torch.cuda.empty_cache()

        latent_id = torch.argmax(latent_acts).item()
        latent_title = latent_descriptions[latent_id]["title"]
        latent_key = latent_descriptions[latent_id]["key"]
        activation_value = latent_acts[latent_id].item()
        steering_vector = steering_vectors[latent_key]
        del latent_acts
        torch.cuda.empty_cache()

        if verbose and (generated_tokens % 20 == 0):
            print(f"Token {generated_tokens}: latent={latent_title} (value={activation_value:.3f})")

        # 2) BASE MODEL — build candidate tokens across (coef, window)
        candidate_tokens = []
        if use_perplexity_guardrail:
            coef_list = coefficients if (coefficients is not None and len(coefficients) > 0) else [coefficient]
            win_list = token_windows if (token_windows is not None and len(token_windows) > 0) else [-1]
            for coef in coef_list:
                for win in win_list:
                    window_size = abs(int(win)) if int(win) != 0 else 0
                    with torch.inference_mode():
                        with base_model.trace(base_output_ids) as tracer:
                            layer_out = base_model.model.layers[steering_layer].output[0]
                            if window_size > 0:
                                if bias_vector is not None:
                                    layer_out[:, -window_size:, :] += coef * bias_vector
                                layer_out[:, -window_size:, :] += coef * steering_vector
                            else:
                                if bias_vector is not None:
                                    layer_out[:, :, :] += coef * bias_vector
                                layer_out[:, :, :] += coef * steering_vector
                            logits_steered_c = base_model.lm_head.output.save()
                    tok_c, tok_c_str = get_token_and_string(
                        logits_steered_c[0, -1],
                        temperature,
                        base_tokenizer,
                        base_output_ids,
                    )
                    del logits_steered_c
                    candidate_tokens.append({
                        "type": "steered",
                        "coef": float(coef),
                        "window": int(win),
                        "tok": tok_c,
                        "tok_str": tok_c_str,
                    })
        else:
            coef_to_use = (coefficients[0] if (coefficients is not None and len(coefficients) > 0) else coefficient)
            window_to_use = (token_windows[0] if (token_windows is not None and len(token_windows) > 0) else -1)
            window_size = abs(int(window_to_use)) if int(window_to_use) != 0 else 0
            with torch.inference_mode():
                with base_model.trace(base_output_ids) as tracer:
                    layer_out = base_model.model.layers[steering_layer].output[0]
                    if window_size > 0:
                        if bias_vector is not None:
                            layer_out[:, -window_size:, :] += coef_to_use * bias_vector
                        layer_out[:, -window_size:, :] += coef_to_use * steering_vector
                    else:
                        if bias_vector is not None:
                            layer_out[:, :, :] += coef_to_use * bias_vector
                        layer_out[:, :, :] += coef_to_use * steering_vector
                    logits_steered = base_model.lm_head.output.save()
            tok_steered, tok_steered_str = get_token_and_string(
                logits_steered[0, -1],
                temperature,
                base_tokenizer,
                base_output_ids,
            )
            del logits_steered
            candidate_tokens.append({
                "type": "steered",
                "coef": float(coef_to_use),
                "window": int(window_to_use),
                "tok": tok_steered,
                "tok_str": tok_steered_str,
            })

        # 3) Evaluate perplexity of each candidate under thinking model and pick best
        with torch.inference_mode():
            with thinking_model.trace(thinking_output_ids) as tracer:
                logits_thinking_curr = thinking_model.lm_head.output.save()

        if use_perplexity_guardrail:
            best = None  # tuple(perplexity, index)
            for idx, cand in enumerate(candidate_tokens):
                p = get_perplexity(cand["tok_str"], logits_thinking_curr, thinking_model)
                cand["perplexity"] = p
                if best is None or p < best[0]:
                    best = (p, idx)
            assert best is not None
            chosen_cand = candidate_tokens[best[1]]
            next_tok = chosen_cand["tok"]
            next_tok_str = chosen_cand["tok_str"]
            token_perpl = chosen_cand["perplexity"]
            chosen = chosen_cand["type"]
            chosen_coef = chosen_cand.get("coef", None)
            chosen_window = chosen_cand.get("window", None)
        else:
            p = get_perplexity(candidate_tokens[0]["tok_str"], logits_thinking_curr, thinking_model)
            candidate_tokens[0]["perplexity"] = p
            next_tok = candidate_tokens[0]["tok"]
            next_tok_str = candidate_tokens[0]["tok_str"]
            token_perpl = p
            chosen = candidate_tokens[0]["type"]
            chosen_coef = candidate_tokens[0].get("coef", None)
            chosen_window = candidate_tokens[0].get("window", None)

        del logits_thinking_curr
        torch.cuda.empty_cache()

        # 4) Append chosen token to both sequences
        base_tok_ids = base_tokenizer.encode(
            next_tok_str, return_tensors="pt", add_special_tokens=False
        ).to(base_model.device).to(torch.long)
        thinking_tok_ids = thinking_model.tokenizer.encode(
            next_tok_str, return_tensors="pt", add_special_tokens=False
        ).to(thinking_model.device).to(torch.long)

        base_output_ids = torch.cat([base_output_ids, base_tok_ids], dim=1)
        thinking_output_ids = torch.cat([thinking_output_ids, thinking_tok_ids], dim=1)
        del base_tok_ids, thinking_tok_ids
        torch.cuda.empty_cache()

        # 5) Book-keeping
        steering_selection.append(chosen)
        per_token_perplexity.append(token_perpl)
        token_position.append(len(token_latent_info))
        token_latent_info.append(
            {
                "token": next_tok_str,
                "latent_id": latent_id if chosen == "steered" else None,
                "latent_title": latent_title if chosen == "steered" else "No Steering",
                "latent_key": latent_key if chosen == "steered" else None,
                "activation_value": activation_value if chosen == "steered" else 0.0,
                "perplexity": token_perpl,
                "coefficient": chosen_coef if chosen == "steered" else None,
                "window": chosen_window if chosen == "steered" else None,
                "selection": chosen,
                "future_token": None,
            }
        )

        generated_tokens += 1

        # Stop if EOS
        if next_tok == base_tokenizer.eos_token_id:
            break

    # Final cleanup
    del steering_vector
    gc.collect()
    torch.cuda.empty_cache()

    return (
        base_output_ids,
        token_latent_info,
        per_token_perplexity,
        token_position,
        steering_selection,
    )

def load_steering_vectors(model_id, thinking_model_id, sae_layer, n_clusters):
    """Load steering vectors from train_vectors output"""
    hyperparams_dir_abs = os.path.join(os.path.dirname(__file__), "../train-vectors/results/vars/hyperparams")
    vectors_dir_abs = os.path.join(os.path.dirname(__file__), "../train-vectors/results/vars/optimized_vectors")

    all_vectors = _load_all_steering_vectors(
        hyperparams_dir=hyperparams_dir_abs,
        vectors_dir=vectors_dir_abs,
        verbose=False,
    )

    descriptions = get_latent_descriptions(thinking_model_id, sae_layer, n_clusters)
    steering_vectors = {}
    for idx, desc in descriptions.items():
        latent_title = desc.get("key", "")
        if not latent_title:
            continue
        key = latent_title.lower().replace(" ", "-")
        if key in all_vectors:
            steering_vectors[key] = all_vectors[key]

    # Include general bias vector (if present)
    if "bias" in all_vectors and "bias" not in steering_vectors:
        steering_vectors["bias"] = all_vectors["bias"]
    else:
        print("No bias vector found")

    print(steering_vectors.keys())

    return steering_vectors

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
        patch = Rectangle((0, 0), 1, 1, fc=color)
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
    print(f"Loading SAE for model {thinking_model_id}, layer {args.sae_layer}...")
    sae, _ = load_sae(thinking_model_id, args.sae_layer, args.n_clusters)
    sae = sae.to(thinking_model.device)
    print(f"Loading steering vectors and layer effects...")
    descriptions = get_latent_descriptions(thinking_model_id, args.sae_layer, args.n_clusters)
    steering_vectors = load_steering_vectors(base_model_id, thinking_model_id, args.sae_layer, args.n_clusters)
    # Move steering vectors to base model device and dtype
    base_device = base_model.device
    base_dtype = next(base_model.parameters()).dtype if hasattr(base_model, "parameters") else torch.float32
    for k, v in list(steering_vectors.items()):
        if isinstance(v, torch.Tensor):
            steering_vectors[k] = v.to(device=base_device, dtype=base_dtype, non_blocking=True)
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
    with thinking_model.generate(thinking_input_ids, max_new_tokens=args.max_thinking_tokens, temperature=args.temperature, pad_token_id=thinking_tokenizer.eos_token_id) as gen:
        thinking_outputs = thinking_model.generator.output.save()
    thinking_response = thinking_tokenizer.decode(thinking_outputs[0][len(thinking_input_ids[0]):], skip_special_tokens=True)
    print(thinking_response)
    
    # Generate with base model
    print("\n===== Generating with Base Model =====")
    base_input_with_cold_start, _, cold_start_text = prepare_cold_start(
        thinking_outputs,
        thinking_input_ids,
        base_input_ids,
        thinking_tokenizer=thinking_tokenizer,
        base_tokenizer=base_tokenizer,
        n_cold_start_tokens=args.n_cold_start_tokens,
    )
    
    with base_model.generate(base_input_with_cold_start, max_new_tokens=100, temperature=args.temperature, pad_token_id=base_tokenizer.eos_token_id) as gen:
        base_outputs = base_model.generator.output.save()
    base_response = f"{cold_start_text}{base_tokenizer.decode(base_outputs[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
    print(base_response)
    
    # Clean up base model outputs
    del base_outputs
    torch.cuda.empty_cache()
    
    # Generate with hybrid approach
    print("\n===== Generating with Hybrid Approach (Token-Level) =====")
    base_input_with_cold_start, thinking_input_with_cold_start, cold_start_text = prepare_cold_start(
        thinking_outputs,
        thinking_input_ids,
        base_input_ids,
        thinking_tokenizer=thinking_tokenizer,
        base_tokenizer=base_tokenizer,
        n_cold_start_tokens=args.n_cold_start_tokens,
    )
    
    hybrid_output_ids, token_latent_info, per_token_perplexity, token_position, steering_selection = hybrid_generate_token(
        thinking_model=thinking_model,
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        thinking_input_ids=thinking_input_with_cold_start,
        base_input_ids=base_input_with_cold_start,
        max_new_tokens=args.max_new_tokens,
        steering_layer=args.steering_layer,
        sae_layer=args.sae_layer,
        sae=sae,
        steering_vectors=steering_vectors,
        latent_descriptions=descriptions,
        temperature=args.temperature,
        coefficient=(args.coefficients[0] if args.coefficients else 0.3),
        coefficients=args.coefficients,
        token_windows=args.token_windows,
        verbose=False,
        use_perplexity_guardrail=args.use_perplexity_guardrail
    )
    hybrid_response = f"{cold_start_text}{base_tokenizer.decode(hybrid_output_ids[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
    print(hybrid_response)
    
    # Clean up hybrid outputs
    del hybrid_output_ids
    torch.cuda.empty_cache()
    
    # Print correct answer for reference
    print("\n===== Correct Answer =====")
    print(answer)
    
    # Visualize results
    latent_colors = generate_latent_colors(descriptions)
    visualize_generation_results(token_latent_info, steering_selection, per_token_perplexity, token_position, latent_colors)
    
    # Clean up example-specific variables
    del latent_colors, per_token_perplexity, token_position
    
    return thinking_response, base_response, hybrid_response, token_latent_info, steering_selection

def clean_answer(text):
    return re.sub(r'\s+', ' ', text).strip()

def safe_chat_batch(prompts, model_name: str = "gpt-4o", max_tokens: int = 1024, **kwargs):
    import asyncio
    import concurrent.futures
    async def _run():
        return await chat_batch(
            prompts,
            model=model_name,
            max_tokens=max_tokens,
            **kwargs,
        )
    try:
        loop = asyncio.get_running_loop()
        def _thread_runner():
            return asyncio.run(_run())
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(_thread_runner)
            return fut.result()
    except RuntimeError:
        return asyncio.run(_run())

def evaluate_answer(model_answer, correct_answer, question, model_name):
    prompt = f"""Please evaluate whether the following answer to a math problem is correct.

Question: {question}

Correct answer: {correct_answer}

Model's answer: {model_answer}

First, extract the final numerical answer from both the correct answer and model's answer. 
Then determine if the model's final numerical answer is equivalent to the correct final numerical answer.
Just answer YES if the model's answer is correct, or NO if it's incorrect. Nothing else.
"""
    
    response_list = safe_chat_batch([prompt], model_name="openai/gpt-4o", max_tokens=100)
    response = response_list[0] if isinstance(response_list, (list, tuple)) else response_list
    is_correct = "yes" in response.lower()
    print(f"{model_name} evaluated as: {response}")
    return is_correct, response

def append_rolling_result(record: dict, args, base_model_id: str):
    os.makedirs(f"{args.results_dir}/rolling", exist_ok=True)
    path = f"{args.results_dir}/rolling/rolling_{base_model_id}_{args.dataset}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def analyze_hybrid_stats(token_latent_info, steering_selection):
    steered_count = steering_selection.count("steered")
    unsteered_count = steering_selection.count("unsteered")
    total = steered_count + unsteered_count
    no_steering_fraction = unsteered_count / total if total > 0 else 0
    latent_counts = Counter()
    for info in token_latent_info:
        if info["latent_title"] != "No Steering":
            latent_counts[info["latent_title"]] += 1
    latent_percentages = {}
    if steered_count > 0:
        for latent, count in latent_counts.items():
            latent_percentages[latent] = (count / steered_count) * 100
    steering_stats = {
        "steered_count": steered_count,
        "unsteered_count": unsteered_count,
        "total_tokens": total,
        "steering_fraction": steered_count / total if total > 0 else 0,
        "no_steering_fraction": no_steering_fraction
    }
    return no_steering_fraction, latent_counts, latent_percentages, steering_stats

def save_detailed_results(results, args, thinking_model_id, base_model_id):
    os.makedirs(f"{args.results_dir}/detailed", exist_ok=True)
    filename = f"{args.results_dir}/detailed/hybrid_stats_{base_model_id}_{args.dataset}.json"
    avg_steering_stats = {
        "steered_count": sum(stat["steered_count"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "unsteered_count": sum(stat["unsteered_count"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "total_tokens": sum(stat["total_tokens"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "steering_fraction": sum(stat["steering_fraction"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "no_steering_fraction": sum(stat["no_steering_fraction"] for stat in results["steering_stats"]) / len(results["steering_stats"])
    }
    detailed_data = {
        "metadata": {
            "base_model": args.base_model,
            "thinking_model": args.thinking_model,
            "dataset": args.dataset,
            "temperature": args.temperature,
            "coefficients": args.coefficients,
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
    with open(filename, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    print(f"Detailed results saved to {filename}")
    return detailed_data

def run_evaluation(thinking_model, thinking_tokenizer, base_model, base_tokenizer, 
                  sae, steering_vectors, descriptions, args, dataset, thinking_model_id, base_model_id):

    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

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
        "steering_stats": [],
        "token_latent_info": [],
        "steering_selection": []
    }
    
    task_counter = 0
    for i, item in enumerate(dataset):
        if i < args.eval_start_idx:
            continue
        if task_counter >= args.n_tasks:
            break
        task_counter += 1
        print(f"\n===== Processing Task {task_counter}/{args.n_tasks} =====")
        
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
        print(f"Correct answer: {correct_answer}")
        results["questions"].append(question)
        results["correct_answers"].append(correct_answer)
        
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
        
        # Thinking model
        print("Generating with Thinking Model...")
        clear_gpu_memory()
        with thinking_model.generate(thinking_input_ids, max_new_tokens=args.max_thinking_tokens, temperature=args.temperature, pad_token_id=thinking_tokenizer.eos_token_id) as gen:
            thinking_outputs = thinking_model.generator.output.save()
        thinking_response = thinking_tokenizer.decode(thinking_outputs[0][len(thinking_input_ids[0]):], skip_special_tokens=True)
        results["thinking_answers"].append(thinking_response)
        results["thinking_lengths"].append(len(thinking_response.split()))

        # Base model
        print("Generating with Base Model...")
        clear_gpu_memory()
        base_input_with_cold_start, _, cold_start_text = prepare_cold_start(
            thinking_outputs,
            thinking_input_ids,
            base_input_ids,
            thinking_tokenizer=thinking_tokenizer,
            base_tokenizer=base_tokenizer,
            n_cold_start_tokens=args.n_cold_start_tokens,
        )
        with base_model.generate(base_input_with_cold_start, max_new_tokens=args.max_new_tokens, temperature=args.temperature, pad_token_id=base_tokenizer.eos_token_id) as gen:
            base_outputs = base_model.generator.output.save()
        base_response = f"{cold_start_text}{base_tokenizer.decode(base_outputs[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
        del base_outputs, base_input_with_cold_start
        clear_gpu_memory()
        results["base_answers"].append(base_response)
        results["base_lengths"].append(len(base_response.split()))
        
        # Hybrid token-level
        print("Generating with Hybrid Approach (Token-Level)...")
        clear_gpu_memory()
        base_input_with_cold_start, thinking_input_with_cold_start, cold_start_text = prepare_cold_start(
            thinking_outputs,
            thinking_input_ids,
            base_input_ids,
            thinking_tokenizer=thinking_tokenizer,
            base_tokenizer=base_tokenizer,
            n_cold_start_tokens=args.n_cold_start_tokens,
        )
        hybrid_output_ids, token_latent_info, per_token_perplexity, token_position, steering_selection = hybrid_generate_token(
            thinking_model=thinking_model,
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            thinking_input_ids=thinking_input_with_cold_start,
            base_input_ids=base_input_with_cold_start,
            max_new_tokens=args.max_new_tokens,
            steering_layer=args.steering_layer,
            sae_layer=args.sae_layer,
            sae=sae,
            steering_vectors=steering_vectors,
            latent_descriptions=descriptions,
            temperature=args.temperature,
            coefficient=(args.coefficients[0] if args.coefficients else 0.3),
            coefficients=args.coefficients,
            token_windows=args.token_windows,
            verbose=False,
            use_perplexity_guardrail=args.use_perplexity_guardrail
        )
        hybrid_response = f"{cold_start_text}{base_tokenizer.decode(hybrid_output_ids[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
        del hybrid_output_ids, base_input_with_cold_start, thinking_input_with_cold_start
        clear_gpu_memory()
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
        thinking_correct, thinking_judge_raw = evaluate_answer(clean_thinking_answer, correct_answer, question, "Thinking Model")
        base_correct, base_judge_raw = evaluate_answer(clean_base_answer, correct_answer, question, "Base Model")
        hybrid_correct, hybrid_judge_raw = evaluate_answer(clean_hybrid_answer, correct_answer, question, "Hybrid Model")
        
        if thinking_correct:
            results["thinking_correct"] += 1
        if base_correct:
            results["base_correct"] += 1
        if hybrid_correct:
            results["hybrid_correct"] += 1
        
        # Rolling save for this task
        rolling_record = {
            "ts": time.time(),
            "dataset": args.dataset,
            "question": question,
            "gold_answer": correct_answer,
            "answers": {
                "thinking": thinking_response,
                "base": base_response,
                "hybrid": hybrid_response,
            },
            "judges": {
                "thinking": {"correct": bool(thinking_correct), "raw": thinking_judge_raw},
                "base": {"correct": bool(base_correct), "raw": base_judge_raw},
                "hybrid": {"correct": bool(hybrid_correct), "raw": hybrid_judge_raw},
            },
            "hybrid_details": {
                "per_token": token_latent_info,
                "steering_selection": steering_selection,
                "coefficients": args.coefficients,
                "token_windows": args.token_windows,
                "sae_layer": args.sae_layer,
                "steering_layer": args.steering_layer,
            },
        }
        append_rolling_result(rolling_record, args, base_model_id)

        # Print current results
        print(f"\nCurrent Results after {task_counter} tasks:")
        print(f"Thinking Model: {results['thinking_correct']}/{task_counter} correct ({results['thinking_correct']/task_counter*100:.1f}%)")
        print(f"Base Model: {results['base_correct']}/{task_counter} correct ({results['base_correct']/task_counter*100:.1f}%)")
        print(f"Hybrid Model: {results['hybrid_correct']}/{task_counter} correct ({results['hybrid_correct']/task_counter*100:.1f}%)")
        
        # Clean up to prevent memory leaks
        del thinking_input_ids, base_input_ids, thinking_outputs
        del token_latent_info, per_token_perplexity, token_position, steering_selection
        del thinking_response, base_response, hybrid_response
        del clean_thinking_answer, clean_base_answer, clean_hybrid_answer
        del latent_counts, latent_percentages, steering_stats
        del cold_start_text
        torch.cuda.empty_cache()
        gc.collect()

    thinking_accuracy = results["thinking_correct"] / task_counter * 100
    base_accuracy = results["base_correct"] / task_counter * 100
    hybrid_accuracy = results["hybrid_correct"] / task_counter * 100

    print("\n===== Final Results =====")
    print(f"Thinking Model: {results['thinking_correct']}/{task_counter} correct ({thinking_accuracy:.1f}%)")
    print(f"Base Model: {results['base_correct']}/{task_counter} correct ({base_accuracy:.1f}%)")
    print(f"Hybrid Model: {results['hybrid_correct']}/{task_counter} correct ({hybrid_accuracy:.1f}%)")

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
    plt.savefig(f"{args.results_dir}/accuracy_{base_model_id}_{args.dataset}.png")
    plt.show()

    benchmark_data = {
        "metadata": {
            "base_model": args.base_model,
            "thinking_model": args.thinking_model,
            "n_tasks": task_counter,
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
    json_path = f"{args.results_dir}/benchmark_results_{base_model_id}_{args.dataset}.json"
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

