# %%
import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import NNsight
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
import sys
import math
import gc
from utils.utils import model_mapping

# Add parent directory to path for imports
sys.path.append('..')
from utils import utils

# Add argparse for model selection
parser = argparse.ArgumentParser(description="Analyze layer effects for SAE decoder latents")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                    help="Model to analyze")
parser.add_argument("--n_examples", type=int, default=100,
                    help="Number of sentences to analyze per latent")
parser.add_argument("--load_in_8bit", type=bool, default=False,
                    help="Load the model in 8-bit mode")
parser.add_argument("--layer", type=int, default=14,
                    help="Layer number for SAE")
parser.add_argument("--n_clusters", type=int, default=19,
                    help="Number of clusters for SAE")
args, _ = parser.parse_known_args()

# %%
def split_into_sentences(original_text, thinking_text, tokenizer):
    """Split text into sentences and find their token positions using character to token mapping"""
    # Split into sentences using regex
    sentences = re.split(r'[.!?;]', thinking_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if len(s.split()) >= 3]    

    # Get character to token mapping
    char_to_token = utils.get_char_to_token_map(original_text, tokenizer)
    
    # Process each sentence
    sentence_positions = []
    for sentence in sentences:
        # Find this sentence in the original text
        text_pos = original_text.find(sentence)
        if text_pos >= 0:
            # Get start and end token positions
            token_start = char_to_token.get(text_pos, None)
            if token_start <= 0:
                continue
            token_end = char_to_token.get(text_pos + len(sentence) - 1, None)
            
            if token_start is not None and token_end is not None and token_start < token_end:
                # Add 1 to token_end to make it inclusive
                sentence_positions.append((token_start, token_end))
    
    return sentence_positions

def compute_kl_divergence_metric(logits):
    """Compute KL divergence between predicted distribution and detached version"""
    probs = F.log_softmax(logits, dim=-1)
    detached_probs = F.log_softmax(logits.detach(), dim=-1)
    return F.kl_div(probs, detached_probs, reduction='batchmean')

def analyze_layer_effects(model, tokenizer, text, sae_decoder_weights, label_positions, max_sentences=None):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    
    # Initialize effects for all latents and layers
    patching_effects = {latent_idx: [0 for _ in range(model.config.num_hidden_layers)] 
                       for latent_idx in range(sae_decoder_weights.shape[0])}

    if len(label_positions) == 0:
        return None

    # Limit number of sentences to process if max_sentences is specified
    if max_sentences is not None:
        label_positions = label_positions[:max_sentences]

    for pos in label_positions:
        layer_gradients = []
        layer_activations = []
        
        start, end = pos

        with model.trace() as tracer:
            with tracer.invoke(
                {
                    "input_ids": input_ids[:, :end], 
                    "attention_mask": (input_ids[:, :end] != tokenizer.pad_token_id).long()
                }
            ) as invoker:
                # Collect activations from each layer
                for layer_idx in range(model.config.num_hidden_layers):
                    layer_gradients.append(model.model.layers[layer_idx].output[0].grad.detach().save())
                    layer_activations.append(model.model.layers[layer_idx].output[0][:,1:].detach().save())

                # Get logits for the endpoints
                logits = model.lm_head.output.save()
                
                # Compute cross entropy metric for each labeled section
                value = compute_kl_divergence_metric(logits[0, start-1])

                # Backward pass
                value.backward()
    
        layer_gradients = [layer_gradients[i].value for i in range(model.config.num_hidden_layers)]
        layer_activations = [layer_activations[i] for i in range(model.config.num_hidden_layers)]

        # Process all latents for this example
        for layer_idx in range(model.config.num_hidden_layers):
            # Get activations and gradients for the entire labeled section
            gradients = layer_gradients[layer_idx][0, start-1:start]
        
            if args.model in list(model_mapping.keys()):
                decoder_weights = sae_decoder_weights[:, layer_idx].to(torch.bfloat16) - feature_vectors["overall"][layer_idx].unsqueeze(0).to(torch.bfloat16).to("cuda")
            else:
                decoder_weights = sae_decoder_weights.to(torch.bfloat16)

            # Normalize decoder weights
            normalized_decoder_weights = decoder_weights / decoder_weights.norm(dim=-1, keepdim=True)
            
            effects = torch.einsum('ld,sd->ls', normalized_decoder_weights, gradients).abs()

            # Add effects to running total
            for latent_idx in range(sae_decoder_weights.shape[0]):
                patching_effects[latent_idx][layer_idx] += effects[latent_idx].mean().item()
        
            # Clean up layer-specific tensors
            del gradients, normalized_decoder_weights
        
        # Clean up batch-specific tensors
        del layer_gradients
        torch.cuda.empty_cache()
        gc.collect()

    # Average effects across sentences
    for latent_idx in patching_effects:
        patching_effects[latent_idx] = [effect / len(label_positions) for effect in patching_effects[latent_idx]]

    return patching_effects

def plot_layer_effects_grid(layer_effects, model_name, latent_descriptions):
    # Calculate grid dimensions
    n_latents = sum(1 for latent_idx, effects in layer_effects.items() if effects)
    
    # Handle case where no latents have effects
    if n_latents == 0:
        print("No layer effects found for any latent. Skipping visualization.")
        return
    
    # Calculate grid dimensions - square-ish grid
    n_cols = int(math.ceil(math.sqrt(n_latents)))
    n_rows = int(math.ceil(n_latents / n_cols))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), facecolor='white')
    
    # Handle case where there's only one subplot
    if n_latents == 1:
        axes = np.array([axes])
    
    # Flatten the axes for easier indexing
    axes = axes.flatten()
    
    # Get model ID for title
    model_id = model_name.split('/')[-1]
    
    # Use a colormap for variety when there are many latents
    cmap = plt.cm.tab20
    
    # Counter for valid latents
    valid_latent_idx = 0
    
    for latent_idx, effects in layer_effects.items():
        if not effects:  # Skip if no effects for this latent
            continue
            
        # Get current axis
        ax = axes[valid_latent_idx]
        ax.set_facecolor('white')
        
        # Get color from colormap
        color = cmap(valid_latent_idx % 20)
        
        effects_array = np.array(effects)
        
        # Handle NaN values by replacing them with 0
        effects_array = np.nan_to_num(effects_array, nan=0.0)
        
        # Compute mean and std, ignoring NaN values
        mean_effects = np.nanmean(effects_array, axis=0)
        std_effects = np.nanstd(effects_array, axis=0)
        
        # Apply smoothing using convolution
        window_size = 1  # Increase coarseness by reducing window size
        kernel = np.ones(window_size) / window_size
        smoothed_effects = np.convolve(mean_effects, kernel, mode='valid')
        std_smoothed = np.convolve(std_effects, kernel, mode='valid')
        
        x = range(len(smoothed_effects))
        
        # Get latent title from descriptions
        latent_title = f"Latent {latent_idx}"
        if latent_descriptions and latent_idx in latent_descriptions:
            latent_title = f"Latent {latent_idx}: {latent_descriptions[latent_idx]['title']}"
        
        ax.fill_between(x, 
                        smoothed_effects - std_smoothed,
                        smoothed_effects + std_smoothed,
                        alpha=0.2, 
                        color=color)
        
        ax.plot(x, smoothed_effects, 
                color=color,
                linewidth=2.5,
                marker='o',
                markersize=4)
        
        # Set title and labels for each subplot
        ax.set_title(latent_title, 
                    fontsize=14, 
                    pad=5, 
                    color='black')
        
        ax.set_xlabel('Layer', fontsize=10, labelpad=5, color='black')
        
        # Set y-label for leftmost subplots
        if valid_latent_idx % n_cols == 0:
            ax.set_ylabel('Mean KL-Divergence', fontsize=10, labelpad=5, color='black')
        
        ax.tick_params(axis='both', which='major', labelsize=9, colors='black')
        
        # Remove offset on x-axis
        ax.margins(x=0)
        
        # Add box and grid with stronger visibility
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)  # Make the box lines thicker
            spine.set_color('black')  # Set explicit color
        
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        
        # Enhanced grid settings
        ax.grid(True, 
                linestyle='--',      # Dashed lines
                alpha=0.4,           # More opaque
                color='gray',        # Gray color
                which='major')       # Show major grid lines
        
        valid_latent_idx += 1
    
    # Hide unused subplots
    for i in range(valid_latent_idx, len(axes)):
        axes[i].set_visible(False)
    
    # Add a common title for all subplots
    fig.suptitle(f"Layer Effects for SAE Latents - {model_id}", fontsize=20, y=0.98, color='black')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the suptitle
    
    model_id_lower = model_name.split('/')[-1].lower()
    
    plt.savefig(f'results/figures/layer_effects_{model_id_lower}_layer_{layer}_n_clusters_{n_clusters}.pdf', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.show()
    plt.close()

# %%
# Load model and data
model_name = args.model
layer = args.layer
n_clusters = args.n_clusters
print(f"Loading model {model_name}...")
model, tokenizer = utils.load_model(model_name=model_name)

# Get model ID from model name
model_id = model_name.split('/')[-1].lower()

# %% Load SAE
thinking_model_id = model_mapping[args.model].split("/")[-1].lower() if args.model in list(model_mapping.keys()) else model_id
if args.model in list(model_mapping.keys()):
    print(f"Loading feature vectors for layer {layer} with {n_clusters} clusters...")
    feature_vectors = torch.load(f"../train-vectors/results/vars/mean_vectors_{model_id}_fs3.pt")
else:
    print(f"Loading SAE for layer {layer} with {n_clusters} clusters...")
    sae, checkpoint = utils.load_sae(thinking_model_id, layer, n_clusters)

latent_descriptions = utils.get_latent_descriptions(thinking_model_id, layer, n_clusters)

# Create directories
os.makedirs('results/vars', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# %%
# Get responses data
prefix = "base_" if args.model in list(model_mapping.keys()) else ""
responses_path = f'../generate-responses/results/vars/{prefix}eval_responses_{model_id}.json'

with open(responses_path, 'r') as f:
    results = json.load(f)

# %%
# Number of sentences to analyze per latent
n_examples = args.n_examples

# Analyze layer effects for all latents at once
print("Analyzing layer effects for all latents...")
if args.model in list(model_mapping.keys()):
    layer_effects = {latent_idx: [] for latent_idx in range(len(feature_vectors))}
    w_dec = torch.stack([feature_vectors[x["title"].lower().replace(" ", "-")] for x in latent_descriptions.values()]).to(model.device)
else:
    layer_effects = {latent_idx: [] for latent_idx in range(sae.W_dec.shape[0])}
    w_dec = sae.W_dec.to(model.device)

total_sentences = 0
for example in tqdm(results):
    if total_sentences >= n_examples:
        break
    
    original_text = example['full_response'] if not args.model in list(model_mapping.keys()) else f"Respond to the following questions step by step.\n\nQuestion:\n{example['original_message']['content']}\nStep by step answer:\n{example['thinking_process']}"
    thinking_text = example['thinking_process']
    
    # Split text into sentences instead of using annotations
    sentence_positions = split_into_sentences(original_text, thinking_text, tokenizer)

    if sentence_positions:  # Only process if we found sentences
        # Calculate how many more sentences we can process
        remaining_sentences = n_examples - total_sentences
        effects = analyze_layer_effects(
            model,
            tokenizer,
            original_text,
            w_dec,
            sentence_positions,
            max_sentences=remaining_sentences
        )
        
        if effects:
            for latent_idx, latent_effects in effects.items():
                layer_effects[latent_idx].append(latent_effects)
            total_sentences += min(len(sentence_positions), remaining_sentences)

# %% Plot results
plot_layer_effects_grid(layer_effects, model_name, latent_descriptions)

# %% Save average layer effects to JSON
avg_layer_effects = {}
for latent_idx, effects_list in layer_effects.items():
    if effects_list:  # Only process if we have effects for this latent
        # Convert to numpy array and compute mean across examples
        effects_array = np.array(effects_list)
        avg_effects = np.nanmean(effects_array, axis=0).tolist()
        
        # Calculate the maximum average score for this latent
        max_avg_score = np.nanmax(np.nanmean(effects_array, axis=0))
        
        latent_title = latent_descriptions[latent_idx]['title']
            
        avg_layer_effects[str(latent_idx)] = {
            "title": latent_title,
            "avg_effects": avg_effects,
            "max_layer": np.argmax(avg_effects).item()  # Convert to float for JSON serialization
        }

# Save to JSON file
output_path = f'results/vars/layer_effects_{model_id}_layer_{layer}_n_clusters_{n_clusters}.json'
with open(output_path, 'w') as f:
    json.dump(avg_layer_effects, f, indent=2)

print(f"Saved average layer effects to {output_path}")
# %%
