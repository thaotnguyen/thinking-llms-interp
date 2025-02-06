# %%
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import NNsight
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import utils

# %%
def find_label_positions(annotated_response, original_text, tokenizer, label):
    """Parse annotations and find token positions for each label"""
    label_positions = []
    pattern = f'\\["{label}"\\]([^\\[]+?)(?=\\[|$)'
    matches = re.finditer(pattern, annotated_response)
    thinking_tokens = tokenizer.encode(original_text)[1:]
    
    for match in matches:

        text = match.group(1).strip()
        text_tokens = tokenizer.encode(text)[1:]
        
        for j in range(len(thinking_tokens) - len(text_tokens) + 1):
            if thinking_tokens[j:j + len(text_tokens)] == text_tokens:
                token_start = j
                token_end = j + len(text_tokens)
                label_positions.append((token_start, token_end))
                continue
    
    return label_positions

def compute_kl_divergence_metric(logits):
    """Compute KL divergence between predicted distribution and detached version"""
    probs = F.softmax(logits, dim=-1)
    detached_probs = F.softmax(logits.detach(), dim=-1)
    return F.kl_div(probs.log(), detached_probs, reduction='batchmean')

def analyze_layer_effects(model, tokenizer, text, label, mean_vectors_dict, label_positions):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    
    patching_effects = [0 for _ in range(model.config.num_hidden_layers)]

    if len(label_positions) == 0:
        return None

    for pos in label_positions:
        layer_activations = []
        layer_gradients = []
        
        start, end = pos

        with model.trace() as tracer:
            with tracer.invoke(input_ids[:, :start]) as invoker:
                # Collect activations from each layer
                for layer_idx in range(model.config.num_hidden_layers):
                    layer_activations.append(model.model.layers[layer_idx].output[0].save())
                    layer_gradients.append(model.model.layers[layer_idx].output[0].grad.save())
                
                # Get logits for the endpoints
                logits = model.lm_head.output.save()
                
                # Compute cross entropy metric for each labeled section
                value = compute_kl_divergence_metric(logits[0, start-1])

                # Backward pass
                value.backward()
    
        layer_activations = [layer_activations[i].value for i in range(model.config.num_hidden_layers)]
        layer_gradients = [layer_gradients[i].value for i in range(model.config.num_hidden_layers)]

        mean_activation = mean_vectors_dict['overall']['mean'].to(torch.bfloat16).to("cuda")

        for layer_idx in range(model.config.num_hidden_layers):
            # Get activations and gradients for the entire labeled section
            activations = layer_activations[layer_idx][0, start-1:end]
            gradients = layer_gradients[layer_idx][0, start-1:end]
            
            effect = torch.einsum('sd,sd->s', (activations - mean_activation[layer_idx].unsqueeze(0)), gradients).mean().abs()
            
            patching_effects[layer_idx] += effect.cpu().item()
        
    patching_effects = [effect / len(label_positions) for effect in patching_effects]

    return patching_effects

def plot_layer_effects(layer_effects, model_name):
    plt.figure(figsize=(12, 8))
    
    # Set style
    plt.style.use('ggplot')
    
    # Color scheme
    colors = ['#2E86C1', '#E67E22', '#27AE60', '#C0392B']
    
    for (label, effects), color in zip(layer_effects.items(), colors):
        if not effects:  # Skip if no effects for this label
            continue
            
        # Convert list of lists to numpy array
        effects_array = np.array(effects)
        
        # Calculate mean across examples
        mean_effects = np.mean(effects_array, axis=0)
        
        # Apply smoothing using convolution
        window_size = 4
        kernel = np.ones(window_size) / window_size
        smoothed_effects = np.convolve(mean_effects, kernel, mode='valid')
        
        # Adjust x-axis for smoothed data
        x = range(len(smoothed_effects))
        
        # Calculate standard deviation across examples
        std_effects = np.std(effects_array, axis=0)
        std_smoothed = np.convolve(std_effects, kernel, mode='valid')
        
        plt.fill_between(x, 
                        smoothed_effects - std_smoothed,
                        smoothed_effects + std_smoothed,
                        alpha=0.2, 
                        color=color)
        
        plt.plot(x, smoothed_effects, 
                label=f"{label.replace('-', ' ').title()} (n={len(effects)})",
                color=color,
                linewidth=2.5,
                marker='o',
                markersize=4)
    
    plt.xlabel('Layer', fontsize=12, fontweight='bold')
    plt.ylabel('Average Patching Effect', fontsize=12, fontweight='bold')
    plt.title('Layer-wise Attribution Effects Across Model', 
             fontsize=14, 
             fontweight='bold', 
             pad=20)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(bbox_to_anchor=(1.05, 1), 
              loc='upper left', 
              borderaxespad=0.,
              frameon=True,
              fontsize=10)
    
    plt.tight_layout()
    
    model_id = model_name.split('/')[-1].lower()
    
    plt.savefig(f'figures/layer_effects_{model_id}.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    plt.show()
    plt.close()

# %%
# Load model and data
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # Can be changed to use different models
model, tokenizer, mean_vectors_dict = utils.load_model_and_vectors(compute_features=False, model_name=model_name)

# %%
with open(f'data/responses_{model_name.split("/")[-1].lower()}.json', 'r') as f:
    results = json.load(f)

# %%
labels = ['uncertainty-estimation','adding-knowledge', 'example-testing', 'backtracking']
n_examples = 100 # Number of examples to analyze per label

# Store results
layer_effects = {label: [] for label in labels}

# Analyze each label
for label in labels:
    print(f"Analyzing label: {label}")
    for example in tqdm(results[:n_examples]):
        original_text = example['thinking_process']
        annotated_text = example['annotated_thinking']

        
        # Find token positions of labeled sentences
        label_positions = find_label_positions(annotated_text, original_text, tokenizer, label)

        if label_positions:  # Only process if we found labeled sentences
            effects = analyze_layer_effects(
                model, 
                tokenizer, 
                original_text, 
                label, 
                mean_vectors_dict,
                label_positions
            )

            if effects:
                layer_effects[label].append(effects)

# %% Plot results
plot_layer_effects(layer_effects, model_name)

# %%
