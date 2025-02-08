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
            with tracer.invoke(input_ids[:, :end]) as invoker:
                # Collect activations from each layer
                for layer_idx in range(model.config.num_hidden_layers):
                    layer_activations.append(model.model.layers[layer_idx].output[0].save())
                    layer_gradients.append(model.model.layers[layer_idx].output[0].grad.save())
                
                # Get logits for the endpoints
                logits = model.lm_head.output.save()
                
                # Compute cross entropy metric for each labeled section
                value = compute_kl_divergence_metric(logits[0, start])

                # Backward pass
                value.backward()
    
        layer_activations = [layer_activations[i].value for i in range(model.config.num_hidden_layers)]
        layer_gradients = [layer_gradients[i].value for i in range(model.config.num_hidden_layers)]

        feature_activation = mean_vectors_dict[label]['mean'].to(torch.bfloat16).to("cuda") - mean_vectors_dict['overall']['mean'].to(torch.bfloat16).to("cuda")

        for layer_idx in range(model.config.num_hidden_layers):
            # Get activations and gradients for the entire labeled section
            activations = layer_activations[layer_idx][0, start-1:start]
            gradients = layer_gradients[layer_idx][0, start-1:start]
            
            effect = torch.einsum('d,sd->s', feature_activation[layer_idx], gradients).mean().abs()
            
            patching_effects[layer_idx] += effect.cpu().item()
        
    patching_effects = [effect / len(label_positions) for effect in patching_effects]

    return patching_effects

def plot_layer_effects(layer_effects, model_name):
    # Set white background
    plt.figure(figsize=(12, 8), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Remove ggplot style to keep default lines
    # plt.style.use('ggplot')  # Removing this line
    
    # Color scheme
    colors = ['#2E86C1', '#E67E22', '#27AE60', '#C0392B']
    
    for (label, effects), color in zip(layer_effects.items(), colors):
        if not effects:  # Skip if no effects for this label
            continue
            
        effects_array = np.array(effects)
        mean_effects = np.mean(effects_array, axis=0)
        
        # Apply smoothing using convolution
        window_size = 4  # Increase coarseness by reducing window size
        kernel = np.ones(window_size) / window_size
        smoothed_effects = np.convolve(mean_effects, kernel, mode='valid')
        
        x = range(len(smoothed_effects))

        std_effects = np.std(effects_array, axis=0)
        std_smoothed = np.convolve(std_effects, kernel, mode='valid')
        
        plt.fill_between(x, 
                        np.maximum(0, smoothed_effects - std_smoothed),
                        smoothed_effects + std_smoothed,
                        alpha=0.2, 
                        color=color)
        
        plt.plot(x, smoothed_effects, 
                label="{}".format(label.replace('-', '\n').title()),
                color=color,
                linewidth=2.5,
                marker='o',
                markersize=4)
    
    plt.xlabel('Layer', fontsize=24, labelpad=10, color='black')  # Set font color to black
    plt.ylabel('Mean KL-Divergence', fontsize=24, labelpad=10, color='black')  # Set font color to black
    plt.title('DeepSeek-R1-Distill-Llama-8B', fontsize=24, pad=20, color='black')  # Set font color to black
    plt.xticks(fontsize=24, color='black')  # Set font color to black
    plt.yticks(fontsize=24, color='black')  # Set font color to black
    
    # Remove offset on x-axis
    ax.margins(x=0)
    
    # Set y-axis limits to start at 0
    ax.set_ylim(bottom=0)
    
    # Add box and grid with stronger visibility
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # Make the box lines thicker
        spine.set_color('black')  # Set explicit color
    
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Enhanced grid settings
    plt.grid(True, 
             linestyle='--',      # Dashed lines
             alpha=0.4,           # More opaque
             color='gray',        # Gray color
             which='major')       # Show major grid lines
    
    plt.legend(bbox_to_anchor=(1, 1), 
              loc='upper right', 
              borderaxespad=0.,
              frameon=True,
              fontsize=22,
              facecolor='#f5f5f5')  # Set light gray background for the axes
    
    plt.tight_layout()
    
    model_id = model_name.split('/')[-1].lower()
    
    plt.savefig(f'figures/layer_effects_{model_id}.pdf', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.show()
    plt.close()

# %%
# Load model and data
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Can be changed to use different models
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
