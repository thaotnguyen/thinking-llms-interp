# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nnsight import NNsight
from utils import chat
import re
import json
import random
from messages import eval_messages
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import gc
import os
import utils
# %%
def load_model_and_vectors(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    model, tokenizer, feature_vectors = utils.load_model_and_vectors(compute_features=True, model_name=model_name)
    return model, tokenizer, feature_vectors

def get_label_counts(thinking_process, labels):
    # Get annotated version using chat function
    annotated_response = chat(f"""
    Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.

    Available labels:
    0. initializing -> The model is rephrasing the given task and states initial thoughts.
    1. deduction -> The model is performing a deduction step based on its current approach and assumptions.
    2. adding-knowledge -> The model is enriching the current approach with recalled facts.
    3. example-testing -> The model generates examples to test its current approach.
    4. uncertainty-estimation -> The model is stating its own uncertainty.
    5. backtracking -> The model decides to change its approach.

    The reasoning chain to analyze:
    {thinking_process}

    Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
    """)
    
    # Initialize token counts for each label
    label_token_counts = {label: 0 for label in labels}
    
    # Find all annotated sections
    pattern = r'\["([\w-]+)"\]([^\[]+)'
    matches = re.finditer(pattern, annotated_response)
    
    # Get tokens for the entire thinking process
    total_tokens = len(tokenizer.encode(thinking_process))
    
    # Count tokens for each label
    for match in matches:
        label = match.group(1)
        text = match.group(2).strip()
        if label != "end-section" and label in labels:
            # Count tokens in this section
            token_count = len(tokenizer.encode(text)) - 1  # Subtract 1 to account for potential BOS token
            label_token_counts[label] += token_count
    
    # Convert to fractions
    label_fractions = {
        label: count / total_tokens if total_tokens > 0 else 0 
        for label, count in label_token_counts.items()
    }
            
    return label_fractions, annotated_response

def generate_and_analyze(model, tokenizer, message, feature_vectors, label, labels, steer_mode="none"):
    input_ids = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    steer_positive = True if steer_mode == "positive" else False

    if(label == "uncertainty-estimation"):
        layers = [9,10,11]
        print(f"Setting layers for {label}: {layers}")
    elif(label == "example-testing"):
        layers = [8,9,10,11]
        print(f"Setting layers for {label}: {layers}")
    elif(label == "backtracking"):
        layers = [9,10,11]
        print(f"Setting layers for {label}: {layers}")
    elif(label == "adding-knowledge"):
        layers = [11,12,13]
        print(f"Setting layers for {label}: {layers}")

    output_ids = utils.custom_generate_with_projection_removal(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=500,
        label=label if steer_mode != "none" else "none",
        feature_vectors=feature_vectors,
        layers=layers,
        coefficient=1,
        steer_positive=steer_positive,
        show_progress=False
    )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract thinking process
    think_start = response.index("<think>") + len("<think>")
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)
    thinking_process = response[think_start:think_end].strip()
    
    label_fractions, annotated_response = get_label_counts(thinking_process, labels)
    
    return {
        "response": response,
        "thinking_process": thinking_process,
        "label_fractions": label_fractions,
        "annotated_response": annotated_response
    }

def plot_label_statistics(results, model_name):
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Get model identifier for file naming
    model_id = model_name.split('/')[-1].lower()
    
    # Use white background
    plt.style.use('seaborn-v0_8-white')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    labels_list = list(results.keys())
    x = np.arange(len(labels_list))
    width = 0.25
    
    # Calculate means as before
    original_means = []
    positive_means = []
    negative_means = []
    
    for label in labels_list:
        orig_fracs = [ex["original"]["label_fractions"].get(label, 0) for ex in results[label]]
        pos_fracs = [ex["positive"]["label_fractions"].get(label, 0) for ex in results[label]]
        neg_fracs = [ex["negative"]["label_fractions"].get(label, 0) for ex in results[label]]
        
        original_means.append(np.mean(orig_fracs))
        positive_means.append(np.mean(pos_fracs))
        negative_means.append(np.mean(neg_fracs))
    
    # Plot bars with black edges
    ax.bar(x - width, original_means, width, label='Original', color='#2E86C1', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x, positive_means, width, label='Positive Steering', color='#27AE60', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x + width, negative_means, width, label='Negative Steering', color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add percentage labels on top of bars
    def add_labels(positions, values):
        for pos, val in zip(positions, values):
            ax.text(pos, val, f'{val*100:.1f}%', ha='center', va='bottom', fontsize=14)
    
    add_labels(x - width, original_means)
    add_labels(x, positive_means)
    add_labels(x + width, negative_means)
    
    # Improve styling with larger font sizes and bold title
    ax.set_ylabel('Average Token Fraction (%)', fontsize=24, labelpad=10)
    ax.set_title('DeepSeek-R1-Distill-Llama-8B', fontsize=24, pad=20, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace('-', '\n') for label in labels_list], rotation=0, fontsize=24)
    ax.tick_params(axis='y', labelsize=16)
    
    # Convert y-axis to percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
    
    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Customize legend with larger font
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=20)
    
    # Show all spines (lines around the plot)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    plt.tight_layout()
    plt.savefig(f'figures/steering_results_{model_id}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %% Parameters
n_examples = 8
random.seed(42)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load model and vectors
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Can be changed to use different models
model_id = model_name.split('/')[-1].lower()

# %%
model, tokenizer, feature_vectors = load_model_and_vectors(model_name)

# %% Randomly sample evaluation examples
eval_indices = random.sample(range(len(eval_messages)), n_examples)

# Store results
labels = ['adding-knowledge', 'uncertainty-estimation', 'example-testing', 'backtracking']
results = {label: [] for label in labels}

# Evaluate each label
for label in labels:
    for idx in tqdm(eval_indices, desc=f"Processing examples for {label}"):
        message = eval_messages[idx]
        
        example_results = {
            "original": generate_and_analyze(model, tokenizer, message, feature_vectors, label, labels, "none"),
            "positive": generate_and_analyze(model, tokenizer, message, feature_vectors, label, labels, "positive"),
            "negative": generate_and_analyze(model, tokenizer, message, feature_vectors, label, labels, "negative")
        }
        
        results[label].append(example_results)

# Save results
with open(f'data/steering_evaluation_results_{model_id}.json', 'w') as f:
    json.dump(results, f, indent=2)

# %% Plot statistics
results = json.load(open(f'data/steering_evaluation_results_{model_id}.json'))
plot_label_statistics(results, model_name)

# %%
