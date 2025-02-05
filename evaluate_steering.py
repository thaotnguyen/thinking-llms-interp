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

# %%
def load_model_and_vectors():
    # ... existing code from evaluate_vectors.py ...
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", torch_dtype=torch.bfloat16)
    model = NNsight(model).to("cuda")
    
    mean_vectors_dict = torch.load("mean_vectors.pt")
    overall_mean = mean_vectors_dict['overall']['mean']
    feature_vectors = {}
    
    for label in mean_vectors_dict:
        if label != 'overall':
            feature_vectors[label] = mean_vectors_dict[label]['mean'] - overall_mean
            
    return model, tokenizer, feature_vectors

def custom_generate_with_projection_removal(model, tokenizer, input_ids, max_new_tokens, label, feature_vectors, steer_positive=False):
    generated_ids = input_ids.clone().cpu()
    if label in feature_vectors:
        # Move feature vectors to GPU only once, outside the loop
        feature_vector = feature_vectors[label].to("cuda").to(torch.bfloat16)
        normalized_features = feature_vector / torch.norm(feature_vector, dim=1, keepdim=True)
    else:
        normalized_features = None
            
    for k in range(max_new_tokens):
        # Clear cache at start of each iteration
        input_chunk = generated_ids.to("cuda")
        
        with torch.no_grad():  # Add this to reduce memory usage
            with model.trace(input_chunk) as trace:
                if normalized_features is not None:
                    for layer_idx in range(model.config.num_hidden_layers):
                        hidden_states = model.model.layers[layer_idx].output[0]
                        # Compute projections more efficiently
                        if steer_positive:
                            if layer_idx in range(4,16):
                                projection = torch.einsum('sh,h->s', hidden_states[0], normalized_features[layer_idx])
                                projection_vector = projection[-1:].unsqueeze(-1) * normalized_features[layer_idx]  # Outer product
                                model.model.layers[layer_idx].output[0][:, -1:] += 0.25 * projection_vector
                        else:
                            projection = torch.einsum('sh,h->s', hidden_states[0], normalized_features[layer_idx])
                            projection_vector = projection[-1:].unsqueeze(-1) * normalized_features[layer_idx]  # Outer product
                            model.model.layers[layer_idx].output[0][:, -1:] -= 0.25 * projection_vector

                        del hidden_states
                
                outputs = model.lm_head.output.save()
        
        next_token = outputs[:, -1, :].argmax(dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).cpu()], dim=1)

        # Explicitly delete tensors
        del trace, outputs, next_token, input_chunk
       
        torch.cuda.empty_cache()
        if k % 10 == 0:
            gc.collect()
    
    gc.collect()
    return generated_ids.cpu()

def get_label_counts(thinking_process, labels):
    # Get annotated version using chat function
    annotated_response = chat(f"""
    Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.

    Available labels:
    0. initializing -> The model is rephrasing the given task and states initial thoughts.
    1. deduction -> The model is performing a deduction step based on its current approach and assumptions.
    2. adding-knowledge -> The model is enriching the current approach with recalled facts.
    3. example-testing -> The model generates examples to test its current approach.
    3. uncertainty-estimation -> The model is stating its own uncertainty.
    4. backtracking -> The model decides to change its approach.

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
    output_ids = custom_generate_with_projection_removal(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=500,
        label=label if steer_mode != "none" else "none",
        feature_vectors=feature_vectors,
        steer_positive=steer_positive
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

def plot_label_statistics(results):
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    labels_list = list(results.keys())
    x = np.arange(len(labels_list))
    width = 0.25
    
    # For each label, get its scores in original, positive, and negative steering
    original_means = []
    positive_means = []
    negative_means = []
    
    for label in labels_list:
        # Calculate mean fraction for this label in its own steering experiments
        orig_fracs = [ex["original"]["label_fractions"].get(label, 0) for ex in results[label]]
        pos_fracs = [ex["positive"]["label_fractions"].get(label, 0) for ex in results[label]]
        neg_fracs = [ex["negative"]["label_fractions"].get(label, 0) for ex in results[label]]
        
        original_means.append(np.mean(orig_fracs))
        positive_means.append(np.mean(pos_fracs))
        negative_means.append(np.mean(neg_fracs))
    
    # Create bars
    ax.bar(x - width, original_means, width, label='Original')
    ax.bar(x, positive_means, width, label='Positive Steering')
    ax.bar(x + width, negative_means, width, label='Negative Steering')
    
    ax.set_ylabel('Average Token Fraction')
    ax.set_title('Label Token Fractions Under Different Steering Conditions')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('figures/steering_results.png', dpi=300)
    plt.show()
    plt.close()

# %% Parameters
n_examples = 10
random.seed(42)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load model and vectors
model, tokenizer, feature_vectors = load_model_and_vectors()

# %% Randomly sample evaluation examples
eval_indices = random.sample(range(len(eval_messages)), n_examples)

# Store results
labels = ['adding-knowledge', 'deduction', 'uncertainty-estimation', 'example-testing', 'backtracking']
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
with open('data/steering_evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# %% Plot statistics
plot_label_statistics(results)

# %%
