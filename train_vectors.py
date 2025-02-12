# %%
import dotenv
dotenv.load_dotenv(".env")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from nnsight import NNsight
from collections import defaultdict
from tqdm import tqdm
import random
import json
import os
import time  # Add this import at the top

# %% Load model

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # Can be changed to use different models
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model = NNsight(model).to("cuda")

model.generation_config.temperature=None
model.generation_config.top_p=None


mean_vectors = defaultdict(lambda: {
    'mean': torch.zeros(model.config.num_hidden_layers, model.config.hidden_size),
    'count': 0
})

# %% Define functions

def process_model_output(prompt_and_model_response_input_ids, model):
    """Get model output and layer activations"""
    start_time = time.time()
    layer_outputs = []
    with model.trace(prompt_and_model_response_input_ids):
        for layer_idx in range(model.config.num_hidden_layers):
            layer_outputs.append(model.model.layers[layer_idx].output[0].save())
    
    layer_outputs = torch.cat([x.value.cpu().detach().to(torch.float32) for x in layer_outputs], dim=0)
    elapsed = time.time() - start_time
    print(f"process_model_output took {elapsed:.2f} seconds")
    return layer_outputs

def get_label_positions(annotated_response: str, prompt_and_model_response_input_ids: list[int], tokenizer: AutoTokenizer):
    """Parse annotations and find token positions for each label"""
    start_time = time.time()
    label_positions = {}
    pattern = r'\["([\w-]+)"\]([^\[]+)'
    matches = re.finditer(pattern, annotated_response)
    
    for match in matches:
        labels = [label.strip() for label in match.group(1).strip('"').split(',')]
        if "end-section" in labels:
            continue

        # Get the text between the label and the next label or end-section
        text = match.group(2).strip()

        # Encode the text and remove the BOS token
        text_tokens: list[int] = tokenizer.encode(text)[1:]
        
        # Find the position of the text in the thinking tokens
        # Once found, we save the positions for each label
        for j in range(len(prompt_and_model_response_input_ids) - len(text_tokens) + 1):
            fragment = prompt_and_model_response_input_ids[j:j + len(text_tokens)]
            if fragment == text_tokens:
                for label in labels:
                    if label not in label_positions:
                        label_positions[label] = []
                    token_start = j
                    token_end = j + len(text_tokens)
                    label_positions[label].append((token_start, token_end))
                break
    
    elapsed = time.time() - start_time
    print(f"get_label_positions took {elapsed:.2f} seconds")
    return label_positions

def update_mean_vectors(mean_vectors, layer_outputs, label_positions, index):
    """Update mean vectors for overall and individual labels"""
    start_time = time.time()
    # Calculate overall thinking section boundaries
    all_positions = [pos for positions in label_positions.values() for pos in positions]
    if all_positions:
        min_pos = min(start for start, _ in all_positions)
        max_pos = max(end for _, end in all_positions)
        
        # Update overall mean
        overall_vectors = layer_outputs[:, min_pos:max_pos].mean(dim=1)
        current_count = mean_vectors['overall']['count']
        current_mean = mean_vectors['overall']['mean']
        mean_vectors['overall']['mean'] = current_mean + (overall_vectors - current_mean) / (current_count + 1)
        mean_vectors['overall']['count'] += 1

        if torch.isnan(mean_vectors['overall']['mean']).any():
            print(f"NaN in mean_vectors['overall']['mean'] at index {index}")

    elapsed = time.time() - start_time
    print(f"update_mean_vectors (overall) took {elapsed:.2f} seconds")
    start_time = time.time()

    # Process all labels at once
    for label, positions in label_positions.items():
        if not positions:
            continue
            
        # Stack all positions for this label
        starts = torch.tensor([pos[0]-1 for pos in positions])
        
        # Gather all vectors at once using index_select
        vectors = torch.index_select(layer_outputs, 1, starts.to(layer_outputs.device))
        mean_vector = vectors.mean(dim=1)
        
        # Update mean for this label
        current_count = mean_vectors[label]['count']
        current_mean = mean_vectors[label]['mean']
        mean_vectors[label]['mean'] = current_mean + (mean_vector - current_mean) / (current_count + len(positions))
        mean_vectors[label]['count'] += len(positions)
        
        if torch.isnan(mean_vectors[label]['mean']).any():
            print(f"NaN in mean_vectors['{label}']['mean'] at index {index}")

    elapsed = time.time() - start_time
    print(f"update_mean_vectors (individual labels) took {elapsed:.2f} seconds")


def calculate_next_token_frequencies(responses_data, tokenizer):
    """Calculate frequencies of next tokens for each label"""
    label_token_frequencies = defaultdict(lambda: defaultdict(int))
    
    for response in responses_data:
        annotated_text = response["annotated_response"]
        pattern = r'\["([\w-]+)"\](.*?)\["end-section"\]'
        matches = re.finditer(pattern, annotated_text, re.DOTALL)
        
        for match in matches:
            label = match.group(1)
            text = match.group(2).strip()
            # Get first token after label
            tokens = tokenizer.encode(text)[1:2]  # Just get the first token
            if tokens:
                next_token_text = tokenizer.decode(tokens)
                label_token_frequencies[label][next_token_text] += 1
    
    return label_token_frequencies

def should_skip_example(label, next_token, used_counts, max_examples=50):
    """Determine if we should skip this example based on frequency caps"""
    if used_counts[label][next_token] >= max_examples:
        return True
    return False

# %% Load data

save_every = 10
save_path = f"data/mean_vectors_{model_name.split('/')[-1].lower()}.pt"

annotated_responses_json_path = f"data/annotated_responses_{model_name.split('/')[-1].lower()}.json"
original_messages_json_path = f"data/base_responses_{model_name.split('/')[-1].lower()}.json"

tasks_json_path = "data/tasks.json"

if not os.path.exists(annotated_responses_json_path):
    raise FileNotFoundError(f"Annotated responses file not found at {annotated_responses_json_path}")
if not os.path.exists(original_messages_json_path):
    raise FileNotFoundError(f"Original messages file not found at {original_messages_json_path}")
if not os.path.exists(tasks_json_path):
    raise FileNotFoundError(f"Tasks file not found at {tasks_json_path}")

print(f"Loading existing annotated responses from {annotated_responses_json_path}")
with open(annotated_responses_json_path, 'r') as f:
    annotated_responses_data = json.load(f)["responses"]
random.shuffle(annotated_responses_data)

print(f"Loading existing original messages from {original_messages_json_path}")
with open(original_messages_json_path, 'r') as f:
    original_messages_data = json.load(f)["responses"]
random.shuffle(original_messages_data)

print(f"Loading existing tasks from {tasks_json_path}")
with open(tasks_json_path, 'r') as f:
    tasks_data = json.load(f)

# %% Calculate token frequencies for each label
label_token_frequencies = calculate_next_token_frequencies(annotated_responses_data, tokenizer)

# %%

# Track how many times we've used each token for each label
used_counts = defaultdict(lambda: defaultdict(int))

for i, annotated_response_data in tqdm(enumerate(annotated_responses_data), total=len(annotated_responses_data), desc="Processing annotated responses"):
    iter_start_time = time.time()
    response_uuid = annotated_response_data["response_uuid"]

    # Fetch the task and base response data
    task_data = next((task for task in tasks_data if task["task_uuid"] == annotated_response_data["task_uuid"]), None)
    base_response_data = next((msg for msg in original_messages_data if msg["response_uuid"] == response_uuid), None)

    # Build prompt message, appending the task prompt and the original response
    prompt_message = [task_data["prompt_message"]]
    prompt_message_input_ids = tokenizer.apply_chat_template(prompt_message, add_generation_prompt=True, return_tensors="pt").to("cuda")
    base_response_str = base_response_data["response_str"]
    if base_response_str.startswith("<think>"):
        # Remove the <think> tag prefix, since we already added it to the prompt_message_input_ids
        base_response_str = base_response_str[len("<think>"):]
    base_response_input_ids = tokenizer.encode(base_response_str, add_special_tokens=False, return_tensors="pt").to("cuda")

    prompt_and_model_response_input_ids = torch.cat([prompt_message_input_ids, base_response_input_ids], dim=1)

    # Get activations for each layer on this prompt
    layer_outputs = process_model_output(prompt_and_model_response_input_ids, model)

    # Get the positions for each label in the combined tokenized prompt and model response
    label_positions = get_label_positions(annotated_response_data["annotated_response"], prompt_and_model_response_input_ids[0].tolist(), tokenizer)
    
    # Check frequencies and skip if needed
    should_process = False
    for label, positions in label_positions.items():
        for start, end in positions:
            # Get the first token of the labeled sequence
            text = tokenizer.decode(prompt_and_model_response_input_ids[0][start:start+1])
            if label_token_frequencies[label][text] > 50:
                if not should_skip_example(label, text, used_counts):
                    should_process = True
            else:
                # Always process examples with frequency < 50
                should_process = True
            
            if should_process:
                used_counts[label][text] += 1

    if should_process:
        update_mean_vectors(mean_vectors, layer_outputs, label_positions, i)
        
        if i % save_every == 0:
            save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
            torch.save(save_dict, save_path)
            print(f"Current mean_vectors: {mean_vectors.keys()}. Saved...")
            print("Token usage statistics:")
            for label in used_counts:
                print(f"{label}: {dict(used_counts[label])}")
            iter_elapsed = time.time() - iter_start_time
            print(f"Iteration {i} took {iter_elapsed:.2f} seconds")

# Save final results
save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
torch.save(save_dict, save_path)
print("Saved final mean vectors")

# %%
