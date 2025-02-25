# %%
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import random
from tqdm import tqdm
from typing import List, Dict, Any
from tiny_dashboard.visualization_utils import activation_visualization
from IPython.display import HTML, display
import pickle
import numpy as np
from deepseek_steering.running_mean import RunningMeanStd

# %% Set experiment parameters
EXPERIMENT_PARAMS = {
    # Model parameters
    "deepseek_model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "original_model_name": "Qwen/Qwen2.5-14B",
    # "original_model_name": "Qwen/Qwen2.5-14B-Instruct",
    
    # Analysis parameters
    "responses_to_analyze": 1000,  # Number of responses to analyze
    "top_tokens_to_show": 30,     # Number of top tokens to display
    "seed": 42,                   # Random seed
    
    # Token filtering
    "tokens_to_exclude": ["\n", "I", ":", "'m", ".\n"]
}

# Set random seed
random.seed(EXPERIMENT_PARAMS["seed"])

# %% Set model names
deepseek_model_name = EXPERIMENT_PARAMS["deepseek_model_name"]
original_model_name = EXPERIMENT_PARAMS["original_model_name"]
# original_model_name = "Qwen/Qwen2.5-14B-Instruct"

tokens_to_exclude = EXPERIMENT_PARAMS["tokens_to_exclude"]

# %%

seed = EXPERIMENT_PARAMS["seed"]
random.seed(seed)

# %% Load models

deepseek_tokenizer = AutoTokenizer.from_pretrained(deepseek_model_name)
deepseek_model = AutoModelForCausalLM.from_pretrained(
    deepseek_model_name,
    torch_dtype=torch.float16,  # Use float16 for memory efficiency
    device_map="auto"  # Automatically handle device placement
)

original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
original_model = AutoModelForCausalLM.from_pretrained(
    original_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# %% Load data

annotated_responses_json_path = f"../data/annotated_responses_{deepseek_model_name.split('/')[-1].lower()}.json"
original_messages_json_path = f"../data/base_responses_{deepseek_model_name.split('/')[-1].lower()}.json"

tasks_json_path = "../data/tasks.json"

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

# %% Prepare model input

available_labels = ["initializing", "deduction", "adding-knowledge", "example-testing", "uncertainty-estimation", "backtracking"]

def prepare_model_input(
    response_uuid: str,
    annotated_responses_data: List[Dict[str, Any]],
    tasks_data: List[Dict[str, Any]],
    original_messages_data: List[Dict[str, Any]],
    tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    """
    Prepare model input for a given response UUID.
    Returns the tokenized input ready for the model.
    
    Returns:
        Dict with keys:
            'prompt_and_response_ids': Tensor of shape (1, sequence_length)
            'annotated_response': str
    """
    # Fetch the relevant response data
    annotated_response_data = next((r for r in annotated_responses_data if r["response_uuid"] == response_uuid), None)
    if not annotated_response_data:
        raise ValueError(f"Could not find annotated response data for UUID {response_uuid}")
    
    task_data = next((t for t in tasks_data if t["task_uuid"] == annotated_response_data["task_uuid"]), None)
    if not task_data:
        raise ValueError(f"Could not find task data for UUID {annotated_response_data['task_uuid']}")
    
    base_response_data = next((m for m in original_messages_data if m["response_uuid"] == response_uuid), None)
    if not base_response_data:
        raise ValueError(f"Could not find base response data for UUID {response_uuid}")
    
    # Build prompt message
    prompt_message = [task_data["prompt_message"]]
    prompt_message_input_ids = tokenizer.apply_chat_template(
        conversation=prompt_message,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # Process base response
    base_response_str = base_response_data["response_str"]
    if base_response_str.startswith("<think>"):
        base_response_str = base_response_str[len("<think>"):]
    base_response_input_ids = tokenizer.encode(
        text=base_response_str,
        add_special_tokens=False,
        return_tensors="pt"
    )

    prompt_and_response_ids = torch.cat(
        tensors=[prompt_message_input_ids, base_response_input_ids],
        dim=1
    )

    # Find start and end positions of thinking process (-1 if not found)
    thinking_start_token_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
    thinking_end_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]

    prompt_and_response_ids_list = prompt_and_response_ids.tolist()[0]
    thinking_start_token_index = next((i + 1 for i, token in enumerate(prompt_and_response_ids_list) if token == thinking_start_token_id), -1)
    thinking_end_token_index = next((i for i, token in enumerate(prompt_and_response_ids_list) if token == thinking_end_token_id), -1)

    thinking_token_ids = prompt_and_response_ids[:, thinking_start_token_index:thinking_end_token_index]

    # Build token position to label mapping
    token_to_label = {}
    # Remove end-section markers from annotated response
    annotated_response = annotated_response_data["annotated_response"].replace('[\"end-section\"]', '')
    current_pos = 0
    current_label = None
    last_token_index = None  # Track the last token index we processed

    # Process tokens from thinking start to end
    i = 0
    while i < thinking_token_ids.size(1):
        current_token = deepseek_tokenizer.decode(thinking_token_ids[0, i])
        next_token = deepseek_tokenizer.decode(thinking_token_ids[0, i + 1]) if i + 1 < thinking_token_ids.size(1) else None
        
        # Search for labels and tokens from current position
        while current_pos < len(annotated_response):
            # Check for label markers, accounting for whitespace
            if annotated_response[current_pos:].strip().startswith('["'):
                # Find the actual start of the label marker after current_pos
                label_start = annotated_response.find('["', current_pos)
                label_end = annotated_response.find('"]', label_start)
                if label_end != -1:
                    # get the new label
                    label = annotated_response[label_start + 2:label_end]
                    current_label = label

                    if current_label not in available_labels:
                        current_label = None

                    current_pos = label_end + 2
                    # print(f"New label: {current_label} starting at {current_pos}: `{annotated_response[current_pos:current_pos+5]}`")

                    # Assign the new label to the last token processed
                    if last_token_index is not None:
                        token_to_label[last_token_index] = current_label
                        last_token_index = None

                    continue
            
            # Try to find current token
            found_current = annotated_response.find(current_token, current_pos)
            found_next = -1 if next_token is None else annotated_response.find(next_token, current_pos)
            
            # If next token is found before current token
            if found_next != -1 and (found_current == -1 or found_next < found_current):
                token_to_label[i] = current_label
                if i + 1 < thinking_token_ids.size(1):
                    token_to_label[i + 1] = current_label
                current_pos = found_next + len(next_token)
                last_token_index = i + 1
                # print(f"Assigning  label {current_label} to `{current_token}` and next token `{next_token}`. We are now at {current_pos}: `{annotated_response[current_pos:current_pos+5]}`")
                i += 2  # Skip the next token since we've processed it
                break
            
            # If current token is found
            elif found_current != -1:
                token_to_label[i] = current_label
                current_pos = found_current + len(current_token)
                # print(f"Assigning label {current_label} to `{current_token}`. We are now at {current_pos}: `{annotated_response[current_pos:current_pos+5]}`")
                last_token_index = i
                i += 1  # Move to next token
                break
            
            # If neither token is found, move to next character
            else:
                current_pos += 1
                
        # If we've reached the end of annotated response
        if current_pos >= len(annotated_response):
            token_to_label[i] = None
            i += 1  # Move to next token
    
    return {
        'prompt_and_response_ids': prompt_and_response_ids,
        'annotated_response': annotated_response_data["annotated_response"],
        'thinking_start_token_index': thinking_start_token_index,
        'thinking_end_token_index': thinking_end_token_index,
        'thinking_token_ids': thinking_token_ids,
        'token_to_label': token_to_label
    }


# %% Feed the input to both models and get the logits for all tokens

def get_logits(prompt_and_response_ids, thinking_start_token_index, thinking_end_token_index):
    # Clear CUDA cache before processing
    torch.cuda.empty_cache()
    
    # Get logits from both models
    with torch.no_grad():
        # DeepSeek model logits
        deepseek_outputs = deepseek_model(
            input_ids=prompt_and_response_ids.to(deepseek_model.device)
        )
        deepseek_logits = deepseek_outputs.logits.cpu()  # Move to CPU immediately
        del deepseek_outputs  # Free memory
        torch.cuda.empty_cache()
        
        # Original model logits
        original_outputs = original_model(
            input_ids=prompt_and_response_ids.to(original_model.device)
        )
        original_logits = original_outputs.logits.cpu()  # Move to CPU immediately
        del original_outputs  # Free memory
        torch.cuda.empty_cache()

    # Assert both logits have the same shape
    assert deepseek_logits.shape == original_logits.shape

    # Return only the logits for the thinking tokens
    deepseek_logits = deepseek_logits[0, thinking_start_token_index:thinking_end_token_index]
    original_logits = original_logits[0, thinking_start_token_index:thinking_end_token_index]

    return deepseek_logits, original_logits

# %% Calculate the KL divergence between the logits

def calculate_kl_divergence(p_logits, q_logits):
    """
    Calculate KL divergence between two distributions given their logits.
    Uses PyTorch's built-in KL divergence function with log_softmax.
    """
    # Convert logits directly to log probabilities
    p_log = torch.nn.functional.log_softmax(p_logits, dim=-1)
    q_log = torch.nn.functional.log_softmax(q_logits, dim=-1)
    
    # Calculate KL divergence using PyTorch's function
    kl_div = torch.nn.functional.kl_div(
        p_log,      # input in log-space
        q_log,      # target in log-space
        reduction='none',
        log_target=True
    )

    # Sum over vocabulary dimension
    kl_div = kl_div.sum(dim=-1)
    
    kl_div = kl_div.squeeze()

    # Convert to float32 and ensure positive values
    kl_div = kl_div.float()
    kl_div = torch.clamp(kl_div, min=0.0)

    return kl_div

# %% Pick a random response uuid and visualize the heatmap

response_uuid = random.choice(annotated_responses_data)["response_uuid"]

model_input = prepare_model_input(
    response_uuid=response_uuid,
    annotated_responses_data=annotated_responses_data,
    tasks_data=tasks_data,
    original_messages_data=original_messages_data,
    tokenizer=deepseek_tokenizer
)

print(f"\nResponse UUID: {response_uuid}")
print(f"Prompt and response IDs: `{deepseek_tokenizer.decode(model_input['prompt_and_response_ids'][0], skip_special_tokens=False)}`")
print(f"Thinking response: `{deepseek_tokenizer.decode(model_input['thinking_token_ids'][0], skip_special_tokens=False)}`")

for i, token in enumerate(model_input['thinking_token_ids'][0]):
    print(f"{i}: {deepseek_tokenizer.decode(token)} -> {model_input['token_to_label'][i]}")

deepseek_logits, original_logits = get_logits(
    prompt_and_response_ids=model_input['prompt_and_response_ids'],
    thinking_start_token_index=model_input['thinking_start_token_index'],
    thinking_end_token_index=model_input['thinking_end_token_index']
)

# Calculate KL divergence for each position
kl_divergence = calculate_kl_divergence(deepseek_logits, original_logits)

# Get the tokens for visualization
thinking_tokens = deepseek_tokenizer.convert_ids_to_tokens(
    model_input['thinking_token_ids'][0]
)

html = activation_visualization(
    thinking_tokens,
    kl_divergence,
    tokenizer=deepseek_tokenizer,
    title="KL Divergence between Models",
    relative_normalization=False,
)
display(HTML(html))

# %%

def get_kl_stats_path(deepseek_model_name: str, original_model_name: str) -> str:
    """Get the path to the KL stats file."""
    return f"../data/kl_stats/normalized_kl_scores_{deepseek_model_name.split('/')[-1].lower()}_{original_model_name.split('/')[-1].lower()}.pkl"

def save_kl_stats(stats: dict, experiment_params: dict, deepseek_model_name: str, original_model_name: str) -> None:
    """Save KL stats and experiment parameters to a pickle file."""
    output_path = get_kl_stats_path(deepseek_model_name, original_model_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_to_save = {
        'stats': stats,
        'experiment_params': experiment_params
    }
    with open(output_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"\nSaved normalized KL scores and parameters to {output_path}")

def load_kl_stats(deepseek_model_name: str, original_model_name: str) -> dict:
    """Load KL stats from a pickle file if it exists."""
    stats_path = get_kl_stats_path(deepseek_model_name, original_model_name)
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            data = pickle.load(f)
            # For backwards compatibility with old saved files
            if isinstance(data, dict) and 'stats' in data:
                return data
            return {'stats': data, 'experiment_params': None}
    return None

def collect_kl_stats(
    experiment_params: dict,
    annotated_responses_data: list,
    tasks_data: list,
    original_messages_data: list,
    deepseek_tokenizer,
) -> dict:
    """Collect KL divergence statistics across responses."""
    # Try to load existing stats first
    existing_data = load_kl_stats(
        experiment_params["deepseek_model_name"], 
        experiment_params["original_model_name"]
    )
    if existing_data is not None:
        # Check if parameters match
        if (existing_data.get('experiment_params') == experiment_params):
            print(f"Loaded existing KL stats for {len(existing_data['stats'])} tokens")
            return existing_data['stats']
        else:
            print("Found existing stats but parameters don't match. Recomputing...")

    # Dictionary to store KL divergence sums and counts for next tokens
    next_token_stats = {}

    all_response_uuids = [response["response_uuid"] for response in annotated_responses_data]
    response_uuids_to_analyze = random.sample(
        all_response_uuids, 
        experiment_params["responses_to_analyze"]
    )

    print(f"Analyzing {experiment_params['responses_to_analyze']} responses from {len(all_response_uuids)} total responses")

    for response_uuid in tqdm(response_uuids_to_analyze):
        # Clear CUDA cache at the start of each iteration
        torch.cuda.empty_cache()
        
        model_input = prepare_model_input(
            response_uuid=response_uuid, 
            annotated_responses_data=annotated_responses_data, 
            tasks_data=tasks_data, 
            original_messages_data=original_messages_data, 
            tokenizer=deepseek_tokenizer
        )

        deepseek_logits, original_logits = get_logits(
            prompt_and_response_ids=model_input['prompt_and_response_ids'],
            thinking_start_token_index=model_input['thinking_start_token_index'],
            thinking_end_token_index=model_input['thinking_end_token_index']
        )

        kl_divergence = calculate_kl_divergence(deepseek_logits, original_logits)
        thinking_token_ids = model_input['thinking_token_ids'][0]
        
        # Process each token pair in the response
        response_kl_stats = {}
        for i in range(len(thinking_token_ids) - 1):
            # Get the next token and normalize it
            next_token = deepseek_tokenizer.decode(thinking_token_ids[i + 1]).strip()

            if next_token in tokens_to_exclude:
                continue

            current_kl = kl_divergence[i].item()

            if next_token not in response_kl_stats:
                response_kl_stats[next_token] = {
                    'kl_sum': 0.0,
                    'total_occurrences': 0
                }
            
            response_kl_stats[next_token]['kl_sum'] += current_kl
            response_kl_stats[next_token]['total_occurrences'] += 1

        for token, stats in response_kl_stats.items():
            if token not in next_token_stats:
                next_token_stats[token] = {
                    'sum_of_avg_kl_div': 0.0,
                    'response_uuids': set()
                }
            next_token_stats[token]['sum_of_avg_kl_div'] += stats['kl_sum'] / stats['total_occurrences']
            next_token_stats[token]['response_uuids'].add(response_uuid)

    # Calculate normalized KL divergence for each token
    for token, stats in next_token_stats.items():
        normalized_kl = stats['sum_of_avg_kl_div'] / len(response_uuids_to_analyze)
        next_token_stats[token]['normalized_kl'] = normalized_kl

    # Save the collected stats with parameters
    save_kl_stats(
        next_token_stats, 
        experiment_params,
        experiment_params["deepseek_model_name"], 
        experiment_params["original_model_name"]
    )

    return next_token_stats

# Replace the analysis section with:
next_token_stats = collect_kl_stats(
    experiment_params=EXPERIMENT_PARAMS,
    annotated_responses_data=annotated_responses_data,
    tasks_data=tasks_data,
    original_messages_data=original_messages_data,
    deepseek_tokenizer=deepseek_tokenizer,
)

# Sort tokens by normalized KL divergence
sorted_tokens = sorted(
    next_token_stats.items(),
    key=lambda x: x[1]['normalized_kl'],
    reverse=True
)

def get_display_token(token):
    token = token.replace("\n", "\\n")
    token = f"`{token}`"
    return token

# Display results
print(f"\nTop {EXPERIMENT_PARAMS['top_tokens_to_show']} tokens by normalized KL divergence across {EXPERIMENT_PARAMS['responses_to_analyze']} responses:")
print("\nFormat: Token: Normalized KL (Responses, Total Occurrences)")
print("-" * 60)
for token, stats in sorted_tokens[:EXPERIMENT_PARAMS['top_tokens_to_show']]:
    print(f"{get_display_token(token)}: {stats['normalized_kl']:.4f} ({len(stats['response_uuids'])})")

# Visualize results
plt.figure(figsize=(15, 8))
tokens = [get_display_token(t[0]) for t in sorted_tokens[:EXPERIMENT_PARAMS['top_tokens_to_show']]]
scores = [t[1]['normalized_kl'] for t in sorted_tokens[:EXPERIMENT_PARAMS['top_tokens_to_show']]]

plt.bar(range(len(tokens)), scores)
plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
plt.title(f'Top {EXPERIMENT_PARAMS["top_tokens_to_show"]} Tokens by Normalized KL Divergence of previous token')
plt.xlabel('Token')
plt.ylabel('Normalized KL Divergence across all responses')
plt.tight_layout()
plt.show()

# %%
