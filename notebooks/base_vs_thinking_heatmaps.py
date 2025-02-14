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

# %% Set model names
deepseek_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
original_model_name = "Qwen/Qwen2.5-14B"
# original_model_name = "Qwen/Qwen2.5-14B-Instruct"

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

# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
class RunningMeanStd:
    def __init__(self):
        """
        Calculates the running mean, std, and sum of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        self.mean = None
        self.var = None
        self.count = 0
        self.sum = None  # Add sum tracking

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd()
        if self.mean is not None:
            new_object.mean = self.mean.clone()
            new_object.var = self.var.clone()
            new_object.count = float(self.count)
            new_object.sum = self.sum.clone()  # Copy sum
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.
        """
        self.update_from_moments(other.mean, other.var, other.count)
        if self.sum is None:
            self.sum = other.sum.clone()
        else:
            self.sum += other.sum

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = arr.double().mean(dim=0)
        batch_var = arr.double().var(dim=0)
        batch_count = arr.shape[0]
        batch_sum = arr.double().sum(dim=0)  # Calculate batch sum
        
        if batch_count == 0:
            return
        if self.mean is None:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
            self.sum = batch_sum  # Initialize sum
        else:
            self.update_from_moments(batch_mean, batch_var, batch_count)
            self.sum += batch_sum  # Update sum

    def update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float
    ) -> None:
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def compute(
        self, return_dict=False
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor] | dict[str, float]:
        """
        Compute the running mean, variance, count and sum

        Returns:
            mean, var, count, sum if return_dict=False
            dict with keys 'mean', 'var', 'count', 'sum' if return_dict=True
        """
        if return_dict:
            return {
                "mean": self.mean.item(),
                "var": self.var.item(),
                "count": self.count,
                "sum": self.sum.item(),
            }
        return self.mean, self.var, self.count, self.sum

# %%

responses_to_collect = 1000
seed = 42
random.seed(seed)

all_response_uuids = [response["response_uuid"] for response in annotated_responses_data]

print(f"Collecting {responses_to_collect} responses from {len(all_response_uuids)} responses")

# randomly sample responses_to_collect response uuids from all_response_uuids
response_uuids_to_collect = random.sample(all_response_uuids, responses_to_collect)

kl_stats_per_token = {}
kl_stats_per_token_pair = {}
kl_stats_per_next_token = {}
kl_stats_per_label = {}
kl_stats_per_next_token_and_label = {}  # New: tracks stats per (current_token, next_token, next_token_label)

for response_uuid in tqdm(response_uuids_to_collect):
    # Clear CUDA cache at the start of each iteration
    torch.cuda.empty_cache()
    
    model_input = prepare_model_input(response_uuid=response_uuid, annotated_responses_data=annotated_responses_data, tasks_data=tasks_data, original_messages_data=original_messages_data, tokenizer=deepseek_tokenizer)

    # Move input tensors to CPU after use
    deepseek_logits, original_logits = get_logits(
        prompt_and_response_ids=model_input['prompt_and_response_ids'],
        thinking_start_token_index=model_input['thinking_start_token_index'],
        thinking_end_token_index=model_input['thinking_end_token_index']
    )

    kl_divergence = calculate_kl_divergence(deepseek_logits, original_logits)

    thinking_tokens = deepseek_tokenizer.batch_decode(model_input['thinking_token_ids'][0])

    # Add ticks and replace new lines with \n
    thinking_tokens = [token.replace('\n', '\\n') for token in thinking_tokens]
    thinking_tokens = [f"`{token}`" for token in thinking_tokens]

    assert len(thinking_tokens) == len(kl_divergence)

    # Update stats for individual tokens
    for token, kl_divergence_value in zip(thinking_tokens, kl_divergence):
        if token not in kl_stats_per_token:
            kl_stats_per_token[token] = RunningMeanStd()
        # Convert to float64 before passing to update
        kl_stats_per_token[token].update(kl_divergence_value.to(torch.float64).unsqueeze(0))

    # Update stats for token pairs and next tokens
    for i in range(len(thinking_tokens) - 1):
        current_token = thinking_tokens[i]
        next_token = thinking_tokens[i + 1]
        current_kl = kl_divergence[i]
        next_kl = kl_divergence[i + 1]
        next_token_label = model_input['token_to_label'].get(i + 1)  # Get label of next token

        # Update token pair and label stats (using current token's KL)
        if next_token_label is not None:  # Only track if next token has a label
            triple_key = (current_token, next_token, next_token_label)
            if triple_key not in kl_stats_per_next_token_and_label:
                kl_stats_per_next_token_and_label[triple_key] = RunningMeanStd()
            # Convert to float64 before passing to update
            kl_stats_per_next_token_and_label[triple_key].update(current_kl.to(torch.float64).unsqueeze(0))

        # Update token pair stats (using current token's KL)
        pair_key = (current_token, next_token)
        if pair_key not in kl_stats_per_token_pair:
            kl_stats_per_token_pair[pair_key] = RunningMeanStd()
        # Convert to float64 before passing to update
        kl_stats_per_token_pair[pair_key].update(current_kl.to(torch.float64).unsqueeze(0))

        # Update next token stats (using previous token's KL)
        if next_token not in kl_stats_per_next_token:
            kl_stats_per_next_token[next_token] = RunningMeanStd()
        # Convert to float64 before passing to update
        kl_stats_per_next_token[next_token].update(current_kl.to(torch.float64).unsqueeze(0))

    # Get labels for each token
    token_labels = [model_input['token_to_label'].get(i) for i in range(len(thinking_tokens))]

    # Update stats for labels
    for label, kl_divergence_value in zip(token_labels, kl_divergence):
        if label is not None:  # Only track tokens that have a label
            if label not in kl_stats_per_label:
                kl_stats_per_label[label] = RunningMeanStd()
            kl_stats_per_label[label].update(kl_divergence_value.to(torch.float64).unsqueeze(0))

# %% Save KL stats to disk

# Create directory if it doesn't exist
os.makedirs("../data/kl_stats", exist_ok=True)

# Get model ID for filenames
deepseek_model_id = deepseek_model_name.split('/')[-1].lower()
original_model_id = original_model_name.split('/')[-1].lower()

# Save each stats dictionary
stats_to_save = {
    "token": kl_stats_per_token,
    "token_pair": kl_stats_per_token_pair,
    "next_token": kl_stats_per_next_token,
    "label": kl_stats_per_label,
    "next_token_and_label": kl_stats_per_next_token_and_label  # Add new stats
}

for stats_type, stats_dict in stats_to_save.items():
    output_path = f"../data/kl_stats/kl_stats_per_{stats_type}_{deepseek_model_id}_{original_model_id}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(stats_dict, f)
    print(f"Saved {stats_type} KL stats to {output_path}")

# %% Add visualization for token pairs and next tokens

def plot_top_stats(stats_dict, title, n=20, pair_keys=False, metric='mean', top_count_pct=0.1):
    """
    Plot statistics for tokens/pairs
    
    Args:
        stats_dict: Dictionary of statistics
        title: Title for the plot
        n: Number of top items to show
        pair_keys: Whether the keys are pairs/tuples
        metric: 'mean' or 'sum' to determine which metric to sort by
        top_count_pct: Filter to keep only top percentage by count (0.1 = top 10%)
    """
    # Create a list of (key, value, count) tuples
    values = []
    for key, stats in stats_dict.items():
        mean, _, count, sum_val = stats.compute()
        value = sum_val.item() if metric == 'sum' else mean.item()
        values.append((key, value, count))
    
    # First filter by count - keep only top percentage
    values.sort(key=lambda x: x[2], reverse=True)
    cutoff_idx = max(1, int(len(values) * top_count_pct))
    values = values[:cutoff_idx]
    
    # Then sort by the chosen metric
    values.sort(key=lambda x: x[1], reverse=True)
    top_values = values[:n]
    
    # Create lists for plotting
    if pair_keys:
        if len(top_values[0][0]) == 2:
            keys = [f"{t[0][0]}\n{t[0][1]}" for t in top_values]
        else:
            keys = [f"{t[0][0]}\n{t[0][1]}\n{t[0][2]}" for t in top_values]
    else:
        keys = [t[0] for t in top_values]
    metric_values = [t[1] for t in top_values]
    
    # Create the plot
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(keys)), metric_values)
    plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
    plt.title(f'{title} by {metric.capitalize()} KL Divergence (Top {n}, from top {int(top_count_pct*100)}% by count)')
    plt.xlabel('Token' if not pair_keys else 'Token Pair')
    plt.ylabel(f'{metric.capitalize()} KL Divergence')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../plots/{title.replace(' ', '_').lower()}_{metric}_{deepseek_model_id}_{original_model_id}.png")
    
    # Print the list
    print(f"\nTop {n} {title} by {metric} (filtered to top {int(top_count_pct*100)}% by count):")
    for key, value, count in top_values:
        if pair_keys:
            if len(key) == 2:
                print(f"({key[0]}, {key[1]}): {value:.4f} (count: {count})")
            else:
                print(f"({key[0]}, {key[1]}, {key[2]}): {value:.4f} (count: {count})")
        else:
            print(f"{key}: {value:.4f} (count: {count})")

# Plot all statistics with both mean and sum
plot_top_stats(kl_stats_per_token, "Tokens", metric='mean')
plot_top_stats(kl_stats_per_token, "Tokens", metric='sum')

plot_top_stats(kl_stats_per_token_pair, "Token Pairs", pair_keys=True, metric='mean')
plot_top_stats(kl_stats_per_token_pair, "Token Pairs", pair_keys=True, metric='sum')

plot_top_stats(kl_stats_per_next_token, "Next Tokens (Previous Token's KL)", metric='mean')
plot_top_stats(kl_stats_per_next_token, "Next Tokens (Previous Token's KL)", metric='sum')

plot_top_stats(kl_stats_per_next_token_and_label, "Next Tokens and Labels", pair_keys=True, metric='mean')
plot_top_stats(kl_stats_per_next_token_and_label, "Next Tokens and Labels", pair_keys=True, metric='sum')

# %% Add visualization for label statistics

def plot_label_stats(stats_dict, metric='mean', top_count_pct=0.1):
    """
    Plot statistics for labels
    
    Args:
        stats_dict: Dictionary of statistics
        metric: 'mean' or 'sum' to determine which metric to sort by
        top_count_pct: Filter to keep only top percentage by count (0.1 = top 10%)
    """
    values = []
    for label, stats in stats_dict.items():
        mean, var, count, sum_val = stats.compute()
        value = sum_val.item() if metric == 'sum' else mean.item()
        std_dev = torch.sqrt(var).item()
        values.append((label, value, count, std_dev))
    
    # First filter by count - keep only top percentage
    values.sort(key=lambda x: x[2], reverse=True)
    cutoff_idx = max(1, int(len(values) * top_count_pct))
    values = values[:cutoff_idx]
    
    # Then sort by the chosen metric
    values.sort(key=lambda x: x[1], reverse=True)
    
    # Create lists for plotting
    labels = [t[0] for t in values]
    metric_values = [t[1] for t in values]
    counts = [t[2] for t in values]
    std_devs = [t[3] for t in values]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot bars with error bars
    bars = plt.bar(range(len(labels)), metric_values, yerr=std_devs, capsize=5)
    
    # Add count annotations on top of each bar
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'n={int(count)}',
                ha='center', va='bottom')
    
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.title(f'KL Divergence by Label ({metric.capitalize()}, from top {int(top_count_pct*100)}% by count)')
    plt.xlabel('Label')
    plt.ylabel(f'{metric.capitalize()} KL Divergence')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../plots/kl_divergence_by_label_{metric}_{deepseek_model_id}_{original_model_id}.png")
    
    # Print detailed statistics
    print(f"\nLabel Statistics ({metric}, filtered to top {int(top_count_pct*100)}% by count):")
    for label, value, count, std_dev in values:
        print(f"{label:20} {value:.4f} ± {std_dev:.4f} (count: {count})")

# Plot label statistics with both metrics
plot_label_stats(kl_stats_per_label, metric='mean')
plot_label_stats(kl_stats_per_label, metric='sum')

# %% Add stacked bar plot for token pairs by label

def plot_stacked_token_pairs_by_label(
    stats_dict, 
    n=20, 
    metric='sum', 
    top_count_pct=0.1,
    ignore_categories=["initializing", "deduction"]  # New parameter
):
    """
    Create a stacked bar plot showing token pairs with different colors for each label
    
    Args:
        stats_dict: Dictionary of statistics
        n: Number of top pairs to show
        metric: Either 'sum' or 'mean' to determine which metric to use for plotting
        top_count_pct: Filter to keep only top percentage by count (0.1 = top 10%)
        ignore_categories: List of categories to ignore when calculating statistics
    """
    # First, organize data by token pairs
    pair_data = {}
    pair_counts = {}  # Track total counts for each pair
    for (current_token, next_token, label), stats in stats_dict.items():
        # Skip if label is in ignore_categories
        if label in ignore_categories:
            continue
            
        pair_key = (current_token, next_token)
        if pair_key not in pair_data:
            pair_data[pair_key] = {}
            pair_counts[pair_key] = 0
        mean, _, count, sum_val = stats.compute()
        value = sum_val.item() if metric == 'sum' else mean.item()
        pair_data[pair_key][label] = value
        pair_counts[pair_key] += count
    
    # First filter by count - keep only top percentage of pairs
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    cutoff_idx = max(1, int(len(sorted_pairs) * top_count_pct))
    top_pairs_by_count = sorted_pairs[:cutoff_idx]
    filtered_pairs = {pair: count for pair, count in top_pairs_by_count}
    
    # Calculate total for filtered pairs and sort
    pair_totals = {
        pair: sum(label_values.values()) 
        for pair, label_values in pair_data.items() 
        if pair in filtered_pairs
    }
    top_pairs = sorted(pair_totals.items(), key=lambda x: x[1], reverse=True)[:n]
    
    # Prepare data for plotting
    pairs = []
    for p, _ in top_pairs:
        current_token = p[0].replace('\n', '\\n')  # Escape newlines in tokens
        next_token = p[1].replace('\n', '\\n')
        pairs.append(f"{current_token}\n{next_token}")
    
    # Create data arrays for each label
    label_data = {label: [] for label in set(label for pair_dict in pair_data.values() for label in pair_dict.keys())}
    for pair, _ in top_pairs:
        pair_dict = pair_data[pair]
        for label in label_data:
            label_data[label].append(pair_dict.get(label, 0))
    
    # Create the stacked bar plot
    plt.figure(figsize=(20, 10))  # Make figure wider
    
    # Use a colormap for different labels
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(label_data)))
    
    bottom = np.zeros(len(pairs))
    bars = []
    for i, (label, values) in enumerate(label_data.items()):
        bar = plt.bar(range(len(pairs)), values, bottom=bottom, 
                     label=label, color=colors[i])
        bars.append(bar)
        bottom += np.array(values)
    
    # Adjust the text sizes
    plt.xticks(range(len(pairs)), pairs, rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=16)
    plt.title(f'Stacked KL Divergence by Token Pairs and Labels ({metric.capitalize()}, from top {int(top_count_pct*100)}% by count)', 
              fontsize=16)
    plt.xlabel('Token Pair', fontsize=16)
    plt.ylabel(f'{metric.capitalize()} KL Divergence', fontsize=16)
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', fontsize=16)
    
    plt.subplots_adjust(bottom=0.2)  # Add more space at bottom for labels
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../plots/stacked_kl_divergence_by_token_pairs_and_labels_{metric}_{deepseek_model_id}_{original_model_id}.png", 
                bbox_inches='tight', dpi=300)
    
    # Print the values with counts
    print(f"\nTop token pairs by label contributions ({metric}, filtered to top {int(top_count_pct*100)}% by count):")
    for pair, total in top_pairs:
        print(f"\n{pair[0]}, {pair[1]}: {total:.4f} total (count: {filtered_pairs[pair]})")
        pair_labels = pair_data[pair]
        for label, value in sorted(pair_labels.items(), key=lambda x: x[1], reverse=True):
            # Get the count from the original stats dictionary
            count = stats_dict[(pair[0], pair[1], label)].count
            print(f"  {label}: {value:.4f} (count: {int(count)})")

# Create the stacked bar plots for both metrics with default ignored categories
plot_stacked_token_pairs_by_label(kl_stats_per_next_token_and_label, metric='sum')
plot_stacked_token_pairs_by_label(kl_stats_per_next_token_and_label, metric='mean')

# Create plots without ignoring any categories
plot_stacked_token_pairs_by_label(kl_stats_per_next_token_and_label, metric='sum', ignore_categories=[])
plot_stacked_token_pairs_by_label(kl_stats_per_next_token_and_label, metric='mean', ignore_categories=[])

# %%
