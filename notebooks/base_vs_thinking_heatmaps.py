# %%
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import random
from tqdm import tqdm
from typing import List, Dict, Any
from tiny_dashboard.visualization_utils import activation_visualization
from IPython.display import HTML, display

# %% Set model names
deepseek_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
original_model_name = "Qwen/Qwen2.5-14B"

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

# %% Pick a random response uuid
response_uuid = random.choice(annotated_responses_data)["response_uuid"]

# %% Prepare model input

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
    
    return {
        'prompt_and_response_ids': prompt_and_response_ids,
        'annotated_response': annotated_response_data["annotated_response"],
        'thinking_start_token_index': thinking_start_token_index,
        'thinking_end_token_index': thinking_end_token_index,
        'thinking_token_ids': thinking_token_ids
    }

model_input = prepare_model_input(
    response_uuid=response_uuid,
    annotated_responses_data=annotated_responses_data,
    tasks_data=tasks_data,
    original_messages_data=original_messages_data,
    tokenizer=deepseek_tokenizer
)

print(f"Prompt and response IDs: `{deepseek_tokenizer.decode(model_input['prompt_and_response_ids'][0])}`")
print(f"Thinking response: `{deepseek_tokenizer.decode(model_input['thinking_token_ids'][0])}`")

# %% Feed the input to both models and get the logits for all tokens

def get_logits(prompt_and_response_ids, thinking_start_token_index, thinking_end_token_index):
    # Get logits from both models
    with torch.no_grad():
        # DeepSeek model logits
        deepseek_outputs = deepseek_model(
            input_ids=prompt_and_response_ids.to(deepseek_model.device)
        )
        deepseek_logits = deepseek_outputs.logits
        
        # Original model logits
        original_outputs = original_model(
            input_ids=prompt_and_response_ids.to(original_model.device)
        )
        original_logits = original_outputs.logits

    # Move logits to CPU for easier processing
    deepseek_logits = deepseek_logits.cpu()
    original_logits = original_logits.cpu()

    # Assert both logits have the same shape
    assert deepseek_logits.shape == original_logits.shape

    # Return only the logits for the thinking tokens
    deepseek_logits = deepseek_logits[0, thinking_start_token_index:thinking_end_token_index]
    original_logits = original_logits[0, thinking_start_token_index:thinking_end_token_index]

    return deepseek_logits, original_logits

deepseek_logits, original_logits = get_logits(
    prompt_and_response_ids=model_input['prompt_and_response_ids'],
    thinking_start_token_index=model_input['thinking_start_token_index'],
    thinking_end_token_index=model_input['thinking_end_token_index']
)

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

# Calculate KL divergence for each position
kl_divergence = calculate_kl_divergence(deepseek_logits, original_logits)

# %% Create interactive visualization using activation_visualization

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
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = None
        self.var = None
        self.count = 0

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd()
        if self.mean is not None:
            new_object.mean = self.mean.clone()
            new_object.var = self.var.clone()
            new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = arr.double().mean(dim=0)
        batch_var = arr.double().var(dim=0)
        batch_count = arr.shape[0]
        if batch_count == 0:
            return
        if self.mean is None:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
        else:
            self.update_from_moments(batch_mean, batch_var, batch_count)

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
    ) -> tuple[torch.Tensor, torch.Tensor, float] | dict[str, float]:
        """
        Compute the running mean and variance and also return the count

        Returns:
            mean, var, count
        """
        if return_dict:
            return {
                "mean": self.mean.item(),
                "var": self.var.item(),
                "count": self.count,
            }
        return self.mean, self.var, self.count

# %%

responses_to_collect = 2

all_response_uuids = [response["response_uuid"] for response in annotated_responses_data]

print(f"Collecting {responses_to_collect} responses from {len(all_response_uuids)} responses")

# randomly sample responses_to_collect response uuids from all_response_uuids
response_uuids_to_collect = random.sample(all_response_uuids, responses_to_collect)

kl_stats_per_token = {}

for response_uuid in tqdm(response_uuids_to_collect):
    model_input = prepare_model_input(response_uuid=response_uuid, annotated_responses_data=annotated_responses_data, tasks_data=tasks_data, original_messages_data=original_messages_data, tokenizer=deepseek_tokenizer)

    deepseek_logits, original_logits = get_logits(
        prompt_and_response_ids=model_input['prompt_and_response_ids'],
        thinking_start_token_index=model_input['thinking_start_token_index'],
        thinking_end_token_index=model_input['thinking_end_token_index']
    )

    kl_divergence = calculate_kl_divergence(deepseek_logits, original_logits)

    # thinking_tokens = [str(deepseek_tokenizer.decode(token_id)) for token_id in model_input['thinking_token_ids'][0]]
    thinking_tokens = deepseek_tokenizer.batch_decode(model_input['thinking_token_ids'][0])

    # Add ticks and replace new lines with \n
    thinking_tokens = [token.replace('\n', '\\n') for token in thinking_tokens]
    thinking_tokens = [f"`{token}`" for token in thinking_tokens]

    assert len(thinking_tokens) == len(kl_divergence)

    for token, kl_divergence_value in zip(thinking_tokens, kl_divergence):
        if token not in kl_stats_per_token:
            kl_stats_per_token[token] = RunningMeanStd()
        kl_stats_per_token[token].update(kl_divergence_value.unsqueeze(0))

# %%

tokens_to_show = 20

# Create a list of (token, mean_kl) tuples
token_kl_means = []
for token, stats in kl_stats_per_token.items():
    mean, _, _ = stats.compute()
    token_kl_means.append((token, mean.item()))

# Sort by mean KL divergence
token_kl_means.sort(key=lambda x: x[1], reverse=True)

# Take top 20
top_tokens_to_show_tokens = token_kl_means[:tokens_to_show]

# Create lists for plotting
tokens = [t[0] for t in top_tokens_to_show_tokens]
means = [t[1] for t in top_tokens_to_show_tokens]

# Create the plot
plt.figure(figsize=(15, 6))
plt.bar(range(len(tokens)), means)
plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
plt.title('Top 20 Tokens by Mean KL Divergence')
plt.xlabel('Token')
plt.ylabel('Mean KL Divergence')
plt.tight_layout()
plt.show()

# Print the list
print("\nTop 20 tokens by mean KL divergence:")
for token, mean in top_tokens_to_show_tokens:
    count = kl_stats_per_token[token].count
    print(f"{token}: {mean:.4f} (count: {count})")

# %%
