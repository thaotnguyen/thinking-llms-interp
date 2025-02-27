# %%
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import random
from tqdm import tqdm
from typing import List, Dict, Any
# from tiny_dashboard.visualization_utils import activation_visualization
# from IPython.display import HTML, display
import pickle
# import numpy as np
# from deepseek_steering.running_mean import RunningMeanStd

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

# Disable gradients globally
torch.set_grad_enabled(False)

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
# deepseek_model = AutoModelForCausalLM.from_pretrained(
#     deepseek_model_name,
#     torch_dtype=torch.float16,  # Use float16 for memory efficiency
#     device_map="auto"  # Automatically handle device placement
# )

# original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
# original_model = AutoModelForCausalLM.from_pretrained(
#     original_model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

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
    
    return {
        'prompt_and_response_ids': prompt_and_response_ids,
        'annotated_response': annotated_response_data["annotated_response"],
        'thinking_start_token_index': thinking_start_token_index,
        'thinking_end_token_index': thinking_end_token_index,
        'thinking_token_ids': thinking_token_ids,
    }


# %% Feed the input to both models and get the logits for all tokens

def get_logits(prompt_and_response_ids, thinking_start_token_index, thinking_end_token_index):
    """Get logits from both models with memory-efficient handling."""
    # Clear CUDA cache before processing
    torch.cuda.empty_cache()
    
    # Process models one at a time to reduce memory usage
    # DeepSeek model logits
    with torch.no_grad():
        deepseek_outputs = deepseek_model(
            input_ids=prompt_and_response_ids.to(deepseek_model.device)
        )
        # Only keep the logits we need and move to CPU immediately
        deepseek_logits = deepseek_outputs.logits[
            0, 
            thinking_start_token_index:thinking_end_token_index
        ].cpu()
        del deepseek_outputs
    
    # Clear memory before processing next model
    torch.cuda.empty_cache()
    
    # Original model logits
    with torch.no_grad():
        original_outputs = original_model(
            input_ids=prompt_and_response_ids.to(original_model.device)
        )
        # Only keep the logits we need and move to CPU immediately
        original_logits = original_outputs.logits[
            0,
            thinking_start_token_index:thinking_end_token_index
        ].cpu()
        del original_outputs
    
    torch.cuda.empty_cache()
    
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

# response_uuid = random.choice(annotated_responses_data)["response_uuid"]

# model_input = prepare_model_input(
#     response_uuid=response_uuid,
#     annotated_responses_data=annotated_responses_data,
#     tasks_data=tasks_data,
#     original_messages_data=original_messages_data,
#     tokenizer=deepseek_tokenizer
# )

# print(f"\nResponse UUID: {response_uuid}")
# print(f"Prompt and response IDs: `{deepseek_tokenizer.decode(model_input['prompt_and_response_ids'][0], skip_special_tokens=False)}`")
# print(f"Thinking response: `{deepseek_tokenizer.decode(model_input['thinking_token_ids'][0], skip_special_tokens=False)}`")

# deepseek_logits, original_logits = get_logits(
#     prompt_and_response_ids=model_input['prompt_and_response_ids'],
#     thinking_start_token_index=model_input['thinking_start_token_index'],
#     thinking_end_token_index=model_input['thinking_end_token_index']
# )

# # Calculate KL divergence for each position
# kl_divergence = calculate_kl_divergence(deepseek_logits, original_logits)

# # Get the tokens for visualization
# thinking_tokens = deepseek_tokenizer.convert_ids_to_tokens(
#     model_input['thinking_token_ids'][0]
# )

# html = activation_visualization(
#     thinking_tokens,
#     kl_divergence,
#     tokenizer=deepseek_tokenizer,
#     title="KL Divergence between Models",
#     relative_normalization=False,
# )
# display(HTML(html))

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
            # Print the parameters that don't match
            print(f"Existing parameters: {existing_data['experiment_params']}")
            print(f"New parameters: {experiment_params}")

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

        # Move input tensors to CPU until needed
        model_input['prompt_and_response_ids'] = model_input['prompt_and_response_ids'].cpu()
        model_input['thinking_token_ids'] = model_input['thinking_token_ids'].cpu()

        deepseek_logits, original_logits = get_logits(
            prompt_and_response_ids=model_input['prompt_and_response_ids'],
            thinking_start_token_index=model_input['thinking_start_token_index'],
            thinking_end_token_index=model_input['thinking_end_token_index']
        )

        kl_divergence = calculate_kl_divergence(deepseek_logits, original_logits)
        
        # Clean up large tensors we don't need anymore
        del deepseek_logits
        del original_logits
        torch.cuda.empty_cache()

        # Rest of the processing remains the same...
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

# %% Sort tokens by normalized KL divergence
sorted_tokens = sorted(
    next_token_stats.items(),
    key=lambda x: x[1]['normalized_kl'],
    reverse=True
)

def get_display_token(token):
    token = token.replace("\n", "\\n")
    token = f"`{token}`"
    return token

# %% Display results
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
plt.savefig(f"../figures/kl_div_analysis_{deepseek_model_name.split('/')[-1].lower()}_{original_model_name.split('/')[-1].lower()}.png")
plt.show()

# %% 

def collect_sentences_with_tokens(sorted_tokens: list, original_messages_data: list, top_n: int = 30) -> dict:
    """
    Collect sentences containing the top tokens, along with their surrounding context.
    
    Args:
        sorted_tokens: List of (token, stats) tuples sorted by KL divergence
        original_messages_data: List of original response data
        top_n: Number of top tokens to analyze
    
    Returns:
        Dict mapping tokens to lists of sentence contexts (prev, current, next)
    """
    token_sentences = {}
    top_tokens = [token for token, _ in sorted_tokens[:top_n]]
    
    for response in original_messages_data:
        # Get the response text and clean it
        text = response.get("response_str", "")
        if not text:
            continue
            
        # Remove <think> tags if present
        if text.startswith("<think>"):
            text = text[len("<think>"):]
        if "</think>" in text:
            text = text.split("</think>")[0]
            
        # Remove newlines and normalize spaces
        text = text.replace("\n", "").replace("  ", " ").strip()
            
        # Split into sentences (simple split on dots)
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
        
        # Look for tokens in sentences
        for i, sentence in enumerate(sentences):
            for token in top_tokens:
                if token in sentence:
                    # Get surrounding context
                    prev_sentence = sentences[i-1] if i > 0 else None
                    next_sentence = sentences[i+1] if i < len(sentences)-1 else None
                    
                    if token not in token_sentences:
                        token_sentences[token] = []
                    
                    token_sentences[token].append({
                        "prev": prev_sentence,
                        "current": sentence,
                        "next": next_sentence,
                        "response_uuid": response["response_uuid"]
                    })
    
    return token_sentences

# Collect sentences
token_sentences = collect_sentences_with_tokens(
    sorted_tokens,
    original_messages_data,
    EXPERIMENT_PARAMS["top_tokens_to_show"]
)

# Save results
output_path = f"../data/token_sentences_{deepseek_model_name.split('/')[-1].lower()}_{original_model_name.split('/')[-1].lower()}.json"
with open(output_path, 'w') as f:
    json.dump({
        "token_sentences": token_sentences,
        "experiment_params": EXPERIMENT_PARAMS
    }, f, indent=2)

print(f"\nSaved token sentences to {output_path}")

# Print some example contexts
print("\nExample contexts for top tokens:")
print("-" * 60)
for token, contexts in list(token_sentences.items())[:5]:
    print(f"\nToken: {token}")
    for context in contexts[:2]:  # Show first 2 examples
        print("\nContext:")
        if context["prev"]:
            print(f"Prev: {context['prev']}")
        print(f"Current: {context['current']}")
        if context["next"]:
            print(f"Next: {context['next']}")
    print("-" * 30)

# %%

def create_reflex_analysis_prompt(token_sentences: dict, examples_per_token: int = 10, n_labels: int = 3) -> str:
    """
    Creates a prompt for analyzing individual behavioral reflexes in AI responses.
    First identifies core behavioral patterns, then maps them to sentences.
    
    Args:
        token_sentences: Dictionary mapping tokens to lists of sentence contexts
        examples_per_token: Number of example sentences to include per token
    
    Returns:
        A formatted prompt string for the LLM
    """
    # Collect and shuffle all sentences
    all_sentences = set()
    for contexts in token_sentences.values():
        for context in contexts[:examples_per_token]:
            all_sentences.add(context["current"])
    
    all_sentences = list(all_sentences)
    
    random.seed(EXPERIMENT_PARAMS["seed"])
    random.shuffle(all_sentences)
    
    prompt = f"""You are analyzing behavioral reflexes in the reasoning traces of AI models. Your task is to:
1. First identify a small set of {n_labels} core cognitive patterns exhibited in the sentences below. The labels should capture high-level reasoning patterns or cognitive processes. Each label should be concise (1-3 words). Make sure the labels are NOT domain-specific. In other words, labels that represent specific problem-solving strategies, not specific to a domain.
2. Then map each sentence to one of these labels.

Here are the sentences to analyze:"""

    # Add numbered sentences
    for i, sentence in enumerate(all_sentences, 1):
        prompt += f'\n{i}. "{sentence}"'
    
    prompt += """

Output your analysis in this format:

Reasoning Patterns:"""

    for i in range(n_labels):
        prompt += f"\n{chr(65 + i)}. [label {i + 1}]"

    prompt += """

Sentence Classifications (only the letter, no other text):
1. [letter]
2. [letter]
etc."""
    
    return prompt

# Example usage:
analysis_prompt = create_reflex_analysis_prompt(token_sentences, examples_per_token=3)
print(analysis_prompt)

# %%
