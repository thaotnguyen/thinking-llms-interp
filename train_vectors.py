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
import gc
from typing import Dict, List, Tuple, Any, DefaultDict, Optional
from torch import Tensor

# %% Load model

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # Can be changed to use different models
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model = NNsight(model).to("cuda")

model.generation_config.temperature=None
model.generation_config.top_p=None

model.eval()  # Ensure model is in eval mode
torch.set_grad_enabled(False)  # Disable gradient computation

mean_vectors: DefaultDict[str, Dict[str, Any]] = defaultdict(lambda: {
    'mean': torch.zeros(model.config.num_hidden_layers, model.config.hidden_size),
    'count': 0
})

# %% Define functions

allowed_labels = [
    'initializing',
    'example-testing',
    'backtracking',
    'deduction',
    'uncertainty-estimation',
    'adding-knowledge'
 ]

def get_label_positions(
    annotated_response: str, 
    prompt_and_model_response_input_ids: List[int], 
    tokenizer: AutoTokenizer
) -> Dict[str, List[Tuple[int, int]]]:
    """Parse annotations and find token positions for each label"""
    label_positions = {}
    pattern = r'\["([\w-]+)"\]([^\[]+)'
    matches = re.finditer(pattern, annotated_response)
    
    for match in matches:
        labels = [label.strip() for label in match.group(1).strip('"').split(',')]
        if "end-section" in labels:
            continue

        if not any(label in allowed_labels for label in labels):
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

    return label_positions

def update_mean_vectors(
    mean_vectors: DefaultDict[str, Dict[str, Tensor]], 
    layer_outputs: Tensor, 
    positions_to_update: Dict[str, List[Tuple[int, int]]]
) -> None:
    """
    Update mean vectors only for specified positions
    positions_to_update: dict of label -> list of positions to update
    """
    for label, positions in positions_to_update.items():
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

def calculate_next_token_frequencies(
    responses_data: List[Dict[str, Any]], 
    tokenizer: AutoTokenizer
) -> DefaultDict[str, DefaultDict[str, int]]:
    """Calculate frequencies of next tokens for each label"""
    label_token_frequencies = defaultdict(lambda: defaultdict(int))
    
    for response in tqdm(responses_data, desc="Calculating token frequencies"):
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

def should_skip_example(
    label: str, 
    next_token: str, 
    used_counts: DefaultDict[str, DefaultDict[str, int]], 
    max_examples: int = 50
) -> bool:
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

# %% Select tokens to process

used_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

# Add constants
MAX_EXAMPLES_PER_NEXT_TOKEN = 5
MAX_EXAMPLES_PER_LABEL = 15  # -1 for using all next tokens

# Add assertion
assert MAX_EXAMPLES_PER_LABEL == -1 or MAX_EXAMPLES_PER_LABEL % MAX_EXAMPLES_PER_NEXT_TOKEN == 0, \
    f"MAX_EXAMPLES_PER_LABEL ({MAX_EXAMPLES_PER_LABEL}) must be divisible by MAX_EXAMPLES_PER_NEXT_TOKEN ({MAX_EXAMPLES_PER_NEXT_TOKEN})"

# Add this new function after the other helper functions
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
    
    return {
        'prompt_and_response_ids': torch.cat(
            tensors=[prompt_message_input_ids, base_response_input_ids],
            dim=1
        ),
        'annotated_response': annotated_response_data["annotated_response"]
    }

# Update the collect_label_positions_by_token function to use prepare_model_input
def collect_label_positions_by_token(
    annotated_responses_data: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    tasks_data: List[Dict[str, Any]],
    original_messages_data: List[Dict[str, Any]]
) -> DefaultDict[str, DefaultDict[str, List[Dict[str, Any]]]]:
    """
    Collect all positions for each (label, next_token) pair across all responses.
    Returns a nested dictionary: label -> next_token -> list of (response_uuid, positions)
    
    Returns:
        DefaultDict[str, DefaultDict[str, List[Dict[str, Any]]]] where the inner Dict has keys:
            'response_uuid': str
            'position': Tuple[int, int]
    """
    label_token_positions = defaultdict(lambda: defaultdict(list))
    
    for annotated_response in tqdm(annotated_responses_data, desc="Collecting label positions"):
        response_uuid = annotated_response["response_uuid"]
        
        # Get model input
        model_input = prepare_model_input(
            response_uuid=response_uuid,
            annotated_responses_data=annotated_responses_data,
            tasks_data=tasks_data,
            original_messages_data=original_messages_data,
            tokenizer=tokenizer
        )
        
        # Get positions for each label
        label_positions = get_label_positions(
            annotated_response=model_input['annotated_response'],
            prompt_and_model_response_input_ids=model_input['prompt_and_response_ids'][0].tolist(),
            tokenizer=tokenizer
        )
        
        # Store positions by label and first token
        for label, positions in label_positions.items():
            for start, end in positions:
                first_token = model_input['prompt_and_response_ids'][0][start:start+1]
                first_token_str = tokenizer.decode(first_token)
                label_token_positions[label][first_token_str].append({
                    'response_uuid': response_uuid,
                    'position': (start, end)
                })
    
    return label_token_positions

# Update the main processing section
# First collect all positions
label_token_positions = collect_label_positions_by_token(
    annotated_responses_data=annotated_responses_data,
    tokenizer=tokenizer,
    tasks_data=tasks_data,
    original_messages_data=original_messages_data
)

# Group examples by response_uuid to process them efficiently
required_examples_by_label_and_token: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

print("Grouping examples by response UUID...")
# For each label and token, randomly sample up to MAX_EXAMPLES_PER_NEXT_TOKEN examples
for label in tqdm(label_token_positions, desc="Processing labels"):
    # Sort all tokens by frequency
    sorted_tokens = sorted(
        label_token_positions[label].items(),
        key=lambda x: label_token_frequencies[label][x[0]],
        reverse=True
    )
    
    # Calculate how many different tokens to use
    num_tokens_to_use = (MAX_EXAMPLES_PER_LABEL // MAX_EXAMPLES_PER_NEXT_TOKEN) if MAX_EXAMPLES_PER_LABEL != -1 else len(sorted_tokens)
    
    # Process selected tokens
    for token_str, examples in tqdm(sorted_tokens[:num_tokens_to_use], 
                                  desc=f"Processing tokens for {label}",
                                  leave=False):
        # Randomly shuffle examples
        random.shuffle(examples)
        # Take up to MAX_EXAMPLES_PER_NEXT_TOKEN examples
        selected_examples = examples[:MAX_EXAMPLES_PER_NEXT_TOKEN]
        
        required_examples_by_label_and_token[label][token_str] += len(selected_examples)

# Update the logging to show selected tokens
print("\nSelected tokens by label:")
for label in label_token_frequencies:
    num_tokens_to_use = (MAX_EXAMPLES_PER_LABEL // MAX_EXAMPLES_PER_NEXT_TOKEN) if MAX_EXAMPLES_PER_LABEL != -1 else len(label_token_frequencies[label])
    print(f"\n{label} (top {num_tokens_to_use} tokens):")
    for token, freq in sorted(label_token_frequencies[label].items(), key=lambda x: x[1], reverse=True)[:num_tokens_to_use]:
        print(f"  {token}: {required_examples_by_label_and_token[label][token]} out of {freq}")

# %%

def process_model_output(prompt_and_model_response_input_ids: Tensor, model: Any) -> Tensor:
    """Get model output and layer activations"""
    start_time = time.time()
    layer_outputs = []
    
    with model.trace(prompt_and_model_response_input_ids):
        for layer_idx in tqdm(range(model.config.num_hidden_layers), 
                            desc="Processing layers",
                            leave=False):
            layer_outputs.append(model.model.layers[layer_idx].output[0].save())
    
    # Stack tensors and move to CPU immediately
    layer_outputs = torch.cat([x.value.cpu().detach().to(torch.float32) for x in layer_outputs], dim=0)
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()
    
    elapsed = time.time() - start_time
    print(f"process_model_output took {elapsed:.2f} seconds")
    return layer_outputs

# %%

print(f"Processing {len(annotated_responses_data)} responses...")

# Initialize the processed examples counter to have zero for all labels and tokens in required_examples_by_label_and_token
processed_examples_by_label_and_token: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
for label in required_examples_by_label_and_token:
    for token in required_examples_by_label_and_token[label]:
        processed_examples_by_label_and_token[label][token] = 0

def select_best_response(
    label_token_positions: DefaultDict[str, DefaultDict[str, List[Dict[str, Any]]]],
    required_examples_by_label_and_token: DefaultDict[str, DefaultDict[str, int]],
    processed_examples_by_label_and_token: DefaultDict[str, DefaultDict[str, int]],
    processed_response_uuids: set[str]
) -> Optional[Tuple[str, Dict[str, List[Tuple[int, int]]]]]:
    """
    Select the response that will fill the most needed (label, token) pairs.
    Returns (response_uuid, {label: [(start, end)]}) or None if no more responses needed.
    """
    # Check if we have met all requirements
    all_requirements_met = True
    for label, token_dict in required_examples_by_label_and_token.items():
        for token, required_count in token_dict.items():
            if processed_examples_by_label_and_token[label][token] < required_count:
                all_requirements_met = False
                break
        if not all_requirements_met:
            break
    
    if all_requirements_met:
        return None

    # Track responses and their contribution scores
    response_scores: Dict[str, Tuple[int, Dict[str, List[Tuple[int, int]]]]] = {}
    
    # For each label and token
    for label, token_positions in label_token_positions.items():
        for token, examples in token_positions.items():
            # Skip if we don't need more examples for this (label, token)
            required = required_examples_by_label_and_token[label][token]
            processed = processed_examples_by_label_and_token[label][token]
            if processed >= required:
                continue
                
            # Check each example that could contribute to this (label, token)
            for example in examples:
                response_uuid = example['response_uuid']
                if response_uuid in processed_response_uuids:
                    continue
                    
                # Initialize score and positions dict for this response if needed
                if response_uuid not in response_scores:
                    response_scores[response_uuid] = (0, {})
                
                # Add this position to the response's positions
                score, positions = response_scores[response_uuid]
                if label not in positions:
                    positions[label] = []
                positions[label].append(example['position'])
                
                # Increment score
                response_scores[response_uuid] = (score + 1, positions)
    
    if not response_scores:
        return None
        
    # Get the highest scoring responses
    max_score = max(score for score, _ in response_scores.values())
    best_responses = [(uuid, positions) for uuid, (score, positions) in response_scores.items() if score == max_score]
    
    # Randomly select one of the best responses
    selected_uuid, selected_positions = random.choice(best_responses)
    return selected_uuid, selected_positions

# Initialize set to track processed responses
processed_response_uuids: set[str] = set()

def count_missing_examples(
    required_examples_by_label_and_token: DefaultDict[str, DefaultDict[str, int]],
    processed_examples_by_label_and_token: DefaultDict[str, DefaultDict[str, int]]
) -> int:
    """Count total number of missing examples across all (label, token) pairs"""
    total_missing = 0
    for label, token_dict in required_examples_by_label_and_token.items():
        for token, required_count in token_dict.items():
            processed = processed_examples_by_label_and_token[label][token]
            total_missing += max(0, required_count - processed)
    return total_missing

print("Processing responses...")

# Initialize progress bar
total_missing = count_missing_examples(
    required_examples_by_label_and_token=required_examples_by_label_and_token,
    processed_examples_by_label_and_token=processed_examples_by_label_and_token
)
progress_bar = tqdm(total=total_missing, desc="Processing examples")
previous_missing = total_missing

while True:
    # Select best response to process next
    result = select_best_response(
        label_token_positions=label_token_positions,
        required_examples_by_label_and_token=required_examples_by_label_and_token,
        processed_examples_by_label_and_token=processed_examples_by_label_and_token,
        processed_response_uuids=processed_response_uuids
    )
    
    if result is None:
        print("\nAll required examples have been processed")
        break
        
    response_uuid, labels_positions = result
    iter_start_time = time.time()
    
    # Get model input
    model_input = prepare_model_input(
        response_uuid=response_uuid,
        annotated_responses_data=annotated_responses_data,
        tasks_data=tasks_data,
        original_messages_data=original_messages_data,
        tokenizer=tokenizer
    )
    
    # Move to GPU
    model_input['prompt_and_response_ids'] = model_input['prompt_and_response_ids'].to("cuda")
    
    # Get activations and update mean vectors only for the selected positions
    layer_outputs = process_model_output(
        prompt_and_model_response_input_ids=model_input['prompt_and_response_ids'],
        model=model
    )
    update_mean_vectors(
        mean_vectors=mean_vectors,
        layer_outputs=layer_outputs,
        positions_to_update=labels_positions
    )
    
    # Update processed counts and add to processed set
    for label, positions in labels_positions.items():
        for start, end in positions:
            first_token = model_input['prompt_and_response_ids'][0][start:start+1]
            first_token_str = tokenizer.decode(first_token)
            processed_examples_by_label_and_token[label][first_token_str] += 1
    
    processed_response_uuids.add(response_uuid)
    
    # Update progress bar
    current_missing = count_missing_examples(
        required_examples_by_label_and_token=required_examples_by_label_and_token,
        processed_examples_by_label_and_token=processed_examples_by_label_and_token
    )
    progress_bar.update(previous_missing - current_missing)
    previous_missing = current_missing
    
    # Save periodically
    if len(processed_response_uuids) % save_every == 0:
        save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
        torch.save(save_dict, save_path)
        iter_elapsed = time.time() - iter_start_time
        print(f"\nProcessed {len(processed_response_uuids)} responses. Last iteration took {iter_elapsed:.2f} seconds")
    
    # Clear memory
    del layer_outputs
    del model_input
    torch.cuda.empty_cache()
    gc.collect()

progress_bar.close()

# Save final results
save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
torch.save(save_dict, save_path)
print("Saved final mean vectors")

# %%
