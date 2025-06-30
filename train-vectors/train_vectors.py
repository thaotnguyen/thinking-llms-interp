# %%
import argparse
import dotenv
dotenv.load_dotenv("../.env")

from transformers import AutoTokenizer
import torch
import re
from nnsight import NNsight
from collections import defaultdict
import os
import random
import json
import utils
import math
import gc
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser(description="Train steering vectors from annotated responses")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                    help="Model to train steering vectors for")
parser.add_argument("--save_every", type=int, default=100, 
                    help="Save checkpoints every n batches")
parser.add_argument("--n_samples", type=int, default=500,
                    help="Number of samples to process")
parser.add_argument("--n_few_shot_examples", type=int, default=3,
                    help="Number of few-shot examples to include before each example")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--batch_size", type=int, default=2,
                    help="Batch size for processing messages")
args, _ = parser.parse_known_args()

# %%
def get_batched_message_ids(tokenizer, messages_list):
    max_token_length = max([len(tokenizer.encode(msg, return_tensors="pt")[0]) for msg in messages_list])
    input_ids = torch.cat([tokenizer.encode(msg, padding="max_length", max_length=max_token_length, return_tensors="pt").to("cuda") for msg in messages_list])
    return input_ids

def process_saved_responses_batch(question_responses_list, tokenizer, model, few_shot_examples=None):
    """Get layer activations for a batch of saved responses without generation"""
    # If few-shot examples are provided, prepend them to each response
    full_responses_with_few_shot = []
    target_token_offsets = []  # Track where the actual example starts for each item
    
    for question_response in question_responses_list:
        if few_shot_examples:
            few_shot_text = "Respond to the following questions step by step.\n\n"
            for ex in few_shot_examples:
                few_shot_text += f"Question:\n{ex['original_message']['content']}\nStep by step answer:\n{ex['thinking_process'][:3000]}...\n\n\n\n"
            
            full_text = few_shot_text + f"{question_response}"
            
            # Calculate the token offset for the target example
            few_shot_token_length = len(tokenizer.encode(few_shot_text, add_special_tokens=False, return_tensors="pt")[0])
            target_token_offsets.append(few_shot_token_length)  # -1 to account for the first token
        else:
            full_text = question_response
            target_token_offsets.append(0)  # No offset if no few-shot examples
            
        full_responses_with_few_shot.append(full_text)
    
    tokenized_responses = get_batched_message_ids(tokenizer, full_responses_with_few_shot)
    
    # Process the inputs through the model to get activations
    layer_outputs = []
    with model.trace(
        {
            "input_ids": tokenized_responses, 
            "attention_mask": (tokenized_responses != tokenizer.pad_token_id).long()
        }
    ) as tracer:
        
        # Capture layer outputs
        for layer_idx in range(model.config.num_hidden_layers):
            layer_outputs.append(model.model.layers[layer_idx].output[0].save())
    
    layer_outputs = [x.value.detach().to(torch.float32) for x in layer_outputs]

    batch_layer_outputs = []
    
    for batch_idx in range(len(question_responses_list)):
        # get length of padding tokens
        attention_mask = (tokenized_responses[batch_idx] != tokenizer.pad_token_id).long()
        padding_length = (attention_mask.squeeze() == 0).sum().item()
        
        # Get the offset for this example
        target_offset = target_token_offsets[batch_idx]
        
        example_outputs = torch.stack([
            layer_output[batch_idx][padding_length + target_offset:] 
            for layer_output in layer_outputs
        ])
        
        batch_layer_outputs.append(example_outputs)
    
    return batch_layer_outputs

def get_label_positions(annotated_thinking, response_text, tokenizer):
    """Parse SAE annotations and find token positions for each label"""
    label_positions = {}
    
    # Use a pattern that captures labeled segments in the format [category-name] text [end-section]
    pattern = r'\["(\S+?)"\](.*?)\["end-section"\]'
    matches = list(re.finditer(pattern, annotated_thinking, re.DOTALL))
    
    # Create character to token mapping once
    char_to_token = utils.get_char_to_token_map(response_text, tokenizer)
    
    for match in matches:
        label = match.group(1).strip()
        text = match.group(2).strip()
        
        if not text:  # Skip empty text
            continue
            
        # Find this text in the original response
        text_pos = response_text.find(text)
        if text_pos >= 0:
            # Get start and end token positions
            token_start = char_to_token.get(text_pos, None)
            token_end = char_to_token.get(text_pos + len(text) - 1, None)
            
            # Adjust token_end to include the entire token
            if token_end is not None:
                token_end += 1

            if token_start is None or token_end is None or token_start >= token_end:
                continue
            
            # If we found valid token positions
            if label not in label_positions:
                label_positions[label] = []
            label_positions[label].append((token_start, token_end))
    
    return label_positions

def update_mean_vectors(mean_vectors, layer_outputs, label_positions, index):
    """Update mean vectors for overall and individual labels"""
    # Calculate overall thinking section boundaries
    all_positions = [pos for positions in label_positions.values() for pos in positions]
    if all_positions:
        min_pos = min(start for start, _ in all_positions)
        max_pos = max(end for _, end in all_positions)
        
        # Update overall mean
        overall_vectors = layer_outputs[:, min_pos:max_pos]

        for i in range(overall_vectors.shape[1]):
            current_count = mean_vectors['overall']['count']
            current_mean = mean_vectors['overall']['mean']
            mean_vectors['overall']['mean'] = current_mean + (overall_vectors[:, i] - current_mean) / (current_count + 1)
            mean_vectors['overall']['count'] += 1

        if torch.isnan(mean_vectors['overall']['mean']).any():
            print(f"NaN in mean_vectors['overall']['mean'] at index {index}")
    
    # Update individual labels
    for label, positions in label_positions.items():
        for position in positions:
            start, end = position
            if start >= layer_outputs.shape[1] or end > layer_outputs.shape[1] or start >= end or start <= 0:
                continue
                
            vectors = layer_outputs[:, start-1:start+1]
            if torch.isnan(vectors).any():
                print(f"NaN in vectors for label '{label}' at index {index}")
                continue
            
            if label not in mean_vectors:
                mean_vectors[label] = {
                    'mean': torch.zeros_like(vectors[:,0]).to(model.device),
                    'count': 0
                }

            for i in range(vectors.shape[1]):
                    
                current_count = mean_vectors[label]['count']
                current_mean = mean_vectors[label]['mean']
                mean_vectors[label]['mean'] = current_mean + (vectors[:, i] - current_mean) / (current_count + 1)
                mean_vectors[label]['count'] += 1

# %% Main execution
model_name = args.model
thinking_model_name = utils.model_mapping[model_name]

# Create directories
os.makedirs('results/vars', exist_ok=True)

save_every = args.save_every
save_path = f"results/vars/mean_vectors_{model_name.split('/')[-1].lower()}.pt"

# Add few-shot suffix to save path if using few-shot examples
if args.n_few_shot_examples > 0:
    save_path = save_path.replace('.pt', f'_fs{args.n_few_shot_examples}.pt')

# Default responses path if not provided
responses_json_path = f"../generate-responses/results/vars/responses_{thinking_model_name.split('/')[-1].lower()}.json"

if not os.path.exists(responses_json_path):
    raise FileNotFoundError(f"Annotated responses file not found at {responses_json_path}. Please annotate responses first.")

# Load model using utils function
print(f"Loading model {model_name}...")
model, tokenizer = utils.load_model(model_name=model_name, load_in_8bit=args.load_in_8bit)

mean_vectors = defaultdict(lambda: {
    'mean': torch.zeros(model.config.num_hidden_layers, model.config.hidden_size).to(model.device),
    'count': 0
})

# Load existing responses
print(f"Loading annotated responses from {responses_json_path}")
with open(responses_json_path, 'r') as f:
    responses_data = json.load(f)

# Filter out responses without annotations
valid_responses = [resp for resp in responses_data if resp.get('annotated_thinking')]
print(f"Found {len(valid_responses)} responses with annotations out of {len(responses_data)} total responses")

random.seed(args.seed)
random.shuffle(valid_responses)

# Process in batches
num_samples = min(len(valid_responses), args.n_samples)
num_batches = math.ceil(num_samples / args.batch_size)

print(f"Processing {num_samples} annotated responses in {num_batches} batches...")

# %%
for batch_idx in tqdm(range(num_batches), desc="Processing responses"):
    start_idx = batch_idx * args.batch_size
    end_idx = min(start_idx + args.batch_size, num_samples)
    
    batch_responses = valid_responses[start_idx:end_idx]
    batch_full_question_responses = [f"Question:\n{data['original_message']['content']}\nStep by step answer:\n{data['thinking_process']}" for data in batch_responses]
    batch_indices = list(range(start_idx, end_idx))
    
    # Select few-shot examples if enabled
    selected_few_shot = None
    if args.n_few_shot_examples > 0:
        few_shot_pool = valid_responses[:start_idx] + valid_responses[end_idx:]
        selected_few_shot = random.sample(few_shot_pool, args.n_few_shot_examples)
    
    # Process saved responses to calculate vectors
    batch_layer_outputs = process_saved_responses_batch(
        batch_full_question_responses, 
        tokenizer, 
        model, 
        few_shot_examples=selected_few_shot
    )
    
    # Update vectors based on annotations
    for i, (response_data, layer_outputs) in enumerate(zip(batch_responses, batch_layer_outputs)):
        annotated_thinking = response_data.get("annotated_thinking", "")
        if annotated_thinking:
            question_response = f"Question:\n{response_data['original_message']['content']}\nStep by step answer:\n{response_data['thinking_process']}"
            label_positions = get_label_positions(annotated_thinking, question_response, tokenizer)
            update_mean_vectors(mean_vectors, layer_outputs, label_positions, batch_indices[i])
            
    del batch_layer_outputs
    
    if batch_idx % save_every == 0:
        # Save updated vectors
        save_dict = {k: v['mean'] for k, v in mean_vectors.items()}
        for k, v in save_dict.items():
            save_dict[k] = save_dict[k].cpu()
        torch.save(save_dict, save_path)
        print(f"Saved checkpoint at batch {batch_idx+1}/{num_batches}")

    torch.cuda.empty_cache()
    gc.collect()

# Save final results
save_dict = {k: v['mean'] for k, v in mean_vectors.items()}
for k, v in save_dict.items():
    save_dict[k] = save_dict[k].cpu()
torch.save(save_dict, save_path)

# Print statistics about collected vectors
print("\nCollected vectors statistics:")
for label, data in mean_vectors.items():
    print(f"  {label}: {data['count']} examples")
    
print(f"\nSaved final vectors to {save_path}")
# %%
