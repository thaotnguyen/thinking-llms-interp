# %%
import dotenv
dotenv.load_dotenv(".env")

from transformers import AutoTokenizer, AutoModelForCausalLM
from deepseek_steering.utils import chat
import torch
import re
from nnsight import NNsight
from collections import defaultdict
from deepseek_steering.messages import messages
from tqdm import tqdm
from deepseek_steering.messages import labels
import random
import json


# %%
def process_model_output(message, tokenizer, model):
    """Get model output and layer activations"""
    tokenized_messages = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
    output = model.generate(
        tokenized_messages, 
        max_new_tokens=500,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    layer_outputs = []
    with model.trace(output):
        for layer_idx in range(model.config.num_hidden_layers):
            layer_outputs.append(model.model.layers[layer_idx].output[0].save())
    
    layer_outputs = torch.cat([x.value.cpu().detach().to(torch.float32) for x in layer_outputs], dim=0)
    return output, layer_outputs

def get_label_positions(annotated_response, thinking_tokens, tokenizer):
    """Parse annotations and find token positions for each label"""
    label_positions = {}
    pattern = r'\["([\w-]+)"\]([^\[]+)'
    matches = re.finditer(pattern, annotated_response)
    
    for match in matches:
        labels = [label.strip() for label in match.group(1).strip('"').split(',')]
        if "end-section" in labels:
            continue

        text = match.group(2).strip()
        text_tokens = tokenizer.encode(text)[1:]
        
        for j in range(len(thinking_tokens) - len(text_tokens) + 1):
            if thinking_tokens[j:j + len(text_tokens)] == text_tokens:
                for label in labels:
                    if label not in label_positions:
                        label_positions[label] = []
                    token_start = j
                    token_end = j + len(text_tokens)
                    label_positions[label].append((token_start, token_end))
                break
    
    return label_positions

def update_mean_vectors(mean_vectors, layer_outputs, label_positions, index):
    """Update mean vectors for overall and individual labels"""
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
    
    # Update individual labels
    for label, positions in label_positions.items():
        for position in positions:
            start, end = position
            vectors = layer_outputs[:, start-1:start].mean(dim=1)
            current_count = mean_vectors[label]['count']
            current_mean = mean_vectors[label]['mean']
            mean_vectors[label]['mean'] = current_mean + (vectors - current_mean) / (current_count + 1)
            mean_vectors[label]['count'] += 1
            if torch.isnan(mean_vectors[label]['mean']).any():
                print(f"NaN in mean_vectors['{label}']['mean'] at index {index}")


def calculate_next_token_frequencies(responses_data, tokenizer):
    """Calculate frequencies of next tokens for each label"""
    label_token_frequencies = defaultdict(lambda: defaultdict(int))
    
    for response in responses_data:
        annotated_text = response["annotated_thinking"]
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

# %% Main execution
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

# %%
save_every = 10
save_path = f"data/mean_vectors_{model_name.split('/')[-1].lower()}.pt"
responses_json_path = f"data/annotated_responses_{model_name.split('/')[-1].lower()}.json"

if not os.path.exists(responses_json_path):
    raise FileNotFoundError(f"Responses file not found at {responses_json_path}")

print(f"Loading existing responses from {responses_json_path}")
with open(responses_json_path, 'r') as f:
    responses_data = json.load(f)
random.shuffle(responses_data)

# Calculate token frequencies for each label
label_token_frequencies = calculate_next_token_frequencies(responses_data, tokenizer)

# Track how many times we've used each token for each label
used_counts = defaultdict(lambda: defaultdict(int))

for i, response_data in tqdm(enumerate(responses_data), total=len(responses_data), desc="Processing saved responses"):
    output, layer_outputs = process_model_output(response_data["original_message"], tokenizer, model)
    label_positions = get_label_positions(response_data["annotated_thinking"], output[0].tolist(), tokenizer)
    
    # Check frequencies and skip if needed
    should_process = False
    for label, positions in label_positions.items():
        for start, end in positions:
            # Get the first token of the labeled sequence
            text = tokenizer.decode(output[0][start:start+1])
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

    print(label_token_frequencies)

# Save final results
save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
torch.save(save_dict, save_path)
print("Saved final mean vectors")

# %%
