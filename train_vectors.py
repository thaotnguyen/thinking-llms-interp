# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import chat
import torch
import re
from nnsight import NNsight
from collections import defaultdict
from messages import messages
from tqdm import tqdm
import os
from messages import labels
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

def extract_thinking_process(response):
    """Extract thinking process from response"""
    think_start = response.index("<think>") + len("<think>")
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)
    return response[think_start:think_end].strip()

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
            vectors = layer_outputs[:, start-1:end].mean(dim=1)
            current_count = mean_vectors[label]['count']
            current_mean = mean_vectors[label]['mean']
            mean_vectors[label]['mean'] = current_mean + (vectors - current_mean) / (current_count + 1)
            mean_vectors[label]['count'] += 1
            if torch.isnan(mean_vectors[label]['mean']).any():
                print(f"NaN in mean_vectors['{label}']['mean'] at index {index}")

def process_single_message(message, tokenizer, model, mean_vectors, get_annotation=True):
    """Process a single message and update mean vectors"""
    output, layer_outputs = process_model_output(message, tokenizer, model)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    thinking_process = extract_thinking_process(response)
    
    if get_annotation:
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
    
    return {
        "original_message": message,
        "full_response": response,
        "thinking_process": thinking_process,
        "annotated_thinking": annotated_response if get_annotation else None
    }

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
save_path = f"mean_vectors_{model_name.split('/')[-1].lower()}.pt"

load_from_json = False
responses_json_path = f"data/responses_{model_name.split('/')[-1].lower()}.json"

responses_data = []

if load_from_json and os.path.exists(responses_json_path):
    print(f"Loading existing responses from {responses_json_path}")
    with open(responses_json_path, 'r') as f:
        responses_data = json.load(f)
    random.shuffle(responses_data)
    
    for i, response_data in tqdm(enumerate(responses_data), total=len(responses_data), desc="Processing saved responses"):
        output, layer_outputs = process_model_output(response_data["original_message"], tokenizer, model)
        label_positions = get_label_positions(response_data["annotated_thinking"], output[0].tolist(), tokenizer)
        update_mean_vectors(mean_vectors, layer_outputs, label_positions, i)
        
        if i % save_every == 0:
            save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
            torch.save(save_dict, save_path)
            print(f"Current mean_vectors: {mean_vectors.keys()}. Saved...")
else:
    random.shuffle(messages)
    for i, message in tqdm(enumerate(messages), total=len(messages), desc="Processing problems"):
        response_data = process_single_message(message, tokenizer, model, mean_vectors)
        responses_data.append(response_data)
        
        output, layer_outputs = process_model_output(message, tokenizer, model)
        label_positions = get_label_positions(response_data["annotated_thinking"], output[0].tolist(), tokenizer)
        update_mean_vectors(mean_vectors, layer_outputs, label_positions, i)
        
        if i % save_every == 0:
            save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
            torch.save(save_dict, save_path)
            with open(responses_json_path, "w") as f:
                json.dump(responses_data, f, indent=2)
            print(f"Current mean_vectors: {mean_vectors.keys()}. Saved...")

# Save final results
with open(responses_json_path, "w") as f:
    json.dump(responses_data, f, indent=2)
print("Saved final responses data")

save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
torch.save(save_dict, save_path)
print("Saved final mean vectors")

# %%
