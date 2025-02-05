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

# %%
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", torch_dtype=torch.bfloat16)
model = NNsight(model).to("cuda")

model.generation_config.temperature=None
model.generation_config.top_p=None

# %%
mean_vectors = defaultdict(lambda: {
    'mean': torch.zeros(model.config.num_hidden_layers, model.config.hidden_size),
    'count': 0
})

save_every = 10
save_path = "mean_vectors.pt"

random.shuffle(messages)

for i, message in tqdm(enumerate(messages), total=len(messages), desc="Processing problems"):
    # Get model response
    tokenized_messages = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
    output = model.generate(
        tokenized_messages, 
        max_new_tokens=500,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    layer_outputs = []
    with model.trace(
        output
    ):
        for layer_idx in range(model.config.num_hidden_layers):
            layer_outputs.append(model.model.layers[layer_idx].output[0].save())

    layer_outputs = torch.cat([x.value.cpu().detach().to(torch.float32) for x in layer_outputs], dim=0)

    # Extract thinking process
    think_start = response.index("<think>") + len("<think>")
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)  # Use full length if no end tag
    thinking_process = response[think_start:think_end].strip()
    
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

    # Parse annotations and find positions
    # Dictionary to store token positions for each label
    label_positions = {}
    
    # Find all annotated sections
    pattern = r'\["([\w-]+)"\]([^\[]+)'  # Match labels with quotes, only words and hyphens
    matches = re.finditer(pattern, annotated_response)
    
    # First, find the token indices for the entire thinking process
    thinking_tokens = output[0].tolist()
    
    for match in matches:
        labels = [label.strip() for label in match.group(1).strip('"').split(',')]
        if("end-section" in labels):
            continue

        text = match.group(2).strip()
        
        # Get token indices for this text segment
        text_tokens = tokenizer.encode(text)[1:]
        
        # Find the position of these tokens in the full thinking process tokens
        for j in range(len(thinking_tokens) - len(text_tokens) + 1):
            if thinking_tokens[j:j + len(text_tokens)] == text_tokens:
                # Add token positions to each label
                for label in labels:
                    if label not in label_positions:
                        label_positions[label] = []
                    # Store token positions relative to the start of the response
                    token_start = j
                    token_end = token_start + len(text_tokens)
                    label_positions[label].append((token_start, token_end))
                break
    
    # Calculate overall thinking section boundaries
    all_positions = [pos for positions in label_positions.values() for pos in positions]
    if all_positions:  # Check if we have any positions
        min_pos = min(start for start, _ in all_positions)
        max_pos = max(end for _, end in all_positions)
        
        # Calculate overall mean activation for thinking section
        overall_vectors = layer_outputs[:, min_pos:max_pos].mean(dim=1)
        current_count = mean_vectors['overall']['count']
        current_mean = mean_vectors['overall']['mean']
        mean_vectors['overall']['mean'] = current_mean + (overall_vectors - current_mean) / (current_count + 1)
        mean_vectors['overall']['count'] += 1

        if torch.isnan(mean_vectors['overall']['mean']).any():
            print(f"NaN in mean_vectors['overall']['mean'] at index {i}")
    
    # Update using running mean formula for individual labels
    for label, positions in label_positions.items():

        for position in positions:
            vectors = layer_outputs[:, position[0]:position[1]].mean(dim=1)
            current_count = mean_vectors[label]['count']
            current_mean = mean_vectors[label]['mean']
            mean_vectors[label]['mean'] = current_mean + (vectors - current_mean) / (current_count + 1)
            mean_vectors[label]['count'] += 1
            if torch.isnan(mean_vectors[label]['mean']).any():
                print(f"NaN in mean_vectors['{label}']['mean'] at index {i}")
            
    # After updating mean vectors, check if we should save
    if i % save_every == 0:
        save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
        torch.save(save_dict, save_path)
        print(f"Current mean_vectors: {mean_vectors.keys()}. Saved...")

# Save one final time after all processing
save_path = "mean_vectors.pt"
save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
torch.save(save_dict, save_path)
print("Saved final mean vectors")

# %%
