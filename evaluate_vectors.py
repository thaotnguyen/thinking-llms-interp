# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nnsight import NNsight
import matplotlib.pyplot as plt
from utils import chat
import re
import numpy as np
from messages import eval_messages, labels
from tqdm import tqdm
import gc
import random
import os
import utils

os.system('')  # Enable ANSI support on Windows

random.shuffle(eval_messages)

# %% Evaluation examples - 3 from each category
def load_model_and_vectors(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    model, tokenizer, feature_vectors = utils.load_model_and_vectors(compute_features=True, model_name=model_name)
    return model, tokenizer, feature_vectors

def get_thinking_activations(model, tokenizer, message_idx):
    """Get activations for a specific evaluation example"""
    message = eval_messages[message_idx]
    
    # Generate response
    tokenized_messages = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
    output = model.generate(
        tokenized_messages,
        max_new_tokens=500,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract thinking process
    think_start = response.index("<think>") + len("<think>")
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)
    thinking_process = response[think_start:think_end].strip()
    
    # Get activations
    layer_outputs = []
    with model.trace(output):
        for layer_idx in range(model.config.num_hidden_layers):
            layer_outputs.append(model.model.layers[layer_idx].output[0].save())
    
    layer_outputs = torch.cat([x.value.cpu().detach().to(torch.float32) for x in layer_outputs], dim=0)
    
    return layer_outputs, thinking_process, response

# %%
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Can be changed to use different models
model, tokenizer, feature_vectors = load_model_and_vectors(model_name)

# %% Get activations and response
data_idx = 3
activations, thinking_process, full_response = get_thinking_activations(model, tokenizer, data_idx)

# %%
print("Original response:")
input_ids = tokenizer.apply_chat_template([eval_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")
output_ids = utils.custom_generate_with_projection_removal(
    model,
    tokenizer,
    input_ids,
    max_new_tokens=500,
    label="none", 
    feature_vectors=feature_vectors,
    show_progress=True
)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
print("\n================\n")

# %%
input_ids = tokenizer.apply_chat_template([eval_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")
output_ids = utils.custom_generate_with_projection_removal(
    model,
    tokenizer,
    input_ids,
    max_new_tokens=300,
    label="uncertainty-estimation",
    feature_vectors=feature_vectors,
    layers=[8,9,10],
    coefficient=1,
    steer_positive=True,
    show_progress=True
)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
print("\n================\n")

# %%
