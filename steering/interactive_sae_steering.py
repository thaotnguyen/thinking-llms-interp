# %%
import sys
import os
import dotenv
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from utils.utils import model_mapping

# Add parent directory to path for imports
sys.path.append('..')
dotenv.load_dotenv("../.env")

from utils import utils
from utils.utils import SAE
from messages import eval_messages

# %% Main interactive script
#model_name = "Qwen/Qwen2.5-Math-1.5B"
model_name = "meta-llama/Llama-3.1-8B"

layer = 14
n_clusters = 19

# %% Get model ID from model name
model_id = model_name.split('/')[-1].lower()
is_base_model = False if "deepseek" in model_name else True

# %% Load model and tokenizer
print(f"Loading models {model_name}...")
model, tokenizer = utils.load_model(model_name=model_name)

# %% Load SAE
sae_model_id = model_mapping[model_name].split("/")[-1].lower() if model_name in list(model_mapping.keys()) else model_id
latent_descriptions = utils.get_latent_descriptions(sae_model_id, layer, n_clusters)

if not is_base_model:
    sae, checkpoint = utils.load_sae(sae_model_id, layer, n_clusters)
else: 
    feature_vector = torch.load(f"../train-vectors/results/vars/optimized_vectors_{model_id}.pt")

# %%
print("\nAvailable SAE latents:")
if not is_base_model:
    for idx in range(sae.encoder.weight.data.shape[0]):
        title = latent_descriptions.get(idx, {}).get('title', 'No title available')
        print(f"{idx}: {title}")
else:
    for idx, latent_title in enumerate(feature_vector.keys()):
        print(f"{idx}: {latent_title}")

# %% Format input for generation
latent_idx = 14
data_idx = 15

if not is_base_model:
    latent_title = latent_descriptions.get(latent_idx, {}).get('title', 'No title available')
    input_ids = tokenizer.apply_chat_template([eval_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")
else:
    latent_title = list(feature_vector.keys())[latent_idx]
    input_ids = tokenizer.encode(f"Respond to the following question step by step:\n\nQuestion:\n{eval_messages[data_idx]['content']}\nStep by step answer:\n", return_tensors="pt").to("cuda")

print(f"Latent title: {latent_title}")

# %% Generate original response once
print("\nGenerating original response...")
original_output_ids = utils.custom_generate_with_steering(
    model,
    tokenizer,
    input_ids,
    max_new_tokens=250,
    steering_vector=None,
    layer=None
)
original_response = tokenizer.decode(original_output_ids[0])

# %%
effects_path = f"../vector-layer-attribution/results/vars/layer_effects_{model_id}_layer_{layer}_n_clusters_{n_clusters}.json"

try:
    with open(effects_path, 'r') as f:
        effects = json.load(f)

    steering_layer = None
    for i, effect in enumerate(effects.values()):
        if effect["title"].lower().replace(" ", "-") == latent_title:
            steering_layer = effect["max_layer"]
            break
except FileNotFoundError:
    print(f"Effects file not found: {effects_path}")
    steering_layer = layer

if steering_layer is None:
    print(f"Latent not found in effects: {latent_title}")
    steering_layer = layer

steering_layer = 4

print(f"Steering layer: {steering_layer}")

if not is_base_model:
    vector = sae.W_dec[latent_idx].detach()
else:
    vector = feature_vector[latent_title.lower().replace(" ", "-")]

print("Top 5 closest token embeddings:")
embed_similarity = (model.model.embed_tokens.weight.data.to(torch.float32) @ vector.T.to("cuda"))
for i in embed_similarity.argsort(descending=True)[:5]:
    print(f"Token: {tokenizer.decode(i)}, Similarity: {embed_similarity[i]}")

print("Top 5 closes token unembeddings:")
unembed_similarity = (model.lm_head.weight.data.to(torch.float32) @ vector.T.to("cuda"))
for i in unembed_similarity.argsort(descending=True)[:5]:
    print(f"Token: {tokenizer.decode(i)}, Similarity: {unembed_similarity[i]}")


# %% Print original response
print(f"Steering with latent: {latent_title}")

print("\nOriginal response:")
print(original_response)
print("\n================\n")

# Generate positively steered response
print("Generating positively steered response...")
pos_output_ids = utils.custom_generate_with_steering(
    model,
    tokenizer,
    input_ids,
    max_new_tokens=100,
    steering_vector=vector,
    layer=steering_layer,
    coefficient=1
)
pos_response = tokenizer.decode(pos_output_ids[0])
print(pos_response)
print("\n================\n") 

# %% Generate negatively steered response
print("Generating negatively steered response...")
neg_output_ids = utils.custom_generate_with_steering(
    model,
    tokenizer,
    input_ids,
    max_new_tokens=500,
    steering_vector=vector,
    layer=steering_layer,
    coefficient=-0.5
)
neg_response = tokenizer.decode(neg_output_ids[0])
print(neg_response)
print("\n================\n")
