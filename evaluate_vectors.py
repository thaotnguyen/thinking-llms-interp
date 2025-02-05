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

os.system('')  # Enable ANSI support on Windows

random.shuffle(eval_messages)

# %% Evaluation examples - 3 from each category
def load_model_and_vectors():
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", torch_dtype=torch.bfloat16)
    model = NNsight(model).to("cuda")
    
    # Load mean vectors
    mean_vectors_dict = torch.load("mean_vectors.pt")
    print(mean_vectors_dict.keys())
    
    # Compute feature vectors by subtracting overall mean
    overall_mean = mean_vectors_dict['overall']['mean']
    feature_vectors = {}
    
    for label in mean_vectors_dict:
        if label != 'overall':
            feature_vectors[label] = mean_vectors_dict[label]['mean'] - overall_mean
            
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

def custom_generate_with_projection_removal(model, tokenizer, input_ids, max_new_tokens, label, feature_vectors, steer_positive=False):
    generated_ids = input_ids.clone().cpu()
    if label in feature_vectors:
        # Move feature vectors to GPU only once, outside the loop
        feature_vector = feature_vectors[label].to("cuda").to(torch.bfloat16)
        normalized_features = feature_vector / torch.norm(feature_vector, dim=1, keepdim=True)
    else:
        normalized_features = None
            
    for k in tqdm(range(max_new_tokens), desc="Generating response"):
        # Clear cache at start of each iteration
        input_chunk = generated_ids.to("cuda")
        
        with torch.no_grad():  # Add this to reduce memory usage
            with model.trace(input_chunk) as trace:
                if normalized_features is not None:
                    for layer_idx in range(model.config.num_hidden_layers):
                        hidden_states = model.model.layers[layer_idx].output[0]
                        # Compute projections more efficiently
                        if steer_positive:
                            if layer_idx in range(4,16):
                                projection = torch.einsum('sh,h->s', hidden_states[0], normalized_features[layer_idx])
                                projection_vector = projection[1:].unsqueeze(-1) * normalized_features[layer_idx]  # Outer product
                                model.model.layers[layer_idx].output[0][:, 1:] += 0.1 * projection_vector
                        else:
                            projection = torch.einsum('sh,h->s', hidden_states[0], normalized_features[layer_idx])
                            projection_vector = projection[1:].unsqueeze(-1) * normalized_features[layer_idx]  # Outer product
                            model.model.layers[layer_idx].output[0][:, 1:] -= 0.1 * projection_vector

                        del hidden_states
                
                outputs = model.lm_head.output.save()
        
        next_token = outputs[:, -1, :].argmax(dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).cpu()], dim=1)

        # Explicitly delete tensors
        del trace, outputs, next_token, input_chunk
       
        torch.cuda.empty_cache()
        if k % 10 == 0:
            gc.collect()
    
    gc.collect()
    return generated_ids.cpu()

# %%
model, tokenizer, feature_vectors = load_model_and_vectors()

# %% Get activations and response
data_idx = 2
activations, thinking_process, full_response = get_thinking_activations(model, tokenizer, data_idx)

# %%
print("Original response:")
input_ids = tokenizer.apply_chat_template([eval_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")
output_ids = custom_generate_with_projection_removal(
        model, 
        tokenizer, 
        input_ids, 
        max_new_tokens=500, 
        label="none",
        feature_vectors=feature_vectors
    )
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
print("\n================\n")

# %%
input_ids = tokenizer.apply_chat_template([eval_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")
output_ids = custom_generate_with_projection_removal(
        model,
        tokenizer, 
        input_ids, 
        max_new_tokens=500, 
        label="backtracking",
        feature_vectors=feature_vectors,
        steer_positive=True
    )
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
print("\n================\n")


# %% Create heatmap of cosine similarities
def plot_cosine_similarity_heatmap(feature_vectors, layer_idx):
    labels = list(feature_vectors.keys())
    n_labels = len(labels)
    
    # Create similarity matrix
    similarity_matrix = torch.zeros((n_labels, n_labels))
    for i, label_1 in enumerate(labels):
        for j, label_2 in enumerate(labels):
            similarity_matrix[i, j] = torch.cosine_similarity(
                feature_vectors[label_1][layer_idx], 
                feature_vectors[label_2][layer_idx], 
                dim=-1
            )
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    
    # Add labels
    plt.xticks(range(n_labels), labels, rotation=45, ha='right')
    plt.yticks(range(n_labels), labels)
    
    plt.title(f'Cosine Similarity Between Feature Vectors (Layer {layer_idx})')
    plt.tight_layout()
    plt.show()

# Plot heatmaps for all layers
for layer_idx in range(model.config.num_hidden_layers):
    plot_cosine_similarity_heatmap(feature_vectors, layer_idx)



# %%
