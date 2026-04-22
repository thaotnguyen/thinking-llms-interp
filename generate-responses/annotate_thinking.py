# %%
import argparse
import sys
import pickle as pkl
import json
import re
import torch
import numpy as np
from tqdm import tqdm
import dotenv
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils import load_model, get_char_to_token_map, split_into_sentences, get_token_offset_mapping, char_span_to_token_span
from utils.sae import load_sae
from utils.responses import extract_thinking_process

dotenv.load_dotenv("../.env")

# Add parent directory to path for imports
sys.path.append('..')
from utils import utils

parser = argparse.ArgumentParser(description="Annotate thinking processes in generated responses")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model used to generate responses")
parser.add_argument("--layer", type=int, default=6,
                    help="Layer to analyze")
parser.add_argument("--n_clusters", type=int, default=15,
                    help="Number of clusters in the SAE")
parser.add_argument("--max_tokens", type=int, default=None,
                    help="Maximum number of tokens to process (truncate if longer). None = no limit.")
parser.add_argument("--load_in_8bit", action="store_true",
                    help="Load model in 8-bit precision to save memory")
parser.add_argument("--flash_attn", action="store_true", default=False,
                    help="Try to enable FlashAttention 2 for lower memory use")
parser.add_argument("--disable_cache", action="store_true", default=False,
                    help="Disable KV cache during forward passes to reduce VRAM")
args, _ = parser.parse_known_args()


def process_responses(responses_file, model, tokenizer, sae, layer, output_file, model_name, mean_vector=None):
    """Process responses and annotate thinking processes"""
    # Load responses
    with open(responses_file, 'r') as f:
        responses_data = json.load(f)
    
    device = model.device
    
    if mean_vector is not None:
        # Move mean_vector to the target device initially
        mean_vector = mean_vector.to(sae.b_dec.device)
        print("Using mean vector for centering activations")
    else:
        print("Warning: No mean vector provided. Activations will not be centered.")
    
    print(f"Processing {len(responses_data)} responses...")
    
    # Create new structure for annotated responses
    annotated_responses = []
    
    for idx, response_item in tqdm(enumerate(responses_data), total=len(responses_data)):
        try:
            _annotate_one(idx, response_item, model, tokenizer, sae, layer, mean_vector, device, annotated_responses, output_file)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at response {idx} (pmcid={response_item.get('pmcid')}). Skipping.")
            torch.cuda.empty_cache()
            import gc; gc.collect()
            continue
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at response {idx} (pmcid={response_item.get('pmcid')}). Skipping.")
                torch.cuda.empty_cache()
                import gc; gc.collect()
            else:
                print(f"RuntimeError at response {idx}: {e}")
            continue
        except Exception as e:
            print(f"Error at response {idx} (pmcid={response_item.get('pmcid')}): {e}")
            continue

    # Save final results
    with open(output_file, 'w') as f:
        json.dump(annotated_responses, f, indent=2)
    
    return annotated_responses


def _annotate_one(idx, response_item, model, tokenizer, sae, layer, mean_vector, device, annotated_responses, output_file):
        # Get the full response and thinking process
        full_response = response_item['full_response']
        thinking_process = extract_thinking_process(full_response)

        if not thinking_process:
            return

        # Split into sentences (min_words=1 to match generate_activations.py)
        sentences = split_into_sentences(thinking_process, min_words=1)

        # Tokenize only the thinking portion — the case prompt is long and
        # irrelevant; all sentences we need are inside thinking_process.
        offset_mapping = get_token_offset_mapping(
            thinking_process, tokenizer,
            max_length=args.max_tokens if args.max_tokens else None
        )

        if args.max_tokens:
            tokens = tokenizer.encode(thinking_process, return_tensors="pt", truncation=True, max_length=args.max_tokens).to(device)
        else:
            tokens = tokenizer.encode(thinking_process, return_tensors="pt").to(device)

        # Run through model to get activations
        with torch.no_grad():
            with model.trace({
                "input_ids": tokens,
                "attention_mask": (tokens != tokenizer.pad_token_id).long()
            }) as tracer:
                activations = model.model.layers[layer].output.save()

        # Move to CPU immediately — keeps GPU free for the next forward pass
        # while we do the (cheap) per-sentence slicing on CPU.
        activations = activations.detach().cpu()
        del tokens
        torch.cuda.empty_cache()

        # Process each sentence
        annotated_thinking = ""
        local_cursor = 0

        for sentence in sentences:
            # Find this sentence within thinking_process (positions are now local)
            local_pos = thinking_process.find(sentence.strip(), local_cursor)
            if local_pos < 0:
                local_pos = thinking_process.find(sentence.strip())
            if local_pos < 0:
                print(f"Warning: Sentence not found in thinking process: {sentence}")
                continue
            local_cursor = local_pos + len(sentence.strip())
            sentence_pos = local_pos
            if sentence_pos < 0:
                # Sentence not found in full response
                print(f"Warning: Sentence not found in response: {sentence}")
                continue
                
            # Get token positions for this sentence (exclusive end, matches generate_activations.py)
            token_start, token_end = char_span_to_token_span(
                offset_mapping, sentence_pos, sentence_pos + len(sentence.strip())
            )
            
            if token_start is None or token_end is None or token_start >= token_end or token_start >= activations.shape[1] or token_end > activations.shape[1]:
                # Invalid token range
                continue
            
            # Get activations for this sentence
            # Use token_start-1 as slice start (context token) and token_end as exclusive end,
            # consistent with generate_activations.py
            slice_start = max(0, token_start - 1)
            sentence_activations = activations[0, slice_start:token_end, :]
            
            # Average the activations across all tokens in the sentence first
            avg_sentence_activation = torch.mean(sentence_activations, dim=0)

            # Move to target device before centering
            avg_sentence_activation = avg_sentence_activation.to(sae.b_dec.device)
            
            # Center the activation if mean vector is available
            if mean_vector is not None:
                # Ensure mean_vector is on the same device
                if mean_vector.device != avg_sentence_activation.device:
                    mean_vector = mean_vector.to(avg_sentence_activation.device)
                
                avg_sentence_activation = avg_sentence_activation - mean_vector

            # Normalize the activation
            avg_sentence_activation = avg_sentence_activation / (torch.norm(avg_sentence_activation) + 1e-8)
            
            # Apply SAE to the average activation
            avg_activation = avg_sentence_activation - sae.b_dec
            latent_activation = sae.encoder(avg_activation.unsqueeze(0))
        
            # Find the latent with highest activation
            
            top_latent_idx = torch.argmax(latent_activation.squeeze(0)).item()
            del latent_activation, avg_activation, avg_sentence_activation, sentence_activations
            
            # Use idx<number> instead of category title
            # top_activation = round(latent_activation[0, top_latent_idx].item(), 2)
            
            # Format idx tag
            idx_tag = f"idx{top_latent_idx}"
            
            # Add to annotated thinking
            annotated_thinking += f'["{'0'}:{idx_tag}"]{sentence}["end-section"]'
        
        # Create new annotated response item with only required fields
        annotated_item = {
            'pmcid': response_item['pmcid'],
            'question_id': response_item['question_id'],
            'category': response_item['category'] if 'category' in response_item else 'diagnosis',
            'dataset_name': response_item['dataset_name'],
            'annotated_thinking': annotated_thinking.strip()
        }
        annotated_responses.append(annotated_item)

        # Free the activation tensor and flush GPU cache before the next response.
        del activations
        torch.cuda.empty_cache()
        import gc; gc.collect()

        # Save intermediate results every 10 items
        if (len(annotated_responses)) % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(annotated_responses, f, indent=2)

# %% Get model ID from model name
model_name = args.model
model_id = model_name.split('/')[-1].lower()

# %%
responses_file = f"results/vars/responses_{model_id}.json"
output_file = f"results/vars/annotated_responses_{model_id}.json"
mean_vector_file = f"results/vars/activations_{model_id}_100000_{args.layer}.pkl"

# Load model and tokenizer
print(f"Loading model {model_name}...")
model, tokenizer = load_model(model_name=model_name, load_in_8bit=args.load_in_8bit,
                              enable_flash_attn=args.flash_attn, disable_cache=args.disable_cache)

# %% Load SAE
print(f"Loading SAE for model {model_id}, layer {args.layer}, clusters {args.n_clusters}...")
sae, checkpoint = load_sae(model_id, args.layer, args.n_clusters)
sae = sae.to(model.device)

# Get mean vector from checkpoint (preferred) or activations file (fallback)
mean_vector = None
if 'mean_vector' in checkpoint and checkpoint['mean_vector'] is not None:
    print("Loading mean_vector from SAE checkpoint")
    mv = checkpoint['mean_vector']
    if isinstance(mv, np.ndarray):
        mean_vector = torch.from_numpy(mv)
    elif isinstance(mv, torch.Tensor):
        mean_vector = mv
    else:
        print(f"Warning: mean_vector in checkpoint has unexpected type {type(mv)}")
        
    if mean_vector is not None:
        mean_vector = mean_vector.to(model.device)
else:
    print(f"mean_vector not found in SAE checkpoint. Checking {mean_vector_file}...")
    if os.path.exists(mean_vector_file):
        try:
            with open(mean_vector_file, 'rb') as f:
                data = pkl.load(f)
                if len(data) >= 3 and data[2] is not None:
                    print(f"Loading mean_vector from {mean_vector_file}")
                    # Handle if data[2] is numpy array or tensor
                    if isinstance(data[2], np.ndarray):
                        mean_vector = torch.from_numpy(data[2]).to(model.device)
                    else:
                        mean_vector = torch.tensor(data[2], device=model.device)
                else:
                    print("No mean_vector found in activations file (old format?)")
        except Exception as e:
            print(f"Error loading mean_vector from pickle: {e}")
    else:
         print(f"Activations file {mean_vector_file} not found.")

# %% Process responses
processed_data = process_responses(
    responses_file, 
    model, 
    tokenizer, 
    sae, 
    args.layer,
    output_file,
    model_name,
    mean_vector
)

print(f"Annotation complete. Annotated data saved to {output_file}") 

# %%
