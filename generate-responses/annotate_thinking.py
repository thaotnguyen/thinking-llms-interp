# %%
import argparse
import sys
import pickle as pkl
import json
import re
import torch
from tqdm import tqdm
import dotenv
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils import load_model, get_char_to_token_map
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
args, _ = parser.parse_known_args()


def split_into_sentences(text):
    """Split text into sentences using regex while preserving delimiters"""
    # Split on sentence boundaries but keep the delimiters
    # Split after punctuation+space or after newline
    sentences = re.split(r'(?<=[.!?;] )|(?<=\n)', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    # If the last sentence is just an <answer>...</answer> span (possibly
    # spanning multiple lines), drop it so only reasoning remains.
    if sentences:
        last = sentences[-1]
        if re.match(r"^<answer>[\s\S]*</answer>$", last):
            sentences = sentences[:-1]

    return sentences


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
        # Get the full response and thinking process
        full_response = response_item['full_response']
        thinking_process = extract_thinking_process(full_response)
   
        # Split into sentences
        sentences = split_into_sentences(thinking_process)
 
        # Create mapping from character positions to token positions
        char_to_token = get_char_to_token_map(full_response, tokenizer)
        
        # Tokenize the full response (with optional truncation)
        if args.max_tokens:
            tokens = tokenizer.encode(full_response, return_tensors="pt", truncation=True, max_length=args.max_tokens).to(device)
        else:
            tokens = tokenizer.encode(full_response, return_tensors="pt").to(device)
        
        # Run through model to get activations
        with torch.no_grad():
            # Use single-line with-header to keep nnsight's tracer happy
            with model.trace({
                "input_ids": tokens,
                "attention_mask": (tokens != tokenizer.pad_token_id).long()
            }) as tracer:
                activations = model.model.layers[layer].output.save()
        
        # Process each sentence
        annotated_thinking = ""
        
        for sentence in sentences:
            # Find this sentence in the full response
            # Pattern to match either at start of string or after punctuation/newlines
            pattern = r'(?:^|(?:[.?!;\n]|\n\n))\s*(' + re.escape(sentence.strip()) + ')'
            match = re.search(pattern, full_response)
            sentence_pos = match.start(1) if match else -1
            if sentence_pos < 0:
                # Sentence not found in full response
                print(f"Warning: Sentence not found in response: {sentence}")
                print(f"Full response: {sentences}")
                continue
                
            # Get token positions for this sentence
            token_start = char_to_token.get(sentence_pos)
            token_end = char_to_token.get(sentence_pos + len(sentence) - 1)
            
            if token_start is None or token_end is None or token_start >= token_end or token_start >= activations.shape[1] or token_end > activations.shape[1]:
                # Invalid token range
                continue
            
            # Get activations for this sentence
            sentence_activations = activations[0, token_start-1:token_end, :]
            
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
            
            # Use idx<number> instead of category title
            # top_activation = round(latent_activation[0, top_latent_idx].item(), 2)
            
            # Format idx tag
            idx_tag = f"idx{top_latent_idx}"
            
            # Add to annotated thinking
            annotated_thinking += f'["{'0'}:{idx_tag}"]{sentence}["end-section"]'
        
        # Create new annotated response item with only required fields
        annotated_item = {
            'question_id': response_item['question_id'],
            'category': response_item['category'],
            'dataset_name': response_item['dataset_name'],
            'annotated_thinking': annotated_thinking.strip()
        }
        annotated_responses.append(annotated_item)
        
        # Save intermediate results every 10 items
        if (idx + 1) % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(annotated_responses, f, indent=2)
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(annotated_responses, f, indent=2)
    
    return annotated_responses

# %% Get model ID from model name
model_name = args.model
model_id = model_name.split('/')[-1].lower()

# %%
responses_file = f"results/vars/responses_{model_id}.json"
output_file = f"results/vars/annotated_responses_{model_id}.json"
mean_vector_file = f"results/vars/activations_{model_id}_100000_{args.layer}.pkl"

# Load model and tokenizer
print(f"Loading model {model_name}...")
model, tokenizer = load_model(model_name=model_name, load_in_8bit=args.load_in_8bit)

# %% Load SAE
print(f"Loading SAE for model {model_id}, layer {args.layer}, clusters {args.n_clusters}...")
sae, checkpoint = load_sae(model_id, args.layer, args.n_clusters)
sae = sae.to(model.device)

# Get mean vector from checkpoint if available
with open(mean_vector_file, 'rb') as f:
    mean_vector = torch.tensor(pkl.load(f)[2], device=model.device)

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
