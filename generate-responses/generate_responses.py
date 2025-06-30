import argparse
import dotenv
dotenv.load_dotenv("../.env")

from transformers import AutoTokenizer
import torch
import os
from messages import messages, eval_messages
from tqdm import tqdm
import random
import json
import utils
import math
import gc

# Parse arguments
parser = argparse.ArgumentParser(description="Generate responses from models without steering vectors")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="Model to generate responses from")
parser.add_argument("--save_every", type=int, default=1, 
                    help="Save checkpoint every n examples")
parser.add_argument("--max_tokens", type=int, default=1000,
                    help="Maximum number of tokens to generate")
parser.add_argument("--n_samples", type=int, default=500,
                    help="Number of samples to process")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for processing messages")
parser.add_argument("--generate_eval", action="store_true", default=False,
                    help="Generate responses for eval_messages instead of regular messages")
parser.add_argument("--is_base_model", action="store_true", default=False,
                    help="Whether the model is a base model")
args, _ = parser.parse_known_args()

def get_batched_message_ids(tokenizer, messages_list):
    if args.is_base_model:
        max_token_length = max([len(tokenizer.encode(msg["content"], return_tensors="pt")[0]) for msg in messages_list])
        input_ids = torch.cat([tokenizer.encode(msg["content"], padding="max_length", max_length=max_token_length, return_tensors="pt").to("cuda") for msg in messages_list])
    else:
        max_token_length = max([len(tokenizer.apply_chat_template([msg], add_generation_prompt=True, return_tensors="pt")[0]) for msg in messages_list])
        input_ids = torch.cat([tokenizer.apply_chat_template([msg], add_generation_prompt=True, padding="max_length", max_length=max_token_length, return_tensors="pt").to("cuda") for msg in messages_list])

    return input_ids

def process_model_output_batch(messages_batch, tokenizer, model):
    """Get model output for a batch of messages without collecting activations"""
    tokenized_messages = get_batched_message_ids(tokenizer, messages_batch)
    
    with model.generate(
        {
            "input_ids": tokenized_messages, 
            "attention_mask": (tokenized_messages != tokenizer.pad_token_id).long()
        },
        max_new_tokens=args.max_tokens,
        pad_token_id=tokenizer.pad_token_id
    ) as tracer:
        outputs = model.generator.output.save()

    return outputs

def extract_thinking_process(response):
    """Extract thinking process from response"""
    try:
        think_start = response.index("<think>") + len("<think>")
    except ValueError:
        think_start = 0
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)
    return response[think_start:think_end].strip()


def process_message_batch(messages_batch, tokenizer, model):
    """Process a batch of messages and return response data"""
    outputs = process_model_output_batch(messages_batch, tokenizer, model)
    
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    thinking_processes = [extract_thinking_process(response) for response in responses]
    
    batch_data = []
    for message, response, thinking in zip(messages_batch, responses, thinking_processes):
        batch_data.append({
            "original_message": message,
            "full_response": response,
            "thinking_process": thinking,
            "annotated_thinking": ""  # Empty string for annotated_thinking
        })
    
    return batch_data

# Main execution
if __name__ == "__main__":
    model_name = args.model

    # Load model using utils function
    print(f"Loading model {model_name}...")
    model, tokenizer = utils.load_model(model_name=model_name, load_in_8bit=args.load_in_8bit)

    # Create directories
    os.makedirs('results/vars', exist_ok=True)

    save_every = args.save_every
    eval_prefix = "eval_responses" if args.generate_eval else "responses"
    model_prefix = "base_" if args.is_base_model else ""
    responses_json_path = f"results/vars/{model_prefix}{eval_prefix}_{model_name.split('/')[-1].lower()}.json"

    responses_data = []
    random.seed(args.seed)

    # Choose message set based on --generate_eval flag
    selected_messages = eval_messages if args.generate_eval else messages
    
    print(f"Processing {args.n_samples} {'evaluation' if args.generate_eval else ''} messages to generate responses")
    random.shuffle(selected_messages)
    selected_messages = selected_messages[:args.n_samples]
    num_batches = math.ceil(len(selected_messages) / args.batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc="Processing message batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(selected_messages))
        
        batch_messages = selected_messages[start_idx:end_idx]
        
        batch_data = process_message_batch(batch_messages, tokenizer, model)
        responses_data.extend(batch_data)
        
        if batch_idx % save_every == 0 or batch_idx == num_batches - 1:
            with open(responses_json_path, "w") as f:
                json.dump(responses_data, f, indent=2)
            print(f"Saved responses after batch {batch_idx+1}/{num_batches}")
            
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()

    # Save final results
    with open(responses_json_path, "w") as f:
        json.dump(responses_data, f, indent=2)
    print(f"Saved {len(responses_data)} responses to {responses_json_path}") 