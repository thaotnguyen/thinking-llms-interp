import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
import os
from datasets import load_dataset
from tqdm import tqdm
import random
import json
import utils
import math
import gc

# Parse arguments
parser = argparse.ArgumentParser(description="Generate responses from models without steering vectors")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to generate responses from")
parser.add_argument("--dataset", type=str, default="TIGER-Lab/MMLU-Pro",
                    help="Dataset in HuggingFace to generate responses from")
parser.add_argument("--dataset_split", type=str, default="test",
                    help="Split of dataset to generate responses from")
parser.add_argument("--save_every", type=int, default=1, 
                    help="Save checkpoint every n batches")
parser.add_argument("--max_tokens", type=int, default=1000,
                    help="Maximum number of tokens to generate")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for processing messages")
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

def get_batch_messages_from_dataset_rows(dataset_name, rows) -> dict[str, dict[str, str]]:
    if dataset_name != "TIGER-Lab/MMLU-Pro":
        raise ValueError(f"Dataset {dataset_name} not supported")

    messages_by_question_id = {}
    for row in rows:
        question = row["question"]
        question_id = row["question_id"]
        messages_by_question_id[question_id] = {"role": "user", "content": question}
    return messages_by_question_id

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


def process_message_batch(dataset_name, rows, question_ids, messages_by_question_id, tokenizer, model):
    """Process a batch of messages and return response data"""
    messages_batch = [messages_by_question_id[question_id] for question_id in question_ids]
    outputs = process_model_output_batch(messages_batch, tokenizer, model)
    
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    thinking_processes = [extract_thinking_process(response) for response in responses]
    
    batch_data = []
    for message, response, thinking, question_id in zip(messages_batch, responses, thinking_processes, question_ids):
        row = next(row for row in rows if row["question_id"] == question_id)
        batch_data.append({
            "original_message": message,
            "full_response": response,
            "thinking_process": thinking,
            "question_id": question_id,
            "category": row["category"],
            "question": row["question"],
            "dataset_name": dataset_name
        })
    
    return batch_data


def save_responses(responses_data, responses_json_path):
    with open(responses_json_path, "w") as f:
        json.dump(responses_data, f, indent=2)
    print(f"Saved {len(responses_data)} responses to {responses_json_path}") 


# Main execution
if __name__ == "__main__":
    model_name = args.model

    # Load model using utils function
    print(f"Loading model {model_name}...")
    model, tokenizer = utils.load_model(model_name=model_name, load_in_8bit=args.load_in_8bit)

    # Create directories
    os.makedirs('results/vars', exist_ok=True)

    save_every = args.save_every
    model_prefix = "base_" if args.is_base_model else ""
    responses_json_path = f"results/vars/{model_prefix}responses_{model_name.split('/')[-1].lower()}.json"

    responses_data = []
    random.seed(args.seed)

    ds = load_dataset(args.dataset)
    rows = ds[args.dataset_split]
    messages_by_question_id = get_batch_messages_from_dataset_rows(args.dataset, rows)
    question_ids = list(messages_by_question_id.keys())
    
    print(f"Processing {len(question_ids)} questions in {args.dataset_split} split of {args.dataset} to generate responses")
    random.shuffle(question_ids)
    question_ids = question_ids[:len(question_ids)]
    num_batches = math.ceil(len(question_ids) / args.batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches of messages"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(question_ids))
        
        batch_question_ids = question_ids[start_idx:end_idx]
        batch_data = process_message_batch(args.dataset, rows, batch_question_ids, messages_by_question_id, tokenizer, model)
        responses_data.extend(batch_data)
        
        if batch_idx % save_every == 0 or batch_idx == num_batches - 1:
            save_responses(responses_data, responses_json_path)
            
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()

    # Save final results
    save_responses(responses_data, responses_json_path)