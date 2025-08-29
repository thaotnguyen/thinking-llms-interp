import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
import os
from datasets import load_dataset
from tqdm import tqdm
import random
import json
import math
import gc
from messages import messages
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils.responses import extract_thinking_process

# Parse arguments
parser = argparse.ArgumentParser(description="Generate responses from models without steering vectors")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to generate responses from")
parser.add_argument("--dataset", type=str, default="TIGER-Lab/MMLU-Pro",
                    help="Dataset in HuggingFace to generate responses from", choices=["legacy-messages", "TIGER-Lab/MMLU-Pro"])
parser.add_argument("--dataset_split", type=str, default="test",
                    help="Split of dataset to generate responses from")
parser.add_argument("--max_tokens", type=int, default=1000,
                    help="Maximum number of tokens to generate")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--is_base_model", action="store_true", default=False,
                    help="Whether the model is a base model")
parser.add_argument("--tensor_parallel_size", type=int, default=-1,
                    help="Number of GPUs to use for tensor parallelism")
parser.add_argument("--dtype", type=str, default="auto",
                    help="Data type for model")
parser.add_argument("--temperature", type=float, default=0.0,
                    help="Temperature for sampling")
parser.add_argument("--top_p", type=float, default=1.0,
                    help="Top-p for sampling")
args, _ = parser.parse_known_args()

if args.tensor_parallel_size == -1:
    args.tensor_parallel_size = torch.cuda.device_count()

def get_prompts(tokenizer, messages_list):
    if args.is_base_model:
        prompts = [msg["content"] for msg in messages_list]
    else:
        prompts = [tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages_list]
    return prompts

def get_messages_from_dataset(dataset_name, rows) -> dict[str, dict[str, str]]:
    if dataset_name != "TIGER-Lab/MMLU-Pro":
        raise ValueError(f"Dataset {dataset_name} not supported")

    messages_by_question_id = {}
    for row in rows:
        question_id = row["question_id"]
        messages_by_question_id[question_id] = {
            "role": "user", 
            "content": row["question"],
            "category": row["category"],
            "question": row["question"],
        }
    return messages_by_question_id

def process_model_output_batch(messages_batch, tokenizer, model):
    """Get model output for a batch of messages using VLLM"""
    prompts = get_prompts(tokenizer, messages_batch)
    
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    outputs = model.generate(prompts, sampling_params)
    
    return [output.outputs[0].text for output in outputs]


def process_messages(dataset_name, question_ids, messages_by_question_id, tokenizer, model):
    """Process a batch of messages and return response data"""
    messages_batch = [messages_by_question_id[question_id] for question_id in question_ids]
    responses = process_model_output_batch(messages_batch, tokenizer, model)
    
    thinking_processes = [extract_thinking_process(response) for response in responses]
    
    all_data = []
    for message, response, thinking, question_id in zip(messages_batch, responses, thinking_processes, question_ids):
        all_data.append({
            "original_message": {"role": message["role"], "content": message["content"]},
            "full_response": response,
            "thinking_process": thinking,
            "question_id": question_id,
            "category": message["category"],
            "question": message["question"],
            "dataset_name": dataset_name
        })
    
    return all_data


def save_responses(responses_data, responses_json_path):
    with open(responses_json_path, "w") as f:
        json.dump(responses_data, f, indent=2)
    print(f"Saved {len(responses_data)} responses to {responses_json_path}") 


# Main execution
if __name__ == "__main__":
    model_name = args.model

    # Load model using vllm
    print(f"Loading model {model_name}...")
    model = LLM(
        model=model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create directories
    os.makedirs('results/vars', exist_ok=True)

    model_prefix = "base_" if args.is_base_model else ""
    responses_json_path = f"results/vars/{model_prefix}responses_{model_name.split('/')[-1].lower()}.json"

    random.seed(args.seed)

    if args.dataset == "legacy-messages":
        # Use the index in messages as question_id
        question_ids = list(range(len(messages)))
        messages_by_question_id = {}
        for i, msg in enumerate(messages):
            messages_by_question_id[i] = {
                "role": "user",
                "content": msg["content"],
                "category": "legacy-messages",
                "question": msg["content"],
            }
        print(f"Processing {len(question_ids)} questions from legacy-messages to generate responses")
    else:
        ds = load_dataset(args.dataset)
        rows = ds[args.dataset_split]
        messages_by_question_id = get_messages_from_dataset(args.dataset, rows)
        question_ids = list(messages_by_question_id.keys())
        print(f"Processing {len(question_ids)} questions in {args.dataset_split} split of {args.dataset} to generate responses")
    
    random.shuffle(question_ids)
    
    responses_data = process_messages(args.dataset, question_ids, messages_by_question_id, tokenizer, model)
        
    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()

    # Save final results
    save_responses(responses_data, responses_json_path)