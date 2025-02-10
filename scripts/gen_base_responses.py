import json
import os
import uuid
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import NNsight
import torch
from tqdm import tqdm
import random
import click

def generate_base_response(model, tokenizer, task_uuid, message, max_tokens):
    """Generate a base response for a given message"""
    input_ids = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        tokenizer=tokenizer,
        stop_strings=["</think>"]
    )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return {
        "response_uuid": str(uuid.uuid4()),
        "response_str": response,
        "task_uuid": task_uuid
    }

@click.command()
@click.option(
    '--model-name',
    "-m",
    default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    help='Name of the model to use'
)
@click.option(
    '--output-dir',
    "-o",
    default="data",
    help='Directory to save the responses'
)
@click.option(
    '--seed',
    "-s",
    default=42,
    help='Random seed for reproducibility'
)
@click.option(
    '--max-tokens',
    default=5_000,
    help='Maximum number of tokens to generate per response'
)
def main(model_name: str, output_dir: str, seed: int, max_tokens: int):
    """Generate base responses using the specified model and save them to a JSON file."""
    model_id = model_name.split('/')[-1].lower()
    random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tasks from JSON file
    with open('data/tasks.json', 'r') as f:
        tasks = json.load(f)
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = NNsight(model).to("cuda")
    
    # Generate responses
    responses = []
    for task in tqdm(tasks, desc="Generating responses"):
        response_data = generate_base_response(model, tokenizer, task["task_uuid"], task["prompt_message"], max_tokens)
        responses.append(response_data)
        
        # Free up memory
        torch.cuda.empty_cache()
    
    # Save results
    output_path = os.path.join(output_dir, f'base_responses_{model_id}.json')
    results = {
        "model_name": model_name,
        "responses": responses,
        "max_tokens": max_tokens,
        "seed": seed
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(responses)} responses to {output_path}")

if __name__ == "__main__":
    main()
