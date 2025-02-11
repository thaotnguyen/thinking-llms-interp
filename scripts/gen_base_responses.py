import json
import os
import uuid
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import NNsight
import torch
from tqdm import tqdm
import click
from deepseek_steering.utils import chat


def save_responses(responses, output_path, model_name, max_tokens, temperature, n_gen, top_p):
    """Save generated responses to a JSON file.
    
    Args:
        responses (list): List of response dictionaries
        output_path (str): Path to save the JSON file
        model_name (str): Name of the model used
        max_tokens (int): Maximum tokens used for generation
        temperature (float): Temperature used for generation
        n_gen (int): Number of generations per task
        top_p (float): Top-p used for generation
    """
    results = {
        "model_name": model_name,
        "responses": responses,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n_gen": n_gen,
        "top_p": top_p
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def generate_base_response(model, tokenizer, task_uuid, message, max_tokens, temperature, top_p, n_gen):
    """Generate a base response for a given message"""
    input_ids = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n_gen,
        use_cache=True
    )
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    return [{
        "response_uuid": str(uuid.uuid4()),
        "response_str": response,
        "task_uuid": task_uuid
    } for response in responses]


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer from the specified model name.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        tuple: (model, tokenizer) loaded and configured for generation
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = NNsight(model).to("cuda")
    return model, tokenizer


def generate_responses_locally(model, tokenizer, tasks, output_path, model_name, max_tokens, temperature, n_gen, save_every, top_p):
    """Generate responses for all tasks with periodic saving.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        tasks: List of tasks to generate responses for
        output_path: Path to save the responses
        model_name: Name of the model used
        max_tokens: Maximum tokens for generation
        temperature: Temperature for generation
        n_gen: Number of generations per task
        save_every: Save checkpoint every N iterations
    
    Returns:
        list: List of generated responses
    """
    responses = []
    
    for i, task in enumerate(tqdm(tasks, desc="Generating responses")):
        response_data = generate_base_response(model, tokenizer, task["task_uuid"], task["prompt_message"], max_tokens, temperature, top_p, n_gen)
        responses.append(response_data)
        
        # Save periodically
        if (i + 1) % save_every == 0:
            save_responses(responses, output_path, model_name, max_tokens, temperature, n_gen, top_p)
            print(f"Saved {len(responses)} responses after iteration {i + 1}")
            
        # Free up memory
        torch.cuda.empty_cache()
    
    return responses


def generate_openai_base_response(model, task_uuid, task_content, max_tokens, temperature, top_p):
    """Generate a base response for a given message"""
    prompt = f"""
    Please answer the following question:
    
    Question:
    `{task_content}`
    
    Please format your response like this:
    <think>
    ...
    </think>
    [Your answer here]
    """
    
    response = chat(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    
    return {
        "response_uuid": str(uuid.uuid4()),
        "response_str": response,
        "task_uuid": task_uuid
    }


def generate_openai_responses(tasks, output_path, model_name, max_tokens, temperature, n_gen, save_every, top_p):
    """Generate responses for all tasks with periodic saving.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        tasks: List of tasks to generate responses for
        output_path: Path to save the responses
        model_name: Name of the model used
        max_tokens: Maximum tokens for generation
        temperature: Temperature for generation
        n_gen: Number of generations per task
        save_every: Save checkpoint every N iterations
    
    Returns:
        list: List of generated responses
    """
    
    responses = []
    for i, task in tqdm(enumerate(tasks), desc="Generating responses"):
        for _ in range(n_gen):
            response_data = generate_openai_base_response(model_name, task["task_uuid"], task["prompt_message"]["content"], max_tokens, temperature, top_p)
            responses.append(response_data)

        if (i + 1) * n_gen % save_every == 0:
            save_responses(responses, output_path, model_name, max_tokens, temperature, n_gen, top_p)
            print(f"Saved {len(responses)} responses after iteration {i + 1}")

    return responses


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
    '--max-tokens',
    default=3_000,
    help='Maximum number of tokens to generate per response'
)
@click.option(
    '--temperature',
    default=0.7,
    help='Temperature for the model'
)
@click.option(
    '--top-p',
    default=0.90,
    help='Top-p for the model'
)
@click.option(
    '--n-gen',
    default=10,
    help='Number of responses to generate per task'
)
@click.option(
    '--save-every',
    default=10,
    help='Save responses every N iterations'
)
def main(model_name: str, output_dir: str, max_tokens: int, temperature: float, n_gen: int, save_every: int, top_p: float):
    """Generate base responses using the specified model and save them to a JSON file."""
    model_id = model_name.split('/')[-1].lower()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tasks from JSON file
    with open('data/tasks.json', 'r') as f:
        tasks = json.load(f)

    tasks = tasks[:3]
    
    # Generate responses
    output_path = os.path.join(output_dir, f'base_responses_{model_id}.json')
    if model_id == "gpt-4o" or "openai" in model_name:
        responses = generate_openai_responses(
            tasks, output_path, model_id,
            max_tokens, temperature, n_gen, save_every, top_p
        )
    else:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_name)
        responses = generate_responses_locally(
            model, tokenizer, tasks, output_path, model_name,
            max_tokens, temperature, n_gen, save_every, top_p
        )
    
    # Save final results
    save_responses(responses, output_path, model_name, max_tokens, temperature, n_gen, top_p)

if __name__ == "__main__":
    main()
