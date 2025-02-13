import json
import os
import uuid
import click
import asyncio
import logging
from deepseek_steering.open_router_utils import ORBatchProcessor


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


def build_task_prompt(task_content: str) -> str:
    """Build the prompt for a given task content"""
    return f"""
    Please answer the following question:
    
    Question:
    `{task_content}`
    
    Please format your response like this:
    <think>
    ...
    </think>
    [Your answer here]
    """


def load_existing_responses(output_path: str) -> list | None:
    """Load existing responses from a JSON file if it exists."""
    if not os.path.exists(output_path):
        return None
    
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
        return data.get("responses", [])
    except Exception as e:
        logging.warning(f"Failed to load existing responses from {output_path}: {e}")
        return None


async def generate_openai_responses_async(
    tasks, 
    output_path: str, 
    model_id: str,
    max_tokens: int, 
    temperature: float, 
    n_gen: int, 
    top_p: float,
    max_retries: int = 15,
    existing_responses: list | None = None,
):
    """Generate responses using OpenRouter batch processor"""
    def process_response(or_response: str | tuple[str, str], task: dict) -> dict:
        assert isinstance(or_response, tuple), f"ORBatchProcessor should return a tuple: (reasoning, response) for model {model_id}. Found {type(or_response)}"
        
        reasoning, response = or_response
        if reasoning is not None:
            response_str = f"<think>{reasoning}</think>{response}"
        else:
            response_str = f"<think>{response}</think>"

        return {
            "response_uuid": str(uuid.uuid4()),
            "response_str": response_str,
            "task_uuid": task["task_uuid"]
        }

    processor = ORBatchProcessor(
        model_id=model_id,
        temperature=temperature,
        max_new_tokens=max_tokens,
        rate_limiter=None,
        max_retries=max_retries,
        process_response=process_response,
    )

    # Create a map of existing responses per task
    existing_response_counts = {}
    if existing_responses:
        for response in existing_responses:
            task_uuid = response["task_uuid"]
            existing_response_counts[task_uuid] = existing_response_counts.get(task_uuid, 0) + 1

    # Prepare batch items, considering existing responses
    batch_items = []
    for task in tasks:
        existing_count = existing_response_counts.get(task["task_uuid"], 0)
        needed_generations = max(0, n_gen - existing_count)
        
        for _ in range(needed_generations):
            prompt = build_task_prompt(task["prompt_message"]["content"])
            batch_items.append((task, prompt))

    # Process batch
    results = await processor.process_batch(batch_items)

    responses = existing_responses or []
    for (task, result) in results:
        if result is not None:
            responses.append(result)
    
    save_responses(responses, output_path, model_id, max_tokens, temperature, n_gen, top_p)

    return responses


@click.command()
@click.option(
    '--model-name',
    "-m",
    default="deepseek/deepseek-r1-distill-llama-8b", 
    help='Name of the model to use (needs to be in OpenRouter format)'
)
@click.option(
    '--test', 
    is_flag=True,
    help='Run in test mode with reduced tasks and generations'
)
@click.option(
    '--output-dir',
    "-o",
    default="data",
    help='Directory to save the responses'
)
@click.option(
    '--max-tokens',
    default=5_000,
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
    '--verbose',
    "-v",
    is_flag=True,
    help='Verbose output'
)
@click.option(
    '--append',
    is_flag=True,
    help='Append to existing results instead of starting fresh'
)
def main(
    model_name: str, 
    output_dir: str, 
    max_tokens: int, 
    temperature: float, 
    n_gen: int, 
    top_p: float,
    test: bool,
    verbose: bool,
    append: bool
):
    """Generate base responses using OpenRouter batch processor"""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    model_id = model_name.split('/')[-1].lower()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tasks from JSON file
    with open('data/tasks.json', 'r') as f:
        tasks = json.load(f)

    if test:
        tasks = tasks[:3]
        n_gen = 3
        max_tokens = 250
    
    # Generate responses
    output_path = os.path.join(output_dir, f'base_responses_{model_id}.json')
    
    # Load existing responses if append flag is set
    existing_responses = load_existing_responses(output_path) if append else None
    
    responses = asyncio.run(generate_openai_responses_async(
        tasks=tasks,
        output_path=output_path,
        model_id=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        n_gen=n_gen,
        top_p=top_p,
        existing_responses=existing_responses,
    ))
    
    # Save final results
    save_responses(responses, output_path, model_name, max_tokens, temperature, n_gen, top_p)

if __name__ == "__main__":
    main()
