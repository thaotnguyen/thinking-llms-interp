import json
import os
import uuid
from tqdm import tqdm
import click
from deepseek_steering.utils import chat

def generate_openai_base_response(model, task_uuid, task_content, max_tokens, temperature):
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
        temperature=temperature
    )
    
    return {
        "response_uuid": str(uuid.uuid4()),
        "response_str": response,
        "task_uuid": task_uuid
    }

@click.command()
@click.option(
    '--model-name',
    "-m",
    default="gpt-4o",
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
    default=5_000,
    help='Maximum number of tokens to generate per response'
)
@click.option(
    '--temperature',
    "-t",
    default=0.01,
    help='Temperature for the model'
)
def main(model_name: str, output_dir: str, max_tokens: int, temperature: float):
    """Generate base responses using the specified model and save them to a JSON file."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tasks from JSON file
    with open('data/tasks.json', 'r') as f:
        tasks = json.load(f)
    
    # Generate responses
    responses = []
    for task in tqdm(tasks, desc="Generating responses"):
        response_data = generate_openai_base_response(model_name, task["task_uuid"], task["prompt_message"]["content"], max_tokens, temperature)
        responses.append(response_data)
    
    # Save results
    output_path = os.path.join(output_dir, f'base_responses_{model_name}.json')
    results = {
        "model_name": model_name,
        "responses": responses,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(responses)} responses to {output_path}")

if __name__ == "__main__":
    main()
