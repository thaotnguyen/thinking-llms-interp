import json
import os
import uuid
from deepseek_steering.messages import messages, eval_messages

# Categories based on the comment headers in messages.py
CATEGORIES = [
    "mathematical-logic",
    "spatial-reasoning", 
    "verbal-logic",
    "pattern-recognition",
    "lateral-thinking",
    "causal-reasoning",
    "probabilistic-thinking",
    "systems-thinking",
    "creative-problem-solving",
    "scientific-reasoning"
]

def generate_tasks():
    """Generate task objects from messages"""
    tasks = []
    
    for i, message in enumerate(messages):
        # Each category has 30 questions
        category_index = i // 30
        category = CATEGORIES[category_index]

        task = {
            "task_uuid": str(uuid.uuid4()),
            "prompt_message": message,
            "task_category": category,
            "split": "train"
        }
        tasks.append(task)

    for i, message in enumerate(eval_messages):
        # Each category has 3 questions
        category_index = i // 3
        category = CATEGORIES[category_index]

        task = {
            "task_uuid": str(uuid.uuid4()),
            "prompt_message": message,
            "task_category": category,
            "split": "eval"
        }
        tasks.append(task)
    
    return tasks

def main():
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate tasks
    tasks = generate_tasks()
    
    # Save to JSON file
    output_path = os.path.join("data", "tasks.json")
    with open(output_path, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Generated {len(tasks)} tasks and saved to {output_path}")

if __name__ == "__main__":
    main()
