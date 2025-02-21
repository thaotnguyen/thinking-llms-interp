# %%
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import random
from tqdm import tqdm
import pickle

# %% Set model names and parameters
deepseek_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
original_model_name = "Qwen/Qwen2.5-14B"
# original_model_name = "Qwen/Qwen2.5-14B-Instruct"

# Generation parameters
max_tokens = 5000  # Maximum number of tokens to generate

# Which labels are we forcing?
thinking_labels = ["example-testing", "uncertainty-estimation", "backtracking"]

# Which top k diverging tokens are we forcing per label?
top_k_diverging_tokens = 5

# %%

seed = 42
random.seed(seed)

# %% Load models

deepseek_tokenizer = AutoTokenizer.from_pretrained(deepseek_model_name)
deepseek_model = AutoModelForCausalLM.from_pretrained(
    deepseek_model_name,
    torch_dtype=torch.float16,  # Use float16 for memory efficiency
    device_map="auto"  # Automatically handle device placement
)

original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
original_model = AutoModelForCausalLM.from_pretrained(
    original_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# %% Load the answer_tasks.json file

answer_tasks_path = "../data/answer_tasks.json"

with open(answer_tasks_path, "r") as f:
    answer_tasks = json.load(f)

# %%

# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
class RunningMeanStd:
    def __init__(self):
        """
        Calculates the running mean, std, and sum of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        self.mean = None
        self.var = None
        self.count = 0
        self.sum = None  # Add sum tracking

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd()
        if self.mean is not None:
            new_object.mean = self.mean.clone()
            new_object.var = self.var.clone()
            new_object.count = float(self.count)
            new_object.sum = self.sum.clone()  # Copy sum
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.
        """
        self.update_from_moments(other.mean, other.var, other.count)
        if self.sum is None:
            self.sum = other.sum.clone()
        else:
            self.sum += other.sum

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = arr.double().mean(dim=0)
        batch_var = arr.double().var(dim=0)
        batch_count = arr.shape[0]
        batch_sum = arr.double().sum(dim=0)  # Calculate batch sum
        
        if batch_count == 0:
            return
        if self.mean is None:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
            self.sum = batch_sum  # Initialize sum
        else:
            self.update_from_moments(batch_mean, batch_var, batch_count)
            self.sum += batch_sum  # Update sum

    def update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float
    ) -> None:
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def compute(
        self, return_dict=False
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor] | dict[str, float]:
        """
        Compute the running mean, variance, count and sum

        Returns:
            mean, var, count, sum if return_dict=False
            dict with keys 'mean', 'var', 'count', 'sum' if return_dict=True
        """
        if return_dict:
            return {
                "mean": self.mean.item(),
                "var": self.var.item(),
                "count": self.count,
                "sum": self.sum.item(),
            }
        return self.mean, self.var, self.count, self.sum

# %% Load kl stats

# Get model IDs for filenames
deepseek_model_id = deepseek_model_name.split('/')[-1].lower()
original_model_id = original_model_name.split('/')[-1].lower()

# Load each stats dictionary
kl_stats = {}
stats_types = ["token", "token_pair", "next_token", "label", "next_token_and_label"]

for stats_type in stats_types:
    stats_path = f"../data/kl_stats/kl_stats_per_{stats_type}_{deepseek_model_id}_{original_model_id}.pkl"
    try:
        with open(stats_path, 'rb') as f:
            kl_stats[stats_type] = pickle.load(f)
        print(f"Loaded {stats_type} KL stats from {stats_path}")
    except FileNotFoundError:
        print(f"Warning: Could not find KL stats file at {stats_path}")
        kl_stats[stats_type] = {}

# Print some basic statistics about the loaded data

print("\nKL Stats Summary:")
for stats_type, stats_dict in kl_stats.items():
    print(f"\n{stats_type.replace('_', ' ').title()}:")
    print(f"Number of unique entries: {len(stats_dict)}")
    
    # Get a few example entries
    examples = list(stats_dict.items())[:3]
    print("Example entries:")
    for key, stats in examples:
        mean, var, count, sum_val = stats.compute()
        print(f"  {key}: mean={mean:.4f}, count={count}")
    
# %% Find top diverging tokens per label

def get_top_tokens_for_label(stats_dict, label, k=5, min_count=10):
    """
    Get the top k most diverging tokens for a given label.
    Only considers tokens that appear at least min_count times.
    Uses sum of KL divergence for ranking to account for both frequency and divergence.
    
    Returns:
        List of tuples (token_pair, sum_kl, count)
    """
    # Filter entries for this label and with sufficient count
    label_entries = []
    for (current_token, next_token, token_label), stats in stats_dict.items():
        # Remove backticks before and after tokens
        current_token = current_token.strip("`")
        next_token = next_token.strip("`")
        if token_label == label:
            _, _, count, sum_val = stats.compute()
            if count >= min_count:
                label_entries.append(((current_token, next_token), sum_val.item(), int(count)))
    
    # Sort by sum of KL divergence and take top k
    label_entries.sort(key=lambda x: x[1], reverse=True)
    return label_entries[:k]

# Get top diverging tokens for each thinking label
print("\nTop diverging token pairs per thinking label:")
for label in thinking_labels:
    print(f"\n{label}:")
    top_tokens = get_top_tokens_for_label(kl_stats["next_token_and_label"], label, k=top_k_diverging_tokens)
    
    for (current_token, next_token), kl_value, count in top_tokens:
        print(f"  `{current_token}` -> `{next_token}`: {kl_value:.4f} (count: {count})")
    
# %% Define evaluation functions

def generate_user_message(task):
    question = task["prompt_message"]["content"]
    return f"Here is a question: '{question}'\n\nPlease think step by step about it. Once you have thought carefully about it, provide your final answer as 'Answer: <answer>'."

def generate_original_model_response(model, tokenizer, task):
    """Generate a response from the model for a given task"""
   
    user_message = generate_user_message(task)
    
    # Format the prompt using the chat template
    if "Instruct" in original_model_name:
        prompt_message = [
            {"role": "user", "content": user_message}
        ]
        input_ids = tokenizer.apply_chat_template(
            [prompt_message],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
    else:
        # For non-Instruct models, directly encode the user message
        input_ids = tokenizer.encode(
            user_message,
            return_tensors="pt",
            add_special_tokens=True
        ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract the generated response (excluding the prompt)
    response_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    num_tokens = len(response_ids)
    
    return response.strip(), num_tokens

def generate_thinking_model_response(model, tokenizer, task):
    """Generate a response from the model for a given task"""
    prompt_message = [
        {"role": "user", "content": generate_user_message(task)}
    ]

    # Format the prompt using the chat template
    input_ids = tokenizer.apply_chat_template(
        [prompt_message],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract the generated response (excluding the prompt)
    response_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    num_tokens = len(response_ids)
    
    return response.strip(), num_tokens

def evaluate_answer(raw_response, correct_answer, model_name):
    """
    Compare the model's response to the correct answer.
    Returns True if the response contains the correct answer.
    """
    # Convert both to lowercase for case-insensitive comparison
    response = raw_response.lower()
    correct_answer = correct_answer.lower()
    
    answer_prefixes = ["answer:", "the answer is"]

    for prefix in answer_prefixes:
        if prefix in response:
            response = response.split(prefix)[1].strip()
            # Remove common punctuation and whitespace after answer prefix
            response = ''.join(c for c in response if c.isalnum())
            return correct_answer in response
    
    print(f"No answer found in response of {model_name}. Expected answer: `{correct_answer}`\nResponse:\n`{raw_response}`")
    return False    

# %% Evaluate deepseek and original models

results = {
    "deepseek": {"correct": 0, "total": 0, "responses": []},
    "original": {"correct": 0, "total": 0, "responses": []},
    "original_with_thinking_tokens": {"correct": 0, "total": 0, "responses": []}
}

print("Evaluating deepseek and original models...")
for task in tqdm(answer_tasks):
    # Generate responses from both models
    original_response, original_num_tokens = generate_original_model_response(original_model, original_tokenizer, task)
    deepseek_response, deepseek_num_tokens = generate_thinking_model_response(deepseek_model, deepseek_tokenizer, task)
    
    # Evaluate responses
    original_correct = evaluate_answer(original_response, task["answer"], "original")
    deepseek_correct = evaluate_answer(deepseek_response, task["answer"], "deepseek")
    
    # Update results
    results["original"]["correct"] += original_correct
    results["original"]["total"] += 1
    results["original"]["responses"].append({
        "task_uuid": task["task_uuid"],
        "task_category": task["task_category"],
        "question": task["prompt_message"]["content"],
        "correct_answer": task["answer"],
        "model_response": original_response,
        "is_correct": original_correct,
        "num_tokens": original_num_tokens
    })

    results["deepseek"]["correct"] += deepseek_correct
    results["deepseek"]["total"] += 1
    results["deepseek"]["responses"].append({
        "task_uuid": task["task_uuid"],
        "task_category": task["task_category"],
        "question": task["prompt_message"]["content"],
        "correct_answer": task["answer"],
        "model_response": deepseek_response,
        "is_correct": deepseek_correct,
        "num_tokens": deepseek_num_tokens
    })
    

# %% Create modified prompting function that forces the thinking tokens

def get_all_force_tokens():
    """Get all token pairs to force across all thinking labels"""
    force_tokens = {} # current_token -> next_token -> labels
    for label in thinking_labels:
        top_tokens = get_top_tokens_for_label(kl_stats["next_token_and_label"], label, k=top_k_diverging_tokens)
        for (current_token, next_token), _, _ in top_tokens:
            # If multiple labels suggest different next tokens for the same trigger,
            # keep the one with higher KL divergence (it's already sorted)
            if current_token not in force_tokens:
                force_tokens[current_token] = {next_token: [label]}
            else:
                if next_token not in force_tokens[current_token]:
                    force_tokens[current_token][next_token] = [label]
                else:
                    force_tokens[current_token][next_token].append(label)
    return force_tokens

def generate_thinking_model_response_with_forcing(model, tokenizer, task):
    """
    Generate a response with token forcing based on deepseek model preferences.
    Uses token-by-token generation to check and potentially force specific token sequences.
    Considers token pairs from all thinking labels.
    Forces a token only when it's the most likely next token according to deepseek.
    """
    # Get all token pairs to watch for
    force_tokens = get_all_force_tokens()

    user_message = generate_user_message(task)
    
    # Format the prompt using the chat template
    if "Instruct" in original_model_name:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
    else:
        input_ids = tokenizer.encode(
            user_message,
            return_tensors="pt",
            add_special_tokens=True
        ).to(model.device)

    deepseek_input_ids = deepseek_tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(deepseek_model.device)
    
    # Generate response token by token
    generated_ids = input_ids.clone()

    for _ in range(max_tokens):
        # Get next token distribution from original model
        with torch.no_grad():
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[0, -1, :]
            original_next_token_id = torch.argmax(next_token_logits).item()
        
        # Check if last generated token is a trigger token
        forced_token = False
        if len(generated_ids[0]) > len(input_ids[0]):  # Only check after first generated token
            last_token = tokenizer.decode(generated_ids[0, -1:])
            if last_token in force_tokens:
                # Get deepseek model's prediction for next token
                with torch.no_grad():
                    deepseek_outputs = deepseek_model(deepseek_input_ids)
                    deepseek_next_token_logits = deepseek_outputs.logits[0, -1, :]
                    
                # Get the target next token
                target_next_tokens = list(force_tokens[last_token].keys())
                
                # Force token only if it's the most likely next token according to deepseek
                deepseek_next_token_id = torch.argmax(deepseek_next_token_logits).item()
                for target_next_token in target_next_tokens:
                    target_token_id = tokenizer.encode(target_next_token, add_special_tokens=False)[0]
                    if deepseek_next_token_id == target_token_id:
                        # Log the forcing event
                        response_so_far = tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
                        original_next_token = tokenizer.decode(original_next_token_id)
                        
                        print(f"\n### Forcing token in task: {task['task_uuid']}")
                        print(f"Response so far: `{response_so_far}`")
                        print(f"Labels forcing token: {force_tokens[last_token][target_next_token]}")
                        print(f"Trigger token: {last_token}")
                        print(f"Forced next token: {target_next_token}")
                        print(f"Original model would have generated: {original_next_token}")
                        print("-" * 80)
                        
                        next_token_id = torch.tensor([[target_token_id]]).to(model.device)
                        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                        deepseek_input_ids = torch.cat([deepseek_input_ids, next_token_id], dim=1)
                        forced_token = True
                        break
        
        if not forced_token:
            # If no forcing occurred, continue with original model's prediction
            next_token_id = torch.tensor([[original_next_token_id]]).to(model.device)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            deepseek_input_ids = torch.cat([deepseek_input_ids, next_token_id], dim=1)

        # Check if we've hit the end token
        if next_token_id == tokenizer.eos_token_id:
            break
    
    # Extract the generated response (excluding the prompt)
    response_ids = generated_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=False)
    num_tokens = len(response_ids)
    
    return response.strip(), num_tokens

# %% Evaluate original model with forced thinking tokens

print("\nEvaluating original model with forced thinking tokens...")
results["original_with_thinking_tokens"] = {"correct": 0, "total": 0, "responses": []}

for task in tqdm(answer_tasks):
    response, num_tokens = generate_thinking_model_response_with_forcing(
        original_model, 
        original_tokenizer, 
        task
    )
    
    # Evaluate response
    is_correct = evaluate_answer(response, task["answer"], "original_with_thinking_tokens")
    
    # Update results
    results["original_with_thinking_tokens"]["correct"] += is_correct
    results["original_with_thinking_tokens"]["total"] += 1
    results["original_with_thinking_tokens"]["responses"].append({
        "task_uuid": task["task_uuid"],
        "task_category": task["task_category"],
        "question": task["prompt_message"]["content"],
        "correct_answer": task["answer"],
        "model_response": response,
        "is_correct": is_correct,
        "num_tokens": num_tokens
    })

# %% Print overall results
for model_name, model_results in results.items():
    accuracy = model_results["correct"] / model_results["total"]
    print(f"\n{model_name.capitalize()} Model Results:")
    print(f"Overall Accuracy: {accuracy:.2%} ({model_results['correct']}/{model_results['total']})")
    
    # Calculate per-category accuracies
    category_results = {}
    for response in model_results["responses"]:
        category = response["task_category"]
        if category not in category_results:
            category_results[category] = {"correct": 0, "total": 0}
        category_results[category]["correct"] += response["is_correct"]
        category_results[category]["total"] += 1
    
    print("\nAccuracy by Category:")
    for category, stats in category_results.items():
        cat_accuracy = stats["correct"] / stats["total"]
        print(f"{category}: {cat_accuracy:.2%} ({stats['correct']}/{stats['total']})")

# %% Save results to file

output_path = f"../data/reasoning_tokens_forcing_{deepseek_model_name.split('/')[-1].lower()}.json"
with open(output_path, "w") as f:
    json.dump({
        "deepseek_model": deepseek_model_name,
        "original_model": original_model_name,
        "results": results
    }, f, indent=2)

print(f"\nResults saved to {output_path}")