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
max_tokens_deepseek = 3000  # Maximum number of tokens for deepseek model
max_tokens_original = 500  # Maximum number of tokens for original model
max_tokens_forced = 2000  # Maximum number of tokens for original model with forced tokens

# Token forcing parameters
top_k_for_checking_eos = 1  # End generation if EOS token is in top-k predictions of original model
thinking_labels = ["example-testing", "uncertainty-estimation", "backtracking"]
top_k_diverging_tokens = 10  # How many top diverging tokens to force per label
top_p_predictions = 0.75  # Probability mass to consider from deepseek predictions for forcing
min_token_count = 10  # Minimum count for a token to be considered for forcing

# Experiment parameters
dataset_name = "math"  # "gsm8k" or "math"
num_tasks = 20  # Number of tasks to randomly sample
save_every_n_tasks = 20  # How often to save intermediate results
seed = 42  # Random seed for reproducibility

# Answer evaluation parameters
answer_prefixes = ["answer:", "the answer is"]  # Prefixes to look for when extracting answers

# Model configuration
model_dtype = torch.float16  # Data type for model weights
device_map = "auto"  # Device placement strategy

overwrite_evaluation_existing_results = False

# %%

# Output configuration
output_dir = "../data"
output_path = os.path.join(
    output_dir,
    f"reasoning_tokens_forcing_{deepseek_model_name.split('/')[-1].lower()}_{original_model_name.split('/')[-1].lower()}.json"
)

# %%

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

if dataset_name == "gsm8k":
    tasks_path = "../data/gsm8k.json"
else:
    tasks_path = "../data/math.json"

with open(tasks_path, "r") as f:
    tasks_dataset = json.load(f)

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

# Load next_token_and_label stats
stats_type = "next_token_and_label"
stats_path = f"../data/kl_stats/kl_stats_per_{stats_type}_{deepseek_model_id}_{original_model_id}.pkl"
try:
    with open(stats_path, 'rb') as f:
        kl_stats = pickle.load(f)
    print(f"Loaded {stats_type} KL stats from {stats_path}")
except FileNotFoundError:
    print(f"Warning: Could not find KL stats file at {stats_path}")
    kl_stats = {}

# Print some basic statistics about the loaded data
print("\nKL Stats Summary:")
print(f"Number of unique entries: {len(kl_stats)}")

# Get a few example entries
examples = list(kl_stats.items())[:3]
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
        List of tuples (token, sum_kl, count)
    """
    # Filter entries for this label and with sufficient count
    label_entries = []
    for (_, next_token, token_label), stats in stats_dict.items():
        # Remove backticks before and after tokens
        next_token = next_token.strip("`")
        if token_label == label:
            _, _, count, sum_val = stats.compute()
            if count >= min_count:
                label_entries.append((next_token, sum_val.item(), int(count)))
    
    # Sort by sum of KL divergence and take top k
    label_entries.sort(key=lambda x: x[1], reverse=True)
    return label_entries[:k]

# Get top diverging tokens for each thinking label
print("\nTop diverging token pairs per thinking label:")
for label in thinking_labels:
    print(f"\n{label}:")
    top_tokens = get_top_tokens_for_label(kl_stats, label, k=top_k_diverging_tokens)
    
    for token, kl_value, count in top_tokens:
        print(f"  `{token}`: {kl_value:.4f} (count: {count})")
    
# %% Define evaluation functions

def generate_user_message(task):
    question = task["q-str"]
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
            max_new_tokens=max_tokens_original,
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
            max_new_tokens=max_tokens_deepseek,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract the generated response (excluding the prompt)
    response_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    num_tokens = len(response_ids)
    
    return response.strip(), num_tokens


# %%
def evaluate_answer(raw_response, raw_correct_answer, model_name):
    """
    Compare the model's response to the correct answer.
    Returns True if the response contains the correct answer.
    """
    # Convert both to lowercase for case-insensitive comparison
    response = raw_response.lower()
    correct_answer = str(raw_correct_answer).lower()  # Ensure correct_answer is a string

    # Remove LaTeX from both
    stuff_to_remove = ["\\boxed", "\\text", "\\frac"]
    for stuff in stuff_to_remove:
        response = response.replace(stuff, "")
        correct_answer = correct_answer.replace(stuff, "")

    # Clean up correct answer
    correct_answer = ''.join(c for c in correct_answer if c.isalnum() or c.isspace())
    correct_answer = correct_answer.strip()
    
    for prefix in answer_prefixes:
        if prefix in response:
            # Get everything after the prefix
            response = response.split(prefix)[-1].strip()

            # Clean up the response - remove common punctuation and whitespace
            response = ''.join(c for c in response if c.isalnum() or c.isspace())
            response = response.strip()

            # Check if they match exactly
            is_correct = correct_answer in response
            return is_correct

    print(f"No answer prefix found in response of {model_name}. Expected answer: `{correct_answer}`\nResponse:\n`{raw_response}`")    
    
    return raw_correct_answer in raw_response

assert evaluate_answer("""Let's think step by step about this question:                   
                                                                                                                                                                                                                   
Step 1: Determine the total amount of the items before sales tax, which is given as $150.                                                                                                                          
                                                    
Step 2: Calculate the sales tax amount. To do this, I will multiply the total amount of the items by the sales tax rate, which is 8%. So, I will calculate 8% of $150.
                                                                                                         
Step 3: Convert the sales tax rate to a decimal by dividing it by 100, which gives me 0.08.

Step 4: Multiply the total amount of the items, $150, by the sales tax rate, 0.08, to find the sales tax amount: $150 * 0.08 = $12.

Step 5: Add the sales tax amount to the total amount of the items before sales tax to find the total amount Pauline will spend: $150 + $12 = $162.

I have now calculated the total amount, including sales tax, that Pauline will spend on all the items. I can now provide my final answer:

Answer: $162""", "162", "original")

# %%

def save_results(results, deepseek_model_name, original_model_name, output_dir="../data"):
    """
    Save experiment results to a file, creating the output directory if it doesn't exist.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add experiment parameters to results
    experiment_params = {
        # Model parameters
        "deepseek_model": deepseek_model_name,
        "original_model": original_model_name,
        "model_dtype": str(model_dtype),
        "device_map": device_map,
        
        # Generation parameters
        "max_tokens_deepseek": max_tokens_deepseek,
        "max_tokens_original": max_tokens_original,
        "max_tokens_forced": max_tokens_forced,
        
        # Token forcing parameters
        "top_k_for_checking_eos": top_k_for_checking_eos,
        "thinking_labels": thinking_labels,
        "top_k_diverging_tokens": top_k_diverging_tokens,
        "top_p_predictions": top_p_predictions,
        "min_token_count": min_token_count,
        
        # Experiment parameters
        "dataset_name": dataset_name,
        "num_tasks": num_tasks,
        "save_every_n_tasks": save_every_n_tasks,
        "seed": seed,
        
        # Answer evaluation parameters
        "answer_prefixes": answer_prefixes
    }
    
    # Save results using the global output_path
    with open(output_path, "w") as f:
        json.dump({
            "experiment_parameters": experiment_params,
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

def load_results(deepseek_model_name, original_model_name, output_dir="../data"):
    """
    Load existing results file if it exists.
    
    Returns:
        dict: Loaded results or None if file doesn't exist
    """
    try:
        with open(output_path, "r") as f:
            loaded = json.load(f)
            print(f"Loaded existing results from {output_path}")

            results = loaded["results"]

            if "deepseek" not in results:
                results["deepseek"] = {"correct": 0, "total": 0, "responses": []}
            if "original" not in results:
                results["original"] = {"correct": 0, "total": 0, "responses": []}
            if "original_with_thinking_tokens" not in results:
                results["original_with_thinking_tokens"] = {"correct": 0, "total": 0, "responses": []}

            return results
    except FileNotFoundError:
        print(f"No existing results found at {output_path}")
        return {
            "deepseek": {"correct": 0, "total": 0, "responses": []},
            "original": {"correct": 0, "total": 0, "responses": []},
            "original_with_thinking_tokens": {"correct": 0, "total": 0, "responses": []}
        }

# Load existing results or initialize new ones
results = load_results(deepseek_model_name, original_model_name)

# %%
all_tasks = list(tasks_dataset["problems-by-qid"].items())

# randomly sample tasks
tasks_to_evaluate = random.sample(all_tasks, num_tasks)

# %% Create modified prompting function that forces the thinking tokens

def get_all_force_tokens():
    """Get all tokens to force across all thinking labels"""
    force_tokens = {}  # token -> labels
    for label in thinking_labels:
        top_tokens = get_top_tokens_for_label(kl_stats, label, k=top_k_diverging_tokens)
        for token, _, _ in top_tokens:
            if token not in force_tokens:
                force_tokens[token] = [label]
            else:
                force_tokens[token].append(label)
    return force_tokens

def generate_thinking_model_response_with_forcing(model, tokenizer, task):
    """
    Generate a response with token forcing based on deepseek model preferences.
    Uses token-by-token generation to check and potentially force specific tokens.
    Forces a token if any of deepseek's top-p predictions is a forcing token.
    
    Returns:
        tuple: (response, num_tokens, forced_tokens_info)
        where forced_tokens_info is a list of dicts containing info about each forced token
    """
    # Get all tokens to watch for
    force_tokens = get_all_force_tokens()
    forced_tokens_info = []  # Track info about forced tokens

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

    for token_pos in range(max_tokens_forced):
        # Get next token distribution from original model
        with torch.no_grad():
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[0, -1, :]
            original_next_token_id = torch.argmax(next_token_logits).item()

            # Get top k tokens and their probabilities from original model
            original_probs = torch.softmax(next_token_logits, dim=0)
            top_probs, top_indices = torch.topk(original_probs, top_k_for_checking_eos)

            # print("\nTop 10 original model predictions:")
            # for j, (token_idx, prob) in enumerate(zip(top_indices, top_probs)):
            #     token = tokenizer.decode(token_idx, skip_special_tokens=False)
            #     print(f"{j+1}. `{token}` (p={prob:.4f})")

            # Check if EOS token is in top k predictions
            if tokenizer.eos_token_id in top_indices:
                print("EOS token found in top k predictions - ending generation")
                generated_ids = torch.cat([generated_ids, torch.tensor([[tokenizer.eos_token_id]]).to(model.device)], dim=1)
                break

            # Create a temporary tensor with the next token to check completion conditions
            temp_ids = torch.cat([generated_ids[0, input_ids.shape[1]:], torch.tensor([original_next_token_id]).to(model.device)])
            response_so_far_with_original_token = tokenizer.decode(temp_ids, skip_special_tokens=False)

            # Check if we've completed generating the answer
            if "Answer: " in response_so_far_with_original_token:
                answer_pos = response_so_far_with_original_token.find("Answer: ") + len("Answer: ")
                if "\n" in response_so_far_with_original_token[answer_pos:]:
                    print("Answer found - ending generation")
                    generated_ids = torch.cat([generated_ids, torch.tensor([[original_next_token_id]]).to(model.device)], dim=1)
                    break

            original_next_token = tokenizer.decode(original_next_token_id, skip_special_tokens=False)

            # Calculate probability of original token
            original_token_prob = original_probs[original_next_token_id].item()

        check_forcing = True
        if response_so_far_with_original_token.endswith("Answer") or \
            response_so_far_with_original_token.endswith("Answer:") or \
            response_so_far_with_original_token.endswith("Answer: "):
            # Model is about to generate the answer, so we don't need to force any more tokens
            check_forcing = False
        
        forced_token = False
        if check_forcing:
            # Get deepseek model's top-p predictions
            with torch.no_grad():
                deepseek_outputs = deepseek_model(deepseek_input_ids)
                deepseek_next_token_logits = deepseek_outputs.logits[0, -1, :]
                deepseek_probs = torch.softmax(deepseek_next_token_logits, dim=0)
                
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(deepseek_probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                # Find indices where cumsum is less than top_p
                nucleus_mask = cumsum_probs <= top_p_predictions
                # Include the first probability after top_p to ensure we don't cut off mid-word
                if not nucleus_mask.any():
                    nucleus_mask[0] = True
                else:
                    nucleus_mask[torch.where(nucleus_mask)[0][-1] + 1] = True
                
                # Get the indices and probabilities within the nucleus
                nucleus_indices = sorted_indices[nucleus_mask]
                
            # Check each prediction in the nucleus
            for pred_idx, deepseek_next_token_id in enumerate(nucleus_indices):
                deepseek_next_token_id = deepseek_next_token_id.item()
                deepseek_next_token = deepseek_tokenizer.decode(deepseek_next_token_id, skip_special_tokens=False)
                
                # Check if this prediction is in our forcing set
                if deepseek_next_token in force_tokens:
                    # Only force if it's different from what the original model would do
                    if deepseek_next_token != original_next_token:
                        # Find the probability the original model assigns to the forced token
                        # First find the token ID in the original model's vocabulary
                        forced_token_id = tokenizer.encode(deepseek_next_token, add_special_tokens=False)[0]
                        forced_token_prob = original_probs[forced_token_id].item()
                        
                        # Store forcing event information
                        forced_tokens_info.append({
                            "labels": force_tokens[deepseek_next_token],
                            "forced_token": deepseek_next_token,
                            "position": token_pos,
                            "original_next_token": original_next_token,
                            "deepseek_prediction_rank": pred_idx + 1,  # 1-based rank
                            "deepseek_prediction_probability": torch.softmax(deepseek_next_token_logits, dim=0)[deepseek_next_token_id].item(),
                            "original_model_forced_token_prob": forced_token_prob,
                            "original_model_original_token_prob": original_token_prob
                        })
                        
                        # Log the forcing event
                        response_so_far = tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
                        
                        print(f"\n### Forcing token in task: {task_id}")
                        print(f"Response so far: `{response_so_far}`")
                        print(f"Labels forcing token: {force_tokens[deepseek_next_token]}")
                        print(f"Forced token: `{deepseek_next_token}`")
                        print(f"Original model would have generated: `{original_next_token}`")
                        print(f"Token was prediction #{pred_idx + 1} with probability {forced_tokens_info[-1]['deepseek_prediction_probability']:.4f}")
                        print(f"Original model probability of forced token: {forced_token_prob:.4f}")
                        print(f"Original model probability of its preferred token: {original_token_prob:.4f}")
                        print("-" * 80)
                        
                        next_token_id = torch.tensor([[deepseek_next_token_id]]).to(model.device)
                        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                        deepseek_input_ids = torch.cat([deepseek_input_ids, next_token_id], dim=1)
                        forced_token = True
                        break
        
        if not forced_token:
            # If no forcing occurred, continue with original model's prediction
            next_token_id = torch.tensor([[original_next_token_id]]).to(model.device)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            deepseek_input_ids = torch.cat([deepseek_input_ids, next_token_id], dim=1)

    # Extract the generated response (excluding the prompt)
    response_ids = generated_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=False)
    num_tokens = len(response_ids)
    
    return response.strip(), num_tokens, forced_tokens_info

# %% Generate responses if needed
models_with_new_results = []
if len(results["deepseek"]["responses"]) > 0:
    print("Skipping deepseek model generation - results already exist")
else:
    print("Generating deepseek model responses...")
    models_with_new_results.append("deepseek")
    for i, (task_id, task) in enumerate(tqdm(tasks_to_evaluate)):
        expected_answer = task["answer-without-reasoning"]
        deepseek_response, deepseek_num_tokens = generate_thinking_model_response(deepseek_model, deepseek_tokenizer, task)
        
        # Store results without evaluation
        results["deepseek"]["total"] += 1
        results["deepseek"]["responses"].append({
            "task_uuid": task_id,
            "question": task["q-str"],
            "correct_answer": expected_answer,
            "model_response": deepseek_response,
            "is_correct": None,  # Will be evaluated later
            "num_tokens": deepseek_num_tokens
        })
        
        # Save partial results
        if (i + 1) % save_every_n_tasks == 0:
            save_results(results, deepseek_model_name, original_model_name)

if len(results["original"]["responses"]) > 0:
    print("Skipping original model generation - results already exist")
else:
    print("Generating original model responses...")
    models_with_new_results.append("original")
    for i, (task_id, task) in enumerate(tqdm(tasks_to_evaluate)):
        expected_answer = task["answer-without-reasoning"]
        response, num_tokens = generate_original_model_response(original_model, original_tokenizer, task)
        
        # Store results without evaluation
        results["original"]["total"] += 1
        results["original"]["responses"].append({
            "task_uuid": task_id,
            "question": task["q-str"],
            "correct_answer": expected_answer,
            "model_response": response,
            "is_correct": None,  # Will be evaluated later
            "num_tokens": num_tokens
        })
        
        # Save partial results
        if (i + 1) % save_every_n_tasks == 0:
            save_results(results, deepseek_model_name, original_model_name)

if len(results["original_with_thinking_tokens"]["responses"]) > 0:
    print("Skipping forced thinking generation - results already exist")
else:
    print("\nGenerating original model responses with forced thinking tokens...")
    models_with_new_results.append("original_with_thinking_tokens")
    for i, (task_id, task) in enumerate(tqdm(tasks_to_evaluate)):
        expected_answer = task["answer-without-reasoning"]
        response, num_tokens, forced_tokens_info = generate_thinking_model_response_with_forcing(
            original_model, 
            original_tokenizer, 
            task
        )

        print(f"### Task {task_id}")
        print(f"Generated response: `{response}`")
        print(f"Expected answer: `{expected_answer}`")
        print("-" * 80)

        # Store results without evaluation
        results["original_with_thinking_tokens"]["total"] += 1
        results["original_with_thinking_tokens"]["responses"].append({
            "task_uuid": task_id,
            "question": task["q-str"],
            "correct_answer": expected_answer,
            "model_response": response,
            "is_correct": None,  # Will be evaluated later
            "num_tokens": num_tokens,
            "forced_tokens_info": forced_tokens_info
        })
        
        # Save partial results
        if (i + 1) % save_every_n_tasks == 0:
            save_results(results, deepseek_model_name, original_model_name)

# %% Evaluate all responses
print("\nEvaluating all responses...")

# Check if we should evaluate
should_evaluate = overwrite_evaluation_existing_results
if not should_evaluate:
    # Check if any responses need evaluation
    for model_name, model_results in results.items():
        if model_name in models_with_new_results:
            should_evaluate = True
            break

        for response_data in model_results["responses"]:
            if response_data["is_correct"] is None:
                should_evaluate = True
                break

        if should_evaluate:
            break

if not should_evaluate:
    print("Skipping evaluation - results already exist and overwrite_evaluation_existing_results=False")
else:
    # Evaluate each model's responses
    for model_name, model_results in results.items():
        print(f"\nEvaluating {model_name} responses...")
        for response_data in tqdm(model_results["responses"]):
            # Only evaluate if overwriting or not yet evaluated
            if overwrite_evaluation_existing_results or \
                model_name in models_with_new_results or \
                response_data["is_correct"] is None:

                is_correct = evaluate_answer(
                    response_data["model_response"], 
                    response_data["correct_answer"], 
                    model_name
                )
                response_data["is_correct"] = is_correct

    # Save results after evaluation
    save_results(results, deepseek_model_name, original_model_name)

# %% Reset correctness counters
for model_name, model_results in results.items():
    model_results["correct"] = 0
    model_results["total"] = len(model_results["responses"])

    for response_data in model_results["responses"]:
        if response_data["is_correct"]:
            model_results["correct"] += 1

save_results(results, deepseek_model_name, original_model_name)

# %% Print overall results
for model_name, model_results in results.items():
    accuracy = model_results["correct"] / model_results["total"]
    print(f"\n{model_name.capitalize()} Model Results:")
    print(f"Overall Accuracy: {accuracy:.2%} ({model_results['correct']}/{model_results['total']})")