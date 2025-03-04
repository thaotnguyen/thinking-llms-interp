# %%
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import dotenv
dotenv.load_dotenv(".env")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import random
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from tiny_dashboard.visualization_utils import activation_visualization
import openai
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# %% Set model names and parameters
deepseek_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
original_model_name = "Qwen/Qwen2.5-14B"
# original_model_name = "Qwen/Qwen2.5-14B-Instruct"

# Generation parameters
max_tokens_deepseek = 3000  # Maximum number of tokens for deepseek model
max_tokens_original = 500  # Maximum number of tokens for original model
max_tokens_forced = 3000  # Maximum number of tokens for original model with forced tokens

# Token forcing parameters
top_k_for_checking_eos = 1  # End generation if EOS token is in top-k predictions of original model
thinking_labels = ["example-testing", "uncertainty-estimation", "backtracking"]
top_k_diverging_tokens = 10  # How many top diverging tokens to force per label
top_p_predictions = 0.4  # Probability mass to consider from deepseek predictions for forcing
min_token_count = 10  # Minimum count for a token to be considered for forcing

# Experiment parameters
dataset_name = "math"  # "gsm8k" or "math"
num_tasks = 500  # Number of tasks to randomly sample
save_every_n_tasks = 20  # How often to save intermediate results
seed = 42  # Random seed for reproducibility

# Answer evaluation parameters
answer_prefixes = ["answer:", "the answer is"]  # Prefixes to look for when extracting answers

# Model configuration
model_dtype = torch.float16  # Data type for model weights
device_map = "auto"  # Device placement strategy

overwrite_evaluation_existing_results = False

tasks_where_forced_thinking_did_not_help = {'math_train_counting_and_probability_93', 'math_train_counting_and_probability_1', 'math_test_geometry_52', 'math_test_number_theory_495', 'math_test_algebra_867', 'math_test_algebra_829', 'math_train_precalculus_100', 'math_test_geometry_305', 'math_train_counting_and_probability_644', 'math_train_algebra_1483', 'math_test_counting_and_probability_239', 'math_test_algebra_773', 'math_train_number_theory_218', 'math_test_intermediate_algebra_230', 'math_test_counting_and_probability_141', 'math_train_algebra_1513', 'math_train_algebra_57', 'math_test_counting_and_probability_210', 'math_train_precalculus_735', 'math_train_intermediate_algebra_589', 'math_train_precalculus_291', 'math_test_intermediate_algebra_499', 'math_train_intermediate_algebra_994', 'math_test_counting_and_probability_1', 'math_train_intermediate_algebra_1282', 'math_test_prealgebra_123', 'math_train_intermediate_algebra_194', 'math_train_algebra_1121', 'math_train_intermediate_algebra_652', 'math_train_precalculus_271', 'math_train_counting_and_probability_266', 'math_train_intermediate_algebra_116', 'math_test_counting_and_probability_293', 'math_train_algebra_1162', 'math_test_precalculus_105', 'math_test_prealgebra_808', 'math_test_precalculus_244', 'math_train_number_theory_692', 'math_train_prealgebra_1086', 'math_train_intermediate_algebra_454', 'math_test_geometry_119', 'math_test_precalculus_412', 'math_train_number_theory_330', 'math_test_geometry_292', 'math_test_intermediate_algebra_101', 'math_test_geometry_22'}

tasks_where_forced_thinking_helped = {'math_train_counting_and_probability_93', 'math_train_counting_and_probability_1', 'math_test_geometry_52', 'math_test_number_theory_495', 'math_test_algebra_867', 'math_test_algebra_829', 'math_train_precalculus_100', 'math_test_geometry_305', 'math_train_counting_and_probability_644', 'math_train_algebra_1483', 'math_test_counting_and_probability_239', 'math_test_algebra_773', 'math_train_number_theory_218', 'math_test_intermediate_algebra_230', 'math_test_counting_and_probability_141', 'math_train_algebra_1513', 'math_train_algebra_57', 'math_test_counting_and_probability_210', 'math_train_precalculus_735', 'math_train_intermediate_algebra_589', 'math_train_precalculus_291', 'math_test_intermediate_algebra_499', 'math_train_intermediate_algebra_994', 'math_test_counting_and_probability_1', 'math_train_intermediate_algebra_1282', 'math_test_prealgebra_123', 'math_train_intermediate_algebra_194', 'math_train_algebra_1121', 'math_train_intermediate_algebra_652', 'math_train_precalculus_271', 'math_train_counting_and_probability_266', 'math_train_intermediate_algebra_116', 'math_test_counting_and_probability_293', 'math_train_algebra_1162', 'math_test_precalculus_105', 'math_test_prealgebra_808', 'math_test_precalculus_244', 'math_train_number_theory_692', 'math_train_prealgebra_1086', 'math_train_intermediate_algebra_454', 'math_test_geometry_119', 'math_test_precalculus_412', 'math_train_number_theory_330', 'math_test_geometry_292', 'math_test_intermediate_algebra_101', 'math_test_geometry_22'}

tasks_where_forced_thinking_hurt = {'math_test_algebra_402', 'math_train_counting_and_probability_335', 'math_test_prealgebra_810', 'math_train_prealgebra_761', 'math_test_number_theory_519', 'math_train_algebra_686', 'math_train_counting_and_probability_719', 'math_test_prealgebra_165', 'math_test_algebra_147', 'math_test_prealgebra_359', 'math_train_prealgebra_794', 'math_train_prealgebra_981', 'math_test_algebra_1005', 'math_train_prealgebra_714', 'math_train_prealgebra_349', 'math_test_prealgebra_78', 'math_test_prealgebra_120', 'math_test_prealgebra_557', 'math_train_number_theory_75', 'math_train_prealgebra_858', 'math_train_algebra_204', 'math_test_algebra_1167', 'math_test_geometry_198', 'math_train_algebra_921', 'math_test_algebra_1112', 'math_train_intermediate_algebra_193', 'math_train_prealgebra_592', 'math_train_intermediate_algebra_65', 'math_train_prealgebra_298', 'math_train_prealgebra_932', 'math_train_precalculus_535', 'math_train_algebra_157', 'math_train_prealgebra_450', 'math_test_prealgebra_503', 'math_train_intermediate_algebra_1271', 'math_test_intermediate_algebra_555', 'math_train_intermediate_algebra_80', 'math_test_prealgebra_553', 'math_train_algebra_1396', 'math_test_intermediate_algebra_504', 'math_train_algebra_116', 'math_train_intermediate_algebra_893', 'math_train_number_theory_546', 'math_train_counting_and_probability_385', 'math_train_geometry_150', 'math_train_prealgebra_18', 'math_train_algebra_818', 'math_test_number_theory_325', 'math_test_intermediate_algebra_807', 'math_train_prealgebra_809', 'math_test_prealgebra_364', 'math_train_precalculus_105', 'math_train_intermediate_algebra_557', 'math_train_counting_and_probability_687', 'math_train_number_theory_160', 'math_train_counting_and_probability_652', 'math_train_algebra_1553', 'math_train_number_theory_289', 'math_test_geometry_245', 'math_test_counting_and_probability_11', 'math_train_intermediate_algebra_328', 'math_test_algebra_516', 'math_train_algebra_602', 'math_train_number_theory_739', 'math_test_intermediate_algebra_817', 'math_train_number_theory_519', 'math_test_number_theory_470', 'math_train_intermediate_algebra_564', 'math_train_algebra_1721', 'math_test_intermediate_algebra_152', 'math_test_counting_and_probability_391', 'math_test_number_theory_99', 'math_test_prealgebra_425', 'math_test_algebra_1029', 'math_test_algebra_906', 'math_train_algebra_517', 'math_train_prealgebra_136', 'math_test_algebra_217', 'math_train_counting_and_probability_44', 'math_train_number_theory_369', 'math_train_counting_and_probability_274', 'math_train_prealgebra_254', 'math_test_counting_and_probability_93', 'math_test_algebra_824'}

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
def evaluate_answer(task_uuid, raw_response, raw_correct_answer, model_name):
    """
    Use GPT-4 to evaluate if the model's response contains the correct answer.
    
    Args:
        raw_response: The model's complete response
        raw_correct_answer: The expected correct answer
        model_name: Name of the model being evaluated (for logging)
    
    Returns:
        tuple: (is_correct: bool, explanation: str | None)
        where is_correct is True if the response is deemed correct, False otherwise
        and explanation is the LLM's explanation or None if parsing failed
    """
    task = tasks_dataset["problems-by-qid"][task_uuid]
    # Construct the prompt for GPT-4
    evaluation_prompt = f"""Please evaluate if the following response arrives at the correct answer.

Question: `{task["q-str"]}`
Ground truth answer: `{raw_correct_answer}`
Response to evaluate: `{raw_response}`

Please provide your evaluation in the following format:
<explanation>Your detailed explanation of whether the response arrives at the correct answer</explanation>
<correct>YES/NO/UNKNOWN</correct>"""

    try:
        # Call GPT-4 for evaluation
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise mathematical answer evaluator. Your task is to determine if a given response arrives at the correct answer, regardless of the reasoning path taken."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0
        )
        
        # Extract the evaluation result
        evaluation = response.choices[0].message.content
        
        # Parse the XML-like tags
        explanation_start = evaluation.find("<explanation>") + len("<explanation>")
        explanation_end = evaluation.find("</explanation>")
        correct_start = evaluation.find("<correct>") + len("<correct>")
        correct_end = evaluation.find("</correct>")
        
        if correct_start == -1 or correct_end == -1:
            print(f"Error parsing GPT-4 response for {model_name}. Response: {evaluation}")
            return False, None
            
        explanation = evaluation[explanation_start:explanation_end].strip()
        correct_str = evaluation[correct_start:correct_end].strip().upper()
        
        # Map the response to boolean
        if correct_str == "YES":
            return True, explanation
        elif correct_str == "NO":
            return False, explanation
        else:
            print(f"Unclear evaluation from GPT-4 for {model_name}. Response: {evaluation}")
            return False, explanation
            
    except Exception as e:
        print(f"Error during GPT-4 evaluation for {model_name}: {str(e)}")
        return False, None

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

def load_results():
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
results = load_results()

# %%
all_tasks = list(tasks_dataset["problems-by-qid"].items())

# tasks_to_evaluate = [t for t in all_tasks if t[0] in tasks_where_forced_thinking_did_not_help]
# tasks_to_evaluate = [t for t in all_tasks if t[0] == "math_train_prealgebra_298"]

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
    Also forces tokens based on newline handling rules:
    - If deepseek's top prediction has newline, force it (unless original has same newline)
    - If original's top prediction has newline, force deepseek's top prediction (unless deepseek has same newline)
    
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
    
    # Initialize past key values for both models
    with torch.no_grad():
        # Get initial past key values for original model
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[0, -1, :]

        # Get initial past key values for deepseek model
        deepseek_outputs = deepseek_model(deepseek_input_ids, use_cache=True)
        deepseek_past_key_values = deepseek_outputs.past_key_values
        deepseek_next_token_logits = deepseek_outputs.logits[0, -1, :]

    # Initialize generated sequences
    generated_ids = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    deepseek_attention_mask = torch.ones_like(deepseek_input_ids)

    for token_pos in range(max_tokens_forced):
        # Get next token distribution from original model using cached values
        with torch.no_grad():
            # Get next token from original model using past key values
            original_probs = torch.softmax(next_token_logits, dim=0)
            original_next_token_id = torch.argmax(next_token_logits).item()
            original_next_token = tokenizer.decode(original_next_token_id, skip_special_tokens=False)

            # Get top k tokens and their probabilities from original model
            top_probs, top_indices = torch.topk(original_probs, top_k_for_checking_eos)

            # print("\nTop 10 original model predictions:")
            # for j, (token_idx, prob) in enumerate(zip(top_indices, top_probs)):
            #     token = tokenizer.decode(token_idx, skip_special_tokens=False)
            #     print(f"{j+1}. `{token}` (p={prob:.4f})")

            # Check if EOS token is in top k predictions
            if tokenizer.eos_token_id in top_indices:
                print("EOS token found in top k predictions - ending generation")
                break

            # Create a temporary tensor with the next token to check completion conditions
            temp_ids = torch.cat([generated_ids[0, input_ids.shape[1]:], torch.tensor([original_next_token_id]).to(model.device)])
            response_so_far_with_original_token = tokenizer.decode(temp_ids, skip_special_tokens=False)

            # Check if we've completed generating the answer
            if "Answer: " in response_so_far_with_original_token:
                answer_pos = response_so_far_with_original_token.find("Answer: ") + len("Answer: ")
                if "\n" in response_so_far_with_original_token[answer_pos:]:
                    print("Answer found - ending generation")
                    break

            # Calculate probability of original token
            original_token_prob = original_probs[original_next_token_id].item()

            # Get deepseek model's top prediction
            deepseek_probs = torch.softmax(deepseek_next_token_logits, dim=0)
            deepseek_next_token_id = torch.argmax(deepseek_next_token_logits).item()
            deepseek_next_token = deepseek_tokenizer.decode(deepseek_next_token_id, skip_special_tokens=False)
        
            # Create a temporary tensor with deepseek's next token to check if it would start the answer
            temp_ids = torch.cat([generated_ids[0, input_ids.shape[1]:], torch.tensor([deepseek_next_token_id]).to(model.device)])
            response_so_far_with_deepseek_token = tokenizer.decode(temp_ids, skip_special_tokens=False)

        check_forcing = True
        if response_so_far_with_original_token.endswith("Answer") or \
            response_so_far_with_original_token.endswith("Answer:") or \
            response_so_far_with_original_token.endswith("Answer: "):
            # Model is about to generate the answer, so we don't need to force any more tokens
            check_forcing = False

        forced_token = False
        if check_forcing:
            if response_so_far_with_deepseek_token.endswith("Answer") or \
                response_so_far_with_deepseek_token.endswith("Answer:") or \
                response_so_far_with_deepseek_token.endswith("Answer: "):
                # Force deepseek's token since it would start the answer
                forced_tokens_info.append({
                    "labels": ["deepseek-starting-answer"],
                    "forced_token": deepseek_next_token,
                    "position": token_pos,
                    "original_next_token": original_next_token,
                    "deepseek_prediction_rank": 1,
                    "deepseek_prediction_probability": deepseek_probs[deepseek_next_token_id].item(),
                    "original_model_forced_token_prob": original_probs[deepseek_next_token_id].item(),
                    "original_model_original_token_prob": original_token_prob
                })

                print(f"\n### Forcing token in task: {task_id}")
                print("Reason: deepseek model would start the answer")
                print(f"Response so far: `{response_so_far_with_original_token}`")
                print(f"Forced token: `{deepseek_next_token}`")
                print(f"Original model would have generated: `{original_next_token}`")
                print(f"Token was prediction #{1} with probability {deepseek_probs[deepseek_next_token_id].item():.4f}")
                print(f"Original model probability of forced token: {original_probs[deepseek_next_token_id].item():.4f}")
                print(f"Original model probability of its preferred token: {original_token_prob:.4f}")
                print("-" * 80)

                next_token_id = torch.tensor([[deepseek_next_token_id]]).to(model.device)
                forced_token = True

        if not forced_token:
            # Check if the deepseek model wants to end thinking
            if deepseek_next_token == "</think>":
                print("Deepseek model wants to end thinking")
                print(f"Response so far: `{response_so_far_with_original_token}`")
                print(f"Forced token: `{deepseek_next_token}`")
                print(f"Original model would have generated: `{original_next_token}`")
                print("-" * 80)
                break
                
        if not forced_token:
            # First check newline forcing rules
            deepseek_has_newline = "\n" in deepseek_next_token
            original_has_newline = "\n" in original_next_token
            
            # Only skip newline forcing if both have same newline token
            if not (deepseek_has_newline and original_has_newline and deepseek_next_token == original_next_token):
                if deepseek_has_newline:
                    # Force deepseek's newline token
                    forced_tokens_info.append({
                        "labels": ["deepseek-model-newline"],
                        "forced_token": deepseek_next_token,
                        "position": token_pos,
                        "original_next_token": original_next_token,
                        "deepseek_prediction_rank": 1,
                        "deepseek_prediction_probability": deepseek_probs[deepseek_next_token_id].item(),
                        "original_model_forced_token_prob": original_probs[deepseek_next_token_id].item(),
                        "original_model_original_token_prob": original_token_prob
                    })

                    print(f"\n### Forcing token in task: {task_id}")
                    print("Reason: deepseek model would have generated a newline")
                    print(f"Response so far: `{response_so_far_with_original_token}`")
                    print(f"Forced token: `{deepseek_next_token}`")
                    print(f"Original model would have generated: `{original_next_token}`")
                    print(f"Token was prediction #{1} with probability {deepseek_probs[deepseek_next_token_id].item():.4f}")
                    print(f"Original model probability of forced token: {original_probs[deepseek_next_token_id].item():.4f}")
                    print(f"Original model probability of its preferred token: {original_token_prob:.4f}")
                    print("-" * 80)
                    
                    next_token_id = torch.tensor([[deepseek_next_token_id]]).to(model.device)
                    forced_token = True
                elif original_has_newline:
                    # Force deepseek's top prediction (even if not a forcing token)
                    forced_tokens_info.append({
                        "labels": ["original-model-newline"],
                        "forced_token": deepseek_next_token,
                        "position": token_pos,
                        "original_next_token": original_next_token,
                        "deepseek_prediction_rank": 1,
                        "deepseek_prediction_probability": deepseek_probs[deepseek_next_token_id].item(),
                        "original_model_forced_token_prob": original_probs[deepseek_next_token_id].item(),
                        "original_model_original_token_prob": original_token_prob
                    })

                    print(f"\n### Forcing token in task: {task_id}")
                    print("Reason: original model would have generated a newline")
                    print(f"Response so far: `{response_so_far_with_original_token}`")
                    print(f"Forced token: `{deepseek_next_token}`")
                    print(f"Original model would have generated: `{original_next_token}`")
                    print(f"Token was prediction #{1} with probability {deepseek_probs[deepseek_next_token_id].item():.4f}")
                    print(f"Original model probability of forced token: {original_probs[deepseek_next_token_id].item():.4f}")
                    print(f"Original model probability of its preferred token: {original_token_prob:.4f}")
                    print("-" * 80)

                    next_token_id = torch.tensor([[deepseek_next_token_id]]).to(model.device)
                    forced_token = True

            # If no newline forcing occurred, check regular forcing tokens
            if not forced_token:
                # Get deepseek model's top-p predictions using cached values
                with torch.no_grad():
                    # Sort probabilities in descending order
                    sorted_probs, sorted_indices = torch.sort(deepseek_probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                    # Find indices where cumsum is less than top_p
                    nucleus_mask = cumsum_probs <= top_p_predictions
                    # Include the first probability after top_p
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
                            forced_token_id = tokenizer.encode(deepseek_next_token, add_special_tokens=False)[0]
                            forced_token_prob = original_probs[forced_token_id].item()
                            
                            # Store forcing event information
                            forced_tokens_info.append({
                                "labels": force_tokens[deepseek_next_token],
                                "forced_token": deepseek_next_token,
                                "position": token_pos,
                                "original_next_token": original_next_token,
                                "deepseek_prediction_rank": pred_idx + 1,
                                "deepseek_prediction_probability": deepseek_probs[deepseek_next_token_id].item(),
                                "original_model_forced_token_prob": forced_token_prob,
                                "original_model_original_token_prob": original_token_prob
                            })
                            
                            # Log the forcing event
                            response_so_far = tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
                            
                            print(f"\n### Forcing token in task: {task_id}")
                            print("Reason: reasoning token in top-p deepseek predictions")
                            print(f"Response so far: `{response_so_far}`")
                            print(f"Labels forcing token: {force_tokens[deepseek_next_token]}")
                            print(f"Forced token: `{deepseek_next_token}`")
                            print(f"Original model would have generated: `{original_next_token}`")
                            print(f"Token was prediction #{pred_idx + 1} with probability {forced_tokens_info[-1]['deepseek_prediction_probability']:.4f}")
                            print(f"Original model probability of forced token: {forced_token_prob:.4f}")
                            print(f"Original model probability of its preferred token: {original_token_prob:.4f}")
                            print("-" * 80)
                            
                            next_token_id = torch.tensor([[forced_token_id]]).to(model.device)
                            forced_token = True
                            break

        if not forced_token:
            # If no forcing occurred, use original model's prediction
            next_token_id = torch.tensor([[original_next_token_id]]).to(model.device)

        # Update generated sequences and attention masks
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)
        deepseek_input_ids = torch.cat([deepseek_input_ids, next_token_id], dim=1)
        deepseek_attention_mask = torch.cat([deepseek_attention_mask, torch.ones_like(next_token_id)], dim=1)

        # Update past key values for both models
        with torch.no_grad():
            # Update original model's cache
            outputs = model(
                next_token_id,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[0, -1, :]

            # Update deepseek model's cache
            deepseek_outputs = deepseek_model(
                next_token_id,
                attention_mask=deepseek_attention_mask,
                past_key_values=deepseek_past_key_values,
                use_cache=True
            )
            deepseek_past_key_values = deepseek_outputs.past_key_values
            deepseek_next_token_logits = deepseek_outputs.logits[0, -1, :]

    # Extract the generated response (excluding the prompt)
    response_ids = generated_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=False)
    num_tokens = len(response_ids)
    
    return response.strip(), num_tokens, forced_tokens_info

# %% Generate responses if needed
def get_tasks_needing_evaluation(existing_responses, tasks_to_evaluate):
    """
    Returns a list of tasks that need evaluation by comparing against existing responses.
    
    Args:
        existing_responses: List of response dictionaries from results
        tasks_to_evaluate: List of (task_id, task) tuples to check
        
    Returns:
        List of (task_id, task) tuples that need evaluation
    """
    # Build set of task IDs that already have responses
    existing_task_ids = {r["task_uuid"] for r in existing_responses}
    
    # Filter tasks_to_evaluate to only include tasks not yet evaluated
    return [(task_id, task) for task_id, task in tasks_to_evaluate 
            if task_id not in existing_task_ids]

# For each model, get list of tasks needing evaluation
tasks_for_deepseek = get_tasks_needing_evaluation(
    results["deepseek"]["responses"], 
    tasks_to_evaluate
)
tasks_for_original = get_tasks_needing_evaluation(
    results["original"]["responses"], 
    tasks_to_evaluate
)
tasks_for_forced = get_tasks_needing_evaluation(
    results["original_with_thinking_tokens"]["responses"], 
    tasks_to_evaluate
)

print(f"\nTasks needing evaluation:")
print(f"Deepseek model: {len(tasks_for_deepseek)} tasks")
print(f"Original model: {len(tasks_for_original)} tasks")
print(f"Forced thinking: {len(tasks_for_forced)} tasks")

if tasks_for_deepseek:
    print("\nGenerating deepseek model responses...")
    for i, (task_id, task) in enumerate(tqdm(tasks_for_deepseek)):
        expected_answer = task["answer-without-reasoning"]
        deepseek_response, deepseek_num_tokens = generate_thinking_model_response(
            deepseek_model, 
            deepseek_tokenizer, 
            task
        )
        
        # Store results without evaluation
        results["deepseek"]["total"] += 1
        results["deepseek"]["responses"].append({
            "task_uuid": task_id,
            "question": task["q-str"],
            "correct_answer": expected_answer,
            "model_response": deepseek_response,
            "is_correct": None,  # Will be evaluated later
            "is_correct_explanation": None,  # Will be evaluated later
            "num_tokens": deepseek_num_tokens
        })
        
        # Save partial results
        if (i + 1) % save_every_n_tasks == 0:
            save_results(results, deepseek_model_name, original_model_name)
else:
    print("\nNo new tasks for deepseek model")

if tasks_for_original:
    print("\nGenerating original model responses...")
    for i, (task_id, task) in enumerate(tqdm(tasks_for_original)):
        expected_answer = task["answer-without-reasoning"]
        response, num_tokens = generate_original_model_response(
            original_model, 
            original_tokenizer, 
            task
        )
        
        # Store results without evaluation
        results["original"]["total"] += 1
        results["original"]["responses"].append({
            "task_uuid": task_id,
            "question": task["q-str"],
            "correct_answer": expected_answer,
            "model_response": response,
            "is_correct": None,  # Will be evaluated later
            "is_correct_explanation": None,  # Will be evaluated later
            "num_tokens": num_tokens
        })
        
        # Save partial results
        if (i + 1) % save_every_n_tasks == 0:
            save_results(results, deepseek_model_name, original_model_name)
else:
    print("\nNo new tasks for original model")

if tasks_for_forced:
    print("\nGenerating original model responses with forced thinking tokens...")
    for i, (task_id, task) in enumerate(tqdm(tasks_for_forced)):
        expected_answer = task["answer-without-reasoning"]
        response, num_tokens, forced_tokens_info = generate_thinking_model_response_with_forcing(
            original_model, 
            original_tokenizer, 
            task
        )

        print(f"\n### Task {task_id}")
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
            "is_correct_explanation": None,  # Will be evaluated later
            "num_tokens": num_tokens,
            "forced_tokens_info": forced_tokens_info
        })
        
        # Save partial results
        if (i + 1) % save_every_n_tasks == 0:
            save_results(results, deepseek_model_name, original_model_name)
else:
    print("\nNo new tasks for forced thinking")

# %% Evaluate responses
print("\nEvaluating responses...")

# Get list of responses needing evaluation
responses_to_evaluate = []
for model_name, model_results in results.items():
    for response_data in model_results["responses"]:
        if response_data["is_correct"] is None or overwrite_evaluation_existing_results:
            responses_to_evaluate.append((model_name, response_data))

if responses_to_evaluate:
    print(f"Evaluating {len(responses_to_evaluate)} responses...")
    for model_name, response_data in tqdm(responses_to_evaluate):
        is_correct, explanation = evaluate_answer(
            response_data["task_uuid"],
            response_data["model_response"],
            response_data["correct_answer"],
            model_name
        )
        response_data["is_correct"] = is_correct
        response_data["is_correct_explanation"] = explanation

    # Save results after evaluation
    save_results(results, deepseek_model_name, original_model_name)
else:
    print("No responses need evaluation")

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

# %%

# Get sets of correctly answered tasks for each model
deepseek_correct = {r["task_uuid"] for r in results["deepseek"]["responses"] if r["is_correct"]}
base_correct = {r["task_uuid"] for r in results["original"]["responses"] if r["is_correct"]}
forced_correct = {r["task_uuid"] for r in results["original_with_thinking_tokens"]["responses"] if r["is_correct"]}

# Get all task IDs
all_tasks = {r["task_uuid"] for r in results["deepseek"]["responses"]}

# Create sets for incorrect answers
deepseek_wrong = all_tasks - deepseek_correct
base_wrong = all_tasks - base_correct
forced_wrong = all_tasks - forced_correct

show_task_ids = False

print("Analysis of all possible combinations:")
print("True = correct, False = wrong\n")

# All 8 possible combinations
combinations = {
    (True, True, True): deepseek_correct & base_correct & forced_correct,
    (True, True, False): deepseek_correct & base_correct & forced_wrong,
    (True, False, True): deepseek_correct & base_wrong & forced_correct,
    (True, False, False): deepseek_correct & base_wrong & forced_wrong,
    (False, True, True): deepseek_wrong & base_correct & forced_correct,
    (False, True, False): deepseek_wrong & base_correct & forced_wrong,
    (False, False, True): deepseek_wrong & base_wrong & forced_correct,
    (False, False, False): deepseek_wrong & base_wrong & forced_wrong
}

# Print results in a formatted table
if show_task_ids:
    print(f"{'Deep':5} | {'Base':5} | {'Force':5} | {'Count':6} | {'Task IDs'}")
    print("-" * 70)
else:
    print(f"{'Deep':5} | {'Base':5} | {'Force':5} | {'Count':6}")
    print("-" * 30)

for (d, b, f), task_set in combinations.items():
    if show_task_ids:
        print(f"{str(d):5} | {str(b):5} | {str(f):5} | {len(task_set):6d} | {task_set if task_set else 'None'}")
    else:
        print(f"{str(d):5} | {str(b):5} | {str(f):5} | {len(task_set):6d}")

# Verify that all tasks are accounted for
total_from_combinations = sum(len(s) for s in combinations.values())
print(f"\nVerification - Total tasks: {len(all_tasks)}")
print(f"Sum of all combinations: {total_from_combinations}")
assert total_from_combinations == len(all_tasks), "Error: Not all tasks accounted for!"

# Print some interesting insights
print("\nKey insights:")
print(f"Tasks all models got correct: {len(combinations[(True, True, True)])}")
print(f"Tasks all models got wrong: {len(combinations[(False, False, False)])}")

deepseek_correct_tasks = combinations[(True, True, True)] | combinations[(True, True, False)] | combinations[(True, False, True)] | combinations[(True, False, False)]
print(f"\nTasks Deepseek got correct: {len(deepseek_correct_tasks)} ({len(deepseek_correct_tasks) / len(all_tasks):.2%})")

base_correct_tasks = combinations[(False, True, True)] | combinations[(False, True, False)] | combinations[(True, True, True)] | combinations[(True, True, False)]
print(f"Tasks Base got correct: {len(base_correct_tasks)} ({len(base_correct_tasks) / len(all_tasks):.2%})")

forced_correct_tasks = combinations[(False, False, True)] | combinations[(True, False, True)] | combinations[(False, True, True)] | combinations[(True, True, True)]
print(f"Tasks Forced got correct: {len(forced_correct_tasks)} ({len(forced_correct_tasks) / len(all_tasks):.2%})")

forced_helped_tasks = combinations[(True, False, True)] | combinations[(False, False, True)]
print(f"\nTasks where forced thinking helped: {len(combinations[(True, False, True)])} + {len(combinations[(False, False, True)])} = {len(forced_helped_tasks)}")
print(f"Task IDs: {forced_helped_tasks}")

forced_hurt_tasks = combinations[(True, True, False)] | combinations[(False, True, False)]
print(f"\nTasks where forced thinking hurt: {len(combinations[(True, True, False)])} + {len(combinations[(False, True, False)])} = {len(forced_hurt_tasks)}")
print(f"Task IDs: {forced_hurt_tasks}")

forced_did_not_help_tasks = combinations[(True, False, False)] | combinations[(False, False, False)]
print(f"\nTasks where forced thinking did not help: {len(combinations[(True, False, False)])}")
print(f"Task IDs: {forced_did_not_help_tasks}")

forced_did_not_hurt_tasks = combinations[(False, True, True)] | combinations[(True, True, True)]
print(f"\nTasks where forced thinking did not hurt: {len(combinations[(False, True, True)])} + {len(combinations[(True, True, True)])} = {len(forced_did_not_hurt_tasks)}")
print(f"Task IDs: {forced_did_not_hurt_tasks}")

# %% Feed the input to both models and get the logits for all tokens

def get_logits(task_id: str, response_text: str):
    """Get logits from both models with memory-efficient handling."""
    # Clear CUDA cache before processing
    torch.cuda.empty_cache()

    task = tasks_dataset["problems-by-qid"][task_id]
    user_message = generate_user_message(task)

    # Format the prompt for the deepseek model
    prompt_message = [
        {"role": "user", "content": user_message}
    ]
    deepseek_input_ids = deepseek_tokenizer.apply_chat_template(
        conversation=prompt_message,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(deepseek_model.device)
    
    # Format the prompt for the original model
    if "Instruct" in original_model_name:
        prompt_message = [
            {"role": "user", "content": user_message}
        ]
        original_input_ids = original_tokenizer.apply_chat_template(
            [prompt_message],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(original_model.device)
    else:
        # For non-Instruct models, directly encode the user message
        original_input_ids = original_tokenizer.encode(
            user_message,
            return_tensors="pt",
            add_special_tokens=True
        ).to(original_model.device)

    # Add the response text to the original input ids
    deepseek_response_start_pos = len(deepseek_input_ids[0])
    deepseek_response_ids = deepseek_tokenizer.encode(response_text, add_special_tokens=False, return_tensors="pt").to(deepseek_model.device)
    response_input_ids = deepseek_response_ids[:]
    deepseek_input_ids = torch.cat([deepseek_input_ids, deepseek_response_ids], dim=1)

    # print(f"Deepseek input ids: {deepseek_input_ids.shape}\nDecoded: {deepseek_tokenizer.decode(deepseek_input_ids[0], skip_special_tokens=False)}")

    # Add the response text to the original input ids
    original_response_start_pos = len(original_input_ids[0])
    original_response_ids = original_tokenizer.encode(response_text, add_special_tokens=False, return_tensors="pt").to(original_model.device)
    original_input_ids = torch.cat([original_input_ids, original_response_ids], dim=1)

    # print(f"Original input ids: {original_input_ids.shape}\nDecoded: {original_tokenizer.decode(original_input_ids[0], skip_special_tokens=False)}")
    
    # Process models one at a time to reduce memory usage
    # DeepSeek model logits
    with torch.no_grad():
        deepseek_outputs = deepseek_model(
            input_ids=deepseek_input_ids.to(deepseek_model.device)
        )
        # Only keep the logits we need and move to CPU immediately
        deepseek_logits = deepseek_outputs.logits[
            0, 
            deepseek_response_start_pos:
        ].cpu()
        del deepseek_outputs
    
    # Clear memory before processing next model
    torch.cuda.empty_cache()
    
    # Original model logits
    with torch.no_grad():
        original_outputs = original_model(
            input_ids=original_input_ids.to(original_model.device)
        )
        # Only keep the logits we need and move to CPU immediately
        original_logits = original_outputs.logits[
            0,
            original_response_start_pos:
        ].cpu()
        del original_outputs
    
    torch.cuda.empty_cache()

    assert deepseek_logits.shape == original_logits.shape, f"Error: Logits have different shapes: {deepseek_logits.shape} != {original_logits.shape}.\n Decoded input to deepseek: {deepseek_tokenizer.decode(deepseek_input_ids[0], skip_special_tokens=False)}\n Decoded input to original: {original_tokenizer.decode(original_input_ids[0], skip_special_tokens=False)}"
    
    return deepseek_logits, original_logits, response_input_ids

# %% Calculate the KL divergence between the logits

def calculate_kl_divergence(p_logits, q_logits):
    """
    Calculate KL divergence between two distributions given their logits.
    Uses PyTorch's built-in KL divergence function with log_softmax.
    """
    # Convert logits directly to log probabilities
    p_log = torch.nn.functional.log_softmax(p_logits, dim=-1)
    q_log = torch.nn.functional.log_softmax(q_logits, dim=-1)
    
    # Calculate KL divergence using PyTorch's function
    kl_div = torch.nn.functional.kl_div(
        p_log,      # input in log-space
        q_log,      # target in log-space
        reduction='none',
        log_target=True
    )

    # Sum over vocabulary dimension
    kl_div = kl_div.sum(dim=-1)
    
    kl_div = kl_div.squeeze()

    # Convert to float32 and ensure positive values
    kl_div = kl_div.float()
    kl_div = torch.clamp(kl_div, min=0.0)

    return kl_div

# %%

# Grab one of the tasks where forced thinking did not help but it should have, and analyze its KL div heatmap
task_id = random.choice(list(combinations[(True, True, False)]))
# task_id = "math_train_prealgebra_298"

metric = "prob" # "prob" or "kl_div"

# Focus on analyzing KL divergence for a forced thinking response
print(f"\nAnalyzing KL divergence for task ID: {task_id} (expected answer: {tasks_dataset['problems-by-qid'][task_id]['answer-without-reasoning']})")
response_data = next(r for r in results["original_with_thinking_tokens"]["responses"] if r["task_uuid"] == task_id)

# Get logits and calculate KL divergence
deepseek_logits, original_logits, response_input_ids = get_logits(
    task_id,
    response_data["model_response"]
)

assert deepseek_logits.shape == original_logits.shape, f"Error: Logits have different shapes: {deepseek_logits.shape} != {original_logits.shape}"
assert response_input_ids.shape[1] == deepseek_logits.shape[0], f"Error: Response input ids have different length than logits: {response_input_ids.shape[1]} != {deepseek_logits.shape[0]}"

if metric == "kl_div":
    metric_values = calculate_kl_divergence(deepseek_logits, original_logits).cpu()
    assert metric_values.shape == (response_input_ids.shape[1],), f"Error: Metric values have different shape than response input ids: {metric_values.shape} != {response_input_ids.shape[1]}"
elif metric == "prob":
    # Gather the probabilities of the response_input_ids from the probs tensor. We need to account for the fact that we don't have next token logits for first token
    # Convert logits to probabilities using softmax
    metric_values = torch.nn.functional.softmax(deepseek_logits, dim=-1)
    
    # For each position, get the probability of the actual token that was generated
    token_probs = []
    token_probs.append(0) # Add a dummy token with probability 0 at the beginning
    for i in range(1, response_input_ids.shape[1]):
        token_id = response_input_ids[0, i]
        token_prob = metric_values[i - 1, token_id]
        token_probs.append(token_prob)
    
    metric_values = torch.tensor(token_probs)
    assert metric_values.shape == (response_input_ids.shape[1],), f"Error: Metric values have different shape than response input ids: {metric_values.shape} != {response_input_ids.shape[1]}"

# Get the tokens for visualization
thinking_tokens = deepseek_tokenizer.convert_ids_to_tokens(
    response_input_ids[0]
)

print(f"Number of forced tokens: {len(response_data['forced_tokens_info'])}")
for forced_token_info in response_data["forced_tokens_info"]:
    pos = forced_token_info["position"]
    print(f"Forced token: `{forced_token_info['forced_token']}` at position {pos}")
    thinking_tokens[pos] = f"**{thinking_tokens[pos]}**"

    # Show top 5 tokens with highest probability for the deepseek model in the forced token position.
    # We are gonna print them as <decoded_token> (<probability>)
    
    pos = pos - 1 # Shift by one to the left to account for the fact that we don't have next token logits for first token
    top_5_tokens = torch.argsort(deepseek_logits[pos], descending=True)[:5]
    top_5_tokens_probs = torch.nn.functional.softmax(deepseek_logits[pos], dim=-1)[top_5_tokens]
    print("Top 5 tokens with highest probability for the deepseek model in position previous to the forced token:")
    for token, prob in zip(top_5_tokens, top_5_tokens_probs):
        print(f" `{deepseek_tokenizer.decode([token])}` ({prob:.2%})")
    
# for pos in range(1, response_input_ids.shape[1]):
#     pos = pos - 1 # Shift by one to the left to account for the fact that we don't have next token logits for first token
#     top_5_tokens = torch.argsort(deepseek_logits[pos], descending=True)[:5]
#     top_5_tokens_probs = torch.nn.functional.softmax(deepseek_logits[pos], dim=-1)[top_5_tokens]
#     print(f"Top 5 tokens with highest probability for the deepseek model in position {pos}:")
#     for token, prob in zip(top_5_tokens, top_5_tokens_probs):
#         print(f" `{deepseek_tokenizer.decode([token])}` ({prob:.2%})")

# Create heatmap visualization
html = activation_visualization(
    thinking_tokens,
    metric_values,
    tokenizer=deepseek_tokenizer,
    title="KL Divergence Heatmap for Task Where Forced Thinking Did Not Help",
    relative_normalization=False,
)
display(HTML(html))
# %%
