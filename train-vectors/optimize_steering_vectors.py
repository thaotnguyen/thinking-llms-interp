# %%
import argparse
import dotenv
dotenv.load_dotenv("../.env")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os
import random
import json
import utils
import gc
from utils import steering_opt
import math
from tqdm import tqdm
import traceback

# %% Parse arguments
parser = argparse.ArgumentParser(description="Optimize steering vectors from annotated responses")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                    help="Model to train steering vectors for")
parser.add_argument("--test_examples_pct", type=float, default=0.30,
                    help="Percentage of examples to use for testing (rest used for training)")
parser.add_argument("--save_path", type=str, default="results/vars/optimized_vectors",
                    help="Path to save optimized vectors")
parser.add_argument("--layer", type=int, default=6,
                    help="Layer to optimize steering vector for")
parser.add_argument("--max_iters", type=int, default=50,
                    help="Maximum optimization iterations")
parser.add_argument("--lr", type=str, default="1e-1",
                    help="Learning rate(s) for optimization. Can be a single value or comma-separated list")
parser.add_argument("--min_lr", type=float, default=1e-9,
                    help="Minimum learning rate for optimization")
parser.add_argument("--warmup_iters", type=int, default=0,
                    help="Number of warmup iterations")
parser.add_argument("--context_sentences", type=int, default=1,
                    help="Number of additional sentences to include after target completion")
parser.add_argument("--test_max_tokens", type=int, default=100,
                    help="Maximum tokens to generate when testing")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--has_bos_token", action="store_true", default=True,
                    help="Whether the model has a BOS token")
parser.add_argument("--minibatch_size", type=int, default=32,
                    help="Size of minibatches for optimization (0 means all datapoints at once)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--steering_vector_idx", type=int, default=0,
                    help="Index of the specific steering vector to optimize")
parser.add_argument("--grad_clip", type=float, default=1.0,
                    help="Maximum L2 norm of gradients for gradient clipping. None means no clipping.")
parser.add_argument("--generation_batch_size", type=int, default=3,
                    help="Batch size for generating completions when creating training and evaluation datapoints")
parser.add_argument("--perplexity_batch_size", type=int, default=4,
                    help="Batch size for computing perplexity scores")
args, _ = parser.parse_known_args()

# At module level
ANNOTATION_PATTERN = re.compile(r'\["([\d.]+):(\S+?)"\](.*?)\["end-section"\]', re.DOTALL)
CATEGORY_PATTERN = re.compile(r'\["[\d.]+:(\S+?)"\]')

def get_label_positions(annotated_thinking, response_text, tokenizer, context_sentences=0):
    """Parse SAE annotations and find token positions for each label"""
    label_positions = {}
    
    # Use a pattern that captures labeled segments in the format [activation:category-name] text [end-section]
    # Now supporting activation strength values in the format [56.86:category-name]
    matches = list(ANNOTATION_PATTERN.finditer(annotated_thinking))
    
    # Create character to token mapping once
    char_to_token = utils.get_char_to_token_map(response_text, tokenizer)
    
    # Split response into sentences for context
    sentences = utils.split_into_sentences(response_text, min_words=0)
    
    for match in matches[:-1]:
        activation_str = match.group(1).strip()
        label = match.group(2).strip()
        text = match.group(3).strip()
        
        try:
            activation = float(activation_str)
        except ValueError:
            print(f"Warning: Could not parse activation value '{activation_str}' for category '{label}'")
            continue
            
        if not text:  # Skip empty text
            continue
            
        # Find this text in the original response
        pattern = r'(?:[.?!;\n]|\n\n)\s*(' + re.escape(text) + ')'
        match = re.search(pattern, response_text)
        text_pos = match.start(1) if match else -1
        if text_pos >= 0:
            # Get start and end token positions
            token_start = char_to_token.get(text_pos, None)
            token_end = char_to_token.get(text_pos + len(text) - 1, None)
            
            # Adjust token_end to include the entire token
            if token_end is not None:
                token_end += 1

            if token_start is None or token_end is None or token_start >= token_end:
                continue
            
            # Find the sentence containing our target text
            target_sentence_idx = -1
            for i, sentence in enumerate(sentences):
                if text in sentence:
                    target_sentence_idx = i
                    break
            
            if target_sentence_idx == -1:
                continue
                
            # Get additional context sentences if requested
            additional_context = ""
            if context_sentences > 0 and target_sentence_idx < len(sentences) - 1:
                end_idx = min(target_sentence_idx + context_sentences + 1, len(sentences))
                additional_sentences = sentences[target_sentence_idx + 1:end_idx]
                
                # Find the original whitespace between sentences
                if additional_sentences:
                    # Get the text up to the end of our target text
                    text_end_pos = text_pos + len(text)
                    # Get the text up to the start of the next sentence
                    next_sentence_start = response_text.find(additional_sentences[0], text_end_pos)
                    if next_sentence_start > text_end_pos:
                        # Extract the original whitespace
                        original_whitespace = response_text[text_end_pos:next_sentence_start]
                        # Use the original whitespace to join sentences
                        additional_context = original_whitespace + original_whitespace.join(additional_sentences)
                    else:
                        # Fallback to space if we can't find the original whitespace
                        additional_context = " " + " ".join(additional_sentences)
                
                # Update token_end to include additional context
                if additional_context:
                    context_end_pos = text_pos + len(text) + len(additional_context)
                    context_token_end = char_to_token.get(context_end_pos - 1, None)
                    if context_token_end is not None:
                        token_end = context_token_end + 1
            
            # If we found valid token positions
            if label not in label_positions:
                label_positions[label] = []
            label_positions[label].append((token_start, token_end, text + additional_context, activation, text_pos))
    
    return label_positions

def get_batched_perplexity_inputs(tokenizer, prompts, target_completions, model):
    """Tokenize and pad a batch of prompt+completion pairs for perplexity calculation"""
    full_texts = [prompt + completion for prompt, completion in zip(prompts, target_completions)]
    
    # Tokenize all full texts to find max length
    tokenized_texts = [tokenizer(text, return_tensors='pt') for text in full_texts]
    max_token_length = max([tokens['input_ids'].shape[1] for tokens in tokenized_texts])
    
    # Create padded batch and track prompt lengths
    input_ids_list = []
    attention_mask_list = []
    prompt_lengths = []
    
    for i, (tokens, prompt) in enumerate(zip(tokenized_texts, prompts)):
        input_ids = tokens['input_ids'][0]  # Remove batch dimension
        current_length = input_ids.shape[0]
        prompt_length = len(tokenizer.encode(prompt))
        prompt_lengths.append(prompt_length)
        
        # Pad to max length on the right for perplexity calculation
        if current_length < max_token_length:
            padding_length = max_token_length - current_length
            padded_input_ids = torch.cat([
                input_ids,
                torch.full((padding_length,), tokenizer.pad_token_id, dtype=input_ids.dtype)
            ])
            padded_attention_mask = torch.cat([
                torch.ones(current_length, dtype=torch.long),
                torch.zeros(padding_length, dtype=torch.long)
            ])
        else:
            padded_input_ids = input_ids
            padded_attention_mask = torch.ones(current_length, dtype=torch.long)
        
        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(padded_attention_mask)
    
    # Stack into batches
    batch_input_ids = torch.stack(input_ids_list).to(model.device)
    batch_attention_mask = torch.stack(attention_mask_list).to(model.device)
    
    return {
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask
    }, prompt_lengths

def calculate_perplexity(model, tokenizer, examples):
    """Calculate perplexity for a batch of examples"""
    if not examples:
        return []
    
    # Extract prompts and target completions from examples
    if isinstance(examples, dict) and 'prompt' in examples:
        # Single example case
        examples = [examples]
    
    prompts = [ex['prompt'] if isinstance(ex, dict) else ex[0] for ex in examples]
    target_completions = [ex['target_completion'] if isinstance(ex, dict) else ex[1] for ex in examples]
    
    try:
        # Get batched inputs
        batch_inputs, prompt_lengths = get_batched_perplexity_inputs(tokenizer, prompts, target_completions, model)
        
        with torch.no_grad():
            outputs = model(**batch_inputs)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Calculate perplexity for each example in the batch
            perplexities = []
            
            for i, (prompt_length, target_completion) in enumerate(zip(prompt_lengths, target_completions)):
                try:
                    # Get logits and labels for this example
                    example_logits = logits[i]  # [seq_len, vocab_size]
                    example_input_ids = batch_inputs['input_ids'][i]  # [seq_len]
                    example_attention_mask = batch_inputs['attention_mask'][i]  # [seq_len]
                    
                    # Find the actual sequence length (excluding padding)
                    actual_length = example_attention_mask.sum().item()
                    
                    # Calculate loss only on the target completion part
                    # We need to shift the logits and labels to predict next token
                    shift_logits = example_logits[:-1, :].contiguous()  # Remove last token from logits
                    shift_labels = example_input_ids[1:].contiguous()  # Remove first token from labels
                    
                    # Get the relevant portion for target completion
                    # -1 to account for the shift in both start and end
                    target_start = prompt_length - 1
                    target_end = actual_length - 1
                    
                    if target_start >= target_end or target_start < 0:
                        perplexities.append(float('inf'))
                        continue
                    
                    target_logits = shift_logits[target_start:target_end, :]
                    target_labels = shift_labels[target_start:target_end]
                    
                    # Skip if we have no valid tokens to compute perplexity on
                    if target_logits.size(0) == 0:
                        perplexities.append(float('inf'))
                        continue
                    
                    # Calculate loss with numerical stability
                    loss = torch.nn.functional.cross_entropy(
                        target_logits,
                        target_labels,
                        reduction='mean'
                    )
                    
                    # Check for NaN or inf values
                    if torch.isnan(loss) or torch.isinf(loss):
                        perplexities.append(float('inf'))
                        continue
                    
                    # Calculate perplexity with clipping to avoid overflow
                    perplexity = torch.exp(torch.clamp(loss, max=100)).item()
                    
                    # Final check for NaN or inf
                    if math.isnan(perplexity) or math.isinf(perplexity):
                        perplexities.append(float('inf'))
                    else:
                        perplexities.append(perplexity)
                        
                except Exception as e:
                    print(f"Error calculating perplexity for example {i}: {e}")
                    perplexities.append(float('inf'))
            
            return perplexities
            
    except Exception as e:
        print(f"Error in batch perplexity calculation: {e}")
        return [float('inf')] * len(examples)

def extract_examples_for_category(responses_data, category_name, test_examples_pct, tokenizer, model):
    """Extract examples for a specific category from the responses data and split into training and test sets"""
    examples_for_category = []
    
    # Process each response to extract labeled segments for the specified category
    n_annotated_thinking_containing_category = 0
    for resp in tqdm(responses_data, desc="Extracting examples for category"):
        if not resp.get('annotated_thinking'):
            continue

        full_text = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{resp['original_message']['content']}\n\nStep by step answer:\n{resp['thinking_process']}"
        
        # Look for the specific category in the annotated thinking
        if category_name not in resp['annotated_thinking']:
            continue
            
        n_annotated_thinking_containing_category += 1
        label_positions = get_label_positions(resp['annotated_thinking'], full_text, tokenizer, args.context_sentences)
        
        if category_name in label_positions:
            for start, end, text, activation, text_pos in label_positions[category_name]:
                # Get the text up to this point using the saved text_pos
                context = full_text[:text_pos]

                # Check if context ends properly
                if context[-1] not in ['.', '?', '!', ';', '\n', '\n\n'] and context[-2] not in ['.', '?', '!', ';', '\n', '\n\n'] and context.strip()[-1] not in ['.', '?', '!', ';', '\n', '\n\n']:
                    continue
                
                examples_for_category.append({
                    'prompt': context,
                    'target_completion': text,
                    'original_question': resp['original_message']['content'],
                    'full_thinking': resp['thinking_process'],
                    'activation': activation
                })
    
    if not examples_for_category:
        return [], []

    print(f"Found {n_annotated_thinking_containing_category} annotated thinking containing category {category_name}")
    print(f"Found {len(examples_for_category)} examples for category {category_name}")
        
    # First sort by activation and use all available examples
    sorted_by_activation = sorted(examples_for_category, key=lambda x: x['activation'], reverse=True)
    top_candidates = sorted_by_activation
    
    # Compute perplexity using batched calculation
    perplexity_batch_size = args.perplexity_batch_size
    num_batches = math.ceil(len(top_candidates) / perplexity_batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc="Computing perplexity"):
        start_idx = batch_idx * perplexity_batch_size
        end_idx = min(start_idx + perplexity_batch_size, len(top_candidates))
        
        batch_examples = top_candidates[start_idx:end_idx]
        batch_perplexities = calculate_perplexity(model, tokenizer, batch_examples)
        
        # Assign perplexities back to examples
        for example, perplexity in zip(batch_examples, batch_perplexities):
            example['perplexity'] = perplexity
    
    # Filter out examples with infinite perplexity
    valid_candidates = [ex for ex in top_candidates if not math.isinf(ex['perplexity'])]
    
    if not valid_candidates:
        print(f"Warning: No valid perplexity scores for category {category_name}, using activation only")
        valid_candidates = top_candidates
        for ex in valid_candidates:
            ex['perplexity'] = 0.0  # Set perplexity to 0 for invalid cases
    
    # Calculate z-scores for perplexity
    perplexities = [ex['perplexity'] for ex in valid_candidates]
    mean_perplexity = sum(perplexities) / len(perplexities)
    std_perplexity = math.sqrt(sum((p - mean_perplexity) ** 2 for p in perplexities) / len(perplexities))
    
    # Calculate perplexity score for filtered candidates
    max_perplexity = max(ex['perplexity'] for ex in valid_candidates)
    
    for example in valid_candidates:
        example['perplexity_score'] = example['perplexity'] / max_perplexity if max_perplexity > 0 else 0
    
    # Sort by perplexity score and use all valid candidates
    sorted_by_perplexity = sorted(valid_candidates, key=lambda x: x['perplexity_score'], reverse=True)
    
    selected_examples = sorted_by_perplexity

    #random.shuffle(selected_examples)

    # Split based on test percentage
    n_test_examples = int(len(selected_examples) * test_examples_pct)
    n_training_examples = len(selected_examples) - n_test_examples
    
    training_examples = selected_examples[:n_training_examples]
    test_examples = selected_examples[n_training_examples:]
        
    return training_examples, test_examples

def get_sorted_categories(responses_data):
    """Extract all unique categories from responses data and return them sorted alphabetically"""
    categories = set()
    
    for resp in responses_data:
        if not resp.get('annotated_thinking'):
            continue
            
        # Extract category names from annotated thinking
        matches = CATEGORY_PATTERN.finditer(resp['annotated_thinking'])
        for match in matches:
            category = match.group(1).strip()
            categories.add(category)
    
    return sorted(list(categories))

def get_batched_prompt_inputs(tokenizer, prompts, model):
    """Tokenize and pad a batch of prompts for generation"""
    # Tokenize all prompts to find max length
    tokenized_prompts = [tokenizer(prompt, return_tensors='pt') for prompt in prompts]
    max_token_length = max([tokens['input_ids'].shape[1] for tokens in tokenized_prompts])
    
    # Create padded batch
    input_ids_list = []
    attention_mask_list = []
    
    for tokens in tokenized_prompts:
        input_ids = tokens['input_ids'][0]  # Remove batch dimension
        current_length = input_ids.shape[0]
        
        # Pad to max length
        if current_length < max_token_length:
            padding_length = max_token_length - current_length
            # Pad on the left for generation
            padded_input_ids = torch.cat([
                torch.full((padding_length,), tokenizer.pad_token_id, dtype=input_ids.dtype),
                input_ids
            ])
            padded_attention_mask = torch.cat([
                torch.zeros(padding_length, dtype=torch.long),
                torch.ones(current_length, dtype=torch.long)
            ])
        else:
            padded_input_ids = input_ids
            padded_attention_mask = torch.ones(current_length, dtype=torch.long)
        
        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(padded_attention_mask)
    
    # Stack into batches
    batch_input_ids = torch.stack(input_ids_list).to(model.device)
    batch_attention_mask = torch.stack(attention_mask_list).to(model.device)
    
    return {
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask
    }

def create_datapoints(model, tokenizer, examples):
    """Create training datapoints for a batch of examples"""
    if not examples:
        return []
    
    prompts = [example['prompt'] for example in examples]
    target_completions = [example['target_completion'] for example in examples]
    
    # Get max tokens to generate based on target completions
    max_new_tokens = max([len(tokenizer.encode(completion)) for completion in target_completions])
    
    # Get batched inputs
    batch_inputs = get_batched_prompt_inputs(tokenizer, prompts, model)
    
    # Generate completions for the batch while suppressing EOS token
    with torch.no_grad():
        generated_tokens = model.generate(
            **batch_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            suppress_tokens=[tokenizer.eos_token_id],
            do_sample=False
        )
    
    # Decode generated outputs
    generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    # Extract only the generated part (remove the original prompt)
    og_completions = []
    for i, (full_prompt, full_generated) in enumerate(zip(prompts, generated_texts)):
        # Remove padding tokens from the beginning if any
        full_generated_clean = full_generated.replace(tokenizer.pad_token, "").strip()

        # Remove the original prompt from the generated text
        og_completion = full_generated_clean.replace(full_prompt, "")
        
        if og_completion == "":
            print(f"Warning: No completion generated for prompt {i}")
            og_completion = "."  # Use a simple character instead of EOS token
        
        og_completions.append(og_completion)
    
    # Create training datapoints
    datapoints = []
    for example, og_completion in zip(examples, og_completions):
        target_completion = example['target_completion']
        full_prompt = example['prompt']
        datapoint = steering_opt.TrainingDatapoint(
            full_prompt,
            src_completions=[og_completion],  # completions to decrease probability
            dst_completions=[target_completion],  # completions to increase probability
            token=slice(max(1, -30), None) if args.has_bos_token else slice(-30, None)
        )
        datapoints.append(datapoint)
    
    return datapoints

def test_on_unseen_example(model, tokenizer, vector, layer, test_example, max_new_tokens=50):
    """Test the optimized vector on an unseen example"""
    prompt = test_example['prompt']
    
    print("\n--- Testing on unseen example ---")
    print(f"Prompt: ...{prompt[-100:]}")
    print("\n--- Generated Completions ---")
    
    # Generate without steering
    generated_tokens = model.generate(
        **tokenizer(prompt, return_tensors='pt').to(model.device), 
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        suppress_tokens=[tokenizer.eos_token_id]
    )
    unsteered_completion = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    unsteered_generated_part = unsteered_completion[len(prompt):]
    print("UNSTEERED:")
    print(unsteered_generated_part)
    print()
    
    # Generate with steering
    steering_hook = (layer, steering_opt.make_steering_hook_hf(vector, token=slice(max(1, -30), None) if args.has_bos_token else slice(-30, None)))
    with steering_opt.hf_hooks_contextmanager(model, [steering_hook]): 
        generated_tokens = model.generate(
            **tokenizer(prompt, return_tensors='pt').to(model.device), 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        steered_completion = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        steered_generated_part = steered_completion[len(prompt):]
        print("STEERED:")
        print(steered_generated_part)
    
    print("\nTARGET:")
    print(test_example['target_completion'])
    print(f"Activation strength: {test_example['activation']}")
    print("\n" + "-"*50 + "\n")
    
    return unsteered_completion, steered_completion

def save_hyperparameters(hyperparams, model_name_short, steering_vector_idx, category):
    """Save hyperparameters for a specific steering vector for reproducibility"""
    hp_dir = "results/vars"
    os.makedirs(hp_dir, exist_ok=True)
    
    hp_file = f"{hp_dir}/steering_vector_hyperparams.json"
    
    # Load existing hyperparameters if file exists
    if os.path.exists(hp_file):
        with open(hp_file, 'r') as f:
            all_hyperparams = json.load(f)
    else:
        all_hyperparams = {}
    
    # Create model entry if it doesn't exist
    if model_name_short not in all_hyperparams:
        all_hyperparams[model_name_short] = {}
    
    # Update hyperparameters for this vector
    all_hyperparams[model_name_short][str(steering_vector_idx)] = {
        "category": category,
        "hyperparameters": hyperparams
    }
    
    # Save updated hyperparameters
    with open(hp_file, 'w') as f:
        json.dump(all_hyperparams, f, indent=2)
    
    print(f"Saved hyperparameters to {hp_file}")

def main():
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Parse learning rates
    try:
        learning_rates = [float(lr.strip()) for lr in args.lr.split(',')]
    except ValueError:
        raise ValueError("Learning rates must be comma-separated numbers")
    
    # Create directories
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
        torch_dtype=torch.bfloat16
    )

    torch.set_default_device(model.device)

    # Default responses path
    model_name_short = args.model.split('/')[-1].lower()
    thinking_model_name = utils.model_mapping.get(args.model, model_name_short)
    if thinking_model_name is None:
        thinking_model_name = model_name_short
    thinking_model_short = thinking_model_name.split('/')[-1].lower()
    responses_json_path = f"../generate-responses/results/vars/responses_{thinking_model_short}.json"
    annotated_responses_json_path = f"../generate-responses/results/vars/annotated_responses_{thinking_model_short}.json"
    
    if not os.path.exists(responses_json_path):
        raise FileNotFoundError(f"Annotated responses file not found at {responses_json_path}. Please annotate responses first.")
    
    # Load responses
    print(f"Loading annotated responses from {responses_json_path}")
    with open(responses_json_path, 'r') as f:
        responses_data = json.load(f)

    # Load annotated responses
    print(f"Loading annotated responses from {annotated_responses_json_path}")
    with open(annotated_responses_json_path, 'r') as f:
        annotated_responses_data = json.load(f)
    
    # Match responses with their annotations by index and validate
    valid_responses = []
    for i, resp in enumerate(responses_data):
        if i < len(annotated_responses_data):
            annotated_resp = annotated_responses_data[i]
            # Verify that the responses match by question_id and dataset_name
            if (resp['question_id'] == annotated_resp['question_id'] and 
                resp['dataset_name'] == annotated_resp['dataset_name'] and
                annotated_resp.get('annotated_thinking')):
                # Create merged response with annotated_thinking
                merged_resp = resp.copy()
                merged_resp['annotated_thinking'] = annotated_resp['annotated_thinking']
                valid_responses.append(merged_resp)
            else:
                print(f"Warning: Mismatch at index {i} - response question_id: {resp['question_id']}, annotated question_id: {annotated_resp.get('question_id')}")
    
    print(f"Found {len(valid_responses)} responses with annotations out of {len(responses_data)} total responses")
    
    # Get all available categories sorted alphabetically
    all_categories = get_sorted_categories(valid_responses)
    
    # Print all available categories with their indices
    print("\nAvailable categories with indices:")
    for idx, category in enumerate(all_categories):
        print(f"  [{idx}] {category}")
    
    # Check if steering_vector_idx is valid
    if args.steering_vector_idx < 0 or args.steering_vector_idx >= len(all_categories):
        raise ValueError(f"Invalid steering_vector_idx: {args.steering_vector_idx}. Must be between 0 and {len(all_categories)-1}")
    
    # Get the target category name
    target_category = all_categories[args.steering_vector_idx]
    print(f"\nOptimizing vector for category: {target_category} (index {args.steering_vector_idx})")
    
    # Extract examples only for the target category
    print(f"Extracting examples for category {target_category}...")
    training_examples, test_examples = extract_examples_for_category(
        valid_responses, 
        target_category, 
        args.test_examples_pct, 
        tokenizer, 
        model
    )

    ex_activations = [ex['activation'] for ex in training_examples]
    max_activation = max(ex_activations)
    
    if not training_examples:
        print(f"No valid examples found for category {target_category}. Exiting.")
        return
        
    print(f"Found {len(training_examples)} training examples and {len(test_examples)} test examples")
        
    # Create datapoints for this category using batched generation
    datapoints = []
    batch_size = args.generation_batch_size
    num_batches = math.ceil(len(training_examples) / batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc="Creating training datapoints"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(training_examples))
        
        batch_examples = training_examples[start_idx:end_idx]
        
        batch_datapoints = create_datapoints(
            model, 
            tokenizer, 
            batch_examples
        )
        datapoints.extend(batch_datapoints)
    
    # Create evaluation datapoints using batched generation
    eval_datapoints = []
    num_eval_batches = math.ceil(len(test_examples) / batch_size)
    
    for batch_idx in tqdm(range(num_eval_batches), desc="Creating evaluation datapoints"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_examples))
        
        batch_examples = test_examples[start_idx:end_idx]
        
        batch_datapoints = create_datapoints(
            model,
            tokenizer,
            batch_examples
        )
        eval_datapoints.extend(batch_datapoints)
    
    if not datapoints:
        print(f"No valid datapoints for category {target_category}, exiting")
        return
    
    # Store results for each learning rate
    all_results = {}
    
    # Run optimization for each learning rate
    for lr in tqdm(learning_rates, desc="Optimizing with learning rates"):
        print(f"\nOptimizing with learning rate: {lr}")
        try:
            vector, loss_info = steering_opt.optimize_vector(
                model, 
                datapoints, 
                args.layer,
                tokenizer=tokenizer,
                max_iters=args.max_iters,
                lr=lr,
                minibatch_size=args.minibatch_size,
                eval_datapoints=eval_datapoints,
                return_loss_history=True,
                do_target_loss_sum=False,
                target_loss=None,
                warmup_steps=args.warmup_iters,
                min_lr=args.min_lr,
                max_norm=max_activation*2,
                do_output_constr=False,
                target_loss_target_iters=float('inf'),  # type: ignore
                early_stopping_min_delta=1,
                early_stopping_patience=5,
                early_stopping_metric='eval_loss',
                grad_clip=args.grad_clip
            )
            
            # Store results
            if lr not in all_results:
                all_results[lr] = []
            
            all_results[lr].append({
                'vector': vector.detach().cpu(),
                'loss_info': loss_info,
                'final_eval_loss': loss_info['eval_losses'][-1] if 'eval_losses' in loss_info else float('inf')
            })
            
        except Exception as e:
            print(f"Error optimizing vector for category {target_category} with lr {lr}: {e}")
            traceback.print_exc()
            continue
    
    if not all_results:
        print("No successful optimization runs. Exiting.")
        return
    
    # Find best learning rate and run based on final evaluation loss
    best_lr = None
    best_result = None
    best_eval_loss = float('inf')
    
    for lr, results in all_results.items():
        for result in results:
            if result['final_eval_loss'] < best_eval_loss:
                best_eval_loss = result['final_eval_loss']
                best_lr = lr
                best_result = result
    
    if best_result is None:
        print("No valid optimization results found. Exiting.")
        return
    
    print(f"\nBest learning rate: {best_lr} (final eval loss: {best_result['final_eval_loss']:.4f})")
    
    # Save hyperparameters for this vector
    hyperparams = {
        "layer": args.layer,
        "max_iters": args.max_iters,
        "learning_rates": learning_rates,
        "best_lr": best_lr,
        "context_sentences": args.context_sentences,
        "seed": args.seed,
        "test_examples_pct": args.test_examples_pct,
        "n_training_examples": len(training_examples),
        "n_test_examples": len(test_examples),
        "vector_norm": best_result['vector'].norm().item(),
        "grad_clip": args.grad_clip
    }
    save_hyperparameters(hyperparams, model_name_short, args.steering_vector_idx, target_category)
    
    # Save training and evaluation losses for all learning rates
    losses_path = f"results/vars/losses_{model_name_short}_idx_{args.steering_vector_idx}.pt"
    losses = {
        best_lr: [{
            'train_losses': best_result['loss_info']['train_losses'],
            'eval_losses': best_result['loss_info']['eval_losses'] if 'eval_losses' in best_result['loss_info'] else None,
            'final_eval_loss': best_result['final_eval_loss']
        }]
    }
    torch.save(losses, losses_path)
    print(f"\nSaved training and evaluation losses for best learning rate {best_lr} to {losses_path}")
    
    # Test the best vector on an unseen example
    if test_examples:
        print(f"\nTesting {target_category} vector on an unseen example (best lr: {best_lr}):")
        # Use the highest activating test example
        test_example = random.choice(test_examples)
        test_on_unseen_example(
            model, 
            tokenizer, 
            best_result['vector'], 
            args.layer, 
            test_example,
            max_new_tokens=args.test_max_tokens
        )
    
    # Free memory
    del datapoints
    del eval_datapoints
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save all vectors
    # Load existing optimized vectors if they exist
    vectors_path = f"{args.save_path}_{model_name_short}.pt"
    optimized_vectors = {}
    if os.path.exists(vectors_path):
        print(f"\nLoading existing optimized vectors from {vectors_path}")
        optimized_vectors = torch.load(vectors_path)

    optimized_vectors[target_category] = best_result['vector']

    torch.save(optimized_vectors, vectors_path)
    print(f"\nSaved optimized vectors to {vectors_path}")

if __name__ == "__main__":
    main() 
# %%
