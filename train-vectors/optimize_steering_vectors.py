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
import sys
import os
# Add the parent directory to the path to ensure we import from the correct utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils
import gc
from utils import steering_opt
from tqdm import tqdm
import traceback
import wandb

# %% Parse arguments
parser = argparse.ArgumentParser(description="Optimize steering vectors from annotated responses")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                    help="Model to train steering vectors for")
parser.add_argument("--n_training_examples", type=int, default=2048,
                    help="Number of training examples to use per category")
parser.add_argument("--test_examples_pct", type=float, default=1,
                    help="Percentage of examples to use for testing (rest used for training)")
parser.add_argument("--save_path", type=str, default="results/vars",
                    help="Path to save optimized vectors")
parser.add_argument("--layer", type=int, default=6,
                    help="Layer to optimize steering vector for")
parser.add_argument("--max_iters", type=int, default=5,
                    help="Maximum optimization iterations")
parser.add_argument("--lr", type=str, default="1e-1",
                    help="Learning rate(s) for optimization. Can be a single value or comma-separated list")
parser.add_argument("--min_lr", type=float, default=0,
                    help="Minimum learning rate for optimization")
parser.add_argument("--warmup_iters", type=int, default=0,
                    help="Number of warmup iterations")
parser.add_argument("--context_sentences", type=int, default=0,
                    help="Number of additional sentences to include after target completion")
parser.add_argument("--test_max_tokens", type=int, default=100,
                    help="Maximum tokens to generate when testing")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--minibatch_size", type=int, default=6,
                    help="Size of minibatches for optimization")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--steering_vector_idx", type=int, default=0,
                    help="Index of the specific steering vector to optimize")
parser.add_argument("--grad_clip", type=float, default=None,
                    help="Maximum L2 norm of gradients for gradient clipping. None means no clipping.")
parser.add_argument("--steering_token_window", type=int, default=50,
                    help="Number of previous tokens in the target completion to apply the steering vector to (None means all)")
parser.add_argument("--use_wandb", action="store_true", default=False,
                    help="Use wandb for logging")
parser.add_argument("--wandb_project", type=str, default="optimize-steering-vectors",
                    help="Wandb project name")
parser.add_argument("--use_activation_perplexity_selection", action="store_true", default=False,
                    help="If set, use activation/perplexity-based selection. If not set, use random sampling over the full set.")
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

def extract_examples_for_category(responses_data, category_name, test_examples_pct, tokenizer, n_training_examples, model):
    """Extract examples for a specific category from the responses data and split into training and test sets
    Now: select by top perplexity after activation pre-filter, unless random sampling is selected.
    """
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
        print(f"No valid examples found for category {category_name}. Exiting.")
        return [], [], 0.0

    print(f"Found {n_annotated_thinking_containing_category} annotated thinking containing category {category_name}")
    print(f"Found {len(examples_for_category)} examples for category {category_name}")

    # Calculate how many total examples we need to get the desired number of training and test examples
    n_test_examples = int(test_examples_pct * n_training_examples)
    total_examples_needed = n_training_examples + n_test_examples

    if not args.use_activation_perplexity_selection:
        # Use random sampling over the full set
        if len(examples_for_category) > total_examples_needed:
            final_examples = random.sample(examples_for_category, total_examples_needed)
        else:
            final_examples = examples_for_category.copy()
        random.shuffle(final_examples)
        n_train = min(n_training_examples, len(final_examples))
        n_test = min(n_test_examples, len(final_examples) - n_train)
        training_examples = final_examples[:n_train]
        test_examples = final_examples[n_train:n_train + n_test]
        print(f"Final selection (random): {len(training_examples)} training examples, {len(test_examples)} test examples")
        return training_examples, test_examples, 1.0

    # --- CORRECTED LOGIC: Activation pre-filter, then select by perplexity ---
    # 1. Sort all examples by activation (descending)
    examples_for_category_sorted = sorted(examples_for_category, key=lambda x: x['activation'], reverse=True)
    # 2. Take the top 4x the data needed
    sample_size = min(len(examples_for_category_sorted), total_examples_needed * 4)
    sampled_examples = examples_for_category_sorted[:sample_size]

    print(f"Top {sample_size} examples by activation selected for perplexity ranking")

    # 3. Calculate perplexity for each of these examples
    print("Calculating perplexity for sampled examples...")
    examples_with_metrics = []
    for example in tqdm(sampled_examples, desc="Computing perplexity"):
        try:
            # Tokenize the full sequence (prompt + target completion)
            full_text = example['prompt'] + example['target_completion']
            inputs = tokenizer(full_text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0]  # Remove batch dimension
                prompt_tokens = tokenizer(example['prompt'], return_tensors='pt')['input_ids'][0]
                prompt_len = len(prompt_tokens)
                shift_logits = logits[:-1, :].contiguous()
                shift_labels = inputs['input_ids'][0][1:].contiguous()
                target_start = prompt_len - 1
                target_end = len(shift_labels)
                if target_start >= target_end or target_start < 0:
                    continue
                target_logits = shift_logits[target_start:target_end, :]
                target_labels = shift_labels[target_start:target_end]
                if target_logits.size(0) == 0:
                    continue
                loss = torch.nn.functional.cross_entropy(target_logits, target_labels, reduction='mean')
                perplexity = torch.exp(loss).item()
                examples_with_metrics.append({
                    **example,
                    'perplexity': perplexity,
                    'activation': example['activation']
                })
        except Exception as e:
            print(f"Error calculating perplexity for example: {e}")
            continue

    if not examples_with_metrics:
        print("No valid examples with perplexity found. Falling back to random sampling.")
        # Fall back to random sampling if perplexity calculation fails
        max_examples = min(len(examples_for_category), total_examples_needed)
        selected_examples = random.sample(examples_for_category, max_examples)
        # Split based on test percentage
        n_test_examples = int(len(selected_examples) * test_examples_pct)
        n_training_examples_actual = len(selected_examples) - n_test_examples
        training_examples = selected_examples[:n_training_examples_actual]
        test_examples = selected_examples[n_training_examples_actual:]
        return training_examples, test_examples, 1.0

    # 4. From the top N (train + test) by perplexity (descending), split into train and test sets
    final_examples = sorted(examples_with_metrics, key=lambda x: x['perplexity'], reverse=True)[:total_examples_needed]

    # 5. Split into train and test
    n_train = min(n_training_examples, len(final_examples))
    n_test = min(n_test_examples, len(final_examples) - n_train)
    training_examples = final_examples[:n_train]
    test_examples = final_examples[n_train:n_train + n_test]

    print(f"Final selection (perplexity): {len(training_examples)} training examples, {len(test_examples)} test examples")

    # Calculate max activation before removing the field
    max_activation = max(ex['activation'] for ex in final_examples) if final_examples else 1.0

    # Remove perplexity/activation from the final examples
    for ex in final_examples:
        ex.pop('perplexity', None)
        ex.pop('activation', None)

    return training_examples, test_examples, max_activation

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

def test_on_unseen_example(model, tokenizer, vector, layer, test_example, max_new_tokens=50, steering_token_window=None):
    """Test the optimized vector on an unseen example"""
    prompt = test_example['prompt']
    target_completion = test_example['target_completion']
    
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
    
    # Figure out the correct slice for steering
    prompt_tok = tokenizer(prompt, return_tensors='pt')['input_ids'][0]
    target_tok = tokenizer(target_completion, return_tensors='pt')['input_ids'][0]
    prompt_len = len(prompt_tok)
    target_len = len(target_tok)
    if steering_token_window is None:
        steering_start = prompt_len
    else:
        steering_start = prompt_len + max(0, target_len - steering_token_window)
    steering_token_slice = slice(steering_start, None)
    
    # Generate with steering
    steering_hook = (layer, steering_opt.make_steering_hook_hf(vector, token=steering_token_slice))
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

    for param in model.parameters():
        param.requires_grad = False

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
    
    # Initialize wandb
    wandb_run = None
    lrs_str = "-".join([str(lr) for lr in learning_rates])
    run_name = f"{model_name_short}_layer{args.layer}_idx{args.steering_vector_idx}_lr{lrs_str}_n{args.n_training_examples}"
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Please install it with `pip install wandb`")
        
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )

    # Extract examples only for the target category
    print(f"Extracting examples for category {target_category}...")
    training_examples, test_examples, max_activation = extract_examples_for_category(
        valid_responses, 
        target_category, 
        args.test_examples_pct, 
        tokenizer,
        args.n_training_examples,
        model
    )
    
    if not training_examples:
        print(f"No valid examples found for category {target_category}. Exiting.")
        return
        
    print(f"Found {len(training_examples)} training examples and {len(test_examples)} test examples")
    
    # Extract prompts and target completions for training
    train_prompts = [ex['prompt'] for ex in training_examples]
    train_target_completions = [ex['target_completion'] for ex in training_examples]
    
    # Extract prompts and target completions for evaluation
    eval_prompts = [ex['prompt'] for ex in test_examples]
    eval_target_completions = [ex['target_completion'] for ex in test_examples]
    
    # Store results for each learning rate
    all_results = {}
    
    # Run optimization for each learning rate
    for lr in tqdm(learning_rates, desc="Optimizing with learning rates"):
        print(f"\nOptimizing with learning rate: {lr}")
        try:
            vector, loss_info = steering_opt.optimize_vector_simple(
                model, 
                tokenizer,
                train_prompts, 
                train_target_completions, 
                args.layer,
                lr=lr,
                max_iters=args.max_iters,
                minibatch_size=args.minibatch_size,
                warmup_steps=args.warmup_iters,
                min_lr=args.min_lr,
                starting_norm=1,
                max_norm=None,
                grad_clip=args.grad_clip,
                early_stopping_patience=5,
                early_stopping_min_delta=0.1,
                return_info=True,
                return_loss_history=True,
                steering_token_window=args.steering_token_window,
                eval_prompts=eval_prompts,
                eval_target_completions=eval_target_completions,
                wandb_run=wandb_run
            )
            
            # Store results
            if lr not in all_results:
                all_results[lr] = []
            
            all_results[lr].append({
                'vector': vector.detach().cpu(),
                'loss_info': loss_info,
                'final_loss': loss_info['final_loss']
            })
            
        except Exception as e:
            print(f"Error optimizing vector for category {target_category} with lr {lr}: {e}")
            traceback.print_exc()
            continue
    
    if not all_results:
        print("No successful optimization runs. Exiting.")
        return
    
    # Find best learning rate and run based on final loss
    best_lr = None
    best_result = None
    best_loss = float('inf')
    
    for lr, results in all_results.items():
        for result in results:
            # Use evaluation loss for selection if available, otherwise use training loss
            final_loss = result['loss_info'].get('final_eval_loss', result['final_loss'])
            if final_loss < best_loss:
                best_loss = final_loss
                best_lr = lr
                best_result = result
    
    if best_result is None:
        print("No valid optimization results found. Exiting.")
        return
    
    print(f"\nBest learning rate: {best_lr} (final loss: {best_result['final_loss']:.4f})")
    if 'final_eval_loss' in best_result['loss_info']:
        print(f"Best evaluation loss: {best_result['loss_info']['final_eval_loss']:.4f}")
    
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
        "grad_clip": args.grad_clip,
        "minibatch_size": args.minibatch_size,
        "steering_token_window": args.steering_token_window
    }
    save_hyperparameters(hyperparams, model_name_short, args.steering_vector_idx, target_category)
    
    # Log best hyperparameters and summary to wandb
    if wandb_run:
        wandb_run.config.update(hyperparams, allow_val_change=True)
        wandb_run.summary['best_lr'] = best_lr
        wandb_run.summary['final_loss'] = best_result['final_loss']
        if 'final_eval_loss' in best_result['loss_info']:
            wandb_run.summary['final_eval_loss'] = best_result['loss_info']['final_eval_loss']

    # Save training and evaluation losses for best learning rate in the format expected by visualization
    losses_path = f"results/vars/losses_{model_name_short}_idx_{args.steering_vector_idx}.pt"
    
    # Prepare loss data in the format expected by visualize_vector_losses.py
    loss_data = {
        best_lr: [{
            'train_losses': best_result['loss_info']['loss_history'],
            'eval_losses': best_result['loss_info'].get('eval_loss_history', []),
            'final_loss': best_result['final_loss'],
            'final_eval_loss': best_result['loss_info'].get('final_eval_loss', best_result['final_loss'])
        }]
    }
    
    torch.save(loss_data, losses_path)
    print(f"\nSaved training and evaluation losses for best learning rate {best_lr} to {losses_path}")
    
    # Test the best vector on an unseen example
    if test_examples:
        print(f"\nTesting {target_category} vector on an unseen example (best lr: {best_lr}):")
        # Use a random test example
        test_example = random.choice(test_examples)
        test_on_unseen_example(
            model, 
            tokenizer, 
            best_result['vector'], 
            args.layer, 
            test_example,
            max_new_tokens=args.test_max_tokens,
            steering_token_window=args.steering_token_window
        )
    
    # Free memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save best vector
    # TODO: make this dynamic
    vectors_path = f"{args.save_path}/{model_name_short}_layer6_idx{args.steering_vector_idx}.pt"
    optimized_vectors = {}
    optimized_vectors[target_category] = best_result['vector']
    torch.save(optimized_vectors, vectors_path)
    print(f"\nSaved optimized vectors to {vectors_path}")

    if wandb_run:
        artifact = wandb.Artifact(run_name, type='model')
        artifact.add_file(vectors_path)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

if __name__ == "__main__":
    main() 
# %%
