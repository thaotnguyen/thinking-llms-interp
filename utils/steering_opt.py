import torch
from typing import List, Tuple, Callable, Optional, Union
import dataclasses
from contextlib import contextmanager
import numpy as np
from tqdm import tqdm
import random
import gc
from torch.optim.lr_scheduler import LambdaLR

# context manager for running a HuggingFace Llama model with hooks
@contextmanager
def hf_hooks_contextmanager(model, hook_infos : List[Tuple[int, Callable]]):
	"""
	A context manager for running a HuggingFace Llama-like model with hooks (particularly steering hooks).

	Args:
		model (HuggingFace model): the model to hook into
		hook_infos: a list of pairs. The first element of each pair is the layer to hook into, and the second element is the hook function to attach.
	
	Example:
		# make and apply a steering hook to a HuggingFace model
		layer = 10
		hook_fn = steering_opt.make_steering_hook_hf(vector)
		# generate tokens while hooked
		with steering_opt.hf_hooks_contextmanager(model, [(layer, hook_fn)]):
			input_tokens = tokenizer("Hello, world.", return_tensors="pt")
			generated_tokens = model.generate(**input_tokens, max_new_tokens=10)
		# print generated tokens
		print(tokenizer.batch_decode(generated_tokens)[0])
	"""

	# set up hooks
	hooks = [ model.model.layers[cur_layer].register_forward_pre_hook(hook_fn) for cur_layer, hook_fn in hook_infos]
	# yield execution
	try:
		yield
	finally:
		# make sure to remove all hooks
		for hook in hooks: hook.remove()

# functions for making steering hooks
def make_steering_hook_hf(vector_, matrix=None, token=None):
	"""
	Makes a hook for steering the activations of a HuggingFace model.

	Args:
		vector_: a vector which will be added to the activations
		matrix (optional): a matrix, such that the product of that matrix with the activations will be added to the activations
		token (optional): an int or a slice denoting which tokens to apply steering to.
	"""
	if token is None:
		token = slice(None)
	def hook_fn(module, args):
		x = args[0]
		vector = vector_.to(x) if isinstance(vector_, torch.Tensor) else vector_
		x_sliced = x[:, token].detach().clone()
		x[:, token] = x_sliced + vector

		if matrix is not None:
			affine_term = torch.zeros_like(x)
			affine_term[:, token] = torch.einsum('...n, mn -> ...m', x_sliced, matrix.to(x))
			x = x + affine_term

		return x
	return hook_fn

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0):
	"""
	Creates a learning rate scheduler that:
	1. Linearly increases the learning rate from 0 to the initial LR during warmup
	2. Uses cosine annealing to decrease the learning rate from the initial LR to min_lr after warmup
	
	Args:
		optimizer: The optimizer for which to schedule the learning rate
		num_warmup_steps: The number of steps for the warmup phase
		num_training_steps: The total number of training steps
		min_lr: Minimum learning rate to decay to (default: 0)
	
	Returns:
		A PyTorch LambdaLR scheduler
	"""
	# Get initial learning rate from optimizer
	initial_lr = optimizer.param_groups[0]['lr']
	
	def lr_lambda(current_step):
		if current_step < num_warmup_steps:
			# Linear warmup
			return float(current_step) / float(max(1, num_warmup_steps))
		else:
			# Cosine decay from initial_lr to min_lr
			progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
			cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
			
			# Calculate factor that scales from initial_lr to min_lr
			return cosine_decay * (1.0 - min_lr / initial_lr) + min_lr / initial_lr
			
	return LambdaLR(optimizer, lr_lambda)

def optimize_vector_simple(model, tokenizer, prompts, target_completions, layer,
                          lr=0.01, max_iters=50, minibatch_size=32, 
                          warmup_steps=0, min_lr=0, grad_clip=None,
                          early_stopping_patience=5, early_stopping_min_delta=1e-4,
                          max_norm=None, starting_norm=1, debug=False,
                          return_info=True, return_loss_history=False,
                          steering_token_window=None,
                          eval_prompts=None, eval_target_completions=None):
    """
    Simplified steering vector optimization that minimizes next token prediction loss
    for target completions with proper minibatching.

	Args:
        model: HuggingFace model to optimize for
        tokenizer: Associated tokenizer
        prompts: List of prompts
        target_completions: List of target completions (same length as prompts)
        layer: Layer to apply steering vector to
        lr: Learning rate
        max_iters: Maximum optimization iterations
        minibatch_size: Size of minibatches for optimization
        warmup_steps: Number of warmup steps for learning rate
        min_lr: Minimum learning rate
        grad_clip: Gradient clipping value
        early_stopping_patience: Number of iterations to wait before early stopping
        early_stopping_min_delta: Minimum improvement required to reset early stopping counter
        max_norm: Maximum norm of the steering vector
        starting_norm: Starting norm of the steering vector
        debug: Whether to print debug information
        return_info: Whether to return optimization info
        return_loss_history: Whether to return loss history
        steering_token_window: If not None, apply the steering vector only to the last N tokens of the target completion.
        eval_prompts: List of evaluation prompts (optional)
        eval_target_completions: List of evaluation target completions (optional)
    
    Returns:
        vector: Optimized steering vector
        info: Optimization info (if return_info=True)
    """
    if len(prompts) != len(target_completions):
        raise ValueError("Number of prompts must equal number of target completions")
    
    if eval_prompts is not None and eval_target_completions is not None:
        if len(eval_prompts) != len(eval_target_completions):
            raise ValueError("Number of eval prompts must equal number of eval target completions")
    
    # Initialize steering vector
    d_model = model.config.hidden_size
    with torch.no_grad():
        vector = torch.randn(d_model, device=model.device)
        vector = starting_norm * vector / vector.norm()
    vector.requires_grad_(True)

    # Tokenize all prompts and target completions
    prompt_tokens = []
    target_tokens = []
    prompt_lengths = []
    target_lengths = []

    for prompt, target_completion in zip(prompts, target_completions):
        # Tokenize prompt
        prompt_tokenized = tokenizer(prompt, return_tensors='pt')
        prompt_tokens.append(prompt_tokenized['input_ids'][0])
        prompt_lengths.append(len(prompt_tokenized['input_ids'][0]))

        # Tokenize target completion
        target_tokenized = tokenizer(target_completion, return_tensors='pt')
        target_tokens.append(target_tokenized['input_ids'][0])
        target_lengths.append(len(target_tokenized['input_ids'][0]))

    # Tokenize evaluation data if provided
    eval_prompt_tokens = []
    eval_target_tokens = []
    eval_prompt_lengths = []
    eval_target_lengths = []

    if eval_prompts is not None and eval_target_completions is not None:
        for prompt, target_completion in zip(eval_prompts, eval_target_completions):
            # Tokenize prompt
            prompt_tokenized = tokenizer(prompt, return_tensors='pt')
            eval_prompt_tokens.append(prompt_tokenized['input_ids'][0])
            eval_prompt_lengths.append(len(prompt_tokenized['input_ids'][0]))

            # Tokenize target completion
            target_tokenized = tokenizer(target_completion, return_tensors='pt')
            eval_target_tokens.append(target_tokenized['input_ids'][0])
            eval_target_lengths.append(len(target_tokenized['input_ids'][0]))

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam([vector], lr=lr)

    if max_iters is not None:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_iters,
            min_lr=min_lr
        )

    # Training loop variables
    loss_history = []
    eval_loss_history = []
    best_loss = float('inf')
    best_vector = vector.detach().clone()
    early_stopping_counter = 0
    prev_loss = None
    
    # Setup progress bar (epochs)
    pbar = tqdm(total=max_iters, desc="Optimizing vector", dynamic_ncols=True)

    latest_train_loss = None
    latest_eval_loss = None
    
    for iteration in range(max_iters):
        # Shuffle data
        indices = list(range(len(prompts)))
        random.shuffle(indices)
        
        total_loss = 0
        num_batches = 0
        
        # Process in minibatches
        for batch_start in range(0, len(indices), minibatch_size):
            batch_end = min(batch_start + minibatch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Prepare batch data
            batch_prompts = [prompt_tokens[idx] for idx in batch_indices]
            batch_targets = [target_tokens[idx] for idx in batch_indices]
            batch_prompt_lengths = [prompt_lengths[idx] for idx in batch_indices]
            batch_target_lengths = [target_lengths[idx] for idx in batch_indices]
            
            # Create full sequences: prompt + target for each example
            batch_sequences = []
            for prompt_tok, target_tok in zip(batch_prompts, batch_targets):
                full_sequence = torch.cat([prompt_tok, target_tok])
                batch_sequences.append(full_sequence)
            
            # Find max sequence length for padding
            max_seq_len = max(len(seq) for seq in batch_sequences)
            
            # Left-pad sequences
            padded_sequences = []
            attention_masks = []
            
            for seq in batch_sequences:
                seq_len = len(seq)
                if seq_len < max_seq_len:
                    # Left-pad with pad_token_id
                    padding_length = max_seq_len - seq_len
                    padded_seq = torch.cat([
                        torch.full((padding_length,), tokenizer.pad_token_id, dtype=seq.dtype, device=seq.device),
                        seq
                    ])
                    attention_mask = torch.cat([
                        torch.zeros(padding_length, dtype=torch.long, device=seq.device),
                        torch.ones(seq_len, dtype=torch.long, device=seq.device)
                    ])
                else:
                    padded_seq = seq
                    attention_mask = torch.ones(seq_len, dtype=seq.dtype, device=seq.device)
                
                padded_sequences.append(padded_seq)
                attention_masks.append(attention_mask)
            
            # Stack into batch tensors
            batch_input_ids = torch.stack(padded_sequences).to(model.device)
            batch_attention_mask = torch.stack(attention_masks).to(model.device)
            
            # Determine steering token positions for each example in the batch
            steering_token_slices = []
            for i, (prompt_len, target_len) in enumerate(zip(batch_prompt_lengths, batch_target_lengths)):
                if steering_token_window is None:
                    # Apply to all of the target completion
                    steering_start = prompt_len
                else:
                    # Apply to the last N tokens of the target completion
                    steering_start = prompt_len + max(0, target_len - steering_token_window)
                
                # Adjust for left padding
                padding_length = max_seq_len - (prompt_len + target_len)
                steering_start += padding_length
                steering_token_slices.append(slice(steering_start, None))
            
            # Create steering hook that applies different slices to different examples
            def batch_steering_hook(module, args):
                x = args[0]  # [batch_size, seq_len, hidden_dim]
                batch_size = x.shape[0]
                
                # Apply steering vector to each example according to its slice
                for i in range(batch_size):
                    steering_slice = steering_token_slices[i]
                    x[i, steering_slice] = x[i, steering_slice] + vector
                
                return x
            
            hook_infos = [(layer, batch_steering_hook)]
            
            # Forward pass with steering
            with hf_hooks_contextmanager(model, hook_infos):
                inputs = {
                    'input_ids': batch_input_ids,
                    'attention_mask': batch_attention_mask
                }
                outputs = model(**inputs)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Calculate loss for each example in the batch
            batch_loss = 0
            valid_examples = 0
            
            for i, (prompt_len, target_len) in enumerate(zip(batch_prompt_lengths, batch_target_lengths)):
                # Get logits and labels for this example
                example_logits = logits[i]  # [seq_len, vocab_size]
                example_input_ids = batch_input_ids[i]  # [seq_len]
                example_attention_mask = batch_attention_mask[i]  # [seq_len]
                
                # Find the actual sequence length (excluding padding)
                actual_length = example_attention_mask.sum().item()
                
                # Calculate loss only on the target completion part
                # We need to shift the logits and labels to predict next token
                shift_logits = example_logits[:-1, :].contiguous()  # Remove last token from logits
                shift_labels = example_input_ids[1:].contiguous()  # Remove first token from labels
                
                # Get the relevant portion for target completion
                # -1 to account for the shift in both start and end
                target_start = prompt_len - 1
                target_end = actual_length - 1
                
                if target_start >= target_end or target_start < 0:
                    continue
                
                target_logits = shift_logits[target_start:target_end, :]
                target_labels = shift_labels[target_start:target_end]
                
                # Skip if we have no valid tokens to compute loss on
                if target_logits.size(0) == 0:
                    continue
                
                # Calculate cross entropy loss
                loss = torch.nn.functional.cross_entropy(target_logits, target_labels, reduction='mean')
                batch_loss += loss
                valid_examples += 1
            
            # Average loss over valid examples in batch
            if valid_examples > 0:
                batch_loss = batch_loss / valid_examples
                
                # Backward pass
                batch_loss.backward()
                
                # Gradient clipping
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_([vector], grad_clip)
                
                # Optimizer step
                optimizer.step()
                
                # Apply norm constraint
                if max_norm is not None:
                    with torch.no_grad():
                        current_norm = vector.norm()
                        if current_norm > max_norm:
                            vector[:] = max_norm * vector / current_norm
                
                total_loss += batch_loss.item()
                num_batches += 1
                
                # Store batch loss in history
                if return_loss_history:
                    loss_history.append(batch_loss.item())
                
                # Update latest train loss
                latest_train_loss = batch_loss.item()
                
                # Update progress bar description
                if latest_eval_loss is not None:
                    pbar.set_description(f"Epoch {iteration+1}/{max_iters} - Train: {latest_train_loss:.6f} Eval: {latest_eval_loss:.6f}")
                else:
                    pbar.set_description(f"Epoch {iteration+1}/{max_iters} - Train: {latest_train_loss:.6f}")
            else:
                # Skip this batch if no valid examples
                continue
        
        # Compute evaluation loss at the end of each epoch if evaluation data is provided
        eval_loss = None
        if eval_prompts is not None and eval_target_completions is not None:
            eval_loss = compute_evaluation_loss(
                model, tokenizer, vector, layer,
                eval_prompt_tokens, eval_target_tokens,
                eval_prompt_lengths, eval_target_lengths,
                steering_token_window
            )
            eval_loss_history.append(eval_loss)
            latest_eval_loss = eval_loss
            # Update progress bar description with latest eval loss
            if latest_train_loss is not None:
                pbar.set_description(f"Epoch {iteration+1}/{max_iters} - Train: {latest_train_loss:.6f} Eval: {latest_eval_loss:.6f}")
            else:
                pbar.set_description(f"Epoch {iteration+1}/{max_iters} - Eval: {latest_eval_loss:.6f}")
        
        # Early stopping logic - use eval loss if available, otherwise use train loss
        current_loss = eval_loss if eval_loss is not None else total_loss / num_batches if num_batches > 0 else float('inf')
        
        if prev_loss is not None:
            if prev_loss - current_loss > early_stopping_min_delta:
                # Improvement found, update best model
                best_loss = current_loss
                best_vector = vector.detach().clone()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if debug: 
                    print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
        
        prev_loss = current_loss

        # Check early stopping
        if early_stopping_counter >= early_stopping_patience:
            if debug:
                print(f"Early stopping triggered. No improvement of more than {early_stopping_min_delta} for {early_stopping_patience} iterations.")
            # Restore best parameters
            vector.data.copy_(best_vector.data)
            break

        # Step scheduler
        if max_iters is not None:
            scheduler.step()

        # Increment progress bar at the end of the epoch
        pbar.update(1)
    
    pbar.close()

    if debug:
        final_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Final train loss: {final_train_loss}")
        if eval_loss is not None:
            print(f"Final eval loss: {eval_loss}")
        print(f"Number of iterations: {iteration + 1}")
    
    # Prepare return values
    if return_info:
        final_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        info = {
            'iters': iteration + 1,
            'final_loss': final_train_loss,
            'norm': vector.norm().item(),
            'best_loss': best_loss
        }
        if eval_loss is not None:
            info['final_eval_loss'] = eval_loss
        if return_loss_history:
            info['loss_history'] = loss_history
            if eval_loss_history:
                info['eval_loss_history'] = eval_loss_history
        return vector, info
    else:
        return vector

def compute_evaluation_loss(model, tokenizer, vector, layer,
                          eval_prompt_tokens, eval_target_tokens,
                          eval_prompt_lengths, eval_target_lengths,
                          steering_token_window=None):
    """
    Compute evaluation loss on evaluation data.

    Args:
        model: HuggingFace model
        tokenizer: Associated tokenizer
        vector: Current steering vector
        layer: Layer to apply steering vector to
        eval_prompt_tokens: List of tokenized evaluation prompts
        eval_target_tokens: List of tokenized evaluation target completions
        eval_prompt_lengths: List of evaluation prompt lengths
        eval_target_lengths: List of evaluation target lengths
        steering_token_window: If not None, apply steering to last N tokens

    Returns:
        float: Average evaluation loss
    """
    model.eval()
    total_eval_loss = 0
    valid_eval_examples = 0
    
    with torch.no_grad():
        # Process evaluation data in batches
        for i in range(len(eval_prompt_tokens)):
            prompt_tok = eval_prompt_tokens[i]
            target_tok = eval_target_tokens[i]
            prompt_len = eval_prompt_lengths[i]
            target_len = eval_target_lengths[i]
            
            # Create full sequence
            full_sequence = torch.cat([prompt_tok, target_tok])
            
            # Determine steering token positions
            if steering_token_window is None:
                steering_start = prompt_len
            else:
                steering_start = prompt_len + max(0, target_len - steering_token_window)
            steering_token_slice = slice(steering_start, None)
            
            # Create steering hook
            hook_fn = make_steering_hook_hf(vector, token=steering_token_slice)
            hook_infos = [(layer, hook_fn)]
            
            # Forward pass with steering
            with hf_hooks_contextmanager(model, hook_infos):
                inputs = {'input_ids': full_sequence.unsqueeze(0).to(model.device)}
                outputs = model(**inputs)
                logits = outputs.logits[0]  # Remove batch dimension
            
            # Calculate loss for target tokens only
            # We need to shift the logits and labels to predict next token
            shift_logits = logits[:-1, :].contiguous()  # Remove last token from logits
            shift_labels = full_sequence[1:].contiguous()  # Remove first token from labels
            
            # Get the relevant portion for target completion
            target_start = prompt_len - 1
            target_end = len(full_sequence) - 1
            
            if target_start >= target_end or target_start < 0:
                continue
            
            target_logits = shift_logits[target_start:target_end, :]
            target_labels = shift_labels[target_start:target_end]
            
            # Skip if we have no valid tokens to compute loss on
            if target_logits.size(0) == 0:
                continue
            
            # Calculate cross entropy loss
            loss = torch.nn.functional.cross_entropy(target_logits, target_labels, reduction='mean')
            total_eval_loss += loss.item()
            valid_eval_examples += 1
    
    model.train()
    
    # Return average evaluation loss
    return total_eval_loss / valid_eval_examples if valid_eval_examples > 0 else float('inf')