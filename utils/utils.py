import dotenv
dotenv.load_dotenv("../.env")

import torch
from nnsight import LanguageModel
from tqdm import tqdm
import gc
import time
import random
import torch.nn as nn
import anthropic
import os
from openai import OpenAI
import json
import re
import numpy as np

class LinearProbe(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)
        
    def forward(self, x):
        return self.linear(x)


def chat(prompt, model="gpt-4.1", max_tokens=28000):

    model_provider = ""

    if model in ["gpt-4o", "gpt-4.1"]:
        model_provider = "openai"
        client = OpenAI()
    elif model in ["claude-3-opus", "claude-3-7-sonnet", "claude-3-5-haiku"]:
        model_provider = "anthropic"
        client = anthropic.Anthropic()
    elif model in ["deepseek-v3", "gemini-2-0-think", "gemini-2-0-flash", "deepseek-r1"]:
        model_provider = "openrouter"
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    # try 3 times with 3 second sleep between attempts
    for _ in range(3):
        try:
            if model_provider == "openai":
                client = OpenAI()
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=1e-19,
                )
                return response.choices[0].message.content
            elif model_provider == "anthropic":
                model_mapping = {
                    "claude-3-opus": "claude-3-opus-latest",
                    "claude-3-7-sonnet": "claude-3-7-sonnet-latest",
                    "claude-3-5-haiku": "claude-3-5-haiku-latest"
                }

                if model == "claude-3-7-sonnet":
                    response = client.messages.create(
                        model=model_mapping[model],
                        temperature=1,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        thinking = {
                            "type": "enabled",
                            "budget_tokens": max_tokens
                        },
                        max_tokens=max_tokens+1
                    )

                    thinking_response = response.content[0].thinking
                    answer_response = response.content[1].text

                    return f"<think>{thinking_response}\n</think>\n{answer_response}"

                else:
                    response = client.messages.create(
                        model=model_mapping[model],
                        temperature=1e-19,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        max_tokens=max_tokens
                    )

                    return response.content[0].text
            elif model_provider == "openrouter":
                # Map model names to OpenRouter model IDs
                model_mapping = {
                    "deepseek-r1": "deepseek/deepseek-r1",
                    "deepseek-v3": "deepseek/deepseek-chat",
                    "gemini-2-0-think": "google/gemini-2.0-flash-thinking-exp:free",
                    "gemini-2-0-flash": "google/gemini-2.0-flash-001"
                }
                
                response = client.chat.completions.create(
                    model=model_mapping[model],
                    extra_body={},
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=1e-19,
                    max_tokens=max_tokens
                )

                if hasattr(response.choices[0].message, "reasoning"):
                    thinking_response = response.choices[0].message.reasoning
                    answer_response = response.choices[0].message.content

                    return f"<think>{thinking_response}\n</think>\n{answer_response}"
                else:
                    return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(20)

    return None

def load_sae(model_id, layer, n_clusters, load_base_decoder=False):
    sae_path = f'../train-saes/results/vars/saes/sae_{model_id}_layer{layer}_clusters{n_clusters}.pt'
    if not os.path.exists(sae_path):
        raise FileNotFoundError(f"SAE model not found at {sae_path}")
        
    checkpoint = torch.load(sae_path)
    
    # Create SAE model
    sae = SAE(checkpoint['input_dim'], checkpoint['num_latents'], k=checkpoint.get('topk', 3))
    
    # Load weights
    sae.encoder.weight.data = checkpoint['encoder_weight']
    sae.encoder.bias.data = checkpoint['encoder_bias']
    sae.W_dec.data = checkpoint['decoder_weight']
    sae.b_dec.data = checkpoint['b_dec']
    
    print(f"Loaded SAE model from {sae_path}")

    return sae, checkpoint

def generate_cluster_description(examples, model="gpt-4.1", n_trace_examples=0, model_name=None):
    """
    Generate a concise title and description for a cluster based on the top k examples.
    
    Args:
        examples (list): List of text examples from the cluster
        model (str): Model to use for generating the description
        n_trace_examples (int): Number of full reasoning trace examples to include in the prompt
        model_name (str): Name of the model whose responses should be loaded for trace examples
        
    Returns:
        tuple: (title, description) where both are strings
    """    
    # Prepare trace examples if requested
    trace_examples_text = ""
    if n_trace_examples > 0 and model_name is not None:
        try:
            # Get model identifier for file naming
            model_id = model_name.split('/')[-1].lower()
            responses_json_path = f"../generate-responses/results/vars/responses_{model_id}.json"
            
            # Load responses
            with open(responses_json_path, 'r') as f:
                responses_data = json.load(f)
            
            # Select random examples
            trace_samples = random.sample(responses_data, min(n_trace_examples, len(responses_data)))
            
            # Extract thinking processes
            trace_examples = []
            for sample in trace_samples:
                if sample.get("thinking_process"):
                    trace_examples.append(sample["thinking_process"])
            
            if trace_examples:
                trace_examples_text = "Here are some full reasoning traces to help understand the context:\n'''\n"
                for i, trace in enumerate(trace_examples):
                    trace_examples_text += f"TRACE {i+1}:\n{trace}\n\n"
                trace_examples_text += "'''"
        except Exception as e:
            print(f"Error loading trace examples: {e}")
    
    # Create a prompt for the model
    prompt = f"""Analyze the following {len(examples)} sentences from an LLM reasoning trace. These sentences are grouped into a cluster based on their similar role or function in the reasoning process.

Your task is to identify the precise cognitive function these sentences serve in the reasoning process. Consider:
1. The reasoning strategy or cognitive operation being performed
2. Whether these sentences tend to appear in a specific position in reasoning (if applicable)

{trace_examples_text}

Examples:
'''
{chr(10).join([f"- {example}" for example in examples])}
'''

Look for:
- Shared reasoning strategies or cognitive mechanisms
- Common linguistic patterns or structures
- Positional attributes (only if clearly evident)
- Functional role within the overall reasoning process

Your response should be in this exact format:
Title: [concise title naming the specific reasoning function]
Description: [2-3 sentences explaining (1) what this function does, (2) what is INCLUDED and NOT INCLUDED in this category, and (3) position in reasoning if relevant]

Avoid overly general descriptions. Be precise enough that someone could reliably identify new examples of this reasoning function.
"""
        
    # Get the response from the model
    response = chat(prompt, model=model)
    
    # Parse the response to extract title and description
    title = "Unnamed Cluster"
    description = "No description available"
    
    title_match = re.search(r"Title:\s*(.*?)(?:\n|$)", response)
    if title_match:
        title = title_match.group(1).strip()
        
    desc_match = re.search(r"Description:\s*(.*?)(?:\n|$)", response)
    if desc_match:
        description = desc_match.group(1).strip()
    
    return title, description

def simplify_category_name(category_name):
    """
    Simplify a category name by extracting just the number if it matches 'Category N'
    or return the original name if not.
    """
    import re
    match = re.match(r'Category\s+(\d+)', category_name)
    if match:
        return match.group(1)
    return category_name

def completeness_autograder(sentences, categories, model="gpt-4.1"):
    """
    Autograder that evaluates if sentences belong to any of the provided categories.
    
    Args:
        sentences (list): List of sentences to evaluate
        categories (list): List of tuples where each tuple is (cluster_id, title, description)
    
    Returns:
        dict: Statistics about category assignments including the fraction of sentences assigned/not assigned
    """
    # Format the categories into a readable list for the prompt
    categories_text = "\n\n".join([f"Category {cluster_id}: {title}\nDescription: {description}" 
                                  for cluster_id, title, description in categories])
    
    # Format the sentences into a numbered list
    sentences_text = "\n\n".join([f"Sentence {i}: {sentence}" for i, sentence in enumerate(sentences)])

    prompt = f"""# Task: Categorize Sentences of Reasoning Traces

You are an expert at categorizing the sentences of reasoning traces into predefined categories. Your task is to analyze each sentence and assign it to the most appropriate category based on the provided descriptions. If a sentence does not fit into any category, label it as "None".

## Categories:
{categories_text}

## Sentences to Categorize:
{sentences_text}

## Instructions:
1. For each sentence, carefully consider if it fits into one of the defined categories.
2. Assign exactly ONE category to each sentence if applicable, or "None" if it doesn't fit any category.
3. Provide your response in the exact format specified below.

## Response Format:
Your response must follow this exact JSON format:
```json
{{
  "categorizations": [
    {{
      "sentence_id": <sentence idx>,
      "assigned_category": "Category <category idx>" (not the title, just the category index) or "None",
      "explanation": "Brief explanation of your reasoning"
    }},
    ... (repeat for all sentences)
  ]
}}
```

Only include the JSON object in your response, with no additional text before or after.
"""
        
    # Call the chat API to get the categorization results
    response = chat(prompt, model=model)
    
    # Parse the response to extract the JSON
    try:
        import re
        import json
        
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find just the JSON object
            json_match = re.search(r'{\s*"categorizations":\s*\[[\s\S]*?\]\s*}', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                # If all else fails, just try to use the entire response
                json_str = response
        
        result = json.loads(json_str)
        
        # Count the number of sentences assigned to each category and those not assigned
        total_sentences = len(sentences)
        assigned = 0
        not_assigned = 0
        category_counts = {str(cluster_id): 0 for cluster_id, _, _ in categories}
        category_counts["None"] = 0
        
        for item in result["categorizations"]:
            category = item["assigned_category"]
            if category == "None":
                not_assigned += 1
                category_counts["None"] += 1
            else:
                assigned += 1
                # Extract just the cluster ID from "Category N" format
                category_id = simplify_category_name(category)
                category_counts[category_id] = category_counts.get(category_id, 0) + 1
        
        # Calculate fractions
        assigned_fraction = assigned / total_sentences if total_sentences > 0 else 0
        not_assigned_fraction = not_assigned / total_sentences if total_sentences > 0 else 0
        
        return {
            "total_sentences": total_sentences,
            "assigned": assigned,
            "not_assigned": not_assigned,
            "assigned_fraction": assigned_fraction,
            "not_assigned_fraction": not_assigned_fraction,
            "category_counts": category_counts,
            "categorizations": result["categorizations"]
        }
    
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {response}")
        return {
            "error": str(e),
            "total_sentences": len(sentences),
            "assigned": 0,
            "not_assigned": len(sentences),
            "assigned_fraction": 0,
            "not_assigned_fraction": 1.0,
            "raw_response": response
        }

def accuracy_autograder(sentences, categories, ground_truth_labels, model="gpt-4.1", n_autograder_examples=30):
    """
    Binary autograder that evaluates each cluster independently against examples from outside the cluster.
    
    Args:
        sentences (list): List of all sentences to potentially sample from
        categories (list): List of tuples where each tuple is (cluster_id, title, description)
        ground_truth_labels (list): List of cluster IDs (as strings) for each sentence in sentences
        model (str): Model to use for the autograding
        n_autograder_examples (int): Number of examples to sample from each cluster for testing
    
    Returns:
        dict: Metrics including precision, recall, accuracy and F1 score for each category
    """
    results = {}
    
    # Get a mapping from sentence index to cluster ID for easy lookup
    sentence_to_cluster = {i: label for i, label in enumerate(ground_truth_labels)}
    
    # For each category, evaluate independently
    for cluster_id, title, description in categories:
        cluster_id_str = str(cluster_id)
        
        # Find all examples in this cluster and not in this cluster
        in_cluster_indices = [i for i, label in enumerate(ground_truth_labels) if label == cluster_id_str]
        out_cluster_indices = [i for i, label in enumerate(ground_truth_labels) if label != cluster_id_str]
        
        # Get n_autograder_examples from the current cluster
        from_cluster_count = min(len(in_cluster_indices), n_autograder_examples)
        in_cluster_sample = random.sample(in_cluster_indices, from_cluster_count)
        
        # Get equal number of examples from outside the cluster
        from_outside_count = min(len(out_cluster_indices), n_autograder_examples)
        out_cluster_sample = random.sample(out_cluster_indices, from_outside_count)
        
        # Combine the samples and remember the ground truth
        test_indices = in_cluster_sample + out_cluster_sample
        test_sentences = [sentences[i] for i in test_indices]
        test_ground_truth = ["Yes" if i in in_cluster_sample else "No" for i in test_indices]
        
        # Shuffle to avoid position bias
        combined = list(zip(range(len(test_indices)), test_sentences, test_ground_truth))
        random.shuffle(combined)
        shuffled_indices, test_sentences, test_ground_truth = zip(*combined)
        
        # Create a prompt for binary classification
        prompt = f"""# Task: Binary Classification of Reasoning Sentences

You are an expert at analyzing reasoning traces. I'll provide a description of a specific reasoning function or pattern, along with several example sentences. Your task is to determine whether each sentence belongs to this category or not.

## Category Description:
Title: {title}
Description: {description}

## Sentences to Classify:
{chr(10).join([f"Sentence {i}: {sentence}" for i, sentence in enumerate(test_sentences)])}

## Instructions:
1. For each sentence, determine if it belongs to the described category.
2. Respond with "Yes" if it belongs to the category, or "No" if it does not.
3. Provide your response in the exact format specified below.

## Response Format:
Your response must follow this exact JSON format:
```json
{{
  "classifications": [
    {{
      "sentence_id": <sentence idx>,
      "belongs_to_category": "Yes" or "No",
      "explanation": "Brief explanation of your reasoning"
    }},
    ... (repeat for all sentences)
  ]
}}
```

Only include the JSON object in your response, with no additional text before or after.
"""
        
        # Call the chat API to get the classification results
        response = chat(prompt, model=model)
        
        # Parse the response to extract the JSON
        try:
            import re
            import json
            
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find just the JSON object
                json_match = re.search(r'{\s*"classifications":\s*\[[\s\S]*?\]\s*}', response)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # If all else fails, just try to use the entire response
                    json_str = response
            
            result = json.loads(json_str)
            
            # Compute metrics for this cluster
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            predictions = []
            
            for item in result["classifications"]:
                sentence_idx = item["sentence_id"]
                belongs = item["belongs_to_category"]
                predictions.append(belongs)
                
                true_label = test_ground_truth[sentence_idx]
                
                if belongs == "Yes" and true_label == "Yes":
                    true_positives += 1
                elif belongs == "Yes" and true_label == "No":
                    false_positives += 1
                elif belongs == "No" and true_label == "Yes":
                    false_negatives += 1
                elif belongs == "No" and true_label == "No":
                    true_negatives += 1
            
            # Calculate metrics
            accuracy = (true_positives + true_negatives) / len(test_sentences) if test_sentences else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[cluster_id_str] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
                "predictions": predictions,
                "classifications": result["classifications"]
            }
        
        except Exception as e:
            print(f"Error in accuracy autograder for cluster {cluster_id}: {e}")
            print(f"Raw response: {response}")
            results[cluster_id_str] = {
                "error": str(e),
                "raw_response": response
            }
    
    # Calculate overall averages across all clusters
    if results:
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            avg_accuracy = sum(r["accuracy"] for r in valid_results.values()) / len(valid_results)
            avg_precision = sum(r["precision"] for r in valid_results.values()) / len(valid_results)
            avg_recall = sum(r["recall"] for r in valid_results.values()) / len(valid_results)
            avg_f1 = sum(r["f1"] for r in valid_results.values()) / len(valid_results)
            
            results["avg"] = {
                "accuracy": avg_accuracy,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            }
    
    return results


def get_char_to_token_map(text, tokenizer):
    """Create a mapping from character positions to token positions"""
    token_offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)['offset_mapping']
    
    # Create mapping from character position to token index
    char_to_token = {}
    for token_idx, (start, end) in enumerate(token_offsets):
        for char_pos in range(start, end):
            char_to_token[char_pos] = token_idx
            
    return char_to_token

def process_saved_responses(model_name, n_examples, model, tokenizer, layer):
    """Load and process saved responses to get activations"""
    print(f"Processing saved responses for {model_name}...")
    
    # Load model and tokenizer
    model_id = model_name.split('/')[-1].lower()
    responses_json_path = f"../generate-responses/results/vars/responses_{model_id}.json"
    
    print(f"Loading responses from {responses_json_path}...")
    try:
        with open(responses_json_path, 'r') as f:
            responses_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {responses_json_path} not found.")
        return [], []
    
    # Limit to n_examples
    import random
    random.shuffle(responses_data)
    responses_data = responses_data[:n_examples]
        
    # Extract text segments and their activations
    all_activations = []
    all_texts = []
    
    overall_running_mean = torch.zeros(1, model.config.hidden_size)
    overall_running_count = 0

    print(f"Extracting activations for {n_examples} sentences...")
    from tqdm import tqdm
    for response_data in tqdm(responses_data):
        if not response_data.get("thinking_process"):
            continue
            
        # Get the thinking process text
        thinking_text = response_data["thinking_process"]
        full_response = response_data["full_response"]
        
        # Split into sentences using regex
        sentences = re.split(r'[.!?;]', thinking_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [s for s in sentences if len(s.split()) >= 3]
        
        # Encode the full response to get input_ids
        input_ids = tokenizer.encode(full_response, return_tensors="pt").to(model.device)
        
        # Get layer activations
        with model.trace({
            "input_ids": input_ids, 
            "attention_mask": (input_ids != tokenizer.pad_token_id).long()
        }) as tracer:
            layer_outputs = model.model.layers[layer].output[0].save()
        
        # Convert layer outputs to numpy arrays
        layer_outputs = layer_outputs.detach().to(torch.float32)
        
        # Create character to token mapping
        char_to_token = get_char_to_token_map(full_response, tokenizer)
        
        # Process each sentence
        min_token_start = float('inf')
        max_token_end = -float('inf')
        for sentence in sentences:
            # Find this sentence in the original text
            text_pos = full_response.find(sentence)
            if text_pos >= 0:
                # Get start and end token positions
                token_start = char_to_token.get(text_pos, None)
                token_end = char_to_token.get(text_pos + len(sentence), None)
                
                if token_start is not None and token_end is not None and token_start < token_end:
                    if token_start < min_token_start:
                        min_token_start = token_start
                    if token_end > max_token_end:
                        max_token_end = token_end

                    # Extract activations for this segment
                    segment_activations = layer_outputs[:, token_start-1:token_end, :].mean(dim=1).cpu()  # Average over tokens
                                        
                    # Save the result
                    all_activations.append(segment_activations)  # Store as torch tensor
                    all_texts.append(sentence)
    
        if min_token_start < layer_outputs.shape[1] and max_token_end > 0:
            vector = layer_outputs[:,min_token_start:max_token_end,:].mean(dim=1).cpu()
            overall_running_mean = overall_running_mean + (vector - overall_running_mean) / (overall_running_count + 1)
            overall_running_count += 1

    print(f"Found {len(all_activations)} sentences with activations across {overall_running_count} examples")

    return all_activations, all_texts, overall_running_mean

def load_model(device="cuda:0", load_in_8bit=False, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    """
    Load model, tokenizer and mean vectors. Optionally compute feature vectors.
    
    Args:
        load_in_8bit (bool): If True, load the model in 8-bit mode
        model_name (str): Name/path of the model to load
    """
    model = LanguageModel(model_name, dispatch=True, load_in_8bit=load_in_8bit, device_map=device, torch_dtype=torch.bfloat16)
    
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.top_k=None
    model.generation_config.do_sample=False
    
    tokenizer = model.tokenizer

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def get_latent_descriptions(model_id, layer, n_clusters):
    """Get titles and descriptions for SAE latents"""
    results_path = f'../train-saes/results/vars/sae_topk_results_{model_id}_layer{layer}.json'
    
    if not os.path.exists(results_path):
        print(f"Warning: Results file not found at {results_path}")
        return {}
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Extract category descriptions for optimal_n_clusters
        if str(n_clusters) in results.get('detailed_results', {}):
            optimal_results = results['detailed_results'][str(n_clusters)]
            if 'categories' in optimal_results:
                categories = {}
                for cluster_id, title, description in optimal_results['categories']:
                    categories[int(cluster_id)] = {'title': title, 'description': description}
                return categories
    except Exception as e:
        print(f"Error loading cluster descriptions: {e}")
    
    return {}

def custom_generate_with_steering(model, tokenizer, input_ids, max_new_tokens, steering_vector=None, layer=None, normalize=False, coefficient=1.0):
    """
    Generate text while steering with a specific feature vector.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        input_ids: Input token ids
        max_new_tokens: Maximum number of tokens to generate
        steering_vector: Vector to use for steering (should match model hidden size)
        layer: Layer index to apply steering to
        coefficient: Strength of steering (higher values = stronger effect)
        sae: Sparse Autoencoder model to use for activation-based steering
    """
    model_layers = model.model.layers

    with model.generate(
        {
            "input_ids": input_ids, 
            "attention_mask": (input_ids != tokenizer.pad_token_id).long()
        },
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    ) as tracer:
        # Apply .all() to model to ensure interventions work across all generations
        model_layers.all()

        if steering_vector is not None and layer is not None:
            # Convert steering vector to correct device and dtype if needed
            steering_vector = steering_vector.to(model.device).to(model.dtype)
            avg_norm = model.model.layers[layer].output[0][:, 1:, :].norm(dim=-1).mean(dim=1)
            if normalize:
                steering_vector = steering_vector.unsqueeze(0).unsqueeze(0) * avg_norm
            model.model.layers[layer].output[0][:, 1:, :] += coefficient * steering_vector
        
        outputs = model.generator.output.save()
                    
    return outputs

def get_random_distinct_colors(labels):
    """
    Generate random distinct ANSI colors for each label.
    
    Args:
        labels: List of label names
        
    Returns:
        Dictionary mapping labels to ANSI color codes
    """
    import random
    
    # List of distinct ANSI colors (excluding black, white, and hard-to-see colors)
    # Format is "\033[COLORm" where COLOR is a number between 31-96
    distinct_colors = [
        "\033[31m",  # Red
        "\033[32m",  # Green
        "\033[33m",  # Yellow
        "\033[34m",  # Blue
        "\033[35m",  # Magenta
        "\033[36m",  # Cyan
        "\033[91m",  # Bright Red
        "\033[92m",  # Bright Green
        "\033[93m",  # Bright Yellow
        "\033[94m",  # Bright Blue
        "\033[95m",  # Bright Magenta
        "\033[96m",  # Bright Cyan
    ]
    
    # Shuffle the colors to randomize them
    random.shuffle(distinct_colors)
    
    # Ensure we have enough colors
    if len(labels) > len(distinct_colors):
        # If we need more colors, create additional ones with random RGB values
        additional_needed = len(labels) - len(distinct_colors)
        for _ in range(additional_needed):
            # Generate random RGB foreground color (38;2;r;g;b)
            r, g, b = random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)
            # Ensure colors are distinct by checking minimum distance from existing colors
            # (simplified approach)
            distinct_colors.append(f"\033[38;2;{r};{g};{b}m")
    
    # Assign colors to labels
    label_colors = {}
    for i, label in enumerate(labels):
        label_colors[label] = distinct_colors[i % len(distinct_colors)]
    
    return label_colors

def custom_hybrid_generate(
        thinking_model, 
        base_model,
        base_tokenizer,
        input_ids, 
        max_new_tokens, 
        baseline_method="probe",  # Options: "probe", "random", "norm_diff", "kl_div"
        baseline_config=None,  # Configuration for the baseline method
        warmup=0,
        show_progress=True,
        color_output=False):
    """
    Unified hybrid generate function that supports different baseline methods.
    
    Args:
        thinking_model: The thinking model to use
        base_model: The base model to use
        base_tokenizer: The tokenizer
        input_ids: Input token ids (can be batched)
        max_new_tokens: Maximum number of tokens to generate
        baseline_method: The baseline method to use ("probe", "random", "norm_diff", "kl_div")
        baseline_config: Configuration for the baseline method
        warmup: Number of warmup tokens
        show_progress: Whether to show progress bar
        color_output: Whether to color the output
    """
    # Get the device of the thinking model
    device = thinking_model.device
    
    # Handle batched input
    batch_size = input_ids.shape[0]
    base_generated_ids = input_ids.clone().cpu()
    
    # Get random distinct colors for labels
    if baseline_method == "probe" and baseline_config is not None:
        all_labels = list(baseline_config.get("label_to_idx", {}).keys())
        all_labels.append("warmup")  # Add warmup to the list of labels
        label_colors = get_random_distinct_colors(all_labels)
    else:
        # For other methods, just create colors for the methods themselves
        all_labels = ["warmup", "random", "norm_diff", "kl_div"]
        label_colors = get_random_distinct_colors(all_labels)
    
    # Color for warmup
    if "warmup" not in label_colors:
        label_colors["warmup"] = "\033[90m"  # Default to gray if not in random assignment
    
    iterator = range(max_new_tokens)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating response")
    
    # Track model usage and forced tokens for each sequence in batch
    base_model_tokens = [0] * batch_size
    thinking_model_tokens = [0] * batch_size
    forced_tokens = [{} for _ in range(batch_size)]  # List of dictionaries to track token frequencies
    
    # Three possible states: 0 = not forced, 1 = attempted force (models agreed), 2 = successful force (models disagreed)
    forced_states = [[] for _ in range(batch_size)]  
    forced_labels = [[] for _ in range(batch_size)]  # List of lists to track which label forced each token
    potential_forced_labels = [[] for _ in range(batch_size)]  # List of lists to track which label would have forced each token
    seen_end_think = [False] * batch_size  # Track if we've seen </think> token for each sequence

    for k in iterator:
        base_input_chunk = base_generated_ids.to(device)

        with torch.no_grad():
            with thinking_model.trace({
                        "input_ids": base_input_chunk, 
                        "attention_mask": (base_input_chunk != base_tokenizer.pad_token_id).long()
            }) as tracer:
                thinking_outputs = thinking_model.lm_head.output.save()
                
                # Get hidden states if using probe or norm_diff baseline
                if baseline_method == "probe":
                    probe_layer = baseline_config["probe_layer"]
                    hidden_states = thinking_model.model.layers[probe_layer].output[0][:, -1, :].save()
                elif baseline_method == "norm_diff":
                    target_layer = baseline_config["target_layer"]
                    thinking_hidden_states = thinking_model.model.layers[target_layer].output[0][:, -1, :].save()

        # Now run base model
        with torch.no_grad():
            with base_model.trace({
                        "input_ids": base_input_chunk, 
                        "attention_mask": (base_input_chunk != base_tokenizer.pad_token_id).long()
            }) as tracer:
                base_outputs = base_model.lm_head.output.save()
                
                # Get base model hidden states if using norm_diff
                if baseline_method == "norm_diff":
                    target_layer = baseline_config["target_layer"]
                    base_hidden_states = base_model.model.layers[target_layer].output[0][:, -1, :].save()

        # Get baseline predictions for each sequence in batch
        if baseline_method == "probe":
            # Get probe predictions
            hidden_states = hidden_states.to(torch.float32).to(device)
            probe = baseline_config["probe"].to(device)
            label_to_idx = baseline_config["label_to_idx"]
            force_categories = baseline_config.get("forcing", None)
            logits = probe(hidden_states)
            probs = torch.sigmoid(logits)
            max_probs, max_indices = torch.max(probs, dim=-1)
            max_labels = [label for idx in max_indices for label, label_idx in label_to_idx.items() if label_idx == idx]
            should_force = max_probs > baseline_config.get("threshold", 0.5)
            should_force = [0 if label not in force_categories else should_force[i] for i, label in enumerate(max_labels)]
            forced_labels_batch = [label if force else None for label, force in zip(max_labels, should_force)]
        elif baseline_method == "random":
            # Get random baseline predictions
            forced_token_rate = baseline_config.get("forced_token_rate", 0.5)
            should_force = [random.random() < forced_token_rate for _ in range(batch_size)]
            forced_labels_batch = ["random" if force else None for force in should_force]
        elif baseline_method == "norm_diff":
            # Calculate norm difference
            threshold = baseline_config.get("threshold", 0.1)
            
            # Calculate norms for both models' hidden states
            base_norms = torch.norm(base_hidden_states, dim=-1)
            thinking_norms = torch.norm(thinking_hidden_states, dim=-1)
            
            # Calculate relative difference (abs(base_norm - thinking_norm) / base_norm)
            norm_diff = torch.abs(base_norms - thinking_norms) / base_norms
            should_force = norm_diff > threshold
            forced_labels_batch = ["norm_diff" if force else None for force in should_force]
        elif baseline_method == "kl_div":
            # Calculate KL divergence
            threshold = baseline_config.get("threshold", 1.0)
            
            # Get softmax distributions
            base_probs = torch.nn.functional.softmax(base_outputs[:, -1, :], dim=-1)
            thinking_probs = torch.nn.functional.softmax(thinking_outputs[:, -1, :], dim=-1)
            
            # Calculate KL divergence: KL(thinking || base)
            kl_divs = torch.nn.functional.kl_div(
                thinking_probs.log(), 
                base_probs, 
                reduction='none'
            ).sum(dim=-1)
            
            should_force = kl_divs > threshold
            forced_labels_batch = ["kl_div" if force else None for force in should_force]
        
        # Get next tokens from both models for each sequence in batch
        base_next_tokens = base_outputs[:, -1, :].argmax(dim=-1)
        thinking_next_tokens = thinking_outputs[:, -1, :].argmax(dim=-1)

        # Process each sequence in the batch
        next_tokens = []
        for i in range(batch_size):
            # During warmup, use thinking model's predictions
            if k < warmup:
                next_token = thinking_next_tokens[i]
                thinking_model_tokens[i] += 1
                forced_states[i].append(2)  # Successfully forced (warmup)
                forced_labels[i].append("warmup")
                potential_forced_labels[i].append("warmup")
            else:
                # Check if we've seen </think> token
                current_text = base_tokenizer.decode(base_generated_ids[i], skip_special_tokens=True)
                if "</think>" in current_text:
                    seen_end_think[i] = True
                
                # Track potential forced label regardless of whether we force it
                if not seen_end_think[i] and forced_labels_batch[i] is not None:
                    potential_forced_labels[i].append(forced_labels_batch[i])
                else:
                    potential_forced_labels[i].append(None)
                
                # Determine forced state and next token
                if not seen_end_think[i] and forced_labels_batch[i] is not None:
                    if thinking_next_tokens[i] != base_next_tokens[i]:
                        # Criterion met and models disagree - successful force
                        next_token = thinking_next_tokens[i]
                        thinking_model_tokens[i] += 1
                        forced_states[i].append(2)  # 2 = successful force
                        forced_labels[i].append(forced_labels_batch[i])
                        # Track forced token
                        token_text = base_tokenizer.decode(next_token)
                        forced_tokens[i][token_text] = forced_tokens[i].get(token_text, 0) + 1
                    else:
                        # Criterion met but models agree - attempted force
                        next_token = base_next_tokens[i]
                        base_model_tokens[i] += 1
                        forced_states[i].append(1)  # 1 = attempted force
                        forced_labels[i].append(forced_labels_batch[i])
                else:
                    # No forcing criterion - use base model
                    next_token = base_next_tokens[i]
                    base_model_tokens[i] += 1
                    forced_states[i].append(0)  # 0 = not forced
                    forced_labels[i].append(None)
            
            next_tokens.append(next_token)

        # Stack next tokens and append to sequences
        next_tokens = torch.stack(next_tokens)
        base_generated_ids = torch.cat([base_generated_ids, next_tokens.unsqueeze(1).cpu()], dim=1)
        
        # Check for end of sequence for each sequence in batch
        if all(next_tokens == base_tokenizer.eos_token_id):
            break

        del trace, thinking_outputs, base_outputs, base_next_tokens, thinking_next_tokens, base_input_chunk
        if baseline_method == "norm_diff":
            del base_hidden_states, thinking_hidden_states
        elif baseline_method == "probe":
            del hidden_states
       
        torch.cuda.empty_cache()
        if k % 50 == 0:
            gc.collect()
    
    gc.collect()
    
    if color_output:
        # Print model usage statistics
        total_tokens = [base + think for base, think in zip(base_model_tokens, thinking_model_tokens)]
        print(f"\nModel Usage Statistics:")
        for i in range(batch_size):
            print(f"\nSequence {i+1}:")
            
            # Calculate forcing statistics as percentage of total tokens
            total_attempted = sum(1 for state in forced_states[i] if state == 1 or state == 2)
            successful_forced = sum(1 for state in forced_states[i] if state == 2)
            
            total_rate = total_attempted / total_tokens[i]
            successful_rate = successful_forced / total_tokens[i]
            
            print(f"Total attempted: {total_attempted} ({total_rate*100:.1f}%)")
            print(f"Successful forced: {successful_forced} ({successful_rate*100:.1f}%)")
        
        # Print top forced tokens for each sequence
        print("\nTop 10 Most Frequent Forced Tokens:")
        for i in range(batch_size):
            print(f"\nSequence {i+1}:")
            sorted_tokens = sorted(forced_tokens[i].items(), key=lambda x: x[1], reverse=True)
            for token, freq in sorted_tokens[:10]:
                print(f"'{token}': {freq} times")
        
        # Print colored output for each sequence
        print("\nColored Output (Colors indicate which label forced the token):")
        for i in range(batch_size):
            print(f"\nSequence {i+1}:")
            base_text = base_tokenizer.decode(base_generated_ids[i], skip_special_tokens=True)
            
            # Split into tokens and color them, skipping input tokens
            base_tokens = base_tokenizer.encode(base_text)
            input_length = len(base_tokenizer.encode(base_tokenizer.decode(input_ids[i], skip_special_tokens=True)))
            colored_base = []
            
            for j, token in enumerate(base_tokens):
                if j < input_length:
                    colored_base.append(base_tokenizer.decode(token))
                else:
                    token_idx = j - input_length
                    token_text = base_tokenizer.decode(token)
                    if token_idx < len(forced_states[i]):
                        state = forced_states[i][token_idx]
                        if state == 2:  # Successfully forced
                            # Token was actually forced - color and underline
                            label = forced_labels[i][token_idx]
                            if label == "warmup":
                                colored_base.append(f"\033[90m\033[4m{token_text}\033[0m")  # Gray for warmup with underline
                            else:
                                colored_base.append(f"{label_colors[label]}\033[4m{token_text}\033[0m")  # Colored with underline
                        elif state == 1:  # Attempted force
                            # Token wasn't forced but would have been - just color
                            label = forced_labels[i][token_idx]
                            colored_base.append(f"{label_colors[label]}{token_text}\033[0m")
                        else:  # Not forced
                            colored_base.append(token_text)
                    else:
                        colored_base.append(token_text)
            
            print("Base (with forced tokens colored and underlined by label):")
            print(base_tokenizer.convert_tokens_to_string(colored_base))
        
        # Print color legend
        print("\nColor Legend:")
        for label, color in label_colors.items():
            print(f"{color}{label}\033[0m")
        print("\nUnderlined tokens were successfully forced (models disagreed)")
        print("Colored but not underlined tokens were attempted forces (models agreed)")
    
    return base_generated_ids.cpu(), forced_states, forced_labels, forced_tokens


# Create NumpyEncoder for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Function to convert numpy types to Python native types
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def process_batch_annotations(thinking_processes):
    """Annotate a batch of reasoning chains using the 7-category reasoning framework."""
    annotated_responses = []
    for thinking in thinking_processes:
        annotated_response = chat(f"""
Please annotate the following reasoning trace by marking segments with categories from the reasoning framework below. Use this format: ["<category>"] ... ["<end-section>"]. A sentence can be split into multiple segments if it exhibits different behaviors. Only use the categories provided below.

Reasoning Framework Categories:

1. problem-identification-framing – Problem Identification and Framing
   *Description:* This reflects the model's initial orientation toward the problem—an explicit commitment to focus attention on a particular question or task. It's not solving yet; it's mentally staking out the terrain and clarifying the goal.
   *Includes:* Explicit declarations of the question or topic to be addressed; clarifying scope or rephrasing the goal of the reasoning.
   *Excludes:* Any move toward analysis, solution generation, or speculation.
   *Examples:* "Okay, so I'm trying to figure out how pressure affects the boiling point of water.", "Okay, so I'm trying to figure out the ripple effects of making college education free."

2. metacognitive-setup – Metacognitive Setup and Decomposition Initiation
   *Description:* This captures the model's pre-analytic cognitive preparation—noticing uncertainty or complexity and deciding to plan, organize, or scaffold the reasoning process before diving in.
   *Includes:* Metacognitive statements about strategy or planning; moves to mentally break a problem into manageable parts.
   *Excludes:* Execution of any actual reasoning steps or guesses.
   *Examples:* "Hmm, let me think about this step by step.", "Let me try to visualize this.", "I'm not entirely sure where to start, but I think it's important to break it down step by step."

3. stepwise-calculation – Stepwise Calculation / Enumeration / Local Inference
   *Description:* This cluster captures the model's mechanistic reasoning—applying rules, performing arithmetic, listing possibilities. It's executing a mental algorithm.
   *Includes:* Arithmetic, combinatorics, enumeration of cases; explicit inferences from rules or facts.
   *Excludes:* High-level summaries or contextual reasoning.
   *Examples:* "3 times 7 is 21, and 21 times 11 is 231.", "So, the probability of drawing a red on the first draw is 4 out of 7, which is 4/7.", "Each face is a base for one pyramid, so 6 pyramids."

4. generating-alternatives – Generating Alternatives / Hypotheses
   *Description:* This cluster reflects the model's attempt to expand the hypothesis space. It's not committing to an answer—it's surfacing possible explanations, mechanisms, or paths forward.
   *Includes:* Generative thinking under uncertainty; multiple speculative branches or mechanisms.
   *Excludes:* Final answers or rule-based deductions.
   *Examples:* "Or maybe it's about controlling invasive species.", "Maybe it's just the body's way of fighting off the infection.", "I should also consider different scenarios."

5. information-seeking – Information-Seeking and Epistemic Uncertainty
   *Description:* The model confronts a knowledge gap and initiates action to resolve it. This is a pivot away from internal reasoning toward acquiring more information.
   *Includes:* Statements of uncertainty paired with information-seeking intent; declarations that external info is needed.
   *Excludes:* Internal speculation without intent to learn more; passive confusion without action.
   *Examples:* "I should probably look up some information to get a better understanding.", "Maybe I should ask someone or look it up to find out more information.", "I think I'll just have to check online or maybe ask a friend."

6. consequence-projection – Consequence Projection / Scenario Elaboration
   *Description:* This is forward simulation. The model is running a mental model of the world to ask: "What would happen if...?"
   *Includes:* Counterfactuals, conditionals, and policy simulation; exploration of second- or third-order effects.
   *Excludes:* Simple cause-effect or binary conclusions.
   *Examples:* "Also, with more free time, people might pursue further education.", "If the species affects farming, there might be compensation programs.", "Cities might save money on road repairs due to AVs."

7. conclusion-articulation – (Sub)-Conclusion Articulation
   *Description:* This is the "wrap up this step" reflex. It's when the model finishes part of the reasoning and states a result—before continuing onward.
   *Includes:* (Partial) conclusions or intermediate inferences; logic checkpoints or sanity checks.
   *Excludes:* Problem framing or speculative reasoning.
   *Examples:* "So, each face is a base for one pyramid, so 6 pyramids.", "So, the next month is December, which is D.", "So, if the surgeon is the mother, then yes, the patient is her son."

Reasoning trace to annotate:
{thinking}

Only return the annotated text using the specified format. Do not include any explanation or commentary outside the annotations.
If the last sentence is not finished, do not include it in the annotations.
""")
        annotated_responses.append(annotated_response)
    
    return annotated_responses


class SAE(nn.Module):
    def __init__(self, d_in, num_latents, k=1):
        super().__init__()
        self.encoder = nn.Linear(d_in, num_latents, bias=True)
        self.encoder.bias.data.zero_()
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.k = k
        self.set_decoder_norm_to_unit_norm()
        
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + 1e-5
        
    def encode(self, x):
        forward = self.encoder(x - self.b_dec)
        top_acts, top_indices = forward.topk(self.k, dim=-1)
        return top_acts, top_indices
        
    def decode(self, top_acts, top_indices):
        batch_size = top_indices.shape[0]
        
        # Reshape for embedding_bag
        top_acts_flat = top_acts.view(-1)
        top_indices_flat = top_indices.view(-1)
        
        # For embedding_bag we need offsets that point to the start of each sample
        offsets = torch.arange(0, batch_size, device=top_indices.device) * self.k
        
        # Use embedding_bag
        res = nn.functional.embedding_bag(
            top_indices_flat, self.W_dec, offsets=offsets, 
            per_sample_weights=top_acts_flat, mode="sum"
        )
        
        return res + self.b_dec
        
    def forward(self, x):
        top_acts, top_indices = self.encode(x)
        return self.decode(top_acts, top_indices)

model_mapping = {
    "meta-llama/Llama-3.1-8B":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen/Qwen2.5-Math-1.5B":"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/Qwen2.5-14B":"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}

#  problem-framing
#  analytical-decomposition
#  structural-decomposition
#  possibility-checking
#  calculation-computation
#  hypothesis-generation
#  generating-additional-considerations
#  logical-structure-testing

steering_config = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "backtracking": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "backtracking": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "backtracking": {"vector_layer": 44, "pos_layers": [44], "neg_layers": [44], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 44, "pos_layers": [44], "neg_layers": [44], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 44, "pos_layers": [44], "neg_layers": [44], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 44, "pos_layers": [44], "neg_layers": [44], "pos_coefficient": 1, "neg_coefficient": 1},
    }
}
