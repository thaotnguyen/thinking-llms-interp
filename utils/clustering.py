"""
Clustering utilities for neural activation analysis.
This module contains common functions for clustering analysis, model saving/loading,
and evaluation metrics.
"""

import os
import json
import re
import numpy as np
import pickle
import time
import random
from tqdm import tqdm
from utils.utils import print_and_flush, chat, chat_batch, convert_numpy_types


def load_trained_clustering_data(model_id, layer, n_clusters, method):
    """
    Load trained clustering data for a specific model, layer, and clustering method.
    
    Parameters:
    -----------
    model_id : str
        Model identifier
    layer : int
        Layer number
    n_clusters : int
        Number of clusters
    method : str
        Clustering method name
        
    Returns:
    --------
    dict
        Dictionary containing the clustering data
    """
    if method == 'sae_topk':
        # SAE uses a different path and torch.load
        sae_path = f'results/vars/saes/sae_{model_id}_layer{layer}_clusters{n_clusters}.pt'
        
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE clustering model not found at {sae_path}")
        
        # Load the SAE model
        import torch
        sae_data = torch.load(sae_path, map_location='cpu')
        
        # Return the data with method information
        sae_data['method'] = method
        return sae_data
    
    else:
        # All other methods use pickle files
        clustering_path = f'results/vars/{method}/{model_id}_layer{layer}_clusters{n_clusters}.pkl'
        
        if not os.path.exists(clustering_path):
            raise FileNotFoundError(f"Clustering model not found at {clustering_path}")
        
        # Load the clustering model
        with open(clustering_path, 'rb') as f:
            clustering_data = pickle.load(f)
        
        # Add method information
        clustering_data['method'] = method
        return clustering_data


def predict_clusters(activations, clustering_data):
    """
    Predict cluster labels for new activations using loaded clustering data.
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Normalized activation vectors to predict clusters for
    clustering_data : dict
        Dictionary containing the clustering data loaded from file
        
    Returns:
    --------
    numpy.ndarray
        cluster_labels array of cluster assignments
    """
    method = clustering_data['method']
    
    if method == 'agglomerative':
        # For agglomerative clustering, assign to nearest cluster center using cosine similarity
        cluster_centers = clustering_data['cluster_centers']
        
        # Normalize both activations and cluster centers
        activations_norm = activations / np.linalg.norm(activations, axis=1, keepdims=True)
        centers_norm = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        
        # Compute cosine similarity and assign to nearest center
        similarities = np.dot(activations_norm, centers_norm.T)
        cluster_labels = np.argmax(similarities, axis=1)
        
        return cluster_labels
    
    elif method == 'spherical_kmeans':
        # Use KMeans predict method on normalized data
        model = clustering_data['model']
        
        # Normalize activations for spherical kmeans
        activations_norm = activations / np.linalg.norm(activations, axis=1, keepdims=True)
        
        # Predict using the KMeans model
        cluster_labels = model.predict(activations_norm)
        
        return cluster_labels
    
    elif method == 'gmm':
        # Use GMM predict method
        model = clustering_data['model']
        
        # Predict using the GMM model
        cluster_labels = model.predict(activations)
        
        return cluster_labels
    
    elif method == 'pca_kmeans':
        # Apply PCA transform first, then use KMeans predict
        pca = clustering_data['pca']
        kmeans = clustering_data['kmeans']
        
        # Apply PCA transformation
        reduced_data = pca.transform(activations)
        
        # Predict using KMeans
        cluster_labels = kmeans.predict(reduced_data)
        
        return cluster_labels
    
    elif method == 'pca_gmm':
        # Apply PCA transform first, then use GMM predict
        pca = clustering_data['pca']
        gmm = clustering_data['gmm']
        
        # Apply PCA transformation
        reduced_data = pca.transform(activations)
        
        # Predict using GMM
        cluster_labels = gmm.predict(reduced_data)
        
        return cluster_labels
    
    elif method == 'pca_agglomerative':
        # Apply PCA transform first, then assign to nearest cluster center
        pca = clustering_data['pca']
        cluster_centers = clustering_data['cluster_centers']
        
        # Apply PCA transformation
        reduced_data = pca.transform(activations)
        
        # Normalize reduced data for cosine similarity
        reduced_norm = reduced_data / np.linalg.norm(reduced_data, axis=1, keepdims=True)
        
        # Get reduced cluster centers by applying PCA to original centers
        reduced_centers = pca.transform(cluster_centers)
        centers_norm = reduced_centers / np.linalg.norm(reduced_centers, axis=1, keepdims=True)
        
        # Compute cosine similarity and assign to nearest center
        similarities = np.dot(reduced_norm, centers_norm.T)
        cluster_labels = np.argmax(similarities, axis=1)
        
        return cluster_labels
    
    elif method == 'sae_topk':
        # Use the encoder to get activations, then take argmax
        import torch
        
        # Extract SAE parameters
        encoder_weight = clustering_data['encoder_weight']
        encoder_bias = clustering_data['encoder_bias']
        b_dec = clustering_data['b_dec']
        
        # Convert to torch tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        activations_tensor = torch.from_numpy(activations).float().to(device)
        encoder_weight = encoder_weight.to(device)
        encoder_bias = encoder_bias.to(device)
        b_dec = b_dec.to(device)
        
        # Apply encoder: activations = W * (x - b_dec) + b_enc
        with torch.no_grad():
            # Center the activations
            centered_activations = activations_tensor - b_dec
            
            # Apply encoder transformation
            encoded_activations = torch.matmul(centered_activations, encoder_weight.T) + encoder_bias
            
            # Get the cluster assignment as the argmax of encoder activations
            cluster_labels = encoded_activations.argmax(dim=1).cpu().numpy()
        
        return cluster_labels
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")


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

def generate_cluster_descriptions(cluster_examples_list, model="gpt-4.1", n_trace_examples=0, model_name=None):
    """
    Generate descriptions for multiple clusters in batch.
    
    Args:
        cluster_examples_list (list): List of tuples (cluster_idx, examples) for each cluster
        model (str): Model to use for generating descriptions
        n_trace_examples (int): Number of full reasoning trace examples to include in prompts
        model_name (str): Name of the model whose responses should be loaded for trace examples
        
    Returns:
        list: List of tuples (cluster_idx, title, description) for each cluster
    """
    # Prepare trace examples if requested (shared across all clusters)
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
    
    # Create prompts for all clusters
    batch_prompts = []
    cluster_indices = []
    
    for cluster_idx, examples in cluster_examples_list:
        # Create a prompt for this cluster
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
        
        batch_prompts.append(prompt)
        cluster_indices.append(cluster_idx)
    
    # Process all prompts in batch
    print(f"Processing {len(batch_prompts)} cluster description prompts in batch...")
    # Run the async batch processing
    import asyncio
    
    # Handle both Jupyter (with running event loop) and regular scripts
    try:
        # Check if there's already a running event loop
        loop = asyncio.get_running_loop()
        # If we get here, we're in an environment with a running loop (like Jupyter)
        # We need to create a task and run it
        import concurrent.futures
        import threading
        
        def run_in_thread():
            return asyncio.run(chat_batch(batch_prompts, model=model))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            responses = future.result()
    except RuntimeError:
        # No running event loop, safe to use asyncio.run()
        responses = asyncio.run(chat_batch(batch_prompts, model=model))
    
    # Parse responses to extract titles and descriptions
    results = []
    for i, response in enumerate(responses):
        cluster_idx = cluster_indices[i]
        
        # Parse the response to extract title and description
        title = "Unnamed Cluster"
        description = "No description available"
        
        title_match = re.search(r"Title:\s*(.*?)(?:\n|$)", response)
        if title_match:
            title = title_match.group(1).strip()
            
        desc_match = re.search(r"Description:\s*(.*?)(?:\n|$)", response)
        if desc_match:
            description = desc_match.group(1).strip()
        
        results.append((str(cluster_idx), title, description))
    
    return results

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
        
        # Sanitize the JSON string to escape invalid backslash sequences.
        json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
        
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

def accuracy_autograder(sentences, categories, ground_truth_labels, model="gpt-4.1", n_autograder_examples=100):
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
    
    # Collect all prompts and metadata for batch processing
    batch_prompts = []
    batch_metadata = []
    
    # For each category, prepare data and prompts
    for cluster_id, title, description in tqdm(categories, desc="Preparing batch prompts"):
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
      "explanation": "Brief explanation of your reasoning",
      "belongs_to_category": "Yes" or "No"
    }},
    ... (repeat for all sentences)
  ]
}}
```

Only include the JSON object in your response, with no additional text before or after.
"""
        
        # Add prompt and metadata to batch
        batch_prompts.append(prompt)
        batch_metadata.append({
            "cluster_id": cluster_id,
            "cluster_id_str": cluster_id_str,
            "title": title,
            "description": description,
            "test_sentences": test_sentences,
            "test_ground_truth": test_ground_truth,
            "test_indices": test_indices  # Store original indices for reference
        })
    
    # Process all prompts in batch
    print(f"Processing {len(batch_prompts)} prompts in batch for evaluating accuracy...")
    # Run the async batch processing
    import asyncio
    
    # Handle both Jupyter (with running event loop) and regular scripts
    try:
        # Check if there's already a running event loop
        loop = asyncio.get_running_loop()
        # If we get here, we're in an environment with a running loop (like Jupyter)
        # We need to create a task and run it
        import concurrent.futures
        import threading
        
        def run_in_thread():
            return asyncio.run(chat_batch(batch_prompts, model=model))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            responses = future.result()
    except RuntimeError:
        # No running event loop, safe to use asyncio.run()
        responses = asyncio.run(chat_batch(batch_prompts, model=model))
    
    # Process all responses
    for i, response in enumerate(responses):
        metadata = batch_metadata[i]
        cluster_id = metadata["cluster_id"]
        cluster_id_str = metadata["cluster_id_str"]
        test_sentences = metadata["test_sentences"]
        test_ground_truth = metadata["test_ground_truth"]
        
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
            
            # Sanitize the JSON string to escape invalid backslash sequences.
            json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
            
            result = json.loads(json_str)
            
            # Compute metrics for this cluster
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            predictions = []
            
            # Add sentence texts to classifications for detailed analysis
            enhanced_classifications = []
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
                
                # Create enhanced classification with sentence text and ground truth
                enhanced_item = item.copy()
                enhanced_item["sentence_text"] = test_sentences[sentence_idx]
                enhanced_item["ground_truth"] = true_label
                enhanced_classifications.append(enhanced_item)
            
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
                "classifications": enhanced_classifications  # Use enhanced classifications with sentence texts
            }
        
        except Exception as e:
            print(f"Error in accuracy autograder for cluster {cluster_id}: {e}")
            import traceback
            traceback.print_exc()
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


def compute_centroid_orthogonality(cluster_centers):
    """
    Compute the orthogonality of cluster centroids using 1 - cosine similarity.
    Uses pairwise_distances from sklearn to explicitly compute all pairwise similarities.
    
    Parameters:
    -----------
    cluster_centers : numpy.ndarray
        Cluster center vectors
        
    Returns:
    --------
    float
        Average orthogonality (1 - cosine similarity) between centroids
    """
    start_time = time.time()
    # First compute cosine similarity (not distance)
    norm_cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    # Use dot product for cosine similarity
    cosine_sim = np.dot(norm_cluster_centers, norm_cluster_centers.T)
    # Take absolute value to treat opposite directions as similar
    abs_cosine_sim = np.abs(cosine_sim)
    # Calculate orthogonality as 1 - absolute similarity
    orthogonality = 1 - abs_cosine_sim
    
    # Get the indices of the upper triangular part (excluding diagonal)
    # This ensures we only count each pair once and exclude self-similarities
    indices = np.triu_indices(orthogonality.shape[0], k=1)
    
    # Extract the upper triangular values
    upper_tri_values = orthogonality[indices]
    
    # Calculate average orthogonality
    avg_orthogonality = np.mean(upper_tri_values) if len(upper_tri_values) > 0 else 0.0
    
    print_and_flush(f"Computed centroid orthogonality in {time.time() - start_time} seconds")
    return avg_orthogonality


def generate_representative_examples(cluster_centers, texts, cluster_labels, example_activations):
    """
    Generate representative examples for each cluster based on distance to centroid.
    
    Parameters:
    -----------
    cluster_centers : numpy.ndarray
        Cluster centers
    texts : list
        List of texts
    cluster_labels : numpy.ndarray
        Cluster labels for each text
    example_activations : numpy.ndarray
        Normalized activation vectors
        
    Returns:
    --------
    dict
        Dictionary mapping cluster_idx to list of representative examples
    """
    start_time = time.time()
    representative_examples = {}
    
    for cluster_idx in tqdm(range(len(cluster_centers)), desc="Generating representative examples"):
        # Get indices of texts in this cluster
        cluster_indices = np.where(cluster_labels == cluster_idx)[0]
        
        # Skip empty clusters
        if len(cluster_indices) == 0:
            representative_examples[cluster_idx] = []
            print_and_flush(f"WARNING:Skipping empty cluster {cluster_idx} in generate_representative_examples")
            continue
            
        # Get all examples in this cluster
        cluster_texts = [texts[i] for i in cluster_indices]
        
        # Calculate distances to centroid
        cluster_vectors = np.stack([example_activations[i] for i in cluster_indices])
        centroid = cluster_centers[cluster_idx]
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        
        # Sort examples by distance to centroid
        sorted_indices = np.argsort(distances)
        sorted_examples = [cluster_texts[i] for i in sorted_indices]
        
        representative_examples[cluster_idx] = sorted_examples
    
    print_and_flush(f"Generated representative examples in {time.time() - start_time} seconds")
    return representative_examples


def generate_category_descriptions(cluster_centers, model_name, n_description_examples, representative_examples):
    """
    Generate descriptions for each cluster based on most representative sentences.
    Uses half top examples and half random examples from the cluster.
    
    Parameters:
    -----------
    cluster_centers : numpy.ndarray
        Cluster centers
    texts : list
        List of texts
    cluster_labels : numpy.ndarray
        Cluster labels for each text
    example_activations : numpy.ndarray
        Normalized activation vectors
    model_name : str
        Name of the model to use for generating descriptions
    n_description_examples : int
        Number of examples to use for generating descriptions
        
    Returns:
    --------
    list
        List of tuples (cluster_id, category_title, category_description)
    """
    start_time = time.time()    
    # Prepare batch data for all non-empty clusters
    cluster_examples_list = []
    for cluster_idx in range(len(cluster_centers)):
        # Skip empty clusters
        if len(representative_examples[cluster_idx]) == 0:
            print_and_flush(f"WARNING:Skipping empty cluster {cluster_idx} in generate_category_descriptions")
            continue
        
        # Get top examples
        examples = representative_examples[cluster_idx][:n_description_examples]
        cluster_examples_list.append((cluster_idx, examples))
    
    # Generate descriptions in batch
    categories = generate_cluster_descriptions(
        cluster_examples_list, 
        model_name=model_name, 
        n_trace_examples=3
    )
    
    print_and_flush(f"Generated category descriptions in {time.time() - start_time} seconds")
    return categories


def evaluate_clustering_accuracy(texts, cluster_labels, categories, n_autograder_examples):
    """
    Evaluate clustering using the binary accuracy autograder.
    Tests each cluster independently against examples from other clusters.
    
    Parameters:
    -----------
    texts : list
        List of texts
    cluster_labels : numpy.ndarray
        Cluster labels for each text
    categories : list
        List of tuples (cluster_id, title, description)
    n_autograder_examples : int
        Number of examples from each cluster to use for autograding
        
    Returns:
    --------
    dict
        Autograder results including precision, recall, accuracy and F1 for each cluster
    """
    start_time = time.time()
    # Convert cluster_labels to list of strings for compatibility
    str_cluster_labels = [str(label) for label in cluster_labels]
    
    # Run binary autograder
    for _ in range(3):
        try:
            results = accuracy_autograder(texts, categories, str_cluster_labels, 
                                               n_autograder_examples=n_autograder_examples)
            break
        except Exception as e:
            print_and_flush(f"Error running accuracy autograder: {e}")
            time.sleep(5)
    
    print_and_flush(results["avg"])
    print_and_flush(f"Evaluated clustering accuracy in {time.time() - start_time} seconds")
    return results

def evaluate_clustering_completeness(texts, categories, n_test_examples=50):
    """
    Evaluate clustering using the completeness autograder with a random sample of texts.
    
    Parameters:
    -----------
    texts : list
        List of texts
    categories : list
        List of tuples (cluster_id, title, description)
    n_test_examples : int
        Number of examples to use for testing completeness
        
    Returns:
    --------
    dict
        Autograder results
    """
    start_time = time.time()
    # Sample n_test_examples randomly from all texts
    if len(texts) > n_test_examples:
        test_texts = random.sample(texts, n_test_examples)
    else:
        test_texts = texts
    
    # Run autograder on the sampled texts
    for _ in range(3):
        try:
            results = completeness_autograder(test_texts, categories)
            break
        except Exception as e:
            print_and_flush(f"Error running completeness autograder: {e}")
            time.sleep(5)
    
    print_and_flush(f"Evaluated clustering completeness in {time.time() - start_time} seconds")
    return results


def evaluate_clustering_scoring_metrics(texts, cluster_labels, n_clusters, example_activations, cluster_centers, 
                       model_name, n_autograder_examples, n_description_examples):
    """
    Evaluate clustering using both accuracy and optionally completeness autograders.
    
    Parameters:
    -----------
    texts : list
        List of texts
    cluster_labels : numpy.ndarray
        Cluster labels for each text
    n_clusters : int
        Number of clusters
    example_activations : numpy.ndarray
        Normalized activation vectors
    cluster_centers : numpy.ndarray
        Cluster centers
    model_name : str
        Name of the model to use for generating descriptions
    n_autograder_examples : int
        Number of examples from each cluster to use for autograding
    n_description_examples : int
        Number of examples to use for generating descriptions
        
    Returns:
    --------
    dict
        Combined evaluation results
    """
    # Generate representative examples
    representative_examples = generate_representative_examples(
        cluster_centers, texts, cluster_labels, example_activations
    )
    
    # Generate category descriptions
    categories = generate_category_descriptions(
        cluster_centers, model_name, n_description_examples, representative_examples
    )
    
    # Run binary accuracy autograder (evaluates each cluster independently)
    accuracy_results = evaluate_clustering_accuracy(
        texts, cluster_labels, categories, n_autograder_examples
    )
    
    # Compute centroid orthogonality
    orthogonality = compute_centroid_orthogonality(cluster_centers)
    
    # Get average accuracy from accuracy_results["avg"]
    avg_accuracy = accuracy_results.get("avg", {}).get("accuracy", 0)
    
    results = {
        "accuracy": avg_accuracy,
        "categories": categories,
        "orthogonality": orthogonality  # Add orthogonality to results
    }
    
    # Optionally run completeness autograder
    completeness_results = evaluate_clustering_completeness(texts, categories)
    results["assigned_fraction"] = completeness_results["assigned_fraction"]
    results["category_counts"] = completeness_results["category_counts"]

    # Create detailed results by cluster
    detailed_results = {}
    for cluster_id, title, description in tqdm(categories, desc="Creating detailed results"):
        cluster_id_str = str(cluster_id)
        cluster_metrics = accuracy_results.get(cluster_id_str, {})
        cluster_idx = int(cluster_id)
        cluster_examples = representative_examples[cluster_idx]
        
        detailed_results[cluster_id_str] = {
            'title': title,
            'description': description,
            'size': len(np.where(cluster_labels == cluster_idx)[0]),
            'precision': cluster_metrics.get('precision', 0),
            'recall': cluster_metrics.get('recall', 0),
            'accuracy': cluster_metrics.get('accuracy', 0),
            'f1': cluster_metrics.get('f1', 0),
            'examples': cluster_examples[:15]  # Top 15 examples
        }
    
    results['detailed_results'] = detailed_results
    
    return results


def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Parameters:
    -----------
    obj : any
        Object to convert
        
    Returns:
    --------
    any
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


# Dictionary mapping clustering method names to their implementations
CLUSTERING_METHODS = {
    'agglomerative': 'clustering_agglomerative',
    'pca_agglomerative': 'clustering_pca_agglomerative',
    'gmm': 'clustering_gmm',
    'pca_gmm': 'clustering_pca_gmm',
    'spherical_kmeans': 'clustering_spherical_kmeans',
    'pca_kmeans': 'clustering_pca_kmeans',
    'sae_topk': 'clustering_sae_topk'
}

# Set of supported clustering methods
SUPPORTED_CLUSTERING_METHODS = {
    'agglomerative',
    'pca_agglomerative',
    'gmm',
    'pca_gmm',
    'spherical_kmeans',
    'pca_kmeans',
    'sae_topk'
} 