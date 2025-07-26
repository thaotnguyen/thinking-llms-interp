"""
Batch API versions of clustering evaluation methods using OpenAI's batch API.
This module contains methods for generating cluster descriptions and evaluating
clustering accuracy, completeness, and semantic orthogonality using batched requests.
"""

import os
import json
import re
import numpy as np
import time
import random
import tempfile
from openai import OpenAI
from utils.utils import print_and_flush
from utils.clustering import (
    categories_examples, parse_json_response, simplify_category_name,
)


def submit_openai_batch(prompts_with_ids, batch_description="Clustering evaluation batch", model="gpt-4.1", temperature=1e-19, max_tokens=28000):
    """
    Submit a batch of prompts to OpenAI's batch API.
    
    Args:
        prompts_with_ids (dict): Dictionary mapping custom_id to prompt text
        batch_description (str): Description for the batch
        model (str): OpenAI model to use
        temperature (float): Temperature parameter
        max_tokens (int): Maximum tokens per response
        
    Returns:
        str: Batch ID for tracking the submitted batch
    """
    client = OpenAI()

    if model.startswith("o3") or model.startswith("o4"):
        temperature = 1.0
    
    # Create batch requests
    batch_requests = []
    for custom_id, prompt in prompts_with_ids.items():
        batch_requests.append({
            "custom_id": custom_id,
            "method": "POST", 
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        })
    
    # Create temporary JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for request in batch_requests:
            json.dump(request, f)
            f.write('\n')
        batch_input_path = f.name
    
    try:
        # Upload batch input file
        with open(batch_input_path, 'rb') as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        
        # Create the batch
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions", 
            completion_window="24h",
            metadata={"description": batch_description}
        )
        
        print(f"Submitted batch {batch.id} with {len(batch_requests)} requests")
        return batch.id
        
    finally:
        # Clean up temporary file
        os.unlink(batch_input_path)


def process_batch_results(batch_id):
    """
    Process results from a completed OpenAI batch.
    
    Args:
        batch_id (str): The batch ID to process
        
    Returns:
        dict: Dictionary mapping custom_id to response content
    """
    client = OpenAI()
    
    # Get batch status
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise ValueError(f"Batch not completed. Current status: {batch.status}")
    
    # Get the output file  
    if not batch.output_file_id:
        raise ValueError("No output file available")
    
    # Download and process results
    file_response = client.files.content(batch.output_file_id)
    results = {}
    
    # Process each line of the output file
    for line in file_response.text.splitlines():
        result = json.loads(line)
        custom_id = result.get("custom_id")
        
        if custom_id:
            # Check for errors
            if result.get("error"):
                print(f"Error in request {custom_id}: {result['error']}")
                results[custom_id] = None
                continue
                
            # Extract response content
            response = result.get("response", {}).get("body", {})
            choices = response.get("choices", [])
            if choices and choices[0].get("message", {}).get("content"):
                content = choices[0]["message"]["content"]
                results[custom_id] = content
            else:
                print(f"Invalid content in request {custom_id}")
                results[custom_id] = None
    
    # Check for errors file
    if batch.error_file_id:
        error_response = client.files.content(batch.error_file_id)
        for line in error_response.text.splitlines():
            error = json.loads(line)
            print(f"Batch error: {error}")
    
    print(f"Processed batch {batch_id} with {len(results)} results")
    return results


def check_batch_status(batch_id):
    """
    Check the status of an OpenAI batch.
    
    Args:
        batch_id (str): The batch ID to check
        
    Returns:
        str: The current status of the batch
    """
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    return batch.status


def generate_cluster_descriptions_batch(model_name, cluster_examples_list, n_trace_examples=0, n_categories_examples=3, model="o4-mini"):
    """
    Generate descriptions for multiple clusters using batch API.
    
    Args:
        model_name (str): Name of the model whose responses should be loaded for trace examples
        cluster_examples_list (list): List of tuples (cluster_idx, examples) for each cluster
        n_trace_examples (int): Number of full reasoning trace examples to include in prompts
        n_categories_examples (int): Number of category examples to include
        model (str): Model to use for generating descriptions
        
    Returns:
        tuple: (batch_id, cluster_indices) - batch_id for tracking, cluster_indices for processing results
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
    
    # Prepare category examples text if requested
    category_examples_text = ""
    if n_categories_examples > 0:
        # Select up to n_categories_examples from the categories_examples list
        selected_examples = categories_examples[:n_categories_examples]
        
        if selected_examples:
            category_examples_text = "Here are some example categories to help guide your analysis:\n\n"
            for i, category in enumerate(selected_examples):
                category_examples_text += f"**Example Category {i+1}: {category['title']}**\n"
                category_examples_text += f"Description: {category['description']}\n"
                category_examples_text += "Examples:\n"
                for example in category['examples']:
                    category_examples_text += f"- {example}\n"
                category_examples_text += "\n"
    
    # Create prompts for all clusters
    prompts_with_ids = {}
    cluster_indices = []
    
    for cluster_idx, examples in cluster_examples_list:
        # Create a prompt for this cluster
        prompt = f"""Analyze the following {len(examples)} sentences from an LLM reasoning trace. These sentences are grouped into a cluster based on their similar role or function in the reasoning process.

Your task is to identify the precise cognitive function these sentences serve in the reasoning process. Consider the reasoning strategy or cognitive operation being performed.
""" + (f"\n{trace_examples_text}" if trace_examples_text else "") + f"""

Sentences:
'''
{chr(10).join([f"- {example}" for example in examples])}
'''

Look for:
- Shared reasoning strategies or cognitive mechanisms
- Common linguistic patterns or structures
- Functional role within the overall reasoning process"""  + (f"\n\n{category_examples_text}" if category_examples_text else "") + """

Your response should be in this exact format:
Title: [concise title naming the specific reasoning function]
Description: [2-3 sentences explaining (1) what is the reasoning process that this cluster is about, (2) what is INCLUDED and NOT INCLUDED in this category]

Avoid overly general descriptions. Be precise enough that someone could reliably identify new examples of this reasoning function.
"""
        
        custom_id = f"cluster_description_{cluster_idx}"
        prompts_with_ids[custom_id] = prompt
        cluster_indices.append(cluster_idx)
    
    # Submit batch
    print(f"Submitting batch for {len(prompts_with_ids)} cluster descriptions...")
    batch_id = submit_openai_batch(
        prompts_with_ids, 
        f"Cluster descriptions for {model_name}",
        model=model
    )
    
    return batch_id, cluster_indices


def process_cluster_descriptions_batch(batch_id, cluster_indices):
    """
    Process results from cluster descriptions batch.
    
    Args:
        batch_id (str): The batch ID to process
        cluster_indices (list): List of cluster indices in the same order as submitted
        
    Returns:
        list: List of tuples (cluster_id, title, description) for each cluster
    """
    # Process batch results
    results = process_batch_results(batch_id)
    
    # Parse responses to extract titles and descriptions
    categories = []
    for i, cluster_idx in enumerate(cluster_indices):
        custom_id = f"cluster_description_{cluster_idx}"
        response = results.get(custom_id)
        
        if response:
            # Parse the response to extract title and description
            title = "Unnamed Cluster"
            description = "No description available"
            
            title_match = re.search(r"Title:\s*(.*?)(?:\n|$)", response)
            if title_match:
                title = title_match.group(1).strip()
                
            desc_match = re.search(r"Description:\s*(.*?)(?:\n|$)", response)
            if desc_match:
                description = desc_match.group(1).strip()
            
            categories.append((str(cluster_idx), title, description))
        else:
            print(f"No response received for cluster {cluster_idx}")
            categories.append((str(cluster_idx), "Unnamed Cluster", "No description available"))
    
    return categories


def completeness_autograder_batch(sentences, categories, ground_truth_labels=None, max_sentences_per_prompt=50, model="gpt-4.1-mini"):
    """
    Autograder that evaluates if sentences belong to any of the provided categories using batch API.
    
    Args:
        sentences (list): List of sentences to evaluate
        categories (list): List of tuples where each tuple is (cluster_id, title, description)
        ground_truth_labels (list, optional): Ground truth cluster labels for each sentence
        max_sentences_per_prompt (int): Maximum number of sentences to process per prompt
        model (str): Model to use for evaluation
    
    Returns:
        tuple: (batch_id, metadata) for processing results later
    """
    # Format the categories into a readable list for the prompt
    categories_text = "\n\n".join([f"Category {cluster_id}: {title}\nDescription: {description}" 
                                  for cluster_id, title, description in categories])
    
    # Split sentences into chunks if necessary
    total_sentences = len(sentences)
    if total_sentences <= max_sentences_per_prompt:
        # Process all sentences in a single prompt
        sentence_chunks = [sentences]
        ground_truth_chunks = [ground_truth_labels] if ground_truth_labels is not None else [None]
        chunk_start_indices = [0]
    else:
        # Split into multiple chunks
        sentence_chunks = []
        ground_truth_chunks = []
        chunk_start_indices = []
        
        for i in range(0, total_sentences, max_sentences_per_prompt):
            end_idx = min(i + max_sentences_per_prompt, total_sentences)
            sentence_chunks.append(sentences[i:end_idx])
            chunk_start_indices.append(i)
            
            if ground_truth_labels is not None:
                ground_truth_chunks.append(ground_truth_labels[i:end_idx])
            else:
                ground_truth_chunks.append(None)
    
    # Create prompts for all chunks
    prompts_with_ids = {}
    metadata = {
        "sentence_chunks": sentence_chunks,
        "ground_truth_chunks": ground_truth_chunks, 
        "chunk_start_indices": chunk_start_indices,
        "categories": categories,
        "total_sentences": total_sentences
    }
    
    for chunk_idx, (sentence_chunk, chunk_start_idx) in enumerate(zip(sentence_chunks, chunk_start_indices)):
        # Format the sentences into a numbered list (using original sentence indices)
        sentences_text = "\n\n".join([f"Sentence {chunk_start_idx + i}: {sentence}" 
                                     for i, sentence in enumerate(sentence_chunk)])

        prompt = f"""# Task: Categorize Sentences of Reasoning Traces

You are a highly selective expert at categorizing reasoning sentences. Your task is to STRICTLY evaluate whether each sentence fits into one of the predefined categories. You should be CONSERVATIVE and PRECISE - only assign a category if there is a clear, unambiguous match.

**CRITICAL INSTRUCTIONS:**
- BE STRICT: Only assign a category if the sentence is a clear, strong example of that category
- PREFER "None": When in doubt, choose "None" rather than forcing an assignment
- AVOID false positives: It is better to miss a borderline case than to incorrectly categorize
- REQUIRE precise match: The sentence must clearly demonstrate the specific reasoning function described
- NO loose interpretations: Don't stretch categories to accommodate sentences that don't clearly fit

## Categories:
{categories_text}

## Sentences to Categorize:
{sentences_text}

## Evaluation Criteria:
1. Does the sentence CLEARLY and UNAMBIGUOUSLY demonstrate the exact reasoning function described?
2. Would this sentence serve as a good TEACHING EXAMPLE of the category?
3. Is there ANY doubt about whether it fits the category description?

If you answer "no" to questions 1-2 or "yes" to question 3, assign "None".

**Remember: False positives (incorrect assignments) are worse than false negatives (missed assignments). When uncertain, choose "None".**

## Confidence Scoring:
For each sentence, you must also provide a confidence score:
- If assigning to a category: Use a score from 1-10 (1 = barely fits, 10 = perfect example)
- If assigning "None": Always use confidence score 0

## Response Format:
Your response must follow this exact JSON format:
```json
{{
  "categorizations": [
    {{
      "sentence_id": <sentence idx>,
      "explanation": "Brief explanation of your reasoning and why you were certain/uncertain",
      "assigned_category": "Category <category idx>" (not the title, just the category index) or "None",
      "confidence": <integer from 0-10>
    }},
    ... (repeat for all sentences)
  ]
}}
```

Only include the JSON object in your response, with no additional text before or after.
"""
        
        custom_id = f"completeness_chunk_{chunk_idx}"
        prompts_with_ids[custom_id] = prompt
    
    # Submit batch
    print(f"Submitting batch for completeness evaluation with {len(prompts_with_ids)} chunks...")
    batch_id = submit_openai_batch(
        prompts_with_ids, 
        "Completeness evaluation",
        model=model,
        max_tokens=4000
    )
    
    return batch_id, metadata


def process_completeness_batch(batch_id, metadata):
    """
    Process results from completeness evaluation batch.
    
    Args:
        batch_id (str): The batch ID to process
        metadata (dict): Metadata from the batch submission
        
    Returns:
        dict: Statistics about category assignments
    """
    # Process batch results
    results = process_batch_results(batch_id)
    
    # Extract metadata
    sentence_chunks = metadata["sentence_chunks"]
    ground_truth_chunks = metadata["ground_truth_chunks"]
    chunk_start_indices = metadata["chunk_start_indices"]
    categories = metadata["categories"]
    total_sentences = metadata["total_sentences"]
    sentences = [sentence for chunk in sentence_chunks for sentence in chunk]
    ground_truth_labels = [label for chunk in ground_truth_chunks for label in chunk] if ground_truth_chunks[0] is not None else None
    
    # Aggregate results from all chunks
    all_categorizations = []
    parsing_errors = []
    
    # Process each response from the batch
    for chunk_idx in range(len(sentence_chunks)):
        custom_id = f"completeness_chunk_{chunk_idx}"
        response = results.get(custom_id)
        
        if response:
            try:
                result = parse_json_response(response, expected_field='categorizations')
                
                # Add the categorizations from this chunk to the overall list
                all_categorizations.extend(result["categorizations"])
                
            except Exception as e:
                print(f"Error parsing response for chunk {chunk_idx}: {e}")
                print(f"Raw response: {response}")
                parsing_errors.append({
                    "chunk_idx": chunk_idx,
                    "error": str(e),
                    "raw_response": response
                })
        else:
            print(f"No response received for chunk {chunk_idx}")
            parsing_errors.append({
                "chunk_idx": chunk_idx,
                "error": "No response received",
                "raw_response": None
            })
    
    # If we had parsing errors, return error information
    if parsing_errors and not all_categorizations:
        return {
            "error": f"Failed to parse {len(parsing_errors)} chunks",
            "total_sentences": total_sentences,
            "assigned": 0,
            "not_assigned": total_sentences,
            "assigned_fraction": 0,
            "not_assigned_fraction": 1.0,
            "avg_confidence": 0.0,
            "parsing_errors": parsing_errors
        }
    
    # Aggregate statistics from all categorizations
    assigned = 0
    not_assigned = 0
    category_counts = {str(cluster_id): 0 for cluster_id, _, _ in categories}
    category_counts["None"] = 0
    
    # Confidence tracking
    total_confidence = 0.0
    category_confidences = {str(cluster_id): [] for cluster_id, _, _ in categories}
    category_confidences["None"] = []
    
    # Enhanced analysis if ground truth labels are provided
    detailed_analysis = {
        "correct_assignments": [],      # Assigned to correct category
        "incorrect_assignments": [],    # Assigned to wrong category  
        "missed_assignments": [],       # Should have been assigned but weren't (assigned "None")
    }
    
    # Create a mapping of category IDs for quick lookup
    valid_category_ids = {str(cluster_id) for cluster_id, _, _ in categories}
    
    for item in all_categorizations:
        sentence_idx = item["sentence_id"]
        assigned_category = item["assigned_category"]
        confidence = item.get("confidence", 0)
        explanation = item.get("explanation", "")
        sentence_text = sentences[sentence_idx] if sentence_idx < len(sentences) else ""
        
        # Validate confidence scores
        if assigned_category == "None" and confidence != 0:
            print(f"Warning: Sentence {sentence_idx} assigned 'None' but has non-zero confidence {confidence}")
            confidence = 0  # Force to 0 for "None" assignments
        elif assigned_category != "None" and (confidence < 1 or confidence > 10):
            print(f"Warning: Sentence {sentence_idx} has invalid confidence {confidence}, clamping to valid range")
            confidence = max(1, min(10, confidence))

        # Normalize confidence to 0-1
        confidence = confidence / 10.0
        
        total_confidence += confidence
        
        # Process assignment counts
        if assigned_category == "None":
            not_assigned += 1
            category_counts["None"] += 1
            category_confidences["None"].append(confidence)
        else:
            assigned += 1
            # Extract just the cluster ID from "Category N" format
            category_id = simplify_category_name(assigned_category)
            category_counts[category_id] = category_counts.get(category_id, 0) + 1
            category_confidences[category_id].append(confidence)
        
        # Enhanced analysis with ground truth if available
        if ground_truth_labels is not None and sentence_idx < len(ground_truth_labels) and sentence_idx < len(sentences):
            ground_truth = str(ground_truth_labels[sentence_idx])
            assigned_category_id = simplify_category_name(assigned_category) if assigned_category != "None" else "None"
            
            # Create detailed item for analysis
            detailed_item = {
                "sentence_id": sentence_idx,
                "sentence_text": sentence_text,
                "assigned_category": assigned_category,
                "ground_truth_category": ground_truth,
                "confidence": confidence,
                "explanation": explanation
            }
            
            assert ground_truth in valid_category_ids, f"Ground truth {ground_truth} not in valid category ids {valid_category_ids}"

            # Sentence should have been assigned to a category
            if assigned_category_id == ground_truth:
                # Correctly assigned
                detailed_analysis["correct_assignments"].append(detailed_item)
            elif assigned_category_id == "None":
                # Should have been assigned but wasn't
                detailed_analysis["missed_assignments"].append(detailed_item)
            else:
                # Assigned to wrong category
                detailed_analysis["incorrect_assignments"].append(detailed_item)
    
    # Calculate fractions
    assigned_fraction = assigned / total_sentences if total_sentences > 0 else 0
    not_assigned_fraction = not_assigned / total_sentences if total_sentences > 0 else 0
    
    # Calculate confidence metrics
    avg_confidence = total_confidence / total_sentences if total_sentences > 0 else 0
    
    # Calculate average confidence by category
    category_avg_confidences = {}
    for category_id, confidences in category_confidences.items():
        if confidences:
            category_avg_confidences[category_id] = sum(confidences) / len(confidences)
        else:
            category_avg_confidences[category_id] = 0.0
    
    # Calculate detailed metrics if ground truth is available
    completeness_metrics = {}
    if ground_truth_labels is not None:
        n_correct = len(detailed_analysis["correct_assignments"])
        n_incorrect = len(detailed_analysis["incorrect_assignments"])
        n_missed = len(detailed_analysis["missed_assignments"])
        
        # Sentences that should have been assigned (have valid ground truth categories)
        should_be_assigned = n_correct + n_incorrect + n_missed
        # Sentences that were assigned
        were_assigned = n_correct + n_incorrect
        
        # Calculate confidence-based metrics
        correct_confidences = [item["confidence"] for item in detailed_analysis["correct_assignments"]]
        incorrect_confidences = [item["confidence"] for item in detailed_analysis["incorrect_assignments"]]
        
        completeness_metrics = {
            "correct_assignments": n_correct,
            "incorrect_assignments": n_incorrect,
            "missed_assignments": n_missed,
            "assignment_accuracy": n_correct / were_assigned if were_assigned > 0 else 0,
            "assignment_recall": n_correct / should_be_assigned if should_be_assigned > 0 else 0,
            "assignment_precision": n_correct / were_assigned if were_assigned > 0 else 0,
            "avg_correct_confidence": sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0,
            "avg_incorrect_confidence": sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0
        }
    
    result_dict = {
        "total_sentences": total_sentences,
        "assigned": assigned,
        "not_assigned": not_assigned,
        "assigned_fraction": assigned_fraction,
        "not_assigned_fraction": not_assigned_fraction,
        "avg_confidence": avg_confidence,
        "category_counts": category_counts,
        "category_confidences": category_confidences,
        "category_avg_confidences": category_avg_confidences,
        "categorizations": all_categorizations,
        "detailed_analysis": detailed_analysis,
        "completeness_metrics": completeness_metrics
    }
    
    # Add information about batching if applicable
    if len(sentence_chunks) > 1:
        result_dict["batch_info"] = {
            "num_batches": len(sentence_chunks),
            "max_sentences_per_prompt": max_sentences_per_prompt,
            "parsing_errors": parsing_errors
        }
    
    return result_dict


def accuracy_autograder_batch(sentences, categories, ground_truth_labels, n_autograder_examples, max_sentences_per_prompt=50, model="gpt-4.1-mini"):
    """
    Binary autograder that evaluates each cluster independently against examples from outside the cluster using batch API.
    
    Args:
        sentences (list): List of all sentences to potentially sample from
        categories (list): List of tuples where each tuple is (cluster_id, title, description)
        ground_truth_labels (list): List of cluster IDs (as strings) for each sentence in sentences
        n_autograder_examples (int): Number of examples to sample from each cluster for testing
        max_sentences_per_prompt (int): Maximum number of sentences to process per prompt
        model (str): Model to use for the autograding
    
    Returns:
        tuple: (batch_id, metadata) for processing results later
    """
    # Get a mapping from sentence index to cluster ID for easy lookup
    sentence_to_cluster = {i: label for i, label in enumerate(ground_truth_labels)}
    
    # Collect all prompts and metadata for batch processing
    prompts_with_ids = {}
    metadata = {
        "categories": categories,
        "batch_metadata": []
    }
    
    # For each category, prepare data and prompts
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
        
        # Split sentences into chunks if necessary
        total_test_sentences = len(test_sentences)
        if total_test_sentences <= max_sentences_per_prompt:
            # Process all sentences in a single prompt
            sentence_chunks = [test_sentences]
            ground_truth_chunks = [test_ground_truth]
        else:
            # Split into multiple chunks
            sentence_chunks = []
            ground_truth_chunks = []
            
            for i in range(0, total_test_sentences, max_sentences_per_prompt):
                end_idx = min(i + max_sentences_per_prompt, total_test_sentences)
                sentence_chunks.append(test_sentences[i:end_idx])
                ground_truth_chunks.append(test_ground_truth[i:end_idx])
        
        # Create prompts for all chunks of this cluster
        for chunk_idx, (sentence_chunk, ground_truth_chunk) in enumerate(zip(sentence_chunks, ground_truth_chunks)):
            # Format the sentences into a numbered list (starting from 0 for each chunk)
            sentences_text = chr(10).join([f"Sentence {i}: {sentence}" for i, sentence in enumerate(sentence_chunk)])

            # Create a prompt for binary classification
            prompt = f"""# Task: Binary Classification of Reasoning Sentences by Function

You are an expert at analyzing the *function* of sentences within a longer chain of reasoning. Your task is to determine if each sentence below performs the specific cognitive or procedural role described.

**Core Principle:** Do not focus on the surface-level topic of the sentence. Instead, abstract away from the specific content and ask: "What *job* is this sentence doing in the reasoning trace?"

## Category Description:
Title: {title}
Description: {description}

## Sentences to Classify:
{sentences_text}

## Instructions:
1. For each sentence, identify its functional role in a potential reasoning process.
2. Compare this role to the category description provided.
3. If the sentence's function matches the description, assign "Yes". Importantly, a sentence might not match a description word-for-word, but it might serve the same underlying purpose.
4. If the sentence's function does not align with the category, assign it "No".
5. Respond with "Yes" or "No" for each sentence.

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
            custom_id = f"accuracy_{cluster_id}_{chunk_idx}"
            prompts_with_ids[custom_id] = prompt
            metadata["batch_metadata"].append({
                "custom_id": custom_id,
                "cluster_id": cluster_id,
                "cluster_id_str": cluster_id_str,
                "title": title,
                "description": description,
                "chunk_idx": chunk_idx,
                "test_sentences": sentence_chunk,
                "test_ground_truth": ground_truth_chunk,
                "test_indices": test_indices  # Store original indices for reference
            })
    
    # Submit batch
    print(f"Submitting batch for accuracy evaluation with {len(prompts_with_ids)} prompts...")
    batch_id = submit_openai_batch(
        prompts_with_ids, 
        "Accuracy evaluation",
        model=model,
        max_tokens=4000
    )
    
    return batch_id, metadata


def process_accuracy_batch(batch_id, metadata):
    """
    Process results from accuracy evaluation batch.
    
    Args:
        batch_id (str): The batch ID to process
        metadata (dict): Metadata from the batch submission
        
    Returns:
        dict: Metrics including precision, recall, accuracy and F1 score for each category
    """
    # Process batch results
    results = process_batch_results(batch_id)
    
    # Extract metadata
    categories = metadata["categories"]
    batch_metadata = metadata["batch_metadata"]
    
    # Group responses by cluster_id for aggregation
    cluster_responses = {}
    parsing_errors = []
    
    # Process each response from the batch
    for item in batch_metadata:
        custom_id = item["custom_id"]
        cluster_id_str = item["cluster_id_str"]
        
        if cluster_id_str not in cluster_responses:
            cluster_responses[cluster_id_str] = {
                "metadata": item,  # Store metadata from first chunk (contains cluster info)
                "all_classifications": []
            }
        
        response = results.get(custom_id)
        if response:
            try:
                result = parse_json_response(response, expected_field='classifications')
                
                # Add sentence text and ground truth to each classification
                for classification in result["classifications"]:
                    sentence_idx = classification["sentence_id"]
                    if sentence_idx < len(item["test_sentences"]):
                        classification["sentence_text"] = item["test_sentences"][sentence_idx]
                        classification["ground_truth"] = item["test_ground_truth"][sentence_idx]
                    else:
                        print(f"Warning: sentence_id {sentence_idx} out of range for chunk with {len(item['test_sentences'])} sentences")
                
                # Add the classifications from this chunk to the cluster's overall list
                cluster_responses[cluster_id_str]["all_classifications"].extend(result["classifications"])
                
            except Exception as e:
                print(f"Error parsing response for cluster {item['cluster_id']} chunk {item['chunk_idx']}: {e}")
                print(f"Raw response: {response}")
                parsing_errors.append({
                    "cluster_id": item["cluster_id"],
                    "chunk_idx": item["chunk_idx"],
                    "error": str(e),
                    "raw_response": response
                })
        else:
            print(f"No response received for {custom_id}")
            parsing_errors.append({
                "cluster_id": item["cluster_id"],
                "chunk_idx": item["chunk_idx"],
                "error": "No response received",
                "raw_response": None
            })
    
    # Process aggregated results for each cluster
    final_results = {}
    for cluster_id_str, cluster_data in cluster_responses.items():
        cluster_metadata = cluster_data["metadata"]
        cluster_id = cluster_metadata["cluster_id"]
        all_classifications = cluster_data["all_classifications"]
        
        if not all_classifications:
            print(f"No valid classifications for cluster {cluster_id}")
            final_results[cluster_id_str] = {
                "error": "No valid classifications",
                "parsing_errors": [e for e in parsing_errors if e["cluster_id"] == cluster_id]
            }
            continue
        
        # Compute metrics for this cluster
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        predictions = []
        enhanced_classifications = []
        
        for item in all_classifications:
            if "belongs_to_category" not in item or "ground_truth" not in item:
                print(f"Warning: Missing required fields in classification item: {item}")
                continue
                
            belongs = item["belongs_to_category"]
            predictions.append(belongs)
            
            true_label = item["ground_truth"]
            
            if belongs == "Yes" and true_label == "Yes":
                true_positives += 1
            elif belongs == "Yes" and true_label == "No":
                false_positives += 1
            elif belongs == "No" and true_label == "Yes":
                false_negatives += 1
            elif belongs == "No" and true_label == "No":
                true_negatives += 1
            
            enhanced_classifications.append(item)
        
        # Calculate metrics
        total_predictions = len(enhanced_classifications)
        accuracy = (true_positives + true_negatives) / total_predictions if total_predictions > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        final_results[cluster_id_str] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "predictions": predictions,
            "classifications": enhanced_classifications
        }
    
    # Calculate overall averages across all clusters
    if final_results:
        valid_results = {k: v for k, v in final_results.items() if "error" not in v}
        if valid_results:
            avg_accuracy = sum(r["accuracy"] for r in valid_results.values()) / len(valid_results)
            avg_precision = sum(r["precision"] for r in valid_results.values()) / len(valid_results)
            avg_recall = sum(r["recall"] for r in valid_results.values()) / len(valid_results)
            avg_f1 = sum(r["f1"] for r in valid_results.values()) / len(valid_results)
            
            final_results["avg"] = {
                "accuracy": avg_accuracy,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            }
    
    # Add parsing error summary if any occurred
    if parsing_errors:
        final_results["parsing_errors"] = parsing_errors
    
    return final_results


def compute_semantic_orthogonality_batch(categories, orthogonality_threshold=0.5, model="gpt-4.1-mini"):
    """
    Compute the semantic orthogonality of categories using LLM-based similarity evaluation with batch API.
    
    Parameters:
    -----------
    categories : list
        List of tuples (cluster_id, title, description) for each category
    orthogonality_threshold : float
        Threshold for counting orthogonal pairs (default: 0.5)
    model : str
        Model to use for semantic similarity evaluation
        
    Returns:
    --------
    tuple: (batch_id, metadata) for processing results later
    """
    start_time = time.time()
    n_categories = len(categories)
    
    if n_categories <= 1:
        return None, {
            "orthogonality_matrix": np.array([[0.0]]) if n_categories == 1 else np.array([]),
            "explanations": {},
            "orthogonality_score": 0.0,
            "orthogonality_threshold": orthogonality_threshold,
            "categories": categories
        }
    
    # Prepare batch prompts for all pairs in upper triangle
    prompts_with_ids = {}
    batch_pairs = []
    
    for i in range(n_categories):
        for j in range(i + 1, n_categories):  # Only upper triangle (i < j)
            category1 = categories[i]
            category2 = categories[j]
            
            cluster_id1, title1, description1 = category1
            cluster_id2, title2, description2 = category2
            
            prompt = f"""# Task: Semantic Similarity Evaluation

You are an expert at analyzing the semantic similarity between different reasoning functions. Your task is to evaluate how similar two categories of reasoning sentences are in terms of their underlying cognitive or functional purpose.

## Category 1:
Title: {title1}
Description: {description1}

## Category 2:
Title: {title2}
Description: {description2}

## Instructions:
Rate the semantic similarity between these two categories on a scale from 0 to 10, where:
- 0 = Completely different reasoning functions
- 5 = Somewhat related but distinct functions
- 10 = Essentially the same reasoning function, just described differently

Consider:
1. The underlying cognitive process or reasoning operation
2. The functional role within a reasoning trace
3. Whether sentences from one category could reasonably belong to the other

Focus on functional similarity rather than surface-level word overlap.

## Response Format:
Your response must follow this exact JSON format:
```json
{{
  "explanation": "Brief explanation of your reasoning for this score",
  "similarity_score": <integer from 0-10>
}}
```

Only include the JSON object in your response, with no additional text before or after.
"""
            
            custom_id = f"similarity_{i}_{j}"
            prompts_with_ids[custom_id] = prompt
            batch_pairs.append((i, j))
    
    metadata = {
        "categories": categories,
        "batch_pairs": batch_pairs,
        "orthogonality_threshold": orthogonality_threshold,
        "n_categories": n_categories
    }
    
    # Submit batch
    print(f"Submitting batch for semantic similarity evaluation with {len(prompts_with_ids)} pairs...")
    batch_id = submit_openai_batch(
        prompts_with_ids, 
        "Semantic similarity evaluation",
        model=model,
        max_tokens=1000
    )
    
    print_and_flush(f"Submitted semantic orthogonality batch in {time.time() - start_time} seconds")
    
    return batch_id, metadata


def process_semantic_orthogonality_batch(batch_id, metadata):
    """
    Process results from semantic orthogonality evaluation batch.
    
    Args:
        batch_id (str): The batch ID to process
        metadata (dict): Metadata from the batch submission
        
    Returns:
        dict: Dictionary containing orthogonality matrix and metrics
    """
    start_time = time.time()
    
    # Handle single category case
    if batch_id is None:
        return metadata
    
    # Process batch results
    results = process_batch_results(batch_id)
    
    # Extract metadata
    categories = metadata["categories"]
    batch_pairs = metadata["batch_pairs"]
    orthogonality_threshold = metadata["orthogonality_threshold"]
    n_categories = metadata["n_categories"]
    
    # Initialize similarity matrix and explanations dictionary
    similarity_matrix = np.zeros((n_categories, n_categories))
    explanations = {}
    
    # Fill diagonal with 1.0 (perfect self-similarity)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # Parse responses and fill similarity matrix
    for idx, (i, j) in enumerate(batch_pairs):
        custom_id = f"similarity_{i}_{j}"
        response = results.get(custom_id)
        
        if response:
            try:
                result = parse_json_response(response)
                similarity_score = result.get('similarity_score', 0)
                explanation = result.get('explanation', '')
                
                # Validate score is in range
                if not isinstance(similarity_score, (int, float)) or similarity_score < 0 or similarity_score > 10:
                    print(f"Warning: Invalid similarity score {similarity_score}, clamping to valid range")
                    similarity_score = max(0, min(10, int(similarity_score)))
                
                # Normalize to 0-1
                normalized_score = similarity_score / 10.0
                
                # Fill both positions in the matrix (symmetric)
                similarity_matrix[i, j] = normalized_score
                similarity_matrix[j, i] = normalized_score
                
                # Store explanations for both pairs (symmetric)
                explanations[f"{i},{j}"] = explanation
                explanations[f"{j},{i}"] = explanation
                
            except Exception as e:
                print(f"Error parsing semantic similarity response for pair ({i}, {j}): {e}")
                print(f"Raw response: {response}")
                # Default to 0 similarity on error
                similarity_matrix[i, j] = 0.0
                similarity_matrix[j, i] = 0.0
                # Default explanation on error
                explanations[f"{i},{j}"] = "Error parsing response"
                explanations[f"{j},{i}"] = "Error parsing response"
        else:
            print(f"No response received for similarity pair ({i}, {j})")
            # Default to 0 similarity on no response
            similarity_matrix[i, j] = 0.0
            similarity_matrix[j, i] = 0.0
            # Default explanation on no response
            explanations[f"{i},{j}"] = "No response received"
            explanations[f"{j},{i}"] = "No response received"
    
    # Calculate orthogonality as 1 - similarity
    orthogonality_matrix = 1 - similarity_matrix
    
    # Get the indices of the upper triangular part (excluding diagonal)
    indices = np.triu_indices(orthogonality_matrix.shape[0], k=1)
    
    # Extract the upper triangular values
    upper_tri_values = orthogonality_matrix[indices]
    
    # Calculate orthogonality score (fraction of pairs above threshold)
    orthogonality_score = np.sum(upper_tri_values > orthogonality_threshold) / len(upper_tri_values) if len(upper_tri_values) > 0 else 0
    
    print_and_flush(f"Processed semantic orthogonality batch in {time.time() - start_time} seconds")
    
    return {
        "semantic_orthogonality_matrix": orthogonality_matrix,
        "semantic_orthogonality_explanations": explanations,
        "semantic_orthogonality_score": orthogonality_score,
        "semantic_orthogonality_threshold": orthogonality_threshold
    }