#!/usr/bin/env python3

import argparse
import json
import os
from utils.utils import print_and_flush
from utils.clustering_batched import check_batch_status
from utils.clustering import (
    load_trained_clustering_data, predict_clusters, 
    generate_representative_examples,
    generate_cluster_descriptions
)
from utils.clustering_batched import (
    generate_cluster_descriptions_batch, process_cluster_descriptions_batch
)
from utils import utils
import numpy as np
import gc
import torch

# %%

parser = argparse.ArgumentParser(description="Generate cluster titles using batch API")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to analyze")
parser.add_argument("--layer", type=int, default=12,
                    help="Layer to analyze")
parser.add_argument("--n_examples", type=int, default=500,
                    help="Number of examples to analyze")
parser.add_argument("--clustering_methods", type=str, nargs='+', 
                    default=["gmm", "pca_gmm", "spherical_kmeans", "pca_kmeans", "agglomerative", "pca_agglomerative", "sae_topk"],
                    help="Clustering methods to evaluate")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--description_examples", type=int, default=200,
                    help="Number of examples to use for generating cluster descriptions")
parser.add_argument("--evaluator_model", type=str, default="o4-mini",
                    help="Model to use for generating descriptions")
parser.add_argument("--command", type=str, choices=["submit", "process", "direct"], required=True,
                    help="Command to run: submit batch jobs, process results, or generate directly")
parser.add_argument("--batch_file", type=str, default=None,
                    help="JSON file containing batch information (for process command)")
parser.add_argument("--check_status", action="store_true", default=False,
                    help="Check status of pending batches before processing")
parser.add_argument("--repetitions", type=int, default=5,
                    help="Number of repetitions for generating different category sets")
parser.add_argument("--cluster_sizes", type=int, nargs='+', default=None,
                    help="Specific cluster sizes to process (if None, process all available cluster sizes)")

args, _ = parser.parse_known_args()


def submit_description_batches():
    """Submit batch jobs for generating cluster descriptions."""
    print_and_flush("=== SUBMITTING CLUSTER DESCRIPTION BATCHES ===")
    
    # Get model identifier for file naming
    model_id = args.model.split('/')[-1].lower()
    
    # Load model and process activations
    print_and_flush("Loading model and processing activations...")
    model, tokenizer = utils.load_model(
        model_name=args.model,
        load_in_8bit=args.load_in_8bit
    )

    # Process saved responses
    all_activations, all_texts, overall_mean = utils.process_saved_responses(
        args.model, 
        args.n_examples,
        model,
        tokenizer,
        args.layer
    )

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # Center activations
    all_activations = [x - overall_mean for x in all_activations]
    all_activations = np.stack([a.reshape(-1) for a in all_activations])
    norms = np.linalg.norm(all_activations, axis=1, keepdims=True)
    all_activations = all_activations / norms
    
    batch_info = {}
    
    # Process each clustering method
    for method in args.clustering_methods:
        print_and_flush(f"\n=== Processing {method.upper()} ===")
        
        # Load existing results to get cluster sizes
        results_json_path = f'results/vars/{method}_results_{model_id}_layer{args.layer}.json'
        if not os.path.exists(results_json_path):
            print_and_flush(f"No existing results found for {method} at {results_json_path}. Skipping.")
            continue
            
        with open(results_json_path, 'r') as f:
            existing_results = json.load(f)
        
        cluster_sizes = list(existing_results.get("results_by_cluster_size", {}).keys())
        if not cluster_sizes:
            print_and_flush(f"No clustering results found for {method}. Skipping.")
            continue
            
        # Filter cluster sizes if specified
        if args.cluster_sizes is not None:
            requested_sizes = [str(size) for size in args.cluster_sizes]
            cluster_sizes = [size for size in cluster_sizes if size in requested_sizes]
            if not cluster_sizes:
                print_and_flush(f"None of the requested cluster sizes {args.cluster_sizes} found for {method}. Skipping.")
                continue
            
        method_batches = {}
        
        # Process each cluster size
        for cluster_size in cluster_sizes:
            n_clusters = int(cluster_size)
            print_and_flush(f"Submitting batch for {method} with {n_clusters} clusters...")
            
            try:
                # Load the trained clustering model
                clustering_data = load_trained_clustering_data(model_id, args.layer, n_clusters, method)
                cluster_centers = clustering_data['cluster_centers']
                
                # Predict cluster labels for current activations
                if method == 'sae_topk':
                    cluster_labels = predict_clusters(all_activations, clustering_data, model_id, args.layer, n_clusters)
                else:
                    cluster_labels = predict_clusters(all_activations, clustering_data)
                
                # Generate representative examples
                representative_examples = generate_representative_examples(
                    cluster_centers, all_texts, cluster_labels, all_activations, 
                    clustering_data=clustering_data, model_id=model_id, layer=args.layer, n_clusters=n_clusters
                )
                
                # Prepare examples for batch processing
                cluster_examples_list = []
                for cluster_idx in range(n_clusters):
                    if len(representative_examples[cluster_idx]) == 0:
                        print_and_flush(f"WARNING: Skipping empty cluster {cluster_idx}")
                        continue
                    
                    # Sample examples from across the cluster for diversity
                    cluster_examples = representative_examples[cluster_idx]
                    total_examples = len(cluster_examples)
                    
                    if total_examples <= args.description_examples:
                        examples = cluster_examples
                    else:
                        # Sample from different deciles
                        examples = []
                        examples_per_decile = args.description_examples // 10
                        remainder = args.description_examples % 10
                        
                        for decile in range(10):
                            start_idx = (decile * total_examples) // 10
                            end_idx = ((decile + 1) * total_examples) // 10
                            
                            decile_examples = cluster_examples[start_idx:end_idx]
                            
                            num_to_sample = examples_per_decile
                            if decile < remainder:
                                num_to_sample += 1
                                
                            examples.extend(decile_examples[:num_to_sample])
                    
                    # Shuffle examples for diversity
                    import random
                    random.shuffle(examples)
                    cluster_examples_list.append((cluster_idx, examples))
                
                # Submit batches for multiple repetitions (different category sets)
                cluster_size_batches = {}
                
                for rep_idx in range(args.repetitions):
                    print_and_flush(f"  Submitting repetition {rep_idx + 1}/{args.repetitions}...")
                    
                    # Submit batch for this repetition
                    batch_id, cluster_indices = generate_cluster_descriptions_batch(
                        args.model, cluster_examples_list, model=args.evaluator_model
                    )
                    
                    cluster_size_batches[f"rep_{rep_idx}"] = {
                        "batch_id": batch_id,
                        "cluster_indices": cluster_indices,
                        "n_clusters": n_clusters,
                        "method": method
                    }
                    
                    print_and_flush(f"  Submitted batch {batch_id} for repetition {rep_idx + 1}")
                
                method_batches[cluster_size] = cluster_size_batches
                
                print_and_flush(f"Submitted all repetition batches for {method} with {n_clusters} clusters")
                
            except Exception as e:
                print_and_flush(f"Error processing {method} with {n_clusters} clusters: {e}")
                continue
        
        batch_info[method] = method_batches
    
    # Save batch information
    batch_info_file = f"batch_info_titles_{model_id}_layer{args.layer}.json"
    with open(batch_info_file, 'w') as f:
        json.dump(batch_info, f, indent=2)
    
    print_and_flush(f"\nBatch information saved to {batch_info_file}")
    print_and_flush("All batches submitted successfully!")


def process_description_batches():
    """Process completed batch jobs and update results with cluster descriptions."""
    print_and_flush("=== PROCESSING CLUSTER DESCRIPTION BATCHES ===")
    
    # Get model identifier for file naming
    model_id = args.model.split('/')[-1].lower()
    
    # Load batch information
    if args.batch_file:
        batch_info_file = args.batch_file
    else:
        batch_info_file = f"batch_info_titles_{model_id}_layer{args.layer}.json"
    
    if not os.path.exists(batch_info_file):
        print_and_flush(f"Batch information file {batch_info_file} not found!")
        return
    
    with open(batch_info_file, 'r') as f:
        batch_info = json.load(f)
    
    # Check status of all batches first if requested
    if args.check_status:
        print_and_flush("Checking batch status...")
        all_completed = True
        for method, method_batches in batch_info.items():
            for cluster_size, cluster_data in method_batches.items():
                for rep_key, rep_data in cluster_data.items():
                    if not rep_key.startswith("rep_"):
                        continue
                    batch_id = rep_data["batch_id"]
                    status = check_batch_status(batch_id)
                    print_and_flush(f"{method} {cluster_size} {rep_key}: {batch_id} -> {status}")
                    if status != "completed":
                        all_completed = False
        
        if not all_completed:
            print_and_flush("Not all batches are completed. Exiting.")
            return
    
    # Process each method's batches
    for method, method_batches in batch_info.items():
        print_and_flush(f"\n=== Processing {method.upper()} ===")
        
        # Load existing results
        results_json_path = f'results/vars/{method}_results_{model_id}_layer{args.layer}.json'
        if not os.path.exists(results_json_path):
            print_and_flush(f"No existing results found for {method} at {results_json_path}. Skipping.")
            continue
            
        with open(results_json_path, 'r') as f:
            existing_results = json.load(f)
        
        # Filter cluster sizes if specified
        if args.cluster_sizes is not None:
            requested_sizes = [str(size) for size in args.cluster_sizes]
            method_batches = {size: data for size, data in method_batches.items() if size in requested_sizes}
            if not method_batches:
                print_and_flush(f"None of the requested cluster sizes {args.cluster_sizes} found for {method}. Skipping.")
                continue
        
        # Process each cluster size
        for cluster_size, cluster_data in method_batches.items():
            n_clusters = None
            all_categories = []  # Store categories from all repetitions
            
            print_and_flush(f"Processing batches for {cluster_size} clusters...")
            
            # Process each repetition
            for rep_idx in range(args.repetitions):
                rep_key = f"rep_{rep_idx}"
                if rep_key not in cluster_data:
                    print_and_flush(f"Missing repetition {rep_idx} for {method} {cluster_size}. Skipping.")
                    continue
                
                rep_data = cluster_data[rep_key]
                batch_id = rep_data["batch_id"]
                cluster_indices = rep_data["cluster_indices"]
                if n_clusters is None:
                    n_clusters = rep_data["n_clusters"]
                
                print_and_flush(f"  Processing repetition {rep_idx + 1} batch {batch_id}...")
                
                try:
                    # Check batch status
                    status = check_batch_status(batch_id)
                    if status != "completed":
                        print_and_flush(f"  Batch {batch_id} not completed (status: {status}). Skipping.")
                        continue
                    
                    # Process batch results
                    categories = process_cluster_descriptions_batch(batch_id, cluster_indices)
                    
                    print_and_flush(f"  Generated descriptions for {len(categories)} clusters:")
                    for cluster_id, title, description in categories:
                        print_and_flush(f"    Cluster {cluster_id}: {title}")
                        print_and_flush(f"      {description}")
                    
                    all_categories.append(categories)
                    print_and_flush(f"  Successfully processed repetition {rep_idx + 1}")
                    
                except Exception as e:
                    print_and_flush(f"  Error processing batch {batch_id}: {e}")
                    continue
            
            # Update existing results with category information
            if all_categories and "results_by_cluster_size" in existing_results:
                cluster_results = existing_results["results_by_cluster_size"].get(cluster_size, {})
                
                # Create empty all_results structure with correct number of repetitions
                if len(all_categories) > 0:
                    cluster_results["all_results"] = []
                    for rep_idx, categories in enumerate(all_categories):
                        cluster_results["all_results"].append({"categories": categories})
                    
                    existing_results["results_by_cluster_size"][cluster_size] = cluster_results
                    print_and_flush(f"Updated results with {len(all_categories)} different category sets for {n_clusters} clusters")
                else:
                    print_and_flush(f"No valid categories generated for {cluster_size} clusters")
            
            print_and_flush(f"Completed processing for {cluster_size} clusters")
        
        # Save updated results
        with open(results_json_path, 'w') as f:
            json.dump(existing_results, f, indent=2)
        print_and_flush(f"Updated results saved to {results_json_path}")
    
    print_and_flush("All batch results processed successfully!")


def generate_descriptions_direct():
    """Generate cluster descriptions directly without using batch API."""
    print_and_flush("=== GENERATING CLUSTER DESCRIPTIONS DIRECTLY ===")
    
    # Get model identifier for file naming
    model_id = args.model.split('/')[-1].lower()
    
    # Load model and process activations
    print_and_flush("Loading model and processing activations...")
    model, tokenizer = utils.load_model(
        model_name=args.model,
        load_in_8bit=args.load_in_8bit
    )

    # Process saved responses
    all_activations, all_texts, overall_mean = utils.process_saved_responses(
        args.model, 
        args.n_examples,
        model,
        tokenizer,
        args.layer
    )

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # Center activations
    all_activations = [x - overall_mean for x in all_activations]
    all_activations = np.stack([a.reshape(-1) for a in all_activations])
    norms = np.linalg.norm(all_activations, axis=1, keepdims=True)
    all_activations = all_activations / norms
    
    # Process each clustering method
    for method in args.clustering_methods:
        print_and_flush(f"\n=== Processing {method.upper()} ===")
        
        # Load existing results to get cluster sizes
        results_json_path = f'results/vars/{method}_results_{model_id}_layer{args.layer}.json'
        if not os.path.exists(results_json_path):
            print_and_flush(f"No existing results found for {method} at {results_json_path}. Skipping.")
            continue
            
        with open(results_json_path, 'r') as f:
            existing_results = json.load(f)
        
        cluster_sizes = list(existing_results.get("results_by_cluster_size", {}).keys())
        if not cluster_sizes:
            print_and_flush(f"No clustering results found for {method}. Skipping.")
            continue
            
        # Filter cluster sizes if specified
        if args.cluster_sizes is not None:
            requested_sizes = [str(size) for size in args.cluster_sizes]
            cluster_sizes = [size for size in cluster_sizes if size in requested_sizes]
            if not cluster_sizes:
                print_and_flush(f"None of the requested cluster sizes {args.cluster_sizes} found for {method}. Skipping.")
                continue
        
        # Process each cluster size
        for cluster_size in cluster_sizes:
            n_clusters = int(cluster_size)
            print_and_flush(f"Generating descriptions for {method} with {n_clusters} clusters...")
            
            try:
                # Load the trained clustering model
                clustering_data = load_trained_clustering_data(model_id, args.layer, n_clusters, method)
                cluster_centers = clustering_data['cluster_centers']
                
                # Predict cluster labels for current activations
                if method == 'sae_topk':
                    cluster_labels = predict_clusters(all_activations, clustering_data, model_id, args.layer, n_clusters)
                else:
                    cluster_labels = predict_clusters(all_activations, clustering_data)
                
                # Generate representative examples
                representative_examples = generate_representative_examples(
                    cluster_centers, all_texts, cluster_labels, all_activations, 
                    clustering_data=clustering_data, model_id=model_id, layer=args.layer, n_clusters=n_clusters
                )
                
                # Prepare examples for description generation
                cluster_examples_list = []
                for cluster_idx in range(n_clusters):
                    if len(representative_examples[cluster_idx]) == 0:
                        print_and_flush(f"WARNING: Skipping empty cluster {cluster_idx}")
                        continue
                    
                    # Sample examples from across the cluster for diversity
                    cluster_examples = representative_examples[cluster_idx]
                    total_examples = len(cluster_examples)
                    
                    if total_examples <= args.description_examples:
                        examples = cluster_examples
                    else:
                        # Sample from different deciles
                        examples = []
                        examples_per_decile = args.description_examples // 10
                        remainder = args.description_examples % 10
                        
                        for decile in range(10):
                            start_idx = (decile * total_examples) // 10
                            end_idx = ((decile + 1) * total_examples) // 10
                            
                            decile_examples = cluster_examples[start_idx:end_idx]
                            
                            num_to_sample = examples_per_decile
                            if decile < remainder:
                                num_to_sample += 1
                                
                            examples.extend(decile_examples[:num_to_sample])
                    
                    # Shuffle examples for diversity
                    import random
                    random.shuffle(examples)
                    cluster_examples_list.append((cluster_idx, examples))
                
                # Generate descriptions for multiple repetitions
                all_categories = []
                
                for rep_idx in range(args.repetitions):
                    print_and_flush(f"  Generating repetition {rep_idx + 1}/{args.repetitions}...")
                    
                    # Generate cluster descriptions directly
                    categories = generate_cluster_descriptions(
                        args.model, 
                        cluster_examples_list, 
                        args.evaluator_model,
                        n_trace_examples=0,
                        n_categories_examples=5
                    )
                    
                    print_and_flush(f"  Generated descriptions for {len(categories)} clusters:")
                    for cluster_id, title, description in categories:
                        print_and_flush(f"    Cluster {cluster_id}: {title}")
                        print_and_flush(f"      {description}")
                    
                    all_categories.append(categories)
                    print_and_flush(f"  Successfully completed repetition {rep_idx + 1}")
                
                # Update existing results with category information
                if all_categories and "results_by_cluster_size" in existing_results:
                    cluster_results = existing_results["results_by_cluster_size"].get(cluster_size, {})
                    
                    # Create all_results structure with generated categories
                    if len(all_categories) > 0:
                        cluster_results["all_results"] = []
                        for rep_idx, categories in enumerate(all_categories):
                            cluster_results["all_results"].append({"categories": categories})
                        
                        existing_results["results_by_cluster_size"][cluster_size] = cluster_results
                        print_and_flush(f"Updated results with {len(all_categories)} different category sets for {n_clusters} clusters")
                    else:
                        print_and_flush(f"No valid categories generated for {cluster_size} clusters")
                
                print_and_flush(f"Completed processing for {cluster_size} clusters")
                
            except Exception as e:
                print_and_flush(f"Error processing {method} with {n_clusters} clusters: {e}")
                continue
        
        # Save updated results
        with open(results_json_path, 'w') as f:
            json.dump(existing_results, f, indent=2)
        print_and_flush(f"Updated results saved to {results_json_path}")
    
    print_and_flush("All cluster descriptions generated successfully!")


def main():
    if args.command == "submit":
        submit_description_batches()
    elif args.command == "process":
        process_description_batches()
    elif args.command == "direct":
        generate_descriptions_direct()


if __name__ == "__main__":
    main()