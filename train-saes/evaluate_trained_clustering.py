# %%
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
import torch
import gc
from utils.utils import print_and_flush
from utils.clustering_batched import check_batch_status
from utils.clustering import (
    SUPPORTED_CLUSTERING_METHODS,
    load_trained_clustering_data, predict_clusters, save_clustering_results, compute_centroid_orthogonality
)
from utils.clustering_batched import (
    accuracy_autograder_batch, process_accuracy_batch,
    completeness_autograder_batch, process_completeness_batch,
    compute_semantic_orthogonality_batch, process_semantic_orthogonality_batch
)
from utils import utils
import random

# %%

parser = argparse.ArgumentParser(description="Evaluate saved clustering models using batch API")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="Model to analyze")
parser.add_argument("--layer", type=int, default=12,
                    help="Layer to analyze")
parser.add_argument("--n_examples", type=int, default=500,
                    help="Number of examples to analyze")
parser.add_argument("--clustering_methods", type=str, nargs='+', 
                    default=list(SUPPORTED_CLUSTERING_METHODS),
                    help="Clustering methods to evaluate")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--re_compute_cluster_labels", action="store_true", default=False,
                    help="Re-compute cluster labels and centers, and save them to the existing file")
parser.add_argument("--n_autograder_examples", type=int, default=100,
                    help="Number of examples from each cluster to use for autograding")
parser.add_argument("--n_completeness_examples", type=int, default=500,
                    help="Number of examples to use for completeness evaluation")
parser.add_argument("--description_examples", type=int, default=200,
                    help="Number of examples to use for generating cluster descriptions")
parser.add_argument("--evaluator_model", type=str, default="gpt-4.1-mini",
                    help="Model to use for evaluations")
parser.add_argument("--command", type=str, choices=["submit", "process"], required=True,
                    help="Command to run: submit batch jobs or process results")
parser.add_argument("--batch_file", type=str, default=None,
                    help="JSON file containing batch information (for process command)")
parser.add_argument("--check_status", action="store_true", default=False,
                    help="Check status of pending batches before processing")
parser.add_argument("--repetitions", type=int, default=5,
                    help="Number of repetitions for evaluation")
args, _ = parser.parse_known_args()

# %% Get model identifier for file naming
model_id = args.model.split('/')[-1].lower()

clustering_methods = [method for method in args.clustering_methods if method in SUPPORTED_CLUSTERING_METHODS]

def submit_evaluation_batches():
    """Submit batch jobs for evaluating clustering methods."""
    print_and_flush("=== SUBMITTING CLUSTERING EVALUATION BATCHES ===")
    
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
    
    # Filter clustering methods based on args
    clustering_methods = [method for method in args.clustering_methods if method in SUPPORTED_CLUSTERING_METHODS]
    
    # Process each clustering method
    for method in clustering_methods:
        print_and_flush(f"\n=== Processing {method.upper()} ===")
        
        # Load existing results to get cluster sizes and categories
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
            
        method_batches = {}
        
        # Process each cluster size
        for cluster_size in cluster_sizes:
            n_clusters = int(cluster_size)
            print_and_flush(f"Submitting batches for {method} with {n_clusters} clusters...")
            
            try:
                # Load the trained clustering model
                clustering_data = load_trained_clustering_data(model_id, args.layer, n_clusters, method)
                cluster_centers = clustering_data['cluster_centers']
                
                # Predict cluster labels for current activations
                if args.re_compute_cluster_labels:
                    cluster_labels = predict_clusters(all_activations, clustering_data)
                    clustering_data['cluster_labels'] = cluster_labels
                else:
                    cluster_labels = clustering_data['cluster_labels']
                
                # Get all categories from existing results (one set per repetition)
                cluster_results = existing_results["results_by_cluster_size"].get(cluster_size, {})
                all_categories = []
                if "all_results" in cluster_results and len(cluster_results["all_results"]) > 0:
                    for repetition_data in cluster_results["all_results"]:
                        if "categories" in repetition_data:
                            all_categories.append(repetition_data["categories"])
                
                if not all_categories:
                    print_and_flush(f"No categories found for {method} with {n_clusters} clusters. Skipping evaluation.")
                    continue
                
                # Ensure we have enough category sets for the number of repetitions
                if len(all_categories) < args.repetitions:
                    print_and_flush(f"Only {len(all_categories)} category sets found, but {args.repetitions} repetitions requested. Using available categories.")
                    # Repeat the last category set if we don't have enough
                    while len(all_categories) < args.repetitions:
                        all_categories.append(all_categories[-1])
                
                cluster_size_batches = {}
                
                # Submit batches for multiple repetitions, each with its own categories
                for rep_idx in range(args.repetitions):
                    categories = all_categories[rep_idx]  # Use categories specific to this repetition
                    print_and_flush(f"  Submitting repetition {rep_idx + 1}/{args.repetitions} with {len(categories)} categories...")
                    
                    rep_batches = {}
                    
                    # Submit accuracy evaluation batch
                    str_cluster_labels = [str(label) for label in cluster_labels]
                    acc_batch_id, acc_metadata = accuracy_autograder_batch(
                        all_texts, categories, str_cluster_labels, args.n_autograder_examples,
                        model=args.evaluator_model
                    )
                    rep_batches["accuracy"] = {
                        "batch_id": acc_batch_id,
                        "metadata": acc_metadata
                    }
                    
                    # Submit completeness evaluation batch  
                    # Sample texts for completeness evaluation if needed
                    if len(all_texts) > args.n_completeness_examples:
                        sample_indices = random.sample(range(len(all_texts)), args.n_completeness_examples)
                        completeness_texts = [all_texts[i] for i in sample_indices]
                        completeness_labels = [str_cluster_labels[i] for i in sample_indices]
                    else:
                        completeness_texts = all_texts
                        completeness_labels = str_cluster_labels
                    
                    comp_batch_id, comp_metadata = completeness_autograder_batch(
                        completeness_texts, categories, completeness_labels, 
                        model=args.evaluator_model
                    )
                    rep_batches["completeness"] = {
                        "batch_id": comp_batch_id,
                        "metadata": comp_metadata
                    }
                    
                    # Submit semantic orthogonality batch (only once per cluster size, using first repetition's categories)
                    if rep_idx == 0:
                        sem_batch_id, sem_metadata = compute_semantic_orthogonality_batch(
                            categories, model=args.evaluator_model
                        )
                        rep_batches["semantic_orthogonality"] = {
                            "batch_id": sem_batch_id,
                            "metadata": sem_metadata
                        }
                    
                    # Store categories used for this repetition
                    rep_batches["categories"] = categories
                    cluster_size_batches[f"rep_{rep_idx}"] = rep_batches
                
                # Store other data needed for processing
                cluster_size_batches["cluster_centers"] = cluster_centers.tolist()
                cluster_size_batches["n_clusters"] = n_clusters
                cluster_size_batches["method"] = method
                
                method_batches[cluster_size] = cluster_size_batches
                
                print_and_flush(f"Submitted all batches for {method} with {n_clusters} clusters")
                
            except Exception as e:
                print_and_flush(f"Error processing {method} with {n_clusters} clusters: {e}")
                continue
        
        batch_info[method] = method_batches
    
    # Save batch information
    batch_info_file = f"batch_info_eval_{model_id}_layer{args.layer}.json"
    with open(batch_info_file, 'w') as f:
        json.dump(batch_info, f, indent=2)
    
    print_and_flush(f"\nBatch information saved to {batch_info_file}")
    print_and_flush("All evaluation batches submitted successfully!")


def process_evaluation_batches():
    """Process completed batch jobs and update results with evaluation metrics."""
    print_and_flush("=== PROCESSING CLUSTERING EVALUATION BATCHES ===")
    
    # Load batch information
    if args.batch_file:
        batch_info_file = args.batch_file
    else:
        batch_info_file = f"batch_info_eval_{model_id}_layer{args.layer}.json"
    
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
                    for eval_type, batch_data in rep_data.items():
                        if "batch_id" in batch_data:
                            batch_id = batch_data["batch_id"]
                            status = check_batch_status(batch_id)
                            print_and_flush(f"{method} {cluster_size} {rep_key} {eval_type}: {batch_id} -> {status}")
                            if status != "completed":
                                all_completed = False
        
        if not all_completed:
            print_and_flush("Not all batches are completed. Exiting.")
            return
    
    # Process each method's batches
    for method, method_batches in batch_info.items():
        print_and_flush(f"\n=== Processing {method.upper()} ===")
        
        eval_results_by_cluster_size = {}
        
        # Process each cluster size
        for cluster_size, cluster_data in method_batches.items():
            n_clusters = cluster_data["n_clusters"]
            cluster_centers = np.array(cluster_data["cluster_centers"])
            
            print_and_flush(f"Processing {n_clusters} clusters...")
            
            all_results = []
            semantic_orthogonality_result = None
            
            # Process each repetition
            for rep_idx in range(args.repetitions):
                rep_key = f"rep_{rep_idx}"
                if rep_key not in cluster_data:
                    print_and_flush(f"Missing repetition {rep_idx} for {method} {cluster_size}. Skipping.")
                    continue
                
                rep_data = cluster_data[rep_key]
                
                # Get categories specific to this repetition
                categories = rep_data.get("categories", [])
                if not categories:
                    print_and_flush(f"No categories found for repetition {rep_idx} in {method} {cluster_size}. Skipping.")
                    continue
                
                rep_results = {}
                
                try:
                    # Process accuracy batch
                    if "accuracy" in rep_data:
                        acc_batch_id = rep_data["accuracy"]["batch_id"]
                        acc_metadata = rep_data["accuracy"]["metadata"]
                        
                        status = check_batch_status(acc_batch_id)
                        if status == "completed":
                            accuracy_results = process_accuracy_batch(acc_batch_id, acc_metadata)
                            rep_results["avg_accuracy"] = accuracy_results["avg"]["accuracy"]
                            rep_results["avg_f1"] = accuracy_results["avg"]["f1"]
                            rep_results["avg_precision"] = accuracy_results["avg"]["precision"]
                            rep_results["avg_recall"] = accuracy_results["avg"]["recall"]
                        else:
                            print_and_flush(f"Accuracy batch {acc_batch_id} not completed (status: {status})")
                            continue
                    
                    # Process completeness batch
                    if "completeness" in rep_data:
                        comp_batch_id = rep_data["completeness"]["batch_id"]
                        comp_metadata = rep_data["completeness"]["metadata"]
                        
                        status = check_batch_status(comp_batch_id)
                        if status == "completed":
                            completeness_results = process_completeness_batch(comp_batch_id, comp_metadata)
                            rep_results["assigned_fraction"] = completeness_results["assigned_fraction"]
                            rep_results["avg_confidence"] = completeness_results["avg_confidence"]
                            rep_results["category_counts"] = completeness_results["category_counts"]
                            rep_results["category_confidences"] = completeness_results["category_confidences"]
                            rep_results["category_avg_confidences"] = completeness_results["category_avg_confidences"]
                            rep_results["completeness_detailed"] = completeness_results["detailed_analysis"]
                            rep_results["completeness_metrics"] = completeness_results["completeness_metrics"]
                        else:
                            print_and_flush(f"Completeness batch {comp_batch_id} not completed (status: {status})")
                            continue
                    
                    # Process semantic orthogonality batch (only once per cluster size)
                    if "semantic_orthogonality" in rep_data and semantic_orthogonality_result is None:
                        sem_batch_id = rep_data["semantic_orthogonality"]["batch_id"]
                        sem_metadata = rep_data["semantic_orthogonality"]["metadata"]
                        
                        if sem_batch_id:  # Check if batch was actually submitted
                            status = check_batch_status(sem_batch_id)
                            if status == "completed":
                                semantic_orthogonality_result = process_semantic_orthogonality_batch(sem_batch_id, sem_metadata)
                            else:
                                print_and_flush(f"Semantic orthogonality batch {sem_batch_id} not completed (status: {status})")
                        else:
                            # Handle single category case
                            semantic_orthogonality_result = sem_metadata
                    
                    # Compute centroid orthogonality
                    orthogonality = compute_centroid_orthogonality(cluster_centers)
                    rep_results["orthogonality"] = orthogonality
                    
                    # Add semantic orthogonality results
                    if semantic_orthogonality_result:
                        rep_results["semantic_orthogonality_matrix"] = semantic_orthogonality_result["semantic_orthogonality_matrix"]
                        rep_results["semantic_orthogonality_explanations"] = semantic_orthogonality_result["semantic_orthogonality_explanations"]
                        rep_results["semantic_orthogonality_score"] = semantic_orthogonality_result["semantic_orthogonality_score"]
                        rep_results["semantic_orthogonality_threshold"] = semantic_orthogonality_result["semantic_orthogonality_threshold"]
                    else:
                        rep_results["semantic_orthogonality_score"] = 0.0
                    
                    # Add categories (repetition-specific)
                    rep_results["categories"] = categories
                    
                    # Calculate final score
                    final_score = (rep_results["avg_f1"] + rep_results["avg_confidence"] + rep_results["semantic_orthogonality_score"]) / 3
                    rep_results["final_score"] = final_score
                    
                    all_results.append(rep_results)
                    
                    print_and_flush(f"  Processed repetition {rep_idx + 1}: final_score={final_score:.4f}")
                    
                except Exception as e:
                    print_and_flush(f"Error processing repetition {rep_idx} for {method} {cluster_size}: {e}")
                    continue
            
            if not all_results:
                print_and_flush(f"No valid results for {method} with {n_clusters} clusters")
                continue
            
            # Calculate average final score and statistics
            avg_final_score = np.mean([result['final_score'] for result in all_results])
            
            # Compute statistics across all repetitions
            statistics = {}
            metrics_to_stat = [
                'avg_accuracy', 'avg_f1', 'avg_precision', 'avg_recall', 'orthogonality',
                'semantic_orthogonality_score', 'assigned_fraction', 'avg_confidence', 'final_score'
            ]

            for metric in metrics_to_stat:
                values = [res[metric] for res in all_results if metric in res]
                if not values:
                    continue

                mean = np.mean(values)
                
                n = len(values)
                if n > 1:
                    from scipy import stats
                    std_dev = np.std(values, ddof=1)  # Sample standard deviation
                    sem = stats.sem(values)
                    # 95% confidence interval using the t-distribution
                    conf = 0.95
                    t_score = stats.t.ppf((1+conf)/2, df=n - 1)
                    ci_95 = (mean - t_score * sem, mean + t_score * sem)
                else:
                    std_dev = 0
                    sem = 0
                    ci_95 = (mean, mean)
                
                statistics[metric] = {
                    'mean': mean,
                    'std_dev': std_dev,
                    'sem': sem,
                    'ci_95': ci_95
                }
            
            eval_results_by_cluster_size[cluster_size] = {
                "all_results": all_results,
                "avg_final_score": avg_final_score,
                "statistics": statistics
            }
            
            print_and_flush(f"Completed {method} with {n_clusters} clusters: avg_score={avg_final_score:.4f}")
        
        # Save results using the existing function
        if eval_results_by_cluster_size:
            save_clustering_results(model_id, args.layer, method, eval_results_by_cluster_size)
            print_and_flush(f"Saved results for {method}")
        else:
            print_and_flush(f"No results to save for {method}")
    
    print_and_flush("All evaluation batch results processed successfully!")


def print_evaluation_summary(results, method):
    """
    Print a summary of evaluation results.
    
    Parameters:
    -----------
    results : dict
        Evaluation results
    method : str
        Clustering method name
    """
    
    print_and_flush("\n" + "="*50)
    print_and_flush(f"{method.upper()} EVALUATION SUMMARY")
    print_and_flush("="*50)
    print_and_flush(f"Model: {model_id.upper()}, Layer: {args.layer}")
    
    # Extract data from new format
    best_cluster = results['best_cluster']
    results_by_cluster_size = results['results_by_cluster_size']
    
    print_and_flush(f"Optimal clusters: {best_cluster['size']}")
    print_and_flush(f"Optimal final score: {best_cluster['avg_final_score']:.4f}")
    print_and_flush(f"Optimal accuracy: {best_cluster['avg_accuracy']:.4f}")
    print_and_flush(f"Optimal precision: {best_cluster['avg_precision']:.4f}")
    print_and_flush(f"Optimal recall: {best_cluster['avg_recall']:.4f}")
    print_and_flush(f"Optimal F1: {best_cluster['avg_f1']:.4f}")
    print_and_flush(f"Optimal completeness: {best_cluster['completeness']:.4f}")
    print_and_flush(f"Optimal orthogonality: {best_cluster['orthogonality']:.4f}")
    print_and_flush(f"Optimal semantic orthogonality: {best_cluster['semantic_orthogonality']:.4f}")
    
    print_and_flush("\nMetrics for all cluster sizes:")
    print_and_flush(f"{'Clusters':<10} {'Final':<8} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'Complet':<8} {'Orthog':<8} {'SemOrth':<8}")
    print_and_flush(f"{'-'*10} {'-'*8} {'-'*10} {'-'*11} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    # Extract metrics for all cluster sizes from new format
    cluster_sizes = sorted([int(k) for k in results_by_cluster_size.keys()])
    optimal_n_clusters = int(best_cluster['size'])
    
    for n_clusters in cluster_sizes:
        cluster_results = results_by_cluster_size[str(n_clusters)]
        avg_final_score = cluster_results['avg_final_score']
        
        # Get average metrics across all repetitions
        all_repetitions = cluster_results['all_results']
        avg_accuracy = np.mean([rep['avg_accuracy'] for rep in all_repetitions])
        avg_precision = np.mean([rep['avg_precision'] for rep in all_repetitions])
        avg_recall = np.mean([rep['avg_recall'] for rep in all_repetitions])
        avg_f1 = np.mean([rep['avg_f1'] for rep in all_repetitions])
        avg_completeness = np.mean([rep['assigned_fraction'] for rep in all_repetitions])
        avg_orthogonality = np.mean([rep['orthogonality'] for rep in all_repetitions])
        avg_semantic_orthogonality = np.mean([rep['semantic_orthogonality_score'] for rep in all_repetitions])
        
        prefix = "* " if n_clusters == optimal_n_clusters else "  "
        print_and_flush(f"{prefix}{n_clusters:<8} "
                f"{avg_final_score:<8.4f} {avg_accuracy:<10.4f} {avg_precision:<11.4f} "
                f"{avg_recall:<8.4f} {avg_f1:<8.4f} "
                f"{avg_completeness:<8.4f} {avg_orthogonality:<8.4f} {avg_semantic_orthogonality:<8.4f}")

def main():
    if args.command == "submit":
        submit_evaluation_batches()
    elif args.command == "process":
        process_evaluation_batches()


if __name__ == "__main__":
    main() 