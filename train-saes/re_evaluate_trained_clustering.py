# %%
import numpy as np
import argparse
import json
from tqdm import tqdm
import torch
import gc
from utils.utils import print_and_flush
from utils.clustering import (
    convert_numpy_types, SUPPORTED_CLUSTERING_METHODS,
    load_trained_clustering_data, predict_clusters, evaluate_clustering_scoring_metrics
)
from utils import utils

# %%

parser = argparse.ArgumentParser(description="Evaluate saved clustering models")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="Model to analyze")
parser.add_argument("--layer", type=int, default=12,
                    help="Layer to analyze")
parser.add_argument("--n_examples", type=int, default=500,
                    help="Number of examples to analyze")
parser.add_argument("--min_clusters", type=int, default=4,
                    help="Minimum number of clusters")
parser.add_argument("--max_clusters", type=int, default=20,
                    help="Maximum number of clusters")
parser.add_argument("--clustering_methods", type=str, nargs='+', 
                    default=list(SUPPORTED_CLUSTERING_METHODS),
                    help="Clustering methods to evaluate")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--re_compute_cluster_labels", action="store_true", default=False,
                    help="Re-compute cluster labels and centers, and save them to the existing file")
parser.add_argument("--n_autograder_examples", type=int, default=100,
                    help="Number of examples from each cluster to use for autograding")
parser.add_argument("--description_examples", type=int, default=200,
                    help="Number of examples to use for generating cluster descriptions")
args, _ = parser.parse_known_args()

# %% Get model identifier for file naming
model_id = args.model.split('/')[-1].lower()

clustering_methods = [method for method in args.clustering_methods if method in SUPPORTED_CLUSTERING_METHODS]

# %%

def re_evaluate_clustering_method_with_n_clusters(model_id, layer, n_clusters, method, activations, texts, re_compute_cluster_labels):
    """
    Evaluate a saved clustering model using comprehensive scoring metrics.
    
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
    activations : numpy.ndarray
        Test data to evaluate on (should be normalized)
    texts : list
        List of texts corresponding to the activations
    re_compute_cluster_labels : bool
        Whether to recompute cluster labels
        
    Returns:
    --------
    dict
        Dictionary containing comprehensive evaluation metrics
    """
    # Load the saved clustering model
    clustering_data = load_trained_clustering_data(model_id, layer, n_clusters, method)
    cluster_centers = clustering_data['cluster_centers']
    
    if re_compute_cluster_labels:
        # Predict cluster labels with new activations
        cluster_labels = predict_clusters(activations, clustering_data)
        clustering_data['cluster_labels'] = cluster_labels
    else:
        cluster_labels = clustering_data['cluster_labels']
    
    # Use evaluate_clustering_scoring_metrics for comprehensive evaluation with repetitions
    scoring_results = evaluate_clustering_scoring_metrics(
        texts, 
        cluster_labels, 
        n_clusters, 
        activations,
        cluster_centers,
        args.model,
        args.n_autograder_examples,
        args.description_examples,
        repetitions=5  # Use 5 repetitions for evaluation
    )
    
    # Extract the best repetition (highest final score) for return values
    best_repetition = max(scoring_results["all_results"], key=lambda x: x["final_score"])
    
    return {
        'avg_final_score': scoring_results['avg_final_score'],
        'best_final_score': best_repetition['final_score'],
        'best_repetition': best_repetition,
        'all_repetitions': scoring_results['all_results'],
        'n_repetitions': len(scoring_results['all_results']),
        'n_clusters': n_clusters,
        'method': method
    }


def re_evaluate_clustering_method(model_id, layer, method, min_clusters, max_clusters, activations, all_texts, re_compute_cluster_labels):
    """
    Evaluate a clustering method across different numbers of clusters and update the existing JSON file.
    
    Parameters:
    -----------
    model_id : str
        Model identifier
    layer : int
        Layer number
    method : str
        Clustering method name
    min_clusters : int
        Minimum number of clusters
    max_clusters : int
        Maximum number of clusters
    activations : numpy.ndarray
        Activations to evaluate on (should be normalized)
    all_texts : list
        List of texts corresponding to the activations
    re_compute_cluster_labels : bool
        Whether to recompute cluster labels
        
    Returns:
    --------
    dict
        Dictionary containing evaluation results
    """
    print_and_flush(f"\nEvaluating {method.upper()} clustering method...")
    
    # Load existing JSON file
    results_json_path = f'results/vars/{method}_results_{model_id}_layer{layer}.json'
    
    with open(results_json_path, 'r') as f:
        existing_results = json.load(f)
    print_and_flush(f"Loaded existing results from {results_json_path}")
    
    # Get existing detailed results or create new dict
    existing_detailed_results = existing_results.get("detailed_results", {})
    
    cluster_range = list(range(min_clusters, max_clusters + 1))
    print_and_flush(f"Testing {len(cluster_range)} different cluster counts...")
    
    # Process each cluster count
    for n_clusters in tqdm(cluster_range, desc=f"{method.capitalize()} evaluation"):
        print_and_flush(f"Processing {n_clusters} clusters...")
        
        # Evaluate this cluster count
        results = re_evaluate_clustering_method_with_n_clusters(
            model_id, layer, n_clusters, method, activations, all_texts, re_compute_cluster_labels
        )
        
        # Store results for this cluster count in the new format
        existing_detailed_results[str(n_clusters)] = {
            "avg_final_score": results['avg_final_score'],
            "best_final_score": results['best_final_score'],
            "best_repetition": results['best_repetition'],
            "all_repetitions": results['all_repetitions'],
            "n_repetitions": results['n_repetitions']
        }
    
    # Calculate summary metrics across all cluster counts
    if existing_detailed_results:
        # Extract key metrics for easy access
        cluster_counts = sorted([int(k) for k in existing_detailed_results.keys()])
        avg_final_scores = [existing_detailed_results[str(n)]["avg_final_score"] for n in cluster_counts]
        best_final_scores = [existing_detailed_results[str(n)]["best_final_score"] for n in cluster_counts]
        
        # Find optimal cluster count based on average final score
        optimal_n_clusters = cluster_counts[np.argmax(avg_final_scores)]
        optimal_idx = cluster_counts.index(optimal_n_clusters)
        
        # Get metrics from the best repetition of the optimal cluster count
        optimal_best_rep = existing_detailed_results[str(optimal_n_clusters)]["best_repetition"]
        
        # Extract metrics from all cluster counts for backward compatibility
        accuracy_scores = [existing_detailed_results[str(n)]["best_repetition"]["avg_accuracy"] for n in cluster_counts]
        precision_scores = [existing_detailed_results[str(n)]["best_repetition"]["avg_precision"] for n in cluster_counts]
        recall_scores = [existing_detailed_results[str(n)]["best_repetition"]["avg_recall"] for n in cluster_counts]
        f1_scores = [existing_detailed_results[str(n)]["best_repetition"]["avg_f1"] for n in cluster_counts]
        assignment_rates = [existing_detailed_results[str(n)]["best_repetition"]["assigned_fraction"] for n in cluster_counts]
        confidence_scores = [existing_detailed_results[str(n)]["best_repetition"]["avg_confidence"] for n in cluster_counts]
        orthogonality_scores = [existing_detailed_results[str(n)]["best_repetition"]["orthogonality"] for n in cluster_counts]
        semantic_orthogonality_scores = [existing_detailed_results[str(n)]["best_repetition"]["semantic_orthogonality_score"] for n in cluster_counts]
        
        # Update the existing results with summary metrics for backward compatibility
        existing_results.update({
            'cluster_range': cluster_counts,
            'avg_final_scores': avg_final_scores,
            'best_final_scores': best_final_scores,
            'accuracy_scores': accuracy_scores,
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'f1_scores': f1_scores,
            'assignment_rates': assignment_rates,
            'confidence_scores': confidence_scores,
            'orthogonality_scores': orthogonality_scores,
            'semantic_orthogonality_scores': semantic_orthogonality_scores,
            'final_scores': best_final_scores,  # Use best scores as final scores
            'optimal_n_clusters': optimal_n_clusters,
            'optimal_avg_final_score': avg_final_scores[optimal_idx],
            'optimal_best_final_score': best_final_scores[optimal_idx],
            # Extract key metrics from optimal best repetition
            'optimal_accuracy': optimal_best_rep["avg_accuracy"],
            'optimal_precision': optimal_best_rep["avg_precision"],
            'optimal_recall': optimal_best_rep["avg_recall"],
            'optimal_f1': optimal_best_rep["avg_f1"],
            'optimal_assignment_rate': optimal_best_rep["assigned_fraction"],
            'optimal_confidence': optimal_best_rep["avg_confidence"],
            'optimal_orthogonality': optimal_best_rep["orthogonality"],
            'optimal_semantic_orthogonality': optimal_best_rep["semantic_orthogonality_score"],
            'optimal_final_score': best_final_scores[optimal_idx],
            'detailed_results': existing_detailed_results
        })
        
        # Convert numpy types and save updated results
        existing_results = convert_numpy_types(existing_results)
        
        with open(results_json_path, 'w') as f:
            json.dump(existing_results, f, indent=2)
        print_and_flush(f"Updated results saved to {results_json_path}")
        
        return existing_results
    else:
        print_and_flush("No clustering results to process.")
        return existing_results


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
    print_and_flush(f"Optimal clusters: {results['optimal_n_clusters']}")
    print_and_flush(f"Optimal final score: {results.get('optimal_final_score', 0.0):.4f}")
    print_and_flush(f"Optimal accuracy: {results['optimal_accuracy']:.4f}")
    print_and_flush(f"Optimal precision: {results['optimal_precision']:.4f}")
    print_and_flush(f"Optimal recall: {results['optimal_recall']:.4f}")
    print_and_flush(f"Optimal F1: {results['optimal_f1']:.4f}")
    print_and_flush(f"Optimal assignment rate: {results['optimal_assignment_rate']:.4f}")
    print_and_flush(f"Optimal confidence: {results.get('optimal_confidence', 0.0):.4f}")
    print_and_flush(f"Optimal orthogonality: {results['optimal_orthogonality']:.4f}")
    print_and_flush(f"Optimal semantic orthogonality: {results['optimal_semantic_orthogonality']:.4f}")
    
    print_and_flush("\nMetrics for all cluster sizes:")
    print_and_flush(f"{'Clusters':<10} {'Final':<8} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'Assign%':<8} {'Confid':<8} {'Orthog':<8} {'SemOrth':<8}")
    print_and_flush(f"{'-'*10} {'-'*8} {'-'*10} {'-'*11} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    cluster_range = results['cluster_range']
    confidence_scores = results['confidence_scores']
    final_scores = results['final_scores']
    semantic_orthogonality_scores = results['semantic_orthogonality_scores']
    for i, n_clusters in enumerate(cluster_range):
        prefix = "* " if n_clusters == results['optimal_n_clusters'] else "  "
        print_and_flush(f"{prefix}{n_clusters:<8} "
                f"{final_scores[i]:<8.4f} {results['accuracy_scores'][i]:<10.4f} {results['precision_scores'][i]:<11.4f} "
                f"{results['recall_scores'][i]:<8.4f} {results['f1_scores'][i]:<8.4f} "
                f"{results['assignment_rates'][i]:<8.4f} {confidence_scores[i]:<8.4f} {results['orthogonality_scores'][i]:<8.4f} {semantic_orthogonality_scores[i]:<8.4f}")

# %% Load model and process activations
print_and_flush("Loading model and processing activations...")
model, tokenizer = utils.load_model(
    model_name=args.model,
    load_in_8bit=args.load_in_8bit
)

# %% Process saved responses
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

# %% Center activations
all_activations = [x - overall_mean for x in all_activations]
all_activations = np.stack([a.reshape(-1) for a in all_activations])
norms = np.linalg.norm(all_activations, axis=1, keepdims=True)
all_activations = all_activations / norms

# %% Main evaluation loop
print_and_flush("\n" + "="*50)
print_and_flush("EVALUATING SAVED CLUSTERING MODELS")
print_and_flush("="*50)

all_results = {}

for method in clustering_methods:
    results = re_evaluate_clustering_method(model_id, args.layer, method, args.min_clusters, args.max_clusters, all_activations, all_texts, args.re_compute_cluster_labels)
    all_results[method] = results
    print_evaluation_summary(results, method)

# %% Print overall comparison
if len(all_results) > 1:
    print_and_flush("\n" + "="*50)
    print_and_flush("OVERALL COMPARISON")
    print_and_flush("="*50)
    print_and_flush(f"{'Method':<20} {'Optimal K':<10} {'Final':<8} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'Assign%':<8} {'Confid':<8} {'Orthog':<8} {'SemOrth':<8}")
    print_and_flush(f"{'-'*20} {'-'*10} {'-'*8} {'-'*10} {'-'*11} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for method, results in all_results.items():
        print_and_flush(f"{method.capitalize():<20} {results['optimal_n_clusters']:<10} "
              f"{results.get('optimal_final_score', 0.0):<8.4f} {results['optimal_accuracy']:<10.4f} "
              f"{results['optimal_precision']:<11.4f} {results['optimal_recall']:<8.4f} "
              f"{results['optimal_f1']:<8.4f} {results['optimal_assignment_rate']:<8.4f} "
              f"{results.get('optimal_confidence', 0.0):<8.4f} {results['optimal_orthogonality']:<8.4f} {results.get('optimal_semantic_orthogonality', 0.0):<8.4f}")

print_and_flush("\nEvaluation complete!") 