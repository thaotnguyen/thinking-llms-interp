# %%
import numpy as np
import argparse
import json
from tqdm import tqdm
import torch
import gc
from utils.utils import print_and_flush
from utils.clustering import (
    compute_centroid_orthogonality, 
    compute_silhouette_score, convert_numpy_types, SUPPORTED_CLUSTERING_METHODS,
    load_trained_clustering_data, predict_clusters
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
parser.add_argument("--silhouette_sample_size", type=int, default=100_000,
                    help="Number of samples to use for silhouette score calculation")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--re_compute_cluster_labels", action="store_true", default=False,
                    help="Re-compute cluster labels and centers, and save them to the existing file")
args, _ = parser.parse_known_args()

# %% Get model identifier for file naming
model_id = args.model.split('/')[-1].lower()

clustering_methods = [method for method in args.clustering_methods if method in SUPPORTED_CLUSTERING_METHODS]

# %%

def re_evaluate_clustering_method_with_n_clusters(model_id, layer, n_clusters, method, activations, re_compute_cluster_labels):
    """
    Evaluate a saved clustering model by computing silhouette score and orthogonality.
    
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
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
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
    
    # Compute silhouette score using utility function
    silhouette = compute_silhouette_score(activations, cluster_labels, 
                                        sample_size=args.silhouette_sample_size, random_state=42)
    
    # Compute orthogonality
    orthogonality = compute_centroid_orthogonality(cluster_centers)
    
    # Compute cluster sizes
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))
    
    return {
        'silhouette': silhouette,
        'orthogonality': orthogonality,
        'cluster_sizes': cluster_sizes,
        'n_clusters': n_clusters,
        'method': method
    }


def re_evaluate_clustering_method(model_id, layer, method, min_clusters, max_clusters, activations, re_compute_cluster_labels):
    """
    Evaluate a clustering method across different numbers of clusters.
    
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
        
    Returns:
    --------
    dict
        Dictionary containing evaluation results
    """
    print_and_flush(f"\nEvaluating {method.upper()} clustering method...")
    cluster_range = list(range(min_clusters, max_clusters + 1))
    silhouette_scores = []
    orthogonality_scores = []
    cluster_size_stats = []

    print_and_flush(f"Testing {len(cluster_range)} different cluster counts...")
    for n_clusters in tqdm(cluster_range, desc=f"{method.capitalize()} evaluation"):
        results = re_evaluate_clustering_method_with_n_clusters(model_id, layer, n_clusters, method, activations, re_compute_cluster_labels)

        silhouette_scores.append(results['silhouette'])
        orthogonality_scores.append(results['orthogonality'])
        cluster_size_stats.append(results['cluster_sizes'])
    
    # Find optimal number of clusters based on silhouette score
    if len(silhouette_scores) > 0 and max(silhouette_scores) > 0:
        optimal_idx = np.argmax(silhouette_scores)
        optimal_n_clusters = cluster_range[optimal_idx]
    else:
        optimal_n_clusters = min_clusters
    
    return {
        'method': method,
        'model_id': model_id,
        'layer': layer,
        'cluster_range': cluster_range,
        'silhouette_scores': silhouette_scores,
        'orthogonality_scores': orthogonality_scores,
        'cluster_size_stats': cluster_size_stats,
        'optimal_n_clusters': optimal_n_clusters,
        'optimal_silhouette': silhouette_scores[cluster_range.index(optimal_n_clusters)] if optimal_n_clusters in cluster_range else 0.0,
        'optimal_orthogonality': orthogonality_scores[cluster_range.index(optimal_n_clusters)] if optimal_n_clusters in cluster_range else 0.0
    }

def save_evaluation_results(results, method):
    """
    Save evaluation results to JSON file.
    
    Parameters:
    -----------
    results : dict
        Evaluation results
    method : str
        Clustering method name
    """
    if results is None:
        return
    
    # Convert numpy types to Python native types for JSON serialization
    results = convert_numpy_types(results)
    
    # Save results to JSON
    results_json_path = f'results/vars/{method}_evaluation_{model_id}_layer{args.layer}.json'
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_and_flush(f"Saved {method} evaluation results to {results_json_path}")

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
    if results is None:
        return
    
    print_and_flush("\n" + "="*50)
    print_and_flush(f"{method.upper()} EVALUATION SUMMARY")
    print_and_flush("="*50)
    print_and_flush(f"Model: {model_id.upper()}, Layer: {args.layer}")
    print_and_flush(f"Optimal clusters: {results['optimal_n_clusters']}")
    print_and_flush(f"Optimal silhouette: {results['optimal_silhouette']:.4f}")
    print_and_flush(f"Optimal orthogonality: {results['optimal_orthogonality']:.4f}")
    
    print_and_flush("\nMetrics for all cluster sizes:")
    print_and_flush(f"{'Clusters':<10} {'Silhouette':<12} {'Orthogonality':<15}")
    print_and_flush(f"{'-'*10} {'-'*12} {'-'*15}")
    
    for i, n_clusters in enumerate(results['cluster_range']):
        prefix = "* " if n_clusters == results['optimal_n_clusters'] else "  "
        print_and_flush(f"{prefix}{n_clusters:<8} {results['silhouette_scores'][i]:<12.4f} "
                f"{results['orthogonality_scores'][i]:<15.4f}")

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
all_activations = np.stack([a.cpu().numpy().reshape(-1) for a in all_activations])
norms = np.linalg.norm(all_activations, axis=1, keepdims=True)
all_activations = all_activations / norms

# %% Main evaluation loop
print_and_flush("\n" + "="*50)
print_and_flush("EVALUATING SAVED CLUSTERING MODELS")
print_and_flush("="*50)

all_results = {}

for method in clustering_methods:
    try:
        results = re_evaluate_clustering_method(model_id, args.layer, method, args.min_clusters, args.max_clusters, all_activations, args.re_compute_cluster_labels)
        if results is not None:
            all_results[method] = results
            save_evaluation_results(results, method)
            print_evaluation_summary(results, method)
    except Exception as e:
        print_and_flush(f"Error evaluating {method}: {e}")

# %% Print overall comparison
if len(all_results) > 1:
    print_and_flush("\n" + "="*50)
    print_and_flush("OVERALL COMPARISON")
    print_and_flush("="*50)
    print_and_flush(f"{'Method':<20} {'Optimal K':<10} {'Silhouette':<12} {'Orthogonality':<15}")
    print_and_flush(f"{'-'*20} {'-'*10} {'-'*12} {'-'*15}")
    
    for method, results in all_results.items():
        print_and_flush(f"{method.capitalize():<20} {results['optimal_n_clusters']:<10} "
              f"{results['optimal_silhouette']:<12.4f} {results['optimal_orthogonality']:<15.4f}")

print_and_flush("\nEvaluation complete!") 