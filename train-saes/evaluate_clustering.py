# %%
import numpy as np
import argparse
import json
from tqdm import tqdm
from utils.clustering import (
    print_and_flush, load_clustering_model, predict_clusters, 
    compute_centroid_orthogonality, generate_test_data, 
    compute_silhouette_score, convert_numpy_types, SUPPORTED_CLUSTERING_METHODS
)

# %%

parser = argparse.ArgumentParser(description="Evaluate saved clustering models")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="Model to analyze")
parser.add_argument("--layer", type=int, default=12,
                    help="Layer to analyze")
parser.add_argument("--min_clusters", type=int, default=4,
                    help="Minimum number of clusters")
parser.add_argument("--max_clusters", type=int, default=20,
                    help="Maximum number of clusters")
parser.add_argument("--clustering_methods", type=str, nargs='+', 
                    default=list(SUPPORTED_CLUSTERING_METHODS),
                    help="Clustering methods to evaluate")
parser.add_argument("--silhouette_sample_size", type=int, default=50_000,
                    help="Number of samples to use for silhouette score calculation")
args, _ = parser.parse_known_args()

# %% Get model identifier for file naming
model_id = args.model.split('/')[-1].lower()

clustering_methods = [method for method in args.clustering_methods if method in SUPPORTED_CLUSTERING_METHODS]

# %%

def evaluate_saved_clustering_model(model_id, layer, n_clusters, method, test_data):
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
    test_data : numpy.ndarray
        Test data to evaluate on (should be normalized)
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    try:
        # Load the saved clustering model
        clustering_data = load_clustering_model(model_id, layer, n_clusters, method)
        
        # Predict cluster labels for test data
        cluster_labels = predict_clusters(test_data, clustering_data)
        
        # Compute silhouette score using utility function
        silhouette = compute_silhouette_score(test_data, cluster_labels, 
                                            sample_size=args.silhouette_sample_size, random_state=42)
        
        # Compute orthogonality
        cluster_centers = clustering_data['cluster_centers']
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
        
    except Exception as e:
        print_and_flush(f"Error evaluating {method} with {n_clusters} clusters: {e}")
        return None



def evaluate_clustering_method(model_id, layer, method, min_clusters, max_clusters):
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
        
    Returns:
    --------
    dict
        Dictionary containing evaluation results
    """
    print_and_flush(f"\nEvaluating {method.upper()} clustering method...")
    
    # Try to get input dimension from the first available model
    input_dim = None
    for n_clusters in range(min_clusters, max_clusters + 1):
        try:
            clustering_data = load_clustering_model(model_id, layer, n_clusters, method)
            input_dim = clustering_data['input_dim']
            break
        except FileNotFoundError:
            continue
    
    if input_dim is None:
        print_and_flush(f"No saved models found for {method}")
        return None
    
    # Generate test data
    test_data = generate_test_data(input_dim)
    
    # Evaluate across cluster range
    cluster_range = list(range(min_clusters, max_clusters + 1))
    silhouette_scores = []
    orthogonality_scores = []
    cluster_size_stats = []
    
    print_and_flush(f"Testing {len(cluster_range)} different cluster counts...")
    for n_clusters in tqdm(cluster_range, desc=f"{method.capitalize()} evaluation"):
        results = evaluate_saved_clustering_model(model_id, layer, n_clusters, method, test_data)
        
        if results is not None:
            silhouette_scores.append(results['silhouette'])
            orthogonality_scores.append(results['orthogonality'])
            cluster_size_stats.append(results['cluster_sizes'])
        else:
            silhouette_scores.append(0.0)
            orthogonality_scores.append(0.0)
            cluster_size_stats.append({})
    
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

# %% Main evaluation loop
print_and_flush("\n" + "="*50)
print_and_flush("EVALUATING SAVED CLUSTERING MODELS")
print_and_flush("="*50)

all_results = {}

for method in clustering_methods:
    try:
        results = evaluate_clustering_method(model_id, args.layer, method, args.min_clusters, args.max_clusters)
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