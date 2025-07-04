# %%
import numpy as np
import torch
import argparse
import json
from tqdm import tqdm
from utils import utils
import gc
from utils.utils import print_and_flush
from utils.clustering import (
    evaluate_clustering
)
from utils.clustering_methods import CLUSTERING_METHODS

# %%

parser = argparse.ArgumentParser(description="K-means clustering and autograding of neural activations")
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
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--n_autograder_examples", type=int, default=100,
                    help="Number of examples from each cluster to use for autograding")
parser.add_argument("--description_examples", type=int, default=50,
                    help="Number of examples to use for generating cluster descriptions")
parser.add_argument("--clustering_methods", type=str, nargs='+', 
                    default=["gmm", "pca_gmm", "spherical_kmeans", "pca_kmeans", "agglomerative", "pca_agglomerative", "sae_topk"],
                    help="Clustering methods to use")
parser.add_argument("--clustering_pilot_size", type=int, default=50_000,
                    help="Number of samples to use for pilot fitting with GMM")
parser.add_argument("--clustering_pilot_n_init", type=int, default=10,
                    help="Number of initializations for pilot fitting with GMM")
parser.add_argument("--clustering_pilot_max_iter", type=int, default=100,
                    help="Maximum iterations for pilot fitting with GMM")
parser.add_argument("--clustering_full_n_init", type=int, default=1,
                    help="Number of initializations for full fitting with GMM")
parser.add_argument("--clustering_full_max_iter", type=int, default=100,
                    help="Maximum iterations for full fitting with GMM")
parser.add_argument("--silhouette_sample_size", type=int, default=50_000,
                    help="Number of samples to use for silhouette score calculation")
args, _ = parser.parse_known_args()

# %%

def run_clustering_experiment(clustering_method, clustering_func, all_texts, activations, args, model_id=None):
    """
    Run a clustering experiment using the specified clustering method.
    
    Parameters:
    -----------
    clustering_method : str
        Name of the clustering method
    clustering_func : function
        Function that implements the clustering algorithm
    all_texts : list
        List of texts to cluster
    activations : numpy.ndarray
        Normalized activation vectors
    args : argparse.Namespace
        Command line arguments
    model_id : str
        Model identifier for file naming
        
    Returns:
    --------
    dict
        Results of the clustering experiment
    """
    print_and_flush(f"\nRunning {clustering_method.upper()} clustering experiment...")
    
    # For methods that require n_clusters, use the original code
    silhouette_scores = []
    accuracy_scores = []
    f1_scores = []
    assignment_rates = []
    orthogonality_scores = []  # Add orthogonality scores
    precision_scores = []  # Add precision scores
    recall_scores = []  # Add recall scores
    detailed_results_dict = {}
    
    cluster_range = list(range(args.min_clusters, args.max_clusters + 1))
    
    print_and_flush(f"Testing {len(cluster_range)} different cluster counts...")
    for n_clusters in tqdm(cluster_range, desc=f"{clustering_method.capitalize()} progress"):
        # Perform clustering
        cluster_labels, cluster_centers, silhouette = clustering_func(activations, n_clusters, args)
        
        # Evaluate clustering
        evaluation_results = evaluate_clustering(
            all_texts, 
            cluster_labels, 
            n_clusters, 
            activations,
            cluster_centers,
            args.model,
            args.n_autograder_examples,
            args.description_examples,
        )
        
        # Store metrics
        silhouette_scores.append(silhouette)
        accuracy_scores.append(evaluation_results['accuracy'])
        orthogonality_scores.append(evaluation_results['orthogonality'])
        
        # Calculate average F1 score, precision, and recall across all clusters
        f1_sum = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        f1_count = 0
        for cluster_id, metrics in evaluation_results['detailed_results'].items():
            if metrics['f1'] > 0:  # Only count non-zero F1 scores
                f1_sum += metrics['f1']
                precision_sum += metrics['precision']
                recall_sum += metrics['recall']
                f1_count += 1
        avg_f1 = f1_sum / f1_count if f1_count > 0 else 0
        avg_precision = precision_sum / f1_count if f1_count > 0 else 0
        avg_recall = recall_sum / f1_count if f1_count > 0 else 0
        
        f1_scores.append(avg_f1)
        precision_scores.append(avg_precision)
        recall_scores.append(avg_recall)
        
        # Store assignment rate if completeness was run
        assignment_rates.append(evaluation_results.get('assigned_fraction', 0))
        
        # Store detailed results
        detailed_results_dict[n_clusters] = evaluation_results

    # Identify optimal number of clusters based on accuracy only
    optimal_n_clusters = cluster_range[np.argmax(accuracy_scores)]

    # Create a concise results JSON
    results_data = {
        "clustering_method": clustering_method,
        "model_id": model_id,
        "layer": args.layer,
        "cluster_range": cluster_range,
        "silhouette_scores": silhouette_scores,
        "accuracy_scores": accuracy_scores,
        "precision_scores": precision_scores,
        "recall_scores": recall_scores,
        "f1_scores": f1_scores,
        "assignment_rates": assignment_rates,
        "orthogonality_scores": orthogonality_scores,
        "optimal_n_clusters": optimal_n_clusters,
        "optimal_silhouette": silhouette_scores[cluster_range.index(optimal_n_clusters)],
        "optimal_accuracy": accuracy_scores[cluster_range.index(optimal_n_clusters)],
        "optimal_precision": precision_scores[cluster_range.index(optimal_n_clusters)],
        "optimal_recall": recall_scores[cluster_range.index(optimal_n_clusters)],
        "optimal_f1": f1_scores[cluster_range.index(optimal_n_clusters)],
        "optimal_assignment_rate": assignment_rates[cluster_range.index(optimal_n_clusters)],
        "optimal_orthogonality": orthogonality_scores[cluster_range.index(optimal_n_clusters)],
        "detailed_results": detailed_results_dict
    }

    # Convert any numpy types to Python native types for JSON serialization
    results_data = utils.convert_numpy_types(results_data)
    
    # Save results to JSON
    results_json_path = f'results/vars/{clustering_method}_results_{model_id}_layer{args.layer}.json'
    with open(results_json_path, 'w') as f:
        json.dump(results_data, f, indent=2, cls=utils.NumpyEncoder)
    print_and_flush(f"Saved {clustering_method} results to {results_json_path}")
    
    return results_data


# %% Load model and process activations
print_and_flush("Loading model and processing activations...")
model, tokenizer = utils.load_model(
    model_name=args.model,
    load_in_8bit=args.load_in_8bit
)

# %% Get model identifier for file naming
model_id = args.model.split('/')[-1].lower()

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

# %% Filter clustering methods based on args
clustering_methods = [method for method in args.clustering_methods if method in CLUSTERING_METHODS]

# Run each clustering method
current_results = {}
for method in clustering_methods:
    try:
        clustering_func = CLUSTERING_METHODS[method]
        results = run_clustering_experiment(method, clustering_func, all_texts, all_activations, args, model_id)
        current_results[method] = results
    except Exception as e:
        print_and_flush(f"Error running {method}: {e}")
