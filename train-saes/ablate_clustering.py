# %%
import numpy as np
import torch
import argparse
import json
import os
from tqdm import tqdm
from utils import utils
import gc
from utils.utils import print_and_flush
from utils.clustering import (
    evaluate_clustering_scoring_metrics
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
parser.add_argument("--description_examples", type=int, default=200,
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
    
    # Define results path and load existing data
    results_json_path = f'results/vars/{clustering_method}_results_{model_id}_layer{args.layer}.json'
    existing_results_data = {}
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, 'r') as f:
                existing_results_data = json.load(f)
            print_and_flush(f"Loaded existing results from {results_json_path}")
        except json.JSONDecodeError:
            print_and_flush(f"Warning: Could not decode JSON from {results_json_path}. Starting fresh.")
    
    # Get existing detailed results if any
    existing_detailed_results = existing_results_data.get("detailed_results", {})
    
    # Define cluster range to test
    cluster_range = [10, 20, 30, 40, 50]
    
    print_and_flush(f"Testing {len(cluster_range)} different cluster counts...")
    
    # Process each cluster count
    for n_clusters in tqdm(cluster_range, desc=f"{clustering_method.capitalize()} progress"):
        # Skip if we already have results for this cluster count
        if str(n_clusters) in existing_detailed_results:
            print_and_flush(f"Skipping n_clusters={n_clusters} (already have results)")
            continue
            
        print_and_flush(f"Processing {n_clusters} clusters...")
        
        # Perform clustering
        cluster_labels, cluster_centers = clustering_func(activations, n_clusters, args)
        
        # Evaluate clustering with repetitions
        scoring_results = evaluate_clustering_scoring_metrics(
            all_texts, 
            cluster_labels, 
            n_clusters, 
            activations,
            cluster_centers,
            args.model,
            args.n_autograder_examples,
            args.description_examples,
            repetitions=5  # Use 5 repetitions for robust evaluation
        )
        
        # Extract the best repetition (highest final score) for detailed storage
        best_repetition = max(scoring_results["all_results"], key=lambda x: x["final_score"])
        
        # Store results for this cluster count
        existing_detailed_results[str(n_clusters)] = {
            "avg_final_score": scoring_results["avg_final_score"],
            "best_final_score": best_repetition["final_score"],
            "best_repetition": best_repetition,
            "all_repetitions": scoring_results["all_results"],
            "n_repetitions": len(scoring_results["all_results"])
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
        
        # Create summary results
        results_data = {
            "clustering_method": clustering_method,
            "model_id": model_id,
            "layer": args.layer,
            "cluster_range": cluster_counts,
            "avg_final_scores": avg_final_scores,
            "best_final_scores": best_final_scores,
            "optimal_n_clusters": optimal_n_clusters,
            "optimal_avg_final_score": avg_final_scores[optimal_idx],
            "optimal_best_final_score": best_final_scores[optimal_idx],
            # Extract key metrics from optimal best repetition
            "optimal_accuracy": optimal_best_rep["avg_accuracy"],
            "optimal_precision": optimal_best_rep["avg_precision"],
            "optimal_recall": optimal_best_rep["avg_recall"],
            "optimal_f1": optimal_best_rep["avg_f1"],
            "optimal_orthogonality": optimal_best_rep["orthogonality"],
            "optimal_semantic_similarity": optimal_best_rep["avg_semantic_similarity"],
            "optimal_semantic_orthogonality": optimal_best_rep["avg_semantic_orthogonality"],
            "optimal_assigned_fraction": optimal_best_rep["assigned_fraction"],
            "optimal_confidence": optimal_best_rep["avg_confidence"],
            "detailed_results": existing_detailed_results
        }
        
        # Convert any numpy types to Python native types for JSON serialization
        results_data = utils.convert_numpy_types(results_data)
        
        # Save results to JSON
        with open(results_json_path, 'w') as f:
            json.dump(results_data, f, indent=2, cls=utils.NumpyEncoder)
        print_and_flush(f"Saved {clustering_method} results to {results_json_path}")
        
        return results_data
    else:
        print_and_flush("No clustering results to process.")
        return {}


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
all_activations = np.stack([a.reshape(-1) for a in all_activations])
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
        import traceback
        print(traceback.format_exc())
