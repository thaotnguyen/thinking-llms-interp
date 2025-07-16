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
    
    # Use evaluate_clustering_scoring_metrics for comprehensive evaluation
    scoring_results = evaluate_clustering_scoring_metrics(
        texts, 
        cluster_labels, 
        n_clusters, 
        activations,
        cluster_centers,
        args.model,
        args.n_autograder_examples,
        args.description_examples,
    )
    
    # Calculate average F1 score, precision, and recall across all clusters
    f1_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_count = 0
    for cluster_id, metrics in scoring_results['detailed_results'].items():
        if metrics['f1'] > 0:  # Only count non-zero F1 scores
            f1_sum += metrics['f1']
            precision_sum += metrics['precision']
            recall_sum += metrics['recall']
            f1_count += 1
    avg_f1 = f1_sum / f1_count if f1_count > 0 else 0
    avg_precision = precision_sum / f1_count if f1_count > 0 else 0
    avg_recall = recall_sum / f1_count if f1_count > 0 else 0
    
    return {
        'accuracy': scoring_results['accuracy'],
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'assignment_rate': scoring_results['assigned_fraction'],
        'orthogonality': scoring_results['orthogonality'],
        'detailed_results': scoring_results,
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
    cluster_range = list(range(min_clusters, max_clusters + 1))
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    assignment_rates = []
    orthogonality_scores = []
    detailed_results_dict = {}

    print_and_flush(f"Testing {len(cluster_range)} different cluster counts...")
    for n_clusters in tqdm(cluster_range, desc=f"{method.capitalize()} evaluation"):
        results = re_evaluate_clustering_method_with_n_clusters(model_id, layer, n_clusters, method, activations, all_texts, re_compute_cluster_labels)

        accuracy_scores.append(results['accuracy'])
        precision_scores.append(results['precision'])
        recall_scores.append(results['recall'])
        f1_scores.append(results['f1'])
        assignment_rates.append(results['assignment_rate'])
        orthogonality_scores.append(results['orthogonality'])
        detailed_results_dict[n_clusters] = results['detailed_results']
    
    # Load existing JSON file and update it
    results_json_path = f'results/vars/{method}_results_{model_id}_layer{layer}.json'
    
    with open(results_json_path, 'r') as f:
        existing_results = json.load(f)
    print_and_flush(f"Loaded existing results from {results_json_path}")
    
    # Extract confidence scores from detailed results
    confidence_scores = []
    for n_clusters in cluster_range:
        detailed_result = detailed_results_dict[n_clusters]
        confidence_scores.append(detailed_result.get('avg_confidence', 0.0))
    
    # Calculate final scores (average of F1, confidence, and orthogonality)
    final_scores = [(f1 + conf + orth) / 3 for f1, conf, orth in 
                   zip(f1_scores, confidence_scores, orthogonality_scores)]

    # Find optimal number of clusters based on final score (same as ablate_clustering.py)
    optimal_n_clusters = cluster_range[np.argmax(final_scores)]
    
    # Update the existing results with new metrics
    optimal_idx = cluster_range.index(optimal_n_clusters)
    existing_results.update({
        'accuracy_scores': accuracy_scores,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'f1_scores': f1_scores,
        'assignment_rates': assignment_rates,
        'confidence_scores': confidence_scores,
        'orthogonality_scores': orthogonality_scores,
        'final_scores': final_scores,
        'optimal_n_clusters': optimal_n_clusters,
        'optimal_accuracy': accuracy_scores[optimal_idx],
        'optimal_precision': precision_scores[optimal_idx],
        'optimal_recall': recall_scores[optimal_idx],
        'optimal_f1': f1_scores[optimal_idx],
        'optimal_assignment_rate': assignment_rates[optimal_idx],
        'optimal_confidence': confidence_scores[optimal_idx],
        'optimal_orthogonality': orthogonality_scores[optimal_idx],
        'optimal_final_score': final_scores[optimal_idx],
        'detailed_results': detailed_results_dict
    })
    
    # Convert numpy types and save updated results
    existing_results = convert_numpy_types(existing_results)
    
    with open(results_json_path, 'w') as f:
        json.dump(existing_results, f, indent=2)
    print_and_flush(f"Updated results saved to {results_json_path}")
    
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
    
    print_and_flush("\nMetrics for all cluster sizes:")
    print_and_flush(f"{'Clusters':<10} {'Final':<8} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'Assign%':<8} {'Confid':<8} {'Orthog':<8}")
    print_and_flush(f"{'-'*10} {'-'*8} {'-'*10} {'-'*11} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    cluster_range = results['cluster_range']
    confidence_scores = results.get('confidence_scores', [0.0] * len(cluster_range))
    final_scores = results.get('final_scores', [0.0] * len(cluster_range))
    for i, n_clusters in enumerate(cluster_range):
        prefix = "* " if n_clusters == results['optimal_n_clusters'] else "  "
        print_and_flush(f"{prefix}{n_clusters:<8} "
                f"{final_scores[i]:<8.4f} {results['accuracy_scores'][i]:<10.4f} {results['precision_scores'][i]:<11.4f} "
                f"{results['recall_scores'][i]:<8.4f} {results['f1_scores'][i]:<8.4f} "
                f"{results['assignment_rates'][i]:<8.4f} {confidence_scores[i]:<8.4f} {results['orthogonality_scores'][i]:<8.4f}")

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
    print_and_flush(f"{'Method':<20} {'Optimal K':<10} {'Final':<8} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'Assign%':<8} {'Confid':<8} {'Orthog':<8}")
    print_and_flush(f"{'-'*20} {'-'*10} {'-'*8} {'-'*10} {'-'*11} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for method, results in all_results.items():
        print_and_flush(f"{method.capitalize():<20} {results['optimal_n_clusters']:<10} "
              f"{results.get('optimal_final_score', 0.0):<8.4f} {results['optimal_accuracy']:<10.4f} "
              f"{results['optimal_precision']:<11.4f} {results['optimal_recall']:<8.4f} "
              f"{results['optimal_f1']:<8.4f} {results['optimal_assignment_rate']:<8.4f} "
              f"{results.get('optimal_confidence', 0.0):<8.4f} {results['optimal_orthogonality']:<8.4f}")

print_and_flush("\nEvaluation complete!") 