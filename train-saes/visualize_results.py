# %%
import sys
import os
import matplotlib.pyplot as plt
import argparse
import json
import numpy as np

# %%
def print_and_flush(message):
    """Prints a message and flushes stdout."""
    print(message)
    sys.stdout.flush()

parser = argparse.ArgumentParser(description="K-means clustering and autograding of neural activations")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="Model to analyze")
parser.add_argument("--layer", type=int, default=12,
                    help="Layer to analyze")
parser.add_argument("--min_clusters", type=int, default=4,
                    help="Minimum number of clusters")
parser.add_argument("--max_clusters", type=int, default=20,
                    help="Maximum number of clusters")
parser.add_argument("--clustering_methods", type=str, nargs='+', 
                    default=["gmm", "pca_gmm", "spherical_kmeans", "pca_kmeans", "agglomerative", "pca_agglomerative", "sae_topk"],
                    help="Clustering methods to use")
args, _ = parser.parse_known_args()

# %% Get model identifier for file naming
model_id = args.model.split('/')[-1].lower()

CLUSTERING_METHODS = {
    # 'agglomerative',
    # 'pca_agglomerative',
    # 'gmm',
    # 'pca_gmm',
    # 'spherical_kmeans',
    # 'pca_kmeans',
    'sae_topk'
}

clustering_methods = [method for method in args.clustering_methods if method in CLUSTERING_METHODS]

# %%

def visualize_results(results_json_path):
    """
    Create a comprehensive visualization of clustering results from the results JSON.
    Shows median values with uncertainty bands (min/max) across repetitions.
    
    Parameters:
    -----------
    results_json_path : str
        Path to the JSON file containing the clustering results
    """
    # Load results from JSON
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    # Extract basic info
    model_id = results['model_id']
    layer = results['layer']
    method = results['clustering_method']
    detailed_results = results['detailed_results']
    
    # Extract cluster range from available data
    available_clusters = sorted([int(k) for k in detailed_results.keys()])
    print(f"Cluster range available: {available_clusters}")
    
    # Filter to desired range
    cluster_range_to_keep = [10, 20, 30, 40, 50]
    cluster_range = [c for c in cluster_range_to_keep if c in available_clusters]
    
    if not cluster_range:
        print("No data available for the desired cluster range")
        return
    
    # Initialize lists for metrics
    metrics = {
        'final_scores': {'median': [], 'min': [], 'max': []},
        'f1_scores': {'median': [], 'min': [], 'max': []},
        'accuracy_scores': {'median': [], 'min': [], 'max': []},
        'confidence_scores': {'median': [], 'min': [], 'max': []},
        'orthogonality_scores': {'median': [], 'min': [], 'max': []},
        'semantic_similarity_scores': {'median': [], 'min': [], 'max': []}
    }
    
    # Extract data for each cluster count
    for n_clusters in cluster_range:
        cluster_data = detailed_results[str(n_clusters)]
        
        if 'all_repetitions' in cluster_data:
            # New structure with repetitions
            repetitions = cluster_data['all_repetitions']
            
            # Extract metrics from each repetition
            rep_final_scores = [rep['final_score'] for rep in repetitions]
            rep_f1_scores = [rep['avg_f1'] for rep in repetitions]
            rep_accuracy_scores = [rep['avg_accuracy'] for rep in repetitions]
            rep_confidence_scores = [rep['avg_confidence'] for rep in repetitions]
            rep_orthogonality_scores = [rep['orthogonality'] for rep in repetitions]
            rep_semantic_similarity_scores = [rep['avg_semantic_similarity'] for rep in repetitions]
            
        else:
            # Old structure fallback - treat as single "repetition"
            rep_final_scores = [cluster_data.get('final_score', 0)]
            rep_f1_scores = [cluster_data.get('f1', 0)]
            rep_accuracy_scores = [cluster_data.get('accuracy', 0)]
            rep_confidence_scores = [cluster_data.get('avg_confidence', 0)]
            rep_orthogonality_scores = [cluster_data.get('orthogonality', 0)]
            rep_semantic_similarity_scores = [cluster_data.get('semantic_similarity', 0)]
        
        # Calculate statistics across repetitions
        for metric_name, values in [
            ('final_scores', rep_final_scores),
            ('f1_scores', rep_f1_scores),
            ('accuracy_scores', rep_accuracy_scores),
            ('confidence_scores', rep_confidence_scores),
            ('orthogonality_scores', rep_orthogonality_scores),
            ('semantic_similarity_scores', rep_semantic_similarity_scores)
        ]:
            metrics[metric_name]['median'].append(np.median(values))
            metrics[metric_name]['min'].append(np.min(values))
            metrics[metric_name]['max'].append(np.max(values))
    
    # Find optimal cluster count based on median final score
    optimal_n_clusters = cluster_range[np.argmax(metrics['final_scores']['median'])]
    
    # Create figure with 3x2 subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    
    # Define x-coordinates for vertical lines
    vertical_lines_x = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    
    def plot_with_uncertainty(ax, x, median, min_vals, max_vals, color, xlabel, ylabel, title):
        """Helper function to plot line with uncertainty band"""
        # Plot uncertainty band
        ax.fill_between(x, min_vals, max_vals, alpha=0.2, color=color, label='Min-Max Range')
        # Plot median line
        ax.plot(x, median, 'o-', color=color, linewidth=2, label='Median')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axvline(x=optimal_n_clusters, color='gray', linestyle='--', alpha=0.7, label='Optimal')
        
        # Add vertical grid lines
        for x_line in vertical_lines_x:
            ax.axvline(x=x_line, color='red', linestyle='--', alpha=0.15)
        
        ax.legend()
    
    # Final Score (combined metric) - Top Left
    plot_with_uncertainty(
        axs[0, 0], cluster_range, 
        metrics['final_scores']['median'], 
        metrics['final_scores']['min'], 
        metrics['final_scores']['max'], 
        'blue', 'Number of Clusters', 'Final Score', 
        'Final Score vs. Number of Clusters'
    )
    
    # F1 Score - Top Right
    plot_with_uncertainty(
        axs[0, 1], cluster_range,
        metrics['f1_scores']['median'],
        metrics['f1_scores']['min'],
        metrics['f1_scores']['max'],
        'red', 'Number of Clusters', 'Average F1 Score',
        'Average F1 Score vs. Number of Clusters'
    )
    
    # Semantic Similarity - Middle Left
    plot_with_uncertainty(
        axs[1, 0], cluster_range,
        metrics['semantic_similarity_scores']['median'],
        metrics['semantic_similarity_scores']['min'],
        metrics['semantic_similarity_scores']['max'],
        'brown', 'Number of Clusters', 'Semantic Similarity',
        'Semantic Similarity vs. Number of Clusters'
    )
    
    # Completeness - Middle Right
    plot_with_uncertainty(
        axs[1, 1], cluster_range,
        metrics['confidence_scores']['median'],
        metrics['confidence_scores']['min'],
        metrics['confidence_scores']['max'],
        'purple', 'Number of Clusters', 'Completeness',
        'Completeness vs. Number of Clusters'
    )
    
    # Centroid Orthogonality - Bottom Left
    plot_with_uncertainty(
        axs[2, 0], cluster_range,
        metrics['orthogonality_scores']['median'],
        metrics['orthogonality_scores']['min'],
        metrics['orthogonality_scores']['max'],
        'orange', 'Number of Clusters', 'Orthogonality',
        'Centroid Orthogonality vs. Number of Clusters'
    )
    
    # Accuracy - Bottom Right
    plot_with_uncertainty(
        axs[2, 1], cluster_range,
        metrics['accuracy_scores']['median'],
        metrics['accuracy_scores']['min'],
        metrics['accuracy_scores']['max'],
        'green', 'Number of Clusters', 'Accuracy',
        'Autograder Accuracy vs. Number of Clusters'
    )
    
    # Add overall title
    method_name = method.capitalize()
    plt.suptitle(f'{method_name} Clustering Metrics Summary (Model: {model_id}, Layer: {layer})\nLines show median across repetitions, bands show min-max range', fontsize=14)
    
    # Save figure
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust for suptitle
    os.makedirs('results/figures', exist_ok=True)
    save_path = f'results/figures/{method}_summary_{model_id}_layer{layer}.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print_and_flush(f"Saved {method_name} summary visualization to {save_path}")
    
    # Also save as PNG for easier viewing
    save_path_png = f'results/figures/{method}_summary_{model_id}_layer{layer}.png'
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print_and_flush(f"Saved {method_name} summary visualization to {save_path_png}")
    
    plt.show()

# %%

def print_concise_summary(results_data, clustering_method):
    """
    Print a concise summary of the clustering results.
    
    Parameters:
    -----------
    results_data : dict
        Dictionary containing the clustering results
    clustering_method : str
        Name of the clustering method
    """
    # Extract data for printing
    model_id = results_data['model_id']
    layer = results_data['layer']
    detailed_results_dict = results_data['detailed_results']
    
    # Get available cluster range and optimal from the results
    available_clusters = sorted([int(k) for k in detailed_results_dict.keys()])
    
    # Try to get optimal from summary, otherwise calculate from available data
    optimal_n_clusters = results_data.get('optimal_n_clusters')
    if optimal_n_clusters is None:
        # Calculate optimal based on median final scores
        final_scores = []
        for n_clusters in available_clusters:
            cluster_data = detailed_results_dict[str(n_clusters)]
            if 'all_repetitions' in cluster_data:
                rep_final_scores = [rep['final_score'] for rep in cluster_data['all_repetitions']]
                final_scores.append(np.median(rep_final_scores))
            else:
                final_scores.append(cluster_data.get('final_score', 0))
        optimal_n_clusters = available_clusters[np.argmax(final_scores)]

    # Print concise summary of experiment
    print_and_flush("\n" + "="*50)
    model_display = model_id.upper() if model_id else "UNKNOWN"
    print_and_flush(f"{clustering_method.upper()} CLUSTERING SUMMARY - {model_display} Layer {layer}")
    print_and_flush("="*50)

    print_and_flush(f"- Tested cluster range: {min(available_clusters)} to {max(available_clusters)}")
    print_and_flush(f"- Optimal number of clusters: {optimal_n_clusters}")

    # Print results for all cluster sizes
    print_and_flush("\nMetrics for all cluster sizes:")
    print_and_flush(f"{'Clusters':<10} {'Accuracy':<12} {'Avg F1':<12} {'Orthogonality':<15}")
    print_and_flush(f"{'-'*10} {'-'*12} {'-'*12} {'-'*15}")

    for n_clusters in available_clusters:
        cluster_data = detailed_results_dict[str(n_clusters)]
        
        # Extract metrics from the structure
        if 'best_repetition' in cluster_data:
            # New structure with repetitions - use best repetition
            best_rep = cluster_data['best_repetition']
            accuracy = best_rep['avg_accuracy']
            avg_f1 = best_rep['avg_f1']
            orthogonality = best_rep['orthogonality']
        else:
            # Old structure fallback
            accuracy = cluster_data.get('accuracy', 0)
            orthogonality = cluster_data.get('orthogonality', 0)
            
            # Calculate average F1 from detailed results
            cluster_results = cluster_data.get('detailed_results', {})
            f1s = []
            for cluster_id, metrics in cluster_results.items():
                if isinstance(metrics, dict) and metrics.get('f1', 0) > 0:
                    f1s.append(metrics['f1'])
            avg_f1 = sum(f1s) / len(f1s) if f1s else 0
        
        # Highlight the optimal cluster size
        prefix = "* " if n_clusters == optimal_n_clusters else "  "
        
        print_and_flush(f"{prefix}{n_clusters:<8} {accuracy:<12.4f} "
                f"{avg_f1:<12.4f} {orthogonality:<15.4f}")

    # Print top clusters in optimal clustering
    print_and_flush("\nTop clusters with optimal clustering (K={}):\n".format(optimal_n_clusters))
    print_and_flush(f"{'Cluster ID':<10} {'Title':<40} {'Size':<8} {'Precision':<12} {'Recall':<12} {'F1':<8}")
    print_and_flush(f"{'-'*10} {'-'*40} {'-'*8} {'-'*12} {'-'*12} {'-'*8}")

    optimal_cluster_data = detailed_results_dict[str(optimal_n_clusters)]
    
    # Get detailed results from the appropriate structure
    if 'best_repetition' in optimal_cluster_data:
        optimal_detailed_results = optimal_cluster_data['best_repetition']['detailed_results']
    else:
        optimal_detailed_results = optimal_cluster_data.get('detailed_results', {})

    # Sort clusters by size
    sorted_clusters = sorted(
        optimal_detailed_results.items(),
        key=lambda x: x[1].get('size', 0) if isinstance(x[1], dict) else 0,
        reverse=True
    )

    for cluster_id, cluster_data in sorted_clusters:
        if not isinstance(cluster_data, dict):
            continue
            
        title_short = cluster_data.get('title', 'Unknown')[:37] + "..." if len(cluster_data.get('title', '')) > 40 else cluster_data.get('title', 'Unknown')
        size = cluster_data.get('size', 0)
        precision = cluster_data.get('precision', 0)
        recall = cluster_data.get('recall', 0)
        f1 = cluster_data.get('f1', 0)
        
        print_and_flush(f"{cluster_id:<10} {title_short:<40} {size:<8} "
                f"{precision:<12.4f} {recall:<12.4f} {f1:<.4f}")

# %%

for clustering_method in clustering_methods:
    results_json_path = f'results/vars/{clustering_method}_results_{model_id}_layer{args.layer}.json'
    if not os.path.exists(results_json_path):
        print_and_flush(f"Results file {results_json_path} does not exist")
        continue
    
    with open(results_json_path, 'r') as f:
        results_data = json.load(f)
    print_and_flush(f"Loaded {clustering_method} results from {results_json_path}")

    # Visualize results
    visualize_results(results_json_path)

# %%