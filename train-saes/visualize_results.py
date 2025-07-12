# %%
import sys
import os
import matplotlib.pyplot as plt
import argparse
import json

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
    
    Parameters:
    -----------
    results_json_path : str
        Path to the JSON file containing the clustering results
    """
    # Load results from JSON
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    # Extract data for plotting
    cluster_range = results['cluster_range']
    accuracy_scores = results['accuracy_scores']
    f1_scores = results['f1_scores']
    assignment_rates = results['assignment_rates']
    orthogonality_scores = results['orthogonality_scores']
    optimal_n_clusters = results['optimal_n_clusters']
    model_id = results['model_id']
    layer = results['layer']
    method = results['clustering_method']
    
    cluster_range_to_keep = [10,20,30,40,50]
    indices_to_keep = [cluster_range.index(x) for x in cluster_range_to_keep]
    cluster_range = [cluster_range[i] for i in indices_to_keep]
    accuracy_scores = [accuracy_scores[i] for i in indices_to_keep]
    f1_scores = [f1_scores[i] for i in indices_to_keep]
    assignment_rates = [assignment_rates[i] for i in indices_to_keep]
    orthogonality_scores = [orthogonality_scores[i] for i in indices_to_keep]
    optimal_n_clusters = cluster_range_to_keep[indices_to_keep.index(optimal_n_clusters)]

    
    # Create figure with 3x2 subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    
    # Define x-coordinates for vertical lines
    vertical_lines_x = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    
    # Calculate final score (cluster score from analyze-clusters.py)
    final_scores = [(f1 + assignment + orthogonality) / 3 
                   for f1, assignment, orthogonality in zip(f1_scores, assignment_rates, orthogonality_scores)]
    
    # Final Score (combined metric) - Top Left
    axs[0, 0].plot(cluster_range, final_scores, 'o-', color='blue')
    axs[0, 0].set_xlabel('Number of Clusters')
    axs[0, 0].set_ylabel('Final Score')
    axs[0, 0].set_title('Final Score vs. Number of Clusters')
    axs[0, 0].axvline(x=optimal_n_clusters, color='gray', linestyle='--')
    for x in vertical_lines_x:
        axs[0, 0].axvline(x=x, color='red', linestyle='--', alpha=0.15)
    
    # Accuracy - Top Right
    axs[0, 1].plot(cluster_range, accuracy_scores, 'o-', color='green')
    axs[0, 1].set_xlabel('Number of Clusters')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_title('Autograder Accuracy vs. Number of Clusters')
    axs[0, 1].axvline(x=optimal_n_clusters, color='gray', linestyle='--')
    for x in vertical_lines_x:
        axs[0, 1].axvline(x=x, color='red', linestyle='--', alpha=0.15)
    
    # F1 Score - Middle Left
    axs[1, 0].plot(cluster_range, f1_scores, 'o-', color='red')
    axs[1, 0].set_xlabel('Number of Clusters')
    axs[1, 0].set_ylabel('Average F1 Score')
    axs[1, 0].set_title('Average F1 Score vs. Number of Clusters')
    axs[1, 0].axvline(x=optimal_n_clusters, color='gray', linestyle='--')
    for x in vertical_lines_x:
        axs[1, 0].axvline(x=x, color='red', linestyle='--', alpha=0.15)
    
    # Centroid Orthogonality - Middle Right
    axs[1, 1].plot(cluster_range, orthogonality_scores, 'o-', color='orange')
    axs[1, 1].set_xlabel('Number of Clusters')
    axs[1, 1].set_ylabel('Orthogonality')
    axs[1, 1].set_title('Centroid Orthogonality vs. Number of Clusters')
    axs[1, 1].axvline(x=optimal_n_clusters, color='gray', linestyle='--')
    for x in vertical_lines_x:
        axs[1, 1].axvline(x=x, color='red', linestyle='--', alpha=0.15)
    
    # Assignment Rate (Completeness) - Bottom Left
    axs[2, 0].plot(cluster_range, assignment_rates, 'o-', color='purple')
    axs[2, 0].set_xlabel('Number of Clusters')
    axs[2, 0].set_ylabel('Assignment Rate')
    axs[2, 0].set_title('Completeness: Assignment Rate vs. Number of Clusters')
    axs[2, 0].axvline(x=optimal_n_clusters, color='gray', linestyle='--')
    for x in vertical_lines_x:
        axs[2, 0].axvline(x=x, color='red', linestyle='--', alpha=0.15)
    
    # Hide the empty subplot in the bottom-right
    axs[2, 1].axis('off')
    
    # Add overall title
    method_name = method.capitalize()
    plt.suptitle(f'{method_name} Clustering Metrics Summary (Model: {model_id}, Layer: {layer})', fontsize=16)
    
    # Save figure
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust for suptitle
    save_path = f'results/figures/{method}_summary_{model_id}_layer{layer}.pdf'
    plt.savefig(save_path)
    print_and_flush(f"Saved {method_name} summary visualization to {save_path}")

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
    cluster_range = results_data['cluster_range']
    accuracy_scores = results_data['accuracy_scores']
    f1_scores = results_data['f1_scores']
    optimal_n_clusters = results_data['optimal_n_clusters']
    detailed_results_dict = results_data['detailed_results']
    model_id = results_data['model_id']
    layer = results_data['layer']
    orthogonality_scores = results_data['orthogonality_scores']

    # Print concise summary of experiment
    print_and_flush("\n" + "="*50)
    model_display = model_id.upper() if model_id else "UNKNOWN"
    print_and_flush(f"{clustering_method.upper()} CLUSTERING SUMMARY - {model_display} Layer {args.layer}")
    print_and_flush("="*50)

    print_and_flush(f"- Tested cluster range: {args.min_clusters} to {args.max_clusters}")
    print_and_flush(f"- Optimal number of clusters: {optimal_n_clusters}")

    # Print results for all cluster sizes
    print_and_flush("\nMetrics for all cluster sizes:")
    print_and_flush(f"{'Clusters':<10} {'Accuracy':<12} {'Avg F1':<12} {'Orthogonality':<15}")
    print_and_flush(f"{'-'*10} {'-'*12} {'-'*12} {'-'*15}")

    for i, n_clusters in enumerate(cluster_range):
        # Calculate average precision and recall for this cluster size
        cluster_results = detailed_results_dict[n_clusters]
        f1s = []
        
        for cluster_id, metrics in cluster_results['detailed_results'].items():
            if metrics['f1'] > 0:
                f1s.append(metrics['f1'])
        
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0
        
        # Highlight the optimal cluster size
        prefix = "* " if n_clusters == optimal_n_clusters else "  "
        
        print_and_flush(f"{prefix}{n_clusters:<8} {accuracy_scores[i]:<12.4f} "
                f"{avg_f1:<12.4f} {orthogonality_scores[i]:<15.4f}")

    # Print top clusters in optimal clustering
    print_and_flush("\nTop clusters with optimal clustering (K={}):\n".format(optimal_n_clusters))
    print_and_flush(f"{'Cluster ID':<10} {'Title':<40} {'Size':<8} {'Precision':<12} {'Recall':<12} {'F1':<8}")
    print_and_flush(f"{'-'*10} {'-'*40} {'-'*8} {'-'*12} {'-'*12} {'-'*8}")

    optimal_results = detailed_results_dict[optimal_n_clusters]
    sorted_clusters = sorted(
        optimal_results['detailed_results'].items(),
        key=lambda x: x[1]['size'],
        reverse=True
    )

    for cluster_id, cluster_data in sorted_clusters:  # Show all clusters, not just top 5
        title_short = cluster_data['title'][:37] + "..." if len(cluster_data['title']) > 40 else cluster_data['title']
        print_and_flush(f"{cluster_id:<10} {title_short:<40} {cluster_data['size']:<8} "
                f"{cluster_data['precision']:<12.4f} {cluster_data['recall']:<12.4f} {cluster_data['f1']:<.4f}")

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