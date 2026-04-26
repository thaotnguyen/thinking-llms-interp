# %%
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import argparse
    
parser = argparse.ArgumentParser(description="Visualize SAE grid search results")
parser.add_argument("--model", type=str, default="all",
                    help="Model identifier")

args, _ = parser.parse_known_args()

def load_sae_grid_search_results(model_id, method="sae_topk"):
    """
    Load all SAE grid search results for a specific model across all layers and cluster sizes.
    
    Parameters:
    -----------
    model_id : str
        Model identifier (e.g., "deepseek-r1-distill-qwen-1.5b")
    method : str
        Clustering method to load (default: "sae_topk")
        
    Returns:
    --------
    DataFrame
        DataFrame containing metrics for each layer/n_clusters configuration
    """
    results_dir = 'results/vars'
    results_data = []
    pattern = re.compile(rf"^{re.escape(method)}_results_{re.escape(model_id)}_layer(\d+)\.json$")
    
    # Find all result files for this model and method
    for filename in os.listdir(results_dir):
        match = pattern.match(filename)
        if match:
            try:
                layer = int(match.group(1))
                
                file_path = os.path.join(results_dir, filename)
                with open(file_path, 'r') as f:
                    results = json.load(f)
                
                # New structure: results_by_cluster_size contains cluster sizes as keys
                for n_clusters_str, cluster_data in results["results_by_cluster_size"].items():
                    n_clusters = int(n_clusters_str)
                    
                    # Extract all repetitions for this cluster size
                    all_results = cluster_data.get("all_results", [])
                    statistics = cluster_data.get("statistics", {})
                    
                    required_statistics = {
                        "orthogonality",
                        "avg_accuracy",
                        "avg_f1",
                        "avg_confidence",
                        "semantic_orthogonality_score",
                        "final_score",
                    }

                    if not all_results or not statistics or not required_statistics.issubset(statistics):
                        print(f"Warning: No results found for layer {layer}, clusters {n_clusters}")
                        continue
                    
                    # Calculate average metrics across all repetitions
                    avg_orthogonality = statistics["orthogonality"]["mean"]
                    avg_accuracy = statistics["avg_accuracy"]["mean"]
                    avg_f1 = statistics["avg_f1"]["mean"]
                    avg_completeness = statistics["avg_confidence"]["mean"]
                    avg_semantic_orthogonality = statistics["semantic_orthogonality_score"]["mean"]
                    avg_final_score = statistics["final_score"]["mean"]
                    
                    # Check for dead latents by examining detailed results from first repetition
                    has_dead_latents = False
                    active_clusters = n_clusters  # Default assumption
                    
                    if all_results and "detailed_results" in all_results[0]:
                        detailed = all_results[0]["detailed_results"]
                        active_clusters = 0
                        
                        # Count clusters with size > 0
                        for cluster_id_str, cluster_info in detailed.items():
                            if cluster_info.get("size", 0) > 0:
                                active_clusters += 1
                        
                        has_dead_latents = active_clusters < n_clusters
                    
                    metrics = {
                        "layer": layer,
                        "n_clusters": n_clusters,
                        "orthogonality": avg_orthogonality,
                        "semantic_orthogonality": avg_semantic_orthogonality,
                        "accuracy": avg_accuracy,
                        "f1": avg_f1,
                        "completeness": avg_completeness,
                        "final_score": avg_final_score,
                        "has_dead_latents": has_dead_latents,
                        "active_clusters": active_clusters
                    }
                    results_data.append(metrics)
                
                print(f"Loaded results for layer {layer}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if not results_data:
        print(f"No grid search results found for model {model_id} with method {method}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results_data)
    return df

def visualize_grid_search(results_df, model_id, output_dir="results/figures"):
    """
    Visualize the SAE grid search results with heatmaps.
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame containing metrics for each layer/n_clusters configuration
    model_id : str
        Model identifier for the plot title
    output_dir : str
        Directory to save the visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to visualize
    metrics = ["avg_f1", "avg_confidence", "semantic_orthogonality_score", "final_score"]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Define custom colormap that goes from red (weak) to white to blue (strong)
    cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        [(0.8, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.8)],
        N=256
    )
    
    # Pivot the DataFrame for each metric
    for i, metric in enumerate(metrics):
        # Create a pivot table
        pivot = results_df.pivot_table(
            index='layer', 
            columns='n_clusters', 
            values=metric,
            aggfunc='mean'  # In case there are duplicates
        )
        
        # Sort indices to make sure they're in ascending order
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)
        
        # Create heatmap with colors for all cells but annotations only for best/worst
        ax = axes[i]
        
        # First create heatmap with all colors but no annotations
        hm = sns.heatmap(
            pivot,
            annot=False,  # No annotations initially
            cmap=cmap,
            center=0.5,
            vmin=0,
            vmax=1,
            cbar=False,  # No individual colorbars
            ax=ax
        )
        
        # Find top 3 and bottom 3 configurations
        flat_pivot = pivot.values.flatten()
        top_indices = np.argsort(flat_pivot)[-3:][::-1]  # Indices of top 3 values (highest first)
        bottom_indices = np.argsort(flat_pivot)[:3]  # Indices of bottom 3 values (lowest first)
        
        # Convert flat indices to 2D coordinates
        top_positions = [np.unravel_index(idx, pivot.shape) for idx in top_indices]
        bottom_positions = [np.unravel_index(idx, pivot.shape) for idx in bottom_indices]
        
        # Colors for top 3 (green gradient) and bottom 3 (red gradient)
        top_colors = ['darkgreen', 'green', 'forestgreen']  
        bottom_colors = ['darkred', 'red', 'firebrick']
        
        # Add annotations for top 3 and bottom 3
        for i, (pos, color) in enumerate(zip(top_positions, top_colors)):
            text = f"{pivot.iloc[pos[0], pos[1]]:.2f}"
            ax.text(pos[1] + 0.5, pos[0] + 0.5, text,
                   ha="center", va="center",
                   fontsize=20, weight="bold", color="black")
            ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, fill=False, 
                                      edgecolor=color, lw=5, linestyle='-'))
        
        for i, (pos, color) in enumerate(zip(bottom_positions, bottom_colors)):
            text = f"{pivot.iloc[pos[0], pos[1]]:.2f}"
            ax.text(pos[1] + 0.5, pos[0] + 0.5, text,
                   ha="center", va="center",
                   fontsize=20, weight="bold", color="black")
            ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, fill=False, 
                                      edgecolor=color, lw=5, linestyle='-'))
        
        # Grey out cells with dead latents
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                if (row < has_dead_latents_pivot.shape[0] and 
                    col < has_dead_latents_pivot.shape[1] and
                    pd.notna(has_dead_latents_pivot.iloc[row, col]) and 
                    has_dead_latents_pivot.iloc[row, col]):
                    # Use hatching pattern instead of gray fill
                    ax.add_patch(plt.Rectangle((col, row), 1, 1, 
                                              fill=True, 
                                              color='#000000',  # Black
                                              alpha=0.2,
                                              hatch='///', 
                                              edgecolor='#000000',  # Black
                                              linewidth=0.5))
        
        # Set labels
        ax.set_ylabel("Number of Clusters", fontsize=20) if i == 0 else ax.set_ylabel("")
        ax.set_xlabel("Layer", fontsize=12)
        
        # Add text annotations about best/worst config
        ax.text(
            0.02, 0.02, 
            f"Best: Layer {max_pos[0]}, Clusters {max_pos[1]} ({max_val:.2f})\nWorst: Layer {min_pos[0]}, Clusters {min_pos[1]} ({min_val:.2f})",
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    # Add overall title
    plt.suptitle(f"SAE Grid Search Results for {model_id.upper()}", fontsize=20)
    
    # Adjust tight_layout to reduce white borders
    plt.tight_layout(rect=[-0.05, 0, 1.05, 0.97])
    
    # Create summary table of best configurations per metric
    print("\nBest configurations per metric:")
    
    summary_data = []
    for metric in metrics:
        pivot = results_df.pivot_table(
            index='layer', 
            columns='n_clusters', 
            values=metric,
            aggfunc='mean'
        )
        
        max_pos = np.unravel_index(pivot.values.argmax(), pivot.shape)
        max_layer = pivot.index[max_pos[0]]
        max_clusters = pivot.columns[max_pos[1]]
        max_val = pivot.values[max_pos]
        
        summary_data.append({
            "metric": metric,
            "best_layer": max_layer,
            "best_clusters": max_clusters,
            "value": max_val
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    
    # Find best overall configuration using final_score from JSON
    best_config = results_df.loc[results_df['final_score'].idxmax()]
    
    print("\nBest overall configuration (based on final_score):")
    print(f"Layer: {int(best_config['layer'])}, Clusters: {int(best_config['n_clusters'])}")
    print(f"Metrics: Orthogonality={best_config['orthogonality']:.2f}, F1={best_config['f1']:.2f}, " +
          f"Accuracy={best_config['accuracy']:.2f}, Completeness={best_config['completeness']:.2f}")
    print(f"Final Score: {best_config['final_score']:.3f}")
    
    return fig

def visualize_combined_grid_search(results_df, model_id, output_dir="results/figures"):
    """
    Visualize the SAE grid search results with a single heatmap of final scores.
    Grey out configurations with dead latents.
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame containing metrics for each layer/n_clusters configuration
    model_id : str
        Model identifier for the plot title
    output_dir : str
        Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define metrics to combine
    metrics = ["f1", "completeness", "semantic_orthogonality"]
    
    # Normalize each metric to 0-1 range
    for metric in metrics:
        min_val = results_df[metric].min()
        max_val = results_df[metric].max()
        results_df[f"{metric}_norm"] = (results_df[metric] - min_val) / (max_val - min_val)
        print(f"Metric: {metric}, Min: {min_val}, Max: {max_val}, Norm Min: {results_df[f'{metric}_norm'].min()}, Norm Max: {results_df[f'{metric}_norm'].max()}")
    
    # Calculate combined score (equal weight to all metrics)
    results_df['normalized_final_score'] = (results_df['f1_norm'] + 
                                   results_df['completeness_norm'] + 
                                   results_df['semantic_orthogonality_norm']) / len(metrics)
    
    # Create figure - taller than wide
    plt.figure(figsize=(10, 14))
    
    # Define custom colormap that goes from red (weak) to white to blue (strong)
    # Using paler colors for better readability of scores
    cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        [(0.9, 0.5, 0.5), (1.0, 1.0, 1.0), (0.5, 0.5, 0.9)],
        N=256
    )
    
    # Create a pivot table for the final score - flip axes by putting n_clusters as index and layer as columns
    pivot = results_df.pivot_table(
        index='n_clusters', 
        columns='layer', 
        values='normalized_final_score',
        aggfunc='mean'
    )
    
    # Create a mask for cells with dead latents
    has_dead_latents_pivot = results_df.pivot_table(
        index='n_clusters',
        columns='layer',
        values='has_dead_latents',
        aggfunc=lambda x: any(x)
    )
    
    # Create a pivot for active clusters (for annotations)
    active_clusters_pivot = results_df.pivot_table(
        index='n_clusters',
        columns='layer',
        values='active_clusters',
        aggfunc='mean'
    )
    
    # Sort indices to make sure they're in ascending order
    pivot = pivot.sort_index(axis=1).sort_index(axis=0, ascending=False)
    has_dead_latents_pivot = has_dead_latents_pivot.sort_index(axis=1).sort_index(axis=0, ascending=False)
    active_clusters_pivot = active_clusters_pivot.sort_index(axis=1).sort_index(axis=0, ascending=False)
    
    # Create heatmap
    ax = plt.gca()
    hm = sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Normalized Final Score'},
        annot_kws={"size": 14, "color": "black"}
    )
    
    # Increase colorbar label font size
    cbar = hm.collections[0].colorbar
    cbar.ax.set_ylabel('Normalized Final Score', fontsize=16)
    
    # Grey out cells with dead latents without adding text
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if i < has_dead_latents_pivot.shape[0] and j < has_dead_latents_pivot.shape[1]:
                if pd.notna(has_dead_latents_pivot.iloc[i, j]) and has_dead_latents_pivot.iloc[i, j]:
                    # Add grey overlay for cells with dead latents
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='grey', alpha=0.3))
    
    # Find the best and worst configurations
    max_val = pivot.max().max()
    min_val = pivot.min().min()
    max_pos = np.unravel_index(pivot.values.argmax(), pivot.shape)
    min_pos = np.unravel_index(pivot.values.argmin(), pivot.shape)
    
    # Convert positions to actual layer and n_clusters values
    max_n_clusters = pivot.index[max_pos[0]]
    max_layer = pivot.columns[max_pos[1]]
    min_n_clusters = pivot.index[min_pos[0]]
    min_layer = pivot.columns[min_pos[1]]
    
    # Mark best and worst cells with more visible colored outlines
    ax.add_patch(plt.Rectangle((min_pos[1], min_pos[0]), 1, 1, fill=False, edgecolor='darkred', lw=4))
    ax.add_patch(plt.Rectangle((max_pos[1], max_pos[0]), 1, 1, fill=False, edgecolor='darkgreen', lw=4))
    
    # Set title and labels with larger font sizes - flipped axis labels
    ax.set_ylabel("Number of Clusters", fontsize=18)
    ax.set_xlabel("Layer", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.suptitle(f"SAE Grid Search Results for {model_id.upper()}", fontsize=22)
    
    # Use tight_layout with adjusted left and right margins to reduce white space
    plt.tight_layout(rect=[-0.05, 0, 1.05, 0.95])
    
    # Save figure
    save_path = os.path.join(output_dir, f"sae_combined_grid_search_{model_id}.pdf")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        
    # Count configurations with dead latents
    dead_latent_count = results_df['has_dead_latents'].sum()
    print(f"Number of configurations with dead latents: {dead_latent_count} / {len(results_df)}")
    
    return plt.gcf()

# Usage example
def main(model_id):
    """
    Main function to load and visualize SAE grid search results.
    
    Parameters:
    -----------
    model_id : str
        Model identifier (e.g., "deepseek-r1-distill-qwen-1.5b")
    """
    # Load grid search results
    results_df = load_sae_grid_search_results(model_id)
    
    if results_df is not None:
        print(f"Column names: {results_df.columns}")
        # Print overview of available data
        print(f"\nFound {len(results_df)} configurations across {results_df['layer'].nunique()} layers " +
              f"and {results_df['n_clusters'].nunique()} cluster sizes")
        
        # Visualize combined grid search results
        visualize_combined_grid_search(results_df, model_id)
        
        # Also visualize individual metrics if desired
        # visualize_grid_search(results_df, model_id)
    else:
        print("No results to visualize.")

def get_all_model_ids():
    """
    Extract all unique model IDs from the result files in the directory.
    
    Returns:
    --------
    list
        List of unique model IDs
    """
    results_dir = 'results/vars'
    model_ids = set()
    pattern = re.compile(r"^sae_topk_results_(.+)_layer(\d+)\.json$")
    
    for filename in os.listdir(results_dir):
        match = pattern.match(filename)
        if match:
            model_ids.add(match.group(1))
    
    return sorted(list(model_ids))

def visualize_all_models(output_dir="results/figures"):
    """
    Load and visualize results for all models, displaying them side by side.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all model IDs
    model_ids = get_all_model_ids()
    # Filter out models with "Llama-70B" in title/id
    model_ids = [m for m in model_ids if "llama-70b" not in m.lower()]
    
    if not model_ids:
        print("No model results found.")
        return
    
    print(f"Found {len(model_ids)} models: {', '.join(model_ids)}")
    
    # Sort model IDs by estimated model size
    def get_model_size(model_id):
        # Extract size indicators (like '1.5b', '7b', etc.)
        size_indicators = ['125m', '350m', '770m', '1.5b', '3b', '7b', '13b', '30b', '70b']
        for i, size in enumerate(size_indicators):
            if size in model_id.lower():
                return i
        return len(size_indicators)  # Default to largest if no size found
    
    model_ids = sorted(model_ids, key=get_model_size)
    
    # Load results for all models
    model_results = {}
    for model_id in model_ids:
        results_df = load_sae_grid_search_results(model_id)
        if results_df is not None:
            model_results[model_id] = results_df
            print(f"Loaded results for {model_id}")
        else:
            print(f"No results found for {model_id}")
    
    if not model_results:
        print("No results loaded for any model.")
        return
    
    # Define custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        [(0.9, 0.5, 0.5), (1.0, 1.0, 1.0), (0.5, 0.5, 0.9)],
        N=256
    )
    
    # Create multi-panel figure with up to 3 columns and shared colorbar
    n_models = len(model_results)
    if n_models == 0:
        print("No results loaded for any model.")
        return
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / n_cols))
    fig = plt.figure(figsize=(12 * n_cols, 11 * n_rows))
    
    # Set font sizes globally for better readability in a paper
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 32,    # Increased from 28
        'axes.labelsize': 28,    # Increased from 24
        'xtick.labelsize': 24,   # Increased from 22
        'ytick.labelsize': 24,   # Increased from 22
        'legend.fontsize': 24,   # Increased from 22
        'figure.titlesize': 36   # Increased from 32
    })
    
    # Create a GridSpec layout with space for the colorbar
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=n_cols * [1] + [0.05], height_ratios=n_rows * [1], top=0.88)
    
    # Create axes for each model in row-major order, centering last row if not full
    axes = []
    items_in_last_row = n_models - (n_rows - 1) * n_cols
    for i in range(n_models):
        row = i // n_cols
        idx_in_row = i - row * n_cols
        if row == n_rows - 1 and items_in_last_row < n_cols:
            # Center for 3-column grids: place 1 in middle, 2 at sides
            if n_cols == 3:
                if items_in_last_row == 1:
                    col = 1
                elif items_in_last_row == 2:
                    # Place two items adjacent near the center-left: columns 0 and 1
                    col = 0 if idx_in_row == 0 else 1
                else:
                    col = idx_in_row
            else:
                # Generic offset for other n_cols
                offset = (n_cols - items_in_last_row) // 2
                col = offset + idx_in_row
        else:
            col = idx_in_row
        axes.append(fig.add_subplot(gs[row, col]))
    
    # Create a separate axis for the colorbar spanning all rows
    cbar_ax = fig.add_subplot(gs[:, -1])
    
    # Create visualizations for each model
    for j, (model_id, results_df) in enumerate(model_results.items()):
        metrics = ["f1", "completeness", "semantic_orthogonality"]
        for metric in metrics:
            if metric not in results_df.columns:
                raise ValueError(f"Metric {metric} not found in results for {model_id}. Columns: {results_df.columns}")
            
            min_val = results_df[metric].min()
            max_val = results_df[metric].max()
            results_df[f"{metric}_norm"] = (results_df[metric] - min_val) / (max_val - min_val)
            print(f"Metric: {metric}, Min: {min_val}, Max: {max_val}, Norm Min: {results_df[f'{metric}_norm'].min()}, Norm Max: {results_df[f'{metric}_norm'].max()}")

        results_df['normalized_final_score'] = (results_df['f1_norm'] + 
                                   results_df['completeness_norm'] + 
                                   results_df['semantic_orthogonality_norm']) / len(metrics)
        
        # Create pivot table
        pivot = results_df.pivot_table(
            index='n_clusters', 
            columns='layer', 
            values='normalized_final_score',
            aggfunc='mean'
        )
        
        # Create mask for cells with dead latents
        has_dead_latents_pivot = results_df.pivot_table(
            index='n_clusters',
            columns='layer',
            values='has_dead_latents',
            aggfunc=lambda x: any(x)
        )
        
        # Sort indices
        pivot = pivot.sort_index(axis=1).sort_index(axis=0, ascending=False)
        has_dead_latents_pivot = has_dead_latents_pivot.sort_index(axis=1).sort_index(axis=0, ascending=False)
        
        # Create heatmap with colors for all cells but annotations only for top/bottom
        ax = axes[j]
        
        # First create heatmap with all colors but no annotations
        hm = sns.heatmap(
            pivot,
            annot=False,  # No annotations initially
            cmap=cmap,
            center=0.5,
            vmin=0,
            vmax=1,
            cbar=False,  # No individual colorbars
            ax=ax
        )
        
        # Find top 3 and bottom 3 configurations
        flat_pivot = pivot.values.flatten()
        top_indices = np.argsort(flat_pivot)[-3:][::-1]  # Indices of top 3 values (highest first)
        bottom_indices = np.argsort(flat_pivot)[:3]  # Indices of bottom 3 values (lowest first)
        
        # Convert flat indices to 2D coordinates
        top_positions = [np.unravel_index(idx, pivot.shape) for idx in top_indices]
        bottom_positions = [np.unravel_index(idx, pivot.shape) for idx in bottom_indices]
        
        # Colors for top 3 (green gradient) and bottom 3 (red gradient)
        top_colors = ['darkgreen', 'green', 'forestgreen']  
        bottom_colors = ['darkred', 'red', 'firebrick']
        
        # Add annotations for top 3 and bottom 3
        for i, (pos, color) in enumerate(zip(top_positions, top_colors)):
            text = f"{pivot.iloc[pos[0], pos[1]]:.2f}"
            ax.text(pos[1] + 0.5, pos[0] + 0.5, text,
                   ha="center", va="center",
                   fontsize=28, weight="bold", color="black")
            ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, fill=False, 
                                      edgecolor=color, lw=5, linestyle='-'))
        
        for i, (pos, color) in enumerate(zip(bottom_positions, bottom_colors)):
            text = f"{pivot.iloc[pos[0], pos[1]]:.2f}"
            ax.text(pos[1] + 0.5, pos[0] + 0.5, text,
                   ha="center", va="center",
                   fontsize=28, weight="bold", color="black")
            ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, fill=False, 
                                      edgecolor=color, lw=5, linestyle='-'))
        
        # Grey out cells with dead latents
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                if (row < has_dead_latents_pivot.shape[0] and 
                    col < has_dead_latents_pivot.shape[1] and
                    pd.notna(has_dead_latents_pivot.iloc[row, col]) and 
                    has_dead_latents_pivot.iloc[row, col]):
                    # Use hatching pattern instead of gray fill
                    ax.add_patch(plt.Rectangle((col, row), 1, 1, 
                                              fill=True, 
                                              color='#000000',  # Black
                                              alpha=0.2,
                                              hatch='///', 
                                              edgecolor='#000000',  # Black
                                              linewidth=0.5))
        
        # Set labels
        ax.set_ylabel("Number of Clusters", fontsize=28) if j == 0 else ax.set_ylabel("")
        ax.set_xlabel("Layer", fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=24)
        
        # Set title (keep model name on one line)
        display_model_id = model_id.upper()
        ax.set_title(display_model_id, fontsize=32, pad=10)  # Reduced padding
    
    # Add a single colorbar
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Normalized Final Score', fontsize=28)
    cbar.ax.tick_params(labelsize=24)
    
    # Adjust layout to reduce white borders on left and right
    plt.tight_layout(rect=[-0.05, 0, 1.05, 0.98])  
    
    # Save figure with reduced padding
    save_path = os.path.join(output_dir, "sae_grid_search_all_models.pdf")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    
    return fig

if __name__ == "__main__":
    if args.model == "all":
        visualize_all_models()
    else:
        main(args.model.split('/')[-1].lower())
# %%
