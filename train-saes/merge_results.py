import json
import argparse
import numpy as np
from utils import utils

def merge_json_results(file1_path, file2_path, output_path):
    """
    Merges two JSON result files, combining their detailed results and recalculating summary statistics.

    Args:
        file1_path (str): Path to the first JSON file.
        file2_path (str): Path to the second JSON file.
        output_path (str): Path to save the merged JSON file.
    """
    try:
        with open(file1_path, 'r') as f:
            data1 = json.load(f)
        with open(file2_path, 'r') as f:
            data2 = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure file paths are correct.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # Merge detailed_results, with data2 overwriting duplicates
    merged_detailed_results = data1.get("detailed_results", {})
    merged_detailed_results.update(data2.get("detailed_results", {}))

    # Re-create sorted cluster range and score lists from merged results
    final_cluster_range = sorted([int(k) for k in merged_detailed_results.keys()])
    
    if not final_cluster_range:
        print("No results to process.")
        return

    # Re-calculate all scores from the merged detailed results
    final_accuracy_scores = [merged_detailed_results[str(n)]['accuracy'] for n in final_cluster_range]
    final_orthogonality_scores = [merged_detailed_results[str(n)]['orthogonality'] for n in final_cluster_range]
    final_assignment_rates = [merged_detailed_results[str(n)].get('assigned_fraction', 0) for n in final_cluster_range]

    final_f1_scores = []
    final_precision_scores = []
    final_recall_scores = []

    for n_clusters in final_cluster_range:
        scoring_results = merged_detailed_results[str(n_clusters)]
        f1_sum = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        f1_count = 0
        for cluster_id, metrics in scoring_results['detailed_results'].items():
            if metrics.get('f1', 0) > 0:
                f1_sum += metrics['f1']
                precision_sum += metrics['precision']
                recall_sum += metrics['recall']
                f1_count += 1
        avg_f1 = f1_sum / f1_count if f1_count > 0 else 0
        avg_precision = precision_sum / f1_count if f1_count > 0 else 0
        avg_recall = recall_sum / f1_count if f1_count > 0 else 0
        final_f1_scores.append(avg_f1)
        final_precision_scores.append(avg_precision)
        final_recall_scores.append(avg_recall)

    # Re-identify optimal number of clusters based on accuracy
    optimal_n_clusters = final_cluster_range[np.argmax(final_accuracy_scores)]
    optimal_idx = final_cluster_range.index(optimal_n_clusters)

    # Use metadata from the first file, but update where necessary
    merged_data = {
        "clustering_method": data1.get("clustering_method"),
        "model_id": data1.get("model_id"),
        "layer": data1.get("layer"),
        "cluster_range": final_cluster_range,
        "accuracy_scores": final_accuracy_scores,
        "precision_scores": final_precision_scores,
        "recall_scores": final_recall_scores,
        "f1_scores": final_f1_scores,
        "assignment_rates": final_assignment_rates,
        "orthogonality_scores": final_orthogonality_scores,
        "optimal_n_clusters": optimal_n_clusters,
        "optimal_accuracy": final_accuracy_scores[optimal_idx],
        "optimal_precision": final_precision_scores[optimal_idx],
        "optimal_recall": final_recall_scores[optimal_idx],
        "optimal_f1": final_f1_scores[optimal_idx],
        "optimal_assignment_rate": final_assignment_rates[optimal_idx],
        "optimal_orthogonality": final_orthogonality_scores[optimal_idx],
        "detailed_results": merged_detailed_results
    }

    # Convert any numpy types to Python native types for JSON serialization
    merged_data = utils.convert_numpy_types(merged_data)

    # Save the merged data
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2, cls=utils.NumpyEncoder)
    print(f"Successfully merged {file1_path} and {file2_path} into {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two JSON result files.")
    parser.add_argument("file1", type=str, help="Path to the first JSON results file.")
    parser.add_argument("file2", type=str, help="Path to the second JSON results file.")
    parser.add_argument("output_file", type=str, help="Path to save the merged output file.")
    args = parser.parse_args()

    merge_json_results(args.file1, args.file2, args.output_file) 