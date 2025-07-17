# %%
import json

# %%
total_sentences = 517662.0

layers = [6, 10, 14, 18, 22, 26]
n_cluster_range = [10, 20, 30, 40, 50]

best_final_score = 0.0
best_layer_and_cluster_size = None

final_scores_by_layer_n_clusters = {}  # (layer, n_clusters) -> final_score
avg_f1_by_layer_n_clusters = {}  # (layer, n_clusters) -> avg_f1
best_scores_by_layer_n_clusters = {}  # (layer, n_clusters) -> best_final_score

for layer in layers:
    try:
        with open(f'../train-saes/results/vars/sae_topk_results_deepseek-r1-distill-llama-8b_layer{layer}.json') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Results file not found for layer {layer}")
        continue

    clusters_detailed_results = results['detailed_results']

    for n_clusters in n_cluster_range:
        if str(n_clusters) not in clusters_detailed_results:
            print(f"Warning: No data for layer {layer}, {n_clusters} clusters")
            continue
            
        print(f"=== Layer {layer}, {n_clusters} clusters ===")
        cluster_data = clusters_detailed_results[str(n_clusters)]
        
        # Extract metrics from the best repetition
        best_rep = cluster_data['best_repetition']
        avg_final_score = cluster_data['avg_final_score']
        best_final_score = cluster_data['best_final_score']
        
        accuracy = best_rep['avg_accuracy']
        orthogonality = best_rep['orthogonality']
        semantic_orthogonality = best_rep['semantic_orthogonality_score']
        completeness = best_rep['avg_confidence']
        cluster_detailed_results = best_rep['detailed_results']
        
        print(f"Average Final Score: {avg_final_score:.4f}")
        print(f"Best Final Score: {best_final_score:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Orthogonality: {orthogonality:.4f}")
        print(f"Semantic Orthogonality: {semantic_orthogonality:.4f}")
        print(f"Completeness: {completeness:.4f}")
        
        # Calculate average F1 across clusters
        avg_f1 = 0.0
        valid_clusters = 0
        for cluster_id, data in cluster_detailed_results.items():
            if isinstance(data, dict) and 'f1' in data:
                cluster_size = data['size']
                cluster_percentage = cluster_size / total_sentences * 100
                cluster_title = data['title']
                print(f"  Cluster {cluster_id}: {cluster_size} examples ({cluster_percentage:.2f}%) - {cluster_title}")
                avg_f1 += data['f1']
                valid_clusters += 1

        if valid_clusters > 0:
            avg_f1 /= valid_clusters
            avg_f1_by_layer_n_clusters[(layer, n_clusters)] = avg_f1
            print(f"Average F1: {avg_f1:.4f}")
        else:
            avg_f1 = 0.0
            print("No valid cluster data found")

        final_scores_by_layer_n_clusters[(layer, n_clusters)] = avg_final_score
        best_scores_by_layer_n_clusters[(layer, n_clusters)] = best_final_score
        print()

print("=== Clusters sorted by Average Final Score ===")
sorted_clusters = sorted(final_scores_by_layer_n_clusters.items(), key=lambda x: x[1], reverse=True)
for (layer, n_clusters), score in sorted_clusters:
    print(f"Layer {layer}, {n_clusters} clusters: {score:.4f}")

print("\n=== Clusters sorted by Best Final Score ===")
sorted_clusters = sorted(best_scores_by_layer_n_clusters.items(), key=lambda x: x[1], reverse=True)
for (layer, n_clusters), score in sorted_clusters:
    print(f"Layer {layer}, {n_clusters} clusters: {score:.4f}")

print("\n=== Clusters sorted by Avg F1 ===")
sorted_clusters = sorted(avg_f1_by_layer_n_clusters.items(), key=lambda x: x[1], reverse=True)
for (layer, n_clusters), avg_f1 in sorted_clusters:
    print(f"Layer {layer}, {n_clusters} clusters: {avg_f1:.4f}")

# Find the overall best configuration
if final_scores_by_layer_n_clusters:
    best_config = max(final_scores_by_layer_n_clusters.items(), key=lambda x: x[1])
    best_layer, best_n_clusters = best_config[0]
    best_score = best_config[1]
    print(f"\n=== Best Configuration ===")
    print(f"Layer {best_layer}, {best_n_clusters} clusters: {best_score:.4f}")

# %%