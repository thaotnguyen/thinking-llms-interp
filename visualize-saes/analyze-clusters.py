# %%
import json

# %%

# Found 517662 sentences with activations across 12032 examples
with open('../train-saes/results/vars/sae_topk_results_deepseek-r1-distill-llama-8b_layer6.json') as f:
    results = json.load(f)

total_sentences = 517662.0

clusters_detailed_results = results['detailed_results']
n_cluster_range = [int(n) for n in clusters_detailed_results.keys()]
n_cluster_range.sort()
# n_cluster_range = [20]

final_scores_by_n_clusters = {}
for n_clusters in n_cluster_range:
    print(f"=== {n_clusters} clusters ===")
    cluster_data = clusters_detailed_results[str(n_clusters)] # dict_keys(['accuracy', 'categories', 'orthogonality', 'assigned_fraction', 'category_counts', 'detailed_results'])
    accuracy = cluster_data['accuracy']
    orthogonality = cluster_data['orthogonality']
    completeness = cluster_data['assigned_fraction']
    
    print(f"Accuracy: {accuracy}")
    # print(f"Categories: {cluster_data['categories']}")
    print(f"Orthogonality: {orthogonality}")
    print(f"Completeness: {completeness}")
    # print(f"Category Counts: {cluster_data['category_counts']}")
    
    cluster_detailed_results = cluster_data['detailed_results'] # dict_keys(['title', 'description', 'size', 'precision', 'recall', 'accuracy', 'f1', 'examples'])
    avg_f1 = 0.0
    for cluster_id, data in cluster_detailed_results.items():
        cluster_size = data['size']
        cluster_percentage = cluster_size/total_sentences*100
        cluster_title = data['title']
        # print(f"Cluster {cluster_id}: {cluster_size} examples ({cluster_percentage:.2f}%) - {cluster_title}")
        avg_f1 += data['f1']

        # Print the first 10 sentences in the cluster
        # for sentence in data['examples'][:10]:
        #     print(f"  - {sentence}")
        # print()

    avg_f1 /= len(cluster_detailed_results)
    print(f"Average F1: {avg_f1}")

    cluster_score = (avg_f1 + completeness + orthogonality) / 3
    print(f"Cluster Score: {cluster_score}")

    final_scores_by_n_clusters[n_clusters] = cluster_score

print("=== Clusters sorted by Final score ===")
# worst first
sorted_clusters = sorted(final_scores_by_n_clusters.items(), key=lambda x: x[1], reverse=False)
for n_clusters, cluster_score in sorted_clusters:
    print(f"Cluster {n_clusters}: {cluster_score}")




# %%