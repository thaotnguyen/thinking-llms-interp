MIN_CLUSTERS=4
MAX_CLUSTERS=20
N_EXAMPLES=100000  # all responses
CLUSTERING_METHODS="gmm pca_gmm spherical_kmeans pca_kmeans agglomerative pca_agglomerative sae_topk"

# for LAYER in 4 8 12 16 20 24; do
#     python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer $LAYER --min_clusters $MIN_CLUSTERS --max_clusters $MAX_CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS
#     python visualize_results.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer $LAYER
#     python visualize_comparison.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer $LAYER
# done

for LAYER in 6 10 14 18 22 26; do
    python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layer $LAYER --min_clusters $MIN_CLUSTERS --max_clusters $MAX_CLUSTERS --n_examples $N_EXAMPLES --clustering_methods sae_topk
    # python re_evaluate_trained_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layer $LAYER --min_clusters $MIN_CLUSTERS --max_clusters $MAX_CLUSTERS --n_examples $N_EXAMPLES --clustering_methods sae_topk
    python visualize_results.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layer $LAYER
    python visualize_comparison.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layer $LAYER
done

# for LAYER in 8 14 20 26 32 38; do
#     python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer $LAYER --min_clusters $MIN_CLUSTERS --max_clusters $MAX_CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS
#     python visualize_results.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer $LAYER
#     python visualize_comparison.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer $LAYER
# done