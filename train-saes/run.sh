CLUSTERS="5,10,15,20,25,30,35,40,45,50"
N_EXAMPLES=100000  # all responses

# CLUSTERING_METHODS="gmm pca_gmm spherical_kmeans pca_kmeans agglomerative pca_agglomerative sae_topk"
CLUSTERING_METHODS="sae_topk"

# MODELS="deepseek-ai/DeepSeek-R1-Distill-Llama-8B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODELS="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

REPETITIONS=5

get_layers() {
    local model=$1
    case "$model" in
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") echo "6 10 14 18 22 26" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") echo "4 8 12 16 20 24" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B") echo "8 14 20 26 32 38" ;;
        *) echo "" ;;
    esac
}

# Train all clustering methods for all models and layers
for MODEL in $MODELS; do
    for LAYER in $(get_layers $MODEL); do
        python train_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS
        python visualize_results.py --model $MODEL --layer $LAYER --clusters $CLUSTERS
        python visualize_comparison.py --model $MODEL --layer $LAYER
    done
done

# # Generate titles for all clustering methods for all models and layers
# for MODEL in $MODELS; do
#     for LAYER in $(get_layers $MODEL); do
#         python generate_titles_trained_clustering.py --model $MODEL --layer $LAYER --cluster_sizes $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS
#     done
# done

# # Evaluate all clustering methods for all models and layers
# for MODEL in $MODELS; do
#     for LAYER in $(get_layers $MODEL); do
#         python evaluate_titles_trained_clustering.py --model $MODEL --layer $LAYER --cluster_sizes $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS
#     done
# done

# Visualize all clustering methods for all models and layers
# python visualize_clusters.py --model all