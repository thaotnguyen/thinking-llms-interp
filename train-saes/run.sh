CLUSTERS="10 12 14 16 18 20" # 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
N_EXAMPLES=100000  # all responses

# CLUSTERING_METHODS="gmm pca_gmm spherical_kmeans pca_kmeans agglomerative pca_agglomerative sae_topk"
CLUSTERING_METHODS="sae_topk"

MODELS="Qwen/QwQ-32B openai/gpt-oss-20b deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B deepseek-ai/DeepSeek-R1-Distill-Llama-8B FreedomIntelligence/HuatuoGPT-o1-8B deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# MODELS="openai/gpt-oss-20b"

REPETITIONS=1

get_layers() {
    local model=$1
    case "$model" in
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") echo "6 10 14 18 22 26" ;; # Total layers: 32
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") echo "4 8 12 16 20 24" ;; # Total layers: 28
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B") echo "8 14 20 26 32 38" ;; # Total layers: 48
        "Qwen/QwQ-32B") echo "9 18 27 36 45 54" ;; # Total layers: 64
        "FreedomIntelligence/HuatuoGPT-o1-8B") echo "6 10 14 18 22 26" ;; # Total layers: 32
        "openai/gpt-oss-20b") echo "5 8 11 14 17 20" ;; # Total layers: 24
        *) echo "" ;;
    esac
}

# Generate activations for all models and layers
for MODEL in $MODELS; do
    LAYERS_TO_PROCESS=$(get_layers "$MODEL")
    if [ -n "$LAYERS_TO_PROCESS" ]; then
        if [ "$MODEL" = "openai/gpt-oss-20b" ]; then
            python generate_activations.py --model "$MODEL" --layers $LAYERS_TO_PROCESS --n_examples $N_EXAMPLES --batch_size 1 
        else
            python generate_activations.py --model "$MODEL" --layers $LAYERS_TO_PROCESS --n_examples $N_EXAMPLES --load_in_8bit --batch_size 1
        fi
    fi
done

# Train all clustering methods for all models and layers
for MODEL in $MODELS; do
    for LAYER in $(get_layers $MODEL); do
        python train_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS
    done
done

# Generate titles for all clustering methods for all models and layers
# Uses OpenAI's batch API by default (except for gpt-5-mini which uses parallel API). Change to --command direct if you want to generate titles directly, and ommit the next loop that calls the command "process"
for MODEL in $MODELS; do
    for LAYER in $(get_layers $MODEL); do
        python generate_titles_trained_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS --command submit --evaluator_model gpt-5-mini --max_workers 30
    done
done

# Wait for titles to be generated
for MODEL in $MODELS; do
    for LAYER in $(get_layers $MODEL); do
        python generate_titles_trained_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS --command process --wait-batch-completion
    done
done

# Evaluate all clustering methods for all models and layers
# Uses OpenAI's batch API by default (except for gpt-5-mini which uses parallel API). Change to --command direct if you want to generate titles directly, and ommit the next loop that calls the command "process"
for MODEL in $MODELS; do
    for LAYER in $(get_layers $MODEL); do
        # Extra flags to disable re-computing some of the evaluation metrics, use as needed: --no-accuracy --no-completeness --no-orth --no-sem-orth
        python evaluate_trained_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS --command submit --accuracy_target_cluster_percentage 0.2 --max_workers 30
    done
done

# Wait for evaluation to complete
for MODEL in $MODELS; do
    for LAYER in $(get_layers $MODEL); do
        python evaluate_trained_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS --command process --wait-batch-completion
    done
done

# Visualize all clustering methods for all models and layers
for MODEL in $MODELS; do
    for LAYER in $(get_layers $MODEL); do
        python visualize_results.py --model $MODEL --layer $LAYER --clusters 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --clustering_methods $CLUSTERING_METHODS
        python visualize_comparison.py --model $MODEL --layer $LAYER
    done
    python visualize_clusters.py --model $MODEL
done
python visualize_clusters.py --model all