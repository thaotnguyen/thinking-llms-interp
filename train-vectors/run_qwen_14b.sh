python optimize_steering_vectors.py \
    --model Qwen/Qwen2.5-14B \
    --max_iters 50 \
    --n_training_examples 2048 \
    --n_eval_examples 512 \
    --optim_minibatch_size 1 \
    --layer 18 \
    --steering_vector_idx -1 \
    --lr "1e-2"

for cluster in {0..4}; do  
    echo "Processing cluster: $cluster"
    python optimize_steering_vectors.py \
        --model Qwen/Qwen2.5-14B \
        --max_iters 50 \
        --n_training_examples 2048 \
        --n_eval_examples 512 \
        --optim_minibatch_size 6 \
        --layer 18 \
        --steering_vector_idx $cluster \
        --lr "1e-2" \
        --use_activation_perplexity_selection
done

python visualize_vector_losses.py --model Qwen/Qwen2.5-14B --smoothing_sigma 100 --steering_strategy linear

python evaluate_steering_vectors.py --model Qwen/Qwen2.5-14B --steering_strategy linear

