for cluster in {0..35}; do
    echo "Processing cluster: $cluster"
    python optimize_steering_vectors.py \
        --model meta-llama/Llama-3.1-8B \
        --max_iters 50 \
        --n_training_examples 1024 \
        --minibatch_size 4 \
        --layer 12 \
        --steering_vector_idx $cluster \
        --lr "1e-2" \
        --use_activation_perplexity_selection
        #--use_wandb
done

# python visualize_vector_losses.py --model meta-llama/Llama-3.1-8B

# python evaluate_steering_vectors.py

