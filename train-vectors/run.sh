
MINIBATCH_SIZE_PER_GPU=6
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
TOTAL_MINIBATCHES=$((MINIBATCH_SIZE_PER_GPU * NUM_GPUS))
echo "Total minibatches: $TOTAL_MINIBATCHES"
for cluster in 0; do
    python optimize_steering_vectors.py --model meta-llama/Llama-3.1-8B --max_iters 50 --n_training_examples 2048 --layer 6 --has_bos_token True --steering_vector_idx $cluster --lr "5e-2" --use_wandb --minibatch_size $TOTAL_MINIBATCHES
done

python visualize_vector_losses.py --model meta-llama/Llama-3.1-8B --layer 6 --n_clusters 20