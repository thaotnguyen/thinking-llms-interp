
for cluster in {0..19}; do
    python optimize_steering_vectors.py --model meta-llama/Llama-3.1-8B --layer 6 --has_bos_token True --steering_vector_idx $cluster --lr "5e-2"
done
