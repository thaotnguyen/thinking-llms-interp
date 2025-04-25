cd ..
cd hybrid

python train_probes.py --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
                       --epochs 15 \
                       --batch_size 32 \
                       --num_samples 500 \
                       --probe_layer 20


python evaluate_hybrid.py --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
                         --base_model "meta-llama/Llama-3.1-8B" \
                         --probe_layer 20 \
                         --n_batches 1000 \
                         --batch_size 4 \
                         --max_tokens 1000 \
                         --n_gpus 1 \
                         --seed 42