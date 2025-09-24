python hybrid_token.py --dataset gsm8k --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --base_model meta-llama/Llama-3.1-8B --steering_layer 12  --sae_layer 6 --n_clusters 15 --max_new_tokens 2000 --max_thinking_tokens 2000

python hybrid_token.py --dataset math500 --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --base_model meta-llama/Llama-3.1-8B --steering_layer 12  --sae_layer 6 --n_clusters 15 --max_new_tokens 2000 --max_thinking_tokens 2000
