# python generate_responses.py --model meta-llama/Llama-3.1-8B --save_every 1 --max_tokens 1000 --batch_size 64 --is_base_model
# python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --save_every 1 --max_tokens 1000 --batch_size 64


python annotate_thinking.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layer 6 --n_clusters 15