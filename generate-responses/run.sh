# python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --save_every 1 --batch_size 32 --load_in_8bit
python annotate_thinking.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer 12 --n_clusters 14 --load_in_8bit

# python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --save_every 1 --batch_size 8 --load_in_8bit
python annotate_thinking.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layer 10 --n_clusters 20 --load_in_8bit

# python generate_responses.py --model FreedomIntelligence/HuatuoGPT-o1-8B --save_every 1 --batch_size 8 --load_in_8bit
python annotate_thinking.py --model FreedomIntelligence/HuatuoGPT-o1-8B --layer 10 --n_clusters 16

# python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --save_every 1 --batch_size 4 --load_in_8bit
python annotate_thinking.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer 32 --n_clusters 16 --load_in_8bit

# python generate_responses.py --model Qwen/QwQ-32B --save_every 1 --batch_size 2 --load_in_8bit 
python annotate_thinking.py --model Qwen/QwQ-32B --layer 36 --n_clusters 18 --load_in_8bit

# python generate_responses.py --model openai/gpt-oss-20b --save_every 1 --batch_size 2
python annotate_thinking.py --model openai/gpt-oss-20b --layer 17 --n_clusters 20
