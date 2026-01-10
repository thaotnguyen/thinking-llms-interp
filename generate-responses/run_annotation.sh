# python annotate_thinking.py --model Qwen/QwQ-32B --layer 27 --n_clusters 14 --load_in_8bit
# python analyze_transitions.py --model Qwen/QwQ-32B --layer 27 --n_clusters 14

python annotate_thinking.py --model openai/gpt-oss-20b --layer 17 --n_clusters 16
python analyze_transitions.py --model openai/gpt-oss-20b --layer 17 --n_clusters 16

# python annotate_thinking.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer 20 --n_clusters 12 --load_in_8bit
# python analyze_transitions.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer 20

# python annotate_thinking.py --model FreedomIntelligence/HuatuoGPT-o1-8B --layer 18 --n_clusters 10
# python analyze_transitions.py --model FreedomIntelligence/HuatuoGPT-o1-8B --layer 18

python annotate_thinking.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layer 14 --n_clusters 12 --load_in_8bit
python analyze_transitions.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layer 14 --n_clusters 12

# python annotate_thinking.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer 16 --n_clusters 8 --load_in_8bit
# python analyze_transitions.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer 16
