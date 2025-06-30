cd train-saes

python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer 8

python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer 14

python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer 20

python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer 26

python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer 32

python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer 38