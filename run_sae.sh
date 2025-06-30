cd train-saes

python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer 4

python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer 8

python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer 12

python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer 16

python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer 20

python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer 24