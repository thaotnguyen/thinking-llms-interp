#!/bin/bash
# Example script to train and sweep SAE-based steering vectors

# Configuration
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
LAYER=20
N_CLUSTERS=10
FEATURE_IDX=9
COEFFICIENTS="-2.0 -1.0 -0.5 0.0 0.5 1.0 2.0 5.0"

# Training configuration
N_TRAINING_EXAMPLES=8
MAX_ITERS=1000
LR="1e-1,5e-2,1e-2"
STEERING_TYPE="linear"

# Evaluation configuration
DATASET="tmknguyen/MedCaseReasoning-filtered"
LIMIT=100
TEMPERATURE=0.8
JUDGE_MODEL="gpt-5-nano"

# Output
OUTPUT_DIR="results/sae_vector_sweep"

echo "========================================"
echo "SAE Vector Training and Coefficient Sweep"
echo "========================================"
echo "Model: $MODEL"
echo "Layer: $LAYER"
echo "SAE clusters: $N_CLUSTERS"
echo "Feature: $FEATURE_IDX"
echo "Coefficients: $COEFFICIENTS"
echo ""

python train_and_sweep_sae_vectors.py \
    --model "$MODEL" \
    --layer $LAYER \
    --n_clusters $N_CLUSTERS \
    --feature_idx $FEATURE_IDX \
    --coefficients $COEFFICIENTS \
    --n_training_examples $N_TRAINING_EXAMPLES \
    --max_iters $MAX_ITERS \
    --lr "$LR" \
    --steering_type "$STEERING_TYPE" \
    --dataset "$DATASET" \
    --limit $LIMIT \
    --temperature $TEMPERATURE \
    --judge_model "$JUDGE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42

echo ""
echo "========================================"
echo "Sweep complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

