#!/bin/bash
# Example: Run coefficient sweep to find optimal steering strength

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LAYER=10
N_CLUSTERS=10
FEATURE_IDX=0  # Change this to test different features

# Default coefficients: sweep from negative to positive
# Include 0.0 for baseline comparison
COEFFICIENTS="-2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0"

# Test on 100 questions (set higher or remove --limit for full evaluation)
LIMIT=100

echo "========================================="
echo "Coefficient Sweep Experiment"
echo "========================================="
echo "Model: $MODEL"
echo "Layer: $LAYER"
echo "Feature: idx${FEATURE_IDX}"
echo "Coefficients: $COEFFICIENTS"
echo "Questions: $LIMIT"
echo ""
echo "This will:"
echo "  1. Generate responses with each coefficient"
echo "  2. Grade all responses (requires OPENAI_API_KEY)"
echo "  3. Plot accuracy vs coefficient"
echo ""
echo "Estimated time: ~30-60 minutes depending on dataset size"
echo "========================================="
echo ""

# Run the sweep
python coefficient_sweep.py \
    --model "$MODEL" \
    --layer $LAYER \
    --n_clusters $N_CLUSTERS \
    --feature_idx $FEATURE_IDX \
    --coefficients $COEFFICIENTS \
    --limit $LIMIT \
    --load_in_8bit \
    --judge_model gpt-4o-mini

echo ""
echo "========================================="
echo "Sweep Complete!"
echo "========================================="
echo ""
echo "Results saved to: results/coefficient_sweep/"
echo ""
echo "Next steps:"
echo "  1. Check the plot: results/coefficient_sweep/accuracy_vs_coefficient_*.png"
echo "  2. Review CSV: results/coefficient_sweep/results_*.csv"
echo "  3. Run for other features by changing FEATURE_IDX"

