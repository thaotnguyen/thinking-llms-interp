#!/bin/bash
# Run steering experiments for different features and coefficients

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LAYER=10
N_CLUSTERS=10
DATASET="jhu-clsp/medcon-qa"
SPLIT="valid"
MAX_TOKENS=2048
TEMP=0.8
LIMIT=50  # Set to empty string for all questions

# Common flags
COMMON_FLAGS="--model $MODEL --layer $LAYER --n_clusters $N_CLUSTERS --dataset $DATASET --dataset_split $SPLIT --max_tokens $MAX_TOKENS --temperature $TEMP --load_in_8bit"

if [ ! -z "$LIMIT" ]; then
    COMMON_FLAGS="$COMMON_FLAGS --limit $LIMIT"
fi

echo "========================================="
echo "Running Steering Experiments"
echo "========================================="
echo "Model: $MODEL"
echo "Layer: $LAYER"
echo "N_clusters: $N_CLUSTERS"
echo "Limit: ${LIMIT:-all}"
echo ""

# Example: Steer towards different features with coefficient 1.0
for FEATURE_IDX in 0 1 2; do
    echo "----------------------------------------"
    echo "Steering towards idx${FEATURE_IDX} with coefficient 1.0"
    echo "----------------------------------------"
    python generate_responses_with_steering.py $COMMON_FLAGS --feature_idx $FEATURE_IDX --coefficient 1.0
    echo ""
done

# Example: Try different coefficients for feature 0
for COEF in 0.5 1.5 2.0; do
    echo "----------------------------------------"
    echo "Steering towards idx0 with coefficient $COEF"
    echo "----------------------------------------"
    python generate_responses_with_steering.py $COMMON_FLAGS --feature_idx 0 --coefficient $COEF
    echo ""
done

echo "========================================="
echo "All steering experiments complete!"
echo "========================================="
echo ""
echo "Results saved to: results/vars/responses_*_layer${LAYER}_idx*_coef*.json"
echo ""
echo "Next steps:"
echo "  1. Grade responses: cd .. && python grade_responses.py --input generate-responses/results/vars/responses_*.json --output ..."
echo "  2. Annotate steered responses to verify feature usage"
echo "  3. Compare with baseline (non-steered) responses"

