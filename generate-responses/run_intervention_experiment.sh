#!/bin/bash
# Run intervention experiment: truncate and retry on toxic phrases

set -e

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LIMIT=500
MAX_TOKENS=2048
TEMP=0.7
MAX_RETRIES=3
SPLIT="test"
SAMPLES=1
WORKERS=1000

echo "================================================================"
echo "TOXIC PHRASE INTERVENTION EXPERIMENT"
echo "================================================================"
echo "Model: $MODEL"
echo "Test cases: $LIMIT"
echo "Split: $SPLIT"
echo "Toxic phrases: 'think likely', 'alternatively', 'wait', 'maybe'"
echo "================================================================"
echo ""

# Step 1: Generate responses (baseline + intervention)
echo "Step 1: Generating responses..."
python generate_with_intervention.py \
    --model "$MODEL" \
    --limit $LIMIT \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMP \
    --max_retries $MAX_RETRIES \
    --split $SPLIT \
    --samples $SAMPLES \
    --use_vllm \
    --batch_size 8 \
    --vllm_gpu_memory_utilization 0.95

MODEL_ID=$(echo "$MODEL" | awk -F'/' '{print tolower($NF)}')
BASELINE_FILE="results/intervention_experiment/responses_${MODEL_ID}_baseline.json"
INTERVENTION_FILE="results/intervention_experiment/responses_${MODEL_ID}_intervention.json"

# Step 2: Grade baseline
echo ""
echo "Step 2: Grading baseline responses..."
python ../grade_responses.py \
    --input "$BASELINE_FILE" \
    --output "${BASELINE_FILE%.json}.graded.json" \
    --model "gpt-5-nano" \
    --workers $WORKERS

# Step 3: Grade intervention
echo ""
echo "Step 3: Grading intervention responses..."
python ../grade_responses.py \
    --input "$INTERVENTION_FILE" \
    --output "${INTERVENTION_FILE%.json}.graded.json" \
    --model "gpt-5-nano" \
    --workers $WORKERS

# Step 4: Compare results
echo ""
echo "Step 4: Comparing results..."
python compare_intervention_results.py \
    --baseline "${BASELINE_FILE%.json}.graded.json" \
    --intervention "${INTERVENTION_FILE%.json}.graded.json" \
    --output "results/intervention_experiment/comparison_${MODEL_ID}.json"

echo ""
echo "================================================================"
echo "EXPERIMENT COMPLETE!"
echo "================================================================"
echo "Results saved to: results/intervention_experiment/"

