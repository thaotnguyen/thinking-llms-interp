#!/usr/bin/env python3
"""
Run coefficient sweep experiment: generate responses with different steering
coefficients, grade them, and plot accuracy vs coefficient.
"""
import argparse
import subprocess
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Sweep through steering coefficients and measure accuracy")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                    help="Model to use")
parser.add_argument("--layer", type=int, required=True,
                    help="Layer to apply steering at")
parser.add_argument("--n_clusters", type=int, required=True,
                    help="Number of clusters in SAE")
parser.add_argument("--feature_idx", type=int, required=True,
                    help="Feature index to steer towards")
parser.add_argument("--coefficients", type=float, nargs="+", 
                    default=[-10.0, -2.0, -1.0, -0.5, 0.0, 1.0, 2.0],
                    help="List of coefficients to test")
parser.add_argument("--dataset", type=str, default="tmknguyen/MedCaseReasoning-filtered",
                    help="Dataset to use")
parser.add_argument("--dataset_split", type=str, default="train",
                    help="Dataset split")
parser.add_argument("--max_tokens", type=int, default=2048,
                    help="Max tokens per response")
parser.add_argument("--temperature", type=float, default=0.8,
                    help="Sampling temperature")
parser.add_argument("--limit", type=int, default=100,
                    help="Number of questions to test (None = all)")
parser.add_argument('--engine', type=str, default='nnsight', choices=['nnsight','hf','vllm'], 
                    help='Generation engine to use for steering runs (only applied to steering runs)')
parser.add_argument("--load_in_8bit", action="store_true",
                    help="Load model in 8-bit")
parser.add_argument("--skip_generation", action="store_true",
                    help="Skip generation if files exist, only grade and plot")
parser.add_argument("--skip_grading", action="store_true",
                    help="Skip grading if graded files exist, only plot")
parser.add_argument("--judge_model", type=str, default="gpt-5-nano",
                    help="Model to use for grading")
parser.add_argument("--output_dir", type=str, default="results/coefficient_sweep",
                    help="Directory to save results")
parser.add_argument("--use_raw_sae", action="store_true",
                    help="Use raw SAE decoder vectors instead of optimized steering vectors")
args = parser.parse_args()


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Warning: Command failed with return code {result.returncode}")
        return False
    return True


def generate_responses(model, layer, n_clusters, feature_idx, coefficient, 
                      dataset, dataset_split, max_tokens, temperature, 
                      limit, load_in_8bit):
    """Generate responses with given steering coefficient."""
    model_id = model.split('/')[-1].lower()
    
    # Check if coefficient is 0 (baseline, no steering)
    if coefficient == 0.0:
        output_file = f"results/vars/responses_{model_id}_baseline.json"
        
        if os.path.exists(output_file) and args.skip_generation:
            print(f"Skipping generation: {output_file} already exists")
            return output_file
        
        # Generate baseline without steering
        cmd = [
            "python", "generate_responses.py",
            "--model", model,
            "--dataset", dataset,
            "--dataset_split", dataset_split,
            "--max_tokens", str(max_tokens),
            "--temperature", str(temperature),
        ]
        
        if limit:
            cmd.extend(["--limit", str(limit)])
        
        description = f"Generating BASELINE responses (no steering)"
    else:
        output_file = f"results/vars/responses_{model_id}_layer{layer}_idx{feature_idx}_coef{coefficient}.json"
        
        if os.path.exists(output_file) and args.skip_generation:
            print(f"Skipping generation: {output_file} already exists")
            return output_file
        
        # Generate with steering
        cmd = [
            "python", "generate_responses_with_steering.py",
            "--model", model,
            "--layer", str(layer),
            "--n_clusters", str(n_clusters),
            "--feature_idx", str(feature_idx),
            "--coefficient", str(coefficient),
            "--dataset", dataset,
            "--dataset_split", dataset_split,
            "--max_tokens", str(max_tokens),
            "--temperature", str(temperature),
        ]
        
        if limit:
            cmd.extend(["--limit", str(limit)])
        
        if load_in_8bit:
            cmd.append("--load_in_8bit")
        
        if args.use_raw_sae:
            cmd.append("--use_raw_sae")
        
        description = f"Generating responses with coefficient={coefficient}"
        if args.engine:
            cmd.extend(['--engine', args.engine])
    
    success = run_command(cmd, description)
    if success:
        return output_file
    return None


def grade_responses(responses_file, judge_model):
    """Grade responses using grade_responses.py."""
    graded_file = responses_file.replace('.json', '.graded.json')
    
    if os.path.exists(graded_file) and args.skip_grading:
        print(f"Skipping grading: {graded_file} already exists")
        return graded_file
    
    cmd = [
        "python", "../grade_responses.py",
        "--input", responses_file,
        "--output", graded_file,
        "--model", judge_model,
    ]
    
    description = f"Grading responses: {os.path.basename(responses_file)}"
    success = run_command(cmd, description)
    if success:
        return graded_file
    return None


def extract_accuracy(graded_file):
    """Extract accuracy from graded responses."""
    if not os.path.exists(graded_file):
        print(f"Warning: Graded file not found: {graded_file}")
        return None
    
    try:
        with open(graded_file, 'r') as f:
            data = json.load(f)
        
        total = len(data)
        correct = sum(1 for item in data if item.get('is_correct', False))
        accuracy = correct / total if total > 0 else 0.0
        
        print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
        return accuracy, correct, total
    except Exception as e:
        print(f"Error extracting accuracy from {graded_file}: {e}")
        return None


def plot_results(results_df, output_dir, model_id, layer, feature_idx):
    """Create visualization of accuracy vs coefficient."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot line with markers
    ax.plot(results_df['coefficient'], results_df['accuracy'], 
            marker='o', linewidth=2, markersize=8, label='Accuracy')
    
    # Add baseline reference line if coefficient 0 is included
    if 0.0 in results_df['coefficient'].values:
        baseline_acc = results_df[results_df['coefficient'] == 0.0]['accuracy'].values[0]
        ax.axhline(y=baseline_acc, color='red', linestyle='--', 
                   label=f'Baseline (no steering): {baseline_acc:.1%}', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Steering Coefficient', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'Accuracy vs Steering Coefficient\n'
                 f'Model: {model_id}, Layer: {layer}, Feature: idx{feature_idx}',
                 fontsize=16, fontweight='bold')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add value labels on points
    for idx, row in results_df.iterrows():
        ax.annotate(f"{row['accuracy']:.1%}\n({row['correct']}/{row['total']})",
                   (row['coefficient'], row['accuracy']),
                   textcoords="offset points",
                   xytext=(0, 10),
                   ha='center',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Legend
    ax.legend(fontsize=12)
    
    # Save plot
    plot_file = os.path.join(output_dir, 
                             f"accuracy_vs_coefficient_{model_id}_layer{layer}_idx{feature_idx}.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    # Also save as PDF
    pdf_file = plot_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"PDF saved to: {pdf_file}")
    
    plt.close()
    
    return plot_file


def main():
    print("="*80)
    print("COEFFICIENT SWEEP EXPERIMENT")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Feature: idx{args.feature_idx}")
    print(f"Coefficients: {args.coefficients}")
    print(f"Questions: {args.limit if args.limit else 'all'}")
    print(f"Judge Model: {args.judge_model}")
    print()
    
    model_id = args.model.split('/')[-1].lower()
    results = []
    
    # Process each coefficient
    for coef in tqdm(args.coefficients, desc="Processing coefficients"):
        print(f"\n{'='*80}")
        print(f"COEFFICIENT: {coef}")
        print(f"{'='*80}")
        
        # 1. Generate responses
        if not args.skip_generation:
            responses_file = generate_responses(
                args.model, args.layer, args.n_clusters, args.feature_idx,
                coef, args.dataset, args.dataset_split, args.max_tokens,
                args.temperature, args.limit, args.load_in_8bit
            )
        else:
            # Construct expected filename
            if coef == 0.0:
                responses_file = f"results/vars/responses_{model_id}_baseline.json"
            else:
                responses_file = f"results/vars/responses_{model_id}_layer{args.layer}_idx{args.feature_idx}_coef{coef}.json"
        
        if responses_file is None or not os.path.exists(responses_file):
            print(f"Warning: Failed to generate or find responses for coefficient {coef}")
            continue
        
        # 2. Grade responses
        graded_file = grade_responses(responses_file, args.judge_model)
        
        if graded_file is None or not os.path.exists(graded_file):
            print(f"Warning: Failed to grade responses for coefficient {coef}")
            continue
        
        # 3. Extract accuracy
        result = extract_accuracy(graded_file)
        if result is not None:
            accuracy, correct, total = result
            results.append({
                'coefficient': coef,
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'responses_file': responses_file,
                'graded_file': graded_file,
            })
    
    if not results:
        print("\nError: No results were generated. Exiting.")
        sys.exit(1)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('coefficient')
    
    # Save results to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_file = os.path.join(args.output_dir, 
                            f"results_{model_id}_layer{args.layer}_idx{args.feature_idx}.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"\nResults saved to: {csv_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Coefficient':<15} {'Accuracy':<15} {'Correct/Total':<20}")
    print("-"*50)
    for _, row in results_df.iterrows():
        print(f"{row['coefficient']:<15.2f} {row['accuracy']:<15.1%} {row['correct']}/{row['total']}")
    
    # Find best coefficient
    best_row = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"\nBest coefficient: {best_row['coefficient']:.2f} (accuracy: {best_row['accuracy']:.1%})")
    
    # Compare to baseline if available
    if 0.0 in results_df['coefficient'].values:
        baseline_row = results_df[results_df['coefficient'] == 0.0].iloc[0]
        print(f"Baseline (coef=0): {baseline_row['accuracy']:.1%}")
        delta = best_row['accuracy'] - baseline_row['accuracy']
        print(f"Best improvement: {delta:+.1%}")
    
    # Plot results
    plot_file = plot_results(results_df, args.output_dir, model_id, args.layer, args.feature_idx)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results CSV: {csv_file}")
    print(f"Plot: {plot_file}")


if __name__ == "__main__":
    main()

