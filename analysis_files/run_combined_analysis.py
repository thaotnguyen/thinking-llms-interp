#!/usr/bin/env python3
"""Run combined analysis on existing model outputs.

This script aggregates all labeled and graded responses from all models and rounds,
then runs the combined transition and differential analyses.

Usage:
    python run_combined_analysis.py --out_root analysis_runs
"""
import argparse
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from multi_model_pipeline import aggregate_all_models_analysis, short_name, MODELS

def find_round_outputs(out_root: str) -> dict:
    """Find all round outputs from existing model directories."""
    all_round_outputs = {}
    
    # Look for round1_10 and round2_100 directories
    for round_tag in ["round1_10", "round2_100"]:
        round_outputs = {}
        for model in MODELS:
            model_short = short_name(model)
            model_dir = os.path.join(out_root, model_short, round_tag)
            labeled_csv = os.path.join(model_dir, "results.labeled.csv")
            results_csv = os.path.join(model_dir, f"results_{model_short}.csv")
            
            paths = {}
            if os.path.isfile(labeled_csv):
                paths["labeled_csv"] = labeled_csv
            if os.path.isfile(results_csv):
                paths["results_csv"] = results_csv
            
            if paths:
                round_outputs[model_short] = paths
        
        if round_outputs:
            all_round_outputs[round_tag] = round_outputs
    
    return all_round_outputs


def main():
    ap = argparse.ArgumentParser(description="Run combined analysis on existing model outputs")
    ap.add_argument("--out_root", default="analysis_runs", help="Root output directory")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if output files already exist")
    args = ap.parse_args()
    
    print(f"Finding model outputs in {args.out_root}...")
    all_round_outputs = find_round_outputs(args.out_root)
    
    if not all_round_outputs:
        print("No model outputs found. Make sure you have run the pipeline first.")
        return
    
    print(f"Found outputs for {len(all_round_outputs)} rounds:")
    for round_tag, round_outputs in all_round_outputs.items():
        print(f"  {round_tag}: {len(round_outputs)} models")
    
    print("\nRunning combined analysis...")
    aggregate_all_models_analysis(
        all_round_outputs,
        args.out_root,
        skip_existing=args.skip_existing,
        skip_labeling=False,
        skip_grading=False,
    )
    
    print("\nCombined analysis complete!")
    combined_dir = os.path.join(args.out_root, "combined", "all_models_analysis")
    print(f"Results in: {combined_dir}")
    print(f"\nTo run classification:")
    print(f"  python classify_traces_transition.py \\")
    print(f"    --combined_dir {combined_dir} \\")
    print(f"    --out_dir transition_classifier_results")


if __name__ == "__main__":
    main()

