
import matplotlib
matplotlib.use('Agg')
import os
import glob
import json
import pandas as pd
import numpy as np
import sys
from typing import Any, Optional

# Add current dir to path to allow import
sys.path.append(os.getcwd())

from analyze_traces_state_transition import run_analysis, VISIBLE_STATES

BASE_DIR = "white_box_medqa-edited"
OUT_ROOT = "comparison_results_white_box_medqa-edited"
OUT_DIR = os.path.join(OUT_ROOT, "global_aggregate")

def _to_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    try:
        if isinstance(x, float) and np.isnan(x):
            return None
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    try:
        return float(s) > 0.5
    except Exception:
        return None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Collect ALL traces from ALL models
    print(f"Scanning {BASE_DIR}...")
    pattern = os.path.join(BASE_DIR, "**", "results.labeled.json")
    files = glob.glob(pattern, recursive=True)
    
    all_rows = []
    
    for json_file in files:
        # Infer model name
        parts = json_file.split(os.sep)
        try:
            if BASE_DIR in parts:
                idx = parts.index(BASE_DIR)
                if idx + 1 < len(parts):
                    model_name = parts[idx + 1]
                else:
                    model_name = "unknown"
            else:
                model_name = "unknown"
        except IndexError:
            model_name = "unknown"
            
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                
            traces = data.get("traces", []) if isinstance(data, dict) else data
            if isinstance(traces, list):
                for t in traces:
                    if isinstance(t, dict):
                        # Ensure minimal fields presence
                        if "verified_correct" not in t:
                             # Default to None if not present
                             t["verified_correct"] = None
                        all_rows.append(t)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            
    print(f"Total traces collected: {len(all_rows)}")
    if not all_rows:
        return

    # 2. Save combined JSON for "Global Aggregate"
    combined_json_path = os.path.join(OUT_DIR, "global_combined_traces.json")
    with open(combined_json_path, "w") as f:
        json.dump({"traces": all_rows}, f)
        
    # 3. Run Analysis on the combined dataset
    # This will generate "circos_plot.png", "transition_matrix_heatmap.png" for the AGGREGATE.
    # It also generates "circos_correct.png"/"circos_incorrect.png" automatically if "verified_correct" is present.
    # It generates "transition_matrix_diff_Correct_-_Incorrect.png" automatically.
    
    print("Running global aggregate analysis (this might take a while)...")
    
    # run_analysis will perform:
    # - Standard analysis for "All"
    # - Split analysis for Correct vs Incorrect
    # - Split analysis for Difficulty (if present)
    # - Correlation analysis
    run_analysis(
        labeled_csv=combined_json_path, # Accepts json despite name
        out_dir=OUT_DIR
    )
    
    print(f"Completed global aggregate analysis. Outputs in {OUT_DIR}")

if __name__ == "__main__":
    main()
