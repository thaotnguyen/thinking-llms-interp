import os
import glob
import json
import sys
import pandas as pd

# Add current dir to path to allow import
sys.path.append(os.getcwd())

from analyze_traces_state_transition import run_analysis

BASE_DIR = "analysis_runs"
OUT_ROOT = "comparison_results_analysis_runs/global_aggregate"

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    
    # 1. Identify all files
    print(f"Scanning {BASE_DIR}...")
    pattern = os.path.join(BASE_DIR, "**", "results.labeled.json")
    files = glob.glob(pattern, recursive=True)
    
    all_traces = []
    
    # 2. Collect all traces
    for json_file in files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            traces = data.get("traces", []) if isinstance(data, dict) else data
            if isinstance(traces, list):
                # Infer model name for context if needed, but for global aggregation we just want the traces
                # We can add a source field if we want to debug, but run_analysis aggregates everything given to it
                for t in traces:
                    if isinstance(t, dict):
                         all_traces.append(t)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            
    print(f"Combined {len(all_traces)} traces from all models.")
    
    if not all_traces:
        print("No traces found.")
        return
        
    # Write temporary combined file
    combined_json_path = os.path.join(OUT_ROOT, "global_combined_traces.json")
    # For very large files, dumping to string might be slow, but 35k traces is manageable (~100MB)
    combined_data = {"traces": all_traces}
    with open(combined_json_path, "w") as f:
        json.dump(combined_data, f)
        
    # Run analysis
    try:
        print(f"Running global aggregate analysis...")
        print(f"Outputs will be in {OUT_ROOT}")
        run_analysis(
            labeled_csv=combined_json_path,
            out_dir=OUT_ROOT
        )
        print("Completed global aggregate analysis.")
    except Exception as e:
        print(f"Failed analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
