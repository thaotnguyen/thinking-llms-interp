import os
import glob
import json
import sys
import shutil

# Add current dir to path to allow import
sys.path.append(os.getcwd())

from analyze_traces_state_transition import run_analysis

BASE_DIR = "analysis_runs"
OUT_ROOT = "comparison_results_analysis_runs"

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    
    # 1. Identify models and their files
    print(f"Scanning {BASE_DIR}...")
    pattern = os.path.join(BASE_DIR, "**", "results.labeled.json")
    files = glob.glob(pattern, recursive=True)
    
    model_files = {}
    
    for json_file in files:
        # Infer model name
        parts = json_file.split(os.sep)
        try:
            if BASE_DIR in parts:
                idx = parts.index(BASE_DIR)
                if idx + 1 < len(parts):
                    model_name = parts[idx + 1]
                    if model_name not in model_files:
                        model_files[model_name] = []
                    model_files[model_name].append(json_file)
        except IndexError:
            pass
            
    print(f"Found {len(model_files)} models: {list(model_files.keys())}")
    
    # 2. Process each model
    for model_name, file_paths in model_files.items():
        print(f"\nProcessing {model_name}...")
        
        # Combine traces
        all_traces = []
        for fp in file_paths:
            try:
                with open(fp, "r") as f:
                    data = json.load(f)
                traces = data.get("traces", []) if isinstance(data, dict) else data
                if isinstance(traces, list):
                    # Add dataset info if useful, though run_analysis mainly uses pmcid/verified_correct
                    for t in traces:
                        if isinstance(t, dict):
                            # Try to infer dataset name from path
                            parts = fp.split(os.sep)
                            # analysis_runs/model/dataset/...
                            if model_name in parts:
                                idx = parts.index(model_name)
                                if idx + 1 < len(parts):
                                    t["dataset"] = parts[idx+1]
                            all_traces.append(t)
            except Exception as e:
                print(f"  Error reading {fp}: {e}")
        
        print(f"  Combined {len(all_traces)} traces.")
        if not all_traces:
            continue
            
        # Create output dir for this model
        model_out_dir = os.path.join(OUT_ROOT, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        
        # Write temporary combined file
        combined_json_path = os.path.join(model_out_dir, "combined_traces.json")
        combined_data = {"traces": all_traces}
        with open(combined_json_path, "w") as f:
            json.dump(combined_data, f)
            
        # Run analysis
        try:
            print(f"  Running analysis for {model_name}...")
            # We call run_analysis directly. 
            # Note: run_analysis prints to stdout, so we might see a lot of output.
            run_analysis(
                labeled_csv=combined_json_path,
                out_dir=model_out_dir
            )
            print(f"  Completed {model_name}.")
        except Exception as e:
            print(f"  Failed analysis for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll models processed.")

if __name__ == "__main__":
    main()
