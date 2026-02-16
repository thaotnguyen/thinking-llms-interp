import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

# Add current dir to path to allow import
sys.path.append(os.getcwd())

from analyze_traces_state_transition import (
    extract_sequence, 
    circos_for_sequences, 
    VISIBLE_STATES, 
    STATE_COLORS
)

BASE_DIR = "analysis_runs"
OUT_DIR = "comparison_results_analysis_runs"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Collect all data
    print(f"Scanning {BASE_DIR}...")
    pattern = os.path.join(BASE_DIR, "**", "results.labeled.json")
    files = glob.glob(pattern, recursive=True)
    
    all_rows = []
    
    for json_file in files:
        # Infer model name
        # Path structure: analysis_runs/model_name/dataset/...
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
            
        print(f"Loading {model_name} from {json_file}...")
        
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                
            traces = data.get("traces", []) if isinstance(data, dict) else data
            if isinstance(traces, list):
                for t in traces:
                    if isinstance(t, dict):
                        t["model"] = model_name
                        all_rows.append(t)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    df = pd.DataFrame(all_rows)
    print(f"Total traces: {len(df)}")
    
    if len(df) == 0:
        print("No traces found.")
        return
    
    if "label_json" not in df.columns:
        print("Error: label_json column missing")
        return

    # 2. Extract sequences
    print("Extracting sequences...")
    # extract_sequence returns list of states including 'other', filter them here
    df["seq"] = df["label_json"].apply(lambda s: [st for st in extract_sequence(s) if st != "other"])
    
    # 3. Generate Combined Plot
    models = sorted(df["model"].unique())
    print(f"Models found: {models}")
    
    if not models:
        print("No models found.")
        return

    def generate_combined_plot(min_prob, out_filename):
        print(f"Generating {out_filename} with min_prob={min_prob}...")
        
        # Create grid
        n_models = len(models)
        cols_grid = 3 if n_models >= 3 else n_models
        rows_grid = (n_models + cols_grid - 1) // cols_grid
        
        # Increase figure size significantly
        fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(6 * cols_grid, 6 * rows_grid + 2), subplot_kw=dict(polar=True))
        
        # Ensure axes is always a flat list
        if n_models == 1:
            axes_flat = [axes]
        else:
            axes_flat = axes.flatten()
        
        for i, m in enumerate(models):
            ax = axes_flat[i]
            sub_m = df[df["model"] == m]
            seqs_m = [s for s in sub_m["seq"].tolist() if s]
            
            if not seqs_m:
                ax.text(0.5, 0.5, f"{m}\n(No valid sequences)", ha="center", va="center")
                ax.axis("off")
                continue
                
            # Call the plotting function imported from the main script
            try:
                circos_for_sequences(
                    seqs_m, 
                    title=str(m), 
                    states=VISIBLE_STATES, 
                    ax=ax, 
                    show_legend=False, 
                    exclude_self=True,
                    min_prob=min_prob
                )
            except Exception as e:
                print(f"Error plotting {m}: {e}")
                ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
        
        # Turn off unused axes
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")
        
        # Add shared legend
        colors = [STATE_COLORS.get(s, "#dddddd") for s in VISIBLE_STATES]
        handles = [
            Line2D([0], [0], color=c, lw=4, label=s)
            for s, c in zip(VISIBLE_STATES, colors)
        ]
        # Place legend outside
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.92, 0.5), fontsize=12, title="States")
        title_suffix = f" (trim < {int(min_prob*100)}%)" if min_prob > 0 else ""
        plt.suptitle(f"Flow Comparison by Model (Analysis Runs){title_suffix}", fontsize=20, y=0.98)
        
        # Adjust layout - closer together
        plt.subplots_adjust(top=0.92, right=0.9, hspace=0.1, wspace=0.1)
        
        out_path = os.path.join(OUT_DIR, out_filename)
        print(f"Saving to {out_path}...")
        fig.savefig(out_path)
        
        # Also save as PNG
        png_filename = out_filename.replace(".pdf", ".png")
        if out_filename == "circos_combined_all_models.pdf":
            png_filename = "model_comparison_grids.png"
            
        out_png_path = os.path.join(OUT_DIR, png_filename)
        print(f"Saving to {out_png_path}...")
        fig.savefig(out_png_path, dpi=150)
        
        plt.close(fig)

    # Generate plots for different thresholds
    generate_combined_plot(0.0, "circos_combined_all_models.pdf")
    generate_combined_plot(0.10, "circos_combined_all_models_trim10.pdf")
    generate_combined_plot(0.25, "circos_combined_all_models_trim25.pdf")

    print("Done.")

if __name__ == "__main__":
    main()
