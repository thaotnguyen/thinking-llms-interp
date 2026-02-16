import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
from typing import Any, Optional

# Add current dir to path to allow import
sys.path.append(os.getcwd())

from analyze_traces_state_transition import (
    extract_sequence, 
    circos_for_sequences, 
    transition_matrix,
    heatmap_matrix,
    heatmap_diff,
    VISIBLE_STATES, 
    STATE_COLORS
)

BASE_DIR = "white_box_medqa-edited"
OUT_DIR = "comparison_results"

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
    
    # 1. Collect all data
    pattern = os.path.join(BASE_DIR, "**", "results.labeled.json")
    files = glob.glob(pattern, recursive=True)
    
    all_rows = []
    
    for json_file in files:
        # Infer model name
        parts = json_file.split(os.sep)
        try:
            if parts[0] == BASE_DIR:
                model_name = parts[1]
            else:
                try:
                    idx = parts.index("white_box_medqa-edited")
                    model_name = parts[idx + 1]
                except ValueError:
                    model_name = "unknown"
        except IndexError:
            model_name = "unknown"
            
        print(f"Loading {model_name}...")
        
        with open(json_file, "r") as f:
            data = json.load(f)
            
        traces = data.get("traces", []) if isinstance(data, dict) else data
        if isinstance(traces, list):
            for t in traces:
                if isinstance(t, dict):
                    t["model"] = model_name
                    all_rows.append(t)
    
    df = pd.DataFrame(all_rows)
    print(f"Total traces: {len(df)}")
    
    if "label_json" not in df.columns:
        print("Error: label_json column missing")
        return

    # 2. Extract sequences and Correctness
    print("Extracting sequences and correctness...")
    df["seq"] = df["label_json"].apply(lambda s: [st for st in extract_sequence(s) if st != "other"])
    df["is_correct"] = df["verified_correct"].apply(_to_bool)
    
    # Split
    seqs_correct = [s for s, c in zip(df["seq"], df["is_correct"]) if c is True and s]
    seqs_incorrect = [s for s, c in zip(df["seq"], df["is_correct"]) if c is False and s]
    
    print(f"Correct traces: {len(seqs_correct)}")
    print(f"Incorrect traces: {len(seqs_incorrect)}")
    
    # 3. Aggregated Circos Plot (Correct vs Incorrect)
    print("Generating Aggregated Circos Plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))
    
    # Correct
    try:
        circos_for_sequences(
            seqs_correct, 
            title=f"All Correct (n={len(seqs_correct)})", 
            states=VISIBLE_STATES, 
            ax=axes[0], 
            show_legend=False, 
            exclude_self=True
        )
    except Exception as e:
        print(f"Error plotting correct: {e}")
        
    # Incorrect
    try:
        circos_for_sequences(
            seqs_incorrect, 
            title=f"All Incorrect (n={len(seqs_incorrect)})", 
            states=VISIBLE_STATES, 
            ax=axes[1], 
            show_legend=False, 
            exclude_self=True
        )
    except Exception as e:
        print(f"Error plotting incorrect: {e}")
        
    # Shared Legend
    colors = [STATE_COLORS.get(s, "#dddddd") for s in VISIBLE_STATES]
    handles = [
        Line2D([0], [0], color=c, lw=4, label=s)
        for s, c in zip(VISIBLE_STATES, colors)
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.92, 0.5), fontsize=10, title="States")
    plt.suptitle("Aggregate Flow: Correct vs Incorrect (All Models)", fontsize=16)
    plt.subplots_adjust(right=0.9, wspace=0.1)
    
    out_circos = os.path.join(OUT_DIR, "circos_aggregate_correct_vs_incorrect.pdf")
    fig.savefig(out_circos, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_circos}")
    
    # 4. Transition Matrices & Heatmaps
    print("Generating Transition Matrices...")
    
    # Compute matrices
    P_corr, C_corr = transition_matrix(seqs_correct, exclude_self=True, states=VISIBLE_STATES)
    P_inc, C_inc = transition_matrix(seqs_incorrect, exclude_self=True, states=VISIBLE_STATES)
    
    # Save CSVs
    pd.DataFrame(P_corr, index=VISIBLE_STATES, columns=VISIBLE_STATES).to_csv(os.path.join(OUT_DIR, "transition_probs_aggregate_correct.csv"))
    pd.DataFrame(P_inc, index=VISIBLE_STATES, columns=VISIBLE_STATES).to_csv(os.path.join(OUT_DIR, "transition_probs_aggregate_incorrect.csv"))
    
    # Plot Heatmaps
    heatmap_matrix(
        P_corr, C_corr, 
        title="Transition Matrix: All Correct", 
        out_pdf=os.path.join(OUT_DIR, "heatmap_aggregate_correct.pdf"),
        states=VISIBLE_STATES
    )
    heatmap_matrix(
        P_inc, C_inc, 
        title="Transition Matrix: All Incorrect", 
        out_pdf=os.path.join(OUT_DIR, "heatmap_aggregate_incorrect.pdf"),
        states=VISIBLE_STATES
    )
    
    # 5. Difference Matrix
    print("Generating Difference Matrix...")
    Diff = P_corr - P_inc
    pd.DataFrame(Diff, index=VISIBLE_STATES, columns=VISIBLE_STATES).to_csv(os.path.join(OUT_DIR, "transition_diff_aggregate_correct_vs_incorrect.csv"))
    
    heatmap_diff(
        Diff, 
        title="ΔP: Correct - Incorrect (Aggregate)", 
        out_pdf=os.path.join(OUT_DIR, "heatmap_diff_aggregate_correct_vs_incorrect.pdf"),
        states=VISIBLE_STATES
    )
    
    print("Done.")

if __name__ == "__main__":
    main()
