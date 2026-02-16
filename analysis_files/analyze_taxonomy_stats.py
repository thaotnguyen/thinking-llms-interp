import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from typing import Any, Optional, List, Dict

# Add current dir to path to allow import
sys.path.append(os.getcwd())

from analyze_traces_state_transition import (
    extract_ordered_chunks,
    STATE_ORDER,
    STATE_COLORS,
    VISIBLE_STATES
)

BASE_DIR = "analysis_runs"
OUT_DIR = "comparison_results_analysis_runs/taxonomy_stats"

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

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

def count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.split())

def load_data():
    print(f"Scanning {BASE_DIR}...")
    pattern = os.path.join(BASE_DIR, "**", "results.labeled.json")
    files = glob.glob(pattern, recursive=True)
    
    chunk_rows = []
    trace_meta_rows = []
    
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
                for t_idx, t in enumerate(traces):
                    if isinstance(t, dict):
                        is_correct = _to_bool(t.get("verified_correct"))
                        label_json = t.get("label_json")
                        if not label_json:
                            continue
                            
                        trace_id = f"{model_name}_{t_idx}_{json_file}"
                        
                        # Store trace metadata
                        trace_meta_rows.append({
                            "model": model_name,
                            "is_correct": is_correct,
                            "trace_id": trace_id
                        })
                        
                        # Extract chunks
                        chunks = extract_ordered_chunks(label_json)
                        for c in chunks:
                            state = c.get("state")
                            if state == "other":
                                continue
                                
                            text = c.get("text", "")
                            word_count = count_words(text)
                            
                            chunk_rows.append({
                                "model": model_name,
                                "is_correct": is_correct,
                                "state": state,
                                "word_count": word_count,
                                "trace_id": trace_id
                            })
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            
    return pd.DataFrame(chunk_rows), pd.DataFrame(trace_meta_rows)

def plot_frequencies(df_chunks, df_traces, suffix="aggregated", title_suffix="Aggregated", filter_model=None):
    """
    Plot frequency of taxonomy items.
    We calculate: Count of State / Total Count of All States (Proportion)
    Split by Correct vs Incorrect.
    """
    
    data = df_chunks.copy()
    if filter_model:
        data = data[data["model"] == filter_model]
        
    if len(data) == 0:
        return

    # Filter out None correctness if we represent correct vs incorrect
    data_clean = data.dropna(subset=["is_correct"]).copy()
    data_clean["is_correct_str"] = data_clean["is_correct"].apply(lambda x: "Correct" if x else "Incorrect")
    
    # 1. Calculate proportions per group (Correct/Incorrect)
    # Group by [is_correct_str, state] -> count
    stats = data_clean.groupby(["is_correct_str", "state"]).size().reset_index(name="count")
    
    # Calculate totals per correct/incorrect group to normalize
    totals = stats.groupby("is_correct_str")["count"].transform("sum")
    stats["proportion"] = stats["count"] / totals
    
    # Save CSV
    stats.to_csv(os.path.join(OUT_DIR, f"frequency_stats_{suffix}.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=stats,
        x="state",
        y="proportion",
        hue="is_correct_str",
        order=VISIBLE_STATES,
        palette={"Correct": "#2ecc71", "Incorrect": "#e74c3c"}
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Taxonomy Item Frequency ({title_suffix})")
    plt.ylabel("Proportion of Total Steps")
    plt.xlabel("State")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"frequency_plot_{suffix}.pdf"))
    plt.close()

def plot_lengths(df_chunks, suffix="aggregated", title_suffix="Aggregated", filter_model=None):
    """
    Plot lengths (word counts) of taxonomic sentences.
    """
    data = df_chunks.copy()
    if filter_model:
        data = data[data["model"] == filter_model]

    if len(data) == 0:
        return

    # Filter out None correctness
    data_clean = data.dropna(subset=["is_correct"]).copy()
    data_clean["is_correct_str"] = data_clean["is_correct"].apply(lambda x: "Correct" if x else "Incorrect")
    
    # Calculate stats for CSV
    stats = data_clean.groupby(["is_correct_str", "state"])["word_count"].describe().reset_index()
    stats.to_csv(os.path.join(OUT_DIR, f"length_stats_{suffix}.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # We truncate outlines to avoid messy plots if there are extreme outliers, 
    # but showfliers=False in boxplot is often cleaner for report
    sns.boxplot(
        data=data_clean,
        x="state",
        y="word_count",
        hue="is_correct_str",
        order=VISIBLE_STATES,
        palette={"Correct": "#2ecc71", "Incorrect": "#e74c3c"},
        showfliers=False 
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Taxonomic Sentence Lengths ({title_suffix})")
    plt.ylabel("Word Count")
    plt.xlabel("State")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"length_plot_{suffix}.pdf"))
    plt.close()

def main():
    print("Loading data...")
    df_chunks, df_traces = load_data()
    
    print(f"Loaded {len(df_chunks)} chunks from {len(df_traces)} traces.")
    
    if len(df_chunks) == 0:
        print("No data found.")
        return

    # --- Aggregated Analysis ---
    print("Generating Aggregated Plots...")
    plot_frequencies(df_chunks, df_traces, suffix="aggregated", title_suffix="All Models")
    plot_lengths(df_chunks, suffix="aggregated", title_suffix="All Models")
    
    # --- Per-Model Analysis ---
    models = df_chunks["model"].unique()
    print(f"Generating Per-Model Plots for: {models}")
    
    # Create sub-directory for per-model files to avoid clutter
    model_out_dir = os.path.join(OUT_DIR, "per_model")
    os.makedirs(model_out_dir, exist_ok=True)
    
    # We will also generate a single big figure for per-model comparison if possible
    # But first, individual files requested: "per-model ... plots + csvs"
    
    for m in models:
        safe_m = m.replace(" ", "_").replace("/", "_")
        # Save in the per_model subdir
        
        # We temporarily change global OUT_DIR or pass paths?
        # Easier to just pass a suffix that includes the path, but my functions assume distinct suffix/filename logic.
        # Let's just adjust the functions to return data, or handle pathing manually inside loop? 
        # Actually I hardcoded OUT_DIR in functions. I should have passed it.
        # I'll just temporarily hack the output path logic by hacking the suffix or overriding OUT_DIR if I refactored.
        # Instead, let's just save them in the main OUT_DIR with prefix "model_X_"
        
        plot_frequencies(df_chunks, df_traces, suffix=f"model_{safe_m}", title_suffix=m, filter_model=m)
        plot_lengths(df_chunks, suffix=f"model_{safe_m}", title_suffix=m, filter_model=m)

    # --- Summary Comparison Plot (All models in one big grid) ---
    # Frequency Comparison (Heatmap or Grid of Bars)
    print("Generating Model Comparison Plots...")
    
    # 1. Proportion of each state per model (Correct Only vs Incorrect Only or Combined)
    # Let's do a grouped bar plot or heatmap for "Proportion of Time Spent in State X by Model"
    
    # Group by [model, state] -> normalized count
    model_stats = df_chunks.groupby(["model", "state"]).size().reset_index(name="count")
    model_totals = model_stats.groupby("model")["count"].transform("sum")
    model_stats["proportion"] = model_stats["count"] / model_totals
    
    # Pivot for heatmap
    pivoted = model_stats.pivot(index="model", columns="state", values="proportion")
    # Reorder columns
    pivoted = pivoted[[c for c in VISIBLE_STATES if c in pivoted.columns]]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivoted, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Proportion of States by Model (Aggregate)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison_frequency_heatmap.pdf"))
    plt.close()
    
    # 2. Length Comparison
    # Boxplot of lengths, x=Model, y=Length, hue=State? Too busy.
    # Boxplot of lengths, x=State, y=Length, hue=Model? Too busy.
    # Let's just output the csv data for cross-model comparison
    
    print("Done.")

if __name__ == "__main__":
    main()
