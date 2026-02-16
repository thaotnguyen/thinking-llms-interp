import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
import networkx as nx
from scipy import stats

# For FDR correction
try:
    from statsmodels.stats.multitest import multipletests
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False
    # Fallback FDR implementation (Benjamini-Hochberg)
    def multipletests(pvals, alpha=0.05, method='fdr_bh'):
        if method != 'fdr_bh':
            raise ValueError(f"Only fdr_bh supported without statsmodels")
        pvals = np.asarray(pvals)
        n = len(pvals)
        if n == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        sorted_indices = np.argsort(pvals)
        sorted_pvals = pvals[sorted_indices]
        # Benjamini-Hochberg procedure
        corrected = np.zeros(n)
        for i in range(n-1, -1, -1):
            corrected[sorted_indices[i]] = min(1.0, sorted_pvals[i] * n / (i + 1))
            if i < n - 1:
                corrected[sorted_indices[i]] = min(corrected[sorted_indices[i]], corrected[sorted_indices[i+1]])
        rejected = corrected <= alpha
        return rejected, corrected, None, None

try:
    import tiktoken
except Exception:
    tiktoken = None

# Optional: pyCirclize for chord (circos-style) diagrams
try:
    from pycirclize import Circos  # type: ignore
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception:
    Circos = None  # type: ignore
    plt = None
    Line2D = None

# New taxonomy states for analysis aligned with labeling prompt
# Universal Taxonomy v2 (U1-U7)
STATE_ORDER = [
    "orchestration_meta_control",          # U1
    "case_evidence_extraction",           # U2
    "evidence_processing",                # U3
    "medical_knowledge_general",          # U4
    "hypothesis_generation",              # U5
    "hypothesis_evaluation_narrowing",    # U6
    "final_answer_commitment",            # U7
    "other",
]
STATE_TO_IDX = {s: i for i, s in enumerate(STATE_ORDER)}

STATE_COLORS = {
    "orchestration_meta_control": "#f9cb9c",                 # U1
    "case_evidence_extraction": "#fce5cd",                   # U2
    "evidence_processing": "#fff2cc",                        # U3
    "medical_knowledge_general": "#d9ead3",                  # U4
    "hypothesis_generation": "#cfe2f3",                      # U5
    "hypothesis_evaluation_narrowing": "#d5a6d5",            # U6
    "final_answer_commitment": "#e1d5e7",                    # U7
    "other": "#EFEFEF",
}

# Ensure consistent color list for VISIBLE_STATES
VISIBLE_STATES = [s for s in STATE_ORDER if s != "other"]
VISIBLE_COLORS = [STATE_COLORS.get(s, "#dddddd") for s in VISIBLE_STATES]

@dataclass
class TraceStats:
    n_chunks: int
    counts: Dict[str, int]
    has_loops: bool
    token_count: Optional[int]


def extract_ordered_chunks(label_json_str: Any) -> List[Dict[str, Any]]:
    if isinstance(label_json_str, dict):
        obj = label_json_str
    else:
        try:
            obj = json.loads(label_json_str)
        except Exception:
            return []
    chunks: List[Dict[str, Any]] = []
    # Map short codes and legacy names to canonical set
    legacy_to_new = {
        # --- NEW Universal Taxonomy v2 (Self-mapping) ---
        "orchestration_meta_control": "orchestration_meta_control",                 # U1
        "case_evidence_extraction": "case_evidence_extraction",                     # U2
        "evidence_processing": "evidence_processing",                               # U3
        "medical_knowledge_general": "medical_knowledge_general",                   # U4
        "hypothesis_generation": "hypothesis_generation",                           # U5
        "hypothesis_evaluation_narrowing": "hypothesis_evaluation_narrowing",       # U6
        "final_answer_commitment": "final_answer_commitment",                       # U7
        "other": "other",

        # --- Explicit Long Names from JSON ---
        "Orchestration / meta-control": "orchestration_meta_control",
        "Case evidence extraction (patient-specific facts)": "case_evidence_extraction",
        "Evidence processing (salience + interpretation + summarization)": "evidence_processing",
        "Medical knowledge / templates / criteria (general facts)": "medical_knowledge_general",
        "Hypothesis generation (differential expansion)": "hypothesis_generation",
        "Hypothesis evaluation & narrowing (support OR exclude)": "hypothesis_evaluation_narrowing",
        "Final answer commitment": "final_answer_commitment",

        # --- Short Codes (U1-U7) ---
        "U1": "orchestration_meta_control",
        "U2": "case_evidence_extraction",
        "U3": "evidence_processing",
        "U4": "medical_knowledge_general",
        "U5": "hypothesis_generation",
        "U6": "hypothesis_evaluation_narrowing",
        "U7": "final_answer_commitment",

        # --- Legacy Mappings to v2 ---
        # U1
        "meta_reasoning_orchestration": "orchestration_meta_control",
        "task_goal_framing": "orchestration_meta_control",
        "process_management": "orchestration_meta_control",
        "initialization": "orchestration_meta_control",
        "process_organization": "orchestration_meta_control",
        "plan_generation": "orchestration_meta_control",
        "IN": "orchestration_meta_control",
        "PO": "orchestration_meta_control",
        
        # U2
        "case_framing_and_problem_formulation": "case_evidence_extraction", # closest fit
        "case_presentation_framing": "case_evidence_extraction",
        "extracting_current_case_evidence": "case_evidence_extraction",
        "background_context": "case_evidence_extraction",
        "case_fact_reporting": "case_evidence_extraction",
        "case_setup": "case_evidence_extraction",
        "problem_setup": "case_evidence_extraction",
        "time_course_analysis": "case_evidence_extraction",
        "DA": "case_evidence_extraction",
        
        # U3
        "fact_extraction_and_reorganization": "evidence_processing", # closest fit to generic processing
        "salience_estimation_and_feature_selection": "evidence_processing",
        "semantic_and_clinical_interpretation": "evidence_processing",
        "salience_problem_targeting": "evidence_processing",
        "interpreting_individual_data_elements": "evidence_processing",
        "local_evidence_interpretation": "evidence_processing",
        "data_acquisition": "evidence_processing",
        "case_interpretation": "evidence_processing",
        "EG": "evidence_processing",

        # U4
        "medical_knowledge_background_reasoning": "medical_knowledge_general",
        "medical_knowledge_retrieval": "medical_knowledge_general",
        "domain_knowledge_recall": "medical_knowledge_general",
        "stored_medical_knowledge": "medical_knowledge_general",
        "medical_facts": "medical_knowledge_general",
        "fact_retrieval": "medical_knowledge_general",
        "external_resource_retrieval_simulation": "medical_knowledge_general",
        "SK": "medical_knowledge_general",
        "MF": "medical_knowledge_general",

        # U5
        "hypothesis_generation_and_refinement": "hypothesis_generation",
        "hypothesis_generation": "hypothesis_generation",
        "HG": "hypothesis_generation",

        # U6
        "hypothesis_evaluation_and_integrative_support": "hypothesis_evaluation_narrowing",
        "rule_out_logic_and_constraint_handling": "hypothesis_evaluation_narrowing",
        "hypothesis_evaluation": "hypothesis_evaluation_narrowing",
        "evidence_synthesis": "hypothesis_evaluation_narrowing",
        "eliminating_alternatives": "hypothesis_evaluation_narrowing",
        "integrative_support": "hypothesis_evaluation_narrowing",
        "hypothesis_weighing": "hypothesis_evaluation_narrowing",
        "result_consolidation": "hypothesis_evaluation_narrowing",
        "uncertainty_management": "hypothesis_evaluation_narrowing",
        "self_checking": "hypothesis_evaluation_narrowing",
        "prompting_reconsideration": "hypothesis_evaluation_narrowing",
        "HT": "hypothesis_evaluation_narrowing",
        "ES": "hypothesis_evaluation_narrowing",
        "WD": "hypothesis_evaluation_narrowing",

        # U7
        "final_diagnostic_conclusion": "final_answer_commitment",
        "final_conclusion": "final_answer_commitment",
        "final_answer": "final_answer_commitment",
        "final_answer_emission": "final_answer_commitment",
        "diagnostic_commitment": "final_answer_commitment",
        "diagnosis": "final_answer_commitment",
        "DC": "final_answer_commitment",
        "FA": "final_answer_commitment",

        "OT": "other",
    }
    def key_ord(x: Any) -> Any:
        try:
            return int(x)
        except Exception:
            return x
    for k in sorted(obj.keys(), key=key_ord):
        entry = obj[k]
        # Handle new format where function is directly in 'function' key, or in 'function_tags'
        tags = entry.get("function") if isinstance(entry, dict) else None
        if not tags:
             tags = entry.get("function_tags", []) if isinstance(entry, dict) else []

        text = entry.get("text") if isinstance(entry, dict) else None
        if isinstance(tags, str):
            tags = [tags]
        # normalize tags to new canonical set
        norm: List[str] = []
        for t in tags or []:
            t_str = str(t).strip()
            variants = [
                t_str,
                t_str.lower(),
                t_str.upper(),
                t_str.replace(" ", "_"),
                t_str.lower().replace(" ", "_"),
                t_str.replace("-", "_")
            ]
            mapped = None
            for v in variants:
                if v in legacy_to_new:
                    mapped = legacy_to_new[v]
                    break
            # additional special-cases for medical facts spellings
            if mapped is None and t_str.lower() in ("medical facts", "medical_facts", "medicalfacts"):
                mapped = "stored_medical_knowledge"
            if mapped and mapped in STATE_TO_IDX:
                norm.append(mapped)
        if not norm:
            # if nothing mapped, treat as 'other'
            mapped = "other"
            if mapped in STATE_TO_IDX:
                norm.append(mapped)
        
        # Extract depends_on field (list of dependency indices)
        depends_on = entry.get("depends_on", []) if isinstance(entry, dict) else []
        if not isinstance(depends_on, list):
            depends_on = []
        # Normalize to string keys
        depends_on_normalized = [str(d) for d in depends_on]
        
        chunks.append({"state": norm[0], "text": text or "", "depends_on": depends_on_normalized, "chunk_id": k})
    return chunks


def extract_sequence(label_json_str: str) -> List[str]:
    chunks = extract_ordered_chunks(label_json_str)
    return [c["state"] for c in chunks]


def transition_matrix(seqs: List[List[str]], exclude_self: bool = True, states: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if states is None:
        states = VISIBLE_STATES
    IDX = {s: i for i, s in enumerate(states)}
    n = len(states)
    counts = np.zeros((n, n), dtype=np.int64)
    for seq in seqs:
        for a, b in zip(seq, seq[1:]):
            if exclude_self and a == b:
                continue
            if a in IDX and b in IDX:
                i, j = IDX[a], IDX[b]
                counts[i, j] += 1
    # row-normalize to probabilities
    P = counts.astype(float)
    row_sums = P.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums != 0)
    return P, counts


def seq_transition_counts(seq: List[str], exclude_self: bool = True, states: Optional[List[str]] = None) -> np.ndarray:
    """Return a transition COUNT matrix for a single sequence.
    Excludes self transitions if exclude_self is True.
    """
    if states is None:
        states = VISIBLE_STATES
    IDX = {s: i for i, s in enumerate(states)}
    n = len(states)
    C = np.zeros((n, n), dtype=float)
    if not seq or len(seq) < 2:
        return C
    for a, b in zip(seq, seq[1:]):
        if exclude_self and a == b:
            continue
        if a in IDX and b in IDX:
            i, j = IDX[a], IDX[b]
            C[i, j] += 1.0
    return C


def weighted_transition_matrix(
    seqs: List[List[str]],
    groups: List[Optional[int]],
    target_dist: Optional[Dict[int, float]] = None,
    exclude_self: bool = True,
    states: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute difficulty-adjusted transition matrix using inverse-probability weighting.

    Args:
        seqs: list of state sequences (same length as groups; empty sequences ignored).
        groups: per-sequence difficulty indicator (e.g., case_correct_count). None values are skipped.
        target_dist: mapping of group -> target probability mass. If None, uses the empirical
            distribution of provided groups.
        exclude_self: whether to exclude self transitions.

    Returns:
        (P_adj, Cw): P_adj is the row-normalized transition probabilities after weighting.
        Cw is the weighted transition count matrix (float), prior to row normalization.
    """
    # Filter to valid pairs
    valid = [(s, g) for s, g in zip(seqs, groups) if s and g is not None]
    if not valid:
        if states is None:
            states = VISIBLE_STATES
        n = len(states)
        return np.zeros((n, n), dtype=float), np.zeros((n, n), dtype=float)

    seqs_f = [s for s, _ in valid]
    groups_f = [int(g) for _, g in valid]

    # Empirical subset distribution over present groups
    present_groups, counts = np.unique(groups_f, return_counts=True)
    subset_total = float(counts.sum())
    subset_probs = {int(g): (c / subset_total if subset_total > 0 else 0.0) for g, c in zip(present_groups, counts)}

    # Target distribution restricted to present groups; if None, use subset distribution
    if target_dist is None:
        t_probs_raw = subset_probs.copy()
    else:
        t_probs_raw = {int(g): float(target_dist.get(int(g), 0.0)) for g in present_groups}

    # Renormalize target probs over present groups only
    t_sum = sum(t_probs_raw.values())
    if t_sum <= 0:
        # fallback to subset distribution if target has no mass on present groups
        t_probs = subset_probs.copy()
    else:
        t_probs = {g: v / t_sum for g, v in t_probs_raw.items()}

    # Compute group weights w(g) = t_probs(g) / subset_probs(g)
    w_by_group: Dict[int, float] = {}
    for g in present_groups:
        p_s = subset_probs.get(int(g), 0.0)
        p_t = t_probs.get(int(g), 0.0)
        if p_s > 0:
            w_by_group[int(g)] = float(p_t / p_s)
        else:
            # If somehow absent (shouldn't happen due to present_groups), set zero
            w_by_group[int(g)] = 0.0

    # Weighted sum of transition counts over sequences
    if states is None:
        states = VISIBLE_STATES
    n = len(states)
    Cw = np.zeros((n, n), dtype=float)
    for s, g in zip(seqs_f, groups_f):
        w = w_by_group.get(int(g), 0.0)
        if w <= 0:
            continue
        C = seq_transition_counts(s, exclude_self=exclude_self, states=states)
        Cw += w * C

    # Row-normalize to probabilities
    row_sums = Cw.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P_adj = np.divide(Cw, row_sums, out=np.zeros_like(Cw), where=row_sums != 0)
    return P_adj, Cw


def dwell_times(seq: List[str]) -> List[Tuple[str, int]]:
    if not seq:
        return []
    runs = []
    curr = seq[0]
    k = 1
    for s in seq[1:]:
        if s == curr:
            k += 1
        else:
            runs.append((curr, k))
            curr, k = s, 1
    runs.append((curr, k))
    return runs


def seq_transition_matrix(seq: List[str], exclude_self: bool = True, states: Optional[List[str]] = None) -> np.ndarray:
    if states is None:
        states = VISIBLE_STATES
    IDX = {s: i for i, s in enumerate(states)}
    n = len(states)
    C = np.zeros((n, n), dtype=float)
    if not seq or len(seq) < 2:
        return C
    for a, b in zip(seq, seq[1:]):
        if exclude_self and a == b:
            continue
        if a in IDX and b in IDX:
            i, j = IDX[a], IDX[b]
            C[i, j] += 1.0
    row_sums = C.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.divide(C, row_sums, out=np.zeros_like(C), where=row_sums != 0)
    return P


def flatten_matrix(P: np.ndarray, prefix: str, states: Optional[List[str]] = None) -> Dict[str, float]:
    if states is None:
        states = VISIBLE_STATES
    feats: Dict[str, float] = {}
    for i, fi in enumerate(states):
        for j, fj in enumerate(states):
            feats[f"{prefix}_{fi}_{fj}"] = float(P[i, j])
    return feats


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # lightweight sentence splitter
    import re
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    if not text:
        return 0
    if tiktoken is None:
        # fallback approximate whitespace tokenization
        return max(1, len(text.split()))
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text.split()))


def compute_trace_stats(df_row: pd.Series) -> TraceStats:
    label_json_str = df_row["label_json"] if isinstance(df_row["label_json"], str) else json.dumps(df_row["label_json"])
    seq = extract_sequence(label_json_str)
    counts = {s: 0 for s in STATE_ORDER}
    for s in seq:
        counts[s] += 1
    has_loops = any(a == b for a, b in zip(seq, seq[1:]))
    token_count = count_tokens(str(df_row.get("reasoning_trace", "")))
    return TraceStats(n_chunks=len(seq), counts=counts, has_loops=has_loops, token_count=token_count)


def build_dag_from_chunks(chunks: List[Dict[str, Any]]) -> nx.DiGraph:
    """Build a directed acyclic graph from chunk dependency annotations.
    
    Returns:
        NetworkX DiGraph with nodes labeled by chunk_id and state attributes.
    """
    G = nx.DiGraph()
    
    # Create a mapping from chunk_id (string) to chunk data
    chunk_map = {c["chunk_id"]: c for c in chunks}
    
    # Add all nodes
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        G.add_node(chunk_id, state=chunk["state"], text=chunk.get("text", ""))
    
    # Add edges based on depends_on
    for chunk in chunks:
        target = chunk["chunk_id"]
        for dep in chunk.get("depends_on", []):
            if dep in chunk_map:
                G.add_edge(dep, target)
    
    return G


def dag_metrics(G: nx.DiGraph, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute comprehensive DAG metrics for causal reasoning analysis.
    
    Args:
        G: NetworkX DiGraph representing the dependency structure
        chunks: List of chunk dictionaries with state information
    
    Returns:
        Dictionary of DAG metrics
    """
    metrics: Dict[str, float] = {}
    
    # Basic structure
    metrics["dag_n_nodes"] = float(G.number_of_nodes())
    metrics["dag_n_edges"] = float(G.number_of_edges())
    
    if G.number_of_nodes() == 0:
        # Return zeros for empty graphs
        return {k: 0.0 for k in [
            "dag_n_nodes", "dag_n_edges", "dag_avg_in_degree", "dag_avg_out_degree",
            "dag_avg_in_degree_filtered", "dag_highest_pagerank", "dag_avg_betweenness",
            "dag_n_sources", "dag_n_sinks", "dag_longest_path", "dag_width", "dag_layerwidth"
        ]}
    
    # Degree statistics
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    metrics["dag_avg_in_degree"] = float(np.mean(in_degrees)) if in_degrees else 0.0
    metrics["dag_avg_out_degree"] = float(np.mean(out_degrees)) if out_degrees else 0.0
    
    # Filtered in-degree (excluding U1 orchestration and U7 final answer)
    chunk_states = {c["chunk_id"]: c["state"] for c in chunks}
    filtered_in_degrees = [
        d for n, d in G.in_degree()
        if chunk_states.get(n) not in ["orchestration_meta_control", "final_answer_commitment"]
    ]
    metrics["dag_avg_in_degree_filtered"] = float(np.mean(filtered_in_degrees)) if filtered_in_degrees else 0.0
    
    # Centrality measures
    try:
        pagerank = nx.pagerank(G)
        metrics["dag_highest_pagerank"] = float(max(pagerank.values())) if pagerank else 0.0
    except Exception:
        metrics["dag_highest_pagerank"] = 0.0
    
    try:
        betweenness = nx.betweenness_centrality(G)
        metrics["dag_avg_betweenness"] = float(np.mean(list(betweenness.values()))) if betweenness else 0.0
    except Exception:
        metrics["dag_avg_betweenness"] = 0.0
    
    # Sources and sinks
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    metrics["dag_n_sources"] = float(len(sources))
    metrics["dag_n_sinks"] = float(len(sinks))
    
    # Longest path (height of DAG)
    try:
        if nx.is_directed_acyclic_graph(G):
            longest_path = nx.dag_longest_path_length(G)
            metrics["dag_longest_path"] = float(longest_path)
        else:
            # Graph has cycles, compute approximate longest simple path
            metrics["dag_longest_path"] = 0.0
    except Exception:
        metrics["dag_longest_path"] = 0.0
    
    # Width: maximum number of nodes at any level in a topological ordering
    try:
        if nx.is_directed_acyclic_graph(G) and G.number_of_nodes() > 0:
            # Compute level for each node (longest path from any source)
            levels: Dict[Any, int] = {}
            for node in nx.topological_sort(G):
                predecessors = list(G.predecessors(node))
                if not predecessors:
                    levels[node] = 0
                else:
                    levels[node] = max(levels[p] for p in predecessors) + 1
            
            # Count nodes at each level
            from collections import Counter
            level_counts = Counter(levels.values())
            metrics["dag_width"] = float(max(level_counts.values())) if level_counts else 0.0
        else:
            metrics["dag_width"] = 0.0
    except Exception:
        metrics["dag_width"] = 0.0
    
    # Layerwidth: minimum width needed for a layered partition
    # This is approximated by computing the width of a greedy topological layering
    try:
        if nx.is_directed_acyclic_graph(G) and G.number_of_nodes() > 0:
            # Use longest path layering for pathwidth-like metric
            layer_assignment: Dict[Any, int] = {}
            for node in nx.topological_sort(G):
                preds = list(G.predecessors(node))
                if not preds:
                    layer_assignment[node] = 0
                else:
                    layer_assignment[node] = max(layer_assignment[p] for p in preds) + 1
            
            # Maximum nodes in any layer
            from collections import Counter
            layer_sizes = Counter(layer_assignment.values())
            metrics["dag_layerwidth"] = float(max(layer_sizes.values())) if layer_sizes else 0.0
        else:
            metrics["dag_layerwidth"] = 0.0
    except Exception:
        metrics["dag_layerwidth"] = 0.0
    
    return metrics


def compute_dag_features(label_json_str: str) -> Dict[str, float]:
    """Extract DAG features from a labeled trace.
    
    Args:
        label_json_str: JSON string containing labeled chunks with depends_on annotations
    
    Returns:
        Dictionary of DAG metrics
    """
    chunks = extract_ordered_chunks(label_json_str)
    if not chunks:
        return {k: 0.0 for k in [
            "dag_n_nodes", "dag_n_edges", "dag_avg_in_degree", "dag_avg_out_degree",
            "dag_avg_in_degree_filtered", "dag_highest_pagerank", "dag_avg_betweenness",
            "dag_n_sources", "dag_n_sinks", "dag_longest_path", "dag_width", "dag_layerwidth"
        ]}
    
    G = build_dag_from_chunks(chunks)
    return dag_metrics(G, chunks)


def dwell_runs_with_durations(label_json_str: str) -> List[Dict[str, Any]]:
    """Return runs with durations in chunks, sentences, and tokens for each contiguous state run."""
    # Filter out 'other' chunks entirely
    chunks_raw = extract_ordered_chunks(label_json_str)
    chunks = [c for c in chunks_raw if c.get("state") != "other"]
    if not chunks:
        return []
    
    def normalize_text(text: Any) -> str:
        """Convert text to string, handling lists and other types."""
        if text is None:
            return ""
        if isinstance(text, str):
            return text
        if isinstance(text, list):
            # Join list items with spaces
            return " ".join(str(item) for item in text if item)
        return str(text)
    
    runs: List[Dict[str, Any]] = []
    curr_state = chunks[0]["state"]
    accum_texts: List[str] = [normalize_text(chunks[0].get("text"))]
    for ch in chunks[1:]:
        if ch["state"] == curr_state:
            accum_texts.append(normalize_text(ch.get("text")))
        else:
            full_text = " ".join(accum_texts).strip()
            runs.append({
                "state": curr_state,
                "duration_chunks": len(accum_texts),
                "duration_sentences": sum(len(split_sentences(t)) for t in accum_texts),
                "duration_tokens": count_tokens(full_text),
            })
            curr_state = ch["state"]
            accum_texts = [normalize_text(ch.get("text"))]
    # flush last
    full_text = " ".join(accum_texts).strip()
    runs.append({
        "state": curr_state,
        "duration_chunks": len(accum_texts),
        "duration_sentences": sum(len(split_sentences(t)) for t in accum_texts),
        "duration_tokens": count_tokens(full_text),
    })
    return runs


def sankey_for_sequences(
    seqs: List[List[str]],
    title: str,
    out_pdf: Optional[str] = None,
    states: Optional[List[str]] = None,
    *,
    min_prob: float = 0.0,
    exclude_self: bool = True,
):
    """Build a Sankey diagram from sequences.

    Args:
        seqs: list of state label sequences
        title: figure title
        out_pdf: optional path to save image (PDF)
        states: ordering of states (defaults to VISIBLE_STATES)
        min_prob: drop links with conditional probability P(j|i) < min_prob
        exclude_self: whether to exclude self-transitions
    """
    if states is None:
        states = VISIBLE_STATES
    P, counts = transition_matrix(seqs, exclude_self=exclude_self, states=states)
    n = len(states)
    sources: List[int] = []
    targets: List[int] = []
    values: List[int] = []
    labels = states
    for i in range(n):
        for j in range(n):
            if counts[i, j] > 0 and (P[i, j] >= float(min_prob)):
                sources.append(i)
                targets.append(j)
                values.append(int(counts[i, j]))

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig.update_layout(title_text=title, font_size=12)
    if out_pdf:
        fig.write_image(out_pdf)
    return fig


def circos_for_sequences(
    seqs: List[List[str]],
    title: str,
    out_pdf: Optional[str] = None,
    states: Optional[List[str]] = None,
    ax: Optional[Any] = None,  # matplotlib Axes
    *,
    min_prob: float = 0.0,
    exclude_self: bool = True,
    show_legend: bool = True,
):
    """Create a circos-style chord diagram from sequences using pyCirclize.

    Verified against docs examples:
    - https://moshi4.github.io/pyCirclize/chord_diagram/
    - Circos.chord_diagram(matrix_df, space=..., r_lim=..., cmap=..., ...)

    We threshold links by conditional probability P(j|i) and pass the
    raw transition counts for thickness. Self-links can be excluded.
    """
    if states is None:
        states = VISIBLE_STATES
    if Circos is None:
        raise RuntimeError("pycirclize is required for chord diagrams. Install with `pip install pycirclize`.")

    # Compute probability matrix for thresholding and counts for thickness
    P, counts = transition_matrix(seqs, exclude_self=False, states=states)
    C = counts.astype(float).copy()

    # Apply filters
    n = len(states)
    if exclude_self:
        for i in range(n):
            C[i, i] = 0.0
    if min_prob and min_prob > 0.0:
        for i in range(n):
            for j in range(n):
                if P[i, j] < float(min_prob):
                    C[i, j] = 0.0

    if not np.any(C > 0):
        # Even if empty, if we have an ax, we might want to clear it or show "No Data"
        if ax:
            ax.text(0.5, 0.5, "No links", ha="center", va="center")
            ax.axis("off")
            ax.set_title(title)
        else:
            print(f"[circos_for_sequences] No links after filtering (min_prob={min_prob}, exclude_self={exclude_self}); skipping plot: {title}")
        return None

    # DataFrame format expected by pyCirclize (index: from, columns: to)
    matrix_df = pd.DataFrame(C, index=states, columns=states)

    # Use our consistent colors corresponding to states (pass as dict to cmap)
    colors = [STATE_COLORS.get(s, "#dddddd") for s in states]
    color_dict = {s: c for s, c in zip(states, colors)}

    circos = Circos.chord_diagram(
        matrix_df,
        space=4,
        r_lim=(90, 100),
        cmap=color_dict,
        # Remove labels by setting size=0 and color=none. label_kws=None might default.
        label_kws=dict(size=0, color="none"),
        link_kws=dict(direction=1, ec="black", lw=0.5),
    )
    
    # Plot on existing ax or new figure
    if ax is not None:
        circos.plotfig(ax=ax)
        ax.set_title(title)
        # If legend requested on subplot
        if show_legend:
            handles = [
                Line2D([0], [0], color=c, lw=4, label=s)
                for s, c in zip(states, colors)
            ]
            # Place legend outside? or simply
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8, title="States")
    else:
        fig = circos.plotfig()
        fig.suptitle(title)
        if show_legend:
            handles = [
                Line2D([0], [0], color=c, lw=4, label=s)
                for s, c in zip(states, colors)
            ]
            fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.85, 0.5), fontsize=9, title="States")
            # Adjust layout to make room for legend
            plt.subplots_adjust(right=0.85)

        if out_pdf:
            try:
                fig.savefig(out_pdf, bbox_inches="tight")
            except Exception:
                fig.savefig(out_pdf)
            plt.close(fig)
    
    return circos


def heatmap_matrix(P: np.ndarray, counts: np.ndarray, title: str, out_pdf: Optional[str] = None, states: Optional[List[str]] = None):
    if states is None:
        states = VISIBLE_STATES
    # Create text overlay with probabilities (%) and counts in hover
    text = np.vectorize(lambda v: f"{v*100:.1f}%")(P)
    def _fmt_count(v: float) -> str:
        try:
            if float(v).is_integer():
                return f"{int(v)}"
        except Exception:
            pass
        return f"{float(v):.1f}"
    hover = [[f"{states[i]}→{states[j]}<br>p={P[i,j]:.3f}<br>n={_fmt_count(counts[i,j])}" for j in range(P.shape[1])] for i in range(P.shape[0])]
    fig = go.Figure(data=go.Heatmap(
        z=P,
        x=states,
        y=states,
        colorscale="Blues",
        zmin=0,
        zmax=max(1e-9, float(P.max())),
        text=text,
        texttemplate="%{text}",
        hoverinfo="text",
        hovertext=hover,
        showscale=True,
        colorbar=dict(title="P(j|i)"),
    ))
    fig.update_layout(title=title, xaxis_title="to j", yaxis_title="from i")
    fig.update_yaxes(autorange="reversed")
    if out_pdf:
        fig.write_image(out_pdf)
    return fig


def heatmap_diff(D: np.ndarray, title: str, out_pdf: Optional[str] = None, states: Optional[List[str]] = None):
    if states is None:
        states = VISIBLE_STATES
    # Diverging colormap centered at 0, overlay delta in percentage points
    maxabs = float(np.max(np.abs(D))) if D.size else 0.0
    text = np.vectorize(lambda v: f"{v*100:+.1f}pp")(D)
    hover = [[f"{states[i]}→{states[j]}<br>Δp={D[i,j]:+.3f}" for j in range(D.shape[1])] for i in range(D.shape[0])]
    fig = go.Figure(data=go.Heatmap(
        z=D,
        x=states,
        y=states,
        colorscale="RdBu",
        zmid=0,
        zmin=-maxabs,
        zmax=maxabs,
        text=text,
        texttemplate="%{text}",
        hoverinfo="text",
        hovertext=hover,
        showscale=True,
        colorbar=dict(title="ΔP"),
    ))
    fig.update_layout(title=title, xaxis_title="to j", yaxis_title="from i")
    fig.update_yaxes(autorange="reversed")
    if out_pdf:
        fig.write_image(out_pdf)
    return fig


def compute_correlations_with_pvalues(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute correlations with p-values for features against target.
    
    Returns DataFrame with columns: feature, correlation, pvalue
    """
    if target_col not in df.columns:
        return pd.DataFrame(columns=["feature", "correlation", "pvalue"])
    
    if feature_cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in num_cols if c != target_col]
    
    y = df[target_col].astype(float).values
    results = []
    
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        x = df[feat].astype(float).values
        # Remove NaN pairs
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3:  # Need at least 3 points
            continue
        x_clean = x[mask]
        y_clean = y[mask]
        try:
            corr, pval = stats.pearsonr(x_clean, y_clean)
            results.append({"feature": feat, "correlation": corr, "pvalue": pval})
        except Exception:
            continue
    
    return pd.DataFrame(results)


def apply_multiple_testing_corrections(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Apply Bonferroni and FDR (Benjamini-Hochberg) corrections.
    
    Returns dict with keys: 'bonferroni', 'fdr_bh'
    """
    pvals = np.asarray(pvalues)
    n = len(pvals)
    
    if n == 0:
        return {"bonferroni": np.array([]), "fdr_bh": np.array([])}
    
    # Bonferroni correction
    bonferroni = np.minimum(pvals * n, 1.0)
    
    # FDR (Benjamini-Hochberg) correction
    if _HAS_STATSMODELS:
        _, fdr_bh, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    else:
        # Manual BH procedure
        sorted_indices = np.argsort(pvals)
        sorted_pvals = pvals[sorted_indices]
        fdr_bh = np.zeros(n)
        for i in range(n-1, -1, -1):
            fdr_bh[sorted_indices[i]] = min(1.0, sorted_pvals[i] * n / (i + 1))
            if i < n - 1:
                fdr_bh[sorted_indices[i]] = min(fdr_bh[sorted_indices[i]], fdr_bh[sorted_indices[i+1]])
    
    return {"bonferroni": bonferroni, "fdr_bh": fdr_bh}


def adjusted_correlations(df: pd.DataFrame, out_csv: str) -> Optional[pd.DataFrame]:
    # Compute correlation with verified_correct after residualizing features on case_correct_count
    if "verified_correct" not in df.columns:
        return None
    if "case_correct_count" not in df.columns:
        return None
    if df["case_correct_count"].nunique() <= 1:
        return None
    y = df["verified_correct"].astype(float).values
    d = df["case_correct_count"].astype(float).values
    res = []
    num_cols = df.select_dtypes(include=[np.number]).columns
    feats = [c for c in num_cols if c not in ("verified_correct", "case_correct_count")]
    if not feats:
        return None
    d1 = np.c_[np.ones(len(d)), d]
    for c in feats:
        x = df[c].astype(float).values
        # OLS residuals of x ~ a + b*d
        try:
            # Remove NaN pairs
            mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(d))
            if mask.sum() < 3:
                continue
            x_clean = x[mask]
            y_clean = y[mask]
            d_clean = d[mask]
            d1_clean = np.c_[np.ones(len(d_clean)), d_clean]
            beta, *_ = np.linalg.lstsq(d1_clean, x_clean, rcond=None)
            x_hat = d1_clean @ beta
            r = x_clean - x_hat
            # Pearson corr between residuals and y with p-value
            if np.std(r) > 0 and np.std(y_clean) > 0:
                corr, pval = stats.pearsonr(r, y_clean)
                res.append({"feature": c, "correlation": corr, "pvalue": pval})
            else:
                res.append({"feature": c, "correlation": np.nan, "pvalue": np.nan})
        except Exception:
            res.append({"feature": c, "correlation": np.nan, "pvalue": np.nan})
    
    if not res:
        return None
    
    result_df = pd.DataFrame(res)
    # Apply corrections
    valid_mask = ~(np.isnan(result_df["pvalue"]))
    if valid_mask.sum() > 0:
        corrections = apply_multiple_testing_corrections(result_df.loc[valid_mask, "pvalue"].values)
        result_df["pvalue_bonferroni"] = np.nan
        result_df["pvalue_fdr_bh"] = np.nan
        result_df.loc[valid_mask, "pvalue_bonferroni"] = corrections["bonferroni"]
        result_df.loc[valid_mask, "pvalue_fdr_bh"] = corrections["fdr_bh"]
    else:
        result_df["pvalue_bonferroni"] = np.nan
        result_df["pvalue_fdr_bh"] = np.nan
    
    # Sort by p-value
    result_df = result_df.sort_values("pvalue")
    result_df.to_csv(out_csv, index=False)
    return result_df


def dwell_distribution_plots(runs_df: pd.DataFrame, title_prefix: str, out_prefix: Optional[str] = None, states: Optional[List[str]] = None):
    if states is None:
        states = VISIBLE_STATES
    if runs_df.empty:
        return []
    figs = []
    for col, label in [("duration_chunks", "chunks"), ("duration_sentences", "sentences"), ("duration_tokens", "tokens")]:
        fig = px.box(runs_df, x="state", y=col, category_orders={"state": states}, title=f"{title_prefix} — {label}")
        if out_prefix:
            fig.write_image(f"{out_prefix}_{label}.pdf")
        figs.append(fig)
    return figs


def prepare_subsets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    subsets = {
        "all": df,
        "correct": df[df["verified_correct"] == True],
        "incorrect": df[df["verified_correct"] == False],
    }
    # Difficulty by pmcid: compute number of correct across its 10 traces
    if "pmcid" in df.columns:
        grp = df.groupby("pmcid")["verified_correct"].sum(min_count=1).fillna(0).astype(int)
        diff0 = grp[grp == 0].index
        diff5 = grp[grp == 5].index
        diff10 = grp[grp == 10].index
        subsets["difficulty_0"] = df[df["pmcid"].isin(diff0)]
        subsets["difficulty_5"] = df[df["pmcid"].isin(diff5)]
        subsets["difficulty_10"] = df[df["pmcid"].isin(diff10)]
    return subsets


def run_analysis(labeled_csv: str = "results.labeled.csv", out_dir: str = "analysis_outputs", difficulty_filter: Optional[Tuple[int, int]] = None):
    os.makedirs(out_dir, exist_ok=True)
    
    if labeled_csv.lower().endswith(".json"):
        with open(labeled_csv, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "traces" in data:
            df = pd.DataFrame(data["traces"])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("JSON file must contain 'traces' list or be a list of records.")
    else:
        df = pd.read_csv(labeled_csv)
    
    # Collection for all correlations
    all_correlations: List[Dict[str, Any]] = []
    
    # Apply difficulty filter if specified
    if difficulty_filter is not None:
        if "case_correct_count" not in df.columns:
            # Compute case_correct_count if not present
            if "pmcid" not in df.columns or "verified_correct" not in df.columns:
                raise ValueError("Cannot compute case_correct_count: missing pmcid or verified_correct")
            case_correct = df.groupby("pmcid")["verified_correct"].sum(min_count=1).fillna(0).astype(int)
            df = df.merge(case_correct.rename("case_correct_count"), left_on="pmcid", right_index=True, how="left")
        min_difficulty, max_difficulty = difficulty_filter
        df = df[(df["case_correct_count"] >= min_difficulty) & (df["case_correct_count"] <= max_difficulty)]
        if df.empty:
            print(f"No traces found with difficulty_filter range {difficulty_filter}")
            return

    # Ensure label_json column exists
    if "label_json" not in df.columns:
        raise ValueError("Expected label_json column in labeled CSV. Run labeling first.")

    # Coerce verified_correct to boolean dtype (handles strings like 'True'/'False', 1/0, yes/no)
    def _to_bool(x: Any) -> Optional[bool]:
        if isinstance(x, bool):
            return x
        if x is None:
            return None
        # Handle pandas NaN
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

    if "verified_correct" in df.columns:
        vc = df["verified_correct"].apply(_to_bool)
        # Use pandas nullable boolean dtype so sums/correlations work and NA is preserved
        try:
            df["verified_correct"] = vc.astype("boolean")
        except Exception:
            df["verified_correct"] = vc

    # Build sequences per row and drop 'other' states
    df["seq"] = df["label_json"].apply(lambda s: [st for st in extract_sequence(s) if st != "other"])
    # per-case correct count (difficulty) and attach to df
    if "pmcid" in df.columns:
        case_correct = df.groupby("pmcid")["verified_correct"].sum(min_count=1).fillna(0).astype(int)
        df = df.merge(case_correct.rename("case_correct_count"), left_on="pmcid", right_index=True, how="left")

    subsets = prepare_subsets(df)

    # Global transition matrix across all sequences for difference baselines
    all_seqs = [s for s in df["seq"].tolist() if s]
    P_all, _counts_all = transition_matrix(all_seqs, exclude_self=True, states=VISIBLE_STATES)

    # Stats table
    stat_rows = []

    subset_results: Dict[str, Dict[str, Any]] = {}

    # Build target difficulty distribution over all traces (case_correct_count)
    target_dist: Optional[Dict[int, float]] = None
    if "case_correct_count" in df.columns:
        cc_all = df["case_correct_count"].dropna().astype(int)
        if not cc_all.empty:
            vc = cc_all.value_counts().sort_index()
            target_dist = {int(k): float(v) / float(vc.sum()) for k, v in vc.items()}
    for name, sub in subsets.items():
        seqs = [s for s in sub["seq"].tolist() if s]
        if not seqs:
            continue
        P, counts = transition_matrix(seqs, exclude_self=True, states=VISIBLE_STATES)
        # collect runs with durations across traces by state
        runs_records: List[Dict[str, Any]] = []
        for _, r in sub.iterrows():
            lj = r["label_json"]
            if not isinstance(lj, str):
                lj = json.dumps(lj)
            runs = dwell_runs_with_durations(lj)
            for rec in runs:
                runs_records.append({
                    **rec,
                    "pmcid": r.get("pmcid"),
                    "sample_index": r.get("sample_index"),
                    "verified_correct": r.get("verified_correct"),
                    "case_correct_count": r.get("case_correct_count"),
                })
        runs_df = pd.DataFrame(runs_records)
        # Save numeric outputs
        pd.DataFrame(counts, index=VISIBLE_STATES, columns=VISIBLE_STATES).to_csv(os.path.join(out_dir, f"transition_counts_{name}.csv"))
        pd.DataFrame(P, index=VISIBLE_STATES, columns=VISIBLE_STATES).to_csv(os.path.join(out_dir, f"transition_probs_{name}.csv"))
        runs_df.to_csv(os.path.join(out_dir, f"dwell_runs_{name}.csv"), index=False)
        
        # Dwell time correlations with accuracy
        if not runs_df.empty and "verified_correct" in runs_df.columns:
            if "verified_correct" in runs_df.columns:
                corr_df = compute_correlations_with_pvalues(runs_df, "verified_correct")
                if len(corr_df) > 0:
                    # Apply corrections
                    valid_mask = ~(np.isnan(corr_df["pvalue"]))
                    if valid_mask.sum() > 0:
                        corrections = apply_multiple_testing_corrections(corr_df.loc[valid_mask, "pvalue"].values)
                        corr_df["pvalue_bonferroni"] = np.nan
                        corr_df["pvalue_fdr_bh"] = np.nan
                        corr_df.loc[valid_mask, "pvalue_bonferroni"] = corrections["bonferroni"]
                        corr_df.loc[valid_mask, "pvalue_fdr_bh"] = corrections["fdr_bh"]
                    # Sort by p-value
                    corr_df = corr_df.sort_values("pvalue")
                    corr_df.to_csv(os.path.join(out_dir, f"correlation_dwell_accuracy_{name}.csv"), index=False)
                    for _, row in corr_df.iterrows():
                        if pd.notna(row["correlation"]):
                            all_correlations.append({
                                "subset": name,
                                "type": "dwell_accuracy_unadjusted",
                                "feature": row["feature"],
                                "correlation": row["correlation"],
                                "pvalue": row["pvalue"],
                                "pvalue_bonferroni": row.get("pvalue_bonferroni", np.nan),
                                "pvalue_fdr_bh": row.get("pvalue_fdr_bh", np.nan),
                            })
            if "case_correct_count" in runs_df.columns:
                corr_df = compute_correlations_with_pvalues(runs_df, "case_correct_count")
                if len(corr_df) > 0:
                    # Apply corrections
                    valid_mask = ~(np.isnan(corr_df["pvalue"]))
                    if valid_mask.sum() > 0:
                        corrections = apply_multiple_testing_corrections(corr_df.loc[valid_mask, "pvalue"].values)
                        corr_df["pvalue_bonferroni"] = np.nan
                        corr_df["pvalue_fdr_bh"] = np.nan
                        corr_df.loc[valid_mask, "pvalue_bonferroni"] = corrections["bonferroni"]
                        corr_df.loc[valid_mask, "pvalue_fdr_bh"] = corrections["fdr_bh"]
                    # Sort by p-value
                    corr_df = corr_df.sort_values("pvalue")
                    corr_df.to_csv(os.path.join(out_dir, f"correlation_dwell_difficulty_{name}.csv"), index=False)
                    for _, row in corr_df.iterrows():
                        if pd.notna(row["correlation"]):
                            all_correlations.append({
                                "subset": name,
                                "type": "dwell_difficulty",
                                "feature": row["feature"],
                                "correlation": row["correlation"],
                                "pvalue": row["pvalue"],
                                "pvalue_bonferroni": row.get("pvalue_bonferroni", np.nan),
                                "pvalue_fdr_bh": row.get("pvalue_fdr_bh", np.nan),
                            })
            
            # Adjusted correlations for dwell times (partialing out difficulty)
            adj_corr = adjusted_correlations(runs_df, os.path.join(out_dir, f"correlation_dwell_accuracy_adjusted_{name}.csv"))
            if adj_corr is not None and len(adj_corr) > 0:
                for _, row in adj_corr.iterrows():
                    if pd.notna(row["correlation"]):
                        all_correlations.append({
                            "subset": name,
                            "type": "dwell_accuracy_adjusted",
                            "feature": row["feature"],
                            "correlation": row["correlation"],
                            "pvalue": row["pvalue"],
                            "pvalue_bonferroni": row.get("pvalue_bonferroni", np.nan),
                            "pvalue_fdr_bh": row.get("pvalue_fdr_bh", np.nan),
                        })

        # Visualizations
        heatmap_matrix(P, counts, title=f"Transition Matrix (no self) — {name}", out_pdf=os.path.join(out_dir, f"transitions_{name}.pdf"), states=VISIBLE_STATES)
        # Also compute and save matrices including self-transitions
        P_self, counts_self = transition_matrix(seqs, exclude_self=False, states=VISIBLE_STATES)
        pd.DataFrame(counts_self, index=VISIBLE_STATES, columns=VISIBLE_STATES).to_csv(os.path.join(out_dir, f"transition_counts_including_self_{name}.csv"))
        pd.DataFrame(P_self, index=VISIBLE_STATES, columns=VISIBLE_STATES).to_csv(os.path.join(out_dir, f"transition_probs_including_self_{name}.csv"))
        heatmap_matrix(P_self, counts_self, title=f"Transition Matrix (including self) — {name}", out_pdf=os.path.join(out_dir, f"transitions_including_self_{name}.pdf"), states=VISIBLE_STATES)

        dwell_distribution_plots(runs_df, title_prefix=f"Dwell durations — {name}", out_prefix=os.path.join(out_dir, f"dwell_{name}"), states=VISIBLE_STATES)

        # Sankey diagrams (baseline + trimmed at 10% and 25%)
        sankey_for_sequences(seqs, title=f"Flows — {name}", out_pdf=os.path.join(out_dir, f"sankey_{name}.pdf"), states=VISIBLE_STATES, min_prob=0.0, exclude_self=True)
        sankey_for_sequences(seqs, title=f"Flows (trim <10%) — {name}", out_pdf=os.path.join(out_dir, f"sankey_trim10_{name}.pdf"), states=VISIBLE_STATES, min_prob=0.10, exclude_self=True)
        sankey_for_sequences(seqs, title=f"Flows (trim <25%) — {name}", out_pdf=os.path.join(out_dir, f"sankey_trim25_{name}.pdf"), states=VISIBLE_STATES, min_prob=0.25, exclude_self=True)

        # Circos diagrams with same trimming rules
        circos_for_sequences(seqs, title=f"Circos — {name}", out_pdf=os.path.join(out_dir, f"circos_{name}.pdf"), states=VISIBLE_STATES, min_prob=0.0, exclude_self=True)
        circos_for_sequences(seqs, title=f"Circos (trim <10%) — {name}", out_pdf=os.path.join(out_dir, f"circos_trim10_{name}.pdf"), states=VISIBLE_STATES, min_prob=0.10, exclude_self=True)
        circos_for_sequences(seqs, title=f"Circos (trim <25%) — {name}", out_pdf=os.path.join(out_dir, f"circos_trim25_{name}.pdf"), states=VISIBLE_STATES, min_prob=0.25, exclude_self=True)

        # Difficulty-adjusted transition matrix (if target distribution is available)
        P_adj = None
        Cw_adj = None
        if target_dist is not None and "case_correct_count" in sub.columns:
            # Align groups to the seqs list (skip rows with empty seq)
            aligned_groups: List[Optional[int]] = []
            aligned_seqs: List[List[str]] = []
            for _, rr in sub.iterrows():
                seq = rr.get("seq")
                if not seq:
                    continue
                g = rr.get("case_correct_count")
                aligned_seqs.append(seq)
                try:
                    aligned_groups.append(int(g) if pd.notna(g) else None)
                except Exception:
                    aligned_groups.append(None)
            if aligned_seqs:
                P_adj, Cw_adj = weighted_transition_matrix(aligned_seqs, aligned_groups, target_dist=target_dist, exclude_self=True, states=VISIBLE_STATES)
                # Save adjusted outputs
                pd.DataFrame(Cw_adj, index=VISIBLE_STATES, columns=VISIBLE_STATES).to_csv(os.path.join(out_dir, f"transition_counts_adjusted_{name}.csv"))
                pd.DataFrame(P_adj, index=VISIBLE_STATES, columns=VISIBLE_STATES).to_csv(os.path.join(out_dir, f"transition_probs_adjusted_{name}.csv"))
                heatmap_matrix(P_adj, Cw_adj, title=f"Transition Matrix (difficulty-adjusted) — {name}", out_pdf=os.path.join(out_dir, f"transitions_adjusted_{name}.pdf"), states=VISIBLE_STATES)

        # Trace-level stats
        sub_stats = []
        for _, r in sub.iterrows():
            seq = r["seq"]
            if not seq:
                continue
            counts_dict = {s: seq.count(s) for s in VISIBLE_STATES}
            loops = sum(1 for a, b in zip(seq, seq[1:]) if a == b)
            # sentence/token totals for entire trace (approximate by whole reasoning_trace)
            total_sentences = len(split_sentences(str(r.get("reasoning_trace", ""))))
            total_tokens = count_tokens(str(r.get("reasoning_trace", "")))
            sub_stats.append({
                "pmcid": r.get("pmcid"),
                "sample_index": r.get("sample_index"),
                "verified_correct": r.get("verified_correct"),
                "case_correct_count": r.get("case_correct_count"),
                "trace_len_chunks": len(seq),
                "trace_len_sentences": total_sentences,
                "trace_len_tokens": total_tokens,
                "loops": loops,
                **{f"cnt_{s}": counts_dict[s] for s in VISIBLE_STATES},
            })
        if sub_stats:
            s_df = pd.DataFrame(sub_stats)
            s_df.to_csv(os.path.join(out_dir, f"stats_{name}.csv"), index=False)
            # Correlations with accuracy where applicable
            if "verified_correct" in s_df.columns:
                corr_df = compute_correlations_with_pvalues(s_df, "verified_correct")
                if len(corr_df) > 0:
                    # Apply corrections
                    valid_mask = ~(np.isnan(corr_df["pvalue"]))
                    if valid_mask.sum() > 0:
                        corrections = apply_multiple_testing_corrections(corr_df.loc[valid_mask, "pvalue"].values)
                        corr_df["pvalue_bonferroni"] = np.nan
                        corr_df["pvalue_fdr_bh"] = np.nan
                        corr_df.loc[valid_mask, "pvalue_bonferroni"] = corrections["bonferroni"]
                        corr_df.loc[valid_mask, "pvalue_fdr_bh"] = corrections["fdr_bh"]
                    # Sort by p-value
                    corr_df = corr_df.sort_values("pvalue")
                    corr_df.to_csv(os.path.join(out_dir, f"correlation_accuracy_{name}.csv"), index=False)
                    for _, row in corr_df.iterrows():
                        if pd.notna(row["correlation"]):
                            all_correlations.append({
                                "subset": name,
                                "type": "stats_accuracy_unadjusted",
                                "feature": row["feature"],
                                "correlation": row["correlation"],
                                "pvalue": row["pvalue"],
                                "pvalue_bonferroni": row.get("pvalue_bonferroni", np.nan),
                                "pvalue_fdr_bh": row.get("pvalue_fdr_bh", np.nan),
                            })
            if "case_correct_count" in s_df.columns:
                corr_df = compute_correlations_with_pvalues(s_df, "case_correct_count")
                if len(corr_df) > 0:
                    # Apply corrections
                    valid_mask = ~(np.isnan(corr_df["pvalue"]))
                    if valid_mask.sum() > 0:
                        corrections = apply_multiple_testing_corrections(corr_df.loc[valid_mask, "pvalue"].values)
                        corr_df["pvalue_bonferroni"] = np.nan
                        corr_df["pvalue_fdr_bh"] = np.nan
                        corr_df.loc[valid_mask, "pvalue_bonferroni"] = corrections["bonferroni"]
                        corr_df.loc[valid_mask, "pvalue_fdr_bh"] = corrections["fdr_bh"]
                    # Sort by p-value
                    corr_df = corr_df.sort_values("pvalue")
                    corr_df.to_csv(os.path.join(out_dir, f"correlation_difficulty_{name}.csv"), index=False)
                    for _, row in corr_df.iterrows():
                        if pd.notna(row["correlation"]):
                            all_correlations.append({
                                "subset": name,
                                "type": "stats_difficulty",
                                "feature": row["feature"],
                                "correlation": row["correlation"],
                                "pvalue": row["pvalue"],
                                "pvalue_bonferroni": row.get("pvalue_bonferroni", np.nan),
                                "pvalue_fdr_bh": row.get("pvalue_fdr_bh", np.nan),
                            })

            # Adjusted for difficulty (partial out case_correct_count)
            adj_corr = adjusted_correlations(s_df, os.path.join(out_dir, f"correlation_accuracy_adjusted_{name}.csv"))
            if adj_corr is not None and len(adj_corr) > 0:
                for _, row in adj_corr.iterrows():
                    if pd.notna(row["correlation"]):
                        all_correlations.append({
                            "subset": name,
                            "type": "stats_accuracy_adjusted",
                            "feature": row["feature"],
                            "correlation": row["correlation"],
                            "pvalue": row["pvalue"],
                            "pvalue_bonferroni": row.get("pvalue_bonferroni", np.nan),
                            "pvalue_fdr_bh": row.get("pvalue_fdr_bh", np.nan),
                        })

        # store matrices for difference plots later
        subset_results[name] = {"P": P, "counts": counts}
        if P_adj is not None:
            subset_results[name]["P_adj"] = P_adj

    # Differences between matrices
    def ensure_present(key: str) -> bool:
        return key in subset_results and subset_results[key]["P"].size > 0

    # Differences between matrices
    def ensure_present(key: str) -> bool:
        return key in subset_results and subset_results[key]["P"].size > 0

    # New: Combined Circos Plot for Correct vs Incorrect
    if ensure_present("correct") and ensure_present("incorrect") and plt:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Correct
            seqs_correct = [s for s in subsets["correct"]["seq"].tolist() if s]
            circos_for_sequences(seqs_correct, title="Correct Traces", states=VISIBLE_STATES, ax=axes[0], show_legend=False, exclude_self=True)
            
            # Incorrect
            seqs_incorrect = [s for s in subsets["incorrect"]["seq"].tolist() if s]
            circos_for_sequences(seqs_incorrect, title="Incorrect Traces", states=VISIBLE_STATES, ax=axes[1], show_legend=False, exclude_self=True)
            
            # Add shared legend
            colors = [STATE_COLORS.get(s, "#dddddd") for s in VISIBLE_STATES]
            handles = [
                Line2D([0], [0], color=c, lw=4, label=s)
                for s, c in zip(VISIBLE_STATES, colors)
            ]
            fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.92, 0.5), fontsize=10, title="States")
            plt.suptitle("Flow Comparison: Correct vs Incorrect", fontsize=14)
            plt.subplots_adjust(right=0.9, wspace=0.3)
            
            out_comb = os.path.join(out_dir, "circos_combined_correct_vs_incorrect.pdf")
            fig.savefig(out_comb, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"Error creating combined Correct vs Incorrect plot: {e}")

    # New: Combined Circos Plot for All Models (if 'model' column exists)
    # Check if 'model' column exists in df and create subsets
    model_col = None
    for c in df.columns:
        if c.lower() == "model":
            model_col = c
            break
            
    if model_col and plt:
        models = sorted(df[model_col].unique())
        if len(models) > 1:
            try:
                # Determine grid size
                n_models = len(models)
                cols_grid = 3 if n_models >= 3 else n_models
                rows_grid = (n_models + cols_grid - 1) // cols_grid
                
                fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(5 * cols_grid, 5 * rows_grid + 1))
                axes_flat = axes.flatten() if n_models > 1 else [axes]
                
                for i, m in enumerate(models):
                    ax = axes_flat[i]
                    sub_m = df[df[model_col] == m]
                    seqs_m = [s for s in sub_m["seq"].tolist() if s]
                    if not seqs_m:
                        ax.axis("off")
                        continue
                    circos_for_sequences(seqs_m, title=str(m), states=VISIBLE_STATES, ax=ax, show_legend=False, exclude_self=True)
                
                # Turn off unused axes
                for j in range(i + 1, len(axes_flat)):
                    axes_flat[j].axis("off")
                
                # Add shared legend
                colors = [STATE_COLORS.get(s, "#dddddd") for s in VISIBLE_STATES]
                handles = [
                    Line2D([0], [0], color=c, lw=4, label=s)
                    for s, c in zip(VISIBLE_STATES, colors)
                ]
                fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.92, 0.5), fontsize=10, title="States")
                plt.suptitle("Flow Comparison by Model", fontsize=16)
                plt.subplots_adjust(right=0.9, hspace=0.3, wspace=0.3)
                
                out_comb = os.path.join(out_dir, "circos_combined_all_models.pdf")
                fig.savefig(out_comb, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"Error creating combined All Models plot: {e}")

    # Correct vs Incorrect
    if ensure_present("correct") and ensure_present("incorrect"):
        D = subset_results["correct"]["P"] - subset_results["incorrect"]["P"]
        pd.DataFrame(D, index=VISIBLE_STATES, columns=VISIBLE_STATES).to_csv(os.path.join(out_dir, "transition_diff_probs_correct_vs_incorrect.csv"))
        heatmap_diff(D, title="ΔP: Correct - Incorrect", out_pdf=os.path.join(out_dir, "transitions_diff_correct_vs_incorrect.pdf"), states=VISIBLE_STATES)

        # Adjusted differences (if available)
        if "P_adj" in subset_results.get("correct", {}) and "P_adj" in subset_results.get("incorrect", {}):
            D_adj = subset_results["correct"]["P_adj"] - subset_results["incorrect"]["P_adj"]
            pd.DataFrame(D_adj, index=VISIBLE_STATES, columns=VISIBLE_STATES).to_csv(os.path.join(out_dir, "transition_diff_probs_adjusted_correct_vs_incorrect.csv"))
            heatmap_diff(D_adj, title="ΔP (difficulty-adjusted): Correct - Incorrect", out_pdf=os.path.join(out_dir, "transitions_diff_adjusted_correct_vs_incorrect.pdf"), states=VISIBLE_STATES)

    # Difficulty diffs (if present)
    pairs = [("difficulty_10", "difficulty_0"), ("difficulty_5", "difficulty_0"), ("difficulty_10", "difficulty_5")]
    for a, b in pairs:
        if ensure_present(a) and ensure_present(b):
            D = subset_results[a]["P"] - subset_results[b]["P"]
            pd.DataFrame(D, index=VISIBLE_STATES, columns=VISIBLE_STATES).to_csv(os.path.join(out_dir, f"transition_diff_probs_{a}_minus_{b}.csv"))
            heatmap_diff(D, title=f"ΔP: {a} - {b}", out_pdf=os.path.join(out_dir, f"transitions_diff_{a}_minus_{b}.pdf"), states=VISIBLE_STATES)

    # Per-trace transition features and dwell time features combined for correlations
    trans_rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        seq = r.get("seq")
        if not seq:
            continue
        P_seq = seq_transition_matrix(seq, exclude_self=True, states=VISIBLE_STATES)
        feats = {
            **flatten_matrix(P_seq, prefix="P", states=VISIBLE_STATES),
            **flatten_matrix(P_seq - P_all, prefix="D_all", states=VISIBLE_STATES),
        }
        
        # Add DAG features
        lj = r["label_json"]
        if not isinstance(lj, str):
            lj = json.dumps(lj)
        dag_feats = compute_dag_features(lj)
        feats.update(dag_feats)
        
        # Add aggregated dwell time features (per state)
        runs = dwell_runs_with_durations(lj)
        if runs:
            for run in runs:
                state = run["state"]
                feats[f"dwell_{state}_chunks"] = run["duration_chunks"]
                feats[f"dwell_{state}_sentences"] = run["duration_sentences"]
                feats[f"dwell_{state}_tokens"] = run["duration_tokens"]
        
        trans_rows.append({
            "pmcid": r.get("pmcid"),
            "sample_index": r.get("sample_index"),
            "verified_correct": r.get("verified_correct"),
            "case_correct_count": r.get("case_correct_count"),
            **feats,
        })
    if trans_rows:
        trans_df = pd.DataFrame(trans_rows)
        trans_path = os.path.join(out_dir, "per_trace_transition_features.csv")
        trans_df.to_csv(trans_path, index=False)

        # Unadjusted correlations with accuracy
        if "verified_correct" in trans_df.columns:
            corr_df = compute_correlations_with_pvalues(trans_df, "verified_correct")
            if len(corr_df) > 0:
                # Apply corrections
                valid_mask = ~(np.isnan(corr_df["pvalue"]))
                if valid_mask.sum() > 0:
                    corrections = apply_multiple_testing_corrections(corr_df.loc[valid_mask, "pvalue"].values)
                    corr_df["pvalue_bonferroni"] = np.nan
                    corr_df["pvalue_fdr_bh"] = np.nan
                    corr_df.loc[valid_mask, "pvalue_bonferroni"] = corrections["bonferroni"]
                    corr_df.loc[valid_mask, "pvalue_fdr_bh"] = corrections["fdr_bh"]
                # Sort by p-value
                corr_df = corr_df.sort_values("pvalue")
                corr_df.to_csv(os.path.join(out_dir, "corr_unadjusted_transitions_accuracy.csv"), index=False)
                for _, row in corr_df.iterrows():
                    if pd.notna(row["correlation"]):
                        all_correlations.append({
                            "subset": "all",
                            "type": "transition_accuracy_unadjusted",
                            "feature": row["feature"],
                            "correlation": row["correlation"],
                            "pvalue": row["pvalue"],
                            "pvalue_bonferroni": row.get("pvalue_bonferroni", np.nan),
                            "pvalue_fdr_bh": row.get("pvalue_fdr_bh", np.nan),
                        })

        # Adjusted correlations for difficulty
        adj_corr = adjusted_correlations(trans_df, os.path.join(out_dir, "corr_adjusted_transitions_accuracy.csv"))
        if adj_corr is not None and len(adj_corr) > 0:
            for _, row in adj_corr.iterrows():
                if pd.notna(row["correlation"]):
                    all_correlations.append({
                        "subset": "all",
                        "type": "transition_accuracy_adjusted",
                        "feature": row["feature"],
                        "correlation": row["correlation"],
                        "pvalue": row["pvalue"],
                        "pvalue_bonferroni": row.get("pvalue_bonferroni", np.nan),
                        "pvalue_fdr_bh": row.get("pvalue_fdr_bh", np.nan),
                    })
        
        # Separate correlations for just dwell features
        num_cols = trans_df.select_dtypes(include=[np.number]).columns
        dwell_cols = [c for c in num_cols if c.startswith("dwell_")]
        if dwell_cols and "verified_correct" in trans_df.columns:
            dwell_df = trans_df[dwell_cols + ["verified_correct", "case_correct_count"]].copy()
            corr_df = compute_correlations_with_pvalues(dwell_df, "verified_correct")
            if len(corr_df) > 0:
                # Apply corrections
                valid_mask = ~(np.isnan(corr_df["pvalue"]))
                if valid_mask.sum() > 0:
                    corrections = apply_multiple_testing_corrections(corr_df.loc[valid_mask, "pvalue"].values)
                    corr_df["pvalue_bonferroni"] = np.nan
                    corr_df["pvalue_fdr_bh"] = np.nan
                    corr_df.loc[valid_mask, "pvalue_bonferroni"] = corrections["bonferroni"]
                    corr_df.loc[valid_mask, "pvalue_fdr_bh"] = corrections["fdr_bh"]
                # Sort by p-value
                corr_df = corr_df.sort_values("pvalue")
                corr_df.to_csv(os.path.join(out_dir, "corr_unadjusted_dwell_vs_transitions_accuracy.csv"), index=False)
                for _, row in corr_df.iterrows():
                    if pd.notna(row["correlation"]):
                        all_correlations.append({
                            "subset": "all",
                            "type": "dwell_vs_transition_accuracy_unadjusted",
                            "feature": row["feature"],
                            "correlation": row["correlation"],
                            "pvalue": row["pvalue"],
                            "pvalue_bonferroni": row.get("pvalue_bonferroni", np.nan),
                            "pvalue_fdr_bh": row.get("pvalue_fdr_bh", np.nan),
                        })
            
            # Adjusted dwell correlations
            adj_corr = adjusted_correlations(dwell_df, os.path.join(out_dir, "corr_adjusted_dwell_vs_transitions_accuracy.csv"))
            if adj_corr is not None and len(adj_corr) > 0:
                for _, row in adj_corr.iterrows():
                    if pd.notna(row["correlation"]):
                        all_correlations.append({
                            "subset": "all",
                            "type": "dwell_vs_transition_accuracy_adjusted",
                            "feature": row["feature"],
                            "correlation": row["correlation"],
                            "pvalue": row["pvalue"],
                            "pvalue_bonferroni": row.get("pvalue_bonferroni", np.nan),
                            "pvalue_fdr_bh": row.get("pvalue_fdr_bh", np.nan),
                        })
        
        # DAG-specific correlations
        dag_cols = [c for c in num_cols if c.startswith("dag_")]
        if dag_cols and "verified_correct" in trans_df.columns:
            dag_df = trans_df[dag_cols + ["verified_correct", "case_correct_count"]].copy()
            corr_df = compute_correlations_with_pvalues(dag_df, "verified_correct")
            if len(corr_df) > 0:
                # Apply corrections
                valid_mask = ~(np.isnan(corr_df["pvalue"]))
                if valid_mask.sum() > 0:
                    corrections = apply_multiple_testing_corrections(corr_df.loc[valid_mask, "pvalue"].values)
                    corr_df["pvalue_bonferroni"] = np.nan
                    corr_df["pvalue_fdr_bh"] = np.nan
                    corr_df.loc[valid_mask, "pvalue_bonferroni"] = corrections["bonferroni"]
                    corr_df.loc[valid_mask, "pvalue_fdr_bh"] = corrections["fdr_bh"]
                # Sort by p-value
                corr_df = corr_df.sort_values("pvalue")
                corr_df.to_csv(os.path.join(out_dir, "corr_unadjusted_dag_accuracy.csv"), index=False)
                for _, row in corr_df.iterrows():
                    if pd.notna(row["correlation"]):
                        all_correlations.append({
                            "subset": "all",
                            "type": "dag_accuracy_unadjusted",
                            "feature": row["feature"],
                            "correlation": row["correlation"],
                            "pvalue": row["pvalue"],
                            "pvalue_bonferroni": row.get("pvalue_bonferroni", np.nan),
                            "pvalue_fdr_bh": row.get("pvalue_fdr_bh", np.nan),
                        })
            
            # Adjusted DAG correlations
            adj_corr = adjusted_correlations(dag_df, os.path.join(out_dir, "corr_adjusted_dag_accuracy.csv"))
            if adj_corr is not None and len(adj_corr) > 0:
                for _, row in adj_corr.iterrows():
                    if pd.notna(row["correlation"]):
                        all_correlations.append({
                            "subset": "all",
                            "type": "dag_accuracy_adjusted",
                            "feature": row["feature"],
                            "correlation": row["correlation"],
                            "pvalue": row["pvalue"],
                            "pvalue_bonferroni": row.get("pvalue_bonferroni", np.nan),
                            "pvalue_fdr_bh": row.get("pvalue_fdr_bh", np.nan),
                        })

    # Print all correlations sorted by p-value
    if all_correlations:
        corr_summary = pd.DataFrame(all_correlations)
        corr_summary["abs_correlation"] = corr_summary["correlation"].abs()
        # Sort by p-value (NaN p-values go to end)
        corr_summary["pvalue_for_sort"] = corr_summary["pvalue"].fillna(1.0)
        corr_summary_sorted = corr_summary.sort_values("pvalue_for_sort", ascending=True)
        corr_summary_sorted = corr_summary_sorted.drop("pvalue_for_sort", axis=1)
        
        print("\n" + "="*100)
        print("ALL CORRELATIONS SORTED BY P-VALUE")
        print("="*100)
        print(f"\nTotal correlations computed: {len(corr_summary_sorted)}\n")
        
        # Count significant correlations
        sig_bonf = (corr_summary_sorted["pvalue_bonferroni"] <= 0.05).sum()
        sig_fdr = (corr_summary_sorted["pvalue_fdr_bh"] <= 0.05).sum()
        sig_uncorrected = (corr_summary_sorted["pvalue"] <= 0.05).sum()
        print(f"Significant at p<0.05 (uncorrected): {sig_uncorrected}")
        print(f"Significant at p<0.05 (Bonferroni): {sig_bonf}")
        print(f"Significant at p<0.05 (FDR-BH): {sig_fdr}\n")
        
        # Print in a formatted table
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 80)
        
        display_cols = ["subset", "type", "feature", "correlation", "pvalue", "pvalue_bonferroni", "pvalue_fdr_bh"]
        print(corr_summary_sorted[display_cols].to_string(index=False))
        print("\n" + "="*100 + "\n")
        
        # Save to CSV
        corr_summary_path = os.path.join(out_dir, "all_correlations_sorted.csv")
        corr_summary_sorted.to_csv(corr_summary_path, index=False)
        print(f"All correlations saved to: {corr_summary_path}")

    print(f"\nAnalysis complete. Outputs in {out_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="MCMC-style analysis of labeled traces")
    p.add_argument("--labeled_csv", default="results.labeled.csv")
    p.add_argument("--out_dir", default="analysis_outputs")
    p.add_argument("--difficulty_filter", type=int, nargs=2, default=None, help="Filter traces to only those with case_correct_count in range [min max] (e.g., --difficulty_filter 4 6)")
    args = p.parse_args()

    difficulty_filter_tuple = tuple(args.difficulty_filter) if args.difficulty_filter else None
    run_analysis(labeled_csv=args.labeled_csv, out_dir=args.out_dir, difficulty_filter=difficulty_filter_tuple)
