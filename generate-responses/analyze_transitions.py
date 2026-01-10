#!/usr/bin/env python3
"""
Analyze state transitions in annotated thinking processes.

This script extracts state transitions from annotated thinking, computes statistics,
correlates with correctness, and generates visualizations.
"""
import argparse
import sys
import json
import re
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import dotenv
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

dotenv.load_dotenv("../.env")

# Add parent directory to path for imports
sys.path.append('..')
from utils.utils import load_model
from utils.clustering import get_latent_descriptions

# Pattern matching the annotation format
ANNOTATION_PATTERN = re.compile(r'\["([\d.]+):(\S+?)"\](.*?)\["end-section"\]', re.DOTALL)


def parse_annotated_thinking(annotated_thinking: str) -> List[Tuple[str, str, float]]:
    """
    Parse annotated thinking to extract state segments.
    
    Returns: List of (state_label, text, activation) tuples
    """
    matches = list(ANNOTATION_PATTERN.finditer(annotated_thinking))
    segments = []
    
    for match in matches:
        activation_str = match.group(1).strip()
        state_label = match.group(2).strip()
        text = match.group(3).strip()
        
        try:
            activation = float(activation_str)
        except ValueError:
            continue
            
        if not text:
            continue
            
        segments.append((state_label, text, activation))
    
    return segments


def count_tokens_in_text(text: str, tokenizer) -> int:
    """Count tokens in a text segment using the tokenizer."""
    if not text:
        return 0
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def count_sentences_in_text(text: str) -> int:
    """Count sentences in a text segment."""
    if not text:
        return 0
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?;])', text)
    return len([s for s in sentences if s.strip()])


def analyze_trace(segments: List[Tuple[str, str, float]], tokenizer) -> Dict:
    """
    Analyze a single reasoning trace to extract transition statistics.
    
    Returns a dictionary with:
    - state_sequence: List of states in order
    - tokens_per_segment: Tokens emitted in each segment before transition
    - total_tokens_per_state: Total tokens in each state
    - total_sentences_per_state: Total sentences in each state
    - sentence_count_per_state: Count of sentence segments per state
    - token_count_per_state: Count of token segments per state
    - transitions: List of (from_state, to_state) tuples
    - avg_consecutive_tokens_per_state: Average tokens spent consecutively in each state
    - avg_consecutive_segments_per_state: Average segments spent consecutively in each state
    """
    if not segments:
        return {
            'state_sequence': [],
            'tokens_per_segment': [],
            'total_tokens_per_state': {},
            'total_sentences_per_state': {},
            'sentence_count_per_state': {},
            'token_count_per_state': {},
            'transitions': [],
            'avg_consecutive_tokens_per_state': {},
            'avg_consecutive_segments_per_state': {},
        }
    
    state_sequence = []
    tokens_per_segment = []
    total_tokens_per_state = defaultdict(int)
    total_sentences_per_state = defaultdict(int)
    sentence_count_per_state = defaultdict(int)
    token_count_per_state = defaultdict(int)
    transitions = []
    
    # Track consecutive runs of each state
    consecutive_runs = defaultdict(list)  # state -> [list of (token_count, segment_count)]
    current_run_state = None
    current_run_tokens = 0
    current_run_segments = 0
    
    prev_state = None
    for state_label, text, activation in segments:
        state_sequence.append(state_label)
        
        # Count tokens in this segment
        token_count = count_tokens_in_text(text, tokenizer)
        tokens_per_segment.append(token_count)
        
        # Update state statistics
        total_tokens_per_state[state_label] += token_count
        sentence_count = count_sentences_in_text(text)
        total_sentences_per_state[state_label] += sentence_count
        sentence_count_per_state[state_label] += 1
        token_count_per_state[state_label] += 1
        
        # Track consecutive runs
        if state_label == current_run_state:
            # Continue current run
            current_run_tokens += token_count
            current_run_segments += 1
        else:
            # Save previous run and start new one
            if current_run_state is not None:
                consecutive_runs[current_run_state].append((current_run_tokens, current_run_segments))
            current_run_state = state_label
            current_run_tokens = token_count
            current_run_segments = 1
        
        # Track transitions
        if prev_state is not None and prev_state != state_label:
            transitions.append((prev_state, state_label))
        prev_state = state_label
    
    # Save final run
    if current_run_state is not None:
        consecutive_runs[current_run_state].append((current_run_tokens, current_run_segments))
    
    # Compute averages
    avg_consecutive_tokens = {}
    avg_consecutive_segments = {}
    for state, runs in consecutive_runs.items():
        if runs:
            avg_consecutive_tokens[state] = sum(r[0] for r in runs) / len(runs)
            avg_consecutive_segments[state] = sum(r[1] for r in runs) / len(runs)
    
    return {
        'state_sequence': state_sequence,
        'tokens_per_segment': tokens_per_segment,
        'total_tokens_per_state': dict(total_tokens_per_state),
        'total_sentences_per_state': dict(total_sentences_per_state),
        'sentence_count_per_state': dict(sentence_count_per_state),
        'token_count_per_state': dict(token_count_per_state),
        'transitions': transitions,
        'avg_consecutive_tokens_per_state': avg_consecutive_tokens,
        'avg_consecutive_segments_per_state': avg_consecutive_segments,
    }


def build_transition_matrix(traces: List[Dict], correct_only: Optional[bool] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Build transition matrix from traces.
    
    Args:
        traces: List of trace analysis dictionaries
        correct_only: If True, only use correct traces; if False, only incorrect; if None, use all
    
    Returns:
        (transition_matrix, state_labels)
    """
    # Collect all unique states
    all_states = set()
    for trace in traces:
        all_states.update(trace['state_sequence'])
    
    state_labels = sorted(all_states)
    state_to_idx = {state: idx for idx, state in enumerate(state_labels)}
    
    # Initialize transition count matrix
    transition_counts = defaultdict(int)
    
    # Count transitions
    for trace in traces:
        for from_state, to_state in trace['transitions']:
            if from_state in state_to_idx and to_state in state_to_idx:
                transition_counts[(from_state, to_state)] += 1
    
    # Build matrix
    n_states = len(state_labels)
    matrix = np.zeros((n_states, n_states), dtype=int)
    
    for (from_state, to_state), count in transition_counts.items():
        from_idx = state_to_idx[from_state]
        to_idx = state_to_idx[to_state]
        matrix[from_idx, to_idx] = count
    
    return matrix, state_labels


def compute_transition_probabilities(transition_matrix: np.ndarray) -> np.ndarray:
    """Compute transition probabilities P(B|A) from transition counts."""
    # Add small epsilon to avoid division by zero
    row_sums = transition_matrix.sum(axis=1, keepdims=True) + 1e-10
    probabilities = transition_matrix / row_sums
    return probabilities


def load_graded_responses(model_id: str) -> Dict[str, bool]:
    """
    Load graded responses and return a mapping from question_id to is_correct.
    
    Checks for graded file, if not found, tries to grade responses.
    """
    graded_file = f"results/vars/responses_{model_id}.graded.json"
    responses_file = f"results/vars/responses_{model_id}.json"
    
    # Try to load graded responses
    if os.path.exists(graded_file):
        with open(graded_file, 'r') as f:
            graded_data = json.load(f)
        return {item['question_id']: item.get('is_correct', False) for item in graded_data}
    
    # If not found, check if we can grade
    if os.path.exists(responses_file):
        print(f"Graded responses not found. Please run grade_responses.py first.")
        print(f"Expected file: {graded_file}")
        return {}
    
    return {}


def compute_correlations(traces: List[Dict], correctness: List[bool], transition_matrix: np.ndarray, state_labels: List[str]) -> Dict[str, float]:
    """
    Compute correlations between trace metrics and correctness.
    
    Returns dictionary of metric_name -> correlation_coefficient
    """
    if len(traces) != len(correctness):
        raise ValueError(f"Length mismatch: {len(traces)} traces vs {len(correctness)} correctness labels")
    
    correlations = {}
    
    # Collect all unique states
    all_states = set()
    for trace in traces:
        all_states.update(trace['state_sequence'])
    
    # For each state, compute metrics
    for state in all_states:
        # Total tokens in state
        tokens_in_state = [trace['total_tokens_per_state'].get(state, 0) for trace in traces]
        if any(tokens_in_state):
            corr, pval = stats.pearsonr(tokens_in_state, correctness)
            correlations[f'{state}_total_tokens'] = {'correlation': corr, 'pvalue': pval}
        
        # Total sentences in state
        sentences_in_state = [trace['total_sentences_per_state'].get(state, 0) for trace in traces]
        if any(sentences_in_state):
            corr, pval = stats.pearsonr(sentences_in_state, correctness)
            correlations[f'{state}_total_sentences'] = {'correlation': corr, 'pvalue': pval}
        
        # Count of segments in state
        segment_count = [trace['sentence_count_per_state'].get(state, 0) for trace in traces]
        if any(segment_count):
            corr, pval = stats.pearsonr(segment_count, correctness)
            correlations[f'{state}_segment_count'] = {'correlation': corr, 'pvalue': pval}
        
        # Average consecutive tokens in state
        avg_consec_tokens = [trace['avg_consecutive_tokens_per_state'].get(state, 0) for trace in traces]
        if any(avg_consec_tokens):
            corr, pval = stats.pearsonr(avg_consec_tokens, correctness)
            correlations[f'{state}_avg_consecutive_tokens'] = {'correlation': corr, 'pvalue': pval}
        
        # Average consecutive segments in state
        avg_consec_segments = [trace['avg_consecutive_segments_per_state'].get(state, 0) for trace in traces]
        if any(avg_consec_segments):
            corr, pval = stats.pearsonr(avg_consec_segments, correctness)
            correlations[f'{state}_avg_consecutive_segments'] = {'correlation': corr, 'pvalue': pval}
    
    # Transition probabilities for each state pair
    state_to_idx = {s: i for i, s in enumerate(state_labels)}
    transition_probs = compute_transition_probabilities(transition_matrix)
    
    for from_state in all_states:
        if from_state not in state_to_idx:
            continue
        from_idx = state_to_idx[from_state]
        
        for to_state in all_states:
            if to_state not in state_to_idx:
                continue
            to_idx = state_to_idx[to_state]
            
            # Get transition probability from this pair
            trans_prob = transition_probs[from_idx, to_idx]
            
            # For each trace, check if this transition occurs
            trans_usage = []
            for trace in traces:
                # Count how many times this transition occurs in the trace
                count = sum(1 for t in trace['transitions'] if t == (from_state, to_state))
                trans_usage.append(count)
            
            # Only correlate if this transition occurs in some traces
            if any(trans_usage) and trans_prob > 0:
                corr, pval = stats.pearsonr(trans_usage, correctness)
                correlations[f'transition_{from_state}_to_{to_state}'] = {'correlation': corr, 'pvalue': pval}
    
    # Overall trace metrics
    total_tokens = [sum(trace['total_tokens_per_state'].values()) for trace in traces]
    if any(total_tokens):
        corr, pval = stats.pearsonr(total_tokens, correctness)
        correlations['total_tokens_all_states'] = {'correlation': corr, 'pvalue': pval}
    
    total_sentences = [sum(trace['total_sentences_per_state'].values()) for trace in traces]
    if any(total_sentences):
        corr, pval = stats.pearsonr(total_sentences, correctness)
        correlations['total_sentences_all_states'] = {'correlation': corr, 'pvalue': pval}
    
    num_transitions = [len(trace['transitions']) for trace in traces]
    if any(num_transitions):
        corr, pval = stats.pearsonr(num_transitions, correctness)
        correlations['num_transitions'] = {'correlation': corr, 'pvalue': pval}
    
    trace_length = [len(trace['state_sequence']) for trace in traces]
    if any(trace_length):
        corr, pval = stats.pearsonr(trace_length, correctness)
        correlations['trace_length'] = {'correlation': corr, 'pvalue': pval}
    
    return correlations


def generate_visualizations(transition_matrix_all: np.ndarray,
                           transition_matrix_correct: np.ndarray,
                           transition_matrix_incorrect: np.ndarray,
                           state_labels: List[str],
                           output_prefix: str):
    """Generate visualization plots for transition matrices."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # 1. Overall transition matrix heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(transition_matrix_all, 
                xticklabels=state_labels, 
                yticklabels=state_labels,
                annot=True, 
                fmt='d',
                cmap='Blues',
                cbar_kws={'label': 'Transition Count'})
    plt.title('Overall Transition Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('To State', fontsize=12)
    plt.ylabel('From State', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_overall.png")
    plt.close()
    
    # 2. Correct transition matrix
    if transition_matrix_correct.sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(transition_matrix_correct,
                    xticklabels=state_labels,
                    yticklabels=state_labels,
                    annot=True,
                    fmt='d',
                    cmap='Greens',
                    cbar_kws={'label': 'Transition Count'})
        plt.title('Transition Matrix (Correct Answers)', fontsize=16, fontweight='bold')
        plt.xlabel('To State', fontsize=12)
        plt.ylabel('From State', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_correct.png")
        plt.close()
    
    # 3. Incorrect transition matrix
    if transition_matrix_incorrect.sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(transition_matrix_incorrect,
                    xticklabels=state_labels,
                    yticklabels=state_labels,
                    annot=True,
                    fmt='d',
                    cmap='Reds',
                    cbar_kws={'label': 'Transition Count'})
        plt.title('Transition Matrix (Incorrect Answers)', fontsize=16, fontweight='bold')
        plt.xlabel('To State', fontsize=12)
        plt.ylabel('From State', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_incorrect.png")
        plt.close()
    
    # 4. Difference matrix (correct - incorrect)
    if transition_matrix_correct.shape == transition_matrix_incorrect.shape:
        diff_matrix = transition_matrix_correct.astype(float) - transition_matrix_incorrect.astype(float)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(diff_matrix,
                    xticklabels=state_labels,
                    yticklabels=state_labels,
                    annot=True,
                    fmt='.1f',
                    cmap='RdBu_r',
                    center=0,
                    cbar_kws={'label': 'Difference (Correct - Incorrect)'})
        plt.title('Transition Matrix Difference (Correct - Incorrect)', fontsize=16, fontweight='bold')
        plt.xlabel('To State', fontsize=12)
        plt.ylabel('From State', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_difference.png")
        plt.close()
    
    # 5. Transition probabilities (overall)
    prob_matrix = compute_transition_probabilities(transition_matrix_all)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(prob_matrix,
                xticklabels=state_labels,
                yticklabels=state_labels,
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Transition Probability'})
    plt.title('Transition Probabilities P(To|From)', fontsize=16, fontweight='bold')
    plt.xlabel('To State', fontsize=12)
    plt.ylabel('From State', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_probabilities.png")
    plt.close()


def load_state_labels(model_name: str, layer: int, n_clusters: int) -> Dict[str, str]:
    """
    Load semantic labels for states from SAE clustering results.
    
    Returns a mapping from idx labels (e.g., 'idx0') to semantic labels (e.g., 'stepwise-calculation').
    Falls back to idx labels if semantic labels are not available.
    """
    try:
        descriptions = get_latent_descriptions(model_name, layer, n_clusters, clustering_method='sae_topk', sorted=True)
        
        if descriptions:
            # descriptions is a dict with positions as keys and {'key', 'title', 'description'} as values
            label_map = {}
            for pos, info in descriptions.items():
                idx_key = info['key']  # e.g., 'idx0'
                title = info['title']  # e.g., 'stepwise-calculation'
                label_map[idx_key] = title
            
            print(f"Loaded semantic labels for {len(label_map)} states")
            return label_map
        else:
            print("No semantic labels found in SAE results. Using idx labels.")
            return {}
    except Exception as e:
        print(f"Warning: Could not load semantic labels: {e}")
        print("Using idx labels instead.")
        return {}


def apply_state_labels(state: str, label_map: Dict[str, str]) -> str:
    """Apply semantic label to a state, falling back to idx label if not found.
    Format: 'semantic-label (idx0)' or just 'idx0' if no semantic label."""
    semantic = label_map.get(state, None)
    if semantic and semantic != state:
        return f"{semantic} ({state})"
    return state


def main():
    parser = argparse.ArgumentParser(description="Analyze state transitions in annotated thinking")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Model used to generate responses")
    parser.add_argument("--layer", type=int, default=6,
                        help="Layer to analyze (for loading tokenizer)")
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Number of clusters (for loading semantic labels). If not specified, inferred from annotated data.")
    args = parser.parse_args()
    
    # Get model ID
    model_name = args.model
    model_id = model_name.split('/')[-1].lower()
    
    # File paths
    annotated_file = f"results/vars/annotated_responses_{model_id}.json"
    responses_file = f"results/vars/responses_{model_id}.json"
    
    if not os.path.exists(annotated_file):
        print(f"Error: Annotated responses file not found: {annotated_file}")
        print("Please run annotate_thinking.py first.")
        return
    
    # Load annotated responses
    print(f"Loading annotated responses from {annotated_file}...")
    with open(annotated_file, 'r') as f:
        annotated_data = json.load(f)
    
    # Load original responses to get full_response for token counting
    print(f"Loading original responses from {responses_file}...")
    if os.path.exists(responses_file):
        with open(responses_file, 'r') as f:
            responses_data = json.load(f)
        # Create mapping from question_id to full_response
        response_map = {item['question_id']: item.get('full_response', '') 
                       for item in responses_data}
    else:
        response_map = {}
    
    # Load graded responses for correctness
    print("Loading graded responses...")
    correctness_map = load_graded_responses(model_id)
    
    if not correctness_map:
        print("Warning: No correctness labels found. Some analyses will be skipped.")
        print("Please run grade_responses.py first.")
    
    # Load tokenizer for token counting
    print(f"Loading tokenizer for {model_name}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Analyze traces
    print("Analyzing traces...")
    traces = []
    trace_metadata = []
    
    for item in tqdm(annotated_data, desc="Processing traces"):
        question_id = item['question_id']
        annotated_thinking = item.get('annotated_thinking', '')
        
        if not annotated_thinking:
            continue
        
        # Parse segments
        segments = parse_annotated_thinking(annotated_thinking)
        if not segments:
            continue
        
        # Analyze trace
        trace_analysis = analyze_trace(segments, tokenizer)
        
        # Get correctness
        is_correct = correctness_map.get(question_id, None)
        
        # Store trace
        traces.append(trace_analysis)
        trace_metadata.append({
            'question_id': question_id,
            'is_correct': is_correct,
            'trace_analysis': trace_analysis
        })
    
    if not traces:
        print("No valid traces found. Exiting.")
        return
    
    print(f"Analyzed {len(traces)} traces.")
    
    # Infer n_clusters from the data if not specified
    if args.n_clusters is None:
        # Collect all unique states from traces
        all_states = set()
        for trace in traces:
            all_states.update(trace['state_sequence'])
        # Count states that match idxN pattern
        idx_states = [s for s in all_states if s.startswith('idx') and s[3:].isdigit()]
        if idx_states:
            max_idx = max(int(s[3:]) for s in idx_states)
            inferred_n_clusters = max_idx + 1
            print(f"Inferred n_clusters={inferred_n_clusters} from annotated data")
        else:
            inferred_n_clusters = len(all_states)
            print(f"Could not infer n_clusters from idx pattern. Using {inferred_n_clusters}")
    else:
        inferred_n_clusters = args.n_clusters
    
    # Load semantic labels for states
    print(f"Loading semantic labels for {inferred_n_clusters} clusters...")
    label_map = load_state_labels(model_name, args.layer, inferred_n_clusters)
    
    # Build transition matrices
    print("Building transition matrices...")
    transition_matrix_all, state_labels = build_transition_matrix(traces)
    
    # Apply semantic labels to state_labels
    state_labels_semantic = [apply_state_labels(s, label_map) for s in state_labels]
    print(f"State labels: {', '.join(state_labels_semantic)}")
    
    # Build CSV data now that we know all states
    trace_data = []
    for meta in trace_metadata:
        trace_analysis = meta['trace_analysis']
        row = {
            'question_id': meta['question_id'],
            'is_correct': meta['is_correct'],
            'trace_length': len(trace_analysis['state_sequence']),
            'num_transitions': len(trace_analysis['transitions']),
            'total_tokens': sum(trace_analysis['total_tokens_per_state'].values()),
            'total_sentences': sum(trace_analysis['total_sentences_per_state'].values()),
        }
        # Add per-state metrics with semantic labels
        for state, semantic_label in zip(state_labels, state_labels_semantic):
            row[f'{semantic_label}_tokens'] = trace_analysis['total_tokens_per_state'].get(state, 0)
            row[f'{semantic_label}_sentences'] = trace_analysis['total_sentences_per_state'].get(state, 0)
            row[f'{semantic_label}_segment_count'] = trace_analysis['sentence_count_per_state'].get(state, 0)
            row[f'{semantic_label}_avg_consecutive_tokens'] = trace_analysis['avg_consecutive_tokens_per_state'].get(state, 0)
            row[f'{semantic_label}_avg_consecutive_segments'] = trace_analysis['avg_consecutive_segments_per_state'].get(state, 0)
        trace_data.append(row)
    
    # Separate correct and incorrect traces
    correct_traces = []
    incorrect_traces = []
    correctness_list = []
    
    for meta in trace_metadata:
        is_correct = meta['is_correct']
        trace_analysis = meta['trace_analysis']
        if is_correct is True:
            correct_traces.append(trace_analysis)
            correctness_list.append(1)
        elif is_correct is False:
            incorrect_traces.append(trace_analysis)
            correctness_list.append(0)
        else:
            correctness_list.append(None)
    
    transition_matrix_correct, state_labels_correct = build_transition_matrix(correct_traces) if correct_traces else (np.zeros_like(transition_matrix_all), state_labels)
    transition_matrix_incorrect, state_labels_incorrect = build_transition_matrix(incorrect_traces) if incorrect_traces else (np.zeros_like(transition_matrix_all), state_labels)
    
    print(f"Correct traces: {len(correct_traces)}, states: {state_labels_correct if correct_traces else 'N/A'}")
    print(f"Incorrect traces: {len(incorrect_traces)}, states: {state_labels_incorrect if incorrect_traces else 'N/A'}")
    print(f"Correct transitions: {transition_matrix_correct.sum()}")
    print(f"Incorrect transitions: {transition_matrix_incorrect.sum()}")
    
    # Ensure all matrices have same shape by remapping to the full state space
    if correct_traces and state_labels_correct != state_labels:
        # Remap to full state space
        full_matrix = np.zeros_like(transition_matrix_all)
        state_map = {s: i for i, s in enumerate(state_labels)}
        for i, from_state in enumerate(state_labels_correct):
            for j, to_state in enumerate(state_labels_correct):
                if from_state in state_map and to_state in state_map:
                    full_matrix[state_map[from_state], state_map[to_state]] = transition_matrix_correct[i, j]
        transition_matrix_correct = full_matrix
    elif not correct_traces:
        transition_matrix_correct = np.zeros_like(transition_matrix_all)
    
    if incorrect_traces and state_labels_incorrect != state_labels:
        # Remap to full state space
        full_matrix = np.zeros_like(transition_matrix_all)
        state_map = {s: i for i, s in enumerate(state_labels)}
        for i, from_state in enumerate(state_labels_incorrect):
            for j, to_state in enumerate(state_labels_incorrect):
                if from_state in state_map and to_state in state_map:
                    full_matrix[state_map[from_state], state_map[to_state]] = transition_matrix_incorrect[i, j]
        transition_matrix_incorrect = full_matrix
    elif not incorrect_traces:
        transition_matrix_incorrect = np.zeros_like(transition_matrix_all)
    
    # Compute correlations
    print("Computing correlations...")
    valid_traces = []
    valid_correctness = []
    for meta, is_correct in zip(trace_metadata, correctness_list):
        if is_correct is not None:
            valid_traces.append(meta['trace_analysis'])
            valid_correctness.append(is_correct)
    
    correlations = {}
    if valid_traces and valid_correctness:
        correlations = compute_correlations(valid_traces, valid_correctness, transition_matrix_all, state_labels)
    
    # Generate visualizations
    print("Generating visualizations...")
    output_prefix = f"results/vars/transition_matrix_{model_id}"
    generate_visualizations(transition_matrix_all,
                           transition_matrix_correct,
                           transition_matrix_incorrect,
                           state_labels_semantic,
                           output_prefix)
    
    # Compute transition probabilities
    transition_probabilities = compute_transition_probabilities(transition_matrix_all)
    
    # Calculate accuracy
    accuracy = None
    if len(correct_traces) + len(incorrect_traces) > 0:
        accuracy = len(correct_traces) / (len(correct_traces) + len(incorrect_traces))
    
    # Prepare output data
    output_data = {
        'model_id': model_id,
        'num_traces': len(traces),
        'num_correct': len(correct_traces),
        'num_incorrect': len(incorrect_traces),
        'num_unlabeled': len(traces) - len(correct_traces) - len(incorrect_traces),
        'accuracy': accuracy,
        'num_unique_states': len(state_labels),
        'state_labels': state_labels_semantic,
        'state_labels_idx': state_labels,  # Keep original idx labels for reference
        'transition_matrix_all': transition_matrix_all.tolist(),
        'transition_matrix_correct': transition_matrix_correct.tolist(),
        'transition_matrix_incorrect': transition_matrix_incorrect.tolist(),
        'transition_probabilities': transition_probabilities.tolist(),
        'correlations': correlations,
    }
    
    # Save JSON output
    json_output_file = f"results/vars/transition_analysis_{model_id}.json"
    print(f"Saving JSON output to {json_output_file}...")
    with open(json_output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save CSV output
    csv_output_file = f"results/vars/transition_stats_{model_id}.csv"
    print(f"Saving CSV output to {csv_output_file}...")
    if trace_data:
        df = pd.DataFrame(trace_data)
        df.to_csv(csv_output_file, index=False)
    
    # Generate text report
    report_file = f"results/vars/transition_report_{model_id}.txt"
    print(f"Generating text report to {report_file}...")
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRANSITION MATRIX ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: {model_name}\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Total Traces: {len(traces)}\n")
        f.write(f"Correct Answers: {len(correct_traces)}\n")
        f.write(f"Incorrect Answers: {len(incorrect_traces)}\n")
        f.write(f"Unlabeled: {len(traces) - len(correct_traces) - len(incorrect_traces)}\n")
        if accuracy is not None:
            f.write(f"Accuracy: {accuracy:.2%} ({len(correct_traces)}/{len(correct_traces) + len(incorrect_traces)})\n")
        f.write(f"Unique States: {len(state_labels)}\n")
        f.write(f"States: {', '.join(state_labels_semantic)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("OVERALL TRANSITION MATRIX (Counts)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Rows = From State, Columns = To State\n\n")
        
        # Print transition matrix
        header = " " * 15 + " ".join(f"{s:>15}" for s in state_labels_semantic)
        f.write(header + "\n")
        for i, from_state in enumerate(state_labels_semantic):
            row = f"{from_state:>15} " + " ".join(f"{transition_matrix_all[i, j]:>15}" for j in range(len(state_labels_semantic)))
            f.write(row + "\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("TRANSITION PROBABILITIES P(To|From)\n")
        f.write("=" * 80 + "\n\n")
        
        header = " " * 15 + " ".join(f"{s:>15}" for s in state_labels_semantic)
        f.write(header + "\n")
        for i, from_state in enumerate(state_labels_semantic):
            row = f"{from_state:>15} " + " ".join(f"{transition_probabilities[i, j]:>15.3f}" for j in range(len(state_labels_semantic)))
            f.write(row + "\n")
        
        # Add difference matrix info
        if len(correct_traces) > 0 and len(incorrect_traces) > 0:
            diff_matrix = transition_matrix_correct.astype(float) - transition_matrix_incorrect.astype(float)
            f.write("\n" + "=" * 80 + "\n")
            f.write("TRANSITION MATRIX DIFFERENCE (Correct - Incorrect)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total transitions in correct: {transition_matrix_correct.sum()}\n")
            f.write(f"Total transitions in incorrect: {transition_matrix_incorrect.sum()}\n")
            f.write(f"Max positive difference: {diff_matrix.max():.1f}\n")
            f.write(f"Max negative difference: {diff_matrix.min():.1f}\n")
            f.write(f"Mean absolute difference: {np.abs(diff_matrix).mean():.2f}\n\n")
            
            header = " " * 15 + " ".join(f"{s:>15}" for s in state_labels_semantic)
            f.write(header + "\n")
            for i, from_state in enumerate(state_labels_semantic):
                row = f"{from_state:>15} " + " ".join(f"{diff_matrix[i, j]:>15.1f}" for j in range(len(state_labels_semantic)))
                f.write(row + "\n")
        
        if correlations:
            f.write("\n" + "=" * 80 + "\n")
            f.write("CORRELATIONS WITH CORRECTNESS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'Metric':<40} {'Correlation':>12} {'P-value':>12}\n")
            f.write("-" * 80 + "\n")
            
            for metric, stats_dict in sorted(correlations.items()):
                corr = stats_dict.get('correlation', 0)
                pval = stats_dict.get('pvalue', 1.0)
                f.write(f"{metric:<40} {corr:>12.4f} {pval:>12.4e}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")
        
        # Find strongest correlations
        if correlations:
            sorted_corrs = sorted(correlations.items(), 
                                key=lambda x: abs(x[1].get('correlation', 0)),
                                reverse=True)
            f.write("Strongest correlations with correctness:\n")
            for metric, stats_dict in sorted_corrs[:10]:
                corr = stats_dict.get('correlation', 0)
                pval = stats_dict.get('pvalue', 1.0)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                f.write(f"  {metric}: {corr:.4f} (p={pval:.4e}) {sig}\n")
        
        f.write("\nVisualizations saved:\n")
        f.write(f"  - {output_prefix}_overall.png\n")
        f.write(f"  - {output_prefix}_correct.png\n")
        f.write(f"  - {output_prefix}_incorrect.png\n")
        f.write(f"  - {output_prefix}_difference.png\n")
        f.write(f"  - {output_prefix}_probabilities.png\n")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total traces analyzed: {len(traces)}")
    print(f"Correct: {len(correct_traces)}, Incorrect: {len(incorrect_traces)}, Unlabeled: {len(traces) - len(correct_traces) - len(incorrect_traces)}")
    if accuracy is not None:
        print(f"Accuracy: {accuracy:.2%}")
    print(f"Unique states found: {len(state_labels)}")
    print(f"States: {', '.join(state_labels_semantic)}")
    print(f"\nOutputs:")
    print(f"  JSON: {json_output_file}")
    print(f"  CSV: {csv_output_file}")
    print(f"  Report: {report_file}")
    print(f"  Visualizations: {output_prefix}_*.png")


if __name__ == "__main__":
    main()

