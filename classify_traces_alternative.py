#!/usr/bin/env python3
"""Alternative classifier focusing on features that ARE likely to be predictive.

Since state distributions and transitions are NOT different between correct/incorrect,
we focus on:
1. Sequence-level patterns (order, structure)
2. Case difficulty features
3. Model-specific patterns
4. Sequence length and complexity
5. State sequence patterns (not just transitions)

Usage:
    python classify_traces_alternative.py \
        --labeled_csv analysis_runs/combined/all_models_analysis/results.labeled.combined.csv \
        --out_dir alternative_classifier_results
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import analyze_traces_state_transition as transition_analysis

TAXONOMY_CATEGORIES = [
    "initialization",
    "case_setup",
    "hypothesis_generation",
    "hypothesis_weighing",
    "stored_medical_knowledge",
    "case_interpretation",
    "prompting_reconsideration",
    "diagnostic_commitment",
    "final_answer_emission",
]


def extract_sequence_structure_features(sequence: List[str]) -> Dict[str, float]:
    """Extract structural features about the sequence pattern."""
    if not sequence:
        return {
            "has_initialization": 0.0,
            "has_diagnostic_commitment": 0.0,
            "has_final_answer": 0.0,
            "num_state_changes": 0.0,
            "max_consecutive_same": 0.0,
            "sequence_entropy": 0.0,
        }
    
    features = {
        "has_initialization": 1.0 if "initialization" in sequence else 0.0,
        "has_diagnostic_commitment": 1.0 if "diagnostic_commitment" in sequence else 0.0,
        "has_final_answer": 1.0 if "final_answer_emission" in sequence else 0.0,
    }
    
    # Count state changes
    state_changes = sum(1 for i in range(len(sequence) - 1) if sequence[i] != sequence[i+1])
    features["num_state_changes"] = float(state_changes)
    
    # Max consecutive same state
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    features["max_consecutive_same"] = float(max_consecutive)
    
    # Sequence entropy (diversity measure)
    state_counts = Counter(sequence)
    total = len(sequence)
    entropy = -sum((count/total) * np.log2(count/total + 1e-10) for count in state_counts.values())
    features["sequence_entropy"] = entropy
    
    return features


def extract_sequence_order_features(sequence: List[str]) -> Dict[str, float]:
    """Extract features about the order/structure of states."""
    if not sequence:
        return {}
    
    features = {}
    
    # Position of key states
    key_states = ["initialization", "diagnostic_commitment", "final_answer_emission"]
    for state in key_states:
        if state in sequence and len(sequence) > 0:
            pos = sequence.index(state)
            features[f"position_{state}"] = float(pos) / len(sequence)  # Normalized position
            features[f"early_{state}"] = 1.0 if pos < len(sequence) / 3 else 0.0
            features[f"late_{state}"] = 1.0 if pos > 2 * len(sequence) / 3 else 0.0
        else:
            features[f"position_{state}"] = 1.0  # Not present = end
            features[f"early_{state}"] = 0.0
            features[f"late_{state}"] = 0.0
    
    # Check for "proper" reasoning flow
    # Good flow: initialization -> case_setup -> hypothesis -> weighing -> commitment -> answer
    has_init = "initialization" in sequence
    has_setup = "case_setup" in sequence
    has_hyp = "hypothesis_generation" in sequence or "hypothesis_weighing" in sequence
    has_commit = "diagnostic_commitment" in sequence
    has_answer = "final_answer_emission" in sequence
    
    # Check order
    if has_init and has_setup:
        init_pos = sequence.index("initialization")
        setup_pos = sequence.index("case_setup")
        features["init_before_setup"] = 1.0 if init_pos < setup_pos else 0.0
    else:
        features["init_before_setup"] = 0.0
    
    if has_commit and has_answer:
        commit_pos = sequence.index("diagnostic_commitment")
        answer_pos = sequence.index("final_answer_emission")
        features["commit_before_answer"] = 1.0 if commit_pos < answer_pos else 0.0
    else:
        features["commit_before_answer"] = 0.0
    
    # Count how many key stages are present
    key_stages = sum([has_init, has_setup, has_hyp, has_commit, has_answer])
    features["key_stages_present"] = float(key_stages) / 5.0
    
    return features


def extract_repetition_features(sequence: List[str]) -> Dict[str, float]:
    """Extract features about repetition and cycling."""
    if not sequence or len(sequence) < 2:
        return {
            "repetition_rate": 0.0,
            "cycles": 0.0,
            "backtracking": 0.0,
        }
    
    # Repetition: same state appears multiple times
    state_counts = Counter(sequence)
    repeated_states = sum(1 for count in state_counts.values() if count > 1)
    features = {
        "repetition_rate": float(repeated_states) / len(state_counts) if state_counts else 0.0,
    }
    
    # Cycles: state A -> B -> A
    cycles = 0
    for i in range(len(sequence) - 2):
        if sequence[i] == sequence[i+2] and sequence[i] != sequence[i+1]:
            cycles += 1
    features["cycles"] = float(cycles) / (len(sequence) - 2) if len(sequence) > 2 else 0.0
    
    # Backtracking: going back to earlier states (simplified)
    # Count how many times we see a state that appeared earlier
    seen_states = set()
    backtrack_count = 0
    for state in sequence:
        if state in seen_states:
            backtrack_count += 1
        seen_states.add(state)
    features["backtracking"] = float(backtrack_count) / len(sequence) if sequence else 0.0
    
    return features


def extract_state_sequence_patterns(sequence: List[str], max_pattern_length: int = 4) -> Dict[str, float]:
    """Extract specific state sequence patterns (subsequences)."""
    if not sequence:
        return {}
    
    features = {}
    
    # Common patterns that might indicate good/bad reasoning
    patterns = [
        # Good patterns
        (["case_setup", "hypothesis_generation"], "setup_to_hypothesis"),
        (["hypothesis_generation", "hypothesis_weighing"], "hyp_to_weighing"),
        (["hypothesis_weighing", "diagnostic_commitment"], "weighing_to_commitment"),
        (["diagnostic_commitment", "final_answer_emission"], "commitment_to_answer"),
        
        # Potentially problematic patterns
        (["diagnostic_commitment", "hypothesis_generation"], "commitment_to_hypothesis"),  # Changing mind after commitment
        (["final_answer_emission", "hypothesis_generation"], "answer_to_hypothesis"),  # Generating hypotheses after answer
        (["case_setup", "case_setup"], "setup_repetition"),
        (["hypothesis_weighing", "hypothesis_weighing"], "weighing_repetition"),
    ]
    
    for pattern, name in patterns:
        count = 0
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                count += 1
        features[f"pattern_{name}"] = float(count) / max(1, len(sequence) - len(pattern) + 1)
    
    return features


def prepare_alternative_features(
    labeled_csv: str,
) -> Tuple[pd.DataFrame, pd.Series, List[List[str]], pd.DataFrame]:
    """Prepare features focusing on what might actually be predictive."""
    df = pd.read_csv(labeled_csv)
    
    # Coerce verified_correct
    def _to_bool(x):
        if isinstance(x, bool):
            return x
        if pd.isna(x):
            return None
        s = str(x).strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
        return None
    
    if "verified_correct" in df.columns:
        vc = df["verified_correct"].apply(_to_bool)
        try:
            df["verified_correct"] = vc.astype("boolean")
        except Exception:
            df["verified_correct"] = vc
    
    # Extract sequences and metadata
    all_sequences = []
    all_labels = []
    all_metadata = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting sequences"):
        label_json = row.get("label_json")
        if pd.isna(label_json):
            continue
        
        try:
            sequence = transition_analysis.extract_sequence(str(label_json))
            sequence = [s for s in sequence if s != "other"]
        except Exception:
            continue
        
        if not sequence:
            continue
        
        true_correct = row.get("verified_correct")
        if pd.isna(true_correct):
            continue
        
        all_sequences.append(sequence)
        all_labels.append(bool(true_correct))
        
        # Collect metadata
        metadata = {
            "pmcid": row.get("pmcid"),
            "sample_index": row.get("sample_index"),
            "model": row.get("model", "unknown"),
            "sequence_length": len(sequence),
        }
        all_metadata.append(metadata)
        valid_indices.append(idx)
    
    # Extract features
    all_features = []
    
    for sequence in tqdm(all_sequences, desc="Computing features"):
        features = {
            # Basic sequence features
            "sequence_length": float(len(sequence)),
            "num_unique_states": float(len(set(sequence))),
            "state_diversity": float(len(set(sequence))) / len(sequence) if sequence else 0.0,
            
            # Structural features
            **extract_sequence_structure_features(sequence),
            **extract_sequence_order_features(sequence),
            **extract_repetition_features(sequence),
            **extract_state_sequence_patterns(sequence),
            
            # Proportion features (might still be useful in combination)
            **{f"proportion_{cat}": sequence.count(cat) / len(sequence) if sequence else 0.0
               for cat in TAXONOMY_CATEGORIES},
        }
        
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    labels_series = pd.Series(all_labels, index=valid_indices)
    metadata_df = pd.DataFrame(all_metadata, index=valid_indices)
    
    # Add case-level features if available
    if "pmcid" in metadata_df.columns:
        # Group by pmcid to get case-level stats
        case_stats = df.groupby("pmcid").agg({
            "verified_correct": lambda x: x.astype(int).sum() if x.notna().any() else 0,
        }).reset_index()
        case_stats.columns = ["pmcid", "case_correct_count"]
        
        # Merge with metadata
        metadata_df = metadata_df.merge(case_stats, on="pmcid", how="left")
        features_df["case_correct_count"] = metadata_df["case_correct_count"].fillna(0).values
    
    # NOTE: Model features are NOT included - we want to predict correctness regardless of model
    
    # Handle NaN values - fill with 0 for numeric features, drop columns that are all NaN
    print(f"\nHandling NaN values...")
    print(f"  Features before cleaning: {len(features_df.columns)}")
    print(f"  NaN count per column:")
    nan_counts = features_df.isna().sum()
    for col, count in nan_counts[nan_counts > 0].items():
        print(f"    {col}: {count} NaNs")
    
    # Drop columns that are all NaN
    features_df = features_df.dropna(axis=1, how='all')
    
    # Fill remaining NaN values with 0 (for numeric features) or median
    for col in features_df.columns:
        if features_df[col].isna().any():
            if features_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                # Fill with median for numeric columns
                median_val = features_df[col].median()
                if pd.isna(median_val):
                    features_df[col] = features_df[col].fillna(0.0)
                else:
                    features_df[col] = features_df[col].fillna(median_val)
            else:
                # Fill with 0 for other types
                features_df[col] = features_df[col].fillna(0.0)
    
    print(f"  Features after cleaning: {len(features_df.columns)}")
    print(f"  Remaining NaNs: {features_df.isna().sum().sum()}")
    
    return features_df, labels_series, all_sequences, metadata_df


def train_ensemble_classifier(
    features_df: pd.DataFrame,
    labels: pd.Series,
    sequences: Optional[List[List[str]]] = None,
    metadata_df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train an ensemble of classifiers including interpretable models.
    
    Args:
        features_df: Feature matrix
        labels: Target labels
        sequences: Optional sequence data for ROCKET
        metadata_df: Optional metadata DataFrame with 'pmcid' column for case-based splitting
        test_size: Test set size
        random_state: Random seed
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
    )
    
    # Ensure no NaN values
    if features_df.isna().any().any():
        print("Warning: Found NaN values, filling with 0...")
        features_df = features_df.fillna(0.0)
    
    # Convert to numeric, replacing any non-numeric with 0
    for col in features_df.columns:
        if features_df[col].dtype == 'object':
            try:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0)
            except Exception:
                features_df[col] = 0.0
    
    X = features_df.values.astype(float)
    y = labels.values
    
    # Final check for NaN/inf
    if np.isnan(X).any() or np.isinf(X).any():
        print("Warning: Found NaN/Inf in X, replacing...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Split by case (pmcid) if metadata is available, otherwise by sample
    # This prevents data leakage when multiple samples exist per case
    train_mask = None
    test_mask = None
    split_by_case = False
    
    if metadata_df is not None and 'pmcid' in metadata_df.columns:
        print("\nSplitting by case (pmcid) to prevent data leakage...")
        case_ids = metadata_df['pmcid'].unique()
        
        # Get case-level labels for stratification
        case_labels = {}
        for case_id in case_ids:
            case_mask = metadata_df['pmcid'] == case_id
            case_indices = metadata_df[case_mask].index
            # Map metadata indices to feature indices (they should align)
            case_y = y[case_indices]
            case_label = case_y.mode() if len(case_y) > 0 else pd.Series([0])
            if len(case_label) > 0:
                case_labels[case_id] = case_label.iloc[0]
            else:
                case_labels[case_id] = 0  # Default to incorrect if no label
        
        case_label_array = np.array([case_labels[cid] for cid in case_ids])
        
        # Split cases
        train_cases, test_cases = train_test_split(
            case_ids, test_size=test_size, random_state=random_state,
            stratify=case_label_array if len(np.unique(case_label_array)) > 1 else None
        )
        
        # Create masks for samples (align with features_df index)
        train_mask = metadata_df['pmcid'].isin(train_cases).values
        test_mask = metadata_df['pmcid'].isin(test_cases).values
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        split_by_case = True
        
        # Verify no case overlap
        train_pmcids = set(metadata_df[train_mask]['pmcid'])
        test_pmcids = set(metadata_df[test_mask]['pmcid'])
        overlap = train_pmcids & test_pmcids
        if overlap:
            raise ValueError(f"Data leakage detected: {len(overlap)} cases in both train and test!")
        
        print(f"  Train: {len(train_cases)} cases, {len(X_train)} samples")
        print(f"  Test: {len(test_cases)} cases, {len(X_test)} samples")
        print(f"  Train accuracy: {y_train.mean():.4f}, Test accuracy: {y_test.mean():.4f}")
    else:
        print("\nSplitting by sample (no case metadata available)...")
        # Fallback to sample-based split if no pmcid available
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        if metadata_df is not None and 'pmcid' in metadata_df.columns:
            # Warn if we have pmcid but didn't use it
            print("  WARNING: pmcid available but not used for splitting - potential data leakage!")
    
    # Scale for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    models_to_save = {}
    confusion_matrices = {}
    all_probas = {}  # Store probabilities for ensemble
    
    # 1. Random Forest
    print("  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    all_probas["random_forest"] = y_proba_rf
    
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    confusion_matrices["random_forest"] = cm_rf
    
    results["random_forest"] = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf, zero_division=0),
        "recall": recall_score(y_test, y_pred_rf, zero_division=0),
        "f1": f1_score(y_test, y_pred_rf, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba_rf) if len(np.unique(y_test)) > 1 else 0.0,
        "feature_importance": dict(zip(features_df.columns, rf.feature_importances_)),
    }
    models_to_save["random_forest"] = rf
    
    # 2. Gradient Boosting
    print("  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state,
    )
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    y_proba_gb = gb.predict_proba(X_test)[:, 1]
    all_probas["gradient_boosting"] = y_proba_gb
    
    cm_gb = confusion_matrix(y_test, y_pred_gb)
    confusion_matrices["gradient_boosting"] = cm_gb
    
    results["gradient_boosting"] = {
        "accuracy": accuracy_score(y_test, y_pred_gb),
        "precision": precision_score(y_test, y_pred_gb, zero_division=0),
        "recall": recall_score(y_test, y_pred_gb, zero_division=0),
        "f1": f1_score(y_test, y_pred_gb, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba_gb) if len(np.unique(y_test)) > 1 else 0.0,
        "feature_importance": dict(zip(features_df.columns, gb.feature_importances_)),
    }
    models_to_save["gradient_boosting"] = gb
    
    # 3. Logistic Regression (scaled)
    print("  Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=0.1,  # Stronger regularization
        random_state=random_state,
    )
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    all_probas["logistic_regression"] = y_proba_lr
    
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    confusion_matrices["logistic_regression"] = cm_lr
    
    results["logistic_regression"] = {
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "precision": precision_score(y_test, y_pred_lr, zero_division=0),
        "recall": recall_score(y_test, y_pred_lr, zero_division=0),
        "f1": f1_score(y_test, y_pred_lr, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba_lr) if len(np.unique(y_test)) > 1 else 0.0,
        "feature_importance": dict(zip(features_df.columns, lr.coef_[0])),
    }
    models_to_save["logistic_regression"] = lr
    
    # 4. Decision Tree (interpretable)
    print("  Training Decision Tree...")
    dt = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=random_state,
    )
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_proba_dt = dt.predict_proba(X_test)[:, 1]
    all_probas["decision_tree"] = y_proba_dt
    
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    confusion_matrices["decision_tree"] = cm_dt
    
    results["decision_tree"] = {
        "accuracy": accuracy_score(y_test, y_pred_dt),
        "precision": precision_score(y_test, y_pred_dt, zero_division=0),
        "recall": recall_score(y_test, y_pred_dt, zero_division=0),
        "f1": f1_score(y_test, y_pred_dt, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba_dt) if len(np.unique(y_test)) > 1 else 0.0,
        "feature_importance": dict(zip(features_df.columns, dt.feature_importances_)),
    }
    models_to_save["decision_tree"] = dt
    
    # 5. ExplainableBoostingMachine (interpretml)
    print("  Training ExplainableBoostingMachine...")
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        ebm = ExplainableBoostingClassifier(
            random_state=random_state,
            n_jobs=-1,
        )
        ebm.fit(X_train, y_train)
        y_pred_ebm = ebm.predict(X_test)
        y_proba_ebm = ebm.predict_proba(X_test)[:, 1]
        all_probas["explainable_boosting"] = y_proba_ebm
        
        cm_ebm = confusion_matrix(y_test, y_pred_ebm)
        confusion_matrices["explainable_boosting"] = cm_ebm
        
        # EBM uses term_importances(), not feature_importances_
        # Get global feature importance from term importances
        try:
            # EBM provides term_importances() which returns a dict
            term_imps = ebm.term_importances()
            # Extract feature-level importance (sum over all terms involving each feature)
            feature_imp_dict = {}
            for feat_name in features_df.columns:
                # Sum importance of all terms that include this feature
                feat_imp = 0.0
                if isinstance(term_imps, dict):
                    for term_name, imp in term_imps.items():
                        # Term names can be strings (single features) or tuples (interactions)
                        if isinstance(term_name, str):
                            if term_name == feat_name:
                                feat_imp += abs(imp)
                        elif isinstance(term_name, (tuple, list)):
                            # Interaction term - include if feature is in it
                            if feat_name in term_name:
                                feat_imp += abs(imp) / len(term_name)  # Divide by interaction size
                        elif term_name == feat_name:
                            feat_imp += abs(imp)
                feature_imp_dict[feat_name] = feat_imp
        except Exception as e:
            # Fallback: use a simple measure if term_importances doesn't work
            print(f"    Warning: Could not extract EBM feature importance: {e}")
            feature_imp_dict = {feat: 0.0 for feat in features_df.columns}
        
        results["explainable_boosting"] = {
            "accuracy": accuracy_score(y_test, y_pred_ebm),
            "precision": precision_score(y_test, y_pred_ebm, zero_division=0),
            "recall": recall_score(y_test, y_pred_ebm, zero_division=0),
            "f1": f1_score(y_test, y_pred_ebm, zero_division=0),
            "auc": roc_auc_score(y_test, y_proba_ebm) if len(np.unique(y_test)) > 1 else 0.0,
            "feature_importance": feature_imp_dict,
        }
        models_to_save["explainable_boosting"] = ebm
    except ImportError:
        print("    Warning: interpretml not installed, skipping ExplainableBoostingMachine")
        print("    Install with: pip install interpret")
    
    # 6. Ridge Classifier (linear, interpretable)
    print("  Training Ridge Classifier...")
    ridge = RidgeClassifier(
        alpha=1.0,
        class_weight='balanced',
        random_state=random_state,
    )
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    # Ridge doesn't have predict_proba, use decision function
    y_score_ridge = ridge.decision_function(X_test_scaled)
    y_proba_ridge = 1 / (1 + np.exp(-y_score_ridge))  # Sigmoid transform
    
    cm_ridge = confusion_matrix(y_test, y_pred_ridge)
    confusion_matrices["ridge_classifier"] = cm_ridge
    
    # Handle coef_ shape - RidgeClassifier.coef_ is 1D for binary classification
    coef = ridge.coef_
    # For binary classification, coef_ is shape (n_features,)
    # For multi-class, it's shape (n_classes, n_features) or (n_classes-1, n_features)
    if coef.ndim == 1:
        coef_values = coef
    elif coef.ndim == 2:
        # Take first row for binary (shape is (1, n_features))
        coef_values = coef[0] if coef.shape[0] == 1 else coef.flatten()
    else:
        coef_values = np.array(coef).flatten()
    
    # Ensure we have the right number of coefficients
    n_features = len(features_df.columns)
    if len(coef_values) != n_features:
        print(f"    Warning: coef shape mismatch: coef shape {coef.shape}, got {len(coef_values)} values, expected {n_features}")
        # Pad or truncate to match
        if len(coef_values) < n_features:
            coef_values = np.pad(coef_values, (0, n_features - len(coef_values)), mode='constant')
        else:
            coef_values = coef_values[:n_features]
    
    results["ridge_classifier"] = {
        "accuracy": accuracy_score(y_test, y_pred_ridge),
        "precision": precision_score(y_test, y_pred_ridge, zero_division=0),
        "recall": recall_score(y_test, y_pred_ridge, zero_division=0),
        "f1": f1_score(y_test, y_pred_ridge, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba_ridge) if len(np.unique(y_test)) > 1 else 0.0,
        "feature_importance": dict(zip(features_df.columns, coef_values)),
    }
    models_to_save["ridge_classifier"] = ridge
    
    # 7. ROCKET + Ridge (aeon-based, interpretable)
    # ROCKET converts sequences to features, then Ridge is interpretable
    if sequences is not None and len(sequences) > 0:
        print("  Training ROCKET + Ridge (aeon-based)...")
        try:
            from aeon.transformations.collection.convolution_based import Rocket
            from sklearn.linear_model import RidgeClassifierCV
            
            # Convert sequences to fixed-length format for ROCKET
            # We need to one-hot encode sequences and pad to same length
            max_len = max(len(seq) for seq in sequences) if sequences else 1
            n_states = len(TAXONOMY_CATEGORIES)
            
            # Create one-hot encoded sequences
            def sequence_to_onehot(seq, max_length, n_states):
                state_to_idx = {cat: i for i, cat in enumerate(TAXONOMY_CATEGORIES)}
                onehot = np.zeros((max_length, n_states))
                for i, state in enumerate(seq[:max_length]):
                    if state in state_to_idx:
                        onehot[i, state_to_idx[state]] = 1.0
                return onehot
            
            X_rocket_train = np.array([sequence_to_onehot(seq, max_len, n_states) for seq in sequences])
            # ROCKET expects shape (n_samples, n_channels, n_timepoints)
            # We have (n_samples, n_timepoints, n_channels), so transpose
            X_rocket_train = np.transpose(X_rocket_train, (0, 2, 1))
            
            # Split for ROCKET (use same split as main models)
            # If we split by case above, use the same train/test masks
            if split_by_case and train_mask is not None and test_mask is not None:
                # Use the same case-based split
                X_rocket_train_split = X_rocket_train[train_mask]
                X_rocket_test_split = X_rocket_train[test_mask]
            else:
                # Fallback to sample-based split
                train_indices, test_indices = train_test_split(
                    np.arange(len(X_rocket_train)), test_size=test_size, random_state=random_state, stratify=y
                )
                X_rocket_train_split = X_rocket_train[train_indices]
                X_rocket_test_split = X_rocket_train[test_indices]
            
            # Apply ROCKET transformation
            rocket = Rocket(num_kernels=1000, random_state=random_state)
            X_rocket_features_train = rocket.fit_transform(X_rocket_train_split)
            X_rocket_features_test = rocket.transform(X_rocket_test_split)
            
            # Combine ROCKET features with existing features
            X_combined_train = np.hstack([X_train, X_rocket_features_train])
            X_combined_test = np.hstack([X_test, X_rocket_features_test])
            
            # Scale combined features
            scaler_rocket = StandardScaler()
            X_combined_train_scaled = scaler_rocket.fit_transform(X_combined_train)
            X_combined_test_scaled = scaler_rocket.transform(X_combined_test)
            
            # Train Ridge classifier (interpretable)
            rocket_ridge = RidgeClassifierCV(
                alphas=[0.1, 1.0, 10.0],
                class_weight='balanced',
            )
            rocket_ridge.fit(X_combined_train_scaled, y_train)
            y_pred_rocket = rocket_ridge.predict(X_combined_test_scaled)
            y_score_rocket = rocket_ridge.decision_function(X_combined_test_scaled)
            y_proba_rocket = 1 / (1 + np.exp(-y_score_rocket))
            all_probas["rocket_ridge"] = y_proba_rocket
            
            cm_rocket = confusion_matrix(y_test, y_pred_rocket)
            confusion_matrices["rocket_ridge"] = cm_rocket
            
            # Handle coef_ shape for ROCKET ridge
            rocket_coef = rocket_ridge.coef_
            if rocket_coef.ndim == 1:
                rocket_coef_values = rocket_coef
            elif rocket_coef.ndim == 2:
                rocket_coef_values = rocket_coef[0] if rocket_coef.shape[0] == 1 else rocket_coef.flatten()
            else:
                rocket_coef_values = np.array(rocket_coef).flatten()
            
            # Combine feature names
            all_feature_names = list(features_df.columns) + [f"rocket_{i}" for i in range(X_rocket_features_train.shape[1])]
            if len(rocket_coef_values) != len(all_feature_names):
                # Adjust to match
                if len(rocket_coef_values) < len(all_feature_names):
                    rocket_coef_values = np.pad(rocket_coef_values, (0, len(all_feature_names) - len(rocket_coef_values)), mode='constant')
                else:
                    rocket_coef_values = rocket_coef_values[:len(all_feature_names)]
            
            results["rocket_ridge"] = {
                "accuracy": accuracy_score(y_test, y_pred_rocket),
                "precision": precision_score(y_test, y_pred_rocket, zero_division=0),
                "recall": recall_score(y_test, y_pred_rocket, zero_division=0),
                "f1": f1_score(y_test, y_pred_rocket, zero_division=0),
                "auc": roc_auc_score(y_test, y_proba_rocket) if len(np.unique(y_test)) > 1 else 0.0,
                "feature_importance": dict(zip(all_feature_names, np.abs(rocket_coef_values))),
            }
            models_to_save["rocket_ridge"] = rocket_ridge
            models_to_save["rocket_transformer"] = rocket
            models_to_save["rocket_scaler"] = scaler_rocket
            print(f"    ROCKET extracted {X_rocket_features_train.shape[1]} features from sequences")
        except ImportError:
            print("    Warning: aeon not installed, skipping ROCKET classifier")
            print("    Install with: pip install aeon")
        except Exception as e:
            print(f"    Warning: ROCKET classifier failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Skipping ROCKET (no sequences provided)")
    
    # 8. Ensemble (voting)
    print("  Creating ensemble...")
    # Manual ensemble: average probabilities from best models
    base_models = ["random_forest", "gradient_boosting", "logistic_regression", "decision_tree"]
    proba_list = [all_probas[model] for model in base_models if model in all_probas]
    weights = [0.3, 0.3, 0.2, 0.2][:len(proba_list)]
    
    # Add optional models
    if "explainable_boosting" in all_probas:
        proba_list.append(all_probas["explainable_boosting"])
        weights = [0.25, 0.25, 0.2, 0.15, 0.15]  # Adjust weights
    
    if "rocket_ridge" in all_probas:
        proba_list.append(all_probas["rocket_ridge"])
        # Rebalance weights
        n_models = len(proba_list)
        weights = [1.0 / n_models] * n_models
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    
    y_proba_ensemble = np.average(proba_list, axis=0, weights=weights)
    y_pred_ensemble = (y_proba_ensemble > 0.5).astype(int)
    
    cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
    confusion_matrices["ensemble"] = cm_ensemble
    
    results["ensemble"] = {
        "accuracy": accuracy_score(y_test, y_pred_ensemble),
        "precision": precision_score(y_test, y_pred_ensemble, zero_division=0),
        "recall": recall_score(y_test, y_pred_ensemble, zero_division=0),
        "f1": f1_score(y_test, y_pred_ensemble, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba_ensemble) if len(np.unique(y_test)) > 1 else 0.0,
    }
    
    return {
        "results": results,
        "models": models_to_save,
        "scaler": scaler,
        "feature_names": list(features_df.columns),
        "confusion_matrices": confusion_matrices,
        "X_test": X_test,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
    }


def main():
    ap = argparse.ArgumentParser(description="Alternative classifier focusing on predictive features")
    ap.add_argument(
        "--labeled_csv",
        required=True,
        help="Path to labeled CSV file",
    )
    ap.add_argument(
        "--out_dir",
        default="alternative_classifier_results",
        help="Output directory",
    )
    ap.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set size",
    )
    ap.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state",
    )
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Preparing alternative features...")
    features_df, labels, sequences, metadata_df = prepare_alternative_features(args.labeled_csv)
    
    print(f"\nPrepared {len(features_df)} traces with {len(features_df.columns)} features")
    print(f"  Correct: {labels.sum()}")
    print(f"  Incorrect: {(~labels).sum()}")
    print(f"  Feature columns: {list(features_df.columns)[:10]}...")
    
    print("\n" + "="*80)
    print("TRAINING ENSEMBLE CLASSIFIERS")
    print("="*80)
    
    results = train_ensemble_classifier(
        features_df, labels, sequences, metadata_df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for model_name, metrics in results["results"].items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        
        if "feature_importance" in metrics:
            top_features = sorted(metrics["feature_importance"].items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            print(f"  Top features:")
            for feat, importance in top_features:
                print(f"    {feat}: {importance:.4f}")
    
    # Save confusion matrices
    print("\nSaving confusion matrices...")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    
    confusion_dir = os.path.join(args.out_dir, "confusion_matrices")
    os.makedirs(confusion_dir, exist_ok=True)
    
    for model_name, cm in results["confusion_matrices"].items():
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Incorrect", "Correct"])
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
        ax.set_title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
        plt.tight_layout()
        plt.savefig(os.path.join(confusion_dir, f"{model_name}_confusion_matrix.png"), dpi=150)
        plt.close()
        print(f"  Saved {model_name} confusion matrix")
    
    # Save models
    print("\nSaving trained models...")
    models_dir = os.path.join(args.out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    from joblib import dump
    
    # Save each model individually
    for model_name, model in results["models"].items():
        model_path = os.path.join(models_dir, f"{model_name}.joblib")
        dump(model, model_path)
        print(f"  Saved {model_name} -> {model_path}")
    
    # Save scaler
    if "scaler" in results:
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        dump(results["scaler"], scaler_path)
        print(f"  Saved scaler -> {scaler_path}")
    
    # Save all models together for easy loading
    all_models = {
        "models": results["models"],
        "scaler": results.get("scaler"),
        "feature_names": results["feature_names"],
    }
    all_models_path = os.path.join(args.out_dir, "classifier_models.joblib")
    dump(all_models, all_models_path)
    print(f"  Saved all models -> {all_models_path}")
    
    # Save results
    summary = {
        "results": {k: {m: v for m, v in metrics.items() if m != "feature_importance"} 
                    for k, metrics in results["results"].items()},
        "top_features": {
            model: sorted(metrics["feature_importance"].items(), key=lambda x: abs(x[1]), reverse=True)[:20]
            for model, metrics in results["results"].items()
            if "feature_importance" in metrics
        },
        "confusion_matrices": {
            model: cm.tolist() for model, cm in results["confusion_matrices"].items()
        },
    }
    
    with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    features_df.to_csv(os.path.join(args.out_dir, "features.csv"), index=False)
    labels.to_csv(os.path.join(args.out_dir, "labels.csv"), index=True)
    
    print(f"\nSaved all results to {args.out_dir}")
    print(f"  - Models: {models_dir}/")
    print(f"  - Confusion matrices: {confusion_dir}/")
    print(f"  - Summary: results_summary.json")


if __name__ == "__main__":
    main()

