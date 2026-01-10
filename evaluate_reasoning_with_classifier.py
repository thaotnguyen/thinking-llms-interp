#!/usr/bin/env python3
"""Evaluate accuracy improvement using classifier-guided re-reasoning.

This script:
1. Generates test set responses using deepseek-r1-distill-llama-70b
2. For each case, performs a multi-turn conversation:
   - First turn: initial diagnosis
   - Second turn: re-prompt saying it's wrong, get another diagnosis
3. Labels both reasoning traces using the taxonomy
4. Uses a trained classifier to predict correctness of the first trace
5. Computes 3 accuracies:
   - Accuracy of all first answers
   - Accuracy of all second answers
   - Accuracy using classifier guidance (first if classifier thinks correct, else second)

IMPORTANT: The classifier MUST be trained on the TRAIN split using the exact same
approach as classify_traces_alternative.py. The classifier is loaded from
--classifier_dir/classifier_models.pkl, or trained from --labeled_csv (which must
be from the train split).

Caching: Labeled traces and graded responses are saved to results.json and automatically
reloaded on subsequent runs to avoid redoing expensive API calls.

Usage:
    # First, train classifier on train split:
    python classify_traces_alternative.py \
        --labeled_csv <train_split_labeled.csv> \
        --out_dir alternative_classifier_results
    
    # Then evaluate on test split:
    python evaluate_reasoning_with_classifier.py \
        --dataset tmknguyen/MedCaseReasoning-filtered \
        --split test \
        --classifier_dir alternative_classifier_results \
        --out_dir re_reasoning_eval_results
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from joblib import load

# Setup paths
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_root)

# Import local modules
import analyze_traces_state_transition as transition_analysis
import classify_traces_alternative as classifier_module
import label_traces as labeler

# Import grading function - try eval.py first, fallback to grade_responses
try:
    from eval import verify_three_step
except ImportError:
    try:
        import grade_responses as grader
        def verify_three_step(case_prompt, predicted, true_diag, model="gpt-5-nano"):
            # Fallback implementation using grade_responses
            return grader.grade_item({}, {"case_prompt": case_prompt, "predicted": predicted, "true": true_diag}, model)
    except ImportError:
        # Last resort: simple string matching
        def verify_three_step(case_prompt, predicted, true_diag, model="gpt-5-nano"):
            pred_norm = predicted.lower().strip()
            true_norm = true_diag.lower().strip()
            is_correct = pred_norm == true_norm or pred_norm in true_norm or true_norm in pred_norm
            return is_correct, "", "", 8.0 if is_correct else 0.0, ""

# Model configuration (from multi_model_pipeline.py)
MODELS = [
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    # "FreedomIntelligence/HuatuoGPT-o1-8B",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "qwen/qwq-32b",
    "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-r1",
]

# Mapping from HuggingFace model names to OpenRouter model names
OPENROUTER_MODEL_MAPPING = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "deepseek/deepseek-r1-distill-qwen-1.5b",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek/deepseek-r1-distill-llama-8b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "deepseek/deepseek-r1-distill-qwen-14b",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "qwen/qwen3-32b": "qwen/qwen3-32b",
    "qwen/qwq-32b": "qwen/qwq-32b",
    "deepseek/deepseek-r1-distill-llama-70b": "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-r1": "deepseek/deepseek-r1",
    "FreedomIntelligence/HuatuoGPT-o1-8B": "freedomintelligence/huatuogpt-o1-8b",
}


def short_name(model_id: str) -> str:
    """Get short name for model (for file paths)."""
    model_id = (model_id or "").strip().replace("/", "_").replace(" ", "_")
    return model_id[-60:] if len(model_id) > 60 else model_id


def get_openrouter_model(model_id: str) -> str:
    """Get OpenRouter model name from HuggingFace model ID."""
    return OPENROUTER_MODEL_MAPPING.get(model_id, model_id)

# Prompt templates (matching multi_model_pipeline.py - uses <think>)
PROMPT_TEMPLATE = (
    "Read the following case presentation and give the most likely diagnosis.\n"
    "First, provide your internal reasoning for the diagnosis within the tags <think> ... </think>.\n"
    "Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>.\n\n"
    "----------------------------------------\nCASE PRESENTATION\n----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\nOUTPUT TEMPLATE\n----------------------------------------\n"
    "<think>\n...your internal reasoning for the diagnosis...\n</think><answer>\n...the name of the disease/entity...\n</answer>"
)

RE_REASONING_PROMPT_TEMPLATE = (
    "You previously provided a diagnosis for this case, but that diagnosis was incorrect.\n"
    "Please reconsider the case and provide a different diagnosis.\n"
    "First, provide your internal reasoning for the new diagnosis within the tags <think> ... </think>.\n"
    "Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>.\n\n"
    "----------------------------------------\nCASE PRESENTATION\n----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\nOUTPUT TEMPLATE\n----------------------------------------\n"
    "<think>\n...your internal reasoning for the new diagnosis...\n</think><answer>\n...the name of the disease/entity...\n</answer>"
)

# Regex patterns (matching multi_model_pipeline.py)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def exists_nonempty(path: str) -> bool:
    """Check if file exists and is non-empty."""
    return os.path.isfile(path) and os.path.getsize(path) > 0


def ensure_json_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable Python types.
    
    Compatible with both NumPy 1.x and 2.x (avoids deprecated types).
    """
    # Check for numpy boolean types
    if isinstance(obj, np.bool_):
        return bool(obj)
    # Check for numpy integer types (np.int_ removed in NumPy 2.0, use np.integer)
    elif isinstance(obj, np.integer):
        return int(obj)
    # Check for numpy float types (np.float_ removed in NumPy 2.0, use np.floating)
    elif isinstance(obj, np.floating):
        return float(obj)
    # Check for numpy number type (fallback)
    elif isinstance(obj, np.number):
        # Try to convert based on whether it's integer or float-like
        try:
            if np.issubdtype(type(obj), np.integer):
                return int(obj)
            elif np.issubdtype(type(obj), np.floating):
                return float(obj)
        except Exception:
            pass
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    try:
        # Check if it's a pandas NA value
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def extract_answer(text: str) -> str:
    """Extract the last <answer>...</answer> block."""
    if not text:
        return ""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return ""
    ans = matches[-1].strip()
    placeholder = "...the name of the disease/entity..."
    if placeholder.lower() in ans.lower():
        return ""
    return ans


def extract_think(text: str) -> str:
    """Extract the last <think>...</think> block."""
    if not text:
        return ""
    matches = THINK_RE.findall(text)
    if not matches:
        return ""
    think = matches[-1].strip()
    placeholder = "...your internal reasoning for the diagnosis..."
    if placeholder.lower() in think.lower():
        return ""
    return think


def generate_with_openrouter(
    messages: List[Dict[str, str]],
    model_id: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    retries: int = 3,
) -> str:
    """Generate response using OpenRouter API."""
    import openai
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    
    backoff = 1.0
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and getattr(choice.message, "content", None):
                    return choice.message.content
                elif hasattr(choice, "text"):
                    return choice.text
            return ""
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2
    
    return ""


def generate_two_diagnoses(
    case_prompt: str,
    model_id: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Tuple[str, str, str, str]:
    """Generate first and second diagnoses for a case.
    
    Returns:
        (first_reasoning, first_answer, second_reasoning, second_answer)
    """
    # First turn
    first_prompt = PROMPT_TEMPLATE.format(case_prompt=case_prompt.strip())
    first_messages = [{"role": "user", "content": first_prompt}]
    first_response = generate_with_openrouter(first_messages, model_id, temperature, max_tokens)
    first_reasoning = extract_think(first_response)
    first_answer = extract_answer(first_response)
    
    # Second turn - neutral re-prompt to get alternative reasoning
    # NOTE: We don't claim the first answer is wrong, as it might be correct.
    # This allows the model to provide an alternative perspective without bias.
    second_prompt = RE_REASONING_PROMPT_TEMPLATE.format(case_prompt=case_prompt.strip())
    second_messages = [
        {"role": "user", "content": first_prompt},
        {"role": "assistant", "content": first_response},
        {"role": "user", "content": second_prompt},
    ]
    second_response = generate_with_openrouter(second_messages, model_id, temperature, max_tokens)
    second_reasoning = extract_think(second_response)
    second_answer = extract_answer(second_response)
    
    return first_reasoning, first_answer, second_reasoning, second_answer


def label_trace(
    case_prompt: str,
    reasoning_trace: str,
    cache_dir: str,
    workers: int = 100,
) -> Optional[str]:
    """Label a single reasoning trace using the taxonomy.
    
    Returns:
        JSON string of labeled trace, or None if labeling fails
    """
    if not reasoning_trace or not reasoning_trace.strip():
        return None
    
    # Use label_traces functionality
    # We'll create a temporary CSV row, label it, and extract the label_json
    # Match the exact format used in multi_model_pipeline.py convert_to_results()
    import tempfile
    import csv
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8', newline='') as tmp_csv:
        # Use the same fieldnames as multi_model_pipeline.py convert_to_results()
        fieldnames = [
            "pmcid", "sample_index", "prompt_edit", "prompt_insert", "case_prompt",
            "diagnostic_reasoning", "true_diagnosis", "predicted_diagnosis",
            "reasoning_trace", "posthoc_reasoning_trace", "verification_response",
            "verification_similarity", "verified_correct"
        ]
        writer = csv.DictWriter(tmp_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "pmcid": "temp",
            "sample_index": 0,  # Use int like multi_model_pipeline.py
            "prompt_edit": "",  # Empty string like multi_model_pipeline.py
            "prompt_insert": "",  # Empty string like multi_model_pipeline.py
            "case_prompt": case_prompt,
            "diagnostic_reasoning": None,  # None like multi_model_pipeline.py
            "true_diagnosis": "",  # Not needed for labeling, but required column
            "predicted_diagnosis": "",  # Not needed for labeling, but required column
            "reasoning_trace": reasoning_trace,
            "posthoc_reasoning_trace": "",  # Empty string like multi_model_pipeline.py
            "verification_response": "",  # Not needed for labeling, but required column
            "verification_similarity": "",  # Empty string like multi_model_pipeline.py
            "verified_correct": False,  # Use boolean False like multi_model_pipeline.py
        })
        tmp_csv_path = tmp_csv.name
    
    try:
        # Label using label_traces
        # Suppress tqdm output by monkey-patching it in label_traces module
        ensure_dir(cache_dir)
        labeled_jsonl = os.path.join(cache_dir, "temp_labeled.jsonl")
        labeled_csv = os.path.join(cache_dir, "temp_labeled.csv")
        
        # Temporarily patch tqdm in label_traces to be a no-op
        import label_traces
        original_tqdm = getattr(label_traces, 'tqdm', None)
        
        # Create a no-op tqdm wrapper that just returns the iterable
        class SilentTqdm:
            def __init__(self, iterable=None, *args, **kwargs):
                self.iterable = iterable if iterable is not None else (args[0] if args else [])
                self.total = kwargs.get('total', len(self.iterable) if hasattr(self.iterable, '__len__') else None)
                self.desc = kwargs.get('desc', '')
                self.unit = kwargs.get('unit', 'it')
            
            def __iter__(self):
                return iter(self.iterable)
            
            def __enter__(self):
                return self
            
            def __exit__(self, *args):
                return False
            
            def update(self, n=1):
                pass
            
            def close(self):
                pass
        
        # Patch tqdm in label_traces module
        if original_tqdm:
            label_traces.tqdm = SilentTqdm
        
        try:
            labeler.label_csv(
                csv_path=tmp_csv_path,
                out_jsonl=labeled_jsonl,
                out_csv=labeled_csv,
                out_html=None,
                cache_dir=cache_dir,
                resume=False,
                workers=workers,
                max_tokens=8000,
                model="deepseek-chat",
            )
        finally:
            # Restore original tqdm
            if original_tqdm:
                label_traces.tqdm = original_tqdm
        
        # Read back the labeled result
        if os.path.exists(labeled_csv):
            df = pd.read_csv(labeled_csv)
            if len(df) > 0 and "label_json" in df.columns:
                label_json = df.iloc[0]["label_json"]
                if pd.notna(label_json):
                    return str(label_json)
    except Exception as exc:
        print(f"Warning: Labeling failed: {exc}", file=sys.stderr)
    finally:
        # Cleanup temp files
        try:
            if os.path.exists(tmp_csv_path):
                os.unlink(tmp_csv_path)
            if os.path.exists(labeled_jsonl):
                os.unlink(labeled_jsonl)
            if os.path.exists(labeled_csv):
                os.unlink(labeled_csv)
        except Exception:
            pass
    
    return None


def extract_features_from_labeled_trace(label_json: str) -> Optional[Dict[str, float]]:
    """Extract features from a labeled trace for classifier input."""
    if not label_json:
        return None
    
    try:
        # Parse label_json if it's a string
        if isinstance(label_json, str):
            import json
            try:
                # Try to parse as JSON first to validate
                parsed = json.loads(label_json)
                label_json_str = label_json
            except json.JSONDecodeError:
                # If not valid JSON, use as-is (might already be parsed)
                label_json_str = str(label_json)
        else:
            # Already parsed, convert to string for extract_sequence
            import json
            label_json_str = json.dumps(label_json)
        
        sequence = transition_analysis.extract_sequence(label_json_str)
        sequence = [s for s in sequence if s != "other"]
    except Exception as exc:
        # Debug: log the exception for troubleshooting
        print(f"Warning: Failed to extract sequence from label_json: {exc}", file=sys.stderr)
        return None
    
    if not sequence:
        print(f"Warning: Empty sequence after extraction (label_json length: {len(str(label_json))})", file=sys.stderr)
        return None
    
    # Extract features using same functions as classify_traces_alternative.py
    try:
        features = {
            "sequence_length": float(len(sequence)),
            "num_unique_states": float(len(set(sequence))),
            "state_diversity": float(len(set(sequence))) / len(sequence) if sequence else 0.0,
            
            **classifier_module.extract_sequence_structure_features(sequence),
            **classifier_module.extract_sequence_order_features(sequence),
            **classifier_module.extract_repetition_features(sequence),
            **classifier_module.extract_state_sequence_patterns(sequence),
            
            **{f"proportion_{cat}": sequence.count(cat) / len(sequence) if sequence else 0.0
               for cat in classifier_module.TAXONOMY_CATEGORIES},
        }
    except Exception as exc:
        print(f"Warning: Failed to extract features from sequence: {exc}", file=sys.stderr)
        return None
    
    return features


def load_all_classifiers(classifier_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all trained classifiers from classify_traces_alternative.py output.
    
    Returns:
        Dictionary mapping model_name -> {"model": ..., "scaler": ..., "feature_names": ..., "needs_scaling": bool, "is_rocket": bool}
    """
    models_dir = os.path.join(classifier_dir, "models")
    all_models_path = os.path.join(classifier_dir, "classifier_models.joblib")
    
    classifiers = {}
    
    # Try to load the combined models file first
    if os.path.exists(all_models_path):
        try:
            saved_data = load(all_models_path)
            all_models = saved_data.get("models", {})
            shared_scaler = saved_data.get("scaler")
            shared_feature_names = saved_data.get("feature_names", [])
            
            print(f"Loaded combined models from {all_models_path}")
            print(f"  Found {len(all_models)} models")
            print(f"  Feature names: {len(shared_feature_names)}")
            
            # Determine which models need scaling
            models_need_scaling = {
                "logistic_regression", "ridge_classifier", "rocket_ridge"
            }
            
            for model_name, model in all_models.items():
                if model_name == "rocket_transformer" or model_name == "rocket_scaler":
                    continue  # Skip these, they're stored separately
                
                is_rocket = model_name == "rocket_ridge"
                needs_scaling = model_name in models_need_scaling
                
                # For ROCKET, we need the transformer and scaler too
                if is_rocket:
                    rocket_transformer = all_models.get("rocket_transformer")
                    rocket_scaler = all_models.get("rocket_scaler")
                    classifiers[model_name] = {
                        "model": model,
                        "scaler": shared_scaler,  # Main scaler for base features
                        "rocket_transformer": rocket_transformer,
                        "rocket_scaler": rocket_scaler,  # Scaler for combined features
                        "feature_names": shared_feature_names,
                        "needs_scaling": needs_scaling,
                        "is_rocket": True,
                    }
                else:
                    classifiers[model_name] = {
                        "model": model,
                        "scaler": shared_scaler if needs_scaling else None,
                        "feature_names": shared_feature_names,
                        "needs_scaling": needs_scaling,
                        "is_rocket": False,
                    }
            
            return classifiers
        except Exception as exc:
            print(f"Warning: Failed to load combined models: {exc}", file=sys.stderr)
    
    # Fallback: try loading individual models
    if os.path.exists(models_dir):
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        shared_scaler = None
        if os.path.exists(scaler_path):
            try:
                shared_scaler = load(scaler_path)
            except Exception:
                pass
        
        # Load feature names from results_summary if available
        shared_feature_names = []
        results_summary_path = os.path.join(classifier_dir, "results_summary.json")
        if os.path.exists(results_summary_path):
            try:
                with open(results_summary_path, "r") as f:
                    summary = json.load(f)
                    # Try to extract feature names from top_features
                    if "top_features" in summary and summary["top_features"]:
                        first_model = list(summary["top_features"].keys())[0]
                        shared_feature_names = [feat for feat, _ in summary["top_features"][first_model]]
            except Exception:
                pass
        
        # Load individual model files
        model_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib") and f != "scaler.joblib"]
        
        models_need_scaling = {
            "logistic_regression", "ridge_classifier", "rocket_ridge"
        }
        
        for model_file in model_files:
            model_name = model_file.replace(".joblib", "")
            if model_name in ["rocket_transformer", "rocket_scaler"]:
                continue
            
            try:
                model_path = os.path.join(models_dir, model_file)
                model = load(model_path)
                
                is_rocket = model_name == "rocket_ridge"
                needs_scaling = model_name in models_need_scaling
                
                # For ROCKET, try to load transformer and scaler
                if is_rocket:
                    rocket_transformer_path = os.path.join(models_dir, "rocket_transformer.joblib")
                    rocket_scaler_path = os.path.join(models_dir, "rocket_scaler.joblib")
                    rocket_transformer = load(rocket_transformer_path) if os.path.exists(rocket_transformer_path) else None
                    rocket_scaler = load(rocket_scaler_path) if os.path.exists(rocket_scaler_path) else None
                    
                    classifiers[model_name] = {
                        "model": model,
                        "scaler": shared_scaler,
                        "rocket_transformer": rocket_transformer,
                        "rocket_scaler": rocket_scaler,
                        "feature_names": shared_feature_names,
                        "needs_scaling": needs_scaling,
                        "is_rocket": True,
                    }
                else:
                    classifiers[model_name] = {
                        "model": model,
                        "scaler": shared_scaler if needs_scaling else None,
                        "feature_names": shared_feature_names,
                        "needs_scaling": needs_scaling,
                        "is_rocket": False,
                    }
            except Exception as exc:
                print(f"Warning: Failed to load {model_file}: {exc}", file=sys.stderr)
    
    if not classifiers:
        raise RuntimeError(
            f"No classifiers found in {classifier_dir}. "
            f"Please train classifiers first using classify_traces_alternative.py"
        )
    
    print(f"Loaded {len(classifiers)} classifiers: {list(classifiers.keys())}")
    return classifiers


def predict_trace_correctness(
    label_json: str,
    classifier_info: Dict[str, Any],
    debug: bool = False,
) -> Tuple[bool, float]:
    """Predict correctness of a trace using the classifier.
    
    Args:
        label_json: Labeled trace JSON string
        classifier_info: Dictionary with "model", "scaler", "feature_names", "needs_scaling", "is_rocket", etc.
    
    Returns:
        (is_correct_predicted, confidence_probability)
    """
    model = classifier_info["model"]
    scaler = classifier_info.get("scaler")
    feature_names = classifier_info["feature_names"]
    needs_scaling = classifier_info.get("needs_scaling", False)
    is_rocket = classifier_info.get("is_rocket", False)
    
    features = extract_features_from_labeled_trace(label_json)
    if features is None:
        if debug:
            print(f"Warning: Could not extract features from label_json", file=sys.stderr)
        return False, 0.0
    
    # Build feature vector matching training order
    feature_vec = np.array([features.get(name, 0.0) for name in feature_names], dtype=float)
    
    feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)
    feature_vec = feature_vec.reshape(1, -1)
    
    # Check feature vector shape matches
    if feature_vec.shape[1] != len(feature_names):
        if debug:
            print(f"Error: Feature vector shape mismatch: got {feature_vec.shape[1]} features, expected {len(feature_names)}", file=sys.stderr)
        return False, 0.0
    
    try:
        # Handle ROCKET model (needs sequence transformation)
        if is_rocket:
            rocket_transformer = classifier_info.get("rocket_transformer")
            rocket_scaler = classifier_info.get("rocket_scaler")
            
            if rocket_transformer is None:
                if debug:
                    print("Warning: ROCKET transformer not available", file=sys.stderr)
                return False, 0.0
            
            # Extract sequence for ROCKET
            try:
                sequence = transition_analysis.extract_sequence(str(label_json))
                sequence = [s for s in sequence if s != "other"]
            except Exception:
                if debug:
                    print("Warning: Could not extract sequence for ROCKET", file=sys.stderr)
                return False, 0.0
            
            # Convert sequence to one-hot (same as training)
            from classify_traces_alternative import TAXONOMY_CATEGORIES
            max_len = 200  # Use a reasonable max length
            n_states = len(TAXONOMY_CATEGORIES)
            state_to_idx = {cat: i for i, cat in enumerate(TAXONOMY_CATEGORIES)}
            onehot = np.zeros((max_len, n_states))
            for i, state in enumerate(sequence[:max_len]):
                if state in state_to_idx:
                    onehot[i, state_to_idx[state]] = 1.0
            
            # ROCKET expects (n_samples, n_channels, n_timepoints)
            X_rocket = np.transpose(onehot.reshape(1, max_len, n_states), (0, 2, 1))
            
            # Transform with ROCKET
            X_rocket_features = rocket_transformer.transform(X_rocket)
            
            # Combine with base features
            X_combined = np.hstack([feature_vec, X_rocket_features])
            
            # Scale combined features
            if rocket_scaler:
                X_combined_scaled = rocket_scaler.transform(X_combined)
            else:
                X_combined_scaled = X_combined
            
            # Predict
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_combined_scaled)[0, 1]
            else:
                # RidgeClassifier - use decision function
                y_score = model.decision_function(X_combined_scaled)
                prob = 1 / (1 + np.exp(-y_score[0]))
        else:
            # Regular model
            if needs_scaling and scaler:
                feature_vec_scaled = scaler.transform(feature_vec)
            else:
                feature_vec_scaled = feature_vec
            
            # Predict
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(feature_vec_scaled)[0, 1]
            else:
                # RidgeClassifier - use decision function
                y_score = model.decision_function(feature_vec_scaled)
                prob = 1 / (1 + np.exp(-y_score[0]))
        
        is_correct = prob > 0.5
        
        if debug:
            print(f"Prediction: is_correct={is_correct}, prob={prob:.4f}", file=sys.stderr)
        
        return is_correct, prob
    except Exception as exc:
        if debug:
            print(f"Error during prediction: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        return False, 0.0


def grade_answer(predicted: str, true_diagnosis: str, case_prompt: str = "") -> bool:
    """Grade if predicted diagnosis matches true diagnosis using verify_three_step."""
    if not predicted or not true_diagnosis:
        return False
    
    try:
        # Use the three-step verification from eval.py
        is_correct, _, _, score, _ = verify_three_step(
            case_prompt=case_prompt or "Medical case",
            predicted_diagnosis=predicted,
            true_diagnosis=true_diagnosis,
            model="gpt-5-nano",
        )
        return is_correct
    except Exception as exc:
        print(f"Warning: Grading failed: {exc}", file=sys.stderr)
        # Fallback: simple string comparison (normalized)
        pred_norm = predicted.lower().strip()
        true_norm = true_diagnosis.lower().strip()
        return pred_norm == true_norm or pred_norm in true_norm or true_norm in pred_norm


def evaluate_classifier_on_dataset(
    labeled_csv: str,
    classifier_name: str,
    classifier_info: Dict[str, Any],
    out_dir: str,
    dataset_name: str = "dataset",
) -> Dict[str, Any]:
    """Evaluate a classifier on a labeled dataset and generate confusion matrix.
    
    Returns:
        Dictionary with metrics and confusion matrix
    """
    print(f"\nEvaluating {classifier_name} on {dataset_name}...")
    
    df = pd.read_csv(labeled_csv)
    
    # Extract sequences and labels
    all_labels = []
    all_predictions = []
    all_probas = []
    all_label_jsons = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {classifier_name}"):
        label_json = row.get("label_json")
        if pd.isna(label_json):
            continue
        
        true_correct = row.get("verified_correct")
        if pd.isna(true_correct):
            continue
        
        try:
            is_correct_pred, prob = predict_trace_correctness(str(label_json), classifier_info, debug=False)
            all_labels.append(bool(true_correct))
            all_predictions.append(is_correct_pred)
            all_probas.append(prob)
            all_label_jsons.append(str(label_json))
        except Exception as exc:
            print(f"Warning: Prediction failed for row {idx}: {exc}", file=sys.stderr)
            continue
    
    if not all_labels:
        print(f"Warning: No valid predictions for {classifier_name} on {dataset_name}")
        return {}
    
    # Compute metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probas)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Save confusion matrix plot
    confusion_dir = os.path.join(out_dir, "confusion_matrices", dataset_name)
    os.makedirs(confusion_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Incorrect", "Correct"])
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    ax.set_title(f'Confusion Matrix - {classifier_name.replace("_", " ").title()} ({dataset_name})')
    plt.tight_layout()
    cm_path = os.path.join(confusion_dir, f"{classifier_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Saved confusion matrix to {cm_path}")
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "n_samples": len(all_labels),
        "n_correct_true": int(y_true.sum()),
        "n_incorrect_true": int((~y_true).sum()),
        "n_correct_pred": int(y_pred.sum()),
        "n_incorrect_pred": int((~y_pred).sum()),
    }


def compute_accuracy_with_ci(correct: List[bool], alpha: float = 0.05) -> Tuple[float, float, float]:
    """Compute accuracy with confidence interval using normal approximation.
    
    Returns:
        (accuracy, lower_bound, upper_bound)
    """
    if not correct:
        return 0.0, 0.0, 0.0
    
    n = len(correct)
    acc = sum(correct) / n if n > 0 else 0.0
    
    # Wilson score interval for binomial proportion
    import scipy.stats as stats
    z = stats.norm.ppf(1 - alpha / 2)
    
    # Normal approximation (OK for large n)
    se = np.sqrt(acc * (1 - acc) / n) if n > 0 else 0.0
    margin = z * se
    lower = max(0.0, acc - margin)
    upper = min(1.0, acc + margin)
    
    return acc, lower, upper


def run_evaluation_for_model(
    model_id: str,
    args: argparse.Namespace,
    classifier_info: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run evaluation for a single model.
    
    Returns:
        Dictionary with model name and summary statistics
    """
    model_short = short_name(model_id)
    openrouter_model = get_openrouter_model(model_id)
    
    print(f"\n{'='*80}")
    print(f"Processing model: {model_id}")
    print(f"Short name: {model_short}")
    print(f"{'='*80}")
    
    # Model-specific output directory
    model_out_dir = os.path.join(args.out_dir, model_short)
    ensure_dir(model_out_dir)
    
    # Results files (model-specific)
    results_json = os.path.join(model_out_dir, "results.json")
    results_csv = os.path.join(model_out_dir, "results.csv")
    
    # Load existing results if they exist (auto-skip generation)
    all_results: List[Dict[str, Any]] = []
    if exists_nonempty(results_json) and not args.force_regeneration:
        print(f"Found existing results at {results_json}, loading...")
        try:
            with open(results_json, "r") as f:
                all_results = json.load(f)
            print(f"Loaded {len(all_results)} existing results")
        except Exception as exc:
            print(f"Warning: Failed to load existing results: {exc}", file=sys.stderr)
            print("Will regenerate...", file=sys.stderr)
            all_results = []
    
    if not all_results or args.force_regeneration:
        # Generate responses
        if args.force_regeneration:
            print(f"Force regeneration requested for {model_short}, regenerating responses...")
        else:
            print(f"No existing results found for {model_short}, generating responses (first and second diagnoses)...")
        
        tasks = []
        for idx, row in enumerate(rows):
            pmcid = str(row.get("pmcid", idx))
            case_prompt = row.get("case_prompt") or row.get("case_presentation") or ""
            true_diagnosis = row.get("final_diagnosis", "")
            tasks.append((pmcid, case_prompt, true_diagnosis, idx))
        
        # Process tasks in parallel
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(generate_two_diagnoses, case_prompt, openrouter_model): task
                for task in tasks
            }
            
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=f"Generating ({model_short})"):
                pmcid, case_prompt, true_diagnosis, idx = future_to_task[future]
                try:
                    first_reasoning, first_answer, second_reasoning, second_answer = future.result()
                    
                    all_results.append({
                        "pmcid": pmcid,
                        "case_index": idx,
                        "case_prompt": case_prompt,
                        "true_diagnosis": true_diagnosis,
                        "first_reasoning": first_reasoning,
                        "first_answer": first_answer,
                        "second_reasoning": second_reasoning,
                        "second_answer": second_answer,
                        "model": model_id,
                        "model_short": model_short,
                    })
                except Exception as exc:
                    print(f"Error processing {pmcid}: {exc}", file=sys.stderr)
                    all_results.append({
                        "pmcid": pmcid,
                        "case_index": idx,
                        "case_prompt": case_prompt,
                        "true_diagnosis": true_diagnosis,
                        "error": str(exc),
                        "model": model_id,
                        "model_short": model_short,
                    })
        
        # Save generation results
        with open(results_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved generation results to {results_json}")
    
    # Label traces (parallelized) - with caching
    print(f"Labeling reasoning traces for {model_short}...")
    labeled_results: List[Dict[str, Any]] = []
    
    def label_result_traces(result: Dict[str, Any]) -> Dict[str, Any]:
        """Label traces for a single result."""
        if "error" in result:
            return result
        
        pmcid = result["pmcid"]
        case_prompt = result["case_prompt"]
        cache_dir = os.path.join(model_out_dir, ".cache", "labels", pmcid)
        
        if not args.skip_labeling:
            # Check if labels already exist in the result (from previous run)
            if result.get("first_label_json") and result.get("second_label_json"):
                # Labels already cached, skip
                return result
            
            # Label first trace (only if not already labeled)
            first_label_json = result.get("first_label_json")
            if not first_label_json and result.get("first_reasoning"):
                first_label_json = label_trace(
                    case_prompt,
                    result["first_reasoning"],
                    os.path.join(cache_dir, "first"),
                    workers=1,  # Single trace, doesn't need workers
                )
                result["first_label_json"] = first_label_json
            
            # Label second trace (only if not already labeled)
            second_label_json = result.get("second_label_json")
            if not second_label_json and result.get("second_reasoning"):
                second_label_json = label_trace(
                    case_prompt,
                    result["second_reasoning"],
                    os.path.join(cache_dir, "second"),
                    workers=1,  # Single trace, doesn't need workers
                )
                result["second_label_json"] = second_label_json
        else:
            # Use existing labels if present
            first_label_json = result.get("first_label_json")
            second_label_json = result.get("second_label_json")
        
        return result
    
    # Parallelize labeling (since each call makes API requests)
    # Filter to only label traces that don't already have labels
    results_to_label = [r for r in all_results if not (r.get("first_label_json") and r.get("second_label_json"))]
    results_already_labeled = [r for r in all_results if r.get("first_label_json") and r.get("second_label_json")]
    
    if results_already_labeled:
        print(f"  {len(results_already_labeled)} results already have labels, skipping...")
        labeled_results.extend(results_already_labeled)
    
    if not args.skip_labeling and results_to_label:
        print(f"  Labeling {len(results_to_label)} traces...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_result = {
                executor.submit(label_result_traces, result): result
                for result in results_to_label
            }
            
            for future in tqdm(as_completed(future_to_result), total=len(results_to_label), desc=f"Labeling ({model_short})"):
                try:
                    labeled_result = future.result()
                    labeled_results.append(labeled_result)
                except Exception as exc:
                    result = future_to_result[future]
                    print(f"Error labeling traces for {result.get('pmcid', 'unknown')}: {exc}", file=sys.stderr)
                    labeled_results.append(result)
        
        # Save labeled results back to JSON for caching
        with open(results_json, "w") as f:
            json.dump(labeled_results, f, indent=2)
        print(f"Saved labeled results to {results_json}")
    elif args.skip_labeling:
        # Sequential if skipping labeling
        for result in tqdm(all_results, desc=f"Labeling ({model_short})"):
            labeled_results.append(result)
    
    # Grade answers and make predictions (parallelized)
    print(f"Grading answers and making classifier predictions for {model_short}...")
    final_results: List[Dict[str, Any]] = []
    
    def process_result_for_grading(result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single result: grade answers and make classifier predictions."""
        if "error" in result:
            return result
        
        true_diagnosis = result["true_diagnosis"]
        first_answer = result.get("first_answer", "")
        second_answer = result.get("second_answer", "")
        first_label_json = result.get("first_label_json")
        second_label_json = result.get("second_label_json")
        
        # Grade answers (API calls - will be parallelized)
        # Check if already graded (cached)
        case_prompt = result.get("case_prompt", "")
        if "first_correct" in result and "second_correct" in result:
            # Already graded, use cached values
            first_correct = result["first_correct"]
            second_correct = result["second_correct"]
        else:
            # Need to grade
            first_correct = grade_answer(first_answer, true_diagnosis, case_prompt) if first_answer else False
            second_correct = grade_answer(second_answer, true_diagnosis, case_prompt) if second_answer else False
        
        # Classifier prediction on first trace (CPU-bound, fast)
        classifier_thinks_first_correct = False
        classifier_confidence = 0.0
        if first_label_json:
            # Enable debug for first few predictions to diagnose issues
            debug = False  # Set to True for detailed debugging
            classifier_thinks_first_correct, classifier_confidence = predict_trace_correctness(
                first_label_json,
                classifier_info,
                debug=debug,
            )
        
        # Select final answer based on classifier
        if classifier_thinks_first_correct:
            final_answer = first_answer
            final_correct = first_correct
            selection_strategy = "first"
        else:
            final_answer = second_answer
            final_correct = second_correct
            selection_strategy = "second"
        
        result.update({
            "first_correct": bool(first_correct),
            "second_correct": bool(second_correct),
            "classifier_thinks_first_correct": bool(classifier_thinks_first_correct),
            "classifier_confidence": float(classifier_confidence),
            "final_answer": str(final_answer) if final_answer else "",
            "final_correct": bool(final_correct),
            "selection_strategy": str(selection_strategy),
        })
        
        return result
    
    # Parallelize grading (since it involves API calls)
    # Filter to only grade results that haven't been graded yet
    results_to_grade = [r for r in labeled_results if "first_correct" not in r or "second_correct" not in r]
    results_already_graded = [r for r in labeled_results if "first_correct" in r and "second_correct" in r]
    
    if results_already_graded:
        print(f"  {len(results_already_graded)} results already graded, skipping...")
        final_results.extend(results_already_graded)
    
    if results_to_grade:
        print(f"  Grading {len(results_to_grade)} results...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_result = {
                executor.submit(process_result_for_grading, result): result
                for result in results_to_grade
            }
            
            for future in tqdm(as_completed(future_to_result), total=len(results_to_grade), desc=f"Grading ({model_short})"):
                try:
                    processed_result = future.result()
                    final_results.append(processed_result)
                except Exception as exc:
                    result = future_to_result[future]
                    print(f"Error processing result {result.get('pmcid', 'unknown')}: {exc}", file=sys.stderr)
                    final_results.append(result)
        
        # Save graded results back to JSON for caching
        json_serializable_results = [ensure_json_serializable(r) for r in final_results]
        with open(results_json, "w") as f:
            json.dump(json_serializable_results, f, indent=2)
        print(f"Saved graded results to {results_json}")
    else:
        # All results already graded, just use them
        final_results = labeled_results
    
    # Ensure final results are JSON serializable
    json_serializable_results = [ensure_json_serializable(r) for r in final_results]
    with open(results_json, "w") as f:
        json.dump(json_serializable_results, f, indent=2)
    
    # Convert to CSV for analysis
    csv_rows = []
    for r in final_results:
        first_answer = str(r.get("first_answer", "")).strip().lower()
        second_answer = str(r.get("second_answer", "")).strip().lower()
        answer_changed = first_answer != second_answer if (first_answer and second_answer) else None
        
        # Determine correctness transition
        first_correct = r.get("first_correct", False)
        second_correct = r.get("second_correct", False)
        if first_correct is True and second_correct is False:
            correctness_transition = "right_to_wrong"
        elif first_correct is False and second_correct is True:
            correctness_transition = "wrong_to_right"
        elif first_correct is True and second_correct is True:
            correctness_transition = "right_to_right"
        elif first_correct is False and second_correct is False:
            correctness_transition = "wrong_to_wrong"
        else:
            correctness_transition = "unknown"
        
        csv_rows.append({
            "pmcid": r.get("pmcid"),
            "model": model_short,
            "true_diagnosis": r.get("true_diagnosis"),
            "first_answer": r.get("first_answer", ""),
            "second_answer": r.get("second_answer", ""),
            "final_answer": r.get("final_answer", ""),
            "first_correct": r.get("first_correct", False),
            "second_correct": r.get("second_correct", False),
            "final_correct": r.get("final_correct", False),
            "classifier_thinks_first_correct": r.get("classifier_thinks_first_correct", False),
            "classifier_confidence": r.get("classifier_confidence", 0.0),
            "selection_strategy": r.get("selection_strategy", ""),
            "answer_changed": answer_changed,
            "correctness_transition": correctness_transition,
        })
    
    df = pd.DataFrame(csv_rows)
    df.to_csv(results_csv, index=False)
    print(f"Saved results to {results_csv}")
    
    # Compute accuracies and return summary
    valid_results = [r for r in final_results if "error" not in r]
    
    summary = {
        "model": model_id,
        "model_short": model_short,
        "out_dir": model_out_dir,
        "n_cases": len(valid_results),
    }
    
    if not valid_results:
        print(f"No valid results to compute accuracies for {model_short}")
        return summary
    
    # First answer accuracy
    first_correct = [r["first_correct"] for r in valid_results if r.get("first_answer")]
    if first_correct:
        acc1, lower1, upper1 = compute_accuracy_with_ci(first_correct)
        summary["first_accuracy"] = acc1
        summary["first_accuracy_ci"] = (lower1, upper1)
        print(f"\n1. First Answer Accuracy: {acc1:.1%} (95% CI: {lower1:.1%} - {upper1:.1%})")
        print(f"   N = {len(first_correct)}")
    
    # Second answer accuracy
    second_correct = [r["second_correct"] for r in valid_results if r.get("second_answer")]
    if second_correct:
        acc2, lower2, upper2 = compute_accuracy_with_ci(second_correct)
        summary["second_accuracy"] = acc2
        summary["second_accuracy_ci"] = (lower2, upper2)
        print(f"\n2. Second Answer Accuracy: {acc2:.1%} (95% CI: {lower2:.1%} - {upper2:.1%})")
        print(f"   N = {len(second_correct)}")
    
    # Prompt effectiveness metrics
    results_with_both_answers = [
        r for r in valid_results 
        if r.get("first_answer") and r.get("second_answer")
    ]
    
    if results_with_both_answers:
        # Count how many times the answer changed
        answer_changed = [
            r for r in results_with_both_answers
            if str(r.get("first_answer", "")).strip().lower() != str(r.get("second_answer", "")).strip().lower()
        ]
        answer_unchanged = [
            r for r in results_with_both_answers
            if str(r.get("first_answer", "")).strip().lower() == str(r.get("second_answer", "")).strip().lower()
        ]
        
        # Count correctness transitions
        right_to_wrong = [
            r for r in results_with_both_answers
            if r.get("first_correct") is True and r.get("second_correct") is False
        ]
        wrong_to_right = [
            r for r in results_with_both_answers
            if r.get("first_correct") is False and r.get("second_correct") is True
        ]
        right_to_right = [
            r for r in results_with_both_answers
            if r.get("first_correct") is True and r.get("second_correct") is True
        ]
        wrong_to_wrong = [
            r for r in results_with_both_answers
            if r.get("first_correct") is False and r.get("second_correct") is False
        ]
        
        n_total = len(results_with_both_answers)
        n_changed = len(answer_changed)
        n_unchanged = len(answer_unchanged)
        
        print(f"\nPrompt Effectiveness:")
        print(f"   Total cases with both answers: {n_total}")
        print(f"   Answer changed: {n_changed} ({n_changed/n_total:.1%})")
        print(f"   Answer unchanged: {n_unchanged} ({n_unchanged/n_total:.1%})")
        print(f"\n   Correctness Transitions:")
        print(f"     Right → Wrong: {len(right_to_wrong)} ({len(right_to_wrong)/n_total:.1%})")
        print(f"     Wrong → Right: {len(wrong_to_right)} ({len(wrong_to_right)/n_total:.1%})")
        print(f"     Right → Right: {len(right_to_right)} ({len(right_to_right)/n_total:.1%})")
        print(f"     Wrong → Wrong: {len(wrong_to_wrong)} ({len(wrong_to_wrong)/n_total:.1%})")
        
        # Net improvement
        net_improvement = len(wrong_to_right) - len(right_to_wrong)
        print(f"\n   Net Improvement: {net_improvement:+d} cases ({net_improvement/n_total:+.1%})")
        if net_improvement > 0:
            print(f"     ✓ Re-reasoning prompt improved accuracy")
        elif net_improvement < 0:
            print(f"     ✗ Re-reasoning prompt decreased accuracy")
        else:
            print(f"     = Re-reasoning prompt had no net effect")
        
        summary["prompt_effectiveness"] = {
            "total_cases": n_total,
            "answer_changed": n_changed,
            "answer_unchanged": n_unchanged,
            "right_to_wrong": len(right_to_wrong),
            "wrong_to_right": len(wrong_to_right),
            "right_to_right": len(right_to_right),
            "wrong_to_wrong": len(wrong_to_wrong),
            "net_improvement": net_improvement,
            "net_improvement_pct": net_improvement / n_total if n_total > 0 else 0.0,
        }
    
    # Classifier prediction statistics
    classifier_predictions = [r.get("classifier_thinks_first_correct", False) for r in valid_results if r.get("first_label_json")]
    classifier_confidences = [r.get("classifier_confidence", 0.0) for r in valid_results if r.get("first_label_json")]
    
    if classifier_predictions:
        n_predicted_correct = sum(classifier_predictions)
        n_predicted_incorrect = len(classifier_predictions) - n_predicted_correct
        avg_confidence = np.mean(classifier_confidences) if classifier_confidences else 0.0
        print(f"\nClassifier Statistics:")
        print(f"   Total predictions: {len(classifier_predictions)}")
        print(f"   Predicted correct: {n_predicted_correct} ({n_predicted_correct/len(classifier_predictions):.1%})")
        print(f"   Predicted incorrect: {n_predicted_incorrect} ({n_predicted_incorrect/len(classifier_predictions):.1%})")
        print(f"   Average confidence: {avg_confidence:.4f}")
        
        # Check if classifier is always predicting incorrect
        if n_predicted_correct == 0:
            print(f"   ⚠ WARNING: Classifier predicted ALL traces as incorrect!")
            print(f"   This suggests a potential issue with:")
            print(f"     - Feature extraction (check if label_json format is correct)")
            print(f"     - Feature mismatch (check if feature names match training)")
            print(f"     - Classifier threshold or model issue")
        
        summary["classifier_predicted_correct"] = n_predicted_correct
        summary["classifier_predicted_incorrect"] = n_predicted_incorrect
        summary["classifier_avg_confidence"] = float(avg_confidence)
    
    # Classifier-guided accuracy
    final_correct = [r["final_correct"] for r in valid_results if r.get("final_answer")]
    if final_correct:
        acc3, lower3, upper3 = compute_accuracy_with_ci(final_correct)
        summary["classifier_guided_accuracy"] = acc3
        summary["classifier_guided_accuracy_ci"] = (lower3, upper3)
        print(f"\n3. Classifier-Guided Accuracy: {acc3:.1%} (95% CI: {lower3:.1%} - {upper3:.1%})")
        print(f"   N = {len(final_correct)}")
        
        used_first = sum(1 for r in valid_results if r.get("selection_strategy") == "first")
        used_second = sum(1 for r in valid_results if r.get("selection_strategy") == "second")
        summary["used_first"] = used_first
        summary["used_second"] = used_second
        print(f"   Selected first answer: {used_first} cases")
        print(f"   Selected second answer: {used_second} cases")
    
    # Best case accuracy (either first OR second was correct)
    # This represents the upper bound of what the classifier could achieve
    best_case_results = [
        r for r in valid_results
        if r.get("first_answer") and r.get("second_answer")
    ]
    if best_case_results:
        best_case_correct = [
            r.get("first_correct") or r.get("second_correct")
            for r in best_case_results
        ]
        acc4, lower4, upper4 = compute_accuracy_with_ci(best_case_correct)
        summary["best_case_accuracy"] = acc4
        summary["best_case_accuracy_ci"] = (lower4, upper4)
        print(f"\n4. Best Case Accuracy (Either First or Second Correct): {acc4:.1%} (95% CI: {lower4:.1%} - {upper4:.1%})")
        print(f"   N = {len(best_case_correct)}")
        print(f"   This represents the upper bound if classifier always picks the correct answer when available")
        
        # Breakdown: how many cases had at least one correct answer
        n_at_least_one_correct = sum(best_case_correct)
        n_both_wrong = len(best_case_correct) - n_at_least_one_correct
        print(f"   Cases with at least one correct: {n_at_least_one_correct} ({n_at_least_one_correct/len(best_case_correct):.1%})")
        print(f"   Cases with both wrong: {n_both_wrong} ({n_both_wrong/len(best_case_correct):.1%})")
        
        summary["best_case_breakdown"] = {
            "n_at_least_one_correct": n_at_least_one_correct,
            "n_both_wrong": n_both_wrong,
            "pct_at_least_one_correct": n_at_least_one_correct / len(best_case_correct) if best_case_correct else 0.0,
        }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy improvement using classifier-guided re-reasoning"
    )
    parser.add_argument(
        "--dataset",
        default="tmknguyen/MedCaseReasoning-filtered",
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (train/val/test)",
    )
    parser.add_argument(
        "--classifier_dir",
        default="alternative_classifier_results",
        help="Directory containing trained classifier",
    )
    parser.add_argument(
        "--labeled_csv",
        default=None,
        help="Optional: labeled CSV from TRAIN split to train classifier if not pre-trained. "
             "MUST be from train split to match classify_traces_alternative.py training.",
    )
    parser.add_argument(
        "--train_labeled_csv",
        default=None,
        help="Path to labeled CSV from TRAIN split for evaluating classifiers on train set. "
             "If not provided, uses --labeled_csv if available.",
    )
    parser.add_argument(
        "--train_split",
        default="train",
        help="Dataset split used for training classifier (default: train). "
             "Classifier MUST be trained on train split.",
    )
    parser.add_argument(
        "--out_dir",
        default="re_reasoning_eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID to evaluate (default: deepseek-r1-distill-llama-70b). Use --all_models for all models.",
    )
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Run evaluation for all models",
    )
    parser.add_argument(
        "--limit_cases",
        type=int,
        default=None,
        help="Limit number of cases for testing (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=100,
        help="Number of parallel workers for generation",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip generation (use existing results). Default: auto-skip if results.json exists.",
    )
    parser.add_argument(
        "--force_regeneration",
        action="store_true",
        help="Force regeneration even if results.json exists",
    )
    parser.add_argument(
        "--skip_labeling",
        action="store_true",
        help="Skip labeling (use existing labels)",
    )
    args = parser.parse_args()
    
    ensure_dir(args.out_dir)
    
    # Determine which models to evaluate
    if args.all_models:
        models_to_evaluate = MODELS
        print(f"Running evaluation for all {len(models_to_evaluate)} models")
    elif args.model:
        models_to_evaluate = [args.model]
        print(f"Running evaluation for model: {args.model}")
    else:
        # Default: use deepseek-r1-distill-llama-70b
        default_model = "deepseek/deepseek-r1-distill-llama-70b"
        models_to_evaluate = [default_model]
        print(f"No model specified, using default: {default_model}")
    
    # Load all classifiers
    print("\nLoading all classifiers...")
    print(f"NOTE: Classifiers must be trained on '{args.train_split}' split to match classify_traces_alternative.py")
    all_classifiers = load_all_classifiers(args.classifier_dir)
    
    # Evaluate each classifier on train set (if provided) to verify performance
    train_csv = args.train_labeled_csv or args.labeled_csv
    if train_csv and os.path.exists(train_csv):
        print("\n" + "="*80)
        print("EVALUATING CLASSIFIERS ON TRAIN SET (VERIFICATION)")
        print("="*80)
        train_eval_results = {}
        for classifier_name, classifier_info in all_classifiers.items():
            try:
                train_results = evaluate_classifier_on_dataset(
                    train_csv,
                    classifier_name,
                    classifier_info,
                    args.out_dir,
                    dataset_name="train",
                )
                train_eval_results[classifier_name] = train_results
            except Exception as exc:
                print(f"Error evaluating {classifier_name} on train set: {exc}", file=sys.stderr)
                import traceback
                traceback.print_exc()
        
        # Save train evaluation results
        train_eval_path = os.path.join(args.out_dir, "train_evaluation_results.json")
        with open(train_eval_path, "w") as f:
            json.dump(train_eval_results, f, indent=2, default=str)
        print(f"\nSaved train evaluation results to {train_eval_path}")
    
    # Load dataset (shared across all models and classifiers)
    print(f"\nLoading dataset {args.dataset} split {args.split}...")
    dataset = load_dataset(args.dataset)[args.split]
    rows = list(dataset)
    if args.limit_cases:
        rows = rows[:args.limit_cases]
    print(f"Loaded {len(rows)} cases")
    
    # Run evaluation for each classifier and each model
    all_classifier_summaries: Dict[str, List[Dict[str, Any]]] = {}
    
    for classifier_name, classifier_info in all_classifiers.items():
        print("\n" + "="*80)
        print(f"EVALUATING WITH CLASSIFIER: {classifier_name}")
        print("="*80)
        
        classifier_summaries = []
        
        for model_id in models_to_evaluate:
            try:
                # Create classifier-specific output directory
                classifier_out_dir = os.path.join(args.out_dir, "classifiers", classifier_name)
                ensure_dir(classifier_out_dir)
                
                # Temporarily override out_dir for this classifier
                original_out_dir = args.out_dir
                args.out_dir = classifier_out_dir
                
                summary = run_evaluation_for_model(
                    model_id=model_id,
                    args=args,
                    classifier_info=classifier_info,
                    rows=rows,
                )
                summary["classifier_name"] = classifier_name
                classifier_summaries.append(summary)
                
                # Restore original out_dir
                args.out_dir = original_out_dir
                
            except Exception as exc:
                print(f"Error processing model {model_id} with classifier {classifier_name}: {exc}", file=sys.stderr)
                import traceback
                traceback.print_exc()
        
        all_classifier_summaries[classifier_name] = classifier_summaries
        
        # Save classifier-specific summary
        classifier_summary_path = os.path.join(args.out_dir, "classifiers", classifier_name, "summary.json")
        with open(classifier_summary_path, "w") as f:
            json.dump(classifier_summaries, f, indent=2, default=str)
    
    # Save combined summary across all classifiers and models
    combined_summary_path = os.path.join(args.out_dir, "all_classifiers_summary.json")
    with open(combined_summary_path, "w") as f:
        json.dump(all_classifier_summaries, f, indent=2, default=str)
    print(f"\nSaved combined summary to {combined_summary_path}")
    
    # Print summary comparison
    print("\n" + "="*80)
    print("COMPARISON ACROSS ALL CLASSIFIERS AND MODELS")
    print("="*80)
    for classifier_name, summaries in all_classifier_summaries.items():
        print(f"\n{classifier_name.upper()}:")
        for summary in summaries:
            model_short = summary.get("model_short", "unknown")
            if "classifier_guided_accuracy" in summary:
                acc = summary["classifier_guided_accuracy"]
                first_acc = summary.get("first_accuracy", 0.0)
                second_acc = summary.get("second_accuracy", 0.0)
                best_case_acc = summary.get("best_case_accuracy", 0.0)
                improvement = acc - first_acc
                potential_improvement = best_case_acc - acc
                print(f"  {model_short:40s}: First={first_acc:.1%}, Second={second_acc:.1%}, Guided={acc:.1%} (Δ={improvement:+.1%}), Best={best_case_acc:.1%} (potential={potential_improvement:+.1%})")
    
    print(f"\nAll results saved to {args.out_dir}")


if __name__ == "__main__":
    main()

