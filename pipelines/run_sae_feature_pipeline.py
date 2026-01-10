#!/usr/bin/env python3
"""
End-to-end pipeline:
- Train a steering vector for a chosen SAE feature (using existing training code)
- Place vector where the coefficient sweep expects it
- Run coefficient sweep (generation + grading) and produce CSV + plot
- Leave per-coefficient responses saved for later inspection

This script orchestrates existing components without re-implementing core logic.

Inputs
- --model: HF model id for generation
- --layer: layer to apply steering at
- --n_clusters: number of SAE clusters used for feature indexing
- --feature_idx: idxN (integer N) of the SAE/taxonomy feature to target

Notes
- We reuse generate-responses/train_and_sweep_sae_vectors.py for vector training only
  (skip generation and grading there). It saves the trained vector under
  <output_dir>/vectors/<model_id>_layer<L>_idx<N>_<strategy>.pt.
- Coefficient sweep expects an optimized vector to be accessible from
  generate-responses/generate_responses_with_steering.py at path:
  ../train-vectors/results/vars/optimized_vectors/<model_id>_idx<N>.pt (legacy unsuffixed name).
  We copy the trained vector to that path so the sweep can find it.
- We then invoke generate-responses/coefficient_sweep.py which handles response
  generation, grading (via grade_responses.py), CSV, and plotting.

Outputs
- Vector: <pipeline_output>/vectors/<model_id>_layer<L>_idx<N>_<strategy>.pt
- Copy to sweep location: train-vectors/results/vars/optimized_vectors/<model_id>_idx<N>.pt
- Sweep CSV + plot under: generate-responses/results/coefficient_sweep (or your --sweep_output_dir)
- Responses for each coefficient at: generate-responses/results/vars/*.json

Requirements
- Annotated responses must exist for the chosen (thinking) model if training is performed.
- OPENAI_API_KEY set in environment for grading (used by grade_responses.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
import json
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
GEN_RESP = ROOT / "generate-responses"
TRAIN_VECT = ROOT / "train-vectors"


def run(cmd: list[str], cwd: Path | None = None, desc: str | None = None) -> None:
    if desc:
        print("\n" + "=" * 80)
        print(desc)
        print("=" * 80)
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {' '.join(cmd)}")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def train_vector(
    model: str,
    layer: int,
    n_clusters: int,
    feature_idx: int,
    steering_type: str,
    n_training_examples: int,
    batch_size: int,
    max_iters: int,
    lr: str,
    output_dir: Path,
    load_in_8bit: bool,
    thinking_model: str | None,
    seed: int,
    ) -> Path:
    """Invoke the existing trainer to learn a vector and return its saved path."""
    ensure_dir(output_dir)
    trainer = GEN_RESP / "train_and_sweep_sae_vectors.py"
    if not trainer.exists():
        raise FileNotFoundError(f"Trainer not found: {trainer}")

    # Vector is saved at: <out>/vectors/<model_id>_layer<L>_idx<N>_<strategy>.pt
    cmd = [
        sys.executable, str(trainer),
        "--model", model,
        "--layer", str(layer),
        "--n_clusters", str(n_clusters),
        "--feature_idx", str(feature_idx),
        "--n_training_examples", str(n_training_examples),
        "--batch_size", str(batch_size),
        "--max_iters", str(max_iters),
        "--lr", lr,
        "--steering_type", steering_type,
        "--output_dir", str(output_dir),
        "--seed", str(seed),
    ]
    if thinking_model:
        cmd += ["--thinking_model", thinking_model]
    if load_in_8bit:
        cmd.append("--load_in_8bit")

    run(cmd, cwd=GEN_RESP, desc="Training steering vector (train_and_sweep_sae_vectors.py)")

    model_id = model.split('/')[-1].lower()
    vec_path = output_dir / "vectors" / f"{model_id}_layer{layer}_idx{feature_idx}_{steering_type}.pt"
    if not vec_path.exists():
        raise FileNotFoundError(f"Expected vector not found at {vec_path}")
    return vec_path


def place_vector_for_sweep(trained_vector_path: Path, model: str, feature_idx: int) -> Path:
    """Copy the trained vector to the legacy unsuffixed path used by the sweep generator."""
    model_id = model.split('/')[-1].lower()
    legacy_dir = TRAIN_VECT / "results" / "vars" / "optimized_vectors"
    ensure_dir(legacy_dir)
    legacy_path = legacy_dir / f"{model_id}_idx{feature_idx}.pt"

    # Load and re-save as a simple tensor under {category: tensor} if needed
    obj = torch.load(trained_vector_path, map_location='cpu')
    category = f"idx{feature_idx}"
    if isinstance(obj, torch.Tensor):
        payload = {category: obj}
    elif isinstance(obj, dict):
        # train_and_sweep saves raw dict for non-linear types; keep as {category: params}
        payload = {category: obj}
    else:
        raise ValueError(f"Unexpected vector format at {trained_vector_path}: {type(obj)}")

    torch.save(payload, legacy_path)
    print(f"Placed sweep vector at: {legacy_path}")
    return legacy_path


def run_coefficient_sweep(
    model: str,
    layer: int,
    n_clusters: int,
    feature_idx: int,
    coefficients: list[float],
    dataset: str,
    dataset_split: str,
    max_tokens: int,
    temperature: float,
    limit: int,
    batch_size: int,
    load_in_8bit: bool,
    judge_model: str,
    sweep_output_dir: Path,
    engine: str,
    ) -> None:
    """Invoke the coefficient_sweep.py which will also call grade_responses.py."""
    sweeper = GEN_RESP / "coefficient_sweep.py"
    if not sweeper.exists():
        raise FileNotFoundError(f"Coefficient sweeper not found: {sweeper}")

    coeff_strs = [str(c) for c in coefficients]
    cmd = [
        sys.executable, str(sweeper),
        "--model", model,
        "--layer", str(layer),
        "--n_clusters", str(n_clusters),
        "--feature_idx", str(feature_idx),
        "--coefficients", *coeff_strs,
        "--dataset", dataset,
        "--dataset_split", dataset_split,
        "--max_tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--limit", str(limit),
        "--batch_size", str(batch_size),
        "--judge_model", judge_model,
        "--output_dir", str(sweep_output_dir),
    ]
    if engine:
        cmd.extend(["--engine", engine])
    if load_in_8bit:
        cmd.append("--load_in_8bit")

    # Important: do NOT pass --use_raw_sae; we want our optimized vector
    run(cmd, cwd=GEN_RESP, desc="Running coefficient sweep (generation + grading + plot)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SAE feature pipeline: train vector, sweep coefficients, grade, plot.")

    # Core config
    ap.add_argument("--model", required=True, help="HF model id for generation")
    ap.add_argument("--layer", required=True, type=int, help="Layer to apply steering")
    ap.add_argument("--n_clusters", required=True, type=int, help="SAE/taxonomy cluster count")
    ap.add_argument("--feature_idx", required=True, type=int, help="Feature index (e.g., 9 for idx9)")

    # Vector training knobs
    ap.add_argument("--thinking_model", default=None, help="Model to train vector on (defaults to --model)")
    ap.add_argument("--steering_type", choices=["linear", "adaptive_linear", "resid_lora"], default="linear")
    ap.add_argument("--n_training_examples", type=int, default=8, help="Number of training examples (reduce for less memory usage)")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size for training and sweep (reduce for less memory usage)")
    ap.add_argument("--max_iters", type=int, default=1000)
    ap.add_argument("--lr", default="1e-1", help="Comma-separated learning rates (as in train-vectors)")
    ap.add_argument("--skip_training", action="store_true", help="Skip training if vector already exists at pipeline output")

    # Sweep + grading
    ap.add_argument("--coefficients", type=float, nargs="+", default=[-10, -2.0, -1.0, -0.5, 0.0])
    ap.add_argument("--dataset", default="tmknguyen/MedCaseReasoning-filtered")
    ap.add_argument("--dataset_split", default="train")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--judge_model", default="gpt-5-nano")

    # System
    ap.add_argument("--output_dir", default=str(ROOT / "results" / "sae_feature_pipeline"))
    ap.add_argument("--sweep_output_dir", default=str(GEN_RESP / "results" / "coefficient_sweep"))
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--engine", type=str, default="nnsight", choices=["nnsight","hf","vllm"], help="Generation engine for sweep generation")

    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    sweep_output_dir = Path(args.sweep_output_dir)
    ensure_dir(output_dir)

    # 1) Train vector (or reuse)
    model_id = args.model.split('/')[-1].lower()
    trained_vec = output_dir / "vectors" / f"{model_id}_layer{args.layer}_idx{args.feature_idx}_{args.steering_type}.pt"

    if args.skip_training and trained_vec.exists():
        print(f"Skipping training; using existing vector: {trained_vec}")
    else:
        _ = train_vector(
            model=args.model,
            layer=args.layer,
            n_clusters=args.n_clusters,
            feature_idx=args.feature_idx,
            steering_type=args.steering_type,
            n_training_examples=args.n_training_examples,
            batch_size=args.batch_size,
            max_iters=args.max_iters,
            lr=args.lr,
            output_dir=output_dir,
            load_in_8bit=args.load_in_8bit,
            thinking_model=args.thinking_model,
            seed=args.seed,
        )

    # 2) Copy to the sweep-expected location (unsuffixed legacy path)
    legacy_vec_path = place_vector_for_sweep(trained_vec, args.model, args.feature_idx)

    # 3) Run sweep (this will generate responses per coefficient, grade them, and plot)
    run_coefficient_sweep(
        model=args.model,
        layer=args.layer,
        n_clusters=args.n_clusters,
        feature_idx=args.feature_idx,
        coefficients=args.coefficients,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        limit=args.limit,
        batch_size=args.batch_size,
        load_in_8bit=args.load_in_8bit,
        judge_model=args.judge_model,
        sweep_output_dir=sweep_output_dir,
        engine=args.engine,
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Vector (trained): {trained_vec}")
    print(f"Vector (for sweep): {legacy_vec_path}")
    print(f"Sweep outputs: {sweep_output_dir}")
    print("Per-coefficient responses are saved under generate-responses/results/vars/")


if __name__ == "__main__":
    main()
