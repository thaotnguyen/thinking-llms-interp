#!/usr/bin/env bash
# Serve openai/gpt-oss-120b (native MXFP4) on 4x RTX 5090 for taxonomy titling.
#
# Notes:
#  - .venv-vllm has torch 2.11.0+cu128; Blackwell (sm120, cc 12.0) needs CUDA >= 12.8.
#  - vllm 0.24.0 pulls a cu13-built binding whose libs live in nvidia/cu13/lib and are
#    NOT on the default loader path -> "ImportError: libcudart.so.13". Hence LD_LIBRARY_PATH.
#  - 8 KV heads / TP 4 = 2 per GPU (divides cleanly).
#  - max-model-len 65536: prompts run ~15-35k tokens (200 capped examples + trace prefix).
#  - MOE BACKEND: "auto" picks Marlin, whose fp4 repack kernel does NOT support sm120
#    (consumer Blackwell) and dies with
#        RuntimeError: gptq_marlin_repack, .../gptq_marlin_repack.cu:344
#    "flashinfer_b12x" looks right for SM12x but is rejected for MXFP4 MoE. The
#    backends MXFP4 actually accepts are: deep_gemm, flashinfer_trtllm(_afp8),
#    flashinfer_cutlass(_afp8), triton, triton_unfused, humming, marlin, aiter*,
#    xpu, cpu, emulation. "triton" is the portable gpt-oss MXFP4 path -> use it.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SP="$REPO/.venv-vllm/lib/python3.12/site-packages"
MOE_BACKEND="${MOE_BACKEND:-emulation}"

export HF_HOME=/workspace/.hf_home
export LD_LIBRARY_PATH="$SP/nvidia/cu13/lib:$SP/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export VLLM_LOGGING_LEVEL=INFO
# FlashInfer misdetects this GPU and dies with the (nonsensical) "FlashInfer
# requires GPUs with sm75 or higher" on sm120 -- its jit/core.py check_cuda_arch
# rejects cc 12.0. It gets pulled in from two independent places, so both are
# disabled: the attention backend, and the SAMPLER (flashinfer/sampling.py
# top_k_top_p_sampling_from_logits). We decode greedily (temperature 0) anyway.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER=0

exec "$REPO/.venv-vllm/bin/vllm" serve openai/gpt-oss-120b \
  --served-model-name openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --kernel-config "{\"moe_backend\": \"${MOE_BACKEND}\"}" \
  --port 8000
