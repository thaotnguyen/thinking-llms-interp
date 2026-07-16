#!/usr/bin/env bash
# Serve gpt-oss-120b with llama.cpp on 4x RTX 5090, for taxonomy titling.
#
# WHY llama.cpp AND NOT vLLM
#   vLLM cannot run this model on sm120 (consumer Blackwell). Every MXFP4 MoE
#   backend fails, on both 0.24.0 and 0.25.1:
#     marlin            gptq_marlin_repack fails on sm120 (reproduced standalone)
#     triton/_unfused   emit `.tile::scatter4`, an SM100-only TMA op; ptxas rejects
#                       it for sm_120a. Forcing the Hopper path then hits
#                       `assert num_stages >= 1` -- the kernels assume ~228 KB of
#                       shared memory (SM90/SM100); the 5090 has ~100 KB.
#     humming           cuModuleLoad -> CUDA_ERROR_ILLEGAL_ADDRESS
#     flashinfer_*      reject MXFP4 (want quantized activations); deep_gemm is FP8-only
#     emulation         illegal memory access in the MoE kernel (stock triton_kernels,
#                       eager and cudagraph, under two attention backends)
#   And it cannot be sidestepped by dropping the quantization: bf16 is 234 GB and
#   fp8 is 117 GB, against 130 GB of total VRAM. llama.cpp has its own CUDA kernels
#   and compiles for `120a` natively.
#
# PRECISION: NO ADDED QUANTIZATION.
#   gpt-oss-120b ships natively in MXFP4 -- that IS the original release, not a
#   post-hoc quantization. We use the official GGUF as published and apply no
#   further quantization (no Q4_K_M / Q8_0 repack). The only "less quantized"
#   option would be upcasting to bf16, which does not fit in 130 GB.
#   Non-expert tensors in this GGUF are already full precision.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${LLAMA:-/workspace/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:?set MODEL=/path/to/gpt-oss-120b-mxfp4.gguf}"

# Throughput knobs (the job is ~4,800 large-prefill requests):
#   -np 8          8 parallel slots -> continuous batching
#   -c 262144      total KV pool, split across slots => 32k ctx per slot.
#                  Prompts run ~15k tokens, so this leaves generous headroom.
#   -fa on         flash attention
#   -ngl 999       every layer on GPU
#   --split-mode layer  spread layers across the 4 GPUs (they are PCIe-only, no NVLink,
#                  so layer split beats row split: far less cross-GPU traffic)
#   --cont-batching + --cache-reuse: reuse the KV of the shared prompt prefix
#                  (the per-model reasoning-trace block is identical across prompts)
exec "$LLAMA" \
  --model "$MODEL" \
  --alias openai/gpt-oss-120b \
  --host 127.0.0.1 --port ${PORT:-8090} \
  -ngl 999 \
  --split-mode layer \
  -c 393216 \
  -np 16 \
  -fa on \
  --cont-batching \
  --cache-reuse 256 \
  -b 8192 -ub 2048 \
  --jinja \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --no-warmup \
  --metrics
