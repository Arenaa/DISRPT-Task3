#!/usr/bin/env bash
# Instruction SFT for Qwen/Qwen3-4B (from repo root: scripts/run_finetune_qwen3_4b.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python scripts/finetune_qwen_instruction_sft.py \
  --qwen3-4b \
  --bf16 \
  --gradient-checkpointing \
  --max-length 1024 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --eval-batch-size 1 \
  "$@"
