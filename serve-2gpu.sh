#!/usr/bin/env bash
# ============================================================================
# serve-2gpu.sh — serve Qwen3.8-Flash-Next EXL3 4.05 on TWO GPUs (autosplit).
# Applies configs/config-2gpu.yml, then runs TabbyAPI in the foreground.
#
#   GPU_ORDER=0,1 ./serve-2gpu.sh   card order           (default: 0,1)
#   MODEL_DIR=/path ./serve-2gpu.sh                     (default: $HOME/models/Qwen3.8-Flash-Next)
#
# The author's measured numbers were taken with GPU_ORDER=1,0 (second card as the
# primary autosplit device on that motherboard); performance was the same in both
# orders, so the default here is the natural one.
#
# Stop with Ctrl-C. Verify from another terminal with ./verify.sh
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
TABBY_DIR="$REPO_DIR/tabbyAPI"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next}"
MODEL_NAME="Qwen3.8-Flash-Next-EXL3-405"
GPU_ORDER="${GPU_ORDER:-0,1}"

[ -x "$VENV_DIR/bin/python" ] || { echo "run ./install-2gpu.sh first (.venv missing)" >&2; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "model not found at $MODEL_DIR (use MODEL_DIR=... or re-run the installer)" >&2; exit 1; }

# 1) stop a running instance (graceful unload takes ~20 s)
if pgrep -f "$VENV_DIR/bin/python main.py" >/dev/null; then
  echo "[2gpu] stopping the running TabbyAPI instance..."
  pkill -f "$VENV_DIR/bin/python main.py" || true
  for _ in $(seq 1 40); do
    pgrep -f "$VENV_DIR/bin/python main.py" >/dev/null || break
    sleep 1
  done
fi

# 2) apply the canonical 2-GPU config and the forced sampler preset
cp "$REPO_DIR/configs/config-2gpu.yml" "$TABBY_DIR/config.yml"
mkdir -p "$TABBY_DIR/models" "$TABBY_DIR/sampler_overrides"
ln -sfn "$MODEL_DIR" "$TABBY_DIR/models/$MODEL_NAME"
cp "$REPO_DIR/sampler_overrides/qwen38_flash_next_thinking.yml" "$TABBY_DIR/sampler_overrides/"
echo "[2gpu] applied config-2gpu.yml -> tabbyAPI/config.yml"
grep -E "cpu_moe_split_experts:|cpu_moe_threads:|chunk_size:|max_seq_len:|cache_mode:|draft_mode:|draft_num_tokens:|dynamic_draft:|autosplit_reserve:|ngram_ram:|override_preset:" "$TABBY_DIR/config.yml" | sed 's/^/    /'

# 3) start (foreground, no timeout wrapper — stop it yourself with Ctrl-C / pkill)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU_ORDER"
cd "$TABBY_DIR"
echo "[2gpu] starting TabbyAPI on 0.0.0.0:8001 (device order: $GPU_ORDER) — loading takes a minute or two..."
exec "$VENV_DIR/bin/python" main.py
