#!/usr/bin/env bash
# ============================================================================
# serve-1gpu.sh — serve Qwen3.8-Flash-Next EXL3 4.05 on ONE GPU.
# Applies configs/config-1gpu.yml, then runs TabbyAPI in the foreground.
#
#   GPU_ID=0 ./serve-1gpu.sh      choose the card   (default: 0)
#   MODEL_DIR=/path ./serve-1gpu.sh                (default: $HOME/models/Qwen3.8-Flash-Next)
#
# Stop with Ctrl-C. Verify from another terminal with ./verify.sh
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
TABBY_DIR="$REPO_DIR/tabbyAPI"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next}"
MODEL_NAME="Qwen3.8-Flash-Next-EXL3-405"
GPU_ID="${GPU_ID:-0}"

[ -x "$VENV_DIR/bin/python" ] || { echo "run ./install-1gpu.sh first (.venv missing)" >&2; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "model not found at $MODEL_DIR (use MODEL_DIR=... or re-run the installer)" >&2; exit 1; }

# 1) stop a running instance (graceful unload takes ~20 s)
if pgrep -f "$VENV_DIR/bin/python main.py" >/dev/null; then
  echo "[1gpu] stopping the running TabbyAPI instance..."
  pkill -f "$VENV_DIR/bin/python main.py" || true
  for _ in $(seq 1 40); do
    pgrep -f "$VENV_DIR/bin/python main.py" >/dev/null || break
    sleep 1
  done
fi

# 2) apply the canonical 1-GPU config and the forced sampler preset
cp "$REPO_DIR/configs/config-1gpu.yml" "$TABBY_DIR/config.yml"
mkdir -p "$TABBY_DIR/models" "$TABBY_DIR/sampler_overrides"
ln -sfn "$MODEL_DIR" "$TABBY_DIR/models/$MODEL_NAME"
cp "$REPO_DIR/sampler_overrides/qwen38_flash_next_thinking.yml" "$TABBY_DIR/sampler_overrides/"
echo "[1gpu] applied config-1gpu.yml -> tabbyAPI/config.yml"
grep -E "cpu_moe_split_experts:|cpu_moe_threads:|chunk_size:|max_seq_len:|cache_mode:|draft_mode:|ngram_ram:|override_preset:" "$TABBY_DIR/config.yml" | sed 's/^/    /'

# 3) start (foreground, no timeout wrapper — stop it yourself with Ctrl-C / pkill)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU_ID"
cd "$TABBY_DIR"
echo "[1gpu] starting TabbyAPI on 0.0.0.0:8001 (GPU $GPU_ID) — loading takes a minute or two..."
exec "$VENV_DIR/bin/python" main.py
