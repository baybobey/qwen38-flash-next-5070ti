#!/usr/bin/env bash
# ============================================================================
# serve-strata.sh — serve Qwen3.8-Flash-Next IQ3_S with Strata.
# Applies the canonical run config (paths resolved + the local API key
# injected), then runs the Strata server in the foreground.
#
#   ./serve-2gpu-strata.sh              two cards, layer split, 262144 ctx
#   ./serve-1gpu-strata.sh                  one card, 262144 ctx
#   CTX=500k ./serve-2gpu-strata.sh         524,288 ctx (experimental yarn rope scaling)
#   CTX=1m   ./serve-2gpu-strata.sh         1,048,576 ctx (experimental; ~95 GiB RAM, ~1.1 GiB VRAM free)
#
#   GPU_IDS=1,0 ./serve-strata.sh --gpus 2  which cards (nvidia-smi numbering)
#   PORT=8001 ./serve-strata.sh --gpus 2            the serve port
#
# Run ./install-strata-<N>gpu.sh first. Stop with Ctrl-C, or
# ./stop-strata.py 8001 from another terminal. Verify with ./verify-strata.sh
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$REPO_DIR/Strata"
DATA_DIR="${DATA_DIR:-$REPO_DIR/Strata-data}"
MODELS_ROOT="$DATA_DIR/models"
GGUF_DIR="${MODEL_DIR:-${GGUF_DIR:-$MODELS_ROOT/IQ3_S}}"
KEY_FILE="$STRATA_DIR/api_key.txt"
CTX="${CTX:-262k}"

GPUS="${GPUS:-}"
PORT="${PORT:-8001}"
GPU_IDS="${GPU_IDS:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    *) echo "unknown option: $1 (see the header comment)" >&2; exit 2 ;;
  esac
done
case "$GPUS" in
  1|2) ;;
  *) echo "error: --gpus is required (1 or 2) — use ./serve-1gpu-strata.sh or ./serve-2gpu-strata.sh" >&2; exit 2 ;;
esac
[ -x "$STRATA_DIR/.venv/bin/python" ] || { echo "Strata not installed under $STRATA_DIR — run ./install-strata-${GPUS}gpu.sh first" >&2; exit 1; }

# the model must be where the config says it is
SHARD1="$GGUF_DIR/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
SHARD2="$GGUF_DIR/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
[ -f "$SHARD1" ] && [ -f "$SHARD2" ] || \
  { echo "model GGUFs not found under $GGUF_DIR (run ./install-strata-${GPUS}gpu.sh, or MODEL_DIR=/path for an existing download)" >&2; exit 1; }

# 1) pick the profile + apply the canonical config (resolve paths + inject key)
case "$CTX" in
  262k)  CFG_SRC="$REPO_DIR/configs/config-${GPUS}gpu.json" ;;
  500k)  CFG_SRC="$REPO_DIR/configs/config-${GPUS}gpu-500k.json" ;;
  1m)    CFG_SRC="$REPO_DIR/configs/config-${GPUS}gpu-1m.json" ;;
  *) echo "[serve-strata] unknown CTX '$CTX' (use 262k | 500k | 1m)" >&2; exit 2 ;;
esac
# the extended windows were validated on the 2-GPU layout only
[ -f "$CFG_SRC" ] || { echo "[serve-strata] no $CTX profile for $GPUS GPU(s) (the extended windows are tuned for 2 GPUs; use CTX=262k, or add configs/$(basename "$CFG_SRC"))" >&2; exit 2; }
[ "$CTX" != "262k" ] && echo "[serve-strata] NOTE: experimental rope scaling (yarn) — the window is past the trained 262k."
FINAL_ARGS=()
[ -n "$GPU_IDS" ] && FINAL_ARGS+=(--gpu-ids "$GPU_IDS")
"$STRATA_DIR/.venv/bin/python" "$REPO_DIR/scripts/finalize-strata.py" "$CFG_SRC" \
  --out "$STRATA_DIR/strata-iq3_s.json" \
  --strata-dir "$STRATA_DIR" --data-dir "$DATA_DIR" --gguf-dir "$GGUF_DIR" \
  --key-file "$KEY_FILE" --port "$PORT" "${FINAL_ARGS[@]}"
echo "[serve-strata] applied $(basename "$CFG_SRC") -> Strata/strata-iq3_s.json"

# 2) stop a running instance on this port (the engine child exits with it)
"$STRATA_DIR/.venv/bin/python" "$REPO_DIR/stop-strata.py" "$PORT" --quiet || true

# 3) start (foreground, no timeout wrapper — stop it yourself with Ctrl-C)
cd "$STRATA_DIR"
echo "[serve-strata] starting the Strata server on 0.0.0.0:$PORT ($GPUS GPU layout, $CTX) — loading the model takes a minute or two..."
exec "$STRATA_DIR/.venv/bin/python" serve/server.py --engine strata \
  --config "$STRATA_DIR/strata-iq3_s.json" --port "$PORT"
