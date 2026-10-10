#!/usr/bin/env bash
# ============================================================================
# install-strata.sh — installer for the Qwen3.8-Flash-Next Strata IQ3_S stack.
#
# Use the wrappers instead of calling this directly:
#     ./install-strata-1gpu.sh [options]     1x 16 GB GPU
#     ./install-strata-2gpu.sh [options]        2x 16 GB GPUs
#
# Installs into this directory:
#     Strata/             the Strata engine clone (pinned commit) — setup.sh
#                         builds the CUDA engine there and creates Strata/.venv
#     Strata-data/        the model files + the prepared pack + the MTP draft
#                         (under Strata-data/models by default; --model-dir D moves them)
#     Strata/strata-iq3_s.json   the run config, written from
#                         configs/config-<N>gpu.json with this machine's paths
#     Strata/api_key.txt  a random API key, chmod 600, never printed, never in git
#
# What it tunes beyond Strata's stock setup (all measured on
# 2x RTX 5070 Ti — see README.md):
#     --prefill 12288 (borrows cache slots)   --kv int8 --kv-resident 65536
#     --pipeline-windows 2 (2-GPU only)                       --ple-io ram (26.9 GiB table resident)
#     conversation parking on every profile    calibrated --pcie-frac / --spec-min-p
#     server-side sampler defaults (temp 1.0 / top-k 20 / top-p 0.95) + froggeric's fixed chat
#     template with reasoning effort pinned to xhigh
#     images on every profile: the Strata vision encoder (--vision gpu, strata-vision
#     built for CUDA + mmproj downloaded) with --vram-reserve-mib 700 for it
#
# Options:
#     --gpus N          1 or 2 (set by the wrapper)
#     --model-dir D     the IQ3_S folder that already holds both GGUF shards
#                       (implies --skip-model; the installer then never downloads)
#     --data-dir D      where Strata puts its derived data (default: ./Strata-data)
#     --skip-model      do not download the GGUF (it must already exist under
#                       Strata-data/models/IQ3_S or the --model-dir folder)
#     --calibrate       run Strata's ~10-min per-PC calibration (recommended;
#                       it re-measures --pcie-frac / --spec-min-p for your box)
#     --context N       262144 (default), 524288 or 1048576 — which profile to finalize
#                       (all three stay installed; serve-strata.sh picks at start)
#     --port N          the serve port (default 8001)
#     --start           start the server when the install finishes
#     --dry-run         show what would be installed, change nothing
#     -y, --yes         assume yes (no prompts)
#     -h, --help        show this help
#
# Nothing secret is stored in this repository: the API key is generated on this
# machine at install time and written only to Strata/api_key.txt (and, at serve time,
# into the git-ignored run config Strata/strata-iq3_s.json).
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- pinned versions (override via env if you know what you are doing) ------
STRATA_REPO="${STRATA_REPO:-https://github.com/Niko1221/Strata}"
STRATA_COMMIT="${STRATA_COMMIT:-61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406}"   # tag v0.1.42
# The GGUF: the ISTA-DASLab GSQ-RCO quant of Qwen3.8-Flash-Next. Informational here:
# Strata's own setup carries the pin for the download (same revision as of
# v0.1.42); kept to document what this stack was built and measured against.
MODEL_REPO="${MODEL_REPO:-ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF}"
MODEL_REV="${MODEL_REV:-ed59f92082b1e93c0e96d60a8b11aab089b52f09}"
MODEL_VARIANT="IQ3_S"
MODEL_TOTAL_BYTES="${MODEL_TOTAL_BYTES:-83617662656}"      # shard1 54,817,524,224 + shard2 (PLE table) 28,800,138,432
# froggeric's fixed Qwen chat template. Fetched from the author's own repository at
# install time — never bundled here — pinned by revision and verified by sha256.
TEMPLATE_REPO="${TEMPLATE_REPO:-froggeric/Qwen-Fixed-Chat-Templates}"
TEMPLATE_REV="${TEMPLATE_REV:-855bffc49448e299789730ff92c9b8d834d6cc14}"
TEMPLATE_FILE="${TEMPLATE_FILE:-chat_template.jinja}"
TEMPLATE_SHA256="${TEMPLATE_SHA256:-e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2}"

# ---- defaults / args -------------------------------------------------------
GPUS=""
DATA_DIR="${DATA_DIR:-$REPO_DIR/Strata-data}"
MODELS_ROOT="$DATA_DIR/models"        # setup creates MODELS_ROOT/IQ3_S/ for the download
GGUF_DIR=""                           # the folder holding both shards (resolved below)
SKIP_MODEL=0
SKIP_MODEL_SET=0
ASSUME_YES=0
CALIBRATE=0
START=0
DRY_RUN=0
CONTEXT=262144
PORT=8001

usage() { sed -n '3,43p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus)      GPUS="${2:?}"; shift 2 ;;
    --model-dir|--gguf-dir) GGUF_DIR="${2:?}"; SKIP_MODEL=1; SKIP_MODEL_SET=1; shift 2 ;;
    --data-dir)  DATA_DIR="${2:?}"; shift 2 ;;
    --skip-model) SKIP_MODEL=1; SKIP_MODEL_SET=1; shift ;;
    --calibrate) CALIBRATE=1; shift ;;
    --context)   CONTEXT="${2:?}"; shift 2 ;;
    --port)      PORT="${2:?}"; shift 2 ;;
    --start)     START=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    -y|--yes)    ASSUME_YES=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$GPUS" ]; then
  echo "error: --gpus is required; use ./install-strata-1gpu.sh or ./install-strata-2gpu.sh" >&2
  exit 2
fi
case "$GPUS" in 1|2) ;; *) echo "error: --gpus must be 1 or 2" >&2; exit 2 ;; esac
case "$CONTEXT" in 262144|524288|1048576) ;; *) echo "error: --context must be 262144, 524288 or 1048576" >&2; exit 2 ;; esac
STRATA_DIR="$REPO_DIR/Strata"
case "$CONTEXT" in
  262144)  CFG_SRC="$REPO_DIR/configs/config-${GPUS}gpu.json" ;;
  524288)  CFG_SRC="$REPO_DIR/configs/config-${GPUS}gpu-500k.json" ;;
  1048576) CFG_SRC="$REPO_DIR/configs/config-${GPUS}gpu-1m.json" ;;
esac
GGUF_DIR="${GGUF_DIR:-$MODELS_ROOT/IQ3_S}"   # setup's own layout for a downloaded variant
MODELS_DIR="$MODELS_ROOT"                    # where setup drops the mmproj vision encoder

say()  { printf '\n== %s\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die()  { printf '\nerror: %s\n' "$*" >&2; exit 1; }

# ---- preflight -------------------------------------------------------------
say "Preflight ($GPUS GPU layout, Strata)"

[ "$(uname -s)" = "Linux" ] || die "this installer targets Linux (found $(uname -s))"
[ "$(uname -m)" = "x86_64" ] || die "x86_64 only for the engine build (found $(uname -m))"

for c in nvidia-smi git curl python3 cmake ninja; do
  command -v "$c" >/dev/null 2>&1 || die "missing required command: $c (Strata has no prebuilt Linux engine: the installer compiles it with cmake + ninja)"
done
command -v nvcc >/dev/null 2>&1 || die "the engine build needs the CUDA toolkit (nvcc not found)"
note "tools: ok (nvidia-smi, git, curl, python3, cmake, ninja, nvcc)"

GPU_LINES="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
GPU_COUNT="$(printf '%s\n' "$GPU_LINES" | grep -c . || true)"
note "GPUs detected: $GPU_COUNT"
printf '%s\n' "$GPU_LINES" | sed 's/^/     /'
[ "$GPU_COUNT" -ge "$GPUS" ] || die "this layout needs $GPUS GPU(s), found $GPU_COUNT"
MIN_VRAM_MIB=14336    # ~14 GiB; the tuned configs assume 16 GB cards
while IFS= read -r line; do
  vram="$(printf '%s' "$line" | awk -F', ' '{gsub(/ MiB/,"",$2); print $2}')"
  if [ "${vram:-0}" -lt "$MIN_VRAM_MIB" ]; then
    note "warning: a GPU reports ${vram} MiB VRAM; these configs are tuned for 16 GB cards"
  fi
done <<< "$GPU_LINES"

# CUDA: sm_120 (RTX 50) builds with CUDA 13; CUDA >= 12.4 covers sm_86+.
CUDA_VER="$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
MIN_CAP="$(LC_ALL=C nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
  | tr -d ' ' | LC_ALL=C awk 'NF{print $1+0}' | LC_ALL=C sort -n | head -1 || true)"
note "nvcc: ${CUDA_VER:-unknown}; lowest GPU compute capability: ${MIN_CAP:-unknown}"
if [ -n "${MIN_CAP:-}" ] && LC_ALL=C awk -v c="$MIN_CAP" 'BEGIN{exit !(c+0 >= 12.0)}' && \
   [ -n "${CUDA_VER:-}" ] && LC_ALL=C awk -v v="$CUDA_VER" 'BEGIN{exit !(v+0 < 13.0)}'; then
  die "sm_120 (RTX 50-series) needs the CUDA 13 toolkit to build the engine (found nvcc $CUDA_VER)"
fi

MEM_TOTAL_KIB="$(awk '/^MemTotal:/{print $2}' /proc/meminfo)"
MEM_TOTAL_GIB=$(( MEM_TOTAL_KIB / 1024 / 1024 ))
note "system RAM: ${MEM_TOTAL_GIB} GiB"
# experts (~46.8 GiB) + the 26.9 GiB n-gram table resident + KV pages + OS
[ "$MEM_TOTAL_GIB" -ge 62 ] || die "at least ~62 GiB RAM is required (128 GB recommended); found ${MEM_TOTAL_GIB} GiB"
[ "$MEM_TOTAL_GIB" -ge 110 ] || note "warning: 128 GiB is recommended; ${MEM_TOTAL_GIB} GiB will be tight with --ple-io ram and parking"

mkdir -p "$MODELS_ROOT"
AVAIL_GB=$(( $(df -B1 --output=avail "$MODELS_ROOT" | tail -1) / 1000 / 1000 / 1000 ))
if [ "$SKIP_MODEL" -eq 0 ]; then
  note "free disk at $MODELS_ROOT: ${AVAIL_GB} GB (need ~122 GB: model 83.6 GB + pack ~1.5 GB + MTP ~5 GB + vision mmproj ~0.9 GB + engine build)"
  [ "$AVAIL_GB" -ge 115 ] || die "not enough free disk space for the model download"
else
  note "free disk at $MODELS_ROOT: ${AVAIL_GB} GB (model skipped)"
fi

PY_TAG="$(python3 -c 'import sys; print(".".join(map(str,sys.version_info[:3])))')"
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' \
  || die "python >= 3.10 required for Strata's setup (found $PY_TAG)"
note "python: $PY_TAG (Strata's setup.sh creates its own venv inside the clone)"

if [ "$DRY_RUN" -eq 1 ]; then
  say "Dry run — nothing will be changed"
  note "Strata clone:   $STRATA_DIR @ ${STRATA_COMMIT:0:10} (tag v0.1.42), CUDA engine built locally (--build)"
  note "model:          $MODEL_REPO @ ${MODEL_REV:0:10}, variant $MODEL_VARIANT"
  note "GGUF dir:       $GGUF_DIR"
  note "data dir:       $DATA_DIR (packs/, mtp/)"
  note "vision:         setup --vision gpu: the strata-vision CUDA encoder is built and the"
  note "                mmproj encoder (~0.9 GB) downloaded; every profile serves images"
  note "calibration:    $([ "$CALIBRATE" -eq 1 ] && echo "./setup.sh --calibrate (per-PC, ~10 min)" || echo "skipped (--calibrate adds it; the configs keep the author's measured values)")"
  note "run config:     $(basename "$CFG_SRC") -> Strata/strata-iq3_s.json (paths resolved for this machine)"
  note "chat template:  froggeric's fixed template, revision ${TEMPLATE_REV:0:10}, sha256-verified,"
  note "                written into the pack tokenizer dir (with the xhigh effort patch)"
  note "api key:        generated locally into Strata/api_key.txt if absent"
  note "after install:  ./serve-2gpu-strata.sh  (or 1gpu; CTX=500k|1m for the extended windows)"
  exit 0
fi

# ---- 1. Strata clone -------------------------------------------------------
say "Strata clone (pinned ${STRATA_COMMIT:0:10})"
if [ -d "$STRATA_DIR/.git" ]; then
  git -C "$STRATA_DIR" fetch --all --quiet
else
  git clone --quiet "$STRATA_REPO" "$STRATA_DIR"
fi
git -C "$STRATA_DIR" checkout --quiet "$STRATA_COMMIT"
note "HEAD: $(git -C "$STRATA_DIR" log -1 --format='%h %s')"

# ---- 2. Strata setup (engine build, model download, pack, MTP) ----
# setup.sh is idempotent: completed steps (venv, engine build,
# download, pack, MTP) are recognized and skipped on a re-run.
say "Strata setup (engine build + model + pack + MTP draft; takes a while)"
SETUP_ARGS=(--yes --family qwen --model "$MODEL_VARIANT" --context "$CONTEXT" --kv int8 --vision gpu --no-start --build
            --data-dir "$DATA_DIR" --models-dir "$MODELS_ROOT")
if [ "$GPUS" = "2" ]; then
  SETUP_ARGS+=(--gpus "0,1")
else
  SETUP_ARGS+=(--gpu 0)
fi
if [ "$SKIP_MODEL" -eq 1 ]; then
  SETUP_ARGS+=(--gguf-dir "$GGUF_DIR")
  [ -d "$GGUF_DIR" ] || die "--skip-model/--model-dir: no such folder: $GGUF_DIR (it must hold both IQ3_S shards)"
  note "--skip-model: setup reuses the GGUFs already in $GGUF_DIR"
fi
( cd "$STRATA_DIR" && ./setup.sh "${SETUP_ARGS[@]}" )

if [ "$CALIBRATE" -eq 1 ]; then
  say "Per-PC calibration (--pcie-frac / --spec-min-p, ~10 min)"
  ( cd "$STRATA_DIR" && ./setup.sh --calibrate --no-start )
fi

# ---- 3. the API key --------------------------------------------------------
say "API key (generated locally; never printed, never in git)"
KEY_FILE="$STRATA_DIR/api_key.txt"
if [ ! -f "$KEY_FILE" ]; then
  umask 077
  python3 -c 'import secrets; print(secrets.token_hex(32))' > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  note "generated a fresh key in Strata/api_key.txt"
else
  note "keeping the existing Strata/api_key.txt"
fi

# ---- 4. chat template (froggeric's fixed Qwen template, with the xhigh patch)
say "Chat template (froggeric's fixed Qwen template)"
PACK_TOK_DIR="$DATA_DIR/packs/$(printf '%s' "$MODEL_VARIANT" | tr 'A-Z' 'a-z')/tokenizer"
[ -d "$PACK_TOK_DIR" ] || die "pack tokenizer dir missing: $PACK_TOK_DIR (did the setup finish?)"
TEMPLATE_PATH="$PACK_TOK_DIR/chat_template.jinja"
if grep -q 'froggeric-v' "$TEMPLATE_PATH" 2>/dev/null; then
  note "the pack already carries froggeric's template; refreshing it from the pinned revision"
fi
# verify the pinned upstream file BEFORE touching the pack
TMP_TPL="$(mktemp)"
curl -sL "https://huggingface.co/${TEMPLATE_REPO}/resolve/${TEMPLATE_REV}/${TEMPLATE_FILE}" -o "$TMP_TPL" \
  || die "could not download the chat template from $TEMPLATE_REPO"
GOT_SHA="$(sha256sum "$TMP_TPL" | cut -d' ' -f1)"
if [ "$GOT_SHA" != "$TEMPLATE_SHA256" ]; then
  rm -f "$TMP_TPL"
  die "chat template checksum mismatch (expected ${TEMPLATE_SHA256:0:12}…, got ${GOT_SHA:0:12}…); upstream may have moved — check ${TEMPLATE_REPO} before trusting the pin"
fi
TEMPLATE_STAMP="$(grep -oE 'froggeric-v[0-9]+(\.[0-9]+)?' "$TMP_TPL" | head -1)"
# keep the original template alongside (Strata's packs carry the stock one)
if [ -f "$TEMPLATE_PATH" ] && ! grep -q 'froggeric-v' "$TEMPLATE_PATH" && [ ! -f "$PACK_TOK_DIR/chat_template.orig.jinja.bak" ]; then
  cp "$TEMPLATE_PATH" "$PACK_TOK_DIR/chat_template.orig.jinja.bak"
  note "the pack's stock template is kept as chat_template.orig.jinja.bak"
fi
# the one local change: pin the default reasoning effort to xhigh
# (upstream's default renders 'medium'; per-request reasoning_effort still wins).
# Exact string replace + verify — a silently un-patched template would serve
# medium effort with no visible error.
if ! python3 - "$TMP_TPL" <<'PY'
import sys
p = sys.argv[1]
src = open(p, encoding="utf-8").read()
MID = "{%- set _default_reasoning_effort = 'medium' %}"
XHI = ("{#- fixed install: default reasoning effort pinned to xhigh "
       "(per-request reasoning_effort still overrides) -#}\n"
       "{%- set _default_reasoning_effort = 'xhigh' %}")
if src.count(MID) != 1:
    sys.exit(f"expected exactly one default-effort line, found {src.count(MID)}")
if "_default_reasoning_effort = 'xhigh'" in src:
    sys.exit(0)          # already patched (idempotent re-run)
open(p, "w", encoding="utf-8").write(src.replace(MID, XHI))
PY
then
  die "could not apply the xhigh reasoning-effort patch to the chat template"
fi
grep -q "_default_reasoning_effort = 'xhigh'" "$TMP_TPL" \
  || die "the xhigh patch did not take effect on the chat template"
note "default reasoning effort patched: medium -> xhigh"
PATCHED_SHA="$(sha256sum "$TMP_TPL" | cut -d' ' -f1)"
cp "$TMP_TPL" "$TEMPLATE_PATH"
rm -f "$TMP_TPL"
note "templates: $TEMPLATE_PATH  ${TEMPLATE_STAMP:-} (upstream sha256 ${TEMPLATE_SHA256:0:12}…, installed ${PATCHED_SHA:0:12}…)"
note "author: froggeric — https://huggingface.co/${TEMPLATE_REPO} (Apache-2.0)"
note "downloaded at install time, not redistributed with this repo"
cat > "$PACK_TOK_DIR/chat_template.README.txt" <<EOF
chat_template.jinja — fixed Qwen chat template by froggeric.
Version stamp: ${TEMPLATE_STAMP:-unknown}
Source: https://huggingface.co/${TEMPLATE_REPO}
Revision: $TEMPLATE_REV (pinned)
Upstream sha256: $TEMPLATE_SHA256
Installed sha256: $PATCHED_SHA  (one local change: _default_reasoning_effort 'medium' -> 'xhigh')
License: Apache-2.0 (see the upstream repository)

Fetched by this repository's installer from the author's own repository; it is not
redistributed with the installer. Please credit froggeric if you pass it on.
The pack's stock template is kept next to it as chat_template.orig.jinja.bak.
EOF

# ---- 5. the run config (paths resolved + key injected) -------
say "Run config"
python3 "$REPO_DIR/scripts/finalize-strata.py" "$CFG_SRC" \
  --out "$STRATA_DIR/strata-iq3_s.json" \
  --strata-dir "$STRATA_DIR" --data-dir "$DATA_DIR" --gguf-dir "$GGUF_DIR" \
  --models-dir "$MODELS_DIR" --key-file "$KEY_FILE" --port "$PORT"
# the server-side default for clients that send no reasoning_effort (the template
# default already pins xhigh; this matches the shared Chat-settings file too)
SHARED="$STRATA_DIR/strata-iq3_s.shared-settings.json"
if [ ! -f "$SHARED" ]; then
  printf '{\n "reasoning_effort": "high"\n}\n' > "$SHARED"   # "high" maps to xhigh here
  note "wrote $SHARED (reasoning_effort=high for clients that send none)"
else
  note "keeping the existing $SHARED"
fi

# ---- done ------------------------------------------------------------------
cat <<EOF

============================================================================
Install complete ($GPUS GPU layout, Strata IQ3_S, ${CONTEXT} context finalized)

Start:
    $REPO_DIR/serve-${GPUS}gpu-strata.sh                    # 262k (default)
    CTX=500k $REPO_DIR/serve-${GPUS}gpu-strata.sh               # 524,288 (experimental yarn 2)
    CTX=1m   $REPO_DIR/serve-${GPUS}gpu-strata.sh               # 1,048,576 (experimental yarn 4)

Then, in another terminal:
    $REPO_DIR/verify-strata.sh                # health + context length + prompts

The API key lives in $KEY_FILE (chmod 600).
Read it with:  cat $KEY_FILE
Point any OpenAI-compatible client at http://127.0.0.1:$PORT/v1 with
model name qwen3.8-flash-next-iq3_s. The chat UI is at http://127.0.0.1:$PORT/.

Notes:
  * Re-running this installer is safe and idempotent: the clone, engine, venv,
    pack, configs and template are refreshed; the model and the API key are kept.
  * If you move the model later, re-run this installer with --model-dir <new path>.
  * Chat template: froggeric's fixed Qwen template (Apache-2.0, fetched at
    install time; xhigh pinned) — see the install log above.
  * Tuning notes and measured numbers: README.md
============================================================================
EOF

if [ "$START" -eq 1 ]; then
  say "Starting the server"
  exec "$REPO_DIR/serve-${GPUS}gpu-strata.sh"
fi
