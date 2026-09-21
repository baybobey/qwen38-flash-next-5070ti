#!/usr/bin/env bash
# ============================================================================
# install.sh — installer for the Qwen3.8-Flash-Next EXL3 serving stack.
#
# Use the wrappers instead of calling this directly:
#     ./install-1gpu.sh [options]     1x 16 GB GPU
#     ./install-2gpu.sh [options]     2x 16 GB GPUs
#
# Installs into this directory (plus the model, which defaults to
# $HOME/models/Qwen3.8-Flash-Next):
#     .venv/            torch + exllamav3 + TabbyAPI dependencies + hf CLI
#     tabbyAPI/         TabbyAPI clone, pinned to a known-good commit
#     config.yml        written from configs/config-<N>gpu.yml (+ models/ symlink,
#                       + sampler_overrides/ preset)
#     api_tokens.yml    generated locally with a random API key, chmod 600
#
# Options:
#     --gpus N         1 or 2 (set by the wrapper)
#     --model-dir D    where the model lives / is downloaded to
#     --venv D         venv location (default: .venv next to this script)
#     --skip-model     do not download the model (must already exist in D)
#     --start          start the server when the install finishes
#     --dry-run        show what would be installed, change nothing
#     --latest-template
#                      download upstream's current template file instead of the pinned
#                      revision (unpinned; the sha256 it received is printed)
#     --official-template
#                      use Qwen's own chat template from the model dir instead of
#                      downloading froggeric's fixed template
#     -y, --yes        assume yes (no prompts)
#     -h, --help       show this help
#
# Nothing secret is stored in this repository: the API key is generated on this
# machine at install time and written only to tabbyAPI/api_tokens.yml.
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- pinned versions (override via env if you know what you are doing) ------
MODEL_REPO="${MODEL_REPO:-turboderp/Qwen3.8-Flash-Next-exl3}"
MODEL_BRANCH="${MODEL_BRANCH:-4.05bpw_h6_ng6}"
MODEL_REV="${MODEL_REV:-55a732e0c4c3d4614bc42b68493bb930d9b02c0a}"   # branch head 2026-09-12
MODEL_NAME="${MODEL_NAME:-Qwen3.8-Flash-Next-EXL3-405}"              # directory name under tabbyAPI/models
MODEL_TOTAL_BYTES="${MODEL_TOTAL_BYTES:-107463600896}"               # 100.1 GiB (manifest total)
TABBY_REPO="${TABBY_REPO:-https://github.com/theroyallab/tabbyAPI}"
TABBY_COMMIT="${TABBY_COMMIT:-72082731339f8b8a04601bdf1b58605914d01fcb}"
# froggeric's fixed Qwen chat template. Fetched from the author's own repository at
# install time — never bundled here — pinned by revision and verified by sha256.
# Upstream: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates (Apache-2.0)
TEMPLATE_REPO="${TEMPLATE_REPO:-froggeric/Qwen-Fixed-Chat-Templates}"
TEMPLATE_REV="${TEMPLATE_REV:-855bffc49448e299789730ff92c9b8d834d6cc14}"
TEMPLATE_FILE="${TEMPLATE_FILE:-chat_template.jinja}"
TEMPLATE_SHA256="${TEMPLATE_SHA256:-e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2}"
TEMPLATE_NAME="${TEMPLATE_NAME:-froggeric-qwen38-v225}"   # no dots: tabbyAPI resolves it with Path.with_suffix
EXL3_VERSION="${EXL3_VERSION:-1.5.1}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

# ---- defaults / args -------------------------------------------------------
GPUS=""
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next}"
VENV_DIR=""
SKIP_MODEL=0
ASSUME_YES=0
START=0
DRY_RUN=0
OFFICIAL_TEMPLATE=0
LATEST_TEMPLATE=0

usage() { sed -n '3,31p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus)      GPUS="${2:?}"; shift 2 ;;
    --model-dir) MODEL_DIR="${2:?}"; shift 2 ;;
    --venv)      VENV_DIR="${2:?}"; shift 2 ;;
    --skip-model) SKIP_MODEL=1; shift ;;
    --start)     START=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --official-template) OFFICIAL_TEMPLATE=1; shift ;;
    --latest-template) LATEST_TEMPLATE=1; shift ;;
    -y|--yes)    ASSUME_YES=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$GPUS" ]; then
  echo "error: --gpus is required; use ./install-1gpu.sh or ./install-2gpu.sh" >&2
  exit 2
fi
case "$GPUS" in 1|2) ;; *) echo "error: --gpus must be 1 or 2" >&2; exit 2 ;; esac
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
TABBY_DIR="$REPO_DIR/tabbyAPI"
CONFIG_SRC="$REPO_DIR/configs/config-${GPUS}gpu.yml"
CANONICAL_MODEL_DIR="$HOME/models/Qwen3.8-Flash-Next"

say()  { printf '\n== %s\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die()  { printf '\nerror: %s\n' "$*" >&2; exit 1; }

# ---- preflight -------------------------------------------------------------
say "Preflight ($GPUS GPU layout)"

[ "$(uname -s)" = "Linux" ] || die "this installer targets Linux (found $(uname -s))"
[ "$(uname -m)" = "x86_64" ] || die "x86_64 only for the prebuilt wheels (found $(uname -m))"

for c in nvidia-smi git curl python3; do
  command -v "$c" >/dev/null 2>&1 || die "missing required command: $c"
done
note "tools: ok (nvidia-smi, git, curl, python3)"

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

DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)"
note "driver CUDA version: ${DRIVER_CUDA:-unknown}"
if [ -n "$DRIVER_CUDA" ]; then
  major="${DRIVER_CUDA%%.*}"; minor="${DRIVER_CUDA##*.}"
  if [ "$major" -lt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -lt 8 ]; }; then
    die "CUDA ${DRIVER_CUDA} driver is too old for the ${TORCH_CUDA} packages (need >= 12.8, driver >= 570)"
  fi
fi

MEM_TOTAL_KIB="$(awk '/^MemTotal:/{print $2}' /proc/meminfo)"
MEM_TOTAL_GIB=$(( MEM_TOTAL_KIB / 1024 / 1024 ))
note "system RAM: ${MEM_TOTAL_GIB} GiB"
[ "$MEM_TOTAL_GIB" -ge 90 ] || die "at least ~90 GiB RAM is required (128 GB recommended); found ${MEM_TOTAL_GIB} GiB"
[ "$MEM_TOTAL_GIB" -ge 110 ] || note "warning: 128 GiB is recommended; ${MEM_TOTAL_GIB} GiB will be tight with ngram_ram enabled"

PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_TAG="$(python3 -c 'import sys; print("cp%d%d" % sys.version_info[:2])')"
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' \
  || die "python >= 3.10 required (found $PY_VERSION)"
note "python: $PY_VERSION ($PY_TAG)"

# exllamav3 release wheels. The prebuilt ones carry kernels for sm_80/86/89/90/100/120
# (checked with cuobjdump --list-elf); a GPU older than sm_80 has no kernel image in
# them and must use the JIT wheel, which compiles the CUDA sources for the local arch
# and therefore needs the CUDA toolkit (nvcc).
EXL3_BASE="https://github.com/turboderp-org/exllamav3/releases/download/v${EXL3_VERSION}"
EXL3_PREBUILT="${EXL3_BASE}/exllamav3-${EXL3_VERSION}%2B${TORCH_CUDA}.torch${TORCH_VERSION}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
EXL3_JIT="${EXL3_BASE}/exllamav3-${EXL3_VERSION}-py3-none-any.whl"

http_code() { curl -sIL -o /dev/null -w '%{http_code}' "$1"; }

MIN_CAP="$(LC_ALL=C nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
  | tr -d ' ' | LC_ALL=C awk 'NF{print $1+0}' | LC_ALL=C sort -n | head -1 || true)"
PREFER_JIT=0
if [ -n "${EXL3_FORCE_JIT:-}" ]; then
  PREFER_JIT=1
  note "EXL3_FORCE_JIT is set: using the JIT engine build"
fi
if [ -n "$MIN_CAP" ]; then
  note "compute capability (lowest GPU): $(LC_ALL=C awk -v c="$MIN_CAP" 'BEGIN{printf "%.1f", c}')"
  if LC_ALL=C awk -v c="$MIN_CAP" 'BEGIN{exit !(c+0 < 8.0)}'; then
    PREFER_JIT=1
    note "note: the prebuilt engine wheel only carries kernels for sm_80+; this GPU needs"
    note "      the JIT build (compiled at first import, requires the CUDA toolkit)"
  fi
else
  note "compute capability: could not be queried, assuming the prebuilt wheel fits"
fi
if [ "$PREFER_JIT" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
  command -v nvcc >/dev/null 2>&1 || die "this GPU needs the JIT engine build, which requires the CUDA toolkit (nvcc not found)"
fi

MODEL_CHECK_DIR="$MODEL_DIR"
while [ ! -d "$MODEL_CHECK_DIR" ] && [ "$MODEL_CHECK_DIR" != "/" ]; do MODEL_CHECK_DIR="$(dirname "$MODEL_CHECK_DIR")"; done
AVAIL_GB=$(( $(df -B1 --output=avail "$MODEL_CHECK_DIR" | tail -1) / 1000 / 1000 / 1000 ))
if [ "$SKIP_MODEL" -eq 0 ] && [ ! -f "$MODEL_DIR/config.json" ]; then
  note "free disk at $MODEL_CHECK_DIR: ${AVAIL_GB} GB (need ~$(( MODEL_TOTAL_BYTES / 1000 / 1000 / 1000 + 15 )) GB for model + venv)"
  [ "$AVAIL_GB" -ge 115 ] || die "not enough free disk space for the model download"
else
  note "free disk at $MODEL_CHECK_DIR: ${AVAIL_GB} GB"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  say "Dry run — nothing will be changed"
  note "venv:           $VENV_DIR"
  note "torch:          ${TORCH_VERSION}+${TORCH_CUDA} (index ${TORCH_INDEX})"
  if [ "$PREFER_JIT" -eq 1 ]; then
    note "engine:         JIT build (CUDA kernels compile at first import, nvcc required)"
  else
    note "engine:         prebuilt wheel preferred"
  fi
  if [ -n "${EXLL3_WHEEL_URL:-}" ]; then
    note "wheel:          ${EXLL3_WHEEL_URL} (from the environment)"
  elif [ "$PREFER_JIT" -eq 0 ] && [ "$(http_code "$EXL3_PREBUILT")" = "200" ]; then
    note "wheel:          $EXL3_PREBUILT"
  elif [ "$(http_code "$EXL3_JIT")" = "200" ]; then
    note "wheel:          $EXL3_JIT"
  else
    note "wheel:          not found on the v${EXL3_VERSION} release page"
  fi
  note "tabbyAPI:       $TABBY_DIR @ ${TABBY_COMMIT:0:10}"
  note "config:         $(basename "$CONFIG_SRC") -> tabbyAPI/config.yml"
  if [ "$OFFICIAL_TEMPLATE" -eq 1 ]; then
    note "template:       Qwen's own (--official-template); nothing downloaded"
  else
    note "template:       froggeric's fixed template, revision ${TEMPLATE_REV:0:10},"
    note "                downloaded to tabbyAPI/templates/${TEMPLATE_NAME}.jinja and sha256-verified"
  fi
  if [ "$SKIP_MODEL" -eq 1 ]; then
    note "model:          skipped (--skip-model)"
  elif [ -f "$MODEL_DIR/config.json" ]; then
    note "model:          already present at $MODEL_DIR"
  else
    note "model:          download ~$(( MODEL_TOTAL_BYTES / 1000 / 1000 / 1000 + 1 )) GB to $MODEL_DIR"
  fi
  note "api key:        generated locally if tabbyAPI/api_tokens.yml is absent"
  exit 0
fi

# ---- venv + python packages ------------------------------------------------
say "Python venv"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
  note "created $VENV_DIR"
else
  note "reusing $VENV_DIR"
fi
PIP="$VENV_DIR/bin/pip"
"$PIP" install --upgrade pip wheel setuptools >/dev/null
note "pip: $("$VENV_DIR/bin/python" -m pip --version | awk '{print $2}')"

say "Installing torch ${TORCH_VERSION}+${TORCH_CUDA}"
if "$VENV_DIR/bin/python" -c 'import torch' 2>/dev/null; then
  note "torch already importable: $("$VENV_DIR/bin/python" -c 'import torch; print(torch.__version__)')"
else
  "$PIP" install "torch==${TORCH_VERSION}+${TORCH_CUDA}" --index-url "$TORCH_INDEX"
  note "installed $("$VENV_DIR/bin/python" -c 'import torch; print(torch.__version__)')"
fi

say "Installing exllamav3 ${EXL3_VERSION}"
EXL3_URL=""
if [ -n "${EXLL3_WHEEL_URL:-}" ]; then
  EXL3_URL="$EXLL3_WHEEL_URL"; note "using EXLL3_WHEEL_URL from the environment"
elif [ "$PREFER_JIT" -eq 0 ] && [ "$(http_code "$EXL3_PREBUILT")" = "200" ]; then
  EXL3_URL="$EXL3_PREBUILT"; note "prebuilt wheel for ${TORCH_CUDA}/torch${TORCH_VERSION}/${PY_TAG} (sm_80+; no nvcc needed)"
elif [ "$(http_code "$EXL3_JIT")" = "200" ]; then
  EXL3_URL="$EXL3_JIT"
  note "using the JIT wheel (CUDA kernels compile at first import)"
  command -v nvcc >/dev/null 2>&1 || die "the JIT wheel needs the CUDA toolkit (nvcc not found); install it, or install a matching torch/python and re-run"
else
  die "could not find an exllamav3 v${EXL3_VERSION} wheel on the release page"
fi
# let pip resolve the wheel's own requirements (torch is already installed and
# satisfies its constraint, so pip leaves it alone)
"$PIP" install "$EXL3_URL"
note "installed $("$VENV_DIR/bin/python" -c 'from importlib.metadata import version; print("exllamav3", version("exllamav3"))')"

say "Import check (first import compiles CUDA kernels; can take several minutes)"
"$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata as m
import exllamav3
print("   exllamav3", m.version("exllamav3"), "imports OK")
PY

say "TabbyAPI clone (pinned ${TABBY_COMMIT:0:10})"
if [ -d "$TABBY_DIR/.git" ]; then
  git -C "$TABBY_DIR" fetch --all --quiet
else
  git clone --quiet "$TABBY_REPO" "$TABBY_DIR"
fi
git -C "$TABBY_DIR" checkout --quiet "$TABBY_COMMIT"
note "HEAD: $(git -C "$TABBY_DIR" log -1 --format='%h %s')"

say "TabbyAPI dependencies"
"$VENV_DIR/bin/python" -c 'import tomllib' 2>/dev/null || "$PIP" install tomli >/dev/null
"$VENV_DIR/bin/python" - "$TABBY_DIR/pyproject.toml" "$PIP" <<'PY'
import subprocess, sys
try:
    import tomllib
except ModuleNotFoundError:          # Python 3.10
    import tomli as tomllib
pyproject, pip = sys.argv[1], sys.argv[2]
deps = tomllib.load(open(pyproject, "rb"))["project"]["dependencies"]
print(f"   installing {len(deps)} dependencies")
subprocess.run([pip, "install", *deps], check=True)
PY

say "Hugging Face CLI"
"$PIP" install --upgrade "huggingface_hub[hf_xet]" >/dev/null 2>&1 || "$PIP" install --upgrade huggingface_hub >/dev/null
note "hf: $("$VENV_DIR/bin/hf" version 2>/dev/null | head -1 || echo installed)"

# ---- model -----------------------------------------------------------------
say "Model: $MODEL_REPO @ $MODEL_BRANCH (${MODEL_REV:0:10})"
if [ "$SKIP_MODEL" -eq 1 ]; then
  note "skipped (--skip-model)"
elif [ -f "$MODEL_DIR/config.json" ]; then
  note "already present: $MODEL_DIR"
else
  note "target: $MODEL_DIR  (~100 GiB, resumable — re-run this script if it is interrupted)"
  if [ "$ASSUME_YES" -eq 0 ] && [ -t 0 ]; then
    read -r -p "   download now? [y/N] " answer || answer="n"
    case "$answer" in y|Y|yes|YES) ;; *) SKIP_MODEL=1; note "skipped" ;; esac
  fi
  if [ "$SKIP_MODEL" -eq 0 ]; then
    "$VENV_DIR/bin/hf" download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
    note "verifying the download against the Hugging Face manifest"
    "$VENV_DIR/bin/python" "$REPO_DIR/scripts/check_download.py" "$MODEL_DIR" \
      --repo "$MODEL_REPO" --revision "$MODEL_REV" || die "download verification failed"
  fi
fi
if [ "$SKIP_MODEL" -eq 1 ] && [ ! -f "$MODEL_DIR/config.json" ]; then
  note "note: $MODEL_DIR has no config.json yet"
fi

# ---- wire up tabbyAPI ------------------------------------------------------
say "Chat template (froggeric's fixed Qwen template)"
if [ "$OFFICIAL_TEMPLATE" -eq 1 ]; then
  note "skipped (--official-template): the model's own Qwen template is used"
else
  mkdir -p "$TABBY_DIR/templates"
  TEMPLATE_PATH="$TABBY_DIR/templates/${TEMPLATE_NAME}.jinja"
  if [ "$LATEST_TEMPLATE" -eq 1 ]; then
    TEMPLATE_URL="https://huggingface.co/${TEMPLATE_REPO}/resolve/main/${TEMPLATE_FILE}"
    note "downloading from ${TEMPLATE_REPO} @ main (unpinned, as requested)"
  else
    TEMPLATE_URL="https://huggingface.co/${TEMPLATE_REPO}/resolve/${TEMPLATE_REV}/${TEMPLATE_FILE}"
    note "downloading from ${TEMPLATE_REPO} @ ${TEMPLATE_REV:0:10} (pinned)"
  fi
  curl -sL "$TEMPLATE_URL" -o "$TEMPLATE_PATH.part" \
    || die "could not download the chat template from $TEMPLATE_REPO"
  GOT_SHA="$(sha256sum "$TEMPLATE_PATH.part" | cut -d' ' -f1)"
  TEMPLATE_STAMP="$(grep -oE 'froggeric-v[0-9]+(\.[0-9]+)?' "$TEMPLATE_PATH.part" | head -1)"
  if [ "$LATEST_TEMPLATE" -eq 1 ]; then
    note "got ${TEMPLATE_STAMP:-an unstamped revision}, sha256 ${GOT_SHA:0:16}… — not checked against a pin;"
    note "if you keep it, pin it by exporting TEMPLATE_REV and TEMPLATE_SHA256 before installing"
  else
    if [ "$GOT_SHA" != "$TEMPLATE_SHA256" ]; then
      rm -f "$TEMPLATE_PATH.part"
      die "chat template checksum mismatch (expected ${TEMPLATE_SHA256:0:12}…, got ${GOT_SHA:0:12}…); upstream may have moved — check ${TEMPLATE_REPO} before trusting the pin"
    fi
    note "templates/${TEMPLATE_NAME}.jinja  ${TEMPLATE_STAMP:-}  (sha256 verified against the pin)"
    UPSTREAM_REV="$(curl -s "https://huggingface.co/api/models/${TEMPLATE_REPO}" \
      | "$VENV_DIR/bin/python" -c 'import json,sys; print(json.load(sys.stdin).get("sha",""))' 2>/dev/null || true)"
    if [ -n "${UPSTREAM_REV:-}" ] && [ "$UPSTREAM_REV" != "$TEMPLATE_REV" ]; then
      note "note: upstream has a newer revision (${UPSTREAM_REV:0:10}); this install used the pinned"
      note "      ${TEMPLATE_REV:0:10} — re-run with --latest-template to take what upstream serves now"
    fi
  fi
  note "author: froggeric — https://huggingface.co/${TEMPLATE_REPO} (Apache-2.0)"
  note "downloaded at install time, not redistributed with this repo"
  cat > "$TABBY_DIR/templates/${TEMPLATE_NAME}.README.txt" <<EOF
${TEMPLATE_NAME}.jinja — fixed Qwen chat template by froggeric.
Version stamp: ${TEMPLATE_STAMP:-unknown}
Source: https://huggingface.co/${TEMPLATE_REPO}
Revision: $([ "$LATEST_TEMPLATE" -eq 1 ] && echo "upstream main (unpinned, fetched $(date -I))" || echo "$TEMPLATE_REV (pinned)")
sha256: ${GOT_SHA}
License: Apache-2.0 (see the upstream repository)

Fetched by this repository's installer from the author's own repository; it is not
redistributed with the installer. Please credit froggeric if you pass it on.
EOF
fi

say "TabbyAPI config + API key"
cp "$CONFIG_SRC" "$TABBY_DIR/config.yml"
note "applied $(basename "$CONFIG_SRC") -> tabbyAPI/config.yml"
if [ "$OFFICIAL_TEMPLATE" -eq 1 ]; then
  sed -i 's/^\( *\)prompt_template:.*/\1prompt_template:/' "$TABBY_DIR/config.yml"
  note "config prompt_template cleared -> the model's own Qwen template is used"
else
  note "config prompt_template: ${TEMPLATE_NAME} -> templates/${TEMPLATE_NAME}.jinja"
fi
mkdir -p "$TABBY_DIR/models" "$TABBY_DIR/sampler_overrides"
if [ -f "$MODEL_DIR/config.json" ] || [ "$SKIP_MODEL" -eq 0 ]; then
  ln -sfn "$MODEL_DIR" "$TABBY_DIR/models/$MODEL_NAME"
  note "models/$MODEL_NAME -> $MODEL_DIR"
fi
cp "$REPO_DIR/sampler_overrides/qwen38_flash_next_thinking.yml" "$TABBY_DIR/sampler_overrides/"
note "installed the forced sampler preset (temp 1.0 / top-k 20 / top-p 0.95)"

if [ ! -f "$TABBY_DIR/api_tokens.yml" ]; then
  umask 077
  "$VENV_DIR/bin/python" - > "$TABBY_DIR/api_tokens.yml" <<'PY'
import secrets
print("api_key: " + secrets.token_hex(32))
print("admin_key: " + secrets.token_hex(16))
PY
  chmod 600 "$TABBY_DIR/api_tokens.yml"
  note "generated a fresh API key in tabbyAPI/api_tokens.yml (not printed, not in git)"
else
  note "keeping the existing tabbyAPI/api_tokens.yml"
fi

# ---- done ------------------------------------------------------------------
cat <<EOF

============================================================================
Install complete ($GPUS GPU layout)

Start:
    $REPO_DIR/serve-${GPUS}gpu.sh

Then, in another terminal:
    $REPO_DIR/verify.sh                 # health + context length + a test prompt

The API key lives in $TABBY_DIR/api_tokens.yml (chmod 600).
Read it with:  grep api_key $TABBY_DIR/api_tokens.yml
Point any OpenAI-compatible client at http://127.0.0.1:8001/v1 with
model name $MODEL_NAME.

Notes:
  * The model is $(LC_ALL=C awk "BEGIN{printf \"%.1f\", $MODEL_TOTAL_BYTES/1073741824}") GiB on disk and must stay next to the config:
    $MODEL_DIR
  * If you re-run this installer it is idempotent: venv, clone and config are
    refreshed, the model and the API key are kept.
  * Chat template: $([ "$OFFICIAL_TEMPLATE" -eq 1 ] && echo "Qwen's own (--official-template)" || echo "froggeric's fixed Qwen template (froggeric, Apache-2.0), fetched at install time")
    Upstream: https://huggingface.co/$TEMPLATE_REPO
  * Tuning notes and measured numbers: README.md
============================================================================
EOF

if [ "$START" -eq 1 ]; then
  say "Starting the server"
  exec "$REPO_DIR/serve-${GPUS}gpu.sh"
fi
