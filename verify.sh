#!/usr/bin/env bash
# ============================================================================
# verify.sh — smoke-test a running server: health, context length, sampler.
#
#   ./verify.sh                 (defaults: port 8001, model Qwen3.8-Flash-Next-EXL3-405)
#   PORT=8001 ./verify.sh
#
# The API key is read from tabbyAPI/api_tokens.yml at runtime and never printed.
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
TABBY_DIR="$REPO_DIR/tabbyAPI"
PORT="${PORT:-8001}"
BASE="http://127.0.0.1:$PORT"
MODEL_NAME="${MODEL_NAME:-Qwen3.8-Flash-Next-EXL3-405}"

[ -f "$TABBY_DIR/api_tokens.yml" ] || { echo "no tabbyAPI/api_tokens.yml — run the installer first" >&2; exit 1; }
KEY="$("$VENV_DIR/bin/python" - "$TABBY_DIR/api_tokens.yml" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["api_key"])
PY
)"

echo "== GET /health =="
curl -sS --max-time 15 "$BASE/health" || { echo; echo "server not reachable on $BASE"; exit 1; }
echo

echo "== GET /v1/models (context length must be 262144) =="
curl -sS --max-time 15 "$BASE/v1/models" -H "Authorization: Bearer $KEY" \
  | "$VENV_DIR/bin/python" -c '
import json, sys
d = json.load(sys.stdin)
for m in d.get("data", []):
    print(f"   {m.get(\"id\")}  n_ctx={m.get(\"n_ctx\")}")
'

echo
echo "== POST /v1/chat/completions (small prompt; sampler preset applies) =="
curl -sS --max-time 120 "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one short sentence: are you ready?\"}],\"max_tokens\":64}" \
  | "$VENV_DIR/bin/python" -c '
import json, sys
d = json.load(sys.stdin)
ch = (d.get("choices") or [{}])[0]
msg = ch.get("message") or {}
print("   finish_reason:", ch.get("finish_reason"))
print("   reasoning:", (msg.get("reasoning_content") or "")[:160].replace("\n", " "))
print("   content:  ", (msg.get("content") or "").strip()[:160].replace("\n", " "))
print("   usage:    ", d.get("usage"))
'

cat <<'EOF'

Expected: healthy, n_ctx 262144, a streamed/returned answer with reasoning kept
out of content. On the server's startup log also check that it does NOT print
"No sampler override preset is configured" — that warning means clients without
sampler parameters will run untruncated (token-soup risk).
EOF
