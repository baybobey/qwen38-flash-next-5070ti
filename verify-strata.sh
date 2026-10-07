#!/usr/bin/env bash
# ============================================================================
# verify-strata.sh — smoke-test the running Strata server.
#
#   ./verify-strata.sh                      (defaults: port 8001)
#   PORT=8002 ./verify-strata.sh
#
# The API key is read from Strata/api_key.txt at runtime and never printed.
# The expected context length is read from the active run config
# (Strata/strata-iq3_s.json, --max-context), so it matches the profile you started
# (262144 | 524288 | 1048576).
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$REPO_DIR/Strata"
PY="$STRATA_DIR/.venv/bin/python"
PORT="${PORT:-8001}"
BASE="http://127.0.0.1:$PORT"

[ -f "$STRATA_DIR/api_key.txt" ] || { echo "no Strata/api_key.txt — run ./install-strata-<N>gpu.sh first" >&2; exit 1; }
[ -x "$PY" ] || { echo "Strata venv missing — run the installer first" >&2; exit 1; }
KEY="$(cat "$STRATA_DIR/api_key.txt")"

CTX_EXPECT="$("$PY" - "$STRATA_DIR/strata-iq3_s.json" <<'PY'
import json, sys
args = json.load(open(sys.argv[1]))["args"]
print(args[args.index("--max-context") + 1] if "--max-context" in args else "?")
PY
)"
MODEL_NAME="$("$PY" - "$STRATA_DIR/strata-iq3_s.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("model_name", "qwen3.8-flash-next-iq3_s"))
PY
)"

echo "== GET /health =="
curl -sS --max-time 15 "$BASE/health" || { echo; echo "server not reachable on $BASE"; exit 1; }
echo

echo "== GET /v1/models (context length must be $CTX_EXPECT) =="
curl -sS --max-time 15 "$BASE/v1/models" -H "Authorization: Bearer ***" \
  | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
for m in d.get("data", []):
    meta = m.get("meta") or {}
    print("   %s  n_ctx=%s" % (m.get("id"), meta.get("n_ctx")))
'

echo
echo "== POST /v1/chat/completions (small prompt; sampler defaults apply) =="
curl -sS --max-time 180 "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer ***" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one short sentence: are you ready?\"}],\"max_tokens\":256}" \
  | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
ch = (d.get("choices") or [{}])[0]
msg = ch.get("message") or {}
print("   finish_reason:", ch.get("finish_reason"))
print("   reasoning:", (msg.get("reasoning_content") or "")[:160].replace("\n", " "))
print("   content:  ", (msg.get("content") or "").strip()[:160].replace("\n", " "))
print("   usage:    ", d.get("usage"))
'

echo
echo "== functional suite (health, tool calls, multi-turn, Anthropic) =="
"$PY" "$REPO_DIR/scripts/smoke-strata.py" --port "$PORT"

cat <<EOF

Expected: healthy, n_ctx $CTX_EXPECT, a short answer with reasoning kept out of
content, and smoke 4/4. Streaming answers with 'strata serve: prompt ...'
lines in Strata/strata-iq3_s.log show the engine's own prompt/decode tok/s.
EOF
