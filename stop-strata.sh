#!/usr/bin/env bash
# Stop the Strata server on a port (default 8001). Thin wrapper around
# stop-strata.py so it can run without finding a venv python first.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for PY in "$REPO_DIR/Strata/.venv/bin/python" "$(command -v python3 || true)"; do
  [ -x "$PY" ] && exec "$PY" "$REPO_DIR/stop-strata.py" "$@"
done
echo "no python found" >&2
exit 1
