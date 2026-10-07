#!/usr/bin/env bash
# Wrapper: serve with Strata on ONE 16 GB GPU (262144 window; CTX=500k|1m for the experimental ones).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/serve-strata.sh" --gpus 1 "$@"
