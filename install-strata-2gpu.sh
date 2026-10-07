#!/usr/bin/env bash
# Wrapper: install the Strata stack for TWO 16 GB GPUs.
# All arguments are forwarded to install-strata.sh  (./install-strata.sh --help for options).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install-strata.sh" --gpus 2 "$@"
