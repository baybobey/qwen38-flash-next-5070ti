#!/usr/bin/env bash
# Wrapper: install the stack for TWO 16 GB GPUs.
# All arguments are forwarded to install.sh  (./install.sh --help for options).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install.sh" --gpus 2 "$@"
