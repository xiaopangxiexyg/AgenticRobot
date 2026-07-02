#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACTIVE_SIDES="${ACTIVE_SIDES:-both}"
exec "${SCRIPT_DIR}/run_policy.sh" "$@"

