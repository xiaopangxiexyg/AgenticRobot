#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REQUIRE_TACTILE_DEFAULT="${REQUIRE_TACTILE_DEFAULT:-0}"
export HAND_TACTILE_REGIONS_DEFAULT="${HAND_TACTILE_REGIONS_DEFAULT:-}"
exec "${SCRIPT_DIR}/run_policy.sh" "$@"
