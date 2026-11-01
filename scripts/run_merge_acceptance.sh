#!/usr/bin/env bash
set -euo pipefail

# Same lifecycle and evidence implementation as full acceptance, core scope.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash scripts/run_merge_acceptance.sh [--durations=N] [-q|-v|-vv]"
  echo "Run core merge checks in two bounded pytest processes; full matrices remain separate."
  exit 0
fi
exec bash "$SCRIPT_DIR/run_full_acceptance.sh" --core "$@"
