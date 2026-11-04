#!/usr/bin/env bash
set -euo pipefail

# Same lifecycle and evidence implementation as full acceptance, core scope.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash scripts/run_merge_acceptance.sh [--durations=N] [-q|-v|-vv]"
  echo "Run core merge checks in four batches with two live pytest processes by default;"
  echo "DEXTELEOP_FULL_ACCEPTANCE_SHARDS/_MAX_JOBS may select up to four live processes"
  echo "under the memory guard in tests/README.md. Full matrices remain separate."
  exit 0
fi
exec bash "$SCRIPT_DIR/run_full_acceptance.sh" --core "$@"
