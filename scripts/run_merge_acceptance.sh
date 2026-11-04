#!/usr/bin/env bash
set -euo pipefail

# Same lifecycle and evidence implementation as full acceptance, core scope.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash scripts/run_merge_acceptance.sh [--durations=N] [-q|-v|-vv]"
  echo "Run core merge checks: four batches and two live pytest processes by default."
  echo "Only on the local merge host, under the memory guard in tests/README.md, set"
  echo "DEXTELEOP_FULL_ACCEPTANCE_SHARDS=8 DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS=4."
  echo "Full matrices remain separate."
  exit 0
fi
exec bash "$SCRIPT_DIR/run_full_acceptance.sh" --core "$@"
