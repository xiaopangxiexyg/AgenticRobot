#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONDA_ENV="${CONDA_ENV:-ManipRep_310}"
CONFIG_NAME="${CONFIG_NAME:-realrobot_paper_fold_vanilla_tactile}"
case "$CONFIG_NAME" in
  realrobot_paper_fold_vanilla_tactile|realrobot_paper_fold_vanilla_tactile_norm) ;;
  *)
    printf 'Invalid CONFIG_NAME for paper_fold tactile script: %s\n' "$CONFIG_NAME" >&2
    printf 'Use realrobot_paper_fold_vanilla_tactile or realrobot_paper_fold_vanilla_tactile_norm.\n' >&2
    exit 2
    ;;
esac
export CONFIG_NAME
export RUN_NAME="${RUN_NAME:-realrobot_paper_fold_vtactile_vla_ac16_nohis_vis}"

exec "$SCRIPT_DIR/train_realrobot_chips_vanilla_tactile_vla.sh" "$@"
