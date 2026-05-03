#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONDA_ENV="${CONDA_ENV:-ManipRep_310}"
CONFIG_NAME="${CONFIG_NAME:-realrobot_paper_fold_vanilla_vla}"
if [[ "$CONFIG_NAME" != "realrobot_paper_fold_vanilla_vla" ]]; then
  printf 'Invalid CONFIG_NAME for paper_fold no-tactile script: %s\n' "$CONFIG_NAME" >&2
  printf 'Use CONFIG_NAME=realrobot_paper_fold_vanilla_vla, or use Scripts/train_realrobot_paper_fold_vanilla_tactile_vla.sh for tactile configs.\n' >&2
  exit 2
fi
export CONFIG_NAME
export RUN_NAME="${RUN_NAME:-realrobot_paper_fold_vanilla_vla_ac16_nohis_vis_notac}"

exec "$SCRIPT_DIR/train_realrobot_chips_vanilla_vla.sh" "$@"
