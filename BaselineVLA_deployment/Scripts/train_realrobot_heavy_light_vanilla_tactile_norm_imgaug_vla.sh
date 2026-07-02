#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONDA_ENV="${CONDA_ENV:-ManipRep_310}"
CONFIG_NAME="${CONFIG_NAME:-realrobot_heavy_light_vanilla_tactile_norm}"
if [[ "$CONFIG_NAME" != "realrobot_heavy_light_vanilla_tactile_norm" ]]; then
  printf 'Invalid CONFIG_NAME for heavy_light tactile-norm image-augment script: %s\n' "$CONFIG_NAME" >&2
  printf 'Use CONFIG_NAME=realrobot_heavy_light_vanilla_tactile_norm.\n' >&2
  exit 2
fi
export CONFIG_NAME
export IMAGE_AUGMENT="${IMAGE_AUGMENT:-true}"
export RUN_NAME="${RUN_NAME:-realrobot_heavy_light_right_vtactile_vla_ac16_nohis_vis_imgaug_tacnorm}"

exec "$SCRIPT_DIR/train_realrobot_heavy_light_vanilla_tactile_vla.sh" "$@"
