#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../Scripts/load_local_env.sh
source "${SCRIPT_DIR}/../../Scripts/load_local_env.sh"
PROJECT_ROOT="${PROJECT_ROOT:-$MANIPREP_PROJECT_ROOT}"
DEPLOY_PACKAGE_ROOT="${DEPLOY_PACKAGE_ROOT:-${MANIPREP_DEPLOY_PACKAGE_ROOT:-}}"
CONDA_ENV="${CONDA_ENV:-ManipRep_310}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINT="${CHECKPOINT:-${CKPT:-${MODEL_CKPT:-}}}"
DEVICE="${DEVICE:-cuda}"
RIGHT_HAND_IP="${RIGHT_HAND_IP:-}"
FR3_HOST="${FR3_HOST:-192.168.20.172}"
CAMERA_KEY="${CAMERA_KEY:-front}"
CAMERA_MODE="${CAMERA_MODE:-single}"
HZ="${HZ:-30}"
POLICY_HZ="${POLICY_HZ:-5}"
MIN_ACTION_QUEUE="${MIN_ACTION_QUEUE:-8}"
MAX_ARM_STEP_RAD="${MAX_ARM_STEP_RAD:-0.02}"
MAX_HAND_STEP="${MAX_HAND_STEP:-0.03}"
MAX_FINGERTIP_FORCE="${MAX_FINGERTIP_FORCE:-${INSPIRE_HAND_MAX_FORCE:-${MAX_FORCE:-1000}}}"
HAND_TACTILE_REGIONS="${HAND_TACTILE_REGIONS:-}"
TACTILE_SIDE="${TACTILE_SIDE:-right}"
REQUIRE_IMAGE="${REQUIRE_IMAGE:-1}"
REQUIRE_CAMERA_FRAME="${REQUIRE_CAMERA_FRAME:-1}"
REQUIRE_READY_OBSERVATION="${REQUIRE_READY_OBSERVATION:-1}"
REQUIRE_TACTILE="${REQUIRE_TACTILE:-0}"
STRICT_STATE_DIM="${STRICT_STATE_DIM:-1}"
ACTION_CLIP_NAN_TO_HOLD="${ACTION_CLIP_NAN_TO_HOLD:-1}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-all}"
APPEND_ACTION_CHUNKS="${APPEND_ACTION_CHUNKS:-0}"
SEND_ACTIONS="${SEND_ACTIONS:-0}"
AUTO_START="${AUTO_START:-0}"
INIT_TIME_S="${INIT_TIME_S:-5.0}"
INIT_SOURCE="${INIT_SOURCE:-auto}"
INIT_DATASET="${INIT_DATASET:-}"
INIT_PARQUET="${INIT_PARQUET:-}"
INIT_EPISODE="${INIT_EPISODE:-0}"
INIT_FRAME="${INIT_FRAME:-0}"
INIT_ACTION_COLUMN="${INIT_ACTION_COLUMN:-action}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "Set CHECKPOINT=/path/to/best.pt or CHECKPOINT=/path/to/checkpoints/epoch_XXXX.pt" >&2
  exit 2
fi
if [[ -z "$DEPLOY_PACKAGE_ROOT" ]]; then
  echo "Set DEPLOY_PACKAGE_ROOT or MANIPREP_DEPLOY_PACKAGE_ROOT in local_paths.env" >&2
  exit 2
fi

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

maniprep_activate_conda "$CONDA_ENV"

export PYTHONPATH="${DEPLOY_PACKAGE_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
POLICY_PATH="${PROJECT_ROOT}/PolicyDeploy/Model_FR3_deploy/policy.py:Policy"
export CHECKPOINT DEVICE CAMERA_KEY TACTILE_SIDE REQUIRE_IMAGE REQUIRE_TACTILE STRICT_STATE_DIM ACTION_CLIP_NAN_TO_HOLD ACTION_CHUNK_SIZE
POLICY_KWARGS_JSON="$("$PYTHON_BIN" "${PROJECT_ROOT}/PolicyDeploy/Model_FR3_deploy/build_policy_kwargs.py" --require-tactile-default 0)"

extra_args=()
if is_true "$SEND_ACTIONS"; then
  extra_args+=(--send-actions)
fi
if is_true "$AUTO_START"; then
  extra_args+=(--auto-start)
fi
if is_true "$APPEND_ACTION_CHUNKS"; then
  extra_args+=(--append-action-chunks)
fi
if is_true "$REQUIRE_CAMERA_FRAME"; then
  extra_args+=(--require-camera-frame)
fi
if is_true "$REQUIRE_READY_OBSERVATION"; then
  extra_args+=(--require-ready-observation)
fi
extra_args+=(
  --init-time-s "$INIT_TIME_S"
  --init-source "$INIT_SOURCE"
  --init-episode "$INIT_EPISODE"
  --init-frame "$INIT_FRAME"
  --init-action-column "$INIT_ACTION_COLUMN"
)
if [[ -n "$INIT_DATASET" ]]; then
  extra_args+=(--init-dataset "$INIT_DATASET")
fi
if [[ -n "$INIT_PARQUET" ]]; then
  extra_args+=(--init-parquet "$INIT_PARQUET")
fi

"$PYTHON_BIN" -m inspire_data_collect.deploy.run_policy \
  --policy "$POLICY_PATH" \
  --policy-kwargs-json "$POLICY_KWARGS_JSON" \
  --active-sides right \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --enable-arm \
  --fr3-host "$FR3_HOST" \
  --camera-mode "$CAMERA_MODE" \
  --camera-key "$CAMERA_KEY" \
  --hand-tactile-regions "$HAND_TACTILE_REGIONS" \
  --inspire-hand-max-force "$MAX_FINGERTIP_FORCE" \
  --hz "$HZ" \
  --policy-hz "$POLICY_HZ" \
  --min-action-queue "$MIN_ACTION_QUEUE" \
  --max-arm-step-rad "$MAX_ARM_STEP_RAD" \
  --max-hand-step "$MAX_HAND_STEP" \
  "${extra_args[@]}" \
  "$@"
