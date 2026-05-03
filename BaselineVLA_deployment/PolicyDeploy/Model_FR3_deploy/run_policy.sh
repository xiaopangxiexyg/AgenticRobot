#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../Scripts/load_local_env.sh
source "${SCRIPT_DIR}/../../Scripts/load_local_env.sh"
PROJECT_ROOT="${PROJECT_ROOT:-$MANIPREP_PROJECT_ROOT}"
DEPLOY_PACKAGE_ROOT="${DEPLOY_PACKAGE_ROOT:-${MANIPREP_DEPLOY_PACKAGE_ROOT:-}}"
CONDA_ENV="${CONDA_ENV:-ManipRep_310}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DEPLOY_MODE="${DEPLOY_MODE:-${FR3_DEPLOY_MODE:-model}}"
case "${DEPLOY_MODE,,}" in
  model|policy|vla)
    DEPLOY_MODE="model"
    ;;
  move|move_frame|frame|init|init_frame|hold)
    DEPLOY_MODE="move_frame"
    ;;
  replay|debug_replay|trajectory)
    DEPLOY_MODE="replay"
    ;;
  *)
    echo "DEPLOY_MODE must be one of: model, move_frame, replay; got '${DEPLOY_MODE}'" >&2
    exit 2
    ;;
esac

case "$DEPLOY_MODE" in
  model)
    DEFAULT_CAMERA_MODE="single"
    DEFAULT_POLICY_HZ="5"
    DEFAULT_MIN_ACTION_QUEUE="8"
    DEFAULT_REQUIRE_IMAGE="1"
    DEFAULT_REQUIRE_CAMERA_FRAME="1"
    DEFAULT_INIT_SOURCE="auto"
    DEFAULT_APPEND_ACTION_CHUNKS="0"
    ;;
  move_frame)
    DEFAULT_CAMERA_MODE="none"
    DEFAULT_POLICY_HZ="10"
    DEFAULT_MIN_ACTION_QUEUE="0"
    DEFAULT_REQUIRE_IMAGE="0"
    DEFAULT_REQUIRE_CAMERA_FRAME="0"
    DEFAULT_INIT_SOURCE="dataset"
    DEFAULT_APPEND_ACTION_CHUNKS="0"
    ;;
  replay)
    DEFAULT_CAMERA_MODE="none"
    DEFAULT_POLICY_HZ="10"
    DEFAULT_MIN_ACTION_QUEUE="8"
    DEFAULT_REQUIRE_IMAGE="0"
    DEFAULT_REQUIRE_CAMERA_FRAME="0"
    DEFAULT_INIT_SOURCE="replay"
    DEFAULT_APPEND_ACTION_CHUNKS="1"
    ;;
esac

CHECKPOINT="${CHECKPOINT:-${CKPT:-${MODEL_CKPT:-}}}"
DEVICE="${DEVICE:-cuda}"
ACTIVE_SIDES="${ACTIVE_SIDES:-right}"
LEFT_HAND_IP="${LEFT_HAND_IP:-}"
RIGHT_HAND_IP="${RIGHT_HAND_IP:-}"
ENABLE_ARM="${ENABLE_ARM:-1}"
ARM_SIDE="${ARM_SIDE:-right}"
FR3_HOST="${FR3_HOST:-192.168.20.172}"
FR3_PORT="${FR3_PORT:-5558}"
CAMERA_KEY="${CAMERA_KEY:-front}"
CAMERA_MODE="${CAMERA_MODE:-$DEFAULT_CAMERA_MODE}"
CAMERA_CONFIG="${CAMERA_CONFIG:-}"
CAMERA_SERIAL="${CAMERA_SERIAL:-}"
HZ="${HZ:-30}"
POLICY_HZ="${POLICY_HZ:-$DEFAULT_POLICY_HZ}"
MIN_ACTION_QUEUE="${MIN_ACTION_QUEUE:-$DEFAULT_MIN_ACTION_QUEUE}"
MAX_STEPS="${MAX_STEPS:-0}"
MAX_ARM_STEP_RAD="${MAX_ARM_STEP_RAD:-0.02}"
MAX_HAND_STEP="${MAX_HAND_STEP:-0.03}"
MAX_FINGERTIP_FORCE="${MAX_FINGERTIP_FORCE:-${INSPIRE_HAND_MAX_FORCE:-${MAX_FORCE:-1000}}}"
if [[ -z "${HAND_TACTILE_REGIONS+x}" ]]; then
  HAND_TACTILE_REGIONS="${HAND_TACTILE_REGIONS_DEFAULT-all}"
fi
TACTILE_SIDE="${TACTILE_SIDE:-right}"
REQUIRE_IMAGE="${REQUIRE_IMAGE:-$DEFAULT_REQUIRE_IMAGE}"
REQUIRE_CAMERA_FRAME="${REQUIRE_CAMERA_FRAME:-$DEFAULT_REQUIRE_CAMERA_FRAME}"
REQUIRE_READY_OBSERVATION="${REQUIRE_READY_OBSERVATION:-1}"
REQUIRE_TACTILE="${REQUIRE_TACTILE:-${REQUIRE_TACTILE_DEFAULT:-1}}"
STRICT_STATE_DIM="${STRICT_STATE_DIM:-1}"
ACTION_CLIP_NAN_TO_HOLD="${ACTION_CLIP_NAN_TO_HOLD:-1}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-all}"
APPEND_ACTION_CHUNKS="${APPEND_ACTION_CHUNKS:-$DEFAULT_APPEND_ACTION_CHUNKS}"
ACTION_QUEUE_POLICY="${ACTION_QUEUE_POLICY:-}"
SEND_ACTIONS="${SEND_ACTIONS:-0}"
AUTO_START="${AUTO_START:-0}"
INIT_TIME_S="${INIT_TIME_S:-5.0}"
INIT_SOURCE="${INIT_SOURCE:-$DEFAULT_INIT_SOURCE}"
INIT_DATASET="${INIT_DATASET:-}"
INIT_PARQUET="${INIT_PARQUET:-}"
INIT_EPISODE="${INIT_EPISODE:-0}"
INIT_FRAME="${INIT_FRAME:-0}"
INIT_ACTION_COLUMN="${INIT_ACTION_COLUMN:-action}"
ALLOW_ARM_CONNECT_FAILURE="${ALLOW_ARM_CONNECT_FAILURE:-0}"
ALLOW_HAND_CONNECT_FAILURE="${ALLOW_HAND_CONNECT_FAILURE:-0}"
VIS_CAM="${VIS_CAM:-0}"
VISUALIZE_TACTILE="${VISUALIZE_TACTILE:-0}"

DEBUG_REPLAY_DATASET="${DEBUG_REPLAY_DATASET:-${REPLAY_DATASET:-}}"
DEBUG_REPLAY_PARQUET="${DEBUG_REPLAY_PARQUET:-${REPLAY_PARQUET:-}}"
DEBUG_REPLAY_EPISODE="${DEBUG_REPLAY_EPISODE:-${REPLAY_EPISODE:-$INIT_EPISODE}}"
DEBUG_REPLAY_ACTION_COLUMN="${DEBUG_REPLAY_ACTION_COLUMN:-${REPLAY_ACTION_COLUMN:-$INIT_ACTION_COLUMN}}"
DEBUG_REPLAY_START_INDEX="${DEBUG_REPLAY_START_INDEX:-${REPLAY_START_INDEX:-0}}"
DEBUG_REPLAY_END_INDEX="${DEBUG_REPLAY_END_INDEX:-${REPLAY_END_INDEX:-}}"
DEBUG_REPLAY_CHUNK_SIZE="${DEBUG_REPLAY_CHUNK_SIZE:-${REPLAY_CHUNK_SIZE:-32}}"
DEBUG_REPLAY_LOOP="${DEBUG_REPLAY_LOOP:-${REPLAY_LOOP:-0}}"

if [[ -z "$DEPLOY_PACKAGE_ROOT" ]]; then
  echo "Set DEPLOY_PACKAGE_ROOT or MANIPREP_DEPLOY_PACKAGE_ROOT in local_paths.env" >&2
  exit 2
fi
if [[ "$DEPLOY_MODE" == "model" && -z "$CHECKPOINT" ]]; then
  echo "Set CHECKPOINT=/path/to/best.pt or CHECKPOINT=/path/to/checkpoints/epoch_XXXX.pt" >&2
  exit 2
fi
if [[ "$DEPLOY_MODE" == "move_frame" && -z "$INIT_DATASET" && -z "$INIT_PARQUET" ]]; then
  echo "DEPLOY_MODE=move_frame requires INIT_DATASET=/path/to/dataset or INIT_PARQUET=/path/to/episode.parquet" >&2
  exit 2
fi
if [[ "$DEPLOY_MODE" == "replay" && -z "$DEBUG_REPLAY_DATASET" && -z "$DEBUG_REPLAY_PARQUET" ]]; then
  DEBUG_REPLAY_DATASET="$INIT_DATASET"
  DEBUG_REPLAY_PARQUET="$INIT_PARQUET"
fi
if [[ "$DEPLOY_MODE" == "replay" && -z "$DEBUG_REPLAY_DATASET" && -z "$DEBUG_REPLAY_PARQUET" ]]; then
  echo "DEPLOY_MODE=replay requires DEBUG_REPLAY_DATASET/REPLAY_DATASET or DEBUG_REPLAY_PARQUET/REPLAY_PARQUET" >&2
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
if [[ -n "$ACTION_QUEUE_POLICY" ]]; then
  extra_args+=(--action-queue-policy "$ACTION_QUEUE_POLICY")
fi
if is_true "$REQUIRE_CAMERA_FRAME"; then
  extra_args+=(--require-camera-frame)
fi
if is_true "$REQUIRE_READY_OBSERVATION"; then
  extra_args+=(--require-ready-observation)
fi
if is_true "$ALLOW_ARM_CONNECT_FAILURE"; then
  extra_args+=(--allow-arm-connect-failure)
fi
if is_true "$ALLOW_HAND_CONNECT_FAILURE"; then
  extra_args+=(--allow-hand-connect-failure)
fi
if is_true "$ENABLE_ARM"; then
  extra_args+=(--enable-arm --arm-side "$ARM_SIDE" --fr3-host "$FR3_HOST" --fr3-port "$FR3_PORT")
fi
if is_true "$VIS_CAM"; then
  extra_args+=(--vis-cam)
fi
if is_true "$VISUALIZE_TACTILE"; then
  extra_args+=(--visualize-tactile)
fi
if [[ -n "$CAMERA_CONFIG" ]]; then
  extra_args+=(--camera-config "$CAMERA_CONFIG")
fi
if [[ -n "$CAMERA_SERIAL" ]]; then
  extra_args+=(--camera-serial "$CAMERA_SERIAL")
fi
if [[ "$MAX_STEPS" != "0" ]]; then
  extra_args+=(--max-steps "$MAX_STEPS")
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

policy_args=()
case "$DEPLOY_MODE" in
  model)
    POLICY_PATH="${PROJECT_ROOT}/PolicyDeploy/Model_FR3_deploy/policy.py:Policy"
    export CHECKPOINT DEVICE CAMERA_KEY TACTILE_SIDE REQUIRE_IMAGE REQUIRE_TACTILE STRICT_STATE_DIM ACTION_CLIP_NAN_TO_HOLD ACTION_CHUNK_SIZE
    POLICY_KWARGS_JSON="$("$PYTHON_BIN" "${PROJECT_ROOT}/PolicyDeploy/Model_FR3_deploy/build_policy_kwargs.py" --require-tactile-default "${REQUIRE_TACTILE_DEFAULT:-1}")"
    policy_args+=(--policy "$POLICY_PATH" --policy-kwargs-json "$POLICY_KWARGS_JSON")
    ;;
  move_frame)
    policy_args+=(--policy hold)
    ;;
  replay)
    if [[ -n "$DEBUG_REPLAY_DATASET" ]]; then
      policy_args+=(--debug-replay-dataset "$DEBUG_REPLAY_DATASET")
    fi
    if [[ -n "$DEBUG_REPLAY_PARQUET" ]]; then
      policy_args+=(--debug-replay-parquet "$DEBUG_REPLAY_PARQUET")
    fi
    policy_args+=(
      --debug-replay-episode "$DEBUG_REPLAY_EPISODE"
      --debug-replay-action-column "$DEBUG_REPLAY_ACTION_COLUMN"
      --debug-replay-start-index "$DEBUG_REPLAY_START_INDEX"
      --debug-replay-chunk-size "$DEBUG_REPLAY_CHUNK_SIZE"
    )
    if [[ -n "$DEBUG_REPLAY_END_INDEX" ]]; then
      policy_args+=(--debug-replay-end-index "$DEBUG_REPLAY_END_INDEX")
    fi
    if is_true "$DEBUG_REPLAY_LOOP"; then
      policy_args+=(--debug-replay-loop)
    fi
    ;;
esac

"$PYTHON_BIN" -m inspire_data_collect.deploy.run_policy \
  "${policy_args[@]}" \
  --active-sides "$ACTIVE_SIDES" \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
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
