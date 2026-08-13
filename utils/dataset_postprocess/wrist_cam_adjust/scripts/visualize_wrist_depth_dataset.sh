#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${1:-datasets/2026-07-05_vive_rm65_real_teleop}"
EPISODE="${2:-0}"
AT_SEC="${AT_SEC:-5.0}"
MASK_SCALE="${MASK_SCALE:-1.0}"
OPEN_GUI="${OPEN_GUI:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-utils/dataset_postprocess/wrist_cam_adjust/outputs}"

EP_TAG="$(printf 'episode_%06d' "${EPISODE}")"
DATASET_NAME="$(basename "${DATASET}")"
RUN_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/${EP_TAG}"
mkdir -p "${RUN_DIR}"

PREVIEW="${RUN_DIR}/${EP_TAG}_wrist_depth_mask_preview.mp4"
CONTACT="${RUN_DIR}/${EP_TAG}_rgb_depth_mask_contact_${AT_SEC}s.png"

LEFT_RGB="${DATASET}/videos/observation.images.left_wrist/chunk-000/${EP_TAG}.mp4"
RIGHT_RGB="${DATASET}/videos/observation.images.right_wrist/chunk-000/${EP_TAG}.mp4"

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${PYTHON_BIN}" -m utils.dataset_postprocess.wrist_cam_adjust visualize-depth-mask \
  --session "${DATASET}" \
  --episode "${EPISODE}" \
  --output "${PREVIEW}" \
  --fps 15 \
  --scale "${MASK_SCALE}" \
  --overwrite

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${PYTHON_BIN}" utils/portable_vis/multi_view/multi_view_video_player.py \
  "${LEFT_RGB}" \
  "${RIGHT_RGB}" \
  "${PREVIEW}" \
  --sort none \
  --labels left_rgb,right_rgb,depth_mask \
  --layout 1x3 \
  --canvas-size 1920x720 \
  --contact-sheet "${CONTACT}" \
  --at "${AT_SEC}"

echo "preview=${PREVIEW}"
echo "contact_sheet=${CONTACT}"

if [[ "${OPEN_GUI}" == "1" ]]; then
  exec env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${PYTHON_BIN}" utils/portable_vis/multi_view/multi_view_video_player.py \
    "${LEFT_RGB}" \
    "${RIGHT_RGB}" \
    "${PREVIEW}" \
    --sort none \
    --labels left_rgb,right_rgb,depth_mask \
    --layout 1x3 \
    --canvas-size 1920x720 \
    --start-paused
fi

echo "Open the GUI with:"
echo "  OPEN_GUI=1 ${0} ${DATASET} ${EPISODE}"
