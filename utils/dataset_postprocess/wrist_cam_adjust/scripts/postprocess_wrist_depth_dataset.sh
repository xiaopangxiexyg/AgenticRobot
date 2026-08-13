#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${1:-datasets/2026-07-05_vive_rm65_real_teleop}"
EPISODE="${2:-0}"
AT_SEC="${AT_SEC:-5.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-utils/dataset_postprocess/wrist_cam_adjust/outputs/depth_mask_postprocess}"

CLOSE_RADIUS_PX="${CLOSE_RADIUS_PX:-5}"
OPEN_RADIUS_PX="${OPEN_RADIUS_PX:-0}"
MEDIAN_KSIZE="${MEDIAN_KSIZE:-5}"
FILL_HOLE_MAX_AREA_PX="${FILL_HOLE_MAX_AREA_PX:-4096}"
MIN_COMPONENT_AREA_PX="${MIN_COMPONENT_AREA_PX:-0}"
GAUSSIAN_BLUR_PX="${GAUSSIAN_BLUR_PX:-0}"
GAUSSIAN_THRESHOLD="${GAUSSIAN_THRESHOLD:-0.5}"
TEMPORAL_GAP_FILL_FRAMES="${TEMPORAL_GAP_FILL_FRAMES:-2}"
TEMPORAL_MAJORITY_WINDOW="${TEMPORAL_MAJORITY_WINDOW:-0}"
TEMPORAL_MAJORITY_THRESHOLD="${TEMPORAL_MAJORITY_THRESHOLD:-0.5}"

EP_TAG="$(printf 'episode_%06d' "${EPISODE}")"
DATASET_NAME="$(basename "${DATASET}")"
RUN_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/${EP_TAG}"
POST_PREVIEW="${RUN_DIR}/postprocessed_preview.mp4"
CONTACT="${RUN_DIR}/${EP_TAG}_rgb_depth_mask_postprocessed_contact_${AT_SEC}s.png"

LEFT_RGB="${DATASET}/videos/observation.images.left_wrist/chunk-000/${EP_TAG}.mp4"
RIGHT_RGB="${DATASET}/videos/observation.images.right_wrist/chunk-000/${EP_TAG}.mp4"

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${PYTHON_BIN}" -m utils.dataset_postprocess.wrist_cam_adjust postprocess-depth-mask \
  --session "${DATASET}" \
  --episode "${EPISODE}" \
  --output-root "${OUTPUT_ROOT}" \
  --close-radius-px "${CLOSE_RADIUS_PX}" \
  --open-radius-px "${OPEN_RADIUS_PX}" \
  --median-ksize "${MEDIAN_KSIZE}" \
  --fill-hole-max-area-px "${FILL_HOLE_MAX_AREA_PX}" \
  --min-component-area-px "${MIN_COMPONENT_AREA_PX}" \
  --gaussian-blur-px "${GAUSSIAN_BLUR_PX}" \
  --gaussian-threshold "${GAUSSIAN_THRESHOLD}" \
  --temporal-gap-fill-frames "${TEMPORAL_GAP_FILL_FRAMES}" \
  --temporal-majority-window "${TEMPORAL_MAJORITY_WINDOW}" \
  --temporal-majority-threshold "${TEMPORAL_MAJORITY_THRESHOLD}" \
  --overwrite

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "${PYTHON_BIN}" utils/portable_vis/multi_view/multi_view_video_player.py \
  "${LEFT_RGB}" \
  "${RIGHT_RGB}" \
  "${POST_PREVIEW}" \
  --sort none \
  --labels left_rgb,right_rgb,depth_mask_postprocessed \
  --layout 1x3 \
  --canvas-size 1920x720 \
  --contact-sheet "${CONTACT}" \
  --at "${AT_SEC}"

echo "postprocessed_preview=${POST_PREVIEW}"
echo "contact_sheet=${CONTACT}"
