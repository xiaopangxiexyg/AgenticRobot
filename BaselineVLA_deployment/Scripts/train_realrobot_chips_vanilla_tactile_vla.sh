#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=Scripts/load_local_env.sh
source "${SCRIPT_DIR}/load_local_env.sh"
PROJECT_ROOT="${PROJECT_ROOT:-$MANIPREP_PROJECT_ROOT}"
CONDA_ENV="${CONDA_ENV:-ManipRep_310}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_NAME="${CONFIG_NAME:-realrobot_vanilla_tactile}"
RUN_NAME="${RUN_NAME:-realrobot_chips_med_vtactile_vla_ac16_nohis_vis}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_MODE="${WANDB_MODE:-offline}"
IMAGE_AUGMENT="${IMAGE_AUGMENT:-false}"
IMAGE_AUGMENT_COLOR_JITTER_PROB="${IMAGE_AUGMENT_COLOR_JITTER_PROB:-0.8}"
IMAGE_AUGMENT_BRIGHTNESS="${IMAGE_AUGMENT_BRIGHTNESS:-0.35}"
IMAGE_AUGMENT_CONTRAST="${IMAGE_AUGMENT_CONTRAST:-0.20}"
IMAGE_AUGMENT_SATURATION="${IMAGE_AUGMENT_SATURATION:-0.15}"
IMAGE_AUGMENT_BRIGHTNESS_BOOST_PROB="${IMAGE_AUGMENT_BRIGHTNESS_BOOST_PROB:-0.0}"
IMAGE_AUGMENT_BRIGHTNESS_BOOST_MAX_DELTA="${IMAGE_AUGMENT_BRIGHTNESS_BOOST_MAX_DELTA:-0.0}"
IMAGE_AUGMENT_CROP_PROB="${IMAGE_AUGMENT_CROP_PROB:-0.45}"
IMAGE_AUGMENT_CROP_SCALE_MIN="${IMAGE_AUGMENT_CROP_SCALE_MIN:-0.85}"
IMAGE_AUGMENT_CROP_SCALE_MAX="${IMAGE_AUGMENT_CROP_SCALE_MAX:-1.0}"
IMAGE_AUGMENT_CROP_RATIO_MIN="${IMAGE_AUGMENT_CROP_RATIO_MIN:-0.90}"
IMAGE_AUGMENT_CROP_RATIO_MAX="${IMAGE_AUGMENT_CROP_RATIO_MAX:-1.10}"
IMAGE_AUGMENT_NOISE_PROB="${IMAGE_AUGMENT_NOISE_PROB:-0.20}"
IMAGE_AUGMENT_NOISE_STD="${IMAGE_AUGMENT_NOISE_STD:-0.008}"
IMAGE_AUGMENT_BLUR_PROB="${IMAGE_AUGMENT_BLUR_PROB:-0.05}"
IMAGE_AUGMENT_BLUR_KERNEL="${IMAGE_AUGMENT_BLUR_KERNEL:-3}"
IMAGE_AUGMENT_BLUR_SIGMA="${IMAGE_AUGMENT_BLUR_SIGMA:-0.5}"
IMAGE_AUGMENT_ERASE_PROB="${IMAGE_AUGMENT_ERASE_PROB:-0.10}"
IMAGE_AUGMENT_ERASE_SCALE_MIN="${IMAGE_AUGMENT_ERASE_SCALE_MIN:-0.01}"
IMAGE_AUGMENT_ERASE_SCALE_MAX="${IMAGE_AUGMENT_ERASE_SCALE_MAX:-0.06}"
IMAGE_AUGMENT_ERASE_RATIO_MIN="${IMAGE_AUGMENT_ERASE_RATIO_MIN:-0.30}"
IMAGE_AUGMENT_ERASE_RATIO_MAX="${IMAGE_AUGMENT_ERASE_RATIO_MAX:-3.30}"
IMAGE_AUGMENT_ERASE_CENTER_PROTECT_FRACTION="${IMAGE_AUGMENT_ERASE_CENTER_PROTECT_FRACTION:-0.50}"
IMAGE_AUGMENT_ERASE_VALUE="${IMAGE_AUGMENT_ERASE_VALUE:-0.0}"
SAVE_PERIODIC_CHECKPOINTS="${SAVE_PERIODIC_CHECKPOINTS:-true}"
CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-5}"
PERIODIC_CHECKPOINT_DIR="${PERIODIC_CHECKPOINT_DIR:-checkpoints}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RESUME_LOAD_OPTIMIZER="${RESUME_LOAD_OPTIMIZER:-true}"
RESUME_STRICT="${RESUME_STRICT:-true}"
RESUME_EPOCHS_ARE_ADDITIONAL="${RESUME_EPOCHS_ARE_ADDITIONAL:-false}"

maniprep_activate_conda "$CONDA_ENV"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/maniprep_matplotlib}"

resolve_auto_resume_checkpoint() {
  local run_dir="$1"
  local periodic_dir="$2"
  local checkpoint_dir
  if [[ "$periodic_dir" = /* ]]; then
    checkpoint_dir="$periodic_dir"
  else
    checkpoint_dir="$run_dir/$periodic_dir"
  fi

  if [[ -f "$run_dir/last.pt" ]]; then
    printf '%s\n' "$run_dir/last.pt"
    return 0
  fi

  local checkpoint_files=()
  if [[ -d "$checkpoint_dir" ]]; then
    shopt -s nullglob
    checkpoint_files=("$checkpoint_dir"/epoch_*.pt)
    shopt -u nullglob
  fi
  if (( ${#checkpoint_files[@]} > 0 )); then
    printf '%s\n' "${checkpoint_files[$((${#checkpoint_files[@]} - 1))]}"
    return 0
  fi

  printf 'No resume checkpoint found: expected %s or %s/epoch_*.pt\n' \
    "$run_dir/last.pt" "$checkpoint_dir" >&2
  return 1
}

cd "$PROJECT_ROOT"
resume_args=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  if [[ "$RESUME_FROM_CHECKPOINT" == "auto" ]]; then
    RESUME_FROM_CHECKPOINT="$(
      resolve_auto_resume_checkpoint "Outputs/vanilla-tactile/${RUN_NAME}" "$PERIODIC_CHECKPOINT_DIR"
    )"
  fi
  resume_args+=(train.resume_from_checkpoint="$RESUME_FROM_CHECKPOINT")
  resume_args+=(train.resume_load_optimizer="$RESUME_LOAD_OPTIMIZER")
  resume_args+=(train.resume_strict="$RESUME_STRICT")
  resume_args+=(train.resume_epochs_are_additional="$RESUME_EPOCHS_ARE_ADDITIONAL")
fi

"$PYTHON_BIN" Training/train.py \
  --config-name="$CONFIG_NAME" \
  train.run_name="$RUN_NAME" \
  data.raw.loader.batch_size="$BATCH_SIZE" \
  data.raw.loader.num_workers="$NUM_WORKERS" \
  data.raw.video.augmentation.enabled="$IMAGE_AUGMENT" \
  data.raw.video.augmentation.color_jitter_prob="$IMAGE_AUGMENT_COLOR_JITTER_PROB" \
  data.raw.video.augmentation.brightness="$IMAGE_AUGMENT_BRIGHTNESS" \
  data.raw.video.augmentation.contrast="$IMAGE_AUGMENT_CONTRAST" \
  data.raw.video.augmentation.saturation="$IMAGE_AUGMENT_SATURATION" \
  data.raw.video.augmentation.brightness_boost_prob="$IMAGE_AUGMENT_BRIGHTNESS_BOOST_PROB" \
  data.raw.video.augmentation.brightness_boost_max_delta="$IMAGE_AUGMENT_BRIGHTNESS_BOOST_MAX_DELTA" \
  data.raw.video.augmentation.random_resized_crop_prob="$IMAGE_AUGMENT_CROP_PROB" \
  data.raw.video.augmentation.crop_scale_min="$IMAGE_AUGMENT_CROP_SCALE_MIN" \
  data.raw.video.augmentation.crop_scale_max="$IMAGE_AUGMENT_CROP_SCALE_MAX" \
  data.raw.video.augmentation.crop_ratio_min="$IMAGE_AUGMENT_CROP_RATIO_MIN" \
  data.raw.video.augmentation.crop_ratio_max="$IMAGE_AUGMENT_CROP_RATIO_MAX" \
  data.raw.video.augmentation.gaussian_noise_prob="$IMAGE_AUGMENT_NOISE_PROB" \
  data.raw.video.augmentation.gaussian_noise_std="$IMAGE_AUGMENT_NOISE_STD" \
  data.raw.video.augmentation.gaussian_blur_prob="$IMAGE_AUGMENT_BLUR_PROB" \
  data.raw.video.augmentation.gaussian_blur_kernel="$IMAGE_AUGMENT_BLUR_KERNEL" \
  data.raw.video.augmentation.gaussian_blur_sigma="$IMAGE_AUGMENT_BLUR_SIGMA" \
  data.raw.video.augmentation.random_erasing_prob="$IMAGE_AUGMENT_ERASE_PROB" \
  data.raw.video.augmentation.random_erasing_scale_min="$IMAGE_AUGMENT_ERASE_SCALE_MIN" \
  data.raw.video.augmentation.random_erasing_scale_max="$IMAGE_AUGMENT_ERASE_SCALE_MAX" \
  data.raw.video.augmentation.random_erasing_ratio_min="$IMAGE_AUGMENT_ERASE_RATIO_MIN" \
  data.raw.video.augmentation.random_erasing_ratio_max="$IMAGE_AUGMENT_ERASE_RATIO_MAX" \
  data.raw.video.augmentation.random_erasing_center_protect_fraction="$IMAGE_AUGMENT_ERASE_CENTER_PROTECT_FRACTION" \
  data.raw.video.augmentation.random_erasing_value="$IMAGE_AUGMENT_ERASE_VALUE" \
  train.save_periodic_checkpoints="$SAVE_PERIODIC_CHECKPOINTS" \
  train.checkpoint_every_n_epochs="$CHECKPOINT_EVERY_N_EPOCHS" \
  train.periodic_checkpoint_dir="$PERIODIC_CHECKPOINT_DIR" \
  train.wandb.enabled="$WANDB_ENABLED" \
  train.wandb.mode="$WANDB_MODE" \
  "${resume_args[@]}" \
  "$@"
