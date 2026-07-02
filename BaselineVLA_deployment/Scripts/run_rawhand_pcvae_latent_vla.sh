#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=Scripts/load_local_env.sh
source "${SCRIPT_DIR}/load_local_env.sh"
PROJECT_ROOT="${PROJECT_ROOT:-$MANIPREP_PROJECT_ROOT}"
CONDA_ENV="${CONDA_ENV:-ManipRep_310}"
PYTHON_BIN="${PYTHON_BIN:-python}"
maniprep_activate_conda "$CONDA_ENV"

cd "$PROJECT_ROOT"

SEED="${SEED:-7}"
VAE_EPOCHS="${VAE_EPOCHS:-2}"
EPOCHS="${EPOCHS:-2}"
LATENT_DIM="${LATENT_DIM:-32}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
VAE_DEVICE="${VAE_DEVICE:-$DEVICE}"
GENMED_DEVICE="${GENMED_DEVICE:-cpu}"
MED_DEVICE="${MED_DEVICE:-cpu}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
RUN_VAE="${RUN_VAE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PIPELINE="${RUN_PIPELINE:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

# Latent MSE is usually much smaller than raw point-cloud MSE; keep it
# supervised without letting it dominate action checkpoint selection.
FULLTRAIN_LATENT_MED_WEIGHT="${FULLTRAIN_LATENT_MED_WEIGHT:-1e-3}"

VANILLA_CHECKPOINT="${VANILLA_CHECKPOINT:-Outputs/vanilla/rawhand_phase3_vla_fulltrain_vanilla_seed7}"

WL_PCVAE_RUN="rawhand_pcvae_wristlocal_f1_8_latent32_seed7"
WL_GTMED_RUN="rawhand_pcvae_latent_wristlocal_vla_gtmed_f1_8_seed7"
WL_GENMED_RUN="rawhand_pcvae_latent_wristlocal_vla_genmed_f1_8_seed7"
WL_FULLTRAIN_RUN="rawhand_pcvae_latent_wristlocal_vla_fulltrain_f1_8_seed7"

CAM_PCVAE_RUN="rawhand_pcvae_camera_f1_8_latent32_seed7"
CAM_GTMED_RUN="rawhand_pcvae_latent_camera_vla_gtmed_f1_8_seed7"
CAM_GENMED_RUN="rawhand_pcvae_latent_camera_vla_genmed_f1_8_seed7"
CAM_FULLTRAIN_RUN="rawhand_pcvae_latent_camera_vla_fulltrain_f1_8_seed7"

OUTPUT_ROOT="Outputs/comparisons/rawhand_pcvae_latent_vla"
EVAL_ROOT="$OUTPUT_ROOT/evaluations"

COMMON_DATA_ARGS=(
  data.raw.root_dirs=[Data/RawData/Human/RawHandPointData]
  data.raw.filters.dataset_name=RawHandPointData
)

COMMON_LIMIT_ARGS=(
  ++data.raw.limits.max_datasets=1
  ++data.raw.limits.max_episodes_per_dataset=64
  ++data.raw.limits.max_frames_per_episode=16
  ++data.raw.limits.max_total_samples=1024
)

VAE_LIMIT_ARGS=(
  ++data.raw.limits.max_datasets=1
  ++data.raw.limits.max_episodes_per_dataset=128
  ++data.raw.limits.max_frames_per_episode=32
  ++data.raw.limits.max_total_samples=8192
)

F1_8_ARGS=(
  data.raw.sequence.med.future_start=1
  data.raw.sequence.med.future_end=8
  data.raw.sequence.med.stride=1
)

run_pcvae_if_needed() {
  local data_preset="$1"
  local run_name="$2"
  local run_dir="Outputs/pcvae/${run_name}"
  if [[ "$RUN_VAE" != "1" ]]; then
    echo "skip pcvae ${run_name}: RUN_VAE=${RUN_VAE}"
    return
  fi
  if [[ -f "${run_dir}/best.pt" && "$FORCE_TRAIN" != "1" ]]; then
    echo "skip pcvae ${run_name}: ${run_dir}/best.pt exists; set FORCE_TRAIN=1 to overwrite"
    return
  fi

  echo "===== train pcvae ${run_name} ====="
  "$PYTHON_BIN" Training/train.py \
    --config-name=pcvae \
    data="$data_preset" \
    model=tactile_point_vae \
    train=real_debug \
    project.seed="$SEED" \
    model.latent_dim="$LATENT_DIM" \
    train.epochs="$VAE_EPOCHS" \
    train.device="$VAE_DEVICE" \
    train.wandb.enabled=true \
    train.wandb.mode=offline \
    train.validation_ratio=1.0 \
    data.raw.loader.batch_size="$VAE_BATCH_SIZE" \
    data.raw.loader.num_workers=0 \
    data.raw.loader.prefetch_factor=null \
    data.raw.loader.persistent_workers=false \
    data.raw.loader.val.num_workers=0 \
    data.raw.loader.val.prefetch_factor=null \
    data.raw.loader.val.persistent_workers=false \
    "${COMMON_DATA_ARGS[@]}" \
    "${VAE_LIMIT_ARGS[@]}" \
    "${F1_8_ARGS[@]}" \
    train.run_name="$run_name"
}

train_latent_if_needed() {
  local experiment="$1"
  local data_preset="$2"
  local pcvae_run="$3"
  local run_name="$4"
  shift 4
  local run_dir="Outputs/${experiment}/${run_name}"
  if [[ "$RUN_TRAIN" != "1" ]]; then
    echo "skip train ${run_name}: RUN_TRAIN=${RUN_TRAIN}"
    return
  fi
  if [[ -f "${run_dir}/best.pt" && "$FORCE_TRAIN" != "1" ]]; then
    echo "skip train ${run_name}: ${run_dir}/best.pt exists; set FORCE_TRAIN=1 to overwrite"
    return
  fi

  echo "===== train ${run_name} ====="
  "$PYTHON_BIN" Training/train.py \
    --config-name=train_runner \
    data="$data_preset" \
    train=real_debug \
    experiment="$experiment" \
    model=basic_vla \
    project.seed="$SEED" \
    model.task.enabled=false \
    train.epochs="$EPOCHS" \
    train.device="$DEVICE" \
    train.wandb.enabled=true \
    train.wandb.mode=offline \
    train.validation_ratio=1.0 \
    data.med.checkpoint="Outputs/pcvae/${pcvae_run}" \
    data.med.checkpoint_kind=best \
    data.med.latent_dim="$LATENT_DIM" \
    data.med.device="$MED_DEVICE" \
    data.raw.loader.batch_size="$BATCH_SIZE" \
    data.raw.loader.num_workers=0 \
    data.raw.loader.prefetch_factor=null \
    data.raw.loader.persistent_workers=false \
    data.raw.loader.val.num_workers=0 \
    data.raw.loader.val.prefetch_factor=null \
    data.raw.loader.val.persistent_workers=false \
    "${COMMON_DATA_ARGS[@]}" \
    "${COMMON_LIMIT_ARGS[@]}" \
    "${F1_8_ARGS[@]}" \
    "$@" \
    train.run_name="$run_name"
}

evaluate_checkpoint_if_needed() {
  local checkpoint_ref="$1"
  local output_dir="$2"
  if [[ "$RUN_EVAL" != "1" ]]; then
    echo "skip eval ${checkpoint_ref}: RUN_EVAL=${RUN_EVAL}"
    return
  fi
  echo "===== eval ${checkpoint_ref} ====="
  "$PYTHON_BIN" TestScripts/evaluate_checkpoint.py \
    "$checkpoint_ref" \
    --checkpoint-kind best \
    --split val \
    --loader-mode safe \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers 0 \
    --save-samples 0 \
    --visualize none \
    --device "$DEVICE" \
    --output-dir "$output_dir"
}

evaluate_pipeline_if_needed() {
  local tag="$1"
  local gtmed_run="$2"
  local genmed_run="$3"
  local output_dir="$4"
  if [[ "$RUN_PIPELINE" != "1" ]]; then
    echo "skip pipeline ${tag}: RUN_PIPELINE=${RUN_PIPELINE}"
    return
  fi

  local vanilla_args=()
  if [[ -f "${VANILLA_CHECKPOINT}/best.pt" || -f "$VANILLA_CHECKPOINT" ]]; then
    vanilla_args=(--vanilla-checkpoint "$VANILLA_CHECKPOINT" --vanilla-checkpoint-kind best)
  else
    echo "warning: vanilla checkpoint not found at ${VANILLA_CHECKPOINT}; pipeline will omit Vanilla comparison"
  fi

  echo "===== pipeline ${tag} ====="
  "$PYTHON_BIN" TestScripts/evaluate_gtmed_pipeline.py \
    "Outputs/gtmed/${gtmed_run}" \
    --genmed-checkpoint "Outputs/genmed/${genmed_run}" \
    "${vanilla_args[@]}" \
    --gtmed-checkpoint-kind best \
    --genmed-checkpoint-kind best \
    --split val \
    --pipeline-mode cached \
    --loader-mode safe \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers 0 \
    --save-samples 0 \
    --visualize none \
    --device "$DEVICE" \
    --genmed-device "$GENMED_DEVICE" \
    --output-dir "$output_dir"
}

run_latent_case() {
  local tag="$1"
  local source_data="$2"
  local latent_data="$3"
  local pcvae_run="$4"
  local gtmed_run="$5"
  local genmed_run="$6"
  local fulltrain_run="$7"

  run_pcvae_if_needed "$source_data" "$pcvae_run"

  train_latent_if_needed gtmed "$latent_data" "$pcvae_run" "$gtmed_run" \
    model.med_branch.projector.type=mlp \
    train.wandb.group=rawhand_pcvae_latent_gtmed

  train_latent_if_needed genmed "$latent_data" "$pcvae_run" "$genmed_run" \
    model.med_branch.predictor.type=mlp \
    train.wandb.group=rawhand_pcvae_latent_genmed

  train_latent_if_needed fulltrain "$latent_data" "$pcvae_run" "$fulltrain_run" \
    experiment.loss_weights.action=1.0 \
    experiment.loss_weights.med="$FULLTRAIN_LATENT_MED_WEIGHT" \
    model.med_branch.predictor.type=mlp \
    model.med_branch.projector.type=mlp \
    train.wandb.group=rawhand_pcvae_latent_fulltrain

  evaluate_checkpoint_if_needed "Outputs/genmed/${genmed_run}" "$EVAL_ROOT/genmed_${tag}_pcvae_latent_f1_8"
  evaluate_pipeline_if_needed "${tag}_pcvae_latent_f1_8" "$gtmed_run" "$genmed_run" "$OUTPUT_ROOT/pipeline_${tag}_pcvae_latent_f1_8"
  evaluate_checkpoint_if_needed "Outputs/fulltrain/${fulltrain_run}" "$EVAL_ROOT/fulltrain_${tag}_pcvae_latent_f1_8"
}

mkdir -p "$EVAL_ROOT"

run_latent_case \
  wristlocal \
  desktasktac_rawhand_point_vae_wrist_local \
  desktasktac_rawhand_point_vae_latent_wrist_local \
  "$WL_PCVAE_RUN" \
  "$WL_GTMED_RUN" \
  "$WL_GENMED_RUN" \
  "$WL_FULLTRAIN_RUN"

run_latent_case \
  camera \
  desktasktac_rawhand_point_vae \
  desktasktac_rawhand_point_vae_latent \
  "$CAM_PCVAE_RUN" \
  "$CAM_GTMED_RUN" \
  "$CAM_GENMED_RUN" \
  "$CAM_FULLTRAIN_RUN"

if [[ "$RUN_SUMMARY" == "1" ]]; then
  "$PYTHON_BIN" Scripts/summarize_rawhand_pcvae_latent_results.py --output-root "$OUTPUT_ROOT"
else
  echo "skip summary: RUN_SUMMARY=${RUN_SUMMARY}"
fi
