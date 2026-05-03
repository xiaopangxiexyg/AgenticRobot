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
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
GENMED_DEVICE="${GENMED_DEVICE:-cpu}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PIPELINE="${RUN_PIPELINE:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

PC_MED_WEIGHT="${PC_MED_WEIGHT:-3e-5}"

VANILLA_CHECKPOINT="${VANILLA_CHECKPOINT:-Outputs/vanilla/rawhand_phase3_vla_fulltrain_vanilla_seed7}"
GTMED_WL_RUN="rawhand_wristlocal_vla_gtmed_pc_f1_8_pn_seed7"
GENMED_WL_RUN="rawhand_wristlocal_vla_genmed_pc_f1_8_pq_seed7"
FULLTRAIN_WL_RUN="rawhand_wristlocal_vla_fulltrain_pc_f1_8_pq_pn_seed7"

OUTPUT_ROOT="Outputs/comparisons/rawhand_wrist_local_pointcloud"
EVAL_ROOT="$OUTPUT_ROOT/evaluations"
PIPELINE_ROOT="$OUTPUT_ROOT/pipeline_wrist_pc_f1_8_pn"

COMMON_TRAIN_ARGS=(
  --config-name=train_runner
  data=desktasktac_rawhand_point_wrist_local_real
  train=real_debug
  model=basic_vla
  project.seed="$SEED"
  model.task.enabled=false
  train.epochs="$EPOCHS"
  train.device="$DEVICE"
  train.wandb.enabled=true
  train.wandb.mode=offline
  train.validation_ratio=1.0
  data.raw.root_dirs=[Data/RawData/Human/RawHandPointData]
  data.raw.filters.dataset_name=RawHandPointData
  data.raw.loader.batch_size="$BATCH_SIZE"
  data.raw.loader.num_workers=0
  data.raw.loader.prefetch_factor=null
  data.raw.loader.persistent_workers=false
  data.raw.loader.val.num_workers=0
  data.raw.loader.val.prefetch_factor=null
  data.raw.loader.val.persistent_workers=false
  ++data.raw.limits.max_datasets=1
  ++data.raw.limits.max_episodes_per_dataset=64
  ++data.raw.limits.max_frames_per_episode=16
  ++data.raw.limits.max_total_samples=1024
)

PC_WRIST_LOCAL_ARGS=(
  data.raw.sequence.med.kind=point_cloud
  data.raw.sequence.med.side=right
  data.raw.sequence.med.intermediate_key=future_right_point_cloud_wrist_local
  data.raw.sequence.med.future_start=1
  data.raw.sequence.med.future_end=8
  data.raw.sequence.med.stride=1
  data.raw.sequence.med.point_cloud.coordinate_frame=wrist_local
  data.raw.sequence.med.point_cloud.reference_position_field=state.eef_position
  data.raw.sequence.med.point_cloud.reference_rotation_field=state.eef_rotation
  data.med.flatten_sequence=auto
)

train_if_needed() {
  local experiment="$1"
  local run_name="$2"
  shift 2
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
  "$PYTHON_BIN" Training/train.py "${COMMON_TRAIN_ARGS[@]}" "${PC_WRIST_LOCAL_ARGS[@]}" "$@" train.run_name="$run_name"
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
  if [[ "$RUN_PIPELINE" != "1" ]]; then
    echo "skip pipeline: RUN_PIPELINE=${RUN_PIPELINE}"
    return
  fi

  local vanilla_args=()
  if [[ -f "${VANILLA_CHECKPOINT}/best.pt" || -f "$VANILLA_CHECKPOINT" ]]; then
    vanilla_args=(--vanilla-checkpoint "$VANILLA_CHECKPOINT" --vanilla-checkpoint-kind best)
  else
    echo "warning: vanilla checkpoint not found at ${VANILLA_CHECKPOINT}; pipeline will omit Vanilla comparison"
  fi

  echo "===== pipeline wrist-local pc_f1_8_pn ====="
  "$PYTHON_BIN" TestScripts/evaluate_gtmed_pipeline.py \
    "Outputs/gtmed/${GTMED_WL_RUN}" \
    --genmed-checkpoint "Outputs/genmed/${GENMED_WL_RUN}" \
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
    --output-dir "$PIPELINE_ROOT"
}

mkdir -p "$EVAL_ROOT"

train_if_needed gtmed "$GTMED_WL_RUN" \
  experiment=gtmed \
  model.med_branch.projector.type=pointnet \
  train.wandb.group=rawhand_wristlocal_vla_gtmed

train_if_needed genmed "$GENMED_WL_RUN" \
  experiment=genmed \
  model.med_branch.predictor.type=point_query \
  train.wandb.group=rawhand_wristlocal_vla_genmed

train_if_needed fulltrain "$FULLTRAIN_WL_RUN" \
  experiment=fulltrain \
  experiment.loss_weights.action=1.0 \
  experiment.loss_weights.med="$PC_MED_WEIGHT" \
  model.med_branch.predictor.type=point_query \
  model.med_branch.projector.type=pointnet \
  train.wandb.group=rawhand_wristlocal_vla_fulltrain

evaluate_checkpoint_if_needed "Outputs/genmed/${GENMED_WL_RUN}" "$EVAL_ROOT/genmed_wrist_pc_f1_8_pq"
evaluate_pipeline_if_needed
evaluate_checkpoint_if_needed "Outputs/fulltrain/${FULLTRAIN_WL_RUN}" "$EVAL_ROOT/fulltrain_wrist_pc_f1_8_pq_pn"

if [[ "$RUN_SUMMARY" == "1" ]]; then
  "$PYTHON_BIN" Scripts/summarize_rawhand_wrist_local_pointcloud_results.py --output-root "$OUTPUT_ROOT"
else
  echo "skip summary: RUN_SUMMARY=${RUN_SUMMARY}"
fi
