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
FORCE_TRAIN="${FORCE_TRAIN:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

# FullTrain optimizes action and med together. Raw med losses are much larger
# than action losses, so these defaults keep the weighted med term on the same
# order as action loss while still training med explicitly.
VEC_MED_WEIGHT="${VEC_MED_WEIGHT:-1e-5}"
PC_MED_WEIGHT="${PC_MED_WEIGHT:-3e-5}"

VANILLA_RUN="rawhand_phase3_vla_fulltrain_vanilla_seed7"
VEC_RUN="rawhand_phase3_vla_fulltrain_vec_f1_8_seed7"
PC_RUN="rawhand_phase3_vla_fulltrain_pc_f1_8_pq_pn_seed7"

OUTPUT_ROOT="Outputs/comparisons/rawhand_fulltrain_vs_vanilla"
EVAL_ROOT="$OUTPUT_ROOT/evaluations"

COMMON_TRAIN_ARGS=(
  --config-name=train_runner
  data=desktasktac_rawhand_point_real
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
  "$PYTHON_BIN" Training/train.py "${COMMON_TRAIN_ARGS[@]}" "$@" train.run_name="$run_name"
}

evaluate_if_needed() {
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

mkdir -p "$EVAL_ROOT"

train_if_needed vanilla "$VANILLA_RUN" \
  experiment=vanilla \
  train.wandb.group=rawhand_phase3_vla_fulltrain_baseline

train_if_needed fulltrain "$VEC_RUN" \
  experiment=fulltrain \
  experiment.loss_weights.action=1.0 \
  experiment.loss_weights.med="$VEC_MED_WEIGHT" \
  data.raw.sequence.med.kind=vector \
  data.raw.sequence.med.side=right \
  data.raw.sequence.med.intermediate_key=future_right_tactile \
  data.raw.sequence.med.future_start=1 \
  data.raw.sequence.med.future_end=8 \
  data.raw.sequence.med.stride=1 \
  data.raw.sequence.med.vector.field=tactile.right_tacglove \
  data.med.flatten_sequence=auto \
  model.med_branch.predictor.type=mlp \
  model.med_branch.projector.type=mlp \
  train.wandb.group=rawhand_phase3_vla_fulltrain

train_if_needed fulltrain "$PC_RUN" \
  experiment=fulltrain \
  experiment.loss_weights.action=1.0 \
  experiment.loss_weights.med="$PC_MED_WEIGHT" \
  data.raw.sequence.med.kind=point_cloud \
  data.raw.sequence.med.side=right \
  data.raw.sequence.med.intermediate_key=future_right_point_cloud \
  data.raw.sequence.med.future_start=1 \
  data.raw.sequence.med.future_end=8 \
  data.raw.sequence.med.stride=1 \
  data.med.flatten_sequence=auto \
  model.med_branch.predictor.type=point_query \
  model.med_branch.projector.type=pointnet \
  train.wandb.group=rawhand_phase3_vla_fulltrain

evaluate_if_needed "Outputs/vanilla/${VANILLA_RUN}" "$EVAL_ROOT/vanilla"
evaluate_if_needed "Outputs/fulltrain/${VEC_RUN}" "$EVAL_ROOT/fulltrain_vec_f1_8"
evaluate_if_needed "Outputs/fulltrain/${PC_RUN}" "$EVAL_ROOT/fulltrain_pc_f1_8_pq_pn"

if [[ "$RUN_SUMMARY" == "1" ]]; then
  "$PYTHON_BIN" Scripts/summarize_rawhand_fulltrain_results.py --output-root "$OUTPUT_ROOT"
else
  echo "skip summary: RUN_SUMMARY=${RUN_SUMMARY}"
fi
