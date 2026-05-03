#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=Scripts/load_local_env.sh
source "${SCRIPT_DIR}/load_local_env.sh"
PROJECT_ROOT="${PROJECT_ROOT:-$MANIPREP_PROJECT_ROOT}"
CONDA_ENV="${CONDA_ENV:-ManipRep_310}"
PYTHON_BIN="${PYTHON_BIN:-python}"
maniprep_activate_conda "$CONDA_ENV"
SUITE_JSON="${SUITE_JSON:-ExperimentSuites/humanmix_vla/humanmix_vla_action_history_grid.json}"
MODE="${MODE:-all}"
FORCE="${FORCE:-0}"
MAX_RUNS="${MAX_RUNS:-}"
ONLY="${ONLY:-}"

BATCH_SIZE="${BATCH_SIZE:-16}"
DATA_RATIO="${DATA_RATIO:-${VAL_RATIO:-0.2}}"
DATA_SEED="${DATA_SEED:-7}"
EPOCHS="${EPOCHS:-1}"
DEVICE="${DEVICE:-cuda}"
WANDB_MODE="${WANDB_MODE:-offline}"
NUM_WORKERS="${NUM_WORKERS:-}"
if [[ -n "$NUM_WORKERS" ]]; then
  TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-$NUM_WORKERS}"
  VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-$NUM_WORKERS}"
  EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-$NUM_WORKERS}"
else
  TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
  VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
  EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
fi
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
PIN_MEMORY="${PIN_MEMORY:-false}"
FRAME_STRIDE="${FRAME_STRIDE:-15}"
VIDEO_ENABLED="${VIDEO_ENABLED:-true}"
VIDEO_CLIP_LENGTH="${VIDEO_CLIP_LENGTH:-4}"
VIDEO_CLIP_STRIDE="${VIDEO_CLIP_STRIDE:-4}"
VIDEO_RESIZE_HEIGHT="${VIDEO_RESIZE_HEIGHT:-90}"
VIDEO_RESIZE_WIDTH="${VIDEO_RESIZE_WIDTH:-160}"
PROGRESS_ENABLED="${PROGRESS_ENABLED:-true}"
SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT:-false}"
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-true}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-}"

SUMMARY_DIR="Outputs/suites/humanmix_vla_action_history_grid"
SUMMARY_TABLE="${SUMMARY_DIR}/humanmix_vla_grid_table.md"

cd "$PROJECT_ROOT"

base_args=("$SUITE_JSON")
base_args+=(--set "train.epochs=${EPOCHS}")
base_args+=(--set "train.device=${DEVICE}")
base_args+=(--set "train.wandb.mode=${WANDB_MODE}")
base_args+=(--set "train.progress.enabled=${PROGRESS_ENABLED}")
base_args+=(--set "train.save_last_checkpoint=${SAVE_LAST_CHECKPOINT}")
base_args+=(--set "train.save_best_checkpoint=${SAVE_BEST_CHECKPOINT}")
base_args+=(--set "train.validation_ratio=1.0")
base_args+=(--set "++data.raw.sampling.sample_ratio=${DATA_RATIO}")
base_args+=(--set "++data.raw.sampling.sample_seed=${DATA_SEED}")
base_args+=(--set "++data.raw.sampling.min_history_frames=16")
base_args+=(--set "++data.raw.sampling.min_future_frames=93")
base_args+=(--set "data.raw.sampling.frame_stride=${FRAME_STRIDE}")
base_args+=(--set "data.raw.video.enabled=${VIDEO_ENABLED}")
base_args+=(--set "data.raw.video.clip_length=${VIDEO_CLIP_LENGTH}")
base_args+=(--set "data.raw.video.clip_stride=${VIDEO_CLIP_STRIDE}")
base_args+=(--set "data.raw.video.resize_height=${VIDEO_RESIZE_HEIGHT}")
base_args+=(--set "data.raw.video.resize_width=${VIDEO_RESIZE_WIDTH}")
base_args+=(--set "data.raw.loader.batch_size=${BATCH_SIZE}")
base_args+=(--set "data.raw.loader.num_workers=${TRAIN_NUM_WORKERS}")
base_args+=(--set "data.raw.loader.prefetch_factor=${PREFETCH_FACTOR}")
base_args+=(--set "data.raw.loader.persistent_workers=${PERSISTENT_WORKERS}")
base_args+=(--set "data.raw.loader.pin_memory=${PIN_MEMORY}")
base_args+=(--set "data.raw.loader.val.num_workers=${VAL_NUM_WORKERS}")
base_args+=(--set "data.raw.loader.val.prefetch_factor=${PREFETCH_FACTOR}")
base_args+=(--set "data.raw.loader.val.persistent_workers=${PERSISTENT_WORKERS}")
base_args+=(--eval-batch-size "${BATCH_SIZE}")
base_args+=(--eval-num-workers "${EVAL_NUM_WORKERS}")
if [[ -n "$EVAL_MAX_BATCHES" && "$EVAL_MAX_BATCHES" != "0" ]]; then
  base_args+=(--eval-max-batches "$EVAL_MAX_BATCHES")
fi

run_args=("${base_args[@]}")
if [[ "$FORCE" == "1" ]]; then
  run_args+=(--force)
else
  run_args+=(--skip-existing)
fi
if [[ -n "$MAX_RUNS" ]]; then
  run_args+=(--max-runs "$MAX_RUNS")
fi

print_plan() {
  printf '\n[HumanMix VLA action/history grid]\n'
  printf 'suite: %s\n' "$SUITE_JSON"
  printf 'python: %s\n' "$PYTHON_BIN"
  printf 'order: Vanilla-tactile VLA first, then FullTrain VLA\n'
  printf 'action_chunk order: 8 -> 16 -> 32\n'
  printf 'history order: no_his -> his1_s2 -> his1_s4 -> his1_s8 -> his2_s2 -> his2_s4 -> his2_s8\n'
  printf 'planned runs: 42 total = 21 Vanilla-tactile + 21 FullTrain\n'
  printf 'batch_size shared by train/eval: %s\n' "$BATCH_SIZE"
  printf 'dataloader workers: train=%s val=%s eval=%s\n' "$TRAIN_NUM_WORKERS" "$VAL_NUM_WORKERS" "$EVAL_NUM_WORKERS"
  printf 'loader prefetch=%s persistent_workers=%s pin_memory=%s\n' "$PREFETCH_FACTOR" "$PERSISTENT_WORKERS" "$PIN_MEMORY"
  printf 'train/eval data ratio: %s\n' "$DATA_RATIO"
  printf 'data subset seed: %s\n' "$DATA_SEED"
  printf 'frame_stride: %s\n' "$FRAME_STRIDE"
  printf 'video: enabled=%s clip_length=%s clip_stride=%s resize=%sx%s\n' "$VIDEO_ENABLED" "$VIDEO_CLIP_LENGTH" "$VIDEO_CLIP_STRIDE" "$VIDEO_RESIZE_HEIGHT" "$VIDEO_RESIZE_WIDTH"
  printf 'common frame window: min_history_frames=16, min_future_frames=93\n'
  printf 'epochs: %s | device: %s | wandb mode: %s | progress=%s\n' "$EPOCHS" "$DEVICE" "$WANDB_MODE" "$PROGRESS_ENABLED"
  printf 'checkpoints: best=%s last=%s\n' "$SAVE_BEST_CHECKPOINT" "$SAVE_LAST_CHECKPOINT"
  if [[ -n "$EVAL_MAX_BATCHES" && "$EVAL_MAX_BATCHES" != "0" ]]; then
    printf 'post-train eval max batches: %s\n' "$EVAL_MAX_BATCHES"
  else
    printf 'post-train eval max batches: unlimited; eval speedup uses DATA_RATIO\n'
  fi
  printf 'summary table: %s\n\n' "$SUMMARY_TABLE"
}

refresh_summary() {
  printf '\n[summary] refreshing canonical 42-run status table\n'
  "$PYTHON_BIN" Scripts/run_experiment_suite.py "${base_args[@]}" --summarize-existing-only
  "$PYTHON_BIN" Scripts/summarize_humanmix_vla_grid.py --summary-dir "$SUMMARY_DIR"
}

run_stage() {
  local stage_name="$1"
  local only_filter="$2"
  local planned_count="$3"
  printf '\n[%s] starting %s planned runs\n' "$stage_name" "$planned_count"
  "$PYTHON_BIN" Scripts/run_experiment_suite.py "${run_args[@]}" --only "$only_filter"
  printf '[%s] finished; refreshing summary\n' "$stage_name"
  refresh_summary
}

print_plan

if [[ -n "$ONLY" ]]; then
  printf '[custom] running only filter: %s\n' "$ONLY"
  "$PYTHON_BIN" Scripts/run_experiment_suite.py "${run_args[@]}" --only "$ONLY"
  refresh_summary
  exit 0
fi

case "$MODE" in
  all)
    run_stage "stage 1/2" "vtat" 21
    run_stage "stage 2/2" "fulltrain" 21
    ;;
  vanilla|vanilla-tactile|vtat)
    run_stage "stage 1/1" "vtat" 21
    ;;
  fulltrain)
    run_stage "stage 1/1" "fulltrain" 21
    ;;
  summary)
    refresh_summary
    ;;
  dry-run)
    "$PYTHON_BIN" Scripts/run_experiment_suite.py "${base_args[@]}" --dry-run
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Expected one of: all, vanilla, fulltrain, summary, dry-run." >&2
    exit 2
    ;;
esac
