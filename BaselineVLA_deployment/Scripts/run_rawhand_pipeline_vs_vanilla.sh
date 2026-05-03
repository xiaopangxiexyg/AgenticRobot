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

# Current direct Vanilla reference from the completed VLA capacity check.
# Note: this run is a historical 5-epoch run. If you later rerun a strict 2-epoch
# vanilla baseline, replace this path and keep the rest of the script unchanged.
VANILLA_REF="Outputs/vanilla/rawhand_phase1_vla_iso_shared_vanilla_seed7"

VEC_GTMED="Outputs/gtmed/rawhand_phase2_vla_oracle_gtmed_vec_f1_8_seed7"
VEC_GENMED="Outputs/genmed/rawhand_phase2_vla_genmed_vec_f1_8_seed7"

PC_GTMED="Outputs/gtmed/rawhand_phase2_vla_oracle_gtmed_pc_f1_8_pn_seed7"
PC_GENMED="Outputs/genmed/rawhand_phase2_vla_genmed_pc_f1_8_pq_seed7"

OUTPUT_ROOT="Outputs/comparisons/rawhand_pipeline_vs_vanilla"

run_case() {
  local case_name="$1"
  local gtmed_run="$2"
  local genmed_run="$3"
  local output_dir="$4"

  echo "===== ${case_name} ====="
  "$PYTHON_BIN" TestScripts/evaluate_gtmed_pipeline.py \
    "$gtmed_run" \
    --genmed-checkpoint "$genmed_run" \
    --vanilla-checkpoint "$VANILLA_REF" \
    --gtmed-checkpoint-kind best \
    --genmed-checkpoint-kind best \
    --vanilla-checkpoint-kind best \
    --split val \
    --pipeline-mode cached \
    --reuse-predicted-med-cache \
    --loader-mode safe \
    --batch-size 8 \
    --num-workers 0 \
    --save-samples 0 \
    --visualize none \
    --device cuda \
    --genmed-device cpu \
    --output-dir "$output_dir"
}

mkdir -p "$OUTPUT_ROOT"

run_case "vec_f1_8" "$VEC_GTMED" "$VEC_GENMED" "$OUTPUT_ROOT/vec_f1_8"
run_case "pc_f1_8_pn" "$PC_GTMED" "$PC_GENMED" "$OUTPUT_ROOT/pc_f1_8_pn"

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

output_root = Path("Outputs/comparisons/rawhand_pipeline_vs_vanilla")
cases = ("vec_f1_8", "pc_f1_8_pn")

print("\n===== compact summary =====")
print("| case | vanilla rmse | oracle rmse | pipeline rmse | pipeline vs vanilla | pipeline vs oracle | summary |")
print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")

for case_name in cases:
    summary_path = output_root / case_name / "summary.json"
    summary = json.loads(summary_path.read_text())
    vanilla_rmse = float(summary["results"]["vanilla"]["val"]["metrics"]["action"]["rmse"])
    oracle_rmse = float(summary["results"]["gtmed_ground_truth_med"]["val"]["metrics"]["action"]["rmse"])
    pipeline_rmse = float(summary["results"]["gtmed_with_genmed_predicted_med"]["val"]["metrics"]["action"]["rmse"])
    pipeline_vs_vanilla = (pipeline_rmse - vanilla_rmse) / vanilla_rmse * 100.0
    pipeline_vs_oracle = (pipeline_rmse - oracle_rmse) / oracle_rmse * 100.0
    print(
        f"| {case_name} | {vanilla_rmse:.6f} | {oracle_rmse:.6f} | {pipeline_rmse:.6f} | "
        f"{pipeline_vs_vanilla:+.2f}% | {pipeline_vs_oracle:+.4f}% | {summary_path} |"
    )
PY
