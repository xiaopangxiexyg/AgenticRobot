#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${DEXTELEOP_ENV_NAME:-dexteleop310}"
UPDATE=0
SKIP_VENDOR_SDK=0
SDK_ROOT=""
CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
MAX_JOBS_VALUE="${MAX_JOBS:-4}"

usage() {
  cat <<'EOF'
Usage: bash scripts/install_point_grasp_env.sh [options]

Create (or update) the canonical dexteleop310 environment, initialize the
reviewed perception submodules, and install the CUDA SAM2, Grounding DINO,
VGN, GraspNet and PointNet2 stack. Model checkpoints are not downloaded.

Options:
  --update                 Update an existing environment instead of creating it.
  --skip-vendor-sdk        Do not install the local DM-Tac SDK.
  --sdk-root PATH          Override the DM-Tac SDK source path.
  --env-name NAME          Conda environment (default: dexteleop310).
  --cuda-arch-list LIST    CUDA architecture, for example 8.9 for RTX 4090.
  --max-jobs N             Parallel CUDA build jobs (default: MAX_JOBS or 4).
  -h, --help               Show this help.

Use --update only after pulling a newer AgenticRobot revision. On machines
without the DM-Tac SDK, pass --skip-vendor-sdk. This script never fetches
model weights; provision assets/perception/ separately.
EOF
}

die() { echo "[point-grasp-env] ERROR: $*" >&2; exit 1; }

while (($#)); do
  case "$1" in
    --update) UPDATE=1; shift ;;
    --skip-vendor-sdk) SKIP_VENDOR_SDK=1; shift ;;
    --sdk-root) SDK_ROOT="${2:?--sdk-root requires a path}"; shift 2 ;;
    --env-name) ENV_NAME="${2:?--env-name requires a name}"; shift 2 ;;
    --cuda-arch-list) CUDA_ARCH_LIST="${2:?--cuda-arch-list requires a value}"; shift 2 ;;
    --max-jobs) MAX_JOBS_VALUE="${2:?--max-jobs requires a positive integer}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$MAX_JOBS_VALUE" =~ ^[1-9][0-9]*$ ]] || die \
  "--max-jobs must be a positive integer"

BASE_ARGS=(--env-name "$ENV_NAME")
if ((UPDATE)); then BASE_ARGS+=(--update); fi
if ((SKIP_VENDOR_SDK)); then BASE_ARGS+=(--skip-vendor-sdk); fi
if [[ -n "$SDK_ROOT" ]]; then BASE_ARGS+=(--sdk-root "$SDK_ROOT"); fi

echo "[point-grasp-env] Synchronizing $ENV_NAME"
bash "$REPO_ROOT/scripts/install_conda_env.sh" "${BASE_ARGS[@]}"

echo "[point-grasp-env] Initializing reviewed perception submodules"
git -C "$REPO_ROOT" submodule update --init --recursive -- \
  model/perception/sam2 \
  model/perception/GroundingDINO \
  model/perception/vgn \
  model/perception/graspnet-baseline

MODEL_ARGS=(
  --env-name "$ENV_NAME"
  --backend cuda
  --with-graspnet
  --max-jobs "$MAX_JOBS_VALUE"
)
if [[ -n "$CUDA_ARCH_LIST" ]]; then
  MODEL_ARGS+=(--cuda-arch-list "$CUDA_ARCH_LIST")
fi

echo "[point-grasp-env] Installing point-grasp model runtime"
bash "$REPO_ROOT/scripts/install_point_grasp_models.sh" "${MODEL_ARGS[@]}"

echo "[point-grasp-env] Complete. Weights remain separate assets."
echo "  conda activate $ENV_NAME"
