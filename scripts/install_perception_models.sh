#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${DEXTELEOP_ENV_NAME:-agenticvla}"
SOURCE_ROOT=""
BACKEND="auto"
MAX_JOBS_VALUE="${MAX_JOBS:-4}"
CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
VERIFY_ONLY=0

SAM2_URL="https://github.com/facebookresearch/sam2.git"
SAM2_REVISION="aa9b8722d0585b661ded4b3dff1bd103540554ae"
GROUNDING_DINO_URL="https://github.com/IDEA-Research/GroundingDINO.git"
GROUNDING_DINO_REVISION="856dde20aee659246248e20734ef9ba5214f5e44"
COMPAT_HEADER="$REPO_ROOT/deployments/perception/grounding_dino_torch_2_10_compat.h"

usage() {
  cat <<'EOF'
Usage: bash scripts/install_perception_models.sh [options]

Clone the reviewed SAM2 and Grounding DINO revisions, install their pinned
Python dependencies into one existing Conda environment, build the optional
Grounding DINO CUDA extension, and run import/revision/operator smoke checks.

Options:
  --env-name NAME              Conda environment (default: agenticvla, or
                               DEXTELEOP_ENV_NAME).
  --source-root PATH           Stable checkout parent (default:
                               ENV_PREFIX/src/perception-models).
  --backend auto|cuda|cpu      Build mode (default: auto).
  --cuda-arch-list LIST        TORCH_CUDA_ARCH_LIST override, for example 8.9.
  --max-jobs N                 Extension build parallelism (default: 4).
  --verify-only                Do not clone/install/build; verify the expected
                               checkouts and installed packages only.
  -h, --help                   Show this help.

The script never downloads model checkpoints. Put the two reviewed weights in
the assets/perception/sam2_grounding_dino bundle as documented in README.md.
EOF
}

die() {
  echo "[perception-models] ERROR: $*" >&2
  exit 1
}

while (($# > 0)); do
  case "$1" in
    --env-name)
      (($# >= 2)) || die "--env-name requires a name"
      ENV_NAME="$2"
      shift 2
      ;;
    --source-root)
      (($# >= 2)) || die "--source-root requires a path"
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --backend)
      (($# >= 2)) || die "--backend requires auto, cuda, or cpu"
      BACKEND="$2"
      shift 2
      ;;
    --cuda-arch-list)
      (($# >= 2)) || die "--cuda-arch-list requires a value"
      CUDA_ARCH_LIST="$2"
      shift 2
      ;;
    --max-jobs)
      (($# >= 2)) || die "--max-jobs requires a positive integer"
      MAX_JOBS_VALUE="$2"
      shift 2
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "$BACKEND" in
  auto|cuda|cpu) ;;
  *) die "--backend must be auto, cuda, or cpu" ;;
esac
[[ "$MAX_JOBS_VALUE" =~ ^[1-9][0-9]*$ ]] || die \
  "--max-jobs must be a positive integer"

command -v conda >/dev/null 2>&1 || die "conda is not on PATH"
command -v git >/dev/null 2>&1 || die "git is not on PATH"
[[ -f "$COMPAT_HEADER" ]] || die "compatibility header is absent: $COMPAT_HEADER"

ENV_PREFIX="$(conda run --name "$ENV_NAME" python -c 'import sys; print(sys.prefix)' 2>/dev/null)" || die \
  "Conda environment '$ENV_NAME' is unavailable"
PYTHON_BIN="$ENV_PREFIX/bin/python"
[[ -x "$PYTHON_BIN" ]] || die "Python is absent from environment: $PYTHON_BIN"

if [[ -z "$SOURCE_ROOT" ]]; then
  SOURCE_ROOT="$ENV_PREFIX/src/perception-models"
elif [[ "$SOURCE_ROOT" != /* ]]; then
  SOURCE_ROOT="$PWD/$SOURCE_ROOT"
fi
SAM2_ROOT="$SOURCE_ROOT/sam2"
GROUNDING_DINO_ROOT="$SOURCE_ROOT/GroundingDINO"

PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
import sys
import torch

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected Python 3.10, got {sys.version.split()[0]}")
if torch.__version__ != "2.10.0+cu126":
    raise SystemExit(
        "AgenticRobot perception install requires torch==2.10.0+cu126; got "
        + torch.__version__
    )
PY

assert_tracked_clean() {
  local checkout="$1"
  git -C "$checkout" diff --quiet -- || die \
    "tracked changes exist in third-party checkout: $checkout"
  git -C "$checkout" diff --cached --quiet -- || die \
    "staged changes exist in third-party checkout: $checkout"
}

ensure_checkout() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local sparse_directory="$4"
  local checkout="$5"

  if [[ ! -e "$checkout" ]]; then
    ((VERIFY_ONLY == 0)) || die "$name checkout is absent: $checkout"
    echo "[perception-models] Cloning $name into $checkout"
    mkdir -p -- "$(dirname -- "$checkout")"
    git clone --filter=blob:none --no-checkout "$url" "$checkout"
    git -C "$checkout" sparse-checkout init --cone
    git -C "$checkout" sparse-checkout set "$sparse_directory"
  elif [[ ! -d "$checkout/.git" && ! -f "$checkout/.git" ]]; then
    die "$name source path exists but is not a Git checkout: $checkout"
  fi

  local actual_url
  actual_url="$(git -C "$checkout" remote get-url origin)" || die \
    "$name checkout has no origin remote"
  [[ "$actual_url" == "$url" ]] || die \
    "$name origin is not the reviewed official URL: $actual_url"
  assert_tracked_clean "$checkout"

  if ! git -C "$checkout" cat-file -e "${revision}^{commit}" 2>/dev/null; then
    ((VERIFY_ONLY == 0)) || die "$name reviewed revision is absent: $revision"
    echo "[perception-models] Fetching $name revision $revision"
    git -C "$checkout" fetch --depth=1 origin "$revision"
  fi
  if [[ "$(git -C "$checkout" rev-parse HEAD 2>/dev/null || true)" != "$revision" ]]; then
    ((VERIFY_ONLY == 0)) || die "$name HEAD does not match $revision"
    echo "[perception-models] Checking out $name revision $revision"
    git -C "$checkout" checkout --detach "$revision"
  fi
  assert_tracked_clean "$checkout"
}

ensure_checkout "SAM2" "$SAM2_URL" "$SAM2_REVISION" "sam2" "$SAM2_ROOT"
ensure_checkout \
  "Grounding DINO" \
  "$GROUNDING_DINO_URL" \
  "$GROUNDING_DINO_REVISION" \
  "groundingdino" \
  "$GROUNDING_DINO_ROOT"

[[ -f "$SAM2_ROOT/sam2/build_sam.py" ]] || die "SAM2 runtime source is incomplete"
[[ -f "$SAM2_ROOT/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" ]] || die \
  "SAM2 model config is incomplete"
[[ -f "$GROUNDING_DINO_ROOT/groundingdino/util/inference.py" ]] || die \
  "Grounding DINO inference API is incomplete"

CUDA_AVAILABLE="$(PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c \
  'import torch; print("1" if torch.cuda.is_available() else "0")')"
CUDA_HOME_VALUE="$(PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c \
  'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or "")')"

BUILD_CUDA=0
if [[ "$BACKEND" == "cuda" ]]; then
  [[ "$CUDA_AVAILABLE" == "1" ]] || die \
    "--backend cuda requested but PyTorch cannot access a CUDA device"
  BUILD_CUDA=1
elif [[ "$BACKEND" == "auto" && "$CUDA_AVAILABLE" == "1" ]]; then
  BUILD_CUDA=1
fi

if ((BUILD_CUDA)); then
  [[ -n "$CUDA_HOME_VALUE" && -x "$CUDA_HOME_VALUE/bin/nvcc" ]] || die \
    "CUDA inference is available but nvcc was not found; pass --backend cpu or install a CUDA toolkit"
  if [[ -z "$CUDA_ARCH_LIST" ]]; then
    CUDA_ARCH_LIST="$(PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c \
      'import torch; p = torch.cuda.get_device_properties(0); print(f"{p.major}.{p.minor}")')"
  fi
fi

if ((VERIFY_ONLY == 0)); then
  echo "[perception-models] Installing reviewed Python dependencies into $ENV_NAME"
  "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cu126 \
    --no-deps "torchvision==0.25.0"
  "$PYTHON_BIN" -m pip install --index-url https://pypi.org/simple \
    "numpy==1.26.4" \
    "hydra-core==1.3.2" \
    "iopath==0.1.10" \
    "transformers==4.39.2" \
    "addict==2.4.0" \
    "yapf==0.40.2" \
    "timm==0.9.16" \
    "supervision==0.22.0" \
    "pycocotools==2.0.8" \
    "ninja==1.13.0"

  echo "[perception-models] Installing SAM2 from the reviewed checkout"
  SAM2_BUILD_CUDA=0 "$PYTHON_BIN" -m pip install \
    --no-deps --no-build-isolation --editable "$SAM2_ROOT"

  if ((BUILD_CUDA)); then
    echo "[perception-models] Building Grounding DINO CUDA extension for arch $CUDA_ARCH_LIST"
    CUDA_HOME="$CUDA_HOME_VALUE" \
    TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST" \
    MAX_JOBS="$MAX_JOBS_VALUE" \
    NVCC_PREPEND_FLAGS="-include=$COMPAT_HEADER ${NVCC_PREPEND_FLAGS:-}" \
    "$PYTHON_BIN" -m pip install \
      --no-deps --no-build-isolation --editable "$GROUNDING_DINO_ROOT"
  else
    echo "[perception-models] Installing Grounding DINO with its CPU fallback"
    CUDA_VISIBLE_DEVICES="" TORCH_CUDA_ARCH_LIST="" \
    "$PYTHON_BIN" -m pip install \
      --no-deps --no-build-isolation --editable "$GROUNDING_DINO_ROOT"
  fi

  # Grounding DINO/supervision may install overlapping OpenCV wheels. Restore
  # the AgenticRobot-locked contrib implementation last so cv2.aruco remains.
  "$PYTHON_BIN" -m pip install --index-url https://pypi.org/simple \
    --force-reinstall --no-deps "opencv-contrib-python==4.10.0.84"
fi

echo "[perception-models] Verifying package provenance and inference APIs"
EXPECTED_SAM2_REVISION="$SAM2_REVISION" \
EXPECTED_GROUNDING_DINO_REVISION="$GROUNDING_DINO_REVISION" \
EXPECT_CUDA_EXTENSION="$BUILD_CUDA" \
PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
import os
import pathlib
import subprocess

import cv2
import groundingdino
import sam2
import torch
import torchvision
from groundingdino.util.inference import load_model, predict
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def revision(package_file: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(pathlib.Path(package_file).resolve().parent), "rev-parse", "HEAD"],
        text=True,
    ).strip()


if revision(sam2.__file__) != os.environ["EXPECTED_SAM2_REVISION"]:
    raise SystemExit("installed SAM2 revision differs from the reviewed revision")
if revision(groundingdino.__file__) != os.environ["EXPECTED_GROUNDING_DINO_REVISION"]:
    raise SystemExit("installed Grounding DINO revision differs from the reviewed revision")
if torchvision.__version__ != "0.25.0+cu126":
    raise SystemExit(f"expected torchvision 0.25.0+cu126, got {torchvision.__version__}")
if cv2.__version__ != "4.10.0" or not hasattr(cv2, "aruco"):
    raise SystemExit("AgenticRobot OpenCV contrib runtime was not preserved")

if os.environ["EXPECT_CUDA_EXTENSION"] == "1":
    from groundingdino import _C

    device = torch.device("cuda")
    value = torch.rand(1, 4, 1, 4, device=device)
    shapes = torch.tensor([[2, 2]], dtype=torch.int64, device=device)
    starts = torch.tensor([0], dtype=torch.int64, device=device)
    locations = torch.rand(1, 2, 1, 1, 1, 2, device=device)
    weights = torch.ones(1, 2, 1, 1, 1, device=device)
    output = _C.ms_deform_attn_forward(
        value, shapes, starts, locations, weights, 1
    )
    if tuple(output.shape) != (1, 2, 4) or not bool(torch.isfinite(output).all()):
        raise SystemExit("Grounding DINO CUDA operator smoke check failed")

print("SAM2/Grounding DINO install verified")
print("SAM2", os.environ["EXPECTED_SAM2_REVISION"])
print("Grounding DINO", os.environ["EXPECTED_GROUNDING_DINO_REVISION"])
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("CUDA available", torch.cuda.is_available())
PY

echo
echo "[perception-models] Installation complete."
echo "  conda activate $ENV_NAME"
echo "  source checkouts: $SOURCE_ROOT"
echo "Weights are separate runtime assets; see README.md for download and inference commands."
