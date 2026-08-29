#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${DEXTELEOP_ENV_NAME:-agenticvla}"
SOURCE_ROOT=""
BACKEND="auto"
VERIFY_ONLY=0
REUSE_INSTALLED_PERCEPTION=0
WITH_GRASPNET=0
ACCEPT_GRASPNET_LICENSE=0
CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
MAX_JOBS_VALUE="${MAX_JOBS:-4}"
VGN_URL="https://github.com/ethz-asl/vgn.git"
VGN_REVISION="d7af0622433f52ae88ebe81533f12b46b33e951a"
GRASPNET_URL="https://github.com/graspnet/graspnet-baseline.git"
GRASPNET_REVISION="280c215129f759ed8649cb4e89fc5dfee55f4f80"

usage() {
  cat <<'EOF'
Usage: bash scripts/install_point_grasp_models.sh [options]

Install the existing reviewed SAM2/Grounding DINO stack, clone the reviewed
BSD-3-Clause VGN revision, install its headless inference dependencies, and verify the
point-grasp APIs in one Conda environment. Checkpoints are separate assets.

Options:
  --env-name NAME          Conda environment (default: agenticvla)
  --source-root PATH       Checkout parent (default: ENV_PREFIX/src/perception-models)
  --backend auto|cuda|cpu  SAM2/Grounding DINO build mode (default: auto)
  --reuse-installed-perception
                           Verify and reuse already imported reviewed SAM2/DINO
  --with-graspnet          Also install the optional GraspNet backend
  --accept-graspnet-noncommercial-license
                           Confirm local noncommercial research-only use; required
                           with --with-graspnet
  --cuda-arch-list LIST    GraspNet pointnet2 CUDA architecture, e.g. 8.9
  --max-jobs N             GraspNet CUDA build parallelism (default: 4)
  --verify-only            Do not clone or install; only verify
  -h, --help               Show this help
EOF
}

die() { echo "[point-grasp-models] ERROR: $*" >&2; exit 1; }

while (($#)); do
  case "$1" in
    --env-name) ENV_NAME="${2:?}"; shift 2 ;;
    --source-root) SOURCE_ROOT="${2:?}"; shift 2 ;;
    --backend) BACKEND="${2:?}"; shift 2 ;;
    --reuse-installed-perception) REUSE_INSTALLED_PERCEPTION=1; shift ;;
    --with-graspnet) WITH_GRASPNET=1; shift ;;
    --accept-graspnet-noncommercial-license) ACCEPT_GRASPNET_LICENSE=1; shift ;;
    --cuda-arch-list) CUDA_ARCH_LIST="${2:?}"; shift 2 ;;
    --max-jobs) MAX_JOBS_VALUE="${2:?}"; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ "$BACKEND" == "auto" || "$BACKEND" == "cuda" || "$BACKEND" == "cpu" ]] || die "--backend must be auto, cuda, or cpu"
[[ "$MAX_JOBS_VALUE" =~ ^[1-9][0-9]*$ ]] || die "--max-jobs must be a positive integer"
if ((WITH_GRASPNET && ! ACCEPT_GRASPNET_LICENSE)); then
  die "--with-graspnet requires --accept-graspnet-noncommercial-license"
fi
if ((ACCEPT_GRASPNET_LICENSE && ! WITH_GRASPNET)); then
  die "the GraspNet license flag requires --with-graspnet"
fi
command -v conda >/dev/null 2>&1 || die "conda is not on PATH"
ENV_PREFIX="$(conda run --name "$ENV_NAME" python -c 'import sys; print(sys.prefix)')" || die "Conda environment is unavailable"
PYTHON_BIN="$ENV_PREFIX/bin/python"
if [[ -z "$SOURCE_ROOT" ]]; then
  SOURCE_ROOT="$ENV_PREFIX/src/perception-models"
elif [[ "$SOURCE_ROOT" != /* ]]; then
  SOURCE_ROOT="$PWD/$SOURCE_ROOT"
fi
VGN_ROOT="$SOURCE_ROOT/vgn"
GRASPNET_ROOT="$SOURCE_ROOT/graspnet-baseline"

if ((REUSE_INSTALLED_PERCEPTION)); then
  POINT_GRASP_MPL_CACHE="${TMPDIR:-/tmp}/agenticvla-point-grasp-matplotlib-${UID}"
  mkdir -p -- "$POINT_GRASP_MPL_CACHE"
  EXPECTED_SAM2_REVISION="aa9b8722d0585b661ded4b3dff1bd103540554ae" \
  EXPECTED_DINO_REVISION="856dde20aee659246248e20734ef9ba5214f5e44" \
  MPLCONFIGDIR="$POINT_GRASP_MPL_CACHE" \
  PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
import os
import pathlib
import subprocess

import groundingdino
import sam2
from groundingdino.util.inference import Model
from sam2.sam2_image_predictor import SAM2ImagePredictor

def checkout(module, expected_url, expected_revision):
    package = pathlib.Path(module.__file__).resolve().parent
    root = subprocess.check_output(
        ["git", "-C", str(package), "rev-parse", "--show-toplevel"], text=True
    ).strip()
    revision = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD"], text=True
    ).strip()
    url = subprocess.check_output(
        ["git", "-C", root, "remote", "get-url", "origin"], text=True
    ).strip()
    if revision != expected_revision or url != expected_url:
        raise SystemExit(f"reviewed source identity mismatch: {root}")

checkout(
    sam2,
    "https://github.com/facebookresearch/sam2.git",
    os.environ["EXPECTED_SAM2_REVISION"],
)
checkout(
    groundingdino,
    "https://github.com/IDEA-Research/GroundingDINO.git",
    os.environ["EXPECTED_DINO_REVISION"],
)
if not callable(getattr(SAM2ImagePredictor, "predict", None)) or Model is None:
    raise SystemExit("installed perception inference APIs are unavailable")
print("reviewed installed SAM2/Grounding DINO reused")
PY
else
  SAM_ARGS=(--env-name "$ENV_NAME" --backend "$BACKEND" --source-root "$SOURCE_ROOT")
  if ((VERIFY_ONLY)); then SAM_ARGS+=(--verify-only); fi
  bash "$REPO_ROOT/scripts/install_perception_models.sh" "${SAM_ARGS[@]}"
fi

if [[ ! -e "$VGN_ROOT" ]]; then
  ((VERIFY_ONLY == 0)) || die "VGN checkout is absent: $VGN_ROOT"
  mkdir -p -- "$SOURCE_ROOT"
  git clone --filter=blob:none "$VGN_URL" "$VGN_ROOT"
elif [[ ! -d "$VGN_ROOT/.git" ]]; then
  die "VGN source path is not a Git checkout: $VGN_ROOT"
fi
[[ "$(git -C "$VGN_ROOT" remote get-url origin)" == "$VGN_URL" ]] || die "VGN origin is not reviewed"
git -C "$VGN_ROOT" diff --quiet -- || die "tracked VGN changes exist"
git -C "$VGN_ROOT" diff --cached --quiet -- || die "staged VGN changes exist"
if ! git -C "$VGN_ROOT" cat-file -e "${VGN_REVISION}^{commit}" 2>/dev/null; then
  ((VERIFY_ONLY == 0)) || die "reviewed VGN revision is absent"
  git -C "$VGN_ROOT" fetch --depth=1 origin "$VGN_REVISION"
fi
if [[ "$(git -C "$VGN_ROOT" rev-parse HEAD)" != "$VGN_REVISION" ]]; then
  ((VERIFY_ONLY == 0)) || die "VGN HEAD differs from the reviewed revision"
  git -C "$VGN_ROOT" checkout --detach "$VGN_REVISION"
fi

if ((VERIFY_ONLY == 0)); then
  "$PYTHON_BIN" -m pip install --index-url https://pypi.org/simple \
    "catkin-pkg==1.0.0" \
    "scipy==1.14.1"
  "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation --editable "$VGN_ROOT"
fi

EXPECTED_VGN_REVISION="$VGN_REVISION" PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
import os
import pathlib
import subprocess
import torch
import vgn
from vgn.networks import get_network

revision = subprocess.check_output(
    ["git", "-C", str(pathlib.Path(vgn.__file__).resolve().parent), "rev-parse", "HEAD"],
    text=True,
).strip()
if revision != os.environ["EXPECTED_VGN_REVISION"]:
    raise SystemExit("installed VGN revision differs from the reviewed revision")
network = get_network("conv")
if not callable(getattr(network, "forward", None)):
    raise SystemExit("VGN inference network is unavailable")
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
print("VGN point-grasp install verified", revision)
PY

if ((WITH_GRASPNET)); then
  if [[ ! -e "$GRASPNET_ROOT" ]]; then
    ((VERIFY_ONLY == 0)) || die "GraspNet checkout is absent: $GRASPNET_ROOT"
    mkdir -p -- "$SOURCE_ROOT"
    git clone --filter=blob:none "$GRASPNET_URL" "$GRASPNET_ROOT"
  elif [[ ! -d "$GRASPNET_ROOT/.git" ]]; then
    die "GraspNet source path is not a Git checkout: $GRASPNET_ROOT"
  fi
  [[ "$(git -C "$GRASPNET_ROOT" remote get-url origin)" == "$GRASPNET_URL" ]] || die "GraspNet origin is not reviewed"
  git -C "$GRASPNET_ROOT" diff --quiet -- || die "tracked GraspNet changes exist"
  git -C "$GRASPNET_ROOT" diff --cached --quiet -- || die "staged GraspNet changes exist"
  if ! git -C "$GRASPNET_ROOT" cat-file -e "${GRASPNET_REVISION}^{commit}" 2>/dev/null; then
    ((VERIFY_ONLY == 0)) || die "reviewed GraspNet revision is absent"
    git -C "$GRASPNET_ROOT" fetch --depth=1 origin "$GRASPNET_REVISION"
  fi
  if [[ "$(git -C "$GRASPNET_ROOT" rev-parse HEAD)" != "$GRASPNET_REVISION" ]]; then
    ((VERIFY_ONLY == 0)) || die "GraspNet HEAD differs from the reviewed revision"
    git -C "$GRASPNET_ROOT" checkout --detach "$GRASPNET_REVISION"
  fi
  CUDA_HOME_VALUE="$(PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or "")')"
  [[ -n "$CUDA_HOME_VALUE" && -x "$CUDA_HOME_VALUE/bin/nvcc" ]] || die "GraspNet pointnet2 requires a CUDA toolkit with nvcc"
  if [[ -z "$CUDA_ARCH_LIST" ]]; then
    CUDA_ARCH_LIST="$(PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c 'import torch; print("" if not torch.cuda.is_available() else f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")')"
  fi
  [[ -n "$CUDA_ARCH_LIST" ]] || die "GraspNet build cannot see a GPU; pass --cuda-arch-list (RTX 4090: 8.9)"

  GRASPNET_SITE_PACKAGES="$(PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])')"
  GRASPNET_PTH="$GRASPNET_SITE_PACKAGES/agenticvla_graspnet_baseline.pth"
  if ((VERIFY_ONLY == 0)); then
    "$PYTHON_BIN" -m pip install --index-url https://pypi.org/simple "ninja==1.11.1.4"
    MAX_JOBS="$MAX_JOBS_VALUE" TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST" \
      "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation --force-reinstall \
      "$GRASPNET_ROOT/pointnet2"
    printf '%s\n' \
      "$GRASPNET_ROOT/models" \
      "$GRASPNET_ROOT/utils" \
      "$GRASPNET_ROOT/pointnet2" > "$GRASPNET_PTH"
  fi
  [[ -f "$GRASPNET_PTH" ]] || die "GraspNet import-path registration is absent: $GRASPNET_PTH"
  EXPECTED_GRASPNET_REVISION="$GRASPNET_REVISION" PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
import importlib
import pathlib
import subprocess

import torch
import pointnet2._ext

spec = importlib.util.find_spec("graspnet")
if spec is None or spec.origin is None:
    raise SystemExit("GraspNet model module is unavailable")
root = subprocess.check_output(
    ["git", "-C", str(pathlib.Path(spec.origin).resolve().parent), "rev-parse", "--show-toplevel"],
    text=True,
).strip()
revision = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
if revision != __import__("os").environ["EXPECTED_GRASPNET_REVISION"]:
    raise SystemExit("installed GraspNet revision differs from the reviewed revision")
print("GraspNet point-grasp install verified", revision)
PY
fi

echo "[point-grasp-models] Installation complete. Checkpoints were not downloaded; see README.md."
