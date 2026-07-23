#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${DEXTELEOP_ENV_NAME:-dexteleop310}"
SDK_ROOT="${DEXTELEOP_DMTAC_SDK_ROOT:-$REPO_ROOT/assets/DM_Tac_SDK_Publish_1.2.10}"
UPDATE=0
SKIP_VENDOR_SDK=0

usage() {
  cat <<'EOF'
Usage: bash scripts/install_conda_env.sh [options]

Create the complete Python 3.10 DexTeleop conda environment.

Options:
  --update                 Update an existing environment and prune packages.
  --skip-vendor-sdk        Do not install the local DM-Tac vendor SDK.
  --sdk-root PATH          Use a different DM-Tac SDK checkout.
  --env-name NAME          Use a different conda environment name.
  -h, --help               Show this help.
EOF
}

die() {
  echo "[dexteleop-env] ERROR: $*" >&2
  exit 1
}

while (($# > 0)); do
  case "$1" in
    --update)
      UPDATE=1
      shift
      ;;
    --skip-vendor-sdk)
      SKIP_VENDOR_SDK=1
      shift
      ;;
    --sdk-root)
      (($# >= 2)) || die "--sdk-root requires a path"
      SDK_ROOT="$2"
      shift 2
      ;;
    --env-name)
      (($# >= 2)) || die "--env-name requires a name"
      ENV_NAME="$2"
      shift 2
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

command -v conda >/dev/null 2>&1 || die "conda is not on PATH"

if ((SKIP_VENDOR_SDK == 0)) && [[ ! -f "$SDK_ROOT/setup.py" ]]; then
  die "DM-Tac SDK not found at '$SDK_ROOT'. Set DEXTELEOP_DMTAC_SDK_ROOT or use --skip-vendor-sdk."
fi

CONDA_BASE="$(conda info --base)"
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
[[ -f "$CONDA_SH" ]] || die "cannot find conda shell integration: $CONDA_SH"
# shellcheck disable=SC1090
source "$CONDA_SH"

cd "$REPO_ROOT"
if conda env list | awk 'NF > 0 && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
  ((UPDATE == 1)) || die "environment '$ENV_NAME' already exists; rerun with --update or choose --env-name"
  echo "[dexteleop-env] Updating and pruning $ENV_NAME from environment.yml"
  conda env update --name "$ENV_NAME" --file "$REPO_ROOT/environment.yml" --prune
else
  echo "[dexteleop-env] Creating $ENV_NAME from environment.yml"
  conda env create --name "$ENV_NAME" --file "$REPO_ROOT/environment.yml"
fi

PIP_LOCK_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$PIP_LOCK_REQUIREMENTS"' EXIT

echo "[dexteleop-env] Checking uv.lock and installing all locked PyPI extras"
conda run --no-capture-output --name "$ENV_NAME" uv lock --check
conda run --no-capture-output --name "$ENV_NAME" \
  uv export --frozen --all-extras --format requirements.txt --no-emit-project \
  >"$PIP_LOCK_REQUIREMENTS"
PYTHONNOUSERSITE=1 conda run --no-capture-output --name "$ENV_NAME" \
  python -m pip install --no-cache-dir --require-hashes --requirement "$PIP_LOCK_REQUIREMENTS"

echo "[dexteleop-env] Installing the repository in editable mode"
PYTHONNOUSERSITE=1 conda run --no-capture-output --name "$ENV_NAME" \
  python -m pip install --no-cache-dir --no-deps --editable "$REPO_ROOT"

echo "[dexteleop-env] Installing pin-pink without its obsolete pip 'pin' dependency"
PYTHONNOUSERSITE=1 conda run --no-capture-output --name "$ENV_NAME" \
  python -m pip install --no-cache-dir --no-deps "pin-pink==4.2.0"

if ((SKIP_VENDOR_SDK == 0)); then
  echo "[dexteleop-env] Installing local DM-Tac SDK from $SDK_ROOT"
  PYTHONNOUSERSITE=1 conda run --no-capture-output --name "$ENV_NAME" \
    python -m pip install --no-cache-dir --no-deps --editable "$SDK_ROOT"
fi

echo "[dexteleop-env] Running import smoke test"
PYTHONNOUSERSITE=1 conda run --no-capture-output --name "$ENV_NAME" python - <<'PY'
import importlib.util
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected Python 3.10, got {sys.version}")

required = (
    "numpy",
    "scipy",
    "pinocchio",
    "pink",
    "qpsolvers",
    "mujoco",
    "gymnasium",
    "torch",
    "pyarrow",
    "zstandard",
    "websockets",
    "msgpack",
    "zmq",
    "serial",
    "cv2",
    "pyrealsense2",
)
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing imports: " + ", ".join(missing))

print("DexTeleop full environment imports ok")
print("python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "wheel CUDA", torch.version.cuda, "CUDA available", torch.cuda.is_available())
PY

if ((SKIP_VENDOR_SDK == 0)); then
  PYTHONNOUSERSITE=1 conda run --no-capture-output --name "$ENV_NAME" python - <<'PY'
import importlib.util

if importlib.util.find_spec("dmrobotics") is None:
    raise SystemExit("dmrobotics SDK was not importable after installation")
print("dmrobotics SDK import path ok")
PY
fi

echo
echo "[dexteleop-env] Installation complete. Activate it with:"
echo "  conda activate $ENV_NAME"
echo "[dexteleop-env] Note: pip check reports pin-pink's intentional missing 'pin' metadata dependency."
