#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="$REPO_ROOT/model/perception/graspgenx"
ASSET_ROOT="${DEXTELEOP_ASSET_ROOT:-$REPO_ROOT/assets}"
PYTHON_BIN="python3.10"
TORCH_BACKEND="auto"
VERIFY_ONLY=0

SOURCE_REVISION="b9429097728cb1c430dd78b92edf17ba318aad03"
CHECKPOINT_REVISION="7c834043c11a11417e31d6d5ea9355801e40a2c1"
CHECKPOINT_URL="https://huggingface.co/adithyamurali/GraspGenXModel"

usage() {
  cat <<'EOF'
Usage: bash scripts/install_graspgenx_env.sh [OPTIONS]

Install the isolated GraspGenX inference service and its reviewed weights.
This script never modifies dexteleop310.

Options:
  --asset-root PATH  Shared AgenticRobot asset root (default: assets/ or
                     DEXTELEOP_ASSET_ROOT).
  --python PATH      Python 3.10/3.11 interpreter used by uv (default: python3.10).
  --torch-backend B  uv PyTorch backend, for example auto, cpu, or cu126
                     (default: auto).
  --verify-only      Verify the existing source, environment, weights, and manifest.
  -h, --help         Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --asset-root)
      ASSET_ROOT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --torch-backend)
      TORCH_BACKEND="$2"
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
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v git >/dev/null || {
  echo "Required command is unavailable: git" >&2
  exit 1
}
if ((VERIFY_ONLY == 0)); then
  command -v uv >/dev/null || {
    echo "Required command is unavailable: uv" >&2
    exit 1
  }
  git lfs version >/dev/null
fi

BUNDLE_ROOT="$ASSET_ROOT/perception/graspgenx-point-grasp"
CHECKPOINT_ROOT="$BUNDLE_ROOT/checkpoints"
GRIPPER_CFG_ROOT="$BUNDLE_ROOT/gripper_descriptions"
VENV_ROOT="$BUNDLE_ROOT/runtime/.venv"
MANIFEST_PATH="$BUNDLE_ROOT/manifest.json"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$BUNDLE_ROOT/runtime/uv-cache}"

if [[ ! -d "$SOURCE_ROOT/.git" && ! -f "$SOURCE_ROOT/.git" ]]; then
  if ((VERIFY_ONLY)); then
    echo "GraspGenX source submodule is missing: $SOURCE_ROOT" >&2
    exit 1
  fi
  GIT_LFS_SKIP_SMUDGE=1 git -C "$REPO_ROOT" submodule update --init --recursive -- \
    model/perception/graspgenx
fi

observed_source_revision="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
if [[ "$observed_source_revision" != "$SOURCE_REVISION" ]]; then
  echo "GraspGenX source revision mismatch: $observed_source_revision" >&2
  exit 1
fi

if ((VERIFY_ONLY == 0)); then
  mkdir -p "$BUNDLE_ROOT" "$GRIPPER_CFG_ROOT" "$(dirname "$VENV_ROOT")"
  if [[ -e "$CHECKPOINT_ROOT" && ! -d "$CHECKPOINT_ROOT/.git" ]]; then
    echo "Refusing non-Git checkpoint directory: $CHECKPOINT_ROOT" >&2
    exit 1
  fi
  if [[ ! -d "$CHECKPOINT_ROOT/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone "$CHECKPOINT_URL" "$CHECKPOINT_ROOT"
  fi
  if [[ -n "$(git -C "$CHECKPOINT_ROOT" status --porcelain)" ]]; then
    echo "Refusing to modify a dirty GraspGenX checkpoint checkout." >&2
    exit 1
  fi
  observed_checkpoint_revision="$(git -C "$CHECKPOINT_ROOT" rev-parse HEAD)"
  if [[ "$observed_checkpoint_revision" != "$CHECKPOINT_REVISION" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git -C "$CHECKPOINT_ROOT" fetch origin \
      "$CHECKPOINT_REVISION" --depth 1
    GIT_LFS_SKIP_SMUDGE=1 git -C "$CHECKPOINT_ROOT" checkout --detach \
      "$CHECKPOINT_REVISION"
  fi
  git -C "$CHECKPOINT_ROOT" lfs pull --include='release/gen/config.yaml,release/gen/epoch_736.pth,release/dis/config.yaml,release/dis/epoch_1056.pth' --exclude=''

  if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
    uv venv --python "$PYTHON_BIN" "$VENV_ROOT"
  fi

  # The reviewed upstream revision cannot be installed from its full dependency
  # metadata: it pins PyOpenGL 3.1.5 while pyrender pins PyOpenGL 3.1.0. Neither
  # renderer is imported by the params-only ZMQ inference path. Install that
  # path's explicit closure. The launcher imports the pinned source submodule
  # through PYTHONPATH, avoiding both the conflict and build artifacts in that
  # read-only source checkout.
  uv pip install \
    --python "$VENV_ROOT/bin/python" \
    --torch-backend "$TORCH_BACKEND" \
    'torch>=2.1,<2.7' \
    'h5py' \
    'hydra-core' \
    'numpy==1.26.4' \
    'webdataset' \
    'scikit-learn' \
    'scipy' \
    'trimesh==4.5.3' \
    'diffusers==0.11.1' \
    'timm==1.0.15' \
    'huggingface-hub==0.25.2' \
    'addict' \
    'imageio' \
    'tqdm' \
    'pyyaml' \
    'yourdfpy==0.0.56' \
    'pyzmq' \
    'msgpack' \
    'msgpack-numpy' \
    'setuptools>=45,<78'

  "$VENV_ROOT/bin/python" - "$BUNDLE_ROOT" "$SOURCE_REVISION" \
    "$CHECKPOINT_REVISION" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
source_revision, checkpoint_revision = sys.argv[2:]
relative_paths = {
    "generator_config": "checkpoints/release/gen/config.yaml",
    "generator_checkpoint": "checkpoints/release/gen/epoch_736.pth",
    "discriminator_config": "checkpoints/release/dis/config.yaml",
    "discriminator_checkpoint": "checkpoints/release/dis/epoch_1056.pth",
}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

files = {
    name: {"relative_path": relative, "sha256": sha256_file(root / relative)}
    for name, relative in relative_paths.items()
}
manifest = {
    "schema_version": "dexteleop.perception_graspgenx_point_grasp_assets.v1",
    "model_profile_id": "graspgenx-release-dmtac-m-sweep-volume-v1",
    "source": {
        "git_url": "https://github.com/NVlabs/GraspGenX.git",
        "revision": source_revision,
        "license": "Apache-2.0",
    },
    "checkpoint_repository": {
        "git_url": "https://huggingface.co/adithyamurali/GraspGenXModel",
        "revision": checkpoint_revision,
        "license": "NVIDIA-Open-Model-License",
    },
    "files": files,
}
temporary = root / ".manifest.json.tmp"
temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, root / "manifest.json")
PY
fi

for path in \
  "$CHECKPOINT_ROOT/release/gen/config.yaml" \
  "$CHECKPOINT_ROOT/release/gen/epoch_736.pth" \
  "$CHECKPOINT_ROOT/release/dis/config.yaml" \
  "$CHECKPOINT_ROOT/release/dis/epoch_1056.pth" \
  "$MANIFEST_PATH" \
  "$VENV_ROOT/bin/python"; do
  [[ -f "$path" ]] || {
    echo "Required GraspGenX asset is missing: $path" >&2
    exit 1
  }
done

observed_checkpoint_revision="$(git -C "$CHECKPOINT_ROOT" rev-parse HEAD)"
if [[ "$observed_checkpoint_revision" != "$CHECKPOINT_REVISION" ]]; then
  echo "GraspGenX checkpoint revision mismatch: $observed_checkpoint_revision" >&2
  exit 1
fi
if [[ -n "$(git -C "$CHECKPOINT_ROOT" status --porcelain)" ]]; then
  echo "GraspGenX checkpoint checkout is dirty." >&2
  exit 1
fi

GRASPGENX_CHECKPOINT_DIR="$CHECKPOINT_ROOT" \
GRASPGENX_GRIPPER_CFG_DIR="$GRIPPER_CFG_ROOT" \
PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$VENV_ROOT/bin/python" -c \
  'from graspgenx.serving.zmq_server import GraspGenXZMQServer; print("GraspGenX service environment ok")'

echo "GraspGenX source: $SOURCE_ROOT @ $SOURCE_REVISION"
echo "GraspGenX assets: $BUNDLE_ROOT @ $CHECKPOINT_REVISION"
