#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${DEXTELEOP_ANNOTATION_VENV:-$REPO_ROOT/.venv-annotation}"
PYTHON_REQUEST="${DEXTELEOP_ANNOTATION_PYTHON:-3.10}"

usage() {
  cat <<'EOF'
Usage: bash scripts/install_annotation_env.sh [options]

Create or synchronize the minimal DexTeleop post-capture annotation
environment. This environment does not install robot, hardware, simulation,
model-training, or policy-deployment dependencies.

Options:
  --venv-path PATH       Virtual environment path (default: .venv-annotation).
  --python PYTHON        Python 3.10 interpreter or uv request (default: 3.10).
  -h, --help             Show this help.

The installer requires uv on PATH and uses the repository's frozen uv.lock.
Re-running the command safely synchronizes the existing environment.
EOF
}

die() {
  echo "[dexteleop-annotation-env] ERROR: $*" >&2
  exit 1
}

while (($# > 0)); do
  case "$1" in
    --venv-path)
      (($# >= 2)) || die "--venv-path requires a path"
      VENV_PATH="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires an interpreter or version"
      PYTHON_REQUEST="$2"
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

command -v uv >/dev/null 2>&1 || die \
  "uv is not on PATH; install uv first (https://docs.astral.sh/uv/)"
command -v ffmpeg >/dev/null 2>&1 || die \
  "ffmpeg is required to play legacy mp4v videos in Chromium"
command -v ffprobe >/dev/null 2>&1 || die \
  "ffprobe is required to inspect video codecs before annotation"

if [[ "$VENV_PATH" != /* ]]; then
  VENV_PATH="$REPO_ROOT/$VENV_PATH"
fi

cd "$REPO_ROOT"
if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  if [[ -e "$VENV_PATH" ]]; then
    die "target exists but is not a Python virtual environment: $VENV_PATH"
  fi
  echo "[dexteleop-annotation-env] Creating $VENV_PATH"
  uv venv --python "$PYTHON_REQUEST" "$VENV_PATH"
fi

VENV_PYTHON="$VENV_PATH/bin/python"
"$VENV_PYTHON" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        "annotation environment requires Python 3.10; got "
        + sys.version.split()[0]
    )
PY

echo "[dexteleop-annotation-env] Synchronizing frozen annotation dependencies"
VIRTUAL_ENV="$VENV_PATH" uv sync \
  --frozen \
  --extra annotation \
  --no-dev \
  --active

echo "[dexteleop-annotation-env] Running import and entrypoint smoke checks"
PYTHONNOUSERSITE=1 "$VENV_PYTHON" - <<'PY'
import importlib.util
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected Python 3.10, got {sys.version}")

required = (
    "numpy",
    "pyarrow",
    "collection_runtime",
    "control_station",
    "dataset_contracts",
    "dataset_interface",
)
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing annotation imports: " + ", ".join(missing))

from control_station.refinement_annotation.cli import build_parser

build_parser()
print("DexTeleop annotation environment imports ok")
print("python", sys.version.split()[0])
PY
PYTHONNOUSERSITE=1 "$VENV_PYTHON" \
  -m control_station refinement-annotation --help >/dev/null

echo
echo "[dexteleop-annotation-env] Installation complete. Activate it with:"
echo "  source \"$VENV_PATH/bin/activate\""
echo "Then start the GUI with:"
echo "  python -m control_station refinement-annotation"
