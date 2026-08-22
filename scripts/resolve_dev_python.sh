#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${DEXTELEOP_ENV_NAME:-dexteleop310}"

die() {
  echo "[dexteleop-python] ERROR: $*" >&2
  exit 1
}

if [[ -n "${DEXTELEOP_PYTHON:-}" ]]; then
  PYTHON_BIN="$DEXTELEOP_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && "$(basename -- "$CONDA_PREFIX")" == "$ENV_NAME" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  PYTHON_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"
else
  die "activate '$ENV_NAME' or set DEXTELEOP_PYTHON to its Python executable"
fi

[[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 10)' \
  || die "'$PYTHON_BIN' is not the supported DexTeleop Python 3.10 interpreter"
printf '%s\n' "$PYTHON_BIN"
