#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${DEXTELEOP_ENV_NAME:-dexteleop310}"

die() {
  echo "[dexteleop-pytest] ERROR: $*" >&2
  exit 1
}

for argument in "$@"; do
  case "$argument" in
    --basetemp|--basetemp=*)
      die "the wrapper owns --basetemp so AF_UNIX paths stay short"
      ;;
  esac
done

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
"$PYTHON_BIN" -c \
  'import sys; assert sys.version_info[:2] == (3, 10); import osqp, pytest' \
  || die "'$PYTHON_BIN' is not the complete DexTeleop Python 3.10 test environment"

BASE_TEMP="$(mktemp -d "/tmp/dt-pytest-${UID:-user}-XXXXXX")"
case "$BASE_TEMP" in
  /tmp/dt-pytest-*) ;;
  *) die "mktemp returned an unexpected path: $BASE_TEMP" ;;
esac
if ((${#BASE_TEMP} > 48)); then
  die "pytest temporary root is too long for bounded AF_UNIX test endpoints"
fi

ASSET_COMPAT_LINK="$REPO_ROOT/assets"
ASSET_COMPAT_ROOT=""
ASSET_COMPAT_LINK_CREATED=0

prepare_asset_compat_link() {
  local requested_root="${DEXTELEOP_ASSET_ROOT:-}"
  local existing_root

  if [[ -z "$requested_root" ]]; then
    return
  fi
  case "$requested_root" in
    /*) ;;
    *) die "DEXTELEOP_ASSET_ROOT must be an absolute directory: $requested_root" ;;
  esac
  [[ -d "$requested_root" ]] || die \
    "DEXTELEOP_ASSET_ROOT is not a directory: $requested_root"
  ASSET_COMPAT_ROOT="$(cd -- "$requested_root" && pwd -P)"

  if [[ -e "$ASSET_COMPAT_LINK" || -L "$ASSET_COMPAT_LINK" ]]; then
    existing_root="$(readlink -f -- "$ASSET_COMPAT_LINK")" || die \
      "cannot resolve existing assets path: $ASSET_COMPAT_LINK"
    [[ "$existing_root" == "$ASSET_COMPAT_ROOT" ]] || die \
      "existing assets path does not match DEXTELEOP_ASSET_ROOT: $existing_root != $ASSET_COMPAT_ROOT"
    return
  fi

  ln -s -- "$ASSET_COMPAT_ROOT" "$ASSET_COMPAT_LINK"
  ASSET_COMPAT_LINK_CREATED=1
}

cleanup() {
  local status=$?
  local current_root

  trap - EXIT
  if ((ASSET_COMPAT_LINK_CREATED)); then
    if [[ -L "$ASSET_COMPAT_LINK" ]]; then
      current_root="$(readlink -f -- "$ASSET_COMPAT_LINK")" || current_root=""
      if [[ "$current_root" == "$ASSET_COMPAT_ROOT" ]]; then
        unlink -- "$ASSET_COMPAT_LINK" || status=1
      else
        echo "[dexteleop-pytest] ERROR: refusing to remove changed assets link: $ASSET_COMPAT_LINK" >&2
        status=1
      fi
    else
      echo "[dexteleop-pytest] ERROR: temporary assets link was replaced: $ASSET_COMPAT_LINK" >&2
      status=1
    fi
  fi
  rm -rf -- "$BASE_TEMP" || status=1
  exit "$status"
}
trap cleanup EXIT

prepare_asset_compat_link

cd "$REPO_ROOT"
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
"$PYTHON_BIN" -m pytest --basetemp="$BASE_TEMP" "$@"
