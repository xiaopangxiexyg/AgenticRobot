#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

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

PYTHON_BIN="$(bash "$REPO_ROOT/scripts/resolve_dev_python.sh")"
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

cleanup() {
  rm -rf -- "$BASE_TEMP"
}
trap cleanup EXIT

cd "$REPO_ROOT"
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
"$PYTHON_BIN" -m pytest --basetemp="$BASE_TEMP" "$@"
