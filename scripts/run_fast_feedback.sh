#!/usr/bin/env bash
set -euo pipefail

# Run the two bounded edit-feedback tiers concurrently.  Each child still
# goes through run_pytest.sh, which owns the Python selection and basetemp.
# This wrapper only owns the concurrency, per-child pytest cache, report
# paths, and the temporary shared assets link needed by a clean worktree.
# DEXTELEOP_FAST_FEEDBACK_RUN_ROOT may name a new absolute directory when a
# caller such as Harness needs the logs and JUnit reports under its Run root.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

die() {
  echo "[dexteleop-fast-feedback] ERROR: $*" >&2
  exit 2
}

for argument in "$@"; do
  case "$argument" in
    --basetemp|--basetemp=*|-m|--junitxml|--junitxml=*|-o|--override-ini|--override-ini=*)
      die "$argument is owned by this wrapper; pass only shared pytest options such as --durations"
      ;;
  esac
done

RUN_ROOT=""
ASSET_LINK="$REPO_ROOT/assets"
ASSET_ROOT=""
ASSET_LINK_CREATED=0
UNIT_PID=""
INTEGRATION_PID=""

prepare_run_root() {
  local requested_root="${DEXTELEOP_FAST_FEEDBACK_RUN_ROOT:-}"
  local parent

  if [[ -z "$requested_root" ]]; then
    RUN_ROOT="$(mktemp -d "/tmp/dt-fast-feedback-${UID:-user}-XXXXXX")"
    return
  fi
  case "$requested_root" in
    /*) ;;
    *) die "DEXTELEOP_FAST_FEEDBACK_RUN_ROOT must be an absolute path: $requested_root" ;;
  esac
  if [[ -e "$requested_root" || -L "$requested_root" ]]; then
    die "DEXTELEOP_FAST_FEEDBACK_RUN_ROOT must not already exist: $requested_root"
  fi
  parent="$(dirname -- "$requested_root")"
  [[ -d "$parent" ]] || die \
    "DEXTELEOP_FAST_FEEDBACK_RUN_ROOT parent is not a directory: $parent"
  mkdir -- "$requested_root" || die \
    "cannot create DEXTELEOP_FAST_FEEDBACK_RUN_ROOT: $requested_root"
  RUN_ROOT="$requested_root"
}

prepare_asset_compat_link() {
  local requested_root="${DEXTELEOP_ASSET_ROOT:-}"
  local existing_root

  if [[ -z "$requested_root" ]]; then
    [[ -e "$ASSET_LINK" ]] || die \
      "assets path is missing; set DEXTELEOP_ASSET_ROOT to an absolute shared asset root"
    return
  fi
  case "$requested_root" in
    /*) ;;
    *) die "DEXTELEOP_ASSET_ROOT must be an absolute directory: $requested_root" ;;
  esac
  [[ -d "$requested_root" ]] || die \
    "DEXTELEOP_ASSET_ROOT is not a directory: $requested_root"
  ASSET_ROOT="$(cd -- "$requested_root" && pwd -P)"

  if [[ -e "$ASSET_LINK" || -L "$ASSET_LINK" ]]; then
    existing_root="$(readlink -f -- "$ASSET_LINK")" || die \
      "cannot resolve existing assets path: $ASSET_LINK"
    [[ "$existing_root" == "$ASSET_ROOT" ]] || die \
      "existing assets path does not match DEXTELEOP_ASSET_ROOT: $existing_root != $ASSET_ROOT"
    return
  fi

  ln -s -- "$ASSET_ROOT" "$ASSET_LINK"
  ASSET_LINK_CREATED=1
}

stop_children() {
  local pid
  for pid in "$UNIT_PID" "$INTEGRATION_PID"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

cleanup() {
  local status=$?
  local current_root

  trap - EXIT INT TERM
  if [[ -n "$UNIT_PID" || -n "$INTEGRATION_PID" ]]; then
    stop_children
  fi
  if ((ASSET_LINK_CREATED)); then
    if [[ -L "$ASSET_LINK" ]]; then
      current_root="$(readlink -f -- "$ASSET_LINK")" || current_root=""
      if [[ "$current_root" == "$ASSET_ROOT" ]]; then
        unlink -- "$ASSET_LINK" || status=1
      else
        echo "[dexteleop-fast-feedback] ERROR: refusing to remove changed assets link: $ASSET_LINK" >&2
        status=1
      fi
    else
      echo "[dexteleop-fast-feedback] ERROR: temporary assets link was replaced: $ASSET_LINK" >&2
      status=1
    fi
  fi
  if [[ -n "$RUN_ROOT" ]]; then
    echo "[dexteleop-fast-feedback] artifacts: $RUN_ROOT"
  fi
  exit "$status"
}

on_signal() {
  stop_children
  wait "$UNIT_PID" 2>/dev/null || true
  wait "$INTEGRATION_PID" 2>/dev/null || true
  UNIT_PID=""
  INTEGRATION_PID=""
  exit 130
}

run_tier() {
  local name="$1"
  local marker="$2"
  shift 2
  PYTHONNOUSERSITE=1 \
  PYTHONPYCACHEPREFIX="$RUN_ROOT/${name}-python-bytecode" \
  SIM_MUJOCO_CACHE_DIR="$RUN_ROOT/${name}-mujoco-cache" \
  "$REPO_ROOT/scripts/run_pytest.sh" \
    -m "$marker" \
    -q \
    -o "cache_dir=$RUN_ROOT/${name}-pytest-cache" \
    --junitxml="$RUN_ROOT/${name}.xml" \
    "$@" \
    >"$RUN_ROOT/${name}.log" 2>&1
}

trap cleanup EXIT
prepare_run_root
prepare_asset_compat_link
trap on_signal INT TERM

run_tier unit 'unit and not optional and not hardware and not slow_sim' "$@" &
UNIT_PID=$!
run_tier integration 'integration and not optional and not hardware and not slow_sim' "$@" &
INTEGRATION_PID=$!

set +e
wait "$UNIT_PID"
UNIT_STATUS=$?
UNIT_PID=""
wait "$INTEGRATION_PID"
INTEGRATION_STATUS=$?
INTEGRATION_PID=""
set -e

echo "===== fast unit (status=$UNIT_STATUS) ====="
cat "$RUN_ROOT/unit.log"
echo "===== fast integration (status=$INTEGRATION_STATUS) ====="
cat "$RUN_ROOT/integration.log"
echo "[dexteleop-fast-feedback] unit JUnit: $RUN_ROOT/unit.xml"
echo "[dexteleop-fast-feedback] integration JUnit: $RUN_ROOT/integration.xml"

if ((UNIT_STATUS != 0 || INTEGRATION_STATUS != 0)); then
  exit 1
fi
