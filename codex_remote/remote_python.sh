#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

PROJECT=""
ENV_NAME=""
PYTHON_BIN="python"
TIMEOUT_SEC=""
LOG_DIR=""
DRY_RUN=false
REMOTE_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  remote_python.sh -H dex-remote --project '~/DexTeleop' --env dexteleop310 -- script.py [args...]
  remote_python.sh -H dex-remote --python /path/to/python -- --version

Options:
      --project DIR       Remote project directory to cd into before running.
      --env NAME_OR_DIR   Conda/micromamba env name, or venv directory.
      --python PYTHON     Python executable after env activation. Default: python.
      --timeout SEC       Wrap the remote program with timeout when available.
      --log-dir DIR       Remote directory for tee'd logs.
      --dry-run           Print the remote command without executing it.
  --                     Everything after -- is passed to python -u.

Safety: L3, non-control remote process execution. Control/device actions require
a purpose-built wrapper.
EOF
  common_ssh_usage
}

while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then
    shift "$COMMON_ARGC"
    continue
  fi

  case "$1" in
    --project)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      PROJECT="$2"
      shift 2
      ;;
    --env)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --timeout)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --log-dir)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      LOG_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --allow-control-risk|--dry-control-ok|--require-safety-ok|--safety-max-age)
      die "$1 is no longer supported by generic L3 wrappers; use a purpose-built wrapper for control/device actions"
      ;;
    --)
      shift
      REMOTE_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument before --: $1"
      ;;
  esac
done

[[ ${#REMOTE_ARGS[@]} -gt 0 ]] || die "missing remote python arguments after --"
if [[ -n "$TIMEOUT_SEC" && ! "$TIMEOUT_SEC" =~ ^[0-9]+$ ]]; then
  die "timeout must be a number: $TIMEOUT_SEC"
fi
assert_python_bin_name "$PYTHON_BIN"
if [[ "$DRY_RUN" != "true" ]]; then
  assert_no_control_risk "${REMOTE_ARGS[@]}"
fi

finalize_common_ssh
print_wrapper_header "L3" "$PROJECT" "$ENV_NAME"

remote_bash "$PROJECT" "$ENV_NAME" "$PYTHON_BIN" "$TIMEOUT_SEC" "$LOG_DIR" "$DRY_RUN" "${REMOTE_ARGS[@]}" <<'REMOTE'
set -Eeuo pipefail

project=$1
env_name=$2
python_bin=$3
timeout_sec=$4
log_dir=$5
dry_run=$6
shift 6

resolve_remote_path() {
  local value="$1"
  case "$value" in
    "") return 1 ;;
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${value#"~/"}" ;;
    /*) printf '%s\n' "$value" ;;
    *) printf '%s/%s\n' "$HOME" "$value" ;;
  esac
}

activate_env() {
  local env_name="$1"
  local conda_sh
  [[ -n "$env_name" ]] || return 0

  if [[ -d "$env_name" && -f "$env_name/bin/activate" ]]; then
    source "$env_name/bin/activate"
    return 0
  fi

  for conda_sh in "$HOME/anaconda3/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/mambaforge/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      source "$conda_sh"
      conda activate "$env_name"
      return 0
    fi
  done

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$env_name"
    return 0
  fi
  if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "$env_name"
    return 0
  fi

  printf 'Cannot activate env: %s\n' "$env_name" >&2
  return 3
}

if [[ -n "$project" ]]; then
  project="$(resolve_remote_path "$project")"
  [[ -d "$project" ]] || { printf 'Project directory does not exist: %s\n' "$project" >&2; exit 3; }
  cd "$project"
fi

activate_env "$env_name"

cmd=("$python_bin" -u "$@")
if [[ -n "$timeout_sec" ]]; then
  command -v timeout >/dev/null 2>&1 || { printf 'timeout requested but timeout command is unavailable\n' >&2; exit 3; }
  cmd=(timeout "$timeout_sec" "${cmd[@]}")
fi

printf '[run]'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$dry_run" == "true" ]]; then
  printf '[result] dry-run only\n'
  exit 0
fi

if [[ -n "$log_dir" ]]; then
  log_dir="$(resolve_remote_path "$log_dir")"
  mkdir -p "$log_dir"
  log_file="$log_dir/remote_python_$(date +%Y%m%d_%H%M%S).log"
  printf '[result] log=%s\n' "$log_file"
  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}
  set -e
  exit "$status"
fi

exec "${cmd[@]}"
REMOTE
