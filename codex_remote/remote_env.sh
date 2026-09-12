#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

PROJECT=""
ENV_NAME=""
PYTHON_BIN="python"
SUMMARY=false

usage() {
  cat <<'EOF'
Usage:
  remote_env.sh -H dex-remote --project '~/DexTeleop' --env dexteleop310
  remote_env.sh -H dex-remote --python /path/to/python

Options:
      --project DIR       Remote project directory to cd into before checks.
      --env NAME_OR_DIR   Conda/micromamba env name, or venv directory.
      --python PYTHON     Python executable after env activation. Default: python.
      --summary           Check only host, cwd and Python; omit environment inventories.

Safety: L1, remote read-only.
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
    --summary)
      SUMMARY=true
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

finalize_common_ssh
assert_python_bin_name "$PYTHON_BIN"
print_wrapper_header "L1" "$PROJECT" "$ENV_NAME"

remote_bash "$PROJECT" "$ENV_NAME" "$PYTHON_BIN" "$SUMMARY" <<'REMOTE'
set -Eeuo pipefail

project=$1
env_name=$2
python_bin=$3
summary=$4

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
    # shellcheck disable=SC1090
    source "$env_name/bin/activate"
    return 0
  fi

  for conda_sh in \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1090
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

printf '[remote] hostname=%s\n' "$(hostname)"
printf '[remote] pwd=%s\n' "$PWD"
if [[ "$summary" != "true" ]]; then
  printf '[remote] shell=%s\n' "${SHELL:-unknown}"
  printf '[remote] path=%s\n' "$PATH"
fi
printf '[python] executable=%s\n' "$(command -v "$python_bin" 2>/dev/null || printf '%s' "$python_bin")"
"$python_bin" --version 2>&1 | sed 's/^/[python] version=/'

if [[ "$summary" == "true" ]]; then
  printf '[scope] summary: shell/PATH omitted; conda inventory, GPU and CUDA not checked\n'
  exit 0
fi

if command -v conda >/dev/null 2>&1; then
  conda info --envs 2>/dev/null | sed 's/^/[conda] /' || true
else
  printf '[conda] command not found\n'
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/[gpu] /' || true
else
  printf '[gpu] nvidia-smi not found\n'
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version 2>/dev/null | tail -n 1 | sed 's/^/[cuda] /' || true
else
  printf '[cuda] nvcc not found\n'
fi
REMOTE
