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
ALLOW_CONTROL_RISK=false
DRY_CONTROL_OK=false
REQUIRE_SAFETY_OK=""
SAFETY_MAX_AGE=60
MODULE_NAME=""
MODULE_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  remote_module.sh -H dex-remote --project '~/DexTeleop' --env dexteleop310 -- sim.mujoco.replay.runner [args...]

Options:
      --project DIR       Remote project directory to cd into before running.
      --env NAME_OR_DIR   Conda/micromamba env name, or venv directory.
      --python PYTHON     Python executable after env activation. Default: python.
      --timeout SEC       Wrap the remote program with timeout when available.
      --log-dir DIR       Remote directory for tee'd logs.
      --dry-run           Print the remote command without executing it.
      --allow-control-risk
                          Allow hardware/control-looking commands, but only with
                          --require-safety-ok.
      --dry-control-ok    Allow hardware/control-looking commands only when
                          explicit dry-run/no-motion markers are present.
      --require-safety-ok PATH
                          Require a fresh remote_safety_check token before running.
      --safety-max-age SEC
                          Max age for safety token. Default: 60.
  --                     Module name and module arguments.

Safety: L3, remote process execution.
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
    --allow-control-risk)
      ALLOW_CONTROL_RISK=true
      shift
      ;;
    --dry-control-ok)
      DRY_CONTROL_OK=true
      shift
      ;;
    --require-safety-ok)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      REQUIRE_SAFETY_OK="$2"
      shift 2
      ;;
    --safety-max-age)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      SAFETY_MAX_AGE="$2"
      shift 2
      ;;
    --)
      shift
      [[ $# -ge 1 ]] || die "missing module name after --"
      MODULE_NAME="$1"
      shift
      MODULE_ARGS=("$@")
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

[[ -n "$MODULE_NAME" ]] || die "missing module name after --"
if [[ -n "$TIMEOUT_SEC" && ! "$TIMEOUT_SEC" =~ ^[0-9]+$ ]]; then
  die "timeout must be a number: $TIMEOUT_SEC"
fi
[[ "$SAFETY_MAX_AGE" =~ ^[0-9]+$ ]] || die "safety max age must be a number: $SAFETY_MAX_AGE"
assert_python_bin_name "$PYTHON_BIN"
if [[ "$DRY_RUN" != "true" ]]; then
  assert_control_risk_allowed "$ALLOW_CONTROL_RISK" "$DRY_CONTROL_OK" "$REQUIRE_SAFETY_OK" "$MODULE_NAME" "${MODULE_ARGS[@]}"
fi

finalize_common_ssh
print_wrapper_header "L3" "$PROJECT" "$ENV_NAME"

remote_bash "$PROJECT" "$ENV_NAME" "$PYTHON_BIN" "$TIMEOUT_SEC" "$LOG_DIR" "$DRY_RUN" "$REQUIRE_SAFETY_OK" "$SAFETY_MAX_AGE" "$MODULE_NAME" "${MODULE_ARGS[@]}" <<'REMOTE'
set -Eeuo pipefail

project=$1
env_name=$2
python_bin=$3
timeout_sec=$4
log_dir=$5
dry_run=$6
safety_token=$7
safety_max_age=$8
module_name=$9
shift 9

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

check_safety_token() {
  local token_path="$1"
  local max_age="$2"
  local now created age

  [[ -n "$token_path" ]] || return 0
  token_path="$(resolve_remote_path "$token_path")"
  case "$token_path" in
    /tmp/codex_*|"$HOME"/.cache/codex_remote_safety/*) ;;
    *) printf 'Refusing safety token path outside /tmp/codex_* or ~/.cache/codex_remote_safety/: %s\n' "$token_path" >&2; exit 5 ;;
  esac
  [[ -f "$token_path" ]] || { printf 'Safety token does not exist: %s\n' "$token_path" >&2; exit 5; }
  created="$(sed -n 's/^created_at=//p' "$token_path" | head -n 1)"
  [[ "$created" =~ ^[0-9]+$ ]] || { printf 'Safety token missing numeric created_at: %s\n' "$token_path" >&2; exit 5; }
  grep -Eq '^guard_checks=[1-9][0-9]*$' "$token_path" || { printf 'Safety token has no hardware/control guard checks: %s\n' "$token_path" >&2; exit 5; }
  now="$(date +%s)"
  age=$((now - created))
  if (( age < 0 || age > max_age )); then
    printf 'Safety token is stale: %ss old, max %ss (%s)\n' "$age" "$max_age" "$token_path" >&2
    exit 5
  fi
  printf '[check] safety_token=%s age=%ss\n' "$token_path" "$age"
}

if [[ -n "$project" ]]; then
  project="$(resolve_remote_path "$project")"
  [[ -d "$project" ]] || { printf 'Project directory does not exist: %s\n' "$project" >&2; exit 3; }
  cd "$project"
fi

activate_env "$env_name"
check_safety_token "$safety_token" "$safety_max_age"

cmd=("$python_bin" -u -m "$module_name" "$@")
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
  log_file="$log_dir/remote_module_$(date +%Y%m%d_%H%M%S).log"
  printf '[result] log=%s\n' "$log_file"
  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}
  set -e
  exit "$status"
fi

exec "${cmd[@]}"
REMOTE
