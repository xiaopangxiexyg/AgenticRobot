#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

PROJECT=""
ENV_NAME=""
PYTHON_BIN="python"
TIMEOUT_SEC="900"
DATASET_ROOT=""
PREPARED_VIEW_ROOT=""
MODE="dry-run"
MODE_EXPLICIT=false
EXPECTED_PLAN_HASH=""
CONFIRM_NO_ACTIVE_READERS=false

usage() {
  cat <<'EOF'
Usage:
  remote_dataset_camera_training_validity.sh -H user@host \
    --project /path/DexTeleop --env /path/env \
    --dataset-root /data/source_dataset \
    --prepared-view-root /data/camera_semantic_prepared_views/source_dataset \
    --dry-run

Apply requires:
  ... --apply --expected-plan-hash SHA256 --confirm-no-active-readers

Safety: fixed L3 non-motion metadata transaction. The command only atomically
updates meta/modality_validity.json from a fully verified, hash-bound camera
prepared-view corruption receipt. It exposes no deletion, payload rewrite,
device, control, or arbitrary module passthrough.
EOF
  common_ssh_usage
}

path_arg() {
  local label="$1"
  local value="$2"
  [[ -n "$value" ]] || die "missing $label"
  case "$value" in
    /*|"~/"*) ;;
    *) die "$label must be an absolute or ~/ remote path" ;;
  esac
  [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || \
    die "$label contains control characters"
}

while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then
    shift "$COMMON_ARGC"
    continue
  fi
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --timeout) TIMEOUT_SEC="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --prepared-view-root) PREPARED_VIEW_ROOT="$2"; shift 2 ;;
    --expected-plan-hash) EXPECTED_PLAN_HASH="$2"; shift 2 ;;
    --confirm-no-active-readers) CONFIRM_NO_ACTIVE_READERS=true; shift ;;
    --dry-run)
      [[ "$MODE_EXPLICIT" != "true" ]] || die "choose exactly one of --dry-run or --apply"
      MODE="dry-run"; MODE_EXPLICIT=true; shift
      ;;
    --apply)
      [[ "$MODE_EXPLICIT" != "true" ]] || die "choose exactly one of --dry-run or --apply"
      MODE="apply"; MODE_EXPLICIT=true; shift
      ;;
    -h|--help) usage; exit 0 ;;
    --) die "argument passthrough is not supported" ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$REMOTE_HOST" ]] || die "missing -H/--host"
path_arg "--project" "$PROJECT"
path_arg "--dataset-root" "$DATASET_ROOT"
path_arg "--prepared-view-root" "$PREPARED_VIEW_ROOT"
assert_python_bin_name "$PYTHON_BIN"
[[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || die "timeout must be numeric"
(( TIMEOUT_SEC >= 1 && TIMEOUT_SEC <= 86400 )) || die "timeout must be in 1..86400"

if [[ "$MODE" == "apply" ]]; then
  [[ "$EXPECTED_PLAN_HASH" =~ ^[0-9a-f]{64}$ ]] || \
    die "--apply requires a lowercase 64-hex --expected-plan-hash"
  [[ "$CONFIRM_NO_ACTIVE_READERS" == "true" ]] || \
    die "--apply requires --confirm-no-active-readers"
else
  [[ -z "$EXPECTED_PLAN_HASH" ]] || \
    die "--expected-plan-hash is only valid with --apply"
  [[ "$CONFIRM_NO_ACTIVE_READERS" == "false" ]] || \
    die "--confirm-no-active-readers is only valid with --apply"
fi

finalize_common_ssh
print_wrapper_header "L3-dataset-non-motion" "$PROJECT" "$ENV_NAME"
printf '[operation] camera-video-training-validity mode=%s timeout_sec=%s\n' \
  "$MODE" "$TIMEOUT_SEC"

remote_bash \
  "$PROJECT" "$ENV_NAME" "$PYTHON_BIN" "$TIMEOUT_SEC" "$MODE" \
  "$DATASET_ROOT" "$PREPARED_VIEW_ROOT" "$EXPECTED_PLAN_HASH" <<'REMOTE'
set -Eeuo pipefail
project=$1
env_name=$2
python_bin=$3
timeout_sec=$4
mode=$5
dataset_root=$6
prepared_view_root=$7
expected_plan_hash=$8

resolve_path() {
  case "$1" in
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${1#"~/"}" ;;
    /*) printf '%s\n' "$1" ;;
    *) return 1 ;;
  esac
}

activate_env() {
  local requested=$1
  local conda_sh
  [[ -n "$requested" ]] || return 0
  if [[ -f "$requested/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$requested/bin/activate"
    return 0
  fi
  for conda_sh in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_sh"
      conda activate "$requested"
      return 0
    fi
  done
  printf 'Cannot activate env: %s\n' "$requested" >&2
  return 3
}

project="$(resolve_path "$project")"
dataset_root="$(resolve_path "$dataset_root")"
prepared_view_root="$(resolve_path "$prepared_view_root")"
[[ -d "$project" ]] || { printf 'Project missing: %s\n' "$project" >&2; exit 3; }
cd "$project"
activate_env "$env_name"

cmd=(
  "$python_bin" -u -m utils.dataset_postprocess
  camera-video-integrity-mark-training-invalid "$dataset_root"
  --prepared-view-root "$prepared_view_root"
)
if [[ "$mode" == "apply" ]]; then
  cmd+=(--apply --expected-plan-hash "$expected_plan_hash")
fi
cmd=(timeout "$timeout_sec" "${cmd[@]}")
printf '[run]'; printf ' %q' "${cmd[@]}"; printf '\n'
exec "${cmd[@]}"
REMOTE
