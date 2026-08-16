#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

PROJECT=""
ENV_NAME=""
PYTHON_BIN="python"
TIMEOUT_SEC="3600"
OPERATION=""
DATASET_ROOT=""
CORRECTED_ROOT=""
RETIREMENT_ROOT=""
REGISTRY_PATH=""
TRANSACTION_PATH=""
MODE="dry-run"
MODE_EXPLICIT=false
EXPECTED_PLAN_HASH=""
CONFIRM_NO_ACTIVE_READERS=false
ALLOW_LEGACY_FORMAT_PRESERVING=false

usage() {
  cat <<'EOF'
Usage:
  remote_dataset_camera_promotion.sh -H user@host \
    --project /path/DexTeleop --env /path/env --promote \
    --dataset-root /data/canonical_old \
    --corrected-root /data/corrected_full \
    --retirement-root /data/retired_swapped \
    --registry /path/camera_registry.json --dry-run

  # Explicit legacy format-preserving promotion (default is strict v2.3):
  remote_dataset_camera_promotion.sh -H user@host \
    --project /path/DexTeleop --env /path/env --promote \
    --dataset-root /data/canonical_old \
    --corrected-root /data/corrected_full \
    --retirement-root /data/retired_swapped \
    --registry /path/camera_registry.json \
    --allow-legacy-format-preserving --dry-run

  remote_dataset_camera_promotion.sh -H user@host \
    --project /path/DexTeleop --env /path/env --rollback \
    --transaction /data/retired_swapped/.transactions/PLAN.json --dry-run

Apply requires:
  ... --apply --expected-plan-hash SHA256 --confirm-no-active-readers

Safety: fixed L3 non-motion dataset transaction. Promotion is admitted only
for independent, self-contained roots where the old dataset is proven swapped
and the materialized dataset is identity. Strict v2.3 is the default. The
explicit legacy option requires an exact format-preserving camera transform and
does not claim or upgrade the dataset to v2.3. Promotion atomically retains the
old root under retirement; rollback is a separate hash-bound operation. No
deletion, overwrite, schema migration, device, or arbitrary passthrough is
exposed.
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

select_operation() {
  [[ -z "$OPERATION" ]] || die "choose exactly one of --promote or --rollback"
  OPERATION="$1"
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
    --promote) select_operation "promote"; shift ;;
    --rollback) select_operation "rollback"; shift ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --corrected-root) CORRECTED_ROOT="$2"; shift 2 ;;
    --retirement-root) RETIREMENT_ROOT="$2"; shift 2 ;;
    --registry) REGISTRY_PATH="$2"; shift 2 ;;
    --transaction) TRANSACTION_PATH="$2"; shift 2 ;;
    --expected-plan-hash) EXPECTED_PLAN_HASH="$2"; shift 2 ;;
    --confirm-no-active-readers) CONFIRM_NO_ACTIVE_READERS=true; shift ;;
    --allow-legacy-format-preserving) ALLOW_LEGACY_FORMAT_PRESERVING=true; shift ;;
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
[[ -n "$OPERATION" ]] || die "choose exactly one of --promote or --rollback"
assert_python_bin_name "$PYTHON_BIN"
[[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || die "timeout must be numeric"
(( TIMEOUT_SEC >= 1 && TIMEOUT_SEC <= 86400 )) || die "timeout must be in 1..86400"

if [[ "$OPERATION" == "promote" ]]; then
  path_arg "--dataset-root" "$DATASET_ROOT"
  path_arg "--corrected-root" "$CORRECTED_ROOT"
  path_arg "--retirement-root" "$RETIREMENT_ROOT"
  path_arg "--registry" "$REGISTRY_PATH"
  [[ -z "$TRANSACTION_PATH" ]] || die "--transaction is only valid with --rollback"
else
  path_arg "--transaction" "$TRANSACTION_PATH"
  [[ -z "$DATASET_ROOT$CORRECTED_ROOT$RETIREMENT_ROOT$REGISTRY_PATH" ]] || \
    die "dataset roots and registry are only valid with --promote"
  [[ "$ALLOW_LEGACY_FORMAT_PRESERVING" == "false" ]] || \
    die "--allow-legacy-format-preserving is only valid with --promote"
fi

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
printf '[operation] %s mode=%s timeout_sec=%s\n' "$OPERATION" "$MODE" "$TIMEOUT_SEC"

remote_bash \
  "$PROJECT" "$ENV_NAME" "$PYTHON_BIN" "$TIMEOUT_SEC" "$OPERATION" "$MODE" \
  "$DATASET_ROOT" "$CORRECTED_ROOT" "$RETIREMENT_ROOT" "$REGISTRY_PATH" \
  "$TRANSACTION_PATH" "$EXPECTED_PLAN_HASH" \
  "$ALLOW_LEGACY_FORMAT_PRESERVING" <<'REMOTE'
set -Eeuo pipefail
project=$1
env_name=$2
python_bin=$3
timeout_sec=$4
operation=$5
mode=$6
dataset_root=$7
corrected_root=$8
retirement_root=$9
registry_path=${10}
transaction_path=${11}
expected_plan_hash=${12}
allow_legacy_format_preserving=${13}

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
[[ -d "$project" ]] || { printf 'Project missing: %s\n' "$project" >&2; exit 3; }
cd "$project"
activate_env "$env_name"

if [[ "$operation" == "promote" ]]; then
  dataset_root="$(resolve_path "$dataset_root")"
  corrected_root="$(resolve_path "$corrected_root")"
  retirement_root="$(resolve_path "$retirement_root")"
  registry_path="$(resolve_path "$registry_path")"
  cmd=(
    "$python_bin" -u -m utils.dataset_postprocess
    camera-semantics-promote "$dataset_root"
    --corrected-root "$corrected_root"
    --retirement-root "$retirement_root"
    --registry "$registry_path"
  )
  if [[ "$allow_legacy_format_preserving" == "true" ]]; then
    cmd+=(--allow-legacy-format-preserving)
  fi
else
  transaction_path="$(resolve_path "$transaction_path")"
  cmd=(
    "$python_bin" -u -m utils.dataset_postprocess
    camera-semantics-rollback-promotion "$transaction_path"
  )
fi
if [[ "$mode" == "apply" ]]; then
  cmd+=(
    --apply
    --expected-plan-hash "$expected_plan_hash"
    --confirm-no-active-readers
  )
fi
cmd=(timeout "$timeout_sec" "${cmd[@]}")
printf '[run]'; printf ' %q' "${cmd[@]}"; printf '\n'
exec "${cmd[@]}"
REMOTE
