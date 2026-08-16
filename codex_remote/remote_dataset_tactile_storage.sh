#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

PROJECT=""
ENV_NAME=""
PYTHON_BIN="python"
TIMEOUT_SEC="86400"
DATASET_ROOT=""
COLD_ROOT=""
OPERATION=""
MODE="dry-run"
MODE_EXPLICIT=false
PLAN_HASH=""
DATASET_FINGERPRINT=""
INVENTORY_HASH=""
MIN_FREE_GB="16"
CONFIRM_NO_ACTIVE_READERS=false

usage() {
  cat <<'EOF'
Usage:
  remote_dataset_tactile_storage.sh -H user@host \
    --project /path/DexTeleop --env /path/env \
    --status --dataset-root /data/dataset

  remote_dataset_tactile_storage.sh -H user@host \
    --project /path/DexTeleop --env /path/env \
    --detach-whole-prefix --dataset-root /data/dataset \
    --cold-root /independent/cold/root --dry-run

  remote_dataset_tactile_storage.sh -H user@host \
    --project /path/DexTeleop --env /path/env \
    --rehydrate --dataset-root /data/corrected_dataset \
    --cold-root /independent/cold/root --inventory-hash SHA256 --dry-run

Apply repeats the selected operation with all frozen confirmations:

  ... --apply --plan-hash SHA256 --dataset-fingerprint SHA256 \
    --confirm-no-active-readers

Options:
      --project DIR       Remote DexTeleop project directory. Required.
      --env NAME_OR_DIR   Conda/micromamba env name, or venv directory.
      --python PYTHON     Python executable after env activation. Default: python.
      --timeout SEC       Remote process timeout, 1..86400. Default: 86400.
      --dataset-root DIR  Existing dataset root. Required.
      --cold-root DIR     Existing independent cold-storage root. Required for
                          detach and rehydrate; forbidden for status.
      --status            Read-only tactile availability/receipt status.
      --detach-whole-prefix
                          Verify an independent cold copy and transactionally
                          retire the complete hot tactile/ prefix.
      --rehydrate         Rebuild a complete hot tactile/ tree from a verified
                          cold inventory using fresh files.
      --inventory-hash SHA256
                          Required for rehydrate; optional status filter.
      --min-free-gb N     Minimum free space for detach/rehydrate. Default: 16.
      --dry-run           Recompute and print a full no-write plan (default).
      --apply             Execute the exact frozen detach/rehydrate plan.
      --plan-hash SHA256  Required with --apply and rejected otherwise.
      --dataset-fingerprint SHA256
                          Required with --apply and rejected otherwise.
      --confirm-no-active-readers
                          Required with --apply; caller confirms collection,
                          training, and other dataset readers are stopped.

Safety: L3, fixed non-motion dataset maintenance only. The wrapper exposes no
module/script passthrough, legacy-hardlink break-glass mode, device option,
deletion, overwrite, or real-hardware entrypoint. Detach is fixed to the
whole-prefix contract needed for a recoverable full-v2.3 migration.
EOF
  common_ssh_usage
}

assert_plain_remote_path() {
  local label="$1"
  local value="$2"
  [[ -n "$value" ]] || die "missing $label"
  case "$value" in
    /*|"~/"*) ;;
    *) die "$label must be an absolute or ~/ remote path: $value" ;;
  esac
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    die "$label contains control characters"
  fi
}

select_operation() {
  local requested="$1"
  [[ -z "$OPERATION" ]] || die "choose exactly one operation"
  OPERATION="$requested"
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
    --dataset-root)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      DATASET_ROOT="$2"
      shift 2
      ;;
    --cold-root)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      COLD_ROOT="$2"
      shift 2
      ;;
    --status)
      select_operation "status"
      shift
      ;;
    --detach-whole-prefix)
      select_operation "detach"
      shift
      ;;
    --rehydrate)
      select_operation "rehydrate"
      shift
      ;;
    --inventory-hash)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      INVENTORY_HASH="$2"
      shift 2
      ;;
    --min-free-gb)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      MIN_FREE_GB="$2"
      shift 2
      ;;
    --plan-hash)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      PLAN_HASH="$2"
      shift 2
      ;;
    --dataset-fingerprint)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      DATASET_FINGERPRINT="$2"
      shift 2
      ;;
    --confirm-no-active-readers)
      CONFIRM_NO_ACTIVE_READERS=true
      shift
      ;;
    --dry-run)
      [[ "$MODE_EXPLICIT" != "true" ]] || die "choose exactly one of --dry-run or --apply"
      MODE="dry-run"
      MODE_EXPLICIT=true
      shift
      ;;
    --apply)
      [[ "$MODE_EXPLICIT" != "true" ]] || die "choose exactly one of --dry-run or --apply"
      MODE="apply"
      MODE_EXPLICIT=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      die "argument passthrough is not supported"
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$REMOTE_HOST" ]] || die "missing -H/--host"
[[ -n "$PROJECT" ]] || die "missing --project"
[[ -n "$OPERATION" ]] || die "choose one of --status, --detach-whole-prefix, or --rehydrate"
assert_plain_remote_path "--project" "$PROJECT"
assert_plain_remote_path "--dataset-root" "$DATASET_ROOT"
assert_python_bin_name "$PYTHON_BIN"
[[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || die "timeout must be a number: $TIMEOUT_SEC"
(( TIMEOUT_SEC >= 1 && TIMEOUT_SEC <= 86400 )) || die "timeout must be in 1..86400 seconds"
[[ "$MIN_FREE_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--min-free-gb must be non-negative"

if [[ "$OPERATION" == "status" ]]; then
  [[ -z "$COLD_ROOT" ]] || die "--cold-root is not valid with --status"
  [[ "$MODE" == "dry-run" ]] || die "--status is read-only and does not accept --apply"
  [[ -z "$PLAN_HASH" && -z "$DATASET_FINGERPRINT" ]] || \
    die "plan confirmations are only valid with detach/rehydrate --apply"
  [[ "$CONFIRM_NO_ACTIVE_READERS" == "false" ]] || \
    die "--confirm-no-active-readers is only valid with --apply"
elif [[ "$OPERATION" == "detach" ]]; then
  assert_plain_remote_path "--cold-root" "$COLD_ROOT"
  [[ -z "$INVENTORY_HASH" ]] || die "--inventory-hash is not valid with detach"
else
  assert_plain_remote_path "--cold-root" "$COLD_ROOT"
  [[ "$INVENTORY_HASH" =~ ^[0-9a-f]{64}$ ]] || \
    die "--rehydrate requires a lowercase 64-hex --inventory-hash"
fi

if [[ -n "$INVENTORY_HASH" ]]; then
  [[ "$INVENTORY_HASH" =~ ^[0-9a-f]{64}$ ]] || \
    die "--inventory-hash must be lowercase 64-hex"
fi

if [[ "$MODE" == "apply" ]]; then
  [[ "$PLAN_HASH" =~ ^[0-9a-f]{64}$ ]] || \
    die "--apply requires a lowercase 64-hex --plan-hash"
  [[ "$DATASET_FINGERPRINT" =~ ^[0-9a-f]{64}$ ]] || \
    die "--apply requires a lowercase 64-hex --dataset-fingerprint"
  [[ "$CONFIRM_NO_ACTIVE_READERS" == "true" ]] || \
    die "--apply requires --confirm-no-active-readers"
else
  [[ -z "$PLAN_HASH" && -z "$DATASET_FINGERPRINT" ]] || \
    die "plan confirmations are only valid with --apply"
  [[ "$CONFIRM_NO_ACTIVE_READERS" == "false" ]] || \
    die "--confirm-no-active-readers is only valid with --apply"
fi

finalize_common_ssh
print_wrapper_header "L3-dataset-non-motion" "$PROJECT" "$ENV_NAME"
printf '[scope] dataset=%s\n' "$DATASET_ROOT"
if [[ -n "$COLD_ROOT" ]]; then
  printf '[scope] cold_storage=%s\n' "$COLD_ROOT"
fi
printf '[operation] %s mode=%s timeout_sec=%s\n' "$OPERATION" "$MODE" "$TIMEOUT_SEC"

remote_bash \
  "$PROJECT" \
  "$ENV_NAME" \
  "$PYTHON_BIN" \
  "$TIMEOUT_SEC" \
  "$DATASET_ROOT" \
  "$COLD_ROOT" \
  "$OPERATION" \
  "$MODE" \
  "$PLAN_HASH" \
  "$DATASET_FINGERPRINT" \
  "$INVENTORY_HASH" \
  "$MIN_FREE_GB" <<'REMOTE'
set -Eeuo pipefail

project=$1
env_name=$2
python_bin=$3
timeout_sec=$4
dataset_root=$5
cold_root=$6
operation=$7
mode=$8
plan_hash=$9
dataset_fingerprint=${10}
inventory_hash=${11}
min_free_gb=${12}

resolve_remote_path() {
  local value="$1"
  case "$value" in
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${value#"~/"}" ;;
    /*) printf '%s\n' "$value" ;;
    *) return 1 ;;
  esac
}

activate_env() {
  local requested="$1"
  local conda_sh
  local named_env
  [[ -n "$requested" ]] || return 0
  if [[ -d "$requested" && -f "$requested/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$requested/bin/activate"
    return 0
  fi
  if [[ "$requested" != */* ]]; then
    for named_env in \
      "$HOME/miniconda3/envs/$requested" \
      "$HOME/anaconda3/envs/$requested" \
      "$HOME/mambaforge/envs/$requested" \
      "/opt/conda/envs/$requested"; do
      if [[ -d "$named_env" && -f "$named_env/bin/activate" ]]; then
        # shellcheck disable=SC1090
        source "$named_env/bin/activate"
        return 0
      fi
    done
  fi
  for conda_sh in \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_sh"
      conda activate "$requested"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$requested"
    return 0
  fi
  if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "$requested"
    return 0
  fi
  printf 'Cannot activate env: %s\n' "$requested" >&2
  return 3
}

project="$(resolve_remote_path "$project")"
dataset_root="$(resolve_remote_path "$dataset_root")"
if [[ -n "$cold_root" ]]; then
  cold_root="$(resolve_remote_path "$cold_root")"
fi

[[ -d "$project" ]] || { printf 'Project directory does not exist: %s\n' "$project" >&2; exit 3; }
[[ -d "$dataset_root" && ! -L "$dataset_root" ]] || {
  printf 'Dataset root is missing or unsafe: %s\n' "$dataset_root" >&2
  exit 4
}
if [[ -n "$cold_root" ]]; then
  [[ -d "$cold_root" && ! -L "$cold_root" ]] || {
    printf 'Cold root is missing or unsafe: %s\n' "$cold_root" >&2
    exit 4
  }
fi

cd "$project"
activate_env "$env_name"

case "$operation" in
  status)
    cmd=(
      "$python_bin" -u -m utils.dataset_postprocess
      modality-status "$dataset_root" --modality tactile
    )
    if [[ -n "$inventory_hash" ]]; then
      cmd+=(--inventory-hash "$inventory_hash")
    fi
    ;;
  detach)
    cmd=(
      "$python_bin" -u -m utils.dataset_postprocess
      modality-detach "$dataset_root"
      --cold-root "$cold_root"
      --modality tactile
      --detach-scope whole-prefix
      --min-free-gb "$min_free_gb"
    )
    ;;
  rehydrate)
    cmd=(
      "$python_bin" -u -m utils.dataset_postprocess
      modality-rehydrate "$dataset_root"
      --cold-root "$cold_root"
      --modality tactile
      --inventory-hash "$inventory_hash"
      --min-free-gb "$min_free_gb"
    )
    ;;
  *)
    printf 'Unexpected operation: %s\n' "$operation" >&2
    exit 2
    ;;
esac

if [[ "$mode" == "apply" ]]; then
  cmd+=(
    --apply
    --plan-hash "$plan_hash"
    --confirm-dataset-fingerprint "$dataset_fingerprint"
    --confirm-no-active-readers
  )
fi

command -v timeout >/dev/null 2>&1 || {
  printf 'timeout command is unavailable\n' >&2
  exit 3
}
cmd=(timeout "$timeout_sec" "${cmd[@]}")

printf '[run]'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
REMOTE
