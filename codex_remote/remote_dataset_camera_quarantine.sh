#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

PROJECT=""
ENV_NAME=""
PYTHON_BIN="python"
TIMEOUT_SEC="300"
VIEW_ROOT=""
AUDIT_ROOT=""
MODE="plan"
EXPECTED_PLAN_HASH=""

usage() {
  cat <<'EOF'
Usage:
  remote_dataset_camera_quarantine.sh -H user@host \
    --project /path/DexTeleop --env /path/env \
    --view-root /path/invalid_view --audit-root /path/_invalid_audit

  remote_dataset_camera_quarantine.sh -H user@host \
    --project /path/DexTeleop --env /path/env \
    --view-root /path/invalid_view --audit-root /path/_invalid_audit \
    --apply --expected-plan-hash SHA256

Safety: fixed L3 non-motion operation. It can only prove a prepared view invalid
and atomically rename it into a same-filesystem audit directory. It cannot
delete, overwrite, or quarantine a view accepted by the strict loader.
EOF
  common_ssh_usage
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
    --view-root) VIEW_ROOT="$2"; shift 2 ;;
    --audit-root) AUDIT_ROOT="$2"; shift 2 ;;
    --expected-plan-hash) EXPECTED_PLAN_HASH="$2"; shift 2 ;;
    --apply) MODE="apply"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) die "argument passthrough is not supported" ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$REMOTE_HOST" ]] || die "missing -H/--host"
[[ "$PROJECT" == /* && "$VIEW_ROOT" == /* && "$AUDIT_ROOT" == /* ]] || \
  die "--project, --view-root, and --audit-root must be absolute paths"
assert_python_bin_name "$PYTHON_BIN"
[[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || die "timeout must be numeric"
if [[ "$MODE" == "apply" ]]; then
  [[ "$EXPECTED_PLAN_HASH" =~ ^[0-9a-f]{64}$ ]] || \
    die "--apply requires a lowercase 64-hex --expected-plan-hash"
else
  [[ -z "$EXPECTED_PLAN_HASH" ]] || \
    die "--expected-plan-hash is only valid with --apply"
fi

finalize_common_ssh
print_wrapper_header "L3-dataset-non-motion" "$PROJECT" "$ENV_NAME"
printf '[scope] invalid_view=%s\n[scope] audit_root=%s\n[mode] %s\n' \
  "$VIEW_ROOT" "$AUDIT_ROOT" "$MODE"

remote_bash \
  "$PROJECT" "$ENV_NAME" "$PYTHON_BIN" "$TIMEOUT_SEC" \
  "$VIEW_ROOT" "$AUDIT_ROOT" "$MODE" "$EXPECTED_PLAN_HASH" <<'REMOTE'
set -Eeuo pipefail
project=$1
env_name=$2
python_bin=$3
timeout_sec=$4
view_root=$5
audit_root=$6
mode=$7
expected_plan_hash=$8

activate_env() {
  local requested=$1
  local conda_sh
  [[ -n "$requested" ]] || return 0
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

[[ -d "$project" ]] || { printf 'Project missing: %s\n' "$project" >&2; exit 3; }
[[ -d "$view_root" && ! -L "$view_root" ]] || {
  printf 'View missing or unsafe: %s\n' "$view_root" >&2
  exit 4
}
cd "$project"
activate_env "$env_name"
cmd=(
  "$python_bin" -u -m utils.dataset_postprocess
  camera-semantics-quarantine-invalid-view "$view_root"
  --audit-root "$audit_root"
)
if [[ "$mode" == "apply" ]]; then
  cmd+=(--expected-plan-hash "$expected_plan_hash" --apply)
fi
cmd=(timeout "$timeout_sec" "${cmd[@]}")
printf '[run]'; printf ' %q' "${cmd[@]}"; printf '\n'
exec "${cmd[@]}"
REMOTE
