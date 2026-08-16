#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

PROJECT=""
ENV_NAME=""
PYTHON_BIN="python"
TIMEOUT_SEC="900"
SOURCE_ROOT=""
OUTPUT_ROOT=""
REGISTRY_PATH=""
MODE="dry-run"
MODE_EXPLICIT=false
EXPECTED_PLAN_HASH=""
VIEW_KIND="camera-only"

usage() {
  cat <<'EOF'
Usage:
  remote_dataset_camera_semantics.sh -H dex-remote \
    --project '~/DexTeleop' --env dexteleop310 \
    --source-root /data/source_dataset \
    --output-root /data/camera_semantic_prepared_views/source_dataset \
    --registry /path/to/frozen_camera_role_registry.json \
    --dry-run

  remote_dataset_camera_semantics.sh -H dex-remote \
    --project '~/DexTeleop' --env dexteleop310 \
    --source-root /data/source_dataset \
    --output-root /data/camera_semantic_full/source_dataset \
    --registry /path/to/frozen_camera_role_registry.json \
    --full-v23 --dry-run

  remote_dataset_camera_semantics.sh -H dex-remote \
    --project '~/DexTeleop' --env dexteleop310 \
    --source-root /data/source_dataset \
    --output-root /data/camera_semantic_prepared_views/source_dataset \
    --registry /path/to/frozen_camera_role_registry.json \
    --apply --expected-plan-hash SHA256

Options:
      --project DIR       Remote DexTeleop project directory. Required.
      --env NAME_OR_DIR   Conda/micromamba env name, or venv directory.
      --python PYTHON     Python executable after env activation. Default: python.
      --timeout SEC       Remote process timeout, 1..86400. Default: 900.
      --source-root DIR   Existing source dataset; the command treats it as read-only.
      --output-root DIR   New corrected-output directory; existing outputs are refused.
      --registry FILE     Frozen camera-role registry used by the correction plan.
      --full-v23          Build an ordinary-loader full-v2.3 clone after verified
                          tactile retirement. Default remains camera-only.
      --dry-run           Recompute and print the complete no-write plan (default).
      --apply             Publish the selected new corrected output.
      --expected-plan-hash SHA256
                          Required with --apply and rejected otherwise.

Safety: L3, fixed non-motion dataset maintenance only. This wrapper exposes no
module/script passthrough, device options, control options, deletion, overwrite,
or real-hardware entrypoint.
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
    --source-root)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --registry)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      REGISTRY_PATH="$2"
      shift 2
      ;;
    --expected-plan-hash)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      EXPECTED_PLAN_HASH="$2"
      shift 2
      ;;
    --full-v23)
      [[ "$VIEW_KIND" == "camera-only" ]] || die "--full-v23 may be specified only once"
      VIEW_KIND="full-v23"
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
assert_plain_remote_path "--project" "$PROJECT"
assert_plain_remote_path "--source-root" "$SOURCE_ROOT"
assert_plain_remote_path "--output-root" "$OUTPUT_ROOT"
assert_plain_remote_path "--registry" "$REGISTRY_PATH"
assert_python_bin_name "$PYTHON_BIN"
[[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || die "timeout must be a number: $TIMEOUT_SEC"
(( TIMEOUT_SEC >= 1 && TIMEOUT_SEC <= 86400 )) || die "timeout must be in 1..86400 seconds"

if [[ "$MODE" == "apply" ]]; then
  [[ "$EXPECTED_PLAN_HASH" =~ ^[0-9a-f]{64}$ ]] || \
    die "--apply requires a lowercase 64-hex --expected-plan-hash"
else
  [[ -z "$EXPECTED_PLAN_HASH" ]] || \
    die "--expected-plan-hash is only valid with --apply"
fi

finalize_common_ssh
print_wrapper_header "L3-dataset-non-motion" "$PROJECT" "$ENV_NAME"
printf '[scope] source_read_only=%s\n' "$SOURCE_ROOT"
printf '[scope] output_new_only=%s\n' "$OUTPUT_ROOT"
printf '[scope] registry_read_only=%s\n' "$REGISTRY_PATH"
printf '[mode] %s view_kind=%s timeout_sec=%s\n' "$MODE" "$VIEW_KIND" "$TIMEOUT_SEC"

remote_bash \
  "$PROJECT" \
  "$ENV_NAME" \
  "$PYTHON_BIN" \
  "$TIMEOUT_SEC" \
  "$SOURCE_ROOT" \
  "$OUTPUT_ROOT" \
  "$REGISTRY_PATH" \
  "$MODE" \
  "$EXPECTED_PLAN_HASH" \
  "$VIEW_KIND" <<'REMOTE'
set -Eeuo pipefail

project=$1
env_name=$2
python_bin=$3
timeout_sec=$4
source_root=$5
output_root=$6
registry_path=$7
mode=$8
expected_plan_hash=$9
view_kind=${10}

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
      if [[ -d "$named_env" && -x "$named_env/bin/python" ]]; then
        for conda_sh in \
          "$HOME/anaconda3/etc/profile.d/conda.sh" \
          "$HOME/miniconda3/etc/profile.d/conda.sh" \
          "$HOME/mambaforge/etc/profile.d/conda.sh" \
          "/opt/conda/etc/profile.d/conda.sh"; do
          if [[ -f "$conda_sh" ]]; then
            # shellcheck disable=SC1090
            source "$conda_sh"
            conda activate "$named_env"
            return 0
          fi
        done
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
source_root="$(resolve_remote_path "$source_root")"
output_root="$(resolve_remote_path "$output_root")"
registry_path="$(resolve_remote_path "$registry_path")"

[[ -d "$project" ]] || { printf 'Project directory does not exist: %s\n' "$project" >&2; exit 3; }
[[ -d "$source_root" ]] || { printf 'Source dataset does not exist: %s\n' "$source_root" >&2; exit 4; }
[[ -f "$registry_path" && ! -L "$registry_path" ]] || { printf 'Registry is missing or unsafe: %s\n' "$registry_path" >&2; exit 4; }
if [[ "$mode" == "apply" && -e "$output_root" ]]; then
  printf 'Refusing existing output path: %s\n' "$output_root" >&2
  exit 5
fi

cd "$project"
activate_env "$env_name"

if [[ "$view_kind" == "full-v23" ]]; then
  subcommand="camera-semantics-materialize"
else
  subcommand="camera-semantics-prepare-view"
fi

cmd=(
  "$python_bin" -u -m utils.dataset_postprocess
  "$subcommand"
  "$source_root"
  --output-root "$output_root"
  --registry "$registry_path"
)
if [[ "$mode" == "apply" ]]; then
  cmd+=(--expected-plan-hash "$expected_plan_hash" --apply)
elif [[ "$view_kind" == "camera-only" ]]; then
  # Full-v2.3 materialization is already plan-only without --apply.
  cmd+=(--dry-run)
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
