#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

PROJECT=""
ENV_NAME=""
PYTHON_BIN="python"
SCRIPT_PATH=""
DRY_FLAG="--dry-run"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  remote_arm_dry_run.sh -H dex-remote --project '~/DexTeleop' --env dexteleop310 --script teleop_runtime/replay/arm_init/dry_run.py -- --config configs/arm.yaml

Options:
      --project DIR       Remote project directory.
      --env NAME_OR_DIR   Conda/micromamba env name, or venv directory.
      --python PYTHON     Python executable after env activation. Default: python.
      --script PATH       Project-relative or absolute dry-run script to execute.
      --dry-flag ARG      Dry-run argument appended before extra args.
                          Allowed: --dry-run, --dry_run, --no-motion,
                          --no_motion, --simulate, --simulation,
                          --check-only, --check_only. Default: --dry-run.
  --                     Extra arguments passed to the script.

Safety: L3. This wrapper previews the exact remote Python command. It does not
execute the target script, because a wrapper cannot prove that script honors a
dry-run flag.
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
    --script)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      SCRIPT_PATH="$2"
      shift 2
      ;;
    --dry-flag)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      DRY_FLAG="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
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

[[ -n "$SCRIPT_PATH" ]] || die "missing --script"
[[ -n "$REMOTE_HOST" ]] || die "missing -H/--host"
assert_python_bin_name "$PYTHON_BIN"
is_allowed_dry_flag "$DRY_FLAG" || die "unsupported --dry-flag for arm dry-run safety: $DRY_FLAG"
assert_safe_dry_script_name "$SCRIPT_PATH"
assert_dry_run_control_args "$SCRIPT_PATH" "$DRY_FLAG" "${EXTRA_ARGS[@]}"
if forbidden_dry_run_override_present "${EXTRA_ARGS[@]}"; then
  die "extra arguments contain a flag that can override dry-run safety"
fi

cmd=("$SCRIPT_DIR/remote_python.sh" -H "$REMOTE_HOST")
if [[ -n "$REMOTE_USER" ]]; then cmd+=(-u "$REMOTE_USER"); fi
if [[ -n "$REMOTE_PORT" ]]; then cmd+=(-p "$REMOTE_PORT"); fi
if [[ -n "$KEY_FILE" ]]; then cmd+=(-k "$KEY_FILE"); fi
if [[ -n "$PROJECT" ]]; then cmd+=(--project "$PROJECT"); fi
if [[ -n "$ENV_NAME" ]]; then cmd+=(--env "$ENV_NAME"); fi
cmd+=(--python "$PYTHON_BIN" --dry-run -- "$SCRIPT_PATH")
if [[ -n "$DRY_FLAG" ]]; then cmd+=("$DRY_FLAG"); fi
cmd+=("${EXTRA_ARGS[@]}")

printf '[local] wrapper=%s safety=L3\n' "$(basename "$0")"
printf '[run]'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
