#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

REMOTE_PATH=""
LOCAL_DIR=""
DRY_RUN=false
EXCLUDES=(
  "__pycache__/"
  ".pytest_cache/"
  "*.pyc"
)

usage() {
  cat <<'EOF'
Usage:
  remote_pull.sh -H dex-remote --remote '~/DexTeleop/debug/latest.jpg' --local ./debug
  remote_pull.sh -H dex-remote --remote '~/DexTeleop/logs' --local ./remote_logs --dry-run

Options:
      --remote PATH     Remote file or directory to pull.
      --local DIR       Local output directory.
      --exclude PATTERN Add an rsync exclude pattern. Can be repeated.
      --dry-run         Print rsync changes without copying.

Safety: L2, local write from remote data.
EOF
  common_ssh_usage
}

while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then
    shift "$COMMON_ARGC"
    continue
  fi

  case "$1" in
    --remote|-r)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      REMOTE_PATH="$2"
      shift 2
      ;;
    --local|-l|--local-out)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      LOCAL_DIR="$2"
      shift 2
      ;;
    --exclude)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      EXCLUDES+=("$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
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

[[ -n "$REMOTE_PATH" ]] || die "missing --remote"
[[ -n "$LOCAL_DIR" ]] || die "missing --local"

finalize_common_ssh
require_cmd rsync

LOCAL_DIR="$(expand_local_path "$LOCAL_DIR")"
mkdir -p "$LOCAL_DIR"
LOCAL_DIR="$(cd "$LOCAL_DIR" && pwd -P)"

print_wrapper_header "L2"
printf '[pull] remote=%s:%s\n' "$(remote_target)" "$REMOTE_PATH"
printf '[pull] local=%s dry_run=%s\n' "$LOCAL_DIR" "$DRY_RUN"

set +e
resolve_output="$(
  remote_bash "$REMOTE_PATH" <<'REMOTE'
set -Eeuo pipefail
path=$1
case "$path" in
  "") printf 'ERROR\tmissing path\n'; exit 2 ;;
  "~") path=$HOME ;;
  "~/"*) path=$HOME/${path#~/} ;;
  /*) ;;
  *) path=$HOME/$path ;;
esac

if [[ ! -e "$path" ]]; then
  printf 'MISSING\t%s\n' "$path"
  exit 4
fi

actual="$(cd "$(dirname "$path")" && pwd -P)/$(basename "$path")"
if [[ -d "$actual" ]]; then
  printf 'DIR\t%s\n' "$actual"
else
  printf 'FILE\t%s\n' "$actual"
fi
REMOTE
)"
resolve_status=$?
set -e

last_line=""
while IFS= read -r line; do
  [[ -n "$line" ]] && last_line="$line"
done <<< "$resolve_output"

kind="${last_line%%$'\t'*}"
remote_actual="${last_line#*$'\t'}"

case "$kind" in
  DIR|FILE) ;;
  MISSING) die "remote path does not exist: $remote_actual" ;;
  *)
    printf 'Remote output:\n%s\n' "$resolve_output" >&2
    die "failed to resolve remote path (ssh exit status: $resolve_status)"
    ;;
esac

rsh="$(build_rsync_rsh)"
args=(rsync -az -s --no-owner --no-group --info=stats2 -e "$rsh")
if [[ "$DRY_RUN" == "true" ]]; then
  args+=(--dry-run)
fi
for pattern in "${EXCLUDES[@]}"; do
  args+=(--exclude="$pattern")
done

if [[ "$kind" == "DIR" ]]; then
  args+=("$(remote_target):$(ensure_trailing_slash "$remote_actual")" "$(ensure_trailing_slash "$LOCAL_DIR")")
else
  args+=("$(remote_target):$remote_actual" "$(ensure_trailing_slash "$LOCAL_DIR")")
fi

printf '[run]'
printf ' %q' "${args[@]}"
printf '\n'

"${args[@]}"
