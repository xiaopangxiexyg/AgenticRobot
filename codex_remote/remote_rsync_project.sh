#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

LOCAL_DIR=""
REMOTE_DIR=""
DELETE=false
DRY_RUN=false
YES=false
CONFIRM_DELETE=""
EXCLUDES=(
  ".git/"
  ".codex/"
  ".ssh/"
  ".gnupg/"
  "__pycache__/"
  ".pytest_cache/"
  ".mypy_cache/"
  ".venv/"
  "node_modules/"
  "*.pyc"
  "*.log"
)

usage() {
  cat <<'EOF'
Usage:
  remote_rsync_project.sh -H dex-remote --local ~/Release --remote '~/Release'
  remote_rsync_project.sh -H dex-remote --local . --remote '~/DexTeleop' --dry-run

Options:
      --local DIR       Local source directory.
      --remote DIR      Remote target directory.
      --delete          Delete remote-only files.
      --confirm-delete PATH
                       Required for real --delete. Must match the resolved
                       remote target path printed by the wrapper.
      --no-delete       Keep remote-only files. Default.
      --exclude PATTERN Add an rsync exclude pattern. Can be repeated.
      --dry-run         Print rsync changes without copying.
      --yes             Skip interactive confirmation for a non-empty remote target.

Safety: L2, remote non-motion write.
If LOCAL/.realtime_sync_ssh.exclude exists, its non-comment patterns are also
applied so one-shot sync cannot bypass machine-local protections.
EOF
  common_ssh_usage
}

while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then
    shift "$COMMON_ARGC"
    continue
  fi

  case "$1" in
    --local|-l)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      LOCAL_DIR="$2"
      shift 2
      ;;
    --remote|-r)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      REMOTE_DIR="$2"
      shift 2
      ;;
    --delete)
      DELETE=true
      shift
      ;;
    --confirm-delete)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      CONFIRM_DELETE="$2"
      shift 2
      ;;
    --no-delete)
      DELETE=false
      shift
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
    --yes|-y)
      YES=true
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

[[ -n "$LOCAL_DIR" ]] || die "missing --local"
[[ -n "$REMOTE_DIR" ]] || die "missing --remote"

finalize_common_ssh
require_cmd rsync

LOCAL_DIR="$(expand_local_path "$LOCAL_DIR")"
[[ -d "$LOCAL_DIR" ]] || die "local directory does not exist: $LOCAL_DIR"
LOCAL_DIR="$(cd "$LOCAL_DIR" && pwd -P)"

project_exclude_file="$LOCAL_DIR/.realtime_sync_ssh.exclude"
if [[ -f "$project_exclude_file" ]]; then
  exclude_count=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    pattern="${line%$'\r'}"
    [[ "$pattern" =~ ^[[:space:]]*$ ]] && continue
    [[ "$pattern" =~ ^[[:space:]]*# ]] && continue
    EXCLUDES+=("$pattern")
    ((exclude_count += 1))
  done < "$project_exclude_file"
  info "Loaded $exclude_count exclude pattern(s) from: $project_exclude_file"
fi

print_wrapper_header "L2"
printf '[sync] local=%s\n' "$LOCAL_DIR"
printf '[sync] remote=%s:%s\n' "$(remote_target)" "$REMOTE_DIR"
printf '[sync] delete=%s dry_run=%s\n' "$DELETE" "$DRY_RUN"

set +e
prepare_output="$(
  remote_bash "$REMOTE_DIR" <<'REMOTE'
set -Eeuo pipefail

target=$1
case "$target" in
  "") printf 'ERROR\tmissing target\n'; exit 2 ;;
  "~") target=$HOME ;;
  "~/"*) target=$HOME/${target#~/} ;;
  /*) ;;
  *) target=$HOME/$target ;;
esac

if ! command -v rsync >/dev/null 2>&1; then
  printf 'MISSING_RSYNC\t%s\n' "$target"
  exit 4
fi

if [[ -e "$target" && ! -d "$target" ]]; then
  printf 'NOT_DIR\t%s\n' "$target"
  exit 3
fi

mkdir -p "$target"
actual="$(cd "$target" && pwd -P)"
if find "$actual" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  printf 'NONEMPTY\t%s\n' "$actual"
else
  printf 'EMPTY\t%s\n' "$actual"
fi
REMOTE
)"
prepare_status=$?
set -e

last_line=""
while IFS= read -r line; do
  [[ -n "$line" ]] && last_line="$line"
done <<< "$prepare_output"

state="${last_line%%$'\t'*}"
remote_actual="${last_line#*$'\t'}"

case "$state" in
  EMPTY)
    info "Remote target is empty: $remote_actual"
    ;;
  NONEMPTY)
    warn "Remote target is not empty: $remote_actual"
    if [[ "$YES" != "true" && "$DRY_RUN" != "true" ]]; then
      confirm_or_die "Continue and rsync into this directory? [y/N]: "
    fi
    ;;
  MISSING_RSYNC)
    die "remote machine is missing rsync. Install rsync on the remote first."
    ;;
  NOT_DIR)
    die "remote target exists but is not a directory: $remote_actual"
    ;;
  *)
    printf 'Remote output:\n%s\n' "$prepare_output" >&2
    die "failed to prepare remote target (ssh exit status: $prepare_status)"
    ;;
esac

if [[ "$DELETE" == "true" && "$DRY_RUN" != "true" ]]; then
  if [[ "$CONFIRM_DELETE" != "$remote_actual" ]]; then
    die "--delete requires --confirm-delete '$remote_actual' after reviewing the resolved remote path"
  fi
fi

rsh="$(build_rsync_rsh)"
source_dir="$(ensure_trailing_slash "$LOCAL_DIR")"
target_dir="$(remote_target):$(ensure_trailing_slash "$remote_actual")"

args=(rsync -az -s --no-owner --no-group --info=stats2 -e "$rsh")
if [[ "$DELETE" == "true" ]]; then
  args+=(--delete)
fi
if [[ "$DRY_RUN" == "true" ]]; then
  args+=(--dry-run)
fi
for pattern in "${EXCLUDES[@]}"; do
  args+=(--exclude="$pattern")
done
args+=("$source_dir" "$target_dir")

printf '[run]'
printf ' %q' "${args[@]}"
printf '\n'

"${args[@]}"
