#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

LOCAL_DIR=""
REMOTE_DIR=""
MAX_FILES=200
CHECK_PATHS=()

usage() {
  cat <<'EOF'
Usage:
  remote_sync_check.sh -H dex-remote --local ~/DexTeleop --remote '~/DexTeleop' --path teleop_runtime/foo.py
  remote_sync_check.sh -H dex-remote --local . --remote '~/DexTeleop' --path src --max-files 500

Options:
      --local DIR       Local synced project/work directory.
      --remote DIR      Remote synced project/work directory.
      --path RELPATH    Relative file or directory path to compare. Can be repeated.
      --max-files N     Max files to compare per directory path. Default: 200.

Safety: L1, read-only sync verification. No remote writes.
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
    --path)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      CHECK_PATHS+=("$2")
      shift 2
      ;;
    --max-files)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      MAX_FILES="$2"
      shift 2
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
[[ "${#CHECK_PATHS[@]}" -gt 0 ]] || die "missing --path"
[[ "$MAX_FILES" =~ ^[0-9]+$ ]] || die "max files must be a number: $MAX_FILES"
(( MAX_FILES > 0 )) || die "max files must be greater than zero"

require_cmd sha256sum
require_cmd stat
require_cmd find
require_cmd sort
require_cmd diff
finalize_common_ssh

LOCAL_DIR="$(expand_local_path "$LOCAL_DIR")"
[[ -d "$LOCAL_DIR" ]] || die "local directory does not exist: $LOCAL_DIR"
LOCAL_DIR="$(cd "$LOCAL_DIR" && pwd -P)"

validate_relpath() {
  local rel="$1"

  case "$rel" in
    ""|/*|*"/../"*|../*|*".."|*/*/..|*$'\n'*|*$'\t'*)
      die "paths must be clean relative paths under --local/--remote: $rel"
      ;;
  esac
}

append_file_manifest() {
  local root="$1"
  local rel="$2"
  local file="$root/$rel"
  local hash size

  [[ -f "$file" ]] || die "local file missing: $rel"
  hash="$(sha256sum "$file" | awk '{print $1}')"
  size="$(stat -c '%s' "$file")"
  printf 'FILE\t%s\t%s\t%s\n' "$rel" "$hash" "$size"
}

append_dir_manifest() {
  local root="$1"
  local rel="$2"
  local dir="$root/$rel"
  local count=0
  local file subrel hash size

  [[ -d "$dir" ]] || die "local directory missing: $rel"
  while IFS= read -r file; do
    count=$((count + 1))
    if (( count > MAX_FILES )); then
      printf 'TRUNCATED\t%s\tmax-files=%s\n' "$rel" "$MAX_FILES"
      break
    fi
    subrel="${file#"$root"/}"
    hash="$(sha256sum "$file" | awk '{print $1}')"
    size="$(stat -c '%s' "$file")"
    printf 'FILE\t%s\t%s\t%s\n' "$subrel" "$hash" "$size"
  done < <(
    find "$dir" \
      \( -path '*/.git' -o -path '*/.git/*' -o \
         -path '*/.codex' -o -path '*/.codex/*' -o \
         -path '*/__pycache__' -o -path '*/__pycache__/*' -o \
         -path '*/.pytest_cache' -o -path '*/.pytest_cache/*' \) -prune -o \
      -type f -print | sort
  )
}

local_manifest="$(mktemp)"
remote_manifest="$(mktemp)"
trap 'rm -f "$local_manifest" "$remote_manifest"' EXIT

for rel in "${CHECK_PATHS[@]}"; do
  validate_relpath "$rel"
  if [[ -f "$LOCAL_DIR/$rel" ]]; then
    append_file_manifest "$LOCAL_DIR" "$rel" >> "$local_manifest"
  elif [[ -d "$LOCAL_DIR/$rel" ]]; then
    append_dir_manifest "$LOCAL_DIR" "$rel" >> "$local_manifest"
  else
    die "local path does not exist: $rel"
  fi
done

print_wrapper_header "L1"
printf '[sync-check] local=%s\n' "$LOCAL_DIR"
printf '[sync-check] remote=%s:%s\n' "$(remote_target)" "$REMOTE_DIR"
printf '[sync-check] paths=%s max_files=%s\n' "${#CHECK_PATHS[@]}" "$MAX_FILES"

set +e
remote_output="$(
  remote_bash "$REMOTE_DIR" "$MAX_FILES" "${#CHECK_PATHS[@]}" "${CHECK_PATHS[@]}" <<'REMOTE'
set -Eeuo pipefail

remote_dir=$1
max_files=$2
path_count=$3
shift 3

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

validate_relpath() {
  local rel="$1"
  case "$rel" in
    ""|/*|*"/../"*|../*|*".."|*/*/..|*$'\n'*|*$'\t'*)
      printf 'Invalid relative path: %s\n' "$rel" >&2
      exit 2
      ;;
  esac
}

append_file_manifest() {
  local root="$1"
  local rel="$2"
  local file="$root/$rel"
  local hash size

  [[ -f "$file" ]] || { printf 'REMOTE_MISSING\t%s\n' "$rel"; return 0; }
  hash="$(sha256sum "$file" | awk '{print $1}')"
  size="$(stat -c '%s' "$file")"
  printf 'FILE\t%s\t%s\t%s\n' "$rel" "$hash" "$size"
}

append_dir_manifest() {
  local root="$1"
  local rel="$2"
  local dir="$root/$rel"
  local count=0
  local file subrel hash size

  [[ -d "$dir" ]] || { printf 'REMOTE_MISSING\t%s\n' "$rel"; return 0; }
  while IFS= read -r file; do
    count=$((count + 1))
    if (( count > max_files )); then
      printf 'TRUNCATED\t%s\tmax-files=%s\n' "$rel" "$max_files"
      break
    fi
    subrel="${file#"$root"/}"
    hash="$(sha256sum "$file" | awk '{print $1}')"
    size="$(stat -c '%s' "$file")"
    printf 'FILE\t%s\t%s\t%s\n' "$subrel" "$hash" "$size"
  done < <(
    find "$dir" \
      \( -path '*/.git' -o -path '*/.git/*' -o \
         -path '*/.codex' -o -path '*/.codex/*' -o \
         -path '*/__pycache__' -o -path '*/__pycache__/*' -o \
         -path '*/.pytest_cache' -o -path '*/.pytest_cache/*' \) -prune -o \
      -type f -print | sort
  )
}

remote_dir="$(resolve_remote_path "$remote_dir")"
[[ -d "$remote_dir" ]] || { printf 'Remote directory does not exist: %s\n' "$remote_dir" >&2; exit 4; }

for ((i = 0; i < path_count; i++)); do
  rel="$1"
  shift
  validate_relpath "$rel"
  if [[ -f "$remote_dir/$rel" ]]; then
    append_file_manifest "$remote_dir" "$rel"
  elif [[ -d "$remote_dir/$rel" ]]; then
    append_dir_manifest "$remote_dir" "$rel"
  else
    printf 'REMOTE_MISSING\t%s\n' "$rel"
  fi
done
REMOTE
)"
remote_status=$?
set -e

if [[ "$remote_status" -ne 0 ]]; then
  printf '%s\n' "$remote_output" >&2
  exit "$remote_status"
fi

printf '%s\n' "$remote_output" > "$remote_manifest"

if diff -u "$local_manifest" "$remote_manifest"; then
  printf '[result] sync-check=ok\n'
else
  printf '[result] sync-check=mismatch\n' >&2
  exit 4
fi
