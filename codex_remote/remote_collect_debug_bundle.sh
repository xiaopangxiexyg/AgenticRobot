#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

PROJECT=""
REMOTE_DIR="/tmp/codex_debug_bundles"
LOCAL_OUT="./debug/remote_bundles"
DMESG_LINES=300
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  remote_collect_debug_bundle.sh -H dex-remote --project '~/DexTeleop' --local-out ./debug/bundles

Options:
      --project DIR      Remote project directory to inspect when present.
      --remote-dir DIR   Remote bundle output directory. Default: /tmp/codex_debug_bundles.
      --local-out DIR    Local output directory. Default: ./debug/remote_bundles.
      --dmesg-lines N    Recent dmesg lines to collect. Default: 300.
      --dry-run          Print actions without writing or pulling.

Safety: L2, remote/local diagnostic file writes.
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
    --remote-dir)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      REMOTE_DIR="$2"
      shift 2
      ;;
    --local-out|--local)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      LOCAL_OUT="$2"
      shift 2
      ;;
    --dmesg-lines)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      DMESG_LINES="$2"
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

[[ "$DMESG_LINES" =~ ^[0-9]+$ ]] || die "dmesg lines must be a number: $DMESG_LINES"

finalize_common_ssh
require_cmd rsync

LOCAL_OUT="$(expand_local_path "$LOCAL_OUT")"
mkdir -p "$LOCAL_OUT"
LOCAL_OUT="$(cd "$LOCAL_OUT" && pwd -P)"

print_wrapper_header "L2" "$PROJECT"
printf '[bundle] remote_dir=%s local_out=%s dry_run=%s\n' "$REMOTE_DIR" "$LOCAL_OUT" "$DRY_RUN"

set +e
bundle_output="$(
  remote_bash "$PROJECT" "$REMOTE_DIR" "$DMESG_LINES" "$DRY_RUN" <<'REMOTE'
set -Eeuo pipefail

project=$1
remote_dir=$2
dmesg_lines=$3
dry_run=$4

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

remote_dir="$(resolve_remote_path "$remote_dir")"
bundle_name="debug_bundle_$(hostname)_$(date +%Y%m%d_%H%M%S)"
work_dir="$remote_dir/$bundle_name"
bundle_path="$remote_dir/${bundle_name}.tar.gz"

printf '[run] collect bundle=%s\n' "$bundle_path"
if [[ "$dry_run" == "true" ]]; then
  printf '[result] dry-run only\n'
  exit 0
fi

mkdir -p "$work_dir"

run_capture() {
  local name="$1"
  shift
  {
    printf '$'
    printf ' %q' "$@"
    printf '\n\n'
    "$@"
  } > "$work_dir/$name.txt" 2>&1 || true
}

{
  printf 'created_at=%s\n' "$(date -Is 2>/dev/null || date)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'user=%s\n' "$(id -un)"
  printf 'pwd=%s\n' "$PWD"
} > "$work_dir/summary.txt"

run_capture uname uname -a
run_capture id id
run_capture groups groups
run_capture ip_addr ip addr
run_capture ip_route ip route
run_capture disk df -h
run_capture memory free -h

if command -v lsusb >/dev/null 2>&1; then run_capture lsusb lsusb; fi
if command -v lspci >/dev/null 2>&1; then run_capture lspci lspci; fi
if command -v nvidia-smi >/dev/null 2>&1; then run_capture nvidia_smi nvidia-smi; fi
if command -v conda >/dev/null 2>&1; then run_capture conda_envs conda env list; fi

{
  for node in /dev/video* /dev/ttyUSB* /dev/ttyACM* /dev/input/event* /dev/hidraw* /dev/nvidia* /dev/dri/renderD*; do
    [[ -e "$node" ]] && ls -ld "$node"
  done
} > "$work_dir/dev_nodes.txt" 2>&1 || true

if command -v dmesg >/dev/null 2>&1; then
  dmesg -T 2>&1 | tail -n "$dmesg_lines" > "$work_dir/dmesg_tail.txt" || true
fi

if command -v python >/dev/null 2>&1; then
  run_capture python_version python --version
  python -m pip freeze > "$work_dir/pip_freeze.txt" 2>&1 || true
fi

if [[ -n "$project" ]]; then
  resolved_project="$(resolve_remote_path "$project")"
  {
    printf 'project=%s\n' "$resolved_project"
    if [[ -d "$resolved_project" ]]; then
      cd "$resolved_project"
      printf 'pwd=%s\n' "$PWD"
      if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git status --short
        git rev-parse --show-toplevel
        git rev-parse HEAD
      fi
    else
      printf 'missing\n'
    fi
  } > "$work_dir/project.txt" 2>&1 || true
fi

tar -czf "$bundle_path" -C "$remote_dir" "$bundle_name"
rm -rf "$work_dir"
printf 'BUNDLE\t%s\n' "$bundle_path"
REMOTE
)"
bundle_status=$?
set -e

printf '%s\n' "$bundle_output"

if [[ "$DRY_RUN" == "true" ]]; then
  exit "$bundle_status"
fi
if [[ "$bundle_status" -ne 0 ]]; then
  exit "$bundle_status"
fi

bundle_path="$(awk -F '\t' '$1 == "BUNDLE" {print $2}' <<< "$bundle_output" | tail -n 1)"
[[ -n "$bundle_path" ]] || die "remote bundle collection did not report a bundle path"

rsh="$(build_rsync_rsh)"
args=(rsync -az -s --no-owner --no-group -e "$rsh" "$(remote_target):$bundle_path" "$(ensure_trailing_slash "$LOCAL_OUT")")
printf '[run]'
printf ' %q' "${args[@]}"
printf '\n'
"${args[@]}"
printf '[result] local_out=%s/%s\n' "$LOCAL_OUT" "$(basename "$bundle_path")"
