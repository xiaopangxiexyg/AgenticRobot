#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

REMOTE_PATH=""
HEAD_LINES=120
TAIL_LINES=""
MAX_BYTES=20000

usage() {
  cat <<'EOF'
Usage:
  remote_view.sh -H dex-remote --path '~/DexTeleop/logs/latest.log'
  remote_view.sh -H dex-remote --path '~/DexTeleop/outputs' --head 80
  remote_view.sh -H dex-remote --path '~/DexTeleop/logs/latest.log' --tail 200

Options:
      --path PATH      Remote file or directory to view.
      --head N         Show first N lines for text files. Default: 120.
      --tail N         Show last N lines for text files.
      --max-bytes N    Cap displayed text bytes. Default: 20000.

Safety: L1, remote read-only. Refuses common secret/key paths.
EOF
  common_ssh_usage
}

while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then
    shift "$COMMON_ARGC"
    continue
  fi

  case "$1" in
    --path|-f)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      REMOTE_PATH="$2"
      shift 2
      ;;
    --head)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      HEAD_LINES="$2"
      TAIL_LINES=""
      shift 2
      ;;
    --tail)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      TAIL_LINES="$2"
      shift 2
      ;;
    --max-bytes)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      MAX_BYTES="$2"
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

[[ -n "$REMOTE_PATH" ]] || die "missing --path"
[[ "$HEAD_LINES" =~ ^[0-9]+$ ]] || die "head lines must be a number: $HEAD_LINES"
if [[ -n "$TAIL_LINES" && ! "$TAIL_LINES" =~ ^[0-9]+$ ]]; then
  die "tail lines must be a number: $TAIL_LINES"
fi
[[ "$MAX_BYTES" =~ ^[0-9]+$ ]] || die "max bytes must be a number: $MAX_BYTES"

finalize_common_ssh
print_wrapper_header "L1"
printf '[view] path=%s\n' "$REMOTE_PATH"

remote_bash "$REMOTE_PATH" "$HEAD_LINES" "$TAIL_LINES" "$MAX_BYTES" <<'REMOTE'
set -Eeuo pipefail

path_arg=$1
head_lines=$2
tail_lines=$3
max_bytes=$4

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

refuse_sensitive_path() {
  local path="$1"
  local lower="${path,,}"

  case "$lower" in
    */.ssh|*/.ssh/*|*/.gnupg|*/.gnupg/*|*/.codex|*/.codex/*|\
    */.aws|*/.aws/*|*/.config/openai|*/.config/openai/*|\
    */.netrc|*.pem|*.key|*.p12|*.pfx|*.kube/config|*/secrets/*|*/secret/*|\
    */.env|*/.env.*|*.env|*.env.*)
      printf 'Refusing to view common secret/key path: %s\n' "$path" >&2
      exit 5
      ;;
  esac
}

is_text_file() {
  local path="$1"
  local mime

  if command -v file >/dev/null 2>&1; then
    mime="$(file -b --mime-type "$path" 2>/dev/null || true)"
    case "$mime" in
      text/*|application/json|application/xml|application/x-yaml|application/yaml|application/toml|application/javascript)
        return 0
        ;;
      inode/x-empty)
        return 0
        ;;
      *)
        return 1
        ;;
    esac
  fi

  LC_ALL=C grep -Iq . "$path"
}

path="$(resolve_remote_path "$path_arg")"
refuse_sensitive_path "$path"

if [[ ! -e "$path" ]]; then
  printf 'Remote path does not exist: %s\n' "$path" >&2
  exit 4
fi

printf '[remote] resolved=%s\n' "$path"

if [[ -d "$path" ]]; then
  printf '[remote] type=directory\n'
  ls -la "$path" | head -n "$head_lines"
  exit 0
fi

if [[ ! -f "$path" ]]; then
  printf '[remote] type=other\n'
  ls -ld "$path"
  exit 0
fi

printf '[remote] type=file\n'
ls -lh "$path"
if command -v file >/dev/null 2>&1; then
  file "$path" || true
fi

if ! is_text_file "$path"; then
  printf '[result] refusing to display non-text file; use remote_pull.sh if you need to fetch it intentionally.\n' >&2
  exit 4
fi

printf '\n[content]\n'
if [[ -n "$tail_lines" ]]; then
  tail -n "$tail_lines" "$path" | head -c "$max_bytes"
else
  sed -n "1,${head_lines}p" "$path" | head -c "$max_bytes"
fi
printf '\n'
REMOTE
