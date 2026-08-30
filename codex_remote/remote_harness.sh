#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

REMOTE_HARNESS=""
HARNESS_ARGS=()
EXPLICIT_HOST=false

usage() {
  cat <<'EOF'
Usage:
  remote_harness.sh -H user@host --harness /home/user/HarnessConfig/bin/harness \
    runner preflight RUNNER_ID [options]
  remote_harness.sh -H user@host --harness /home/user/HarnessConfig/bin/harness \
    transfer exchange-run EXCHANGE [options]

Options:
      --harness PATH  Absolute remote host-private Harness wrapper. Required.

Only `runner preflight` and `transfer exchange-run` are accepted. The latter
may execute the exact authenticated Queue request contained in EXCHANGE; this
wrapper does not grant request, device, motion, or arbitrary-command authority.
EOF
  common_ssh_usage
}

while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then
    case "$1" in
      -H|--host) EXPLICIT_HOST=true ;;
    esac
    shift "$COMMON_ARGC"
    continue
  fi

  case "$1" in
    --harness)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      REMOTE_HARNESS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      HARNESS_ARGS=("$@")
      break
      ;;
    *)
      HARNESS_ARGS=("$@")
      break
      ;;
  esac
done

[[ "$EXPLICIT_HOST" == "true" ]] || die "missing -H/--host"
[[ -n "$REMOTE_HARNESS" ]] || die "missing --harness"
case "$REMOTE_HARNESS" in
  *//*|*/../*|*/./*|*$'\n'*|*$'\r'*)
    die "unsafe --harness path: $REMOTE_HARNESS"
    ;;
esac
case "$REMOTE_HARNESS" in
  /*/HarnessConfig/bin/harness) ;;
  *) die "--harness must be an absolute host-private */HarnessConfig/bin/harness path" ;;
esac

[[ ${#HARNESS_ARGS[@]} -ge 2 ]] || die "missing bounded Harness command"
case "${HARNESS_ARGS[0]} ${HARNESS_ARGS[1]}" in
  "runner preflight") COMMAND_KIND="runner-preflight" ;;
  "transfer exchange-run") COMMAND_KIND="exchange-run" ;;
  *) die "unsupported Harness command; allowed: runner preflight, transfer exchange-run" ;;
esac
for argument in "${HARNESS_ARGS[@]}"; do
  case "$argument" in
    *$'\n'*|*$'\r'*) die "Harness arguments must not contain line breaks" ;;
  esac
done

finalize_common_ssh
case "$REMOTE_HOST" in
  -*|*$'\n'*|*$'\r'*) die "unsafe SSH host: $REMOTE_HOST" ;;
esac
case "$REMOTE_USER" in
  -*|*$'\n'*|*$'\r'*) die "unsafe SSH user: $REMOTE_USER" ;;
esac
print_wrapper_header "L3-protected-harness" "" ""
info "command_kind=$COMMAND_KIND remote_harness=$REMOTE_HARNESS"

remote_bash "$REMOTE_HARNESS" "${HARNESS_ARGS[@]}" <<'REMOTE'
set -Eeuo pipefail

harness_wrapper=$1
shift
if [[ -L "$harness_wrapper" || ! -f "$harness_wrapper" || ! -x "$harness_wrapper" ]]; then
  printf 'Remote Harness wrapper is missing or unsafe: %s\n' "$harness_wrapper" >&2
  exit 3
fi
exec "$harness_wrapper" "$@"
REMOTE
