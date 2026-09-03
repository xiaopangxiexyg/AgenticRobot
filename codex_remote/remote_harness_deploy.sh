#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

DEPLOYMENT_ROOT=""
EXPECTED_COMMIT=""
CONFIRM_PLAN_SHA256=""
ACTION=""
EXPLICIT_HOST=false

usage() {
  cat <<'EOF'
Usage:
  remote_harness_deploy.sh -H user@host --root /home/user/zc \
    --expected-commit FULL_SHA plan
  remote_harness_deploy.sh -H user@host --root /home/user/zc \
    --expected-commit FULL_SHA \
    --confirm-plan-sha256 FULL_SHA256 install

The deployment root must contain these fixed paths:
  HarnessCodingTool             protected Harness source repository
  AgenticRobot                  managed project root
  HarnessControl               protected installation root
  HarnessConfig/bin/harness     host-private Harness launcher

`plan` fetches only origin/agentic-robot, verifies the exact expected commit,
and emits the Harness control install plan. `install` repeats those checks,
requires the exact plan SHA-256, activates the installation, verifies the
current version path, and emits control status.

This L3 maintenance wrapper cannot run project commands, access devices, move
project branches, accept arbitrary refs, or execute arbitrary remote commands.
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
    --root)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      DEPLOYMENT_ROOT="$2"
      shift 2
      ;;
    --expected-commit)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      EXPECTED_COMMIT="$2"
      shift 2
      ;;
    --confirm-plan-sha256)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      CONFIRM_PLAN_SHA256="$2"
      shift 2
      ;;
    plan|install)
      [[ -z "$ACTION" ]] || die "multiple deployment actions are not allowed"
      ACTION="$1"
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

[[ "$EXPLICIT_HOST" == "true" ]] || die "missing -H/--host"
[[ -n "$DEPLOYMENT_ROOT" ]] || die "missing --root"
[[ "$DEPLOYMENT_ROOT" == /* ]] || die "--root must be an absolute path"
case "$DEPLOYMENT_ROOT" in
  /|*//*|*/../*|*/..|*/./*|*/.|*$'\n'*|*$'\r'*)
    die "unsafe --root path: $DEPLOYMENT_ROOT"
    ;;
esac
DEPLOYMENT_ROOT="${DEPLOYMENT_ROOT%/}"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  die "--expected-commit must be a full lowercase Git SHA"
}
[[ -n "$ACTION" ]] || die "missing deployment action: plan or install"
if [[ "$ACTION" == "plan" ]]; then
  [[ -z "$CONFIRM_PLAN_SHA256" ]] || {
    die "plan does not accept --confirm-plan-sha256"
  }
else
  [[ "$CONFIRM_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    die "install requires a full lowercase --confirm-plan-sha256"
  }
fi

finalize_common_ssh
case "$REMOTE_HOST" in
  -*|*$'\n'*|*$'\r'*) die "unsafe SSH host: $REMOTE_HOST" ;;
esac
case "$REMOTE_USER" in
  -*|*$'\n'*|*$'\r'*) die "unsafe SSH user: $REMOTE_USER" ;;
esac

print_wrapper_header "L3-protected-harness-deployment" "$DEPLOYMENT_ROOT" ""
info "action=$ACTION protected_branch=agentic-robot expected_commit=$EXPECTED_COMMIT"

remote_bash \
  "$DEPLOYMENT_ROOT" \
  "$EXPECTED_COMMIT" \
  "$ACTION" \
  "$CONFIRM_PLAN_SHA256" <<'REMOTE'
set -Eeuo pipefail

deployment_root=$1
expected_commit=$2
action=$3
confirm_plan_sha256=$4
source_repository="$deployment_root/HarnessCodingTool"
project_root="$deployment_root/AgenticRobot"
install_root="$deployment_root/HarnessControl"
harness_wrapper="$deployment_root/HarnessConfig/bin/harness"
protected_branch=agentic-robot
expected_identity=CYBER-QAQ/HarnessCodingTool

fail() {
  printf 'Remote Harness deployment refused: %s\n' "$*" >&2
  exit 3
}

canonical_identity() {
  local value=$1
  local repository_path owner repository

  case "$value" in
    *://*)
      repository_path=${value#*://}
      repository_path=${repository_path#*/}
      ;;
    *:*) repository_path=${value#*:} ;;
    *) repository_path=$value ;;
  esac
  repository_path=${repository_path#/}
  repository_path=${repository_path%/}
  owner=${repository_path%%/*}
  repository=${repository_path#*/}
  repository=${repository%.git}
  if [[ -z "$owner" || -z "$repository" || "$repository" == */* ]]; then
    fail "origin URL does not identify one owner/repository"
  fi
  printf '%s/%s\n' "$owner" "$repository"
}

for required_command in git readlink; do
  command -v "$required_command" >/dev/null 2>&1 || {
    fail "missing command: $required_command"
  }
done
[[ -d "$source_repository" && ! -L "$source_repository" ]] || {
  fail "Harness source repository is missing or unsafe: $source_repository"
}
[[ -d "$project_root" && ! -L "$project_root" ]] || {
  fail "project root is missing or unsafe: $project_root"
}
[[ -f "$harness_wrapper" && -x "$harness_wrapper" && ! -L "$harness_wrapper" ]] || {
  fail "host-private Harness wrapper is missing or unsafe: $harness_wrapper"
}

resolved_source=$(git -C "$source_repository" rev-parse --show-toplevel)
resolved_project=$(git -C "$project_root" rev-parse --show-toplevel)
[[ "$resolved_source" == "$source_repository" ]] || {
  fail "Harness source path is not an exact Git root"
}
[[ "$resolved_project" == "$project_root" ]] || {
  fail "project path is not an exact Git root"
}
origin_url=$(git -C "$source_repository" remote get-url origin)
origin_identity=$(canonical_identity "$origin_url")
[[ "$origin_identity" == "$expected_identity" ]] || {
  fail "Harness source identity mismatch: $origin_identity"
}

printf '[remote-check] source=%s identity=%s\n' \
  "$source_repository" "$origin_identity"
printf '[remote-run] fetch refs/heads/%s into its remote-tracking ref\n' \
  "$protected_branch"
GCM_INTERACTIVE=never \
GIT_ASKPASS=/bin/false \
GIT_TERMINAL_PROMPT=0 \
SSH_ASKPASS=/bin/false \
git -c core.hooksPath=/dev/null -C "$source_repository" fetch \
  --no-tags origin \
  "refs/heads/$protected_branch:refs/remotes/origin/$protected_branch"

fetched_commit=$(
  git -C "$source_repository" rev-parse --verify \
    "refs/remotes/origin/$protected_branch^{commit}"
)
[[ "$fetched_commit" == "$expected_commit" ]] || {
  fail "fetched protected commit differs from expected: $fetched_commit"
}
printf '[remote-check] protected_commit=%s\n' "$fetched_commit"

common_arguments=(
  --project-root "$project_root"
  control
)
install_arguments=(
  --install-root "$install_root"
  --source-repository "$source_repository"
  --expected-identity "$expected_identity"
  --protected-branch "$protected_branch"
)

if [[ "$action" == "plan" ]]; then
  exec "$harness_wrapper" "${common_arguments[@]}" plan-install \
    "${install_arguments[@]}" --json
fi

"$harness_wrapper" "${common_arguments[@]}" install \
  "${install_arguments[@]}" \
  --confirm-plan-sha256 "$confirm_plan_sha256" \
  --json

expected_current="$install_root/versions/$expected_commit"
actual_current=$(readlink -f "$install_root/current")
[[ "$actual_current" == "$expected_current" ]] || {
  fail "active protected version differs from expected: $actual_current"
}
printf '[remote-check] active_protected_commit=%s\n' "$expected_commit"
"$harness_wrapper" "${common_arguments[@]}" status \
  --install-root "$install_root" --json
REMOTE
