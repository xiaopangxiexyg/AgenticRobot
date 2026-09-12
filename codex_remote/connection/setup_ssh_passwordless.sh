#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_NAME="$(basename "$0")"
DEFAULT_KEY_FILE="$HOME/.ssh/id_ed25519"

REMOTE_HOST=""
REMOTE_USER=""
REMOTE_PORT="22"
KEY_FILE="$DEFAULT_KEY_FILE"

usage() {
  cat <<'EOF'
Usage:
  setup_ssh_passwordless.sh
  setup_ssh_passwordless.sh -H 192.168.1.20 -u zc
  setup_ssh_passwordless.sh -H 192.168.1.20 -u zc -P 2222
  setup_ssh_passwordless.sh -H zc@192.168.1.20:2222

Options:
  -H, --host HOST        Remote LAN address. Also accepts user@host or host:port.
  -u, --user USER        Remote SSH username.
  -P, --port PORT        Remote SSH port. Default: 22.
  -k, --key FILE         Local SSH private key. Default: ~/.ssh/id_ed25519.
  -h, --help             Show this help.

The remote password is requested silently and is not written to disk. The host
key must already be verified and present in known_hosts; this script never
accepts a new host key.
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '[INFO] %s\n' "$*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

expand_path() {
  case "$1" in
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${1#"~/"}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

parse_host_input() {
  local input="$1"

  if [[ "$input" == *@* && -z "$REMOTE_USER" ]]; then
    REMOTE_USER="${input%@*}"
    input="${input#*@}"
  fi

  if [[ "$input" != *:*:* && "$input" == *:* ]]; then
    REMOTE_PORT="${input##*:}"
    input="${input%:*}"
  fi

  REMOTE_HOST="$input"
}

shell_single_quote() {
  printf "'"
  printf '%s' "$1" | sed "s/'/'\\\\''/g"
  printf "'"
}

ensure_key_pair() {
  local key_dir
  key_dir="$(dirname "$KEY_FILE")"

  mkdir -p "$key_dir"
  chmod 700 "$key_dir"

  if [[ ! -f "$KEY_FILE" ]]; then
    info "No SSH key found at $KEY_FILE; generating an ed25519 key."
    ssh-keygen -t ed25519 -f "$KEY_FILE" -N "" -C "$(whoami)@$(hostname)-lan" >/dev/null
  else
    info "Using existing SSH key: $KEY_FILE"
  fi

  if [[ ! -f "$KEY_FILE.pub" ]]; then
    info "Public key is missing; rebuilding $KEY_FILE.pub from the private key."
    ssh-keygen -y -f "$KEY_FILE" > "$KEY_FILE.pub"
  fi

  chmod 600 "$KEY_FILE"
  chmod 644 "$KEY_FILE.pub"
}

run_remote_install_with_sshpass() {
  local password="$1"
  shift

  SSHPASS="$password" sshpass -e "$@"
}

run_remote_install_with_expect() {
  local password="$1"
  shift

  REMOTE_PASSWORD="$password" expect -f - "$@" <<'EXPECT_EOF'
set timeout 30
set password $env(REMOTE_PASSWORD)
unset env(REMOTE_PASSWORD)
spawn {*}$argv
expect {
  -re {(?i)(password|verification code).*:} {
    send -- "$password\r"
    exp_continue
  }
  eof {
    catch wait result
    exit [lindex $result 3]
  }
  timeout {
    exit 124
  }
}
EXPECT_EOF
}

run_remote_install_with_python() {
  local password="$1"
  shift

  REMOTE_PASSWORD="$password" python3 - "$@" <<'PY_EOF'
import errno
import os
import pty
import select
import signal
import sys

password = os.environ.get("REMOTE_PASSWORD", "")
os.environ.pop("REMOTE_PASSWORD", None)
argv = sys.argv[1:]
if not argv:
    print("missing ssh command", file=sys.stderr)
    sys.exit(2)

pid, fd = pty.fork()
if pid == 0:
    os.execvp(argv[0], argv)

buffer = ""
while True:
    ready, _, _ = select.select([fd], [], [], 30)
    if not ready:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        print("\nError: timeout waiting for ssh.", file=sys.stderr)
        sys.exit(124)

    try:
        data = os.read(fd, 1024)
    except OSError as exc:
        if exc.errno == errno.EIO:
            break
        raise

    if not data:
        break

    text = data.decode(errors="replace")
    sys.stdout.write(text)
    sys.stdout.flush()

    buffer = (buffer + text)[-512:]
    lower = buffer.lower()
    if "password:" in lower or "verification code:" in lower:
        os.write(fd, (password + "\r").encode())
        buffer = ""

_, status = os.waitpid(pid, 0)
if os.WIFEXITED(status):
    sys.exit(os.WEXITSTATUS(status))
if os.WIFSIGNALED(status):
    sys.exit(128 + os.WTERMSIG(status))
sys.exit(1)
PY_EOF
}

install_public_key() {
  local password="$1"
  local public_key quoted_key remote_cmd target
  public_key="$(<"$KEY_FILE.pub")"
  quoted_key="$(shell_single_quote "$public_key")"
  target="${REMOTE_USER}@${REMOTE_HOST}"

  remote_cmd="KEY=${quoted_key}; umask 077; mkdir -p \"\$HOME/.ssh\" || exit 1; touch \"\$HOME/.ssh/authorized_keys\" || exit 1; chmod 700 \"\$HOME/.ssh\"; chmod 600 \"\$HOME/.ssh/authorized_keys\"; if grep -qxF \"\$KEY\" \"\$HOME/.ssh/authorized_keys\"; then printf '%s\n' already-present; else printf '%s\n' \"\$KEY\" >> \"\$HOME/.ssh/authorized_keys\" && printf '%s\n' installed; fi"

  local ssh_cmd=(
    ssh
    -p "$REMOTE_PORT"
    -o ConnectTimeout=10
    -o StrictHostKeyChecking=yes
    -o PubkeyAuthentication=no
    -o PreferredAuthentications=password,keyboard-interactive
    -o NumberOfPasswordPrompts=1
    "$target"
    "$remote_cmd"
  )

  info "Installing public key on $target:$REMOTE_PORT"

  if command -v sshpass >/dev/null 2>&1; then
    run_remote_install_with_sshpass "$password" "${ssh_cmd[@]}"
  elif command -v expect >/dev/null 2>&1; then
    run_remote_install_with_expect "$password" "${ssh_cmd[@]}"
  elif command -v python3 >/dev/null 2>&1; then
    run_remote_install_with_python "$password" "${ssh_cmd[@]}"
  else
    die "need sshpass, expect, or python3 to automate the first password login. Install one, then rerun this script."
  fi
}

verify_passwordless_login() {
  local target output
  target="${REMOTE_USER}@${REMOTE_HOST}"

  info "Verifying passwordless login."
  if output="$(
    ssh \
      -i "$KEY_FILE" \
      -p "$REMOTE_PORT" \
      -o BatchMode=yes \
      -o ConnectTimeout=10 \
      -o StrictHostKeyChecking=yes \
      "$target" \
      "printf '%s\n' passwordless-ok" 2>&1
  )" && [[ "$output" == "passwordless-ok" ]]; then
    printf '\nSuccess. Passwordless SSH is ready:\n'
    if [[ "$REMOTE_PORT" == "22" && "$KEY_FILE" == "$DEFAULT_KEY_FILE" ]]; then
      printf '  ssh %s@%s\n' "$REMOTE_USER" "$REMOTE_HOST"
    else
      printf '  ssh -i %q -p %q %s@%s\n' "$KEY_FILE" "$REMOTE_PORT" "$REMOTE_USER" "$REMOTE_HOST"
    fi
  else
    printf '\nPublic key was installed, but verification failed.\n' >&2
    printf 'Try manually:\n  ssh -i %q -p %q %s@%s\n' "$KEY_FILE" "$REMOTE_PORT" "$REMOTE_USER" "$REMOTE_HOST" >&2
    printf '\nSSH output:\n%s\n' "$output" >&2
    exit 1
  fi
}

prompt_missing_values() {
  local host_input password password_confirm

  if [[ -z "$REMOTE_HOST" ]]; then
    IFS= read -r -p "Remote LAN address: " host_input
    [[ -n "$host_input" ]] || die "remote address cannot be empty"
    parse_host_input "$host_input"
  fi

  if [[ -z "$REMOTE_USER" ]]; then
    IFS= read -r -p "Remote username: " REMOTE_USER
  fi

  if [[ -z "$REMOTE_PORT" ]]; then
    IFS= read -r -p "Remote SSH port [22]: " REMOTE_PORT
    REMOTE_PORT="${REMOTE_PORT:-22}"
  fi

  [[ -n "$REMOTE_HOST" ]] || die "remote address cannot be empty"
  [[ -n "$REMOTE_USER" ]] || die "remote username cannot be empty"
  [[ "$REMOTE_PORT" =~ ^[0-9]+$ ]] || die "SSH port must be a number"

  if [[ -n "${REMOTE_PASSWORD:-}" ]]; then
    password="$REMOTE_PASSWORD"
  else
    IFS= read -r -s -p "Remote password: " password
    printf '\n'
  fi

  [[ -n "$password" ]] || die "remote password cannot be empty"

  if [[ -n "${CONFIRM_REMOTE_PASSWORD:-}" ]]; then
    password_confirm="$CONFIRM_REMOTE_PASSWORD"
  else
    password_confirm="$password"
  fi

  [[ "$password" == "$password_confirm" ]] || die "password confirmation does not match"
  REMOTE_PASSWORD="$password"
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -H|--host)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        parse_host_input "$2"
        shift 2
        ;;
      -u|--user)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        REMOTE_USER="$2"
        shift 2
        ;;
      -P|--port)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        REMOTE_PORT="$2"
        shift 2
        ;;
      -k|--key)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        KEY_FILE="$(expand_path "$2")"
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

  require_cmd ssh
  require_cmd ssh-keygen
  require_cmd sed

  KEY_FILE="$(expand_path "$KEY_FILE")"

  prompt_missing_values
  ensure_key_pair
  install_public_key "$REMOTE_PASSWORD"
  unset REMOTE_PASSWORD
  verify_passwordless_login
}

main "$@"
