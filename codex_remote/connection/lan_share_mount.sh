#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SETUP_SCRIPT="$SCRIPT_DIR/setup_ssh_passwordless.sh"

DEFAULT_KEY_FILE="$HOME/.ssh/id_ed25519"
DEFAULT_LOCAL_DIR="/run/timeshift/backup/share"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/lan_share_mount"
STATE_FILE="$STATE_DIR/mounts.tsv"

REMOTE_HOST=""
REMOTE_USER=""
REMOTE_PORT="22"
KEY_FILE="$DEFAULT_KEY_FILE"
LOCAL_DIR=""
REMOTE_MOUNT_DIR=""
MAIN_HOST=""
PERSIST_REMOTE_FSTAB=false
LIST_ONLY=false
REMOTE_SUDO_PASSWORD=""
REMOTE_WRITE_VERIFIED="unknown"
ALLOW_CODEX=false

usage() {
  cat <<'EOF'
Usage:
  lan_share_mount.sh
  lan_share_mount.sh -l /run/timeshift/backup/share -H 192.168.1.20 -u zc -r /mnt/main-share
  lan_share_mount.sh -H zc@192.168.1.20:2222 -r '~/main-share'

Options:
  -l, --local DIR       Local shared directory. Default: /run/timeshift/backup/share.
  -H, --host HOST       Remote LAN address. Also accepts user@host or host:port.
  -u, --user USER       Remote SSH username. Default prompt value: current local user.
  -P, --port PORT       Remote SSH port. Default: 22.
  -r, --remote DIR      Remote mount directory.
  -k, --key FILE        Local SSH private key. Default: ~/.ssh/id_ed25519.
      --main-host HOST  Host/IP that remote should use to reach this main machine.
      --persist         Also add the NFS mount to the remote machine's /etc/fstab.
      --allow-codex     Allow sharing .codex directories if they exist under local DIR.
      --list            Only show recorded LAN mounts launched by this script.
  -h, --help            Show this help.

This script is run on the main machine. It exports a local directory with NFS,
then uses SSH to mount it on a remote Ubuntu/Linux machine.
If passwordless SSH to the remote is not ready, setup_ssh_passwordless.sh is run.
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
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

exports_quote_path() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

safe_state_field() {
  case "$1" in
    *$'\t'*|*$'\n'*) die "paths and host values cannot contain tabs or newlines: $1" ;;
  esac
}

confirm() {
  local prompt="$1"
  local answer

  while true; do
    IFS= read -r -p "$prompt" answer
    case "$answer" in
      [Yy]|[Yy][Ee][Ss]) return 0 ;;
      ""|[Nn]|[Nn][Oo]) return 1 ;;
      *) printf 'Please answer y or n.\n' >&2 ;;
    esac
  done
}

remote_target() {
  printf '%s@%s\n' "$REMOTE_USER" "$REMOTE_HOST"
}

build_ssh_command() {
  SSH_COMMAND=(ssh)
  if [[ -f "$KEY_FILE" ]]; then
    SSH_COMMAND+=(-i "$KEY_FILE")
  fi
  SSH_COMMAND+=(
    -p "$REMOTE_PORT"
    -o StrictHostKeyChecking=accept-new
    -o BatchMode=yes
    -o NumberOfPasswordPrompts=0
    "$(remote_target)"
  )
}

show_recorded_mounts() {
  local local_dir remote_user remote_host remote_port remote_dir main_host created_at
  local status source target fstype output
  local ssh_args

  printf '\nRecorded LAN mounts launched by this script:\n'

  if [[ ! -s "$STATE_FILE" ]]; then
    printf '  (none recorded yet)\n\n'
    return
  fi

  while IFS=$'\t' read -r local_dir remote_user remote_host remote_port remote_dir main_host created_at; do
    [[ -n "${local_dir:-}" ]] || continue
    status="unknown"
    source="${main_host}:${local_dir}"
    target="$remote_dir"
    fstype="nfs"

    if [[ -n "$remote_user" && -n "$remote_host" && -n "$remote_port" ]]; then
      ssh_args=(ssh)
      if [[ -f "$KEY_FILE" ]]; then
        ssh_args+=(-i "$KEY_FILE")
      fi
      ssh_args+=(
        -p "$remote_port"
        -o BatchMode=yes
        -o ConnectTimeout=3
        -o StrictHostKeyChecking=accept-new
        -o NumberOfPasswordPrompts=0
        "${remote_user}@${remote_host}"
      )

      output="$(
        "${ssh_args[@]}" \
          "findmnt -T $(shell_single_quote "$remote_dir") -n -o SOURCE,TARGET,FSTYPE 2>/dev/null || true" 2>/dev/null || true
      )"
      if [[ -n "$output" ]]; then
        source="$(awk 'NF >= 1 {print $1; exit}' <<< "$output")"
        target="$(awk 'NF >= 2 {print $2; exit}' <<< "$output")"
        fstype="$(awk 'NF >= 3 {print $3; exit}' <<< "$output")"
        if [[ "$source" == "${main_host}:${local_dir}" || "$source" == *":${local_dir}" ]]; then
          status="active"
        else
          status="occupied"
        fi
      fi
    fi

    printf '  [%s] local=%s  remote=%s@%s:%s  source=%s  type=%s\n' \
      "$status" "$local_dir" "$remote_user" "$remote_host" "$target" "$source" "$fstype"
  done < "$STATE_FILE"

  printf '\n'
}

prompt_connection_values() {
  local host_input default_user
  default_user="${USER:-$(whoami)}"

  if [[ -z "$LOCAL_DIR" ]]; then
    IFS= read -r -p "Local shared directory [$DEFAULT_LOCAL_DIR]: " LOCAL_DIR
    LOCAL_DIR="${LOCAL_DIR:-$DEFAULT_LOCAL_DIR}"
  fi

  if [[ -z "$REMOTE_HOST" ]]; then
    IFS= read -r -p "Target LAN address (user@host[:port] accepted): " host_input
    [[ -n "$host_input" ]] || die "target address cannot be empty"
    parse_host_input "$host_input"
  fi

  if [[ -z "$REMOTE_USER" ]]; then
    IFS= read -r -p "Remote username [$default_user]: " REMOTE_USER
    REMOTE_USER="${REMOTE_USER:-$default_user}"
  fi

  if [[ -z "$REMOTE_PORT" ]]; then
    IFS= read -r -p "Remote SSH port [22]: " REMOTE_PORT
    REMOTE_PORT="${REMOTE_PORT:-22}"
  fi

  if [[ -z "$REMOTE_MOUNT_DIR" ]]; then
    while [[ -z "$REMOTE_MOUNT_DIR" ]]; do
      IFS= read -r -p "Remote mount directory: " REMOTE_MOUNT_DIR
    done
  fi

  [[ -n "$REMOTE_HOST" ]] || die "target address cannot be empty"
  [[ -n "$REMOTE_USER" ]] || die "remote username cannot be empty"
  [[ "$REMOTE_PORT" =~ ^[0-9]+$ ]] || die "SSH port must be a number"

  validate_remote_mount_dir_argument
}

validate_remote_mount_dir_argument() {
  local local_user
  local_user="${USER:-$(id -un)}"

  if [[ "$REMOTE_USER" != "$local_user" ]] &&
    { [[ "$REMOTE_MOUNT_DIR" == "$HOME" ]] || [[ "$REMOTE_MOUNT_DIR" == "$HOME/"* ]]; }; then
    die "remote mount directory looks expanded by the local shell: $REMOTE_MOUNT_DIR
If you mean the remote user's home, quote the remote path, for example:
  -r '~/main-share'
Or use an explicit remote absolute path:
  -r /home/$REMOTE_USER/main-share"
  fi
}

check_passwordless_login() {
  local output

  build_ssh_command
  if output="$("${SSH_COMMAND[@]}" "printf '%s\n' passwordless-ok" 2>/dev/null)" &&
    [[ "$output" == *"passwordless-ok"* ]]; then
    return 0
  fi

  return 1
}

ensure_passwordless_login() {
  KEY_FILE="$(expand_path "$KEY_FILE")"

  if check_passwordless_login; then
    info "Passwordless SSH is ready: $(remote_target):$REMOTE_PORT"
    return
  fi

  warn "Passwordless SSH is not ready. Running setup_ssh_passwordless.sh."
  [[ -f "$SETUP_SCRIPT" ]] || die "cannot find setup script: $SETUP_SCRIPT"

  bash "$SETUP_SCRIPT" -H "$REMOTE_HOST" -u "$REMOTE_USER" -P "$REMOTE_PORT" -k "$KEY_FILE"

  if ! check_passwordless_login; then
    die "passwordless SSH setup finished, but verification still failed"
  fi

  info "Passwordless SSH is ready: $(remote_target):$REMOTE_PORT"
}

detect_main_host() {
  local detected

  if [[ -n "$MAIN_HOST" ]]; then
    return
  fi

  detected="$(ip route get "$REMOTE_HOST" 2>/dev/null | sed -n 's/.* src \([^ ]*\).*/\1/p' | head -n 1 || true)"
  [[ -n "$detected" ]] || die "could not detect the local IP used to reach $REMOTE_HOST; pass --main-host"
  MAIN_HOST="$detected"
}

ensure_local_share_dir() {
  LOCAL_DIR="$(expand_path "$LOCAL_DIR")"

  if [[ -e "$LOCAL_DIR" && ! -d "$LOCAL_DIR" ]]; then
    die "local shared path exists but is not a directory: $LOCAL_DIR"
  fi

  if [[ ! -e "$LOCAL_DIR" ]]; then
    info "Creating local shared directory: $LOCAL_DIR"
    sudo mkdir -p "$LOCAL_DIR"
    sudo chown "$(id -u):$(id -g)" "$LOCAL_DIR"
    sudo chmod 775 "$LOCAL_DIR"
  fi

  LOCAL_DIR="$(cd "$LOCAL_DIR" && pwd -P)"

  if [[ ! -w "$LOCAL_DIR" ]]; then
    warn "Current user cannot write to $LOCAL_DIR."
    warn "NFS clients will be mapped to $(id -un):$(id -gn), so remote writes may fail."
    if confirm "Change ownership recursively to $(id -un):$(id -gn)? [y/N]: "; then
      sudo chown -R "$(id -u):$(id -g)" "$LOCAL_DIR"
    fi
  fi
}

check_local_codex_exposure() {
  local found

  if [[ "$ALLOW_CODEX" == "true" ]]; then
    return 0
  fi

  found="$(find "$LOCAL_DIR" -type d -name .codex -print -quit 2>/dev/null || true)"
  [[ -n "$found" ]] || return 0

  warn "The local shared directory contains .codex: $found"
  warn "NFS mount exposes the real directory tree, so the remote machine could read or modify it."

  if [[ ! -t 0 ]]; then
    die "refusing to share .codex non-interactively. Use --allow-codex only if this is intentional."
  fi

  confirm "Continue and expose .codex to the remote machine? [y/N]: " ||
    die "aborted before sharing .codex. Share a narrower directory or rerun with --allow-codex if intentional."
}

ensure_local_nfs_server() {
  if ! command -v exportfs >/dev/null 2>&1; then
    info "Installing local NFS server package: nfs-kernel-server"
    command -v apt-get >/dev/null 2>&1 || die "exportfs is missing and apt-get is unavailable"
    sudo apt-get update
    sudo apt-get install -y nfs-kernel-server
  fi

  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl enable --now nfs-kernel-server >/dev/null
  fi
}

write_local_export() {
  local safe_host hash export_file quoted_path options

  safe_host="$(printf '%s' "$REMOTE_HOST" | tr -c 'A-Za-z0-9_.-' '_')"
  hash="$(printf '%s\t%s' "$LOCAL_DIR" "$REMOTE_HOST" | sha256sum | awk '{print substr($1, 1, 12)}')"
  export_file="/etc/exports.d/lan_share_mount_${safe_host}_${hash}.exports"
  quoted_path="$(exports_quote_path "$LOCAL_DIR")"
  options="rw,sync,no_subtree_check,all_squash,anonuid=$(id -u),anongid=$(id -g)"

  info "Writing NFS export for $REMOTE_HOST: $LOCAL_DIR"
  sudo mkdir -p /etc/exports.d
  {
    printf '# Managed by %s\n' "$(basename "$0")"
    printf '# Remote: %s@%s:%s\n' "$REMOTE_USER" "$REMOTE_HOST" "$REMOTE_MOUNT_DIR"
    printf '%s %s(%s)\n' "$quoted_path" "$REMOTE_HOST" "$options"
  } | sudo tee "$export_file" >/dev/null

  sudo exportfs -ra

  if command -v ufw >/dev/null 2>&1 && sudo ufw status 2>/dev/null | grep -qi '^Status: active'; then
    info "UFW is active; allowing NFS from $REMOTE_HOST"
    sudo ufw allow from "$REMOTE_HOST" to any port nfs >/dev/null
  fi
}

remote_sudo_command() {
  local command

  if [[ -n "$REMOTE_SUDO_PASSWORD" ]]; then
    command="sudo -k -S -p '' sh -s --"
  else
    command="sudo -n sh -s --"
  fi

  while [[ $# -gt 0 ]]; do
    command+=" $(shell_single_quote "$1")"
    shift
  done

  printf '%s\n' "$command"
}

ensure_remote_sudo() {
  build_ssh_command

  if "${SSH_COMMAND[@]}" "sudo -n true" >/dev/null 2>&1; then
    info "Remote sudo is available without an extra password prompt."
    REMOTE_SUDO_PASSWORD=""
    return
  fi

  IFS= read -r -s -p "Remote sudo password for $(remote_target): " REMOTE_SUDO_PASSWORD
  printf '\n'
  [[ -n "$REMOTE_SUDO_PASSWORD" ]] || die "remote sudo password cannot be empty"

  if ! printf '%s\n' "$REMOTE_SUDO_PASSWORD" |
    "${SSH_COMMAND[@]}" "sudo -k -S -p '' true" >/dev/null 2>&1; then
    REMOTE_SUDO_PASSWORD=""
    die "remote sudo verification failed; check the password or sudo privileges"
  fi

  info "Remote sudo is ready."
}

run_remote_sudo_script() {
  local command

  build_ssh_command
  command="$(remote_sudo_command "$@")"

  if [[ -n "$REMOTE_SUDO_PASSWORD" ]]; then
    { printf '%s\n' "$REMOTE_SUDO_PASSWORD"; cat; } | "${SSH_COMMAND[@]}" "$command"
  else
    "${SSH_COMMAND[@]}" "$command"
  fi
}

ensure_remote_nfs_client() {
  info "Checking remote NFS client tools."
  run_remote_sudo_script <<'REMOTE_SH'
set -eu

if command -v mount.nfs >/dev/null 2>&1 || [ -x /sbin/mount.nfs ] || [ -x /usr/sbin/mount.nfs ]; then
  exit 0
fi

if command -v apt-get >/dev/null 2>&1; then
  printf '[INFO] Installing remote package: nfs-common\n'
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y nfs-common
  exit 0
fi

printf 'Remote machine is missing mount.nfs and apt-get is unavailable.\n' >&2
exit 41
REMOTE_SH
}

mount_remote_nfs() {
  local source opts output status last_line state actual
  local persist_value

  source="${MAIN_HOST}:${LOCAL_DIR}"
  opts="rw,vers=4,_netdev,nofail"
  persist_value="$PERSIST_REMOTE_FSTAB"

  info "Mounting $source on $(remote_target):$REMOTE_MOUNT_DIR"
  set +e
  output="$(
    run_remote_sudo_script "$source" "$REMOTE_MOUNT_DIR" "$opts" "$persist_value" <<'REMOTE_SH'
set -eu

source=$1
target=$2
opts=$3
persist=$4

remote_user=${SUDO_USER:-}
if [ -n "$remote_user" ] && getent passwd "$remote_user" >/dev/null 2>&1; then
  remote_home=$(getent passwd "$remote_user" | cut -d: -f6)
else
  remote_home=${HOME:-/root}
fi

case "$target" in
  "") printf 'ERROR\tmissing remote mount directory\n'; exit 2 ;;
  "~") target=$remote_home ;;
  "~/"*) target=$remote_home/${target#~/} ;;
  /*) ;;
  *) target=$remote_home/$target ;;
esac

if [ -e "$target" ] && [ ! -d "$target" ]; then
  printf 'NOT_DIR\t%s\n' "$target"
  exit 3
fi

mkdir -p "$target"
actual=$(cd "$target" && pwd -P)

if mountpoint -q "$actual"; then
  current=$(findmnt -T "$actual" -n -o SOURCE 2>/dev/null || true)
  if [ "$current" = "$source" ]; then
    printf 'ALREADY\t%s\n' "$actual"
    exit 0
  fi

  printf 'BUSY\t%s is already mounted from %s\n' "$actual" "$current"
  exit 4
fi

mount -t nfs -o "$opts" "$source" "$actual"

if [ "$persist" = "true" ]; then
  line="$source $actual nfs $opts 0 0"
  touch /etc/fstab
  if ! grep -qxF "$line" /etc/fstab; then
    printf '%s\n' "$line" >> /etc/fstab
  fi
fi

printf 'MOUNTED\t%s\n' "$actual"
REMOTE_SH
  )"
  status=$?
  set -e

  last_line=""
  while IFS= read -r line; do
    [[ -n "$line" ]] && last_line="$line"
  done <<< "$output"

  state="${last_line%%$'\t'*}"
  actual="${last_line#*$'\t'}"

  case "$state" in
    MOUNTED)
      REMOTE_MOUNT_DIR="$actual"
      info "Remote mount created: $(remote_target):$REMOTE_MOUNT_DIR"
      ;;
    ALREADY)
      REMOTE_MOUNT_DIR="$actual"
      info "Remote mount already exists: $(remote_target):$REMOTE_MOUNT_DIR"
      ;;
    NOT_DIR)
      die "remote path exists but is not a directory: $actual"
      ;;
    BUSY)
      die "$actual"
      ;;
    *)
      printf 'Remote output:\n%s\n' "$output" >&2
      die "failed to mount remote NFS share (ssh exit status: $status)"
      ;;
  esac
}

verify_remote_user_access() {
  local test_file remote_cmd

  test_file="${REMOTE_MOUNT_DIR%/}/.lan_share_mount_write_test_${USER:-user}_$$"
  remote_cmd="test -d $(shell_single_quote "$REMOTE_MOUNT_DIR") && touch $(shell_single_quote "$test_file") && rm -f $(shell_single_quote "$test_file")"

  build_ssh_command
  if "${SSH_COMMAND[@]}" "$remote_cmd" >/dev/null 2>&1; then
    REMOTE_WRITE_VERIFIED="yes"
    info "Remote user write access verified."
  else
    REMOTE_WRITE_VERIFIED="no"
    warn "Remote user write test failed. Check local directory ownership/permissions and NFS export options."
  fi
}

record_mount_state() {
  local tmp created_at

  safe_state_field "$LOCAL_DIR"
  safe_state_field "$REMOTE_USER"
  safe_state_field "$REMOTE_HOST"
  safe_state_field "$REMOTE_PORT"
  safe_state_field "$REMOTE_MOUNT_DIR"
  safe_state_field "$MAIN_HOST"

  mkdir -p "$STATE_DIR"
  tmp="$(mktemp)"
  created_at="$(date '+%Y-%m-%d %H:%M:%S')"

  if [[ -f "$STATE_FILE" ]]; then
    awk -F '\t' \
      -v local_dir="$LOCAL_DIR" \
      -v remote_user="$REMOTE_USER" \
      -v remote_host="$REMOTE_HOST" \
      -v remote_port="$REMOTE_PORT" \
      -v remote_dir="$REMOTE_MOUNT_DIR" \
      '!(($1 == local_dir) && ($2 == remote_user) && ($3 == remote_host) && ($4 == remote_port) && ($5 == remote_dir))' \
      "$STATE_FILE" > "$tmp"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$LOCAL_DIR" "$REMOTE_USER" "$REMOTE_HOST" "$REMOTE_PORT" "$REMOTE_MOUNT_DIR" "$MAIN_HOST" "$created_at" >> "$tmp"
  mv "$tmp" "$STATE_FILE"
}

print_summary() {
  printf '\nLAN share mount is ready.\n'
  printf '  Local real path : %s\n' "$LOCAL_DIR"
  printf '  Main host/IP    : %s\n' "$MAIN_HOST"
  printf '  Remote mount    : %s@%s:%s\n' "$REMOTE_USER" "$REMOTE_HOST" "$REMOTE_MOUNT_DIR"
  case "$REMOTE_WRITE_VERIFIED" in
    yes)
      printf '  Remote can edit : verified, mapped to local uid=%s gid=%s through NFS all_squash\n' "$(id -u)" "$(id -g)"
      ;;
    no)
      printf '  Remote can edit : write test failed; inspect permissions before relying on it\n'
      ;;
    *)
      printf '  Remote can edit : not verified\n'
      ;;
  esac
  if [[ "$PERSIST_REMOTE_FSTAB" == "true" ]]; then
    printf '  Remote fstab    : updated\n'
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -l|--local)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        LOCAL_DIR="$2"
        shift 2
        ;;
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
      -r|--remote)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        REMOTE_MOUNT_DIR="$2"
        shift 2
        ;;
      -k|--key)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        KEY_FILE="$(expand_path "$2")"
        shift 2
        ;;
      --main-host)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        MAIN_HOST="$2"
        shift 2
        ;;
      --persist)
        PERSIST_REMOTE_FSTAB=true
        shift
        ;;
      --allow-codex)
        ALLOW_CODEX=true
        shift
        ;;
      --list)
        LIST_ONLY=true
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
}

main() {
  parse_args "$@"

  require_cmd ssh
  require_cmd sudo
  require_cmd sed
  require_cmd awk
  require_cmd ip
  require_cmd sha256sum
  require_cmd mktemp

  KEY_FILE="$(expand_path "$KEY_FILE")"

  show_recorded_mounts
  if [[ "$LIST_ONLY" == "true" ]]; then
    exit 0
  fi

  prompt_connection_values
  ensure_passwordless_login
  detect_main_host
  ensure_local_share_dir
  check_local_codex_exposure
  ensure_local_nfs_server
  write_local_export
  ensure_remote_sudo
  ensure_remote_nfs_client
  mount_remote_nfs
  verify_remote_user_access
  record_mount_state
  print_summary
}

main "$@"
