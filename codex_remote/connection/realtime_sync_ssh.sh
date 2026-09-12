#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SETUP_SCRIPT="$SCRIPT_DIR/setup_ssh_passwordless.sh"
DEFAULT_KEY_FILE="$HOME/.ssh/id_ed25519"
DEFAULT_EXCLUDE_FILE_NAME=".realtime_sync_ssh.exclude"

REMOTE_HOST=""
REMOTE_USER=""
REMOTE_PORT="22"
KEY_FILE="$DEFAULT_KEY_FILE"
LOCAL_DIR=""
REMOTE_DIR=""
DELETE=false
EXCLUDES=(".codex/")
EXCLUDE_FILES=()

usage() {
  cat <<'EOF'
Usage:
  realtime_sync_ssh.sh
  realtime_sync_ssh.sh -H 192.168.1.20 -u zc -l /local/dir -r /remote/dir
  realtime_sync_ssh.sh -H zc@192.168.1.20:2222 -l ~/Release -r '~/Release'

Options:
  -H, --host HOST        Remote LAN address. Also accepts user@host or host:port.
  -u, --user USER        Remote SSH username.
  -P, --port PORT        Remote SSH port. Default: 22.
  -k, --key FILE         Local SSH private key. Default: ~/.ssh/id_ed25519.
  -l, --local DIR        Local source directory.
  -r, --remote DIR       Remote target directory.
      --delete           Delete remote-only files and make remote mirror local exactly.
      --no-delete        Keep remote-only files. This is the default.
      --exclude PATTERN  Exclude an rsync pattern. Can be repeated. Default excludes: .codex/
      --exclude-from FILE
                          Load exclude patterns from FILE. Can be repeated.
                          Blank lines and lines starting with # are ignored.
  -h, --help             Show this help.

This is a one-way live mirror: local directory -> remote directory.
If passwordless SSH is not ready, this script runs setup_ssh_passwordless.sh.
The host key must already be verified and present in known_hosts.
If LOCAL/.realtime_sync_ssh.exclude exists, it is loaded automatically.
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
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1. Try: sudo apt install lsyncd rsync"
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

lua_quote() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  printf '"%s"' "$value"
}

ensure_trailing_slash() {
  if [[ "$1" == "/" ]]; then
    printf '/\n'
  else
    printf '%s/\n' "${1%/}"
  fi
}

load_exclude_file() {
  local file="$1"
  local line pattern count=0

  file="$(expand_path "$file")"
  [[ -f "$file" ]] || die "exclude file does not exist: $file"
  while IFS= read -r line || [[ -n "$line" ]]; do
    pattern="${line%$'\r'}"
    [[ "$pattern" =~ ^[[:space:]]*$ ]] && continue
    [[ "$pattern" =~ ^[[:space:]]*# ]] && continue
    EXCLUDES+=("$pattern")
    ((count += 1))
  done < "$file"
  info "Loaded $count exclude pattern(s) from: $file"
}

load_exclude_files() {
  local default_file="$LOCAL_DIR/$DEFAULT_EXCLUDE_FILE_NAME"
  local file pattern

  if [[ -f "$default_file" ]]; then
    load_exclude_file "$default_file"
  fi
  for file in "${EXCLUDE_FILES[@]}"; do
    load_exclude_file "$file"
  done
  info "Active exclude patterns:"
  for pattern in "${EXCLUDES[@]}"; do
    printf '  - %s\n' "$pattern"
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
    -o StrictHostKeyChecking=yes
    -o BatchMode=yes
    -o NumberOfPasswordPrompts=0
    "$(remote_target)"
  )
}

build_rsync_rsh() {
  local rsh="ssh"

  if [[ -f "$KEY_FILE" ]]; then
    rsh+=" -i $(shell_single_quote "$KEY_FILE")"
  fi

  rsh+=" -p $(shell_single_quote "$REMOTE_PORT")"
  rsh+=" -o StrictHostKeyChecking=yes"
  rsh+=" -o BatchMode=yes"
  rsh+=" -o NumberOfPasswordPrompts=0"

  printf '%s\n' "$rsh"
}

prompt_connection_values() {
  local host_input

  if [[ -z "$REMOTE_HOST" ]]; then
    IFS= read -r -p "Remote LAN address (user@host[:port] accepted): " host_input
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
}

prompt_sync_dirs() {
  while [[ -z "$LOCAL_DIR" ]]; do
    IFS= read -r -p "Local source directory: " LOCAL_DIR
  done

  LOCAL_DIR="$(expand_path "$LOCAL_DIR")"
  [[ -d "$LOCAL_DIR" ]] || die "local directory does not exist: $LOCAL_DIR"
  LOCAL_DIR="$(cd "$LOCAL_DIR" && pwd -P)"

  while [[ -z "$REMOTE_DIR" ]]; do
    IFS= read -r -p "Remote target directory: " REMOTE_DIR
  done

  validate_remote_dir_argument
}

validate_remote_dir_argument() {
  local local_user
  local_user="${USER:-$(id -un)}"

  if [[ "$REMOTE_USER" != "$local_user" ]] &&
    { [[ "$REMOTE_DIR" == "$HOME" ]] || [[ "$REMOTE_DIR" == "$HOME/"* ]]; }; then
    die "remote directory looks expanded by the local shell: $REMOTE_DIR
If you mean the remote user's home, quote the remote path, for example:
  -r '~/zc/DexTeleop'
Or use an explicit remote absolute path:
  -r /home/$REMOTE_USER/zc/DexTeleop"
  fi
}

count_local_directories() {
  local count
  count="$(find "$LOCAL_DIR" -type d -print 2>/dev/null | wc -l)"
  count="${count//[[:space:]]/}"
  printf '%s\n' "${count:-0}"
}

ensure_inotify_limits() {
  local watches_path="/proc/sys/fs/inotify/max_user_watches"
  local instances_path="/proc/sys/fs/inotify/max_user_instances"
  local events_path="/proc/sys/fs/inotify/max_queued_events"
  local current_watches current_instances current_events
  local dir_count required_watches recommended_watches recommended_instances recommended_events
  local answer need_update

  [[ -r "$watches_path" && -r "$instances_path" && -r "$events_path" ]] || return 0

  current_watches="$(<"$watches_path")"
  current_instances="$(<"$instances_path")"
  current_events="$(<"$events_path")"
  [[ "$current_watches" =~ ^[0-9]+$ ]] || return 0
  [[ "$current_instances" =~ ^[0-9]+$ ]] || return 0
  [[ "$current_events" =~ ^[0-9]+$ ]] || return 0

  dir_count="$(count_local_directories)"
  [[ "$dir_count" =~ ^[0-9]+$ ]] || return 0

  required_watches=$((dir_count * 2 + 1024))
  recommended_watches=$((dir_count * 4 + 4096))
  if (( recommended_watches < 524288 )); then
    recommended_watches=524288
  fi
  if (( recommended_watches < current_watches )); then
    recommended_watches=$current_watches
  fi

  recommended_instances=1024
  if (( recommended_instances < current_instances )); then
    recommended_instances=$current_instances
  fi

  recommended_events=32768
  if (( recommended_events < current_events )); then
    recommended_events=$current_events
  fi

  need_update=false
  if (( current_watches < required_watches )); then
    need_update=true
  fi
  if (( current_instances < 1024 )); then
    need_update=true
  fi
  if (( current_events < 32768 )); then
    need_update=true
  fi

  if [[ "$need_update" == "false" ]]; then
    return 0
  fi

  warn "Local inotify limits are likely too low for lsyncd."
  printf '  Local directories              : %s\n' "$dir_count" >&2
  printf '  fs.inotify.max_user_watches    : %s -> %s\n' "$current_watches" "$recommended_watches" >&2
  printf '  fs.inotify.max_user_instances  : %s -> %s\n' "$current_instances" "$recommended_instances" >&2
  printf '  fs.inotify.max_queued_events   : %s -> %s\n' "$current_events" "$recommended_events" >&2

  if [[ ! -t 0 ]]; then
    die "increase it first, then rerun:
  sudo sysctl -w fs.inotify.max_user_watches=$recommended_watches fs.inotify.max_user_instances=$recommended_instances fs.inotify.max_queued_events=$recommended_events"
  fi

  IFS= read -r -p "Increase local inotify limits now? [y/N]: " answer
  case "$answer" in
    [Yy]|[Yy][Ee][Ss])
      sudo sysctl -w \
        "fs.inotify.max_user_watches=$recommended_watches" \
        "fs.inotify.max_user_instances=$recommended_instances" \
        "fs.inotify.max_queued_events=$recommended_events" >/dev/null
      info "Updated local inotify limits for this boot."
      ;;
    *)
      die "lsyncd may fail until you increase the inotify limits:
  sudo sysctl -w fs.inotify.max_user_watches=$recommended_watches fs.inotify.max_user_instances=$recommended_instances fs.inotify.max_queued_events=$recommended_events"
      ;;
  esac
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

confirm_remote_overwrite() {
  local answer

  if [[ "$DELETE" == "true" ]]; then
    printf 'Remote target already has files. Continuing will mirror local -> remote and delete remote-only files.\n' >&2
  else
    printf 'Remote target already has files. Continuing may overwrite files with matching paths.\n' >&2
  fi

  while true; do
    IFS= read -r -p "Continue? [y/N]: " answer
    case "$answer" in
      [Yy]|[Yy][Ee][Ss]) return 0 ;;
      ""|[Nn]|[Nn][Oo]) return 1 ;;
      *) printf 'Please answer y or n.\n' >&2 ;;
    esac
  done
}

prepare_remote_target_dir() {
  local output status last_line state actual line

  set +e
  output="$(
    build_ssh_command
    "${SSH_COMMAND[@]}" "sh -s" -- "$REMOTE_DIR" <<'REMOTE_SH'
set -eu

target=$1
case "$target" in
  "") printf 'ERROR\tmissing target directory\n'; exit 2 ;;
  "~") target=$HOME ;;
  "~/"*) target=$HOME/${target#~/} ;;
  /*) ;;
  *) target=$HOME/$target ;;
esac

if ! command -v rsync >/dev/null 2>&1; then
  printf 'MISSING_RSYNC\t%s\n' "$target"
  exit 4
fi

if [ -e "$target" ] && [ ! -d "$target" ]; then
  printf 'NOT_DIR\t%s\n' "$target"
  exit 3
fi

if [ ! -e "$target" ]; then
  mkdir -p "$target"
  actual=$(cd "$target" && pwd -P)
  printf 'CREATED\t%s\n' "$actual"
  exit 0
fi

actual=$(cd "$target" && pwd -P)
if find "$actual" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  printf 'NONEMPTY\t%s\n' "$actual"
else
  printf 'EMPTY\t%s\n' "$actual"
fi
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
    CREATED)
      REMOTE_DIR="$actual"
      info "Created remote target directory: $REMOTE_DIR"
      ;;
    EMPTY)
      REMOTE_DIR="$actual"
      info "Remote target directory is empty: $REMOTE_DIR"
      ;;
    NONEMPTY)
      REMOTE_DIR="$actual"
      warn "Remote target directory is not empty: $REMOTE_DIR"
      confirm_remote_overwrite || die "aborted before touching remote target"
      ;;
    MISSING_RSYNC)
      die "remote machine is missing rsync. Install it there first, for example: sudo apt install rsync"
      ;;
    NOT_DIR)
      die "remote target exists but is not a directory: $actual"
      ;;
    *)
      printf 'Remote output:\n%s\n' "$output" >&2
      die "failed to prepare remote target directory (ssh exit status: $status)"
      ;;
  esac
}

run_initial_rsync() {
  local source_dir target_dir rsh
  local args

  source_dir="$(ensure_trailing_slash "$LOCAL_DIR")"
  target_dir="$(remote_target):$(ensure_trailing_slash "$REMOTE_DIR")"
  rsh="$(build_rsync_rsh)"

  args=(rsync -az -s --no-owner --no-group)
  if [[ "$DELETE" == "true" ]]; then
    args+=(--delete)
  fi
  args+=(-e "$rsh")

  local pattern
  for pattern in "${EXCLUDES[@]}"; do
    args+=(--exclude="$pattern")
  done

  args+=("$source_dir" "$target_dir")

  info "Running initial rsync."
  "${args[@]}"
}

write_lsyncd_config() {
  local config_file="$1"
  local source_dir target_dir host rsh log_file status_file
  local pattern

  source_dir="$(ensure_trailing_slash "$LOCAL_DIR")"
  target_dir="$(ensure_trailing_slash "$REMOTE_DIR")"
  host="$(remote_target)"
  rsh="$(build_rsync_rsh)"
  log_file="/tmp/realtime_sync_ssh_${USER:-user}_$$.log"
  status_file="/tmp/realtime_sync_ssh_${USER:-user}_$$.status"

  {
    printf 'settings {\n'
    printf '  logfile = %s,\n' "$(lua_quote "$log_file")"
    printf '  statusFile = %s,\n' "$(lua_quote "$status_file")"
    printf '  statusInterval = 10,\n'
    printf '}\n\n'
    printf 'sync {\n'
    printf '  default.rsyncssh,\n'
    printf '  source = %s,\n' "$(lua_quote "$source_dir")"
    printf '  host = %s,\n' "$(lua_quote "$host")"
    printf '  targetdir = %s,\n' "$(lua_quote "$target_dir")"
    printf '  delay = 1,\n'
    printf '  delete = %s,\n' "$DELETE"
    printf '  rsync = {\n'
    printf '    archive = true,\n'
    printf '    compress = false,\n'
    printf '    rsh = %s,\n' "$(lua_quote "$rsh")"
    printf '    _extra = {\n'
    printf '      "--protect-args",\n'
    printf '      "--no-owner",\n'
    printf '      "--no-group"'
    for pattern in "${EXCLUDES[@]}"; do
      printf ',\n      %s' "$(lua_quote "--exclude=$pattern")"
    done
    printf '\n'
    printf '    }\n'
    printf '  }\n'
    printf '}\n'
  } > "$config_file"

  info "Lsyncd log: $log_file"
  info "Lsyncd status: $status_file"
}

start_lsyncd() {
  local config_file

  config_file="$(mktemp /tmp/realtime_sync_ssh.XXXXXX.lua)"
  trap 'rm -f "$config_file"' EXIT

  write_lsyncd_config "$config_file"

  printf '\nLive sync started. Press Ctrl-C to stop.\n'
  printf '  Local : %s\n' "$LOCAL_DIR"
  printf '  Remote: %s:%s\n\n' "$(remote_target)" "$REMOTE_DIR"

  lsyncd -nodaemon "$config_file"
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
      -l|--local)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        LOCAL_DIR="$2"
        shift 2
        ;;
      -r|--remote)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        REMOTE_DIR="$2"
        shift 2
        ;;
      --delete)
        DELETE=true
        shift
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
      --exclude-from)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        EXCLUDE_FILES+=("$2")
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
  require_cmd rsync
  require_cmd lsyncd
  require_cmd mktemp
  require_cmd sed
  require_cmd find

  prompt_connection_values
  prompt_sync_dirs
  load_exclude_files
  ensure_inotify_limits
  ensure_passwordless_login
  prepare_remote_target_dir
  run_initial_rsync
  start_lsyncd
}

main "$@"
