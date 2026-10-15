#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

die() {
  printf '[foot-pad] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/start_foot_pad.sh

The launcher asks for the number of pedals (1-10), runs the full device
permission repair with sudo, recalibrates that many pedals, and keeps the
forwarding service in the foreground. Press Ctrl-C to stop the service.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    usage >&2
    die "this launcher does not accept arguments"
    ;;
esac

[[ "$EUID" -ne 0 ]] || die "run as your regular user; sudo is used only for the permission repair"
[[ -t 0 && -t 1 ]] || die "run from an interactive terminal so count and sudo prompts are available"
command -v sudo >/dev/null 2>&1 || die "sudo is required for the device permission repair"

cd -- "$PROJECT_ROOT"
if ! PYTHON_BIN="$(bash "$PROJECT_ROOT/scripts/resolve_dev_python.sh")"; then
  exit 1
fi
if ! "$PYTHON_BIN" -c 'import evdev' >/dev/null 2>&1; then
  die "evdev is missing from $PYTHON_BIN; install hardware/foot_pad/requirements.txt in the DexTeleop environment"
fi

while true; do
  if ! IFS= read -r -p '请输入踏板数量 N（1-10）: ' raw_count; then
    die "could not read the pedal count"
  fi
  if [[ "$raw_count" =~ ^[0-9]{1,2}$ ]]; then
    pedal_count=$((10#$raw_count))
    if ((pedal_count >= 1 && pedal_count <= 10)); then
      break
    fi
  fi
  printf '[foot-pad] 请输入 1 到 10 之间的整数。\n' >&2
done

printf '[foot-pad] 正在运行完整设备权限修复；sudo 会在需要时提示输入密码。\n'
sudo -E "$PYTHON_BIN" -m teleop_runtime.configure_devices --fix-permissions

printf '[foot-pad] 开始重新标定 %s 个踏板。按服务提示逐个踩下踏板；标定后转发服务会保持运行，按 Ctrl-C 停止。\n' "$pedal_count"
exec "$PYTHON_BIN" -m hardware.foot_pad --count "$pedal_count" --recalibrate
