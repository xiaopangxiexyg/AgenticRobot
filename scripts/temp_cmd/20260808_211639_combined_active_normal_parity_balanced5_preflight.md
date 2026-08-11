(
set -euo pipefail
cd /home/zc/HandMotion/Developing/DexTeleop
./codex_remote/remote_safety_check.sh \
  -H dex-remote \
  --project /home/user/zc/DexTeleop \
  --arm-ip 192.168.1.20 \
  --arm-port 8080 \
  --token-path /tmp/codex_normal_parity_balanced5_right_safety_ok
./codex_remote/remote_safety_check.sh \
  -H dex-remote \
  --project /home/user/zc/DexTeleop \
  --arm-ip 192.168.101.21 \
  --arm-port 8080 \
  --token-path /tmp/codex_normal_parity_balanced5_left_safety_ok
)
