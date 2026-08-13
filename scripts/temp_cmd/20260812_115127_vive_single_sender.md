(
set -euo pipefail
cd /home/zc/HandMotion/Developing/DexTeleop
SIM_HIL_RECEIVER_IP="${SIM_HIL_RECEIVER_IP:-127.0.0.1}"
/home/zc/anaconda3/envs/dexteleop310/bin/python hardware/VIVE/mocap/vive_data_sender.py \
  --single \
  --host "$SIM_HIL_RECEIVER_IP" \
  --port 9999 \
  --hz 300
)
