set -euo pipefail
cd /home/zc/HandMotion/Developing/DexTeleop

printf '%s\n' '[1/2] normal 15hz-40hz parity shadow; close the viewer to continue'
/home/zc/anaconda3/envs/dexteleop310/bin/python \
  -m teleop_runtime.debug.view_position_admittance_mujoco \
  --log logs/remote_position_admittance_20260807/normal/action_debug_20260807_121302_924.jsonl \
  --side left \
  --admittance-envelope recorded-shadow \
  --interpolation zoh \
  --branch active \
  --speed 0.5 \
  --repeat 1 \
  --modifier-arrow-scale 1 \
  --physical-delta-arrow-scale 5 \
  --hold-viewer-at-end

printf '%s\n' '[2/2] cautious shadow; close the viewer to finish'
/home/zc/anaconda3/envs/dexteleop310/bin/python \
  -m teleop_runtime.debug.view_position_admittance_mujoco \
  --log logs/remote_position_admittance_20260807/cautious/action_debug_20260807_121215_068.jsonl \
  --side left \
  --admittance-envelope recorded-shadow \
  --interpolation zoh \
  --branch active \
  --speed 0.5 \
  --repeat 1 \
  --modifier-arrow-scale 30 \
  --physical-delta-arrow-scale 30 \
  --hold-viewer-at-end
