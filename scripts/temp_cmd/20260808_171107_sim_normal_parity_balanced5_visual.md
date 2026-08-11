cd /home/zc/HandMotion/Developing/DexTeleop && \
exec /home/zc/anaconda3/envs/dexteleop310/bin/python \
  -m teleop_runtime.debug.view_position_admittance_mujoco \
  --log logs/remote_position_admittance_20260807/normal/action_debug_20260807_121302_924.jsonl \
  --side left \
  --admittance-envelope normal-parity-offline \
  --rotation-envelope balanced-5deg \
  --interpolation zoh \
  --branch active \
  --speed 0.5 \
  --repeat 1 \
  --visible-release-tail-sec 2 \
  --status-hz 5 \
  --modifier-arrow-scale 1 \
  --physical-delta-arrow-scale 5 \
  --hold-viewer-at-end
