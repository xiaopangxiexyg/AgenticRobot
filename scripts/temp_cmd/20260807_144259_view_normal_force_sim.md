cd /home/zc/HandMotion/Developing/DexTeleop && \
/home/zc/anaconda3/envs/dexteleop310/bin/python \
  -m teleop_runtime.debug.view_position_admittance_mujoco \
  --log logs/remote_position_admittance_20260807/normal \
  --admittance-envelope recorded-shadow \
  --interpolation zoh \
  --branch active \
  --speed 0.5 \
  --repeat 2 \
  --modifier-arrow-scale 1 \
  --physical-delta-arrow-scale 5 \
  --hold-viewer-at-end
