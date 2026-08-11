cd /home/zc/HandMotion/Developing/DexTeleop && \
exec /home/zc/anaconda3/envs/dexteleop310/bin/python \
  -m teleop_runtime.debug.replay_position_rotational_admittance_mujoco \
  --log /tmp/dexteleop_rotation_wrench_20260807/action_debug_20260807_223418_219.jsonl \
  --side left \
  --envelope balanced-5deg \
  --replay-hz 60 \
  --release-tail-sec 10 \
  --interpolation zoh \
  --offline-combined-normal-parity \
  > /tmp/dexteleop_sim_normal_parity_rotation_wrench_stress_20260808.json
