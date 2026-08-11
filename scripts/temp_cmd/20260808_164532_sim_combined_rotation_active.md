cd /home/zc/HandMotion/Developing/DexTeleop && \
exec /home/zc/anaconda3/envs/dexteleop310/bin/python \
  -m teleop_runtime.debug.replay_position_rotational_admittance_mujoco \
  --log /tmp/dexteleop_rotation_active_20260808/action_debug_20260808_114516_024.jsonl \
  --side left \
  --envelope cautious-3deg \
  --replay-hz 60 \
  --release-tail-sec 10 \
  --interpolation both \
  --combined-translation-rotation-active \
  > /tmp/dexteleop_sim_combined_rotation_active_20260808.json
