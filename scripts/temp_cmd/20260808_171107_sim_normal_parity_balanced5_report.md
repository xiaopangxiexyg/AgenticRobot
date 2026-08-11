cd /home/zc/HandMotion/Developing/DexTeleop && \
exec /home/zc/anaconda3/envs/dexteleop310/bin/python \
  -m teleop_runtime.debug.replay_position_rotational_admittance_mujoco \
  --log logs/remote_position_admittance_20260807/normal/action_debug_20260807_121302_924.jsonl \
  --side left \
  --envelope balanced-5deg \
  --replay-hz 60 \
  --release-tail-sec 10 \
  --interpolation both \
  --offline-combined-normal-parity \
  > /tmp/dexteleop_sim_normal_parity_balanced5_both_20260808.json
