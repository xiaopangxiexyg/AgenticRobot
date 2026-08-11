cd /home/zc/HandMotion/Developing/DexTeleop && \
exec /home/zc/anaconda3/envs/dexteleop310/bin/python \
  -m teleop_runtime.debug.replay_position_rotational_admittance_mujoco \
  --log logs/remote_position_admittance_20260807/cautious/action_debug_20260807_121215_068.jsonl \
  --side left \
  --envelope moderated-8deg \
  --replay-hz 60 \
  --release-tail-sec 10 \
  --interpolation zoh \
  --offline-combined-normal-parity \
  > /tmp/dexteleop_sim_normal_parity_moderated8_low_excitation_20260808.json
