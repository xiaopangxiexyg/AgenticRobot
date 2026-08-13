conda activate \
  "${DEXTELEOP_ENV_NAME:-dexteleop310}"
python -m \
  sim.sim_real_tests.mujoco.view_tcp_pose_envelope \
  --profile \
  relative_tight
