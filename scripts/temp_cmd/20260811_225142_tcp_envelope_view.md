conda activate \
  /home/zc/anaconda3/envs/dexteleop310
python \
  -m \
  sim.sim_real_tests.mujoco.view_tcp_pose_envelope \
  --speed \
  0.35 \
  --repeat \
  0 \
  --pause-between-repeats-sec \
  0.5
