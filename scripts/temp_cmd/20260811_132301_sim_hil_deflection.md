(
cd /home/zc/HandMotion/Developing/DexTeleop
PYTHONPATH=. /home/zc/anaconda3/envs/dexteleop310/bin/python -m control_station sim-hil \
  --enable \
  --control-gui \
  --control-profile policy_human_allocation_scheme_a \
  --checkpoint sim/mujoco_temp_pipeline/checkpoints/cube_pick_goal_delta_xyz_gripper_v2_s211_bootstrap.pt \
  --task-config sim/configs/tasks/rm65_right_cube_pick_goal_delta_xyz_gripper_v2.json \
  --gripper-input-source angle_trigger
)
