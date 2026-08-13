(
set -euo pipefail
cd /home/zc/HandMotion/Developing/DexTeleop
PYTHONPATH=. /home/zc/anaconda3/envs/dexteleop310/bin/python -m control_station sim-hil \
  --enable \
  --control-gui \
  --device cpu \
  --capture-profile human_demonstration \
  --capture-root teleop_records/sim_hil_demonstrations \
  --control-profile fixed_alpha_compatibility \
  --human-alpha 1 \
  --human-timeout-fallback hold \
  --gripper-input-source angle_trigger \
  --angle-trigger-config-root teleop_runtime/configs/hardware/angle_trigger \
  --calibration-profile teleop_records/calibration_profiles/20260627_112114_vive_rm65_virtual_desk.json \
  --vive-coordinate-profile teleop_records/control_station_vive_coordinate_profiles/20260808_220119_vive_coordinate.json \
  --task-config sim/configs/tasks/rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose.json \
  --checkpoint sim/mujoco_temp_pipeline/records/20260728_215045_cube_lift_pregrasp_bridge_v5_center_guard/checkpoints/step_000050176_final.pt \
  --vive-port 9999
)
