conda activate /home/zc/anaconda3/envs/dexteleop310
python /home/zc/HandMotion/Developing/DexTeleop/todos/probes/view_rm65_deploy_ik_rate_comparison_mujoco.py \
  --debug-jsonl /tmp/dexteleop_c2_compare/deploy_debug_20260808_124343_776.jsonl \
  --full-deployment \
  --candidate-mode non-rtc-splice \
  --candidate-trajectory-mode formal \
  --planner-command-hz 60 \
  --non-rtc-splice-initial-jerk-regularization 0 \
  --non-rtc-splice-jerk-delta-regularization 0.0001 \
  --play candidate \
  --reference-hz 120 \
  --speed 1 \
  --repeat 1 \
  --viewer-startup-hold-sec 1
