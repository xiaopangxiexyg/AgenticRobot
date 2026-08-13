(
set -Eeuo pipefail

project=/home/user/zc/DexTeleop
python_bin=/home/user/zc/openpi/.venv/bin/python
profile=deployments/openpi/configs/training/writing_stage1_scenarios_0804_0811_front_h16_absolute_repeat146_v1.json

cd "$project"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

printf '[stage1] computing normalization statistics for the exact 1:4:6 mixture\n'
"$python_bin" -u -m deployments.openpi.training.prepared_skill_launch norm-stats \
  --profile "$profile" \
  --stats-batch-size 256 \
  --stats-num-workers 0

printf '[stage1] training; press S to save and continue, or Ctrl+C once to save and exit\n'
exec "$python_bin" -u -m deployments.openpi.training.prepared_skill_launch train \
  --profile "$profile"
)
