(
set -Eeuo pipefail

training_unit="openpi-writing-stage1-$(date +%Y%m%d-%H%M%S)"

printf '[stage1] system unit: %s.service\n' "$training_unit"
printf '[stage1] press S to save and continue, or Ctrl+C once to save and exit\n'

sudo systemd-run \
  --system \
  --unit="$training_unit" \
  --description='OpenPI writing Stage1 training' \
  --uid=user \
  --slice=openpi-training.slice \
  --working-directory=/home/user/zc/DexTeleop \
  --pty \
  --wait \
  --property=ManagedOOMPreference=avoid \
  --property=MemoryAccounting=yes \
  --property=TasksAccounting=yes \
  --setenv=HOME=/home/user \
  --setenv=USER=user \
  --setenv=LOGNAME=user \
  --setenv=CUDA_VISIBLE_DEVICES=0 \
  --setenv=WANDB_MODE=offline \
  --setenv=PYTHONUNBUFFERED=1 \
  /home/user/zc/openpi/.venv/bin/python -u \
  -m deployments.openpi.training.prepared_skill_launch train \
  --profile deployments/openpi/configs/training/writing_stage1_scenarios_0804_0811_front_h16_absolute_repeat146_v1.json
)
