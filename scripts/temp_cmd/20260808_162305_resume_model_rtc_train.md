cd /home/user/zc/DexTeleop
source /home/user/zc/openpi/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

python -u \
  -m deployments.openpi.training.prepared_skill_launch train \
  --profile /home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage1_0804_0806_front_h16_absolute_v1.json \
  --model-rtc-profile /home/user/zc/DexTeleop/deployments/openpi/configs/model_rtc/writing_stage1_0804_0806_front_h16_absolute_prepared_model_rtc_v1.json \
  --exp-name writing-stage1-0804-0806-front-h16-absolute-model-rtc-b1-nw4-v1 \
  --resume \
  --no-motion
