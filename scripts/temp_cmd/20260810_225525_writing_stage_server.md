(
set -euo pipefail

STAGE=1
CKPT_STEP=150000

case "$CKPT_STEP" in
  30000|60000|90000|120000|150000) ;;
  *) printf 'Unsupported CKPT_STEP: %s\n' "$CKPT_STEP" >&2; exit 2 ;;
esac

case "$STAGE" in
  1)
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage1-0804-0805-front-h16-absolute-prepared-v1/writing-stage1-0804-0805-front-h16-absolute-b1-nw4-meta-v2
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage1_0804_0805_front_h16_absolute_v1.json
    ;;
  2)
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage2-0804-0805-front-h16-absolute-prepared-v1/writing-stage2-0804-0805-front-h16-absolute-b1-nw4-meta-v1
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage2_0804_0805_front_h16_absolute_v1.json
    ;;
  3)
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage3-0804-0806-front-h16-absolute-prepared-v1/writing-stage3-0804-0806-front-h16-absolute-b1-nw4-meta-v1
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage3_0804_0806_front_h16_absolute_v1.json
    ;;
  4)
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage4-0804-0806-front-h16-absolute-prepared-v1/writing-stage4-0804-0806-front-h16-absolute-b1-nw4-meta-v1
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage4_0804_0806_front_h16_absolute_v1.json
    ;;
  *) printf 'Unsupported STAGE: %s\n' "$STAGE" >&2; exit 2 ;;
esac

CHECKPOINT_DIR="$CHECKPOINT_ROOT/$CKPT_STEP"
if [[ ! -f "$CHECKPOINT_DIR/policy_artifact.json" ]]; then
  printf 'Missing checkpoint artifact: %s\n' "$CHECKPOINT_DIR/policy_artifact.json" >&2
  exit 2
fi
if [[ ! -f "$PROFILE" ]]; then
  printf 'Missing training profile: %s\n' "$PROFILE" >&2
  exit 2
fi

cd /home/user/zc/DexTeleop
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

printf '[stage%s] serving checkpoint step %s on port 8078; Ctrl+C returns to the current shell\n' "$STAGE" "$CKPT_STEP"
/home/user/zc/openpi/.venv/bin/python -u -m deployments.openpi.server \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --prepared-training-profile "$PROFILE" \
  --openpi-root /home/user/zc/openpi \
  --host 0.0.0.0 \
  --port 8078 \
  --exclude-cpu 14 \
  --exclude-cpu 30
)
