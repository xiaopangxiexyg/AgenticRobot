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
    QPOS_ARGS=(
      --policy-qpos-init-dataset /home/user/zc/DexTeleop_0803/datasets/2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage
      --policy-qpos-init-episode 0
      --policy-qpos-init-frame 0
      --policy-qpos-init-source auto
      --policy-qpos-init-collision-check
    )
    ;;
  2)
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage2-0804-0805-front-h16-absolute-prepared-v1/writing-stage2-0804-0805-front-h16-absolute-b1-nw4-meta-v1
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage2_0804_0805_front_h16_absolute_v1.json
    QPOS_ARGS=(--no-init-qpos)
    ;;
  3)
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage3-0804-0806-front-h16-absolute-prepared-v1/writing-stage3-0804-0806-front-h16-absolute-b1-nw4-meta-v1
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage3_0804_0806_front_h16_absolute_v1.json
    QPOS_ARGS=(--no-init-qpos)
    ;;
  4)
    CHECKPOINT_ROOT=/home/user/zc/openpi/checkpoints/writing-stage4-0804-0806-front-h16-absolute-prepared-v1/writing-stage4-0804-0806-front-h16-absolute-b1-nw4-meta-v1
    PROFILE=/home/user/zc/DexTeleop/deployments/openpi/configs/training/writing_stage4_0804_0806_front_h16_absolute_v1.json
    QPOS_ARGS=(--no-init-qpos)
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
printf '[stage%s] starting 100-step no-action preflight for checkpoint %s; Ctrl+C returns to the current shell\n' "$STAGE" "$CKPT_STEP"
/home/user/miniconda3/envs/dexteleop310/bin/python -u -m deployments.openpi.client.real \
  --host 127.0.0.1 \
  --port 8078 \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --prepared-training-profile "$PROFILE" \
  --deployment-profile rm65_bimanual_dmtac \
  --arm-execution-profile position_15_60_120hz \
  --arm-control-mode position \
  --position-control-profile rm65_bimanual_position_v1 \
  --position-continuous-waypoint-profile formal_third_order \
  --action-hz 15 \
  "${QPOS_ARGS[@]}" \
  --action-queue-policy replace \
  --min-action-queue 0 \
  --camera-keep-last 4 \
  --camera-frame-max-age-sec 0.2 \
  --position-force-guard \
  --position-force-guard-source tool \
  --position-force-guard-max-force-n 30 \
  --position-force-guard-max-torque-nm 5 \
  --position-force-guard-hz 20 \
  --position-force-guard-consecutive-samples 1 \
  --position-force-guard-max-read-errors 3 \
  --position-force-guard-stale-sec 0.4 \
  --position-force-guard-startup-timeout-sec 2 \
  --max-steps 100 \
  --debug
)
