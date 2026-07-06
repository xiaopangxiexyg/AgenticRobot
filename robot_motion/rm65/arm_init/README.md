# RM65 Arm Init

RM65-specific initialization code lives here only when it needs RM65 action
space, schemas, FK/IK, collision scene, or left/right arm conventions.

Use the new qpos-only init planner when the target is a known qpos:

```bash
PYTHONPATH=. python -m robot_motion.rm65.arm_init.qpos_init \
  --current-qpos current_qpos.json \
  --qpos-target robot_motion/rm65/arm_init/qpos_targets/bench_init_pose.json \
  --arm both \
  --output /tmp/rm65_qpos_init_plan.json \
  --include-steps
```

The command is read-only: it does not connect to hardware and does not send
motion commands. The output steps can be consumed later by a sim or real
executor.

## Batch Qpos Test Sequence

For robustness testing, build a continuous target sequence by sampling valid
arm qpos rows from each dataset episode:

```bash
PYTHONPATH=. python -m robot_motion.rm65.arm_init.qpos_sequence build \
  --dataset-root datasets/2026-07-05_vive_rm65_real_teleop \
  --samples-per-episode 3 \
  --seed 0 \
  --arm both
```

The default output is:

```text
robot_motion/rm65/arm_init/qpos_test_sequences/2026-07-05_vive_rm65_real_teleop_3_per_episode_seed0.json
```

Then dry-run the full chain from the bench init qpos:

```bash
PYTHONPATH=. python -m robot_motion.rm65.arm_init.qpos_sequence dry-run \
  --sequence robot_motion/rm65/arm_init/qpos_test_sequences/2026-07-05_vive_rm65_real_teleop_3_per_episode_seed0.json \
  --current-qpos robot_motion/rm65/arm_init/qpos_targets/bench_init_pose.json \
  --arm both \
  --output /tmp/rm65_qpos_sequence_dry_run.json
```

`build` reads parquet and therefore requires an environment with `pyarrow`
installed, such as `dexteleop310`. `dry-run` only loads JSON and uses the shared
slow qpos transition planner.

Only the qpos target schema and qpos-init entry point are kept here. New
planning logic should be added to `robot_motion/arm_init` first, then adapted
here only when RM65-specific information is required.
