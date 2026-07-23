# Replay Arm Init Runtime Layer

This directory no longer owns arm-init planning.

Current rule:

- shared qpos planning lives in `robot_motion/arm_init`;
- RM65 qpos target loading and RM65-specific planning entry points live in
  `robot_motion/rm65/arm_init`;
- this runtime package may keep real-hardware safety helpers, but must not
  duplicate qpos planning, safe-pose planning, or IK planning logic.

Use the RM65 qpos plan command for offline/dry-run initialization planning:

```bash
PYTHONPATH=. python -m robot_motion.rm65.arm_init.qpos_init \
  --current-qpos current_qpos.json \
  --qpos-target robot_motion/rm65/arm_init/qpos_targets/measured_init_pose.json \
  --arm both \
  --output /tmp/rm65_qpos_init_plan.json \
  --include-steps
```

For real hardware, use the runtime entry point. It reads the current qpos from
the selected RM65 arm(s), builds the plan through `robot_motion`, and only sends
commands when `--execute` and the confirmation token are both present.

Precheck live state and write the plan report without motion:

```bash
PYTHONPATH=. python -m teleop_runtime.replay.arm_init.qpos_execute \
  --qpos-target robot_motion/rm65/arm_init/qpos_targets/measured_init_pose.json \
  --arm both \
  --left-ip 192.168.101.21 \
  --right-ip 192.168.1.20 \
  --output /tmp/rm65_live_qpos_init_precheck.json \
  --include-steps
```

Execute on real hardware:

```bash
PYTHONPATH=. python -m teleop_runtime.replay.arm_init.qpos_execute \
  --qpos-target robot_motion/rm65/arm_init/qpos_targets/measured_init_pose.json \
  --arm both \
  --left-ip 192.168.101.21 \
  --right-ip 192.168.1.20 \
  --execute \
  --confirm INIT_RM65_QPOS_EXECUTE \
  --output /tmp/rm65_live_qpos_init_execute.json
```

The runtime command uses `--left-ip` / `--right-ip` first, then the
`LEFT_ARM_IP` / `RIGHT_ARM_IP` environment variables, then the latest RM65 arm
IP assignment. It does not fall back to hard-coded adapter defaults for
real-motion commands. Reserved documentation networks such as `192.0.2.0/24`
are rejected; fix latest or pass explicit IPs if this happens. Force guard is
enabled by default (`--force-source tool --max-force-n 15 --max-torque-nm 2`)
and triggers a protective stop on limit exceedance or runtime exceptions.

## Sequential Qpos Test List

Run a generated qpos test sequence in order. The runtime reads the current
hardware qpos once at the beginning, plans to the first target, executes it,
then reads hardware qpos again before planning to the next target.

Precheck the whole sequence without sending motion:

```bash
PYTHONPATH=. python -m teleop_runtime.replay.arm_init.qpos_sequence_execute \
  --sequence robot_motion/rm65/arm_init/qpos_test_sequences/2026-07-05_vive_rm65_real_teleop_3_per_episode_seed0.json \
  --arm both \
  --left-ip 192.168.101.21 \
  --right-ip 192.168.1.20 \
  --output /tmp/rm65_qpos_sequence_precheck.json
```

Execute the whole sequence on real hardware:

```bash
PYTHONPATH=. python -m teleop_runtime.replay.arm_init.qpos_sequence_execute \
  --sequence robot_motion/rm65/arm_init/qpos_test_sequences/2026-07-05_vive_rm65_real_teleop_3_per_episode_seed0.json \
  --arm both \
  --left-ip 192.168.101.21 \
  --right-ip 192.168.1.20 \
  --execute \
  --confirm INIT_RM65_QPOS_SEQUENCE_EXECUTE \
  --output /tmp/rm65_qpos_sequence_execute.json
```

Useful bounded smoke test:

```bash
PYTHONPATH=. python -m teleop_runtime.replay.arm_init.qpos_sequence_execute \
  --sequence robot_motion/rm65/arm_init/qpos_test_sequences/2026-07-05_vive_rm65_real_teleop_3_per_episode_seed0.json \
  --arm both \
  --left-ip 192.168.101.21 \
  --right-ip 192.168.1.20 \
  --start-index 0 \
  --max-targets 3 \
  --execute \
  --confirm INIT_RM65_QPOS_SEQUENCE_EXECUTE \
  --output /tmp/rm65_qpos_sequence_execute_first3.json
```

The removed safe-pose/rule-based replay init flow is intentionally not kept as
a compatibility path.
