# RM65 Init Arm Legacy Directory

The old Cartesian standard-pose record/replay scripts were removed.

Initialization planning now follows the qpos mainline:

- shared qpos planning: `robot_motion/arm_init`
- RM65 target schema and qpos init CLI: `robot_motion/rm65/arm_init`
- future real-hardware execution: runtime code should consume the qpos plan
  instead of reimplementing planning here.

Dry-run qpos initialization plan:

```bash
PYTHONPATH=. python -m robot_motion.rm65.arm_init.qpos_init \
  --current-qpos current_qpos.json \
  --qpos-target robot_motion/rm65/arm_init/qpos_targets/bench_init_pose.json \
  --arm both \
  --output /tmp/rm65_qpos_init_plan.json \
  --include-steps
```

This command does not connect to hardware or send motion commands.
