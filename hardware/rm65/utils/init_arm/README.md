# RM65 Teleoperation Init Pose

These scripts record and replay a Cartesian standard pose for Vive tracker
teleoperation alignment. They switch the controller to the `empty` tool frame
and use the bare arm end pose only. The standard file stores pose only:
`[x, y, z, rx, ry, rz]`. It does not store qpos targets.

Run commands from the repository root in the RM65 conda environment:

```bash
conda activate dexteleop310
```

## Record The Standard Pose

Move the arm by hand or teach mode to the desired teleoperation start pose, keep
the workspace clear, then record the current bare-end pose.

Single arm:

```bash
python hardware/rm65/utils/init_arm/rm65_record_standard_pose.py \
  --arm left \
  --tool-frame empty \
  --overwrite \
  --output-json hardware/rm65/utils/init_arm/standard_pose.json
```

Both arms:

```bash
python hardware/rm65/utils/init_arm/rm65_record_standard_pose.py \
  --arm both \
  --tool-frame empty \
  --overwrite \
  --output-json hardware/rm65/utils/init_arm/standard_pose.json
```

## Dry-Run Safety Check

The move script first performs a preflight from the current robot pose to the
recorded standard pose. For `--arm both`, dual-arm collision checking is
required by default. If both arms need to move, the script solves synchronized
joint waypoints with the dual-arm URDF collision model. If one arm is already
within tolerance, the script keeps that arm fixed and only checks the moving
arm's preflight joint waypoints against the fixed peer, which avoids the slow
full dual-arm QP solve. For single-arm motion it keeps the slow `Movep_CANFD`
absolute pose path unless `--command-mode ik-joint` is selected. If only one arm
is online, run with `--arm left` or `--arm right`; dual-arm collision checking
is only used when both arms are selected.

The default collision scene is `dual_arm_hand_proxy_open`, which includes
fixed box collision proxies attached at the arm endpoints,
`left_hand_proxy_0@hand_base` and `right_hand_proxy_0@right_hand_base`,
plus a fixed `table_0` box in the base plane. Use
`--collision-scene-profile dual_arm` for arm-link-only collision pairs, or
explicitly pass `--dual-arm-collision-ik off` to disable the dual-arm guard.

To inspect the exact URDF collision geometries and the configured pair list,
generate a standalone HTML/SVG report:

```bash
python hardware/rm65/utils/scripts/rm65_visualize_dual_arm_collision_scene.py \
  --scene-profile dual_arm_hand_proxy_open \
  --print-pairs \
  --output-html hardware/rm65/utils/results/rm65_dual_arm_collision_scene.html
```

```bash
python hardware/rm65/utils/init_arm/rm65_move_to_standard_pose.py \
  --arm both \
  --pose-json hardware/rm65/utils/init_arm/standard_pose.json
```

## Execute Init Motion

For `--arm both`, the default execution path sends collision-checked joint
waypoints through `Movej_CANFD`; this keeps execution consistent with the
checked plan. For a single arm, the default execution mode remains `pose-canfd`.

```bash
python hardware/rm65/utils/init_arm/rm65_move_to_standard_pose.py \
  --arm both \
  --pose-json hardware/rm65/utils/init_arm/standard_pose.json \
  --execute \
  --confirm INIT_RM65_STANDARD_POSE
```

Useful safety knobs:

- `--root-joint-max-delta-deg 70,90`: max J1/J2 movement from the start state.
- `--root-joint-absolute-limits-deg min1,max1,min2,max2`: optional absolute J1/J2 window.
- `--max-linear-velocity-m-s 0.025` and `--max-angular-velocity-rad-s 0.12`: smooth motion speed.
- `--max-angular-distance-rad 6.5`: relaxed large-orientation-change preflight bound.
- `--max-interp-angular-step-rad 0.02`: large orientation changes automatically add Slerp waypoints.
- `--max-interp-linear-step-m 0.003`: large translation changes automatically add waypoints.
- `--ik-jump-refine-max-depth 6`: if SDK IK changes joint branch between two
  adjacent Cartesian waypoints, recursively split that segment up to 16 substeps
  before aborting. A `refined IK jump...` log means interpolation was enough; an
  `IK jump refinement failed...` log means use a staged intermediate pose or
  move the wrist away from the singular region first.
- `--singularity-elbow-threshold-deg 3`: preflight elbow-singularity threshold. The RM SDK default is 10deg.
- `--singularity-wrist-threshold-deg 0.5`: preflight wrist-singularity threshold. The RM SDK default is 10deg.
- `--controller-avoid-singularity`: best-effort controller-side avoid-singularity mode during the init move.
- `--tool-frame empty`: record and replay in the bare arm end frame.
- `--collision-min-distance-m 0.03`: minimum dual-arm collision distance.
- `--collision-scene-profile dual_arm_hand_proxy_open`: use endpoint proxy boxes and the fixed table box in the collision scene.
- `--hard-stop-force-axis-n 40`: stop if any force axis exceeds this threshold.
- `--hard-stop-force-n 40` and `--hard-stop-torque-nm 2`: runtime force/torque stop thresholds.
- `--force-read-every 1`: read the force sensor every waypoint during execution.
- `--arm left`, `--arm right`, or `--arm both`: choose single-arm or dual-arm init.

Dual-arm collision IK requires `pinocchio`, `pink`, `qpsolvers`, and a QP
backend in the active conda environment. To make dependency failures explicit,
`--dual-arm-collision-ik required` is the default for `--arm both`.
