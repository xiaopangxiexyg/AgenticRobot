# RM65 official branch-anchor probe retirement

## Status

`scripts/probes/rm65_official_branch_anchor_probe.py` was removed from the
versioned source tree on 2026-08-28. It was a 2,707-line, offline-only research
program used to produce the 2026-07-27 RM65 all-solution IK and layered-control
comparison. It was not a supported runtime, deployment entrypoint, test helper,
or reusable library.

The probe is not copied into the repository root `legacy/`. That directory is
strictly local and ignored. Git history remains the source for the retired
implementation; this note and the original result report retain the shared
reasoning and conclusions.

## Retirement evidence

The exact `development` base used for this review was
`77d74b20d29fc1ec2a24a562b70d9920ce90dc3e`.

- Repository-wide search found no import or call site for the probe.
- No test, command map, package export, deployment profile, simulator profile,
  or runtime entrypoint named it.
- Its only external reference was the result narrative in
  [`POSE_LIFT_EXPERIMENTS_0727.md`](../../real/reports/POSE_LIFT_EXPERIMENTS_0727.md).
- The program opened no device and owned no control authority, but keeping a
  large executable copy of a completed experiment made the active probe tree
  look supported and duplicated logic that was subsequently promoted.

## Preserved result and replacements

The original report remains versioned and preserves the evaluated datasets,
numeric outcomes, negative findings, and the resulting layered architecture.
In particular, retirement does not erase the findings that per-frame Cartesian
exact projection is not a safe default, a Pink soft branch anchor did not solve
global branch selection, and qref-to-qcmd feasibility needs an independent
time-law layer.

Current executable owners are:

- `hardware.rm65.official_all_ik.RM65OfficialAllIK` for official all-solution
  candidate production;
- `robot_motion.rm65.source_ik_selection` for collision-safe causal branch
  selection;
- `robot_motion.position_control` for versioned waypoint and bounded-horizon
  trajectory contracts;
- `teleop_runtime.position_control.official_all_ik_governor` for the production
  application adapter, without acquiring dispatch authority;
- `teleop_runtime.debug.view_rm65_pose_lift_sim_replay` for current offline
  MuJoCo comparison and visualization.

Those replacements preserve the promoted capabilities, not every exploratory
CLI flag of the retired script. Historical reproduction requiring the exact
old experiment must use the pinned Git revision and remains offline-only; it
does not authorize a robot connection, motion, or a current scientific claim.
