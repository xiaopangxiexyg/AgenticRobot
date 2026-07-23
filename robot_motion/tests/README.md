# RM65 Tracker Replay Diagnostics

Utilities here are diagnostic/test tools for replaying recorded tracker-derived
targets through the shared RM65 IK stack. They are intentionally kept under
`robot_motion/tests` because they validate tracker-to-IK behavior from offline
datasets; they are not hardware replay entrypoints.

## Automatic Compare

```bash
PYTHONPATH=. python -m robot_motion.tests.compare_tracker_replay \
  --dataset-root datasets/2026-07-03_vive_rm65_real_teleop \
  --episode 0
```

Useful options:

- `--max-frames N`: quick smoke test on the first `N` selected frames.
- `--json-out path.json`: save the full per-frame report.
- `--max-translation-error-m`, `--max-rotation-error-deg`, `--max-qpos-error-deg`: fail with a non-zero exit code when the IK replay exceeds thresholds.
- `--seed-each-frame-from-observation`: reset the IK solver from recorded qpos before every frame for local one-step diagnostics.

The report separates two meanings:

- `source_vs_action_*`: recorded tracker-derived target pose versus the original dataset action pose.
- `ik_vs_action_*`: current shared IK replay output versus the original dataset action pose.

## MuJoCo Viewer

```bash
PYTHONPATH=. python -m robot_motion.tests.view_tracker_replay \
  --dataset-root datasets/2026-07-03_vive_rm65_real_teleop \
  --episode 0
```

This launches a passive MuJoCo viewer and replays the qpos produced by the
offline tracker-to-shared-IK path. By default it uses:

```text
assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf
```

The viewer uses V3 for visualization while the default collision-aware replay
profile remains the separate TAC open-hand proxy scene.

Useful options:

- `--qpos-source tracker_ik`: show the offline tracker IK result. This is the default.
- `--qpos-source dataset_observation`: show the original recorded observation qpos for comparison.
- `--speed N`: playback speed multiplier.
- `--loop`: loop the selected frames until the viewer is closed.
- `--urdf path/to/scene.urdf`: override the MuJoCo scene URDF.
- `--dry-run`: validate the tracker replay and MuJoCo scene load without opening a GUI window.

## HTML Report

```bash
PYTHONPATH=. python -m robot_motion.tests.write_tracker_replay_report \
  --dataset-root datasets/2026-07-03_vive_rm65_real_teleop \
  --episode 0
```

The default report output is:

```text
<dataset-root>/visualizations/offline_tracker_replay_<episode_id>.html
```

Open the HTML in a browser to scrub frames, view camera videos, and inspect
source/action/IK differences frame by frame.
