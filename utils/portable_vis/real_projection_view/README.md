# Real Projection View

Project a DexTeleop episode's arm TCP `state_t`, `action_t`, and
`state_{t+lag}` into the recorded RealSense RGB image. The default camera is
the static `front` camera.

```bash
/home/zc/anaconda3/envs/dexteleop310/bin/python \
  utils/portable_vis/real_projection_view/project_real_trajectory.py \
  datasets/2026-07-06_vive_rm65_real_teleop \
  --episode 10 \
  --lag 8
```

The default output is:

```text
<dataset>/visualizations/real_projection_front_episode_000010_lag8.html
```

Use `--no-open` when running from a headless terminal:

```bash
/home/zc/anaconda3/envs/dexteleop310/bin/python \
  utils/portable_vis/real_projection_view/project_real_trajectory.py \
  datasets/2026-07-06_vive_rm65_real_teleop \
  --episode 10 \
  --lag 8 \
  --output /tmp/real_projection_ep10_lag8.html \
  --no-open
```

## What It Shows

- `state_t`: same-row observed TCP pose.
- `action_t`: same-row action TCP target.
- `state_t+lag`: observed TCP pose `lag` rows later.
- Dashed residuals from `action_t` to `state_t` and `state_t+lag`.
- Short trajectory trails and optional current-pose axes.

The viewer embeds sampled RGB frames as JPEG data URLs, so the generated HTML is
self-contained.

## Coordinate Chain

For each side, the tool treats arm poses as TCP poses in that side's arm base.
It then projects to the front RGB image with:

```text
side base pose
  -> desk via T_D_B(side)
  -> right base via inv(T_D_B(right))
  -> front camera via inv(T_camera_to_parent)
  -> RGB pixel via RealSense intrinsics
```

Sources:

- Episode `metadata.calibration_snapshot.arms.*.T_D_B`; falls back to
  `metadata.load_profile` when needed.
- Runtime camera intrinsics from `<dataset>/meta/devices/realsense.json`.
- Static front camera extrinsic from
  `cameras/configs/rm65/cameras/realsense_registry.json`.

The current implementation intentionally targets the static `front` camera,
whose parent frame is `right_arm_base_link`.
