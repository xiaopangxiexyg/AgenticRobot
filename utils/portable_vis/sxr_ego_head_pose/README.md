# SXR Ego dynamic head/camera extrinsics

This read-only diagnostic replays `head_pose` messages from an SXR Ego
`sensor.mcap` at recorded speed. It draws:

- the dynamic IMU/head coordinate frame;
- the head-position trail;
- the six camera frames derived from `calibration.json` (`rgb`, `tracking`,
  and `ctrl`, each with `cam0` and `cam1`).

The viewer does not connect to a headset or any robot hardware.

## Run

The existing `EgoData` environment contains both the MCAP reader and
Matplotlib:

```bash
conda activate EgoData
python -m utils.portable_vis.sxr_ego_head_pose.visualize_head_extrinsics \
  /home/zc/Downloads/episode_20260529_0100
```

Press Space to pause/resume. Press `q` or Escape to close.

Use a decode-only check when no graphical display is available:

```bash
conda activate EgoData
python -m utils.portable_vis.sxr_ego_head_pose.visualize_head_extrinsics \
  /home/zc/Downloads/episode_20260529_0100 \
  --dry-run
```

By default, the first recorded head pose becomes the display origin. The
recording schema names only `quat[4]`; it does not state quaternion component
order or transform direction. Defaults are therefore explicit and overridable:

```bash
# Defaults:
--quaternion-order xyzw --pose-direction world_from_imu --frame-mode first

# Alternative interpretation:
--quaternion-order wxyz --pose-direction imu_from_world
```

The calibration matrices are interpreted according to their stored names:
`T_ic_camN_to_imu0` maps the calibration camera coordinates into the IMU
frame. The stored camera convention is converted to OpenCV optical axes using:

```text
optical X -> calibration Y
optical Y -> calibration -X
optical Z -> calibration Z
```

This fixed basis change is independently supported by stereo epipolar checks
across all three camera groups. At each replay sample the displayed optical
camera transform is:

```text
T_display_optical_camera =
    T_display_imu @ T_imu_calibration_camera @ T_calibration_camera_optical
```
