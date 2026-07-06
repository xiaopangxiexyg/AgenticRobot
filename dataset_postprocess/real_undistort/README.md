# Real Undistort

Fast OpenCV undistortion for recorded RealSense videos. The tool keeps raw
datasets untouched and writes an undistorted derived dataset with updated
camera intrinsics.

## Inspect Intrinsics

```bash
python -m dataset_postprocess.real_undistort inspect \
  --devices-json datasets/2026-07-03_vive_rm65_real_teleop/meta/devices/realsense.json
```

By default, only cameras with non-zero distortion are selected. Use repeated
`--camera` flags for explicit selection.

## Undistort A Dataset

```bash
python -m dataset_postprocess.real_undistort dataset \
  --dataset-root datasets/2026-07-03_vive_rm65_real_teleop \
  --output-root datasets/2026-07-03_vive_rm65_real_teleop_undistorted \
  --camera left_wrist \
  --camera right_wrist \
  --jobs 2
```

The output dataset preserves the original parquet rows and video paths. Selected
videos are rewritten in place under the output root; unchanged files are
hard-linked by default. The generated files include:

- `meta/devices/realsense_raw_before_undistort.json`
- `meta/devices/realsense.json` with zero distortion and the new pinhole matrix
- `meta/real_undistort.json` with a processing summary

`--alpha` is passed to OpenCV `getOptimalNewCameraMatrix`:

- `--alpha 0`: crop/zoom to avoid black borders, best for model input.
- `--alpha 1`: keep the full field of view, may show black borders.

## Undistort One Video

```bash
python -m dataset_postprocess.real_undistort video \
  --input datasets/2026-07-03_vive_rm65_real_teleop/videos/observation.images.left_wrist/chunk-000/episode_000000.mp4 \
  --output /tmp/left_wrist_undistorted.mp4 \
  --intrinsics-json datasets/2026-07-03_vive_rm65_real_teleop/meta/devices/realsense.json \
  --camera left_wrist \
  --overwrite
```

The implementation precomputes `initUndistortRectifyMap` once per camera and
uses `cv2.remap` for each frame.

