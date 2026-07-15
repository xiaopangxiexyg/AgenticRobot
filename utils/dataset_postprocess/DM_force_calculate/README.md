# DM Force Calculate

Postprocess DM-Tac raw tactile streams saved by DexTeleop into SDK
`distributed_force` streams.

The input is the external tactile format written by
`teleop_runtime.collection.writer_v21`: per-side `manifest.json` plus per-sensor
raw `uint8` streams with shape `[480, 640]`. The output is another external
tactile-compatible stream under `tactile_force/`, with `float32` arrays shaped
`[288, 384, 3]`.

```bash
python -m utils.dataset_postprocess.DM_force_calculate \
  --dataset-root datasets/2026-07-05_vive_rm65_real_teleop_collision \
  --output-root datasets/2026-07-05_vive_rm65_real_teleop_collision \
  --episode 17 \
  --jobs 4 \
  --backend cpu \
  --overwrite
```

The tool uses the official SDK offline path:

1. Reconstruct `DMTacImage(img=raw_uint8, serial=sensor_key)`.
2. Set the stream base frame from `--base-frame-index` (default `0`).
3. Call `Sensor.process(..., getdepth=True, getshear=True, getforce=True)` to
   update SDK inference state for each frame.
4. Read `Sensor.getDistributeForce()` and write `*.float32.bin` streams plus a
   `manifest.json`.

If `dmrobotics` is not installed, pass `--sdk-root
assets/DM_Tac_SDK_Publish_1.2.10` from a Python version supported by the SDK.
