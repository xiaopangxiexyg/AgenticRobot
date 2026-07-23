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
2. By default (`--calibration-policy auto`), look up a lossless fixed base
   frame by sensor serial under
   `hardware/DMTAC_gripper/configs/tactile_calibration/`. If none exists, leave
   the SDK base-frame state untouched. `--calibration-policy required` rejects
   missing serials; `off` forces an uncalibrated diagnostic run.
3. If `--base-frame-index INDEX` is explicitly supplied, install that raw frame
   as a one-run override. Add `--save-calibration` with exactly one `--episode`
   to persist those frames by serial for later runtime and offline use.
4. Call `Sensor.process(..., getdepth=True, getshear=True, getforce=True)` to
   update SDK inference state for each frame.
5. Read `Sensor.getDistributeForce()` and write lossless chunked Zstd float32
   streams plus a `manifest.json`.

Force storage defaults to `--compression zstd --compression-level 3
--chunk-frames 16`. Each chunk is decompressed and compared byte-for-byte before
publication, so float32 values are unchanged. The stream is written to
`*.float32.bin.zst`; the manifest remains authoritative for encoding. Use
`--compression none` only for compatibility or diagnostics.

No episode frame is selected implicitly. To establish fixed calibration from a
known unloaded frame once:

```bash
python -m utils.dataset_postprocess.DM_force_calculate \
  --dataset-root /path/to/dataset \
  --episode 3 \
  --base-frame-index 0 \
  --save-calibration \
  --backend cpu \
  --sdk-root assets/DM_Tac_SDK_Publish_1.2.10 \
  --overwrite
```

Only use a frame verified to be unloaded. Each JSON sidecar records the serial,
source, capture time and SHA-256; the matching `.npy` contains the lossless raw
base frame. Derived stream manifests also record the calibration digest. A new
or changed calibration invalidates the cached force stream automatically.

## SDK 1.2.10 offline compatibility

With no physical sensor connected, SDK 1.2.10 may log that the serial is
disconnected. Its offline `Sensor.process(...)` still returns valid infer,
depth, shear, and global-force values, but may leave the returned
`distributed_force` as `None` and the live getter initialized to an all-zero
placeholder. The processor synchronizes those offline results into the SDK's
own distributed-force composition state before calling the getter. It never
uses the disconnected all-zero placeholder as successful output.

Outputs carry processor revision `sdk_offline_force_fixed_calibration_zstd_v4`. Older
outputs without this revision are recomputed automatically. A stream that is
still all-zero for every frame is reported as failed and is not published.

If `dmrobotics` is not installed, pass `--sdk-root
assets/DM_Tac_SDK_Publish_1.2.10` from a Python version supported by the SDK.
