# Dataset contract boundary

`dataset_contracts` owns stable, hardware-independent dataset formats shared by
online collection, replay, and offline post-processing:

- episode/chunk/channel layout names;
- physical types and row IO for the explicitly declared canonical Parquet fields;
- episode integrity/quality statistics;
- wrist-depth mask/raw codecs and manifests;
- chunked tactile stream decoding.

The dependency direction is intentionally one-way:

```text
teleop_runtime / robot_motion / sim  --->  dataset_contracts  <---  offline postprocess
```

This package may use the standard library and NumPy. Optional codec/IO
dependencies (`pyarrow`, `lz4`, `zstandard`) are imported only by the operation
that needs them. It must not import runtime, hardware, UI, or post-processing
packages.

Existing imports through `teleop_runtime.collection.schema_v21`,
`teleop_runtime.collection.stats`, and
`utils.dataset_postprocess.wrist_cam_adjust.depth` remain compatibility exports;
new format-level code should import this package directly.

`utils.dataset_postprocess.wrist_cam_adjust.smoke_wrist_depth_capture` is a
legacy hardware diagnostic bridge rather than offline processing. It marks
itself with `DIAGNOSTIC_RUNTIME_BRIDGE = True` and loads the camera runtime only
when the CLI runs. It is the sole documented exception under the post-process
tree; normal offline modules may not depend on the runtime.

## Provisional v2.3 pose-frame contract

`dataset_contracts.pose_frames` is the hardware-independent boundary for the
v2.3 absolute-pose source contract. Every source declares a fine-grained
`source_id` (for example `rm65_real_dual_arm_v1`), its native raw absolute
frames, and a source-level `raw_to_desk_transform` with optional per-side
overrides. The resolved profile is embedded in dataset metadata; readers do
not infer it from column names or from the current machine calibration.

v2.2 and older datasets are exposed through `dataset_contracts.v23` as marked
legacy RM65 `raw_absolute` data. v2.3 defaults to `desk_absolute`; forcing raw
storage is allowed only when `pose_storage_frame` is explicit. The physical
policy-I/O layout and layout hash remain the v2.2 contract during this
validation stage. Relative arm action is a loader/replay representation, not
a new parquet layout; `robot_motion.relative_action` performs chunk-local
SE(3) conversion while keeping state in desk absolute coordinates.
