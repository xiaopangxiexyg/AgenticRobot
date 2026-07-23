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

`PoseTrainingDataContract` distinguishes three pose-training origins: native
v2.3 canonical desk, legacy v2.2 raw-identity compatibility, and v2.2 rows
physically migrated to canonical desk. The migrated origin uses an explicit
runtime raw-to-canonical-desk binding like native v2.3, but retains v2.2 source
provenance and must not claim native-v2.3 causal snapshot evidence.

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

`dataset_contracts.causal_snapshot` fixes the native-v2.3 causal cut to
`source_timestamp <= T0 AND available_timestamp <= T0` and provides a canonical
hash embedded at dataset, episode, and row levels. `dataset_contracts.v23_validation`
is the hardware-independent row validator; the read-only finalized-dataset scanner
is exposed as `dataset_interface.validate_v23_dataset` and through
`python -m utils.dataset_postprocess validate-v23`.

## Incremental v2.4 ControlTrace Core

`dataset_contracts.control_trace` and `dataset_contracts.completeness` define the
M4A/M4A.1 logical event registry, immutable fact/envelope boundary, explicit
record schemas for the current policy/human-shadow subsets, branch-aware lineage, and
scoped runtime-mode requirement profiles. Planned event kinds without a producer
and explicit schema fail closed. This is an in-memory contract milestone: it
does not define a WAL, Parquet partition layout, reward target, or episode
commit owner. `teleop_runtime.control_trace.ControlTraceRuntime` is subordinate
to an existing frontend/recorder episode owner and deliberately cannot begin,
finalize, abort, or commit a dataset episode.

M4A.1/M5 also record the passive policy queue-pop emission, human pre-filter
command emission, independent source-adapter receipt, source validation, and typed
episode membership transitions; runtime dataclass or audit-field additions cannot
silently change persisted projections. Journal-issued, sink-accepted, and
sink-durable evidence remain separate.
The in-memory runtime alone has a zero durable high-water mark. M4B.2 adds
per-episode range/dependency seals and an additive capture commit contract;
the human source-specific formal profile has a scoped runtime fixture, while
generic/policy/replay formal profiles remain planned and production activation
is still disabled. Shadow evaluation and authoritative decision are distinct
event types. M5.5a/b now provide explicit authoritative publication, decision,
directive, shared-safety, command, dispatch, and measurement schemas from a
deployable-sim producer. Replay additionally records source/target assessment,
execution authorization, receiver-admitted candidate authorization and its
immutable arbitration-cut snapshot; only authoritative decisions may
materialize an actual selected-action view. Formal-sim episode profiles remain
fail-closed until their durability and branch-specific capture lineage are integrated.

The scoped CS-6b2 component branch adds an explicit
`component_mixing_shadow` arbitration-output record and the fixture-only
`deployable_sim_component_mixing_shadow_v1` profile. It records immutable
arm-pose/hand-qpos strategy snapshots and a preview compound action with
`non_authoritative=true`; it does not admit safety, reference, command,
dispatch, or measured facts. Its episode fixture reuses the existing
subordinate lifecycle owner and remains non-commit-eligible until a downstream
component admission/safety receipt is defined.

The scoped M4B contract adds strict canonical decoding and versioned binary WAL
frames (header CRC, payload SHA-256, sequence trailer). The subordinate runtime
spool is fixed-size, append-only, and fsync-acknowledged; torn tails are detected
and an independent preallocated health ledger records recovery/capacity evidence.
Storage envelope materialization happens outside the control admission lock.
The canonical human fixture uses a dataset-local session WAL plus immutable
per-episode seal/commit sidecars. This is not production activation, Parquet
materialization, or target-machine power-loss acceptance.

The opt-in `native_streams` capability is subordinate to that same episode
lifecycle. Its writer must use the prepared episode UUID/index and seal before
the capture commit; the commit binds the native episode-manifest hash, which is
rechecked by both the manifest store and dataset reader. Native sealing failure
does not invalidate legacy finalization, but it prevents a v2.4 capture commit.
Default commits remain byte-for-byte unchanged. The optional process worker now
supports bounded, non-blocking inline submission for small pose/qpos samples
and records queue/size rejections as native manifest drops. RGBD/tactile arrays
must instead reuse the existing shared-memory snapshot transport; their native
identity binding and real producer activation remain separate gates.

## Learning observability contracts

`dataset_contracts.learning_observability` is the M4L.1 pure contract boundary
for policy-side facts that cannot safely be inferred after capture. It defines
artifact-bound adaptation-state identity, generative sampling evidence,
inference joins, history-action semantics, and model-action codec identity.
These records own no inference hook, tensor/blob store, ControlTrace event,
training view, or learner.

Evidence is capability/profile driven. A stateless or non-generative policy
uses explicit `not_applicable`; a profile that requires actual initial noise
cannot be satisfied by a seed/RNG receipt. Current generic policy clients do
not expose these facts, so no runtime event schema is registered until one
concrete model adapter supplies typed evidence. Runtime object references,
CUDA pointers, and shared-memory handles are never persistent evidence.

## Optional v2.3/v2.4 Plus episode curation overlay

`dataset_contracts.episode_curation` defines the small, episode-scoped Plus
overlay. It is not a raw Parquet schema: an external annotation root contains
`manifest.json` and append-only `episode_curation.jsonl`. Each record carries a
content-derived `annotation_episode_id`, the legacy `episode_index` locator,
source base/episode hashes, an explicit source binding, and only the bounded
curation fields `disposition`, `reason_code`, and optional task-profile
`variant_id`.

`dataset_interface.EpisodeCurationReader` treats a missing overlay as normal,
so existing v2.3 datasets and explicit v2.4 compatibility views remain
readable without migration. The overlay root is external and source data is
read-only; `draft`/`invalid`/`superseded` sets are not training-visible.
RM65 and UMI enable the optional GUI/sidecar path only with an explicit
`--plus-episode-curation` flag and `--plus-annotation-root`; default-off runs
do not create Plus files or alter the raw writer, row schema, or episode
lifecycle. The GUI-side review draft is reset per pending episode, and a
save request must match that episode's opaque pending token. Native v2.4 capture membership remains an explicit adapter boundary
and is never inferred from a directory name.
