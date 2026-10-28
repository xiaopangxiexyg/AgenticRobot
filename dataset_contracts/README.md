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

## Arm hold-intent training contract

`dataset_contracts.arm_mode` freezes the existing
`dexteleop_arm_mode_v2` IDs and defines an additive
`arm_hold_intent_v1` training projection. It does not change the v2.3 row
layout: `fixed_hold` is the positive per-side target, `teleop` is negative,
and `unknown`, `init_pose`, and `stopped` are ignored. A positive label is
valid only when its recorded scope is `arm_and_end_effector`; this means the
current hold intent covers both the arm and its end effector.

Datasets with the known v2 label spec but no `training_projections` entry use
the frozen v1 projection in compatibility mode. Datasets with no arm-mode
label spec remain valid action-training datasets and expose no arm-hold
contract. Missing row columns are never guessed into negative targets.

## Collection progress contracts

Progress 与 retrospective 契约的相同基础校验由私有 `_progress_validation.py`
统一实现；各领域通过标准库 `partial` 绑定原有异常类。这里只共享必填文本、SHA-256、
有限数值、正整数、非负整数和置信度规则，领域错误类型、公开数据类型及序列化/hash
仍由原模块持有。可选字段语义、声明 hash 的错误措辞，以及 retrospective 正整数
先检查非负再拒绝零的顺序继续留在原模块，不能因名称相近而合并。

`dataset_contracts.progress_profile` and `dataset_contracts.progress_facts`
define the first p2 capture-truth slice without importing a runtime or GUI:

- unique coarse progress-step identity with LINEAR or finite ROOTED_TREE
  topology;
- explicit Episode entry/scope/history and sample-clock binding;
- a minimal causal fact envelope and strict tagged payload registry;
- composite Advance, appended Undo, operator-report-time Failure,
  application Hold/Resume and formal-capture Pause/Resume receipts, Retry, and
  scope-aware termination;
- a sealed manifest over the exact ordered fact hashes.

The initial Attempt has an explicit `EpisodeProgressStartedV1` fact because
binding alone cannot reconstruct its Attempt identity.
`dataset_contracts.progress_refinement` now defines only the reusable minimum
needed to validate a salvage anchor: boundary identity and ordering, coarse
progress-profile binding, prefix-certification eligibility, and explicit
predecessor coverage. `dataset_contracts.progress_annotations` adds the
smallest immutable persistence contract that proved necessary after that
resolver prototype: one boundary annotation, one sealed-progress/profile
binding, append-ordered chunks, and a seal manifest over exact ordered hashes.
Episode/profile/set identity is carried once by the binding rather than copied
into every boundary. Annotator provenance, supersession/correction graphs,
coverage receipts, and training-exclusion payloads remain deliberately
unfrozen until their resolver prototypes stabilize.

`dataset_contracts.progress_stages` adds the self-contained ordered execution
stages inside each coarse progress step and an exact capture-time entry binding
for `NOMINAL` or `RECOVERY_FROM_FAILURE`. `progress_stage_annotations` defines
immutable stage occurrences, fixed capture-Failure lineage, abnormal intervals,
privileged coarse-boundary overrides, whole-Episode abnormal exclusion, and one
sealed annotation-set binding/bundle/manifest. These are pure facts: they do
not select an annotation set, operate a GUI, or persist a training index.

`dataset_contracts.retrospective_progress` binds a post-capture Progress window
to immutable legacy Episode artifacts without pretending it was captured
causally. `retrospective_campaign` records one dataset-local eligible set,
selected annotation batch, migration identity, and publication receipt. A
pilot campaign is annotation-process QA metadata, not a dataset split.

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
The additive `component_action_trajectory` source record preserves one complete
immutable component-keyed plan, including every target interval and action
value. It is an actual-data fact emitted once by the source owner, not a
per-control-cut debug snapshot and not evidence of proposal, target, reference,
or dispatch admission. Existing non-trajectory profiles classify it as
optional; a future timed-trajectory capture profile must require it explicitly.
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

The additive formal-component records keep semantic and implementation
contracts distinct: `component_composition_rule_set` stores the complete
pose/qpos component rules, while `formal_component_composition_binding` links
their hash to one CompositionSpec and one formal strategy spec. Completeness
validation requires every v2 formal-component spec to resolve exactly one such
binding and requires that binding to resolve its three typed facts. Historical
spec/strategy v1 record schemas are unchanged.

Role-specific composition additionally publishes low-frequency
`action_payload_schema` and `component_action_schema` definition facts.
Per-frame action records retain compact schema hashes; the definition facts
resolve those hashes to value semantics, frames, coordinates, units, semantic
parameters, topology, and component-to-payload bindings. Completeness checks
require relative/absolute role schemas and the absolute output schema to
resolve, and require every action schema payload reference to resolve exactly
once.

Formal compound execution adds two records without widening those composition
facts. `conservative_compound_authority_profile` binds the exact spec,
rule-set, strategy, producer, components, groups, TTL/fallback class, and
atomic activation contract. `compound_authority_failure` records the failed
safety/resolution/downstream stage before the standard control-domain HOLD
cancellation. Completeness resolves the profile back to the exact three-way
formal binding and every failure back to that profile and its causal facts.

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
The spool sink keeps an accepted-entry offset and scans only newly admitted
journal entries on each worker flush; this removes repeated full-journal scans
but does not release in-memory history. Releasing durable entries remains a
separate protocol change because completeness, episode seal, and record
materialization still read the in-memory journal.

ControlTrace completeness and recorder routing are separate contracts.
`TraceRequirementProfile` declares which facts must exist for one trace mode;
`ControlTraceCaptureRoutingPolicy` classifies already-produced facts as
`causal_core`, `declared_observation`, `diagnostic_debug`, or
`live_telemetry`. A routing policy must classify every registered event exactly
once, but classification never changes the fact payload or hash. The optional
`ControlTraceCaptureRouter` is not installed by any production frontend: it
provides a default-off recorder seam where core facts must remain durable,
declared observations/debug may be selected explicitly, and live telemetry may
use a transient sink. The preserve-all compatibility policy retains every fact
and exists only to model old behavior; it is not a claim that every fact is
required for every replay capability.

The policy + human component Sim-HIL topology uses
`deployable_sim_scheduled_component_policy_human_composer_v1`. It is distinct
from the legacy human + replay Station composer profile: replay and legacy
Station/session-reducer facts are not applicable, while direct composition
activation, both typed source families, component composition, scheduled
execution, and difference observations are required. The profile remains
`planned` until the application-level episode recorder lifecycle and a
complete seal/rebuild fixture are wired.

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

## Provisional v2.5-p capability contracts

v2.5-p now has two deliberately separate branches:

- `v25p_native` is the authoritative provisional design. It records typed
  component streams and causal sample cuts below `meta/v25p/native/` and has no
  dependency on a v2.3 row, `PolicyIOLayout`, or `meta/episodes.jsonl`.
- `v25p` and the v1 `CaptureProjectionProfile` are frozen compatibility
  contracts for already-created, v2.3-based sidecars. Their hashes and readers
  remain valid, but they are not the native storage model.

The frozen `EmbodimentCaptureProfile` v1 still carries target-capability
bindings for old artifacts. `EmbodimentCaptureProfileV2` is the corrected
native topology contract and deliberately omits execution target, authority,
route, reference, and dispatch semantics. Both versions remain explicitly
readable by schema ID; a v1 artifact is never reinterpreted as v2.

`action_schema` is the shared pure contract for `ActionPayloadSchema` and
`ComponentActionSpaceSchema`. Runtime `ComponentAction` and native action
streams therefore use the same semantic definitions and hash domains. The v2
stream profile embeds resolvable action-space, payload-schema, and component-
catalog definitions; its redundant stream fields are retained for simple
readers but must exactly equal those definitions. `cartesian_pose`, rather
than the coordinate spelling `xyz_quat_xyzw`, is the representation family;
the quaternion spelling lives in coordinates and semantic parameters.

A separate native stream profile owns each concrete pose, qpos,
velocity, end-effector, or other captured representation and chooses its
action, state, observation, and audit streams, their semantic stage,
representation, dtype, shape, frames, and storage kind. Small payloads use
`inline_tensor`; media and
other large payloads use a content-hash-bound `external_artifact` reference.
`V25pCausalSample` joins only records available at its cut. This is the common
recording protocol; no robot-specific concatenated action vector is canonical.

Compatibility output is explicit and subordinate. A v2.3 adapter must resolve
each target field by an exact component/role/stage/schema match. It fails on
missing, ambiguous, invalid, or externally stored selected values, and reports
every native stream omitted by v2.3. Only after the old writer finalizes may a
`V25pCompatibilityProjectionReceipt` bind that output. Native capture remains
valid when a compatibility projection is skipped or fails.

`v25p_native_storage` defines an optional physical binding for the same logical
records. `parquet_stream_chunks_v1` binds immutable per-stream Parquet shards,
causal-sample shards, file hashes, logical bundle hashes, schema hashes, and
contiguous sample ranges. These manifests never redefine component semantics or
make Parquet bytes part of logical record identity. Writers and lifecycle
remain outside `dataset_contracts`.

`episode_identity` is still minted once by the authoritative episode owner and
passed unchanged to native storage, task semantics, compatibility writers, and
eventually the v2.4 branch. The current identity retains the field name
`legacy_episode_index` as a transitional owner-assigned ordinal; native storage
does not use it to locate or validate a legacy row. `task_semantics` remains an
append-only attempt timeline, so failure/recovery/success sequences are
preserved and frame-local stages are derived views.

Both stores are default-off and subordinate: neither may start, stop, discard,
or finalize another writer's episode. A native episode seals its sample and
record bundles before optional compatibility receipts are attached. It still
does not claim final `native_v25_eligible`; that requires the later formal v2.5
commit and the planned v2.4 capability merge.

Current built-in adapters expose only values their source sample actually
provides. RealTeleop declares arm pose, measured qpos, and configured
end-effector target/state for its active side set; UMI declares tracker pose and
configured end-effector target/state. Neither invents a velocity stream.
RM65's six-joint qpos shape belongs to the RealTeleop producer profile, not to
the native protocol, which supports arbitrary component counts and declared
stream shapes. These built-in V2 profiles are selected only by the explicit
default-off native-shadow path; frozen V1 profiles and readers remain available
for already-created provisional artifacts.

Any immutable v2.5-p identity, whether under the compatibility or native root,
blocks legacy in-place delete/reindex compaction. A copy-on-write derived
dataset must issue new identities instead of rewriting captured identity facts.

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

## Model continuation data contract

`dataset_contracts.model_continuation` owns the pure, optional artifact
binding. Frozen v1 combines runtime and training declarations and normalizes
only in memory as `legacy_v1_compat`; its serialization and hash never change.
V2 separates a required runtime contract from optional training provenance:
`inference_only_projection_v1` requires `training=null`, while
`trained_prefix_conditioning_v1` requires an exact training contract.
`legacy_v1_compat` cannot be persisted as a new v2 origin.
The first trained model-RTC validator requires explicit trained-v2 provenance,
continuous absolute teacher-overlap data, and the existing `flow_inpaint`
runtime. Training provenance therefore does not create a second RTC runtime;
the reserved `trained_prefix_conditioning` runtime mode needs a separate
future sampler gate.

Both versions bind continuation implementation, delay/prefix ranges,
continuous action and mask semantics, plus codec/layout/frame/normalization
identity. Training prefix provenance (`trajectory_overlap_teacher_v1`,
`recorded_policy_output_v1`, or `synthetic_corruption_v1`) remains distinct
from runtime `producer_local_overlap_v1`.

The contract owns no dataset reader, ndarray/tensor payload, model call,
sampler implementation, runtime request/plan, target, safety, or dispatch.
`PolicyArtifactContract` preserves the source schema and source contract hash;
the normalized view never creates a replacement identity. Its binding remains
default-none, so legacy artifact serialization and raw-RTC hashes are
unchanged.

## Optional policy arm-hold output contract

`dataset_contracts.policy_action_semantics.PolicyArmHoldOutputContract` binds
an additive deployment output to the existing arm-mode training projection.
It fixes left/right order, probability semantics, same-keyframe alignment,
`arm_and_end_effector` positive scope, all four bimanual combinations, and
explicit calibrated enter/exit thresholds plus debounce counts.

`PolicyArtifactContract.arm_hold_output` is omitted from legacy serialization
when absent. When present it is stored beside—not inside—the existing
continuous/discrete `outputs`, so action dimensions and old artifact readers
remain unchanged. The contract owns no tensor, model adapter, runtime state,
safety, or dispatch behavior.

## Optional v2.3/v2.4 Plus episode curation overlay

`dataset_contracts.episode_curation` defines the small, episode-scoped Plus
overlay. It is not a raw Parquet schema: an external annotation root contains
`manifest.json` and append-only `episode_curation.jsonl`. Each record carries a
content-derived `annotation_episode_id`, the legacy `episode_index` locator,
source base/episode hashes, an explicit source binding, and only the bounded
curation fields `disposition`, `reason_code`, and optional task-profile
`variant_id`.

New annotation manifests declare the append-stable base-fingerprint algorithm
`dexteleop.plus.source_dataset_base_fingerprint.v2`. It hashes stable dataset
metadata while excluding the episode/frame counters rewritten at every seal;
exact episode membership remains bound by the separate episode-manifest hash.
This lets one external annotation set accept successive Episodes from the same
live dataset without weakening the per-Episode locator.

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
Once an external Plus record is durable, the live GUI disables source-root
switching and in-place episode deletion/reindexing; use a copy-on-write derived
dataset if those source identities must change.
