# Dataset Interface

## Arm hold-intent derived view

`dataset_interface.arm_hold` projects the optional v2.3 arm-mode columns into
two float32 targets ordered `[left, right]`, plus separate validity, loss, and
transition masks. It is an episode-local read-only view and does not append
columns or rewrite old data.

The baseline auxiliary loss should use BCE-with-logits per side and reduce
only over `arm_hold_loss_mask` (`valid AND not padding`). Missing labels,
unknown/init/stopped modes, and malformed positive scope therefore contribute
zero auxiliary loss without disabling the continuous action loss. Class or
transition weighting is deliberately not hidden in the data interface:
training configuration must declare it explicitly after inspecting
`summarize_arm_hold_coverage()`. In particular, a dataset containing only
paired states `01` and `10` is perfectly complementary and must not be treated
as evidence for independent `00` or `11` behavior.

`dataset_interface.arm_hold_sampling` owns deterministic Episode-local window
batch order and optional exposure repetition. Learner definitions, tensor
collation, losses, and prediction evaluation remain in
`policy_learning.arm_hold`.

## Control-composition projection boundary

`dataset_interface.composition_control_trace` is the optional adapter from
protocol-neutral `CompositionDomainEvent` values to v2.4 `ControlTraceFact`
values. `control_composition` owns the input/context/result/difference domain
objects; this adapter owns only their dataset projection. WAL admission,
episode membership, seal/commit, native media streams, and the final dataset
remain in their existing runtime/collection owners.

Receipt-driven allocation publication uses the compact
`allocation_publication_outcome` projection. It records attempt, decision,
accepted-context, and committed-state identities without duplicating activity
windows or action payloads; ordinary composition context/evaluation facts keep
owning those full records.

Residual-anchor lifecycle is projected as separate spec, admission, record,
and same-cut snapshot facts. These records preserve request/receipt identity,
target boundary, receiver availability, generation, and component binding
lineage, but do not duplicate action vectors; ordinary composition input facts
remain the action-payload owner. Lifecycle-aware formal strategy spec/input
records use development record schema v2 and include the required lifecycle
spec or selected snapshot identity. This is an intentional development
contract update; consumers of the earlier record shape must be updated rather
than treating old and new payloads as interchangeable.

Opt-in source eligibility is projected as separate spec, admission, state,
snapshot, and input-snapshot binding facts. Existing composition input and
input-snapshot record schemas remain unchanged; the binding preserves exact
same-cut lineage without mutating their historical payloads.

Formal component composition follows the same additive rule. The adapter
projects one complete `component_composition_rule_set` fact and one
`formal_component_composition_binding` fact linking the exact
CompositionSpec, rule set, and FormalCompositionStrategySpec hashes. Existing
`composition_spec` retains its existing record shape. Formal strategy
spec/input records use the lifecycle-aware development shape described above.

## Model continuation derived view

`dataset_interface.model_continuation` implements the first model-RTC offline
projection. It accepts one episode-local, artifact-bound absolute pose chunk
and explicit forecast-delay/prefix lengths, then returns continuous
artifact-action prefix and target arrays with three separate masks:

- float prefix mask: one means conditioned;
- boolean target padding mask: true means padding;
- boolean loss mask: true means contributes to loss.

The v1 projection supports only `trajectory_overlap_teacher_v1`. It rejects
relative actions, discrete actions, non-finite values, unsupported delay or
prefix lengths, a delay that leaves no real executable suffix, and
recorded-policy/synthetic provenance without their future typed projections.
It accepts frozen v1 through its normalized training view and trained-prefix
v2, but rejects inference-only v2 because `training=null`.
The sample hash binds the episode/start location, arrays, masks, projection,
layout, frame, chunk, and continuation contract. It does not normalize values,
create framework tensors, rewrite source rows, or own a dataset episode.

The trained-v1 preflight API adds a replayable sampling spec and one combined
observation/action window without changing the D2 sample schema. The spec
lists exact legal `(forecast_delay_steps, prefix_steps)` pairs and uses a
stateless hash over spec/contract/episode/start/draw identity; it owns no
mutable RNG. The resulting window binds observation and generation start to
the same source index, derives the first executable index as `start + delay`,
and exposes spec/causal/window hashes for a later framework adapter. The
chosen pair remains part of the existing training-sample hash rather than a
second selection receipt.

`CompositionInformationPlan` is not a dataset capture profile. A camera or
tactile stream needed only for recording should go directly to the capture
path. If the same immutable sample is also needed by a composition strategy,
composition should receive only its declared inline view or stable blob
reference. Disabling a dataset modality must not change the numerical
composition evaluation, and enabling one must not force its heavy payload through
the composition return interface.

## v2.4 ControlTrace logical readers

`dataset_interface.control_trace` implements the scoped M4C.1 read boundary.
By default it exposes an episode only when the immutable capture commit points
to a valid seal and the canonical session WAL reproduces the sealed range,
dependencies, membership, and hashes. A sealed episode without a commit can
only be opened with `require_commit=False` for explicit audit use.

`materialize_shadow_evaluation_view()` is rebuilt from core facts and contains
`would_select_action` plus legacy-execution and replay-binding lineage. It never
creates authoritative `selected_action` or a bare training `action` column.

`materialize_actual_decision_view()` is the authoritative counterpart. It
accepts only `arbitration_decision` lineage, joins the published directive,
shared-safety result, command, dispatch receipt, and measurement facts, and
keeps real variable decision intervals. Its `decided_at` comes from the
recorded computation completion time rather than the scheduler tick. The row
uses `selected_action` with `action_stage=selected_pre_safety` and never creates
an unqualified `action` training target.

## Collection progress read boundary

`dataset_interface.progress_artifacts` is the strict, read-only disk boundary
for one sealed progress Episode. It loads the existing profile, binding, fact
log, and manifest without creating directories, opening write handles, or
repairing source artifacts. Callers may provide the hashes published by the
Episode owner; the loader verifies those bindings before returning a resolved
timeline. Missing, unsealed, torn, or ambiguously encoded artifacts fail
closed. This module adds no persisted schema and does not own Episode
lifecycle.

`load_dataset_progress_profile()` additionally verifies the dataset-level
`meta/progress/profile.cjson` binding against every Episode-local profile. It
accepts a legacy dataset with identical Episode profiles when explicit
binding is not required, but never guesses across multiple hashes and never
backfills files. This is the read-side enforcement for the runtime rule that
one dataset contains at most one exact progress profile.

`dataset_interface.progress_timeline` is the first p2 read-only resolver. It
independently validates the typed capture-fact sequence and reconstructs:

- canonical Attempt intervals and outcomes;
- effective progress transitions after immediate Undo is applied;
- failed-Attempt/retry boundaries with no active Attempt during adjustment;
- intentional formal-capture pause intervals from applied Pause/Resume
  operation receipts;
- open versus sealed timeline status and exact manifest binding.

The resolver does not import `collection_runtime`, persist a second timeline,
or mutate source facts. Its result dataclasses are provisional derived Python
views rather than frozen dataset schemas.

`dataset_interface.progress_corrections` adds one intentionally narrow
retrospective prototype for sealed LINEAR timelines: it can reassign a terminal
Failure to an earlier Attempt while atomically invalidating the intervening
Advances and their Attempts. It leaves the sealed capture bundle untouched and
fails closed when a Retry/later Attempt would require an explicit suffix
correction. This validates immutable overlay mechanics; it is not the final
post-annotation correction model. The eventual model may replace a bounded
retrospective timeline segment, including stage and time boundaries, after
profile validation.

`dataset_interface.writing_stage1_v23` is the narrow admission adapter for the
legacy writing campaign. It accepts exactly one v2.3 dataset root and resolves
only that dataset's `semantic_layers/<layer-id>` path. It does not combine
dates, choose training views, or weaken camera constraints; those remain work
for the upper abstract-dataset compiler.

Dataset-local retrospective bindings use `.` relative to the one explicitly
allowed owning dataset root, so moving the complete dataset does not invalidate
its semantic layer. `retrospective_campaign_artifacts` admits only Episodes in
a valid definition/publication-receipt pair; partial campaign writes remain
invisible and resumable.

`dataset_interface.sample_axis` provides the version-neutral, read-time mapping
from one Episode clock to its contiguous causal sample indices.
`EpisodeSampleAxis.from_causal_samples(...)` can consume native causal samples
directly; it rejects gaps in `sample_index`, a foreign clock, and decreasing cut
times instead of guessing an ordering. Equal cut times are supported and retain
explicit `first_at_or_after` versus `first_strictly_after` boundary semantics.

`dataset_interface.progress_samples` projects the resolved progress timeline
onto that sample axis without reading or rewriting stream payloads:

- an Advance closes the preceding Attempt as `[start, advance)`, so samples at
  the transition cut belong only to the next Attempt;
- Failure and terminal Attempt cuts are inclusive, matching
  `[first_at_or_after(start), first_strictly_after(end))`;
- an intentional formal-capture Pause is `[pause, resume)`;
- an open Attempt or Pause extends only to the current end of the known sample
  axis.

`dataset_interface.progress_exclusions` builds the first coarse-only training
view from a sealed capture-time timeline. Each online failed Attempt contributes
one exclusion from its Attempt start through its operator-report Failure cut.
Successful Attempts, Retry Attempts, and intentional Pause intervals are not
implicitly added. An exclusion with no matching causal sample remains as an
empty auditable interval instead of deleting the Failure fact. This builder
does not accept an active timeline and does not infer refinement anchors,
corrected onset times, or retrospective stage reassignment.
`ResolvedProgressTimeline.resolution_kind` is a derived-only guard: the capture
resolver emits `CAPTURE_CAUSAL`, while the current retrospective prototype emits
`RETROSPECTIVE_CORRECTION`; the coarse builder rejects the latter instead of
silently losing correction provenance.

`dataset_interface.progress_annotation_jobs` discovers only Episodes already
published in `meta/episodes.jsonl` whose progress manifest and ordered-stage
artifacts are complete; the active/latest partial Episode is invisible.
`progress_stages` validates strict stage order across retries, preserves the
exact capture Failure timestamp, permits only a final posthoc Failure without
a capture fact, and derives loader-ready occurrence/exclusion sample intervals.
A failed stage excludes its own occurrence start through the original Failure
cut. Whole-Episode abnormal produces `EXCLUDED_EPISODE` without deleting
existing annotations. Coverage and exclusions remain derived views rather than
new per-frame columns.

`dataset_interface.training_semantics` is the first composed, read-only
training-selection layer. It projects sealed Attempts, ordered stage
occurrences, abnormal intervals, selected Failure-trigger annotations, and
capture pauses onto one sample axis. A capture-default trigger at Failure end
is treated as unreviewed; Failure exclusion falls back from a reviewed trigger
to the linked failed-stage start and then to the failed-Attempt start. Normal
intervals may cross successful stage and progress-step transitions. Failure,
abnormal, pause, Episode, and caller-supplied continuity boundaries terminate
the interval, after which the existing pose-action chunk contract alone decides
whether an incomplete tail is dropped or padded. Queries assign each window to
the progress step of its observation/start sample and may alternatively select
typed Failure or Recovery anchors. These spans, anchors, and window locations
are rebuildable Python views: they do not rewrite frame rows and are not a new
persisted dataset schema.

`dataset_interface.prepared_training_view` materializes that rebuildable
interpretation during explicit dataset organization, not during live capture.
It can select sealed Episodes from multiple date-based dataset roots, uses the
Episode capture identity as the global key, and retains each source
`episode_index` only as provenance. Its default mode requires one exact
Progress Profile, one exact Stage Profile, and one compatible projected
Parquet field schema. Multiple annotation sets require an explicit per-Episode
selection; identical Episode identities are deduplicated only when their raw
content and selected semantic artifacts are identical.

`dataset_interface.progress_skill_projection` is the explicit exception for
single-subtask training across task Profiles. A minimal
`SkillTrainingContractV1` defines the target Skill independently from task
topology; each `ProgressSkillProjectionV1` binds one exact source Progress and
Stage profile hash, Step/stage subset, controlled components, predicates, and
conditioned context to it. The prepared builder enables this mode only when
both `skill_projection_catalog` and one `target_skill_id` are supplied. It then
adds `skill_intervals.parquet` and deterministic per-Episode receipts. Failure,
abnormal, Pause, Attempt, and omitted-stage boundaries remain discontinuities;
the source datasets are still read-only. Without those two arguments, the
strict single-Profile behavior and physical file set remain unchanged.

The result is an immutable `derived/training_views/<hash>/` snapshot containing
an Episode table, semantic interval/anchor tables, explicit source selection,
and a hashed source inventory. Source roots and files are referenced by
relative locators; no Parquet, video, Progress fact, or annotation is copied or
rewritten. A workspace merge may resolve outside the organized root, while
`require_self_contained=True` provides the stricter release/bundle check. The
prepared view is provisional and may be deleted and rebuilt; it is not a new
dataset truth or a `collection_runtime` responsibility.

The first implementation is a Python API rather than a new production
entrypoint:

```python
from dataset_interface import (
    PreparedSourceDataset,
    build_prepared_training_view,
)

result = build_prepared_training_view(
    "/datasets/organized/writing-v1",
    (
        PreparedSourceDataset("2026-08-01", "/datasets/2026-08-01_writing"),
        PreparedSourceDataset("2026-08-02", "/datasets/2026-08-02_writing"),
    ),
    column_projection=("timestamp", "action", "observation.state"),
)
```

The reviewed writing 3/4/5-step mappings can be selected explicitly:

```python
from dataset_interface import load_builtin_writing_skill_projection_catalog

result = build_prepared_training_view(
    "/datasets/organized/writing-skill",
    sources,
    skill_projection_catalog=(
        load_builtin_writing_skill_projection_catalog()
    ),
    target_skill_id="writing.write_with_brush",
)
```

The writing catalog does not equate `lift_ready` with
`lift_and_present`: those are separate full acquisition Skills. Only their
reviewed `approach + grasp` stage prefix is shared.

This call writes only below the organized root. The date roots are opened as
read-only sources. Callers must use the actual logical column names of their
capture profile; missing or differently typed projected columns fail closed.

`dataset_interface.progress_refinement` is the first validated salvage-anchor
prototype. `dataset_interface.progress_annotations` verifies the minimal
immutable binding/chunk/manifest artifacts and reconstructs one caller-selected
sealed `SelectedRefinementAnnotationSet`; it never writes or repairs annotation
files. The selected view carries both the source progress-manifest hash and the
exact source annotation-manifest hash; salvage, coverage, and cache rows retain
both, so two sealed artifacts that reuse a human-facing annotation-set ID
cannot overwrite one another. `RefinementBoundaryObservation` remains as a compatibility/read-view
name for the single persisted `RefinementBoundaryAnnotationV1` contract, not a
second schema. An exclusion start moves forward only when the refinement
profile marks that boundary as eligible, certifies the preceding prefix, every
required predecessor has one unambiguous annotation in valid time order, all
evidence uses the Episode clock, and confidence meets the explicit resolver
threshold. Missing evidence, conflicts, invalid ordering, low confidence, or a
boundary at/after the Failure cut retain the original coarse Attempt-start
exclusion.

`dataset_interface.progress_coverage` independently derives structural
refinement coverage for every resolved coarse Attempt. It reports
`NOT_APPLICABLE`, `UNANNOTATED`, `PARTIAL`, `COMPLETE`, `CONFLICTED`, or
`INVALID` from profile membership, exact boundary multiplicity, Episode-clock
intervals, and boundary order. Confidence is intentionally not part of
structural completeness; the salvage/training resolver applies its own
explicit confidence threshold.

`dataset_interface.progress_coverage_catalog` is an optional SQLite WAL query
cache over those derived views. It supports transactional per-set replacement,
full rebuild, status/Episode queries, and annotation-set issue lookup. The
cache key includes the exact annotation-manifest hash. The database stores
canonical hashes only as lookup references; it does not define
facts or participate in canonical identity. It may be deleted at any time and
must be rebuilt from sealed progress/annotation artifacts after loss or format
change. Catalog failure therefore cannot change the validity of an annotation
commit.

`dataset_interface.progress_annotation_read_cache` owns the separate,
rebuildable read cache used by the refinement-annotation application. It
reconstructs only sealed lower artifacts and persists no GUI or Station state;
`control_station` supplies paths and presents the resulting read view.

The selected timeline, salvage anchor, and training exclusions remain derived
Python views. General correction contracts, annotation-set selection policy,
canonical coverage receipts, and training-exclusion persistence remain outside
this slice.

`dataset_interface.task_semantics` rebuilds frame-local subtask state from the
authoritative append-only attempt events; it does not write labels back into
source rows. The two v2.5-p read paths are explicit:

- `dataset_interface.v25p_native` verifies the native component records,
  causal-sample joins, bundle hashes, embodiment/task bindings, compatibility
  receipts, and optional v2.4 reference. It never reads `meta/episodes.jsonl`.
  A sealed `native_storage_manifest.cjson` selects the chunked Parquet reader;
  otherwise the frozen per-record CJSON fallback is used.
- `dataset_interface.v25p` verifies the frozen v1 sidecar whose base is an
  exact finalized v2.3 episode-record hash.

`load_v25p_episode(dataset_root, episode_uuid)` additionally verifies the
canonical sidecar domains, embodiment/projection/task profile bindings, the
complete task-event bundle, the exact finalized legacy episode-record hash,
and an optional referenced v2.4 capture commit. Missing or rewritten evidence
fails closed; the loader performs no repair or source mutation.

`project_native_episode_to_v23_rows(...)` is a read-only compatibility adapter,
not an episode writer. It accepts only a dataset-bound v2.3 projection profile,
requires a unique component/role/stage/schema match for each required field,
and exposes omitted native streams as an explicit lossy result. It never treats
v2.3 fields as the source of native semantics.

`load_v25p_parquet_sample(...)` is the bounded random-read surface for the
chunked backend. It resolves the owning chunk from manifest ranges and reads
only the requested component stream shards, while still checking physical and
logical hashes. This storage optimization does not add dataset fields or alter
the generic component/profile contracts.

`component_action_projection` is the version-neutral control-to-data boundary.
It maps a canonical `ComponentAction` to explicitly bound v2 action streams,
checks the shared action-space and payload hashes, and preserves partial
component coverage according to the projection spec. Its result contains no
episode identity, sample index, or time. The existing episode owner supplies
causal time through the thin `teleop_runtime.collection.v25p_component_action`
boundary before persistence. The projector does not compose, arbitrate, route,
dispatch, or create a second episode lifecycle.

`v25p_component_profile` is the deterministic construction boundary for V2
capture profiles. A caller supplies the canonical `ComponentActionSchema`, an
explicit embodiment declaration, ordered action-component-to-stream bindings,
and explicit state/observation stream declarations. The builder copies action
dtype/shape/frames/units/value semantics from the canonical payload schemas,
rejects schema drift or incomplete coverage, and returns the embodiment,
component-stream, and action-projection profiles as one validated immutable
bundle. It never guesses left/right, arm/tool, pose/qpos, source, or target
capability from names. RealTeleop and UMI only provide their source-specific
declarations; both delegate profile/hash construction to this shared builder.

The default-off producer parity gate has two source adapters:
`RealTeleopV25pProducer` consumes the already assembled `CollectorSample`, and
`UmiV25pProducer` consumes the already next-observation-aligned `UmiSample`.
Both construct the shared canonical `ComponentAction` and use the generic
projector for action streams; source-specific state streams and original causal
times are attached only at the collection boundary. Neither adapter starts an
episode or writes a row. `compare_projected_v23_row`
then compares the native projection with the independently generated legacy
row using the projection profile's exact field set and each native stream's
declared dtype/shape. It also checks the per-episode sample index and reports
missing fields, shape/representation failures, and maximum numeric error.
The same comparison is now attached behind default-off switches in UMI and in
both RealTeleop writer modes. The application episode owner supplies one
identity, legacy output remains authoritative, and a native compatibility
receipt becomes `complete` only after the finalized legacy episode record is
resolved and hashed. This remains provisional shadow evidence, not native-v2.5
eligibility and not a v2.4 commit merge.

`dataset_interface` 是训练时输入层，不负责离线生成或修复 mask。它读取 DexTeleop dataset
的 RGB 视频与 wrist final keep-mask，并可在 DataLoader worker 内为 front 派生固定边界 mask，
统一交给同一套在线 masked augmentation。

推荐层级：

```text
utils.dataset_postprocess        # 离线清理、对齐、wrist mask 生成
  -> wrist final_keep_mask
dataset_interface                # 训练时读取视频+mask并增强
  -> front_fixed_border_replace  # 可选的在线固定边界 mask
  -> full_image_photometric_v1   # 可选的全图非几何增强，默认启用
  -> model Dataset/DataLoader
  -> model-specific tensor/normalization transforms
```

默认 mask 语义：

```text
mask > 0  保留原始 RGB
mask == 0 替换/增强背景
```

## 基本用法

```python
from pathlib import Path

from dataset_interface import DatasetInterfaceConfig, MaskConfig, MaskedFramePipeline
from dataset_interface.random_background import FixedBorderMaskConfig, RandomBackgroundAugmentConfig

config = DatasetInterfaceConfig(
    dataset_root=Path("datasets/2026-07-09_vive_rm65_real_teleop"),
    mask=MaskConfig(
        final_mask_root=Path("utils/dataset_postprocess/wrist_cam_adjust/outputs/final_keep_mask"),
    ),
    augmentation=RandomBackgroundAugmentConfig(
        provider="mixed",
        mixed_original_prob=0.1,
        real_image_prob=0.4,
        mixed_black_prob=0.0,
        wrist_pair_mode="shared_scene_family",
        mask_role="keep",
        edge_blur_px=1,
    ),
)

with MaskedFramePipeline(config) as pipeline:
    image = pipeline.read_augmented_frame(
        episode_index=0,
        camera_key="left_wrist",
        frame_index=123,
        epoch=0,
    )
```

`read_augmented_frame` 默认返回 OpenCV BGR `HWC uint8`。如需模型常用格式：

```python
from dataset_interface import VideoConfig

config = DatasetInterfaceConfig(
    dataset_root=...,
    mask=...,
    video=VideoConfig(output_layout="rgb_chw_float32"),
)
```

`provider="mixed"` 的默认有效分布是 10% original / 40% real-image / 50% procedural。
`MaskedFramePipeline` 会自动从仓库默认图片池建立 provider，也可以通过
`DatasetInterfaceConfig(real_image_root=...)` 指定图片根目录；图片池不可用时 fail closed。

## Front 固定边界增强

front 不使用 wrist 的 `final_keep_mask`。启用后，`MaskedFramePipeline` 会在现有
`RandomBackgroundVideoAugmenter` 内生成 `front_fixed_border_replace`：上、下、左、右边界各以
25% 概率独立替换，厚度分别为图像高/宽的 5%（1/20）。每个被选中的边都有自己的随机背景，但同一
episode 和 epoch 内不会随 frame 移动或改变。

```python
config = DatasetInterfaceConfig(
    dataset_root=...,
    mask=MaskConfig(final_mask_root=...),  # front 启用固定边界时不会读取此路径
    augmentation=RandomBackgroundAugmentConfig(
        provider="mixed",
        mixed_original_prob=0.1,
        real_image_prob=0.4,
        mixed_black_prob=0.0,
        fixed_mask=FixedBorderMaskConfig(enabled=True),
    ),
)

with MaskedFramePipeline(config) as pipeline:
    front = pipeline.read_augmented_frame(
        episode_index=0,
        camera_key="front",
        frame_index=123,
        epoch=0,
    )
```

默认固定 mask 的 epoch 变化沿用 `RandomBackgroundAugmentConfig.enable_epoch_variation`：同一
epoch 内静态，不同 epoch 可确定性重采样。设
`FixedBorderMaskConfig(enable_epoch_variation=False)` 可让一个 episode 在整个训练期间保持同一
边界和背景。front 样本的 `mask_manifest_path` 为 `None`，诊断信息位于
`MaskedFrameSample.metadata["fixed_mask"]`；这些信息不会添加到模型输入。

## 全图图像增强

默认的 `RandomBackgroundAugmentConfig` 含有启用状态的 `PhotometricAugmentConfig`。它在 wrist
背景替换或 front 固定边界替换后，对完整 BGR 图像施加非几何的曝光/对比度/Gamma、白平衡、饱和度/低
概率色相、非均匀照明和单一相机质量退化。它不改变图像大小、物体位置、mask 或视角；不会使用 crop、
平移、旋转、翻转、cutout 或随机黑块。

```python
from dataset_interface.random_background import PhotometricAugmentConfig, RandomBackgroundAugmentConfig

training_augmentation = RandomBackgroundAugmentConfig(
    provider="mixed",
    photometric=PhotometricAugmentConfig(
        # Warm/cool BGR gains are respectively (0.85, 1.00, 1.15)
        # and (1.15, 1.00, 0.50).
        enabled=True,
    ),
)
```

背景增强默认以左右腕成对的 `episode + epoch` 场景族确定性缓存；程序背景共享 coarse canvas，真实图片
背景共享 provider/family 但使用独立 donor/crop/运动流，左右腕仍使用独立 view seed。`wrist_pair_mode="independent"`
可回退到旧的逐 camera 行为。Photometric
仍以 `episode + camera + epoch` 为粒度确定性缓存；一个 clip/batch 复用同一套全局颜色、光照和质量
参数。Gaussian/Poisson 噪声仍以 `frame_index` 确定性变化。审计信息仅在
`MaskedFrameSample.metadata["photometric_augmentation"]` 中可见；
`read_augmented_frame`、`read_augmented_clip` 和 `TorchMaskedAugmentDataset` 只交付图像数组，
不会把这项 metadata 加入训练 batch。验证、replay 与部署不应复用训练增强配置，可显式设置：

```python
PhotometricAugmentConfig(enabled=False)
```

## 高分辨率视角遮挡

需要模拟部分遮挡或完整缺失视角时，启用 `ViewOcclusionConfig`，并按相机传入
`occlusion_ratio`。该增强在背景替换和 photometric 之后、layout/tensor 转换之前作用于 source
resolution 图像；模型侧应随后只做固定 resize/letterbox 和 normalization，不得增加 random crop 或
random resized crop。

```python
from dataset_interface.random_background import ViewOcclusionConfig

config = DatasetInterfaceConfig(
    dataset_root=...,
    mask=...,
    augmentation=RandomBackgroundAugmentConfig(
        view_occlusion=ViewOcclusionConfig(enabled=True),
    ),
)

frame = pipeline.read_augmented_frame(
    episode_index=0,
    camera_key="left_wrist",
    frame_index=123,
    occlusion_ratio=0.45,
)
```

`ratio=0` 保持进入遮挡层的图像不变；`ratio=1` 返回 source-resolution 全黑图；中间值默认按
`40% rectangle / 40% Gaussian blob / 20% random grid` 生成接近目标面积的高分辨率 mask。几何计划
在受限工作网格生成后映射回 source resolution，避免全分辨率 float score 和排序成为 DataLoader
瓶颈。真实缺失视角由调用方传
`occlusion_ratio=1, forced_missing_view=True`。`TorchMaskedAugmentDataset` 可通过
`occlusion_ratio_getter(sample, camera_key, epoch)` 为不同相机提供独立 ratio。

### 模型端显式允许缺失视角

数据集选择由模型训练端负责。模型在为缺失视角数据集 B 创建 pipeline 时，可以显式传入参考数据集 A，
或者直接声明缺失相机的 source-resolution 尺寸。尺寸统一使用 `(height, width)`：

```python
from dataset_interface import MissingViewConfig

# A 有 front，B 没有；A 只提供同名相机的分辨率信息，不提供图像内容。
b_config = DatasetInterfaceConfig(
    dataset_root=dataset_b,
    mask=...,
    missing_views=MissingViewConfig(
        enabled=True,
        reference_dataset_roots=(dataset_a,),
    ),
)

# 只有 B 时由模型显式声明尺寸。
b_only_config = DatasetInterfaceConfig(
    dataset_root=dataset_b,
    mask=...,
    missing_views=MissingViewConfig(
        enabled=True,
        explicit_resolutions={"front": (480, 640)},
    ),
)
```

解析在 DataLoader 读取前完成。B 自身已有相机、显式尺寸和所有参考数据集的同名相机必须声明一致的
分辨率，否则 pipeline 初始化立即报错。episode 没有该 camera 字段、字段为空或引用的视频文件不存在
时，不打开视频和 mask，直接生成对应 `H×W×3 uint8` 全黑图；clip 同样直接生成全黑 batch。输出 metadata
记录 `source_view_missing=true`、尺寸来源以及 `forced_missing_view=true`。如果既没有显式尺寸，也没有
任何参考来源，则保持 fail-closed。

`MissingViewConfig` 不会把 `modality_validity.json` 中显式标记为 invalid 的现有视频自动改成黑图；这种
情况仍按数据质量错误拒绝。只有结构性缺失的字段/文件使用上述替代规则。

## 归一化 state 的噪声与 dropout

连续 `observation.state` 的训练增强位于 `dataset_interface.state_augmentation`，但它**必须在
模型自己的 state normalizer 之后**调用；本接口不会猜测不同模型的归一化统计。它只处理 float32
连续 state，不会改动 action、`observation.arm.qpos` 审计列或
`observation.state.discrete` 的 int64 token。

```python
import torch

from dataset_contracts import STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT
from dataset_interface import NormalizedStateInputAdapter, StateAugmentationConfig

# state_norm = model_normalizer(policy_batch.observation_state)
augmentation = StateAugmentationConfig(
    enabled=True,
    noise_std=0.02,              # normalized units; iid N(0, 0.02^2) per element
    dropout_probability=0.10,
    dropout_granularity="whole_state",  # or "side"
)
adapter = NormalizedStateInputAdapter(
    layout=policy_io_layout,
    state_input_mode=STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT,
    config=augmentation,
)
generator = torch.Generator(device=state_norm.device).manual_seed(worker_seed)
model_state = adapter.torch_model_input(state_norm, generator=generator)
```

启用时严格执行：

```text
noise     ~ iid Normal(0, 0.02²)                  # every normalized float element
state_aug = (state_norm + noise) * mask
model_in  = concat(state_aug, mask)                # masked artifact only
```

`whole_state` 以 10% 概率将一条样本的所有连续 state 置零；`side` 则独立以 10% 概率屏蔽
left 或 right 的全部连续 arm pose 与该侧连续末端 state。离散末端 token 保持原值。操作仅生成
批量大小的 float noise/mask，NumPy 对 batch 向量化执行；除最多两个 side/component 写入外没有
逐元素 Python 循环，也不会新增数据集 I/O 或离线文件。若 state normalizer 在 Torch/GPU 中，使用
`torch_model_input(...)`：它以同一算法和配置在原 tensor device 上执行，不会发生 CPU 往返。

默认 `StateAugmentationConfig(enabled=False)`，因此旧训练配方直接使用原 `D` 维 state，不变。
关闭新训练增强只需保持该默认值（或设 `enabled=False`）。只有启用 dropout 的新模型才可使用
`masked_normalized_concat_v1`：其 float 输入是 `2D`，后半是 binary mask。artifact 必须显式绑定
该模式及训练配置：

```python
from dataset_contracts import PolicyArtifactContract

artifact = PolicyArtifactContract(
    policy_io_layout=policy_io_layout,
    state_input_mode="masked_normalized_concat_v1",
    state_augmentation=augmentation.to_dict(),
)
```

部署不会注入噪声或随机 dropout。对该 artifact，v2.2 deployment adapter 自动提供 all-one mask；
兼容的模型/server 必须只归一化前 `D` 个 base-state 元素，并保持后 `D` 个 mask 元素为 0/1。旧
artifact 使用 `raw_state_v1`，部署和模型输入仍是原来的 `D` 维。

## v2.3 pose action chunk

`PoseActionChunkContract` 和 `pose_action_chunk_from_episode_rows(...)` 将同一份绝对 pose
episode 映射成 absolute 或 relative 固定 horizon 训练样本。adapter 返回起点 observation、
action chunk、`action_is_pad` 和 measured TCP anchor；`actions_for_normalization()` 只返回
mask 为 false 的真实 action。`DexTeleopDatasetIndex.pose_action_chunk_locations(...)` 按每个
episode 的 `frame_count` 枚举合法起点，因此不会跨 episode。

默认 `tail_policy="drop"`，只生成完整 chunk。模型可显式选择 `tail_policy="pad_last"`，
在 pose 转换后重复最后一个有效 action。reader 不会自行从 episode 长度猜测或切换策略。

native v2.3 数据进入训练前可运行：

```bash
python -m utils.dataset_postprocess validate-v23 /path/to/dataset
```

这是只读的 finalized-dataset acceptance scan，不会把 v2.2 compatibility view 改写或标记成
native v2.3。

checkpoint artifact 不能只因为 model-facing 接口是 v2.3 就把训练来源统一写成“v2.3”。
从旧 v2.2 raw 通过默认 identity compatibility 构造 relative action 时，应显式使用：

```python
from dataset_contracts import PoseTrainingDataContract

pose_training_data = PoseTrainingDataContract.v22_raw_identity(
    active_sides=("left", "right"),
)
```

只有训练 rows 自身带完整 v2.3 pose-frame contract、且 reader 确实生成 canonical-desk state
时，才使用 `PoseTrainingDataContract.native_v23()`。两者都放入
`PolicyArtifactContract.pose_training_data`；部署据此分别限制 identity compatibility runtime
或要求显式 canonical-desk runtime contract，避免旧 raw state checkpoint 被误当作原生 v2.3。

若训练 rows 来自 `v22-to-v23-desk` 的物理坐标迁移产物，使用
`PoseTrainingDataContract.v22_canonical_desk()`。它要求显式 canonical-desk runtime
binding，但保留原始来源为 v2.2；不得写成 `native_v23()`。

## 训练数据有效性

`meta/modality_validity.json` 是可选文件。文件不存在时，兼容历史数据集：所有 episode 和
所有已声明视频视角默认都可训练。

采集时明确忽视的相机/触觉属于数据集级契约，写在同一个文件的 `capture` 字段，而不是
episode metadata：

```json
{
  "schema_version": "dexteleop.modality_validity.v1",
  "capture": {
    "schema_version": "dexteleop.capture_modalities.v1",
    "scope": "dataset",
    "ignored_modalities": {
      "camera": ["right_wrist"],
      "tactile": ["left"]
    },
    "missing_value_policy": {
      "camera": "frame_absent_valid_false",
      "tactile": "value_absent_valid_false"
    }
  }
}
```

仅全新或空白数据集可以首次绑定、改写该契约；数据集一旦包含帧或 episode，后续追加必须
与已有契约完全一致。没有 `capture` 声明的历史非空数据集按“没有忽视项”处理。训练数据选择器
可通过 `DexTeleopDatasetIndex.ignored_camera_keys` 和 `ignored_tactile_keys` 在加载样本前筛选
数据集，不能假设同一数据集内存在不同的 episode 忽视规则。

当采集故障导致整个 episode 或某个相机视角无效时，可以使用以下 v1 格式标记。`valid: false`
的 episode 会被跳过；某个 `video.<camera_key>` 为 false 时，只有请求该视角的训练样本会被跳过。

```json
{
  "schema_version": "dexteleop.modality_validity.v1",
  "training": {
    "valid": true,
    "allowed_video_camera_keys": ["front", "left_wrist", "right_wrist"],
    "episodes": {
      "42": {
        "valid": false,
        "reason": "tracker reset during recording"
      },
      "57": {
        "modalities": {
          "video": {
            "left_wrist": {
              "valid": false,
              "reason": "camera stream lost"
            }
          }
        }
      }
    }
  }
}
```

`training.valid: false` 表示整个数据集不能用于训练。`allowed_video_camera_keys` 是兼容此前
远端原型的全局 allow-list；未提供时不额外限制视角。所有 episode id 和 camera key 都会在加载
metadata 时校验，错误的 policy 不会被静默忽略。

`MaskedFramePipeline` 在打开视频或 mask 前执行检查；直接调用时发现无效数据会打印红色
warning 并抛出 `InvalidTrainingDataError`。`TorchMaskedAugmentDataset` 会打印同样的 warning，
跳过无效 episode 并选择后续可用样本，因此无效 episode 不会进入本接口的图像加载/增强流程。
为保持 map-style Dataset 的长度，它会以确定性顺序补位，因而有效样本可能被重复采样；若训练器
需要严格改变采样分布，应先用 `DexTeleopDatasetIndex.training_eligibility(...)` 构建自己的 sampler。

默认配置只是在数据加载层拒绝无效样本，不会静默改变模型输入。只有模型端显式启用
`MissingViewConfig` 并提供可验证的参考数据集或尺寸时，结构性缺失视角才会替换为全黑输入。若模型输入
本身需要减少视角，仍须创建匹配的 dataset/model contract 与训练 artifact。

## Clip

连续 clip 使用同一 video reader 和 mask reader，顺序读时避免重复 seek：

```python
clip = pipeline.read_augmented_clip(
    episode_index=0,
    camera_key="right_wrist",
    start_frame=100,
    length=16,
    stride=1,
)
```

## PyTorch-style 包装

```python
from dataset_interface.adapters.torch import TorchMaskedAugmentDataset

wrapped = TorchMaskedAugmentDataset(
    base_dataset,
    pipeline_factory=lambda: MaskedFramePipeline(config),
    camera_keys=("front", "left_wrist", "right_wrist"),
)
```

被包装的 sample 默认需要包含 `episode_index` 和 `frame_index`。`TorchMaskedAugmentDataset`
为兼容旧训练配方默认只包含双 wrist；需要 front 固定边界增强时，像上例一样显式加入 `front`。
输出会新增：

```text
observation.images.front.augmented
observation.images.left_wrist.augmented
observation.images.right_wrist.augmented
```

## 性能约定

- 每个 DataLoader worker 内创建自己的 `MaskedFramePipeline`。
- `VideoReaderCache` 缓存 mp4 reader，避免每帧重新打开视频。
- `BitpackMaskReaderCache` 缓存 wrist final mask 文件句柄。
- `RandomBackgroundVideoAugmenter` 缓存 episode/camera/epoch 的动态背景流，以及固定边界的
  每侧静态背景；同时缓存全图增强的采样计划、照明增益图和 motion-blur kernel。
- wrist final keep-mask 应提前离线生成；训练时不要再合成 depth/table mask。front 固定边界是
  唯一的在线派生 mask，且不需要 bitpack 文件。
- 模型归一化、tensor conversion 和 batch collation 由下游训练代码控制。

## 输入路径

默认 wrist final mask 路径：

```text
<final_mask_root>/<dataset_name>/episode_000000/<camera_key>/manifest.json
```

如果未来 dataset 的 `meta/episodes.jsonl` 中提供 `final_mask_paths`、
`wrist_final_mask_paths` 或 `wrist_keep_mask_paths`，`dataset_interface` 会优先使用这些显式路径。
采集期 `wrist_depth_mask_paths` 不属于训练 final-mask key，缺失 final keep-mask 时会 fail closed，
不会静默退化为 depth-range mask。需要兼容实验时应在上层训练配置显式声明，而不能改变这里的默认契约。
这不影响启用固定边界增强的 front，相应 mask 在训练时在线生成。
