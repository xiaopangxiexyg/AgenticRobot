# Collection Runtime boundary

`collection_runtime` owns device-, control-, and UI-neutral live collection
lifecycle and persistence mechanics. It is below application frontends such as
RealTeleop and UMI, and above pure dataset contracts and storage interfaces.

It may depend on `dataset_contracts`, `dataset_interface`, and pure domain
contracts. It must not import `teleop_runtime`, `control_station`, `hardware`,
`sim`, concrete robot backends, or GUI modules.

## Current maturity

The package boundary is established, but the collection runtime as a whole is
not complete.

Implemented here:

- bounded asynchronous writing and generic stream sinks;
- a writer-neutral accepted-sample receipt and bounded fail-closed retention
  buffer. The concrete writer decides when acceptance occurred; this runtime
  mechanism never infers it from enqueue success or counters;
- immutable artifact publication and episode-maintenance mechanics;
- default-off Plus episode-curation sidecar persistence bound to an immutable
  source dataset.  The RM65/UMI application session and its operator/UI state
  remain in `teleop_runtime`; the historical import path forwards the store;
- fail-closed capture admission plus journaled, reversible quarantine of a
  provably abandoned final tail Episode. Unknown orphan artifacts still block
  append, while a unique unsealed `N+1` legacy artifact group is moved under
  `meta/capture_recovery/` with a durable receipt before that index is reused;
- shared process-level quiescence detection for capture maintenance. It
  recognizes exact formal `python -m` entrypoint arguments rather than
  arbitrary parent-process command text;
- state-aware tactile postprocess planning for mixed raw/compressed/hot/detached
  datasets. It only composes the existing lossless recompression and verified
  modality-storage transactions; interrupted workflows are replanned from
  physical manifests and durable receipts rather than a second progress file;
- dataset/task layout utilities;
- version-neutral component-stream sample production, capture, chunked
  Parquet persistence, seal, and commit mechanics;
- an isolated compatibility sidecar for the frozen v2.3-based provisional
  branch;
- the first application-neutral progress core: linear/rooted-tree profile
  validation, typed fact reduction, immediate Advance undo, durable
  Failure-as-outbox, application Hold/Resume plus formal-capture Pause/Resume
  receipts, bound Episode-clock validation, retry, termination, restart, and
  fail-closed seal;
- a single append-only `facts.cjsonl` per progress Episode. A torn final record
  is reported as incomplete and is never silently truncated or sealed.
- a separate refinement-annotation store for already sealed progress Episodes:
  an immutable binding, bounded append batches published as small canonical
  chunks, restart recovery, idempotent annotation-ID retry, and a manifest that
  seals the exact ordered annotation/chunk hashes. The caller owns the
  directory; no global dataset topology is frozen.
- a read-only managed task-profile catalog under `progress/profiles/`, separated
  into `linear/` and `tree/`. Each task bundle contains its own coarse progress
  profile and may contain an exact-hash-bound refinement profile for internal
  post-annotation stages. RealTeleop currently admits only the linear catalog.

Online Undo is intentionally local: it can only reverse the latest Advance
before any later progress fact. Failure closes the current Attempt; after
Resume, the controller can only create a Retry on that same progress step.
Entering the next step always requires a later explicit successful Advance.
Historical timeline correction belongs to sealed, read-time overlays in
`dataset_interface`, not this runtime reducer.

`ProgressFactStore` treats a write or fsync failure as an uncertain durability
boundary. The live store becomes poisoned and cannot append or seal again;
the caller must close and reopen it so a complete newline-delimited record can
be replayed, or a torn record can fail closed.

采集侧的普通 JSON inventory、压缩 manifest 和恢复 journal 由 `atomic_json.py`
共享临时文件写入、文件同步、原子替换及临时文件清理。调用方仍保留原有的编码和
目录同步策略：bundle 使用目录类型检查，capture recovery 保留目录缺失检查，
modality storage 保留受限的目录同步重试。替换前失败不发布新内容；替换后目录同步
失败仍向调用方报错，但文件已经可见，不能假定尚未发布或已经回滚，也不能据此报告
事务成功；恢复由原事务 owner 决定。事务准入、
不可变 canonical 文件发布和流式数据写入各自保留现有 owner。

触觉重压缩在尝试发布 manifest 前登记回收责任。发布或后续验证失败时，先恢复原始
manifest，再移除本轮生成的压缩载荷；目录同步失败也必须走这条恢复路径，因为原子
替换可能已经生效。恢复失败时继续报错并保留 raw 和新压缩载荷，不删除可能仍被引用
的数据。正常成功后的校验、原始载荷移除和 receipt 顺序不变。

Native tactile bundle 重压缩也使用一个恢复出口，按原 metadata、原 manifests、
本轮压缩载荷的顺序处理；任一恢复失败即中断后续清理，避免内层恢复失败后外层仍删掉
被引用的数据。恢复快照逐 Episode 建立，不回退此前已成功提交的 Episode。恢复失败时
metadata 图可能仍不一致，两套载荷保留不代表完整回滚；成功验证后的 raw 回收失败则
保留已提交的新图，不回退到可能引用已删除 raw 的旧版本。

同一次 native 重压缩调用保留一份 catalog 行集合，逐 Episode 显式原地更新主 binding
及其 metadata 镜像，再编码供发布使用。后续 Episode 直接继承前面已更新的记录，
不依赖外层浅复制保留的嵌套共享引用。内存更新不代表提交成功；任何失败仍中止本次
调用，磁盘状态由该 Episode 的既有备份和恢复顺序负责。

Native 压缩清理以已挂接的全部 Episode component 的 artifact 集合为引用依据，
在删除前一次建立保护集；历史 predecessor 和推导出的压缩目标使用同一保护集，
不随 manifest 遍历顺序逐步重建。候选的实际路径身份必须位于 bundle 的 `tactile/`
范围内，避免历史字段触及 bundle metadata 或外部文件。身份比较解析路径别名，实际
删除仍作用于原路径，保留“删除 symlink 本身”的语义。设备、inode、链接数和哈希
门禁继续执行；该流程仍要求两个存储根静止，不提供并发写入期间的删除保证。

旧的自包含布局同样先读完 finalized Episode 的全部 tactile manifests，并验证所有
raw 流，再清理未提交的压缩目标。保护集包含 manifest 本身、raw 来源及全部流的当前
载荷引用，避免删除被其他 Episode 使用的文件；任一 manifest 或 raw 验证失败时尚未
执行删除。路径身份沿用读者的相对、绝对和外部载荷兼容性；已引用目标直接保留，
未引用目标继续执行设备、inode、链接数门禁，并只 unlink 原候选路径。调用方仍负责
确认数据集静止，不提供并发写入期间的删除保证。

Capture-time progress effective time is monotonic within the bound Episode
clock. Nested Hold/Resume operation times must also be knowable by the receipt's
availability time. The live reducer and read-time resolver enforce the same
rules. Seal fsyncs and closes the append descriptor before publishing the
manifest, so the manifest remains the final commit marker.

Still planned here:

- only generic progress ports/mechanics demonstrated necessary by application
  adapters; application-specific RealTeleop, UMI, and Control Station bindings
  do not belong in this package;
- property/fault fixtures beyond the current deterministic restart and torn-log
  coverage;
- any future physical batching change if progress event volume proves large
  enough to require it. Logical fact and manifest hashes remain independent of
  that future layout.

Owned by other packages and still required for the broader p2 goal:

- broader retrospective correction handling beyond the current sealed,
  terminal-linear Failure-reassignment prototype;
- annotation-set selection policy and broader derived training views in
  `dataset_interface`; structural per-Attempt coverage and its deletable SQLite
  WAL cache are already implemented there;
- any future immutable correction contract and thin resolved-view publication
  contract in `dataset_contracts`, frozen only after real capture data and at
  least two consumers establish their minimum common payload. Current coverage
  and exclusion objects remain rebuildable `dataset_interface` views rather
  than two redundant canonical receipts;
- controlled real-hardware validation. The optional rendered Control Station
  progress panel, typed intents, and application-leaf coordination workflow
  are implemented outside this package; none of them are
  `collection_runtime` responsibilities. UMI remains deferred.

The Episode identity preparation clock and the progress/sample clock are
separate contracts. `ProgressFactStore` binds facts and seal time to
`ProgressEpisodeBindingV1.bound_clock_domain_id`; it does not require that clock
to equal `EpisodeCaptureIdentity.prepared_clock_domain_id`. This permits, for
example, a monotonic preparation clock with a wall-clock UMI sample axis without
weakening per-Episode clock validation.

Known transitional seam:

- `ComponentCaptureEpisodeStore` still preserves the frozen p1
  `task_profile`/`begin_subtask` compatibility behavior so existing provisional
  artifacts and callers remain exact. New progress work must not extend that
  surface; it will use the independent `progress` reducer/store. The old p1
  seam can be removed only after its reader and callers have a verified
  replacement.

Intentionally owned elsewhere:

- Episode identity/lifecycle decisions, operator intents, GUI state, devices,
  robot control, safety, and source production;
- RealTeleop/UMI source adapters and application assembly;
- schemas and identities (`dataset_contracts`);
- read-time projection, verification, and derived training views
  (`dataset_interface`).

Application integration status:

- RealTeleop has an explicitly enabled, `LINEAR` `FULL_TASK` adapter under
  `teleop_runtime.real_teleop`. It accepts one or more progress steps, reuses
  the formal Episode identity, and seals progress only after the legacy
  Episode is published.
- Control Station owns typed Advance/Undo/Failure intents and the application
  workflow. RealTeleop supplies the concrete Session Hold, formal-capture
  Pause/Resume, and writer-accepted-sample ports. The workflow is only
  constructed when a progress profile is configured.
- The independent progress panel is separately default-off. Tree workflows
  and UMI integration remain disabled/deferred.

## Package organization

```text
collection_runtime/
  component_capture/  # generic component/action/state capture and persistence
  compatibility/      # frozen old-format subordinate sidecars only
  progress/           # online progress + distinct sealed annotation stores
    profiles/         # managed linear/tree task profile assets and catalog
  async_writer.py
  atomic_json.py      # atomic JSON replacement with caller-owned directory sync
  dataset_episode_maintenance.py
  dataset_layout.py
  episode_curation.py # application-neutral external curation sidecar store
  immutable_canonical_store.py
  process_quiescence.py # shared formal-collection process exclusion
  sample_acceptance.py # writer-neutral admission receipt + bounded retention
  source_cut.py       # generic non-committable snapshot result
  stream_sink.py
  tactile_postprocess.py # mixed-state lossless compression/storage orchestration
```

`subtask_interface` is deliberately not used as the umbrella name:
component-stream action/state capture is not subtask semantics, and this
package owns runtime mechanics rather than a dataset interface. The
operator-facing “subtask” feature is represented canonically as progress and
belongs in the separate `progress` domain.

See [`progress/README.md`](progress/README.md) for the explicit distinction
between the frozen sparse refinement-boundary store and the current
ordered-stage annotation store used by the independent Control Station tool.

The persisted provisional contracts still contain `V25p*` schema and hash
names. Those identities cannot be silently renamed. Runtime callers should use
the capability-oriented APIs from `collection_runtime.component_capture` and
`collection_runtime.compatibility`. Root `collection_runtime.v25p_*` modules
and old `teleop_runtime.collection` paths are compatibility shims; new
production imports must not depend on them.

This package adds no production entrypoint and does not change the default
v2.3 or provisional native capture behavior.
