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
  dataset_episode_maintenance.py
  dataset_layout.py
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
