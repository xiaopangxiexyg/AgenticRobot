# DexTeleop v2.5-p native landing plan

Status: provisional, default-off, no native-v2.5 claim.

## Contract decision

The source of truth is an embodiment-neutral causal component stream, not a
v2.3 row and not a policy tensor. Each sample references immutable component
records by hash. Action semantics are explicit stages such as source intent,
accepted reference, admitted command, and dispatched command. State and
observation streams retain their own effective/availability time and clock.

The frozen v1 `dataset_contracts.v25p` layout remains readable as a v2.3
compatibility branch. New native artifacts use `meta/v25p/native/`. Conversion
to v2.3 is an adapter operation with a typed receipt; conversion failure never
invalidates a sealed native capture.

The corrected provisional native action contract is schema v2. Its action
payload and action-space definitions are the same pure contracts used by
`robot_motion` and `control_composition`; no HIL-specific representation or
second schema hash exists. Frozen stream-profile/embodiment v1 artifacts remain
readable through explicit schema dispatch. New native action profiles use v2;
the RealTeleop/UMI shadow producers have passed the offline migration gate and
now select V2 only behind their existing default-off switch.

## Ownership and module boundaries

| Package | Owns | Must not own |
| --- | --- | --- |
| `dataset_contracts` | immutable identities, embodiment/stream/task schemas, hashes, manifests, projection receipts | devices, writers, episode lifecycle |
| `teleop_runtime.collection` | subordinate native persistence driven by an external episode owner | robot-specific schema inference, second episode authority, compatibility writer lifecycle |
| `dataset_interface` | verification, read views, explicit compatibility projections | source mutation, episode commit, runtime/device control |
| RealTeleop / UMI recorder | the single episode lifecycle and shared identity | duplicate component semantics or independent sidecar identities |

The current `legacy_episode_index` field in `EpisodeCaptureIdentity` is a
transitional owner ordinal. Native readers do not locate or require a legacy
episode record from it. A future identity schema may rename the field, but the
frozen v1 identity and hashes must remain readable.

## Native artifact topology

The logical contracts do not depend on one physical encoding. The original
`record_cjson_v1` fallback remains readable and is still the default while the
new backend requires an explicit native-shadow storage selection:

```text
meta/v25p/native/
  dataset_manifest.cjson
  profiles/
    embodiment/<hash>.cjson
    component_stream/<hash>.cjson
    task/<hash>.cjson
    compatibility_projection/<hash>.cjson
  episodes/<episode_uuid>/
    identity.cjson
    embodiment_binding.cjson
    # record_cjson_v1 fallback
    streams/<stream_id>/<sequence>.cjson
    samples/<sample_index>.cjson

    # parquet_stream_chunks_v1 alternative
    streams/s<stream_ordinal>/chunk_<chunk_index>.parquet
    samples/chunk_<chunk_index>.parquet
    chunks/chunk_<chunk_index>.cjson
    native_storage_manifest.cjson

    task_events/<sequence>.cjson
    task_manifest.cjson
    native_capture_manifest.cjson
    compatibility_projections/<receipt_hash>.cjson
    episode_manifest.cjson
```

Small numeric/bool component values are canonical inline tensors. Large sensor
payloads are external artifact references containing an immutable content hash,
encoding, and optional selector/range. The native store binds descriptors; the
existing media/blob owner remains responsible for actual artifact durability.

### Parquet chunk physical decision

- A chunk contains one causal-sample Parquet shard and one shard per component
  stream present in that chunk. It does not create one file per component
  record and does not use a sparse union table.
- Inline tensors use a fixed-size Arrow list with the exact declared scalar
  dtype. External media remains a typed content-hash reference rather than a
  byte blob inside Parquet. Record and sample semantic hashes are fixed 32-byte
  columns.
- Logical record/sample/capture hashes remain independent of Parquet byte
  layout, compression, row-group metadata, and future physical migrations.
  Every written table is read back and compared before its immutable chunk
  manifest is published.
- One background storage worker serializes chunks. At most two complete chunks
  may wait behind it; sustained storage lag therefore becomes bounded
  backpressure rather than unbounded memory growth. The active partial chunk is
  also bounded. The native store does not retain the episode's logical record
  and sample objects on this path.
- A storage manifest is published only after all ordered chunk manifests are
  durable. An interruption may leave auditable immutable chunks, but without a
  storage manifest, native capture manifest, and episode manifest it cannot be
  opened as a valid episode.
- Random sample lookup uses manifest sample ranges to open one sample chunk and
  only the requested stream chunks. Full verification still checks every file
  SHA-256, Arrow schema, reconstructed logical hash, sequence, and bundle hash.
- `none` and `zstd` are the only physical compression choices. They produce the
  same logical content and capture manifest.

This is intentionally not a byte-for-byte LeRobot v2.1 layout. The verified
native reader plus the existing explicit v2.3/LeRobot row adapter is the
conversion boundary: it reconstructs only declared compatible fields, keeps
dtype/shape, reports omissions as lossy, and then hands rows to the existing
LeRobot writer. No robot-specific concatenated action vector is introduced into
the native source of truth.

For the six-stream, 256-frame offline fixture with 64-frame chunks, the old
layout publishes 1,792 record/sample data files at seal. The chunk layout uses
28 Parquet data files, four chunk manifests, and one storage manifest, plus the
unchanged episode identity/binding/capture/commit artifacts. Local full-test
duration fell from 18.59 seconds to 0.97 seconds while preserving every native
hash and every projected v2.3 field.

## Landing gates

1. **Pure contract/store/reader gate — landed**
   - native capture seals without `meta/episodes.jsonl`;
   - stream/profile/identity/causal hashes are verified fail-closed;
   - inline and external payload representations are distinct;
   - task attempts share the owner identity;
   - in-place episode reindexing is blocked once native identity exists.
   - corrected v2 action stream profiles embed and validate the canonical
     action-space, payload-schema, and component-catalog definitions;
   - v1/v2 embodiment and stream profiles dispatch by schema ID and both CJSON
     and Parquet stores round-trip v2 profiles without changing logical hashes;
   - multidimensional state/observation tensors restore their declared shape
     for validation after flattened physical storage.

1a. **Canonical ComponentAction projection gate — landed, default-off activated**
   - `dataset_interface` performs explicit component-to-stream projection and
     owns no timing, authority, target resolution, safety, or lifecycle;
   - `teleop_runtime.collection` only attaches caller-supplied effective and
     availability time, retaining the source ComponentAction hash as lineage;
   - full and partial component actions fail closed on required coverage,
     action-space mismatch, or schema drift;
   - offline CJSON and Parquet lifecycle tests verify exact v2 round-trip;
   - RealTeleop thread/process and UMI select this path only with the explicit
     `--v25p-native-shadow` flag; default formal collection is unchanged.

2. **Compatibility adapter gate — landed at component-row boundary**
   - exact component/role/stage/schema matching only;
   - ambiguity, invalid selected evidence, and unsupported external fields fail;
   - omitted native streams are reported as lossy;
   - publication receipts are separate from projection planning.

3. **Recorder producer-adapter gate — landed default-off**
   - RealTeleop and UMI component producer adapters consume their own
     authoritative sample boundaries and do not import each other;
   - explicit producer bindings cover required native streams and fail on
     role/stage/profile mismatch rather than inferring component names;
   - UMI and RealTeleop attach the native store only behind an explicit,
     default-off switch;
   - RealTeleop thread and process modes consume the same assembled sample and
     already-planned legacy row; process mode does not add a second queue;
   - built-in profiles declare only active sides and present end effectors;
     pose/qpos are separate native streams, and velocity is absent until a
     producer supplies it;
   - built-in action streams use the shared canonical `ComponentAction` schema
     and generic dataset-boundary projector rather than a second action hash;
     UMI retains each side's original pose effective/availability clocks;
   - native seal precedes any compatibility finalization receipt;
   - no extra episode owner, action queue, or safety authority.

4. **Dual-write equivalence gate — offline lifecycle and hot-path validation landed**
   - native store round-trip plus adapter-projected v2.3 fields are compared
     against independently generated RealTeleop and UMI row-encoder output;
   - comparison includes sample identity, declared native dtype/shape, missing
     fields, representability, exact/default-zero-tolerance values, and numeric
     error reporting;
   - publish a `complete` compatibility receipt only after exact final output is
     hashed; otherwise publish `failed`/`skipped`;
   - the existing episode owner mints one identity and passes it unchanged to
     both legacy and native writers; legacy finalization remains authoritative;
   - a 256-sample offline episode preserves exact projection for every sample;
   - process-mode interruption cannot publish a native capture/episode commit,
     and a fresh owner identity can restart at the same legacy ordinal;
   - shadow sample files are deferred out of the capture hot path; the measured
     process-worker shadow append over 48 samples was p50 0.55 ms, p95 0.62 ms,
     and max 0.71 ms in the local verification environment;
   - the full v2.5-p/UMI/RealTeleop recorder regression set passes offline;
   - the offline `parquet_stream_chunks_v1` physical gate now provides bounded
     chunk buffering, background publication, crash-incomplete fail-closed
     behavior, exact dtype/hash round trips, per-sample random reads, and exact
     v2.3 projection equivalence;
   - RealTeleop thread/process modes and UMI now accept the same explicit
     `--v25p-native-storage-layout` selection, but only when the already
     default-off `--v25p-native-shadow` path is enabled;
   - a 1,024-sample spawned process-worker soak published eight ordered chunks,
     completed the legacy/native equivalence receipt, reloaded the sealed native
     episode, and kept shadow append p95 below the 10 ms test gate;
   - the runtime-facing default remains `record_cjson_v1`; no formal collection
     command silently changes storage layout. Native-v2.5 eligibility remains
     pending the separate v2.4 merge gate rather than this physical backend.

The explicit provisional selection is the same on both frontends:

```text
--v25p-native-shadow \
--v25p-native-storage-layout parquet_stream_chunks_v1
```

Omitting both flags leaves all existing collection behavior unchanged. Passing
the Parquet layout without enabling the native shadow fails during configuration
instead of being silently ignored.

4a. **v2.5-p2 collection-progress semantics gate — planned, not runtime-enabled**
   - the superseding contract, ownership, atomic advance/undo, failure-derived
     training selection, post-annotation, and migration plan lives in
     `todos/dexteleop_v25p2_collection_semantics_plan.md`;
   - frozen V1 Subtask artifacts remain readable and keep their original hash
     domain; p2 Progress Step/Attempt/refinement contracts use new schema IDs
     rather than widening V1 in place;
   - reusable progress profiles define unique step instances, while each
     episode binding independently declares full-task/window scope, actual
     entry step, and whether prehistory is complete;
   - failure closes the active attempt before the application-owned Hold;
     retry requires typed applied Hold/Resume receipts. The current project
     contract fully pauses capture during Hold rather than introducing a second
     raw-capture path;
   - the existing episode owner remains authoritative and passes the same
     capture identity and sample clock to a subordinate semantic writer;
   - capture progress facts and post-capture refinement annotation sets are
     separately append-only and do not rewrite raw sample/media artifacts;
   - semantic facts are chunked instead of published as one file per event;
   - training exclusion may shrink only at a validated salvage anchor; explicit
     post-annotation corrections may adjust retrospective progress/outcome
     views without mutating capture-time facts;
   - reusable capture-time semantics will move behind a new backend-neutral
     `collection_runtime` boundary instead of remaining owned by
     `teleop_runtime` or being placed in a use-case-shaped `long_horizon`
     package;
   - the first implementation gate is pure p2 contracts and offline validation;
     RealTeleop/UMI attachment remains default-off and long-horizon GUI work is
     explicitly deferred.

5. **v2.4 merge gate — pending**
   - reference an existing immutable v2.4 capture commit; do not widen or mutate
     its frozen capability allowlist;
   - define the final v2.5 commit that jointly verifies native samples, task
     semantics, compatibility receipts, and selected v2.4 capabilities;
   - only that final contract may set native-v2.5 eligibility.

## Rollback

The native path is additive and default-off. The physical layout is selected
when constructing the subordinate store; `record_cjson_v1` remains the default.
Rollback means selecting that layout or disabling native store attachment while
keeping existing v2.3/v2.4 writers unchanged. Readers detect a sealed Parquet
layout from its storage manifest, so both layouts can remain readable. Already
sealed native artifacts remain immutable; no rollback may rewrite their
identities or hashes. Removal requires a copy-on-write derived dataset or
explicit whole-episode disposal before sealing.
