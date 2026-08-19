# Native tactile bundle storage

Native tactile bundle storage separates tactile payloads from the core dataset
at collection time. It is optional and default-off. The core dataset and the
tactile bundle may later be copied, backed up, or restored independently, while
portable content identities prevent an unrelated bundle from being attached.

This mode is intended for a pre-existing local storage-device root. Persisted
contracts never contain its mount point, absolute path, filesystem device ID,
or a NAS address.

## Physical layout

The core dataset keeps actions, states, videos, Parquet, the episode catalog,
and the core half of each tactile commit:

```text
<core-dataset>/
  data/
  videos/
  meta/
    episodes.jsonl
    tactile_bundle_v1/
      dataset_identity.cjson
      episodes/<episode_uuid>/
        identity.cjson
        core_component.cjson
        bundle_commit.cjson
```

The configured tactile storage-device root contains one UUID-addressed bundle:

```text
<tactile-storage-root>/
  datasets/<dataset_uuid>/
    tactile/chunk-.../episode_.../<side>/
      manifest.json
      <sensor-streams>
    meta/tactile_bundle_v1/
      dataset_identity.cjson
      episodes/<episode_uuid>/
        identity.cjson
        tactile_component.cjson
        bundle_commit.cjson
```

Paths stored in the core rows remain logical paths below `tactile/`. The same
dataset identity, episode identity, and bundle commit are mirrored between the
two roots. Artifact size and SHA-256, manifest identity, sample/tactile frame
alignment, and episode identity are content-bound.

## Collection contract

Add these options to an existing formal real-teleop collection command:

```bash
--tactile-storage external \
--tactile-compression zstd \
--tactile-bundle-root /path/to/pre-existing-local-storage-root \
--tactile-bundle-min-free-gb 200
```

The storage root must already exist, be writable, and not be a symlink.
The core and tactile roots must be disjoint directories; they may be on the
same filesystem. `--tactile-bundle-root` is empty by default; omitting it
preserves the established self-contained writer.
Native separation forces artifact integrity sealing. Keep collection-path
compression at `none` when using `--idle-tactile-compression`; the idle worker
resolves the UUID-addressed external bundle and performs verified lossless Zstd
conversion during Hold or other recorder-idle gaps. Inline `zstd`/`lz4` remains
available when idle compression is not requested.

An idle conversion publishes the updated external manifest and tactile component,
mirrors the new episode commit to both roots, updates the core catalog binding,
and only then removes the verified raw predecessor. Cooperative cancellation
before that commit restores the last committed revision; forced termination
remains fail-closed and requires explicit artifact recovery before resume.

Native separation is fail-closed:

- it cannot be enabled in-place on a populated unbound dataset;
- a bound dataset cannot be appended without its matching tactile root;
- missing tactile manifests cannot be committed as a valid native episode;
- there is no fallback that silently writes tactile payloads into the core;
- a materialized core `tactile/` tree is read-only and cannot be appended as a
  native-split dataset.

Episode publication order is:

1. close and hash tactile payloads and manifests;
2. publish core and tactile episode components;
3. publish the tactile-side commit;
4. publish the byte-identical core-side commit;
5. append the authoritative core `meta/episodes.jsonl` row.

The episode is therefore invisible to normal catalog readers until both roots
contain the committed identity.

## Verify or reattach a moved bundle

Activate the project environment and run a full read-only attachment check:

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
python -m utils.dataset_postprocess \
  tactile-bundle-status \
  /path/to/core-dataset \
  --tactile-root /path/to/tactile-storage-root
```

The command discovers `datasets/<dataset_uuid>` from the core identity. It
validates every committed episode, mirrored commit, artifact size and hash,
manifest ownership, and core-to-tactile frame alignment. For a quick admission
check, `--no-verify-payload-hashes` skips payload hashing but still verifies
identities, sizes, manifest hashes, and alignment. Do not use that shortcut as
the final verification before deleting another copy.

To move a bundle, copy the entire
`datasets/<dataset_uuid>` directory so the destination remains
`<new-root>/datasets/<dataset_uuid>`. Then run the full status command against
the new root. A differently named path or a hardlink is not an independent
backup. Do not remove the source until the destination is on the intended
device, the full status check passes, and an independent backup or copy receipt
has been retained.

## Restore a self-contained dataset

Materialization copies all committed artifacts into a fresh staging tree on the
core filesystem, verifies exact hashes and stream decoding, and atomically
publishes it as `<core-dataset>/tactile`. It never hardlinks bundle payloads and
refuses to overwrite an existing target.

First obtain the current content-bound plan:

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
python -m utils.dataset_postprocess \
  tactile-bundle-materialize \
  /path/to/core-dataset \
  --tactile-root /path/to/tactile-storage-root
```

Copy `plan_hash` and `dataset_uuid` from that JSON into the apply command:

```bash
python -m utils.dataset_postprocess \
  tactile-bundle-materialize \
  /path/to/core-dataset \
  --tactile-root /path/to/tactile-storage-root \
  --plan-hash '<plan_hash>' \
  --confirm-dataset-uuid '<dataset_uuid>' \
  --confirm-no-active-writers \
  --confirm-no-active-readers \
  --apply
```

The plan hash binds the dataset identity and the exact set of episode commits,
not only file counts. Stop collection and all other dataset writers before
confirming the apply command. Apply performs another full source verification
and rechecks the exact commit set immediately before atomic publication. A failed
pre-publication attempt leaves a named staging directory and
`RESTORE_FAILED.json` for inspection, while `tactile/` remains absent. A
successful restore writes a materialization receipt below
`meta/tactile_bundle_v1/materializations/`.

Existing readers and training tools that expect `dataset_root/tactile/...` can
use the materialized dataset. Code that intentionally reads a separated bundle
can use `dataset_interface.tactile_bundle.attach_tactile_bundle` to validate and
resolve committed logical paths without materializing.

## Interrupted collection recovery

On restart, a uniquely provable uncommitted next-episode tail is moved—not
deleted—into `meta/tactile_bundle_v1/recovery/abandoned/` on the root that owns
each artifact. Matching receipts are written to both roots before the recovery
journal is cleared. The legacy episode index can then be reused safely.

Recovery stops without moving data when an orphan is not exactly the unique
next episode, identities conflict, the core catalog changed during recovery,
or a source/target path is ambiguous. These cases require explicit offline
inspection rather than heuristic cleanup.
