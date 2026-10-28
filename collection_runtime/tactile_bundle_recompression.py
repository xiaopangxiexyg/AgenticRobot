"""Idle lossless recompression for natively separated tactile bundles.

Native tactile storage deliberately keeps the payload tree away from the core
dataset.  The regular self-contained recompressor therefore cannot be pointed
at a native dataset: its episode catalog and modality tree live on different
roots.  This module binds both roots through the read-only native attachment
contract, performs the same verified chunk conversion on the bundle tree, and
publishes a new tactile component/commit revision before retiring raw files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

from dataset_contracts.canonical_encoding import canonical_bytes, canonical_loads
from dataset_contracts.tactile_bundle import (
    TACTILE_BUNDLE_DATASET_IDENTITY_FILE_DOMAIN_V1,
    TACTILE_BUNDLE_EPISODE_COMMIT_FILE_DOMAIN_V1,
    TACTILE_BUNDLE_TACTILE_COMPONENT_FILE_DOMAIN_V1,
    TactileBundleDatasetIdentityV1,
    TactileBundleArtifactV1,
    TactileEpisodeBundleCommitV1,
    TactileEpisodeComponentV1,
    TactileManifestBindingV1,
    core_dataset_identity_relative_path,
    storage_dataset_identity_relative_path,
    tactile_frame_alignment_hash,
)
from dataset_interface.tactile_bundle import attach_tactile_bundle

from .tactile_recompression import (
    DEFAULT_CHUNK_FRAMES,
    DEFAULT_MIN_FREE_BYTES,
    LOSSLESS_PREDICTOR_V1,
    LOSSLESS_PROFILE_V1,
    TactileRecompressionCancelled,
    _ZstdCodec,
    _atomic_write_bytes,
    _atomic_write_json,
    _compress_stream,
    _raw_manifest_streams,
    _read_json,
    _relative_dataset_path,
)


class NativeTactileBundleRecompressionError(RuntimeError):
    """Raised when a native bundle cannot be safely revised in idle time."""


DEFAULT_IDLE_NATIVE_MAX_EPISODES = 1


def _load_identity(core_root: Path) -> TactileBundleDatasetIdentityV1:
    path = core_root / core_dataset_identity_relative_path()
    if not path.is_file():
        raise NativeTactileBundleRecompressionError(
            f"native tactile dataset identity is missing: {path}"
        )
    try:
        payload = canonical_loads(
            path.read_bytes(),
            expected_domain=TACTILE_BUNDLE_DATASET_IDENTITY_FILE_DOMAIN_V1,
        )
        return TactileBundleDatasetIdentityV1.from_payload(payload)
    except Exception as exc:  # noqa: BLE001 - convert to the native boundary.
        raise NativeTactileBundleRecompressionError(
            f"native tactile dataset identity is invalid: {path}: {exc}"
        ) from exc


def resolve_native_bundle_dataset_root(
    core_dataset_root: str | Path,
    tactile_storage_root: str | Path,
    *,
    require_distinct_device: bool = False,
) -> Path:
    """Resolve ``<storage>/datasets/<dataset_uuid>`` with fail-closed checks."""

    configured_core_root = Path(core_dataset_root).expanduser()
    configured_storage_root = Path(tactile_storage_root).expanduser()
    if configured_core_root.is_symlink() or configured_storage_root.is_symlink():
        raise NativeTactileBundleRecompressionError(
            "native tactile core/storage roots must not be symlinks"
        )
    core_root = configured_core_root.resolve()
    storage_root = configured_storage_root.resolve()
    if not core_root.is_dir():
        raise NativeTactileBundleRecompressionError(
            f"core dataset root is unavailable or unsafe: {core_root}"
        )
    if not storage_root.is_dir() or storage_root.is_symlink():
        raise NativeTactileBundleRecompressionError(
            f"tactile storage root is unavailable or unsafe: {storage_root}"
        )
    if (
        core_root == storage_root
        or core_root.is_relative_to(storage_root)
        or storage_root.is_relative_to(core_root)
    ):
        raise NativeTactileBundleRecompressionError(
            "core dataset and tactile storage roots must be disjoint"
        )
    if bool(require_distinct_device) and int(core_root.stat().st_dev) == int(
        storage_root.stat().st_dev
    ):
        raise NativeTactileBundleRecompressionError(
            "native tactile idle recompression requires a distinct storage filesystem"
        )
    identity = _load_identity(core_root)
    datasets = storage_root / "datasets"
    bundle_root = datasets / identity.dataset_uuid
    if (
        datasets.is_symlink()
        or bundle_root.is_symlink()
        or not bundle_root.is_dir()
        or not bundle_root.resolve().is_relative_to(storage_root)
    ):
        raise NativeTactileBundleRecompressionError(
            "bound native tactile UUID bundle is missing or escapes its storage root"
        )
    storage_identity = bundle_root / storage_dataset_identity_relative_path()
    if not storage_identity.is_file():
        raise NativeTactileBundleRecompressionError(
            f"native tactile storage identity is missing: {storage_identity}"
        )
    return bundle_root.resolve()


def _atomic_write_canonical(path: Path, payload: Mapping[str, Any], *, domain: str) -> None:
    _atomic_write_bytes(path, canonical_bytes(dict(payload), domain=domain))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_catalog_rows(core_root: Path) -> list[dict[str, Any]]:
    path = core_root / "meta" / "episodes.jsonl"
    if not path.is_file():
        return []
    original = path.read_bytes()
    rows: list[dict[str, Any]] = []
    try:
        for line in original.decode("utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("episode catalog row is not an object")
                rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise NativeTactileBundleRecompressionError(
            f"native tactile episode catalog is invalid: {path}: {exc}"
        ) from exc
    return rows


def _update_episode_catalog_in_place(
    rows: list[dict[str, Any]],
    *,
    episode_index: int,
    episode_uuid: str,
    commit_hash: str,
) -> bytes:
    """Update the caller's cumulative rows in place and encode the next catalog."""
    matched = 0
    for row in rows:
        if int(row.get("episode_index", -1)) != int(episode_index):
            continue
        binding = row.get("tactile_bundle")
        if (
            not isinstance(binding, dict)
            or str(binding.get("episode_uuid") or "") != episode_uuid
        ):
            continue
        binding["commit_hash"] = str(commit_hash)
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            metadata_binding = metadata.get("tactile_bundle_episode")
            if isinstance(metadata_binding, dict):
                metadata_binding["commit_hash"] = str(commit_hash)
        matched += 1
    if matched != 1:
        raise NativeTactileBundleRecompressionError(
            "native tactile commit revision must match exactly one core catalog row: "
            f"episode_index={episode_index} episode_uuid={episode_uuid!r} matches={matched}"
        )
    return "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _build_updated_tactile_component(
    *,
    existing: TactileEpisodeComponentV1,
    bundle_root: Path,
) -> TactileEpisodeComponentV1:
    artifacts: list[TactileBundleArtifactV1] = []
    manifests: list[TactileManifestBindingV1] = []
    for binding in existing.manifests:
        manifest_path = bundle_root / binding.manifest_logical_path
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise NativeTactileBundleRecompressionError(
                f"native tactile manifest is missing or unsafe: {manifest_path}"
            )
        manifest = _read_json(manifest_path)
        frames = manifest.get("frames")
        streams = manifest.get("streams")
        if not isinstance(frames, list) or not isinstance(streams, Mapping):
            raise NativeTactileBundleRecompressionError(
                f"native tactile manifest is incomplete: {manifest_path}"
            )
        if str(manifest.get("side") or "") != binding.side:
            raise NativeTactileBundleRecompressionError(
                f"native tactile manifest side changed: {binding.manifest_logical_path}"
            )
        if int(manifest.get("frame_count", -1)) != int(binding.frame_count):
            raise NativeTactileBundleRecompressionError(
                f"native tactile manifest frame count changed: {manifest_path}"
            )
        if tactile_frame_alignment_hash(frames) != binding.frame_alignment_hash:
            raise NativeTactileBundleRecompressionError(
                f"native tactile frame alignment changed: {manifest_path}"
            )
        artifacts.append(
            TactileBundleArtifactV1(
                logical_path=binding.manifest_logical_path,
                role="manifest",
                size_bytes=int(manifest_path.stat().st_size),
                sha256=_sha256_file(manifest_path),
            )
        )
        manifests.append(binding)
        for sensor_key, value in sorted(streams.items(), key=lambda item: str(item[0])):
            if not isinstance(value, Mapping):
                raise NativeTactileBundleRecompressionError(
                    f"native tactile stream is not an object: {sensor_key!r}"
                )
            logical_path = str(value.get("path") or "")
            payload = bundle_root / logical_path
            if payload.is_symlink() or not payload.is_file():
                raise NativeTactileBundleRecompressionError(
                    f"native tactile payload is missing or unsafe: {logical_path}"
                )
            size = int(value.get("stored_size_bytes", -1))
            if size != int(payload.stat().st_size):
                raise NativeTactileBundleRecompressionError(
                    f"native tactile payload size mismatch: {logical_path}"
                )
            digest = _sha256_file(payload)
            if digest != str(value.get("stored_sha256") or ""):
                raise NativeTactileBundleRecompressionError(
                    f"native tactile payload hash mismatch: {logical_path}"
                )
            artifacts.append(
                TactileBundleArtifactV1(
                    logical_path=logical_path,
                    role="payload",
                    size_bytes=size,
                    sha256=digest,
                )
            )
    return TactileEpisodeComponentV1(
        dataset_uuid=existing.dataset_uuid,
        episode_uuid=existing.episode_uuid,
        legacy_episode_index=existing.legacy_episode_index,
        episode_identity_hash=existing.episode_identity_hash,
        sample_count=existing.sample_count,
        artifacts=tuple(artifacts),
        manifests=tuple(manifests),
    )


def _manifest_targets(
    bundle_root: Path,
    component: TactileEpisodeComponentV1,
) -> list[tuple[Path, dict[str, Any], list[tuple[str, dict[str, Any], Path]]]]:
    result: list[tuple[Path, dict[str, Any], list[tuple[str, dict[str, Any], Path]]]] = []
    for binding in component.manifests:
        path = bundle_root / binding.manifest_logical_path
        manifest, raw_streams = _raw_manifest_streams(bundle_root, path)
        result.append((path, manifest, raw_streams))
    return result


def _clean_native_orphans(
    bundle_root: Path,
    components: tuple[TactileEpisodeComponentV1, ...],
    *,
    chunk_frames: int,
) -> int:
    """Remove only deterministic targets or raw predecessors left by a stop."""

    removed = 0
    referenced: set[str] = set()
    for component in components:
        for binding in component.manifests:
            manifest_path = bundle_root / binding.manifest_logical_path
            manifest = _read_json(manifest_path)
            streams = manifest.get("streams", {})
            if not isinstance(streams, Mapping):
                continue
            for value in streams.values():
                if not isinstance(value, Mapping):
                    continue
                target = str(value.get("path") or "")
                if target:
                    referenced.add(target)
                source_info = value.get("recompressed_from")
                if isinstance(source_info, Mapping):
                    source = str(source_info.get("path") or "")
                    if source:
                        source_path = bundle_root / source
                        if source_path.is_file() and source not in referenced:
                            stat = source_path.stat()
                            if int(source_info.get("device", stat.st_dev)) != int(stat.st_dev):
                                raise NativeTactileBundleRecompressionError(
                                    f"native tactile orphan source device changed: {source}"
                                )
                            if int(source_info.get("inode", stat.st_ino)) != int(stat.st_ino):
                                raise NativeTactileBundleRecompressionError(
                                    f"native tactile orphan source inode changed: {source}"
                                )
                            if int(stat.st_nlink) != 1:
                                raise NativeTactileBundleRecompressionError(
                                    f"native tactile orphan source has unexpected links: {source}"
                                )
                            declared_raw_hash = str(
                                source_info.get("raw_sha256") or ""
                            )
                            if (
                                declared_raw_hash
                                and _sha256_file(source_path) != declared_raw_hash
                            ):
                                raise NativeTactileBundleRecompressionError(
                                    f"native tactile orphan source hash changed: {source}"
                                )
                            source_path.unlink()
                            removed += 1
    # A target created before manifest publication is still discoverable from
    # the raw manifest.  Never remove a target that is referenced by a
    # committed manifest.
    for component in components:
        for binding in component.manifests:
            manifest_path = bundle_root / binding.manifest_logical_path
            _manifest, raw_streams = _raw_manifest_streams(bundle_root, manifest_path)
            for _sensor, _spec, source in raw_streams:
                target = source.with_name(f"{source.name}.tdelta{int(chunk_frames)}.zst")
                if not target.is_file():
                    continue
                target_stat = target.stat()
                source_stat = source.stat()
                if target_stat.st_dev != source_stat.st_dev or target_stat.st_nlink != 1:
                    raise NativeTactileBundleRecompressionError(
                        f"native tactile orphan target has unsafe inode: {target}"
                    )
                target.unlink()
                removed += 1
    return removed


def recompress_native_tactile_bundle_zstd_in_place(
    core_dataset_root: str | Path,
    tactile_storage_root: str | Path,
    *,
    level: int = 1,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    backend: str = "auto",
    cancel_requested: Callable[[], bool] | None = None,
    require_distinct_device: bool = False,
    max_episodes: int | None = None,
    verify_historical_payload_hashes: bool = True,
) -> dict[str, Any]:
    """Recompress one native bundle while both roots are quiesced.

    Direct/offline callers keep the historical full-payload validation default.
    The live idle worker passes a bounded episode count and disables repeated
    historical payload reads: all identities, manifest hashes, frame alignment,
    ownership, and artifact sizes are still checked for the whole attachment,
    while the raw/updated episode payload bytes are fully hashed before their
    own commit. This keeps the runtime Home boundary from re-reading every
    previously sealed tactile stream after each new Episode.
    """

    if int(chunk_frames) <= 0:
        raise ValueError("chunk_frames must be positive")
    if max_episodes is not None and int(max_episodes) <= 0:
        raise ValueError("max_episodes must be positive when provided")
    core_root = Path(core_dataset_root).expanduser().resolve()
    storage_root = Path(tactile_storage_root).expanduser().resolve()
    bundle_root = resolve_native_bundle_dataset_root(
        core_root,
        storage_root,
        require_distinct_device=bool(require_distinct_device),
    )
    # This validates the complete pre-maintenance graph, including the UUID,
    # mirrored commits, component hashes, manifest hashes, frame alignment,
    # ownership, and artifact sizes.  Historical payload bytes are optionally
    # skipped for the bounded live-idle path and are verified below only for
    # the episode(s) that will be mutated.
    attached = attach_tactile_bundle(
        core_root,
        storage_root,
        verify_payload_hashes=bool(verify_historical_payload_hashes),
    )
    components = tuple(
        episode.tactile_component
        for episode in sorted(
            attached.episodes.values(),
            key=lambda item: item.legacy_episode_index,
        )
    )
    orphan_count = _clean_native_orphans(
        bundle_root,
        components,
        chunk_frames=int(chunk_frames),
    )
    if cancel_requested is not None and cancel_requested():
        raise TactileRecompressionCancelled("native tactile recompression cancelled before work")

    codec: _ZstdCodec | None = None
    catalog_rows = _read_catalog_rows(core_root)
    catalog_path = core_root / "meta" / "episodes.jsonl"
    manifest_count = 0
    stream_count = 0
    raw_total = 0
    compressed_total = 0
    removed_total = 0
    updated_episodes = 0

    episode_work: dict[
        int,
        tuple[
            list[tuple[Path, dict[str, Any], list[tuple[str, dict[str, Any], Path]]]],
            list[tuple[Path, dict[str, Any], str, dict[str, Any], Path]],
        ],
    ] = {}
    for attached_episode in sorted(
        attached.episodes.values(), key=lambda item: item.legacy_episode_index
    ):
        if cancel_requested is not None and cancel_requested():
            raise TactileRecompressionCancelled(
                "native tactile recompression cancelled before the next episode"
            )
        manifests = _manifest_targets(bundle_root, attached_episode.tactile_component)
        raw_entries = [
            (manifest_path, manifest, sensor, spec, source)
            for manifest_path, manifest, raw_streams in manifests
            for sensor, spec, source in raw_streams
        ]
        if not raw_entries:
            continue
        episode_work[attached_episode.legacy_episode_index] = (
            manifests,
            raw_entries,
        )
        if max_episodes is not None and len(episode_work) >= int(max_episodes):
            break

    if not verify_historical_payload_hashes and episode_work:
        attached.verify_payload_hashes_for_episodes(episode_work)

    for attached_episode in sorted(
        (attached.episodes[index] for index in episode_work),
        key=lambda item: item.legacy_episode_index,
    ):
        manifests, raw_entries = episode_work[attached_episode.legacy_episode_index]
        if cancel_requested is not None and cancel_requested():
            raise TactileRecompressionCancelled(
                "native tactile recompression cancelled before the next episode"
            )
        if codec is None:
            codec = _ZstdCodec(level=int(level), backend=backend)
        required = sum(
            int(source.stat().st_size)
            for _, _, _, _, source in raw_entries
        )
        free = int(shutil.disk_usage(bundle_root).free)
        if free - required < int(min_free_bytes):
            raise OSError(
                "insufficient native tactile bundle free space for atomic conversion: "
                f"free={free} required={required} reserve={int(min_free_bytes)}"
            )
        created: list[Path] = []
        metadata_committed = False
        original_metadata: tuple[tuple[Path, bytes], ...] = ()
        original_manifests: dict[Path, bytes] = {}
        source_stats: dict[Path, os.stat_result] = {}
        replacements_by_manifest: dict[Path, dict[str, dict[str, Any]]] = {}
        manifest_totals: dict[Path, tuple[int, int]] = {}
        try:
            catalog_before_episode = catalog_path.read_bytes()
            for manifest_path, _manifest, sensor, spec, source in raw_entries:
                if cancel_requested is not None and cancel_requested():
                    raise TactileRecompressionCancelled(
                        "native tactile recompression cancelled before stream conversion"
                    )
                initial = source.stat()
                if int(initial.st_nlink) != 1:
                    raise NativeTactileBundleRecompressionError(
                        f"native tactile source has unexpected links: {source}"
                    )
                source_stats[source] = initial
                target = source.with_name(f"{source.name}.tdelta{int(chunk_frames)}.zst")
                chunks, raw_sha, stored_sha, compressed_bytes = _compress_stream(
                    source,
                    target,
                    spec,
                    codec=codec,
                    chunk_frames=int(chunk_frames),
                    cancel_requested=cancel_requested,
                    enforce_global_collection_exclusion=False,
                )
                created.append(target)
                updated = dict(spec)
                updated.update(
                    {
                        "path": _relative_dataset_path(bundle_root, target),
                        "encoding": "chunked_delta_zstd",
                        "compression": "zstd",
                        "storage_dtype": "uint8",
                        "predictor": LOSSLESS_PREDICTOR_V1,
                        "chunk_frame_count": int(chunk_frames),
                        "chunks": chunks,
                        "stored_size_bytes": int(compressed_bytes),
                        "stored_sha256": stored_sha,
                        "stored_hash_profile": "stored_bytes_sha256_v1",
                        "recompressed_from": {
                            "path": _relative_dataset_path(bundle_root, source),
                            "encoding": str(spec.get("encoding") or "raw_c_order"),
                            "compression": "none",
                            "raw_sha256": raw_sha,
                            "device": int(initial.st_dev),
                            "inode": int(initial.st_ino),
                            "link_count": int(initial.st_nlink),
                        },
                    }
                )
                replacements_by_manifest.setdefault(manifest_path, {})[sensor] = updated
                previous_raw, previous_compressed = manifest_totals.get(
                    manifest_path,
                    (0, 0),
                )
                manifest_totals[manifest_path] = (
                    previous_raw + int(initial.st_size),
                    previous_compressed + int(compressed_bytes),
                )

            for manifest_path, _manifest, _raw_streams in manifests:
                original_manifests[manifest_path] = manifest_path.read_bytes()
                updated_manifest = dict(_manifest)
                streams = dict(updated_manifest.get("streams") or {})
                streams.update(replacements_by_manifest.get(manifest_path, {}))
                updated_manifest["streams"] = streams
                updated_manifest["tactile_recompression"] = {
                    "profile": LOSSLESS_PROFILE_V1,
                    "compression": "zstd",
                    "level": int(level),
                    "chunk_frame_count": int(chunk_frames),
                    "predictor": LOSSLESS_PREDICTOR_V1,
                    "backend": codec.backend,
                }
                if replacements_by_manifest.get(manifest_path):
                    _atomic_write_json(manifest_path, updated_manifest)

            for source, initial in source_stats.items():
                current = source.stat()
                if (
                    current.st_dev,
                    current.st_ino,
                    current.st_nlink,
                    current.st_size,
                    current.st_mtime_ns,
                ) != (
                    initial.st_dev,
                    initial.st_ino,
                    initial.st_nlink,
                    initial.st_size,
                    initial.st_mtime_ns,
                ):
                    raise NativeTactileBundleRecompressionError(
                        f"native tactile source changed during recompression: {source}"
                    )
            if cancel_requested is not None and cancel_requested():
                raise TactileRecompressionCancelled(
                    "native tactile recompression cancelled before bundle commit"
                )

            updated_component = _build_updated_tactile_component(
                existing=attached_episode.tactile_component,
                bundle_root=bundle_root,
            )
            old_commit = attached_episode.commit
            updated_commit = TactileEpisodeBundleCommitV1(
                dataset_uuid=old_commit.dataset_uuid,
                episode_uuid=old_commit.episode_uuid,
                legacy_episode_index=old_commit.legacy_episode_index,
                episode_identity_hash=old_commit.episode_identity_hash,
                core_component_hash=old_commit.core_component_hash,
                tactile_component_hash=updated_component.semantic_content_hash,
                core_component_ref=old_commit.core_component_ref,
                tactile_component_ref=old_commit.tactile_component_ref,
            )
            updated_catalog = _update_episode_catalog_in_place(
                catalog_rows,
                episode_index=updated_commit.legacy_episode_index,
                episode_uuid=updated_commit.episode_uuid,
                commit_hash=updated_commit.semantic_content_hash,
            )
            core_commit_path = core_root / old_commit.core_component_ref.replace(
                "core_component.cjson", "bundle_commit.cjson"
            )
            tactile_commit_path = bundle_root / (
                Path(old_commit.tactile_component_ref).parent / "bundle_commit.cjson"
            )
            tactile_component_path = bundle_root / old_commit.tactile_component_ref
            original_metadata = (
                (tactile_component_path, tactile_component_path.read_bytes()),
                (tactile_commit_path, tactile_commit_path.read_bytes()),
                (core_commit_path, core_commit_path.read_bytes()),
                (catalog_path, catalog_before_episode),
            )
            _atomic_write_canonical(
                tactile_component_path,
                updated_component.to_payload(),
                domain=TACTILE_BUNDLE_TACTILE_COMPONENT_FILE_DOMAIN_V1,
            )
            _atomic_write_canonical(
                tactile_commit_path,
                updated_commit.to_payload(),
                domain=TACTILE_BUNDLE_EPISODE_COMMIT_FILE_DOMAIN_V1,
            )
            _atomic_write_canonical(
                core_commit_path,
                updated_commit.to_payload(),
                domain=TACTILE_BUNDLE_EPISODE_COMMIT_FILE_DOMAIN_V1,
            )
            _atomic_write_bytes(catalog_path, updated_catalog)
            # Re-attach before deleting raw bytes.  This proves both commit
            # halves and the updated episode's new payload hashes are
            # coherent.  The idle path deliberately avoids re-reading
            # historical payload bytes here; direct/offline callers retain
            # the full validation default.
            committed_attachment = attach_tactile_bundle(
                core_root,
                storage_root,
                verify_payload_hashes=bool(verify_historical_payload_hashes),
            )
            if not verify_historical_payload_hashes:
                committed_attachment.verify_payload_hashes_for_episodes(
                    (attached_episode.legacy_episode_index,)
                )
            metadata_committed = True

            for source in source_stats:
                source.unlink()
                removed_total += int(source_stats[source].st_size)
            updated_episodes += 1
            manifest_count += len(replacements_by_manifest)
            stream_count += len(raw_entries)
            raw_total += sum(raw for raw, _compressed in manifest_totals.values())
            compressed_total += sum(
                compressed for _raw, compressed in manifest_totals.values()
            )
        except BaseException:
            if not metadata_committed:
                # Stop on any restore failure; remaining metadata may still
                # reference the new manifests and compressed payloads.
                for path, data in original_metadata:
                    _atomic_write_bytes(path, data)
                for path, data in original_manifests.items():
                    _atomic_write_bytes(path, data)
                for target in created:
                    target.unlink(missing_ok=True)
            raise

    return {
        "outcome": "completed" if updated_episodes else "no_work",
        "core_dataset_root": str(core_root),
        "bundle_dataset_root": str(bundle_root),
        "dataset_uuid": _load_identity(core_root).dataset_uuid,
        "episode_count": int(updated_episodes),
        "manifest_count": int(manifest_count),
        "stream_count": int(stream_count),
        "raw_bytes": int(raw_total),
        "compressed_bytes": int(compressed_total),
        "removed_original_bytes": int(removed_total),
        "uncommitted_artifacts_removed": int(orphan_count),
        "compression": "zstd",
        "profile": LOSSLESS_PROFILE_V1,
        "max_episodes": None if max_episodes is None else int(max_episodes),
        "historical_payload_hashes_verified": bool(verify_historical_payload_hashes),
    }


def run_idle_native_tactile_recompression_worker(
    core_dataset_root: str,
    stop_event: Any,
    tactile_storage_root: str,
    min_free_bytes: int | None = None,
    require_distinct_device: bool = False,
) -> Mapping[str, Any]:
    """Controller worker entry point for a native, externally stored bundle."""

    if min_free_bytes is not None and int(min_free_bytes) < 0:
        raise ValueError("min_free_bytes must be non-negative")
    if stop_event.is_set():
        return {
            "outcome": "cancelled",
            "dataset_root": str(Path(core_dataset_root).resolve()),
        }
    result = recompress_native_tactile_bundle_zstd_in_place(
        core_dataset_root,
        tactile_storage_root,
        level=1,
        chunk_frames=DEFAULT_CHUNK_FRAMES,
        min_free_bytes=(
            DEFAULT_MIN_FREE_BYTES
            if min_free_bytes is None
            else int(min_free_bytes)
        ),
        backend="auto",
        cancel_requested=stop_event.is_set,
        require_distinct_device=bool(require_distinct_device),
        max_episodes=DEFAULT_IDLE_NATIVE_MAX_EPISODES,
        verify_historical_payload_hashes=False,
    )
    return result


__all__ = [
    "DEFAULT_IDLE_NATIVE_MAX_EPISODES",
    "NativeTactileBundleRecompressionError",
    "recompress_native_tactile_bundle_zstd_in_place",
    "resolve_native_bundle_dataset_root",
    "run_idle_native_tactile_recompression_worker",
]
