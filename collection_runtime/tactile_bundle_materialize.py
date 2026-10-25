"""Safe fresh-inode materialization of an attached tactile bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.tactile import ChunkedTactileStream, tactile_stream_compression
from dataset_interface.tactile_bundle import (
    AttachedTactileBundle,
    attach_tactile_bundle,
)

from .atomic_json import atomic_write_json
from .immutable_canonical_store import fsync_directory


class TactileBundleMaterializationError(RuntimeError):
    """Raised when safe tactile materialization cannot be completed."""


@dataclass(frozen=True)
class TactileBundleMaterializationPlan:
    dataset_uuid: str
    dataset_identity_hash: str
    episode_commit_set_hash: str
    artifact_count: int
    payload_count: int
    payload_bytes: int
    target_relative_path: str = "tactile"

    @property
    def plan_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(include_hash=False),
            domain="dexteleop.tactile_bundle.materialization_plan.v1",
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "dexteleop.tactile_bundle.materialization_plan.v1",
            "dataset_uuid": self.dataset_uuid,
            "dataset_identity_hash": self.dataset_identity_hash,
            "episode_commit_set_hash": self.episode_commit_set_hash,
            "artifact_count": self.artifact_count,
            "payload_count": self.payload_count,
            "payload_bytes": self.payload_bytes,
            "target_relative_path": self.target_relative_path,
        }
        if include_hash:
            payload["plan_hash"] = self.plan_hash
        return payload


@dataclass(frozen=True)
class TactileBundleMaterializationReport:
    dataset_uuid: str
    artifact_count: int
    payload_count: int
    payload_bytes: int
    target_path: Path
    receipt_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "dexteleop.tactile_bundle.materialization_report.v1",
            "dataset_uuid": self.dataset_uuid,
            "artifact_count": self.artifact_count,
            "payload_count": self.payload_count,
            "payload_bytes": self.payload_bytes,
            "target_path": str(self.target_path),
            "receipt_path": str(self.receipt_path),
            "complete": True,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload, sync_directory=fsync_directory)


def plan_tactile_bundle_materialization(
    core_dataset_root: str | Path,
    tactile_storage_root: str | Path,
) -> TactileBundleMaterializationPlan:
    attached = attach_tactile_bundle(
        core_dataset_root,
        tactile_storage_root,
        verify_payload_hashes=True,
    )
    return _plan_from_attached(attached)


def _plan_from_attached(
    attached: AttachedTactileBundle,
) -> TactileBundleMaterializationPlan:
    report = attached.report
    if report.episode_count <= 0 or report.artifact_count <= 0:
        raise TactileBundleMaterializationError(
            "native tactile bundle has no committed artifacts to materialize"
        )
    episode_commit_set_hash = canonical_content_hash(
        [
            {
                "legacy_episode_index": episode.legacy_episode_index,
                "episode_uuid": episode.episode_uuid,
                "commit_hash": episode.commit.semantic_content_hash,
            }
            for episode in (
                attached.episodes[index]
                for index in sorted(attached.episodes)
            )
        ],
        domain="dexteleop.tactile_bundle.materialization_commit_set.v1",
    )
    return TactileBundleMaterializationPlan(
        dataset_uuid=report.dataset_uuid,
        dataset_identity_hash=attached.dataset_identity.semantic_content_hash,
        episode_commit_set_hash=episode_commit_set_hash,
        artifact_count=report.artifact_count,
        payload_count=report.payload_count,
        payload_bytes=report.payload_bytes,
    )


def _copy_fresh_inode(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_stat = source.stat()
        with source.open("rb") as source_handle, destination.open("xb") as target:
            while True:
                block = source_handle.read(1024 * 1024)
                if not block:
                    break
                written = target.write(block)
                if written != len(block):
                    raise OSError(
                        f"short materialization write: {written}/{len(block)}"
                    )
            target.flush()
            os.fsync(target.fileno())
        destination.chmod(0o644)
        destination_stat = destination.stat()
    except OSError as exc:
        raise TactileBundleMaterializationError(
            f"Cannot copy tactile artifact {source} -> {destination}: {exc}"
        ) from exc
    if (
        int(source_stat.st_dev) == int(destination_stat.st_dev)
        and int(source_stat.st_ino) == int(destination_stat.st_ino)
    ):
        raise TactileBundleMaterializationError(
            f"materialized artifact reused the source inode: {source}"
        )


def _validate_decodable_streams(staging_root: Path) -> None:
    for manifest_path in sorted(staging_root.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TactileBundleMaterializationError(
                f"Cannot read staged tactile manifest {manifest_path}: {exc}"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise TactileBundleMaterializationError(
                f"Staged tactile manifest is not an object: {manifest_path}"
            )
        streams = manifest.get("streams")
        if not isinstance(streams, Mapping):
            raise TactileBundleMaterializationError(
                f"Staged tactile manifest has no stream table: {manifest_path}"
            )
        for sensor_key, raw_spec in streams.items():
            if not isinstance(raw_spec, Mapping):
                raise TactileBundleMaterializationError(
                    f"Invalid staged tactile stream {sensor_key!r}"
                )
            spec = dict(raw_spec)
            logical_path = Path(str(spec.get("path") or ""))
            if not logical_path.parts or logical_path.parts[0] != "tactile":
                raise TactileBundleMaterializationError(
                    f"Invalid staged tactile stream path: {logical_path}"
                )
            stream_path = staging_root.joinpath(*logical_path.parts[1:])
            spec["path"] = str(stream_path)
            frame_count = int(spec.get("frame_count") or 0)
            if frame_count <= 0:
                raise TactileBundleMaterializationError(
                    f"Staged tactile stream has no frames: {logical_path}"
                )
            compression = tactile_stream_compression(spec)
            try:
                if compression == "none":
                    dtype = np.dtype(str(spec.get("dtype") or ""))
                    shape = tuple(int(value) for value in spec.get("shape", ()) or ())
                    expected = int(frame_count * int(np.prod(shape)) * dtype.itemsize)
                    if stream_path.stat().st_size != expected:
                        raise ValueError(
                            f"raw byte count {stream_path.stat().st_size} != {expected}"
                        )
                    array = np.memmap(
                        stream_path,
                        dtype=dtype,
                        mode="r",
                        shape=(frame_count, *shape),
                    )
                    np.asarray(array[0])
                    np.asarray(array[-1])
                else:
                    stream = ChunkedTactileStream(
                        dataset_root=staging_root,
                        spec=spec,
                    )
                    if stream.frame(0) is None or stream.frame(frame_count - 1) is None:
                        raise ValueError("head or tail frame is not decodable")
            except (OSError, TypeError, ValueError) as exc:
                raise TactileBundleMaterializationError(
                    f"Staged tactile stream is not decodable ({sensor_key}): {exc}"
                ) from exc


def materialize_tactile_bundle(
    core_dataset_root: str | Path,
    tactile_storage_root: str | Path,
    *,
    expected_plan_hash: str | None = None,
) -> TactileBundleMaterializationReport:
    """Copy a verified bundle into ``core/tactile`` and atomically expose it.

    The target must not exist.  Any pre-existing staging directory blocks the
    operation for explicit operator inspection; this function never deletes an
    ambiguous prior recovery attempt.
    """

    attached: AttachedTactileBundle = attach_tactile_bundle(
        core_dataset_root,
        tactile_storage_root,
        verify_payload_hashes=True,
    )
    plan = _plan_from_attached(attached)
    if (
        expected_plan_hash is not None
        and str(expected_plan_hash) != plan.plan_hash
    ):
        raise TactileBundleMaterializationError(
            "native tactile bundle changed after the confirmed materialization plan"
        )
    core_root = attached.core_dataset_root
    target = core_root / "tactile"
    if target.exists() or target.is_symlink():
        raise TactileBundleMaterializationError(
            f"Refusing to overwrite existing tactile target: {target}"
        )
    stale_staging = tuple(sorted(core_root.glob(".tactile.native-bundle-staging-*")))
    if stale_staging:
        raise TactileBundleMaterializationError(
            "A previous tactile materialization staging directory requires manual "
            f"inspection: {stale_staging[0]}"
        )
    staging = core_root / f".tactile.native-bundle-staging-{uuid.uuid4()}"
    staging.mkdir(mode=0o755)
    copied = []
    try:
        for artifact in attached.iter_artifacts():
            source = attached.resolve(artifact.logical_path)
            logical = Path(artifact.logical_path)
            destination = staging.joinpath(*logical.parts[1:])
            _copy_fresh_inode(source, destination)
            if int(destination.stat().st_size) != artifact.size_bytes:
                raise TactileBundleMaterializationError(
                    f"Staged artifact size mismatch: {artifact.logical_path}"
                )
            if _sha256_file(destination) != artifact.sha256:
                raise TactileBundleMaterializationError(
                    f"Staged artifact hash mismatch: {artifact.logical_path}"
                )
            copied.append(artifact.logical_path)
        _validate_decodable_streams(staging)
        # Sync all created directories bottom-up before the atomic namespace cut.
        directories = {staging}
        directories.update(path for path in staging.rglob("*") if path.is_dir())
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            fsync_directory(directory)
        # Re-read the portable identities and exact commit set immediately
        # before the namespace cut.  This is deliberately a quick attachment:
        # every copied byte was already hashed above, while commit-set changes
        # (for example an accidentally active collection writer) must abort the
        # materialization rather than publish a stale partial snapshot.
        final_attachment = attach_tactile_bundle(
            core_dataset_root,
            tactile_storage_root,
            verify_payload_hashes=False,
        )
        if _plan_from_attached(final_attachment).plan_hash != plan.plan_hash:
            raise TactileBundleMaterializationError(
                "native tactile bundle changed while it was being materialized"
            )
        os.replace(staging, target)
        fsync_directory(core_root)
    except BaseException as exc:
        if staging.is_dir():
            try:
                _atomic_write_json(
                    staging / "RESTORE_FAILED.json",
                    {
                        "schema": "dexteleop.tactile_bundle.materialization_failure.v1",
                        "dataset_uuid": attached.dataset_identity.dataset_uuid,
                        "copied_artifacts": copied,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            except BaseException:
                pass
        raise

    attachment = attached.report
    receipt_path = (
        core_root
        / "meta"
        / "tactile_bundle_v1"
        / "materializations"
        / f"{attached.dataset_identity.dataset_uuid}.json"
    )
    _atomic_write_json(
        receipt_path,
        {
            "schema": "dexteleop.tactile_bundle.materialization_receipt.v1",
            "dataset_uuid": attachment.dataset_uuid,
            "dataset_identity_hash": attached.dataset_identity.semantic_content_hash,
            "plan_hash": plan.plan_hash,
            "episode_commit_set_hash": plan.episode_commit_set_hash,
            "artifact_count": attachment.artifact_count,
            "payload_count": attachment.payload_count,
            "payload_bytes": attachment.payload_bytes,
            "target_logical_prefix": "tactile",
            "fresh_inode_copy": True,
            "payload_hashes_verified": True,
            "head_tail_decode_verified": True,
            "complete": True,
        },
    )
    return TactileBundleMaterializationReport(
        dataset_uuid=attachment.dataset_uuid,
        artifact_count=attachment.artifact_count,
        payload_count=attachment.payload_count,
        payload_bytes=attachment.payload_bytes,
        target_path=target,
        receipt_path=receipt_path,
    )


__all__ = [
    "TactileBundleMaterializationError",
    "TactileBundleMaterializationPlan",
    "TactileBundleMaterializationReport",
    "materialize_tactile_bundle",
    "plan_tactile_bundle_materialization",
]
