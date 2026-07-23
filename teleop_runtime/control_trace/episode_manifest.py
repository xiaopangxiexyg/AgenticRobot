"""Atomic per-episode publication for v2.4 seal assessments and commits."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from dataset_contracts.canonical_encoding import (
    canonical_bytes,
    canonical_content_hash,
    canonical_loads,
)
from dataset_contracts.control_trace_episode import (
    CONTROL_TRACE_EPISODE_RELATIVE_ROOT_V1,
    EPISODE_COMMIT_FILE_DOMAIN_V1,
    EPISODE_SEAL_FILE_DOMAIN_V1,
    EpisodeTraceSealAssessment,
    V24EpisodeCaptureCommit,
)
from dataset_contracts.stream_catalog import (
    NATIVE_STREAM_EPISODE_MANIFEST_FILE_DOMAIN_V1,
    NATIVE_STREAM_EPISODE_MANIFEST_HASH_DOMAIN_V1,
    NATIVE_STREAM_EPISODE_RELATIVE_ROOT_V1,
)


EPISODE_MANIFEST_RELATIVE_ROOT_V1 = Path(CONTROL_TRACE_EPISODE_RELATIVE_ROOT_V1)


@dataclass(frozen=True)
class EpisodeManifestPublishReceipt:
    episode_uuid: str
    record_kind: str
    semantic_content_hash: str
    relative_path: str
    durable: bool
    idempotent: bool


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        count = os.write(fd, payload[offset:])
        if count <= 0:
            raise OSError("Episode manifest write made no progress.")
        offset += count


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class EpisodeCaptureManifestStore:
    """Publishes immutable metadata; it does not own legacy episode lifecycle."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)
        self.root = self.dataset_root / EPISODE_MANIFEST_RELATIVE_ROOT_V1

    def _publish(
        self,
        *,
        episode_uuid: str,
        record_kind: str,
        semantic_content_hash: str,
        payload: dict[str, object],
        domain: str,
    ) -> EpisodeManifestPublishReceipt:
        episode = str(episode_uuid).strip()
        if not episode or "/" in episode or episode in {".", ".."}:
            raise ValueError("episode_uuid is not safe for manifest publication.")
        episode_dir = self.root / episode
        episode_dir.mkdir(parents=True, exist_ok=True)
        target = episode_dir / f"{record_kind}.cjson"
        encoded = canonical_bytes(payload, domain=domain)
        relative = target.relative_to(self.dataset_root).as_posix()
        if target.exists():
            existing = target.read_bytes()
            if existing != encoded:
                raise ValueError(
                    f"Immutable episode {record_kind} already exists with different content."
                )
            canonical_loads(existing, expected_domain=domain)
            return EpisodeManifestPublishReceipt(
                episode_uuid=episode,
                record_kind=record_kind,
                semantic_content_hash=semantic_content_hash,
                relative_path=relative,
                durable=True,
                idempotent=True,
            )

        temporary = episode_dir / f".{record_kind}.{uuid.uuid4().hex}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(temporary, flags, 0o600)
        try:
            _write_all(fd, encoded)
            os.fsync(fd)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            finally:
                os.close(fd)
            raise
        os.close(fd)
        try:
            os.replace(temporary, target)
            _fsync_directory(episode_dir)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return EpisodeManifestPublishReceipt(
            episode_uuid=episode,
            record_kind=record_kind,
            semantic_content_hash=semantic_content_hash,
            relative_path=relative,
            durable=True,
            idempotent=False,
        )

    def publish_seal_assessment(
        self,
        assessment: EpisodeTraceSealAssessment,
    ) -> EpisodeManifestPublishReceipt:
        if not isinstance(assessment, EpisodeTraceSealAssessment):
            raise TypeError("assessment must be EpisodeTraceSealAssessment.")
        return self._publish(
            episode_uuid=assessment.trace_range.episode_uuid,
            record_kind="seal_assessment",
            semantic_content_hash=assessment.semantic_content_hash,
            payload=assessment.to_payload(),
            domain=EPISODE_SEAL_FILE_DOMAIN_V1,
        )

    def publish_capture_commit(
        self,
        commit: V24EpisodeCaptureCommit,
    ) -> EpisodeManifestPublishReceipt:
        if not isinstance(commit, V24EpisodeCaptureCommit):
            raise TypeError("commit must be V24EpisodeCaptureCommit.")
        episode_dir = self.root / commit.episode_uuid
        seal_path = episode_dir / "seal_assessment.cjson"
        if not seal_path.is_file():
            raise ValueError("Cannot publish capture commit before its seal assessment.")
        seal_payload = canonical_loads(
            seal_path.read_bytes(),
            expected_domain=EPISODE_SEAL_FILE_DOMAIN_V1,
        )
        if not isinstance(seal_payload, dict):
            raise ValueError("Persisted episode seal assessment payload is invalid.")
        if seal_payload.get("semantic_content_hash") != commit.trace_seal_hash:
            raise ValueError("Capture commit does not reference the persisted seal assessment.")
        if not bool(seal_payload.get("capture_commit_eligible")):
            raise ValueError("Persisted seal assessment is not capture-commit eligible.")
        artifact_hashes = dict(commit.capability_artifact_hash_by_name)
        unsupported = set(artifact_hashes) - {"native_streams"}
        if unsupported:
            raise ValueError(
                "Capture commit contains unsupported capability artifacts: "
                f"{sorted(unsupported)}."
            )
        native_stream_hash = artifact_hashes.get("native_streams")
        if native_stream_hash is not None:
            native_manifest_path = (
                self.dataset_root
                / NATIVE_STREAM_EPISODE_RELATIVE_ROOT_V1
                / commit.episode_uuid
                / "manifest.cjson"
            )
            if not native_manifest_path.is_file():
                raise ValueError(
                    "Cannot publish native_streams capability before its "
                    "episode manifest."
                )
            native_payload = canonical_loads(
                native_manifest_path.read_bytes(),
                expected_domain=(
                    NATIVE_STREAM_EPISODE_MANIFEST_FILE_DOMAIN_V1
                ),
            )
            if not isinstance(native_payload, dict):
                raise ValueError("Native stream episode manifest is invalid.")
            native_semantic_payload = dict(native_payload)
            declared_native_hash = native_semantic_payload.pop(
                "semantic_content_hash",
                None,
            )
            if (
                native_payload.get("status") != "sealed"
                or native_payload.get("episode_uuid") != commit.episode_uuid
                or int(native_payload.get("legacy_episode_index", -1))
                != commit.legacy_episode_index
                or declared_native_hash != native_stream_hash
                or canonical_content_hash(
                    native_semantic_payload,
                    domain=NATIVE_STREAM_EPISODE_MANIFEST_HASH_DOMAIN_V1,
                )
                != native_stream_hash
            ):
                raise ValueError(
                    "Capture commit does not match the sealed native stream "
                    "episode manifest."
                )
        return self._publish(
            episode_uuid=commit.episode_uuid,
            record_kind="capture_commit",
            semantic_content_hash=commit.semantic_content_hash,
            payload=commit.to_payload(),
            domain=EPISODE_COMMIT_FILE_DOMAIN_V1,
        )

    def read_seal_payload(self, episode_uuid: str) -> dict[str, object] | None:
        path = self.root / str(episode_uuid) / "seal_assessment.cjson"
        if not path.is_file():
            return None
        value = canonical_loads(path.read_bytes(), expected_domain=EPISODE_SEAL_FILE_DOMAIN_V1)
        if not isinstance(value, dict):
            raise ValueError("Episode seal payload is not a mapping.")
        return value

    def read_commit_payload(self, episode_uuid: str) -> dict[str, object] | None:
        path = self.root / str(episode_uuid) / "capture_commit.cjson"
        if not path.is_file():
            return None
        value = canonical_loads(path.read_bytes(), expected_domain=EPISODE_COMMIT_FILE_DOMAIN_V1)
        if not isinstance(value, dict):
            raise ValueError("Episode commit payload is not a mapping.")
        return value


__all__ = [
    "EPISODE_COMMIT_FILE_DOMAIN_V1",
    "EPISODE_MANIFEST_RELATIVE_ROOT_V1",
    "EPISODE_SEAL_FILE_DOMAIN_V1",
    "EpisodeCaptureManifestStore",
    "EpisodeManifestPublishReceipt",
]
