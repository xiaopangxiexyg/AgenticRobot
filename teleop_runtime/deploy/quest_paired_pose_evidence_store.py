"""Explicit persistence and offline fitting for sealed Quest pose evidence.

This application-leaf module does not participate in capture, robot state
production, calibration acceptance, frame installation, or rendering.  A
caller must first obtain a sealed :class:`PairedPoseCaptureEvidence`, then
explicitly publish, load, or diagnostically fit it.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any

from dataset_contracts.canonical_encoding import (
    canonical_bytes,
    canonical_content_hash,
    canonical_loads,
)
from dataset_contracts.quest_paired_pose_artifact import (
    QUEST_PAIRED_POSE_CAPTURE_ARTIFACT_KIND_V1,
    QUEST_PAIRED_POSE_CAPTURE_FILE_DOMAIN_V1,
    QUEST_PAIRED_POSE_CAPTURE_FILE_SUFFIX_V1,
    QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_HASH_DOMAIN_V1,
    QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_SCHEMA_V1,
    require_sha256_hex_digest,
)
from hardware.quest_dev.features.visualization.paired_pose_calibration import (
    PairedPoseFit,
    fit_paired_pose_calibration,
)
from hardware.quest_dev.features.visualization.paired_pose_capture import (
    PairedPoseCaptureEvidence,
    parse_paired_pose_capture_evidence,
)
from runtime_artifact_paths import RuntimeArtifactLayout


class QuestPairedPoseEvidenceStoreError(RuntimeError):
    """Raised when persisted capture evidence is missing or invalid."""


@dataclass(frozen=True)
class QuestPairedPoseEvidenceCommit:
    """Receipt for one immutable, content-addressed evidence publication."""

    path: Path
    evidence_semantic_content_hash: str
    dataset_semantic_sha256: str
    file_sha256: str
    byte_length: int
    idempotent: bool

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a Path")
        require_sha256_hex_digest(
            self.evidence_semantic_content_hash,
            field="evidence_semantic_content_hash",
        )
        require_sha256_hex_digest(
            self.dataset_semantic_sha256,
            field="dataset_semantic_sha256",
        )
        require_sha256_hex_digest(self.file_sha256, field="file_sha256")
        if (
            isinstance(self.byte_length, bool)
            or not isinstance(self.byte_length, int)
            or self.byte_length < 0
        ):
            raise ValueError("byte_length must be a non-negative integer")
        if not isinstance(self.idempotent, bool):
            raise TypeError("idempotent must be a bool")


@dataclass(frozen=True)
class QuestPairedPoseDiagnosticFit:
    """Candidate-only fit bound to the full capture and its derived dataset."""

    evidence_semantic_content_hash: str
    dataset_semantic_sha256: str
    fit: PairedPoseFit
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        evidence_hash = require_sha256_hex_digest(
            self.evidence_semantic_content_hash,
            field="evidence_semantic_content_hash",
        )
        dataset_hash = require_sha256_hex_digest(
            self.dataset_semantic_sha256,
            field="dataset_semantic_sha256",
        )
        if not isinstance(self.fit, PairedPoseFit):
            raise TypeError("fit must be a PairedPoseFit")
        if self.fit.dataset_sha256 != dataset_hash:
            raise ValueError(
                "diagnostic fit dataset hash differs from capture evidence"
            )
        object.__setattr__(self, "evidence_semantic_content_hash", evidence_hash)
        object.__setattr__(self, "dataset_semantic_sha256", dataset_hash)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self._semantic_payload(),
                domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_HASH_DOMAIN_V1,
            ),
        )

    def _semantic_payload(self) -> dict[str, Any]:
        return {
            "schema": QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_SCHEMA_V1,
            "evidence_semantic_content_hash": self.evidence_semantic_content_hash,
            "dataset_semantic_sha256": self.dataset_semantic_sha256,
            "fit": self.fit.as_dict(),
            "candidate_only": True,
            "calibration_accepted": None,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._semantic_payload(),
            "semantic_content_hash": self.semantic_content_hash,
        }


def _require_layout(layout: RuntimeArtifactLayout) -> RuntimeArtifactLayout:
    if not isinstance(layout, RuntimeArtifactLayout):
        raise TypeError("layout must be a RuntimeArtifactLayout")
    return layout


def _validated_evidence(
    evidence: PairedPoseCaptureEvidence,
) -> PairedPoseCaptureEvidence:
    if not isinstance(evidence, PairedPoseCaptureEvidence):
        raise TypeError("evidence must be a PairedPoseCaptureEvidence")
    return parse_paired_pose_capture_evidence(evidence.as_dict())


def quest_paired_pose_evidence_path(
    *,
    layout: RuntimeArtifactLayout,
    evidence_semantic_content_hash: str,
) -> Path:
    """Allocate the content-addressed path without touching the filesystem."""

    layout = _require_layout(layout)
    digest = require_sha256_hex_digest(
        evidence_semantic_content_hash,
        field="evidence_semantic_content_hash",
    )
    return layout.evidence_dir(
        kind=QUEST_PAIRED_POSE_CAPTURE_ARTIFACT_KIND_V1
    ) / f"{digest}{QUEST_PAIRED_POSE_CAPTURE_FILE_SUFFIX_V1}"


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise ValueError(f"Quest evidence directory is invalid: {path}")
        os.fsync(fd)
    finally:
        os.close(fd)


def _require_real_directory(path: Path) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Quest evidence directory path is invalid: {path}")


def _ensure_directory_chain(path: Path) -> None:
    """Create and durably bind each missing directory component."""

    missing: list[Path] = []
    existing = path
    while not existing.exists():
        missing.append(existing)
        if existing.parent == existing:
            break
        existing = existing.parent
    current = existing
    while True:
        _require_real_directory(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            pass
        _require_real_directory(directory)
        _fsync_directory(directory.parent)


def _read_regular_file_no_follow(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"Quest evidence path is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _existing_matches(path: Path, encoded: bytes, *, domain: str) -> bool:
    existing = _read_regular_file_no_follow(path)
    if existing != encoded:
        raise ValueError(f"immutable Quest evidence differs at {path}")
    canonical_loads(existing, expected_domain=domain)
    return True


def _publish_immutable_canonical(
    path: Path,
    payload: Mapping[str, Any],
    *,
    domain: str,
) -> bool:
    """Publish one Quest artifact without becoming a generic store owner.

    The deploy domain is forbidden from importing ``collection_runtime``.
    Keeping this helper private also avoids presenting a second reusable
    persistence API; a future second consumer should trigger extraction to a
    neutral owner rather than copying this mechanism again.
    """

    encoded = canonical_bytes(dict(payload), domain=domain)
    _ensure_directory_chain(path.parent)
    if path.exists() or path.is_symlink():
        return _existing_matches(path, encoded, domain=domain)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    result: bool
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-directory hard link gives the content-addressed target
            # create-if-absent semantics that os.replace() does not provide.
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            result = _existing_matches(path, encoded, domain=domain)
        else:
            result = False
            _fsync_directory(path.parent)
        temporary.unlink()
        _fsync_directory(path.parent)
        return result
    finally:
        if temporary.exists():
            temporary.unlink()
            _fsync_directory(path.parent)


def _load_evidence_path(
    path: Path,
    *,
    expected_evidence_hash: str,
) -> tuple[PairedPoseCaptureEvidence, bytes]:
    encoded = _read_regular_file_no_follow(path)
    payload = canonical_loads(
        encoded,
        expected_domain=QUEST_PAIRED_POSE_CAPTURE_FILE_DOMAIN_V1,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Quest paired-pose evidence root must be a mapping")
    if canonical_bytes(
        dict(payload),
        domain=QUEST_PAIRED_POSE_CAPTURE_FILE_DOMAIN_V1,
    ) != encoded:
        raise ValueError("Quest paired-pose evidence bytes are not canonical")
    evidence = parse_paired_pose_capture_evidence(payload)
    if evidence.semantic_content_hash != expected_evidence_hash:
        raise ValueError("Quest evidence path does not bind its semantic hash")
    return evidence, encoded


def publish_sealed_quest_paired_pose_evidence(
    *,
    layout: RuntimeArtifactLayout,
    evidence: PairedPoseCaptureEvidence,
) -> QuestPairedPoseEvidenceCommit:
    """Explicitly publish one strictly replayed, candidate-only capture."""

    layout = _require_layout(layout)
    checked = _validated_evidence(evidence)
    path = quest_paired_pose_evidence_path(
        layout=layout,
        evidence_semantic_content_hash=checked.semantic_content_hash,
    )
    try:
        idempotent = _publish_immutable_canonical(
            path,
            checked.as_dict(),
            domain=QUEST_PAIRED_POSE_CAPTURE_FILE_DOMAIN_V1,
        )
        loaded, encoded = _load_evidence_path(
            path,
            expected_evidence_hash=checked.semantic_content_hash,
        )
        if loaded.as_dict() != checked.as_dict():
            raise ValueError("published Quest evidence differs after strict reload")
    except (OSError, TypeError, ValueError) as exc:
        raise QuestPairedPoseEvidenceStoreError(
            f"Cannot publish sealed Quest paired-pose evidence {path}: {exc}"
        ) from exc
    return QuestPairedPoseEvidenceCommit(
        path=path,
        evidence_semantic_content_hash=checked.semantic_content_hash,
        dataset_semantic_sha256=checked.dataset.semantic_sha256,
        file_sha256=hashlib.sha256(encoded).hexdigest(),
        byte_length=len(encoded),
        idempotent=idempotent,
    )


def load_sealed_quest_paired_pose_evidence(
    *,
    layout: RuntimeArtifactLayout,
    evidence_semantic_content_hash: str,
) -> PairedPoseCaptureEvidence:
    """Strictly load one content-addressed candidate evidence artifact."""

    path = quest_paired_pose_evidence_path(
        layout=layout,
        evidence_semantic_content_hash=evidence_semantic_content_hash,
    )
    try:
        evidence, _ = _load_evidence_path(
            path,
            expected_evidence_hash=evidence_semantic_content_hash,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise QuestPairedPoseEvidenceStoreError(
            f"Cannot load sealed Quest paired-pose evidence {path}: {exc}"
        ) from exc
    return evidence


def fit_quest_paired_pose_evidence(
    evidence: PairedPoseCaptureEvidence,
) -> QuestPairedPoseDiagnosticFit:
    """Run the existing AX=XB diagnostic fit on strict capture evidence."""

    checked = _validated_evidence(evidence)
    fit = fit_paired_pose_calibration(checked.dataset)
    if fit.dataset_sha256 != checked.dataset.semantic_sha256:
        raise ValueError("diagnostic fit did not preserve the evidence dataset hash")
    if dict(fit.frames) != dict(checked.dataset.frames):
        raise ValueError("diagnostic fit did not preserve the evidence frames")
    if fit.tool_profile_hash != checked.dataset.tool_profile_hash:
        raise ValueError("diagnostic fit did not preserve the evidence tool profile")
    if fit.sample_count != len(checked.dataset.samples):
        raise ValueError("diagnostic fit sample count differs from capture evidence")
    return QuestPairedPoseDiagnosticFit(
        evidence_semantic_content_hash=checked.semantic_content_hash,
        dataset_semantic_sha256=checked.dataset.semantic_sha256,
        fit=fit,
    )


__all__ = [
    "QuestPairedPoseDiagnosticFit",
    "QuestPairedPoseEvidenceCommit",
    "QuestPairedPoseEvidenceStoreError",
    "fit_quest_paired_pose_evidence",
    "load_sealed_quest_paired_pose_evidence",
    "publish_sealed_quest_paired_pose_evidence",
    "quest_paired_pose_evidence_path",
]
