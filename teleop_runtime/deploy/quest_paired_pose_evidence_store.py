"""Explicit persistence and offline fitting for sealed Quest pose evidence.

This application-leaf module does not participate in capture, robot state
production, calibration acceptance, frame installation, or rendering. Capture
publication and fitting require a sealed :class:`PairedPoseCaptureEvidence`;
an already-published candidate diagnostic report reloads independently by its
own content hash.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
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
    QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_ARTIFACT_KIND_V1,
    QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_DOMAIN_V1,
    QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_NAME_V1,
    QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_HASH_DOMAIN_V1,
    QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_SCHEMA_V1,
    require_sha256_hex_digest,
)
from hardware.quest_dev.features.visualization.paired_pose_calibration import (
    PairedPoseFit,
    fit_paired_pose_calibration,
    parse_paired_pose_fit,
)
from hardware.quest_dev.features.visualization.paired_pose_capture import (
    PairedPoseCaptureEvidence,
    parse_paired_pose_capture_evidence,
)
from runtime_artifact_paths import RuntimeArtifactLayout


class QuestPairedPoseEvidenceStoreError(RuntimeError):
    """Raised when a persisted Quest paired-pose artifact is missing or invalid."""


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
class QuestPairedPoseDiagnosticFitCommit:
    """Receipt for one immutable, content-addressed diagnostic report."""

    path: Path
    diagnostic_fit_semantic_content_hash: str
    evidence_semantic_content_hash: str
    dataset_semantic_sha256: str
    fit_semantic_sha256: str
    file_sha256: str
    byte_length: int
    idempotent: bool

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a Path")
        for field_name in (
            "diagnostic_fit_semantic_content_hash",
            "evidence_semantic_content_hash",
            "dataset_semantic_sha256",
            "fit_semantic_sha256",
            "file_sha256",
        ):
            require_sha256_hex_digest(getattr(self, field_name), field=field_name)
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


_DIAGNOSTIC_FIT_REPORT_KEYS = {
    "schema",
    "evidence_semantic_content_hash",
    "dataset_semantic_sha256",
    "fit",
    "candidate_only",
    "calibration_accepted",
    "semantic_content_hash",
}


def _parse_quest_paired_pose_diagnostic_fit(
    payload: Mapping[str, Any],
) -> QuestPairedPoseDiagnosticFit:
    if not isinstance(payload, Mapping):
        raise TypeError("Quest paired-pose diagnostic fit must be a mapping")
    missing = _DIAGNOSTIC_FIT_REPORT_KEYS.difference(payload)
    unexpected = set(payload).difference(_DIAGNOSTIC_FIT_REPORT_KEYS)
    if missing or unexpected:
        raise ValueError(
            "diagnostic fit report keys mismatch; "
            f"missing={sorted(missing)}, "
            f"unexpected={sorted(str(item) for item in unexpected)}"
        )
    if payload.get("schema") != QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_SCHEMA_V1:
        raise ValueError(
            "diagnostic fit report schema must be "
            f"{QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_SCHEMA_V1!r}"
        )
    if payload.get("candidate_only") is not True:
        raise ValueError("diagnostic fit report candidate_only must remain true")
    if payload.get("calibration_accepted") is not None:
        raise ValueError("diagnostic fit report calibration_accepted must remain null")
    evidence_hash = require_sha256_hex_digest(
        payload.get("evidence_semantic_content_hash"),
        field="evidence_semantic_content_hash",
    )
    dataset_hash = require_sha256_hex_digest(
        payload.get("dataset_semantic_sha256"),
        field="dataset_semantic_sha256",
    )
    semantic_hash = require_sha256_hex_digest(
        payload.get("semantic_content_hash"),
        field="semantic_content_hash",
    )
    raw_fit = payload.get("fit")
    if not isinstance(raw_fit, Mapping):
        raise TypeError("diagnostic fit report fit must be a mapping")
    fit = parse_paired_pose_fit(raw_fit)
    report = QuestPairedPoseDiagnosticFit(
        evidence_semantic_content_hash=evidence_hash,
        dataset_semantic_sha256=dataset_hash,
        fit=fit,
    )
    if report.semantic_content_hash != semantic_hash:
        raise ValueError(
            "diagnostic fit semantic_content_hash does not match the report payload"
        )
    if report.as_dict() != dict(payload):
        raise ValueError("Quest paired-pose diagnostic fit payload is not canonical")
    return report


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


def _validated_diagnostic_fit(
    diagnostic_fit: QuestPairedPoseDiagnosticFit,
) -> QuestPairedPoseDiagnosticFit:
    if not isinstance(diagnostic_fit, QuestPairedPoseDiagnosticFit):
        raise TypeError("diagnostic_fit must be a QuestPairedPoseDiagnosticFit")
    checked = _parse_quest_paired_pose_diagnostic_fit(diagnostic_fit.as_dict())
    encoded = canonical_bytes(
        checked.as_dict(),
        domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_DOMAIN_V1,
    )
    round_trip = canonical_loads(
        encoded,
        expected_domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_DOMAIN_V1,
    )
    if not isinstance(round_trip, Mapping):
        raise TypeError("Quest paired-pose diagnostic fit root must be a mapping")
    reparsed = _parse_quest_paired_pose_diagnostic_fit(round_trip)
    if reparsed.as_dict() != checked.as_dict():
        raise ValueError("diagnostic fit changed during canonical encoding")
    return reparsed


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
    return (
        layout.evidence_dir(kind=QUEST_PAIRED_POSE_CAPTURE_ARTIFACT_KIND_V1)
        / f"{digest}{QUEST_PAIRED_POSE_CAPTURE_FILE_SUFFIX_V1}"
    )


def quest_paired_pose_diagnostic_fit_path(
    *,
    layout: RuntimeArtifactLayout,
    diagnostic_fit_semantic_content_hash: str,
) -> Path:
    """Allocate one derived report path without touching the filesystem."""

    layout = _require_layout(layout)
    digest = require_sha256_hex_digest(
        diagnostic_fit_semantic_content_hash,
        field="diagnostic_fit_semantic_content_hash",
    )
    return (
        layout.derived_dir(
            kind=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_ARTIFACT_KIND_V1,
            artifact_id=digest,
        )
        / QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_NAME_V1
    )


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


def _require_real_directory_chain(path: Path) -> None:
    current = path
    while True:
        _require_real_directory(current)
        if current.parent == current:
            return
        current = current.parent


def _ensure_directory_chain(path: Path) -> None:
    """Create and durably bind each missing directory component."""

    missing: list[Path] = []
    existing = path
    while not existing.exists():
        missing.append(existing)
        if existing.parent == existing:
            break
        existing = existing.parent
    _require_real_directory_chain(existing)
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            pass
        _require_real_directory(directory)
        _fsync_directory(directory.parent)


def _read_regular_file_no_follow(path: Path) -> bytes:
    _require_real_directory_chain(path.parent)
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
                result = b"".join(chunks)
                _require_real_directory_chain(path.parent)
                return result
            chunks.append(chunk)
    finally:
        os.close(fd)


def _existing_matches(
    path: Path,
    encoded: bytes,
    *,
    domain: str,
    artifact_label: str,
) -> bool:
    existing = _read_regular_file_no_follow(path)
    if existing != encoded:
        raise ValueError(f"immutable Quest {artifact_label} differs at {path}")
    canonical_loads(existing, expected_domain=domain)
    return True


def _publish_immutable_canonical(
    path: Path,
    payload: Mapping[str, Any],
    *,
    domain: str,
    artifact_label: str = "evidence",
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
        return _existing_matches(
            path,
            encoded,
            domain=domain,
            artifact_label=artifact_label,
        )

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
            result = _existing_matches(
                path,
                encoded,
                domain=domain,
                artifact_label=artifact_label,
            )
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
        raise TypeError("Quest paired-pose evidence root must be a mapping")
    if (
        canonical_bytes(
            dict(payload),
            domain=QUEST_PAIRED_POSE_CAPTURE_FILE_DOMAIN_V1,
        )
        != encoded
    ):
        raise ValueError("Quest paired-pose evidence bytes are not canonical")
    evidence = parse_paired_pose_capture_evidence(payload)
    if evidence.semantic_content_hash != expected_evidence_hash:
        raise ValueError("Quest evidence path does not bind its semantic hash")
    return evidence, encoded


def _load_diagnostic_fit_path(
    path: Path,
    *,
    expected_diagnostic_fit_hash: str,
) -> tuple[QuestPairedPoseDiagnosticFit, bytes]:
    encoded = _read_regular_file_no_follow(path)
    payload = canonical_loads(
        encoded,
        expected_domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_DOMAIN_V1,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("Quest paired-pose diagnostic fit root must be a mapping")
    if (
        canonical_bytes(
            dict(payload),
            domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_DOMAIN_V1,
        )
        != encoded
    ):
        raise ValueError("Quest paired-pose diagnostic fit bytes are not canonical")
    report = _parse_quest_paired_pose_diagnostic_fit(payload)
    if report.semantic_content_hash != expected_diagnostic_fit_hash:
        raise ValueError("Quest diagnostic fit path does not bind its semantic hash")
    return report, encoded


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


def publish_quest_paired_pose_diagnostic_fit(
    *,
    layout: RuntimeArtifactLayout,
    diagnostic_fit: QuestPairedPoseDiagnosticFit,
) -> QuestPairedPoseDiagnosticFitCommit:
    """Explicitly publish one strictly parsed candidate-only fit report."""

    layout = _require_layout(layout)
    checked = _validated_diagnostic_fit(diagnostic_fit)
    path = quest_paired_pose_diagnostic_fit_path(
        layout=layout,
        diagnostic_fit_semantic_content_hash=checked.semantic_content_hash,
    )
    try:
        idempotent = _publish_immutable_canonical(
            path,
            checked.as_dict(),
            domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_DOMAIN_V1,
            artifact_label="diagnostic fit report",
        )
        loaded, encoded = _load_diagnostic_fit_path(
            path,
            expected_diagnostic_fit_hash=checked.semantic_content_hash,
        )
        if loaded.as_dict() != checked.as_dict():
            raise ValueError(
                "published Quest diagnostic fit differs after strict reload"
            )
    except (OSError, TypeError, ValueError) as exc:
        raise QuestPairedPoseEvidenceStoreError(
            f"Cannot publish Quest paired-pose diagnostic fit {path}: {exc}"
        ) from exc
    return QuestPairedPoseDiagnosticFitCommit(
        path=path,
        diagnostic_fit_semantic_content_hash=checked.semantic_content_hash,
        evidence_semantic_content_hash=checked.evidence_semantic_content_hash,
        dataset_semantic_sha256=checked.dataset_semantic_sha256,
        fit_semantic_sha256=checked.fit.as_dict()["fit_semantic_sha256"],
        file_sha256=hashlib.sha256(encoded).hexdigest(),
        byte_length=len(encoded),
        idempotent=idempotent,
    )


def load_quest_paired_pose_diagnostic_fit(
    *,
    layout: RuntimeArtifactLayout,
    diagnostic_fit_semantic_content_hash: str,
) -> QuestPairedPoseDiagnosticFit:
    """Strictly load one standalone, content-addressed diagnostic report."""

    path = quest_paired_pose_diagnostic_fit_path(
        layout=layout,
        diagnostic_fit_semantic_content_hash=diagnostic_fit_semantic_content_hash,
    )
    try:
        report, _ = _load_diagnostic_fit_path(
            path,
            expected_diagnostic_fit_hash=diagnostic_fit_semantic_content_hash,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise QuestPairedPoseEvidenceStoreError(
            f"Cannot load Quest paired-pose diagnostic fit {path}: {exc}"
        ) from exc
    return report


__all__ = [
    "QuestPairedPoseDiagnosticFit",
    "QuestPairedPoseDiagnosticFitCommit",
    "QuestPairedPoseEvidenceCommit",
    "QuestPairedPoseEvidenceStoreError",
    "fit_quest_paired_pose_evidence",
    "load_quest_paired_pose_diagnostic_fit",
    "load_sealed_quest_paired_pose_evidence",
    "publish_quest_paired_pose_diagnostic_fit",
    "publish_sealed_quest_paired_pose_evidence",
    "quest_paired_pose_diagnostic_fit_path",
    "quest_paired_pose_evidence_path",
]
