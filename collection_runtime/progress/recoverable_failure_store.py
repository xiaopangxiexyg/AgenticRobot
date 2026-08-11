"""Optional immutable recoverable-failure sidecar for stage annotation sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.progress_recoverable_failures import (
    RECOVERABLE_FAILURE_FILE_DOMAIN_V1,
    RecoverableFailureAnnotationSetV1,
)

from ..immutable_canonical_store import publish_immutable_canonical


RECOVERABLE_FAILURE_RELATIVE_PATH = "recoverable_failures.cjson"


class RecoverableFailureStoreError(RuntimeError):
    """Raised when a recoverable-failure sidecar is incomplete or conflicts."""


@dataclass(frozen=True)
class RecoverableFailureCommitReport:
    path: Path
    semantic_content_hash: str
    idempotent: bool


def recoverable_failure_path(annotation_dir: str | Path) -> Path:
    return (
        Path(annotation_dir).expanduser()
        / RECOVERABLE_FAILURE_RELATIVE_PATH
    )


def commit_recoverable_failure_annotation_set(
    *,
    annotation_dir: str | Path,
    annotation_set: RecoverableFailureAnnotationSetV1,
) -> RecoverableFailureCommitReport:
    if not isinstance(annotation_set, RecoverableFailureAnnotationSetV1):
        raise TypeError(
            "annotation_set must be RecoverableFailureAnnotationSetV1."
        )
    path = recoverable_failure_path(annotation_dir)
    try:
        idempotent = publish_immutable_canonical(
            path,
            annotation_set.to_payload(),
            domain=RECOVERABLE_FAILURE_FILE_DOMAIN_V1,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RecoverableFailureStoreError(
            f"Cannot publish recoverable-failure sidecar {path}: {exc}"
        ) from exc
    return RecoverableFailureCommitReport(
        path=path,
        semantic_content_hash=annotation_set.semantic_content_hash,
        idempotent=idempotent,
    )


def load_recoverable_failure_annotation_set(
    annotation_dir: str | Path,
) -> RecoverableFailureAnnotationSetV1:
    path = recoverable_failure_path(annotation_dir)
    try:
        payload = canonical_loads(
            path.read_bytes(),
            expected_domain=RECOVERABLE_FAILURE_FILE_DOMAIN_V1,
        )
        if not isinstance(payload, dict):
            raise ValueError(
                "recoverable-failure sidecar root must be an object."
            )
        return RecoverableFailureAnnotationSetV1.from_payload(payload)
    except (OSError, TypeError, ValueError) as exc:
        raise RecoverableFailureStoreError(
            f"Incomplete or invalid recoverable-failure sidecar {path}: {exc}"
        ) from exc


__all__ = [
    "RECOVERABLE_FAILURE_RELATIVE_PATH",
    "RecoverableFailureCommitReport",
    "RecoverableFailureStoreError",
    "commit_recoverable_failure_annotation_set",
    "load_recoverable_failure_annotation_set",
    "recoverable_failure_path",
]
