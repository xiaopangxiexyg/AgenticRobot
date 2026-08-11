"""Optional immutable failure-trigger sidecar for stage annotation sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.progress_failure_triggers import (
    FAILURE_TRIGGER_FILE_DOMAIN_V1,
    FailureTriggerAnnotationSetV1,
)

from ..immutable_canonical_store import publish_immutable_canonical


FAILURE_TRIGGER_RELATIVE_PATH = "failure_triggers.cjson"


class FailureTriggerStoreError(RuntimeError):
    """Raised when a failure-trigger sidecar is incomplete or conflicts."""


@dataclass(frozen=True)
class FailureTriggerCommitReport:
    path: Path
    semantic_content_hash: str
    idempotent: bool


def failure_trigger_path(annotation_dir: str | Path) -> Path:
    return Path(annotation_dir).expanduser() / FAILURE_TRIGGER_RELATIVE_PATH


def commit_failure_trigger_annotation_set(
    *,
    annotation_dir: str | Path,
    annotation_set: FailureTriggerAnnotationSetV1,
) -> FailureTriggerCommitReport:
    """Publish the trigger sidecar without changing the stage bundle files."""

    if not isinstance(annotation_set, FailureTriggerAnnotationSetV1):
        raise TypeError(
            "annotation_set must be FailureTriggerAnnotationSetV1."
        )
    path = failure_trigger_path(annotation_dir)
    try:
        idempotent = publish_immutable_canonical(
            path,
            annotation_set.to_payload(),
            domain=FAILURE_TRIGGER_FILE_DOMAIN_V1,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise FailureTriggerStoreError(
            f"Cannot publish failure-trigger sidecar {path}: {exc}"
        ) from exc
    return FailureTriggerCommitReport(
        path=path,
        semantic_content_hash=annotation_set.semantic_content_hash,
        idempotent=idempotent,
    )


def load_failure_trigger_annotation_set(
    annotation_dir: str | Path,
) -> FailureTriggerAnnotationSetV1:
    """Load and hash-check one optional trigger sidecar."""

    path = failure_trigger_path(annotation_dir)
    try:
        payload = canonical_loads(
            path.read_bytes(),
            expected_domain=FAILURE_TRIGGER_FILE_DOMAIN_V1,
        )
        if not isinstance(payload, dict):
            raise ValueError("failure-trigger sidecar root must be an object.")
        return FailureTriggerAnnotationSetV1.from_payload(payload)
    except (OSError, TypeError, ValueError) as exc:
        raise FailureTriggerStoreError(
            f"Incomplete or invalid failure-trigger sidecar {path}: {exc}"
        ) from exc


__all__ = [
    "FAILURE_TRIGGER_RELATIVE_PATH",
    "FailureTriggerCommitReport",
    "FailureTriggerStoreError",
    "commit_failure_trigger_annotation_set",
    "failure_trigger_path",
    "load_failure_trigger_annotation_set",
]
