"""Optional capture-time receipts for captured progress failures.

The receipt is deliberately an independent immutable sidecar.  It records
that the default failure-trigger sample was captured at the same bound
Episode time as the existing Failure fact, but it never becomes part of the
append-only progress fact log and cannot change reducer behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash, canonical_loads
from dataset_contracts.progress_failure_triggers import (
    FAILURE_TRIGGER_CAPTURE_FILE_DOMAIN_V1,
    FAILURE_TRIGGER_CAPTURE_PATH_DOMAIN_V1,
    FailureTriggerCaptureRecordV1,
)

from ..immutable_canonical_store import publish_immutable_canonical


FAILURE_TRIGGER_CAPTURE_RELATIVE_ROOT = Path("failure_trigger_capture")


class FailureTriggerCaptureStoreError(RuntimeError):
    """Raised when an optional capture receipt is malformed or conflicts."""


@dataclass(frozen=True)
class FailureTriggerCaptureCommitReport:
    path: Path
    semantic_content_hash: str
    idempotent: bool


def failure_trigger_capture_root(episode_dir: str | Path) -> Path:
    return (
        Path(episode_dir).expanduser().resolve()
        / FAILURE_TRIGGER_CAPTURE_RELATIVE_ROOT
    )


def _source_failure_path_token(source_failure_fact_id: str) -> str:
    return canonical_content_hash(
        {"source_failure_fact_id": str(source_failure_fact_id)},
        domain=FAILURE_TRIGGER_CAPTURE_PATH_DOMAIN_V1,
    )


def failure_trigger_capture_path(
    episode_dir: str | Path,
    *,
    source_failure_fact_id: str,
) -> Path:
    return failure_trigger_capture_root(episode_dir) / (
        f"{_source_failure_path_token(source_failure_fact_id)}.cjson"
    )


def record_failure_trigger_capture(
    *,
    episode_dir: str | Path,
    record: FailureTriggerCaptureRecordV1,
) -> FailureTriggerCaptureCommitReport:
    """Publish one receipt, preserving idempotence by source failure fact."""

    if not isinstance(record, FailureTriggerCaptureRecordV1):
        raise TypeError("record must be FailureTriggerCaptureRecordV1.")
    path = failure_trigger_capture_path(
        episode_dir,
        source_failure_fact_id=record.source_failure_fact_id,
    )
    try:
        idempotent = publish_immutable_canonical(
            path,
            record.to_payload(),
            domain=FAILURE_TRIGGER_CAPTURE_FILE_DOMAIN_V1,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise FailureTriggerCaptureStoreError(
            f"Cannot publish failure-trigger capture receipt {path}: {exc}"
        ) from exc
    return FailureTriggerCaptureCommitReport(
        path=path,
        semantic_content_hash=record.semantic_content_hash,
        idempotent=idempotent,
    )


def load_failure_trigger_capture_records(
    episode_dir: str | Path,
) -> tuple[FailureTriggerCaptureRecordV1, ...]:
    """Load all capture receipts; absent optional storage means no records."""

    root = failure_trigger_capture_root(episode_dir)
    if not root.exists():
        return ()
    if not root.is_dir():
        raise FailureTriggerCaptureStoreError(
            f"Failure-trigger capture root is not a directory: {root}"
        )
    records: list[FailureTriggerCaptureRecordV1] = []
    try:
        paths = tuple(sorted(root.iterdir(), key=lambda value: value.name))
        for path in paths:
            if path.name.startswith("."):
                continue
            if path.suffix != ".cjson" or len(path.stem) != 64:
                raise ValueError(
                    f"Unexpected failure-trigger capture artifact {path.name!r}."
                )
            payload = canonical_loads(
                path.read_bytes(),
                expected_domain=FAILURE_TRIGGER_CAPTURE_FILE_DOMAIN_V1,
            )
            if not isinstance(payload, Mapping):
                raise ValueError("Capture receipt root must be an object.")
            record = FailureTriggerCaptureRecordV1.from_payload(payload)
            expected_path = failure_trigger_capture_path(
                episode_dir,
                source_failure_fact_id=record.source_failure_fact_id,
            )
            if expected_path != path:
                raise ValueError(
                    "Capture receipt path does not bind its source failure fact."
                )
            records.append(record)
    except (OSError, TypeError, ValueError) as exc:
        raise FailureTriggerCaptureStoreError(
            f"Invalid failure-trigger capture receipts under {root}: {exc}"
        ) from exc
    return tuple(records)


__all__ = [
    "FAILURE_TRIGGER_CAPTURE_RELATIVE_ROOT",
    "FailureTriggerCaptureCommitReport",
    "FailureTriggerCaptureStoreError",
    "failure_trigger_capture_path",
    "failure_trigger_capture_root",
    "load_failure_trigger_capture_records",
    "record_failure_trigger_capture",
]
