"""Read-only loading of optional failure-trigger annotation sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.progress_failure_triggers import (
    FAILURE_TRIGGER_FILE_DOMAIN_V1,
    FailureTriggerAnnotationSetV1,
)


FAILURE_TRIGGER_RELATIVE_PATH = "failure_triggers.cjson"


class FailureTriggerAnnotationReadError(ValueError):
    """Raised when a failure-trigger sidecar is absent or inconsistent."""


def _check_expected(
    *,
    label: str,
    actual: str,
    expected: str | None,
) -> None:
    if expected is not None and actual != str(expected).strip():
        raise FailureTriggerAnnotationReadError(
            f"Failure-trigger {label} mismatch."
        )


def load_failure_trigger_annotation_set(
    annotation_dir: str | Path,
    *,
    expected_episode_capture_identity_hash: str | None = None,
    expected_source_progress_manifest_hash: str | None = None,
    expected_task_stage_profile_hash: str | None = None,
    expected_stage_annotation_set_id: str | None = None,
    expected_stage_annotation_bundle_hash: str | None = None,
    expected_sample_axis_hash: str | None = None,
    expected_clock_domain_id: str | None = None,
) -> FailureTriggerAnnotationSetV1:
    """Load one sealed trigger sidecar and optionally bind it to its sources."""

    path = Path(annotation_dir).expanduser().resolve() / FAILURE_TRIGGER_RELATIVE_PATH
    try:
        payload = canonical_loads(
            path.read_bytes(),
            expected_domain=FAILURE_TRIGGER_FILE_DOMAIN_V1,
        )
        if not isinstance(payload, Mapping):
            raise TypeError("Failure-trigger sidecar root must be an object.")
        value = FailureTriggerAnnotationSetV1.from_payload(payload)
    except (OSError, TypeError, ValueError) as exc:
        raise FailureTriggerAnnotationReadError(
            f"Cannot read failure-trigger sidecar {path}: {exc}"
        ) from exc

    try:
        _check_expected(
            label="Episode identity",
            actual=value.episode_capture_identity_hash,
            expected=expected_episode_capture_identity_hash,
        )
        _check_expected(
            label="Progress manifest",
            actual=value.source_progress_manifest_hash,
            expected=expected_source_progress_manifest_hash,
        )
        _check_expected(
            label="stage profile",
            actual=value.task_stage_profile_hash,
            expected=expected_task_stage_profile_hash,
        )
        _check_expected(
            label="stage annotation set",
            actual=value.stage_annotation_set_id,
            expected=expected_stage_annotation_set_id,
        )
        _check_expected(
            label="stage annotation bundle",
            actual=value.stage_annotation_bundle_hash,
            expected=expected_stage_annotation_bundle_hash,
        )
        _check_expected(
            label="sample axis",
            actual=value.sample_axis_hash,
            expected=expected_sample_axis_hash,
        )
        _check_expected(
            label="clock domain",
            actual=value.bound_clock_domain_id,
            expected=expected_clock_domain_id,
        )
    except FailureTriggerAnnotationReadError:
        raise
    return value


def load_optional_failure_trigger_annotation_set(
    annotation_dir: str | Path,
    **kwargs: str | None,
) -> FailureTriggerAnnotationSetV1 | None:
    """Return ``None`` for legacy stage sets without the additive sidecar."""

    path = Path(annotation_dir).expanduser().resolve() / FAILURE_TRIGGER_RELATIVE_PATH
    if not path.is_file():
        return None
    return load_failure_trigger_annotation_set(annotation_dir, **kwargs)


__all__ = [
    "FAILURE_TRIGGER_RELATIVE_PATH",
    "FailureTriggerAnnotationReadError",
    "load_failure_trigger_annotation_set",
    "load_optional_failure_trigger_annotation_set",
]
