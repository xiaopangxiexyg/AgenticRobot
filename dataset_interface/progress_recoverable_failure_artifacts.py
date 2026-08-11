"""Read-only loading of optional recoverable-failure sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.progress_recoverable_failures import (
    RECOVERABLE_FAILURE_FILE_DOMAIN_V1,
    RecoverableFailureAnnotationSetV1,
)

from .progress_stage_annotation_artifacts import (
    StageAnnotationArtifactReadError,
    load_stage_annotation_set,
)


RECOVERABLE_FAILURE_RELATIVE_PATH = "recoverable_failures.cjson"


class RecoverableFailureAnnotationReadError(ValueError):
    """Raised when a present recoverable-failure sidecar is inconsistent."""


def _check_expected(
    *,
    label: str,
    actual: str,
    expected: str | None,
) -> None:
    if expected is not None and actual != str(expected).strip():
        raise RecoverableFailureAnnotationReadError(
            f"Recoverable-failure {label} mismatch."
        )


def load_recoverable_failure_annotation_set(
    annotation_dir: str | Path,
    *,
    expected_episode_capture_identity_hash: str | None = None,
    expected_source_progress_manifest_hash: str | None = None,
    expected_task_stage_profile_hash: str | None = None,
    expected_stage_annotation_set_id: str | None = None,
    expected_stage_annotation_bundle_hash: str | None = None,
    expected_sample_axis_hash: str | None = None,
    expected_clock_domain_id: str | None = None,
) -> RecoverableFailureAnnotationSetV1:
    """Load a present sidecar and verify its optional source bindings."""

    annotation_root = Path(annotation_dir).expanduser().resolve()
    path = (
        annotation_root
        / RECOVERABLE_FAILURE_RELATIVE_PATH
    )
    try:
        stage_set = load_stage_annotation_set(annotation_root)
        payload = canonical_loads(
            path.read_bytes(),
            expected_domain=RECOVERABLE_FAILURE_FILE_DOMAIN_V1,
        )
        if not isinstance(payload, Mapping):
            raise TypeError(
                "Recoverable-failure sidecar root must be an object."
            )
        value = RecoverableFailureAnnotationSetV1.from_payload(payload)
    except (
        OSError,
        TypeError,
        ValueError,
        StageAnnotationArtifactReadError,
    ) as exc:
        raise RecoverableFailureAnnotationReadError(
            f"Cannot read recoverable-failure sidecar {path}: {exc}"
        ) from exc

    sealed_expectations = (
        (
            "Episode identity",
            value.episode_capture_identity_hash,
            stage_set.binding.episode_capture_identity_hash,
        ),
        (
            "Progress manifest",
            value.source_progress_manifest_hash,
            stage_set.binding.source_progress_manifest_hash,
        ),
        (
            "stage profile",
            value.task_stage_profile_hash,
            stage_set.binding.task_stage_profile_hash,
        ),
        (
            "stage annotation set",
            value.stage_annotation_set_id,
            stage_set.binding.annotation_set_id,
        ),
        (
            "stage annotation bundle",
            value.stage_annotation_bundle_hash,
            stage_set.bundle.semantic_content_hash,
        ),
        (
            "clock domain",
            value.bound_clock_domain_id,
            stage_set.binding.bound_clock_domain_id,
        ),
    )
    for label, actual, expected in sealed_expectations:
        _check_expected(label=label, actual=actual, expected=expected)

    for label, actual, expected in (
        (
            "Episode identity",
            value.episode_capture_identity_hash,
            expected_episode_capture_identity_hash,
        ),
        (
            "Progress manifest",
            value.source_progress_manifest_hash,
            expected_source_progress_manifest_hash,
        ),
        (
            "stage profile",
            value.task_stage_profile_hash,
            expected_task_stage_profile_hash,
        ),
        (
            "stage annotation set",
            value.stage_annotation_set_id,
            expected_stage_annotation_set_id,
        ),
        (
            "stage annotation bundle",
            value.stage_annotation_bundle_hash,
            expected_stage_annotation_bundle_hash,
        ),
        ("sample axis", value.sample_axis_hash, expected_sample_axis_hash),
        (
            "clock domain",
            value.bound_clock_domain_id,
            expected_clock_domain_id,
        ),
    ):
        _check_expected(label=label, actual=actual, expected=expected)
    return value


def load_optional_recoverable_failure_annotation_set(
    annotation_dir: str | Path,
    **kwargs: str | None,
) -> RecoverableFailureAnnotationSetV1 | None:
    """Return ``None`` for legacy stage sets without this additive sidecar."""

    annotation_root = Path(annotation_dir).expanduser().resolve()
    path = annotation_root / RECOVERABLE_FAILURE_RELATIVE_PATH
    if not path.is_file():
        try:
            load_stage_annotation_set(annotation_root)
        except StageAnnotationArtifactReadError as exc:
            raise RecoverableFailureAnnotationReadError(
                "Cannot read the sealed stage annotation set for an optional "
                f"recoverable-failure sidecar {path}: {exc}"
            ) from exc
        return None
    return load_recoverable_failure_annotation_set(annotation_dir, **kwargs)


__all__ = [
    "RECOVERABLE_FAILURE_RELATIVE_PATH",
    "RecoverableFailureAnnotationReadError",
    "load_optional_recoverable_failure_annotation_set",
    "load_recoverable_failure_annotation_set",
]
