"""Pure identities for persisted Quest paired-pose candidate evidence.

This module deliberately owns only stable names and digest validation.  The
Quest feature owns the evidence schema and replay validation, while an
application leaf owns filesystem publication and diagnostic fitting.
"""

from __future__ import annotations

from typing import Any


QUEST_PAIRED_POSE_CAPTURE_ARTIFACT_KIND_V1 = "quest_paired_pose_capture"
QUEST_PAIRED_POSE_CAPTURE_FILE_SUFFIX_V1 = ".cjson"
QUEST_PAIRED_POSE_CAPTURE_FILE_DOMAIN_V1 = (
    "dexteleop.quest_paired_pose_capture_evidence.file.v1"
)
QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_SCHEMA_V1 = (
    "dexteleop.quest_paired_pose_diagnostic_fit.v1"
)
QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_HASH_DOMAIN_V1 = (
    "dexteleop.quest_paired_pose_diagnostic_fit.semantic.v1"
)


def require_sha256_hex_digest(value: Any, *, field: str) -> str:
    """Return one raw lowercase SHA-256 digest or fail closed.

    This validator is for the unprefixed capture, dataset, file, and report
    digests.  It must not be used for contracts such as ``tool_profile_hash``
    whose public representation is ``sha256:<digest>``.
    """

    if not isinstance(field, str) or not field.strip():
        raise ValueError("field must be a non-empty string")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must contain 64 lowercase SHA-256 hex chars")
    return value


__all__ = [
    "QUEST_PAIRED_POSE_CAPTURE_ARTIFACT_KIND_V1",
    "QUEST_PAIRED_POSE_CAPTURE_FILE_DOMAIN_V1",
    "QUEST_PAIRED_POSE_CAPTURE_FILE_SUFFIX_V1",
    "QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_HASH_DOMAIN_V1",
    "QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_SCHEMA_V1",
    "require_sha256_hex_digest",
]
