"""Reviewed Quest/robot frame bindings and passive TCP projection.

The calibration solver deliberately stops at candidate-only evidence. This
module is the next, explicit boundary: callers must provide an externally
reviewed acceptance receipt and a binding to the exact Quest reference epoch
and robot pose-coordinate epoch. The module never fits, accepts, installs, or
mutates a calibration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import (
    matrix_to_column_major,
    rigid_inverse,
    validate_rigid_transform,
)


FRAME_BINDING_SCHEMA = "quest_dev.visualization.accepted_frame_binding.v1"
REFERENCE_SPACE_TYPES = {"local", "local-floor", "bounded-floor", "unbounded"}

_BINDING_KEYS = {
    "schema",
    "binding_id",
    "accepted",
    "acceptance_receipt_sha256",
    "acceptance_profile_id",
    "capture_evidence_semantic_sha256",
    "diagnostic_fit_semantic_sha256",
    "dataset_sha256",
    "fit_semantic_sha256",
    "quest",
    "robot",
    "matrix_quest_from_robot_reference",
    "matrix_quest_from_robot_reference_column_major",
    "semantic_sha256",
}
_QUEST_KEYS = {
    "session_id",
    "reference_space_type",
    "reference_space_generation",
    "reference_epoch_id",
    "controller_source_session_id",
    "reference_frame_id",
}
_ROBOT_KEYS = {
    "robot_id",
    "source_instance_id",
    "source_epoch_id",
    "source_binding_hash",
    "component_id",
    "stream_id",
    "reference_frame_id",
    "reference_frame_spec_hash",
    "moving_frame_id",
    "tool_profile_hash",
    "pose_schema_hash",
    "fk_contract_id",
    "fk_contract_hash",
    "coordinate_binding_hash",
    "source_kind",
}


def _identifier(value: Any, name: str, *, max_length: int = 240) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ValueError(f"{name} must be a non-empty string up to {max_length} chars")
    if not all(character.isalnum() or character in "._:/@+-" for character in value):
        raise ValueError(f"{name} contains unsupported characters")
    return value


def _raw_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must contain 64 lowercase SHA-256 hex chars")
    return value


def _sha256_reference(value: Any, name: str) -> str:
    result = _identifier(value, name, max_length=71)
    digest = result.removeprefix("sha256:")
    if (
        not result.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} must use sha256:<64 lowercase hex chars>")
    return result


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_exact_keys(
    value: Mapping[Any, Any],
    required: set[str],
    name: str,
) -> None:
    missing = required.difference(value)
    unexpected = set(value).difference(required)
    if missing or unexpected:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(str(item) for item in unexpected)}"
        )


def _matrix(value: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric 4x4 matrix") from exc
    matrix = validate_rigid_transform(matrix)
    matrix[matrix == 0.0] = 0.0
    matrix.flags.writeable = False
    return matrix


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class QuestFrameIdentity:
    session_id: str
    reference_space_type: str
    reference_space_generation: int
    reference_epoch_id: str
    controller_source_session_id: str
    reference_frame_id: str

    def __post_init__(self) -> None:
        session_id = _identifier(self.session_id, "quest.session_id")
        reference_space_type = _identifier(
            self.reference_space_type,
            "quest.reference_space_type",
        )
        if reference_space_type not in REFERENCE_SPACE_TYPES:
            raise ValueError("quest.reference_space_type is unsupported")
        generation = _non_negative_int(
            self.reference_space_generation,
            "quest.reference_space_generation",
        )
        reference_epoch_id = _identifier(
            self.reference_epoch_id,
            "quest.reference_epoch_id",
        )
        expected_epoch = f"{session_id}:{reference_space_type}:{generation}"
        if reference_epoch_id != expected_epoch:
            raise ValueError(
                "quest.reference_epoch_id must equal "
                "session_id:reference_space_type:reference_space_generation"
            )
        for field_name in (
            "controller_source_session_id",
            "reference_frame_id",
        ):
            _identifier(getattr(self, field_name), f"quest.{field_name}")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "reference_space_type", reference_space_type)
        object.__setattr__(self, "reference_space_generation", generation)
        object.__setattr__(self, "reference_epoch_id", reference_epoch_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "reference_space_type": self.reference_space_type,
            "reference_space_generation": self.reference_space_generation,
            "reference_epoch_id": self.reference_epoch_id,
            "controller_source_session_id": self.controller_source_session_id,
            "reference_frame_id": self.reference_frame_id,
        }


@dataclass(frozen=True)
class RobotFrameIdentity:
    robot_id: str
    source_instance_id: str
    source_epoch_id: str
    source_binding_hash: str
    component_id: str
    stream_id: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    moving_frame_id: str
    tool_profile_hash: str
    pose_schema_hash: str
    fk_contract_id: str
    fk_contract_hash: str
    coordinate_binding_hash: str
    source_kind: str

    def __post_init__(self) -> None:
        for field_name in (
            "robot_id",
            "source_instance_id",
            "source_epoch_id",
            "source_binding_hash",
            "component_id",
            "stream_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "moving_frame_id",
            "pose_schema_hash",
            "fk_contract_id",
            "fk_contract_hash",
            "source_kind",
        ):
            _identifier(getattr(self, field_name), f"robot.{field_name}")
        tool_profile_hash = _sha256_reference(
            self.tool_profile_hash,
            "robot.tool_profile_hash",
        )
        coordinate_binding_hash = _raw_sha256(
            self.coordinate_binding_hash,
            "robot.coordinate_binding_hash",
        )
        object.__setattr__(self, "tool_profile_hash", tool_profile_hash)
        object.__setattr__(self, "coordinate_binding_hash", coordinate_binding_hash)

    def as_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "source_instance_id": self.source_instance_id,
            "source_epoch_id": self.source_epoch_id,
            "source_binding_hash": self.source_binding_hash,
            "component_id": self.component_id,
            "stream_id": self.stream_id,
            "reference_frame_id": self.reference_frame_id,
            "reference_frame_spec_hash": self.reference_frame_spec_hash,
            "moving_frame_id": self.moving_frame_id,
            "tool_profile_hash": self.tool_profile_hash,
            "pose_schema_hash": self.pose_schema_hash,
            "fk_contract_id": self.fk_contract_id,
            "fk_contract_hash": self.fk_contract_hash,
            "coordinate_binding_hash": self.coordinate_binding_hash,
            "source_kind": self.source_kind,
        }


@dataclass(frozen=True)
class AcceptedFrameBinding:
    """One reviewed, epoch-bound T_quest_from_robot_reference transform."""

    binding_id: str
    acceptance_receipt_sha256: str
    acceptance_profile_id: str
    capture_evidence_semantic_sha256: str
    diagnostic_fit_semantic_sha256: str
    dataset_sha256: str
    fit_semantic_sha256: str
    quest: QuestFrameIdentity
    robot: RobotFrameIdentity
    matrix_quest_from_robot_reference: np.ndarray
    semantic_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        binding_id = _identifier(self.binding_id, "binding_id")
        acceptance_profile_id = _identifier(
            self.acceptance_profile_id,
            "acceptance_profile_id",
        )
        for name in (
            "acceptance_receipt_sha256",
            "capture_evidence_semantic_sha256",
            "diagnostic_fit_semantic_sha256",
            "dataset_sha256",
            "fit_semantic_sha256",
        ):
            _raw_sha256(getattr(self, name), name)
        if not isinstance(self.quest, QuestFrameIdentity):
            raise TypeError("quest must be a QuestFrameIdentity")
        if not isinstance(self.robot, RobotFrameIdentity):
            raise TypeError("robot must be a RobotFrameIdentity")
        matrix = _matrix(
            self.matrix_quest_from_robot_reference,
            "matrix_quest_from_robot_reference",
        )
        object.__setattr__(self, "binding_id", binding_id)
        object.__setattr__(self, "acceptance_profile_id", acceptance_profile_id)
        object.__setattr__(
            self,
            "matrix_quest_from_robot_reference",
            matrix,
        )
        object.__setattr__(self, "semantic_sha256", _canonical_sha256(self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": FRAME_BINDING_SCHEMA,
            "binding_id": self.binding_id,
            "accepted": True,
            "acceptance_receipt_sha256": self.acceptance_receipt_sha256,
            "acceptance_profile_id": self.acceptance_profile_id,
            "capture_evidence_semantic_sha256": (
                self.capture_evidence_semantic_sha256
            ),
            "diagnostic_fit_semantic_sha256": self.diagnostic_fit_semantic_sha256,
            "dataset_sha256": self.dataset_sha256,
            "fit_semantic_sha256": self.fit_semantic_sha256,
            "quest": self.quest.as_dict(),
            "robot": self.robot.as_dict(),
            "matrix_quest_from_robot_reference": (
                self.matrix_quest_from_robot_reference.tolist()
            ),
            "matrix_quest_from_robot_reference_column_major": matrix_to_column_major(
                self.matrix_quest_from_robot_reference
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "semantic_sha256": self.semantic_sha256}

    def telemetry_identity(self) -> dict[str, Any]:
        """Return only metadata needed by a browser render gate."""

        return {
            "binding_id": self.binding_id,
            "acceptance_receipt_sha256": self.acceptance_receipt_sha256,
            "quest_session_id": self.quest.session_id,
            "reference_space_type": self.quest.reference_space_type,
            "reference_space_generation": self.quest.reference_space_generation,
            "quest_reference_epoch_id": self.quest.reference_epoch_id,
            "reference_frame_id": self.quest.reference_frame_id,
        }

    def robot_metadata_error(
        self,
        *,
        robot: str,
        metadata: Mapping[str, Any],
    ) -> str | None:
        """Return a stable mismatch reason for one passive pose projection."""

        if not isinstance(metadata, Mapping):
            return "missing_pose_binding"
        if robot != self.robot.robot_id:
            return "robot_mismatch"
        expected = self.robot.as_dict()
        for name, value in expected.items():
            if metadata.get(name) != value:
                return f"{name}_mismatch"
        return None

    def coordinate_binding_error(
        self,
        *,
        robot: str,
        coordinate_binding: Any,
    ) -> str | None:
        """Check a typed passive journal binding without taking ownership."""

        if coordinate_binding is None:
            return "missing_pose_binding"
        metadata = {
            name: getattr(coordinate_binding, name, None)
            for name in _ROBOT_KEYS
            if name != "robot_id"
        }
        metadata["robot_id"] = robot
        metadata["coordinate_binding_hash"] = getattr(
            coordinate_binding,
            "semantic_content_hash",
            None,
        )
        return self.robot_metadata_error(robot=robot, metadata=metadata)


def parse_accepted_frame_binding(
    payload: Mapping[str, Any],
) -> AcceptedFrameBinding:
    """Strictly parse an externally reviewed acceptance binding."""

    if not isinstance(payload, Mapping):
        raise TypeError("accepted frame binding must be a mapping")
    _require_exact_keys(payload, _BINDING_KEYS, "accepted frame binding")
    if payload.get("schema") != FRAME_BINDING_SCHEMA:
        raise ValueError(f"schema must be {FRAME_BINDING_SCHEMA!r}")
    if payload.get("accepted") is not True:
        raise ValueError("accepted frame binding must have accepted=true")
    raw_quest = payload.get("quest")
    if not isinstance(raw_quest, Mapping):
        raise TypeError("quest must be a mapping")
    _require_exact_keys(raw_quest, _QUEST_KEYS, "quest")
    quest = QuestFrameIdentity(
        session_id=raw_quest["session_id"],
        reference_space_type=raw_quest["reference_space_type"],
        reference_space_generation=raw_quest["reference_space_generation"],
        reference_epoch_id=raw_quest["reference_epoch_id"],
        controller_source_session_id=raw_quest["controller_source_session_id"],
        reference_frame_id=raw_quest["reference_frame_id"],
    )
    raw_robot = payload.get("robot")
    if not isinstance(raw_robot, Mapping):
        raise TypeError("robot must be a mapping")
    _require_exact_keys(raw_robot, _ROBOT_KEYS, "robot")
    robot = RobotFrameIdentity(
        robot_id=raw_robot["robot_id"],
        source_instance_id=raw_robot["source_instance_id"],
        source_epoch_id=raw_robot["source_epoch_id"],
        source_binding_hash=raw_robot["source_binding_hash"],
        component_id=raw_robot["component_id"],
        stream_id=raw_robot["stream_id"],
        reference_frame_id=raw_robot["reference_frame_id"],
        reference_frame_spec_hash=raw_robot["reference_frame_spec_hash"],
        moving_frame_id=raw_robot["moving_frame_id"],
        tool_profile_hash=raw_robot["tool_profile_hash"],
        pose_schema_hash=raw_robot["pose_schema_hash"],
        fk_contract_id=raw_robot["fk_contract_id"],
        fk_contract_hash=raw_robot["fk_contract_hash"],
        coordinate_binding_hash=raw_robot["coordinate_binding_hash"],
        source_kind=raw_robot["source_kind"],
    )
    matrix = _matrix(
        payload.get("matrix_quest_from_robot_reference"),
        "matrix_quest_from_robot_reference",
    )
    raw_column = payload.get("matrix_quest_from_robot_reference_column_major")
    if not isinstance(raw_column, list) or len(raw_column) != 16:
        raise ValueError(
            "matrix_quest_from_robot_reference_column_major must contain 16 values"
        )
    if raw_column != matrix_to_column_major(matrix):
        raise ValueError(
            "matrix_quest_from_robot_reference_column_major does not match "
            "matrix_quest_from_robot_reference"
        )
    binding = AcceptedFrameBinding(
        binding_id=payload["binding_id"],
        acceptance_receipt_sha256=payload["acceptance_receipt_sha256"],
        acceptance_profile_id=payload["acceptance_profile_id"],
        capture_evidence_semantic_sha256=payload[
            "capture_evidence_semantic_sha256"
        ],
        diagnostic_fit_semantic_sha256=payload["diagnostic_fit_semantic_sha256"],
        dataset_sha256=payload["dataset_sha256"],
        fit_semantic_sha256=payload["fit_semantic_sha256"],
        quest=quest,
        robot=robot,
        matrix_quest_from_robot_reference=matrix,
    )
    if binding.as_dict() != dict(payload):
        raise ValueError("accepted frame binding is not canonical")
    return binding


def load_accepted_frame_binding(path: str | Path) -> AcceptedFrameBinding:
    """Load one canonical reviewed binding without accepting or modifying it."""

    binding_path = Path(path)
    with binding_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_accepted_frame_binding(payload)


def project_quest_from_tcp(
    matrix_robot_from_tcp: np.ndarray,
    binding: AcceptedFrameBinding,
) -> np.ndarray:
    """Project T_robot_from_tcp into the bound Quest reference frame."""

    if not isinstance(binding, AcceptedFrameBinding):
        raise TypeError("binding must be an AcceptedFrameBinding")
    robot_from_tcp = validate_rigid_transform(matrix_robot_from_tcp)
    return validate_rigid_transform(
        binding.matrix_quest_from_robot_reference @ robot_from_tcp
    )


def project_tcp_from_quest(
    matrix_quest_from_tcp: np.ndarray,
    binding: AcceptedFrameBinding,
) -> np.ndarray:
    """Inverse helper used by offline checks; it never changes the binding."""

    if not isinstance(binding, AcceptedFrameBinding):
        raise TypeError("binding must be an AcceptedFrameBinding")
    quest_from_tcp = validate_rigid_transform(matrix_quest_from_tcp)
    return validate_rigid_transform(
        rigid_inverse(binding.matrix_quest_from_robot_reference) @ quest_from_tcp
    )


__all__ = [
    "FRAME_BINDING_SCHEMA",
    "AcceptedFrameBinding",
    "QuestFrameIdentity",
    "RobotFrameIdentity",
    "load_accepted_frame_binding",
    "parse_accepted_frame_binding",
    "project_quest_from_tcp",
    "project_tcp_from_quest",
]
