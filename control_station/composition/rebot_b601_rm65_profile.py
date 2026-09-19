"""Versioned frame and scale binding for B601-to-RM65 pose residuals.

The profile is an application-level assembly contract.  It does not read the
B601, advance clutch state, solve RM65 IK, apply safety limits, or dispatch a
command.  An enabled profile only constructs the existing
``HumanResidualPoseAdapter`` from explicit calibration evidence.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ComponentActionSchema
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.rebot_b601_rs import (
    B601_RS_BASE_FRAME,
    B601_RS_END_EFFECTOR_FRAME,
)
from teleop_runtime.action.human_residual import HumanResidualPoseAdapter

B601_RM65_FRAME_MAPPING_SCHEMA_V1 = (
    "control_station.b601_rm65_frame_mapping.v1"
)
B601_RM65_FRAME_MAPPING_PROFILE_OFF = "off"
B601_RM65_FRAME_MAPPING_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.b601_rm65_frame_mapping.v1"
)
B601_RM65_COMPONENT_ID = "right_arm"
B601_RM65_TARGET_REFERENCE_FRAME = "right_arm_base"
B601_RM65_TARGET_CONTROLLED_FRAME = "tool_tcp"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "profile_id",
        "enabled",
        "calibration_source_id",
        "calibration_source_sha256",
        "component_id",
        "source_reference_frame_id",
        "source_controlled_frame_id",
        "target_reference_frame_id",
        "target_controlled_frame_id",
        "matrix_target_reference_source_reference",
        "matrix_source_controlled_target_controlled",
        "translation_gain_target_reference_xyz",
        "rotation_gain",
        "semantic_content_hash",
    }
)


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _optional_text(value: object | None, *, name: str) -> str | None:
    if value is None:
        return None
    return _required(value, name=name)


def _optional_sha256(value: object | None, *, name: str) -> str | None:
    normalized = _optional_text(value, name=name)
    if normalized is None:
        return None
    normalized = normalized.lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256.")
    return normalized


def _optional_rigid_pose(
    value: object | None,
    *,
    name: str,
) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    try:
        validated = RigidFrameTransformSpec(
            transform_id=f"{name}:validation",
            source_frame_id=f"{name}:validation-source",
            target_frame_id=f"{name}:validation-target",
            matrix_target_source=value,
        ).matrix_target_source
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric rigid 4x4 pose.") from exc
    return tuple(tuple(float(item) for item in row) for row in validated)


def _translation_gain(value: object) -> tuple[float, float, float]:
    try:
        iterable: Any = value
        raw = tuple(iterable)
    except TypeError as exc:
        raise ValueError(
            "translation_gain_target_reference_xyz must be numeric."
        ) from exc
    if len(raw) != 3 or any(isinstance(item, bool) for item in raw):
        raise ValueError(
            "translation_gain_target_reference_xyz must contain three finite values."
        )
    try:
        normalized = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "translation_gain_target_reference_xyz must be numeric."
        ) from exc
    if any(not math.isfinite(item) for item in normalized):
        raise ValueError(
            "translation_gain_target_reference_xyz must contain three finite values."
        )
    if any(item <= 0.0 for item in normalized):
        raise ValueError(
            "translation_gain_target_reference_xyz values must be positive; "
            "axis direction belongs in the rigid frame transform."
        )
    return (normalized[0], normalized[1], normalized[2])


def _replace_pose_rotation(
    pose: object | None,
    rotation: Sequence[Sequence[float]],
    *,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    """Replace only the 3x3 rotation of an existing rigid pose."""

    if pose is None:
        raise ValueError(f"{name} cannot override a missing frame transform.")
    try:
        base = [list(row) for row in pose]
        matrix = [
            [float(item) for item in row]
            for row in rotation
        ]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric 3x3 matrix.") from exc
    if len(base) != 4 or any(len(row) != 4 for row in base):
        raise ValueError(f"{name} cannot override a non-4x4 frame transform.")
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError(f"{name} must be a numeric 3x3 matrix.")
    if any(not math.isfinite(item) for row in matrix for item in row):
        raise ValueError(f"{name} must contain finite values.")
    for row_index in range(3):
        for column_index in range(3):
            base[row_index][column_index] = matrix[row_index][column_index]
    return tuple(tuple(float(item) for item in row) for row in base)


@dataclass(frozen=True)
class B601RM65FrameMappingProfile:
    """One exact cross-robot frame calibration and residual scale profile."""

    profile_id: str = B601_RM65_FRAME_MAPPING_PROFILE_OFF
    enabled: bool = False
    calibration_source_id: str | None = None
    calibration_source_sha256: str | None = None
    component_id: str = B601_RM65_COMPONENT_ID
    source_reference_frame_id: str = B601_RS_BASE_FRAME
    source_controlled_frame_id: str = B601_RS_END_EFFECTOR_FRAME
    target_reference_frame_id: str = B601_RM65_TARGET_REFERENCE_FRAME
    target_controlled_frame_id: str = B601_RM65_TARGET_CONTROLLED_FRAME
    matrix_target_reference_source_reference: (
        tuple[tuple[float, ...], ...] | None
    ) = None
    matrix_source_controlled_target_controlled: (
        tuple[tuple[float, ...], ...] | None
    ) = None
    translation_gain_target_reference_xyz: tuple[float, float, float] = field(
        default_factory=lambda: (1.0, 1.0, 1.0)
    )
    rotation_gain: float = 1.0
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean.")
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "profile_id",
                "component_id",
                "source_reference_frame_id",
                "source_controlled_frame_id",
                "target_reference_frame_id",
                "target_controlled_frame_id",
            )
        }
        if text["component_id"] != B601_RM65_COMPONENT_ID:
            raise ValueError("B601/RM65 mapping currently requires right_arm.")
        expected_frames = {
            "source_reference_frame_id": B601_RS_BASE_FRAME,
            "source_controlled_frame_id": B601_RS_END_EFFECTOR_FRAME,
            "target_reference_frame_id": B601_RM65_TARGET_REFERENCE_FRAME,
            "target_controlled_frame_id": B601_RM65_TARGET_CONTROLLED_FRAME,
        }
        for name, expected in expected_frames.items():
            if text[name] != expected:
                raise ValueError(f"{name} must be {expected!r}.")
        if len(set(expected_frames.values())) != 4:
            raise ValueError("B601/RM65 mapping requires four distinct frame IDs.")

        calibration_id = _optional_text(
            self.calibration_source_id,
            name="calibration_source_id",
        )
        calibration_sha256 = _optional_sha256(
            self.calibration_source_sha256,
            name="calibration_source_sha256",
        )
        reference_transform = _optional_rigid_pose(
            self.matrix_target_reference_source_reference,
            name="matrix_target_reference_source_reference",
        )
        controlled_transform = _optional_rigid_pose(
            self.matrix_source_controlled_target_controlled,
            name="matrix_source_controlled_target_controlled",
        )
        translation_gain = _translation_gain(
            self.translation_gain_target_reference_xyz
        )
        if isinstance(self.rotation_gain, bool):
            raise TypeError("rotation_gain must be a real number, not bool.")
        rotation_gain = float(self.rotation_gain)
        if not math.isfinite(rotation_gain):
            raise ValueError("rotation_gain must be finite.")
        if not math.isclose(rotation_gain, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "The canonical HumanResidualPoseAdapter currently supports "
                "unscaled rotation only; rotation_gain must be 1.0."
            )
        if self.enabled:
            if text["profile_id"] == B601_RM65_FRAME_MAPPING_PROFILE_OFF:
                raise ValueError("An enabled mapping requires a non-off profile_id.")
            if calibration_id is None or calibration_sha256 is None:
                raise ValueError(
                    "An enabled mapping requires calibration source identity and SHA-256."
                )
            if reference_transform is None or controlled_transform is None:
                raise ValueError(
                    "An enabled mapping requires both rigid frame transforms."
                )

        payload = {
            "schema": B601_RM65_FRAME_MAPPING_SCHEMA_V1,
            **text,
            "enabled": self.enabled,
            "calibration_source_id": calibration_id,
            "calibration_source_sha256": calibration_sha256,
            "matrix_target_reference_source_reference": reference_transform,
            "matrix_source_controlled_target_controlled": controlled_transform,
            "translation_gain_target_reference_xyz": translation_gain,
            "rotation_gain": rotation_gain,
        }
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "calibration_source_id", calibration_id)
        object.__setattr__(self, "calibration_source_sha256", calibration_sha256)
        object.__setattr__(
            self,
            "matrix_target_reference_source_reference",
            reference_transform,
        )
        object.__setattr__(
            self,
            "matrix_source_controlled_target_controlled",
            controlled_transform,
        )
        object.__setattr__(
            self,
            "translation_gain_target_reference_xyz",
            translation_gain,
        )
        object.__setattr__(self, "rotation_gain", rotation_gain)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=B601_RM65_FRAME_MAPPING_HASH_DOMAIN_V1,
            ),
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema": B601_RM65_FRAME_MAPPING_SCHEMA_V1,
            "profile_id": self.profile_id,
            "enabled": self.enabled,
            "calibration_source_id": self.calibration_source_id,
            "calibration_source_sha256": self.calibration_source_sha256,
            "component_id": self.component_id,
            "source_reference_frame_id": self.source_reference_frame_id,
            "source_controlled_frame_id": self.source_controlled_frame_id,
            "target_reference_frame_id": self.target_reference_frame_id,
            "target_controlled_frame_id": self.target_controlled_frame_id,
            "matrix_target_reference_source_reference": (
                self.matrix_target_reference_source_reference
            ),
            "matrix_source_controlled_target_controlled": (
                self.matrix_source_controlled_target_controlled
            ),
            "translation_gain_target_reference_xyz": (
                self.translation_gain_target_reference_xyz
            ),
            "rotation_gain": self.rotation_gain,
            "semantic_content_hash": self.semantic_content_hash,
        }

    def to_payload(self) -> dict[str, Any]:
        payload = self.semantic_payload()
        for name in (
            "matrix_target_reference_source_reference",
            "matrix_source_controlled_target_controlled",
            "translation_gain_target_reference_xyz",
        ):
            value = payload[name]
            if value is None:
                payload[name] = None
            elif name == "translation_gain_target_reference_xyz":
                payload[name] = list(value)
            else:
                payload[name] = [list(row) for row in value]
        return payload

    def with_rotation_overrides(
        self,
        *,
        reference_rotation_matrix: Sequence[Sequence[float]] | None = None,
        controlled_rotation_matrix: Sequence[Sequence[float]] | None = None,
    ) -> B601RM65FrameMappingProfile:
        """Return a profile with in-memory frame-rotation overrides.

        The profile JSON remains the durable calibration source.  This small
        override port is intentionally application-neutral so a foreground
        simulation/commissioning command can sweep frame orientation without
        editing calibration files or maintaining a second mapping
        implementation.  Translations and all non-rotation calibration facts
        are preserved; construction still runs the normal rigid-transform
        validation and derives a fresh semantic hash.
        """

        updates: dict[str, object] = {}
        if reference_rotation_matrix is not None:
            updates["matrix_target_reference_source_reference"] = (
                _replace_pose_rotation(
                    self.matrix_target_reference_source_reference,
                    reference_rotation_matrix,
                    name="reference_rotation_matrix",
                )
            )
        if controlled_rotation_matrix is not None:
            updates["matrix_source_controlled_target_controlled"] = (
                _replace_pose_rotation(
                    self.matrix_source_controlled_target_controlled,
                    controlled_rotation_matrix,
                    name="controlled_rotation_matrix",
                )
            )
        if not updates:
            return self
        return replace(self, **updates)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> B601RM65FrameMappingProfile:
        values = dict(payload)
        unknown = sorted(set(values) - _PROFILE_PAYLOAD_KEYS)
        if unknown:
            raise ValueError(f"Unknown B601/RM65 mapping fields: {unknown}.")
        required = _PROFILE_PAYLOAD_KEYS - {"semantic_content_hash"}
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"Missing B601/RM65 mapping fields: {missing}.")
        if values["schema"] != B601_RM65_FRAME_MAPPING_SCHEMA_V1:
            raise ValueError("Unsupported B601/RM65 frame mapping schema.")
        profile = cls(
            profile_id=values["profile_id"],
            enabled=values["enabled"],
            calibration_source_id=values["calibration_source_id"],
            calibration_source_sha256=values["calibration_source_sha256"],
            component_id=values["component_id"],
            source_reference_frame_id=values["source_reference_frame_id"],
            source_controlled_frame_id=values["source_controlled_frame_id"],
            target_reference_frame_id=values["target_reference_frame_id"],
            target_controlled_frame_id=values["target_controlled_frame_id"],
            matrix_target_reference_source_reference=values[
                "matrix_target_reference_source_reference"
            ],
            matrix_source_controlled_target_controlled=values[
                "matrix_source_controlled_target_controlled"
            ],
            translation_gain_target_reference_xyz=values[
                "translation_gain_target_reference_xyz"
            ],
            rotation_gain=values["rotation_gain"],
        )
        # ``semantic_content_hash`` is derived evidence, not an edit gate.  A
        # calibration operator may tune matrix fields directly while keeping
        # an old receipt in the JSON; the loaded profile always derives its
        # current hash from the actual parameters above.  ``to_payload()``
        # emits that fresh value for new evidence, but stale/missing declared
        # values must not prevent foreground tuning.
        return profile

    def build_frame_graph(self) -> CoordinateFrameGraph:
        """Build the existing immutable graph only for an enabled calibration."""

        self._require_enabled()
        reference_transform = self.matrix_target_reference_source_reference
        controlled_transform = self.matrix_source_controlled_target_controlled
        if reference_transform is None or controlled_transform is None:
            raise RuntimeError(
                "Enabled B601/RM65 mapping lost its validated transforms."
            )
        calibration_parameters = {
            "calibration_source_id": self.calibration_source_id,
            "calibration_source_sha256": self.calibration_source_sha256,
            "mapping_profile_hash": self.semantic_content_hash,
        }
        return CoordinateFrameGraph(
            graph_id=f"{self.profile_id}:frame-graph",
            graph_version=B601_RM65_FRAME_MAPPING_SCHEMA_V1,
            frames=tuple(
                CoordinateFrameSpec(
                    frame_id=frame_id,
                    frame_version=B601_RM65_FRAME_MAPPING_SCHEMA_V1,
                )
                for frame_id in (
                    self.source_reference_frame_id,
                    self.source_controlled_frame_id,
                    self.target_reference_frame_id,
                    self.target_controlled_frame_id,
                )
            ),
            transforms=(
                RigidFrameTransformSpec(
                    transform_id=f"{self.profile_id}:reference",
                    transform_version=B601_RM65_FRAME_MAPPING_SCHEMA_V1,
                    source_frame_id=self.source_reference_frame_id,
                    target_frame_id=self.target_reference_frame_id,
                    matrix_target_source=reference_transform,
                    semantic_parameters=calibration_parameters,
                ),
                RigidFrameTransformSpec(
                    transform_id=f"{self.profile_id}:controlled",
                    transform_version=B601_RM65_FRAME_MAPPING_SCHEMA_V1,
                    source_frame_id=self.target_controlled_frame_id,
                    target_frame_id=self.source_controlled_frame_id,
                    matrix_target_source=controlled_transform,
                    semantic_parameters=calibration_parameters,
                ),
            ),
        )

    def build_residual_adapter(
        self,
        output_action_schema: ComponentActionSchema,
    ) -> HumanResidualPoseAdapter:
        """Bind this calibration to the canonical residual adapter."""

        self._require_enabled()
        if not isinstance(output_action_schema, ComponentActionSchema):
            raise TypeError("output_action_schema must be ComponentActionSchema.")
        if output_action_schema.component_ids != (self.component_id,):
            raise ValueError(
                "B601/RM65 mapping requires an exact single right_arm component."
            )
        payload_schema = output_action_schema.payload_schema_by_component[
            self.component_id
        ]
        if (
            payload_schema.representation_family != "cartesian_pose"
            or payload_schema.value_semantics != "relative_delta"
        ):
            raise ValueError(
                "B601/RM65 mapping requires a relative-delta Cartesian pose schema."
            )
        if (
            payload_schema.reference_frame != self.target_reference_frame_id
            or payload_schema.controlled_frame != self.target_controlled_frame_id
        ):
            raise ValueError(
                "RM65 residual schema frames do not match the mapping profile."
            )
        expected_semantics = {
            "translation_application": "reference_frame_additive",
            "rotation_application": "baseline_local_right_multiply",
        }
        if any(
            payload_schema.semantic_parameters.get(name) != value
            for name, value in expected_semantics.items()
        ):
            raise ValueError(
                "RM65 residual schema must use reference-frame translation and "
                "baseline-local right-multiplied rotation."
            )
        return HumanResidualPoseAdapter(
            output_action_schema=output_action_schema,
            frame_graph=self.build_frame_graph(),
            translation_gain_by_component={
                self.component_id: self.translation_gain_target_reference_xyz
            },
        )

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("B601/RM65 frame mapping profile is disabled.")


def load_b601_rm65_frame_mapping_profile(
    path: str | Path,
) -> B601RM65FrameMappingProfile:
    """Load one explicit UTF-8 JSON profile; there is no implicit default path."""

    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to load B601/RM65 mapping profile {resolved}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TypeError("B601/RM65 mapping profile root must be a JSON object.")
    return B601RM65FrameMappingProfile.from_payload(payload)


__all__ = [
    "B601_RM65_COMPONENT_ID",
    "B601_RM65_FRAME_MAPPING_HASH_DOMAIN_V1",
    "B601_RM65_FRAME_MAPPING_PROFILE_OFF",
    "B601_RM65_FRAME_MAPPING_SCHEMA_V1",
    "B601_RM65_TARGET_CONTROLLED_FRAME",
    "B601_RM65_TARGET_REFERENCE_FRAME",
    "B601RM65FrameMappingProfile",
    "load_b601_rm65_frame_mapping_profile",
]
