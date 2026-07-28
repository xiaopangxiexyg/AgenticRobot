"""Pure embodiment and v2.3 compatibility-projection contracts.

The embodiment profile is reusable by native capture.  The projection profile
in this v1 module is frozen as the explicit v2.3 compatibility target; native
v2.5-p stream semantics live in :mod:`dataset_contracts.v25p_native`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from .canonical_encoding import canonical_content_hash
from .v23 import DEXTELEOP_DATASET_SCHEMA_V23


EMBODIMENT_CAPTURE_PROFILE_SCHEMA_V1 = "dexteleop.embodiment_capture_profile.v1"
EMBODIMENT_CAPTURE_PROFILE_SCHEMA_V2 = "dexteleop.embodiment_capture_profile.v2"
CAPTURE_PROJECTION_PROFILE_SCHEMA_V1 = "dexteleop.capture_projection_profile.v1"
CAPTURE_PROJECTION_PROFILE_SCHEMA_V2 = "dexteleop.capture_projection_profile.v2"
EPISODE_EMBODIMENT_BINDING_SCHEMA_V1 = "dexteleop.episode_embodiment_binding.v1"

_EMBODIMENT_HASH_DOMAIN = "dexteleop.embodiment_capture_profile.semantic.v1"
_EMBODIMENT_HASH_DOMAIN_V2 = "dexteleop.embodiment_capture_profile.semantic.v2"
_PROJECTION_HASH_DOMAIN = "dexteleop.capture_projection_profile.semantic.v1"
_PROJECTION_HASH_DOMAIN_V2 = "dexteleop.capture_projection_profile.semantic.v2"
_BINDING_HASH_DOMAIN = "dexteleop.episode_embodiment_binding.semantic.v1"
_DATASET_ROLES = frozenset({"action", "state", "audit"})
_ACTION_STAGES = frozenset(
    {
        "accepted_reference",
        "admitted_command",
        "dispatched_command",
        "next_accepted_observation",
    }
)
_STATE_STAGES = frozenset({"causal_measured", "estimated", "source_observed"})


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _optional(value: object | None, *, name: str) -> str | None:
    return None if value is None else _required(value, name=name)


def _hash(value: object, *, name: str) -> str:
    text = _required(value, name=name).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return text


def _optional_hash(value: object | None, *, name: str) -> str | None:
    return None if value is None else _hash(value, name=name)


def _strings(values: Sequence[object], *, name: str) -> tuple[str, ...]:
    normalized = tuple(_required(value, name=name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} entries must be unique.")
    return normalized


@dataclass(frozen=True)
class EmbodimentComponentCaptureSpec:
    component_id: str
    component_kind: str
    controlled_entity_id: str
    command_payload_schema_hash: str | None = None
    measurement_payload_schema_hash: str | None = None
    controlled_frame: str | None = None
    reference_frame: str | None = None
    coordinate_names: tuple[str, ...] = ()
    units: str | None = None
    resource_ids: tuple[str, ...] = ()
    required_for_capture: bool = True

    def __post_init__(self) -> None:
        for name in ("component_id", "component_kind", "controlled_entity_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        command_hash = _optional_hash(
            self.command_payload_schema_hash,
            name="command_payload_schema_hash",
        )
        measurement_hash = _optional_hash(
            self.measurement_payload_schema_hash,
            name="measurement_payload_schema_hash",
        )
        # Native component streams own their exact payload schemas.  These
        # optional hashes remain for frozen compatibility profiles that bind a
        # single command/measurement representation, but a structural
        # component must not be forced to choose one measurement schema when
        # it exposes pose, qpos, velocity, or other concurrent streams.
        object.__setattr__(self, "command_payload_schema_hash", command_hash)
        object.__setattr__(self, "measurement_payload_schema_hash", measurement_hash)
        object.__setattr__(
            self,
            "controlled_frame",
            _optional(self.controlled_frame, name="controlled_frame"),
        )
        object.__setattr__(
            self,
            "reference_frame",
            _optional(self.reference_frame, name="reference_frame"),
        )
        object.__setattr__(
            self,
            "coordinate_names",
            _strings(self.coordinate_names, name="coordinate_names"),
        )
        object.__setattr__(self, "units", _optional(self.units, name="units"))
        object.__setattr__(
            self,
            "resource_ids",
            _strings(self.resource_ids, name="resource_ids"),
        )
        object.__setattr__(self, "required_for_capture", bool(self.required_for_capture))

    def to_payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_kind": self.component_kind,
            "controlled_entity_id": self.controlled_entity_id,
            "command_payload_schema_hash": self.command_payload_schema_hash,
            "measurement_payload_schema_hash": self.measurement_payload_schema_hash,
            "controlled_frame": self.controlled_frame,
            "reference_frame": self.reference_frame,
            "coordinate_names": self.coordinate_names,
            "units": self.units,
            "resource_ids": self.resource_ids,
            "required_for_capture": self.required_for_capture,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EmbodimentComponentCaptureSpec":
        return cls(
            component_id=payload.get("component_id", ""),
            component_kind=payload.get("component_kind", ""),
            controlled_entity_id=payload.get("controlled_entity_id", ""),
            command_payload_schema_hash=payload.get("command_payload_schema_hash"),
            measurement_payload_schema_hash=payload.get(
                "measurement_payload_schema_hash"
            ),
            controlled_frame=payload.get("controlled_frame"),
            reference_frame=payload.get("reference_frame"),
            coordinate_names=tuple(payload.get("coordinate_names", ())),
            units=payload.get("units"),
            resource_ids=tuple(payload.get("resource_ids", ())),
            required_for_capture=payload.get("required_for_capture", True),
        )


@dataclass(frozen=True)
class EmbodimentCaptureProfile:
    profile_id: str
    profile_version: str
    topology_id: str
    component_catalog_contract_id: str
    component_catalog_contract_hash: str
    target_capability_contract_id: str
    target_capability_contract_hash: str
    components: tuple[EmbodimentComponentCaptureSpec, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=EMBODIMENT_CAPTURE_PROFILE_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "profile_version",
            "topology_id",
            "component_catalog_contract_id",
            "target_capability_contract_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        for name in (
            "component_catalog_contract_hash",
            "target_capability_contract_hash",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        components = tuple(self.components)
        if not components:
            raise ValueError("components must not be empty.")
        identifiers = tuple(component.component_id for component in components)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Embodiment component_id values must be unique.")
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False), domain=_EMBODIMENT_HASH_DOMAIN
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "topology_id": self.topology_id,
            "component_catalog_contract_id": self.component_catalog_contract_id,
            "component_catalog_contract_hash": self.component_catalog_contract_hash,
            "target_capability_contract_id": self.target_capability_contract_id,
            "target_capability_contract_hash": self.target_capability_contract_hash,
            "components": [component.to_payload() for component in self.components],
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EmbodimentCaptureProfile":
        if payload.get("schema") != EMBODIMENT_CAPTURE_PROFILE_SCHEMA_V1:
            raise ValueError("Unsupported embodiment capture profile schema.")
        result = cls(
            profile_id=payload.get("profile_id", ""),
            profile_version=payload.get("profile_version", ""),
            topology_id=payload.get("topology_id", ""),
            component_catalog_contract_id=payload.get(
                "component_catalog_contract_id", ""
            ),
            component_catalog_contract_hash=payload.get(
                "component_catalog_contract_hash", ""
            ),
            target_capability_contract_id=payload.get(
                "target_capability_contract_id", ""
            ),
            target_capability_contract_hash=payload.get(
                "target_capability_contract_hash", ""
            ),
            components=tuple(
                EmbodimentComponentCaptureSpec.from_payload(component)
                for component in payload.get("components", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ValueError("Embodiment capture profile semantic hash mismatch.")
        return result


@dataclass(frozen=True)
class EmbodimentCaptureProfileV2:
    """Native capture topology without executable-target capability semantics."""

    profile_id: str
    profile_version: str
    topology_id: str
    component_catalog_contract_id: str
    component_catalog_contract_hash: str
    components: tuple[EmbodimentComponentCaptureSpec, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=EMBODIMENT_CAPTURE_PROFILE_SCHEMA_V2, init=False)

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "profile_version",
            "topology_id",
            "component_catalog_contract_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "component_catalog_contract_hash",
            _hash(
                self.component_catalog_contract_hash,
                name="component_catalog_contract_hash",
            ),
        )
        components = tuple(self.components)
        if not components:
            raise ValueError("components must not be empty.")
        identifiers = tuple(component.component_id for component in components)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Embodiment component_id values must be unique.")
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_EMBODIMENT_HASH_DOMAIN_V2,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "topology_id": self.topology_id,
            "component_catalog_contract_id": self.component_catalog_contract_id,
            "component_catalog_contract_hash": self.component_catalog_contract_hash,
            "components": [component.to_payload() for component in self.components],
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EmbodimentCaptureProfileV2":
        if payload.get("schema") != EMBODIMENT_CAPTURE_PROFILE_SCHEMA_V2:
            raise ValueError("Unsupported embodiment capture profile v2 schema.")
        result = cls(
            profile_id=payload.get("profile_id", ""),
            profile_version=payload.get("profile_version", ""),
            topology_id=payload.get("topology_id", ""),
            component_catalog_contract_id=payload.get(
                "component_catalog_contract_id", ""
            ),
            component_catalog_contract_hash=payload.get(
                "component_catalog_contract_hash", ""
            ),
            components=tuple(
                EmbodimentComponentCaptureSpec.from_payload(component)
                for component in payload.get("components", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ValueError("Embodiment capture profile v2 semantic hash mismatch.")
        return result


EmbodimentCaptureProfileLike = EmbodimentCaptureProfile | EmbodimentCaptureProfileV2


def embodiment_capture_profile_from_payload(
    payload: Mapping[str, Any],
) -> EmbodimentCaptureProfileLike:
    schema = payload.get("schema")
    if schema == EMBODIMENT_CAPTURE_PROFILE_SCHEMA_V1:
        return EmbodimentCaptureProfile.from_payload(payload)
    if schema == EMBODIMENT_CAPTURE_PROFILE_SCHEMA_V2:
        return EmbodimentCaptureProfileV2.from_payload(payload)
    raise ValueError("Unsupported embodiment capture profile schema.")


@dataclass(frozen=True)
class CaptureProjectionComponent:
    component_id: str
    dataset_role: str
    dataset_field: str
    value_schema_hash: str
    required: bool = True

    def __post_init__(self) -> None:
        for name in ("component_id", "dataset_role", "dataset_field"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        if self.dataset_role not in _DATASET_ROLES:
            raise ValueError(f"Unsupported dataset_role {self.dataset_role!r}.")
        object.__setattr__(
            self,
            "value_schema_hash",
            _hash(self.value_schema_hash, name="value_schema_hash"),
        )
        object.__setattr__(self, "required", bool(self.required))

    def to_payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "dataset_role": self.dataset_role,
            "dataset_field": self.dataset_field,
            "value_schema_hash": self.value_schema_hash,
            "required": self.required,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CaptureProjectionComponent":
        return cls(
            component_id=payload.get("component_id", ""),
            dataset_role=payload.get("dataset_role", ""),
            dataset_field=payload.get("dataset_field", ""),
            value_schema_hash=payload.get("value_schema_hash", ""),
            required=payload.get("required", True),
        )


@dataclass(frozen=True)
class CaptureProjectionProfile:
    projection_id: str
    projection_version: str
    temporal_alignment_contract_id: str
    temporal_alignment_contract_hash: str
    row_encoder_contract_id: str
    row_encoder_contract_hash: str
    policy_io_layout_hash: str
    pose_frame_contract_hash: str
    causal_snapshot_contract_hash: str
    components: tuple[CaptureProjectionComponent, ...]
    action_stage: str = "accepted_reference"
    state_stage: str = "causal_measured"
    base_dataset_schema: str = DEXTELEOP_DATASET_SCHEMA_V23
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=CAPTURE_PROJECTION_PROFILE_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in (
            "projection_id",
            "projection_version",
            "temporal_alignment_contract_id",
            "row_encoder_contract_id",
            "action_stage",
            "state_stage",
            "base_dataset_schema",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        if self.base_dataset_schema != DEXTELEOP_DATASET_SCHEMA_V23:
            raise ValueError("This v1 compatibility projection must target v2.3.")
        if self.action_stage not in _ACTION_STAGES:
            raise ValueError(f"Unsupported action_stage {self.action_stage!r}.")
        if self.state_stage not in _STATE_STAGES:
            raise ValueError(f"Unsupported state_stage {self.state_stage!r}.")
        for name in (
            "temporal_alignment_contract_hash",
            "row_encoder_contract_hash",
            "policy_io_layout_hash",
            "pose_frame_contract_hash",
            "causal_snapshot_contract_hash",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        components = tuple(self.components)
        if not components:
            raise ValueError("Projection components must not be empty.")
        fields = tuple(component.dataset_field for component in components)
        if len(set(fields)) != len(fields):
            raise ValueError("Projection dataset_field values must be unique.")
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False), domain=_PROJECTION_HASH_DOMAIN
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "projection_id": self.projection_id,
            "projection_version": self.projection_version,
            "base_dataset_schema": self.base_dataset_schema,
            "action_stage": self.action_stage,
            "state_stage": self.state_stage,
            "temporal_alignment_contract_id": self.temporal_alignment_contract_id,
            "temporal_alignment_contract_hash": self.temporal_alignment_contract_hash,
            "row_encoder_contract_id": self.row_encoder_contract_id,
            "row_encoder_contract_hash": self.row_encoder_contract_hash,
            "policy_io_layout_hash": self.policy_io_layout_hash,
            "pose_frame_contract_hash": self.pose_frame_contract_hash,
            "causal_snapshot_contract_hash": self.causal_snapshot_contract_hash,
            "components": [component.to_payload() for component in self.components],
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CaptureProjectionProfile":
        if payload.get("schema") != CAPTURE_PROJECTION_PROFILE_SCHEMA_V1:
            raise ValueError("Unsupported capture projection profile schema.")
        result = cls(
            projection_id=payload.get("projection_id", ""),
            projection_version=payload.get("projection_version", ""),
            base_dataset_schema=payload.get("base_dataset_schema", ""),
            action_stage=payload.get("action_stage", ""),
            state_stage=payload.get("state_stage", ""),
            temporal_alignment_contract_id=payload.get(
                "temporal_alignment_contract_id", ""
            ),
            temporal_alignment_contract_hash=payload.get(
                "temporal_alignment_contract_hash", ""
            ),
            row_encoder_contract_id=payload.get("row_encoder_contract_id", ""),
            row_encoder_contract_hash=payload.get("row_encoder_contract_hash", ""),
            policy_io_layout_hash=payload.get("policy_io_layout_hash", ""),
            pose_frame_contract_hash=payload.get("pose_frame_contract_hash", ""),
            causal_snapshot_contract_hash=payload.get(
                "causal_snapshot_contract_hash", ""
            ),
            components=tuple(
                CaptureProjectionComponent.from_payload(component)
                for component in payload.get("components", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ValueError("Capture projection profile semantic hash mismatch.")
        return result


@dataclass(frozen=True)
class CaptureProjectionComponentV2:
    """One compatibility field bound to one exact native stream.

    Component identity, role, timing stage, payload schema, shape, units, and
    frames remain owned by the native stream profile.  Repeating them here
    would create a second semantic source of truth.
    """

    stream_id: str
    dataset_field: str
    required: bool = True

    def __post_init__(self) -> None:
        for name in ("stream_id", "dataset_field"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        object.__setattr__(self, "required", bool(self.required))

    def to_payload(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "dataset_field": self.dataset_field,
            "required": self.required,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "CaptureProjectionComponentV2":
        return cls(
            stream_id=payload.get("stream_id", ""),
            dataset_field=payload.get("dataset_field", ""),
            required=payload.get("required", True),
        )


@dataclass(frozen=True)
class CaptureProjectionProfileV2:
    """v2.3 compatibility projection by exact native stream identity."""

    projection_id: str
    projection_version: str
    temporal_alignment_contract_id: str
    temporal_alignment_contract_hash: str
    row_encoder_contract_id: str
    row_encoder_contract_hash: str
    policy_io_layout_hash: str
    pose_frame_contract_hash: str
    causal_snapshot_contract_hash: str
    components: tuple[CaptureProjectionComponentV2, ...]
    base_dataset_schema: str = DEXTELEOP_DATASET_SCHEMA_V23
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=CAPTURE_PROJECTION_PROFILE_SCHEMA_V2, init=False)

    def __post_init__(self) -> None:
        for name in (
            "projection_id",
            "projection_version",
            "temporal_alignment_contract_id",
            "row_encoder_contract_id",
            "base_dataset_schema",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        if self.base_dataset_schema != DEXTELEOP_DATASET_SCHEMA_V23:
            raise ValueError("This v2 compatibility projection must target v2.3.")
        for name in (
            "temporal_alignment_contract_hash",
            "row_encoder_contract_hash",
            "policy_io_layout_hash",
            "pose_frame_contract_hash",
            "causal_snapshot_contract_hash",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        components = tuple(self.components)
        if not components or any(
            not isinstance(item, CaptureProjectionComponentV2) for item in components
        ):
            raise ValueError("Projection v2 components must not be empty.")
        fields = tuple(component.dataset_field for component in components)
        if len(fields) != len(set(fields)):
            raise ValueError("Projection dataset_field values must be unique.")
        stream_ids = tuple(component.stream_id for component in components)
        if len(stream_ids) != len(set(stream_ids)):
            raise ValueError("Projection stream_id values must be unique.")
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False), domain=_PROJECTION_HASH_DOMAIN_V2
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "projection_id": self.projection_id,
            "projection_version": self.projection_version,
            "base_dataset_schema": self.base_dataset_schema,
            "temporal_alignment_contract_id": self.temporal_alignment_contract_id,
            "temporal_alignment_contract_hash": self.temporal_alignment_contract_hash,
            "row_encoder_contract_id": self.row_encoder_contract_id,
            "row_encoder_contract_hash": self.row_encoder_contract_hash,
            "policy_io_layout_hash": self.policy_io_layout_hash,
            "pose_frame_contract_hash": self.pose_frame_contract_hash,
            "causal_snapshot_contract_hash": self.causal_snapshot_contract_hash,
            "components": [component.to_payload() for component in self.components],
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CaptureProjectionProfileV2":
        if payload.get("schema") != CAPTURE_PROJECTION_PROFILE_SCHEMA_V2:
            raise ValueError("Unsupported capture projection profile v2 schema.")
        result = cls(
            projection_id=payload.get("projection_id", ""),
            projection_version=payload.get("projection_version", ""),
            base_dataset_schema=payload.get("base_dataset_schema", ""),
            temporal_alignment_contract_id=payload.get(
                "temporal_alignment_contract_id", ""
            ),
            temporal_alignment_contract_hash=payload.get(
                "temporal_alignment_contract_hash", ""
            ),
            row_encoder_contract_id=payload.get("row_encoder_contract_id", ""),
            row_encoder_contract_hash=payload.get("row_encoder_contract_hash", ""),
            policy_io_layout_hash=payload.get("policy_io_layout_hash", ""),
            pose_frame_contract_hash=payload.get("pose_frame_contract_hash", ""),
            causal_snapshot_contract_hash=payload.get(
                "causal_snapshot_contract_hash", ""
            ),
            components=tuple(
                CaptureProjectionComponentV2.from_payload(component)
                for component in payload.get("components", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ValueError("Capture projection profile v2 semantic hash mismatch.")
        return result


CaptureProjectionProfileLike = CaptureProjectionProfile | CaptureProjectionProfileV2


def capture_projection_profile_from_payload(
    payload: Mapping[str, Any],
) -> CaptureProjectionProfileLike:
    schema = payload.get("schema")
    if schema == CAPTURE_PROJECTION_PROFILE_SCHEMA_V1:
        return CaptureProjectionProfile.from_payload(payload)
    if schema == CAPTURE_PROJECTION_PROFILE_SCHEMA_V2:
        return CaptureProjectionProfileV2.from_payload(payload)
    raise ValueError(f"Unsupported capture projection profile schema {schema!r}.")


@dataclass(frozen=True)
class EpisodeEmbodimentBinding:
    episode_uuid: str
    legacy_episode_index: int
    episode_identity_hash: str
    embodiment_profile_hash: str
    capture_projection_profile_hash: str
    bound_at: float
    bound_clock_domain_id: str
    teleop_input_profile_hash: str | None = None
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=EPISODE_EMBODIMENT_BINDING_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_uuid", _required(self.episode_uuid, name="episode_uuid"))
        index = int(self.legacy_episode_index)
        if isinstance(self.legacy_episode_index, bool) or index < 0:
            raise ValueError("legacy_episode_index must be a non-negative integer.")
        object.__setattr__(self, "legacy_episode_index", index)
        for name in (
            "episode_identity_hash",
            "embodiment_profile_hash",
            "capture_projection_profile_hash",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "teleop_input_profile_hash",
            _optional_hash(self.teleop_input_profile_hash, name="teleop_input_profile_hash"),
        )
        bound_at = float(self.bound_at)
        if not math.isfinite(bound_at):
            raise ValueError("bound_at must be finite.")
        object.__setattr__(self, "bound_at", bound_at)
        object.__setattr__(
            self,
            "bound_clock_domain_id",
            _required(self.bound_clock_domain_id, name="bound_clock_domain_id"),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False), domain=_BINDING_HASH_DOMAIN
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "episode_identity_hash": self.episode_identity_hash,
            "embodiment_profile_hash": self.embodiment_profile_hash,
            "capture_projection_profile_hash": self.capture_projection_profile_hash,
            "teleop_input_profile_hash": self.teleop_input_profile_hash,
            "bound_at": self.bound_at,
            "bound_clock_domain_id": self.bound_clock_domain_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EpisodeEmbodimentBinding":
        if payload.get("schema") != EPISODE_EMBODIMENT_BINDING_SCHEMA_V1:
            raise ValueError("Unsupported episode embodiment binding schema.")
        result = cls(
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            episode_identity_hash=payload.get("episode_identity_hash", ""),
            embodiment_profile_hash=payload.get("embodiment_profile_hash", ""),
            capture_projection_profile_hash=payload.get(
                "capture_projection_profile_hash", ""
            ),
            teleop_input_profile_hash=payload.get("teleop_input_profile_hash"),
            bound_at=payload.get("bound_at", float("nan")),
            bound_clock_domain_id=payload.get("bound_clock_domain_id", ""),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ValueError("Episode embodiment binding semantic hash mismatch.")
        return result


# Explicit aliases make the frozen v1 scope visible without breaking existing
# imports or changing any semantic hashes.
V23CompatibilityProjectionComponent = CaptureProjectionComponent
V23CompatibilityProjectionProfile = CaptureProjectionProfile


__all__ = [
    "CAPTURE_PROJECTION_PROFILE_SCHEMA_V1",
    "CAPTURE_PROJECTION_PROFILE_SCHEMA_V2",
    "EMBODIMENT_CAPTURE_PROFILE_SCHEMA_V1",
    "EMBODIMENT_CAPTURE_PROFILE_SCHEMA_V2",
    "EPISODE_EMBODIMENT_BINDING_SCHEMA_V1",
    "CaptureProjectionComponent",
    "CaptureProjectionProfile",
    "CaptureProjectionComponentV2",
    "CaptureProjectionProfileLike",
    "CaptureProjectionProfileV2",
    "EmbodimentCaptureProfile",
    "EmbodimentCaptureProfileLike",
    "EmbodimentCaptureProfileV2",
    "EmbodimentComponentCaptureSpec",
    "EpisodeEmbodimentBinding",
    "V23CompatibilityProjectionComponent",
    "V23CompatibilityProjectionProfile",
    "capture_projection_profile_from_payload",
    "embodiment_capture_profile_from_payload",
]
