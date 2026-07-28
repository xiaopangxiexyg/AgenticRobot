"""Native, embodiment-neutral component-stream contracts for provisional v2.5-p.

These v2 contracts do not depend on a LeRobot/v2.3 row, policy tensor layout,
or legacy episode record.  Compatibility formats are optional projections of
the native causal samples and are bound later by explicit receipts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .action_schema import (
    ACTION_COMPONENT_CATALOG_HASH_DOMAIN_V2,
    ActionPayloadSchema,
    ComponentActionSpaceSchema,
)
from .canonical_encoding import canonical_content_hash


V25P_NATIVE_DATASET_MANIFEST_SCHEMA_V2 = "dexteleop.v25p.native_dataset_manifest.v2"
V25P_NATIVE_COMPONENT_STREAM_PROFILE_SCHEMA_V1 = (
    "dexteleop.v25p.component_stream_profile.v1"
)
V25P_NATIVE_COMPONENT_STREAM_PROFILE_SCHEMA_V2 = (
    "dexteleop.v25p.component_stream_profile.v2"
)
V25P_NATIVE_EMBODIMENT_BINDING_SCHEMA_V1 = (
    "dexteleop.v25p.native_embodiment_binding.v1"
)
V25P_NATIVE_COMPONENT_RECORD_SCHEMA_V1 = "dexteleop.v25p.component_record.v1"
V25P_NATIVE_CAUSAL_SAMPLE_SCHEMA_V1 = "dexteleop.v25p.causal_sample.v1"
V25P_NATIVE_CAPTURE_MANIFEST_SCHEMA_V1 = "dexteleop.v25p.native_capture_manifest.v1"
V25P_NATIVE_EPISODE_MANIFEST_SCHEMA_V2 = "dexteleop.v25p.native_episode_manifest.v2"

V25P_NATIVE_DATASET_STATUS_V1 = "provisional_native_component_streams"
V25P_NATIVE_EPISODE_STATUS_V1 = "sealed_provisional_native"

V25P_NATIVE_DATASET_HASH_DOMAIN_V2 = "dexteleop.v25p.native_dataset_manifest.semantic.v2"
V25P_NATIVE_STREAM_PROFILE_HASH_DOMAIN_V1 = (
    "dexteleop.v25p.component_stream_profile.semantic.v1"
)
V25P_NATIVE_STREAM_PROFILE_HASH_DOMAIN_V2 = (
    "dexteleop.v25p.component_stream_profile.semantic.v2"
)
V25P_NATIVE_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.v25p.native_embodiment_binding.semantic.v1"
)
V25P_NATIVE_COMPONENT_RECORD_HASH_DOMAIN_V1 = (
    "dexteleop.v25p.component_record.semantic.v1"
)
V25P_NATIVE_CAUSAL_SAMPLE_HASH_DOMAIN_V1 = (
    "dexteleop.v25p.causal_sample.semantic.v1"
)
V25P_NATIVE_CAPTURE_MANIFEST_HASH_DOMAIN_V1 = (
    "dexteleop.v25p.native_capture_manifest.semantic.v1"
)
V25P_NATIVE_RECORD_BUNDLE_HASH_DOMAIN_V1 = (
    "dexteleop.v25p.component_record_bundle.semantic.v1"
)
V25P_NATIVE_SAMPLE_BUNDLE_HASH_DOMAIN_V1 = (
    "dexteleop.v25p.causal_sample_bundle.semantic.v1"
)
V25P_NATIVE_EPISODE_MANIFEST_HASH_DOMAIN_V2 = (
    "dexteleop.v25p.native_episode_manifest.semantic.v2"
)

V25P_NATIVE_DATASET_MANIFEST_FILE_DOMAIN_V2 = (
    "dexteleop.v25p.native_dataset_manifest.file.v2"
)
V25P_NATIVE_PROFILE_FILE_DOMAIN_V1 = "dexteleop.v25p.native_profile.file.v1"
V25P_NATIVE_BINDING_FILE_DOMAIN_V1 = "dexteleop.v25p.native_embodiment_binding.file.v1"
V25P_NATIVE_COMPONENT_RECORD_FILE_DOMAIN_V1 = (
    "dexteleop.v25p.component_record.file.v1"
)
V25P_NATIVE_CAUSAL_SAMPLE_FILE_DOMAIN_V1 = "dexteleop.v25p.causal_sample.file.v1"
V25P_NATIVE_CAPTURE_MANIFEST_FILE_DOMAIN_V1 = (
    "dexteleop.v25p.native_capture_manifest.file.v1"
)
V25P_NATIVE_EPISODE_MANIFEST_FILE_DOMAIN_V2 = (
    "dexteleop.v25p.native_episode_manifest.file.v2"
)

V25P_NATIVE_RELATIVE_ROOT_V1 = "meta/v25p/native"
V25P_NATIVE_PROFILE_RELATIVE_ROOT_V1 = f"{V25P_NATIVE_RELATIVE_ROOT_V1}/profiles"
V25P_NATIVE_EPISODE_RELATIVE_ROOT_V1 = f"{V25P_NATIVE_RELATIVE_ROOT_V1}/episodes"

V25P_COMPONENT_ROLES_V1 = frozenset(
    {"action", "state", "observation", "audit"}
)
V25P_COMPONENT_DTYPES_V1 = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "float16",
        "float32",
        "float64",
    }
)
V25P_COMPONENT_STORAGE_KINDS_V1 = frozenset(
    {"inline_tensor", "external_artifact"}
)
V25P_STANDARD_ACTION_STAGES_V1 = frozenset(
    {
        "source_intent",
        "accepted_reference",
        "admitted_command",
        "dispatched_command",
        "next_accepted_observation",
    }
)
V25P_STANDARD_STATE_STAGES_V1 = frozenset(
    {"source_observed", "causal_measured", "estimated"}
)
V25P_STANDARD_OBSERVATION_STAGES_V1 = frozenset(
    {"source_observed", "causal_observed", "causal_measured", "derived_observation"}
)


def _required(value: object, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _path_segment(value: object, *, name: str) -> str:
    result = _required(value, name=name)
    if result in {".", ".."} or "/" in result or "\\" in result or "\x00" in result:
        raise ValueError(f"{name} must be safe for dataset-relative storage.")
    return result


def _optional(value: object | None, *, name: str) -> str | None:
    return None if value is None else _required(value, name=name)


def _hash(value: object, *, name: str) -> str:
    result = _required(value, name=name).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return result


def _optional_hash(value: object | None, *, name: str) -> str | None:
    return None if value is None else _hash(value, name=name)


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _index(value: object, *, name: str, minimum: int = 0) -> int:
    result = int(value)
    if isinstance(value, bool) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def _unique(values: Sequence[object], *, name: str, required: bool = False) -> tuple[str, ...]:
    result = tuple(_required(value, name=name) for value in values)
    if required and not result:
        raise ValueError(f"{name} must not be empty.")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} values must be unique.")
    return result


def _hashes(values: Sequence[object], *, name: str, required: bool = False) -> tuple[str, ...]:
    result = tuple(_hash(value, name=name) for value in values)
    if required and not result:
        raise ValueError(f"{name} must not be empty.")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} values must be unique.")
    return result


def _stage(value: object, *, role: str) -> str:
    result = _required(value, name="value_stage")
    standards = (
        V25P_STANDARD_ACTION_STAGES_V1
        if role == "action"
        else V25P_STANDARD_STATE_STAGES_V1
        if role == "state"
        else V25P_STANDARD_OBSERVATION_STAGES_V1
        if role == "observation"
        else frozenset()
    )
    if result not in standards and "." not in result:
        raise ValueError(
            "A non-standard component value stage must be a namespaced identifier."
        )
    return result


def _freeze_semantic_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _required(key, name="semantic parameter key"): _freeze_semantic_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_semantic_value(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("Semantic parameter arrays must not use object dtype.")
        raw = np.ascontiguousarray(array).tobytes(order="C")
        return np.frombuffer(raw, dtype=array.dtype).reshape(array.shape)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    raise TypeError(
        "Unsupported semantic parameter value: " f"{type(value).__name__}."
    )


@dataclass(frozen=True)
class V25pComponentStreamSpec:
    stream_id: str
    component_id: str
    dataset_role: str
    value_stage: str
    payload_schema_hash: str
    representation_family: str
    dtype: str
    shape: tuple[int, ...]
    coordinate_names: tuple[str, ...]
    units: tuple[str, ...]
    required_for_sample: bool = True
    storage_kind: str = "inline_tensor"
    reference_frame: str | None = None
    controlled_frame: str | None = None
    expressed_in_frame: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stream_id",
            _path_segment(self.stream_id, name="stream_id"),
        )
        for name in ("component_id", "representation_family"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        role = _required(self.dataset_role, name="dataset_role")
        if role not in V25P_COMPONENT_ROLES_V1:
            raise ValueError(f"Unsupported dataset_role {role!r}.")
        object.__setattr__(self, "dataset_role", role)
        object.__setattr__(self, "value_stage", _stage(self.value_stage, role=role))
        object.__setattr__(
            self,
            "payload_schema_hash",
            _hash(self.payload_schema_hash, name="payload_schema_hash"),
        )
        dtype = np.dtype(self.dtype).name
        if dtype not in V25P_COMPONENT_DTYPES_V1:
            raise ValueError(f"Unsupported component dtype {dtype!r}.")
        shape = tuple(_index(item, name="shape dimension", minimum=1) for item in self.shape)
        if not shape:
            raise ValueError("shape must not be empty.")
        size = int(np.prod(shape, dtype=np.int64))
        coordinates = _unique(self.coordinate_names, name="coordinate_name")
        units = tuple(_required(item, name="unit") for item in self.units)
        if coordinates and len(coordinates) != size:
            raise ValueError("coordinate_names must be empty or match flattened payload size.")
        if len(units) not in {1, size}:
            raise ValueError("units must contain one common unit or one unit per value.")
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "coordinate_names", coordinates)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "required_for_sample", bool(self.required_for_sample))
        storage_kind = _required(self.storage_kind, name="storage_kind")
        if storage_kind not in V25P_COMPONENT_STORAGE_KINDS_V1:
            raise ValueError(f"Unsupported component storage_kind {storage_kind!r}.")
        object.__setattr__(self, "storage_kind", storage_kind)
        for name in ("reference_frame", "controlled_frame", "expressed_in_frame"):
            object.__setattr__(self, name, _optional(getattr(self, name), name=name))

    @property
    def flattened_size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def normalize_values(self, values: object) -> tuple[bool | int | float, ...]:
        raw = np.asarray(values)
        target = np.dtype(self.dtype)
        try:
            normalized = raw.astype(target)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Values for stream {self.stream_id!r} have invalid dtype.") from exc
        if target.kind in {"i", "u"} and not np.array_equal(raw, normalized):
            raise ValueError("Integer component values must be exact integers.")
        if target.kind == "b" and not np.all(np.isin(raw, (False, True, 0, 1))):
            raise ValueError("Boolean component values must contain bool/0/1.")
        if target.kind == "f" and not np.all(np.isfinite(normalized)):
            raise ValueError("Float component values must be finite.")
        if normalized.shape != self.shape:
            raise ValueError(
                f"Values for stream {self.stream_id!r} must have shape {self.shape}, "
                f"got {normalized.shape}."
            )
        return tuple(normalized.reshape(-1).tolist())

    def to_payload(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "component_id": self.component_id,
            "dataset_role": self.dataset_role,
            "value_stage": self.value_stage,
            "payload_schema_hash": self.payload_schema_hash,
            "representation_family": self.representation_family,
            "dtype": self.dtype,
            "shape": self.shape,
            "coordinate_names": self.coordinate_names,
            "units": self.units,
            "required_for_sample": self.required_for_sample,
            "storage_kind": self.storage_kind,
            "reference_frame": self.reference_frame,
            "controlled_frame": self.controlled_frame,
            "expressed_in_frame": self.expressed_in_frame,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pComponentStreamSpec":
        return cls(
            stream_id=payload.get("stream_id", ""),
            component_id=payload.get("component_id", ""),
            dataset_role=payload.get("dataset_role", ""),
            value_stage=payload.get("value_stage", ""),
            payload_schema_hash=payload.get("payload_schema_hash", ""),
            representation_family=payload.get("representation_family", ""),
            dtype=payload.get("dtype", ""),
            shape=tuple(payload.get("shape", ())),
            coordinate_names=tuple(payload.get("coordinate_names", ())),
            units=tuple(payload.get("units", ())),
            required_for_sample=payload.get("required_for_sample", True),
            storage_kind=payload.get("storage_kind", "inline_tensor"),
            reference_frame=payload.get("reference_frame"),
            controlled_frame=payload.get("controlled_frame"),
            expressed_in_frame=payload.get("expressed_in_frame"),
        )


@dataclass(frozen=True)
class V25pComponentStreamProfile:
    profile_id: str
    profile_version: str
    embodiment_profile_hash: str
    streams: tuple[V25pComponentStreamSpec, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=V25P_NATIVE_COMPONENT_STREAM_PROFILE_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _required(self.profile_id, name="profile_id"))
        object.__setattr__(
            self, "profile_version", _required(self.profile_version, name="profile_version")
        )
        object.__setattr__(
            self,
            "embodiment_profile_hash",
            _hash(self.embodiment_profile_hash, name="embodiment_profile_hash"),
        )
        streams = tuple(self.streams)
        if not streams or any(not isinstance(item, V25pComponentStreamSpec) for item in streams):
            raise ValueError("streams must contain V25pComponentStreamSpec values.")
        ids = tuple(item.stream_id for item in streams)
        if len(ids) != len(set(ids)):
            raise ValueError("stream_id values must be unique.")
        semantic_keys = tuple(
            (
                item.component_id,
                item.dataset_role,
                item.value_stage,
                item.payload_schema_hash,
            )
            for item in streams
        )
        if len(semantic_keys) != len(set(semantic_keys)):
            raise ValueError(
                "component/role/stage/payload stream bindings must be unique."
            )
        object.__setattr__(self, "streams", streams)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=V25P_NATIVE_STREAM_PROFILE_HASH_DOMAIN_V1,
            ),
        )

    @property
    def stream_by_id(self) -> Mapping[str, V25pComponentStreamSpec]:
        return MappingProxyType({item.stream_id: item for item in self.streams})

    @property
    def required_stream_ids(self) -> tuple[str, ...]:
        return tuple(item.stream_id for item in self.streams if item.required_for_sample)

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "embodiment_profile_hash": self.embodiment_profile_hash,
            "streams": [item.to_payload() for item in self.streams],
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pComponentStreamProfile":
        if payload.get("schema") != V25P_NATIVE_COMPONENT_STREAM_PROFILE_SCHEMA_V1:
            raise ValueError("Unsupported native component stream profile schema.")
        result = cls(
            profile_id=payload.get("profile_id", ""),
            profile_version=payload.get("profile_version", ""),
            embodiment_profile_hash=payload.get("embodiment_profile_hash", ""),
            streams=tuple(
                V25pComponentStreamSpec.from_payload(item)
                for item in payload.get("streams", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ValueError("Native component stream profile hash mismatch.")
        return result


@dataclass(frozen=True)
class V25pComponentStreamSpecV2:
    """One native stream with an optional canonical action-component binding."""

    stream_id: str
    component_id: str
    dataset_role: str
    value_stage: str
    payload_schema_hash: str
    representation_family: str
    dtype: str
    shape: tuple[int, ...]
    coordinate_names: tuple[str, ...]
    units: tuple[str, ...]
    required_for_sample: bool = True
    storage_kind: str = "inline_tensor"
    reference_frame: str | None = None
    controlled_frame: str | None = None
    expressed_in_frame: str | None = None
    value_semantics: str | None = None
    semantic_parameters: Mapping[str, Any] = field(default_factory=dict)
    action_component_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stream_id",
            _path_segment(self.stream_id, name="stream_id"),
        )
        object.__setattr__(
            self,
            "component_id",
            _required(self.component_id, name="component_id"),
        )
        object.__setattr__(
            self,
            "representation_family",
            _required(self.representation_family, name="representation_family"),
        )
        role = _required(self.dataset_role, name="dataset_role")
        if role not in V25P_COMPONENT_ROLES_V1:
            raise ValueError(f"Unsupported dataset_role {role!r}.")
        object.__setattr__(self, "dataset_role", role)
        object.__setattr__(self, "value_stage", _stage(self.value_stage, role=role))
        object.__setattr__(
            self,
            "payload_schema_hash",
            _hash(self.payload_schema_hash, name="payload_schema_hash"),
        )
        dtype = np.dtype(self.dtype).name
        if dtype not in V25P_COMPONENT_DTYPES_V1:
            raise ValueError(f"Unsupported component dtype {dtype!r}.")
        shape = tuple(
            _index(item, name="shape dimension", minimum=1) for item in self.shape
        )
        if not shape:
            raise ValueError("shape must not be empty.")
        size = int(np.prod(shape, dtype=np.int64))
        coordinates = _unique(self.coordinate_names, name="coordinate_name")
        units = tuple(_required(item, name="unit") for item in self.units)
        if coordinates and len(coordinates) != size:
            raise ValueError(
                "coordinate_names must be empty or match flattened payload size."
            )
        if len(units) not in {1, size}:
            raise ValueError("units must contain one common unit or one unit per value.")
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "coordinate_names", coordinates)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "required_for_sample", bool(self.required_for_sample))
        storage_kind = _required(self.storage_kind, name="storage_kind")
        if storage_kind not in V25P_COMPONENT_STORAGE_KINDS_V1:
            raise ValueError(f"Unsupported component storage_kind {storage_kind!r}.")
        object.__setattr__(self, "storage_kind", storage_kind)
        for name in ("reference_frame", "controlled_frame", "expressed_in_frame"):
            object.__setattr__(self, name, _optional(getattr(self, name), name=name))
        value_semantics = _optional(self.value_semantics, name="value_semantics")
        action_component_id = _optional(
            self.action_component_id,
            name="action_component_id",
        )
        if role == "action" and (
            value_semantics is None or action_component_id is None
        ):
            raise ValueError(
                "Action streams require value_semantics and action_component_id."
            )
        if role != "action" and action_component_id is not None:
            raise ValueError(
                "Only action streams may bind a canonical action_component_id."
            )
        object.__setattr__(self, "value_semantics", value_semantics)
        object.__setattr__(self, "action_component_id", action_component_id)
        object.__setattr__(
            self,
            "semantic_parameters",
            _freeze_semantic_value(dict(self.semantic_parameters)),
        )

    @property
    def flattened_size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def normalize_values(self, values: object) -> tuple[bool | int | float, ...]:
        raw = np.asarray(values)
        target = np.dtype(self.dtype)
        try:
            normalized = raw.astype(target)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Values for stream {self.stream_id!r} have invalid dtype."
            ) from exc
        if target.kind in {"i", "u"} and not np.array_equal(raw, normalized):
            raise ValueError("Integer component values must be exact integers.")
        if target.kind == "b" and not np.all(np.isin(raw, (False, True, 0, 1))):
            raise ValueError("Boolean component values must contain bool/0/1.")
        if target.kind == "f" and not np.all(np.isfinite(normalized)):
            raise ValueError("Float component values must be finite.")
        if normalized.shape != self.shape:
            raise ValueError(
                f"Values for stream {self.stream_id!r} must have shape {self.shape}, "
                f"got {normalized.shape}."
            )
        return tuple(normalized.reshape(-1).tolist())

    def to_payload(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "component_id": self.component_id,
            "dataset_role": self.dataset_role,
            "value_stage": self.value_stage,
            "payload_schema_hash": self.payload_schema_hash,
            "representation_family": self.representation_family,
            "dtype": self.dtype,
            "shape": self.shape,
            "coordinate_names": self.coordinate_names,
            "units": self.units,
            "required_for_sample": self.required_for_sample,
            "storage_kind": self.storage_kind,
            "reference_frame": self.reference_frame,
            "controlled_frame": self.controlled_frame,
            "expressed_in_frame": self.expressed_in_frame,
            "value_semantics": self.value_semantics,
            "semantic_parameters": self.semantic_parameters,
            "action_component_id": self.action_component_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pComponentStreamSpecV2":
        return cls(
            stream_id=payload.get("stream_id", ""),
            component_id=payload.get("component_id", ""),
            dataset_role=payload.get("dataset_role", ""),
            value_stage=payload.get("value_stage", ""),
            payload_schema_hash=payload.get("payload_schema_hash", ""),
            representation_family=payload.get("representation_family", ""),
            dtype=payload.get("dtype", ""),
            shape=tuple(payload.get("shape", ())),
            coordinate_names=tuple(payload.get("coordinate_names", ())),
            units=tuple(payload.get("units", ())),
            required_for_sample=payload.get("required_for_sample", True),
            storage_kind=payload.get("storage_kind", "inline_tensor"),
            reference_frame=payload.get("reference_frame"),
            controlled_frame=payload.get("controlled_frame"),
            expressed_in_frame=payload.get("expressed_in_frame"),
            value_semantics=payload.get("value_semantics"),
            semantic_parameters=dict(payload.get("semantic_parameters", {})),
            action_component_id=payload.get("action_component_id"),
        )


@dataclass(frozen=True)
class V25pComponentStreamProfileV2:
    """Self-contained native stream profile bound to one canonical action space."""

    profile_id: str
    profile_version: str
    embodiment_profile_hash: str
    streams: tuple[V25pComponentStreamSpecV2, ...]
    component_action_space_schema: ComponentActionSpaceSchema
    component_catalog_definition: Mapping[str, Any]
    action_payload_schemas: tuple[ActionPayloadSchema, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=V25P_NATIVE_COMPONENT_STREAM_PROFILE_SCHEMA_V2,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _required(self.profile_id, name="profile_id"))
        object.__setattr__(
            self,
            "profile_version",
            _required(self.profile_version, name="profile_version"),
        )
        object.__setattr__(
            self,
            "embodiment_profile_hash",
            _hash(self.embodiment_profile_hash, name="embodiment_profile_hash"),
        )
        if not isinstance(self.component_action_space_schema, ComponentActionSpaceSchema):
            raise TypeError(
                "component_action_space_schema must be ComponentActionSpaceSchema."
            )
        catalog_definition = _freeze_semantic_value(
            dict(self.component_catalog_definition)
        )
        catalog_hash = canonical_content_hash(
            catalog_definition,
            domain=ACTION_COMPONENT_CATALOG_HASH_DOMAIN_V2,
        )
        if catalog_hash != self.component_action_space_schema.component_catalog_hash:
            raise ValueError(
                "Component catalog definition does not reproduce the action-space "
                "component_catalog_hash."
            )
        streams = tuple(self.streams)
        if not streams or any(
            not isinstance(item, V25pComponentStreamSpecV2) for item in streams
        ):
            raise ValueError("streams must contain V25pComponentStreamSpecV2 values.")
        ids = tuple(item.stream_id for item in streams)
        if len(ids) != len(set(ids)):
            raise ValueError("stream_id values must be unique.")
        semantic_keys = tuple(
            (
                item.component_id,
                item.dataset_role,
                item.value_stage,
                item.payload_schema_hash,
            )
            for item in streams
        )
        if len(semantic_keys) != len(set(semantic_keys)):
            raise ValueError(
                "component/role/stage/payload stream bindings must be unique."
            )
        supplied = tuple(self.action_payload_schemas)
        if any(not isinstance(item, ActionPayloadSchema) for item in supplied):
            raise TypeError(
                "action_payload_schemas must contain ActionPayloadSchema values."
            )
        by_hash = {item.semantic_content_hash: item for item in supplied}
        if len(by_hash) != len(supplied):
            raise ValueError("action_payload_schemas must not repeat a schema hash.")
        expected_hashes = set(
            self.component_action_space_schema.payload_schema_hash_by_component.values()
        )
        if set(by_hash) != expected_hashes:
            raise ValueError(
                "Action payload schema definitions must exactly cover the action space."
            )
        for stream in streams:
            if stream.dataset_role != "action":
                continue
            action_component_id = str(stream.action_component_id)
            expected_hash = (
                self.component_action_space_schema.payload_schema_hash_by_component.get(
                    action_component_id
                )
            )
            if expected_hash is None:
                raise ValueError(
                    f"Action stream {stream.stream_id!r} binds unknown component "
                    f"{action_component_id!r}."
                )
            if stream.payload_schema_hash != expected_hash:
                raise ValueError(
                    f"Action stream {stream.stream_id!r} payload schema does not "
                    "match its canonical action component."
                )
            self._validate_action_stream(stream=stream, schema=by_hash[expected_hash])
        # A canonical mapping does not preserve a producer's insertion order
        # across storage boundaries.  Keep the embedded definition list stable
        # by schema hash so a write/read round trip reproduces the profile hash.
        ordered_hashes = tuple(sorted(by_hash))
        object.__setattr__(self, "streams", streams)
        object.__setattr__(
            self,
            "component_catalog_definition",
            catalog_definition,
        )
        object.__setattr__(
            self,
            "action_payload_schemas",
            tuple(by_hash[item] for item in ordered_hashes),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=V25P_NATIVE_STREAM_PROFILE_HASH_DOMAIN_V2,
            ),
        )

    @staticmethod
    def _validate_action_stream(
        *,
        stream: V25pComponentStreamSpecV2,
        schema: ActionPayloadSchema,
    ) -> None:
        comparisons = {
            "representation_family": (stream.representation_family, schema.representation_family),
            "dtype": (stream.dtype, schema.dtype),
            "shape": (stream.shape, schema.shape),
            "coordinate_names": (stream.coordinate_names, schema.coordinate_names),
            "units": (stream.units, schema.units),
            "value_semantics": (stream.value_semantics, schema.value_semantics),
            "reference_frame": (stream.reference_frame, schema.reference_frame),
            "controlled_frame": (stream.controlled_frame, schema.controlled_frame),
            "expressed_in_frame": (stream.expressed_in_frame, schema.expressed_in_frame),
            "semantic_parameters": (
                dict(stream.semantic_parameters),
                dict(schema.semantic_parameters),
            ),
        }
        mismatched = tuple(
            name for name, (actual, expected) in comparisons.items() if actual != expected
        )
        if mismatched:
            raise ValueError(
                f"Action stream {stream.stream_id!r} disagrees with payload schema "
                f"for fields {mismatched}."
            )

    @property
    def stream_by_id(self) -> Mapping[str, V25pComponentStreamSpecV2]:
        return MappingProxyType({item.stream_id: item for item in self.streams})

    @property
    def action_stream_by_component_id(self) -> Mapping[str, V25pComponentStreamSpecV2]:
        result: dict[str, V25pComponentStreamSpecV2] = {}
        for stream in self.streams:
            if stream.dataset_role != "action":
                continue
            component_id = str(stream.action_component_id)
            if component_id in result:
                raise ValueError(
                    "A direct ComponentAction projection requires one action stream "
                    f"per component; {component_id!r} is ambiguous."
                )
            result[component_id] = stream
        return MappingProxyType(result)

    @property
    def required_stream_ids(self) -> tuple[str, ...]:
        return tuple(item.stream_id for item in self.streams if item.required_for_sample)

    @property
    def action_payload_schema_by_hash(self) -> Mapping[str, ActionPayloadSchema]:
        return MappingProxyType(
            {item.semantic_content_hash: item for item in self.action_payload_schemas}
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "embodiment_profile_hash": self.embodiment_profile_hash,
            "component_action_space_schema": (
                self.component_action_space_schema.semantic_payload()
            ),
            "component_catalog_definition": self.component_catalog_definition,
            "action_payload_schemas": [
                item.semantic_payload() for item in self.action_payload_schemas
            ],
            "streams": [item.to_payload() for item in self.streams],
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pComponentStreamProfileV2":
        if payload.get("schema") != V25P_NATIVE_COMPONENT_STREAM_PROFILE_SCHEMA_V2:
            raise ValueError("Unsupported native component stream profile v2 schema.")
        result = cls(
            profile_id=payload.get("profile_id", ""),
            profile_version=payload.get("profile_version", ""),
            embodiment_profile_hash=payload.get("embodiment_profile_hash", ""),
            streams=tuple(
                V25pComponentStreamSpecV2.from_payload(item)
                for item in payload.get("streams", ())
            ),
            component_action_space_schema=ComponentActionSpaceSchema.from_payload(
                payload.get("component_action_space_schema", {})
            ),
            component_catalog_definition=dict(
                payload.get("component_catalog_definition", {})
            ),
            action_payload_schemas=tuple(
                ActionPayloadSchema.from_payload(item)
                for item in payload.get("action_payload_schemas", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ValueError("Native component stream profile v2 hash mismatch.")
        return result


V25pComponentStreamSpecLike = V25pComponentStreamSpec | V25pComponentStreamSpecV2
V25pComponentStreamProfileLike = (
    V25pComponentStreamProfile | V25pComponentStreamProfileV2
)


def component_stream_profile_from_payload(
    payload: Mapping[str, Any],
) -> V25pComponentStreamProfileLike:
    """Parse a frozen v1 or corrected v2 provisional stream profile.

    Dispatch is explicit by schema ID so a v1 artifact can never acquire v2
    action semantics merely because its fields happen to look compatible.
    """

    schema = payload.get("schema")
    if schema == V25P_NATIVE_COMPONENT_STREAM_PROFILE_SCHEMA_V1:
        return V25pComponentStreamProfile.from_payload(payload)
    if schema == V25P_NATIVE_COMPONENT_STREAM_PROFILE_SCHEMA_V2:
        return V25pComponentStreamProfileV2.from_payload(payload)
    raise ValueError("Unsupported native component stream profile schema.")


def normalize_component_stream_stored_values(
    stream: V25pComponentStreamSpecLike,
    values: object,
) -> tuple[bool | int | float, ...]:
    """Validate a record's flattened tensor against its logical stream shape."""

    raw = np.asarray(values)
    if raw.size != stream.flattened_size:
        raise ValueError(
            f"Stored values for stream {stream.stream_id!r} must contain "
            f"{stream.flattened_size} values, got {raw.size}."
        )
    return stream.normalize_values(raw.reshape(stream.shape))


@dataclass(frozen=True)
class V25pNativeEmbodimentBinding:
    episode_uuid: str
    legacy_episode_index: int
    episode_identity_hash: str
    embodiment_profile_hash: str
    component_stream_profile_hash: str
    bound_at: float
    bound_clock_domain_id: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=V25P_NATIVE_EMBODIMENT_BINDING_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_uuid", _required(self.episode_uuid, name="episode_uuid"))
        object.__setattr__(
            self,
            "legacy_episode_index",
            _index(self.legacy_episode_index, name="legacy_episode_index"),
        )
        for name in (
            "episode_identity_hash",
            "embodiment_profile_hash",
            "component_stream_profile_hash",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        object.__setattr__(self, "bound_at", _finite(self.bound_at, name="bound_at"))
        object.__setattr__(
            self,
            "bound_clock_domain_id",
            _required(self.bound_clock_domain_id, name="bound_clock_domain_id"),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False), domain=V25P_NATIVE_BINDING_HASH_DOMAIN_V1
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "episode_identity_hash": self.episode_identity_hash,
            "embodiment_profile_hash": self.embodiment_profile_hash,
            "component_stream_profile_hash": self.component_stream_profile_hash,
            "bound_at": self.bound_at,
            "bound_clock_domain_id": self.bound_clock_domain_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pNativeEmbodimentBinding":
        if payload.get("schema") != V25P_NATIVE_EMBODIMENT_BINDING_SCHEMA_V1:
            raise ValueError("Unsupported native embodiment binding schema.")
        result = cls(
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            episode_identity_hash=payload.get("episode_identity_hash", ""),
            embodiment_profile_hash=payload.get("embodiment_profile_hash", ""),
            component_stream_profile_hash=payload.get("component_stream_profile_hash", ""),
            bound_at=payload.get("bound_at", float("nan")),
            bound_clock_domain_id=payload.get("bound_clock_domain_id", ""),
        )
        if payload.get("semantic_content_hash") not in {None, result.semantic_content_hash}:
            raise ValueError("Native embodiment binding hash mismatch.")
        return result


@dataclass(frozen=True)
class V25pExternalPayloadReference:
    artifact_locator: str
    artifact_content_hash: str
    encoding: str
    sample_selector: str | None = None
    byte_offset: int | None = None
    byte_length: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_locator",
            _required(self.artifact_locator, name="artifact_locator"),
        )
        object.__setattr__(
            self,
            "artifact_content_hash",
            _hash(self.artifact_content_hash, name="artifact_content_hash"),
        )
        object.__setattr__(self, "encoding", _required(self.encoding, name="encoding"))
        object.__setattr__(
            self,
            "sample_selector",
            _optional(self.sample_selector, name="sample_selector"),
        )
        if (self.byte_offset is None) != (self.byte_length is None):
            raise ValueError("byte_offset and byte_length must be declared together.")
        if self.byte_offset is not None:
            object.__setattr__(
                self,
                "byte_offset",
                _index(self.byte_offset, name="byte_offset"),
            )
            object.__setattr__(
                self,
                "byte_length",
                _index(self.byte_length, name="byte_length", minimum=1),
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_locator": self.artifact_locator,
            "artifact_content_hash": self.artifact_content_hash,
            "encoding": self.encoding,
            "sample_selector": self.sample_selector,
            "byte_offset": self.byte_offset,
            "byte_length": self.byte_length,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "V25pExternalPayloadReference":
        return cls(
            artifact_locator=payload.get("artifact_locator", ""),
            artifact_content_hash=payload.get("artifact_content_hash", ""),
            encoding=payload.get("encoding", ""),
            sample_selector=payload.get("sample_selector"),
            byte_offset=payload.get("byte_offset"),
            byte_length=payload.get("byte_length"),
        )


@dataclass(frozen=True)
class V25pComponentRecord:
    episode_uuid: str
    legacy_episode_index: int
    component_stream_profile_hash: str
    stream_id: str
    component_id: str
    dataset_role: str
    value_stage: str
    payload_schema_hash: str
    record_sequence: int
    sample_index: int
    effective_at: float
    available_at: float
    effective_clock_domain_id: str
    available_clock_domain_id: str
    values: tuple[bool | int | float, ...] | None = None
    payload_reference: V25pExternalPayloadReference | None = None
    valid: bool = True
    invalid_reason: str = ""
    source_record_id: str | None = None
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=V25P_NATIVE_COMPONENT_RECORD_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in ("episode_uuid", "stream_id", "component_id", "dataset_role"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        if self.dataset_role not in V25P_COMPONENT_ROLES_V1:
            raise ValueError(f"Unsupported dataset_role {self.dataset_role!r}.")
        object.__setattr__(
            self,
            "legacy_episode_index",
            _index(self.legacy_episode_index, name="legacy_episode_index"),
        )
        object.__setattr__(
            self,
            "component_stream_profile_hash",
            _hash(self.component_stream_profile_hash, name="component_stream_profile_hash"),
        )
        object.__setattr__(
            self, "payload_schema_hash", _hash(self.payload_schema_hash, name="payload_schema_hash")
        )
        object.__setattr__(self, "value_stage", _stage(self.value_stage, role=self.dataset_role))
        object.__setattr__(
            self, "record_sequence", _index(self.record_sequence, name="record_sequence", minimum=1)
        )
        object.__setattr__(self, "sample_index", _index(self.sample_index, name="sample_index"))
        effective = _finite(self.effective_at, name="effective_at")
        available = _finite(self.available_at, name="available_at")
        effective_clock = _required(
            self.effective_clock_domain_id, name="effective_clock_domain_id"
        )
        available_clock = _required(
            self.available_clock_domain_id, name="available_clock_domain_id"
        )
        if effective_clock == available_clock and available < effective:
            raise ValueError("available_at cannot precede effective_at in one clock domain.")
        object.__setattr__(self, "effective_at", effective)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "effective_clock_domain_id", effective_clock)
        object.__setattr__(self, "available_clock_domain_id", available_clock)
        valid = bool(self.valid)
        reason = str(self.invalid_reason).strip()
        values = None if self.values is None else tuple(self.values)
        reference = self.payload_reference
        if reference is not None and not isinstance(
            reference, V25pExternalPayloadReference
        ):
            raise TypeError(
                "payload_reference must be V25pExternalPayloadReference."
            )
        if valid and ((values is None) == (reference is None)):
            raise ValueError(
                "A valid component record requires exactly one inline value or "
                "external payload reference."
            )
        if not valid and (values is not None or reference is not None or not reason):
            raise ValueError(
                "An invalid component record requires only an invalid_reason."
            )
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "payload_reference", reference)
        object.__setattr__(self, "invalid_reason", reason)
        object.__setattr__(
            self, "source_record_id", _optional(self.source_record_id, name="source_record_id")
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=V25P_NATIVE_COMPONENT_RECORD_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "component_stream_profile_hash": self.component_stream_profile_hash,
            "stream_id": self.stream_id,
            "component_id": self.component_id,
            "dataset_role": self.dataset_role,
            "value_stage": self.value_stage,
            "payload_schema_hash": self.payload_schema_hash,
            "record_sequence": self.record_sequence,
            "sample_index": self.sample_index,
            "effective_at": self.effective_at,
            "available_at": self.available_at,
            "effective_clock_domain_id": self.effective_clock_domain_id,
            "available_clock_domain_id": self.available_clock_domain_id,
            "values": self.values,
            "payload_reference": (
                None
                if self.payload_reference is None
                else self.payload_reference.to_payload()
            ),
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "source_record_id": self.source_record_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pComponentRecord":
        if payload.get("schema") != V25P_NATIVE_COMPONENT_RECORD_SCHEMA_V1:
            raise ValueError("Unsupported component record schema.")
        raw_values = payload.get("values")
        raw_reference = payload.get("payload_reference")
        result = cls(
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            component_stream_profile_hash=payload.get("component_stream_profile_hash", ""),
            stream_id=payload.get("stream_id", ""),
            component_id=payload.get("component_id", ""),
            dataset_role=payload.get("dataset_role", ""),
            value_stage=payload.get("value_stage", ""),
            payload_schema_hash=payload.get("payload_schema_hash", ""),
            record_sequence=payload.get("record_sequence", 0),
            sample_index=payload.get("sample_index", -1),
            effective_at=payload.get("effective_at", float("nan")),
            available_at=payload.get("available_at", float("nan")),
            effective_clock_domain_id=payload.get("effective_clock_domain_id", ""),
            available_clock_domain_id=payload.get("available_clock_domain_id", ""),
            values=None if raw_values is None else tuple(raw_values),
            payload_reference=(
                None
                if raw_reference is None
                else V25pExternalPayloadReference.from_payload(raw_reference)
            ),
            valid=payload.get("valid", True),
            invalid_reason=payload.get("invalid_reason", ""),
            source_record_id=payload.get("source_record_id"),
        )
        if payload.get("semantic_content_hash") not in {None, result.semantic_content_hash}:
            raise ValueError("Component record hash mismatch.")
        return result


@dataclass(frozen=True)
class V25pCausalSample:
    episode_uuid: str
    legacy_episode_index: int
    component_stream_profile_hash: str
    sample_index: int
    cut_at: float
    cut_clock_domain_id: str
    record_hash_by_stream: Mapping[str, str]
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=V25P_NATIVE_CAUSAL_SAMPLE_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_uuid", _required(self.episode_uuid, name="episode_uuid"))
        object.__setattr__(
            self,
            "legacy_episode_index",
            _index(self.legacy_episode_index, name="legacy_episode_index"),
        )
        object.__setattr__(
            self,
            "component_stream_profile_hash",
            _hash(self.component_stream_profile_hash, name="component_stream_profile_hash"),
        )
        object.__setattr__(self, "sample_index", _index(self.sample_index, name="sample_index"))
        object.__setattr__(self, "cut_at", _finite(self.cut_at, name="cut_at"))
        object.__setattr__(
            self,
            "cut_clock_domain_id",
            _required(self.cut_clock_domain_id, name="cut_clock_domain_id"),
        )
        records = {
            _required(stream_id, name="stream_id"): _hash(value, name="record hash")
            for stream_id, value in dict(self.record_hash_by_stream).items()
        }
        if not records:
            raise ValueError("A causal sample must reference at least one component record.")
        object.__setattr__(self, "record_hash_by_stream", MappingProxyType(dict(sorted(records.items()))))
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False), domain=V25P_NATIVE_CAUSAL_SAMPLE_HASH_DOMAIN_V1
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "component_stream_profile_hash": self.component_stream_profile_hash,
            "sample_index": self.sample_index,
            "cut_at": self.cut_at,
            "cut_clock_domain_id": self.cut_clock_domain_id,
            "record_hash_by_stream": dict(self.record_hash_by_stream),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pCausalSample":
        if payload.get("schema") != V25P_NATIVE_CAUSAL_SAMPLE_SCHEMA_V1:
            raise ValueError("Unsupported causal sample schema.")
        result = cls(
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            component_stream_profile_hash=payload.get("component_stream_profile_hash", ""),
            sample_index=payload.get("sample_index", -1),
            cut_at=payload.get("cut_at", float("nan")),
            cut_clock_domain_id=payload.get("cut_clock_domain_id", ""),
            record_hash_by_stream=dict(payload.get("record_hash_by_stream", {})),
        )
        if payload.get("semantic_content_hash") not in {None, result.semantic_content_hash}:
            raise ValueError("Causal sample hash mismatch.")
        return result


@dataclass(frozen=True)
class V25pNativeDatasetManifest:
    dataset_id: str
    embodiment_profile_hashes: tuple[str, ...]
    component_stream_profile_hashes: tuple[str, ...]
    created_at: float
    created_clock_domain_id: str
    long_horizon_task_profile_hashes: tuple[str, ...] = ()
    compatibility_projection_profile_hashes: tuple[str, ...] = ()
    status: str = V25P_NATIVE_DATASET_STATUS_V1
    native_v25_eligible: bool = False
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=V25P_NATIVE_DATASET_MANIFEST_SCHEMA_V2, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _required(self.dataset_id, name="dataset_id"))
        for name in (
            "embodiment_profile_hashes",
            "component_stream_profile_hashes",
            "long_horizon_task_profile_hashes",
            "compatibility_projection_profile_hashes",
        ):
            object.__setattr__(
                self,
                name,
                _hashes(
                    getattr(self, name),
                    name=name,
                    required=name in {"embodiment_profile_hashes", "component_stream_profile_hashes"},
                ),
            )
        object.__setattr__(self, "created_at", _finite(self.created_at, name="created_at"))
        object.__setattr__(
            self,
            "created_clock_domain_id",
            _required(self.created_clock_domain_id, name="created_clock_domain_id"),
        )
        if self.status != V25P_NATIVE_DATASET_STATUS_V1 or bool(self.native_v25_eligible):
            raise ValueError("A provisional native v2.5-p dataset cannot claim native v2.5.")
        object.__setattr__(self, "native_v25_eligible", False)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False), domain=V25P_NATIVE_DATASET_HASH_DOMAIN_V2
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "embodiment_profile_hashes": self.embodiment_profile_hashes,
            "component_stream_profile_hashes": self.component_stream_profile_hashes,
            "long_horizon_task_profile_hashes": self.long_horizon_task_profile_hashes,
            "compatibility_projection_profile_hashes": self.compatibility_projection_profile_hashes,
            "created_at": self.created_at,
            "created_clock_domain_id": self.created_clock_domain_id,
            "status": self.status,
            "native_v25_eligible": self.native_v25_eligible,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pNativeDatasetManifest":
        if payload.get("schema") != V25P_NATIVE_DATASET_MANIFEST_SCHEMA_V2:
            raise ValueError("Unsupported native v2.5-p dataset manifest schema.")
        result = cls(
            dataset_id=payload.get("dataset_id", ""),
            embodiment_profile_hashes=tuple(payload.get("embodiment_profile_hashes", ())),
            component_stream_profile_hashes=tuple(
                payload.get("component_stream_profile_hashes", ())
            ),
            long_horizon_task_profile_hashes=tuple(
                payload.get("long_horizon_task_profile_hashes", ())
            ),
            compatibility_projection_profile_hashes=tuple(
                payload.get("compatibility_projection_profile_hashes", ())
            ),
            created_at=payload.get("created_at", float("nan")),
            created_clock_domain_id=payload.get("created_clock_domain_id", ""),
            status=payload.get("status", ""),
            native_v25_eligible=payload.get("native_v25_eligible", False),
        )
        if payload.get("semantic_content_hash") not in {None, result.semantic_content_hash}:
            raise ValueError("Native dataset manifest hash mismatch.")
        return result


@dataclass(frozen=True)
class V25pNativeCaptureManifest:
    episode_uuid: str
    legacy_episode_index: int
    episode_identity_hash: str
    native_embodiment_binding_hash: str
    component_stream_profile_hash: str
    sample_count: int
    record_count_by_stream: Mapping[str, int]
    record_bundle_hash_by_stream: Mapping[str, str]
    sample_bundle_hash: str
    sealed_at: float
    sealed_clock_domain_id: str
    task_semantics_manifest_hash: str | None = None
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=V25P_NATIVE_CAPTURE_MANIFEST_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_uuid", _required(self.episode_uuid, name="episode_uuid"))
        object.__setattr__(
            self,
            "legacy_episode_index",
            _index(self.legacy_episode_index, name="legacy_episode_index"),
        )
        for name in (
            "episode_identity_hash",
            "native_embodiment_binding_hash",
            "component_stream_profile_hash",
            "sample_bundle_hash",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "task_semantics_manifest_hash",
            _optional_hash(
                self.task_semantics_manifest_hash, name="task_semantics_manifest_hash"
            ),
        )
        object.__setattr__(self, "sample_count", _index(self.sample_count, name="sample_count"))
        counts = {
            _required(key, name="stream_id"): _index(value, name="record_count")
            for key, value in dict(self.record_count_by_stream).items()
        }
        bundles = {
            _required(key, name="stream_id"): _hash(value, name="record_bundle_hash")
            for key, value in dict(self.record_bundle_hash_by_stream).items()
        }
        if set(counts) != set(bundles):
            raise ValueError("Record count and bundle maps must bind identical streams.")
        object.__setattr__(self, "record_count_by_stream", MappingProxyType(dict(sorted(counts.items()))))
        object.__setattr__(
            self, "record_bundle_hash_by_stream", MappingProxyType(dict(sorted(bundles.items())))
        )
        object.__setattr__(self, "sealed_at", _finite(self.sealed_at, name="sealed_at"))
        object.__setattr__(
            self,
            "sealed_clock_domain_id",
            _required(self.sealed_clock_domain_id, name="sealed_clock_domain_id"),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=V25P_NATIVE_CAPTURE_MANIFEST_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "episode_identity_hash": self.episode_identity_hash,
            "native_embodiment_binding_hash": self.native_embodiment_binding_hash,
            "component_stream_profile_hash": self.component_stream_profile_hash,
            "sample_count": self.sample_count,
            "record_count_by_stream": dict(self.record_count_by_stream),
            "record_bundle_hash_by_stream": dict(self.record_bundle_hash_by_stream),
            "sample_bundle_hash": self.sample_bundle_hash,
            "task_semantics_manifest_hash": self.task_semantics_manifest_hash,
            "sealed_at": self.sealed_at,
            "sealed_clock_domain_id": self.sealed_clock_domain_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pNativeCaptureManifest":
        if payload.get("schema") != V25P_NATIVE_CAPTURE_MANIFEST_SCHEMA_V1:
            raise ValueError("Unsupported native capture manifest schema.")
        result = cls(
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            episode_identity_hash=payload.get("episode_identity_hash", ""),
            native_embodiment_binding_hash=payload.get("native_embodiment_binding_hash", ""),
            component_stream_profile_hash=payload.get("component_stream_profile_hash", ""),
            sample_count=payload.get("sample_count", -1),
            record_count_by_stream=dict(payload.get("record_count_by_stream", {})),
            record_bundle_hash_by_stream=dict(
                payload.get("record_bundle_hash_by_stream", {})
            ),
            sample_bundle_hash=payload.get("sample_bundle_hash", ""),
            task_semantics_manifest_hash=payload.get("task_semantics_manifest_hash"),
            sealed_at=payload.get("sealed_at", float("nan")),
            sealed_clock_domain_id=payload.get("sealed_clock_domain_id", ""),
        )
        if payload.get("semantic_content_hash") not in {None, result.semantic_content_hash}:
            raise ValueError("Native capture manifest hash mismatch.")
        return result


@dataclass(frozen=True)
class V25pNativeEpisodeManifest:
    episode_uuid: str
    legacy_episode_index: int
    episode_identity_hash: str
    native_embodiment_binding_hash: str
    native_capture_manifest_hash: str
    committed_at: float
    committed_clock_domain_id: str
    compatibility_projection_receipt_hashes: tuple[str, ...] = ()
    v24_capture_commit_hash: str | None = None
    status: str = V25P_NATIVE_EPISODE_STATUS_V1
    native_v25_eligible: bool = False
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=V25P_NATIVE_EPISODE_MANIFEST_SCHEMA_V2, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_uuid", _required(self.episode_uuid, name="episode_uuid"))
        object.__setattr__(
            self,
            "legacy_episode_index",
            _index(self.legacy_episode_index, name="legacy_episode_index"),
        )
        for name in (
            "episode_identity_hash",
            "native_embodiment_binding_hash",
            "native_capture_manifest_hash",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "compatibility_projection_receipt_hashes",
            _hashes(
                self.compatibility_projection_receipt_hashes,
                name="compatibility_projection_receipt_hash",
            ),
        )
        object.__setattr__(
            self,
            "v24_capture_commit_hash",
            _optional_hash(self.v24_capture_commit_hash, name="v24_capture_commit_hash"),
        )
        object.__setattr__(self, "committed_at", _finite(self.committed_at, name="committed_at"))
        object.__setattr__(
            self,
            "committed_clock_domain_id",
            _required(self.committed_clock_domain_id, name="committed_clock_domain_id"),
        )
        if self.status != V25P_NATIVE_EPISODE_STATUS_V1 or bool(self.native_v25_eligible):
            raise ValueError("A provisional native episode cannot claim native v2.5.")
        object.__setattr__(self, "native_v25_eligible", False)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=V25P_NATIVE_EPISODE_MANIFEST_HASH_DOMAIN_V2,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "episode_identity_hash": self.episode_identity_hash,
            "native_embodiment_binding_hash": self.native_embodiment_binding_hash,
            "native_capture_manifest_hash": self.native_capture_manifest_hash,
            "compatibility_projection_receipt_hashes": (
                self.compatibility_projection_receipt_hashes
            ),
            "v24_capture_commit_hash": self.v24_capture_commit_hash,
            "committed_at": self.committed_at,
            "committed_clock_domain_id": self.committed_clock_domain_id,
            "status": self.status,
            "native_v25_eligible": self.native_v25_eligible,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pNativeEpisodeManifest":
        if payload.get("schema") != V25P_NATIVE_EPISODE_MANIFEST_SCHEMA_V2:
            raise ValueError("Unsupported native v2.5-p episode manifest schema.")
        result = cls(
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            episode_identity_hash=payload.get("episode_identity_hash", ""),
            native_embodiment_binding_hash=payload.get("native_embodiment_binding_hash", ""),
            native_capture_manifest_hash=payload.get("native_capture_manifest_hash", ""),
            compatibility_projection_receipt_hashes=tuple(
                payload.get("compatibility_projection_receipt_hashes", ())
            ),
            v24_capture_commit_hash=payload.get("v24_capture_commit_hash"),
            committed_at=payload.get("committed_at", float("nan")),
            committed_clock_domain_id=payload.get("committed_clock_domain_id", ""),
            status=payload.get("status", ""),
            native_v25_eligible=payload.get("native_v25_eligible", False),
        )
        if payload.get("semantic_content_hash") not in {None, result.semantic_content_hash}:
            raise ValueError("Native episode manifest hash mismatch.")
        return result


__all__ = [name for name in globals() if name.startswith("V25P_")] + [
    "V25pCausalSample",
    "V25pComponentRecord",
    "V25pComponentStreamProfile",
    "V25pComponentStreamProfileLike",
    "V25pComponentStreamProfileV2",
    "V25pComponentStreamSpec",
    "V25pComponentStreamSpecLike",
    "V25pComponentStreamSpecV2",
    "V25pExternalPayloadReference",
    "V25pNativeCaptureManifest",
    "V25pNativeDatasetManifest",
    "V25pNativeEmbodimentBinding",
    "V25pNativeEpisodeManifest",
    "component_stream_profile_from_payload",
    "normalize_component_stream_stored_values",
]
