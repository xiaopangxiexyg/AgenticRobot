"""Pure, serialization-stable component action schema contracts.

These definitions are shared by live control and dataset projections.  They
own no source, timing, authority, target capability, execution, or episode
lifecycle semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .canonical_encoding import canonical_content_hash


ACTION_PAYLOAD_SCHEMA_HASH_DOMAIN_V1 = "dexteleop.action_payload_schema.v1"
COMPONENT_ACTION_SCHEMA_HASH_DOMAIN_V1 = "dexteleop.component_action_schema.v1"
ACTION_COMPONENT_CATALOG_HASH_DOMAIN_V2 = "dexteleop.action_component_catalog.v2"

STANDARD_REPRESENTATION_FAMILIES_V1 = frozenset(
    {
        "cartesian_pose",
        "joint_position",
        "spatial_twist",
        "joint_velocity",
        "spatial_wrench",
        "joint_effort",
        "continuous_native",
        "discrete_mode",
    }
)
STANDARD_VALUE_SEMANTICS_V1 = frozenset(
    {
        "absolute_target",
        "relative_delta",
        "rate_target",
        "effort_target",
        "native_target",
        "mode",
    }
)
SUPPORTED_COMPONENT_DTYPES_V1 = frozenset(
    {"float32", "float64", "int64", "bool"}
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _optional_identifier(value: str | None, *, name: str) -> str | None:
    return None if value is None else _required(value, name=name)


def _hash(value: object, *, name: str) -> str:
    normalized = _required(value, name=name).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return normalized


def _identifiers(
    values: tuple[str, ...] | list[str],
    *,
    name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    normalized = tuple(_required(value, name=name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must be non-empty.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique.")
    return normalized


def _dtype_name(value: object) -> str:
    try:
        dtype = np.dtype(value)
    except TypeError as exc:
        raise ValueError(f"Unsupported action payload dtype {value!r}.") from exc
    if dtype.name not in SUPPORTED_COMPONENT_DTYPES_V1:
        raise ValueError(
            "Action payload dtype must be one of "
            f"{sorted(SUPPORTED_COMPONENT_DTYPES_V1)}, got {dtype.name!r}."
        )
    return dtype.name


def _freeze_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Object arrays cannot cross an immutable schema boundary.")
    contiguous = np.ascontiguousarray(array)
    immutable_bytes = contiguous.tobytes(order="C")
    return np.frombuffer(immutable_bytes, dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _freeze_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (str, bytes, int, float, bool, type(None), np.generic)):
        return value.item() if isinstance(value, np.generic) else value
    raise TypeError(
        "Unsupported value at immutable schema boundary: "
        f"{type(value).__name__}."
    )


def _freeze_semantic_mapping(
    values: Mapping[str, Any],
    *,
    name: str,
) -> Mapping[str, Any]:
    try:
        return _deep_freeze(dict(values))
    except TypeError as exc:
        raise TypeError(f"{name} contains a non-canonical value: {exc}") from exc


@dataclass(frozen=True)
class ActionPayloadSchema:
    """Exact numerical and coordinate contract for one action component."""

    schema_id: str
    schema_version: str
    representation_family: str
    dtype: str
    shape: tuple[int, ...]
    coordinate_names: tuple[str, ...]
    units: tuple[str, ...]
    value_semantics: str
    reference_frame: str | None = None
    controlled_frame: str | None = None
    expressed_in_frame: str | None = None
    semantic_parameters: Mapping[str, Any] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        schema_id = _required(self.schema_id, name="schema_id")
        schema_version = _required(self.schema_version, name="schema_version")
        family = _required(
            self.representation_family,
            name="representation_family",
        )
        if family not in STANDARD_REPRESENTATION_FAMILIES_V1 and "." not in family:
            raise ValueError(
                "Non-standard representation families must use a namespaced "
                "identifier containing '.'."
            )
        dtype = _dtype_name(self.dtype)
        shape = tuple(int(value) for value in self.shape)
        if len(shape) != 1 or shape[0] <= 0:
            raise ValueError(
                "Component action v1 payloads must be one-dimensional and non-empty."
            )
        coordinate_names = _identifiers(
            tuple(self.coordinate_names),
            name="coordinate_name",
            allow_empty=False,
        )
        units = tuple(_required(value, name="unit") for value in self.units)
        if len(coordinate_names) != shape[0] or len(units) != shape[0]:
            raise ValueError(
                "coordinate_names and units must match the payload dimension."
            )
        value_semantics = _required(
            self.value_semantics,
            name="value_semantics",
        )
        if value_semantics not in STANDARD_VALUE_SEMANTICS_V1 and (
            "." not in value_semantics
        ):
            raise ValueError(
                "Non-standard value semantics must use a namespaced identifier."
            )
        reference_frame = _optional_identifier(
            self.reference_frame,
            name="reference_frame",
        )
        controlled_frame = _optional_identifier(
            self.controlled_frame,
            name="controlled_frame",
        )
        expressed_in_frame = _optional_identifier(
            self.expressed_in_frame,
            name="expressed_in_frame",
        )
        parameters = _freeze_semantic_mapping(
            self.semantic_parameters,
            name="semantic_parameters",
        )
        self._validate_standard_family(
            family=family,
            dtype=dtype,
            shape=shape,
            coordinate_names=coordinate_names,
            value_semantics=value_semantics,
            reference_frame=reference_frame,
            controlled_frame=controlled_frame,
            expressed_in_frame=expressed_in_frame,
        )
        payload = {
            "schema_id": schema_id,
            "schema_version": schema_version,
            "representation_family": family,
            "dtype": dtype,
            "shape": shape,
            "coordinate_names": coordinate_names,
            "units": units,
            "value_semantics": value_semantics,
            "reference_frame": reference_frame,
            "controlled_frame": controlled_frame,
            "expressed_in_frame": expressed_in_frame,
            "semantic_parameters": parameters,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=ACTION_PAYLOAD_SCHEMA_HASH_DOMAIN_V1,
            ),
        )

    @staticmethod
    def _validate_standard_family(
        *,
        family: str,
        dtype: str,
        shape: tuple[int, ...],
        coordinate_names: tuple[str, ...],
        value_semantics: str,
        reference_frame: str | None,
        controlled_frame: str | None,
        expressed_in_frame: str | None,
    ) -> None:
        if family == "cartesian_pose":
            if dtype not in {"float32", "float64"}:
                raise ValueError("Cartesian pose payloads must use a float dtype.")
            if shape != (7,) or coordinate_names != (
                "x",
                "y",
                "z",
                "qx",
                "qy",
                "qz",
                "qw",
            ):
                raise ValueError(
                    "Cartesian pose v1 requires xyz+quat_xyzw coordinates."
                )
            if value_semantics not in {"absolute_target", "relative_delta"}:
                raise ValueError(
                    "Cartesian pose must be an absolute target or a relative delta."
                )
            if reference_frame is None or controlled_frame is None:
                raise ValueError(
                    "Cartesian pose requires reference_frame and controlled_frame."
                )
        elif family in {"spatial_twist", "spatial_wrench"}:
            expected = (
                ("vx", "vy", "vz", "wx", "wy", "wz")
                if family == "spatial_twist"
                else ("fx", "fy", "fz", "tx", "ty", "tz")
            )
            if dtype not in {"float32", "float64"} or (
                shape != (6,) or coordinate_names != expected
            ):
                raise ValueError(
                    f"{family} requires a six-dimensional float payload."
                )
            expected_semantics = (
                "rate_target" if family == "spatial_twist" else "effort_target"
            )
            if value_semantics != expected_semantics:
                raise ValueError(
                    f"{family} requires {expected_semantics!r} semantics."
                )
            if expressed_in_frame is None or controlled_frame is None:
                raise ValueError(
                    f"{family} requires expressed_in_frame and controlled_frame."
                )
        elif family in {
            "joint_position",
            "joint_velocity",
            "joint_effort",
            "continuous_native",
        }:
            if dtype not in {"float32", "float64"}:
                raise ValueError(f"{family} payloads must use a float dtype.")
            expected = {
                "joint_position": {"absolute_target", "relative_delta"},
                "joint_velocity": "rate_target",
                "joint_effort": "effort_target",
                "continuous_native": "native_target",
            }[family]
            accepted = expected if isinstance(expected, set) else {expected}
            if value_semantics not in accepted:
                raise ValueError(
                    f"{family} requires one of {sorted(accepted)!r} semantics."
                )
        elif family == "discrete_mode":
            if dtype != "int64" or value_semantics != "mode":
                raise ValueError(
                    "discrete_mode requires int64 values with mode semantics."
                )

    def validate_values(self, values: Any) -> np.ndarray:
        """Return one detached immutable array matching this exact schema."""

        raw = np.asarray(values)
        dtype = np.dtype(self.dtype)
        if dtype.kind in {"i", "u"}:
            try:
                normalized = raw.astype(dtype)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Integer action payload contains invalid values."
                ) from exc
            if not np.array_equal(raw, normalized):
                raise ValueError(
                    "Integer action payload must contain exact integer values."
                )
        elif dtype.kind == "b":
            if not np.all(np.isin(raw, (False, True, 0, 1))):
                raise ValueError("Boolean action payload must contain bool/0/1.")
            normalized = raw.astype(dtype)
        else:
            try:
                normalized = raw.astype(dtype)
            except (TypeError, ValueError) as exc:
                raise ValueError("Float action payload contains invalid values.") from exc
            if not np.all(np.isfinite(normalized)):
                raise ValueError(
                    "Canonical component action payloads must be fully finite."
                )
        normalized = np.asarray(normalized).reshape(-1)
        if normalized.shape != self.shape:
            raise ValueError(
                f"Action payload must have shape {self.shape}, got {normalized.shape}."
            )
        if self.representation_family == "cartesian_pose":
            quaternion_norm = float(np.linalg.norm(normalized[3:7]))
            if not math.isfinite(quaternion_norm) or not math.isclose(
                quaternion_norm,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-5,
            ):
                raise ValueError("Cartesian pose quaternion must be normalized.")
        return _freeze_array(np.ascontiguousarray(normalized))

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "representation_family": self.representation_family,
            "dtype": self.dtype,
            "shape": self.shape,
            "coordinate_names": self.coordinate_names,
            "units": self.units,
            "value_semantics": self.value_semantics,
            "reference_frame": self.reference_frame,
            "controlled_frame": self.controlled_frame,
            "expressed_in_frame": self.expressed_in_frame,
            "semantic_parameters": self.semantic_parameters,
            "semantic_content_hash": self.semantic_content_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ActionPayloadSchema":
        result = cls(
            schema_id=payload.get("schema_id", ""),
            schema_version=payload.get("schema_version", ""),
            representation_family=payload.get("representation_family", ""),
            dtype=payload.get("dtype", ""),
            shape=tuple(payload.get("shape", ())),
            coordinate_names=tuple(payload.get("coordinate_names", ())),
            units=tuple(payload.get("units", ())),
            value_semantics=payload.get("value_semantics", ""),
            reference_frame=payload.get("reference_frame"),
            controlled_frame=payload.get("controlled_frame"),
            expressed_in_frame=payload.get("expressed_in_frame"),
            semantic_parameters=dict(payload.get("semantic_parameters", {})),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ValueError("Action payload schema semantic hash mismatch.")
        return result


@dataclass(frozen=True)
class ComponentActionSpaceSchema:
    """Pure definition of one component-keyed action space.

    Runtime component catalog objects remain owned by ``robot_motion``.  This
    definition binds their immutable catalog hash to the exact payload schema
    hash used by every component.
    """

    schema_id: str
    schema_version: str
    component_catalog_hash: str
    payload_schema_hash_by_component: Mapping[str, str]
    semantic_parameters: Mapping[str, Any] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        schema_id = _required(self.schema_id, name="schema_id")
        schema_version = _required(self.schema_version, name="schema_version")
        catalog_hash = _hash(
            self.component_catalog_hash,
            name="component_catalog_hash",
        )
        payload_schemas = {
            _required(component_id, name="component_id"): _hash(
                schema_hash,
                name="payload_schema_hash",
            )
            for component_id, schema_hash in dict(
                self.payload_schema_hash_by_component
            ).items()
        }
        if not payload_schemas:
            raise ValueError(
                "payload_schema_hash_by_component must contain at least one component."
            )
        parameters = _freeze_semantic_mapping(
            self.semantic_parameters,
            name="semantic_parameters",
        )
        semantic_payload = {
            "schema_id": schema_id,
            "schema_version": schema_version,
            "component_catalog_hash": catalog_hash,
            "payload_schema_hash_by_component": payload_schemas,
            "semantic_parameters": parameters,
        }
        object.__setattr__(self, "schema_id", schema_id)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "component_catalog_hash", catalog_hash)
        object.__setattr__(
            self,
            "payload_schema_hash_by_component",
            MappingProxyType(payload_schemas),
        )
        object.__setattr__(self, "semantic_parameters", parameters)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                semantic_payload,
                domain=COMPONENT_ACTION_SCHEMA_HASH_DOMAIN_V1,
            ),
        )

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(self.payload_schema_hash_by_component)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "component_catalog_hash": self.component_catalog_hash,
            "payload_schema_hash_by_component": dict(
                self.payload_schema_hash_by_component
            ),
            "semantic_parameters": self.semantic_parameters,
            "semantic_content_hash": self.semantic_content_hash,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ComponentActionSpaceSchema":
        result = cls(
            schema_id=payload.get("schema_id", ""),
            schema_version=payload.get("schema_version", ""),
            component_catalog_hash=payload.get("component_catalog_hash", ""),
            payload_schema_hash_by_component=dict(
                payload.get("payload_schema_hash_by_component", {})
            ),
            semantic_parameters=dict(payload.get("semantic_parameters", {})),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ValueError("Component action-space schema semantic hash mismatch.")
        return result


__all__ = [
    "ACTION_COMPONENT_CATALOG_HASH_DOMAIN_V2",
    "ACTION_PAYLOAD_SCHEMA_HASH_DOMAIN_V1",
    "COMPONENT_ACTION_SCHEMA_HASH_DOMAIN_V1",
    "STANDARD_REPRESENTATION_FAMILIES_V1",
    "STANDARD_VALUE_SEMANTICS_V1",
    "SUPPORTED_COMPONENT_DTYPES_V1",
    "ActionPayloadSchema",
    "ComponentActionSpaceSchema",
]
