"""DexTeleop v2.2 policy-I/O dataset contract.

The on-disk directory organization remains LeRobot-v2.1-style.  What changes
in v2.2 is the low-dimensional policy contract: dual-arm poses are always
present, while each end effector is described independently by configuration.

This module intentionally has no dependency on runtime, hardware, or model
framework code.  It is the shared source of truth for collection, readers,
training manifests, replay, and deployment preflight checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np


DEXTELEOP_DATASET_SCHEMA_V22 = "dexteleop_dataset_v2.2"
POLICY_IO_LAYOUT_SCHEMA_V22 = "dexteleop_policy_io_layout_v2.2"
LEROBOT_V21_STYLE_FORMAT = "lerobot_v2.1_style"
POLICY_IO_LAYOUT_METADATA_KEY = "policy_io_layout"
POLICY_IO_LAYOUT_HASH_METADATA_KEY = "policy_io_layout_hash"
POLICY_IO_LAYOUT_SCHEMA_METADATA_KEY = "policy_io_layout_schema"

SIDES: tuple[str, str] = ("left", "right")
ARM_POSE_DIM = 7
ARM_QPOS_DIM = 6
ARM_ACTION_DIM = len(SIDES) * ARM_POSE_DIM
ARM_OBSERVATION_POSE_DIM = ARM_ACTION_DIM
ARM_QPOS_AUDIT_DIM = len(SIDES) * ARM_QPOS_DIM
INVALID_DISCRETE_TOKEN = -1

ChannelValueType = Literal["none", "continuous", "discrete"]
LayoutRole = Literal["action", "state"]


class DatasetLayoutError(ValueError):
    """Raised when a v2.2 layout or its data is not self-consistent."""


class DatasetLayoutBindingError(DatasetLayoutError):
    """Raised when dataset/checkpoint/runtime layout identities differ."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _normalize_side(side: Any) -> str:
    normalized = str(side).strip().lower()
    if normalized not in SIDES:
        raise DatasetLayoutError(f"Unsupported side {side!r}; expected one of {SIDES}.")
    return normalized


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetLayoutError(f"{name} must be a mapping.")
    return value


@dataclass(frozen=True)
class DiscreteCategory:
    """One stable integer category for a discrete policy-I/O token."""

    id: int
    label: str
    backend_value: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        category_id = int(self.id)
        if category_id < 0:
            raise DatasetLayoutError("Discrete category ids must be non-negative; -1 is reserved.")
        label = str(self.label).strip()
        if not label:
            raise DatasetLayoutError("Discrete category labels must be non-empty.")
        object.__setattr__(self, "id", category_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "metadata", dict(_jsonable(dict(self.metadata))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "label": self.label,
            "backend_value": _jsonable(self.backend_value),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiscreteCategory":
        return cls(
            id=int(payload["id"]),
            label=str(payload["label"]),
            backend_value=payload.get("backend_value"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class DiscreteTokenSpec:
    """Vocabulary for one discrete end-effector token component."""

    name: str
    categories: tuple[DiscreteCategory, ...]
    source_metadata_key: str | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise DatasetLayoutError("Discrete token names must be non-empty.")
        categories = tuple(self.categories)
        if not categories:
            raise DatasetLayoutError(f"Discrete token {name!r} must declare at least one category.")
        ids = [category.id for category in categories]
        labels = [category.label for category in categories]
        if len(set(ids)) != len(ids):
            raise DatasetLayoutError(f"Discrete token {name!r} has duplicate category ids.")
        if len(set(labels)) != len(labels):
            raise DatasetLayoutError(f"Discrete token {name!r} has duplicate category labels.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "categories", categories)
        source_key = None if self.source_metadata_key is None else str(self.source_metadata_key)
        object.__setattr__(self, "source_metadata_key", source_key or None)

    @property
    def category_ids(self) -> frozenset[int]:
        return frozenset(category.id for category in self.categories)

    def resolve(self, value: Any) -> int:
        if isinstance(value, str):
            label = value.strip()
            for category in self.categories:
                if category.label == label:
                    return int(category.id)
            raise DatasetLayoutError(
                f"Discrete token {self.name!r} does not define label {value!r}."
            )
        try:
            token = int(value)
        except (TypeError, ValueError) as exc:
            raise DatasetLayoutError(
                f"Discrete token {self.name!r} requires an integer id or declared label."
            ) from exc
        if token not in self.category_ids:
            raise DatasetLayoutError(
                f"Discrete token {self.name!r} does not define category id {token}."
            )
        return token

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "categories": [category.to_dict() for category in self.categories],
            "source_metadata_key": self.source_metadata_key,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiscreteTokenSpec":
        raw_categories = payload.get("categories", ())
        if not isinstance(raw_categories, Sequence) or isinstance(raw_categories, (str, bytes)):
            raise DatasetLayoutError("DiscreteTokenSpec.categories must be a sequence.")
        return cls(
            name=str(payload["name"]),
            categories=tuple(
                DiscreteCategory.from_dict(_as_mapping(item, name="discrete category"))
                for item in raw_categories
            ),
            source_metadata_key=payload.get("source_metadata_key"),
        )


@dataclass(frozen=True)
class EndEffectorChannelSpec:
    """One target or measured-state channel for an end effector.

    ``source_indices`` describes extraction from a legacy/raw numeric vector at
    collection time.  It is part of the semantic layout, so changing it changes
    the layout hash. ``legacy_q6_slot_indices`` is deliberately separate: it
    is only an explicit bridge mapping for old q6 deployment/replay paths. A
    collector must never infer a hardware command mapping from its raw capture
    source indices. Discrete channels instead use their token vocabularies;
    they can optionally name a command/state metadata key.
    """

    value_type: ChannelValueType = "none"
    dim: int = 0
    names: tuple[str, ...] = ()
    source_indices: tuple[int, ...] | None = None
    legacy_q6_slot_indices: tuple[int, ...] | None = None
    tokens: tuple[DiscreteTokenSpec, ...] = ()
    units: str | None = None
    normalization: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        value_type = str(self.value_type).strip().lower()
        if value_type not in {"none", "continuous", "discrete"}:
            raise DatasetLayoutError(
                "EndEffectorChannelSpec.value_type must be none, continuous, or discrete."
            )
        dim = int(self.dim)
        if dim < 0:
            raise DatasetLayoutError("EndEffectorChannelSpec.dim must be non-negative.")
        names = tuple(str(name).strip() for name in self.names)
        if any(not name for name in names):
            raise DatasetLayoutError("EndEffectorChannelSpec names must be non-empty.")
        source_indices = (
            None
            if self.source_indices is None
            else tuple(int(index) for index in self.source_indices)
        )
        legacy_q6_slot_indices = (
            None
            if self.legacy_q6_slot_indices is None
            else tuple(int(index) for index in self.legacy_q6_slot_indices)
        )
        tokens = tuple(self.tokens)
        if value_type == "none":
            if (
                dim != 0
                or names
                or tokens
                or source_indices not in (None, ())
                or legacy_q6_slot_indices not in (None, ())
            ):
                raise DatasetLayoutError("A none end-effector channel must have zero dimensions.")
        else:
            if dim <= 0:
                raise DatasetLayoutError("A configured end-effector channel must have dim > 0.")
            if names and len(names) != dim:
                raise DatasetLayoutError(
                    f"EndEffectorChannelSpec.names must have {dim} values, got {len(names)}."
                )
            if not names:
                if value_type == "discrete" and len(tokens) == dim:
                    names = tuple(token.name for token in tokens)
                else:
                    names = tuple(f"value_{index}" for index in range(dim))
            if source_indices is not None and len(source_indices) != dim:
                raise DatasetLayoutError(
                    f"EndEffectorChannelSpec.source_indices must have {dim} values."
                )
            if source_indices is not None and any(index < 0 for index in source_indices):
                raise DatasetLayoutError("EndEffectorChannelSpec.source_indices must be non-negative.")
            if legacy_q6_slot_indices is not None and len(legacy_q6_slot_indices) != dim:
                raise DatasetLayoutError(
                    f"EndEffectorChannelSpec.legacy_q6_slot_indices must have {dim} values."
                )
            if legacy_q6_slot_indices is not None and any(
                index < 0 or index >= 6 for index in legacy_q6_slot_indices
            ):
                raise DatasetLayoutError(
                    "EndEffectorChannelSpec.legacy_q6_slot_indices must be in [0, 5]."
                )
            if legacy_q6_slot_indices is not None and len(set(legacy_q6_slot_indices)) != len(
                legacy_q6_slot_indices
            ):
                raise DatasetLayoutError(
                    "EndEffectorChannelSpec.legacy_q6_slot_indices must not contain duplicates."
                )
            if value_type == "continuous" and tokens:
                raise DatasetLayoutError("Continuous end-effector channels cannot declare discrete tokens.")
            if value_type == "discrete" and len(tokens) != dim:
                raise DatasetLayoutError(
                    f"Discrete end-effector channels require {dim} token specs, got {len(tokens)}."
                )
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(self, "legacy_q6_slot_indices", legacy_q6_slot_indices)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "units", None if self.units is None else str(self.units))
        object.__setattr__(self, "normalization", dict(_jsonable(dict(self.normalization))))

    @classmethod
    def none(cls) -> "EndEffectorChannelSpec":
        return cls()

    @classmethod
    def continuous(
        cls,
        *,
        dim: int,
        names: Sequence[str] = (),
        source_indices: Sequence[int] | None = None,
        legacy_q6_slot_indices: Sequence[int] | None = None,
        units: str | None = None,
        normalization: Mapping[str, Any] | None = None,
    ) -> "EndEffectorChannelSpec":
        return cls(
            value_type="continuous",
            dim=int(dim),
            names=tuple(names),
            source_indices=None if source_indices is None else tuple(source_indices),
            legacy_q6_slot_indices=(
                None if legacy_q6_slot_indices is None else tuple(legacy_q6_slot_indices)
            ),
            units=units,
            normalization={} if normalization is None else dict(normalization),
        )

    @classmethod
    def discrete(
        cls,
        *,
        tokens: Sequence[DiscreteTokenSpec],
        source_indices: Sequence[int] | None = None,
        legacy_q6_slot_indices: Sequence[int] | None = None,
    ) -> "EndEffectorChannelSpec":
        token_tuple = tuple(tokens)
        return cls(
            value_type="discrete",
            dim=len(token_tuple),
            names=tuple(token.name for token in token_tuple),
            source_indices=None if source_indices is None else tuple(source_indices),
            legacy_q6_slot_indices=(
                None if legacy_q6_slot_indices is None else tuple(legacy_q6_slot_indices)
            ),
            tokens=token_tuple,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_type": self.value_type,
            "dim": int(self.dim),
            "names": list(self.names),
            "source_indices": None if self.source_indices is None else list(self.source_indices),
            "legacy_q6_slot_indices": (
                None
                if self.legacy_q6_slot_indices is None
                else list(self.legacy_q6_slot_indices)
            ),
            "tokens": [token.to_dict() for token in self.tokens],
            "units": self.units,
            "normalization": _jsonable(self.normalization),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EndEffectorChannelSpec":
        raw_tokens = payload.get("tokens", ())
        if not isinstance(raw_tokens, Sequence) or isinstance(raw_tokens, (str, bytes)):
            raise DatasetLayoutError("EndEffectorChannelSpec.tokens must be a sequence.")
        source_indices = payload.get("source_indices")
        if source_indices is not None and (
            not isinstance(source_indices, Sequence) or isinstance(source_indices, (str, bytes))
        ):
            raise DatasetLayoutError("EndEffectorChannelSpec.source_indices must be a sequence or null.")
        legacy_q6_slot_indices = payload.get("legacy_q6_slot_indices")
        if legacy_q6_slot_indices is not None and (
            not isinstance(legacy_q6_slot_indices, Sequence)
            or isinstance(legacy_q6_slot_indices, (str, bytes))
        ):
            raise DatasetLayoutError(
                "EndEffectorChannelSpec.legacy_q6_slot_indices must be a sequence or null."
            )
        names = payload.get("names", ())
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            raise DatasetLayoutError("EndEffectorChannelSpec.names must be a sequence.")
        return cls(
            value_type=str(payload.get("value_type", "none")),
            dim=int(payload.get("dim", 0)),
            names=tuple(str(name) for name in names),
            source_indices=None if source_indices is None else tuple(int(index) for index in source_indices),
            legacy_q6_slot_indices=(
                None
                if legacy_q6_slot_indices is None
                else tuple(int(index) for index in legacy_q6_slot_indices)
            ),
            tokens=tuple(
                DiscreteTokenSpec.from_dict(_as_mapping(item, name="discrete token"))
                for item in raw_tokens
            ),
            units=payload.get("units"),
            normalization=dict(payload.get("normalization", {})),
        )


@dataclass(frozen=True)
class EndEffectorSpec:
    """Per-side policy-I/O declaration for one configured end effector."""

    side: str
    adapter: str = "none_v1"
    target: EndEffectorChannelSpec = field(default_factory=EndEffectorChannelSpec.none)
    state: EndEffectorChannelSpec | None = None
    state_source_field: str = "joint_pos"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        side = _normalize_side(self.side)
        adapter = str(self.adapter).strip()
        if not adapter:
            raise DatasetLayoutError("EndEffectorSpec.adapter must be non-empty.")
        target = self.target
        state = target if self.state is None else self.state
        if target.value_type == "none" and state.value_type != "none":
            raise DatasetLayoutError("An absent end effector cannot declare a measured-state channel.")
        if target.value_type != "none" and state.value_type == "none":
            raise DatasetLayoutError("A configured end effector must declare a measured-state channel.")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "state", state)
        state_source_field = str(self.state_source_field).strip()
        object.__setattr__(self, "state_source_field", state_source_field or "joint_pos")
        object.__setattr__(self, "metadata", dict(_jsonable(dict(self.metadata))))

    @property
    def present(self) -> bool:
        return self.target.value_type != "none"

    @classmethod
    def none(cls, side: str) -> "EndEffectorSpec":
        return cls(side=side, adapter="none_v1")

    @classmethod
    def continuous(
        cls,
        side: str,
        *,
        adapter: str,
        target_dim: int,
        state_dim: int | None = None,
        target_names: Sequence[str] = (),
        state_names: Sequence[str] = (),
        target_source_indices: Sequence[int] | None = None,
        state_source_indices: Sequence[int] | None = None,
        target_legacy_q6_slot_indices: Sequence[int] | None = None,
        state_source_field: str = "joint_pos",
        units: str | None = None,
        normalization: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "EndEffectorSpec":
        target = EndEffectorChannelSpec.continuous(
            dim=target_dim,
            names=target_names,
            source_indices=target_source_indices,
            legacy_q6_slot_indices=target_legacy_q6_slot_indices,
            units=units,
            normalization=normalization,
        )
        resolved_state_dim = target_dim if state_dim is None else int(state_dim)
        state = EndEffectorChannelSpec.continuous(
            dim=resolved_state_dim,
            names=state_names or (target_names if resolved_state_dim == target_dim else ()),
            source_indices=(
                target_source_indices
                if state_source_indices is None and resolved_state_dim == target_dim
                else state_source_indices
            ),
            units=units,
            normalization=normalization,
        )
        return cls(
            side=side,
            adapter=adapter,
            target=target,
            state=state,
            state_source_field=state_source_field,
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def discrete(
        cls,
        side: str,
        *,
        adapter: str,
        target_tokens: Sequence[DiscreteTokenSpec],
        state_tokens: Sequence[DiscreteTokenSpec] | None = None,
        target_source_indices: Sequence[int] | None = None,
        state_source_indices: Sequence[int] | None = None,
        target_legacy_q6_slot_indices: Sequence[int] | None = None,
        state_source_field: str = "state_token",
        metadata: Mapping[str, Any] | None = None,
    ) -> "EndEffectorSpec":
        target = EndEffectorChannelSpec.discrete(
            tokens=target_tokens,
            source_indices=target_source_indices,
            legacy_q6_slot_indices=target_legacy_q6_slot_indices,
        )
        state = (
            target
            if state_tokens is None
            else EndEffectorChannelSpec.discrete(
                tokens=state_tokens,
                source_indices=state_source_indices,
            )
        )
        return cls(
            side=side,
            adapter=adapter,
            target=target,
            state=state,
            state_source_field=state_source_field,
            metadata={} if metadata is None else dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        assert self.state is not None
        return {
            "side": self.side,
            "adapter": self.adapter,
            "target": self.target.to_dict(),
            "state": self.state.to_dict(),
            "state_source_field": self.state_source_field,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EndEffectorSpec":
        target = EndEffectorChannelSpec.from_dict(_as_mapping(payload.get("target", {}), name="target"))
        state_payload = payload.get("state")
        state = (
            None
            if state_payload is None
            else EndEffectorChannelSpec.from_dict(_as_mapping(state_payload, name="state"))
        )
        return cls(
            side=str(payload["side"]),
            adapter=str(payload.get("adapter", "none_v1")),
            target=target,
            state=state,
            state_source_field=str(payload.get("state_source_field", "joint_pos")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class PolicyIOComponent:
    """One contiguous component of a typed dense policy vector."""

    name: str
    side: str | None
    value_type: ChannelValueType
    dim: int
    offset: int

    @property
    def stop(self) -> int:
        return int(self.offset) + int(self.dim)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "side": self.side,
            "value_type": self.value_type,
            "dim": int(self.dim),
            "offset": int(self.offset),
            "stop": int(self.stop),
        }


@dataclass(frozen=True)
class PackedPolicyIO:
    """Typed model-facing values produced by a :class:`PolicyIOLayout`."""

    continuous: np.ndarray
    discrete: np.ndarray

    def __post_init__(self) -> None:
        continuous = np.asarray(self.continuous, dtype=np.float32).reshape(-1)
        discrete = np.asarray(self.discrete, dtype=np.int64).reshape(-1)
        object.__setattr__(self, "continuous", continuous.copy())
        object.__setattr__(self, "discrete", discrete.copy())


def _none_specs() -> dict[str, EndEffectorSpec]:
    return {side: EndEffectorSpec.none(side) for side in SIDES}


@dataclass(frozen=True)
class PolicyIOLayout:
    """Immutable v2.2 model-I/O layout with a stable semantic hash.

    The float32 action/state vectors always begin with dual-arm poses.  Discrete
    end-effector values are never coerced into floats: they live in separate
    int64 token vectors and are passed to model adapters as tokens.
    """

    name: str = "dual_arm_end_effector"
    schema: str = POLICY_IO_LAYOUT_SCHEMA_V22
    arm_pose_representation: str = "xyz_quat_xyzw"
    arm_controlled_frame_by_side: Mapping[str, str] = field(
        default_factory=lambda: {"left": "tool_tcp", "right": "tool_tcp"}
    )
    arm_reference_frame_by_side: Mapping[str, str] = field(
        default_factory=lambda: {"left": "left_arm_base", "right": "right_arm_base"}
    )
    end_effectors: Mapping[str, EndEffectorSpec | Mapping[str, Any]] = field(
        default_factory=_none_specs
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise DatasetLayoutError("PolicyIOLayout.name must be non-empty.")
        if str(self.schema) != POLICY_IO_LAYOUT_SCHEMA_V22:
            raise DatasetLayoutError(
                f"Unsupported policy I/O layout schema {self.schema!r}; "
                f"expected {POLICY_IO_LAYOUT_SCHEMA_V22!r}."
            )
        if str(self.arm_pose_representation) != "xyz_quat_xyzw":
            raise DatasetLayoutError("v2.2 requires xyz_quat_xyzw arm poses.")
        controlled = {str(side): str(frame) for side, frame in dict(self.arm_controlled_frame_by_side).items()}
        reference = {str(side): str(frame) for side, frame in dict(self.arm_reference_frame_by_side).items()}
        if set(controlled) != set(SIDES) or any(not controlled[side] for side in SIDES):
            raise DatasetLayoutError("arm_controlled_frame_by_side must define non-empty left/right frames.")
        if set(reference) != set(SIDES) or any(not reference[side] for side in SIDES):
            raise DatasetLayoutError("arm_reference_frame_by_side must define non-empty left/right frames.")
        normalized_effectors: dict[str, EndEffectorSpec] = {}
        raw_effectors = dict(self.end_effectors)
        invalid = set(str(side) for side in raw_effectors) - set(SIDES)
        if invalid:
            raise DatasetLayoutError(f"Unsupported end-effector sides: {sorted(invalid)}.")
        for side in SIDES:
            raw = raw_effectors.get(side)
            if raw is None:
                spec = EndEffectorSpec.none(side)
            elif isinstance(raw, EndEffectorSpec):
                spec = raw
            elif isinstance(raw, Mapping):
                spec = EndEffectorSpec.from_dict(raw)
            else:
                raise DatasetLayoutError(f"end_effectors[{side!r}] must be an EndEffectorSpec or mapping.")
            if spec.side != side:
                raise DatasetLayoutError(
                    f"end_effectors key {side!r} disagrees with spec side {spec.side!r}."
                )
            normalized_effectors[side] = spec
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "schema", POLICY_IO_LAYOUT_SCHEMA_V22)
        object.__setattr__(self, "arm_pose_representation", "xyz_quat_xyzw")
        object.__setattr__(self, "arm_controlled_frame_by_side", controlled)
        object.__setattr__(self, "arm_reference_frame_by_side", reference)
        object.__setattr__(self, "end_effectors", normalized_effectors)
        object.__setattr__(self, "metadata", dict(_jsonable(dict(self.metadata))))

    @classmethod
    def legacy_v21_q6(cls) -> "PolicyIOLayout":
        """Compatibility layout for existing fixed 26D/38D q6 datasets.

        It describes the generic two-q6 legacy form.  A historical dataset that
        actually represented a q6 compatibility vector for a 1D gripper needs a
        declared gripper conversion profile before it can become a canonical
        v2.2 1D-end-effector dataset.
        """

        return cls(
            name="legacy_v21_dual_q6",
            end_effectors={
                side: EndEffectorSpec.continuous(
                    side,
                    adapter="legacy_q6_v1",
                    target_dim=6,
                    target_names=tuple(f"q{index}" for index in range(6)),
                    state_names=tuple(f"q{index}" for index in range(6)),
                    target_source_indices=tuple(range(6)),
                    state_source_indices=tuple(range(6)),
                    target_legacy_q6_slot_indices=tuple(range(6)),
                    metadata={"compatibility_source": "v2.1_fixed_q6"},
                )
                for side in SIDES
            },
            metadata={"compatibility_source": "v2.1_fixed_q6"},
        )

    @property
    def layout_hash(self) -> str:
        digest = hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def end_effector(self, side: str) -> EndEffectorSpec:
        return self.end_effectors[_normalize_side(side)]

    def _channel_for(self, *, role: LayoutRole, side: str) -> EndEffectorChannelSpec:
        spec = self.end_effector(side)
        assert spec.state is not None
        return spec.target if role == "action" else spec.state

    def channel(self, role: LayoutRole, side: str) -> EndEffectorChannelSpec:
        """Return the configured target/state channel for one side."""

        if role not in {"action", "state"}:
            raise DatasetLayoutError(f"Unsupported layout role {role!r}.")
        return self._channel_for(role=role, side=_normalize_side(side))

    def components(self, role: LayoutRole, value_type: ChannelValueType) -> tuple[PolicyIOComponent, ...]:
        if role not in {"action", "state"}:
            raise DatasetLayoutError(f"Unsupported layout role {role!r}.")
        if value_type not in {"continuous", "discrete"}:
            raise DatasetLayoutError("components() only supports continuous or discrete values.")
        items: list[PolicyIOComponent] = []
        offset = 0
        if value_type == "continuous":
            for side in SIDES:
                items.append(
                    PolicyIOComponent(
                        name=f"{side}.arm.pose",
                        side=side,
                        value_type="continuous",
                        dim=ARM_POSE_DIM,
                        offset=offset,
                    )
                )
                offset += ARM_POSE_DIM
        for side in SIDES:
            channel = self._channel_for(role=role, side=side)
            if channel.value_type != value_type:
                continue
            items.append(
                PolicyIOComponent(
                    name=f"{side}.end_effector.{'target' if role == 'action' else 'state'}",
                    side=side,
                    value_type=value_type,
                    dim=channel.dim,
                    offset=offset,
                )
            )
            offset += channel.dim
        return tuple(items)

    def continuous_dim(self, role: LayoutRole) -> int:
        components = self.components(role, "continuous")
        return 0 if not components else components[-1].stop

    def discrete_dim(self, role: LayoutRole) -> int:
        components = self.components(role, "discrete")
        return 0 if not components else components[-1].stop

    @property
    def action_dim(self) -> int:
        return self.continuous_dim("action")

    @property
    def state_dim(self) -> int:
        return self.continuous_dim("state")

    @property
    def action_discrete_dim(self) -> int:
        return self.discrete_dim("action")

    @property
    def state_discrete_dim(self) -> int:
        return self.discrete_dim("state")

    def model_dims(self, role: LayoutRole) -> dict[str, int]:
        return {
            "continuous": self.continuous_dim(role),
            "discrete": self.discrete_dim(role),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POLICY_IO_LAYOUT_SCHEMA_V22,
            "name": self.name,
            "arm": {
                "pose_representation": self.arm_pose_representation,
                "action_dim": ARM_ACTION_DIM,
                "observation_pose_dim": ARM_OBSERVATION_POSE_DIM,
                "qpos_audit_dim": ARM_QPOS_AUDIT_DIM,
                "controlled_frame_by_side": dict(self.arm_controlled_frame_by_side),
                "reference_frame_by_side": dict(self.arm_reference_frame_by_side),
            },
            "end_effectors": {
                side: self.end_effectors[side].to_dict() for side in SIDES
            },
            "model": {
                "action": {
                    "continuous_dim": self.action_dim,
                    "discrete_dim": self.action_discrete_dim,
                    "continuous_components": [
                        component.to_dict() for component in self.components("action", "continuous")
                    ],
                    "discrete_components": [
                        component.to_dict() for component in self.components("action", "discrete")
                    ],
                },
                "observation_state": {
                    "continuous_dim": self.state_dim,
                    "discrete_dim": self.state_discrete_dim,
                    "continuous_components": [
                        component.to_dict() for component in self.components("state", "continuous")
                    ],
                    "discrete_components": [
                        component.to_dict() for component in self.components("state", "discrete")
                    ],
                },
            },
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyIOLayout":
        schema = str(payload.get("schema", POLICY_IO_LAYOUT_SCHEMA_V22))
        arm_payload = _as_mapping(payload.get("arm", {}), name="layout.arm")
        raw_effectors = _as_mapping(payload.get("end_effectors", {}), name="layout.end_effectors")
        end_effectors = {
            _normalize_side(side): EndEffectorSpec.from_dict(
                _as_mapping(spec, name=f"layout.end_effectors[{side!r}]")
            )
            for side, spec in raw_effectors.items()
        }
        layout = cls(
            name=str(payload.get("name", "dual_arm_end_effector")),
            schema=schema,
            arm_pose_representation=str(arm_payload.get("pose_representation", "xyz_quat_xyzw")),
            arm_controlled_frame_by_side=dict(
                arm_payload.get(
                    "controlled_frame_by_side",
                    {"left": "tool_tcp", "right": "tool_tcp"},
                )
            ),
            arm_reference_frame_by_side=dict(
                arm_payload.get(
                    "reference_frame_by_side",
                    {"left": "left_arm_base", "right": "right_arm_base"},
                )
            ),
            end_effectors=end_effectors,
            metadata=dict(payload.get("metadata", {})),
        )
        declared_hash = payload.get("layout_hash")
        if declared_hash is not None and str(declared_hash) != layout.layout_hash:
            raise DatasetLayoutBindingError(
                "policy_io_layout.layout_hash does not match its canonical contents."
            )
        return layout

    def binding_metadata(self) -> dict[str, Any]:
        return {
            POLICY_IO_LAYOUT_SCHEMA_METADATA_KEY: POLICY_IO_LAYOUT_SCHEMA_V22,
            POLICY_IO_LAYOUT_METADATA_KEY: self.to_dict(),
            POLICY_IO_LAYOUT_HASH_METADATA_KEY: self.layout_hash,
        }

    def feature_metadata(self) -> dict[str, Any]:
        features: dict[str, Any] = {
            "action": {
                "shape": [self.action_dim],
                "dtype": "float32",
                "components": [
                    component.to_dict() for component in self.components("action", "continuous")
                ],
            },
            "observation.state": {
                "shape": [self.state_dim],
                "dtype": "float32",
                "components": [
                    component.to_dict() for component in self.components("state", "continuous")
                ],
            },
            "observation.arm.qpos": {
                "shape": [ARM_QPOS_AUDIT_DIM],
                "dtype": "float32",
                "role": "audit_only",
                "components": [
                    {"name": f"{side}.arm.qpos", "side": side, "dim": ARM_QPOS_DIM}
                    for side in SIDES
                ],
            },
            "observation.last_action": {
                "shape": [self.action_dim],
                "dtype": "float32",
                "components": [
                    component.to_dict() for component in self.components("action", "continuous")
                ],
            },
        }
        if self.action_discrete_dim:
            features["action.discrete"] = {
                "shape": [self.action_discrete_dim],
                "dtype": "int64",
                "invalid_value": INVALID_DISCRETE_TOKEN,
                "components": [
                    component.to_dict() for component in self.components("action", "discrete")
                ],
            }
            features["observation.last_action.discrete"] = {
                "shape": [self.action_discrete_dim],
                "dtype": "int64",
                "invalid_value": INVALID_DISCRETE_TOKEN,
                "components": [
                    component.to_dict() for component in self.components("action", "discrete")
                ],
            }
        if self.state_discrete_dim:
            features["observation.state.discrete"] = {
                "shape": [self.state_discrete_dim],
                "dtype": "int64",
                "invalid_value": INVALID_DISCRETE_TOKEN,
                "components": [
                    component.to_dict() for component in self.components("state", "discrete")
                ],
            }
        return features

    def parquet_fixed_list_columns(self) -> dict[str, tuple[str, int]]:
        """Return v2.2 fixed-list physical contracts for a parquet writer.

        The generic parquet helper accepts this compact ``dtype, length`` form
        without importing this module, preserving the dataset-contracts package
        boundary.  Legacy audit columns may still be present in a v2.2 row, but
        these fields are the canonical typed policy-I/O fields.
        """

        contracts: dict[str, tuple[str, int]] = {
            "action": ("float32", self.action_dim),
            "action.arm.pose": ("float32", ARM_ACTION_DIM),
            "observation.state": ("float32", self.state_dim),
            "observation.arm.pose": ("float32", ARM_OBSERVATION_POSE_DIM),
            "observation.arm.qpos": ("float32", ARM_QPOS_AUDIT_DIM),
            "observation.arm.qpos_valid": ("bool", ARM_QPOS_AUDIT_DIM),
            "observation.last_action": ("float32", self.action_dim),
        }
        if self.action_discrete_dim:
            contracts["action.discrete"] = ("int64", self.action_discrete_dim)
            contracts["observation.last_action.discrete"] = (
                "int64",
                self.action_discrete_dim,
            )
        if self.state_discrete_dim:
            contracts["observation.state.discrete"] = ("int64", self.state_discrete_dim)
        for role, prefix in (("action", "action"), ("state", "observation")):
            suffix = "target" if role == "action" else "state"
            for side in SIDES:
                channel = self._channel_for(role=role, side=side)
                if channel.value_type == "continuous":
                    contracts[f"{prefix}.end_effector.{side}.{suffix}"] = (
                        "float32",
                        channel.dim,
                    )
                elif channel.value_type == "discrete":
                    contracts[f"{prefix}.end_effector.{side}.{suffix}_token"] = (
                        "int64",
                        channel.dim,
                    )
        return contracts

    def _pack_arm_poses(self, arm_pose_by_side: Mapping[str, Any] | None) -> np.ndarray:
        values = np.full((ARM_ACTION_DIM,), np.nan, dtype=np.float32)
        source = {} if arm_pose_by_side is None else dict(arm_pose_by_side)
        invalid = set(str(side) for side in source) - set(SIDES)
        if invalid:
            raise DatasetLayoutError(f"Unsupported arm pose sides: {sorted(invalid)}.")
        for index, side in enumerate(SIDES):
            value = source.get(side)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            if array.shape != (ARM_POSE_DIM,):
                raise DatasetLayoutError(
                    f"{side} arm pose must have shape ({ARM_POSE_DIM},), got {array.shape}."
                )
            if not np.all(np.isfinite(array)):
                raise DatasetLayoutError(f"{side} arm pose must be finite when present.")
            start = index * ARM_POSE_DIM
            values[start : start + ARM_POSE_DIM] = array
        return values

    @staticmethod
    def _channel_value(
        channel: EndEffectorChannelSpec,
        value: Any,
        *,
        strict: bool,
    ) -> np.ndarray:
        if channel.value_type == "continuous":
            if value is None:
                if strict:
                    raise DatasetLayoutError("A required continuous end-effector value is missing.")
                return np.full((channel.dim,), np.nan, dtype=np.float32)
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            if array.shape != (channel.dim,):
                raise DatasetLayoutError(
                    f"Continuous end-effector value must have shape ({channel.dim},), got {array.shape}."
                )
            if strict and not np.all(np.isfinite(array)):
                raise DatasetLayoutError("A required continuous end-effector value is non-finite.")
            return array.astype(np.float32, copy=True)
        if channel.value_type == "discrete":
            if value is None:
                if strict:
                    raise DatasetLayoutError("A required discrete end-effector value is missing.")
                return np.full((channel.dim,), INVALID_DISCRETE_TOKEN, dtype=np.int64)
            if isinstance(value, Mapping):
                raw_values = [value.get(name) for name in channel.names]
            elif channel.dim == 1 and not isinstance(value, (list, tuple, np.ndarray)):
                raw_values = [value]
            else:
                raw_values = list(np.asarray(value, dtype=object).reshape(-1))
            if len(raw_values) != channel.dim:
                raise DatasetLayoutError(
                    f"Discrete end-effector value must have {channel.dim} tokens, got {len(raw_values)}."
                )
            tokens = np.empty((channel.dim,), dtype=np.int64)
            for index, raw_value in enumerate(raw_values):
                is_missing = raw_value is None
                if not is_missing and not isinstance(raw_value, str):
                    try:
                        is_missing = int(raw_value) == INVALID_DISCRETE_TOKEN
                    except (TypeError, ValueError):
                        is_missing = False
                if is_missing:
                    if strict:
                        raise DatasetLayoutError("A required discrete end-effector token is missing.")
                    tokens[index] = INVALID_DISCRETE_TOKEN
                    continue
                tokens[index] = channel.tokens[index].resolve(raw_value)
            return tokens
        if value is not None:
            raise DatasetLayoutError("An absent end effector cannot receive a value.")
        return np.zeros((0,), dtype=np.float32)

    def pack(
        self,
        role: LayoutRole,
        *,
        arm_pose_by_side: Mapping[str, Any] | None,
        end_effector_by_side: Mapping[str, Any] | None = None,
        strict: bool = False,
    ) -> PackedPolicyIO:
        if role not in {"action", "state"}:
            raise DatasetLayoutError(f"Unsupported layout role {role!r}.")
        arm_values = self._pack_arm_poses(arm_pose_by_side)
        continuous = np.full((self.continuous_dim(role),), np.nan, dtype=np.float32)
        continuous[:ARM_ACTION_DIM] = arm_values
        discrete = np.full((self.discrete_dim(role),), INVALID_DISCRETE_TOKEN, dtype=np.int64)
        sources = {} if end_effector_by_side is None else dict(end_effector_by_side)
        invalid = set(str(side) for side in sources) - set(SIDES)
        if invalid:
            raise DatasetLayoutError(f"Unsupported end-effector value sides: {sorted(invalid)}.")
        for value_type, vector in (("continuous", continuous), ("discrete", discrete)):
            for component in self.components(role, value_type):
                if component.name.endswith(".arm.pose"):
                    continue
                assert component.side is not None
                channel = self._channel_for(role=role, side=component.side)
                value = sources.get(component.side)
                try:
                    packed = self._channel_value(channel, value, strict=strict)
                except DatasetLayoutError as exc:
                    raise DatasetLayoutError(f"{component.name}: {exc}") from exc
                vector[component.offset : component.stop] = packed
        return PackedPolicyIO(continuous=continuous, discrete=discrete)

    def pack_action(
        self,
        *,
        arm_pose_by_side: Mapping[str, Any] | None,
        end_effector_by_side: Mapping[str, Any] | None = None,
        strict: bool = False,
    ) -> PackedPolicyIO:
        return self.pack(
            "action",
            arm_pose_by_side=arm_pose_by_side,
            end_effector_by_side=end_effector_by_side,
            strict=strict,
        )

    def pack_observation_state(
        self,
        *,
        arm_pose_by_side: Mapping[str, Any] | None,
        end_effector_by_side: Mapping[str, Any] | None = None,
        strict: bool = False,
    ) -> PackedPolicyIO:
        return self.pack(
            "state",
            arm_pose_by_side=arm_pose_by_side,
            end_effector_by_side=end_effector_by_side,
            strict=strict,
        )

    def pack_arm_qpos_audit(self, qpos_by_side: Mapping[str, Any] | None) -> np.ndarray:
        values = np.full((ARM_QPOS_AUDIT_DIM,), np.nan, dtype=np.float32)
        source = {} if qpos_by_side is None else dict(qpos_by_side)
        invalid = set(str(side) for side in source) - set(SIDES)
        if invalid:
            raise DatasetLayoutError(f"Unsupported arm qpos sides: {sorted(invalid)}.")
        for index, side in enumerate(SIDES):
            value = source.get(side)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            if array.shape != (ARM_QPOS_DIM,):
                raise DatasetLayoutError(
                    f"{side} arm qpos must have shape ({ARM_QPOS_DIM},), got {array.shape}."
                )
            if not np.all(np.isfinite(array)):
                raise DatasetLayoutError(f"{side} arm qpos must be finite when present.")
            start = index * ARM_QPOS_DIM
            values[start : start + ARM_QPOS_DIM] = array
        return values

    def unpack(
        self,
        role: LayoutRole,
        *,
        continuous: Any,
        discrete: Any | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        expected_continuous = self.continuous_dim(role)
        expected_discrete = self.discrete_dim(role)
        continuous_values = np.asarray(continuous, dtype=np.float32).reshape(-1)
        if continuous_values.shape != (expected_continuous,):
            raise DatasetLayoutError(
                f"{role} continuous vector must have shape ({expected_continuous},), "
                f"got {continuous_values.shape}."
            )
        if discrete is None:
            discrete_values = np.full((expected_discrete,), INVALID_DISCRETE_TOKEN, dtype=np.int64)
        else:
            discrete_values = np.asarray(discrete, dtype=np.int64).reshape(-1)
        if discrete_values.shape != (expected_discrete,):
            raise DatasetLayoutError(
                f"{role} discrete vector must have shape ({expected_discrete},), "
                f"got {discrete_values.shape}."
            )
        arm_pose_by_side: dict[str, np.ndarray] = {}
        end_effector_by_side: dict[str, np.ndarray] = {}
        for index, side in enumerate(SIDES):
            start = index * ARM_POSE_DIM
            arm_pose_by_side[side] = continuous_values[start : start + ARM_POSE_DIM].copy()
        for value_type, vector in (("continuous", continuous_values), ("discrete", discrete_values)):
            for component in self.components(role, value_type):
                if component.name.endswith(".arm.pose"):
                    continue
                assert component.side is not None
                end_effector_by_side[component.side] = vector[component.offset : component.stop].copy()
        return {
            "arm_pose_by_side": arm_pose_by_side,
            "end_effector_by_side": end_effector_by_side,
        }


def policy_io_layout_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    allow_legacy_v21: bool = True,
) -> PolicyIOLayout:
    payload = {} if metadata is None else dict(metadata)
    raw_layout = payload.get(POLICY_IO_LAYOUT_METADATA_KEY)
    if raw_layout is None:
        if not allow_legacy_v21:
            raise DatasetLayoutBindingError("Dataset/checkpoint is missing policy_io_layout metadata.")
        return PolicyIOLayout.legacy_v21_q6()
    layout = PolicyIOLayout.from_dict(_as_mapping(raw_layout, name=POLICY_IO_LAYOUT_METADATA_KEY))
    declared_schema = payload.get(POLICY_IO_LAYOUT_SCHEMA_METADATA_KEY)
    if declared_schema is not None and str(declared_schema) != POLICY_IO_LAYOUT_SCHEMA_V22:
        raise DatasetLayoutBindingError(
            f"Unsupported declared policy I/O schema {declared_schema!r}."
        )
    declared_hash = payload.get(POLICY_IO_LAYOUT_HASH_METADATA_KEY)
    if declared_hash is not None and str(declared_hash) != layout.layout_hash:
        raise DatasetLayoutBindingError(
            "Declared policy_io_layout_hash does not match policy_io_layout contents."
        )
    return layout


def bind_policy_io_layout(
    metadata: Mapping[str, Any] | None,
    layout: PolicyIOLayout,
) -> dict[str, Any]:
    bound = {} if metadata is None else dict(_jsonable(dict(metadata)))
    bound.update(layout.binding_metadata())
    return bound


def validate_policy_io_layout_binding(
    expected: PolicyIOLayout | str,
    metadata: Mapping[str, Any] | None,
    *,
    context: str,
    allow_legacy_v21: bool = False,
) -> PolicyIOLayout:
    actual = policy_io_layout_from_metadata(metadata, allow_legacy_v21=allow_legacy_v21)
    expected_hash = expected.layout_hash if isinstance(expected, PolicyIOLayout) else str(expected)
    if actual.layout_hash != expected_hash:
        raise DatasetLayoutBindingError(
            f"{context} policy I/O layout mismatch: expected {expected_hash}, "
            f"got {actual.layout_hash}."
        )
    return actual


__all__ = [
    "ARM_ACTION_DIM",
    "ARM_OBSERVATION_POSE_DIM",
    "ARM_POSE_DIM",
    "ARM_QPOS_AUDIT_DIM",
    "ARM_QPOS_DIM",
    "ChannelValueType",
    "DEXTELEOP_DATASET_SCHEMA_V22",
    "DatasetLayoutBindingError",
    "DatasetLayoutError",
    "DiscreteCategory",
    "DiscreteTokenSpec",
    "EndEffectorChannelSpec",
    "EndEffectorSpec",
    "INVALID_DISCRETE_TOKEN",
    "LEROBOT_V21_STYLE_FORMAT",
    "POLICY_IO_LAYOUT_HASH_METADATA_KEY",
    "POLICY_IO_LAYOUT_METADATA_KEY",
    "POLICY_IO_LAYOUT_SCHEMA_METADATA_KEY",
    "POLICY_IO_LAYOUT_SCHEMA_V22",
    "PackedPolicyIO",
    "PolicyIOComponent",
    "PolicyIOLayout",
    "SIDES",
    "bind_policy_io_layout",
    "policy_io_layout_from_metadata",
    "validate_policy_io_layout_binding",
]
