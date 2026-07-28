"""Device-neutral human pose residual adaptation.

This module converts immutable source-native pose samples into canonical,
state-valued ``relative_delta`` component actions. It owns neither a device nor
composition/execution authority. Anchor capture and recenter generation are
explicit inputs, so repeated evaluation cannot integrate a residual twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    ActionCoordinateBinding,
    ComponentAction,
    ComponentActionSchema,
    bind_action_payload_coordinates,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    ResolvedPoseFrameTransform,
)
from robot_motion.runtime.immutable import deep_freeze
from robot_motion.transforms import parse_pose_matrix, validate_pose_matrix


HUMAN_RESIDUAL_POSE_ALGORITHM_DEVELOPMENT = (
    "human_residual_pose.reference_translation_local_rotation.development"
)
HUMAN_POSE_SAMPLE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_pose_sample.development"
)
HUMAN_RESIDUAL_ANCHOR_ENTRY_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_residual_anchor_entry.development"
)
HUMAN_RESIDUAL_ANCHOR_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_residual_anchor.development"
)
HUMAN_RESIDUAL_ADAPTER_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_residual_adapter.development"
)
HUMAN_RESIDUAL_ADAPTATION_RECEIPT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_residual_adaptation_receipt.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _pose(value: object, *, name: str) -> np.ndarray:
    try:
        matrix = parse_pose_matrix(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a supported rigid pose.") from exc
    valid, reason = validate_pose_matrix(matrix)
    if not valid:
        raise ValueError(f"{name} is invalid: {reason}")
    return deep_freeze(np.asarray(matrix, dtype=np.float64))


def _gain(value: object, *, component_id: str) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"translation gain for {component_id!r} must be numeric."
        ) from exc
    if raw.shape == ():
        raw = np.repeat(raw.reshape(1), 3)
    raw = raw.reshape(-1)
    if raw.shape != (3,) or not np.all(np.isfinite(raw)):
        raise ValueError(
            f"translation gain for {component_id!r} must be one finite "
            "scalar or three finite values."
        )
    return deep_freeze(raw)


def _delta_pose_vector(
    translation: np.ndarray,
    rotation: np.ndarray,
    *,
    dtype: str,
) -> np.ndarray:
    quaternion = np.asarray(
        Rotation.from_matrix(rotation).as_quat(),
        dtype=np.float64,
    )
    if quaternion[3] < 0.0 or (
        abs(float(quaternion[3])) <= 1e-12
        and next(
            (
                float(item)
                for item in quaternion[:3]
                if abs(float(item)) > 1e-12
            ),
            0.0,
        )
        < 0.0
    ):
        quaternion = -quaternion
    return np.concatenate((translation, quaternion)).astype(dtype)


@dataclass(frozen=True)
class HumanPoseSample:
    """One source-native pose sample with explicit frame and binding identity."""

    component_id: str
    sample_id: str
    measured_at: float
    measured_clock_domain_id: str
    source_reference_frame_id: str
    source_controlled_frame_id: str
    source_coordinate_binding_id: str
    source_coordinate_binding_hash: str
    pose_source_reference_source_controlled: np.ndarray
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "component_id",
                "sample_id",
                "measured_clock_domain_id",
                "source_reference_frame_id",
                "source_controlled_frame_id",
                "source_coordinate_binding_id",
                "source_coordinate_binding_hash",
            )
        }
        measured_at = _finite(self.measured_at, name="measured_at")
        pose = _pose(
            self.pose_source_reference_source_controlled,
            name="pose_source_reference_source_controlled",
        )
        payload = {
            **values,
            "measured_at": measured_at,
            "pose_source_reference_source_controlled": pose,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "measured_at", measured_at)
        object.__setattr__(
            self,
            "pose_source_reference_source_controlled",
            pose,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=HUMAN_POSE_SAMPLE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "sample_id": self.sample_id,
            "measured_at": self.measured_at,
            "measured_clock_domain_id": self.measured_clock_domain_id,
            "source_reference_frame_id": self.source_reference_frame_id,
            "source_controlled_frame_id": self.source_controlled_frame_id,
            "source_coordinate_binding_id": self.source_coordinate_binding_id,
            "source_coordinate_binding_hash": (
                self.source_coordinate_binding_hash
            ),
            "pose_source_reference_source_controlled": (
                self.pose_source_reference_source_controlled
            ),
            "semantic_content_hash": self.semantic_content_hash,
        }

    @classmethod
    def from_component_action(
        cls,
        action: ComponentAction,
        component_id: str,
        *,
        sample_id: str,
        measured_at: float,
        measured_clock_domain_id: str,
        source_coordinate_binding_id: str,
        source_coordinate_binding_hash: str,
        concrete_reference_frame_id: str | None = None,
        concrete_controlled_frame_id: str | None = None,
    ) -> "HumanPoseSample":
        """Select one already-canonicalized absolute pose component.

        This is the lightweight seam for teleop/replay/model sources that
        already emit a typed absolute ``ComponentAction``.  Device-specific
        calibration and retargeting stay upstream; the exact binding identity
        supplied here prevents those values from being silently reinterpreted.
        """

        if not isinstance(action, ComponentAction):
            raise TypeError("action must be ComponentAction.")
        component = _required(component_id, name="component_id")
        try:
            value = action.value(component)
        except KeyError as exc:
            raise ValueError(
                f"Component action does not contain {component!r}."
            ) from exc
        schema = value.payload_schema
        if (
            schema.representation_family != "cartesian_pose"
            or schema.value_semantics != "absolute_target"
        ):
            raise ValueError(
                "Human pose sample projection requires an absolute-target "
                "cartesian_pose component."
            )
        reference_frame = _required(
            (
                schema.reference_frame
                if concrete_reference_frame_id is None
                else concrete_reference_frame_id
            ),
            name="source_reference_frame_id",
        )
        controlled_frame = _required(
            (
                schema.controlled_frame
                if concrete_controlled_frame_id is None
                else concrete_controlled_frame_id
            ),
            name="source_controlled_frame_id",
        )
        values = np.asarray(value.values, dtype=np.float64).reshape(7)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = values[:3]
        pose[:3, :3] = Rotation.from_quat(values[3:7]).as_matrix()
        return cls(
            component_id=component,
            sample_id=sample_id,
            measured_at=measured_at,
            measured_clock_domain_id=measured_clock_domain_id,
            source_reference_frame_id=reference_frame,
            source_controlled_frame_id=controlled_frame,
            source_coordinate_binding_id=source_coordinate_binding_id,
            source_coordinate_binding_hash=source_coordinate_binding_hash,
            pose_source_reference_source_controlled=pose,
        )


@dataclass(frozen=True)
class HumanResidualPoseAnchorEntry:
    component_id: str
    source_sample_id: str
    source_sample_hash: str
    source_measured_at: float
    source_measured_clock_domain_id: str
    source_reference_frame_id: str
    source_controlled_frame_id: str
    source_coordinate_binding_id: str
    source_coordinate_binding_hash: str
    canonical_reference_frame_id: str
    canonical_controlled_frame_id: str
    pose_canonical_reference_canonical_controlled: np.ndarray
    pose_transform_resolution_id: str
    pose_transform_resolution_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "component_id",
                "source_sample_id",
                "source_sample_hash",
                "source_measured_clock_domain_id",
                "source_reference_frame_id",
                "source_controlled_frame_id",
                "source_coordinate_binding_id",
                "source_coordinate_binding_hash",
                "canonical_reference_frame_id",
                "canonical_controlled_frame_id",
                "pose_transform_resolution_id",
                "pose_transform_resolution_hash",
            )
        }
        measured_at = _finite(
            self.source_measured_at,
            name="source_measured_at",
        )
        pose = _pose(
            self.pose_canonical_reference_canonical_controlled,
            name="pose_canonical_reference_canonical_controlled",
        )
        payload = {
            **text,
            "source_measured_at": measured_at,
            "pose_canonical_reference_canonical_controlled": pose,
        }
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_measured_at", measured_at)
        object.__setattr__(
            self,
            "pose_canonical_reference_canonical_controlled",
            pose,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=HUMAN_RESIDUAL_ANCHOR_ENTRY_HASH_DOMAIN_DEVELOPMENT,
            ),
        )


@dataclass(frozen=True)
class HumanResidualPoseAnchor:
    """Explicit recenter anchor for one exact coordinate graph and component set."""

    anchor_generation: int
    captured_at: float
    captured_clock_domain_id: str
    frame_graph_hash: str
    entries: tuple[HumanResidualPoseAnchorEntry, ...]
    semantic_content_hash: str = field(init=False)
    anchor_id: str = field(init=False)
    entry_by_component: Mapping[str, HumanResidualPoseAnchorEntry] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        generation = int(self.anchor_generation)
        if generation < 0:
            raise ValueError("anchor_generation must be non-negative.")
        captured_at = _finite(self.captured_at, name="captured_at")
        clock = _required(
            self.captured_clock_domain_id,
            name="captured_clock_domain_id",
        )
        graph_hash = _required(self.frame_graph_hash, name="frame_graph_hash")
        entries = tuple(self.entries)
        if not entries or any(
            not isinstance(item, HumanResidualPoseAnchorEntry)
            for item in entries
        ):
            raise TypeError(
                "entries must contain at least one HumanResidualPoseAnchorEntry."
            )
        ordered = tuple(sorted(entries, key=lambda item: item.component_id))
        entry_by_component = {item.component_id: item for item in ordered}
        if len(entry_by_component) != len(ordered):
            raise ValueError("Human residual anchor component IDs must be unique.")
        for item in ordered:
            if (
                item.source_measured_clock_domain_id == clock
                and item.source_measured_at > captured_at
            ):
                raise ValueError(
                    "Human residual anchor capture predates its source sample."
                )
        payload = {
            "anchor_generation": generation,
            "captured_at": captured_at,
            "captured_clock_domain_id": clock,
            "frame_graph_hash": graph_hash,
            "entry_hash_by_component": {
                item.component_id: item.semantic_content_hash for item in ordered
            },
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=HUMAN_RESIDUAL_ANCHOR_HASH_DOMAIN_DEVELOPMENT,
        )
        object.__setattr__(self, "anchor_generation", generation)
        object.__setattr__(self, "captured_at", captured_at)
        object.__setattr__(self, "captured_clock_domain_id", clock)
        object.__setattr__(self, "frame_graph_hash", graph_hash)
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(
            self,
            "entry_by_component",
            MappingProxyType(entry_by_component),
        )
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "anchor_id",
            f"human-residual-anchor:{generation}:{semantic_hash}",
        )


@dataclass(frozen=True)
class HumanResidualPoseAdaptationReceipt:
    adapter_contract_hash: str
    anchor_id: str
    anchor_hash: str
    anchor_generation: int
    adapted_at: float
    adapted_clock_domain_id: str
    source_sample_hash_by_component: Mapping[str, str]
    pose_transform_resolution_hash_by_component: Mapping[str, str]
    action_coordinate_binding_hash_by_component: Mapping[str, str]
    translation_gain_by_component: Mapping[str, tuple[float, float, float]]
    output_action_schema_hash: str
    output_action_hash: str
    frame_graph_hash: str
    semantic_content_hash: str = field(init=False)
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "adapter_contract_hash": _required(
                self.adapter_contract_hash,
                name="adapter_contract_hash",
            ),
            "anchor_id": _required(self.anchor_id, name="anchor_id"),
            "anchor_hash": _required(self.anchor_hash, name="anchor_hash"),
            "anchor_generation": int(self.anchor_generation),
            "adapted_at": _finite(self.adapted_at, name="adapted_at"),
            "adapted_clock_domain_id": _required(
                self.adapted_clock_domain_id,
                name="adapted_clock_domain_id",
            ),
            "source_sample_hash_by_component": dict(
                self.source_sample_hash_by_component
            ),
            "pose_transform_resolution_hash_by_component": dict(
                self.pose_transform_resolution_hash_by_component
            ),
            "action_coordinate_binding_hash_by_component": dict(
                self.action_coordinate_binding_hash_by_component
            ),
            "translation_gain_by_component": {
                str(component_id): tuple(float(value) for value in gain)
                for component_id, gain in self.translation_gain_by_component.items()
            },
            "output_action_schema_hash": _required(
                self.output_action_schema_hash,
                name="output_action_schema_hash",
            ),
            "output_action_hash": _required(
                self.output_action_hash,
                name="output_action_hash",
            ),
            "frame_graph_hash": _required(
                self.frame_graph_hash,
                name="frame_graph_hash",
            ),
            "algorithm_id": HUMAN_RESIDUAL_POSE_ALGORITHM_DEVELOPMENT,
        }
        key_sets = {
            frozenset(payload[name])
            for name in (
                "source_sample_hash_by_component",
                "pose_transform_resolution_hash_by_component",
                "action_coordinate_binding_hash_by_component",
                "translation_gain_by_component",
            )
        }
        if len(key_sets) != 1 or not next(iter(key_sets), frozenset()):
            raise ValueError(
                "Human residual adaptation receipt component mappings must "
                "have the same non-empty key set."
            )
        semantic_hash = canonical_content_hash(
            payload,
            domain=HUMAN_RESIDUAL_ADAPTATION_RECEIPT_HASH_DOMAIN_DEVELOPMENT,
        )
        for name in (
            "source_sample_hash_by_component",
            "pose_transform_resolution_hash_by_component",
            "action_coordinate_binding_hash_by_component",
            "translation_gain_by_component",
        ):
            object.__setattr__(
                self,
                name,
                MappingProxyType(dict(payload[name])),
            )
        for name in (
            "adapter_contract_hash",
            "anchor_id",
            "anchor_hash",
            "anchor_generation",
            "adapted_at",
            "adapted_clock_domain_id",
            "output_action_schema_hash",
            "output_action_hash",
            "frame_graph_hash",
        ):
            object.__setattr__(self, name, payload[name])
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "receipt_id",
            f"human-residual-adaptation:{semantic_hash}",
        )


@dataclass(frozen=True)
class HumanResidualPoseAdaptationResult:
    relative_action: ComponentAction
    receipt: HumanResidualPoseAdaptationReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.relative_action, ComponentAction):
            raise TypeError("relative_action must be ComponentAction.")
        if not isinstance(self.receipt, HumanResidualPoseAdaptationReceipt):
            raise TypeError(
                "receipt must be HumanResidualPoseAdaptationReceipt."
            )
        if (
            self.relative_action.semantic_content_hash
            != self.receipt.output_action_hash
        ):
            raise ValueError(
                "Human residual result action and receipt hashes disagree."
            )


class HumanResidualPoseAdapter:
    """Stateless source adapter for world/reference translation + local rotation."""

    def __init__(
        self,
        *,
        output_action_schema: ComponentActionSchema,
        frame_graph: CoordinateFrameGraph,
        translation_gain_by_component: Mapping[str, object] | None = None,
        concrete_reference_frame_by_component: Mapping[str, str] | None = None,
        concrete_controlled_frame_by_component: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(output_action_schema, ComponentActionSchema):
            raise TypeError(
                "output_action_schema must be ComponentActionSchema."
            )
        if not isinstance(frame_graph, CoordinateFrameGraph):
            raise TypeError("frame_graph must be CoordinateFrameGraph.")
        pose_components = tuple(
            component_id
            for component_id in output_action_schema.component_ids
            if (
                output_action_schema.payload_schema_by_component[
                    component_id
                ].representation_family
                == "cartesian_pose"
                and output_action_schema.payload_schema_by_component[
                    component_id
                ].value_semantics
                == "relative_delta"
            )
        )
        if not pose_components:
            raise ValueError(
                "Human residual adapter requires at least one relative-delta "
                "Cartesian pose component."
            )
        raw_gains = dict(translation_gain_by_component or {})
        unknown_gains = sorted(set(raw_gains) - set(pose_components))
        if unknown_gains:
            raise ValueError(
                f"Translation gains reference non-residual components: "
                f"{unknown_gains}."
            )
        reference_frames = dict(concrete_reference_frame_by_component or {})
        controlled_frames = dict(concrete_controlled_frame_by_component or {})
        for name, values in (
            ("concrete_reference_frame_by_component", reference_frames),
            ("concrete_controlled_frame_by_component", controlled_frames),
        ):
            unknown = sorted(set(values) - set(pose_components))
            if unknown:
                raise ValueError(
                    f"{name} references non-residual components: {unknown}."
                )
            for component_id, frame_id in tuple(values.items()):
                values[component_id] = _required(
                    frame_id,
                    name=f"{name}.{component_id}",
                )
        gains = {
            component_id: _gain(
                raw_gains.get(component_id, 1.0),
                component_id=component_id,
            )
            for component_id in pose_components
        }
        coordinate_bindings: dict[str, ActionCoordinateBinding] = {}
        for component_id in pose_components:
            payload_schema = output_action_schema.payload_schema_by_component[
                component_id
            ]
            coordinate_bindings[component_id] = (
                bind_action_payload_coordinates(
                    payload_schema,
                    frame_graph=frame_graph,
                    reference_frame_id=reference_frames.get(component_id),
                    controlled_frame_id=controlled_frames.get(component_id),
                )
            )
        contract_payload = {
            "algorithm_id": HUMAN_RESIDUAL_POSE_ALGORITHM_DEVELOPMENT,
            "output_action_schema_hash": (
                output_action_schema.semantic_content_hash
            ),
            "frame_graph_hash": frame_graph.semantic_content_hash,
            "pose_components": pose_components,
            "translation_gain_by_component": gains,
            "concrete_reference_frame_by_component": reference_frames,
            "concrete_controlled_frame_by_component": controlled_frames,
            "action_coordinate_binding_hash_by_component": {
                component_id: binding.semantic_content_hash
                for component_id, binding in coordinate_bindings.items()
            },
        }
        self.output_action_schema = output_action_schema
        self.frame_graph = frame_graph
        self.pose_components = pose_components
        self.translation_gain_by_component = MappingProxyType(gains)
        self.concrete_reference_frame_by_component = MappingProxyType(
            reference_frames
        )
        self.concrete_controlled_frame_by_component = MappingProxyType(
            controlled_frames
        )
        self.action_coordinate_binding_by_component = MappingProxyType(
            coordinate_bindings
        )
        self.contract_hash = canonical_content_hash(
            contract_payload,
            domain=HUMAN_RESIDUAL_ADAPTER_HASH_DOMAIN_DEVELOPMENT,
        )

    def capture_anchor(
        self,
        samples: Sequence[HumanPoseSample],
        *,
        anchor_generation: int,
        captured_at: float,
        captured_clock_domain_id: str,
    ) -> HumanResidualPoseAnchor:
        selected = self._samples_by_component(samples)
        entries: list[HumanResidualPoseAnchorEntry] = []
        for component_id in self.pose_components:
            sample = selected[component_id]
            canonical_pose, resolution = self._canonicalize_sample(sample)
            coordinate_binding = (
                self.action_coordinate_binding_by_component[component_id]
            )
            if (
                coordinate_binding.reference_frame is None
                or coordinate_binding.controlled_frame is None
            ):
                raise ValueError(
                    "Human residual pose coordinate binding is incomplete."
                )
            entries.append(
                HumanResidualPoseAnchorEntry(
                    component_id=component_id,
                    source_sample_id=sample.sample_id,
                    source_sample_hash=sample.semantic_content_hash,
                    source_measured_at=sample.measured_at,
                    source_measured_clock_domain_id=(
                        sample.measured_clock_domain_id
                    ),
                    source_reference_frame_id=(
                        sample.source_reference_frame_id
                    ),
                    source_controlled_frame_id=(
                        sample.source_controlled_frame_id
                    ),
                    source_coordinate_binding_id=(
                        sample.source_coordinate_binding_id
                    ),
                    source_coordinate_binding_hash=(
                        sample.source_coordinate_binding_hash
                    ),
                    canonical_reference_frame_id=(
                        coordinate_binding.reference_frame.frame_id
                    ),
                    canonical_controlled_frame_id=(
                        coordinate_binding.controlled_frame.frame_id
                    ),
                    pose_canonical_reference_canonical_controlled=canonical_pose,
                    pose_transform_resolution_id=resolution.resolution_id,
                    pose_transform_resolution_hash=(
                        resolution.semantic_content_hash
                    ),
                )
            )
        return HumanResidualPoseAnchor(
            anchor_generation=anchor_generation,
            captured_at=captured_at,
            captured_clock_domain_id=captured_clock_domain_id,
            frame_graph_hash=self.frame_graph.semantic_content_hash,
            entries=tuple(entries),
        )

    def adapt(
        self,
        samples: Sequence[HumanPoseSample],
        *,
        anchor: HumanResidualPoseAnchor,
        adapted_at: float,
        adapted_clock_domain_id: str,
    ) -> HumanResidualPoseAdaptationResult:
        if not isinstance(anchor, HumanResidualPoseAnchor):
            raise TypeError("anchor must be HumanResidualPoseAnchor.")
        if anchor.frame_graph_hash != self.frame_graph.semantic_content_hash:
            raise ValueError(
                "Human residual anchor belongs to another coordinate graph."
            )
        adapted_time = _finite(adapted_at, name="adapted_at")
        adapted_clock = _required(
            adapted_clock_domain_id,
            name="adapted_clock_domain_id",
        )
        if (
            adapted_clock == anchor.captured_clock_domain_id
            and adapted_time < anchor.captured_at
        ):
            raise ValueError(
                "Human residual adaptation predates its anchor capture."
            )
        if tuple(anchor.entry_by_component) != tuple(
            sorted(self.pose_components)
        ):
            raise ValueError(
                "Human residual anchor component coverage does not match adapter."
            )
        selected = self._samples_by_component(samples)
        values: dict[str, np.ndarray] = {}
        resolution_hashes: dict[str, str] = {}
        for component_id in self.pose_components:
            sample = selected[component_id]
            anchor_entry = anchor.entry_by_component[component_id]
            self._validate_sample_against_anchor(
                sample=sample,
                anchor=anchor_entry,
            )
            current_pose, resolution = self._canonicalize_sample(sample)
            anchor_pose = (
                anchor_entry.pose_canonical_reference_canonical_controlled
            )
            delta_translation = (
                self.translation_gain_by_component[component_id]
                * (current_pose[:3, 3] - anchor_pose[:3, 3])
            )
            delta_rotation = anchor_pose[:3, :3].T @ current_pose[:3, :3]
            schema = self.output_action_schema.payload_schema_by_component[
                component_id
            ]
            values[component_id] = _delta_pose_vector(
                delta_translation,
                delta_rotation,
                dtype=schema.dtype,
            )
            resolution_hashes[component_id] = (
                resolution.semantic_content_hash
            )
        action = ComponentAction(
            action_schema=self.output_action_schema,
            values_by_component=values,
        )
        receipt = HumanResidualPoseAdaptationReceipt(
            adapter_contract_hash=self.contract_hash,
            anchor_id=anchor.anchor_id,
            anchor_hash=anchor.semantic_content_hash,
            anchor_generation=anchor.anchor_generation,
            adapted_at=adapted_time,
            adapted_clock_domain_id=adapted_clock,
            source_sample_hash_by_component={
                component_id: selected[component_id].semantic_content_hash
                for component_id in self.pose_components
            },
            pose_transform_resolution_hash_by_component=resolution_hashes,
            action_coordinate_binding_hash_by_component={
                component_id: self.action_coordinate_binding_by_component[
                    component_id
                ].semantic_content_hash
                for component_id in self.pose_components
            },
            translation_gain_by_component={
                component_id: tuple(
                    float(value)
                    for value in self.translation_gain_by_component[
                        component_id
                    ]
                )
                for component_id in self.pose_components
            },
            output_action_schema_hash=(
                self.output_action_schema.semantic_content_hash
            ),
            output_action_hash=action.semantic_content_hash,
            frame_graph_hash=self.frame_graph.semantic_content_hash,
        )
        return HumanResidualPoseAdaptationResult(
            relative_action=action,
            receipt=receipt,
        )

    def _samples_by_component(
        self,
        samples: Sequence[HumanPoseSample],
    ) -> Mapping[str, HumanPoseSample]:
        values = tuple(samples)
        if any(not isinstance(item, HumanPoseSample) for item in values):
            raise TypeError("samples must contain HumanPoseSample values.")
        selected = {item.component_id: item for item in values}
        if len(selected) != len(values):
            raise ValueError("Human pose sample component IDs must be unique.")
        if set(selected) != set(self.pose_components):
            raise ValueError(
                "Human pose samples must exactly cover adapter pose components."
            )
        return selected

    def _canonicalize_sample(
        self,
        sample: HumanPoseSample,
    ) -> tuple[np.ndarray, ResolvedPoseFrameTransform]:
        coordinate_binding = self.action_coordinate_binding_by_component[
            sample.component_id
        ]
        if (
            coordinate_binding.reference_frame is None
            or coordinate_binding.controlled_frame is None
        ):
            raise ValueError(
                "Human residual pose coordinate binding is incomplete."
            )
        return self.frame_graph.transform_pose_between_frames(
            sample.pose_source_reference_source_controlled,
            source_reference_frame_id=sample.source_reference_frame_id,
            source_controlled_frame_id=sample.source_controlled_frame_id,
            target_reference_frame_id=(
                coordinate_binding.reference_frame.frame_id
            ),
            target_controlled_frame_id=(
                coordinate_binding.controlled_frame.frame_id
            ),
        )

    @staticmethod
    def _validate_sample_against_anchor(
        *,
        sample: HumanPoseSample,
        anchor: HumanResidualPoseAnchorEntry,
    ) -> None:
        for field_name in (
            "source_reference_frame_id",
            "source_controlled_frame_id",
            "source_coordinate_binding_id",
            "source_coordinate_binding_hash",
        ):
            if getattr(sample, field_name) != getattr(anchor, field_name):
                raise ValueError(
                    "Human residual sample and anchor disagree on "
                    f"{field_name}."
                )
        if (
            sample.measured_clock_domain_id
            != anchor.source_measured_clock_domain_id
        ):
            raise ValueError(
                "Human residual sample and anchor use different measured clocks."
            )
        if sample.measured_at < anchor.source_measured_at:
            raise ValueError(
                "Human residual sample predates its recenter anchor."
            )


__all__ = [
    "HUMAN_POSE_SAMPLE_HASH_DOMAIN_DEVELOPMENT",
    "HUMAN_RESIDUAL_ADAPTATION_RECEIPT_HASH_DOMAIN_DEVELOPMENT",
    "HUMAN_RESIDUAL_ADAPTER_HASH_DOMAIN_DEVELOPMENT",
    "HUMAN_RESIDUAL_ANCHOR_ENTRY_HASH_DOMAIN_DEVELOPMENT",
    "HUMAN_RESIDUAL_ANCHOR_HASH_DOMAIN_DEVELOPMENT",
    "HUMAN_RESIDUAL_POSE_ALGORITHM_DEVELOPMENT",
    "HumanPoseSample",
    "HumanResidualPoseAdaptationReceipt",
    "HumanResidualPoseAdaptationResult",
    "HumanResidualPoseAdapter",
    "HumanResidualPoseAnchor",
    "HumanResidualPoseAnchorEntry",
]
