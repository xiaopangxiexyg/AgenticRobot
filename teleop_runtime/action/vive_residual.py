"""Explicit VIVE-coordinate binding for the generic human residual adapter.

The historical VIVE retarget frame is not assumed to be the action/dataset
canonical frame. Callers must provide ``T_canonical_vive_retarget`` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from teleop_runtime.calibration import TeleopCalibration


VIVE_HUMAN_RESIDUAL_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.vive_human_residual_coordinate_binding.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _frame(
    value: CoordinateFrameSpec | None,
    *,
    default_id: str,
    family: str,
    coordinate_profile_hash: str,
) -> CoordinateFrameSpec:
    if value is not None:
        if not isinstance(value, CoordinateFrameSpec):
            raise TypeError(f"{family} frame must be CoordinateFrameSpec.")
        return value
    return CoordinateFrameSpec(
        frame_id=default_id,
        semantic_parameters={
            "frame_family": family,
            "coordinate_profile_hash": coordinate_profile_hash,
        },
    )


@dataclass(frozen=True)
class ViveHumanResidualCoordinateBinding:
    coordinate_profile_id: str
    coordinate_profile_hash: str
    frame_graph: CoordinateFrameGraph
    tracker_role_by_component: Mapping[str, str]
    source_controlled_frame_by_component: Mapping[str, str]
    target_controlled_frame_by_component: Mapping[str, str]
    canonical_reference_frame_id: str
    semantic_content_hash: str = field(init=False)
    binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        profile_id = _required(
            self.coordinate_profile_id,
            name="coordinate_profile_id",
        )
        profile_hash = _required(
            self.coordinate_profile_hash,
            name="coordinate_profile_hash",
        )
        if not isinstance(self.frame_graph, CoordinateFrameGraph):
            raise TypeError("frame_graph must be CoordinateFrameGraph.")
        canonical = _required(
            self.canonical_reference_frame_id,
            name="canonical_reference_frame_id",
        )
        mappings = {
            name: {
                _required(component_id, name=f"{name}.component_id"): _required(
                    value,
                    name=f"{name}.{component_id}",
                )
                for component_id, value in dict(getattr(self, name)).items()
            }
            for name in (
                "tracker_role_by_component",
                "source_controlled_frame_by_component",
                "target_controlled_frame_by_component",
            )
        }
        key_sets = {frozenset(value) for value in mappings.values()}
        if len(key_sets) != 1 or not next(iter(key_sets), frozenset()):
            raise ValueError(
                "VIVE residual component mappings must have the same non-empty "
                "key set."
            )
        if canonical not in self.frame_graph.frame_by_id:
            raise ValueError(
                "VIVE residual canonical reference frame is absent from graph."
            )
        for name in (
            "source_controlled_frame_by_component",
            "target_controlled_frame_by_component",
        ):
            missing = sorted(
                set(mappings[name].values()) - set(self.frame_graph.frame_by_id)
            )
            if missing:
                raise ValueError(
                    f"VIVE residual {name} references unknown frames: {missing}."
                )
        payload = {
            "coordinate_profile_id": profile_id,
            "coordinate_profile_hash": profile_hash,
            "frame_graph_hash": self.frame_graph.semantic_content_hash,
            **mappings,
            "canonical_reference_frame_id": canonical,
            "bridge_policy": "explicit_T_canonical_vive_retarget_required",
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=VIVE_HUMAN_RESIDUAL_BINDING_HASH_DOMAIN_DEVELOPMENT,
        )
        object.__setattr__(self, "coordinate_profile_id", profile_id)
        object.__setattr__(self, "coordinate_profile_hash", profile_hash)
        object.__setattr__(
            self,
            "canonical_reference_frame_id",
            canonical,
        )
        for name, value in mappings.items():
            object.__setattr__(
                self,
                name,
                MappingProxyType(dict(sorted(value.items()))),
            )
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "binding_id",
            f"vive-human-residual-coordinate-binding:{semantic_hash}",
        )


def build_vive_human_residual_coordinate_binding(
    calibration: TeleopCalibration,
    *,
    coordinate_profile_id: str,
    coordinate_profile_hash: str,
    canonical_reference_frame: CoordinateFrameSpec,
    matrix_canonical_vive_retarget: Any,
    tracker_role_by_component: Mapping[str, str],
    source_tracker_frame_by_component: Mapping[str, CoordinateFrameSpec],
    target_controlled_frame_by_component: Mapping[str, CoordinateFrameSpec],
    graph_id: str = "vive-human-residual-coordinate-graph",
    vive_world_frame: CoordinateFrameSpec | None = None,
    vive_operator_frame: CoordinateFrameSpec | None = None,
    vive_retarget_frame: CoordinateFrameSpec | None = None,
) -> ViveHumanResidualCoordinateBinding:
    """Build a cycle-checked graph matching ``TeleopCalibration`` semantics."""

    if not isinstance(calibration, TeleopCalibration):
        raise TypeError("calibration must be TeleopCalibration.")
    profile_id = _required(coordinate_profile_id, name="coordinate_profile_id")
    profile_hash = _required(
        coordinate_profile_hash,
        name="coordinate_profile_hash",
    )
    if not isinstance(canonical_reference_frame, CoordinateFrameSpec):
        raise TypeError(
            "canonical_reference_frame must be CoordinateFrameSpec."
        )
    roles = {
        _required(component_id, name="tracker_role component_id"): _required(
            role,
            name=f"tracker_role_by_component.{component_id}",
        )
        for component_id, role in dict(tracker_role_by_component).items()
    }
    source_specs = dict(source_tracker_frame_by_component)
    target_specs = dict(target_controlled_frame_by_component)
    if (
        not roles
        or set(roles) != set(source_specs)
        or set(roles) != set(target_specs)
    ):
        raise ValueError(
            "Tracker roles, source tracker frames, and target controlled "
            "frames must cover the same non-empty component set."
        )
    for name, specs in (
        ("source_tracker_frame_by_component", source_specs),
        ("target_controlled_frame_by_component", target_specs),
    ):
        if any(not isinstance(value, CoordinateFrameSpec) for value in specs.values()):
            raise TypeError(f"{name} values must be CoordinateFrameSpec.")
    if calibration.T_W_O is None or calibration.T_O_D is None:
        raise ValueError(
            "VIVE residual binding requires explicit T_W_O and T_O_D "
            "calibration transforms."
        )
    missing_offsets = sorted(
        {role for role in roles.values() if role not in calibration.tracker_offsets}
    )
    if missing_offsets:
        raise ValueError(
            f"VIVE residual binding is missing tracker offsets: {missing_offsets}."
        )

    world = _frame(
        vive_world_frame,
        default_id="vive_world",
        family="vive_world",
        coordinate_profile_hash=profile_hash,
    )
    operator = _frame(
        vive_operator_frame,
        default_id="vive_operator",
        family="vive_operator",
        coordinate_profile_hash=profile_hash,
    )
    retarget = _frame(
        vive_retarget_frame,
        default_id="vive_retarget_frame",
        family="vive_retarget_frame",
        coordinate_profile_hash=profile_hash,
    )
    frame_by_id: dict[str, CoordinateFrameSpec] = {}

    def add_frame(value: CoordinateFrameSpec) -> None:
        existing = frame_by_id.get(value.frame_id)
        if (
            existing is not None
            and existing.semantic_content_hash != value.semantic_content_hash
        ):
            raise ValueError(
                f"Conflicting definitions for frame {value.frame_id!r}."
            )
        frame_by_id[value.frame_id] = value

    for value in (
        world,
        operator,
        retarget,
        canonical_reference_frame,
        *source_specs.values(),
        *target_specs.values(),
    ):
        add_frame(value)

    edge_parameters = {
        "coordinate_profile_id": profile_id,
        "coordinate_profile_hash": profile_hash,
    }
    transforms = [
        RigidFrameTransformSpec(
            transform_id="vive-world-to-operator",
            source_frame_id=world.frame_id,
            target_frame_id=operator.frame_id,
            matrix_target_source=np.linalg.inv(
                np.asarray(calibration.T_W_O, dtype=np.float64)
            ),
            semantic_parameters={
                **edge_parameters,
                "calibration_field": "inverse(T_W_O)",
            },
        ),
        RigidFrameTransformSpec(
            transform_id="vive-operator-to-retarget",
            source_frame_id=operator.frame_id,
            target_frame_id=retarget.frame_id,
            matrix_target_source=np.linalg.inv(
                np.asarray(calibration.T_O_D, dtype=np.float64)
            ),
            semantic_parameters={
                **edge_parameters,
                "calibration_field": "inverse(T_O_D)",
            },
        ),
    ]
    bridge = np.asarray(matrix_canonical_vive_retarget, dtype=np.float64)
    if retarget.frame_id == canonical_reference_frame.frame_id:
        if not np.allclose(bridge, np.eye(4), rtol=0.0, atol=1e-9):
            raise ValueError(
                "A shared VIVE-retarget/canonical frame ID requires an "
                "explicit identity bridge."
            )
    else:
        transforms.append(
            RigidFrameTransformSpec(
                transform_id="vive-retarget-to-canonical",
                source_frame_id=retarget.frame_id,
                target_frame_id=canonical_reference_frame.frame_id,
                matrix_target_source=bridge,
                semantic_parameters={
                    **edge_parameters,
                    "bridge_policy": (
                        "explicit_T_canonical_vive_retarget_required"
                    ),
                },
            )
        )
    for component_id in sorted(roles):
        source_frame = source_specs[component_id]
        target_frame = target_specs[component_id]
        tracker_role = roles[component_id]
        offset = calibration.get_tracker_offset(tracker_role)
        if source_frame.frame_id == target_frame.frame_id:
            if not np.allclose(offset, np.eye(4), rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"Component {component_id!r} shares source/target "
                    "controlled frame but has a non-identity tracker offset."
                )
            continue
        transforms.append(
            RigidFrameTransformSpec(
                transform_id=f"{component_id}:target-to-tracker",
                source_frame_id=target_frame.frame_id,
                target_frame_id=source_frame.frame_id,
                matrix_target_source=offset,
                semantic_parameters={
                    **edge_parameters,
                    "component_id": component_id,
                    "tracker_role": tracker_role,
                    "calibration_field": (
                        f"tracker_offsets[{tracker_role!r}]"
                    ),
                },
            )
        )
    graph = CoordinateFrameGraph(
        graph_id=graph_id,
        frames=tuple(frame_by_id.values()),
        transforms=tuple(transforms),
    )
    return ViveHumanResidualCoordinateBinding(
        coordinate_profile_id=profile_id,
        coordinate_profile_hash=profile_hash,
        frame_graph=graph,
        tracker_role_by_component=roles,
        source_controlled_frame_by_component={
            component_id: source_specs[component_id].frame_id
            for component_id in roles
        },
        target_controlled_frame_by_component={
            component_id: target_specs[component_id].frame_id
            for component_id in roles
        },
        canonical_reference_frame_id=canonical_reference_frame.frame_id,
    )


__all__ = [
    "VIVE_HUMAN_RESIDUAL_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "ViveHumanResidualCoordinateBinding",
    "build_vive_human_residual_coordinate_binding",
]
