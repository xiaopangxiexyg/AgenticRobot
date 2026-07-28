"""Atomic pose-plus-joint residual adaptation for canonical human actions.

This source adapter consumes an already-retargeted absolute
``ComponentAction``.  It coordinates the representation-specific pose and
joint-position adapters behind one generic ``ResidualAnchorRequest/Receipt``.
It does not know which role supplies the later baseline and owns no
composition, authority, safety, or execution lifecycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from control_composition.residual_anchor import (
    ResidualAnchorComponentBinding,
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
)
from robot_motion.action import ComponentAction, ComponentActionSchema
from robot_motion.coordinate_frames import CoordinateFrameGraph

from .human_joint_residual import (
    HumanJointPositionResidualAdapter,
    HumanJointPositionResidualAnchor,
    HumanJointPositionResidualResult,
    HumanJointPositionSample,
)
from .human_residual import (
    HumanPoseSample,
    HumanResidualPoseAdaptationResult,
    HumanResidualPoseAdapter,
    HumanResidualPoseAnchor,
)


HUMAN_COMPONENT_RESIDUAL_ADAPTER_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_component_residual_adapter.development"
)
HUMAN_COMPONENT_RESIDUAL_ANCHOR_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_component_residual_anchor.development"
)
HUMAN_COMPONENT_RESIDUAL_ADAPTATION_RECEIPT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.human_component_residual_adaptation_receipt.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class HumanComponentResidualAnchor:
    request: ResidualAnchorRequest
    receipt: ResidualAnchorReceipt
    pose_anchor: HumanResidualPoseAnchor | None
    joint_anchor: HumanJointPositionResidualAnchor | None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ResidualAnchorRequest):
            raise TypeError("request must be ResidualAnchorRequest.")
        if not isinstance(self.receipt, ResidualAnchorReceipt):
            raise TypeError("receipt must be ResidualAnchorReceipt.")
        self.receipt.validate_request(self.request)
        if self.pose_anchor is None and self.joint_anchor is None:
            raise ValueError(
                "Human component residual anchor requires pose or joint state."
            )
        for name, anchor, expected_type in (
            ("pose_anchor", self.pose_anchor, HumanResidualPoseAnchor),
            (
                "joint_anchor",
                self.joint_anchor,
                HumanJointPositionResidualAnchor,
            ),
        ):
            if anchor is not None and not isinstance(anchor, expected_type):
                raise TypeError(f"{name} has an invalid type.")
            if (
                anchor is not None
                and anchor.anchor_generation
                != self.request.request_generation
            ):
                raise ValueError(
                    f"{name} generation disagrees with residual request."
                )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "request_hash": self.request.semantic_content_hash,
                    "receipt_hash": self.receipt.semantic_content_hash,
                    "pose_anchor_hash": (
                        None
                        if self.pose_anchor is None
                        else self.pose_anchor.semantic_content_hash
                    ),
                    "joint_anchor_hash": (
                        None
                        if self.joint_anchor is None
                        else self.joint_anchor.semantic_content_hash
                    ),
                },
                domain=HUMAN_COMPONENT_RESIDUAL_ANCHOR_HASH_DOMAIN_DEVELOPMENT,
            ),
        )


@dataclass(frozen=True)
class HumanComponentResidualAdaptationReceipt:
    adapter_contract_hash: str
    anchor_receipt_id: str
    anchor_receipt_hash: str
    source_action_id: str
    source_action_hash: str
    source_binding_id: str
    source_binding_hash: str
    adapted_at: float
    adapted_clock_domain_id: str
    pose_adaptation_receipt_hash: str | None
    joint_adaptation_receipt_hash: str | None
    output_action_schema_hash: str
    output_action_hash: str
    semantic_content_hash: str = field(init=False)
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "adapter_contract_hash",
                "anchor_receipt_id",
                "anchor_receipt_hash",
                "source_action_id",
                "source_action_hash",
                "source_binding_id",
                "source_binding_hash",
                "adapted_clock_domain_id",
                "output_action_schema_hash",
                "output_action_hash",
            )
        }
        adapted_at = float(self.adapted_at)
        if not math.isfinite(adapted_at):
            raise ValueError("adapted_at must be finite.")
        pose_hash = (
            None
            if self.pose_adaptation_receipt_hash is None
            else _required(
                self.pose_adaptation_receipt_hash,
                name="pose_adaptation_receipt_hash",
            )
        )
        joint_hash = (
            None
            if self.joint_adaptation_receipt_hash is None
            else _required(
                self.joint_adaptation_receipt_hash,
                name="joint_adaptation_receipt_hash",
            )
        )
        if pose_hash is None and joint_hash is None:
            raise ValueError(
                "Component residual adaptation requires pose or joint evidence."
            )
        payload = {
            **text,
            "adapted_at": adapted_at,
            "pose_adaptation_receipt_hash": pose_hash,
            "joint_adaptation_receipt_hash": joint_hash,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=(
                HUMAN_COMPONENT_RESIDUAL_ADAPTATION_RECEIPT_HASH_DOMAIN_DEVELOPMENT
            ),
        )
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "receipt_id",
            f"human-component-residual-adaptation:{semantic_hash}",
        )


@dataclass(frozen=True)
class HumanComponentResidualResult:
    relative_action: ComponentAction
    receipt: HumanComponentResidualAdaptationReceipt
    pose_result: HumanResidualPoseAdaptationResult | None
    joint_result: HumanJointPositionResidualResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.relative_action, ComponentAction):
            raise TypeError("relative_action must be ComponentAction.")
        if not isinstance(
            self.receipt,
            HumanComponentResidualAdaptationReceipt,
        ):
            raise TypeError(
                "receipt must be HumanComponentResidualAdaptationReceipt."
            )
        if (
            self.relative_action.semantic_content_hash
            != self.receipt.output_action_hash
        ):
            raise ValueError(
                "Component residual result action and receipt disagree."
            )


class HumanComponentResidualAdapter:
    """Convert one canonical absolute human action into one relative action."""

    def __init__(
        self,
        *,
        absolute_action_schema: ComponentActionSchema,
        relative_action_schema: ComponentActionSchema,
        source_binding_id: str,
        source_binding_hash: str,
        frame_graph: CoordinateFrameGraph | None = None,
        concrete_reference_frame_by_component: Mapping[str, str] | None = None,
        concrete_controlled_frame_by_component: Mapping[str, str] | None = None,
        pose_translation_gain_by_component: Mapping[str, object] | None = None,
        joint_gain_by_component: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(absolute_action_schema, ComponentActionSchema):
            raise TypeError(
                "absolute_action_schema must be ComponentActionSchema."
            )
        if not isinstance(relative_action_schema, ComponentActionSchema):
            raise TypeError(
                "relative_action_schema must be ComponentActionSchema."
            )
        source_binding_id = _required(
            source_binding_id,
            name="source_binding_id",
        )
        source_binding_hash = _required(
            source_binding_hash,
            name="source_binding_hash",
        )
        components = tuple(relative_action_schema.component_ids)
        if not components:
            raise ValueError("Relative action schema must contain components.")
        if not set(components).issubset(absolute_action_schema.component_ids):
            raise ValueError(
                "Relative components must exist in the absolute source schema."
            )
        pose_components: list[str] = []
        joint_components: list[str] = []
        for component_id in components:
            absolute = absolute_action_schema.payload_schema_by_component[
                component_id
            ]
            relative = relative_action_schema.payload_schema_by_component[
                component_id
            ]
            if (
                absolute.value_semantics != "absolute_target"
                or relative.value_semantics != "relative_delta"
            ):
                raise ValueError(
                    f"Component {component_id!r} must map absolute_target to "
                    "relative_delta."
                )
            if (
                absolute.representation_family
                != relative.representation_family
            ):
                raise ValueError(
                    f"Component {component_id!r} changes representation."
                )
            if (
                absolute.shape != relative.shape
                or absolute.coordinate_names != relative.coordinate_names
                or absolute.units != relative.units
            ):
                raise ValueError(
                    f"Component {component_id!r} changes numerical layout."
                )
            if relative.representation_family == "cartesian_pose":
                pose_components.append(component_id)
            elif relative.representation_family == "joint_position":
                joint_components.append(component_id)
            else:
                raise ValueError(
                    "Initial canonical human residual adapter supports only "
                    "cartesian_pose and joint_position components."
                )
        pose_adapter = None
        if pose_components:
            if not isinstance(frame_graph, CoordinateFrameGraph):
                raise TypeError(
                    "frame_graph is required for pose residual components."
                )
            pose_adapter = HumanResidualPoseAdapter(
                output_action_schema=relative_action_schema,
                frame_graph=frame_graph,
                translation_gain_by_component=(
                    pose_translation_gain_by_component
                ),
                concrete_reference_frame_by_component=(
                    concrete_reference_frame_by_component
                ),
                concrete_controlled_frame_by_component=(
                    concrete_controlled_frame_by_component
                ),
            )
        joint_adapter = None
        if joint_components:
            joint_adapter = HumanJointPositionResidualAdapter(
                output_action_schema=relative_action_schema,
                source_payload_schema_by_component={
                    component_id: (
                        absolute_action_schema.payload_schema_by_component[
                            component_id
                        ]
                    )
                    for component_id in joint_components
                },
                gain_by_component=joint_gain_by_component,
            )
        contract_payload = {
            "absolute_action_schema_hash": (
                absolute_action_schema.semantic_content_hash
            ),
            "relative_action_schema_hash": (
                relative_action_schema.semantic_content_hash
            ),
            "source_binding_id": source_binding_id,
            "source_binding_hash": source_binding_hash,
            "component_ids": components,
            "pose_adapter_contract_hash": (
                None if pose_adapter is None else pose_adapter.contract_hash
            ),
            "joint_adapter_contract_hash": (
                None if joint_adapter is None else joint_adapter.contract_hash
            ),
        }
        self.absolute_action_schema = absolute_action_schema
        self.relative_action_schema = relative_action_schema
        self.source_binding_id = source_binding_id
        self.source_binding_hash = source_binding_hash
        self.component_ids = components
        self.pose_components = tuple(pose_components)
        self.joint_components = tuple(joint_components)
        self.frame_graph = frame_graph
        self.concrete_reference_frame_by_component = MappingProxyType(
            dict(concrete_reference_frame_by_component or {})
        )
        self.concrete_controlled_frame_by_component = MappingProxyType(
            dict(concrete_controlled_frame_by_component or {})
        )
        self.pose_adapter = pose_adapter
        self.joint_adapter = joint_adapter
        self.contract_hash = canonical_content_hash(
            contract_payload,
            domain=HUMAN_COMPONENT_RESIDUAL_ADAPTER_HASH_DOMAIN_DEVELOPMENT,
        )

    def capture_anchor(
        self,
        absolute_action: ComponentAction,
        *,
        request: ResidualAnchorRequest,
        source_action_id: str,
        source_action_hash: str,
        source_available_at: float,
        measured_at: float,
        measured_clock_domain_id: str,
        captured_at: float,
    ) -> HumanComponentResidualAnchor:
        self._validate_request(request)
        self._validate_absolute_action(
            absolute_action,
            source_action_hash=source_action_hash,
        )
        source_action_id = _required(
            source_action_id,
            name="source_action_id",
        )
        pose_samples = self._pose_samples(
            absolute_action,
            source_action_id=source_action_id,
            measured_at=measured_at,
            measured_clock_domain_id=measured_clock_domain_id,
        )
        joint_samples = self._joint_samples(
            absolute_action,
            source_action_id=source_action_id,
            measured_at=measured_at,
            measured_clock_domain_id=measured_clock_domain_id,
        )
        pose_anchor = (
            None
            if self.pose_adapter is None
            else self.pose_adapter.capture_anchor(
                pose_samples,
                anchor_generation=request.request_generation,
                captured_at=captured_at,
                captured_clock_domain_id=request.clock_domain_id,
            )
        )
        joint_anchor = (
            None
            if self.joint_adapter is None
            else self.joint_adapter.capture_anchor(
                joint_samples,
                anchor_generation=request.request_generation,
                captured_at=captured_at,
                captured_clock_domain_id=request.clock_domain_id,
            )
        )
        pose_by_id = {item.component_id: item for item in pose_samples}
        joint_by_id = {item.component_id: item for item in joint_samples}
        bindings: list[ResidualAnchorComponentBinding] = []
        for component_id in self.component_ids:
            if component_id in pose_by_id:
                sample = pose_by_id[component_id]
                anchor_id = pose_anchor.anchor_id
                anchor_hash = pose_anchor.semantic_content_hash
            else:
                sample = joint_by_id[component_id]
                anchor_id = joint_anchor.anchor_id
                anchor_hash = joint_anchor.semantic_content_hash
            bindings.append(
                ResidualAnchorComponentBinding(
                    component_id=component_id,
                    source_sample_id=sample.sample_id,
                    source_sample_hash=sample.semantic_content_hash,
                    absolute_payload_schema_hash=(
                        self.absolute_action_schema.payload_schema_by_component[
                            component_id
                        ].semantic_content_hash
                    ),
                    anchor_id=anchor_id,
                    anchor_hash=anchor_hash,
                )
            )
        receipt = ResidualAnchorReceipt.capture(
            request,
            source_action_id=source_action_id,
            source_action_hash=source_action_hash,
            source_available_at=source_available_at,
            captured_at=captured_at,
            component_bindings=bindings,
        )
        return HumanComponentResidualAnchor(
            request=request,
            receipt=receipt,
            pose_anchor=pose_anchor,
            joint_anchor=joint_anchor,
        )

    def adapt(
        self,
        absolute_action: ComponentAction,
        *,
        anchor: HumanComponentResidualAnchor,
        source_action_id: str,
        source_action_hash: str,
        measured_at: float,
        measured_clock_domain_id: str,
        adapted_at: float,
        adapted_clock_domain_id: str,
    ) -> HumanComponentResidualResult:
        if not isinstance(anchor, HumanComponentResidualAnchor):
            raise TypeError(
                "anchor must be HumanComponentResidualAnchor."
            )
        self._validate_request(anchor.request)
        self._validate_absolute_action(
            absolute_action,
            source_action_hash=source_action_hash,
        )
        source_action_id = _required(
            source_action_id,
            name="source_action_id",
        )
        pose_result = None
        if self.pose_adapter is not None:
            pose_result = self.pose_adapter.adapt(
                self._pose_samples(
                    absolute_action,
                    source_action_id=source_action_id,
                    measured_at=measured_at,
                    measured_clock_domain_id=measured_clock_domain_id,
                ),
                anchor=anchor.pose_anchor,
                adapted_at=adapted_at,
                adapted_clock_domain_id=adapted_clock_domain_id,
            )
        joint_result = None
        if self.joint_adapter is not None:
            joint_result = self.joint_adapter.adapt(
                self._joint_samples(
                    absolute_action,
                    source_action_id=source_action_id,
                    measured_at=measured_at,
                    measured_clock_domain_id=measured_clock_domain_id,
                ),
                anchor=anchor.joint_anchor,
                adapted_at=adapted_at,
                adapted_clock_domain_id=adapted_clock_domain_id,
            )
        values = {}
        for result in (pose_result, joint_result):
            if result is None:
                continue
            for component_id in result.relative_action.component_coverage:
                values[component_id] = result.relative_action.value(
                    component_id
                ).values
        if set(values) != set(self.component_ids):
            raise RuntimeError(
                "Representation adapters did not cover the residual profile."
            )
        action = ComponentAction(
            action_schema=self.relative_action_schema,
            values_by_component=values,
        )
        receipt = HumanComponentResidualAdaptationReceipt(
            adapter_contract_hash=self.contract_hash,
            anchor_receipt_id=anchor.receipt.receipt_id,
            anchor_receipt_hash=anchor.receipt.semantic_content_hash,
            source_action_id=source_action_id,
            source_action_hash=source_action_hash,
            source_binding_id=self.source_binding_id,
            source_binding_hash=self.source_binding_hash,
            adapted_at=adapted_at,
            adapted_clock_domain_id=adapted_clock_domain_id,
            pose_adaptation_receipt_hash=(
                None
                if pose_result is None
                else pose_result.receipt.semantic_content_hash
            ),
            joint_adaptation_receipt_hash=(
                None
                if joint_result is None
                else joint_result.receipt.semantic_content_hash
            ),
            output_action_schema_hash=(
                self.relative_action_schema.semantic_content_hash
            ),
            output_action_hash=action.semantic_content_hash,
        )
        return HumanComponentResidualResult(
            relative_action=action,
            receipt=receipt,
            pose_result=pose_result,
            joint_result=joint_result,
        )

    def _validate_request(self, request: ResidualAnchorRequest) -> None:
        if not isinstance(request, ResidualAnchorRequest):
            raise TypeError("request must be ResidualAnchorRequest.")
        expected = {
            "source_binding_id": self.source_binding_id,
            "source_binding_hash": self.source_binding_hash,
            "absolute_action_schema_hash": (
                self.absolute_action_schema.semantic_content_hash
            ),
            "relative_action_schema_hash": (
                self.relative_action_schema.semantic_content_hash
            ),
        }
        for name, value in expected.items():
            if getattr(request, name) != value:
                raise ValueError(
                    f"Residual anchor request disagrees on {name}."
                )
        if set(request.component_ids) != set(self.component_ids):
            raise ValueError(
                "Residual anchor request component coverage disagrees with "
                "the source adapter."
            )

    def _validate_absolute_action(
        self,
        action: ComponentAction,
        *,
        source_action_hash: str,
    ) -> None:
        if not isinstance(action, ComponentAction):
            raise TypeError("absolute_action must be ComponentAction.")
        if (
            action.action_schema.semantic_content_hash
            != self.absolute_action_schema.semantic_content_hash
        ):
            raise ValueError("Absolute action uses another action schema.")
        if not set(self.component_ids).issubset(action.component_coverage):
            raise ValueError(
                "Absolute action does not cover all residual components."
            )
        if action.semantic_content_hash != _required(
            source_action_hash,
            name="source_action_hash",
        ):
            raise ValueError("Source action hash does not match its payload.")

    def _pose_samples(
        self,
        action: ComponentAction,
        *,
        source_action_id: str,
        measured_at: float,
        measured_clock_domain_id: str,
    ) -> tuple[HumanPoseSample, ...]:
        return tuple(
            HumanPoseSample.from_component_action(
                action,
                component_id,
                sample_id=f"{source_action_id}:{component_id}",
                measured_at=measured_at,
                measured_clock_domain_id=measured_clock_domain_id,
                source_coordinate_binding_id=self.source_binding_id,
                source_coordinate_binding_hash=self.source_binding_hash,
                concrete_reference_frame_id=(
                    self.concrete_reference_frame_by_component.get(
                        component_id
                    )
                ),
                concrete_controlled_frame_id=(
                    self.concrete_controlled_frame_by_component.get(
                        component_id
                    )
                ),
            )
            for component_id in self.pose_components
        )

    def _joint_samples(
        self,
        action: ComponentAction,
        *,
        source_action_id: str,
        measured_at: float,
        measured_clock_domain_id: str,
    ) -> tuple[HumanJointPositionSample, ...]:
        return tuple(
            HumanJointPositionSample.from_component_action(
                action,
                component_id,
                sample_id=f"{source_action_id}:{component_id}",
                measured_at=measured_at,
                measured_clock_domain_id=measured_clock_domain_id,
                source_value_binding_id=self.source_binding_id,
                source_value_binding_hash=self.source_binding_hash,
            )
            for component_id in self.joint_components
        )


__all__ = [
    "HUMAN_COMPONENT_RESIDUAL_ADAPTATION_RECEIPT_HASH_DOMAIN_DEVELOPMENT",
    "HUMAN_COMPONENT_RESIDUAL_ADAPTER_HASH_DOMAIN_DEVELOPMENT",
    "HUMAN_COMPONENT_RESIDUAL_ANCHOR_HASH_DOMAIN_DEVELOPMENT",
    "HumanComponentResidualAdaptationReceipt",
    "HumanComponentResidualAdapter",
    "HumanComponentResidualAnchor",
    "HumanComponentResidualResult",
]
