"""Project nominal demonstration references into canonical ComponentAction.

This is a representation adapter only.  It owns no source scheduling,
composition, authority, safety, target admission, IK, dispatch, or simulation
stepping.  Those identities remain downstream of the returned action and
receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ComponentAction, ComponentActionSchema

from .editor import DemonstrationReference


DEMONSTRATION_COMPONENT_ACTION_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.sim.demonstration_component_action_binding.v1"
)
DEMONSTRATION_COMPONENT_ACTION_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.sim.demonstration_component_action_receipt.v1"
)
DEMONSTRATION_COMPONENT_ACTION_RECEIPT_BINDING_KIND_V1 = (
    "demonstration_component_action_receipt.v1"
)
NORMALIZED_GRIPPER_CLOSE_PROJECTION_HASH_DOMAIN_V1 = (
    "dexteleop.sim.normalized_gripper_close_projection.v1"
)


def _required(value: object, *, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must be non-empty.")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


@dataclass(frozen=True)
class NormalizedGripperCloseToJointPositionProjection:
    """Map one explicit 0=open, 1=closed scalar to native joint targets.

    This is a pure representation conversion.  Values outside the declared
    normalized interval are rejected rather than clipped; target safety and
    execution remain downstream responsibilities.
    """

    projection_id: str
    open_target: tuple[float, ...]
    closed_target: tuple[float, ...]
    semantic_content_hash: str = field(init=False)
    input_representation: str = field(
        default="normalized_gripper_close",
        init=False,
    )
    input_names: tuple[str, ...] = field(
        default=("gripper_close",),
        init=False,
    )
    input_units: tuple[str, ...] = field(default=("1",), init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "projection_id",
            _required(self.projection_id, name="projection_id"),
        )
        opened = tuple(float(value) for value in self.open_target)
        closed = tuple(float(value) for value in self.closed_target)
        if not opened or len(opened) != len(closed):
            raise ValueError(
                "open_target and closed_target must have one equal, non-zero dimension."
            )
        if not all(math.isfinite(value) for value in (*opened, *closed)):
            raise ValueError("Gripper projection endpoints must be finite.")
        if opened == closed:
            raise ValueError("Gripper projection endpoints must be distinct.")
        object.__setattr__(self, "open_target", opened)
        object.__setattr__(self, "closed_target", closed)
        payload = {
            "projection_id": self.projection_id,
            "input_representation": self.input_representation,
            "input_names": self.input_names,
            "input_units": self.input_units,
            "input_interval": (0.0, 1.0),
            "open_target": opened,
            "closed_target": closed,
            "out_of_range_policy": "reject",
            "authority": "none",
            "safety": "none",
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=NORMALIZED_GRIPPER_CLOSE_PROJECTION_HASH_DOMAIN_V1,
            ),
        )

    @property
    def output_size(self) -> int:
        return len(self.open_target)

    def project(self, value: object) -> np.ndarray:
        source = np.asarray(value, dtype=np.float64).reshape(-1)
        if source.shape != (1,) or not np.all(np.isfinite(source)):
            raise ValueError("Normalized gripper close must be one finite scalar.")
        close = float(source[0])
        if close < 0.0 or close > 1.0:
            raise ValueError("Normalized gripper close must lie in [0, 1].")
        opened = np.asarray(self.open_target, dtype=np.float64)
        closed = np.asarray(self.closed_target, dtype=np.float64)
        return opened + close * (closed - opened)


@dataclass(frozen=True)
class DemonstrationComponentActionBinding:
    """Exact pose/hand projection onto one canonical action schema."""

    binding_id: str
    action_schema: ComponentActionSchema
    pose_component_id: str
    hand_component_id: str
    demonstration_hand_target_representation: str
    hand_target_projection: NormalizedGripperCloseToJointPositionProjection | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.action_schema, ComponentActionSchema):
            raise TypeError("action_schema must be a ComponentActionSchema.")
        for name in (
            "binding_id",
            "pose_component_id",
            "hand_component_id",
            "demonstration_hand_target_representation",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        if self.pose_component_id == self.hand_component_id:
            raise ValueError("Pose and hand component IDs must be distinct.")
        missing = sorted(
            {
                self.pose_component_id,
                self.hand_component_id,
            }
            - set(self.action_schema.component_ids)
        )
        if missing:
            raise ValueError(f"Demonstration binding references unknown components: {missing}.")

        pose = self.action_schema.payload_schema_by_component[
            self.pose_component_id
        ]
        if (
            pose.representation_family != "cartesian_pose"
            or pose.shape != (7,)
            or pose.coordinate_names
            != ("x", "y", "z", "qx", "qy", "qz", "qw")
            or pose.units != ("m", "m", "m", "1", "1", "1", "1")
            or pose.value_semantics != "absolute_target"
            or pose.reference_frame is None
            or pose.controlled_frame is None
        ):
            raise ValueError(
                "Pose component must be an absolute xyz+quat_xyzw target with "
                "explicit reference and controlled frames."
            )
        hand = self.action_schema.payload_schema_by_component[
            self.hand_component_id
        ]
        if (
            hand.value_semantics not in {"absolute_target", "native_target"}
            or hand.representation_family
            not in {"joint_position", "continuous_native"}
            or len(hand.shape) != 1
            or hand.shape[0] < 1
        ):
            raise ValueError(
                "Hand component must be a one-dimensional absolute joint or "
                "continuous-native target."
            )
        projection = self.hand_target_projection
        if projection is not None:
            if not isinstance(
                projection,
                NormalizedGripperCloseToJointPositionProjection,
            ):
                raise TypeError(
                    "hand_target_projection must be a normalized gripper projection."
                )
            if (
                self.demonstration_hand_target_representation
                != projection.input_representation
            ):
                raise ValueError(
                    "Demonstration hand representation does not match the projection input."
                )
            if projection.output_size != hand.shape[0]:
                raise ValueError(
                    "Hand projection output dimension does not match the target schema."
                )
            if (
                hand.representation_family != "joint_position"
                or hand.value_semantics != "absolute_target"
            ):
                raise ValueError(
                    "Normalized gripper close projection requires an absolute "
                    "joint-position target schema."
                )
        payload = {
            "binding_id": self.binding_id,
            "action_schema_hash": self.action_schema.semantic_content_hash,
            "pose_component_id": self.pose_component_id,
            "pose_payload_schema_hash": pose.semantic_content_hash,
            "hand_component_id": self.hand_component_id,
            "hand_payload_schema_hash": hand.semantic_content_hash,
            "demonstration_hand_target_representation": (
                self.demonstration_hand_target_representation
            ),
            "authority": "none",
            "safety": "none",
        }
        if projection is not None:
            payload["hand_target_projection_id"] = projection.projection_id
            payload["hand_target_projection_hash"] = projection.semantic_content_hash
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=DEMONSTRATION_COMPONENT_ACTION_BINDING_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class DemonstrationComponentActionReceipt:
    """Immutable lineage for one nominal-reference representation attempt."""

    receipt_id: str
    reference_id: str
    binding_id: str
    binding_hash: str
    source_trajectory_hash: str
    edit_hash: str
    edit_sample_id: str
    edited_trajectory_hash: str
    requested_time_sec: float
    resolved_time_sec: float
    pose_component_id: str
    hand_component_id: str
    input_reference_hash: str
    output_action_hash: str | None
    accepted: bool
    rejection_reasons: tuple[str, ...]
    semantic_content_hash: str


@dataclass(frozen=True)
class DemonstrationComponentActionResult:
    action: ComponentAction | None
    receipt: DemonstrationComponentActionReceipt

    @property
    def accepted(self) -> bool:
        return self.action is not None and self.receipt.accepted


class DemonstrationComponentActionAdapter:
    """Validate and cast one nominal reference into an exact target schema."""

    def __init__(self, binding: DemonstrationComponentActionBinding) -> None:
        if not isinstance(binding, DemonstrationComponentActionBinding):
            raise TypeError(
                "binding must be a DemonstrationComponentActionBinding."
            )
        self.binding = binding

    def adapt(
        self,
        reference: DemonstrationReference,
    ) -> DemonstrationComponentActionResult:
        if not isinstance(reference, DemonstrationReference):
            raise TypeError("reference must be a DemonstrationReference.")
        binding = self.binding
        pose_schema = binding.action_schema.payload_schema_by_component[
            binding.pose_component_id
        ]
        hand_schema = binding.action_schema.payload_schema_by_component[
            binding.hand_component_id
        ]
        input_payload = {
            "source_trajectory_hash": reference.source_trajectory_hash,
            "edit_hash": reference.edit_hash,
            "edit_sample_id": reference.edit_sample_id,
            "edited_trajectory_hash": reference.edited_trajectory_hash,
            "requested_time_sec": reference.requested_time_sec,
            "resolved_time_sec": reference.resolved_time_sec,
            "tool_pose_xyzquat_xyzw": reference.tool_pose_xyzquat_xyzw,
            "hand_target": reference.hand_target,
            "reference_frame": reference.reference_frame,
            "controlled_frame": reference.controlled_frame,
            "hand_component_id": reference.hand_component_id,
            "hand_target_representation": reference.hand_target_representation,
            "hand_target_names": reference.hand_target_names,
            "hand_target_units": reference.hand_target_units,
        }
        input_hash = canonical_content_hash(
            input_payload,
            domain="dexteleop.sim.demonstration_reference.adapter_input.v1",
        )
        reasons: list[str] = []
        if reference.reference_frame != pose_schema.reference_frame:
            reasons.append("pose_reference_frame_mismatch")
        if reference.controlled_frame != pose_schema.controlled_frame:
            reasons.append("pose_controlled_frame_mismatch")
        if reference.hand_component_id != binding.hand_component_id:
            reasons.append("hand_component_id_mismatch")
        if (
            reference.hand_target_representation
            != binding.demonstration_hand_target_representation
        ):
            reasons.append("hand_target_representation_mismatch")
        projection = binding.hand_target_projection
        expected_names = (
            hand_schema.coordinate_names
            if projection is None
            else projection.input_names
        )
        expected_units = (
            hand_schema.units
            if projection is None
            else projection.input_units
        )
        if reference.hand_target_names != expected_names:
            reasons.append("hand_target_names_mismatch")
        if reference.hand_target_units != expected_units:
            reasons.append("hand_target_units_mismatch")
        quaternion_norm = float(
            np.linalg.norm(reference.tool_pose_xyzquat_xyzw[3:7])
        )
        if not math.isclose(
            quaternion_norm,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            reasons.append("pose_quaternion_not_unit")

        action: ComponentAction | None = None
        if not reasons:
            try:
                hand_target = (
                    reference.hand_target
                    if projection is None
                    else projection.project(reference.hand_target)
                )
                action = ComponentAction(
                    action_schema=binding.action_schema,
                    values_by_component={
                        binding.pose_component_id: (
                            reference.tool_pose_xyzquat_xyzw
                        ),
                        binding.hand_component_id: hand_target,
                    },
                )
            except (TypeError, ValueError) as exc:
                reasons.append(
                    f"target_schema_validation_failed:{type(exc).__name__}:{exc}"
                )

        receipt_payload: dict[str, Any] = {
            "binding_id": binding.binding_id,
            "binding_hash": binding.semantic_content_hash,
            "source_trajectory_hash": reference.source_trajectory_hash,
            "edit_hash": reference.edit_hash,
            "edit_sample_id": reference.edit_sample_id,
            "edited_trajectory_hash": reference.edited_trajectory_hash,
            "requested_time_sec": reference.requested_time_sec,
            "resolved_time_sec": reference.resolved_time_sec,
            "pose_component_id": binding.pose_component_id,
            "hand_component_id": binding.hand_component_id,
            "input_reference_hash": input_hash,
            "output_action_hash": (
                None if action is None else action.semantic_content_hash
            ),
            "accepted": action is not None and not reasons,
            "rejection_reasons": tuple(reasons),
        }
        receipt_hash = canonical_content_hash(
            receipt_payload,
            domain=DEMONSTRATION_COMPONENT_ACTION_RECEIPT_HASH_DOMAIN_V1,
        )
        receipt = DemonstrationComponentActionReceipt(
            receipt_id=f"demonstration-component-action-receipt:{receipt_hash}",
            reference_id=f"demonstration-reference:{input_hash}",
            **receipt_payload,
            semantic_content_hash=receipt_hash,
        )
        return DemonstrationComponentActionResult(action=action, receipt=receipt)


__all__ = [
    "DEMONSTRATION_COMPONENT_ACTION_BINDING_HASH_DOMAIN_V1",
    "DEMONSTRATION_COMPONENT_ACTION_RECEIPT_BINDING_KIND_V1",
    "DEMONSTRATION_COMPONENT_ACTION_RECEIPT_HASH_DOMAIN_V1",
    "NORMALIZED_GRIPPER_CLOSE_PROJECTION_HASH_DOMAIN_V1",
    "DemonstrationComponentActionAdapter",
    "DemonstrationComponentActionBinding",
    "DemonstrationComponentActionReceipt",
    "DemonstrationComponentActionResult",
    "NormalizedGripperCloseToJointPositionProjection",
]
