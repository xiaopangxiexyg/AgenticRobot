"""Default-off native v2.5-p producer for aligned UMI samples."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from dataset_contracts.v25p_native import (
    V25pComponentStreamProfileLike,
    V25pComponentStreamProfileV2,
)
from dataset_interface.component_action_projection import (
    ComponentActionProjectionSpec,
    project_component_action_to_v25p,
)
from robot_motion.action import ComponentAction, ComponentActionSchema

from ..v25p_component_action import (
    V25pActionComponentTiming,
    attach_projected_component_action_timings,
)
from ..v25p_native_store import V25pComponentInput
from ..v25p_producer import (
    V25pProducedSample,
    V25pProducerBindings,
    validate_produced_values,
    validate_producer_bindings,
)
from .types import UmiPoseMeasurement, UmiSample


def _finite_timestamp(value: object, *, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return result if np.isfinite(result) else float(fallback)


class UmiV25pProducer:
    """Translate one next-observation-aligned UMI sample into native streams."""

    def __init__(
        self,
        *,
        stream_profile: V25pComponentStreamProfileLike,
        bindings: V25pProducerBindings,
        component_action_schema: ComponentActionSchema | None = None,
        component_action_projection: ComponentActionProjectionSpec | None = None,
        cut_clock_domain_id: str = "local_wall_time",
    ) -> None:
        self.stream_profile = stream_profile
        self.bindings = bindings
        self.component_action_schema = component_action_schema
        self.component_action_projection = component_action_projection
        self.cut_clock_domain_id = str(cut_clock_domain_id).strip()
        if not self.cut_clock_domain_id:
            raise ValueError("cut_clock_domain_id must be non-empty.")
        validate_producer_bindings(
            profile=stream_profile,
            bindings=bindings,
            expected_stage_by_value_key={
                key: (
                    "next_accepted_observation"
                    if key.startswith("action.pose.")
                    else "accepted_reference"
                    if key.startswith("action.end_effector.")
                    else "causal_measured"
                )
                for key in bindings.stream_id_by_value_key
            },
        )
        v2 = isinstance(stream_profile, V25pComponentStreamProfileV2)
        if v2 != (
            component_action_schema is not None
            and component_action_projection is not None
        ):
            raise ValueError(
                "Canonical v2 stream profiles require exactly one ComponentAction "
                "schema and projection spec; frozen v1 profiles require neither."
            )
        if v2 and (
            component_action_schema.semantic_content_hash
            != stream_profile.component_action_space_schema.semantic_content_hash
            or component_action_projection.component_action_space_schema_hash
            != component_action_schema.semantic_content_hash
        ):
            raise ValueError("UMI ComponentAction/profile binding mismatch.")

    @staticmethod
    def _pose_input(
        pose: UmiPoseMeasurement,
        *,
        source_record_id: str,
    ) -> V25pComponentInput:
        return V25pComponentInput(
            values=np.asarray(pose.raw_tcp_pose7, dtype=np.float32),
            effective_at=float(pose.source_timestamp),
            available_at=float(pose.available_timestamp),
            effective_clock_domain_id=str(pose.source_clock),
            available_clock_domain_id=str(pose.available_clock),
            source_record_id=source_record_id,
        )

    def produce(self, sample: UmiSample) -> V25pProducedSample:
        if sample.arm_action_poses is None:
            raise ValueError(
                "UMI native action requires the next accepted observation alignment."
            )
        action_source_timestamp = float(sample.arm_action_source_timestamp)
        cut_at = max(
            float(sample.timestamp),
            action_source_timestamp,
            *(float(pose.available_timestamp) for pose in sample.arm_action_poses.values()),
        )
        values: dict[str, V25pComponentInput] = {}
        source_by_key: dict[str, V25pComponentInput] = {}
        for side in ("left", "right"):
            source_by_key[f"action.pose.{side}"] = self._pose_input(
                sample.arm_action_poses[side],
                source_record_id=(
                    f"umi_next_pose:{int(sample.episode_index)}:"
                    f"{int(sample.arm_action_source_frame_index)}:{side}"
                ),
            )
            source_by_key[f"state.pose.{side}"] = self._pose_input(
                sample.poses[side],
                source_record_id=(
                    f"umi_pose:{int(sample.episode_index)}:"
                    f"{int(sample.frame_index)}:{side}"
                ),
            )
            source_by_key[f"action.end_effector.{side}"] = V25pComponentInput(
                values=np.asarray([sample.gripper_commands[side]], dtype=np.float32),
                effective_at=float(sample.timestamp),
                available_at=float(sample.timestamp),
                effective_clock_domain_id=self.cut_clock_domain_id,
                available_clock_domain_id=self.cut_clock_domain_id,
                source_record_id=(
                    f"umi_gripper_command:{int(sample.episode_index)}:"
                    f"{int(sample.frame_index)}:{side}"
                ),
            )
            hand: Mapping[str, Any] = sample.hand_states[side]
            measured_at = _finite_timestamp(
                hand.get("gripper_measurement_timestamp", hand.get("joint_timestamp")),
                fallback=float(sample.timestamp),
            )
            available_at = _finite_timestamp(
                hand.get("telemetry_snapshot_available_timestamp"),
                fallback=float(sample.timestamp),
            )
            source_by_key[f"state.end_effector.{side}"] = V25pComponentInput(
                values=np.asarray(
                    [float(hand["gripper_measured_scalar"])], dtype=np.float32
                ),
                effective_at=measured_at,
                available_at=available_at,
                effective_clock_domain_id=str(
                    hand.get("telemetry_snapshot_source_clock", self.cut_clock_domain_id)
                ),
                available_clock_domain_id=str(
                    hand.get("telemetry_snapshot_available_clock", self.cut_clock_domain_id)
                ),
                source_record_id=(
                    f"umi_gripper_state:{int(sample.episode_index)}:"
                    f"{int(sample.frame_index)}:{side}"
                ),
            )
        if isinstance(self.stream_profile, V25pComponentStreamProfileV2):
            assert self.component_action_schema is not None
            assert self.component_action_projection is not None
            source_key_by_stream = {
                stream_id: value_key
                for value_key, stream_id in self.bindings.stream_id_by_value_key.items()
            }
            action = ComponentAction(
                action_schema=self.component_action_schema,
                values_by_component={
                    component_id: source_by_key[source_key_by_stream[stream_id]].values
                    for component_id, stream_id in (
                        self.component_action_projection.stream_id_by_action_component.items()
                    )
                },
            )
            projected = project_component_action_to_v25p(
                action=action,
                profile=self.stream_profile,
                spec=self.component_action_projection,
            )
            values.update(
                attach_projected_component_action_timings(
                    projected=projected,
                    profile=self.stream_profile,
                    timing_by_stream={
                        stream_id: V25pActionComponentTiming(
                            effective_at=source_by_key[
                                source_key_by_stream[stream_id]
                            ].effective_at,
                            available_at=source_by_key[
                                source_key_by_stream[stream_id]
                            ].available_at,
                            effective_clock_domain_id=source_by_key[
                                source_key_by_stream[stream_id]
                            ].effective_clock_domain_id,
                            available_clock_domain_id=source_by_key[
                                source_key_by_stream[stream_id]
                            ].available_clock_domain_id,
                        )
                        for stream_id in projected.values_by_stream
                    },
                )
            )
        for value_key, stream_id in self.bindings.stream_id_by_value_key.items():
            if stream_id not in values:
                values[stream_id] = source_by_key[value_key]
        produced = V25pProducedSample(
            sample_index=int(sample.frame_index),
            cut_at=cut_at,
            cut_clock_domain_id=self.cut_clock_domain_id,
            values_by_stream=values,
        )
        validate_produced_values(profile=self.stream_profile, produced=produced)
        return produced


__all__ = ["UmiV25pProducer"]
