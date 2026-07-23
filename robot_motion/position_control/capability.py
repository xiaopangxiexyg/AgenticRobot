"""Execution capability evidence kept separate from Arbiter rate policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dataset_contracts.canonical_encoding import canonical_content_hash


EXECUTION_CAPABILITY_HASH_DOMAIN_V1 = "dexteleop.execution_capability.v1"

CANCELLATION_ACK_LEVELS = (
    "software_queue",
    "sdk_request",
    "backend_state",
    "measured_motion_stopped",
)


@dataclass(frozen=True)
class ExecutionCapabilityProfile:
    profile_id: str
    validated_max_selected_keyframe_hz: float
    configured_selected_keyframe_hz: float
    validated_max_coupled_ik_hz: float | None
    configured_coupled_ik_hz: float | None
    reference_resample_hz: float
    command_hz: float
    backend_servo_hz: float
    validated_action_spaces: tuple[str, ...]
    validated_selection_groups: tuple[str, ...]
    supports_local_group_stop: bool
    acceptance_evidence_id: str
    decision_runtime_budget_sec: float
    ik_runtime_budget_sec: float
    end_to_end_control_budget_sec: float
    cancellation_ack_deadline_sec: float = 0.1
    required_cancellation_ack_levels: tuple[str, ...] = (
        "software_queue",
        "sdk_request",
    )
    supports_backend_state_ack: bool = False
    supports_measured_motion_stopped_ack: bool = False

    def __post_init__(self) -> None:
        if not str(self.profile_id).strip() or not str(self.acceptance_evidence_id).strip():
            raise ValueError("Capability profile/evidence IDs must be non-empty.")
        positive_names = (
            "validated_max_selected_keyframe_hz",
            "configured_selected_keyframe_hz",
            "reference_resample_hz",
            "command_hz",
            "backend_servo_hz",
            "decision_runtime_budget_sec",
            "ik_runtime_budget_sec",
            "end_to_end_control_budget_sec",
            "cancellation_ack_deadline_sec",
        )
        for name in positive_names:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.configured_selected_keyframe_hz > self.validated_max_selected_keyframe_hz:
            raise ValueError("Configured selected-keyframe rate exceeds validated capability.")
        if self.validated_max_coupled_ik_hz is None:
            if self.configured_coupled_ik_hz is not None:
                raise ValueError("Configured coupled IK requires a validated coupled-IK ceiling.")
        else:
            if self.configured_coupled_ik_hz is None:
                raise ValueError("Validated coupled IK capability requires a configured rate.")
            if self.configured_coupled_ik_hz > self.validated_max_coupled_ik_hz:
                raise ValueError("Configured coupled-IK rate exceeds validated capability.")
        if self.command_hz > self.backend_servo_hz:
            raise ValueError("command_hz cannot exceed backend_servo_hz.")
        if self.reference_resample_hz < self.configured_selected_keyframe_hz:
            raise ValueError("reference_resample_hz cannot be below selected-keyframe rate.")
        if self.command_hz < self.reference_resample_hz:
            raise ValueError("command_hz cannot be below reference_resample_hz.")
        object.__setattr__(self, "validated_action_spaces", tuple(self.validated_action_spaces))
        object.__setattr__(self, "validated_selection_groups", tuple(self.validated_selection_groups))
        required_ack_levels = tuple(str(value).strip() for value in self.required_cancellation_ack_levels)
        if not required_ack_levels or len(set(required_ack_levels)) != len(required_ack_levels):
            raise ValueError("required_cancellation_ack_levels must be non-empty and unique.")
        unknown_ack_levels = set(required_ack_levels).difference(CANCELLATION_ACK_LEVELS)
        if unknown_ack_levels:
            raise ValueError(f"Unsupported cancellation acknowledgement levels: {sorted(unknown_ack_levels)!r}.")
        if "backend_state" in required_ack_levels and not self.supports_backend_state_ack:
            raise ValueError("backend_state acknowledgement is required but not supported.")
        if (
            "measured_motion_stopped" in required_ack_levels
            and not self.supports_measured_motion_stopped_ack
        ):
            raise ValueError(
                "measured_motion_stopped acknowledgement is required but not supported."
            )
        object.__setattr__(self, "required_cancellation_ack_levels", required_ack_levels)
        object.__setattr__(self, "supports_backend_state_ack", bool(self.supports_backend_state_ack))
        object.__setattr__(
            self,
            "supports_measured_motion_stopped_ack",
            bool(self.supports_measured_motion_stopped_ack),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            },
            domain=EXECUTION_CAPABILITY_HASH_DOMAIN_V1,
        )


__all__ = ["CANCELLATION_ACK_LEVELS", "ExecutionCapabilityProfile"]
