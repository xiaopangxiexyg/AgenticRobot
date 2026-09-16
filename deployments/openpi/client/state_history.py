"""Default-off OpenPI RTC state-history capture and request projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from dataset_contracts.state_history import PolicyStateHistorySpec
from dataset_contracts.state_input import STATE_INPUT_MODE_RAW
from dataset_interface.state_history import PolicyStateHistoryBuffer
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.schema import STATE_ARM_QPOS_SLICES
from robot_motion.deploy.v22 import V22DeploymentAdapter

HISTORY_STATE_SCHEMA = "dexteleop_history_state_input_v2"
HISTORY_STATE_LAYOUT = (
    "left_arm_qpos_6",
    "right_arm_qpos_6",
    "left_tcp_xyzquat_7",
    "right_tcp_xyzquat_7",
    "left_gripper_opening_1",
    "right_gripper_opening_1",
)
_HISTORY_STATE_FIELDS = frozenset(
    {
        "schema",
        "input_key",
        "age_input_key",
        "frame_offsets",
        "offset_semantics",
        "state_dim",
        "state_layout",
        "age_unit",
        "selection",
        "token_placement",
        "qpos_normalization",
        "policy_state_normalization",
    }
)


@dataclass(frozen=True)
class OpenPIStateHistoryBinding:
    input_key: str
    age_input_key: str
    spec: PolicyStateHistorySpec

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> OpenPIStateHistoryBinding:
        fields = frozenset(value)
        if fields != _HISTORY_STATE_FIELDS:
            raise ValueError(
                "OpenPI state-history metadata fields disagree with the v2 schema: "
                f"missing={sorted(_HISTORY_STATE_FIELDS - fields)!r}, "
                f"unknown={sorted(fields - _HISTORY_STATE_FIELDS)!r}."
            )
        if value.get("schema") != HISTORY_STATE_SCHEMA:
            raise ValueError(f"OpenPI state history requires schema {HISTORY_STATE_SCHEMA!r}.")
        input_key = str(value.get("input_key") or "").strip()
        age_input_key = str(value.get("age_input_key") or "").strip()
        if input_key != "history_states" or age_input_key != "history_state_ages_sec":
            raise ValueError("OpenPI state history requires the canonical state and age input keys.")
        if value.get("offset_semantics") != "causal_control_frame_offset":
            raise ValueError("OpenPI state history requires causal control-frame offsets.")
        if value.get("selection") != "same_episode_same_continuity_segment_exact_past_frame":
            raise ValueError("OpenPI state history requires exact same-segment selection.")
        if value.get("token_placement") != "pi0_action_expert_prefix":
            raise ValueError("OpenPI state history must condition the action expert.")
        if tuple(value.get("state_layout", ())) != HISTORY_STATE_LAYOUT:
            raise ValueError("OpenPI state history requires the canonical 28-D state layout.")
        if value.get("age_unit") != "seconds":
            raise ValueError("OpenPI state-history ages must use seconds.")
        if value.get("qpos_normalization") != "divide_by_pi":
            raise ValueError("OpenPI state history requires divide-by-pi qpos normalization.")
        if value.get("policy_state_normalization") != "reuse_state_norm_stats":
            raise ValueError("OpenPI state history must reuse policy-state normalization stats.")
        return cls(
            input_key=input_key,
            age_input_key=age_input_key,
            spec=PolicyStateHistorySpec(
                frame_offsets=tuple(int(item) for item in value.get("frame_offsets", ())),
                state_dim=int(value.get("state_dim", 0)),
            ),
        )


class OpenPIStateHistoryRecorder:
    """Capture every control tick and freeze exact history per request anchor."""

    def __init__(self, *, binding: OpenPIStateHistoryBinding, adapter: V22DeploymentAdapter) -> None:
        if not isinstance(binding, OpenPIStateHistoryBinding):
            raise TypeError("binding must be OpenPIStateHistoryBinding.")
        self.binding = binding
        self._raw_state_adapter = V22DeploymentAdapter(
            adapter.layout,
            require_action_layout_hash=adapter.require_action_layout_hash,
            state_input_mode=STATE_INPUT_MODE_RAW,
        )
        self._buffer = PolicyStateHistoryBuffer(binding.spec, retention_steps=128)

    def observe(self, observation: DeploymentObservation) -> None:
        typed = self._raw_state_adapter.observation_from_deployment(observation)
        qpos = []
        for side in ("left", "right"):
            values = np.asarray(
                observation.state[STATE_ARM_QPOS_SLICES[side]],
                dtype=np.float32,
            ).reshape(-1)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise ValueError(f"state-history observation lacks finite {side} qpos.")
            qpos.append(values)
        history_state = np.concatenate((*qpos, typed.state)).astype(np.float32, copy=False)
        self._buffer.append(
            timestamp=observation.timestamp,
            step_index=observation.step_index,
            state=history_state,
        )

    def is_ready(self, observation: DeploymentObservation) -> bool:
        return self._buffer.is_ready(
            anchor_timestamp=observation.timestamp,
            anchor_step_index=observation.step_index,
        )

    def request_payload(self, observation: DeploymentObservation) -> dict[str, np.ndarray]:
        snapshot = self._buffer.snapshot(
            anchor_timestamp=observation.timestamp,
            anchor_step_index=observation.step_index,
        )
        if snapshot is None:
            raise RuntimeError(
                "OpenPI state history is not available for this immutable request anchor."
            )
        return {
            self.binding.input_key: np.asarray(snapshot.states, dtype=np.float32),
            self.binding.age_input_key: np.asarray(snapshot.ages_sec, dtype=np.float32),
        }


def history_binding_from_artifact_metadata(
    metadata: Mapping[str, Any],
) -> OpenPIStateHistoryBinding | None:
    value = metadata.get("history_state_input")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("OpenPI history_state_input metadata must be an object.")
    return OpenPIStateHistoryBinding.from_mapping(value)


__all__ = [
    "HISTORY_STATE_LAYOUT",
    "HISTORY_STATE_SCHEMA",
    "OpenPIStateHistoryBinding",
    "OpenPIStateHistoryRecorder",
    "history_binding_from_artifact_metadata",
]
