"""Typed observations for the actual MuJoCo qref→planner→qcmd chain."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.position_control.trajectory import JointTrajectoryProposal


MUJOCO_REFERENCE_ACTIVATION_HASH_DOMAIN_V1 = (
    "dexteleop.mujoco.reference_target_activation.v1"
)
MUJOCO_PLANNER_COMMAND_FRAME_HASH_DOMAIN_V1 = (
    "dexteleop.mujoco.planner_command_frame.v1"
)


def _required_text(value: object, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _generation(value: object, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _qpos_mapping(values: Mapping[str, object], *, name: str):
    result: dict[str, np.ndarray] = {}
    for side, raw in sorted(values.items()):
        key = _required_text(side, name="side")
        qpos = np.asarray(raw, dtype=float).reshape(-1).copy()
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise ValueError(f"{name}[{key!r}] must contain six finite values.")
        qpos.setflags(write=False)
        result[key] = qpos
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return MappingProxyType(result)


def _qpos_payload(values: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    return {side: np.asarray(value, dtype=float).tolist() for side, value in values.items()}


@dataclass(frozen=True)
class MujocoReferenceTargetActivationObservation:
    observation_id: str
    reference_segment_id: str
    generation_activation_id: str
    reference_sequence: int
    execution_epoch: int
    group_segment_generation: int
    scheduled_for: float
    activated_at_worker_time: float
    observed_at: float
    qref_by_side: Mapping[str, np.ndarray]
    activation_status_by_side: Mapping[str, str]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "reference_segment_id",
            "generation_activation_id",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        for name in (
            "reference_sequence",
            "execution_epoch",
            "group_segment_generation",
        ):
            object.__setattr__(self, name, _generation(getattr(self, name), name=name))
        for name in ("scheduled_for", "activated_at_worker_time", "observed_at"):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "qref_by_side",
            _qpos_mapping(self.qref_by_side, name="qref_by_side"),
        )
        object.__setattr__(
            self,
            "activation_status_by_side",
            MappingProxyType(
                {
                    _required_text(side, name="side"): _required_text(
                        status, name="activation_status"
                    )
                    for side, status in sorted(self.activation_status_by_side.items())
                }
            ),
        )


def build_mujoco_reference_target_activation_observation(
    *,
    reference_segment_id: str,
    generation_activation_id: str,
    reference_sequence: int,
    execution_epoch: int,
    group_segment_generation: int,
    scheduled_for: float,
    activated_at_worker_time: float,
    observed_at: float,
    qref_by_side: Mapping[str, object],
    activation_status_by_side: Mapping[str, str],
) -> MujocoReferenceTargetActivationObservation:
    frozen_qref = _qpos_mapping(qref_by_side, name="qref_by_side")
    payload = {
        "reference_segment_id": _required_text(
            reference_segment_id, name="reference_segment_id"
        ),
        "generation_activation_id": _required_text(
            generation_activation_id, name="generation_activation_id"
        ),
        "reference_sequence": _generation(
            reference_sequence, name="reference_sequence"
        ),
        "execution_epoch": _generation(execution_epoch, name="execution_epoch"),
        "group_segment_generation": _generation(
            group_segment_generation, name="group_segment_generation"
        ),
        "scheduled_for": _finite(scheduled_for, name="scheduled_for"),
        "activated_at_worker_time": _finite(
            activated_at_worker_time, name="activated_at_worker_time"
        ),
        "observed_at": _finite(observed_at, name="observed_at"),
        "qref_by_side": _qpos_payload(frozen_qref),
        "activation_status_by_side": dict(sorted(activation_status_by_side.items())),
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=MUJOCO_REFERENCE_ACTIVATION_HASH_DOMAIN_V1,
    )
    return MujocoReferenceTargetActivationObservation(
        observation_id=f"mujoco-reference-target-activation:{semantic_hash}",
        reference_segment_id=payload["reference_segment_id"],
        generation_activation_id=payload["generation_activation_id"],
        reference_sequence=payload["reference_sequence"],
        execution_epoch=payload["execution_epoch"],
        group_segment_generation=payload["group_segment_generation"],
        scheduled_for=payload["scheduled_for"],
        activated_at_worker_time=payload["activated_at_worker_time"],
        observed_at=payload["observed_at"],
        qref_by_side=frozen_qref,
        activation_status_by_side=payload["activation_status_by_side"],
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class MujocoPlannerCommandFrameObservation:
    observation_id: str
    reference_segment_id: str
    reference_sequence: int
    command_sequence: int
    execution_epoch: int
    group_segment_generation: int
    scheduled_for: float
    observed_at: float
    planner_proposal_token_by_side: Mapping[str, int]
    planner_generation_by_side: Mapping[str, int]
    qref_by_side: Mapping[str, np.ndarray]
    planner_qcmd_by_side: Mapping[str, np.ndarray]
    follow_qcmd_by_side: Mapping[str, np.ndarray]
    accepted_qcmd_by_side: Mapping[str, np.ndarray]
    limited_by_side: Mapping[str, tuple[str, ...]]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in ("observation_id", "reference_segment_id", "semantic_content_hash"):
            _required_text(getattr(self, name), name=name)
        for name in (
            "reference_sequence",
            "command_sequence",
            "execution_epoch",
            "group_segment_generation",
        ):
            object.__setattr__(self, name, _generation(getattr(self, name), name=name))
        for name in ("scheduled_for", "observed_at"):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        for name in (
            "qref_by_side",
            "planner_qcmd_by_side",
            "follow_qcmd_by_side",
            "accepted_qcmd_by_side",
        ):
            object.__setattr__(self, name, _qpos_mapping(getattr(self, name), name=name))
        for name in ("planner_proposal_token_by_side", "planner_generation_by_side"):
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        _required_text(side, name="side"): _generation(
                            value, name=name
                        )
                        for side, value in sorted(getattr(self, name).items())
                    }
                ),
            )
        object.__setattr__(
            self,
            "limited_by_side",
            MappingProxyType(
                {
                    _required_text(side, name="side"): tuple(
                        _required_text(value, name="limit_kind") for value in values
                    )
                    for side, values in sorted(self.limited_by_side.items())
                }
            ),
        )


def build_mujoco_planner_command_frame_observation(
    *,
    reference_segment_id: str,
    reference_sequence: int,
    command_sequence: int,
    execution_epoch: int,
    group_segment_generation: int,
    scheduled_for: float,
    observed_at: float,
    qref_by_side: Mapping[str, object],
    proposals_by_side: Mapping[str, JointTrajectoryProposal],
    follow_qcmd_by_side: Mapping[str, object],
    accepted_qcmd_by_side: Mapping[str, object],
) -> MujocoPlannerCommandFrameObservation:
    if not proposals_by_side:
        raise ValueError("proposals_by_side must not be empty.")
    planner_qcmd = {
        side: proposal.q_command for side, proposal in proposals_by_side.items()
    }
    limited = {
        side: tuple(
            name
            for name, active in (
                ("acceleration", proposal.acceleration_limited),
                ("velocity", proposal.velocity_limited),
                ("command_step", proposal.command_step_limited),
                ("safety_envelope", proposal.safety_envelope_limited),
                ("quick_stop", proposal.quick_stop),
            )
            if active
        )
        for side, proposal in proposals_by_side.items()
    }
    frozen = {
        "qref_by_side": _qpos_mapping(qref_by_side, name="qref_by_side"),
        "planner_qcmd_by_side": _qpos_mapping(
            planner_qcmd, name="planner_qcmd_by_side"
        ),
        "follow_qcmd_by_side": _qpos_mapping(
            follow_qcmd_by_side, name="follow_qcmd_by_side"
        ),
        "accepted_qcmd_by_side": _qpos_mapping(
            accepted_qcmd_by_side, name="accepted_qcmd_by_side"
        ),
    }
    payload = {
        "reference_segment_id": _required_text(
            reference_segment_id, name="reference_segment_id"
        ),
        "reference_sequence": _generation(
            reference_sequence, name="reference_sequence"
        ),
        "command_sequence": _generation(command_sequence, name="command_sequence"),
        "execution_epoch": _generation(execution_epoch, name="execution_epoch"),
        "group_segment_generation": _generation(
            group_segment_generation, name="group_segment_generation"
        ),
        "scheduled_for": _finite(scheduled_for, name="scheduled_for"),
        "observed_at": _finite(observed_at, name="observed_at"),
        "planner_proposal_token_by_side": {
            side: int(proposal.token) for side, proposal in proposals_by_side.items()
        },
        "planner_generation_by_side": {
            side: int(proposal.generation) for side, proposal in proposals_by_side.items()
        },
        **{name: _qpos_payload(value) for name, value in frozen.items()},
        "limited_by_side": limited,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=MUJOCO_PLANNER_COMMAND_FRAME_HASH_DOMAIN_V1,
    )
    return MujocoPlannerCommandFrameObservation(
        observation_id=f"mujoco-planner-command-frame:{semantic_hash}",
        reference_segment_id=payload["reference_segment_id"],
        reference_sequence=payload["reference_sequence"],
        command_sequence=payload["command_sequence"],
        execution_epoch=payload["execution_epoch"],
        group_segment_generation=payload["group_segment_generation"],
        scheduled_for=payload["scheduled_for"],
        observed_at=payload["observed_at"],
        planner_proposal_token_by_side=payload["planner_proposal_token_by_side"],
        planner_generation_by_side=payload["planner_generation_by_side"],
        qref_by_side=frozen["qref_by_side"],
        planner_qcmd_by_side=frozen["planner_qcmd_by_side"],
        follow_qcmd_by_side=frozen["follow_qcmd_by_side"],
        accepted_qcmd_by_side=frozen["accepted_qcmd_by_side"],
        limited_by_side=limited,
        semantic_content_hash=semantic_hash,
    )


__all__ = [
    "MujocoPlannerCommandFrameObservation",
    "MujocoReferenceTargetActivationObservation",
    "build_mujoco_planner_command_frame_observation",
    "build_mujoco_reference_target_activation_observation",
]
