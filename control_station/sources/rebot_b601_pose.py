"""Read-only B601-RS joint-state to end-effector-pose source.

This application adapter composes the hardware observation and model packages.
It never connects, enables, disables, or dispatches to the B601-RS.  Each step
uses the backend's host-fresh cache path and emits only after the complete
six-joint snapshot and FK result pass their declared contracts.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_BASE_FRAME,
    B601_RS_END_EFFECTOR_FRAME,
)
from robot_motion.runtime import EffectorCommand, ObservationChannel, TeleopCommand
from robot_motion.runtime.observation_cut import ObservationSampleTracker
from teleop_runtime.action.human_proposal import HumanCommandEmission

B601_READONLY_POSE_SOURCE_CONTRACT_ID_V1 = "control_station_b601_readonly_pose_v1"
B601_READONLY_POSE_SOURCE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.b601_readonly_pose_source.v1"
)
B601_READONLY_POSE_FRAME_CONTRACT_V1 = "b601-rs-base-link-gripper-end-pose-frame-v1"
B601_READONLY_POSE_ACTION_SPEC_V1 = "b601-rs-gripper-end-pose-v1"
B601_READONLY_POSE_UNITS_V1 = "si_m_rad_xyzw_v1"
B601_READONLY_POSE_CLOCK_DOMAIN_V1 = "b601-rs-host-monotonic"
B601_READONLY_FEEDBACK_FRAME_V1 = "b601_rs/vendor_feedback_cache"


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@runtime_checkable
class _B601JointFeedbackSnapshot(Protocol):
    name: str
    frame_id: str
    valid: bool
    stale: bool
    host_receive_freshness_verified: bool
    sample_sequence: int
    status_sequence: int | None
    observed_at_sec: float | None


@runtime_checkable
class _B601FeedbackSnapshot(Protocol):
    sequence: int
    receive_timestamp_sec: float
    joints: tuple[_B601JointFeedbackSnapshot, ...]
    valid: bool
    stale: bool
    freshness_verified: bool
    freshness_semantics: str
    fault_reason: str | None
    host_receive_freshness_verified: bool
    positions_rad: tuple[float, ...] | None


@runtime_checkable
class _B601HostFreshStateReader(Protocol):
    def poll_feedback(
        self,
        *,
        request_feedback: bool = False,
        require_host_freshness: bool = False,
    ) -> _B601FeedbackSnapshot: ...


@runtime_checkable
class _B601ForwardKinematics(Protocol):
    arm_joint_names: tuple[str, ...]
    end_effector_frame: str

    def forward_kinematics(self, q_arm: Sequence[float]) -> object: ...


@dataclass(frozen=True)
class B601ReadOnlyPoseSourceSpec:
    """Source identity before any B601-to-RM65 relative mapping."""

    source_id: str = "b601-rs-readonly-pose"
    effector_name: str = "right_arm"
    reference_frame: str = B601_RS_BASE_FRAME
    pose_frame_contract_hash: str = B601_READONLY_POSE_FRAME_CONTRACT_V1
    action_spec_hash: str = B601_READONLY_POSE_ACTION_SPEC_V1
    units_convention_id: str = B601_READONLY_POSE_UNITS_V1
    clock_domain_id: str = B601_READONLY_POSE_CLOCK_DOMAIN_V1
    contribution_name: str = B601_READONLY_POSE_SOURCE_CONTRACT_ID_V1
    feedback_frame: str = B601_READONLY_FEEDBACK_FRAME_V1

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "effector_name",
            "reference_frame",
            "pose_frame_contract_hash",
            "action_spec_hash",
            "units_convention_id",
            "clock_domain_id",
            "contribution_name",
            "feedback_frame",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        if self.reference_frame != B601_RS_BASE_FRAME:
            raise ValueError(
                "B601 read-only pose source reference_frame must be base_link."
            )
        if self.effector_name not in {"left_arm", "right_arm"}:
            raise ValueError(
                "B601 read-only pose source effector_name must be a canonical arm."
            )

    @property
    def reference_frame_by_effector(self) -> tuple[tuple[str, str], ...]:
        return ((self.effector_name, self.reference_frame),)

    @property
    def action_spec_hash_by_effector(self) -> tuple[tuple[str, str], ...]:
        return ((self.effector_name, self.action_spec_hash),)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "contract_id": B601_READONLY_POSE_SOURCE_CONTRACT_ID_V1,
                "source_id": self.source_id,
                "effector_name": self.effector_name,
                "reference_frame": self.reference_frame,
                "controlled_frame": B601_RS_END_EFFECTOR_FRAME,
                "pose_frame_contract_hash": self.pose_frame_contract_hash,
                "action_spec_hash": self.action_spec_hash,
                "units_convention_id": self.units_convention_id,
                "clock_domain_id": self.clock_domain_id,
                "contribution_name": self.contribution_name,
                "feedback_frame": self.feedback_frame,
            },
            domain=B601_READONLY_POSE_SOURCE_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class B601ReadOnlyPoseStep:
    """One admitted emission or an explicit fail-closed rejection."""

    state: _B601FeedbackSnapshot
    emission: HumanCommandEmission | None
    rejection_reason: str | None
    command_dispatched: bool = False
    authoritative: bool = False
    feedback_poll_elapsed_sec: float | None = None
    fk_elapsed_sec: float | None = None
    emission_elapsed_sec: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, _B601FeedbackSnapshot):
            raise TypeError("state must satisfy the B601 feedback snapshot port.")
        if (self.emission is None) == (self.rejection_reason is None):
            raise ValueError(
                "A B601 pose step must contain exactly one of emission or rejection."
            )
        if self.command_dispatched or self.authoritative:
            raise ValueError(
                "B601 read-only pose steps cannot dispatch or claim authority."
            )

    @property
    def emission_valid(self) -> bool:
        return self.emission is not None


class B601ReadOnlyPoseSource:
    """Project host-fresh B601-RS feedback into one immutable pose emission."""

    def __init__(
        self,
        backend: _B601HostFreshStateReader,
        model: _B601ForwardKinematics,
        spec: B601ReadOnlyPoseSourceSpec | None = None,
    ) -> None:
        if not isinstance(backend, _B601HostFreshStateReader):
            raise TypeError("backend must provide the B601 feedback-reader port.")
        if not isinstance(model, _B601ForwardKinematics):
            raise TypeError("model must provide the B601 FK port.")
        if tuple(model.arm_joint_names) != B601_RS_ARM_JOINT_NAMES:
            raise ValueError("B601 pose model arm-joint order is incompatible.")
        if model.end_effector_frame != B601_RS_END_EFFECTOR_FRAME:
            raise ValueError("B601 pose model end-effector frame is incompatible.")
        self.backend = backend
        self.model = model
        self.spec = B601ReadOnlyPoseSourceSpec() if spec is None else spec
        if not isinstance(self.spec, B601ReadOnlyPoseSourceSpec):
            raise TypeError("spec must be a B601ReadOnlyPoseSourceSpec.")
        self._sample_tracker = ObservationSampleTracker()
        self._next_emission_sequence = 0
        self._last_state_sequence: int | None = None
        self._last_receive_timestamp_sec: float | None = None

    @property
    def next_emission_sequence(self) -> int:
        return self._next_emission_sequence

    def step(self) -> B601ReadOnlyPoseStep:
        """Poll only the host-fresh cache path and emit one admitted FK pose."""

        feedback_started = time.perf_counter()
        state = self.backend.poll_feedback(
            request_feedback=False,
            require_host_freshness=True,
        )
        feedback_poll_elapsed = max(0.0, time.perf_counter() - feedback_started)
        fk_elapsed: float | None = None
        try:
            self._admit_temporal_progress(state)
            qpos, measured_at, status_sequences = self._admit_state(state)
            fk_started = time.perf_counter()
            try:
                pose = self._validated_pose(self.model.forward_kinematics(qpos))
            finally:
                fk_elapsed = max(0.0, time.perf_counter() - fk_started)
        except (TypeError, ValueError, RuntimeError) as error:
            return B601ReadOnlyPoseStep(
                state=state,
                emission=None,
                rejection_reason=f"{type(error).__name__}:{error}",
                feedback_poll_elapsed_sec=feedback_poll_elapsed,
                fk_elapsed_sec=fk_elapsed,
            )

        emission_started = time.perf_counter()
        channel_name = "b601_rs_arm_qpos"
        channel = ObservationChannel(
            channel_name=channel_name,
            capability="joint_position",
            side=None,
            value=qpos,
            frame_name=self.spec.feedback_frame,
            timestamp=measured_at,
            source_name=self.spec.source_id,
            valid=True,
            critical=True,
            metadata={
                "sample_sequence": status_sequences,
                "measured_at": measured_at,
                "measured_clock_domain_id": self.spec.clock_domain_id,
                "host_receive_freshness_verified": True,
                "device_source_timestamp_verified": bool(state.freshness_verified),
                "freshness_semantics": state.freshness_semantics,
            },
        )
        sample_refs = self._sample_tracker.observe(
            {channel_name: channel},
            available_at=state.receive_timestamp_sec,
            available_clock_domain_id=self.spec.clock_domain_id,
        )
        command = TeleopCommand(
            timestamp=state.receive_timestamp_sec,
            metadata={
                "source_contract_hash": self.spec.semantic_content_hash,
                "b601_state_sequence": state.sequence,
                "host_receive_freshness_verified": True,
                "device_source_timestamp_verified": bool(state.freshness_verified),
                "freshness_semantics": state.freshness_semantics,
            },
        )
        command.add_command(
            EffectorCommand(
                effector_name=self.spec.effector_name,
                side=(
                    "left"
                    if self.spec.effector_name == "left_arm"
                    else "right"
                    if self.spec.effector_name == "right_arm"
                    else None
                ),
                command_type="cartesian_pose",
                value=pose,
                frame_name=self.spec.reference_frame,
                metadata={
                    "controlled_frame": B601_RS_END_EFFECTOR_FRAME,
                    "source_state_sequence": state.sequence,
                },
            )
        )
        emission = HumanCommandEmission.capture(
            emission_sequence=self._next_emission_sequence,
            emitted_at=state.receive_timestamp_sec,
            contributions=((self.spec.contribution_name, command),),
            merged_command=command,
            observation_sample_refs=sample_refs,
        )
        self._next_emission_sequence += 1
        emission_elapsed = max(0.0, time.perf_counter() - emission_started)
        return B601ReadOnlyPoseStep(
            state=state,
            emission=emission,
            rejection_reason=None,
            feedback_poll_elapsed_sec=feedback_poll_elapsed,
            fk_elapsed_sec=fk_elapsed,
            emission_elapsed_sec=emission_elapsed,
        )

    def _admit_temporal_progress(self, state: _B601FeedbackSnapshot) -> None:
        if not isinstance(state, _B601FeedbackSnapshot):
            raise TypeError("B601 backend returned an incompatible feedback snapshot.")
        if isinstance(state.sequence, bool) or not isinstance(state.sequence, int):
            raise TypeError("B601 feedback state sequence must be an integer.")
        if state.sequence < 0:
            raise ValueError("B601 feedback state sequence must be non-negative.")
        if not math.isfinite(state.receive_timestamp_sec):
            raise ValueError("B601 feedback receive time must be finite.")
        if (
            self._last_state_sequence is not None
            and state.sequence <= self._last_state_sequence
        ):
            raise ValueError("B601 feedback state sequence did not advance.")
        if (
            self._last_receive_timestamp_sec is not None
            and state.receive_timestamp_sec <= self._last_receive_timestamp_sec
        ):
            raise ValueError("B601 feedback receive time did not advance.")
        self._last_state_sequence = state.sequence
        self._last_receive_timestamp_sec = state.receive_timestamp_sec

    def _admit_state(
        self,
        state: _B601FeedbackSnapshot,
    ) -> tuple[tuple[float, ...], float, tuple[int, ...]]:
        if not isinstance(state, _B601FeedbackSnapshot):
            raise TypeError("B601 backend returned an incompatible feedback snapshot.")
        if not state.valid or state.stale:
            detail = state.fault_reason or "snapshot is invalid or stale"
            raise ValueError(f"B601 feedback rejected: {detail}")
        if not state.host_receive_freshness_verified:
            raise ValueError("B601 feedback lacks verified host receive freshness.")
        joint_names = tuple(joint.name for joint in state.joints)
        if joint_names != B601_RS_ARM_JOINT_NAMES:
            raise ValueError(
                "B601 feedback joint order is incomplete or incompatible: "
                f"{joint_names!r}."
            )
        qpos = state.positions_rad
        if qpos is None:
            raise ValueError("B601 feedback lacks a complete six-joint position.")
        observed_at: list[float] = []
        status_sequences: list[int] = []
        for joint in state.joints:
            if joint.frame_id != self.spec.feedback_frame:
                raise ValueError(
                    f"{joint.name} feedback frame {joint.frame_id!r} is incompatible."
                )
            if (
                not joint.valid
                or joint.stale
                or not joint.host_receive_freshness_verified
            ):
                raise ValueError(f"{joint.name} feedback is not host-fresh and valid.")
            if joint.sample_sequence != state.sequence:
                raise ValueError(f"{joint.name} snapshot sequence is inconsistent.")
            if joint.status_sequence is None:
                raise ValueError(f"{joint.name} lacks a host receive sequence.")
            if isinstance(joint.status_sequence, bool) or not isinstance(
                joint.status_sequence, int
            ):
                raise TypeError(
                    f"{joint.name} host receive sequence must be an integer."
                )
            if joint.observed_at_sec is None or not math.isfinite(
                joint.observed_at_sec
            ):
                raise ValueError(f"{joint.name} lacks a finite observation time.")
            if joint.observed_at_sec > state.receive_timestamp_sec:
                raise ValueError(f"{joint.name} observation time is in the future.")
            observed_at.append(float(joint.observed_at_sec))
            status_sequences.append(int(joint.status_sequence))
        values = tuple(float(value) for value in qpos)
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise ValueError("B601 arm position must be a finite six-vector.")
        return values, min(observed_at), tuple(status_sequences)

    @staticmethod
    def _validated_pose(value: object) -> tuple[tuple[float, ...], ...]:
        try:
            rows = iter(value)
            pose = tuple(tuple(float(component) for component in row) for row in rows)
        except (TypeError, ValueError) as error:
            raise ValueError("B601 FK must return a numeric 4x4 pose.") from error
        if (
            len(pose) != 4
            or any(len(row) != 4 for row in pose)
            or not all(math.isfinite(component) for row in pose for component in row)
        ):
            raise ValueError("B601 FK must return a finite 4x4 pose.")
        if any(
            not math.isclose(actual, expected, abs_tol=1e-9)
            for actual, expected in zip(pose[3], (0.0, 0.0, 0.0, 1.0))
        ):
            raise ValueError("B601 FK pose has an invalid homogeneous row.")
        rotation = tuple(row[:3] for row in pose[:3])
        for column_a in range(3):
            for column_b in range(3):
                dot = sum(
                    rotation[row][column_a] * rotation[row][column_b]
                    for row in range(3)
                )
                expected = 1.0 if column_a == column_b else 0.0
                if not math.isclose(dot, expected, abs_tol=1e-6):
                    raise ValueError("B601 FK pose rotation is not orthonormal.")
        determinant = (
            rotation[0][0]
            * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1]
            * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2]
            * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        if not math.isclose(determinant, 1.0, abs_tol=1e-6):
            raise ValueError("B601 FK pose rotation determinant is not +1.")
        return pose


__all__ = [
    "B601_READONLY_FEEDBACK_FRAME_V1",
    "B601_READONLY_POSE_ACTION_SPEC_V1",
    "B601_READONLY_POSE_CLOCK_DOMAIN_V1",
    "B601_READONLY_POSE_FRAME_CONTRACT_V1",
    "B601_READONLY_POSE_SOURCE_CONTRACT_ID_V1",
    "B601_READONLY_POSE_SOURCE_HASH_DOMAIN_V1",
    "B601_READONLY_POSE_UNITS_V1",
    "B601ReadOnlyPoseSource",
    "B601ReadOnlyPoseSourceSpec",
    "B601ReadOnlyPoseStep",
]
