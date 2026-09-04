"""DMTac-aware lower ports for exact registered shadow replay.

This adapter composes already-started RM65 Skill ports with already-started
DMTac backends.  It never constructs or starts a device and deliberately does
not command either gripper: the first real acceptance stage keeps the measured
opening fixed, monitors it on every causal state cut, and fails closed through
the existing dual-arm HOLD owner when that opening drifts.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.pose import pose_delta_metrics
from robot_motion.runtime import EffectorCommand, TeleopCommand, TeleopObservation
from robot_motion.shadow_execution import (
    AuthoritativeRobotStateSnapshot,
    RegisteredJointTrajectoryPoint,
    RegisteredShadowJointTrajectory,
    ShadowExecutionIdentity,
)

from .control_runtime import (
    RealSkillMotionDispatchReceipt,
    RealSkillMotionHoldReceipt,
    RealSkillMotionObservationCut,
)
from .motion_port import REAL_AGENT_MOTION_COMMAND_METADATA_KEY
from .registered_trajectory import (
    MeasuredGripperBackend,
    RegisteredTrajectoryAuthorityConsumptionReceipt,
    RegisteredTrajectoryPointDispatchReceipt,
    TerminalToolTcpVerificationReceipt,
    TerminalToolTcpVerificationStatus,
    capture_authoritative_shadow_state_from_backends,
    read_measured_gripper_state_cut,
)

_SIDES = ("left", "right")


class StartedRM65ShadowPorts(Protocol):
    """Already-owned RM65 ports; lifecycle remains with their application."""

    def observe(self) -> RealSkillMotionObservationCut: ...

    def admit_command(
        self,
        command: TeleopCommand,
        observation: TeleopObservation,
        now: float,
    ) -> bool: ...

    def dispatch(self, command: TeleopCommand) -> RealSkillMotionDispatchReceipt: ...

    def hold(self, reason: str) -> RealSkillMotionHoldReceipt: ...


class RM65ToolTcpWorldFk(Protocol):
    """The exact lower target-builder FK already bound to the RM65 tool TCP."""

    def tool_tcp_world_targets_for_qpos_pair(
        self,
        *,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
    ) -> Mapping[str, np.ndarray]: ...


class RegisteredTrajectoryToolTcpEnvelopeProjector(Protocol):
    """Project one tool-TCP world pose through the active lower envelope."""

    def __call__(self, side: str, tool_tcp_world: np.ndarray, /) -> np.ndarray: ...


class RegisteredTrajectoryTcpEnvelopeError(RuntimeError):
    """Typed zero-motion rejection for a registered path outside the TCP envelope."""

    reason_code = "registered_trajectory_tcp_envelope_blocked"
    real_motion_may_have_executed = False

    def __init__(self, *, details: Mapping[str, object]) -> None:
        self.details = dict(details)
        super().__init__(self.reason_code)


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class DMTacShadowReplayPortProfile:
    """Trusted fixed-opening limits; none are accepted from an Agent request."""

    maximum_inter_gripper_skew_sec: float = 0.05
    maximum_arm_gripper_skew_sec: float = 0.1
    maximum_gripper_hold_drift: float = 0.02
    terminal_tcp_position_tolerance_m: float = 0.002
    terminal_tcp_orientation_tolerance_deg: float = 0.5
    require_trajectory_tcp_envelope: bool = False

    def __post_init__(self) -> None:
        for name in (
            "maximum_inter_gripper_skew_sec",
            "maximum_arm_gripper_skew_sec",
            "maximum_gripper_hold_drift",
            "terminal_tcp_position_tolerance_m",
            "terminal_tcp_orientation_tolerance_deg",
        ):
            value = _finite(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)
        if not isinstance(self.require_trajectory_tcp_envelope, bool):
            raise TypeError("require_trajectory_tcp_envelope must be boolean.")


class DMTacRM65RegisteredTrajectoryPorts:
    """Exact-qpos replay binding with passive, measured DMTac opening fences."""

    def __init__(
        self,
        *,
        trajectory: RegisteredShadowJointTrajectory,
        identity: ShadowExecutionIdentity,
        rm65_ports: StartedRM65ShadowPorts,
        dmtac_backends_by_side: Mapping[str, MeasuredGripperBackend],
        tool_tcp_world_fk: RM65ToolTcpWorldFk,
        target_tool_tcp_world_by_side: Mapping[str, np.ndarray],
        target_reference_frame_id: str,
        tool_tcp_world_envelope_projector: (
            RegisteredTrajectoryToolTcpEnvelopeProjector | None
        ) = None,
        profile: DMTacShadowReplayPortProfile | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(trajectory, RegisteredShadowJointTrajectory):
            raise TypeError("trajectory must be RegisteredShadowJointTrajectory.")
        if not isinstance(identity, ShadowExecutionIdentity):
            raise TypeError("identity must be ShadowExecutionIdentity.")
        for name in ("observe", "admit_command", "dispatch", "hold"):
            if not callable(getattr(rm65_ports, name, None)):
                raise TypeError(f"rm65_ports must expose {name}().")
        backends = dict(dmtac_backends_by_side)
        if set(backends) != set(_SIDES):
            raise ValueError(
                "dmtac_backends_by_side must contain exactly left and right."
            )
        if not callable(clock):
            raise TypeError("clock must be callable.")
        fk_method = getattr(
            tool_tcp_world_fk,
            "tool_tcp_world_targets_for_qpos_pair",
            None,
        )
        if not callable(fk_method):
            raise TypeError(
                "tool_tcp_world_fk must expose "
                "tool_tcp_world_targets_for_qpos_pair()."
            )
        targets = {
            str(side).strip().lower(): self._pose(
                value,
                name=f"target_tool_tcp_world_by_side.{side}",
            )
            for side, value in dict(target_tool_tcp_world_by_side).items()
        }
        if set(targets) != set(trajectory.moving_sides):
            raise ValueError(
                "terminal TCP targets must contain exactly trajectory.moving_sides."
            )
        frame_id = str(target_reference_frame_id).strip()
        if not frame_id:
            raise ValueError("target_reference_frame_id must be non-empty.")
        self.trajectory = trajectory
        self.identity = identity
        self.rm65_ports = rm65_ports
        self.dmtac_backends_by_side = backends
        self.tool_tcp_world_fk = tool_tcp_world_fk
        self.target_tool_tcp_world_by_side = targets
        self.target_reference_frame_id = frame_id
        self.profile = profile or DMTacShadowReplayPortProfile()
        if (
            self.profile.require_trajectory_tcp_envelope
            and not callable(tool_tcp_world_envelope_projector)
        ):
            raise RegisteredTrajectoryTcpEnvelopeError(
                details={"reason": "active_tcp_envelope_projector_missing"}
            )
        self.tool_tcp_world_envelope_projector = tool_tcp_world_envelope_projector
        self.clock = clock
        self._point_hashes = {
            point.semantic_content_hash for point in self.trajectory.points
        }
        self._cut_lock = threading.Lock()
        self._execution_lock = threading.RLock()
        self._pending_arm_cut: RealSkillMotionObservationCut | None = None
        self._pending_snapshot: AuthoritativeRobotStateSnapshot | None = None
        self._active_consumption: (
            RegisteredTrajectoryAuthorityConsumptionReceipt | None
        ) = None
        self._next_point_index = 0
        self._validate_registered_trajectory_tcp_envelope()

    def _validate_registered_trajectory_tcp_envelope(self) -> None:
        """Reject the complete path before replay if Active safety would alter it."""

        projector = self.tool_tcp_world_envelope_projector
        if projector is None:
            return
        for point_index, point in enumerate(self.trajectory.points):
            observed_raw = self.tool_tcp_world_fk.tool_tcp_world_targets_for_qpos_pair(
                left_qpos=point.qpos_by_side["left"],
                right_qpos=point.qpos_by_side["right"],
            )
            if not isinstance(observed_raw, Mapping):
                raise RegisteredTrajectoryTcpEnvelopeError(
                    details={
                        "reason": "tool_tcp_fk_not_mapping",
                        "point_index": point_index,
                    }
                )
            for side in self.trajectory.moving_sides:
                if side not in observed_raw:
                    raise RegisteredTrajectoryTcpEnvelopeError(
                        details={
                            "reason": "tool_tcp_fk_side_missing",
                            "point_index": point_index,
                            "side": side,
                        }
                    )
                requested = self._pose(
                    observed_raw[side],
                    name=f"trajectory_tool_tcp_world.{point_index}.{side}",
                )
                try:
                    projected = self._pose(
                        projector(side, requested),
                        name=f"projected_tool_tcp_world.{point_index}.{side}",
                    )
                except RegisteredTrajectoryTcpEnvelopeError:
                    raise
                except Exception as exc:
                    raise RegisteredTrajectoryTcpEnvelopeError(
                        details={
                            "reason": "active_tcp_envelope_projection_failed",
                            "point_index": point_index,
                            "side": side,
                            "exception_type": type(exc).__name__,
                            "exception": str(exc),
                        }
                    ) from exc
                position_error_m, orientation_error_deg = pose_delta_metrics(
                    requested,
                    projected,
                )
                if position_error_m > 1e-9 or orientation_error_deg > 1e-7:
                    raise RegisteredTrajectoryTcpEnvelopeError(
                        details={
                            "reason": "active_tcp_envelope_would_modify_path",
                            "point_index": point_index,
                            "side": side,
                            "time_from_start_sec": point.time_from_start_sec,
                            "projection_position_delta_m": position_error_m,
                            "projection_orientation_delta_deg": orientation_error_deg,
                        }
                    )

    @staticmethod
    def _pose(value: object, *, name: str) -> np.ndarray:
        pose = np.asarray(value, dtype=float).copy()
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError(f"{name} must be a finite 4x4 pose.")
        if (
            not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9)
            or not np.allclose(
                pose[:3, :3].T @ pose[:3, :3],
                np.eye(3),
                atol=1e-6,
            )
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6)
        ):
            raise ValueError(f"{name} must be a rigid pose.")
        pose.setflags(write=False)
        return pose

    def verify_terminal_tool_tcp(
        self,
        snapshot: AuthoritativeRobotStateSnapshot,
    ) -> TerminalToolTcpVerificationReceipt:
        """Evaluate the original requested tool TCP from the measured qpos cut."""

        if not isinstance(snapshot, AuthoritativeRobotStateSnapshot):
            raise TypeError("snapshot must be AuthoritativeRobotStateSnapshot.")
        if (
            snapshot.identity.semantic_content_hash
            != self.trajectory.initial_snapshot.identity.semantic_content_hash
        ):
            raise RuntimeError("terminal TCP snapshot identity changed.")
        observed_raw = self.tool_tcp_world_fk.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=snapshot.qpos_by_side["left"],
            right_qpos=snapshot.qpos_by_side["right"],
        )
        if not isinstance(observed_raw, Mapping):
            raise TypeError("tool TCP FK must return a side mapping.")
        position: dict[str, float] = {}
        orientation: dict[str, float] = {}
        for side in self.trajectory.moving_sides:
            if side not in observed_raw:
                raise RuntimeError(f"tool TCP FK is missing {side}.")
            observed = self._pose(
                observed_raw[side],
                name=f"observed_tool_tcp_world_by_side.{side}",
            )
            position[side], orientation[side] = pose_delta_metrics(
                observed,
                self.target_tool_tcp_world_by_side[side],
            )
        within = all(
            position[side]
            <= self.profile.terminal_tcp_position_tolerance_m
            and orientation[side]
            <= self.profile.terminal_tcp_orientation_tolerance_deg
            for side in position
        )
        return TerminalToolTcpVerificationReceipt(
            status=(
                TerminalToolTcpVerificationStatus.VERIFIED
                if within
                else TerminalToolTcpVerificationStatus.TARGET_MISMATCH
            ),
            frame_id=self.target_reference_frame_id,
            observed_snapshot_hash=snapshot.semantic_content_hash,
            position_error_m_by_side=position,
            orientation_error_deg_by_side=orientation,
            position_tolerance_m=(
                self.profile.terminal_tcp_position_tolerance_m
            ),
            orientation_tolerance_deg=(
                self.profile.terminal_tcp_orientation_tolerance_deg
            ),
            evidence_refs=snapshot.arm_evidence_refs,
        )

    def begin_replay_session(
        self,
        consumption: RegisteredTrajectoryAuthorityConsumptionReceipt,
    ) -> tuple[str, ...]:
        """Open one ordered replay session after exact authority consumption."""

        if not isinstance(
            consumption, RegisteredTrajectoryAuthorityConsumptionReceipt
        ):
            raise TypeError(
                "consumption must be RegisteredTrajectoryAuthorityConsumptionReceipt."
            )
        if (
            not consumption.consumed
            or consumption.trajectory_hash != self.trajectory.semantic_content_hash
        ):
            raise RuntimeError("authority consumption does not bind this trajectory.")
        with self._execution_lock:
            if self._active_consumption is not None:
                raise RuntimeError("a registered replay session is already active.")
            self._active_consumption = consumption
            self._next_point_index = 0
            evidence_hash = canonical_content_hash(
                {
                    "consumption_hash": consumption.semantic_content_hash,
                    "trajectory_hash": self.trajectory.semantic_content_hash,
                    "run_id": consumption.run_id,
                    "owner_epoch": consumption.owner_epoch,
                },
                domain="agenticrobot.shadow_execution.dmtac_replay_session.v1",
            )
            return ("dmtac-replay-session-begin:" + evidence_hash,)

    def end_replay_session(self, reason: str) -> tuple[str, ...]:
        """Close the active session and invalidate its point cursor."""

        with self._execution_lock:
            consumption = self._active_consumption
            self._active_consumption = None
            point_index = self._next_point_index
            self._next_point_index = 0
            evidence_hash = canonical_content_hash(
                {
                    "reason": str(reason),
                    "consumption_hash": (
                        None
                        if consumption is None
                        else consumption.semantic_content_hash
                    ),
                    "next_point_index": point_index,
                },
                domain="agenticrobot.shadow_execution.dmtac_replay_session.v1",
            )
            return ("dmtac-replay-session-end:" + evidence_hash,)

    def _read_cut(
        self,
    ) -> tuple[RealSkillMotionObservationCut, AuthoritativeRobotStateSnapshot]:
        arm_cut = self.rm65_ports.observe()
        if not isinstance(arm_cut, RealSkillMotionObservationCut):
            raise TypeError("rm65_ports.observe() must return an observation cut.")
        snapshot = capture_authoritative_shadow_state_from_backends(
            arm_observation_cut=arm_cut,
            gripper_backends_by_side=self.dmtac_backends_by_side,
            identity=self.identity,
            clock=self.clock,
            maximum_inter_side_skew_sec=(
                self.profile.maximum_inter_gripper_skew_sec
            ),
            maximum_arm_gripper_skew_sec=(
                self.profile.maximum_arm_gripper_skew_sec
            ),
        )
        return arm_cut, snapshot

    def current_snapshot(self) -> AuthoritativeRobotStateSnapshot:
        """Return one joined, fresh arm+DMTac cut for replay admission/feedback."""

        arm_cut, snapshot = self._read_cut()
        with self._cut_lock:
            self._pending_arm_cut = arm_cut
            self._pending_snapshot = snapshot
        return snapshot

    def _consume_pending_cut(
        self,
    ) -> tuple[RealSkillMotionObservationCut, AuthoritativeRobotStateSnapshot]:
        with self._cut_lock:
            arm_cut = self._pending_arm_cut
            snapshot = self._pending_snapshot
            self._pending_arm_cut = None
            self._pending_snapshot = None
        if arm_cut is None or snapshot is None:
            return self._read_cut()
        return arm_cut, snapshot

    def dispatch_point(
        self,
        point: RegisteredJointTrajectoryPoint,
        trajectory_hash: str,
        /,
    ) -> RegisteredTrajectoryPointDispatchReceipt:
        """Dispatch the next point after a synchronous measured-state cut."""

        return self._dispatch_registered_point(
            point,
            trajectory_hash,
            synchronous_state_fence=True,
        )

    def dispatch_point_realtime(
        self,
        point: RegisteredJointTrajectoryPoint,
        trajectory_hash: str,
        /,
    ) -> RegisteredTrajectoryPointDispatchReceipt:
        """Dispatch at the registered lattice through RM65 lower health gates.

        Full RM65+DMTac measurements are performed at admission and terminal
        tracking by the replay executor. This path deliberately does not put
        serial gripper reads in the 60 Hz arm-reference loop.
        """

        return self._dispatch_registered_point(
            point,
            trajectory_hash,
            synchronous_state_fence=False,
        )

    def _dispatch_registered_point(
        self,
        point: RegisteredJointTrajectoryPoint,
        trajectory_hash: str,
        *,
        synchronous_state_fence: bool,
    ) -> RegisteredTrajectoryPointDispatchReceipt:
        """Dispatch only the next point of one active authority-bound session."""

        if not isinstance(point, RegisteredJointTrajectoryPoint):
            raise TypeError("point must be RegisteredJointTrajectoryPoint.")
        with self._execution_lock:
            expected = (
                self.trajectory.points[self._next_point_index]
                if self._active_consumption is not None
                and self._next_point_index < len(self.trajectory.points)
                else None
            )
            if (
                expected is None
                or point.semantic_content_hash != expected.semantic_content_hash
                or trajectory_hash != self.trajectory.semantic_content_hash
            ):
                evidence_hash = canonical_content_hash(
                    {
                        "session_active": self._active_consumption is not None,
                        "next_point_index": self._next_point_index,
                        "expected_point_hash": (
                            None
                            if expected is None
                            else expected.semantic_content_hash
                        ),
                        "received_point_hash": point.semantic_content_hash,
                        "trajectory_hash_matches": (
                            trajectory_hash
                            == self.trajectory.semantic_content_hash
                        ),
                    },
                    domain="agenticrobot.shadow_execution.dmtac_session_reject.v1",
                )
                return RegisteredTrajectoryPointDispatchReceipt(
                    point_hash=point.semantic_content_hash,
                    succeeded=False,
                    evidence_refs=("dmtac-replay-session-reject:" + evidence_hash,),
                )
            receipt = self._dispatch_point_in_session(
                point,
                trajectory_hash,
                synchronous_state_fence=synchronous_state_fence,
            )
            if receipt.succeeded:
                self._next_point_index += 1
            return receipt

    def _dispatch_point_in_session(
        self,
        point: RegisteredJointTrajectoryPoint,
        trajectory_hash: str,
        /,
        *,
        synchronous_state_fence: bool,
    ) -> RegisteredTrajectoryPointDispatchReceipt:
        """Admit and dispatch one exact dual-arm point; never command DMTac."""

        if not isinstance(point, RegisteredJointTrajectoryPoint):
            raise TypeError("point must be RegisteredJointTrajectoryPoint.")
        if trajectory_hash != self.trajectory.semantic_content_hash:
            raise ValueError("trajectory hash does not match the bound trajectory.")
        if point.semantic_content_hash not in self._point_hashes:
            raise ValueError("point is not registered by the bound trajectory.")

        now = _finite(self.clock(), name="clock()")
        snapshot_hash: str | None = None
        if synchronous_state_fence:
            arm_cut, snapshot = self._consume_pending_cut()
            observation = arm_cut.observation
            snapshot_hash = snapshot.semantic_content_hash
            identity_matches = (
                snapshot.identity.semantic_content_hash
                == self.trajectory.initial_snapshot.identity.semantic_content_hash
            )
            maximum_gripper_drift = max(
                abs(
                    snapshot.gripper_close_by_side[side]
                    - self.trajectory.initial_snapshot.gripper_close_by_side[side]
                )
                for side in _SIDES
            )
        else:
            observation = TeleopObservation(timestamp=now)
            identity_matches = (
                self.identity.semantic_content_hash
                == self.trajectory.initial_snapshot.identity.semantic_content_hash
            )
            maximum_gripper_drift = 0.0
        command = TeleopCommand(timestamp=now)
        command.metadata[REAL_AGENT_MOTION_COMMAND_METADATA_KEY] = {
            "operation_id": "registered-shadow:" + trajectory_hash,
            "trajectory_hash": trajectory_hash,
            "point_hash": point.semantic_content_hash,
            "gripper_policy": "measured_dmtac_fixed_opening_no_command",
        }
        for side in _SIDES:
            command.add_command(
                EffectorCommand(
                    effector_name=f"{side}_arm",
                    side=side,
                    command_type="joint_vector",
                    value=point.qpos_by_side[side].copy(),
                )
            )

        admitted = bool(
            identity_matches
            and maximum_gripper_drift
            <= self.profile.maximum_gripper_hold_drift
            and self.rm65_ports.admit_command(
                command,
                observation,
                now,
            )
        )
        lower = (
            self.rm65_ports.dispatch(command)
            if admitted
            else RealSkillMotionDispatchReceipt(succeeded=False, evidence_refs=())
        )
        if not isinstance(lower, RealSkillMotionDispatchReceipt):
            raise TypeError("rm65_ports.dispatch() must return a dispatch receipt.")
        evidence_hash = canonical_content_hash(
            {
                "trajectory_hash": trajectory_hash,
                "point_hash": point.semantic_content_hash,
                "admitted": bool(admitted),
                "lower_succeeded": lower.succeeded,
                "identity_matches": identity_matches,
                "maximum_gripper_drift": maximum_gripper_drift,
                "snapshot_hash": snapshot_hash,
                "state_fence": (
                    "synchronous_rm65_dmtac_cut"
                    if synchronous_state_fence
                    else "rm65_lower_health_no_per_point_dmtac_read"
                ),
                "gripper_policy": "measured_dmtac_fixed_opening_no_command",
            },
            domain="agenticrobot.shadow_execution.dmtac_rm65_dispatch.v1",
        )
        return RegisteredTrajectoryPointDispatchReceipt(
            point_hash=point.semantic_content_hash,
            succeeded=bool(admitted and lower.succeeded),
            evidence_refs=(
                *lower.evidence_refs,
                "dmtac-passive-fence-dispatch:" + evidence_hash,
            ),
        )

    def hold(self, reason: str) -> RealSkillMotionHoldReceipt:
        with self._execution_lock:
            return self._hold_locked(reason)

    def _hold_locked(self, reason: str) -> RealSkillMotionHoldReceipt:
        """HOLD both arms and verify DMTac still matches its measured start cut."""

        arm_hold = self.rm65_ports.hold(str(reason))
        if not isinstance(arm_hold, RealSkillMotionHoldReceipt):
            raise TypeError("rm65_ports.hold() must return a hold receipt.")
        gripper_ok = False
        gripper_evidence: tuple[str, ...] = ()
        error_type: str | None = None
        maximum_drift: float | None = None
        try:
            cut = read_measured_gripper_state_cut(
                backends_by_side=self.dmtac_backends_by_side,
                clock=self.clock,
                maximum_inter_side_skew_sec=(
                    self.profile.maximum_inter_gripper_skew_sec
                ),
            )
            maximum_drift = max(
                abs(
                    cut.close_by_side[side]
                    - self.trajectory.initial_snapshot.gripper_close_by_side[side]
                )
                for side in _SIDES
            )
            gripper_ok = maximum_drift <= self.profile.maximum_gripper_hold_drift
            gripper_evidence = (
                *cut.evidence_refs,
                "measured-gripper-cut:" + cut.semantic_content_hash,
            )
        except Exception as exc:  # noqa: BLE001 - compile a failed HOLD receipt.
            error_type = type(exc).__name__
        evidence_hash = canonical_content_hash(
            {
                "reason": str(reason),
                "arm_hold_succeeded": arm_hold.succeeded,
                "gripper_observation_succeeded": gripper_ok,
                "maximum_gripper_drift": maximum_drift,
                "error_type": error_type,
                "gripper_command_dispatched": False,
            },
            domain="agenticrobot.shadow_execution.dmtac_rm65_hold.v1",
        )
        return RealSkillMotionHoldReceipt(
            succeeded=bool(arm_hold.succeeded and gripper_ok),
            evidence_refs=(
                *arm_hold.evidence_refs,
                *gripper_evidence,
                "dmtac-passive-fence-hold:" + evidence_hash,
            ),
        )


__all__ = [
    "DMTacRM65RegisteredTrajectoryPorts",
    "DMTacShadowReplayPortProfile",
    "RM65ToolTcpWorldFk",
    "RegisteredTrajectoryTcpEnvelopeError",
    "RegisteredTrajectoryToolTcpEnvelopeProjector",
    "StartedRM65ShadowPorts",
]
