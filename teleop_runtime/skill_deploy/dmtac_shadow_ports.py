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

from dataset_contracts.canonical_encoding import canonical_content_hash
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
    RegisteredTrajectoryPointDispatchReceipt,
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

    def __post_init__(self) -> None:
        for name in (
            "maximum_inter_gripper_skew_sec",
            "maximum_arm_gripper_skew_sec",
            "maximum_gripper_hold_drift",
        ):
            value = _finite(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)


class DMTacRM65RegisteredTrajectoryPorts:
    """Exact-qpos replay binding with passive, measured DMTac opening fences."""

    def __init__(
        self,
        *,
        trajectory: RegisteredShadowJointTrajectory,
        identity: ShadowExecutionIdentity,
        rm65_ports: StartedRM65ShadowPorts,
        dmtac_backends_by_side: Mapping[str, MeasuredGripperBackend],
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
        self.trajectory = trajectory
        self.identity = identity
        self.rm65_ports = rm65_ports
        self.dmtac_backends_by_side = backends
        self.profile = profile or DMTacShadowReplayPortProfile()
        self.clock = clock
        self._point_hashes = {
            point.semantic_content_hash for point in self.trajectory.points
        }
        self._cut_lock = threading.Lock()
        self._pending_arm_cut: RealSkillMotionObservationCut | None = None
        self._pending_snapshot: AuthoritativeRobotStateSnapshot | None = None

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
        """Admit and dispatch one exact dual-arm point; never command DMTac."""

        if not isinstance(point, RegisteredJointTrajectoryPoint):
            raise TypeError("point must be RegisteredJointTrajectoryPoint.")
        if trajectory_hash != self.trajectory.semantic_content_hash:
            raise ValueError("trajectory hash does not match the bound trajectory.")
        if point.semantic_content_hash not in self._point_hashes:
            raise ValueError("point is not registered by the bound trajectory.")

        arm_cut, snapshot = self._consume_pending_cut()
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
        now = _finite(self.clock(), name="clock()")
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
                arm_cut.observation,
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
                "snapshot_hash": snapshot.semantic_content_hash,
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
    "StartedRM65ShadowPorts",
]
