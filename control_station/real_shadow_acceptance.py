"""Application-level real-state shadow acceptance for DMTac + RM65.

The controller owns sequencing only.  It receives one authoritative measured
state cut, builds and executes the exact registered trajectory in MuJoCo, and
stops there by default.  Optional real replay is delegated to the existing
single-owner lower executor and remains gated by an exact, short-lived,
single-use authority.  This module never constructs, starts, or closes a real
device.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.shadow_execution import (
    AuthoritativeRobotStateSnapshot,
    RegisteredShadowJointTrajectory,
    ShadowSimulationReceipt,
    ShadowSimulationStatus,
)
from teleop_runtime.skill_deploy.control_runtime import RealSkillMotionHoldReceipt
from teleop_runtime.skill_deploy.registered_trajectory import (
    RegisteredTrajectoryAuthorityConsumptionReceipt,
    RegisteredTrajectoryExecutionAuthority,
    RegisteredTrajectoryRealReplayReceipt,
    RegisteredTrajectoryRealReplayStatus,
    execute_registered_shadow_trajectory_with_ports,
)


class RealShadowAcceptanceStatus(str, Enum):
    CAPTURE_FAILED = "capture_failed"
    TRAJECTORY_PREPARATION_FAILED = "trajectory_preparation_failed"
    SIMULATION_INITIALIZATION_FAILED = "simulation_initialization_failed"
    MODEL_BINDING_FAILED = "model_binding_failed"
    SIMULATION_EXECUTION_FAILED = "simulation_execution_failed"
    SIMULATION_SHUTDOWN_FAILED = "simulation_shutdown_failed"
    SIMULATION_REJECTED = "simulation_rejected"
    SIMULATION_ACCEPTED_NO_MOTION = "simulation_accepted_no_motion"
    REAL_REPLAY_REJECTED = "real_replay_rejected"
    REAL_REPLAY_FAILED = "real_replay_failed"
    REAL_REPLAY_COMPLETED = "real_replay_completed"


class RealMotionDisposition(str, Enum):
    NONE = "none"
    EXECUTED = "executed"
    UNKNOWN = "unknown"


class RegisteredTrajectoryReplayPorts(Protocol):
    """Already-owned real replay ports; lifecycle stays with their application."""

    def current_snapshot(self) -> AuthoritativeRobotStateSnapshot: ...

    def begin_replay_session(
        self,
        consumption: RegisteredTrajectoryAuthorityConsumptionReceipt,
    ) -> tuple[str, ...]: ...

    def end_replay_session(self, reason: str) -> tuple[str, ...]: ...

    def dispatch_point(self, point, trajectory_hash: str, /): ...

    def hold(self, reason: str): ...


@dataclass(frozen=True)
class RealShadowAcceptanceReceipt:
    """Terminal application receipt; it is evidence and never motion authority."""

    status: RealShadowAcceptanceStatus
    snapshot_hash: str | None
    trajectory_hash: str | None
    simulation_receipt: ShadowSimulationReceipt | None
    real_replay_receipt: RegisteredTrajectoryRealReplayReceipt | None
    real_motion_disposition: RealMotionDisposition
    terminal_hold_confirmed: bool
    evidence_refs: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()
    non_authoritative: bool = True
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, RealShadowAcceptanceStatus):
            raise TypeError("status must be RealShadowAcceptanceStatus.")
        if not isinstance(self.real_motion_disposition, RealMotionDisposition):
            raise TypeError("real_motion_disposition must be RealMotionDisposition.")
        if not isinstance(self.terminal_hold_confirmed, bool):
            raise TypeError("terminal_hold_confirmed must be boolean.")
        if self.non_authoritative is not True:
            raise ValueError("acceptance receipts must be non-authoritative.")
        snapshot_hash = _optional_text(self.snapshot_hash, name="snapshot_hash")
        trajectory_hash = _optional_text(self.trajectory_hash, name="trajectory_hash")
        refs = tuple(sorted({str(value).strip() for value in self.evidence_refs}))
        if not refs or any(not value for value in refs):
            raise ValueError("acceptance receipt requires non-empty evidence refs.")
        reasons = tuple(str(value).strip() for value in self.reason_codes)
        if not reasons:
            reasons = (self.status.value,)
        if any(not value for value in reasons) or len(reasons) != len(set(reasons)):
            raise ValueError("reason_codes must be non-empty and unique.")
        if (
            self.status is RealShadowAcceptanceStatus.SIMULATION_ACCEPTED_NO_MOTION
            and (
                self.simulation_receipt is None
                or self.simulation_receipt.status is not ShadowSimulationStatus.COMPLETED
                or self.real_replay_receipt is not None
                or self.real_motion_disposition is not RealMotionDisposition.NONE
                or not self.terminal_hold_confirmed
            )
        ):
            raise ValueError("no-motion acceptance requires a completed held simulation.")
        if self.status is RealShadowAcceptanceStatus.REAL_REPLAY_COMPLETED and (
            self.real_replay_receipt is None
            or self.real_replay_receipt.status
            is not RegisteredTrajectoryRealReplayStatus.COMPLETED
            or self.real_motion_disposition is not RealMotionDisposition.EXECUTED
            or not self.terminal_hold_confirmed
        ):
            raise ValueError("completed real replay requires executed motion and HOLD.")
        object.__setattr__(self, "snapshot_hash", snapshot_hash)
        object.__setattr__(self, "trajectory_hash", trajectory_hash)
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "status": self.status.value,
                    "snapshot_hash": snapshot_hash,
                    "trajectory_hash": trajectory_hash,
                    "simulation_receipt_hash": (
                        None
                        if self.simulation_receipt is None
                        else self.simulation_receipt.semantic_content_hash
                    ),
                    "real_replay_status": (
                        None
                        if self.real_replay_receipt is None
                        else self.real_replay_receipt.status.value
                    ),
                    "real_motion_disposition": self.real_motion_disposition.value,
                    "terminal_hold_confirmed": self.terminal_hold_confirmed,
                    "reason_codes": reasons,
                    "evidence_refs": refs,
                    "non_authoritative": True,
                },
                domain="agenticrobot.real_shadow_acceptance.receipt.v1",
            ),
        )

    @property
    def terminal_hold_domain(self) -> str:
        """Name the domain whose terminal HOLD is represented by the boolean."""

        if self.real_replay_receipt is not None:
            return "real"
        if self.real_motion_disposition is RealMotionDisposition.UNKNOWN:
            return "unknown"
        return "simulation"


def _optional_text(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty when present.")
    return normalized


def _error_evidence(stage: str, exc: Exception) -> str:
    digest = canonical_content_hash(
        {
            "stage": stage,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        },
        domain="agenticrobot.real_shadow_acceptance.error.v1",
    )
    return f"real-shadow-acceptance-error:{stage}:{digest}"


def _gate_evidence(kind: str, payload: dict[str, object]) -> str:
    digest = canonical_content_hash(
        payload,
        domain="agenticrobot.real_shadow_acceptance.gate.v1",
    )
    return f"real-shadow-acceptance-gate:{kind}:{digest}"


def _close_simulation(control: object) -> None:
    close = getattr(control, "close", None)
    if callable(close):
        close()


def _hold_simulation_after_error(control: object) -> tuple[str, ...]:
    try:
        rig = control.rig
        hold = rig.hold_motion
        hold("real shadow acceptance simulation error")
        return (
            _gate_evidence(
                "simulation-error-hold-commanded",
                {"hold_commanded": True},
            ),
        )
    except Exception as exc:  # noqa: BLE001 - preserve fail-closed evidence.
        return (_error_evidence("simulation_error_hold", exc),)


def _simulation_receipt_matches(
    trajectory: RegisteredShadowJointTrajectory,
    receipt: ShadowSimulationReceipt,
) -> bool:
    return bool(
        receipt.status is ShadowSimulationStatus.COMPLETED
        and receipt.trajectory_hash == trajectory.semantic_content_hash
        and receipt.initial_snapshot_hash
        == trajectory.initial_snapshot.semantic_content_hash
        and receipt.identity_hash
        == trajectory.initial_snapshot.identity.semantic_content_hash
        and receipt.scene_hash == trajectory.scene_hash
        and receipt.executed_point_count == len(trajectory.points)
        and receipt.terminal_hold_confirmed
        and receipt.exact_planned_qpos_execution_confirmed
        and receipt.model_identity_confirmed
    )


def run_real_shadow_acceptance(
    *,
    snapshot_provider: Callable[[], AuthoritativeRobotStateSnapshot],
    trajectory_builder: Callable[
        [AuthoritativeRobotStateSnapshot], RegisteredShadowJointTrajectory
    ],
    simulation_control_factory: Callable[[AuthoritativeRobotStateSnapshot], object],
    model_binding_provider: Callable[
        [object, RegisteredShadowJointTrajectory], object
    ],
    simulation_executor: Callable[..., ShadowSimulationReceipt],
    real_replay_enabled: bool = False,
    real_ports_factory: (
        Callable[[RegisteredShadowJointTrajectory], RegisteredTrajectoryReplayPorts]
        | None
    ) = None,
    current_scene_hash_provider: Callable[[], str] | None = None,
    authority: RegisteredTrajectoryExecutionAuthority | None = None,
    consume_execution_authority: (
        Callable[
            [RegisteredTrajectoryExecutionAuthority],
            RegisteredTrajectoryAuthorityConsumptionReceipt,
        ]
        | None
    ) = None,
    wall_clock: Callable[[], float] = time.time,
    real_replay_executor: Callable[..., RegisteredTrajectoryRealReplayReceipt] = (
        execute_registered_shadow_trajectory_with_ports
    ),
) -> RealShadowAcceptanceReceipt:
    """Capture, simulate, and optionally replay one exact registered trajectory.

    ``real_replay_enabled`` defaults to false.  When false, neither
    ``real_ports_factory`` nor any real replay callback is called.  When true,
    the exact authority is checked before real ports are requested.
    """

    for name, callback in (
        ("snapshot_provider", snapshot_provider),
        ("trajectory_builder", trajectory_builder),
        ("simulation_control_factory", simulation_control_factory),
        ("model_binding_provider", model_binding_provider),
        ("simulation_executor", simulation_executor),
        ("wall_clock", wall_clock),
        ("real_replay_executor", real_replay_executor),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable.")
    if not isinstance(real_replay_enabled, bool):
        raise TypeError("real_replay_enabled must be boolean.")

    evidence: list[str] = []
    try:
        snapshot = snapshot_provider()
        if not isinstance(snapshot, AuthoritativeRobotStateSnapshot):
            raise TypeError(
                "snapshot_provider must return AuthoritativeRobotStateSnapshot."
            )
    except Exception as exc:  # noqa: BLE001 - compile a typed terminal receipt.
        return RealShadowAcceptanceReceipt(
            status=RealShadowAcceptanceStatus.CAPTURE_FAILED,
            snapshot_hash=None,
            trajectory_hash=None,
            simulation_receipt=None,
            real_replay_receipt=None,
            real_motion_disposition=RealMotionDisposition.NONE,
            terminal_hold_confirmed=False,
            evidence_refs=(_error_evidence("capture", exc),),
        )
    evidence.extend((*snapshot.arm_evidence_refs, *snapshot.gripper_evidence_refs))
    evidence.append(f"authoritative-shadow-snapshot:{snapshot.semantic_content_hash}")

    try:
        trajectory = trajectory_builder(snapshot)
        if not isinstance(trajectory, RegisteredShadowJointTrajectory):
            raise TypeError(
                "trajectory_builder must return RegisteredShadowJointTrajectory."
            )
        if (
            trajectory.initial_snapshot.semantic_content_hash
            != snapshot.semantic_content_hash
        ):
            raise ValueError("trajectory is not bound to the captured snapshot.")
    except Exception as exc:  # noqa: BLE001 - compile a typed terminal receipt.
        evidence.append(_error_evidence("trajectory_preparation", exc))
        return RealShadowAcceptanceReceipt(
            status=RealShadowAcceptanceStatus.TRAJECTORY_PREPARATION_FAILED,
            snapshot_hash=snapshot.semantic_content_hash,
            trajectory_hash=None,
            simulation_receipt=None,
            real_replay_receipt=None,
            real_motion_disposition=RealMotionDisposition.NONE,
            terminal_hold_confirmed=False,
            evidence_refs=tuple(evidence),
        )
    evidence.append(f"registered-shadow-trajectory:{trajectory.semantic_content_hash}")

    try:
        control = simulation_control_factory(snapshot)
    except Exception as exc:  # noqa: BLE001 - fail closed before any real port.
        evidence.append(_error_evidence("simulation_initialization", exc))
        return RealShadowAcceptanceReceipt(
            status=RealShadowAcceptanceStatus.SIMULATION_INITIALIZATION_FAILED,
            snapshot_hash=snapshot.semantic_content_hash,
            trajectory_hash=trajectory.semantic_content_hash,
            simulation_receipt=None,
            real_replay_receipt=None,
            real_motion_disposition=RealMotionDisposition.NONE,
            terminal_hold_confirmed=False,
            evidence_refs=tuple(evidence),
        )

    try:
        binding = model_binding_provider(control, trajectory)
        if binding is None:
            raise TypeError("model_binding_provider must return a model binding.")
    except Exception as exc:  # noqa: BLE001 - fail closed before any real port.
        evidence.extend(_hold_simulation_after_error(control))
        evidence.append(_error_evidence("model_binding", exc))
        try:
            _close_simulation(control)
        except Exception as close_exc:  # noqa: BLE001 - preserve both failures.
            evidence.append(_error_evidence("simulation_close", close_exc))
        return RealShadowAcceptanceReceipt(
            status=RealShadowAcceptanceStatus.MODEL_BINDING_FAILED,
            snapshot_hash=snapshot.semantic_content_hash,
            trajectory_hash=trajectory.semantic_content_hash,
            simulation_receipt=None,
            real_replay_receipt=None,
            real_motion_disposition=RealMotionDisposition.NONE,
            terminal_hold_confirmed=False,
            evidence_refs=tuple(evidence),
        )

    try:
        simulation = simulation_executor(
            control=control,
            trajectory=trajectory,
            model_binding=binding,
        )
        if not isinstance(simulation, ShadowSimulationReceipt):
            raise TypeError(
                "simulation_executor must return ShadowSimulationReceipt."
            )
    except Exception as exc:  # noqa: BLE001 - fail closed before any real port.
        evidence.extend(_hold_simulation_after_error(control))
        evidence.append(_error_evidence("simulation_execution", exc))
        try:
            _close_simulation(control)
        except Exception as close_exc:  # noqa: BLE001 - preserve both failures.
            evidence.append(_error_evidence("simulation_close", close_exc))
        return RealShadowAcceptanceReceipt(
            status=RealShadowAcceptanceStatus.SIMULATION_EXECUTION_FAILED,
            snapshot_hash=snapshot.semantic_content_hash,
            trajectory_hash=trajectory.semantic_content_hash,
            simulation_receipt=None,
            real_replay_receipt=None,
            real_motion_disposition=RealMotionDisposition.NONE,
            terminal_hold_confirmed=False,
            evidence_refs=tuple(evidence),
        )

    evidence.extend(simulation.evidence_refs)
    evidence.append(f"shadow-simulation-receipt:{simulation.semantic_content_hash}")
    try:
        _close_simulation(control)
    except Exception as exc:  # noqa: BLE001 - do not replay after uncertain shutdown.
        evidence.append(_error_evidence("simulation_close", exc))
        return RealShadowAcceptanceReceipt(
            status=RealShadowAcceptanceStatus.SIMULATION_SHUTDOWN_FAILED,
            snapshot_hash=snapshot.semantic_content_hash,
            trajectory_hash=trajectory.semantic_content_hash,
            simulation_receipt=simulation,
            real_replay_receipt=None,
            real_motion_disposition=RealMotionDisposition.NONE,
            terminal_hold_confirmed=simulation.terminal_hold_confirmed,
            evidence_refs=tuple(evidence),
        )
    if not _simulation_receipt_matches(trajectory, simulation):
        simulation_reasons = [simulation.status.value]
        if simulation.trajectory_hash != trajectory.semantic_content_hash:
            simulation_reasons.append("trajectory_hash_mismatch")
        if (
            simulation.initial_snapshot_hash
            != trajectory.initial_snapshot.semantic_content_hash
        ):
            simulation_reasons.append("initial_snapshot_hash_mismatch")
        if (
            simulation.identity_hash
            != trajectory.initial_snapshot.identity.semantic_content_hash
        ):
            simulation_reasons.append("model_identity_hash_mismatch")
        if simulation.scene_hash != trajectory.scene_hash:
            simulation_reasons.append("scene_hash_mismatch")
        if simulation.executed_point_count != len(trajectory.points):
            simulation_reasons.append("executed_point_count_mismatch")
        if not simulation.terminal_hold_confirmed:
            simulation_reasons.append("simulation_terminal_hold_unconfirmed")
        return RealShadowAcceptanceReceipt(
            status=RealShadowAcceptanceStatus.SIMULATION_REJECTED,
            snapshot_hash=snapshot.semantic_content_hash,
            trajectory_hash=trajectory.semantic_content_hash,
            simulation_receipt=simulation,
            real_replay_receipt=None,
            real_motion_disposition=RealMotionDisposition.NONE,
            terminal_hold_confirmed=simulation.terminal_hold_confirmed,
            evidence_refs=tuple(evidence),
            reason_codes=tuple(dict.fromkeys(simulation_reasons)),
        )

    if not real_replay_enabled:
        evidence.append(
            _gate_evidence(
                "default-no-motion",
                {
                    "simulation_receipt_hash": simulation.semantic_content_hash,
                    "real_replay_enabled": False,
                },
            )
        )
        return RealShadowAcceptanceReceipt(
            status=RealShadowAcceptanceStatus.SIMULATION_ACCEPTED_NO_MOTION,
            snapshot_hash=snapshot.semantic_content_hash,
            trajectory_hash=trajectory.semantic_content_hash,
            simulation_receipt=simulation,
            real_replay_receipt=None,
            real_motion_disposition=RealMotionDisposition.NONE,
            terminal_hold_confirmed=True,
            evidence_refs=tuple(evidence),
        )

    now = float(wall_clock())
    authority_reasons: list[str] = []
    current_scene_hash: str | None = None
    if not callable(current_scene_hash_provider):
        authority_reasons.append("current_scene_hash_provider_missing")
    else:
        try:
            current_scene_hash = str(current_scene_hash_provider()).strip()
        except Exception:  # noqa: BLE001 - compile a typed no-motion rejection.
            authority_reasons.append("current_scene_hash_read_failed")
        else:
            if not current_scene_hash:
                authority_reasons.append("current_scene_hash_empty")
            elif current_scene_hash != trajectory.scene_hash:
                authority_reasons.append("current_scene_hash_mismatch")
    if not isinstance(authority, RegisteredTrajectoryExecutionAuthority):
        authority_reasons.append("execution_authority_missing")
    else:
        if not authority.issued_at <= now < authority.expires_at:
            authority_reasons.append("execution_authority_not_current")
        if authority.trajectory_hash != trajectory.semantic_content_hash:
            authority_reasons.append("execution_authority_trajectory_mismatch")
        if authority.simulation_receipt_hash != simulation.semantic_content_hash:
            authority_reasons.append("execution_authority_simulation_mismatch")
    if not callable(consume_execution_authority):
        authority_reasons.append("execution_authority_consumer_missing")
    if not callable(real_ports_factory):
        authority_reasons.append("real_ports_factory_missing")
    if authority_reasons:
        evidence.append(
            _gate_evidence(
                "real-replay-authority-rejected",
                {
                    "authority_present": isinstance(
                        authority, RegisteredTrajectoryExecutionAuthority
                    ),
                    "consumer_present": callable(consume_execution_authority),
                    "ports_factory_present": callable(real_ports_factory),
                    "evaluated_at": now,
                },
            )
        )
        return RealShadowAcceptanceReceipt(
            status=RealShadowAcceptanceStatus.REAL_REPLAY_REJECTED,
            snapshot_hash=snapshot.semantic_content_hash,
            trajectory_hash=trajectory.semantic_content_hash,
            simulation_receipt=simulation,
            real_replay_receipt=None,
            real_motion_disposition=RealMotionDisposition.NONE,
            terminal_hold_confirmed=True,
            evidence_refs=tuple(evidence),
            reason_codes=tuple(authority_reasons),
        )

    assert authority is not None
    assert real_ports_factory is not None
    assert consume_execution_authority is not None
    assert current_scene_hash is not None
    ports: RegisteredTrajectoryReplayPorts | None = None
    try:
        ports = real_ports_factory(trajectory)
        real_receipt = real_replay_executor(
            trajectory=trajectory,
            simulation_receipt=simulation,
            current_scene_hash=current_scene_hash,
            current_scene_hash_provider=current_scene_hash_provider,
            current_snapshot_provider=ports.current_snapshot,
            point_dispatcher=ports.dispatch_point,
            hold_controller=ports.hold,
            authority=authority,
            consume_execution_authority=consume_execution_authority,
            begin_replay_session=ports.begin_replay_session,
            end_replay_session=ports.end_replay_session,
            execution_enabled=True,
            wall_clock=wall_clock,
        )
        if not isinstance(real_receipt, RegisteredTrajectoryRealReplayReceipt):
            raise TypeError(
                "real_replay_executor must return "
                "RegisteredTrajectoryRealReplayReceipt."
            )
    except Exception as exc:  # noqa: BLE001 - motion outcome is conservatively unknown.
        evidence.append(_error_evidence("real_replay", exc))
        real_hold_confirmed = False
        if ports is not None:
            try:
                hold = ports.hold("real shadow acceptance replay executor error")
                if not isinstance(hold, RealSkillMotionHoldReceipt):
                    raise TypeError(
                        "real replay ports hold must return RealSkillMotionHoldReceipt."
                    )
                evidence.extend(hold.evidence_refs)
                real_hold_confirmed = hold.succeeded
            except Exception as hold_exc:  # noqa: BLE001 - preserve both failures.
                evidence.append(_error_evidence("real_replay_error_hold", hold_exc))
        return RealShadowAcceptanceReceipt(
            status=RealShadowAcceptanceStatus.REAL_REPLAY_FAILED,
            snapshot_hash=snapshot.semantic_content_hash,
            trajectory_hash=trajectory.semantic_content_hash,
            simulation_receipt=simulation,
            real_replay_receipt=None,
            real_motion_disposition=RealMotionDisposition.UNKNOWN,
            terminal_hold_confirmed=real_hold_confirmed,
            evidence_refs=tuple(evidence),
        )

    evidence.extend(real_receipt.evidence_refs)
    completed = bool(
        real_receipt.status is RegisteredTrajectoryRealReplayStatus.COMPLETED
        and real_receipt.admission.admissible
        and real_receipt.admission.trajectory_hash
        == trajectory.semantic_content_hash
        and real_receipt.admission.simulation_receipt_hash
        == simulation.semantic_content_hash
        and real_receipt.executed_point_count == len(trajectory.points)
        and real_receipt.real_motion_executed
        and real_receipt.terminal_hold_confirmed
    )
    uncertain_failure = real_receipt.status in {
        RegisteredTrajectoryRealReplayStatus.DISPATCH_FAILED,
        RegisteredTrajectoryRealReplayStatus.LOWER_PORT_ERROR,
        RegisteredTrajectoryRealReplayStatus.TERMINAL_HOLD_FAILED,
    }
    if real_receipt.real_motion_may_have_executed:
        disposition = RealMotionDisposition.UNKNOWN
    elif real_receipt.real_motion_executed:
        disposition = RealMotionDisposition.EXECUTED
    elif uncertain_failure:
        disposition = RealMotionDisposition.UNKNOWN
    else:
        disposition = RealMotionDisposition.NONE
    real_reasons = [real_receipt.status.value]
    if (
        real_receipt.admission.trajectory_hash
        != trajectory.semantic_content_hash
    ):
        real_reasons.append("real_admission_trajectory_mismatch")
    if (
        real_receipt.admission.simulation_receipt_hash
        != simulation.semantic_content_hash
    ):
        real_reasons.append("real_admission_simulation_mismatch")
    if (
        real_receipt.status is RegisteredTrajectoryRealReplayStatus.COMPLETED
        and real_receipt.executed_point_count != len(trajectory.points)
    ):
        real_reasons.append("real_executed_point_count_mismatch")
    return RealShadowAcceptanceReceipt(
        status=(
            RealShadowAcceptanceStatus.REAL_REPLAY_COMPLETED
            if completed
            else (
                RealShadowAcceptanceStatus.REAL_REPLAY_FAILED
                if disposition is not RealMotionDisposition.NONE
                else RealShadowAcceptanceStatus.REAL_REPLAY_REJECTED
            )
        ),
        snapshot_hash=snapshot.semantic_content_hash,
        trajectory_hash=trajectory.semantic_content_hash,
        simulation_receipt=simulation,
        real_replay_receipt=real_receipt,
        real_motion_disposition=disposition,
        terminal_hold_confirmed=real_receipt.terminal_hold_confirmed,
        evidence_refs=tuple(evidence),
        reason_codes=tuple(real_reasons),
    )


__all__ = [
    "RealMotionDisposition",
    "RealShadowAcceptanceReceipt",
    "RealShadowAcceptanceStatus",
    "RegisteredTrajectoryReplayPorts",
    "run_real_shadow_acceptance",
]
