"""RM65 application ports for the independent real Skill control owner.

This module binds already-started dual RM65 backends to the generic Skill
control cycle.  It reuses the policy-deploy state, dispatch, and HOLD owners;
it does not construct RealTeleop, a Gateway, a device, or a control thread.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime import TeleopCommand, TeleopObservation
from teleop_runtime.deploy.arm_safety_runtime import (
    emergency_stop_deploy_arm_backends,
    hold_deploy_arm_backends,
)
from teleop_runtime.deploy.arm_state_reader import ArmStateReader
from teleop_runtime.deploy.dispatch import (
    POSITION_REFERENCE_DISPATCH_AUDIT_POST,
    POSITION_REFERENCE_DISPATCH_AUDIT_CHOICES,
    dispatch_position_reference_command,
    require_arm_backends_ready_for_position_dispatch,
)
from teleop_runtime.interfaces import EffectorBackend
from teleop_runtime.position_force_guard import (
    PositionForceGuardConfig,
    PositionForceGuardFault,
    PositionForceGuardMonitor,
)

from .control_runtime import (
    RealSkillMotionDispatchReceipt,
    RealSkillMotionHoldReceipt,
    RealSkillMotionObservationCut,
)
from .motion_port import REAL_AGENT_MOTION_COMMAND_METADATA_KEY


class RM65SkillTargetBuilder(Protocol):
    """Coupled target solver synchronization used by this concrete binding."""

    def set_current_qpos_pair(
        self,
        *,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
        reset_tracking: bool = True,
    ) -> None: ...


class RM65SkillPositionForceGuard(Protocol):
    """Lower force guard required by the formal real-position profile."""

    @property
    def motion_ready(self) -> bool: ...

    @property
    def fault(self) -> object | None: ...

    def start(
        self,
        *,
        arm_stale_watchdog: bool = True,
    ) -> Mapping[str, object]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ExclusiveRM65SkillPortProfile:
    """Trusted lower profile; none of these fields are Agent arguments."""

    state_refresh_hz: float = 30.0
    max_measurement_age_sec: float = 0.2
    dispatch_audit_mode: str = POSITION_REFERENCE_DISPATCH_AUDIT_POST

    def __post_init__(self) -> None:
        for field_name in ("state_refresh_hz", "max_measurement_age_sec"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be positive and finite.")
            object.__setattr__(self, field_name, value)
        mode = str(self.dispatch_audit_mode).strip().lower()
        if mode not in POSITION_REFERENCE_DISPATCH_AUDIT_CHOICES:
            raise ValueError(
                "dispatch_audit_mode must be one of "
                f"{POSITION_REFERENCE_DISPATCH_AUDIT_CHOICES}."
            )
        object.__setattr__(self, "dispatch_audit_mode", mode)


def _bounded_backend_state(state: Mapping[str, Any]) -> dict[str, Any]:
    background = state.get("background_streaming")
    background_state = (
        dict(background) if isinstance(background, Mapping) else {}
    )
    return {
        "hardware_ready": state.get("hardware_ready"),
        "emergency_stopped": bool(state.get("emergency_stopped", False)),
        "send_result_code": state.get("send_result_code"),
        "background_enabled": bool(background_state.get("enabled", False)),
        "background_healthy": background_state.get("healthy"),
        "background_target_active": background_state.get("target_active"),
        "background_fault": background_state.get("fault"),
    }


def build_exclusive_rm65_skill_position_force_guard(
    *,
    arm_backends: Mapping[str, EffectorBackend],
    config: PositionForceGuardConfig | None = None,
) -> PositionForceGuardMonitor:
    """Build the shared latched guard without importing a RealTeleop host."""

    backends = dict(arm_backends)
    if set(backends) != {"left", "right"}:
        raise ValueError(
            "exclusive RM65 Skill force guard requires exact left/right backends."
        )
    readers: dict[str, Callable[[], Mapping[str, Any]]] = {}
    for side, backend in backends.items():
        reader = getattr(backend, "read_force_data", None)
        if not callable(reader):
            raise TypeError(
                f"{side} RM65 Skill backend does not expose read_force_data."
            )
        readers[side] = reader

    def emergency_stop(fault: PositionForceGuardFault) -> None:
        emergency_stop_deploy_arm_backends(
            backends,
            reason=(
                f"position_force_guard:{fault.reason}:{fault.side or 'all'}"
            ),
        )

    return PositionForceGuardMonitor(
        readers_by_side=readers,
        emergency_stop=emergency_stop,
        config=config,
    )


class ExclusiveRM65SkillDeployPorts:
    """Typed production callbacks over one exact dual-RM65 backend inventory."""

    def __init__(
        self,
        *,
        arm_backends: Mapping[str, EffectorBackend],
        target_builder: RM65SkillTargetBuilder,
        position_force_guard: RM65SkillPositionForceGuard,
        profile: ExclusiveRM65SkillPortProfile | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        backends = dict(arm_backends)
        if set(backends) != {"left", "right"}:
            raise ValueError(
                "exclusive RM65 Skill ports require exact left/right backends."
            )
        if not callable(getattr(target_builder, "set_current_qpos_pair", None)):
            raise TypeError(
                "target_builder must expose set_current_qpos_pair."
            )
        if not callable(clock):
            raise TypeError("clock must be callable.")
        for name in ("start", "close"):
            if not callable(getattr(position_force_guard, name, None)):
                raise TypeError(f"position_force_guard must expose {name}.")
        if not hasattr(position_force_guard, "motion_ready"):
            raise TypeError("position_force_guard must expose motion_ready.")
        if not hasattr(position_force_guard, "fault"):
            raise TypeError("position_force_guard must expose fault.")
        self.arm_backends = backends
        self.target_builder = target_builder
        self.position_force_guard = position_force_guard
        self.profile = profile or ExclusiveRM65SkillPortProfile()
        self.clock = clock
        self.arm_state_reader = ArmStateReader(
            backends,
            refresh_hz=self.profile.state_refresh_hz,
            compute_tcp_pose=True,
            clock=clock,
        )
        self._sync_lock = threading.Lock()
        self._target_builder_synchronized = False
        self._resource_lock = threading.Lock()
        self._resources_started = False
        self._resources_closed = False
        self._force_guard_started = False
        self._startup_hold_receipt: RealSkillMotionHoldReceipt | None = None

    @property
    def resources_started(self) -> bool:
        with self._resource_lock:
            return self._resources_started

    @property
    def startup_hold_receipt(self) -> RealSkillMotionHoldReceipt | None:
        return self._startup_hold_receipt

    def start(self) -> None:
        """Start both lower backends and reject any partial inventory."""

        with self._resource_lock:
            if self._resources_closed:
                raise RuntimeError("closed RM65 Skill resources cannot restart.")
            if self._resources_started:
                return
        started: list[EffectorBackend] = []
        try:
            for side in ("left", "right"):
                backend = self.arm_backends[side]
                backend.start()
                started.append(backend)
                if not backend.is_ready():
                    raise RuntimeError(f"{side} RM65 backend is not ready.")
            require_arm_backends_ready_for_position_dispatch(
                self.arm_backends,
                require_background_target=False,
                stage="exclusive_real_skill_resource_start",
                required_sides=("left", "right"),
            )
            startup_hold = self.hold("exclusive real Skill startup fence")
            if not startup_hold.succeeded:
                raise RuntimeError(
                    "exclusive RM65 Skill startup HOLD was not confirmed."
                )
            self._startup_hold_receipt = startup_hold
            self.position_force_guard.start(arm_stale_watchdog=True)
            self._force_guard_started = True
            self._require_force_guard_motion_ready()
        except Exception:
            try:
                hold_deploy_arm_backends(
                    self.arm_backends,
                    reason="exclusive real Skill resource startup rollback",
                )
            except Exception:
                pass
            if self._force_guard_started:
                try:
                    self.position_force_guard.close()
                except Exception:
                    pass
                self._force_guard_started = False
            for backend in reversed(started):
                try:
                    backend.close()
                except Exception:
                    pass
            with self._resource_lock:
                self._resources_closed = True
            raise
        with self._resource_lock:
            self._resources_started = True

    def close(self) -> None:
        """HOLD before closing both lower resources; always attempt both sides."""

        with self._resource_lock:
            if self._resources_closed:
                return
        hold_receipt = self.hold("exclusive real Skill resource shutdown")
        errors: list[str] = []
        if not hold_receipt.succeeded:
            errors.append("lower HOLD was not confirmed")
        if self._force_guard_started:
            try:
                self.position_force_guard.close()
            except Exception as exc:  # noqa: BLE001 - still close both arms.
                errors.append(f"force_guard: {type(exc).__name__}: {exc}")
            self._force_guard_started = False
        for side in ("right", "left"):
            try:
                self.arm_backends[side].close()
            except Exception as exc:  # noqa: BLE001 - close the other side too.
                errors.append(f"{side}: {type(exc).__name__}: {exc}")
        with self._resource_lock:
            self._resources_closed = True
            self._resources_started = False
        if errors:
            raise RuntimeError(
                "exclusive RM65 Skill resource close failed: " + "; ".join(errors)
            )

    def _require_force_guard_motion_ready(self) -> None:
        fault = self.position_force_guard.fault
        if fault is not None:
            reason = getattr(fault, "reason", type(fault).__name__)
            raise RuntimeError(
                f"RM65 Skill position force guard is faulted: {reason}."
            )
        if not bool(self.position_force_guard.motion_ready):
            raise RuntimeError(
                "RM65 Skill position force guard is not motion-ready."
            )

    def observe(self) -> RealSkillMotionObservationCut:
        """Read one fresh dual hardware cut; command/preview qpos is forbidden."""

        if not self.resources_started:
            raise RuntimeError("exclusive RM65 Skill resources are not started.")
        requested_at = float(self.clock())
        # This is an explicit live observation request, not a historical
        # replay cut.  The reader refreshes first and then captures one shared
        # post-refresh causal cut so a just-measured hardware state is not
        # rejected merely because the SDK call took time.
        states = self.arm_state_reader.snapshot(
            requested_at,
            cut_after_refresh=True,
        )
        available_at = max(requested_at, float(self.clock()))
        qpos_by_side: dict[str, np.ndarray] = {}
        measured_at_by_side: dict[str, float] = {}
        evidence_payload: dict[str, Any] = {}
        for side in ("left", "right"):
            state = states.get(side)
            if not isinstance(state, Mapping):
                raise RuntimeError(f"{side} RM65 hardware state is missing.")
            if state.get("causal_snapshot_missing") is True:
                raise RuntimeError(
                    f"{side} RM65 hardware qpos cut is causally unavailable "
                    f"at {state.get('causal_snapshot_cutoff_timestamp')!r}."
                )
            try:
                qpos = np.asarray(state.get("hardware_qpos"), dtype=float).reshape(6)
                measured_at = float(state.get("hardware_measurement_timestamp"))
                sequence = int(state.get("hardware_measurement_sequence"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"{side} RM65 hardware qpos cut is invalid."
                ) from exc
            if (
                not np.all(np.isfinite(qpos))
                or not math.isfinite(measured_at)
                or sequence <= 0
                or state.get("hardware_measurement_valid") is not True
                or state.get("hardware_read_success") is False
            ):
                raise RuntimeError(
                    f"{side} RM65 hardware qpos cut is not authoritative."
                )
            age_sec = max(0.0, available_at - measured_at)
            if age_sec > self.profile.max_measurement_age_sec:
                raise RuntimeError(
                    f"{side} RM65 hardware qpos cut is stale: {age_sec:.6f}s."
                )
            qpos_by_side[side] = qpos.copy()
            measured_at_by_side[side] = measured_at
            evidence_payload[side] = {
                "measurement_timestamp": measured_at,
                "measurement_sequence": sequence,
                "qpos": tuple(float(value) for value in qpos),
            }
        evidence_hash = canonical_content_hash(
            {
                "available_at": available_at,
                "sides": evidence_payload,
            },
            domain="dexteleop.real_skill.rm65_hardware_qpos_cut.v1",
        )
        return RealSkillMotionObservationCut(
            observation=TeleopObservation(
                timestamp=available_at,
                metadata={
                    "source": "exclusive_rm65_skill_deploy",
                    "hardware_qpos_cut_hash": evidence_hash,
                },
            ),
            qpos_by_side=qpos_by_side,
            measured_at=min(measured_at_by_side.values()),
            available_at=available_at,
            evidence_refs=("rm65-hardware-qpos-cut:" + evidence_hash,),
        )

    def synchronize_target_builder(
        self,
        qpos_by_side: Mapping[str, np.ndarray],
    ) -> None:
        """Seed once, then follow the exact hardware cut without reset churn."""

        if set(qpos_by_side) != {"left", "right"}:
            raise ValueError("target synchronization requires left and right qpos.")
        with self._sync_lock:
            reset_tracking = not self._target_builder_synchronized
            self.target_builder.set_current_qpos_pair(
                left_qpos=np.asarray(qpos_by_side["left"], dtype=float).reshape(6),
                right_qpos=np.asarray(qpos_by_side["right"], dtype=float).reshape(6),
                reset_tracking=reset_tracking,
            )
            self._target_builder_synchronized = True

    def admit_command(
        self,
        command: TeleopCommand,
        observation: TeleopObservation,
        now: float,
    ) -> bool:
        """Require exact coupled joint targets and a ready lower inventory."""

        del observation, now
        if not self.resources_started:
            raise RuntimeError("exclusive RM65 Skill resources are not started.")
        self._require_force_guard_motion_ready()
        if set(command.commands) != {"left_arm", "right_arm"}:
            return False
        if not isinstance(
            command.metadata.get(REAL_AGENT_MOTION_COMMAND_METADATA_KEY),
            Mapping,
        ):
            return False
        for side in ("left", "right"):
            effector = command.commands[f"{side}_arm"]
            try:
                value = np.asarray(effector.value, dtype=float).reshape(6)
            except (TypeError, ValueError):
                return False
            if (
                effector.side != side
                or effector.command_type != "joint_vector"
                or not np.all(np.isfinite(value))
            ):
                return False
        require_arm_backends_ready_for_position_dispatch(
            self.arm_backends,
            require_background_target=False,
            stage="exclusive_real_skill_pre_dispatch",
            required_sides=("left", "right"),
        )
        return True

    def dispatch(self, command: TeleopCommand) -> RealSkillMotionDispatchReceipt:
        """Dispatch through the existing checked real position-reference path."""

        try:
            if not self.resources_started:
                raise RuntimeError(
                    "exclusive RM65 Skill resources are not started."
                )
            self._require_force_guard_motion_ready()
            batch = dispatch_position_reference_command(
                command,
                arm_backends=self.arm_backends,
                required_sides=("left", "right"),
                audit_mode=self.profile.dispatch_audit_mode,
            )
            succeeded = bool(batch.get("accepted", False))
            summary = {
                "accepted": succeeded,
                "audit_mode": batch.get("position_dispatch_audit_mode"),
                "effectors": {
                    name: {
                        "accepted": receipt.get("accepted"),
                        "command_type": receipt.get("command_type"),
                        "rewritten": receipt.get("rewritten"),
                        "result_code": receipt.get("result_code"),
                        "status": receipt.get("status"),
                    }
                    for name, receipt in dict(batch.get("effectors") or {}).items()
                },
            }
        except Exception as exc:  # noqa: BLE001 - convert to fail-closed receipt.
            succeeded = False
            summary = {
                "accepted": False,
                "error_type": type(exc).__name__,
            }
        receipt_hash = canonical_content_hash(
            summary,
            domain="dexteleop.real_skill.rm65_dispatch_receipt.v1",
        )
        return RealSkillMotionDispatchReceipt(
            succeeded=succeeded,
            evidence_refs=("rm65-skill-dispatch:" + receipt_hash,),
        )

    def hold(self, reason: str) -> RealSkillMotionHoldReceipt:
        """Ask both lower backends to HOLD and compile their bounded result."""

        error_type: str | None = None
        try:
            hold_deploy_arm_backends(self.arm_backends, reason=str(reason))
            succeeded = True
        except Exception as exc:  # noqa: BLE001 - preserve a typed failed receipt.
            succeeded = False
            error_type = type(exc).__name__
        state_by_side: dict[str, Any] = {}
        for side, backend in self.arm_backends.items():
            try:
                state_by_side[side] = _bounded_backend_state(backend.get_state())
            except Exception as exc:  # noqa: BLE001 - receipt records unreadable state.
                state_by_side[side] = {
                    "state_error_type": type(exc).__name__,
                }
        receipt_hash = canonical_content_hash(
            {
                "succeeded": succeeded,
                "error_type": error_type,
                "state_by_side": state_by_side,
            },
            domain="dexteleop.real_skill.rm65_hold_receipt.v1",
        )
        return RealSkillMotionHoldReceipt(
            succeeded=succeeded,
            evidence_refs=("rm65-skill-hold:" + receipt_hash,),
        )


__all__ = [
    "ExclusiveRM65SkillDeployPorts",
    "ExclusiveRM65SkillPortProfile",
    "RM65SkillPositionForceGuard",
    "RM65SkillTargetBuilder",
    "build_exclusive_rm65_skill_position_force_guard",
]
