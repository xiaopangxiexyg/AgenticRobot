"""Lifecycle for a default-off B601-to-RM65 MuJoCo motion shadow.

The session owns only activation-local source anchors, clutch/recenter state,
and the ordering of current-sample shadow evaluations.  The injected B601
source owns feedback admission and FK; the injected MuJoCo shadow controller
owns target-baseline admission and composition evaluation.  This module does
not start or close either resource, claim authority, run IK, or dispatch.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from control_composition.activation import (
    CompositionActivationScope,
    build_composition_activation_scope,
)
from control_station.composition.rebot_b601_rm65_mujoco_shadow import (
    B601RM65MujocoShadowActivation,
    B601RM65MujocoShadowCycle,
)
from control_station.composition.rebot_b601_rm65_residual import (
    B601RM65PoseProjection,
    B601RM65ResidualStep,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ExecutionMeasurementObservation
from teleop_runtime.action.human_residual import HumanResidualPoseAnchor

B601_RM65_MUJOCO_SHADOW_SESSION_PROFILE_OFF = "off"
B601_RM65_MUJOCO_SHADOW_SESSION_PROFILE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.b601_rm65_mujoco_shadow_session_profile.development"
)


class _ResidualSourcePort(Protocol):
    def poll_pose(self) -> B601RM65PoseProjection: ...

    def capture_anchor(
        self,
        projection: B601RM65PoseProjection,
        *,
        anchor_generation: int,
        captured_at: float,
        captured_clock_domain_id: str,
    ) -> HumanResidualPoseAnchor: ...

    def adapt(
        self,
        projection: B601RM65PoseProjection,
        *,
        anchor: HumanResidualPoseAnchor,
        adapted_at: float,
        adapted_clock_domain_id: str,
    ) -> B601RM65ResidualStep: ...


class _RuntimeConfigPort(Protocol):
    clock_domain_id: str


class _RuntimePort(Protocol):
    config: _RuntimeConfigPort


class _CompositionSpecPort(Protocol):
    required_roles: tuple[str, ...]


class _ContractsPort(Protocol):
    composition_spec: _CompositionSpecPort


class _ShadowProfilePort(Protocol):
    enabled: bool
    send_actions: bool


class _ShadowControllerPort(Protocol):
    profile: _ShadowProfilePort
    runtime: _RuntimePort | None
    contracts: _ContractsPort | None

    def activate(
        self,
        activation_scope: CompositionActivationScope,
        observation: ExecutionMeasurementObservation,
        *,
        captured_at: float,
        effective_until: float,
        context_ids: tuple[str, ...] = (),
    ) -> B601RM65MujocoShadowActivation: ...

    def evaluate(
        self,
        residual_step: B601RM65ResidualStep,
        *,
        effective_from: float,
        effective_until: float,
        target_effective_at: float,
        context_ids: tuple[str, ...] = (),
    ) -> B601RM65MujocoShadowCycle: ...


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


@dataclass(frozen=True)
class B601RM65MujocoShadowSessionProfile:
    """Default-off identity and timing for one motion-only shadow session."""

    profile_id: str = B601_RM65_MUJOCO_SHADOW_SESSION_PROFILE_OFF
    operation_spec_id: str = "unbound-b601-rm65-shadow-session"
    operation_spec_hash: str = "unbound-b601-rm65-shadow-session-hash"
    clock_domain_id: str = "unbound-b601-rm65-shadow-session-clock"
    effective_horizon_sec: float = 0.1
    enabled: bool = False
    send_actions: bool = False
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        operation_id = _required(
            self.operation_spec_id,
            name="operation_spec_id",
        )
        operation_hash = _required(
            self.operation_spec_hash,
            name="operation_spec_hash",
        )
        clock_domain = _required(
            self.clock_domain_id,
            name="clock_domain_id",
        )
        horizon = _finite(
            self.effective_horizon_sec,
            name="effective_horizon_sec",
        )
        if horizon <= 0.0:
            raise ValueError("effective_horizon_sec must be positive.")
        if not isinstance(self.enabled, bool) or not isinstance(
            self.send_actions,
            bool,
        ):
            raise TypeError("enabled and send_actions must be booleans.")
        if self.send_actions:
            raise ValueError("B601/RM65 shadow sessions cannot send actions.")
        if (
            self.enabled
            and profile_id == B601_RM65_MUJOCO_SHADOW_SESSION_PROFILE_OFF
        ):
            raise ValueError("An enabled shadow session requires a profile id.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "operation_spec_id", operation_id)
        object.__setattr__(self, "operation_spec_hash", operation_hash)
        object.__setattr__(self, "clock_domain_id", clock_domain)
        object.__setattr__(self, "effective_horizon_sec", horizon)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "profile_id": profile_id,
                    "operation_spec_id": operation_id,
                    "operation_spec_hash": operation_hash,
                    "clock_domain_id": clock_domain,
                    "effective_horizon_sec": horizon,
                    "enabled": self.enabled,
                    "send_actions": self.send_actions,
                },
                domain=(
                    B601_RM65_MUJOCO_SHADOW_SESSION_PROFILE_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )


@dataclass(frozen=True)
class B601RM65MujocoShadowSessionSnapshot:
    """Read-only lifecycle projection; never an execution-authority fact."""

    profile_id: str
    state: str
    state_revision: int
    session_epoch: int | None
    anchor_generation: int | None
    activation_id: str | None
    reason: str
    authoritative: bool = False
    promotable: bool = False
    command_dispatched: bool = False

    def __post_init__(self) -> None:
        _required(self.profile_id, name="profile_id")
        if self.state not in {"disabled", "inactive", "active", "clutched"}:
            raise ValueError("Unsupported B601/RM65 shadow session state.")
        revision = int(self.state_revision)
        if (
            isinstance(self.state_revision, bool)
            or revision != self.state_revision
            or revision < 0
        ):
            raise ValueError("state_revision must be a non-negative integer.")
        if self.state == "active":
            if (
                self.session_epoch is None
                or self.anchor_generation is None
                or not self.activation_id
            ):
                raise ValueError("An active session snapshot requires anchor identity.")
        elif any(
            value is not None
            for value in (
                self.session_epoch,
                self.anchor_generation,
                self.activation_id,
            )
        ):
            raise ValueError("A non-active session cannot retain anchor identity.")
        _required(self.reason, name="reason")
        if self.authoritative or self.promotable or self.command_dispatched:
            raise ValueError("A B601/RM65 shadow session cannot execute actions.")


@dataclass(frozen=True)
class B601RM65MujocoShadowSessionActivation:
    """One activation/recenter attempt and its exact two-sided evidence."""

    status: str
    snapshot: B601RM65MujocoShadowSessionSnapshot
    master_projection: B601RM65PoseProjection | None = None
    master_anchor: HumanResidualPoseAnchor | None = None
    target_activation: B601RM65MujocoShadowActivation | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "disabled",
            "source_rejected",
            "target_rejected",
            "activated",
        }:
            raise ValueError("Unsupported shadow session activation status.")
        if not isinstance(self.snapshot, B601RM65MujocoShadowSessionSnapshot):
            raise TypeError("snapshot must be a shadow session snapshot.")
        if self.status == "disabled":
            if any(
                value is not None
                for value in (
                    self.master_projection,
                    self.master_anchor,
                    self.target_activation,
                )
            ):
                raise ValueError("A disabled activation cannot carry source facts.")
        elif self.status == "source_rejected":
            if self.master_projection is None or any(
                value is not None
                for value in (self.master_anchor, self.target_activation)
            ):
                raise ValueError("A source rejection must retain only its projection.")
        elif self.status == "target_rejected":
            if any(
                value is None
                for value in (
                    self.master_projection,
                    self.master_anchor,
                    self.target_activation,
                )
            ):
                raise ValueError("A target rejection requires complete attempted facts.")
        elif any(
            value is None
            for value in (
                self.master_projection,
                self.master_anchor,
                self.target_activation,
            )
        ):
            raise ValueError("An activated session requires both anchor facts.")
        if (self.status == "activated") != (self.snapshot.state == "active"):
            raise ValueError("Activation status and session snapshot disagree.")

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def command_dispatched(self) -> bool:
        return False


@dataclass(frozen=True)
class B601RM65MujocoShadowSessionCycle:
    """One lifecycle-gated current-source shadow cycle."""

    status: str
    snapshot: B601RM65MujocoShadowSessionSnapshot
    master_projection: B601RM65PoseProjection | None = None
    residual_step: B601RM65ResidualStep | None = None
    shadow_cycle: B601RM65MujocoShadowCycle | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "disabled",
            "inactive",
            "clutched",
            "source_rejected",
            "input_rejected",
            "evaluated",
        }:
            raise ValueError("Unsupported B601/RM65 shadow session cycle status.")
        if not isinstance(self.snapshot, B601RM65MujocoShadowSessionSnapshot):
            raise TypeError("snapshot must be a shadow session snapshot.")
        if self.status in {"disabled", "inactive", "clutched"}:
            if any(
                value is not None
                for value in (
                    self.master_projection,
                    self.residual_step,
                    self.shadow_cycle,
                )
            ):
                raise ValueError("A non-evaluating cycle cannot retain source facts.")
        elif any(
            value is None
            for value in (
                self.master_projection,
                self.residual_step,
                self.shadow_cycle,
            )
        ):
            raise ValueError("An evaluated attempt requires exact current facts.")

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def promotable(self) -> bool:
        return False

    @property
    def command_dispatched(self) -> bool:
        return False


class B601RM65MujocoShadowSession:
    """Own activation, clutch, recenter, and current-tick shadow ordering."""

    def __init__(
        self,
        *,
        profile: B601RM65MujocoShadowSessionProfile,
        residual_source: _ResidualSourcePort | None = None,
        shadow_controller: _ShadowControllerPort | None = None,
        clock: Callable[[], float] | None = None,
        first_session_epoch: int = 1,
    ) -> None:
        if not isinstance(profile, B601RM65MujocoShadowSessionProfile):
            raise TypeError("profile must be B601RM65MujocoShadowSessionProfile.")
        epoch = int(first_session_epoch)
        if (
            isinstance(first_session_epoch, bool)
            or epoch != first_session_epoch
            or epoch < 1
        ):
            raise ValueError("first_session_epoch must be a positive integer.")
        if profile.enabled:
            self._validate_enabled_bindings(
                profile=profile,
                residual_source=residual_source,
                shadow_controller=shadow_controller,
                clock=clock,
            )
        self.profile = profile
        self.residual_source = residual_source
        self.shadow_controller = shadow_controller
        self.clock = clock
        self._next_session_epoch = epoch
        self._next_anchor_generation = 0
        self._state = "inactive" if profile.enabled else "disabled"
        self._state_revision = 0
        self._reason = "ready" if profile.enabled else "profile_disabled"
        self._activation_scope: CompositionActivationScope | None = None
        self._master_anchor: HumanResidualPoseAnchor | None = None
        self._anchor_generation: int | None = None
        self._last_target_effective_at: float | None = None
        self._lock = threading.RLock()

    @property
    def snapshot(self) -> B601RM65MujocoShadowSessionSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def activate(
        self,
        target_observation: ExecutionMeasurementObservation,
        *,
        context_ids: tuple[str, ...] = (),
    ) -> B601RM65MujocoShadowSessionActivation:
        """Capture fresh source and target anchors for a new shadow epoch."""

        return self._activate(
            target_observation,
            context_ids=context_ids,
            failure_state="inactive",
        )

    def engage_clutch(
        self,
        *,
        reason: str = "operator_clutch",
    ) -> B601RM65MujocoShadowSessionSnapshot:
        """Withdraw the residual locally without polling or target evaluation."""

        if not self.profile.enabled:
            return self.snapshot
        with self._lock:
            if self._state == "clutched":
                return self._snapshot_locked()
            if self._state != "active":
                raise RuntimeError("Clutch requires an active shadow session.")
            self._clear_active_locked(state="clutched", reason=reason)
            return self._snapshot_locked()

    def release_clutch(
        self,
        target_observation: ExecutionMeasurementObservation,
        *,
        context_ids: tuple[str, ...] = (),
    ) -> B601RM65MujocoShadowSessionActivation:
        """Recenter both sides; any rejection leaves the session clutched."""

        if not self.profile.enabled:
            return self._disabled_activation()
        with self._lock:
            if self._state != "clutched":
                raise RuntimeError("release_clutch() requires a clutched session.")
            # Keep the state precondition and recenter transaction serialized.
            # The lock is reentrant because _activate() owns the same state.
            return self._activate(
                target_observation,
                context_ids=context_ids,
                failure_state="clutched",
            )

    def deactivate(
        self,
        *,
        reason: str = "session_deactivated",
    ) -> B601RM65MujocoShadowSessionSnapshot:
        """Clear local eligibility without closing either injected resource."""

        if not self.profile.enabled:
            return self.snapshot
        with self._lock:
            self._clear_active_locked(state="inactive", reason=reason)
            return self._snapshot_locked()

    def step(
        self,
        *,
        target_effective_at: float,
        context_ids: tuple[str, ...] = (),
    ) -> B601RM65MujocoShadowSessionCycle:
        """Evaluate exactly one fresh B601 sample when the session is active."""

        if not self.profile.enabled:
            return B601RM65MujocoShadowSessionCycle(
                status="disabled",
                snapshot=self.snapshot,
            )
        target = _finite(target_effective_at, name="target_effective_at")
        with self._lock:
            if self._state != "active":
                return B601RM65MujocoShadowSessionCycle(
                    status=self._state,
                    snapshot=self._snapshot_locked(),
                )
            source = self.residual_source
            shadow = self.shadow_controller
            anchor = self._master_anchor
            scope = self._activation_scope
            assert source is not None
            assert shadow is not None
            assert anchor is not None
            assert scope is not None
            try:
                projection = source.poll_pose()
                if not isinstance(projection, B601RM65PoseProjection):
                    raise TypeError("B601 source returned an untyped projection.")
                now = self._now()
                if target + 1e-12 < now:
                    raise ValueError("target_effective_at cannot precede this step.")
                if (
                    self._last_target_effective_at is not None
                    and target <= self._last_target_effective_at
                ):
                    raise ValueError(
                        "Shadow target times must be strictly increasing."
                    )
                residual = source.adapt(
                    projection,
                    anchor=anchor,
                    adapted_at=now,
                    adapted_clock_domain_id=scope.clock_domain_id,
                )
                if not isinstance(residual, B601RM65ResidualStep):
                    raise TypeError("B601 source returned an untyped residual step.")
                cycle = shadow.evaluate(
                    residual,
                    effective_from=now,
                    effective_until=now + self.profile.effective_horizon_sec,
                    target_effective_at=target,
                    context_ids=context_ids,
                )
                if not isinstance(cycle, B601RM65MujocoShadowCycle):
                    raise TypeError("Shadow controller returned an untyped cycle.")
                if cycle.status not in {
                    "source_rejected",
                    "input_rejected",
                    "evaluated",
                }:
                    raise RuntimeError("Shadow controller returned an invalid status.")
                self._last_target_effective_at = target
            except Exception:
                self._clear_active_locked(
                    state="inactive",
                    reason="shadow_step_exception",
                )
                raise
            return B601RM65MujocoShadowSessionCycle(
                status=cycle.status,
                snapshot=self._snapshot_locked(),
                master_projection=projection,
                residual_step=residual,
                shadow_cycle=cycle,
            )

    def _activate(
        self,
        target_observation: ExecutionMeasurementObservation,
        *,
        context_ids: tuple[str, ...],
        failure_state: str,
    ) -> B601RM65MujocoShadowSessionActivation:
        if not self.profile.enabled:
            return self._disabled_activation()
        if not isinstance(target_observation, ExecutionMeasurementObservation):
            raise TypeError("target_observation must be typed.")
        with self._lock:
            source = self.residual_source
            shadow = self.shadow_controller
            assert source is not None
            assert shadow is not None
            self._clear_active_locked(
                state=failure_state,
                reason="activation_in_progress",
            )
            try:
                projection = source.poll_pose()
                if not isinstance(projection, B601RM65PoseProjection):
                    raise TypeError("B601 source returned an untyped projection.")
                if not projection.admitted:
                    self._replace_reason_locked(
                        f"source_rejected:{projection.rejection_reason}"
                    )
                    return B601RM65MujocoShadowSessionActivation(
                        status="source_rejected",
                        snapshot=self._snapshot_locked(),
                        master_projection=projection,
                    )
                captured_at = self._now()
                runtime = shadow.runtime
                contracts = shadow.contracts
                assert runtime is not None
                assert contracts is not None
                epoch = self._next_session_epoch
                generation = self._next_anchor_generation
                scope = build_composition_activation_scope(
                    operation_spec_id=self.profile.operation_spec_id,
                    operation_spec_hash=self.profile.operation_spec_hash,
                    session_epoch=epoch,
                    activated_at=captured_at,
                    input_barrier_at=captured_at,
                    clock_domain_id=runtime.config.clock_domain_id,
                    required_roles=(
                        contracts.composition_spec.required_roles
                    ),
                )
                anchor = source.capture_anchor(
                    projection,
                    anchor_generation=generation,
                    captured_at=captured_at,
                    captured_clock_domain_id=scope.clock_domain_id,
                )
                if not isinstance(anchor, HumanResidualPoseAnchor):
                    raise TypeError("B601 source returned an untyped anchor.")
                self._next_session_epoch += 1
                self._next_anchor_generation += 1
                target = shadow.activate(
                    scope,
                    target_observation,
                    captured_at=captured_at,
                    effective_until=(
                        captured_at + self.profile.effective_horizon_sec
                    ),
                    context_ids=context_ids,
                )
                if not isinstance(target, B601RM65MujocoShadowActivation):
                    raise TypeError(
                        "Shadow controller returned an untyped activation."
                    )
                if target.status not in {"activated", "baseline_rejected"}:
                    raise RuntimeError(
                        "Shadow controller returned an invalid activation status."
                    )
                if target.status != "activated":
                    self._replace_reason_locked(
                        f"target_rejected:{target.status}"
                    )
                    return B601RM65MujocoShadowSessionActivation(
                        status="target_rejected",
                        snapshot=self._snapshot_locked(),
                        master_projection=projection,
                        master_anchor=anchor,
                        target_activation=target,
                    )
                if target.activation_scope != scope:
                    raise RuntimeError(
                        "Shadow target activation differs from this session scope."
                    )
                self._state = "active"
                self._reason = "anchors_captured"
                self._activation_scope = scope
                self._master_anchor = anchor
                self._anchor_generation = generation
                self._last_target_effective_at = None
                self._state_revision += 1
            except Exception:
                self._clear_active_locked(
                    state=failure_state,
                    reason="activation_exception",
                )
                raise
            return B601RM65MujocoShadowSessionActivation(
                status="activated",
                snapshot=self._snapshot_locked(),
                master_projection=projection,
                master_anchor=anchor,
                target_activation=target,
            )

    def _disabled_activation(self) -> B601RM65MujocoShadowSessionActivation:
        return B601RM65MujocoShadowSessionActivation(
            status="disabled",
            snapshot=self.snapshot,
        )

    def _now(self) -> float:
        assert self.clock is not None
        return _finite(self.clock(), name="clock")

    def _clear_active_locked(self, *, state: str, reason: str) -> None:
        normalized_reason = _required(reason, name="reason")
        changed = bool(
            self._state != state
            or self._reason != normalized_reason
            or self._activation_scope is not None
            or self._master_anchor is not None
            or self._anchor_generation is not None
        )
        self._state = state
        self._reason = normalized_reason
        self._activation_scope = None
        self._master_anchor = None
        self._anchor_generation = None
        self._last_target_effective_at = None
        if changed:
            self._state_revision += 1

    def _replace_reason_locked(self, reason: str) -> None:
        normalized = _required(reason, name="reason")
        if normalized != self._reason:
            self._reason = normalized
            self._state_revision += 1

    def _snapshot_locked(self) -> B601RM65MujocoShadowSessionSnapshot:
        scope = self._activation_scope
        return B601RM65MujocoShadowSessionSnapshot(
            profile_id=self.profile.profile_id,
            state=self._state,
            state_revision=self._state_revision,
            session_epoch=None if scope is None else scope.session_epoch,
            anchor_generation=self._anchor_generation,
            activation_id=None if scope is None else scope.activation_id,
            reason=self._reason,
        )

    @staticmethod
    def _validate_enabled_bindings(
        *,
        profile: B601RM65MujocoShadowSessionProfile,
        residual_source: _ResidualSourcePort | None,
        shadow_controller: _ShadowControllerPort | None,
        clock: Callable[[], float] | None,
    ) -> None:
        source_methods = ("poll_pose", "capture_anchor", "adapt")
        if residual_source is None or any(
            not callable(getattr(residual_source, name, None))
            for name in source_methods
        ):
            raise TypeError("Enabled session requires the B601 residual source port.")
        if shadow_controller is None or any(
            not callable(getattr(shadow_controller, name, None))
            for name in ("activate", "evaluate")
        ):
            raise TypeError("Enabled session requires the MuJoCo shadow port.")
        if not shadow_controller.profile.enabled:
            raise ValueError("Enabled session requires an enabled shadow controller.")
        if shadow_controller.profile.send_actions:
            raise ValueError("Shadow controller must remain non-dispatching.")
        if shadow_controller.runtime is None or shadow_controller.contracts is None:
            raise ValueError("Enabled shadow controller bindings are incomplete.")
        if (
            shadow_controller.runtime.config.clock_domain_id
            != profile.clock_domain_id
        ):
            raise ValueError("Session and MuJoCo runtime clock domains differ.")
        if not callable(clock):
            raise TypeError("Enabled session requires a callable clock.")


__all__ = [
    "B601_RM65_MUJOCO_SHADOW_SESSION_PROFILE_HASH_DOMAIN_DEVELOPMENT",
    "B601_RM65_MUJOCO_SHADOW_SESSION_PROFILE_OFF",
    "B601RM65MujocoShadowSession",
    "B601RM65MujocoShadowSessionActivation",
    "B601RM65MujocoShadowSessionCycle",
    "B601RM65MujocoShadowSessionProfile",
    "B601RM65MujocoShadowSessionSnapshot",
]
