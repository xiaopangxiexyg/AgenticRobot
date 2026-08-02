"""Sample-aligned jerk-limited joint execution.

The planner accepts a high-rate stream of requested joint targets, but only
materializes one executable endpoint per source-rate bucket.  A reusable QP
projects the requested endpoint onto the position/velocity/acceleration/jerk
reachable set.  The admitted endpoint is therefore safe to expose as the
materialized action while the high-rate transport consumes the cached control
cut.

This module owns no transport or hardware authority.  Proposals advance only
after the caller commits the exact command accepted by its sink.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .horizon_governor import (
    JointHorizonBatchedWorkspace,
    JointHorizonLimits,
    JointHorizonPlan,
    JointHorizonState,
)
from .trajectory import (
    JointTrajectoryProposal,
    POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
)


def _finite_vector(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1).copy()
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a non-empty finite vector.")
    return vector


@dataclass(frozen=True)
class SampleAlignedTargetAdmission:
    """Prepared source bucket and the reachable endpoint exposed as q_ref."""

    token: int
    source_bucket: int
    source_timestamp: float
    requested_qpos: np.ndarray
    admitted_qpos: np.ndarray
    new_source_bucket: bool
    speed_scale: float
    plan: JointHorizonPlan | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        requested = _finite_vector(self.requested_qpos, name="requested_qpos")
        admitted = _finite_vector(self.admitted_qpos, name="admitted_qpos")
        if requested.shape != admitted.shape:
            raise ValueError("requested_qpos and admitted_qpos must have matching shapes.")
        requested.setflags(write=False)
        admitted.setflags(write=False)
        object.__setattr__(self, "requested_qpos", requested)
        object.__setattr__(self, "admitted_qpos", admitted)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class SampleAlignedJerkLimitedJointTrajectoryPlanner:
    """Latch source-rate qrefs and cache a transactional high-rate control cut."""

    def __init__(
        self,
        limits: JointHorizonLimits,
        *,
        source_hz: float,
        command_hz: float,
        horizon_source_steps: int = 2,
        solver_tolerance: float = 1e-4,
        scale_degraded_target: bool = True,
        jerk_regularization: float = 1e-9,
        jerk_delta_regularization: float = 0.0,
        terminal_velocity_regularization: float = 0.0,
        terminal_acceleration_regularization: float = 0.0,
    ) -> None:
        if not isinstance(limits, JointHorizonLimits):
            raise TypeError("limits must be JointHorizonLimits.")
        self.source_hz = float(source_hz)
        self.command_hz = float(command_hz)
        ratio = self.command_hz / self.source_hz
        if (
            not math.isfinite(self.source_hz)
            or not math.isfinite(self.command_hz)
            or self.source_hz <= 0.0
            or self.command_hz <= 0.0
            or not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9)
        ):
            raise ValueError("command_hz/source_hz must be a positive integer ratio.")
        if (
            isinstance(horizon_source_steps, bool)
            or int(horizon_source_steps) != horizon_source_steps
            or int(horizon_source_steps) < 2
        ):
            raise ValueError("horizon_source_steps must be an integer >= 2.")
        self.commands_per_source = int(round(ratio))
        self.horizon_source_steps = int(horizon_source_steps)
        self.horizon_transitions = (
            self.commands_per_source * self.horizon_source_steps
        )
        self.limits = limits
        self.scale_degraded_target = bool(scale_degraded_target)
        self._workspace = JointHorizonBatchedWorkspace(
            limits=limits,
            dt_sec=1.0 / self.command_hz,
            horizon_transitions=self.horizon_transitions,
            solver_tolerance=float(solver_tolerance),
            jerk_regularization=float(jerk_regularization),
            jerk_delta_regularization=float(jerk_delta_regularization),
            terminal_velocity_regularization=float(
                terminal_velocity_regularization
            ),
            terminal_acceleration_regularization=float(
                terminal_acceleration_regularization
            ),
        )
        self._lock = threading.RLock()
        self._state: JointHorizonState | None = None
        self._source_origin_timestamp: float | None = None
        self._active_source_bucket: int | None = None
        self._active_admitted_qpos: np.ndarray | None = None
        self._active_requested_qpos: np.ndarray | None = None
        self._active_plan: JointHorizonPlan | None = None
        self._active_plan_row = 1
        self._target_sequence: int | None = None
        self._speed_scale = 1.0
        self._mode = "tracking"
        self._reason = "normal"
        self._quick_stop = False
        self._generation = 0
        self._next_token = 0
        self._pending: JointTrajectoryProposal | None = None
        self._last_admission: SampleAlignedTargetAdmission | None = None

    @property
    def dimension(self) -> int:
        return int(self.limits.dimension)

    def reset(
        self,
        qpos: Sequence[float] | np.ndarray,
        *,
        monotonic_time: float | None = None,
    ) -> None:
        del monotonic_time
        q = self._vector(qpos, name="qpos")
        with self._lock:
            self._state = JointHorizonState(
                q_rad=q,
                qd_rad_s=np.zeros(self.dimension, dtype=float),
                qdd_rad_s2=np.zeros(self.dimension, dtype=float),
            )
            self._source_origin_timestamp = None
            self._active_source_bucket = None
            self._active_admitted_qpos = q.copy()
            self._active_requested_qpos = q.copy()
            self._active_plan = None
            self._active_plan_row = 1
            self._target_sequence = None
            self._speed_scale = 1.0
            self._mode = "tracking"
            self._reason = "normal"
            self._quick_stop = False
            self._generation += 1
            self._pending = None
            self._last_admission = None

    def prepare_target(
        self,
        requested_qpos: Sequence[float] | np.ndarray,
        *,
        source_timestamp: float,
        speed_scale: float,
        mode: str,
        reason: str,
    ) -> SampleAlignedTargetAdmission:
        """Return the current bucket endpoint or prepare a new reachable one."""

        requested = self._vector(requested_qpos, name="requested_qpos")
        timestamp = float(source_timestamp)
        scale = float(speed_scale)
        if not math.isfinite(timestamp):
            raise ValueError("source_timestamp must be finite.")
        if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError("speed_scale must be finite and in (0, 1].")
        normalized_mode = str(mode).strip() or "tracking"
        normalized_reason = str(reason).strip() or "normal"
        with self._lock:
            state = self._require_state()
            if self._source_origin_timestamp is None:
                self._source_origin_timestamp = timestamp
            elapsed = max(0.0, timestamp - self._source_origin_timestamp)
            bucket = int(
                math.floor(elapsed * self.source_hz + 1e-7)
            )
            if (
                self._active_source_bucket is not None
                and bucket <= self._active_source_bucket
                and self._active_admitted_qpos is not None
            ):
                admission = SampleAlignedTargetAdmission(
                    token=self._next_token,
                    source_bucket=int(self._active_source_bucket),
                    source_timestamp=timestamp,
                    requested_qpos=requested,
                    admitted_qpos=self._active_admitted_qpos,
                    new_source_bucket=False,
                    speed_scale=self._speed_scale,
                    plan=None,
                    metadata={
                        "planner": POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
                        "source_bucket_held": True,
                        "source_bucket": int(self._active_source_bucket),
                    },
                )
                self._next_token += 1
                self._last_admission = admission
                return admission

            # V1 proportionally shrinks the absolute target in degraded mode.
            # V2 keeps the already-admitted target intact and lets the hard
            # q/qd/qdd/jerk reachable set determine the executable endpoint;
            # this avoids recursively halving a persistent target error.
            scaled_requested = (
                state.q_rad + scale * (requested - state.q_rad)
                if self.scale_degraded_target
                else requested.copy()
            )
            reference = np.repeat(
                scaled_requested.reshape(1, -1),
                self.horizon_transitions + 1,
                axis=0,
            )
            reference[0] = state.q_rad
            for step in range(1, self.commands_per_source + 1):
                alpha = step / float(self.commands_per_source)
                reference[step] = (
                    (1.0 - alpha) * state.q_rad + alpha * scaled_requested
                )
            plan = self._workspace.solve(
                reference,
                initial_state=state,
                commit_count=self.commands_per_source,
            )
            admitted = np.asarray(
                plan.q_path_rad[self.commands_per_source],
                dtype=float,
            ).copy()
            request_error = requested - admitted
            admission = SampleAlignedTargetAdmission(
                token=self._next_token,
                source_bucket=bucket,
                source_timestamp=timestamp,
                requested_qpos=requested,
                admitted_qpos=admitted,
                new_source_bucket=True,
                speed_scale=scale,
                plan=plan,
                metadata={
                    "planner": POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
                    "source_bucket_held": False,
                    "source_bucket": bucket,
                    "source_hz": self.source_hz,
                    "command_hz": self.command_hz,
                    "commands_per_source": self.commands_per_source,
                    "horizon_source_steps": self.horizon_source_steps,
                    "policy_mode": normalized_mode,
                    "policy_reason": normalized_reason,
                    "policy_speed_scale": scale,
                    "degraded_target_scaled": bool(
                        self.scale_degraded_target and scale < 1.0
                    ),
                    "governor_target_qpos_rad": scaled_requested.tolist(),
                    "requested_minus_admitted_qpos_rad": request_error.tolist(),
                    "max_abs_requested_minus_admitted_deg": float(
                        np.max(np.abs(np.rad2deg(request_error)))
                    ),
                    "admission_limited": bool(
                        not np.allclose(requested, admitted, rtol=0.0, atol=1e-10)
                    ),
                    "governor_solver": plan.metadata.get("planner"),
                    "governor_solver_tolerance": plan.metadata.get(
                        "solver_tolerance"
                    ),
                },
            )
            self._next_token += 1
            self._last_admission = admission
            self._mode = normalized_mode
            self._reason = normalized_reason
            return admission

    def activate_target(
        self,
        admission: SampleAlignedTargetAdmission,
        *,
        sequence: int,
    ) -> None:
        if not isinstance(admission, SampleAlignedTargetAdmission):
            raise TypeError("admission must be SampleAlignedTargetAdmission.")
        if admission.plan is None or not admission.new_source_bucket:
            raise ValueError("only a newly prepared source bucket can be activated.")
        with self._lock:
            if self._last_admission is None or admission.token != self._last_admission.token:
                raise RuntimeError("sample-aligned target admission is stale.")
            self._active_source_bucket = int(admission.source_bucket)
            self._active_admitted_qpos = admission.admitted_qpos.copy()
            self._active_requested_qpos = admission.requested_qpos.copy()
            self._active_plan = admission.plan
            self._active_plan_row = 1
            self._target_sequence = int(sequence)
            self._speed_scale = float(admission.speed_scale)
            self._quick_stop = False
            self._generation += 1
            self._pending = None

    def quick_stop(
        self,
        *,
        sequence: int,
        monotonic_time: float | None = None,
        reason: str,
    ) -> None:
        del monotonic_time
        normalized_reason = str(reason).strip() or "quick_stop"
        with self._lock:
            state = self._require_state()
            stopped = JointHorizonState(
                q_rad=state.q_rad,
                qd_rad_s=np.zeros(self.dimension, dtype=float),
                qdd_rad_s2=np.zeros(self.dimension, dtype=float),
            )
            self._state = stopped
            self._active_plan = None
            self._active_plan_row = 1
            self._active_admitted_qpos = np.asarray(stopped.q_rad, dtype=float).copy()
            self._active_requested_qpos = np.asarray(stopped.q_rad, dtype=float).copy()
            self._target_sequence = int(sequence)
            self._speed_scale = 0.0
            self._mode = "quick_stop"
            self._reason = normalized_reason
            self._quick_stop = True
            self._generation += 1
            self._pending = None

    def propose(self, *, monotonic_time: float | None = None) -> JointTrajectoryProposal:
        sample_time = 0.0 if monotonic_time is None else float(monotonic_time)
        if not math.isfinite(sample_time):
            raise ValueError("monotonic_time must be finite.")
        with self._lock:
            state = self._require_state()
            if self._target_sequence is None:
                raise RuntimeError("sample-aligned planner has no active target.")
            if self._pending is not None:
                raise RuntimeError("a trajectory proposal is already pending.")
            row = self._active_plan_row
            if self._quick_stop or self._active_plan is None:
                q = np.asarray(state.q_rad, dtype=float).copy()
                qd = np.zeros(self.dimension, dtype=float)
                qdd = np.zeros(self.dimension, dtype=float)
            else:
                row = min(row, self._active_plan.q_path_rad.shape[0] - 1)
                q = np.asarray(self._active_plan.q_path_rad[row], dtype=float).copy()
                qd = np.asarray(self._active_plan.qd_path_rad_s[row], dtype=float).copy()
                qdd = np.asarray(self._active_plan.qdd_path_rad_s2[row], dtype=float).copy()
            target = (
                np.asarray(state.q_rad, dtype=float).copy()
                if self._active_admitted_qpos is None
                else self._active_admitted_qpos.copy()
            )
            token = self._next_token
            self._next_token += 1
            proposal = JointTrajectoryProposal(
                token=token,
                generation=self._generation,
                target_sequence=int(self._target_sequence),
                sample_monotonic_time=sample_time,
                integration_dt_sec=1.0 / self.command_hz,
                q_command=q,
                q_target=target,
                qd_command_rad_s=qd,
                qd_reference_rad_s=qd,
                qd_desired_rad_s=qd,
                qdd_command_rad_s2=qdd,
                speed_scale=float(self._speed_scale),
                mode=self._mode,
                reason=self._reason,
                quick_stop=bool(self._quick_stop),
                acceleration_limited=True,
                velocity_limited=True,
                command_step_limited=True,
                safety_envelope_limited=False,
                metadata={
                    "planner": POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
                    "sample_aligned_source_bucket": self._active_source_bucket,
                    "sample_aligned_plan_row": int(row),
                    "sample_aligned_commands_per_source": self.commands_per_source,
                    "sample_aligned_admitted_qpos": target.tolist(),
                    "sample_aligned_requested_qpos": (
                        None
                        if self._active_requested_qpos is None
                        else self._active_requested_qpos.tolist()
                    ),
                    "trajectory_max_velocity_rad_s": (
                        self.limits.max_velocity_rad_s.tolist()
                    ),
                    "trajectory_max_acceleration_rad_s2": (
                        self.limits.max_acceleration_rad_s2.tolist()
                    ),
                    "trajectory_max_jerk_rad_s3": self.limits.max_jerk_rad_s3.tolist(),
                },
            )
            self._pending = proposal
            return proposal

    def commit(
        self,
        proposal: JointTrajectoryProposal,
        *,
        accepted_monotonic_time: float | None = None,
        accepted_q_command: Sequence[float] | np.ndarray | None = None,
        allow_bounded_accepted_adjustment: bool = False,
    ) -> None:
        del accepted_monotonic_time
        with self._lock:
            self._require_matching_pending(proposal)
            accepted = (
                np.asarray(proposal.q_command, dtype=float).copy()
                if accepted_q_command is None
                else self._vector(accepted_q_command, name="accepted_q_command")
            )
            adjusted = not np.allclose(
                accepted,
                proposal.q_command,
                rtol=0.0,
                atol=1e-10,
            )
            if adjusted and not allow_bounded_accepted_adjustment:
                raise ValueError("accepted_q_command differs from the transactional proposal.")
            previous = self._require_state()
            dt = 1.0 / self.command_hz
            if adjusted:
                lower = np.minimum(previous.q_rad, proposal.q_command) - 1e-12
                upper = np.maximum(previous.q_rad, proposal.q_command) + 1e-12
                if np.any(accepted < lower) or np.any(accepted > upper):
                    raise ValueError(
                        "adjusted accepted_q_command must remain between the previous "
                        "accepted command and the proposal."
                    )
                qd = (accepted - previous.q_rad) / dt
                qdd = (qd - previous.qd_rad_s) / dt
            else:
                qd = np.asarray(proposal.qd_command_rad_s, dtype=float)
                qdd = np.asarray(proposal.qdd_command_rad_s2, dtype=float)
            self._state = JointHorizonState(
                q_rad=accepted,
                qd_rad_s=qd,
                qdd_rad_s2=qdd,
            )
            if self._active_plan is not None:
                self._active_plan_row = min(
                    self._active_plan_row + 1,
                    self._active_plan.q_path_rad.shape[0] - 1,
                )
            self._pending = None

    def reject(self, proposal: JointTrajectoryProposal) -> bool:
        with self._lock:
            if self._pending is None or self._pending.token != proposal.token:
                return False
            self._pending = None
            return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = self._state
            admission = self._last_admission
            return {
                "planner": POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
                "initialized": state is not None,
                "q_command": None if state is None else state.q_rad.tolist(),
                "qd_command_rad_s": None if state is None else state.qd_rad_s.tolist(),
                "qdd_command_rad_s2": (
                    None if state is None else state.qdd_rad_s2.tolist()
                ),
                "target_q": (
                    None
                    if self._active_admitted_qpos is None
                    else self._active_admitted_qpos.tolist()
                ),
                "target_sequence": self._target_sequence,
                "source_hz": self.source_hz,
                "command_hz": self.command_hz,
                "commands_per_source": self.commands_per_source,
                "horizon_source_steps": self.horizon_source_steps,
                "scale_degraded_target": self.scale_degraded_target,
                "governor_regularization": self._workspace.regularization_metadata,
                "active_source_bucket": self._active_source_bucket,
                "active_plan_row": self._active_plan_row,
                "speed_scale": self._speed_scale,
                "mode": self._mode,
                "reason": self._reason,
                "quick_stop": self._quick_stop,
                "pending_proposal": self._pending is not None,
                "last_admission": (
                    None if admission is None else dict(admission.metadata)
                ),
                "max_velocity_rad_s": self.limits.max_velocity_rad_s.tolist(),
                "max_acceleration_rad_s2": (
                    self.limits.max_acceleration_rad_s2.tolist()
                ),
                "max_jerk_rad_s3": self.limits.max_jerk_rad_s3.tolist(),
            }

    def set_max_command_step(
        self,
        max_command_step_rad: Sequence[float] | np.ndarray,
    ) -> None:
        # The fixed sample-aligned profile folds the configured step into its
        # velocity limits at construction time. Runtime dt scaling must not
        # loosen that hard bound.
        requested = self._vector(
            max_command_step_rad,
            name="max_command_step_rad",
        )
        configured = self.limits.max_velocity_rad_s / self.command_hz
        if np.any(requested + 1e-12 < configured):
            raise RuntimeError(
                "sample-aligned jerk limits are immutable during an active run; "
                "configure the tighter max step before backend construction."
            )

    def set_reference_period_sec(self, reference_period_sec: float) -> None:
        value = float(reference_period_sec)
        if not math.isclose(
            value,
            1.0 / self.source_hz,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "sample-aligned source period is immutable during an active run."
            )

    def _vector(
        self,
        value: Sequence[float] | np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        vector = _finite_vector(value, name=name)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"{name} must contain exactly {self.dimension} finite values."
            )
        return vector

    def _require_state(self) -> JointHorizonState:
        if self._state is None:
            raise RuntimeError("sample-aligned planner must be reset before use.")
        return self._state

    def _require_matching_pending(self, proposal: JointTrajectoryProposal) -> None:
        if not isinstance(proposal, JointTrajectoryProposal):
            raise TypeError("proposal must be JointTrajectoryProposal.")
        if self._pending is None or self._pending.token != proposal.token:
            raise RuntimeError("trajectory proposal is no longer pending.")
        if proposal.generation != self._generation:
            raise RuntimeError("trajectory proposal was superseded by a newer target.")


__all__ = [
    "POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED",
    "SampleAlignedJerkLimitedJointTrajectoryPlanner",
    "SampleAlignedTargetAdmission",
]
