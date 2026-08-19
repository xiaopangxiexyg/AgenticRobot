"""Sample-aligned jerk-limited joint execution.

The planner accepts a high-rate stream of requested joint targets and
materializes one executable endpoint per admitted source cut.  Historical
profiles derive cuts from source-rate time buckets; an explicit successor can
instead use strictly increasing authoritative source timestamps.  A reusable
QP projects the requested endpoint onto the position/velocity/acceleration/
jerk reachable set.  The admitted endpoint is therefore safe to expose as the
materialized action while the high-rate transport consumes the cached control
cut.

This module owns no transport or hardware authority.  Proposals advance only
after the caller commits the exact command accepted by its sink.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .horizon_governor import (
    JointHorizonBatchedWorkspace,
    JointHorizonLimits,
    JointHorizonPlan,
    JointHorizonState,
    _normalize_joint_horizon_position_roundoff,
)
from .trajectory import (
    JointTrajectoryProposal,
    POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
)


SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET = "time_bucket"
SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP = "strictly_increasing_timestamp"
SAMPLE_ALIGNED_SOURCE_LATCH_MODES = (
    SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET,
    SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP,
)

# The horizon certificate accepts only microradian-scale numerical residue.
# A long safety horizon uses normalized position rows, so OSQP's absolute
# tolerance is multiplied by each joint's position scale.  Keeping the safety
# workspace at 1e-6 prevents that normalized residual from crossing the
# configured hard position envelope while leaving the physical limits and
# certificate unchanged.
_SAMPLE_ALIGNED_SAFETY_SOLVER_TOLERANCE = 1e-6

# The safety workspace extends well beyond the four transitions that are
# committed for each 60 Hz source cut.  Giving the remaining transitions only
# a 1% position-tracking weight makes the solver optimize the first cut as if
# the static target did not need to be reached, so each subsequent cut can
# reverse the still-moving tail.  That produces a repeatable right-arm limit
# cycle even when the incoming target is constant.  A 3% tail weight keeps the
# short tracking horizon responsive while making the long horizon converge to
# a stationary target instead of repeatedly correcting it.
_SAMPLE_ALIGNED_SAFETY_TAIL_TRACKING_WEIGHT = 0.03

def normalize_sample_aligned_source_latch_mode(value: object) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in SAMPLE_ALIGNED_SOURCE_LATCH_MODES:
        raise ValueError(
            "source_latch_mode must be one of "
            f"{SAMPLE_ALIGNED_SOURCE_LATCH_MODES}; got {value!r}."
        )
    return normalized


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
        source_latch_mode: str = SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET,
        horizon_source_steps: int = 2,
        safety_horizon_source_steps: int | None = None,
        solver_tolerance: float = 1e-4,
        scale_degraded_target: bool = True,
        jerk_regularization: float = 1e-9,
        jerk_delta_regularization: float = 0.0,
        initial_jerk_delta_regularization: float = 0.0,
        terminal_velocity_regularization: float = 0.0,
        terminal_acceleration_regularization: float = 0.0,
        reference_mode: str = "linear_hold",
        observer_window: int = 4,
        observer_efficiency_power: float = 2.0,
        observer_velocity_ema_alpha: float = 0.5,
        observer_velocity_cap_scale: float = 0.5,
        observer_coherence_min: float = 0.5,
        observer_speed_guard_scale: float = 0.4,
        terminal_velocity_displacement_scale: float = 0.75,
        observer_stop_threshold_rad: float = math.radians(0.01),
        observer_stop_velocity_decay: float = 0.0,
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
        self.source_latch_mode = normalize_sample_aligned_source_latch_mode(
            source_latch_mode
        )
        self.horizon_source_steps = int(horizon_source_steps)
        self.horizon_transitions = (
            self.commands_per_source * self.horizon_source_steps
        )
        normalized_reference_mode = str(reference_mode).strip().lower()
        if safety_horizon_source_steps is None:
            self.safety_horizon_source_steps = None
        else:
            safety_steps = int(safety_horizon_source_steps)
            if (
                isinstance(safety_horizon_source_steps, bool)
                or safety_steps != safety_horizon_source_steps
                or safety_steps <= self.horizon_source_steps
            ):
                raise ValueError(
                    "safety_horizon_source_steps must be an integer greater "
                    "than horizon_source_steps."
                )
            if normalized_reference_mode != "linear_hold":
                raise ValueError(
                    "split tracking/safety horizon currently requires "
                    "reference_mode='linear_hold'."
                )
            self.safety_horizon_source_steps = safety_steps
        self.limits = limits
        self.scale_degraded_target = bool(scale_degraded_target)
        if normalized_reference_mode not in {
            "linear_hold",
            "linear_hold_nullspace_smooth",
            "causal_hermite",
            "causal_hermite_selective",
            "causal_hermite_nullspace_smooth",
            "causal_hermite_endpoint_selective",
            "causal_hermite_endpoint_selective_nullspace",
        }:
            raise ValueError(
                "reference_mode must be 'linear_hold', "
                "'linear_hold_nullspace_smooth', 'causal_hermite' or "
                "'causal_hermite_selective' or "
                "'causal_hermite_nullspace_smooth' or "
                "'causal_hermite_endpoint_selective' or "
                "'causal_hermite_endpoint_selective_nullspace'."
            )
        resolved_observer_window = int(observer_window)
        if (
            isinstance(observer_window, bool)
            or resolved_observer_window != observer_window
            or resolved_observer_window < 2
        ):
            raise ValueError("observer_window must be an integer >= 2.")

        def fraction(value: float, *, name: str) -> float:
            resolved = float(value)
            if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
            return resolved

        efficiency_power = float(observer_efficiency_power)
        if not math.isfinite(efficiency_power) or efficiency_power < 0.0:
            raise ValueError(
                "observer_efficiency_power must be finite and non-negative."
            )
        displacement_scale = float(terminal_velocity_displacement_scale)
        if not math.isfinite(displacement_scale) or displacement_scale <= 0.0:
            raise ValueError(
                "terminal_velocity_displacement_scale must be finite and positive."
            )
        coherence_min = float(observer_coherence_min)
        if not math.isfinite(coherence_min) or not -1.0 <= coherence_min <= 1.0:
            raise ValueError("observer_coherence_min must be finite and in [-1, 1].")
        stop_threshold = float(observer_stop_threshold_rad)
        if not math.isfinite(stop_threshold) or stop_threshold < 0.0:
            raise ValueError(
                "observer_stop_threshold_rad must be finite and non-negative."
            )
        self.reference_mode = normalized_reference_mode
        self.observer_window = resolved_observer_window
        self.observer_efficiency_power = efficiency_power
        self.observer_velocity_ema_alpha = fraction(
            observer_velocity_ema_alpha,
            name="observer_velocity_ema_alpha",
        )
        self.observer_velocity_cap_scale = fraction(
            observer_velocity_cap_scale,
            name="observer_velocity_cap_scale",
        )
        self.observer_coherence_min = coherence_min
        self.observer_speed_guard_scale = fraction(
            observer_speed_guard_scale,
            name="observer_speed_guard_scale",
        )
        self.terminal_velocity_displacement_scale = displacement_scale
        self.observer_stop_threshold_rad = stop_threshold
        self.observer_stop_velocity_decay = fraction(
            observer_stop_velocity_decay,
            name="observer_stop_velocity_decay",
        )
        safety_horizon_transitions = (
            self.horizon_transitions
            if self.safety_horizon_source_steps is None
            else self.commands_per_source * self.safety_horizon_source_steps
        )
        self.safety_horizon_transitions = safety_horizon_transitions
        workspace_kwargs = {
            "limits": limits,
            "dt_sec": 1.0 / self.command_hz,
            "horizon_transitions": safety_horizon_transitions,
            "tracking_horizon_transitions": self.horizon_transitions,
            "tail_tracking_weight": (
                (
                    _SAMPLE_ALIGNED_SAFETY_TAIL_TRACKING_WEIGHT
                    if self.safety_horizon_source_steps is not None
                    else 0.0
                )
            ),
            "position_constraint_interior_rad": (
                1e-5
                if self.safety_horizon_source_steps is not None
                else 1e-6
            ),
            "position_constraint_min_interior_rad": (
                1e-5
                if self.safety_horizon_source_steps is not None
                else 1e-7
            ),
            "solver_tolerance": (
                min(
                    float(solver_tolerance),
                    _SAMPLE_ALIGNED_SAFETY_SOLVER_TOLERANCE,
                )
                if self.safety_horizon_source_steps is not None
                else float(solver_tolerance)
            ),
            "jerk_regularization": float(jerk_regularization),
            "jerk_delta_regularization": float(jerk_delta_regularization),
            "initial_jerk_delta_regularization": float(
                initial_jerk_delta_regularization
            ),
            "terminal_velocity_regularization": float(
                terminal_velocity_regularization
            ),
            "terminal_acceleration_regularization": float(
                terminal_acceleration_regularization
            ),
        }
        self._workspace = JointHorizonBatchedWorkspace(**workspace_kwargs)
        self._candidate_workspace = (
            JointHorizonBatchedWorkspace(
                **{
                    **workspace_kwargs,
                    "horizon_transitions": self.horizon_transitions,
                    "tracking_horizon_transitions": self.horizon_transitions,
                    "tail_tracking_weight": 0.0,
                    "position_constraint_interior_rad": 1e-6,
                    "position_constraint_min_interior_rad": 1e-7,
                }
            )
            if self.reference_mode
            in {
                "causal_hermite_selective",
                "causal_hermite_endpoint_selective",
                "causal_hermite_endpoint_selective_nullspace",
            }
            else None
        )
        self._lock = threading.RLock()
        self._state: JointHorizonState | None = None
        self._source_origin_timestamp: float | None = None
        self._active_source_bucket: int | None = None
        self._active_source_timestamp: float | None = None
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
        self._requested_observations: deque[tuple[float, np.ndarray]] = deque(
            maxlen=self.observer_window + 1
        )
        self._target_velocity_rad_s = np.zeros(self.dimension, dtype=float)
        self._raw_target_velocity_rad_s = np.zeros(self.dimension, dtype=float)
        self._previous_raw_target_velocity_rad_s = np.zeros(
            self.dimension, dtype=float
        )
        self._previous_jerk_rad_s3 = np.zeros(self.dimension, dtype=float)

    def _project_plan_position_roundoff(
        self,
        plan: JointHorizonPlan,
    ) -> JointHorizonPlan:
        return _normalize_joint_horizon_position_roundoff(
            plan,
            limits=self.limits,
        )

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
            self._active_source_timestamp = None
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
            self._requested_observations.clear()
            self._target_velocity_rad_s.fill(0.0)
            self._raw_target_velocity_rad_s.fill(0.0)
            self._previous_raw_target_velocity_rad_s.fill(0.0)
            self._previous_jerk_rad_s3.fill(0.0)

    def prepare_target(
        self,
        requested_qpos: Sequence[float] | np.ndarray,
        *,
        source_timestamp: float,
        speed_scale: float,
        mode: str,
        reason: str,
        requested_velocity_rad_s: Sequence[float] | np.ndarray | None = None,
        requested_velocity_valid: bool | None = None,
    ) -> SampleAlignedTargetAdmission:
        """Return the current bucket endpoint or prepare a new reachable one."""

        requested = self._vector(requested_qpos, name="requested_qpos")
        requested_velocity = (
            None
            if requested_velocity_rad_s is None
            else self._vector(
                requested_velocity_rad_s,
                name="requested_velocity_rad_s",
            )
        )
        if requested_velocity_valid is not None and requested_velocity is None:
            raise ValueError(
                "requested_velocity_valid requires requested_velocity_rad_s."
            )
        velocity_valid = (
            True
            if requested_velocity_valid is None
            else bool(requested_velocity_valid)
        )
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
            self._observe_requested_qpos(timestamp, requested)
            if self._source_origin_timestamp is None:
                self._source_origin_timestamp = timestamp
            if (
                self.source_latch_mode
                == SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP
            ):
                held = bool(
                    self._active_source_timestamp is not None
                    and timestamp <= self._active_source_timestamp
                )
                bucket = (
                    0
                    if self._active_source_bucket is None
                    else int(self._active_source_bucket + (0 if held else 1))
                )
                hold_reason = "source_timestamp_not_increasing"
            else:
                elapsed = max(0.0, timestamp - self._source_origin_timestamp)
                bucket = int(math.floor(elapsed * self.source_hz + 1e-7))
                held = bool(
                    self._active_source_bucket is not None
                    and bucket <= self._active_source_bucket
                )
                hold_reason = "source_time_bucket_not_advanced"
            if (
                held
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
                        "source_latch_mode": self.source_latch_mode,
                        "source_hold_reason": hold_reason,
                        "active_source_timestamp": self._active_source_timestamp,
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
            safety_target_projection = np.zeros(self.dimension, dtype=float)
            if self.safety_horizon_source_steps is not None:
                safety_lower = self.limits.min_position_rad + 1e-4
                safety_upper = self.limits.max_position_rad - 1e-4
                safety_projected = np.clip(
                    scaled_requested,
                    safety_lower,
                    safety_upper,
                )
                safety_target_projection = safety_projected - scaled_requested
                scaled_requested = safety_projected
            reference, reference_metadata = self._build_reference(
                state=state,
                target=scaled_requested,
                requested_velocity_rad_s=requested_velocity,
                requested_velocity_valid=velocity_valid,
            )
            if self.safety_horizon_source_steps is not None:
                reference_metadata = {
                    **reference_metadata,
                    "sample_aligned_safety_target_interior_rad": 1e-4,
                    "sample_aligned_safety_target_projected": bool(
                        np.any(np.abs(safety_target_projection) > 0.0)
                    ),
                    "sample_aligned_safety_target_max_projection_rad": float(
                        np.max(np.abs(safety_target_projection))
                    ),
                }
            selection_metadata: dict[str, Any] = {}
            if self.reference_mode in {
                "causal_hermite_selective",
                "causal_hermite_endpoint_selective",
                "causal_hermite_endpoint_selective_nullspace",
            }:
                hold_reference = self._build_linear_hold_reference(
                    state=state,
                    target=scaled_requested,
                )
                hold_plan = self._workspace.solve(
                    hold_reference,
                    initial_state=state,
                    commit_count=self.commands_per_source,
                    previous_jerk_rad_s3=self._previous_jerk_rad_s3,
                )
                if self._candidate_workspace is None:
                    raise RuntimeError("selective reference workspace is unavailable.")
                hermite_plan = self._candidate_workspace.solve(
                    reference,
                    initial_state=state,
                    commit_count=self.commands_per_source,
                    previous_jerk_rad_s3=self._previous_jerk_rad_s3,
                )
                plan, selection_metadata = self._select_reference_plan(
                    hold_plan=hold_plan,
                    hermite_plan=hermite_plan,
                    target=scaled_requested,
                    terminal_velocity_rad_s=np.asarray(
                        reference_metadata.get(
                            "sample_aligned_terminal_velocity_rad_s",
                            np.zeros(self.dimension, dtype=float),
                        ),
                        dtype=float,
                    ),
                    allow_terminal_dynamics=(
                        self.reference_mode
                        in {
                            "causal_hermite_endpoint_selective",
                            "causal_hermite_endpoint_selective_nullspace",
                        }
                    ),
                )
                if (
                    self.reference_mode
                    == "causal_hermite_endpoint_selective_nullspace"
                ):
                    plan, nullspace_metadata = (
                        self._smooth_committed_cut_nullspace(plan)
                    )
                    selection_metadata = {
                        **selection_metadata,
                        **nullspace_metadata,
                    }
            else:
                plan = self._workspace.solve(
                    reference,
                    initial_state=state,
                    commit_count=self.commands_per_source,
                    previous_jerk_rad_s3=self._previous_jerk_rad_s3,
                )
                if self.safety_horizon_source_steps is not None:
                    selection_metadata = {
                        **selection_metadata,
                        "sample_aligned_safety_horizon_source_steps": int(
                            self.safety_horizon_source_steps
                        ),
                        "sample_aligned_tracking_horizon_source_steps": int(
                            self.horizon_source_steps
                        ),
                    }
                if self.reference_mode in {
                    "linear_hold_nullspace_smooth",
                    "causal_hermite_nullspace_smooth",
                }:
                    plan, selection_metadata = self._smooth_committed_cut_nullspace(
                        plan
                    )
            plan = self._project_plan_position_roundoff(plan)
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
                    "source_latch_mode": self.source_latch_mode,
                    "source_hold_reason": None,
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
                    **reference_metadata,
                    **selection_metadata,
                    "governor_solver": plan.metadata.get("planner"),
                    "governor_solver_tolerance": plan.metadata.get(
                        "solver_tolerance"
                    ),
                    "governor_position_output_roundoff_projected": bool(
                        plan.metadata.get(
                            "position_output_roundoff_projected",
                            False,
                        )
                    ),
                    "governor_position_output_max_projection_rad": float(
                        plan.metadata.get(
                            "position_output_max_projection_rad",
                            0.0,
                        )
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
            self._active_source_timestamp = float(admission.source_timestamp)
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
            self._target_velocity_rad_s.fill(0.0)
            self._raw_target_velocity_rad_s.fill(0.0)
            self._previous_raw_target_velocity_rad_s.fill(0.0)
            self._previous_jerk_rad_s3.fill(0.0)
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
            self._previous_jerk_rad_s3 = (qdd - previous.qdd_rad_s2) / dt
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
                "source_latch_mode": self.source_latch_mode,
                "commands_per_source": self.commands_per_source,
                "horizon_source_steps": self.horizon_source_steps,
                "safety_horizon_source_steps": (
                    self.safety_horizon_source_steps
                ),
                "scale_degraded_target": self.scale_degraded_target,
                "reference_mode": self.reference_mode,
                "observer_window": self.observer_window,
                "observer_efficiency_power": self.observer_efficiency_power,
                "observer_velocity_ema_alpha": self.observer_velocity_ema_alpha,
                "observer_velocity_cap_scale": self.observer_velocity_cap_scale,
                "observer_coherence_min": self.observer_coherence_min,
                "observer_speed_guard_scale": self.observer_speed_guard_scale,
                "terminal_velocity_displacement_scale": (
                    self.terminal_velocity_displacement_scale
                ),
                "observer_stop_threshold_rad": self.observer_stop_threshold_rad,
                "observer_stop_velocity_decay": self.observer_stop_velocity_decay,
                "governor_regularization": self._workspace.regularization_metadata,
                "active_source_bucket": self._active_source_bucket,
                "active_source_timestamp": self._active_source_timestamp,
                "active_plan_row": self._active_plan_row,
                "speed_scale": self._speed_scale,
                "mode": self._mode,
                "reason": self._reason,
                "quick_stop": self._quick_stop,
                "pending_proposal": self._pending is not None,
                "last_admission": (
                    None if admission is None else dict(admission.metadata)
                ),
                "observed_requested_qpos_count": len(
                    self._requested_observations
                ),
                "target_velocity_rad_s": self._target_velocity_rad_s.tolist(),
                "previous_jerk_rad_s3": self._previous_jerk_rad_s3.tolist(),
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

    def _observe_requested_qpos(
        self,
        timestamp: float,
        requested: np.ndarray,
    ) -> None:
        if self._requested_observations:
            previous_timestamp = self._requested_observations[-1][0]
            if timestamp < previous_timestamp - 1e-9:
                return
            if math.isclose(
                timestamp,
                previous_timestamp,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                self._requested_observations.pop()
        self._requested_observations.append((timestamp, requested.copy()))

    def _estimate_requested_velocity(self) -> np.ndarray:
        observations = tuple(self._requested_observations)
        if len(observations) < 3:
            return np.zeros(self.dimension, dtype=float)
        steps = []
        step_velocities = []
        for (time0, q0), (time1, q1) in zip(
            observations[:-1],
            observations[1:],
        ):
            dt = float(time1 - time0)
            if dt <= 1e-6:
                continue
            delta = q1 - q0
            steps.append(delta)
            step_velocities.append(delta / dt)
        if len(steps) < 2:
            return np.zeros(self.dimension, dtype=float)
        step_matrix = np.asarray(steps, dtype=float)
        velocity_matrix = np.asarray(step_velocities, dtype=float)
        path_length = np.sum(np.abs(step_matrix), axis=0)
        efficiency = np.abs(np.sum(step_matrix, axis=0)) / np.maximum(
            path_length,
            1e-12,
        )
        return np.median(velocity_matrix, axis=0) * np.power(
            efficiency,
            self.observer_efficiency_power,
        )

    def _build_reference(
        self,
        *,
        state: JointHorizonState,
        target: np.ndarray,
        requested_velocity_rad_s: np.ndarray | None = None,
        requested_velocity_valid: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        reference = self._build_linear_hold_reference(state=state, target=target)
        if self.reference_mode in {
            "linear_hold",
            "linear_hold_nullspace_smooth",
        }:
            return reference, {
                "sample_aligned_reference_mode": self.reference_mode,
                "sample_aligned_observer_used": False,
            }

        observer_source = (
            "requested_qpos_causal_difference"
            if requested_velocity_rad_s is None
            else "external_causal_velocity"
        )
        if requested_velocity_rad_s is not None and not requested_velocity_valid:
            self._previous_raw_target_velocity_rad_s = (
                self._raw_target_velocity_rad_s.copy()
            )
            self._raw_target_velocity_rad_s.fill(0.0)
            self._target_velocity_rad_s.fill(0.0)
            return reference, {
                "sample_aligned_reference_mode": self.reference_mode,
                "sample_aligned_observer_used": False,
                "sample_aligned_observer_source": observer_source,
                "sample_aligned_observer_sample_count": len(
                    self._requested_observations
                ),
                "sample_aligned_observer_raw_velocity_rad_s": (
                    self._raw_target_velocity_rad_s.tolist()
                ),
                "sample_aligned_observer_velocity_rad_s": (
                    self._target_velocity_rad_s.tolist()
                ),
                "sample_aligned_terminal_velocity_rad_s": (
                    self._target_velocity_rad_s.tolist()
                ),
                "sample_aligned_observer_coherence": 0.0,
                "sample_aligned_observer_source_speed_ratio": 0.0,
                "sample_aligned_observer_fallback_reason": (
                    "external_velocity_invalid"
                ),
                "sample_aligned_observer_stop_detected": False,
            }

        raw_velocity = (
            self._estimate_requested_velocity()
            if requested_velocity_rad_s is None
            else np.asarray(requested_velocity_rad_s, dtype=float).reshape(
                self.dimension
            ).copy()
        )
        previous_raw = self._raw_target_velocity_rad_s.copy()
        self._previous_raw_target_velocity_rad_s = previous_raw
        self._raw_target_velocity_rad_s = raw_velocity.copy()
        observer_velocity_input = raw_velocity.copy()
        stopped = bool(
            self._active_requested_qpos is not None
            and np.max(np.abs(target - self._active_requested_qpos))
            < self.observer_stop_threshold_rad
        )
        if stopped:
            observer_velocity_input = (
                self.observer_stop_velocity_decay
                * self._target_velocity_rad_s
            )
        velocity = (
            (1.0 - self.observer_velocity_ema_alpha)
            * self._target_velocity_rad_s
            + self.observer_velocity_ema_alpha * observer_velocity_input
        )
        velocity_limit = (
            self.observer_velocity_cap_scale * self.limits.max_velocity_rad_s
        )
        velocity = np.clip(velocity, -velocity_limit, velocity_limit)
        self._target_velocity_rad_s = velocity.copy()
        effective_velocity = velocity.copy()
        displacement = target - state.q_rad
        wrong_direction = effective_velocity * displacement <= 0.0
        duration = self.commands_per_source / self.command_hz
        monotone_cap = (
            self.terminal_velocity_displacement_scale
            * np.abs(displacement)
            / duration
        )
        effective_velocity = np.sign(effective_velocity) * np.minimum(
            np.abs(effective_velocity),
            monotone_cap,
        )
        effective_velocity[wrong_direction] = 0.0
        max_velocity = np.maximum(self.limits.max_velocity_rad_s, 1e-12)
        source_speed_ratio = float(np.max(np.abs(raw_velocity) / max_velocity))
        previous_norm = float(np.linalg.norm(previous_raw))
        current_norm = float(np.linalg.norm(raw_velocity))
        coherence = float(
            np.dot(previous_raw, raw_velocity)
            / max(previous_norm * current_norm, 1e-12)
        )
        fallback_reason = None
        if source_speed_ratio > self.observer_speed_guard_scale:
            fallback_reason = "source_speed_guard"
        elif (
            previous_norm <= 1e-6
            or current_norm <= 1e-6
            or coherence < self.observer_coherence_min
        ):
            fallback_reason = "source_velocity_incoherent"
        if fallback_reason is not None:
            self._target_velocity_rad_s.fill(0.0)
            return reference, {
                "sample_aligned_reference_mode": self.reference_mode,
                "sample_aligned_observer_used": False,
                "sample_aligned_observer_source": observer_source,
                "sample_aligned_observer_sample_count": len(
                    self._requested_observations
                ),
                "sample_aligned_observer_raw_velocity_rad_s": raw_velocity.tolist(),
                "sample_aligned_observer_velocity_rad_s": velocity.tolist(),
                "sample_aligned_terminal_velocity_rad_s": (
                    np.zeros(self.dimension, dtype=float).tolist()
                ),
                "sample_aligned_observer_coherence": coherence,
                "sample_aligned_observer_source_speed_ratio": source_speed_ratio,
                "sample_aligned_observer_fallback_reason": fallback_reason,
                "sample_aligned_observer_stop_detected": stopped,
            }

        for step in range(1, self.commands_per_source + 1):
            s = step / float(self.commands_per_source)
            h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
            h10 = s**3 - 2.0 * s**2 + s
            h01 = -2.0 * s**3 + 3.0 * s**2
            h11 = s**3 - s**2
            reference[step] = (
                h00 * state.q_rad
                + h10 * duration * state.qd_rad_s
                + h01 * target
                + h11 * duration * effective_velocity
            )
        for step in range(
            self.commands_per_source + 1,
            self.horizon_transitions + 1,
        ):
            reference[step] = target + (
                (step - self.commands_per_source)
                / self.command_hz
                * effective_velocity
            )
        reference = np.clip(
            reference,
            self.limits.min_position_rad,
            self.limits.max_position_rad,
        )
        return reference, {
            "sample_aligned_reference_mode": self.reference_mode,
            "sample_aligned_observer_used": True,
            "sample_aligned_observer_source": observer_source,
            "sample_aligned_observer_sample_count": len(
                self._requested_observations
            ),
            "sample_aligned_observer_raw_velocity_rad_s": raw_velocity.tolist(),
            "sample_aligned_observer_velocity_rad_s": velocity.tolist(),
            "sample_aligned_terminal_velocity_rad_s": (
                effective_velocity.tolist()
            ),
            "sample_aligned_observer_coherence": coherence,
            "sample_aligned_observer_source_speed_ratio": source_speed_ratio,
            "sample_aligned_observer_fallback_reason": fallback_reason,
            "sample_aligned_observer_stop_detected": stopped,
        }

    def _build_linear_hold_reference(
        self,
        *,
        state: JointHorizonState,
        target: np.ndarray,
        horizon_transitions: int | None = None,
    ) -> np.ndarray:
        transitions = (
            self.safety_horizon_transitions
            if horizon_transitions is None
            else int(horizon_transitions)
        )
        if transitions < self.commands_per_source:
            raise ValueError(
                "horizon_transitions must cover one complete source cut."
            )
        reference = np.repeat(
            target.reshape(1, -1),
            transitions + 1,
            axis=0,
        )
        reference[0] = state.q_rad
        for step in range(1, self.commands_per_source + 1):
            alpha = step / float(self.commands_per_source)
            reference[step] = (1.0 - alpha) * state.q_rad + alpha * target
        return reference

    def _select_reference_plan(
        self,
        *,
        hold_plan: JointHorizonPlan,
        hermite_plan: JointHorizonPlan,
        target: np.ndarray,
        terminal_velocity_rad_s: np.ndarray | None = None,
        allow_terminal_dynamics: bool = False,
    ) -> tuple[JointHorizonPlan, dict[str, Any]]:
        commit = self.commands_per_source
        hold_endpoint = np.asarray(hold_plan.q_path_rad[commit], dtype=float)
        hermite_endpoint = np.asarray(
            hermite_plan.q_path_rad[commit], dtype=float
        )
        hold_error = float(np.max(np.abs(hold_endpoint - target)))
        hermite_error = float(np.max(np.abs(hermite_endpoint - target)))
        endpoint_delta = float(np.max(np.abs(hermite_endpoint - hold_endpoint)))
        terminal_velocity_delta = float(
            np.max(
                np.abs(
                    np.asarray(
                        hermite_plan.qd_path_rad_s[commit], dtype=float
                    )
                    - np.asarray(hold_plan.qd_path_rad_s[commit], dtype=float)
                )
            )
        )
        terminal_acceleration_delta = float(
            np.max(
                np.abs(
                    np.asarray(
                        hermite_plan.qdd_path_rad_s2[commit], dtype=float
                    )
                    - np.asarray(hold_plan.qdd_path_rad_s2[commit], dtype=float)
                )
            )
        )
        jerk_limit = np.maximum(self.limits.max_jerk_rad_s3, 1e-12)

        def smoothness(plan: JointHorizonPlan) -> tuple[float, int, int]:
            jerk = (
                np.diff(plan.qdd_path_rad_s2[: commit + 1], axis=0)
                * self.command_hz
            )
            normalized = jerk / jerk_limit
            chain = np.vstack((self._previous_jerk_rad_s3 / jerk_limit, normalized))
            score = float(
                np.mean(np.square(normalized))
                + 0.25 * np.mean(np.square(np.diff(chain, axis=0)))
            )
            saturation = int(np.sum(np.abs(normalized) >= 0.98))
            step = np.diff(
                np.asarray(plan.q_path_rad[: commit + 1], dtype=float),
                axis=0,
            )
            previous_step = (
                np.asarray(plan.qd_path_rad_s[0], dtype=float)
                / self.command_hz
            )
            step_chain = np.vstack((previous_step, step))
            threshold = math.radians(0.002)
            adjacent_reversal = (
                step_chain[1:] * step_chain[:-1] < 0.0
            ) & (
                np.minimum(
                    np.abs(step_chain[1:]),
                    np.abs(step_chain[:-1]),
                )
                >= threshold
            )
            meaningful_reversals = 0
            for joint in range(self.dimension):
                active = np.flatnonzero(
                    np.abs(step_chain[:, joint]) >= threshold
                )
                if active.size > 1:
                    meaningful_reversals += int(
                        np.sum(
                            np.sign(step_chain[active[1:], joint])
                            * np.sign(step_chain[active[:-1], joint])
                            < 0.0
                        )
                    )
            return score, saturation, max(
                int(np.sum(adjacent_reversal)),
                meaningful_reversals,
            )

        hold_smoothness, hold_saturation, hold_reversals = smoothness(hold_plan)
        (
            hermite_smoothness,
            hermite_saturation,
            hermite_reversals,
        ) = smoothness(hermite_plan)
        endpoint_error_slack = math.radians(0.01)
        # Keep the formal 15 Hz action/next-anchor effectively identical to
        # V2.  Selective shaping may alter the within-cut time law, not drift
        # the closed-loop joint path into a different future problem.
        endpoint_anchor_slack = math.radians(1e-8)
        terminal_velocity_anchor_slack = math.radians(1e-8)
        terminal_acceleration_anchor_slack = math.radians(1e-8)
        predicted_hold_error = predicted_hermite_error = None
        if allow_terminal_dynamics:
            velocity = (
                np.zeros(self.dimension, dtype=float)
                if terminal_velocity_rad_s is None
                else np.asarray(terminal_velocity_rad_s, dtype=float).reshape(
                    self.dimension
                )
            )
            prediction_dt = 1.0 / self.source_hz
            predicted_target = target + prediction_dt * velocity
            predicted_hold = hold_endpoint + prediction_dt * np.asarray(
                hold_plan.qd_path_rad_s[commit], dtype=float
            )
            predicted_hermite = hermite_endpoint + prediction_dt * np.asarray(
                hermite_plan.qd_path_rad_s[commit], dtype=float
            )
            predicted_hold_error = float(
                np.linalg.norm(predicted_hold - predicted_target)
            )
            predicted_hermite_error = float(
                np.linalg.norm(predicted_hermite - predicted_target)
            )
            selected = bool(
                hermite_error <= hold_error + math.radians(0.01)
                and endpoint_delta <= math.radians(0.02)
                and hermite_saturation <= hold_saturation
                and hermite_reversals <= hold_reversals
                and hermite_smoothness <= 1.02 * hold_smoothness
                and predicted_hermite_error
                <= 0.98 * max(predicted_hold_error, 1e-12)
            )
        else:
            selected = bool(
                hermite_error <= hold_error + endpoint_error_slack
                and endpoint_delta <= endpoint_anchor_slack
                and terminal_velocity_delta <= terminal_velocity_anchor_slack
                and terminal_acceleration_delta
                <= terminal_acceleration_anchor_slack
                and hermite_saturation <= hold_saturation
                and hermite_reversals <= hold_reversals
                and hermite_smoothness <= 0.98 * hold_smoothness
            )
        reason = "hermite_better" if selected else "hold_non_regression"
        return (hermite_plan if selected else hold_plan), {
            "sample_aligned_reference_selected": (
                "causal_hermite" if selected else "linear_hold"
            ),
            "sample_aligned_reference_selection_reason": reason,
            "sample_aligned_hold_endpoint_error_deg": math.degrees(hold_error),
            "sample_aligned_hermite_endpoint_error_deg": math.degrees(
                hermite_error
            ),
            "sample_aligned_endpoint_anchor_delta_deg": math.degrees(
                endpoint_delta
            ),
            "sample_aligned_terminal_velocity_anchor_delta_deg_s": math.degrees(
                terminal_velocity_delta
            ),
            "sample_aligned_terminal_acceleration_anchor_delta_deg_s2": (
                math.degrees(terminal_acceleration_delta)
            ),
            "sample_aligned_hold_smoothness": hold_smoothness,
            "sample_aligned_hermite_smoothness": hermite_smoothness,
            "sample_aligned_hold_jerk_saturation_count": hold_saturation,
            "sample_aligned_hermite_jerk_saturation_count": hermite_saturation,
            "sample_aligned_hold_reversal_count": hold_reversals,
            "sample_aligned_hermite_reversal_count": hermite_reversals,
            "sample_aligned_terminal_dynamics_allowed": bool(
                allow_terminal_dynamics
            ),
            "sample_aligned_hold_predicted_error_rad": predicted_hold_error,
            "sample_aligned_hermite_predicted_error_rad": (
                predicted_hermite_error
            ),
        }

    def _smooth_committed_cut_nullspace(
        self,
        plan: JointHorizonPlan,
    ) -> tuple[JointHorizonPlan, dict[str, Any]]:
        """Reduce executed jerk while preserving the V2 terminal q/v/a anchor."""

        commit = int(plan.commit_count)
        dt = float(plan.dt_sec)
        if commit < 4:
            return plan, {
                "sample_aligned_nullspace_smoothing_selected": False,
                "sample_aligned_nullspace_smoothing_reason": "cut_too_short",
            }
        q = np.asarray(plan.q_path_rad, dtype=float).copy()
        qd = np.asarray(plan.qd_path_rad_s, dtype=float).copy()
        qdd = np.asarray(plan.qdd_path_rad_s2, dtype=float).copy()
        baseline_jerk = np.diff(qdd[: commit + 1], axis=0) / dt
        selected_joints = 0
        alpha_by_joint = np.zeros(self.dimension, dtype=float)
        baseline_score_by_joint = np.zeros(self.dimension, dtype=float)
        selected_score_by_joint = np.zeros(self.dimension, dtype=float)

        remaining = np.arange(commit, 0, -1, dtype=float)
        terminal_map = np.vstack(
            (
                dt * np.ones(commit, dtype=float),
                dt**2 * remaining,
                dt**3 * remaining * (remaining + 1.0) / 2.0,
            )
        )
        difference = np.eye(commit, dtype=float)
        difference[1:, :-1] -= np.eye(commit - 1, dtype=float)
        hessian = difference.T @ difference + 1e-6 * np.eye(commit)
        kkt = np.block(
            [
                [hessian, terminal_map.T],
                [
                    terminal_map,
                    np.zeros((terminal_map.shape[0], terminal_map.shape[0])),
                ],
            ]
        )

        for joint in range(self.dimension):
            original = baseline_jerk[:, joint]
            offset = np.zeros(commit, dtype=float)
            offset[0] = -float(self._previous_jerk_rad_s3[joint])
            rhs = np.concatenate(
                (-(difference.T @ offset), terminal_map @ original)
            )
            try:
                ideal = np.linalg.solve(kkt, rhs)[:commit]
            except np.linalg.LinAlgError:
                continue

            def integrate(
                jerk: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                out_q = np.empty(commit + 1, dtype=float)
                out_qd = np.empty(commit + 1, dtype=float)
                out_qdd = np.empty(commit + 1, dtype=float)
                out_q[0] = q[0, joint]
                out_qd[0] = qd[0, joint]
                out_qdd[0] = qdd[0, joint]
                for step in range(commit):
                    out_qdd[step + 1] = out_qdd[step] + dt * jerk[step]
                    out_qd[step + 1] = out_qd[step] + dt * out_qdd[step + 1]
                    out_q[step + 1] = out_q[step] + dt * out_qd[step + 1]
                return out_q, out_qd, out_qdd

            jerk_scale = max(float(self.limits.max_jerk_rad_s3[joint]), 1e-12)

            def score(jerk: np.ndarray) -> float:
                chain = np.r_[float(self._previous_jerk_rad_s3[joint]), jerk]
                return float(
                    np.mean(np.square(jerk / jerk_scale))
                    + 0.25
                    * np.mean(np.square(np.diff(chain) / jerk_scale))
                )

            original_score = score(original)
            baseline_score_by_joint[joint] = original_score
            selected_score_by_joint[joint] = original_score
            for alpha in (1.0, 0.75, 0.5, 0.25, 0.125):
                candidate_jerk = original + alpha * (ideal - original)
                candidate_q, candidate_qd, candidate_qdd = integrate(
                    candidate_jerk
                )
                terminal_error = max(
                    abs(candidate_q[-1] - q[commit, joint]),
                    abs(candidate_qd[-1] - qd[commit, joint]),
                    abs(candidate_qdd[-1] - qdd[commit, joint]),
                )
                if terminal_error > 1e-9:
                    continue
                candidate_q[-1] = q[commit, joint]
                candidate_qd[-1] = qd[commit, joint]
                candidate_qdd[-1] = qdd[commit, joint]
                candidate_jerk = np.diff(candidate_qdd) / dt
                if np.any(
                    candidate_q < self.limits.min_position_rad[joint] - 1e-10
                ) or np.any(
                    candidate_q > self.limits.max_position_rad[joint] + 1e-10
                ):
                    continue
                if np.any(
                    np.abs(candidate_qd)
                    > 0.999 * self.limits.max_velocity_rad_s[joint] + 1e-9
                ):
                    continue
                if np.any(
                    np.abs(candidate_qdd)
                    > 0.999 * self.limits.max_acceleration_rad_s2[joint] + 1e-9
                ):
                    continue
                if np.any(
                    np.abs(candidate_jerk)
                    > 0.999 * self.limits.max_jerk_rad_s3[joint] + 1e-8
                ):
                    continue
                if score(candidate_jerk) > 0.98 * original_score:
                    continue
                previous_step = float(qd[0, joint]) * dt
                original_step = np.r_[
                    previous_step,
                    np.diff(q[: commit + 1, joint]),
                ]
                candidate_step = np.r_[previous_step, np.diff(candidate_q)]
                step_threshold = math.radians(0.002)
                baseline_active = np.abs(original_step) >= step_threshold
                candidate_active = np.abs(candidate_step) >= step_threshold
                if np.any(candidate_active != baseline_active):
                    continue
                if np.any(
                    np.sign(candidate_step[baseline_active])
                    != np.sign(original_step[baseline_active])
                ):
                    continue
                q[: commit + 1, joint] = candidate_q
                qd[: commit + 1, joint] = candidate_qd
                qdd[: commit + 1, joint] = candidate_qdd
                selected_joints += 1
                alpha_by_joint[joint] = alpha
                selected_score_by_joint[joint] = score(candidate_jerk)
                break

        selected = selected_joints > 0
        metadata = {
            **dict(plan.metadata),
            "nullspace_smoothing_selected": selected,
            "nullspace_smoothing_joint_count": selected_joints,
            "nullspace_smoothing_alpha_by_joint": alpha_by_joint.tolist(),
            "nullspace_smoothing_baseline_score_by_joint": (
                baseline_score_by_joint.tolist()
            ),
            "nullspace_smoothing_selected_score_by_joint": (
                selected_score_by_joint.tolist()
            ),
            "nullspace_smoothing_terminal_anchor": "exact_q_qd_qdd",
            "nullspace_smoothing_step_sign_contract_deg": 0.002,
            "nullspace_smoothing_min_score_improvement_ratio": 0.02,
        }
        smoothed = JointHorizonPlan(
            q_path_rad=q,
            qd_path_rad_s=qd,
            qdd_path_rad_s2=qdd,
            commit_count=commit,
            dt_sec=dt,
            metadata=metadata,
        )
        return smoothed, {
            "sample_aligned_reference_selected": (
                "linear_hold_nullspace_smooth" if selected else "linear_hold"
            ),
            "sample_aligned_nullspace_smoothing_selected": selected,
            "sample_aligned_nullspace_smoothing_joint_count": selected_joints,
            "sample_aligned_nullspace_smoothing_alpha_by_joint": (
                alpha_by_joint.tolist()
            ),
            "sample_aligned_nullspace_smoothing_baseline_score_by_joint": (
                baseline_score_by_joint.tolist()
            ),
            "sample_aligned_nullspace_smoothing_selected_score_by_joint": (
                selected_score_by_joint.tolist()
            ),
            "sample_aligned_nullspace_smoothing_terminal_anchor": (
                "exact_q_qd_qdd"
            ),
        }

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
    "SAMPLE_ALIGNED_SOURCE_LATCH_MODES",
    "SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP",
    "SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET",
    "SampleAlignedJerkLimitedJointTrajectoryPlanner",
    "SampleAlignedTargetAdmission",
    "normalize_sample_aligned_source_latch_mode",
]
