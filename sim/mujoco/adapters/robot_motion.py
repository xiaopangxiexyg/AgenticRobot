from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from robot_motion.position_control.execution_policy import (
    classify_position_trajectory_policy,
    final_follow_qpos_target,
    metadata_mapping,
)
from robot_motion.position_control.trajectory import (
    AccelerationLimitedJointTrajectoryPlanner,
    DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
    DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
    DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    JointTrajectoryLimits,
    JointTrajectoryProposal,
)
from robot_motion.rm65.defaults import (
    RM65_POSITION_TRACKING_FOLLOW_ALPHA,
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG,
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
)
from robot_motion.runtime import Rig, TeleopCommand
from robot_motion.runtime.dispatch import CommandDispatchReceipt
from sim.contracts import TimingSpec

from ..control import DirectQposCommand, DirectQposDebugController
from ..scene import MujocoScene


_SIDES = ("left", "right")


def _six_positive(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)) or np.any(vector <= 0.0):
        raise ValueError(f"{name} must contain six finite positive values.")
    return tuple(float(value) for value in vector)


@dataclass(frozen=True)
class MujocoPositionExecutionConfig:
    """Real-aligned qref-to-qcmd settings for prescribed MuJoCo execution."""

    timing: TimingSpec
    active_sides: tuple[str, ...] = _SIDES
    max_velocity_rad_s: tuple[float, ...] = RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S
    max_acceleration_rad_s2: tuple[float, ...] = (
        RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2
    )
    max_command_step_deg: float = RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG
    velocity_ramp_sec: float = DEFAULT_POSITION_VELOCITY_RAMP_SEC
    target_velocity_filter_tau_sec: float = (
        DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC
    )
    degraded_speed_scale: float = DEFAULT_POSITION_DEGRADED_SPEED_SCALE
    follow_qpos_alpha: float = RM65_POSITION_TRACKING_FOLLOW_ALPHA
    realtime_worker: bool = True
    worker_join_timeout_sec: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.timing, TimingSpec):
            raise TypeError("timing must be a TimingSpec.")
        sides = tuple(str(side).strip() for side in self.active_sides)
        if not sides or len(set(sides)) != len(sides) or set(sides) - set(_SIDES):
            raise ValueError("active_sides must be a unique non-empty subset of left/right.")
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(
            self,
            "max_velocity_rad_s",
            _six_positive(self.max_velocity_rad_s, name="max_velocity_rad_s"),
        )
        object.__setattr__(
            self,
            "max_acceleration_rad_s2",
            _six_positive(
                self.max_acceleration_rad_s2,
                name="max_acceleration_rad_s2",
            ),
        )
        for name in (
            "max_command_step_deg",
            "velocity_ramp_sec",
            "worker_join_timeout_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        tau = float(self.target_velocity_filter_tau_sec)
        if not math.isfinite(tau) or tau < 0.0:
            raise ValueError("target_velocity_filter_tau_sec must be finite and non-negative.")
        object.__setattr__(self, "target_velocity_filter_tau_sec", tau)
        degraded = float(self.degraded_speed_scale)
        follow = float(self.follow_qpos_alpha)
        if not math.isfinite(degraded) or not 0.0 < degraded <= 1.0:
            raise ValueError("degraded_speed_scale must be in (0, 1].")
        if not math.isfinite(follow) or not 0.0 < follow <= 1.0:
            raise ValueError("follow_qpos_alpha must be in (0, 1].")
        object.__setattr__(self, "degraded_speed_scale", degraded)
        object.__setattr__(self, "follow_qpos_alpha", follow)


@dataclass(frozen=True)
class _QueuedReference:
    sequence: int
    published_monotonic_time: float
    qref_by_side: Mapping[str, np.ndarray]
    metadata_by_side: Mapping[str, Mapping[str, object]]


class MujocoPositionExecutionRig(Rig):
    """Atomic dual-arm 60 Hz qref to 120 Hz qcmd MuJoCo execution rig.

    The worker is the sole owner of MuJoCo ``MjData``. ``send_commands`` only
    replaces a bounded latest-reference mailbox, so a delayed IK frame cannot
    build an obsolete command backlog.
    """

    def __init__(
        self,
        scene: MujocoScene,
        config: MujocoPositionExecutionConfig,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not callable(monotonic_clock) or not callable(sleep):
            raise TypeError("monotonic_clock and sleep must be callable.")
        self.scene = scene
        self.config = config
        self._clock = monotonic_clock
        self._sleep = sleep
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._started = False
        self._clock_origin = 0.0
        self._next_reference_sequence = 1
        self._next_command_sequence = 1
        self._pending_reference: _QueuedReference | None = None
        self._hold_reason: str | None = None
        self._last_reference: _QueuedReference | None = None
        self._last_worker_error: str | None = None
        self._worker_physics_ticks = 0
        self._worker_deadline_misses = 0
        self._pending_proposals: dict[str, JointTrajectoryProposal] = {}
        self._planners: dict[str, AccelerationLimitedJointTrajectoryPlanner] = {}
        self._direct = DirectQposDebugController(
            scene,
            config.timing,
            active_sides=config.active_sides,
            record_ticks=False,
        )
        self._accepted_qcmd_by_side: dict[str, np.ndarray] = {
            side: np.asarray(scene.get_arm_qpos(side), dtype=float).reshape(6).copy()
            for side in config.active_sides
        }

    @property
    def name(self) -> str:
        return "mujoco_position_execution_rig"

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._clock_origin = float(self._clock())
            max_step = np.full(
                6,
                np.deg2rad(float(self.config.max_command_step_deg)),
                dtype=float,
            )
            for side in self.config.active_sides:
                planner = AccelerationLimitedJointTrajectoryPlanner(
                    JointTrajectoryLimits(
                        max_velocity_rad_s=self.config.max_velocity_rad_s,
                        max_acceleration_rad_s2=self.config.max_acceleration_rad_s2,
                        max_command_step_rad=max_step,
                    ),
                    velocity_ramp_sec=self.config.velocity_ramp_sec,
                    target_velocity_filter_tau_sec=(
                        self.config.target_velocity_filter_tau_sec
                    ),
                    reference_period_sec=1.0 / self.config.timing.reference_hz,
                    monotonic_clock=self._clock,
                )
                initial = self._accepted_qcmd_by_side[side]
                planner.reset(initial, monotonic_time=self._clock_origin)
                # A target is required before the first 120 Hz proposal.
                planner.accept_target(
                    initial,
                    sequence=0,
                    published_monotonic_time=(
                        self._clock_origin - 1.0 / self.config.timing.reference_hz
                    ),
                    monotonic_time=self._clock_origin,
                )
                self._planners[side] = planner
            self._direct.reset(
                DirectQposCommand(
                    sequence=0,
                    qpos_by_side=self._accepted_qcmd_by_side,
                    source="mujoco_position_execution_start",
                )
            )
            self._stop_event.clear()
            self._started = True
            if self.config.realtime_worker:
                self._worker = threading.Thread(
                    target=self._worker_loop,
                    name="mujoco_position_execution",
                    daemon=True,
                )
                self._worker.start()

    def is_ready(self) -> bool:
        with self._lock:
            return self._started and self._last_worker_error is None

    def send_commands(
        self,
        command: TeleopCommand,
        phase_recorder=None,
    ) -> dict[str, CommandDispatchReceipt]:
        del phase_recorder
        expected = {f"{side}_arm" for side in self.config.active_sides}
        present = expected.intersection(command.commands)
        if present != expected:
            raise ValueError(
                "position execution requires one atomic qref for every active arm: "
                f"expected={sorted(expected)} present={sorted(present)}"
            )
        qref_by_side: dict[str, np.ndarray] = {}
        metadata_by_side: dict[str, Mapping[str, object]] = {}
        for side in self.config.active_sides:
            effector = command.commands[f"{side}_arm"]
            if effector.command_type != "joint_vector":
                raise ValueError(f"{effector.effector_name} must be a joint_vector qref.")
            qref = np.asarray(effector.value, dtype=float).reshape(-1)
            if qref.shape != (6,) or not np.all(np.isfinite(qref)):
                raise ValueError(f"{effector.effector_name} qref must contain six finite values.")
            qref_by_side[side] = qref.copy()
            metadata_by_side[side] = dict(effector.metadata)
        published = float(self._clock())
        if not math.isfinite(published):
            raise RuntimeError("monotonic clock returned a non-finite value.")
        with self._lock:
            if not self._started:
                raise RuntimeError(
                    "MujocoPositionExecutionRig.start() is required before dispatch."
                )
            if self._last_worker_error is not None:
                raise RuntimeError(
                    "MuJoCo position execution worker has failed: "
                    f"{self._last_worker_error}"
                )
            sequence = self._next_reference_sequence
            self._next_reference_sequence += 1
            self._pending_reference = _QueuedReference(
                sequence=sequence,
                published_monotonic_time=published,
                qref_by_side=qref_by_side,
                metadata_by_side=metadata_by_side,
            )
            self._hold_reason = None
        return {
            f"{side}_arm": CommandDispatchReceipt(
                effector_name=f"{side}_arm",
                accepted=True,
                operation="queue_mujoco_qref",
                status="queued",
                metadata={"reference_sequence": sequence},
            )
            for side in self.config.active_sides
        }

    def hold_motion(self, reason: str = "manual hold") -> None:
        with self._lock:
            self._pending_reference = None
            self._hold_reason = str(reason).strip() or "manual hold"

    def accepted_qcmd_by_side(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {
                side: qpos.copy()
                for side, qpos in self._accepted_qcmd_by_side.items()
            }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "started": self._started,
                "ready": self._started and self._last_worker_error is None,
                "execution_mode": "kinematic_direct_qpos_debug",
                "robot_dynamics": "not_simulated",
                "timing": self.config.timing.to_dict(),
                "accepted_qcmd_by_side": {
                    side: qpos.tolist()
                    for side, qpos in self._accepted_qcmd_by_side.items()
                },
                "last_reference_sequence": (
                    None if self._last_reference is None else self._last_reference.sequence
                ),
                "pending_reference_sequence": (
                    None if self._pending_reference is None else self._pending_reference.sequence
                ),
                "last_worker_error": self._last_worker_error,
                "physics_ticks": self._worker_physics_ticks,
                "deadline_misses": self._worker_deadline_misses,
                "planner_by_side": {
                    side: planner.snapshot() for side, planner in self._planners.items()
                },
            }

    def step_physics(self) -> None:
        """Advance one deterministic physics tick; used by tests and offline replay."""

        with self._lock:
            if not self._started:
                raise RuntimeError("MujocoPositionExecutionRig.start() is required.")
            if self._last_worker_error is not None:
                raise RuntimeError(self._last_worker_error)
            planner_now = self._clock_origin + self._direct.physics_time_sec
            self._consume_reference_locked(planner_now)
            command = self._build_due_command_locked(planner_now)
            proposals = self._pending_proposals
            try:
                self._direct.step_physics(command)
                if proposals:
                    accepted = {
                        side: self._direct.accepted_arm_qpos(side)
                        for side in self.config.active_sides
                    }
                    # Validate every arm before committing either planner.  The
                    # direct-qpos collision projector is allowed to shorten a
                    # command, but not to move beyond the segment requested by
                    # its planner.  This keeps the bimanual planner update
                    # atomic even if a future projector violates that contract.
                    for side, proposal in proposals.items():
                        previous = self._accepted_qcmd_by_side[side]
                        lower = np.minimum(previous, proposal.q_command) - 1e-9
                        upper = np.maximum(previous, proposal.q_command) + 1e-9
                        if np.any(accepted[side] < lower) or np.any(accepted[side] > upper):
                            raise RuntimeError(
                                f"{side} accepted qcmd is outside its planner segment"
                            )
                    for side, proposal in proposals.items():
                        adjusted = not np.allclose(
                            accepted[side], proposal.q_command, rtol=0.0, atol=1e-10
                        )
                        self._planners[side].commit(
                            proposal,
                            accepted_monotonic_time=planner_now,
                            accepted_q_command=accepted[side],
                            allow_bounded_accepted_adjustment=adjusted,
                        )
                    self._accepted_qcmd_by_side = {
                        side: qpos.copy() for side, qpos in accepted.items()
                    }
            except Exception:
                for side, proposal in proposals.items():
                    try:
                        self._planners[side].reject(proposal)
                    except Exception:
                        # A preceding side may already have committed.  Reset
                        # every planner to the qpos actually held by MuJoCo so
                        # retry/recovery cannot mix two planner generations.
                        pass
                actual = {
                    side: self._direct.accepted_arm_qpos(side)
                    for side in self.config.active_sides
                }
                for side, planner in self._planners.items():
                    planner.reset(actual[side], monotonic_time=planner_now)
                    planner.accept_target(
                        actual[side],
                        sequence=0,
                        published_monotonic_time=(
                            planner_now - 1.0 / self.config.timing.reference_hz
                        ),
                        monotonic_time=planner_now,
                    )
                self._accepted_qcmd_by_side = {
                    side: qpos.copy() for side, qpos in actual.items()
                }
                raise
            finally:
                self._pending_proposals = {}
            self._worker_physics_ticks += 1

    def _consume_reference_locked(self, planner_now: float) -> None:
        reference = self._pending_reference
        hold_reason = self._hold_reason
        self._pending_reference = None
        self._hold_reason = None
        if hold_reason is not None:
            sequence = self._next_reference_sequence
            self._next_reference_sequence += 1
            for planner in self._planners.values():
                planner.quick_stop(
                    sequence=sequence,
                    monotonic_time=planner_now,
                    reason=hold_reason,
                )
            return
        if reference is None:
            return
        for side in self.config.active_sides:
            metadata = reference.metadata_by_side[side]
            policy = classify_position_trajectory_policy(
                metadata,
                metadata_mapping(metadata.get("singularity_metadata")),
                degraded_speed_scale=self.config.degraded_speed_scale,
            )
            planner = self._planners[side]
            if policy.quick_stop:
                planner.quick_stop(
                    sequence=reference.sequence,
                    monotonic_time=planner_now,
                    reason=policy.reason,
                )
            else:
                published = min(reference.published_monotonic_time, planner_now)
                planner.accept_target(
                    reference.qref_by_side[side],
                    sequence=reference.sequence,
                    published_monotonic_time=published,
                    monotonic_time=planner_now,
                    speed_scale=policy.speed_scale,
                    mode=policy.mode,
                    reason=policy.reason,
                )
        self._last_reference = reference

    def _build_due_command_locked(self, planner_now: float) -> DirectQposCommand | None:
        self._pending_proposals = {}
        if not self._direct.command_due:
            return None
        qcmd_by_side: dict[str, np.ndarray] = {}
        for side in self.config.active_sides:
            proposal = self._planners[side].propose(monotonic_time=planner_now)
            accepted = self._accepted_qcmd_by_side[side]
            qcmd, _ = final_follow_qpos_target(
                proposal.q_command,
                accepted,
                alpha=self.config.follow_qpos_alpha,
                bypass=proposal.quick_stop,
            )
            qcmd_by_side[side] = qcmd
            self._pending_proposals[side] = proposal
        sequence = self._next_command_sequence
        self._next_command_sequence += 1
        return DirectQposCommand(
            sequence=sequence,
            qpos_by_side=qcmd_by_side,
            source="mujoco_position_execution",
        )

    def _worker_loop(self) -> None:
        period = 1.0 / self.config.timing.physics_hz
        try:
            while not self._stop_event.is_set():
                scheduled = self._clock_origin + self._worker_physics_ticks * period
                now = float(self._clock())
                remaining = scheduled - now
                if remaining > 0.0:
                    self._sleep(min(remaining, period))
                    continue
                if now - scheduled > period:
                    with self._lock:
                        self._worker_deadline_misses += 1
                self.step_physics()
        except Exception as exc:  # noqa: BLE001 - worker fault must latch visibly.
            with self._lock:
                self._last_worker_error = f"{type(exc).__name__}: {exc}"
            self._stop_event.set()

    def close(self) -> None:
        self._stop_event.set()
        worker = self._worker
        self._worker = None
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=self.config.worker_join_timeout_sec)
        with self._lock:
            self._started = False


__all__ = ["MujocoPositionExecutionConfig", "MujocoPositionExecutionRig"]
