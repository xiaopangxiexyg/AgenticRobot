from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
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
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
    normalize_position_command_planner,
)
from robot_motion.position_control.horizon_governor import JointHorizonLimits
from robot_motion.position_control.sample_aligned_trajectory import (
    SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET,
    SampleAlignedJerkLimitedJointTrajectoryPlanner,
    normalize_sample_aligned_source_latch_mode,
)
from robot_motion.rm65.constants import (
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
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

from ..actuation import (
    GripperActuationProfile,
    MujocoActuatedPositionExecutor,
    MujocoDmtacGripperExecutor,
    PositionActuationProfile,
    identified_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from ..control import DirectQposCommand, DirectQposDebugController
from ..scene import MujocoScene
from .reference_lineage import (
    MujocoPlannerCommandFrameObservation,
    MujocoReferenceTargetActivationObservation,
    build_mujoco_planner_command_frame_observation,
    build_mujoco_reference_target_activation_observation,
)
from .execution_cut import MujocoExecutionStateCut


_SIDES = ("left", "right")


def _six_positive(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)) or np.any(vector <= 0.0):
        raise ValueError(f"{name} must contain six finite positive values.")
    return tuple(float(value) for value in vector)


@dataclass(frozen=True)
class MujocoPositionExecutionConfig:
    """Real-aligned position execution with profile-selected independent clocks.

    The canonical profiles are 15->60->120 and the HIL 60->120 input path;
    both use the same reference/IK/command implementation below.
    """

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
    position_command_planner: str = POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
    sample_aligned_source_hz: float = 15.0
    sample_aligned_source_latch_mode: str = (
        SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET
    )
    sample_aligned_horizon_source_steps: int = 2
    sample_aligned_safety_horizon_source_steps: int | None = None
    sample_aligned_scale_degraded_target: bool = True
    sample_aligned_jerk_regularization: float = 1e-9
    sample_aligned_jerk_delta_regularization: float = 0.0
    # Physical RM65 + DM-Tac execution is the simulation default. Callers that
    # intentionally need the legacy prescribed-qpos path must pass both fields
    # as ``None`` explicitly; absence is no longer an implicit debug switch.
    actuation_profile: PositionActuationProfile | None = field(
        default_factory=rm65_bimanual_udp_feedback_profile
    )
    gripper_actuation_profile: GripperActuationProfile | None = field(
        default_factory=identified_bimanual_dmtac_gripper_profile
    )
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
        planner = normalize_position_command_planner(self.position_command_planner)
        if planner not in {
            POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
        }:
            raise ValueError(
                "MuJoCo position execution requires an acceleration/jerk-limited planner."
            )
        object.__setattr__(self, "position_command_planner", planner)
        source_hz = float(self.sample_aligned_source_hz)
        source_latch_mode = normalize_sample_aligned_source_latch_mode(
            self.sample_aligned_source_latch_mode
        )
        horizon_source_steps = int(self.sample_aligned_horizon_source_steps)
        for name in (
            "sample_aligned_jerk_regularization",
            "sample_aligned_jerk_delta_regularization",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        if planner == POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED:
            ratio = float(self.timing.command_hz) / source_hz
            if (
                not math.isfinite(source_hz)
                or source_hz <= 0.0
                or not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9)
            ):
                raise ValueError(
                    "sample-aligned MuJoCo execution requires command_hz/source_hz "
                    "to be a positive integer ratio."
                )
            if (
                isinstance(self.sample_aligned_horizon_source_steps, bool)
                or horizon_source_steps != self.sample_aligned_horizon_source_steps
                or horizon_source_steps < 2
            ):
                raise ValueError(
                    "sample_aligned_horizon_source_steps must be an integer >= 2."
                )
            safety_horizon_source_steps = (
                None
                if self.sample_aligned_safety_horizon_source_steps is None
                else int(self.sample_aligned_safety_horizon_source_steps)
            )
            if safety_horizon_source_steps is not None and (
                isinstance(
                    self.sample_aligned_safety_horizon_source_steps,
                    bool,
                )
                or safety_horizon_source_steps
                != self.sample_aligned_safety_horizon_source_steps
                or safety_horizon_source_steps <= horizon_source_steps
            ):
                raise ValueError(
                    "sample_aligned_safety_horizon_source_steps must be an "
                    "integer greater than sample_aligned_horizon_source_steps."
                )
            if not math.isclose(follow, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "sample-aligned MuJoCo execution requires follow_qpos_alpha=1."
                )
        object.__setattr__(self, "sample_aligned_source_hz", source_hz)
        object.__setattr__(
            self,
            "sample_aligned_source_latch_mode",
            source_latch_mode,
        )
        object.__setattr__(
            self,
            "sample_aligned_horizon_source_steps",
            horizon_source_steps,
        )
        object.__setattr__(
            self,
            "sample_aligned_safety_horizon_source_steps",
            (
                None
                if self.sample_aligned_safety_horizon_source_steps is None
                else int(self.sample_aligned_safety_horizon_source_steps)
            ),
        )
        object.__setattr__(
            self,
            "sample_aligned_scale_degraded_target",
            bool(self.sample_aligned_scale_degraded_target),
        )
        if self.actuation_profile is not None:
            if not isinstance(self.actuation_profile, PositionActuationProfile):
                raise TypeError(
                    "actuation_profile must be a PositionActuationProfile or None."
                )
            if set(self.actuation_profile.joint_names_by_side) != set(sides):
                raise ValueError(
                    "actuation profile sides must exactly match active_sides."
                )
        if self.gripper_actuation_profile is not None:
            if self.actuation_profile is None:
                raise ValueError(
                    "physical gripper actuation requires the physical arm sender."
                )
            if not isinstance(self.gripper_actuation_profile, GripperActuationProfile):
                raise TypeError(
                    "gripper_actuation_profile must be a GripperActuationProfile or None."
                )
            if set(self.gripper_actuation_profile.side_by_gripper.values()) != set(sides):
                raise ValueError(
                    "gripper actuation profile sides must exactly match active_sides."
                )


@dataclass(frozen=True)
class _QueuedReference:
    sequence: int
    published_monotonic_time: float
    activate_at: float
    reference_segment_id: str | None
    execution_epoch: int | None
    group_segment_generation: int | None
    qref_by_side: Mapping[str, np.ndarray]
    metadata_by_side: Mapping[str, Mapping[str, object]]
    gripper_close_by_name: Mapping[str, float]


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
        self._scheduled_activation_guard: (
            Callable[[str, int, int, float], str] | None
        ) = None
        self._scheduled_active_work_guard: (
            Callable[[str, int, int, float], None] | None
        ) = None
        self._hold_reason: str | None = None
        self._last_reference: _QueuedReference | None = None
        self._last_worker_error: str | None = None
        self._worker_physics_ticks = 0
        self._worker_deadline_misses = 0
        self._pending_proposals: dict[str, JointTrajectoryProposal] = {}
        self._reference_activation_observations: list[
            MujocoReferenceTargetActivationObservation
        ] = []
        self._planner_command_frame_observations: list[
            MujocoPlannerCommandFrameObservation
        ] = []
        self._planners: dict[
            str,
            AccelerationLimitedJointTrajectoryPlanner
            | SampleAlignedJerkLimitedJointTrajectoryPlanner,
        ] = {}
        if config.actuation_profile is None:
            self._execution = DirectQposDebugController(
                scene,
                config.timing,
                active_sides=config.active_sides,
                record_ticks=False,
            )
            # Compatibility seam for existing direct-qpos diagnostics/tests.
            self._direct: DirectQposDebugController | None = self._execution
        else:
            self._execution = MujocoActuatedPositionExecutor(
                scene,
                config.timing,
                config.actuation_profile,
                active_sides=config.active_sides,
            )
            self._direct = None
        self._gripper_execution = (
            None
            if config.gripper_actuation_profile is None
            else MujocoDmtacGripperExecutor(
                scene,
                config.gripper_actuation_profile,
            )
        )
        self._accepted_qcmd_by_side: dict[str, np.ndarray] = {
            side: np.asarray(scene.get_arm_qpos(side), dtype=float).reshape(6).copy()
            for side in config.active_sides
        }

    @property
    def name(self) -> str:
        return "mujoco_position_execution_rig"

    def _new_planner(
        self,
    ) -> (
        AccelerationLimitedJointTrajectoryPlanner
        | SampleAlignedJerkLimitedJointTrajectoryPlanner
    ):
        max_step = np.full(
            6,
            np.deg2rad(float(self.config.max_command_step_deg)),
            dtype=float,
        )
        if (
            self.config.position_command_planner
            == POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED
        ):
            max_acceleration = np.asarray(
                self.config.max_acceleration_rad_s2,
                dtype=float,
            )
            effective_velocity = np.minimum(
                np.asarray(self.config.max_velocity_rad_s, dtype=float),
                max_step * float(self.config.timing.command_hz),
            )
            return SampleAlignedJerkLimitedJointTrajectoryPlanner(
                JointHorizonLimits(
                    min_position_rad=np.asarray(
                        RM65_ABSOLUTE_Q_MIN_RAD,
                        dtype=float,
                    ),
                    max_position_rad=np.asarray(
                        RM65_ABSOLUTE_Q_MAX_RAD,
                        dtype=float,
                    ),
                    max_velocity_rad_s=effective_velocity,
                    max_acceleration_rad_s2=max_acceleration,
                    max_jerk_rad_s3=(
                        max_acceleration / float(self.config.velocity_ramp_sec)
                    ),
                ),
                source_hz=float(self.config.sample_aligned_source_hz),
                command_hz=float(self.config.timing.command_hz),
                source_latch_mode=(
                    self.config.sample_aligned_source_latch_mode
                ),
                horizon_source_steps=int(
                    self.config.sample_aligned_horizon_source_steps
                ),
                safety_horizon_source_steps=(
                    self.config.sample_aligned_safety_horizon_source_steps
                ),
                scale_degraded_target=bool(
                    self.config.sample_aligned_scale_degraded_target
                ),
                jerk_regularization=float(
                    self.config.sample_aligned_jerk_regularization
                ),
                jerk_delta_regularization=float(
                    self.config.sample_aligned_jerk_delta_regularization
                ),
                reference_mode="linear_hold",
            )
        return AccelerationLimitedJointTrajectoryPlanner(
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

    def _reset_and_prime_planner(
        self,
        planner: (
            AccelerationLimitedJointTrajectoryPlanner
            | SampleAlignedJerkLimitedJointTrajectoryPlanner
        ),
        qpos: np.ndarray,
        *,
        monotonic_time: float,
    ) -> None:
        planner.reset(qpos, monotonic_time=monotonic_time)
        if isinstance(planner, SampleAlignedJerkLimitedJointTrajectoryPlanner):
            admission = planner.prepare_target(
                qpos,
                source_timestamp=(
                    monotonic_time
                    - 1.0 / float(self.config.sample_aligned_source_hz)
                ),
                speed_scale=1.0,
                mode="tracking",
                reason="mujoco_position_execution_start",
            )
            planner.activate_target(admission, sequence=0)
            return
        planner.accept_target(
            qpos,
            sequence=0,
            published_monotonic_time=(
                monotonic_time - 1.0 / self.config.timing.reference_hz
            ),
            monotonic_time=monotonic_time,
        )

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._clock_origin = float(self._clock())
            for side in self.config.active_sides:
                planner = self._new_planner()
                initial = self._accepted_qcmd_by_side[side]
                self._reset_and_prime_planner(
                    planner,
                    initial,
                    monotonic_time=self._clock_origin,
                )
                self._planners[side] = planner
            self._execution.reset(
                DirectQposCommand(
                    sequence=0,
                    qpos_by_side=self._accepted_qcmd_by_side,
                    source="mujoco_position_execution_start",
                )
            )
            if self._gripper_execution is not None:
                self._gripper_execution.reset(
                    {
                        name: self.scene.get_gripper_close(name)
                        for name in self.config.gripper_actuation_profile.joint_names_by_gripper
                    }
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
        return self._queue_commands(
            command,
            phase_recorder=phase_recorder,
            activate_at=None,
            reference_segment_id=None,
            execution_epoch=None,
            group_segment_generation=None,
            operation="queue_mujoco_qref",
        )

    def send_scheduled_commands(
        self,
        command: TeleopCommand,
        *,
        activate_at: float,
        reference_segment_id: str,
        execution_epoch: int,
        group_segment_generation: int,
    ) -> dict[str, CommandDispatchReceipt]:
        """Queue one qref without activating it before its segment START."""

        activation = float(activate_at)
        if not math.isfinite(activation):
            raise ValueError("activate_at must be finite.")
        segment_id = str(reference_segment_id).strip()
        if not segment_id:
            raise ValueError("reference_segment_id must be non-empty.")
        return self._queue_commands(
            command,
            phase_recorder=None,
            activate_at=activation,
            reference_segment_id=segment_id,
            execution_epoch=int(execution_epoch),
            group_segment_generation=int(group_segment_generation),
            operation="queue_mujoco_scheduled_qref",
        )

    def bind_scheduled_generation_guards(
        self,
        *,
        activation_guard: Callable[[str, int, int, float], str],
        active_work_guard: Callable[[str, int, int, float], None],
    ) -> None:
        """Bind the receiver-owned generation checks before this rig starts.

        These guards are intentionally receiver callbacks rather than a second
        generation registry in the Rig.  The activation check runs before a
        future qref is accepted by the planners; the active-work check runs
        before every subsequent planner/qcmd proposal derived from it.
        """

        if not callable(activation_guard) or not callable(active_work_guard):
            raise TypeError("Scheduled generation guards must be callable.")
        with self._lock:
            if self._started:
                raise RuntimeError("Scheduled generation guards bind before start().")
            if (
                self._scheduled_activation_guard is not None
                or self._scheduled_active_work_guard is not None
            ):
                raise RuntimeError("Scheduled generation guards are already bound.")
            self._scheduled_activation_guard = activation_guard
            self._scheduled_active_work_guard = active_work_guard

    def _queue_commands(
        self,
        command: TeleopCommand,
        *,
        phase_recorder,
        activate_at: float | None,
        reference_segment_id: str | None,
        execution_epoch: int | None,
        group_segment_generation: int | None,
        operation: str,
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
        gripper_targets: dict[str, float] = {}
        hand_receipts: dict[str, CommandDispatchReceipt] = {}
        hand_commands = {
            side: command.commands.get(f"{side}_hand")
            for side in self.config.active_sides
        }
        present_hands = {side: value for side, value in hand_commands.items() if value is not None}
        if present_hands and self._gripper_execution is None:
            raise ValueError(
                "hand commands require a configured physical gripper actuation profile."
            )
        if self._gripper_execution is not None:
            assert self.config.gripper_actuation_profile is not None
            for side, effector in present_hands.items():
                assert effector is not None
                if effector.command_type not in {"joint_vector", "gripper_scalar"}:
                    raise ValueError(
                        f"{effector.effector_name} must be joint_vector or gripper_scalar."
                    )
                values = np.asarray(effector.value, dtype=float).reshape(-1)
                if values.shape == (1,):
                    close = float(values[0])
                elif values.shape == (6,):
                    slot = int((effector.metadata or {}).get("gripper_slot", 0))
                    if not 0 <= slot < 6:
                        raise ValueError("gripper_slot must be in [0, 5].")
                    close = float(values[slot])
                else:
                    raise ValueError(
                        f"{effector.effector_name} must contain one scalar or six hand slots."
                    )
                if not math.isfinite(close) or not 0.0 <= close <= 1.0:
                    raise ValueError(
                        f"{effector.effector_name} close target must be in [0, 1]."
                    )
                gripper_name = self.config.gripper_actuation_profile.gripper_for_side(side)
                gripper_targets[gripper_name] = close
                hand_receipts[f"{side}_hand"] = CommandDispatchReceipt(
                    effector_name=f"{side}_hand",
                    accepted=True,
                    operation="queue_mujoco_gripper_close",
                    status="queued",
                    metadata={"gripper_name": gripper_name, "target_close": close},
                )
        published = float(self._clock())
        if not math.isfinite(published):
            raise RuntimeError("monotonic clock returned a non-finite value.")
        activation = published if activate_at is None else float(activate_at)
        if activation + 1e-12 < published:
            raise ValueError("A scheduled qref cannot activate before publication.")
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
                activate_at=activation,
                reference_segment_id=reference_segment_id,
                execution_epoch=execution_epoch,
                group_segment_generation=group_segment_generation,
                qref_by_side=qref_by_side,
                metadata_by_side=metadata_by_side,
                gripper_close_by_name=(
                    dict(gripper_targets)
                    if reference_segment_id is not None
                    else {}
                ),
            )
            # Compatibility: immediate/manual dispatch retains the historical
            # behavior of entering the gripper transport queue at publication.
            # Scheduled compound references must wait for the same guarded
            # activation point as their arm qref.
            if gripper_targets and reference_segment_id is None:
                assert self._gripper_execution is not None
                self._gripper_execution.queue_close_targets(gripper_targets)
            self._hold_reason = None
        receipts = {
            f"{side}_arm": CommandDispatchReceipt(
                effector_name=f"{side}_arm",
                accepted=True,
                operation=operation,
                status="queued",
                timestamp=published,
                metadata={
                    "reference_sequence": sequence,
                    "activate_at": activation,
                    "reference_segment_id": reference_segment_id,
                    "execution_epoch": execution_epoch,
                    "group_segment_generation": group_segment_generation,
                },
            )
            for side in self.config.active_sides
        }
        receipts.update(
            {
                name: CommandDispatchReceipt(
                    effector_name=receipt.effector_name,
                    accepted=receipt.accepted,
                    operation=receipt.operation,
                    status=receipt.status,
                    timestamp=published,
                    result_code=receipt.result_code,
                    result_text=receipt.result_text,
                    metadata=receipt.metadata,
                )
                for name, receipt in hand_receipts.items()
            }
        )
        return receipts

    def hold_motion(self, reason: str = "manual hold") -> None:
        with self._lock:
            self._pending_reference = None
            # A receiver cancellation invalidates both future activation and
            # planner work derived from the previously active generation.
            # Drop that reference identity and quick-stop at the last accepted
            # command until a new scheduled reference activates.
            self._last_reference = None
            self._pending_proposals = {}
            self._hold_reason = str(reason).strip() or "manual hold"
            if self._started:
                sequence = self._next_reference_sequence
                self._next_reference_sequence += 1
                planner_now = (
                    self._clock_origin + self._execution.physics_time_sec
                )
                for planner in self._planners.values():
                    planner.quick_stop(
                        sequence=sequence,
                        monotonic_time=planner_now,
                        reason=self._hold_reason,
                    )
            if self._gripper_execution is not None:
                self._gripper_execution.hold_targets()

    def accepted_qcmd_by_side(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {
                side: qpos.copy()
                for side, qpos in self._accepted_qcmd_by_side.items()
            }

    @property
    def reference_activation_observations(
        self,
    ) -> tuple[MujocoReferenceTargetActivationObservation, ...]:
        with self._lock:
            return tuple(self._reference_activation_observations)

    @property
    def planner_command_frame_observations(
        self,
    ) -> tuple[MujocoPlannerCommandFrameObservation, ...]:
        with self._lock:
            return tuple(self._planner_command_frame_observations)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            execution_metadata = dict(self._execution.execution_metadata)
            return {
                "started": self._started,
                "ready": self._started and self._last_worker_error is None,
                **execution_metadata,
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
                "pending_reference_activate_at": (
                    None if self._pending_reference is None else self._pending_reference.activate_at
                ),
                "pending_reference_segment_id": (
                    None
                    if self._pending_reference is None
                    else self._pending_reference.reference_segment_id
                ),
                "reference_activation_observation_count": len(
                    self._reference_activation_observations
                ),
                "planner_command_frame_observation_count": len(
                    self._planner_command_frame_observations
                ),
                "last_worker_error": self._last_worker_error,
                "physics_ticks": self._worker_physics_ticks,
                "deadline_misses": self._worker_deadline_misses,
                "planner_by_side": {
                    side: planner.snapshot() for side, planner in self._planners.items()
                },
                "execution": (
                    self._execution.snapshot()
                    if isinstance(self._execution, MujocoActuatedPositionExecutor)
                    else execution_metadata
                ),
                "gripper_execution": (
                    None
                    if self._gripper_execution is None
                    else self._gripper_execution.snapshot()
                ),
            }

    def sync_viewer(self, viewer_handle) -> None:
        """Synchronize a passive viewer without racing the physics owner."""

        if viewer_handle is None:
            return
        sync = getattr(viewer_handle, "sync", None)
        if not callable(sync):
            raise TypeError("viewer_handle must expose sync().")
        with self._lock:
            sync()

    def capture_scene(self, capture: Callable[[MujocoScene], object]) -> object:
        """Run one read-only scene capture without racing the physics owner."""

        if not callable(capture):
            raise TypeError("capture must be callable.")
        with self._lock:
            return capture(self.scene)

    def capture_active_reference(self) -> dict[str, object] | None:
        """Return a read-only copy of the qref currently owned by execution."""

        with self._lock:
            reference = self._last_reference
            if reference is None:
                return None
            return {
                "sequence": reference.sequence,
                "published_monotonic_time": (
                    reference.published_monotonic_time
                ),
                "activate_at": reference.activate_at,
                "qref_by_side": {
                    side: values.copy()
                    for side, values in reference.qref_by_side.items()
                },
            }

    def capture_execution_state_cut(
        self,
        *,
        sources: Sequence[str] | None = None,
    ) -> MujocoExecutionStateCut:
        """Capture measured qpos, active qref and model qpos under one lock.

        This is a passive observation seam.  In particular it does not call
        ``mj_forward``, alter live ``MjData``, or advance the execution worker.
        """

        requested = (
            self.config.active_sides
            if sources is None
            else tuple(str(value).strip() for value in sources)
        )
        if (
            not requested
            or any(not value for value in requested)
            or len(requested) != len(set(requested))
            or not set(requested).issubset(self.config.active_sides)
        ):
            raise ValueError(
                "sources must be a unique non-empty subset of active_sides."
            )
        with self._lock:
            if not self._started:
                raise RuntimeError(
                    "MujocoPositionExecutionRig.start() is required."
                )
            reference = self._last_reference
            active_qref = (
                None
                if reference is None
                else {
                    source: reference.qref_by_side[source].copy()
                    for source in requested
                }
            )
            return MujocoExecutionStateCut(
                sampled_at=float(self._clock()),
                measured_qpos_by_source={
                    source: self.scene.get_arm_qpos(source)
                    for source in requested
                },
                model_qpos=np.asarray(self.scene.data.qpos, dtype=float).copy(),
                physics_tick=self._worker_physics_ticks,
                active_reference_sequence=(
                    None if reference is None else reference.sequence
                ),
                active_qref_by_source=active_qref,
                active_reference_segment_id=(
                    None
                    if reference is None
                    else reference.reference_segment_id
                ),
                active_execution_fence_sequence=(
                    None if reference is None else reference.execution_epoch
                ),
                active_group_segment_generation=(
                    None
                    if reference is None
                    else reference.group_segment_generation
                ),
            )

    def step_physics(self) -> None:
        """Advance one deterministic physics tick; used by tests and offline replay."""

        with self._lock:
            if not self._started:
                raise RuntimeError("MujocoPositionExecutionRig.start() is required.")
            if self._last_worker_error is not None:
                raise RuntimeError(self._last_worker_error)
            planner_now = self._clock_origin + self._execution.physics_time_sec
            self._consume_reference_locked(planner_now)
            command = self._build_due_command_locked(planner_now)
            proposals = self._pending_proposals
            try:
                if self._gripper_execution is not None:
                    self._gripper_execution.before_physics_step(
                        physics_time_sec=self._execution.physics_time_sec,
                        timestep_sec=self.config.timing.physics_timestep_sec,
                    )
                self._execution.step_physics(command)
                if proposals:
                    accepted = {
                        side: self._execution.accepted_arm_qpos(side)
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
                    active_reference = self._last_reference
                    if (
                        active_reference is not None
                        and active_reference.reference_segment_id is not None
                        and active_reference.execution_epoch is not None
                        and active_reference.group_segment_generation is not None
                        and command is not None
                    ):
                        self._planner_command_frame_observations.append(
                            build_mujoco_planner_command_frame_observation(
                                reference_segment_id=(
                                    active_reference.reference_segment_id
                                ),
                                reference_sequence=active_reference.sequence,
                                command_sequence=command.sequence,
                                execution_epoch=active_reference.execution_epoch,
                                group_segment_generation=(
                                    active_reference.group_segment_generation
                                ),
                                scheduled_for=planner_now,
                                observed_at=float(self._clock()),
                                qref_by_side=active_reference.qref_by_side,
                                proposals_by_side=proposals,
                                follow_qcmd_by_side=command.qpos_by_side,
                                accepted_qcmd_by_side=accepted,
                            )
                        )
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
                    side: self._execution.accepted_arm_qpos(side)
                    for side in self.config.active_sides
                }
                for side, planner in self._planners.items():
                    self._reset_and_prime_planner(
                        planner,
                        actual[side],
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
        if hold_reason is not None:
            self._pending_reference = None
            self._hold_reason = None
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
        if planner_now + 1e-12 < reference.activate_at:
            return
        if reference.reference_segment_id is not None:
            if (
                reference.execution_epoch is None
                or reference.group_segment_generation is None
            ):
                raise RuntimeError(
                    "A scheduled reference requires execution epoch and generation."
                )
            guard = self._scheduled_activation_guard
            if guard is None:
                raise RuntimeError("Scheduled reference activation guard is not bound.")
            generation_activation_id = guard(
                reference.reference_segment_id,
                reference.execution_epoch,
                reference.group_segment_generation,
                planner_now,
            )
            if not str(generation_activation_id).strip():
                raise RuntimeError("Scheduled activation guard returned no fact ID.")
        else:
            generation_activation_id = None
        self._pending_reference = None
        self._hold_reason = None
        sample_aligned_new_bucket: bool | None = None
        activated_qref_by_side: dict[str, np.ndarray] = {}
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
                activated_qref_by_side[side] = self._accepted_qcmd_by_side[
                    side
                ].copy()
            elif isinstance(
                planner,
                SampleAlignedJerkLimitedJointTrajectoryPlanner,
            ):
                admission = planner.prepare_target(
                    reference.qref_by_side[side],
                    source_timestamp=reference.published_monotonic_time,
                    speed_scale=policy.speed_scale,
                    mode=policy.mode,
                    reason=policy.reason,
                )
                if sample_aligned_new_bucket is None:
                    sample_aligned_new_bucket = bool(
                        admission.new_source_bucket
                    )
                elif sample_aligned_new_bucket != bool(
                    admission.new_source_bucket
                ):
                    raise RuntimeError(
                        "bimanual sample-aligned source buckets diverged."
                    )
                if admission.new_source_bucket:
                    planner.activate_target(
                        admission,
                        sequence=reference.sequence,
                    )
                activated_qref_by_side[side] = np.asarray(
                    admission.admitted_qpos,
                    dtype=float,
                ).copy()
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
                activated_qref_by_side[side] = reference.qref_by_side[side].copy()
        if reference.gripper_close_by_name:
            if self._gripper_execution is None:
                raise RuntimeError(
                    "Scheduled gripper targets require a configured gripper executor."
                )
            self._gripper_execution.queue_close_targets(
                reference.gripper_close_by_name
            )
        if sample_aligned_new_bucket is False:
            return
        if sample_aligned_new_bucket is True:
            reference = _QueuedReference(
                sequence=reference.sequence,
                published_monotonic_time=reference.published_monotonic_time,
                activate_at=reference.activate_at,
                reference_segment_id=reference.reference_segment_id,
                execution_epoch=reference.execution_epoch,
                group_segment_generation=reference.group_segment_generation,
                qref_by_side=activated_qref_by_side,
                metadata_by_side=reference.metadata_by_side,
                gripper_close_by_name=reference.gripper_close_by_name,
            )
        self._last_reference = reference
        if (
            reference.reference_segment_id is not None
            and reference.execution_epoch is not None
            and reference.group_segment_generation is not None
        ):
            self._reference_activation_observations.append(
                build_mujoco_reference_target_activation_observation(
                    reference_segment_id=reference.reference_segment_id,
                    generation_activation_id=generation_activation_id,
                    reference_sequence=reference.sequence,
                    execution_epoch=reference.execution_epoch,
                    group_segment_generation=reference.group_segment_generation,
                    scheduled_for=reference.activate_at,
                    activated_at_worker_time=planner_now,
                    observed_at=float(self._clock()),
                    qref_by_side=reference.qref_by_side,
                    activation_status_by_side={
                        side: classify_position_trajectory_policy(
                            reference.metadata_by_side[side],
                            metadata_mapping(
                                reference.metadata_by_side[side].get(
                                    "singularity_metadata"
                                )
                            ),
                            degraded_speed_scale=self.config.degraded_speed_scale,
                        ).mode
                        for side in self.config.active_sides
                    },
                )
            )

    def _build_due_command_locked(self, planner_now: float) -> DirectQposCommand | None:
        self._pending_proposals = {}
        if not self._execution.command_due:
            return None
        active_reference = self._last_reference
        if active_reference is not None and active_reference.reference_segment_id is not None:
            if (
                active_reference.execution_epoch is None
                or active_reference.group_segment_generation is None
            ):
                raise RuntimeError(
                    "An active scheduled reference requires execution epoch and generation."
                )
            guard = self._scheduled_active_work_guard
            if guard is None:
                raise RuntimeError("Scheduled active-work generation guard is not bound.")
            guard(
                active_reference.reference_segment_id,
                active_reference.execution_epoch,
                active_reference.group_segment_generation,
                planner_now,
            )
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
                    missed = int((now - scheduled) // period)
                    with self._lock:
                        self._worker_deadline_misses += missed
                        # Rebase the wall-clock schedule instead of running a
                        # catch-up burst that can starve target publication,
                        # observation capture, and viewer synchronization.
                        self._clock_origin += missed * period
                self.step_physics()
        except Exception as exc:  # noqa: BLE001 - worker fault must latch visibly.
            with self._lock:
                self._last_worker_error = f"{type(exc).__name__}: {exc}"
            self._stop_event.set()

    def close(self) -> None:
        self._stop_event.set()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=self.config.worker_join_timeout_sec)
            if worker.is_alive():
                # Do not follow a bounded join with an unbounded acquisition
                # of the lock the same worker may still own.  Retain the
                # worker identity so a later idempotent close can retry after
                # the native physics call unwinds.
                raise TimeoutError(
                    "MuJoCo position execution worker did not stop before "
                    f"{self.config.worker_join_timeout_sec:.3f}s."
                )
        self._worker = None
        with self._lock:
            self._started = False


__all__ = ["MujocoPositionExecutionConfig", "MujocoPositionExecutionRig"]
