from __future__ import annotations

import math
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.deploy.actions import (
    FilteredNormalizedAction,
    NormalizedAction,
)
from robot_motion.deploy.canonical_action import (
    CanonicalDeployActionPipeline,
    CanonicalDeployActionResult,
)
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
    pose_matrix_to_xyzquat,
    vector_or_none,
    xyzquat_to_pose_matrix,
)
from robot_motion.position_control import (
    CausalMotionResampler,
    DEFAULT_POSITION_COMMAND_HZ,
    DEFAULT_POSITION_REFERENCE_HZ,
    DEFAULT_POSITION_SAMPLE_HZ,
    ResampledMotionReference,
    TimestampedMotionTarget,
)
from robot_motion.rm65 import (
    RM65PositionCollisionPlanResult,
    RM65PositionPlannerProtocol,
)
from robot_motion.runtime import TeleopCommand
from robot_motion.runtime import (
    ExecutionCancellation,
    ExecutionCancellationParticipantSpec,
    ExecutionEpochGuard,
    ExecutionWorkItem,
    WorkerEpochDrainReceipt,
)
from robot_motion.safety_constraints import (
    constrain_joint_vector,
    workspace_distance_l2,
)

from .reference_observation import (
    LegacyPositionReferenceObserver,
    LegacyReferenceDerivation,
    LegacyReferenceGenerationInvalidation,
    LegacyReferenceResolutionFailureDerivation,
    LegacyReferenceResolutionDerivation,
    LegacyReferenceSourceFact,
)


_TIME_EPSILON_SEC = 1e-12
_DEFAULT_SAMPLE_HZ = DEFAULT_POSITION_SAMPLE_HZ
_DEFAULT_REFERENCE_HZ = DEFAULT_POSITION_REFERENCE_HZ
_DEFAULT_COMMAND_HZ = DEFAULT_POSITION_COMMAND_HZ


class PositionDeployQueueFullError(RuntimeError):
    """The bounded policy-keyframe FIFO cannot accept another input."""


class PositionDeployExecutorFault(RuntimeError):
    """A background position-reference failure has been fault-latched."""


@dataclass(frozen=True)
class PositionDeployExecutorConfig:
    """Rates and safety timing for real-only online position deployment.

    ``stale_after_sec`` is measured from the last accepted input keyframe and
    must exceed one sample period.  The compatibility path deliberately runs
    one sample behind; chunk-lookahead mode instead uses an already-available
    same-chunk successor at the current logical time. Collision reset
    freshness uses monotonic hardware measurement timestamps. The
    non-authoritative fallback is dry-run only and is ignored whenever a real
    dispatch callback exists.
    """

    sample_hz: float = _DEFAULT_SAMPLE_HZ
    reference_hz: float = _DEFAULT_REFERENCE_HZ
    command_hz: float = _DEFAULT_COMMAND_HZ
    stale_after_sec: float = 0.25
    queue_size: int = 32
    debug_events_enabled: bool = False
    debug_event_queue_size: int = 1024
    shutdown_timeout_sec: float = 2.0
    collision_required: bool = False
    collision_reset_measurement_max_age_sec: float = 0.25
    collision_reset_pair_max_skew_sec: float = 0.05
    safety_rearm_measurement_max_age_sec: float = 0.1
    allow_non_authoritative_collision_reset_in_dry_run: bool = False
    # Opt-in deployment mode: a same-chunk successor is already known when the
    # current sample-rate action is consumed, so references query the current logical
    # time instead of adding the legacy one-sample causal delay.
    chunk_lookahead_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "sample_hz",
            "reference_hz",
            "command_hz",
            "stale_after_sec",
            "shutdown_timeout_sec",
            "collision_reset_measurement_max_age_sec",
            "collision_reset_pair_max_skew_sec",
            "safety_rearm_measurement_max_age_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.reference_hz + _TIME_EPSILON_SEC < self.sample_hz:
            raise ValueError("reference_hz must be >= sample_hz")
        if self.command_hz + _TIME_EPSILON_SEC < self.reference_hz:
            raise ValueError("command_hz must be >= reference_hz")
        if self.stale_after_sec <= (1.0 / self.sample_hz) + _TIME_EPSILON_SEC:
            raise ValueError("stale_after_sec must exceed one sample period")
        if (
            isinstance(self.queue_size, bool)
            or int(self.queue_size) != self.queue_size
            or int(self.queue_size) <= 0
        ):
            raise ValueError("queue_size must be a positive integer")
        object.__setattr__(self, "queue_size", int(self.queue_size))
        object.__setattr__(
            self,
            "debug_events_enabled",
            bool(self.debug_events_enabled),
        )
        if (
            isinstance(self.debug_event_queue_size, bool)
            or int(self.debug_event_queue_size) != self.debug_event_queue_size
            or int(self.debug_event_queue_size) <= 0
        ):
            raise ValueError("debug_event_queue_size must be a positive integer")
        object.__setattr__(
            self,
            "debug_event_queue_size",
            int(self.debug_event_queue_size),
        )
        object.__setattr__(self, "collision_required", bool(self.collision_required))
        object.__setattr__(
            self,
            "chunk_lookahead_enabled",
            bool(self.chunk_lookahead_enabled),
        )
        object.__setattr__(
            self,
            "allow_non_authoritative_collision_reset_in_dry_run",
            bool(self.allow_non_authoritative_collision_reset_in_dry_run),
        )

    @property
    def sample_period_sec(self) -> float:
        return 1.0 / self.sample_hz

    @property
    def reference_period_sec(self) -> float:
        return 1.0 / self.reference_hz


@dataclass(frozen=True)
class _Submission:
    generation: int
    source_index: int
    timestamp: float
    pose_by_side: Mapping[str, np.ndarray]
    qpos_by_side: Mapping[str, np.ndarray]
    filtered: FilteredNormalizedAction
    observation: DeploymentObservation
    policy_name: str
    discrete_command: TeleopCommand | None = None
    anchor_source_index: int | None = None
    anchor_timestamp: float | None = None
    anchor_pose_by_side: Mapping[str, np.ndarray] | None = None
    anchor_qpos_by_side: Mapping[str, np.ndarray] | None = None
    collision_reset_qpos_by_side: Mapping[str, np.ndarray] | None = None
    collision_reset_metadata: Mapping[str, Any] | None = None
    logical_timestamp: float | None = None
    current_already_staged: bool = False
    lookahead_source_index: int | None = None
    lookahead_timestamp: float | None = None
    lookahead_pose_by_side: Mapping[str, np.ndarray] | None = None
    lookahead_qpos_by_side: Mapping[str, np.ndarray] | None = None
    lookahead_filtered: FilteredNormalizedAction | None = None
    lookahead_observation: DeploymentObservation | None = None
    lookahead_policy_name: str | None = None
    lookahead_discrete_command: TeleopCommand | None = None
    execution_work_item: ExecutionWorkItem | None = None
    lookahead_execution_work_item: ExecutionWorkItem | None = None


@dataclass(frozen=True)
class _ReferenceTemplate:
    filtered: FilteredNormalizedAction
    observation: DeploymentObservation
    policy_name: str
    source_timestamp: float
    discrete_command: TeleopCommand | None = None
    execution_work_item: ExecutionWorkItem | None = None


@dataclass(frozen=True)
class _DiscreteDispatchItem:
    generation: int
    source_index: int
    source_timestamp: float
    logical_emit_monotonic: float
    logical_emit_wall: float
    skipped_before_this: int
    command: TeleopCommand


def _finite_time(value: float, *, name: str) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(timestamp):
        raise ValueError(f"{name} must be finite")
    return timestamp


def _measurement_timestamp_from_arm_state(state: Mapping[str, Any]) -> float | None:
    """Read a successful feedback timestamp without promoting a failed read."""

    for name in (
        "actual_state_timestamp",
        "hardware_measurement_timestamp",
        "hardware_refresh_time",
    ):
        value = state.get(name)
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(timestamp):
            return timestamp
    return None


def _rate_label(value: float) -> str:
    rounded = round(float(value))
    if math.isclose(float(value), float(rounded), rel_tol=0.0, abs_tol=1e-9):
        return str(int(rounded))
    return f"{float(value):g}"


def _copy_filtered_action(filtered: FilteredNormalizedAction) -> FilteredNormalizedAction:
    action = filtered.action
    return FilteredNormalizedAction(
        action=NormalizedAction(
            vector=np.asarray(action.vector, dtype=np.float32).copy(),
            arm_joint_targets={
                str(side): np.asarray(value, dtype=float).copy()
                for side, value in action.arm_joint_targets.items()
            },
            metadata=dict(action.metadata),
        ),
        faults=list(filtered.faults),
        metadata=dict(filtered.metadata),
    )


def _same_execution_scope(
    left: ExecutionWorkItem,
    right: ExecutionWorkItem,
) -> bool:
    return bool(
        left.control_domain_id == right.control_domain_id
        and left.authority_epoch == right.authority_epoch
        and left.execution_epoch == right.execution_epoch
        and dict(left.segment_generation_by_group)
        == dict(right.segment_generation_by_group)
    )


def _copy_teleop_command(command: TeleopCommand) -> TeleopCommand:
    return TeleopCommand(
        commands={
            str(name): effector_command.copy()
            for name, effector_command in command.commands.items()
        },
        timestamp=command.timestamp,
        metadata=dict(command.metadata),
        execution_work_item=command.execution_work_item,
    )


def _target_maps_equal(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> bool:
    if set(left) != set(right):
        return False
    return all(
        np.allclose(
            np.asarray(left[side], dtype=float),
            np.asarray(right[side], dtype=float),
            rtol=0.0,
            atol=1e-7,
        )
        for side in left
    )


def _slim_observation(observation: DeploymentObservation) -> DeploymentObservation:
    """Retain canonical-resolution state without pinning camera frames in the FIFO."""

    return DeploymentObservation(
        timestamp=float(observation.timestamp),
        step_index=int(observation.step_index),
        active_sides=tuple(observation.active_sides),
        arms_enabled=bool(observation.arms_enabled),
        hands_enabled=False,
        state=np.asarray(observation.state, dtype=np.float32).copy(),
        state_valid=np.asarray(observation.state_valid, dtype=bool).copy(),
        last_action=np.asarray(observation.last_action, dtype=np.float32).copy(),
        faults=list(observation.faults),
        metadata=dict(observation.metadata),
    )


class PositionDeployExecutor:
    """Generate causal arm references from already-safe policy keyframes.

    This class intentionally lives under :mod:`teleop_runtime`: its dispatch
    callback is the real-hardware boundary.  Interpolation itself is shared
    with simulation through :class:`robot_motion.position_control.CausalMotionResampler`.

    A missing ``dispatch_callback`` is a pure dry-run.  Canonical resolution,
    reference metadata, timing, fault checks, and counters still run, but no
    backend method is called.
    """

    def __init__(
        self,
        *,
        config: PositionDeployExecutorConfig,
        canonical_pipeline: CanonicalDeployActionPipeline,
        dispatch_callback: Callable[[TeleopCommand], object] | None = None,
        discrete_dispatch_callback: Callable[[TeleopCommand], object] | None = None,
        hold_callback: Callable[[], None] | None = None,
        collision_planner_factory: Callable[[], RM65PositionPlannerProtocol] | None = None,
        reference_observer: LegacyPositionReferenceObserver | None = None,
        execution_epoch_guard: ExecutionEpochGuard | None = None,
        worker_drain_participant: ExecutionCancellationParticipantSpec | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(config, PositionDeployExecutorConfig):
            raise TypeError("config must be a PositionDeployExecutorConfig")
        if not isinstance(canonical_pipeline, CanonicalDeployActionPipeline):
            raise TypeError("canonical_pipeline must be a CanonicalDeployActionPipeline")
        if not bool(canonical_pipeline.arms_enabled):
            raise ValueError("canonical_pipeline must enable arms")
        if bool(canonical_pipeline.hands_enabled):
            raise ValueError("canonical_pipeline must be arm-only (hands_enabled=False)")
        if dispatch_callback is not None and not callable(dispatch_callback):
            raise TypeError("dispatch_callback must be callable or None")
        if discrete_dispatch_callback is not None and not callable(
            discrete_dispatch_callback
        ):
            raise TypeError("discrete_dispatch_callback must be callable or None")
        if discrete_dispatch_callback is not None and dispatch_callback is None:
            raise ValueError(
                "discrete_dispatch_callback requires the arm dispatch_callback"
            )
        if hold_callback is not None and not callable(hold_callback):
            raise TypeError("hold_callback must be callable or None")
        if collision_planner_factory is not None and not callable(
            collision_planner_factory
        ):
            raise TypeError("collision_planner_factory must be callable or None")
        if reference_observer is not None and not callable(
            getattr(reference_observer, "submit_nowait", None)
        ):
            raise TypeError("reference_observer must provide submit_nowait() or be None")
        if (execution_epoch_guard is None) != (worker_drain_participant is None):
            raise ValueError(
                "execution_epoch_guard and worker_drain_participant must be configured together"
            )
        if execution_epoch_guard is not None and not isinstance(
            execution_epoch_guard,
            ExecutionEpochGuard,
        ):
            raise TypeError("execution_epoch_guard must be an ExecutionEpochGuard")
        if worker_drain_participant is not None:
            if not isinstance(
                worker_drain_participant,
                ExecutionCancellationParticipantSpec,
            ):
                raise TypeError(
                    "worker_drain_participant must be an ExecutionCancellationParticipantSpec"
                )
            if worker_drain_participant.worker_kind != "reference_worker":
                raise ValueError("position deploy drain participant must be a reference_worker")
        if config.collision_required and collision_planner_factory is None:
            raise ValueError(
                "collision_required position deploy needs a collision_planner_factory"
            )
        if config.collision_required and tuple(canonical_pipeline.active_sides) != (
            "left",
            "right",
        ):
            raise ValueError(
                "collision-required position deploy needs authoritative left+right state; "
                "single-arm mode cannot guess the passive arm qpos/TCP"
            )
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(wall_clock):
            raise TypeError("wall_clock must be callable")

        self.config = config
        self.canonical_pipeline = canonical_pipeline
        self.dispatch_callback = dispatch_callback
        self.discrete_dispatch_callback = discrete_dispatch_callback
        self.hold_callback = hold_callback
        self._collision_planner_factory = collision_planner_factory
        self._reference_observer = reference_observer
        self._execution_epoch_guard = execution_epoch_guard
        self._worker_drain_participant = worker_drain_participant
        self._collision_planner: RM65PositionPlannerProtocol | None = None
        self._collision_planner_initialized = False
        self._clock = clock
        self._wall_clock = wall_clock

        self._queue: queue.Queue[_Submission] = queue.Queue(maxsize=config.queue_size)
        # A one-slot latest-wins mailbox prevents a slow hand transport from
        # replaying a FIFO backlog as a catch-up burst.
        self._discrete_queue: queue.Queue[_DiscreteDispatchItem] = queue.Queue(maxsize=1)
        max_samples = max(256, config.queue_size * 4)
        self._pose_resampler = CausalMotionResampler(max_samples=max_samples)
        self._qpos_resampler = CausalMotionResampler(max_samples=max_samples)
        self._max_templates = max_samples * 2
        self._templates: OrderedDict[int, _ReferenceTemplate] = OrderedDict()
        self._legacy_reference_source_facts: dict[
            tuple[str, int], LegacyReferenceSourceFact
        ] = {}

        self._lock = threading.RLock()
        self._dispatch_lock = threading.RLock()
        self._discrete_dispatch_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._discrete_thread: threading.Thread | None = None

        self._started = False
        self._closing = False
        self._closed = False
        self._closed_event = threading.Event()
        self._close_error: BaseException | None = None
        self._active = False
        self._generation = 0
        self._next_source_index = 0
        self._submission_stream_initialized = False
        self._stream_signature: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        self._last_submitted_timestamp: float | None = None
        self._last_source_timestamp: float | None = None
        self._next_tick_timestamp: float | None = None
        self._staged_lookahead_source_index: int | None = None
        self._staged_lookahead_timestamp: float | None = None
        self._staged_lookahead_pose_by_side: Mapping[str, np.ndarray] | None = None
        self._staged_lookahead_qpos_by_side: Mapping[str, np.ndarray] | None = None
        self._staged_lookahead_discrete_command: TeleopCommand | None = None
        self._staged_lookahead_execution_work_item: ExecutionWorkItem | None = None
        self._terminal_pose_by_side: Mapping[str, np.ndarray] | None = None
        self._terminal_qpos_by_side: Mapping[str, np.ndarray] | None = None
        self._terminal_timestamp: float | None = None
        self._last_discrete_source_index = -1
        self._fault: BaseException | None = None
        self._fault_stage: str | None = None
        self._fault_metadata: dict[str, Any] | None = None
        self._fault_pending = False
        self._fault_pending_stage: str | None = None
        self._fault_pending_error: str | None = None

        self._submitted_count = 0
        self._reference_count = 0
        self._arm_dispatch_attempt_count = 0
        self._arm_dispatch_failed_count = 0
        self._dispatch_count = 0
        self._dry_run_count = 0
        self._discrete_submitted_count = 0
        self._discrete_reference_count = 0
        self._discrete_dispatch_attempt_count = 0
        self._discrete_dispatch_count = 0
        self._discrete_dispatch_failed_count = 0
        self._discrete_backend_skipped_count = 0
        self._discrete_canceled_count = 0
        self._discrete_dry_run_count = 0
        self._discrete_skipped_count = 0
        self._discrete_worker_coalesced_count = 0
        self._queue_full_count = 0
        self._clear_count = 0
        self._hold_count = 0
        self._hold_failure_count = 0
        self._last_hold_error: str | None = None
        self._hold_retry_required = False
        self._hold_callback_thread_id: int | None = None
        self._missed_reference_ticks = 0
        self._last_reference_metadata: dict[str, Any] | None = None
        self._last_discrete_metadata: dict[str, Any] | None = None
        self._debug_events: deque[dict[str, Any]] = deque(
            maxlen=config.debug_event_queue_size
        )
        self._debug_event_drop_count = 0
        self._collision_reset_count = 0
        self._collision_tracking_count = 0
        self._collision_limited_count = 0
        self._collision_backoff_count = 0
        self._collision_hold_count = 0
        self._last_collision_metadata: dict[str, Any] | None = None
        self._last_collision_reset_metadata: dict[str, Any] | None = None
        self._reference_observer_error_count = 0
        self._last_reference_observer_error: str | None = None
        self._execution_epoch_rejected_count = 0
        self._inflight_execution_work_item: ExecutionWorkItem | None = None
        self._last_completed_execution_work_item: ExecutionWorkItem | None = None
        self._last_old_epoch_invocation_started_at: float | None = None
        self._last_old_epoch_invocation_completed_at: float | None = None

    @property
    def dry_run(self) -> bool:
        return self.dispatch_callback is None and self.discrete_dispatch_callback is None

    def prepare_collision_planner(self) -> None:
        """Construct the coupled planner after real backends loaded active TCPs.

        Construction is intentionally separate from ``__init__`` because RM65
        can auto-load its active tool profile during ``backend.start()``.  The
        first accepted stream still resets the planner from that observation's
        complete measured/authoritative dual-arm qpos.
        """

        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("position deploy executor is closed")
            if self._collision_planner is not None:
                return
            factory = self._collision_planner_factory
        if factory is None:
            if self.config.collision_required:
                raise RuntimeError("required collision planner factory is unavailable")
            return
        planner = factory()
        if planner is None:
            raise RuntimeError("collision planner factory returned None")
        for method_name in ("reset", "plan_pose_reference"):
            if not callable(getattr(planner, method_name, None)):
                raise TypeError(
                    f"collision planner must provide callable {method_name}()"
                )
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("position deploy executor closed while preparing collision planner")
            if self._collision_planner is None:
                self._collision_planner = planner

    def start(self) -> None:
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("position deploy executor is closed")
            if self._started:
                return
            self._started = True
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"position-deploy-{_rate_label(self.config.reference_hz)}hz",
                daemon=True,
            )
            thread = self._thread
            if self.discrete_dispatch_callback is not None:
                self._discrete_thread = threading.Thread(
                    target=self._run_discrete_dispatch,
                    name="position-deploy-discrete-dispatch",
                    daemon=True,
                )
            discrete_thread = self._discrete_thread
            # Publish only started Thread objects; close() can therefore never
            # race into join() on a not-yet-started worker.
            thread.start()
            if discrete_thread is not None:
                discrete_thread.start()

    def submit(
        self,
        filtered: FilteredNormalizedAction,
        observation: DeploymentObservation,
        policy_name: str,
        timestamp_monotonic: float | None = None,
        discrete_command: TeleopCommand | None = None,
        lookahead_filtered: FilteredNormalizedAction | None = None,
        lookahead_observation: DeploymentObservation | None = None,
        lookahead_discrete_command: TeleopCommand | None = None,
        execution_work_item: ExecutionWorkItem | None = None,
        lookahead_execution_work_item: ExecutionWorkItem | None = None,
    ) -> int:
        """Enqueue one safe keyframe without applying its safety filter again.

        Returns the non-negative source index assigned to the keyframe.  Queue
        saturation is reported synchronously via
        :class:`PositionDeployQueueFullError`; no keyframe is overwritten or
        silently dropped.
        """

        if not isinstance(filtered, FilteredNormalizedAction):
            raise TypeError("filtered must be a FilteredNormalizedAction")
        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation must be a DeploymentObservation")
        if lookahead_filtered is not None and not self.config.chunk_lookahead_enabled:
            raise ValueError(
                "lookahead_filtered requires chunk_lookahead_enabled=True"
            )
        if lookahead_filtered is not None and not isinstance(
            lookahead_filtered,
            FilteredNormalizedAction,
        ):
            raise TypeError("lookahead_filtered must be a FilteredNormalizedAction or None")
        if lookahead_observation is not None and not isinstance(
            lookahead_observation,
            DeploymentObservation,
        ):
            raise TypeError("lookahead_observation must be a DeploymentObservation or None")
        if filtered.faults:
            self.latch_safety_fault(
                filtered.faults,
                observation=observation,
                stage="15hz_keyframe_safety",
            )
            raise PositionDeployExecutorFault(
                "cannot submit a keyframe with safety faults: "
                + ", ".join(str(fault) for fault in filtered.faults)
            )
        if lookahead_filtered is not None and lookahead_filtered.faults:
            raise PositionDeployExecutorFault(
                "cannot stage a lookahead keyframe with safety faults: "
                + ", ".join(str(fault) for fault in lookahead_filtered.faults)
            )
        if lookahead_discrete_command is not None and lookahead_filtered is None:
            raise ValueError(
                "lookahead_discrete_command requires lookahead_filtered"
            )
        guard_enabled = self._execution_epoch_guard is not None
        if guard_enabled and execution_work_item is None:
            raise ValueError(
                "epoch-mediated position deploy requires an execution_work_item"
            )
        if execution_work_item is not None:
            if not isinstance(execution_work_item, ExecutionWorkItem):
                raise TypeError("execution_work_item must be an ExecutionWorkItem or None")
            if execution_work_item.work_kind != "reference_segment":
                raise ValueError(
                    "position keyframe execution_work_item must use reference_segment"
                )
        if lookahead_filtered is not None and guard_enabled:
            if lookahead_execution_work_item is None:
                raise ValueError(
                    "epoch-mediated lookahead requires lookahead_execution_work_item"
                )
        elif lookahead_execution_work_item is not None:
            raise ValueError(
                "lookahead_execution_work_item requires lookahead_filtered and an epoch guard"
            )
        if lookahead_execution_work_item is not None:
            if not isinstance(lookahead_execution_work_item, ExecutionWorkItem):
                raise TypeError(
                    "lookahead_execution_work_item must be an ExecutionWorkItem"
                )
            if lookahead_execution_work_item.work_kind != "reference_segment":
                raise ValueError(
                    "lookahead execution work must use reference_segment"
                )
            assert execution_work_item is not None
            if not _same_execution_scope(
                execution_work_item,
                lookahead_execution_work_item,
            ):
                raise ValueError(
                    "current and lookahead execution work must share control epoch/generation"
                )
        for command_name, candidate_command in (
            ("discrete_command", discrete_command),
            ("lookahead_discrete_command", lookahead_discrete_command),
        ):
            if candidate_command is None:
                continue
            if not isinstance(candidate_command, TeleopCommand):
                raise TypeError(f"{command_name} must be a TeleopCommand or None")
            non_hand_effectors = [
                str(name)
                for name in candidate_command.commands
                if not str(name).endswith("_hand")
            ]
            if non_hand_effectors:
                raise ValueError(
                    "discrete_command must contain only hand effectors: "
                    + ", ".join(non_hand_effectors)
                )
            if (
                candidate_command.commands
                and self.dispatch_callback is not None
                and self.discrete_dispatch_callback is None
            ):
                raise ValueError(
                    "a real arm dispatch cannot silently dry-run its discrete hand command"
                )
        timestamp = _finite_time(
            self._clock() if timestamp_monotonic is None else timestamp_monotonic,
            name="timestamp_monotonic",
        )

        pose_by_side, qpos_by_side = self._target_maps(filtered, observation)
        resolved_lookahead_observation = (
            observation
            if lookahead_observation is None
            else lookahead_observation
        )
        lookahead_pose_by_side: Mapping[str, np.ndarray] = {}
        lookahead_qpos_by_side: Mapping[str, np.ndarray] = {}
        if lookahead_filtered is not None:
            (
                lookahead_pose_by_side,
                lookahead_qpos_by_side,
            ) = self._target_maps(
                lookahead_filtered,
                resolved_lookahead_observation,
            )
        try:
            self._validate_observation_reference_domain(
                observation,
                pose_sides=tuple(pose_by_side),
                qpos_sides=tuple(qpos_by_side),
                require_fresh_measurement=False,
            )
        except PositionDeployExecutorFault as exc:
            self.latch_safety_fault(
                ["reference_anchor_out_of_domain", str(exc)],
                observation=observation,
                stage="reference_anchor_safety",
            )
            raise
        if lookahead_filtered is not None:
            self._validate_observation_reference_domain(
                resolved_lookahead_observation,
                pose_sides=tuple(lookahead_pose_by_side),
                qpos_sides=tuple(lookahead_qpos_by_side),
                require_fresh_measurement=False,
            )
        signature = (tuple(sorted(pose_by_side)), tuple(sorted(qpos_by_side)))
        lookahead_signature = (
            tuple(sorted(lookahead_pose_by_side)),
            tuple(sorted(lookahead_qpos_by_side)),
        )
        if lookahead_filtered is not None and lookahead_signature != signature:
            raise ValueError(
                "current and lookahead pose/qpos side modalities must match"
            )
        if self.config.collision_required:
            if qpos_by_side:
                raise ValueError(
                    "collision-required online deploy currently accepts target-pose keyframes only; "
                    "direct qpos policy output has no committed coupled-IK state"
                )
            if set(pose_by_side) != {"left", "right"}:
                raise ValueError(
                    "collision-required online deploy needs left+right target poses at each keyframe"
                )
            if lookahead_filtered is not None and set(lookahead_pose_by_side) != {
                "left",
                "right",
            }:
                raise ValueError(
                    "collision-required lookahead needs left+right target poses"
                )

        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("position deploy executor is closed")
            self._raise_if_fault_locked()
            if (
                self._last_submitted_timestamp is not None
                and timestamp <= self._last_submitted_timestamp + _TIME_EPSILON_SEC
            ):
                raise ValueError("submitted keyframe timestamps must be strictly increasing")
            if self._stream_signature is not None and signature != self._stream_signature:
                raise ValueError(
                    "pose/qpos side modalities cannot change within an active stream; "
                    "call clear() before changing modes"
                )

            first = not self._submission_stream_initialized
            anchor_index: int | None = None
            anchor_timestamp: float | None = None
            anchor_pose: Mapping[str, np.ndarray] | None = None
            anchor_qpos: Mapping[str, np.ndarray] | None = None
            collision_reset_qpos: Mapping[str, np.ndarray] | None = None
            collision_reset_metadata: Mapping[str, Any] | None = None
            next_source_index = self._next_source_index
            logical_timestamp = timestamp
            current_already_staged = False
            staged_discrete_was_present = False
            effective_discrete_command = discrete_command
            if self.config.chunk_lookahead_enabled:
                staged_matches = bool(
                    self._staged_lookahead_source_index is not None
                    and self._staged_lookahead_timestamp is not None
                    and self._staged_lookahead_pose_by_side is not None
                    and self._staged_lookahead_qpos_by_side is not None
                    and _target_maps_equal(
                        pose_by_side,
                        self._staged_lookahead_pose_by_side,
                    )
                    and _target_maps_equal(
                        qpos_by_side,
                        self._staged_lookahead_qpos_by_side,
                    )
                )
                if staged_matches:
                    source_index = int(self._staged_lookahead_source_index)
                    logical_timestamp = float(self._staged_lookahead_timestamp)
                    current_already_staged = True
                    staged_discrete_was_present = bool(
                        self._staged_lookahead_discrete_command is not None
                        and self._staged_lookahead_discrete_command.commands
                    )
                    if (
                        effective_discrete_command is None
                        and self._staged_lookahead_discrete_command is not None
                    ):
                        # Re-submitting a staged arm endpoint must not erase the
                        # hand value that was staged beside it.  The discrete
                        # scheduler still keys on source_index, so it can run
                        # at most once.
                        effective_discrete_command = _copy_teleop_command(
                            self._staged_lookahead_discrete_command
                        )
                else:
                    source_index = next_source_index
                    next_source_index += 1
                    if not first and self._terminal_timestamp is not None:
                        # No same-chunk right endpoint was available.  Insert a
                        # hold anchor at the arrival cut, then interpolate the
                        # new chunk's first action over one accepted 15 Hz wait.
                        anchor_index = source_index
                        source_index = next_source_index
                        next_source_index += 1
                        anchor_timestamp = max(
                            timestamp,
                            float(self._terminal_timestamp)
                            + self.config.reference_period_sec,
                        )
                        anchor_pose = self._terminal_pose_by_side
                        anchor_qpos = self._terminal_qpos_by_side
                        logical_timestamp = (
                            anchor_timestamp + self.config.sample_period_sec
                        )

                if first and self.config.collision_required:
                    (
                        collision_reset_qpos,
                        collision_reset_metadata,
                    ) = self._collision_reset_qpos(
                        observation,
                        now_monotonic=timestamp,
                    )
            else:
                if first:
                    anchor_index = next_source_index
                    source_index = next_source_index + 1
                    next_source_index += 2
                    anchor_timestamp = timestamp - self.config.sample_period_sec
                    anchor_pose, anchor_qpos = self._anchor_maps(
                        observation,
                        pose_sides=tuple(pose_by_side),
                        qpos_sides=tuple(qpos_by_side),
                    )
                    if self.config.collision_required:
                        (
                            collision_reset_qpos,
                            collision_reset_metadata,
                        ) = self._collision_reset_qpos(
                            observation,
                            now_monotonic=timestamp,
                        )
                else:
                    source_index = next_source_index
                    next_source_index += 1

            lookahead_source_index: int | None = None
            lookahead_timestamp: float | None = None
            if lookahead_filtered is not None:
                lookahead_source_index = next_source_index
                next_source_index += 1
                lookahead_timestamp = (
                    float(logical_timestamp) + self.config.sample_period_sec
                )

            submission = _Submission(
                generation=self._generation,
                source_index=source_index,
                timestamp=timestamp,
                pose_by_side=pose_by_side,
                qpos_by_side=qpos_by_side,
                filtered=_copy_filtered_action(filtered),
                observation=_slim_observation(observation),
                policy_name=str(policy_name),
                discrete_command=(
                    None
                    if effective_discrete_command is None
                    else _copy_teleop_command(effective_discrete_command)
                ),
                anchor_source_index=anchor_index,
                anchor_timestamp=anchor_timestamp,
                anchor_pose_by_side=anchor_pose,
                anchor_qpos_by_side=anchor_qpos,
                collision_reset_qpos_by_side=collision_reset_qpos,
                collision_reset_metadata=collision_reset_metadata,
                logical_timestamp=float(logical_timestamp),
                current_already_staged=bool(current_already_staged),
                lookahead_source_index=lookahead_source_index,
                lookahead_timestamp=lookahead_timestamp,
                lookahead_pose_by_side=(
                    None
                    if lookahead_filtered is None
                    else dict(lookahead_pose_by_side)
                ),
                lookahead_qpos_by_side=(
                    None
                    if lookahead_filtered is None
                    else dict(lookahead_qpos_by_side)
                ),
                lookahead_filtered=(
                    None
                    if lookahead_filtered is None
                    else _copy_filtered_action(lookahead_filtered)
                ),
                lookahead_observation=(
                    None
                    if lookahead_filtered is None
                    else _slim_observation(resolved_lookahead_observation)
                ),
                lookahead_policy_name=(
                    None if lookahead_filtered is None else str(policy_name)
                ),
                lookahead_discrete_command=(
                    None
                    if lookahead_discrete_command is None
                    else _copy_teleop_command(lookahead_discrete_command)
                ),
                execution_work_item=execution_work_item,
                lookahead_execution_work_item=lookahead_execution_work_item,
            )
            try:
                self._queue.put_nowait(submission)
            except queue.Full as exc:
                self._queue_full_count += 1
                raise PositionDeployQueueFullError(
                    f"position deploy input FIFO is full ({self.config.queue_size})"
                ) from exc

            self._next_source_index = next_source_index
            self._submission_stream_initialized = True
            self._stream_signature = signature
            self._last_submitted_timestamp = timestamp
            if self.config.chunk_lookahead_enabled:
                self._staged_lookahead_source_index = lookahead_source_index
                self._staged_lookahead_timestamp = lookahead_timestamp
                self._staged_lookahead_pose_by_side = (
                    None
                    if lookahead_filtered is None
                    else dict(lookahead_pose_by_side)
                )
                self._staged_lookahead_qpos_by_side = (
                    None
                    if lookahead_filtered is None
                    else dict(lookahead_qpos_by_side)
                )
                self._staged_lookahead_discrete_command = (
                    None
                    if lookahead_discrete_command is None
                    else _copy_teleop_command(lookahead_discrete_command)
                )
                self._staged_lookahead_execution_work_item = (
                    lookahead_execution_work_item
                )
                self._terminal_pose_by_side = (
                    dict(lookahead_pose_by_side)
                    if lookahead_filtered is not None
                    else dict(pose_by_side)
                )
                self._terminal_qpos_by_side = (
                    dict(lookahead_qpos_by_side)
                    if lookahead_filtered is not None
                    else dict(qpos_by_side)
                )
                self._terminal_timestamp = (
                    float(lookahead_timestamp)
                    if lookahead_timestamp is not None
                    else float(logical_timestamp)
                )
            self._submitted_count += 1
            if (
                effective_discrete_command is not None
                and effective_discrete_command.commands
                and not staged_discrete_was_present
            ):
                self._discrete_submitted_count += 1
            if (
                lookahead_discrete_command is not None
                and lookahead_discrete_command.commands
            ):
                self._discrete_submitted_count += 1
            self._append_debug_event_locked(
                "keyframe_submitted",
                {
                    "source_index": int(source_index),
                    "source_timestamp_monotonic": float(timestamp),
                    "generation": int(self._generation),
                    "has_discrete_command": bool(
                        effective_discrete_command is not None
                        and effective_discrete_command.commands
                    ),
                    "lookahead_source_index": lookahead_source_index,
                    "lookahead_timestamp_monotonic": lookahead_timestamp,
                    "same_chunk_lookahead": bool(lookahead_filtered is not None),
                    "current_already_staged": bool(current_already_staged),
                    "queue_depth": int(self._queue.qsize()),
                },
            )
        self._wake.set()
        return source_index

    def clear(self) -> None:
        """Pause output, clear all queued/re-sampling state, and request hold."""

        self._reject_worker_lifecycle_call("clear")
        should_hold = False
        with self._dispatch_lock:
            with self._lock:
                if self._closing or self._closed:
                    return
                should_hold = bool(
                    self.dispatch_callback is not None
                    or self._active
                    or not self._queue.empty()
                    or not self._discrete_queue.empty()
                    or self._hold_retry_required
                )
                invalidated_generation = self._generation
                self._generation += 1
                self._notify_reference_generation_invalidation_locked(
                    legacy_worker_generation=invalidated_generation,
                    reason="stream_clear",
                    terminal_status="canceled",
                )
                self._clear_count += 1
                self._reset_stream_locked()
                self._append_debug_event_locked(
                    "stream_cleared",
                    {
                        "generation": int(self._generation),
                        "hold_requested": bool(should_hold),
                    },
                )
            # The separate hand worker must finish any in-flight callback
            # before the arm hold is issued. Queued callbacks were discarded
            # under the generation change above, so none can trail the hold.
            with self._discrete_dispatch_lock:
                if should_hold:
                    self._invoke_hold()
        self._wake.set()

    def pause(self) -> None:
        self.clear()

    def drain_execution_cancellation(
        self,
        cancellation: ExecutionCancellation,
    ) -> WorkerEpochDrainReceipt:
        """Invalidate all old-epoch reference work without touching hardware.

        This method is available only when the default-off M1C binding was
        explicitly configured.  Taking ``_dispatch_lock`` waits for any
        callback that already crossed the final reference-worker check; the
        generation change then prevents queue work already copied by the
        worker from becoming dispatchable.
        """

        if not isinstance(cancellation, ExecutionCancellation):
            raise TypeError("cancellation must be an ExecutionCancellation")
        guard = self._execution_epoch_guard
        participant = self._worker_drain_participant
        if guard is None or participant is None:
            raise RuntimeError("position deploy execution drain is not configured")
        snapshot = guard.registry.snapshot()
        if (
            cancellation.control_domain_id != snapshot.control_domain_id
            or cancellation.execution_epoch != snapshot.execution_epoch
        ):
            raise RuntimeError(
                "position deploy drain requires the cancellation epoch to be current"
            )

        discarded: set[str] = set()
        with self._dispatch_lock:
            with self._lock:
                if self._inflight_execution_work_item is not None:
                    raise RuntimeError(
                        "reference callback remained in flight while holding dispatch lock"
                    )
                with self._queue.mutex:
                    queued = tuple(self._queue.queue)
                for submission in queued:
                    for item in (
                        submission.execution_work_item,
                        submission.lookahead_execution_work_item,
                    ):
                        if (
                            item is not None
                            and item.execution_epoch
                            == cancellation.previous_execution_epoch
                        ):
                            discarded.add(item.work_item_id)
                for template in self._templates.values():
                    item = template.execution_work_item
                    if (
                        item is not None
                        and item.execution_epoch
                        == cancellation.previous_execution_epoch
                    ):
                        discarded.add(item.work_item_id)
                invalidated_generation = self._generation
                self._generation += 1
                self._notify_reference_generation_invalidation_locked(
                    legacy_worker_generation=invalidated_generation,
                    reason=f"execution_cancellation:{cancellation.reason}",
                    terminal_status="canceled",
                )
                self._reset_stream_locked()
                completed_item = self._last_completed_execution_work_item
                started_at = (
                    self._last_old_epoch_invocation_started_at
                    if completed_item is not None
                    and completed_item.execution_epoch
                    == cancellation.previous_execution_epoch
                    else None
                )
                completed_at = (
                    self._last_old_epoch_invocation_completed_at
                    if started_at is not None
                    else None
                )
        acknowledged_at = max(float(self._clock()), cancellation.effective_at)
        return WorkerEpochDrainReceipt.evaluate(
            cancellation=cancellation,
            participant=participant,
            observed_execution_epoch=snapshot.execution_epoch,
            discarded_work_ids=tuple(sorted(discarded)),
            last_old_epoch_invocation_started_at=started_at,
            last_old_epoch_invocation_completed_at=completed_at,
            in_flight_old_epoch_count=0,
            no_future_old_epoch_calls=True,
            acknowledged_at=acknowledged_at,
        )

    def latch_safety_fault(
        self,
        faults: Any,
        *,
        observation: DeploymentObservation | None = None,
        stage: str = "safety_gate",
    ) -> None:
        """Latch a keyframe-safety failure and synchronously clear/hold output."""

        self._reject_worker_lifecycle_call("latch_safety_fault")
        if isinstance(faults, str):
            fault_text = faults
        else:
            fault_text = ", ".join(str(fault) for fault in faults)
        if not fault_text:
            fault_text = "unspecified safety fault"
        metadata: dict[str, Any] = {
            "faults": [item for item in fault_text.split(", ") if item],
            "manual_rearm_required": True,
        }
        if observation is not None:
            metadata["observation_timestamp"] = float(observation.timestamp)
            metadata["observation_step_index"] = int(observation.step_index)
        self._latch_fault(
            RuntimeError(f"safety gate rejected keyframe: {fault_text}"),
            stage=str(stage),
            metadata=metadata,
        )

    def manual_rearm(self, observation: DeploymentObservation) -> Mapping[str, Any]:
        """Clear a safety-gate fault after fresh, in-domain measured feedback.

        This deliberately does not re-arm background dispatch faults.  Such
        failures need the operator to restart the runtime/home path.  A safety
        fault can be re-armed only from a new hardware measurement that is
        fresh, finite, within the soft Cartesian workspace, and within the
        hard qpos envelope.
        """

        self._reject_worker_lifecycle_call("manual_rearm")
        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation must be a DeploymentObservation")
        details = self._validate_observation_reference_domain(
            observation,
            pose_sides=tuple(self.canonical_pipeline.active_sides),
            qpos_sides=tuple(self.canonical_pipeline.active_sides),
            require_fresh_measurement=True,
        )
        with self._dispatch_lock:
            with self._lock:
                if self._fault is None:
                    return {"rearmed": False, "reason": "not_faulted", **details}
                if self._fault_stage not in {
                    "safety_gate",
                    "15hz_keyframe_safety",
                    "reference_anchor_safety",
                }:
                    raise PositionDeployExecutorFault(
                        "manual re-arm is permitted only for a latched safety-gate fault; "
                        "restart and re-home after other executor faults"
                    )
                self._fault = None
                self._fault_stage = None
                self._fault_metadata = None
                self._fault_pending = False
                self._fault_pending_stage = None
                self._fault_pending_error = None
                invalidated_generation = self._generation
                self._generation += 1
                self._notify_reference_generation_invalidation_locked(
                    legacy_worker_generation=invalidated_generation,
                    reason="manual_safety_rearm",
                    terminal_status="canceled",
                )
                self._reset_stream_locked()
                self._append_debug_event_locked(
                    "manual_safety_rearmed",
                    dict(details),
                )
        self._wake.set()
        return {"rearmed": True, **details}

    def raise_if_fault(self) -> None:
        with self._lock:
            self._raise_if_fault_locked()

    def _validate_observation_reference_domain(
        self,
        observation: DeploymentObservation,
        *,
        pose_sides: tuple[str, ...],
        qpos_sides: tuple[str, ...],
        require_fresh_measurement: bool,
    ) -> dict[str, Any]:
        """Validate anchors/re-arm state without synthesising an unsafe hold."""

        state = np.asarray(observation.state, dtype=float).reshape(-1)
        valid = np.asarray(observation.state_valid, dtype=bool).reshape(-1)
        safety = self.canonical_pipeline.safety_filter.config
        lower = safety.arm_joint_lower_rad
        upper = safety.arm_joint_upper_rad
        details: dict[str, Any] = {
            "active_sides": list(self.canonical_pipeline.active_sides),
            "workspace_soft_margin_m": float(safety.workspace_soft_margin_m),
            "measurement_max_age_sec": float(self.config.safety_rearm_measurement_max_age_sec),
            "sides": {},
        }
        now_wall = float(self._wall_clock())
        for side in self.canonical_pipeline.active_sides:
            side_details: dict[str, Any] = {}
            arm_state = dict(observation.arm_states.get(side) or {})
            if side in pose_sides:
                pose_slice = STATE_ARM_POSE_SLICES[side]
                pose = vector_or_none(state[pose_slice], length=7)
                if pose is None or not bool(np.all(valid[pose_slice])):
                    raise PositionDeployExecutorFault(
                        f"{side} measured pose is unavailable for safety reference/re-arm"
                    )
                distance = workspace_distance_l2(
                    pose[:3], safety.workspace_min, safety.workspace_max
                )
                side_details["workspace_distance_m"] = float(distance)
                if distance > float(safety.workspace_soft_margin_m) + _TIME_EPSILON_SEC:
                    raise PositionDeployExecutorFault(
                        f"{side} measured pose is outside the allowed {safety.workspace_soft_margin_m:g}m "
                        "workspace soft margin; re-home is required"
                    )
            if side in qpos_sides:
                qpos_slice = STATE_ARM_QPOS_SLICES[side]
                qpos = vector_or_none(state[qpos_slice], length=6)
                if require_fresh_measurement:
                    qpos = vector_or_none(
                        arm_state.get(
                            "actual_hardware_qpos",
                            arm_state.get("hardware_qpos", arm_state.get("measured_qpos")),
                        ),
                        length=6,
                    )
                if qpos is None or (
                    not require_fresh_measurement and not bool(np.all(valid[qpos_slice]))
                ):
                    raise PositionDeployExecutorFault(
                        f"{side} measured qpos is unavailable for safety reference/re-arm"
                    )
                constrained = constrain_joint_vector(
                    qpos,
                    reference=None,
                    lower=lower,
                    upper=upper,
                    max_step_rad=0.0,
                )
                if constrained.value is None or not np.allclose(
                    np.asarray(constrained.value, dtype=float), qpos, atol=1e-10, rtol=0.0
                ):
                    raise PositionDeployExecutorFault(
                        f"{side} measured qpos is outside hard limits; re-home is required"
                    )
            if require_fresh_measurement and self.dispatch_callback is not None:
                measurement_timestamp = _measurement_timestamp_from_arm_state(arm_state)
                measurement_valid = bool(
                    arm_state.get(
                        "actual_state_valid",
                        arm_state.get("hardware_measurement_valid", False),
                    )
                )
                read_success = bool(
                    arm_state.get(
                        "actual_state_read_success",
                        arm_state.get("hardware_read_success", False),
                    )
                )
                if measurement_timestamp is None or not measurement_valid or not read_success:
                    raise PositionDeployExecutorFault(
                        f"{side} has no fresh successful hardware measurement for manual re-arm; "
                        "re-home is required"
                    )
                age = now_wall - measurement_timestamp
                side_details["measurement_age_sec"] = float(age)
                if age < -_TIME_EPSILON_SEC or age > float(
                    self.config.safety_rearm_measurement_max_age_sec
                ):
                    raise PositionDeployExecutorFault(
                        f"{side} hardware measurement age {age:.3f}s exceeds manual re-arm limit "
                        f"{self.config.safety_rearm_measurement_max_age_sec:.3f}s; re-home is required"
                    )
            details["sides"][side] = side_details
        return details

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            fault = self._fault
            return {
                "started": bool(self._started),
                "active": bool(self._active),
                "closing": bool(self._closing),
                "closed": bool(self._closed),
                "dry_run": bool(self.dry_run),
                "queue_depth": int(self._queue.qsize()),
                "queue_capacity": int(self.config.queue_size),
                "submitted_count": int(self._submitted_count),
                "reference_count": int(self._reference_count),
                "arm_dispatch_attempt_count": int(self._arm_dispatch_attempt_count),
                "arm_dispatch_failed_count": int(self._arm_dispatch_failed_count),
                "dispatch_count": int(self._dispatch_count),
                "dry_run_count": int(self._dry_run_count),
                "discrete_submitted_count": int(self._discrete_submitted_count),
                "discrete_reference_count": int(self._discrete_reference_count),
                "discrete_dispatch_attempt_count": int(
                    self._discrete_dispatch_attempt_count
                ),
                "discrete_dispatch_count": int(self._discrete_dispatch_count),
                "discrete_dispatch_failed_count": int(
                    self._discrete_dispatch_failed_count
                ),
                "discrete_backend_skipped_count": int(
                    self._discrete_backend_skipped_count
                ),
                "discrete_canceled_count": int(self._discrete_canceled_count),
                "discrete_dry_run_count": int(self._discrete_dry_run_count),
                "discrete_skipped_count": int(self._discrete_skipped_count),
                "discrete_worker_coalesced_count": int(
                    self._discrete_worker_coalesced_count
                ),
                "discrete_queue_depth": int(self._discrete_queue.qsize()),
                "discrete_queue_capacity": int(self._discrete_queue.maxsize),
                "queue_full_count": int(self._queue_full_count),
                "clear_count": int(self._clear_count),
                "hold_count": int(self._hold_count),
                "hold_failure_count": int(self._hold_failure_count),
                "last_hold_error": self._last_hold_error,
                "hold_retry_required": bool(self._hold_retry_required),
                "missed_reference_ticks": int(self._missed_reference_ticks),
                "last_submitted_timestamp": self._last_submitted_timestamp,
                "last_source_timestamp": self._last_source_timestamp,
                "fault_latched": fault is not None,
                "fault_pending": bool(self._fault_pending),
                "fault_pending_stage": self._fault_pending_stage,
                "fault_pending_error": self._fault_pending_error,
                "fault_stage": self._fault_stage,
                "fault_type": None if fault is None else type(fault).__name__,
                "fault": None if fault is None else str(fault),
                "fault_metadata": (
                    None if self._fault_metadata is None else dict(self._fault_metadata)
                ),
                "last_reference": (
                    None
                    if self._last_reference_metadata is None
                    else dict(self._last_reference_metadata)
                ),
                "last_discrete": (
                    None
                    if self._last_discrete_metadata is None
                    else dict(self._last_discrete_metadata)
                ),
                "debug_event_queue_depth": int(len(self._debug_events)),
                "debug_event_drop_count": int(self._debug_event_drop_count),
                "debug_events_enabled": bool(self.config.debug_events_enabled),
                "reference_observer_attached": self._reference_observer is not None,
                "reference_observer_error_count": int(
                    self._reference_observer_error_count
                ),
                "last_reference_observer_error": self._last_reference_observer_error,
                "execution_epoch_binding_enabled": (
                    self._execution_epoch_guard is not None
                ),
                "execution_epoch_rejected_count": int(
                    self._execution_epoch_rejected_count
                ),
                "inflight_execution_work_item_id": (
                    None
                    if self._inflight_execution_work_item is None
                    else self._inflight_execution_work_item.work_item_id
                ),
                "collision_required": bool(self.config.collision_required),
                "collision_reset_measurement_max_age_sec": float(
                    self.config.collision_reset_measurement_max_age_sec
                ),
                "collision_reset_pair_max_skew_sec": float(
                    self.config.collision_reset_pair_max_skew_sec
                ),
                "collision_reset_dry_run_fallback_enabled": bool(
                    self.config.allow_non_authoritative_collision_reset_in_dry_run
                    and self.dispatch_callback is None
                ),
                "collision_planner_prepared": self._collision_planner is not None,
                "collision_planner_initialized": bool(
                    self._collision_planner_initialized
                ),
                "collision_reset_count": int(self._collision_reset_count),
                "collision_tracking_count": int(self._collision_tracking_count),
                "collision_limited_count": int(self._collision_limited_count),
                "collision_backoff_count": int(self._collision_backoff_count),
                "collision_hold_count": int(self._collision_hold_count),
                "last_collision": (
                    None
                    if self._last_collision_metadata is None
                    else dict(self._last_collision_metadata)
                ),
                "last_collision_reset": (
                    None
                    if self._last_collision_reset_metadata is None
                    else dict(self._last_collision_reset_metadata)
                ),
                "reference_worker_alive": bool(
                    self._thread is not None and self._thread.is_alive()
                ),
                "discrete_worker_alive": bool(
                    self._discrete_thread is not None
                    and self._discrete_thread.is_alive()
                ),
                "sample_hz": float(self.config.sample_hz),
                "reference_hz": float(self.config.reference_hz),
                "command_hz": float(self.config.command_hz),
                "causal_delay_periods": (
                    0 if self.config.chunk_lookahead_enabled else 1
                ),
                "chunk_lookahead_enabled": bool(
                    self.config.chunk_lookahead_enabled
                ),
                "no_temporal_smoothing": True,
                "endpoint_collision_certified": bool(
                    self._last_collision_metadata is not None
                    and self._last_collision_metadata.get(
                        "position_endpoint_collision_passed",
                        False,
                    )
                ),
                "segment_collision_certified": False,
            }

    def drain_debug_events(self) -> list[dict[str, Any]]:
        """Drain bounded observational events without touching control state."""

        with self._lock:
            events = [dict(event) for event in self._debug_events]
            self._debug_events.clear()
            return events

    def close(self) -> None:
        self._reject_worker_lifecycle_call("close")
        wait_for_existing_close = False
        already_closed = False
        completed_close_error: BaseException | None = None
        with self._lock:
            if self._closing:
                wait_for_existing_close = True
            elif self._closed:
                already_closed = True
                completed_close_error = self._close_error
            else:
                # Reject every later submit before waiting for either dispatch
                # barrier. A submit already completed before this assignment is
                # part of the stream that close clears below.
                self._closing = True
        if already_closed:
            if completed_close_error is not None:
                raise completed_close_error
            return
        if wait_for_existing_close:
            timeout = float(self.config.shutdown_timeout_sec)
            if not self._closed_event.wait(timeout=timeout):
                raise RuntimeError(
                    "concurrent position deploy close did not finish within "
                    f"{timeout:.3f}s"
                )
            with self._lock:
                completed_close_error = self._close_error
            if completed_close_error is not None:
                raise completed_close_error
            return

        close_errors: list[BaseException] = []
        should_hold = False
        try:
            try:
                with self._dispatch_lock:
                    with self._lock:
                        should_hold = bool(
                            self._active
                            or not self._queue.empty()
                            or not self._discrete_queue.empty()
                            or self._hold_retry_required
                        )
                        invalidated_generation = self._generation
                        self._generation += 1
                        self._notify_reference_generation_invalidation_locked(
                            legacy_worker_generation=invalidated_generation,
                            reason="executor_close",
                            terminal_status="canceled",
                        )
                        self._reset_stream_locked()
                        self._append_debug_event_locked(
                            "executor_closing",
                            {
                                "generation": int(self._generation),
                                "hold_requested": bool(should_hold),
                            },
                        )
                    with self._discrete_dispatch_lock:
                        if should_hold:
                            try:
                                self._invoke_hold()
                            except BaseException as exc:  # still stop both workers.
                                close_errors.append(exc)
            except BaseException as exc:
                close_errors.append(exc)
        finally:
            with self._lock:
                self._closed = True
                self._active = False
                self._stop.set()
                thread = self._thread
                discrete_thread = self._discrete_thread
            self._wake.set()
            timeout = float(self.config.shutdown_timeout_sec)
            for worker in (thread, discrete_thread):
                if worker is not None and worker is not threading.current_thread():
                    try:
                        worker.join(timeout=timeout)
                        if worker.is_alive():
                            close_errors.append(
                                RuntimeError(
                                    f"position deploy worker {worker.name!r} did not stop "
                                    f"within {timeout:.3f}s"
                                )
                            )
                    except BaseException as exc:
                        close_errors.append(exc)
            if close_errors:
                details = "; ".join(
                    f"{type(error).__name__}: {error}" for error in close_errors
                )
                completed_close_error = RuntimeError(
                    f"position deploy executor close failed: {details}"
                )
                completed_close_error.__cause__ = close_errors[0]
                completed_close_error.__suppress_context__ = True
            with self._lock:
                # Publish the completed close result before waking concurrent
                # waiters. Reusing the exception object keeps every caller's
                # view of a failed close identical, including its root cause.
                self._close_error = completed_close_error
            self._closed_event.set()
        if completed_close_error is not None:
            raise completed_close_error

    def __enter__(self) -> PositionDeployExecutor:
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _target_maps(
        self,
        filtered: FilteredNormalizedAction,
        observation: DeploymentObservation,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        vector = np.asarray(filtered.action.vector, dtype=np.float32).reshape(-1)
        if vector.shape != (ACTION_DIM,):
            raise ValueError(f"filtered action vector must have shape {(ACTION_DIM,)}")
        if not observation.arms_enabled:
            raise ValueError("observation must enable arms")
        active_sides = tuple(self.canonical_pipeline.active_sides)
        if not active_sides:
            raise ValueError("canonical_pipeline has no active arm side")
        missing_sides = sorted(set(active_sides) - set(observation.active_sides))
        if missing_sides:
            raise ValueError(
                "observation is missing canonical pipeline sides: "
                + ", ".join(missing_sides)
            )

        pose_by_side: dict[str, np.ndarray] = {}
        qpos_by_side: dict[str, np.ndarray] = {}
        for side in active_sides:
            if side in filtered.action.arm_joint_targets:
                qpos = vector_or_none(filtered.action.arm_joint_targets[side], length=6)
                if qpos is None:
                    raise ValueError(f"{side} safe qpos keyframe is invalid")
                qpos_by_side[side] = qpos.astype(float, copy=True)
                continue
            xyzquat = vector_or_none(vector[ACTION_ARM_POSE_SLICES[side]], length=7)
            pose = None if xyzquat is None else xyzquat_to_pose_matrix(xyzquat)
            if pose is None:
                raise ValueError(f"{side} safe pose keyframe is invalid")
            pose_by_side[side] = np.asarray(pose, dtype=float).copy()
        return pose_by_side, qpos_by_side

    @staticmethod
    def _anchor_maps(
        observation: DeploymentObservation,
        *,
        pose_sides: tuple[str, ...],
        qpos_sides: tuple[str, ...],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        state = np.asarray(observation.state, dtype=np.float32).reshape(-1)
        anchor_pose: dict[str, np.ndarray] = {}
        anchor_qpos: dict[str, np.ndarray] = {}
        for side in pose_sides:
            xyzquat = vector_or_none(state[STATE_ARM_POSE_SLICES[side]], length=7)
            pose = None if xyzquat is None else xyzquat_to_pose_matrix(xyzquat)
            if pose is None:
                raise ValueError(f"{side} current pose is unavailable for the first causal anchor")
            anchor_pose[side] = np.asarray(pose, dtype=float).copy()
        for side in qpos_sides:
            qpos = vector_or_none(state[STATE_ARM_QPOS_SLICES[side]], length=6)
            if qpos is None:
                raise ValueError(f"{side} current qpos is unavailable for the first causal anchor")
            anchor_qpos[side] = qpos.astype(float, copy=True)
        return anchor_pose, anchor_qpos

    def _collision_reset_qpos(
        self,
        observation: DeploymentObservation,
        *,
        now_monotonic: float,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Select one complete, fresh dual-arm collision reset state.

        Any executor with a real dispatch callback is strict regardless of the
        dry-run compatibility option: cached ``current_qpos`` and observation
        vectors are never accepted as authoritative hardware feedback.
        """

        strict_authoritative = self.dispatch_callback is not None or not bool(
            self.config.allow_non_authoritative_collision_reset_in_dry_run
        )
        try:
            return self._authoritative_collision_reset_qpos(
                observation,
                now_monotonic=now_monotonic,
            )
        except ValueError as exc:
            if strict_authoritative:
                raise
            fallback_reason = str(exc)

        state = np.asarray(observation.state, dtype=np.float32).reshape(-1)
        state_valid = np.asarray(observation.state_valid, dtype=bool).reshape(-1)
        qpos_by_side: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            q_slice = STATE_ARM_QPOS_SLICES[side]
            if not bool(np.all(state_valid[q_slice])):
                raise ValueError(
                    f"{side} dry-run qpos validity is incomplete for collision planner reset"
                )
            qpos = vector_or_none(state[q_slice], length=6)
            if qpos is None:
                raise ValueError(
                    f"{side} dry-run qpos is unavailable for collision planner reset"
                )
            qpos_by_side[side] = qpos.astype(float, copy=True)
        return qpos_by_side, {
            "authoritative": False,
            "source": "dry_run_observation_state_vector",
            "fallback_reason": fallback_reason,
            "real_dispatch_fallback_allowed": False,
        }

    def _authoritative_collision_reset_qpos(
        self,
        observation: DeploymentObservation,
        *,
        now_monotonic: float,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        now = _finite_time(now_monotonic, name="collision reset monotonic time")
        max_age = float(self.config.collision_reset_measurement_max_age_sec)
        qpos_by_side: dict[str, np.ndarray] = {}
        side_metadata: dict[str, dict[str, Any]] = {}
        measurement_times: list[float] = []
        for side in ("left", "right"):
            arm_state = dict(observation.arm_states.get(side) or {})
            if not arm_state:
                raise ValueError(
                    f"{side} authoritative arm state is missing for collision planner reset"
                )
            state_error = arm_state.get("state_error")
            if state_error:
                raise ValueError(
                    f"{side} authoritative arm state has state_error: {state_error}"
                )
            if arm_state.get("hardware_read_success") is not True:
                raise ValueError(
                    f"{side} latest hardware read was not successful for collision planner reset"
                )
            if arm_state.get("hardware_measurement_valid") is not True:
                raise ValueError(
                    f"{side} latest hardware measurement is not valid for collision planner reset"
                )
            if arm_state.get("hardware_ready") is not True:
                raise ValueError(
                    f"{side} hardware backend is not ready for collision planner reset"
                )
            if bool(arm_state.get("emergency_stopped", False)):
                raise ValueError(
                    f"{side} emergency stop is latched for collision planner reset"
                )
            for error_key in ("hardware_arm_err", "hardware_sys_err"):
                error_value = arm_state.get(error_key)
                if error_value not in (None, 0):
                    raise ValueError(
                        f"{side} {error_key}={error_value} blocks collision planner reset"
                    )
            hardware_read_sequence = arm_state.get("hardware_read_sequence")
            measured_sequence = arm_state.get("measured_sequence")
            try:
                hardware_read_sequence = int(hardware_read_sequence)
                measured_sequence = int(measured_sequence)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{side} authoritative hardware/measured sequence is unavailable"
                ) from exc
            if hardware_read_sequence <= 0 or measured_sequence <= 0:
                raise ValueError(
                    f"{side} authoritative hardware/measured sequence must be positive"
                )

            read_time = _finite_time(
                arm_state.get("hardware_read_timestamp_monotonic"),
                name=f"{side} hardware_read_timestamp_monotonic",
            )
            measured_time = _finite_time(
                arm_state.get("measured_timestamp_monotonic"),
                name=f"{side} measured_timestamp_monotonic",
            )
            if not math.isclose(read_time, measured_time, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"{side} latest read timestamp does not match the measured qpos timestamp"
                )
            age_sec = now - measured_time
            if age_sec < -1e-3:
                raise ValueError(
                    f"{side} authoritative qpos timestamp is in the future by {-age_sec:.6f}s"
                )
            if age_sec > max_age + _TIME_EPSILON_SEC:
                raise ValueError(
                    f"{side} authoritative qpos is stale for collision planner reset: "
                    f"age={age_sec:.6f}s limit={max_age:.6f}s"
                )
            hardware_qpos = vector_or_none(arm_state.get("hardware_qpos"), length=6)
            measured_qpos = vector_or_none(arm_state.get("measured_qpos"), length=6)
            if hardware_qpos is None or measured_qpos is None:
                raise ValueError(
                    f"{side} authoritative hardware/measured qpos is unavailable"
                )
            if not np.allclose(
                hardware_qpos,
                measured_qpos,
                rtol=0.0,
                atol=1e-7,
            ):
                raise ValueError(
                    f"{side} hardware_qpos does not match the successful measured_qpos"
                )
            qpos_by_side[side] = hardware_qpos.astype(float, copy=True)
            measurement_times.append(measured_time)
            side_metadata[side] = {
                "source": "successful_hardware_measurement",
                "hardware_read_sequence": hardware_read_sequence,
                "measured_sequence": measured_sequence,
                "measurement_timestamp_monotonic": measured_time,
                "measurement_age_sec": float(max(0.0, age_sec)),
                "hardware_result_code": arm_state.get("hardware_result_code"),
                "hardware_arm_err": arm_state.get("hardware_arm_err"),
                "hardware_sys_err": arm_state.get("hardware_sys_err"),
            }

        pair_skew_sec = max(measurement_times) - min(measurement_times)
        if pair_skew_sec > self.config.collision_reset_pair_max_skew_sec:
            raise ValueError(
                "dual-arm authoritative qpos measurements are too far apart for "
                "collision planner reset: "
                f"skew={pair_skew_sec:.6f}s "
                f"limit={self.config.collision_reset_pair_max_skew_sec:.6f}s"
            )
        return qpos_by_side, {
            "authoritative": True,
            "source": "successful_fresh_dual_hardware_measurement",
            "measurement_max_age_sec": max_age,
            "pair_skew_sec": float(pair_skew_sec),
            "pair_max_skew_sec": float(
                self.config.collision_reset_pair_max_skew_sec
            ),
            "by_side": side_metadata,
            "real_dispatch_fallback_allowed": False,
        }

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._drain_submissions()
                with self._lock:
                    active = bool(
                        self._active
                        and self._fault is None
                        and not self._fault_pending
                        and not self._closing
                    )
                    next_tick = self._next_tick_timestamp
                    last_source = self._last_source_timestamp
                if not active or next_tick is None or last_source is None:
                    self._wake.wait(timeout=0.05)
                    self._wake.clear()
                    continue

                now = _finite_time(self._clock(), name="clock")
                if now - last_source > self.config.stale_after_sec:
                    raise TimeoutError(
                        f"{self.config.sample_hz:g} Hz position keyframe stream is stale: "
                        f"age={now - last_source:.6f}s limit={self.config.stale_after_sec:.6f}s"
                    )
                if now + _TIME_EPSILON_SEC < next_tick:
                    self._wake.wait(timeout=min(next_tick - now, 0.05))
                    self._wake.clear()
                    continue

                self._emit_reference(scheduled_timestamp=next_tick)
                after = _finite_time(self._clock(), name="clock")
                with self._lock:
                    if not self._active:
                        continue
                    following = next_tick + self.config.reference_period_sec
                    if after > following + _TIME_EPSILON_SEC:
                        skipped = (
                            int(
                                math.floor(
                                    (after - following) / self.config.reference_period_sec
                                )
                            )
                            + 1
                        )
                        following += skipped * self.config.reference_period_sec
                        self._missed_reference_ticks += skipped
                    self._next_tick_timestamp = following
        except Exception as exc:  # noqa: BLE001 - a worker failure must be latched.
            self._mark_fault_pending(exc, stage="reference_worker")
            self._latch_fault(
                exc,
                stage="reference_worker",
                metadata={"worker": "reference"},
            )

    def _run_discrete_dispatch(self) -> None:
        try:
            self._run_discrete_dispatch_impl()
        except Exception as exc:  # noqa: BLE001 - every worker exit must fault-latch.
            self._mark_fault_pending(exc, stage="discrete_dispatch_worker_internal")
            self._latch_fault(
                exc,
                stage="discrete_dispatch_worker_internal",
                metadata={"worker": "discrete", "internal_exception": True},
            )

    def _run_discrete_dispatch_impl(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._discrete_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            dispatch_error: BaseException | None = None
            failure_metadata: dict[str, Any] | None = None
            try:
                with self._lock:
                    if (
                        item.generation != self._generation
                        or self._closing
                        or self._closed
                        or self._fault is not None
                        or self._fault_pending
                    ):
                        self._discrete_canceled_count += 1
                        canceled = dict(item.command.metadata)
                        canceled["position_discrete_dispatch_status"] = "canceled"
                        self._last_discrete_metadata = canceled
                        self._append_debug_event_locked(
                            "discrete_dispatch",
                            canceled,
                        )
                        continue

                with self._discrete_dispatch_lock:
                    with self._lock:
                        if (
                            item.generation != self._generation
                            or self._closing
                            or self._closed
                            or self._fault is not None
                            or self._fault_pending
                        ):
                            self._discrete_canceled_count += 1
                            canceled = dict(item.command.metadata)
                            canceled["position_discrete_dispatch_status"] = "canceled"
                            self._last_discrete_metadata = canceled
                            self._append_debug_event_locked(
                                "discrete_dispatch",
                                canceled,
                            )
                            continue
                        self._discrete_dispatch_attempt_count += 1

                    attempt_monotonic = _finite_time(self._clock(), name="clock")
                    attempt_wall = _finite_time(self._wall_clock(), name="wall_clock")
                    callback_result: object = None
                    try:
                        assert self.discrete_dispatch_callback is not None
                        callback_result = self.discrete_dispatch_callback(item.command)
                        accepted, skipped, receipt = self._classify_discrete_result(
                            callback_result
                        )
                    except Exception as exc:  # fault is latched after releasing this lock.
                        # Close the arm-dispatch gate before doing any logging
                        # or receipt formatting. A reference tick that has not yet
                        # crossed its final pre-dispatch check must stop now.
                        self._mark_fault_pending(
                            exc,
                            stage="discrete_dispatch_worker",
                        )
                        dispatch_error = exc
                        accepted = False
                        skipped = False
                        receipt = (
                            dict(callback_result)
                            if isinstance(callback_result, Mapping)
                            else {}
                        )
                    completed_monotonic = _finite_time(self._clock(), name="clock")
                    completed_wall = _finite_time(self._wall_clock(), name="wall_clock")
                    outcome = {
                        **dict(item.command.metadata),
                        "position_discrete_dispatch_status": (
                            "failed"
                            if dispatch_error is not None
                            else ("backend_skipped" if skipped else "accepted")
                        ),
                        "position_discrete_callback_started_monotonic_time": float(
                            attempt_monotonic
                        ),
                        "position_discrete_callback_started_wall_time": float(attempt_wall),
                        "position_discrete_callback_completed_monotonic_time": float(
                            completed_monotonic
                        ),
                        "position_discrete_callback_completed_wall_time": float(
                            completed_wall
                        ),
                        "position_discrete_callback_duration_ms": float(
                            max(0.0, completed_monotonic - attempt_monotonic) * 1000.0
                        ),
                        "position_discrete_queue_lag_ms": float(
                            max(0.0, attempt_monotonic - item.logical_emit_monotonic)
                            * 1000.0
                        ),
                        "position_discrete_actual_causal_lag_sec": float(
                            completed_monotonic - item.source_timestamp
                        ),
                        "position_discrete_backend_receipt": dict(receipt),
                        "position_discrete_callback_accepted": bool(accepted),
                        "position_discrete_callback_skipped": bool(skipped),
                        "position_discrete_commands": self._command_debug_payload(
                            item.command
                        ),
                    }
                    if dispatch_error is not None:
                        outcome.update(
                            {
                                "position_discrete_dispatch_error_type": type(
                                    dispatch_error
                                ).__name__,
                                "position_discrete_dispatch_error": str(dispatch_error),
                            }
                        )
                    item.command.metadata.update(outcome)
                    for effector_command in item.command.commands.values():
                        effector_command.metadata.update(outcome)
                    with self._lock:
                        if dispatch_error is not None:
                            self._discrete_dispatch_failed_count += 1
                            self._fault_pending = True
                            self._fault_pending_stage = "discrete_dispatch_worker"
                        elif skipped:
                            self._discrete_backend_skipped_count += 1
                        elif accepted:
                            self._discrete_dispatch_count += 1
                        self._last_discrete_metadata = dict(outcome)
                        self._append_debug_event_locked(
                            "discrete_dispatch",
                            outcome,
                        )
                        failure_metadata = dict(outcome)
            finally:
                self._discrete_queue.task_done()

            if dispatch_error is not None:
                self._latch_fault(
                    dispatch_error,
                    stage="discrete_dispatch_worker",
                    metadata=failure_metadata,
                )
                return

    @staticmethod
    def _classify_discrete_result(
        result: object,
    ) -> tuple[bool, bool, dict[str, Any]]:
        if result is False:
            raise RuntimeError("discrete dispatch callback rejected the hand keyframe")
        if not isinstance(result, Mapping):
            return True, False, {}
        receipt = dict(result)
        if bool(receipt.get("fatal", False)):
            raise RuntimeError(
                str(receipt.get("error") or "discrete dispatch callback reported a fatal error")
            )
        accepted = bool(receipt.get("accepted", True))
        skipped = bool(receipt.get("skipped", False))
        if not accepted and not skipped:
            raise RuntimeError(
                str(receipt.get("error") or "discrete dispatch callback did not accept the keyframe")
            )
        return accepted, skipped, receipt

    def _validate_collision_arm_dispatch_result(
        self,
        result: object,
        *,
        expected_joint_qpos_by_effector: Mapping[str, np.ndarray],
    ) -> dict[str, Any]:
        """Verify that both collision-planned qpos targets survived dispatch."""

        if not isinstance(result, Mapping):
            raise RuntimeError(
                "collision-required arm dispatch must return structured backend receipts"
            )
        batch = dict(result)
        if not bool(batch.get("accepted", False)):
            raise RuntimeError("collision-required arm dispatch batch was not accepted")
        raw_effectors = batch.get("effectors")
        if not isinstance(raw_effectors, Mapping):
            raise RuntimeError(
                "collision-required arm dispatch receipt is missing per-effector evidence"
            )
        expected_names = {f"{side}_arm" for side in self.canonical_pipeline.active_sides}
        if set(expected_joint_qpos_by_effector) != expected_names:
            raise RuntimeError(
                "collision-required arm dispatch did not produce complete dual joint_vector targets"
            )

        summary_by_effector: dict[str, dict[str, Any]] = {}
        for effector_name in sorted(expected_names):
            raw_receipt = raw_effectors.get(effector_name)
            if not isinstance(raw_receipt, Mapping):
                raise RuntimeError(
                    f"collision-required arm dispatch has no receipt for {effector_name}"
                )
            receipt = dict(raw_receipt)
            if not bool(receipt.get("accepted", False)):
                raise RuntimeError(
                    f"collision-required arm dispatch was rejected by {effector_name}"
                )
            if str(receipt.get("command_type") or "") != "joint_vector":
                raise RuntimeError(
                    f"collision-required arm dispatch changed {effector_name} modality"
                )
            if receipt.get("backend_ready") is not True:
                raise RuntimeError(
                    f"collision-required arm backend {effector_name} is not ready"
                )
            if receipt.get("state_error"):
                raise RuntimeError(
                    f"collision-required arm backend {effector_name} state check failed: "
                    f"{receipt.get('state_error')}"
                )

            expected = np.asarray(
                expected_joint_qpos_by_effector[effector_name],
                dtype=float,
            ).reshape(6)
            requested = vector_or_none(receipt.get("requested_value"), length=6)
            final = vector_or_none(receipt.get("final_value"), length=6)
            if requested is None or final is None:
                raise RuntimeError(
                    f"collision-required arm dispatch receipt for {effector_name} "
                    "does not contain requested/final qpos"
                )
            if not np.allclose(requested, expected, rtol=0.0, atol=1e-7):
                raise RuntimeError(
                    f"collision-required arm dispatcher changed {effector_name} before backend send"
                )
            backend_final_raw = receipt.get("backend_final_target_qpos")
            backend_final = (
                None
                if backend_final_raw is None
                else vector_or_none(backend_final_raw, length=6)
            )
            if backend_final_raw is not None and backend_final is None:
                raise RuntimeError(
                    f"collision-required arm dispatch receipt for {effector_name} "
                    "contains an invalid backend_final_target_qpos"
                )
            rewritten = bool(receipt.get("rewritten", False)) or not np.allclose(
                final,
                expected,
                rtol=0.0,
                atol=1e-7,
            )
            if backend_final is not None:
                rewritten = rewritten or not np.allclose(
                    backend_final,
                    expected,
                    rtol=0.0,
                    atol=1e-7,
                )
            singularity_status = str(receipt.get("singularity_gate_status") or "")
            singularity_blocked = any(
                token in singularity_status.lower() for token in ("hold", "reject")
            )
            if rewritten or singularity_blocked:
                raise RuntimeError(
                    "collision-planned joint_vector was not preserved by "
                    f"{effector_name}: rewritten={rewritten}, "
                    f"singularity_gate_status={singularity_status or 'unknown'}"
                )

            background = receipt.get("background_streaming")
            background = dict(background) if isinstance(background, Mapping) else {}
            if background.get("enabled"):
                if (
                    background.get("fault")
                    or background.get("healthy") is False
                    or background.get("thread_alive") is not True
                    or background.get("target_active") is not True
                    or bool(background.get("target_stale", False))
                ):
                    raise RuntimeError(
                        f"collision-required arm stream {effector_name} became unhealthy "
                        "while queuing the high-rate reference"
                    )
            item = {
                "accepted": True,
                "rewritten": False,
                "backend_ready": True,
                "operation": receipt.get("operation"),
                "status": receipt.get("status"),
                "result_code": receipt.get("result_code"),
                "singularity_gate_status": singularity_status or None,
                "background_streaming": background,
            }
            if self.config.debug_events_enabled:
                item["expected_qpos"] = expected.copy()
                item["requested_qpos"] = requested.copy()
                item["final_qpos"] = final.copy()
                item["backend_final_target_qpos"] = (
                    None if backend_final is None else backend_final.copy()
                )
            summary_by_effector[effector_name] = item

        group_health = batch.get("arm_backend_health")
        if group_health is not None:
            if not isinstance(group_health, Mapping) or not bool(
                group_health.get("ready", False)
            ):
                raise RuntimeError(
                    "collision-required dual-arm health check failed after backend queueing"
                )
        return {
            "position_arm_backend_dispatch_verified": True,
            "position_arm_backend_dispatch_by_effector": summary_by_effector,
            "position_arm_backend_group_health": (
                dict(group_health) if isinstance(group_health, Mapping) else None
            ),
        }

    def _drain_submissions(self) -> None:
        while True:
            try:
                submission = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._accept_submission(submission)
            finally:
                self._queue.task_done()

    def _accept_submission(self, submission: _Submission) -> None:
        guard = self._execution_epoch_guard
        if guard is not None:
            item = submission.execution_work_item
            if item is None:
                raise RuntimeError("epoch-mediated submission lost its execution work item")
            check = guard.check_at_dequeue(item)
            if not check.accepted:
                with self._lock:
                    self._execution_epoch_rejected_count += 1
                    self._append_debug_event_locked(
                        "execution_work_rejected",
                        {
                            "work_item_id": item.work_item_id,
                            "checkpoint": check.checkpoint,
                            "status": check.status,
                        },
                    )
                return
        if submission.collision_reset_qpos_by_side is not None:
            self.prepare_collision_planner()
        with self._lock:
            if (
                submission.generation != self._generation
                or self._closing
                or self._closed
            ):
                return
            if submission.collision_reset_qpos_by_side is not None:
                planner = self._collision_planner
                if planner is None:
                    raise RuntimeError("required collision planner was not prepared")
                applied_reset_metadata = self._collision_reset_metadata_at_apply(
                    submission.collision_reset_metadata
                )
                planner_reset_metadata = dict(
                    planner.reset(submission.collision_reset_qpos_by_side)
                )
                reset_metadata = {
                    **applied_reset_metadata,
                    "planner_reset": planner_reset_metadata,
                    "generation": int(submission.generation),
                    "source_index": int(submission.source_index),
                }
                self._collision_planner_initialized = True
                self._collision_reset_count += 1
                self._last_collision_reset_metadata = dict(reset_metadata)
                self._append_debug_event_locked(
                    "collision_planner_reset",
                    reset_metadata,
                )
            template = _ReferenceTemplate(
                filtered=submission.filtered,
                observation=submission.observation,
                policy_name=submission.policy_name,
                source_timestamp=(
                    submission.timestamp
                    if submission.logical_timestamp is None
                    else float(submission.logical_timestamp)
                ),
                discrete_command=submission.discrete_command,
                execution_work_item=submission.execution_work_item,
            )
            logical_timestamp = (
                submission.timestamp
                if submission.logical_timestamp is None
                else float(submission.logical_timestamp)
            )
            if submission.anchor_source_index is not None:
                assert submission.anchor_timestamp is not None
                if submission.pose_by_side:
                    self._store_legacy_reference_source_fact_locked(
                        modality="pose",
                        source_index=submission.anchor_source_index,
                        source_timestamp=submission.anchor_timestamp,
                        source_stage="current_observation_anchor",
                        target_by_side=dict(submission.anchor_pose_by_side or {}),
                    )
                    self._pose_resampler.push(
                        TimestampedMotionTarget(
                            timestamp=submission.anchor_timestamp,
                            source_index=submission.anchor_source_index,
                            pose_by_side=dict(submission.anchor_pose_by_side or {}),
                            metadata={"stage": "current_observation_anchor"},
                            available_timestamp=(
                                submission.timestamp
                                if self.config.chunk_lookahead_enabled
                                else None
                            ),
                        )
                    )
                if submission.qpos_by_side:
                    self._store_legacy_reference_source_fact_locked(
                        modality="qpos",
                        source_index=submission.anchor_source_index,
                        source_timestamp=submission.anchor_timestamp,
                        source_stage="current_observation_anchor",
                        target_by_side=dict(submission.anchor_qpos_by_side or {}),
                    )
                    self._qpos_resampler.push(
                        TimestampedMotionTarget(
                            timestamp=submission.anchor_timestamp,
                            source_index=submission.anchor_source_index,
                            qpos_by_side=dict(submission.anchor_qpos_by_side or {}),
                            metadata={"stage": "current_observation_anchor"},
                            available_timestamp=(
                                submission.timestamp
                                if self.config.chunk_lookahead_enabled
                                else None
                            ),
                        )
                    )
                self._store_template_locked(
                    submission.anchor_source_index,
                    replace(
                        template,
                        source_timestamp=submission.anchor_timestamp,
                        discrete_command=None,
                    ),
                )

            if submission.pose_by_side and not submission.current_already_staged:
                self._store_legacy_reference_source_fact_locked(
                    modality="pose",
                    source_index=submission.source_index,
                    source_timestamp=logical_timestamp,
                    source_stage=(
                        f"safe_{_rate_label(self.config.sample_hz)}hz_keyframe"
                    ),
                    target_by_side=submission.pose_by_side,
                )
                self._pose_resampler.push(
                    TimestampedMotionTarget(
                        timestamp=logical_timestamp,
                        source_index=submission.source_index,
                        pose_by_side=submission.pose_by_side,
                        metadata={
                            "stage": f"safe_{_rate_label(self.config.sample_hz)}hz_keyframe"
                        },
                        available_timestamp=(
                            submission.timestamp
                            if self.config.chunk_lookahead_enabled
                            else None
                        ),
                    )
                )
            if submission.qpos_by_side and not submission.current_already_staged:
                self._store_legacy_reference_source_fact_locked(
                    modality="qpos",
                    source_index=submission.source_index,
                    source_timestamp=logical_timestamp,
                    source_stage=(
                        f"safe_{_rate_label(self.config.sample_hz)}hz_keyframe"
                    ),
                    target_by_side=submission.qpos_by_side,
                )
                self._qpos_resampler.push(
                    TimestampedMotionTarget(
                        timestamp=logical_timestamp,
                        source_index=submission.source_index,
                        qpos_by_side=submission.qpos_by_side,
                        metadata={
                            "stage": f"safe_{_rate_label(self.config.sample_hz)}hz_keyframe"
                        },
                        available_timestamp=(
                            submission.timestamp
                            if self.config.chunk_lookahead_enabled
                            else None
                        ),
                    )
                )
            self._store_template_locked(submission.source_index, template)
            if submission.lookahead_source_index is not None:
                if (
                    submission.lookahead_timestamp is None
                    or submission.lookahead_filtered is None
                    or submission.lookahead_observation is None
                ):
                    raise RuntimeError("incomplete chunk lookahead submission")
                if submission.lookahead_pose_by_side:
                    self._store_legacy_reference_source_fact_locked(
                        modality="pose",
                        source_index=int(submission.lookahead_source_index),
                        source_timestamp=float(submission.lookahead_timestamp),
                        source_stage=(
                            f"safe_{_rate_label(self.config.sample_hz)}hz_"
                            "chunk_lookahead"
                        ),
                        target_by_side=submission.lookahead_pose_by_side,
                    )
                    self._pose_resampler.push(
                        TimestampedMotionTarget(
                            timestamp=float(submission.lookahead_timestamp),
                            source_index=int(submission.lookahead_source_index),
                            pose_by_side=dict(submission.lookahead_pose_by_side),
                            metadata={
                                "stage": (
                                    f"safe_{_rate_label(self.config.sample_hz)}hz_"
                                    "chunk_lookahead"
                                )
                            },
                            available_timestamp=submission.timestamp,
                        )
                    )
                if submission.lookahead_qpos_by_side:
                    self._store_legacy_reference_source_fact_locked(
                        modality="qpos",
                        source_index=int(submission.lookahead_source_index),
                        source_timestamp=float(submission.lookahead_timestamp),
                        source_stage=(
                            f"safe_{_rate_label(self.config.sample_hz)}hz_"
                            "chunk_lookahead"
                        ),
                        target_by_side=submission.lookahead_qpos_by_side,
                    )
                    self._qpos_resampler.push(
                        TimestampedMotionTarget(
                            timestamp=float(submission.lookahead_timestamp),
                            source_index=int(submission.lookahead_source_index),
                            qpos_by_side=dict(submission.lookahead_qpos_by_side),
                            metadata={
                                "stage": (
                                    f"safe_{_rate_label(self.config.sample_hz)}hz_"
                                    "chunk_lookahead"
                                )
                            },
                            available_timestamp=submission.timestamp,
                        )
                    )
                self._store_template_locked(
                    int(submission.lookahead_source_index),
                    _ReferenceTemplate(
                        filtered=submission.lookahead_filtered,
                        observation=submission.lookahead_observation,
                        policy_name=str(
                            submission.lookahead_policy_name
                            or submission.policy_name
                        ),
                        source_timestamp=float(submission.lookahead_timestamp),
                        discrete_command=submission.lookahead_discrete_command,
                        execution_work_item=(
                            submission.lookahead_execution_work_item
                        ),
                    ),
                )
            self._last_source_timestamp = submission.timestamp
            self._active = True
            if self._next_tick_timestamp is None:
                self._next_tick_timestamp = max(logical_timestamp, self._clock())

    def _collision_reset_metadata_at_apply(
        self,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        applied = dict(metadata or {})
        if not bool(applied.get("authoritative", False)):
            return applied
        by_side = applied.get("by_side")
        if not isinstance(by_side, Mapping):
            raise RuntimeError(
                "authoritative collision reset metadata is missing per-side timestamps"
            )
        now = _finite_time(self._clock(), name="collision reset apply clock")
        max_age = float(self.config.collision_reset_measurement_max_age_sec)
        apply_ages: dict[str, float] = {}
        for side in ("left", "right"):
            side_metadata = by_side.get(side)
            if not isinstance(side_metadata, Mapping):
                raise RuntimeError(
                    f"authoritative collision reset metadata is missing {side}"
                )
            measured_time = _finite_time(
                side_metadata.get("measurement_timestamp_monotonic"),
                name=f"{side} collision reset measurement timestamp",
            )
            age_sec = now - measured_time
            if age_sec < -1e-3:
                raise RuntimeError(
                    f"{side} collision reset measurement moved into the future"
                )
            if age_sec > max_age + _TIME_EPSILON_SEC:
                raise RuntimeError(
                    f"{side} authoritative qpos became stale before planner reset: "
                    f"age={age_sec:.6f}s limit={max_age:.6f}s"
                )
            apply_ages[side] = float(max(0.0, age_sec))
        applied["apply_timestamp_monotonic"] = now
        applied["measurement_age_at_apply_sec_by_side"] = apply_ages
        return applied

    def _execution_work_for_reference_locked(
        self,
        *,
        parent_indices: tuple[int, ...],
        reference_sequence: int,
    ) -> ExecutionWorkItem | None:
        if self._execution_epoch_guard is None:
            return None
        parent_items: list[ExecutionWorkItem] = []
        for source_index in parent_indices:
            template = self._templates.get(int(source_index))
            item = None if template is None else template.execution_work_item
            if item is None:
                raise RuntimeError(
                    "epoch-mediated reference has a parent without execution lineage"
                )
            parent_items.append(item)
        first = parent_items[0]
        if any(not _same_execution_scope(first, item) for item in parent_items[1:]):
            raise RuntimeError(
                "reference parents cross authority/execution epoch or group generation"
            )
        parent_ids = tuple(item.work_item_id for item in parent_items)
        payload = {
            "parent_work_item_ids": parent_ids,
            "reference_sequence": int(reference_sequence),
            "control_domain_id": first.control_domain_id,
            "authority_epoch": first.authority_epoch,
            "execution_epoch": first.execution_epoch,
            "segment_generation_by_group": dict(
                first.segment_generation_by_group
            ),
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain="dexteleop.position_reference_execution_work.v1",
        )
        source_decision_ids = {
            item.source_decision_id
            for item in parent_items
            if item.source_decision_id is not None
        }
        return ExecutionWorkItem(
            work_item_id=f"position-reference-work:{semantic_hash}",
            work_kind="position_command",
            control_domain_id=first.control_domain_id,
            authority_epoch=first.authority_epoch,
            execution_epoch=first.execution_epoch,
            segment_generation_by_group=dict(
                first.segment_generation_by_group
            ),
            source_decision_id=(
                next(iter(source_decision_ids))
                if len(source_decision_ids) == 1
                else None
            ),
            source_reference_segment_id=f"position-reference-lineage:{semantic_hash}",
        )

    def _store_template_locked(self, source_index: int, template: _ReferenceTemplate) -> None:
        self._templates[int(source_index)] = template
        self._templates.move_to_end(int(source_index))
        while len(self._templates) > self._max_templates:
            expired_index, _ = self._templates.popitem(last=False)
            self._legacy_reference_source_facts.pop(("pose", expired_index), None)
            self._legacy_reference_source_facts.pop(("qpos", expired_index), None)

    def _store_legacy_reference_source_fact_locked(
        self,
        *,
        modality: str,
        source_index: int,
        source_timestamp: float,
        source_stage: str,
        target_by_side: Mapping[str, np.ndarray],
    ) -> None:
        """Retain the actual resampler source provenance without re-labelling it."""

        if self._reference_observer is None or not target_by_side:
            return
        key = (str(modality), int(source_index))
        # A same-chunk lookahead later becomes the current keyframe without a
        # second resampler push. Preserve the stage under which it actually
        # entered the worker instead of overwriting provenance at queue drain.
        if key in self._legacy_reference_source_facts:
            return
        target_hash = canonical_content_hash(
            {str(side): np.asarray(value) for side, value in target_by_side.items()},
            domain=f"dexteleop.legacy_position_source_target.{modality}.v1",
        )
        self._legacy_reference_source_facts[key] = LegacyReferenceSourceFact(
            source_index=int(source_index),
            source_timestamp=float(source_timestamp),
            source_stage=str(source_stage),
            target_payload_hash=target_hash,
        )

    def _notify_reference_observer(
        self,
        *,
        generation: int,
        scheduled_timestamp: float,
        actual_emit: float,
        pose_ref: ResampledMotionReference | None,
        qpos_ref: ResampledMotionReference | None,
    ) -> tuple[LegacyReferenceDerivation, ...]:
        """Submit pre-resolver facts; observer failures never affect execution."""

        observer = self._reference_observer
        if observer is None:
            return ()
        derivations: list[LegacyReferenceDerivation] = []
        try:
            for modality, reference, interpolation_contract in (
                (
                    "pose",
                    pose_ref,
                    "xyz_linear+quaternion_shortest_path_slerp",
                ),
                ("qpos", qpos_ref, "linear_qpos_interpolation"),
            ):
                if reference is None:
                    continue
                with self._lock:
                    start = self._legacy_reference_source_facts.get(
                        (modality, int(reference.left_source_index))
                    )
                    end = self._legacy_reference_source_facts.get(
                        (modality, int(reference.right_source_index))
                    )
                if start is None or end is None:
                    raise RuntimeError(
                        "missing legacy source provenance for observed reference "
                        f"{modality}:{reference.left_source_index}->"
                        f"{reference.right_source_index}"
                    )
                derivation = LegacyReferenceDerivation(
                    legacy_worker_generation=int(generation),
                    modality=modality,
                    scheduled_for=float(scheduled_timestamp),
                    emitted_at=float(actual_emit),
                    reference=reference,
                    start_source_fact=start,
                    end_source_fact=end,
                    interpolation_contract_hash=canonical_content_hash(
                        {
                            "modality": modality,
                            "algorithm": interpolation_contract,
                            "sample_hz": float(self.config.sample_hz),
                            "reference_hz": float(self.config.reference_hz),
                            "causal_delay_periods": (
                                0 if self.config.chunk_lookahead_enabled else 1
                            ),
                        },
                        domain=(
                            "dexteleop.legacy_position_interpolation_contract.v1"
                        ),
                    ),
                )
                derivations.append(derivation)
                observer.submit_nowait(derivation)
        except Exception as exc:  # noqa: BLE001 - passive diagnostics only.
            with self._lock:
                self._reference_observer_error_count += 1
                self._last_reference_observer_error = (
                    f"{type(exc).__name__}: {exc}"
                )
        return tuple(derivations)

    @staticmethod
    def _legacy_command_intent_hash(command: TeleopCommand) -> str:
        payload = {
            "commands": {
                str(name): {
                    "effector_name": item.effector_name,
                    "side": item.side,
                    "command_type": item.command_type,
                    "frame_name": item.frame_name,
                    "value": np.asarray(item.value),
                }
                for name, item in sorted(command.commands.items())
            }
        }
        return canonical_content_hash(
            payload,
            domain="dexteleop.legacy_position_command_intent.v1",
        )

    def _notify_reference_resolution_observer(
        self,
        *,
        derivations: tuple[LegacyReferenceDerivation, ...],
        actual_emit: float,
        command: TeleopCommand,
        collision_qpos_by_side: Mapping[str, np.ndarray] | None,
        metadata: Mapping[str, Any],
    ) -> None:
        observer = self._reference_observer
        submit = (
            None
            if observer is None
            else getattr(observer, "submit_resolution_nowait", None)
        )
        if not callable(submit) or not derivations:
            return
        try:
            command_hash = self._legacy_command_intent_hash(command)
            resolver_hash = canonical_content_hash(
                {
                    "resolver": "CanonicalDeployActionPipeline.resolve_filtered",
                    "active_sides": tuple(self.canonical_pipeline.active_sides),
                    "command_intent_hash": command_hash,
                },
                domain="dexteleop.legacy_position_resolver_evidence.v1",
            )
            if collision_qpos_by_side is None:
                planner_status = "not_applicable"
                planner_input_hash = None
                planner_output_hash = None
                planner_evidence_hash = None
                measurement_status = "not_applicable"
            else:
                pose_derivation = next(
                    (
                        item
                        for item in derivations
                        if item.modality == "pose"
                    ),
                    None,
                )
                if pose_derivation is None:
                    raise RuntimeError(
                        "collision planner outcome is missing its pose reference"
                    )
                planner_status = "legacy_collision_planner_accepted"
                planner_input_hash = canonical_content_hash(
                    {
                        str(side): np.asarray(value)
                        for side, value in pose_derivation.reference.pose_by_side.items()
                    },
                    domain="dexteleop.legacy_collision_planner_input.v1",
                )
                planner_output_hash = canonical_content_hash(
                    {
                        str(side): np.asarray(value)
                        for side, value in collision_qpos_by_side.items()
                    },
                    domain="dexteleop.legacy_collision_planner_output.v1",
                )
                planner_evidence_hash = canonical_content_hash(
                    {
                        "input_hash": planner_input_hash,
                        "output_hash": planner_output_hash,
                        "endpoint_collision_checked": bool(
                            metadata.get("position_endpoint_collision_checked", False)
                        ),
                        "endpoint_collision_passed": bool(
                            metadata.get("position_endpoint_collision_passed", False)
                        ),
                        "collision_status": metadata.get(
                            "position_collision_status"
                        ),
                    },
                    domain="dexteleop.legacy_collision_planner_evidence.v1",
                )
                measurement_status = "legacy_planner_state_not_typed"
            for derivation in derivations:
                submit(
                    LegacyReferenceResolutionDerivation(
                        reference_derivation=derivation,
                        resolved_at=float(actual_emit),
                        resolver_evidence_hash=resolver_hash,
                        planner_evidence_status=planner_status,
                        planner_input_hash=planner_input_hash,
                        planner_output_hash=planner_output_hash,
                        planner_evidence_hash=planner_evidence_hash,
                        measured_state_evidence_status=measurement_status,
                        measured_state_evidence_ids=(),
                        command_intent_hash=command_hash,
                        local_pre_backend_generation_check_passed=True,
                        local_pre_backend_check_observed_at=float(actual_emit),
                    )
                )
        except Exception as exc:  # noqa: BLE001 - passive diagnostics only.
            with self._lock:
                self._reference_observer_error_count += 1
                self._last_reference_observer_error = (
                    f"{type(exc).__name__}: {exc}"
                )

    def _notify_reference_resolution_failure_observer(
        self,
        *,
        derivations: tuple[LegacyReferenceDerivation, ...],
        actual_emit: float,
        failure_stage: str,
        error: BaseException,
    ) -> None:
        """Observe an existing explicit failure branch without changing it."""

        observer = self._reference_observer
        submit = (
            None
            if observer is None
            else getattr(observer, "submit_resolution_failure_nowait", None)
        )
        if not callable(submit) or not derivations:
            return
        try:
            evidence_hash = canonical_content_hash(
                {
                    "failure_stage": str(failure_stage),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "legacy_worker_generation": tuple(
                        item.legacy_worker_generation for item in derivations
                    ),
                    "source_reference_sequences": tuple(
                        item.reference.sequence for item in derivations
                    ),
                },
                domain="dexteleop.legacy_position_resolution_failure_evidence.v1",
            )
            for derivation in derivations:
                submit(
                    LegacyReferenceResolutionFailureDerivation(
                        reference_derivation=derivation,
                        failed_at=float(actual_emit),
                        failure_stage=str(failure_stage),
                        error_type=type(error).__name__,
                        error_message=str(error),
                        failure_evidence_hash=evidence_hash,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - passive diagnostics only.
            with self._lock:
                self._reference_observer_error_count += 1
                self._last_reference_observer_error = (
                    f"{type(exc).__name__}: {exc}"
                )

    def _notify_reference_generation_invalidation_locked(
        self,
        *,
        legacy_worker_generation: int,
        reason: str,
        terminal_status: str,
    ) -> None:
        """Observe the real local generation transition without claiming stop ack."""

        observer = self._reference_observer
        submit = (
            None
            if observer is None
            else getattr(observer, "submit_generation_invalidation_nowait", None)
        )
        if not callable(submit):
            return
        try:
            submit(
                LegacyReferenceGenerationInvalidation(
                    legacy_worker_generation=int(legacy_worker_generation),
                    invalidated_at=float(time.monotonic()),
                    reason=str(reason),
                    terminal_status=str(terminal_status),
                )
            )
        except Exception as exc:  # noqa: BLE001 - passive diagnostics only.
            self._reference_observer_error_count += 1
            self._last_reference_observer_error = f"{type(exc).__name__}: {exc}"

    def _emit_reference(self, *, scheduled_timestamp: float) -> None:
        with self._dispatch_lock:
            actual_emit = _finite_time(self._clock(), name="clock")
            wall_emit = _finite_time(self._wall_clock(), name="wall_clock")
            query = (
                actual_emit
                if self.config.chunk_lookahead_enabled
                else actual_emit - self.config.sample_period_sec
            )
            with self._lock:
                if (
                    not self._active
                    or self._fault is not None
                    or self._fault_pending
                    or self._closing
                    or self._closed
                ):
                    return
                generation = self._generation
                pose_ref = (
                    self._pose_resampler.sample_at(
                        query,
                        emitted_timestamp=actual_emit,
                    )
                    if self._stream_signature and self._stream_signature[0]
                    else None
                )
                qpos_ref = (
                    self._qpos_resampler.sample_at(
                        query,
                        emitted_timestamp=actual_emit,
                    )
                    if self._stream_signature and self._stream_signature[1]
                    else None
                )
                parent_indices = self._parent_indices(pose_ref, qpos_ref)
                template_index = max(parent_indices)
                template = self._templates.get(template_index)
                if template is None:
                    raise RuntimeError(
                        f"missing reference template for source index {template_index}"
                    )
                reference_sequence = self._reference_count
                execution_work_item = self._execution_work_for_reference_locked(
                    parent_indices=parent_indices,
                    reference_sequence=reference_sequence,
                )

            metadata = self._reference_metadata(
                reference_sequence=reference_sequence,
                scheduled_timestamp=scheduled_timestamp,
                actual_emit=actual_emit,
                wall_emit=wall_emit,
                query=query,
                pose_ref=pose_ref,
                qpos_ref=qpos_ref,
                parent_indices=parent_indices,
            )
            reference_derivations = self._notify_reference_observer(
                generation=generation,
                scheduled_timestamp=scheduled_timestamp,
                actual_emit=actual_emit,
                pose_ref=pose_ref,
                qpos_ref=qpos_ref,
            )
            collision_qpos_by_side: Mapping[str, np.ndarray] | None = None
            if self.config.collision_required:
                if pose_ref is None or qpos_ref is not None:
                    raise RuntimeError(
                        "collision-required deploy expected a pure high-rate pose reference"
                    )
                with self._lock:
                    planner = self._collision_planner
                    planner_initialized = self._collision_planner_initialized
                if planner is None or not planner_initialized:
                    raise RuntimeError(
                        "collision planner is not reset from complete dual-arm qpos"
                    )
                try:
                    collision_result = planner.plan_pose_reference(
                        pose_ref.pose_by_side,
                        target_reference_frame="arm_base",
                        target_controlled_frame="tool_tcp",
                        source_timestamp=float(query),
                        source=(
                            f"deploy_pose_reference_{_rate_label(self.config.reference_hz)}hz"
                        ),
                        metadata=metadata,
                    )
                except Exception as exc:
                    self._notify_reference_resolution_failure_observer(
                        derivations=reference_derivations,
                        actual_emit=actual_emit,
                        failure_stage="planner",
                        error=exc,
                    )
                    raise
                collision_metadata = self._collision_result_metadata(collision_result)
                metadata.update(collision_metadata)
                with self._lock:
                    self._record_collision_result_locked(
                        collision_result,
                        collision_metadata,
                    )
                endpoint_checked = bool(
                    collision_metadata["position_endpoint_collision_checked"]
                )
                endpoint_passed = bool(
                    collision_metadata["position_endpoint_collision_passed"]
                )
                if (
                    not bool(getattr(collision_result, "valid", False))
                    or bool(getattr(collision_result, "hold", False))
                    or not endpoint_checked
                    or not endpoint_passed
                ):
                    reasons = tuple(
                        str(reason)
                        for reason in getattr(
                            collision_result,
                            "modification_reasons",
                            (),
                        )
                    )
                    planner_error = RuntimeError(
                        "High-rate collision-aware IK failed closed before arm/hand dispatch: "
                        f"endpoint_checked={endpoint_checked}, "
                        f"endpoint_passed={endpoint_passed}, reasons="
                        + (", ".join(reasons) if reasons else "unknown planner failure")
                    )
                    self._notify_reference_resolution_failure_observer(
                        derivations=reference_derivations,
                        actual_emit=actual_emit,
                        failure_stage="planner",
                        error=planner_error,
                    )
                    raise planner_error
                collision_qpos_by_side = {
                    side: np.asarray(collision_result.q_ref_by_side[side], dtype=float).copy()
                    for side in self.canonical_pipeline.active_sides
                }
            reference_filtered = self._build_reference_action(
                template.filtered,
                pose_ref=pose_ref,
                qpos_ref=qpos_ref,
                planned_qpos_by_side=collision_qpos_by_side,
                metadata=metadata,
            )
            reference_observation = replace(template.observation, timestamp=wall_emit)
            try:
                result = self.canonical_pipeline.resolve_filtered(
                    reference_filtered,
                    reference_observation,
                    policy_name=template.policy_name,
                )
            except Exception as exc:
                self._notify_reference_resolution_failure_observer(
                    derivations=reference_derivations,
                    actual_emit=actual_emit,
                    failure_stage="resolver",
                    error=exc,
                )
                raise
            try:
                command = self._validated_command(result)
            except Exception as exc:
                self._notify_reference_resolution_failure_observer(
                    derivations=reference_derivations,
                    actual_emit=actual_emit,
                    failure_stage="command_construction",
                    error=exc,
                )
                raise
            self._annotate_command(command, metadata)
            if execution_work_item is not None:
                command.execution_work_item = execution_work_item
                for effector_command in command.commands.values():
                    effector_command.execution_work_item = execution_work_item

            arm_dispatch_attempted = self.dispatch_callback is not None
            expected_joint_qpos_by_effector = {
                str(name): np.asarray(effector_command.value, dtype=float).reshape(6).copy()
                for name, effector_command in command.commands.items()
                if effector_command.command_type == "joint_vector"
            }
            with self._lock:
                if (
                    generation != self._generation
                    or not self._active
                    or self._fault is not None
                    or self._fault_pending
                    or self._closing
                ):
                    return
                (
                    discrete_source_index,
                    discrete_source_timestamp,
                    discrete_command,
                    discrete_skipped_before_this,
                ) = self._take_due_discrete_command_locked(query)
                if arm_dispatch_attempted:
                    self._arm_dispatch_attempt_count += 1
            self._notify_reference_resolution_observer(
                derivations=reference_derivations,
                actual_emit=actual_emit,
                command=command,
                collision_qpos_by_side=collision_qpos_by_side,
                metadata=metadata,
            )
            if arm_dispatch_attempted:
                dispatch_result: object = None
                try:
                    guard = self._execution_epoch_guard
                    if guard is not None:
                        assert execution_work_item is not None
                        check = guard.check_before_backend_call(execution_work_item)
                        if not check.accepted:
                            with self._lock:
                                self._execution_epoch_rejected_count += 1
                                self._append_debug_event_locked(
                                    "execution_work_rejected",
                                    {
                                        "work_item_id": execution_work_item.work_item_id,
                                        "checkpoint": check.checkpoint,
                                        "status": check.status,
                                    },
                                )
                            return
                        with self._lock:
                            self._inflight_execution_work_item = execution_work_item
                            self._last_old_epoch_invocation_started_at = float(
                                self._clock()
                            )
                    try:
                        dispatch_result = self.dispatch_callback(command)
                    finally:
                        if guard is not None:
                            with self._lock:
                                self._last_old_epoch_invocation_completed_at = float(
                                    self._clock()
                                )
                                self._last_completed_execution_work_item = (
                                    execution_work_item
                                )
                                self._inflight_execution_work_item = None
                    if dispatch_result is False:
                        raise RuntimeError("arm dispatch callback rejected the reference")
                    if self.config.collision_required:
                        dispatch_metadata = self._validate_collision_arm_dispatch_result(
                            dispatch_result,
                            expected_joint_qpos_by_effector=expected_joint_qpos_by_effector,
                        )
                        metadata.update(dispatch_metadata)
                        command.metadata.update(dispatch_metadata)
                        for effector_command in command.commands.values():
                            effector_command.metadata.update(dispatch_metadata)
                except Exception as exc:
                    failed_metadata = {
                        **dict(metadata),
                        "position_arm_dispatch_status": "failed",
                        "position_arm_dispatch_error_type": type(exc).__name__,
                        "position_arm_dispatch_error": str(exc),
                        "position_arm_backend_dispatch_receipt": (
                            self._arm_dispatch_receipt_payload(dispatch_result)
                        ),
                        "position_arm_backend_partial_receipts": (
                            self._arm_dispatch_receipt_payload(
                                {
                                    "accepted": False,
                                    "effectors": command.metadata.get(
                                        "deploy_backend_dispatch_receipts",
                                        {},
                                    ),
                                }
                            )
                        ),
                        "position_arm_commands": self._command_debug_payload(command),
                    }
                    with self._lock:
                        self._reference_count += 1
                        self._arm_dispatch_failed_count += 1
                        self._last_reference_metadata = dict(failed_metadata)
                        self._append_debug_event_locked(
                            "arm_reference_dispatch",
                            failed_metadata,
                        )
                        if discrete_command is not None:
                            self._record_discrete_canceled_locked(
                                discrete_command,
                                source_index=discrete_source_index,
                                source_timestamp=discrete_source_timestamp,
                                actual_emit=actual_emit,
                                wall_emit=wall_emit,
                                skipped_before_this=discrete_skipped_before_this,
                                reason="arm_reference_dispatch_failed",
                            )
                    raise

            if discrete_command is not None:
                self._annotate_discrete_command(
                    discrete_command,
                    source_index=discrete_source_index,
                    source_timestamp=discrete_source_timestamp,
                    actual_emit=actual_emit,
                    wall_emit=wall_emit,
                    skipped_before_this=discrete_skipped_before_this,
                )

            with self._lock:
                abort_after_dispatch = bool(
                    generation != self._generation
                    or not self._active
                    or self._fault is not None
                    or self._fault_pending
                    or self._closing
                )
                if abort_after_dispatch:
                    abort_reason = (
                        "fault_pending"
                        if self._fault_pending or self._fault is not None
                        else ("closing" if self._closing else "generation_changed")
                    )
                    self._reference_count += 1
                    if self.dispatch_callback is None:
                        self._dry_run_count += 1
                    else:
                        self._dispatch_count += 1
                    aborted_metadata = {
                        **dict(metadata),
                        "position_arm_dispatch_status": (
                            "dry_run_aborted"
                            if self.dispatch_callback is None
                            else "accepted_before_async_abort"
                        ),
                        "position_arm_dispatch_partial": bool(
                            self.dispatch_callback is not None
                        ),
                        "position_arm_dispatch_abort_reason": abort_reason,
                        "position_arm_commands": self._command_debug_payload(command),
                    }
                    self._last_reference_metadata = aborted_metadata
                    self._append_debug_event_locked(
                        "arm_reference_dispatch",
                        aborted_metadata,
                    )
                    if discrete_command is not None:
                        self._record_discrete_canceled_locked(
                            discrete_command,
                            source_index=discrete_source_index,
                            source_timestamp=discrete_source_timestamp,
                            actual_emit=actual_emit,
                            wall_emit=wall_emit,
                            skipped_before_this=discrete_skipped_before_this,
                            reason=f"arm_reference_{abort_reason}",
                        )
                    return
                self._reference_count += 1
                if self.dispatch_callback is None:
                    self._dry_run_count += 1
                else:
                    self._dispatch_count += 1
                committed_metadata = {
                    **dict(metadata),
                    "position_arm_dispatch_status": (
                        "dry_run" if self.dispatch_callback is None else "accepted"
                    ),
                    "position_arm_commands": self._command_debug_payload(command),
                }
                self._last_reference_metadata = dict(committed_metadata)
                self._append_debug_event_locked(
                    "arm_reference_dispatch",
                    committed_metadata,
                )
                if discrete_command is not None:
                    self._discrete_reference_count += 1
                    if self.discrete_dispatch_callback is None:
                        self._discrete_dry_run_count += 1
                        dry_metadata = dict(discrete_command.metadata)
                        dry_metadata["position_discrete_dispatch_status"] = "dry_run"
                        self._last_discrete_metadata = dry_metadata
                        self._append_debug_event_locked(
                            "discrete_dispatch",
                            dry_metadata,
                        )

            if (
                discrete_command is not None
                and self.discrete_dispatch_callback is not None
            ):
                assert discrete_source_index is not None
                assert discrete_source_timestamp is not None
                item = _DiscreteDispatchItem(
                    generation=generation,
                    source_index=int(discrete_source_index),
                    source_timestamp=float(discrete_source_timestamp),
                    logical_emit_monotonic=float(actual_emit),
                    logical_emit_wall=float(wall_emit),
                    skipped_before_this=int(discrete_skipped_before_this),
                    command=discrete_command,
                )
                self._enqueue_latest_discrete_item(item)

    def _enqueue_latest_discrete_item(self, item: _DiscreteDispatchItem) -> None:
        dropped_source_indices: list[int] = []
        base_skipped_before_this = int(item.skipped_before_this)
        inherited_skipped_count = 0
        newly_dropped_count = 0
        while True:
            if dropped_source_indices:
                total_skipped = base_skipped_before_this + inherited_skipped_count
                item = replace(item, skipped_before_this=total_skipped)
                coalesce_metadata = {
                    "position_discrete_catch_up": True,
                    "position_discrete_coalesced": True,
                    "position_discrete_skipped_before_this": total_skipped,
                    "position_discrete_worker_coalesced_source_indices": list(
                        dropped_source_indices
                    ),
                }
                item.command.metadata.update(coalesce_metadata)
                for effector_command in item.command.commands.values():
                    effector_command.metadata.update(coalesce_metadata)
            with self._lock:
                try:
                    self._discrete_queue.put_nowait(item)
                except queue.Full:
                    placed = False
                else:
                    placed = True
                    if newly_dropped_count:
                        self._discrete_skipped_count += newly_dropped_count
                        self._discrete_worker_coalesced_count += newly_dropped_count
                        self._append_debug_event_locked(
                            "discrete_dispatch_coalesced",
                            {
                                "retained_source_index": int(item.source_index),
                                "dropped_source_indices": list(dropped_source_indices),
                                "dropped_count": int(newly_dropped_count),
                                "inherited_skipped_count": int(inherited_skipped_count),
                            },
                        )
                    self._append_debug_event_locked(
                        "discrete_dispatch_queued",
                        {
                            **dict(item.command.metadata),
                            "discrete_queue_depth": int(self._discrete_queue.qsize()),
                        },
                    )
            if placed:
                return
            else:
                try:
                    dropped = self._discrete_queue.get_nowait()
                except queue.Empty:
                    continue
                inherited = dropped.command.metadata.get(
                    "position_discrete_worker_coalesced_source_indices",
                    (),
                )
                if isinstance(inherited, (list, tuple)):
                    dropped_source_indices.extend(int(value) for value in inherited)
                dropped_source_indices.append(int(dropped.source_index))
                inherited_skipped_count += 1 + int(dropped.skipped_before_this)
                newly_dropped_count += 1
                self._discrete_queue.task_done()

    def _take_due_discrete_command_locked(
        self,
        query_timestamp: float,
    ) -> tuple[int | None, float | None, TeleopCommand | None, int]:
        due = [
            (source_index, template)
            for source_index, template in self._templates.items()
            if source_index > self._last_discrete_source_index
            and template.source_timestamp <= query_timestamp + _TIME_EPSILON_SEC
        ]
        if not due:
            return None, None, None, 0

        self._last_discrete_source_index = max(source_index for source_index, _ in due)
        commands = [
            (source_index, template)
            for source_index, template in due
            if template.discrete_command is not None and template.discrete_command.commands
        ]
        if not commands:
            return None, None, None, 0

        # Never create a catch-up burst after a late reference tick.  The newest
        # due 15 Hz hand keyframe wins, exactly as the arm reference does.
        skipped_before_this = max(0, len(commands) - 1)
        if skipped_before_this:
            self._discrete_skipped_count += skipped_before_this
        source_index, template = commands[-1]
        assert template.discrete_command is not None
        return (
            int(source_index),
            float(template.source_timestamp),
            _copy_teleop_command(template.discrete_command),
            int(skipped_before_this),
        )

    def _record_discrete_canceled_locked(
        self,
        command: TeleopCommand,
        *,
        source_index: int | None,
        source_timestamp: float | None,
        actual_emit: float,
        wall_emit: float,
        skipped_before_this: int,
        reason: str,
    ) -> None:
        self._annotate_discrete_command(
            command,
            source_index=source_index,
            source_timestamp=source_timestamp,
            actual_emit=actual_emit,
            wall_emit=wall_emit,
            skipped_before_this=skipped_before_this,
        )
        canceled = {
            **dict(command.metadata),
            "position_discrete_dispatch_status": "canceled",
            "position_discrete_cancel_reason": str(reason),
        }
        self._discrete_reference_count += 1
        self._discrete_canceled_count += 1
        self._last_discrete_metadata = canceled
        self._append_debug_event_locked("discrete_dispatch", canceled)

    @staticmethod
    def _parent_indices(
        pose_ref: ResampledMotionReference | None,
        qpos_ref: ResampledMotionReference | None,
    ) -> tuple[int, ...]:
        indices: set[int] = set()
        for reference in (pose_ref, qpos_ref):
            if reference is not None:
                indices.add(reference.left_source_index)
                indices.add(reference.right_source_index)
        if not indices:
            raise RuntimeError("reference contains neither pose nor qpos")
        return tuple(sorted(indices))

    def _reference_metadata(
        self,
        *,
        reference_sequence: int,
        scheduled_timestamp: float,
        actual_emit: float,
        wall_emit: float,
        query: float,
        pose_ref: ResampledMotionReference | None,
        qpos_ref: ResampledMotionReference | None,
        parent_indices: tuple[int, ...],
    ) -> dict[str, Any]:
        references = [ref for ref in (pose_ref, qpos_ref) if ref is not None]
        alpha_by_modality = {
            name: float(reference.interpolation_alpha)
            for name, reference in (("pose", pose_ref), ("qpos", qpos_ref))
            if reference is not None
        }
        unique_alphas = tuple(sorted(set(alpha_by_modality.values())))
        interpolation_alpha: float | None = (
            unique_alphas[0] if len(unique_alphas) == 1 else None
        )
        parent_by_modality = {
            name: [reference.left_source_index, reference.right_source_index]
            for name, reference in (("pose", pose_ref), ("qpos", qpos_ref))
            if reference is not None
        }
        source_timestamps_by_modality = {
            name: [reference.left_source_timestamp, reference.right_source_timestamp]
            for name, reference in (("pose", pose_ref), ("qpos", qpos_ref))
            if reference is not None
        }
        chain = (
            f"safe_{_rate_label(self.config.sample_hz)}"
            f"->causal_{_rate_label(self.config.reference_hz)}"
            f"->{_rate_label(self.config.command_hz)}"
        )
        pose_action_interval_velocity = (
            {}
            if pose_ref is None
            else dict(
                pose_ref.metadata.get("pose_action_interval_velocity", {}) or {}
            )
        )
        return {
            "position_execution_contract": chain,
            # Keep the historical metadata key for consumers, but report the
            # actually configured chain instead of the module's compatibility
            # defaults.  Profile identity is owned by the caller/config
            # resolver; the executor only knows effective rates.
            "position_default_profile_contract": chain,
            "position_configured_profile_contract": chain,
            "position_execution_chain": chain,
            "position_pipeline_stage_order": (
                f"safe_{_rate_label(self.config.sample_hz)}hz_keyframe",
                f"causal_{_rate_label(self.config.reference_hz)}hz_reference",
                f"position_{_rate_label(self.config.command_hz)}hz_command_stream",
            ),
            "position_sample_hz": float(self.config.sample_hz),
            "position_reference_hz": float(self.config.reference_hz),
            "position_command_hz": float(self.config.command_hz),
            "position_reference_sequence": int(reference_sequence),
            "position_reference_wall_time": float(wall_emit),
            "position_reference_emitted_timestamp": float(wall_emit),
            "position_reference_scheduled_monotonic_time": float(scheduled_timestamp),
            "position_reference_emitted_monotonic_time": float(actual_emit),
            "position_reference_query_monotonic_time": float(query),
            "position_causal_output_lag_sec": float(actual_emit - query),
            "position_causal_delay_periods": (
                0 if self.config.chunk_lookahead_enabled else 1
            ),
            "position_chunk_lookahead_enabled": bool(
                self.config.chunk_lookahead_enabled
            ),
            "position_parent_source_indices": list(parent_indices),
            "position_parent_source_indices_by_modality": parent_by_modality,
            "position_parent_source_timestamps_by_modality": source_timestamps_by_modality,
            "position_interpolation_alpha": interpolation_alpha,
            "position_interpolation_alpha_by_modality": alpha_by_modality,
            "position_pose_interpolation": (
                "xyz_linear+quaternion_shortest_path_slerp"
                if pose_ref is not None
                else None
            ),
            "pose_action_interval_velocity": pose_action_interval_velocity,
            "position_pose_speed_source": (
                "adjacent_pose_action_interval"
                if bool(pose_action_interval_velocity.get("quality_valid", False))
                else None
            ),
            "position_qpos_interpolation": "linear" if qpos_ref is not None else None,
            "position_causal": True,
            "position_lookahead_used": bool(
                any(bool(ref.metadata.get("lookahead_used")) for ref in references)
            ),
            "position_lookahead_available_at_emit": bool(
                all(bool(ref.metadata.get("lookahead_available_at_emit")) for ref in references)
            ),
            "position_safety_applied_at": (
                f"{_rate_label(self.config.sample_hz)}hz_keyframe_only"
            ),
            "position_safety_reapplied_at_reference": False,
            "position_endpoint_collision_checked": False,
            "position_segment_collision_checked": False,
            "position_collision_certification_status": "not_checked",
            "no_temporal_smoothing": True,
            "temporal_operation": "timestamp_interpolation_only",
        }

    def _collision_result_metadata(
        self,
        result: RM65PositionCollisionPlanResult,
    ) -> dict[str, Any]:
        planner_metadata = dict(getattr(result, "metadata", {}) or {})
        hold = bool(getattr(result, "hold", False))
        endpoint_checked = bool(planner_metadata.get("endpoint_collision_checked", False))
        endpoint_passed = bool(
            planner_metadata.get("endpoint_collision_passed", False)
        ) and not hold
        summary: dict[str, Any] = {
            "position_collision_planner": str(
                planner_metadata.get("planner", "rm65_coupled_position_collision")
            ),
            "position_collision_planner_sequence": int(getattr(result, "sequence", -1)),
            "position_collision_status": str(getattr(result, "status", "unknown")),
            "position_collision_hold": hold,
            "position_collision_modification_reasons": [
                str(reason)
                for reason in getattr(result, "modification_reasons", ())
            ],
            "position_collision_distance_m": planner_metadata.get(
                "collision_distance_m"
            ),
            "position_collision_scene_profile": planner_metadata.get("scene_profile"),
            "position_collision_pair_count": planner_metadata.get(
                "collision_pair_count"
            ),
            "position_collision_solve_ms": planner_metadata.get("solve_ms"),
            "position_collision_frame_total_ms": planner_metadata.get("frame_total_ms"),
            "position_collision_time_budget_exceeded": bool(
                planner_metadata.get("solve_time_budget_exceeded", False)
            ),
            "position_endpoint_collision_checked": endpoint_checked,
            "position_endpoint_collision_passed": endpoint_passed,
            "position_segment_collision_checked": False,
            "position_segment_collision_certified": False,
            "position_collision_certification_status": (
                "reference_endpoint_only_segment_not_certified"
            ),
            "position_collision_geometry_model": "selected_scene_profile_fixed_geometry",
            "position_arbitrary_tool_geometry_certified": False,
            "temporal_smoothing_applied": False,
            "no_temporal_smoothing": True,
        }
        if self.config.debug_events_enabled:
            summary["position_collision_debug"] = {
                "q_ik_raw_by_side": {
                    str(side): np.asarray(value, dtype=float).copy()
                    for side, value in getattr(result, "q_ik_raw_by_side", {}).items()
                },
                "q_ref_by_side": {
                    str(side): np.asarray(value, dtype=float).copy()
                    for side, value in getattr(result, "q_ref_by_side", {}).items()
                },
                "target_endpoint_pose_world_by_side": {
                    str(side): np.asarray(value, dtype=float).copy()
                    for side, value in getattr(
                        result,
                        "target_endpoint_pose_world_by_side",
                        {},
                    ).items()
                },
                "realized_endpoint_pose_world_by_side": {
                    str(side): np.asarray(value, dtype=float).copy()
                    for side, value in getattr(
                        result,
                        "realized_endpoint_pose_world_by_side",
                        {},
                    ).items()
                },
                "planner_metadata": planner_metadata,
            }
        return summary

    def _record_collision_result_locked(
        self,
        result: RM65PositionCollisionPlanResult,
        metadata: Mapping[str, Any],
    ) -> None:
        status = str(getattr(result, "status", "unknown"))
        if bool(getattr(result, "hold", False)):
            self._collision_hold_count += 1
        elif status == "collision_backoff":
            self._collision_backoff_count += 1
        elif status == "limited":
            self._collision_limited_count += 1
        else:
            self._collision_tracking_count += 1
        self._last_collision_metadata = dict(metadata)
        self._append_debug_event_locked("collision_plan", dict(metadata))

    @staticmethod
    def _build_reference_action(
        template: FilteredNormalizedAction,
        *,
        pose_ref: ResampledMotionReference | None,
        qpos_ref: ResampledMotionReference | None,
        planned_qpos_by_side: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any],
    ) -> FilteredNormalizedAction:
        vector = np.asarray(template.action.vector, dtype=np.float32).copy()
        action_metadata = dict(template.action.metadata)
        action_metadata.update(dict(metadata))
        qpos_targets: dict[str, np.ndarray] = {}
        if pose_ref is not None:
            for side, pose in pose_ref.pose_by_side.items():
                xyzquat = pose_matrix_to_xyzquat(pose)
                if xyzquat is None:
                    raise ValueError(f"{side} interpolated pose is invalid")
                vector[ACTION_ARM_POSE_SLICES[side]] = xyzquat
        if qpos_ref is not None:
            qpos_targets = {
                side: np.asarray(value, dtype=np.float32).copy()
                for side, value in qpos_ref.qpos_by_side.items()
            }
        if planned_qpos_by_side is not None:
            qpos_targets = {
                str(side): np.asarray(value, dtype=np.float32).reshape(6).copy()
                for side, value in planned_qpos_by_side.items()
            }
        return FilteredNormalizedAction(
            action=NormalizedAction(
                vector=vector,
                arm_joint_targets=qpos_targets,
                metadata=action_metadata,
            ),
            faults=list(template.faults),
            metadata=dict(template.metadata),
        )

    @staticmethod
    def _validated_command(result: CanonicalDeployActionResult) -> TeleopCommand:
        if result.faults:
            raise RuntimeError(
                "canonical position reference is blocked: "
                + ", ".join(str(fault) for fault in result.faults)
            )
        if result.command is None or not result.command.commands:
            raise RuntimeError("canonical position reference produced no arm command")
        non_arm = [name for name in result.command.commands if not str(name).endswith("_arm")]
        if non_arm:
            raise RuntimeError(f"position deploy executor received non-arm commands: {non_arm}")
        return result.command

    @staticmethod
    def _annotate_command(command: TeleopCommand, metadata: Mapping[str, Any]) -> None:
        command.metadata.update(dict(metadata))
        for effector_command in command.commands.values():
            effector_command.metadata.update(dict(metadata))
            if bool(metadata.get("position_endpoint_collision_passed", False)):
                # The coupled planner already applied absolute and per-45-Hz
                # joint limits.  Mark this explicitly so RM65 does not apply a
                # second independent per-arm step clip; any later singularity
                # hold/rewrite is instead detected from the dispatch receipt.
                effector_command.metadata["action_pose_source"] = (
                    "limited_solver_output_qpos"
                )
                effector_command.metadata["position_collision_prelimited_qpos"] = True

    def _command_debug_payload(self, command: TeleopCommand) -> dict[str, dict[str, Any]]:
        if not self.config.debug_events_enabled:
            return {}
        payload: dict[str, dict[str, Any]] = {}
        for name, effector_command in command.commands.items():
            value = effector_command.value
            if isinstance(value, np.ndarray):
                value = value.copy()
            payload[str(name)] = {
                "command_type": str(effector_command.command_type),
                "frame_name": effector_command.frame_name,
                "value": value,
            }
        return payload

    def _arm_dispatch_receipt_payload(self, result: object) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            return {"structured": False, "result_type": type(result).__name__}
        raw_effectors = result.get("effectors")
        effectors: dict[str, dict[str, Any]] = {}
        if isinstance(raw_effectors, Mapping):
            for name, raw_receipt in raw_effectors.items():
                if not isinstance(raw_receipt, Mapping):
                    effectors[str(name)] = {"structured": False}
                    continue
                receipt = dict(raw_receipt)
                item = {
                    key: receipt.get(key)
                    for key in (
                        "accepted",
                        "backend_type",
                        "backend_ready",
                        "command_type",
                        "rewritten",
                        "operation",
                        "status",
                        "result_code",
                        "result_text",
                        "state_error",
                        "singularity_gate_status",
                        "background_streaming",
                    )
                    if key in receipt
                }
                if self.config.debug_events_enabled:
                    for key in (
                        "requested_value",
                        "final_value",
                        "backend_final_target_qpos",
                        "receipt_metadata",
                    ):
                        if key in receipt:
                            value = receipt.get(key)
                            item[key] = (
                                value.copy() if isinstance(value, np.ndarray) else value
                            )
                effectors[str(name)] = item
        group_health = result.get("arm_backend_health")
        return {
            "structured": True,
            "accepted": bool(result.get("accepted", False)),
            "effectors": effectors,
            "arm_backend_health": (
                dict(group_health) if isinstance(group_health, Mapping) else None
            ),
        }

    def _annotate_discrete_command(
        self,
        command: TeleopCommand,
        *,
        source_index: int | None,
        source_timestamp: float | None,
        actual_emit: float,
        wall_emit: float,
        skipped_before_this: int,
    ) -> None:
        if source_index is None or source_timestamp is None:
            raise RuntimeError("due discrete command is missing its source identity")
        metadata = {
            "position_discrete_source_index": int(source_index),
            "position_discrete_source_monotonic_time": float(source_timestamp),
            "position_discrete_logical_tick_monotonic_time": float(actual_emit),
            "position_discrete_logical_tick_wall_time": float(wall_emit),
            "position_discrete_logical_causal_lag_sec": float(
                actual_emit - source_timestamp
            ),
            "position_discrete_rate_hz": float(self.config.sample_hz),
            "position_discrete_interpolation": "none",
            "position_discrete_catch_up": bool(skipped_before_this),
            "position_discrete_coalesced": bool(skipped_before_this),
            "position_discrete_skipped_before_this": int(skipped_before_this),
            "position_arm_hand_phase": (
                "same_logical_reference_tick_with_chunk_lookahead"
                if self.config.chunk_lookahead_enabled
                else "same_logical_reference_tick_after_one_action_period_causal_delay"
            ),
        }
        command.timestamp = float(wall_emit)
        command.metadata.update(metadata)
        for effector_command in command.commands.values():
            effector_command.metadata.update(metadata)

    def _raise_if_fault_locked(self) -> None:
        if self._fault_pending and self._fault is None:
            stage = self._fault_pending_stage or "background_worker"
            detail = "" if not self._fault_pending_error else f": {self._fault_pending_error}"
            raise PositionDeployExecutorFault(
                f"position deploy executor fault is pending at {stage}{detail}"
            )
        if self._fault is None:
            return
        message = "position deploy executor background fault is latched"
        if self._fault_stage:
            message += f" at {self._fault_stage}"
        message += f": {type(self._fault).__name__}: {self._fault}"
        raise PositionDeployExecutorFault(message) from self._fault

    def _reject_worker_lifecycle_call(self, operation: str) -> None:
        current = threading.current_thread()
        with self._lock:
            in_hold_callback = self._hold_callback_thread_id == threading.get_ident()
        if (
            current is self._thread
            or current is self._discrete_thread
            or in_hold_callback
        ):
            raise RuntimeError(
                f"position deploy {operation} cannot be called from its worker/hold callback"
            )

    def _mark_fault_pending(self, exc: BaseException, *, stage: str) -> None:
        with self._lock:
            if self._fault is not None:
                return
            self._fault_pending = True
            self._fault_pending_stage = str(stage)
            self._fault_pending_error = f"{type(exc).__name__}: {exc}"
            self._append_debug_event_locked(
                "fault_pending",
                {
                    "stage": str(stage),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

    def _latch_fault(
        self,
        exc: BaseException,
        *,
        stage: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        should_hold = False
        with self._dispatch_lock:
            with self._lock:
                if self._fault is not None:
                    return
                self._fault = exc
                self._fault_stage = str(stage)
                self._fault_pending = False
                self._fault_pending_stage = None
                self._fault_pending_error = None
                self._fault_metadata = dict(metadata or {})
                self._fault_metadata.update(
                    {
                        "stage": str(stage),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                should_hold = bool(
                    self.dispatch_callback is not None
                    or self._active
                    or not self._queue.empty()
                    or not self._discrete_queue.empty()
                    or self._hold_retry_required
                )
                invalidated_generation = self._generation
                self._generation += 1
                self._notify_reference_generation_invalidation_locked(
                    legacy_worker_generation=invalidated_generation,
                    reason=f"fault:{stage}",
                    terminal_status="faulted",
                )
                self._reset_stream_locked()
                self._append_debug_event_locked(
                    "executor_fault",
                    self._fault_metadata,
                )
            with self._discrete_dispatch_lock:
                if should_hold:
                    try:
                        self._invoke_hold()
                    except Exception as hold_exc:
                        with self._lock:
                            self._append_debug_event_locked(
                                "hold_failed",
                                {
                                    "primary_fault_stage": str(stage),
                                    "error_type": type(hold_exc).__name__,
                                    "error": str(hold_exc),
                                },
                            )
        self._wake.set()

    def _invoke_hold(self) -> None:
        if self.hold_callback is None:
            return
        callback_thread_id = threading.get_ident()
        with self._lock:
            self._hold_callback_thread_id = callback_thread_id
        try:
            self.hold_callback()
        except Exception as exc:
            with self._lock:
                self._hold_failure_count += 1
                self._last_hold_error = f"{type(exc).__name__}: {exc}"
                self._hold_retry_required = True
            raise
        else:
            with self._lock:
                self._hold_count += 1
                self._last_hold_error = None
                self._hold_retry_required = False
        finally:
            with self._lock:
                if self._hold_callback_thread_id == callback_thread_id:
                    self._hold_callback_thread_id = None

    def _append_debug_event_locked(
        self,
        event: str,
        payload: Mapping[str, Any],
    ) -> None:
        if not self.config.debug_events_enabled:
            return
        try:
            observed_wall_time = float(self._wall_clock())
        except Exception:
            observed_wall_time = time.time()
        try:
            observed_monotonic_time = float(self._clock())
        except Exception:
            observed_monotonic_time = time.monotonic()
        if len(self._debug_events) >= self._debug_events.maxlen:
            self._debug_event_drop_count += 1
        self._debug_events.append(
            {
                "event": str(event),
                "observed_wall_time": observed_wall_time,
                "observed_monotonic_time": observed_monotonic_time,
                **dict(payload),
            }
        )

    def _drain_discrete_queue_locked(self) -> None:
        while True:
            try:
                self._discrete_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._discrete_canceled_count += 1
                self._discrete_queue.task_done()

    def _reset_stream_locked(self) -> None:
        self._active = False
        self._collision_planner_initialized = False
        self._submission_stream_initialized = False
        self._stream_signature = None
        self._last_submitted_timestamp = None
        self._last_source_timestamp = None
        self._next_tick_timestamp = None
        self._staged_lookahead_source_index = None
        self._staged_lookahead_timestamp = None
        self._staged_lookahead_pose_by_side = None
        self._staged_lookahead_qpos_by_side = None
        self._staged_lookahead_discrete_command = None
        self._staged_lookahead_execution_work_item = None
        self._terminal_pose_by_side = None
        self._terminal_qpos_by_side = None
        self._terminal_timestamp = None
        self._last_discrete_source_index = -1
        self._pose_resampler.clear()
        self._qpos_resampler.clear()
        self._templates.clear()
        self._legacy_reference_source_facts.clear()
        self._drain_discrete_queue_locked()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()


__all__ = [
    "PositionDeployExecutor",
    "PositionDeployExecutorConfig",
    "PositionDeployExecutorFault",
    "PositionDeployQueueFullError",
]
