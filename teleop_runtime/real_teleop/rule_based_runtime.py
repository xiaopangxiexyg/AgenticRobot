"""Lightweight rule-plan extension for the existing RealTeleop owner loop.

This module owns neither hardware lifecycle nor dispatch.  It claims the one
existing command-override slot, consumes the formal recorder's causal arm
feedback, and delegates every command to the normal RealTeleop safety path.
"""

from __future__ import annotations

import copy
import math
import time
from bisect import bisect_right
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from queue import Empty, SimpleQueue
from typing import Any

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract
from robot_motion.pose import pose_delta_metrics
from robot_motion.relative_action import pose_matrix_to_xyzquat, xyzquat_to_pose_matrix
from robot_motion.action import HandActionSpec, RobotAction
from robot_motion.position_control import (
    DEFAULT_POSITION_REFERENCE_HZ,
    MotionTargetResampler,
    PoseActionMotionIntentStateMachine,
    TimestampedMotionTarget,
    build_uniform_time_grid,
    resolve_periodic_tick,
)
from robot_motion.rm65 import (
    BimanualPoseTarget,
    BimanualPoseTargetSource,
    DualArmCollisionAwareRetargeter,
)
from robot_motion.transforms import project_near_rigid_pose
from robot_motion.runtime import (
    EffectorCommand,
    TeleopCommand,
    command_effector_kind_subset,
    format_readiness_report,
    readiness_report_blockers,
)
from robot_motion.replay import (
    CANONICAL_DESK_ACTION_FRAME,
    HARDWARE_ARM_BASE_ACTION_FRAME,
    ReplayTrajectoryGroup,
    RuleBasedActionSegment,
    RuleBasedBoundaryState,
    RuleBasedCollectionCursor,
    RuleBasedCollectionDirective,
    RuleBasedCollectionPlan,
    RuleBasedInitialize,
    RuleBasedReplay,
    replay_action_pose_frame,
)

from ..collection.rule_based_adapter import (
    RuleBasedCollectionAdapter,
    RuleBasedCollectionEmission,
)
from ..backends.grippers import DMTAC_GRIPPER_RECOVERABLE_SEND_STATUSES
from ..collection.action_record import validate_collection_command_action
from ..command_override import TeleopCommandOverride
from ..diagnostics import FaultEvent, FaultSeverity
from .arm_runtime import _sync_dual_arm_retargeter_for_real_control
from .pose_frame_runtime import (
    RealTeleopPoseFrameActionResolver,
    resolve_runtime_pose_frame_contract,
)
from .rule_based_initial_alignment import (
    RuleBasedInitialAlignmentConfig,
    RuleBasedInitialAlignmentPlan,
    RuleBasedInitialAlignmentPlanner,
    RuleBasedInitialAlignmentPlanningError,
    rule_based_coupled_pose_command_fault,
    rule_based_initial_alignment_command_fault,
)
from .runtime_extension import RealTeleopDatasetSampleAdmission

RULE_BASED_REAL_TELEOP_SCHEMA_V1 = "dexteleop.rule_based_real_teleop.v1"

# Collection poses are stored as float32 xyz+quaternion values.  Converting a
# valid unit quaternion to a matrix can therefore leave O(1e-8) orthogonality
# drift, while the final Active TCP envelope intentionally accepts only a
# rigid pose at 1e-9.  Admit only representational drift here; materially bad
# matrices must still fail closed before coupled IK.
_RULE_POSE_RIGID_PROJECTION_LIMIT = 1e-5
_RULE_UI_STATUS_HZ = 15.0
_WRITING_REACHABILITY_MAX_CHECK_SEC = 4.0
_WRITING_REACHABILITY_MAX_AUTO_SKIPS = 32
_WRITING_HEIGHT_DESCENT_STEP_ID = "right_write.pen_down.height_descent"
_RANDOM_SOURCE_SELECTION_ID = "__rule_source_pool_random__"


def _mutable_metadata_copy(value: Any) -> Any:
    """Copy immutable resampling metadata into command-safe containers."""

    if isinstance(value, Mapping):
        return {
            copy.deepcopy(key): _mutable_metadata_copy(item)
            for key, item in value.items()
        }
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(_mutable_metadata_copy(item) for item in value)
    if isinstance(value, list):
        return [_mutable_metadata_copy(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class RuleBasedRealTeleopDeployment:
    """Pre-device-open deployment evidence required by this extension."""

    active_sides: tuple[str, ...]
    collection_pose_frame_contract: PoseFrameContract
    tcp_pose_envelope_profile: str

    def __post_init__(self) -> None:
        sides = tuple(str(side) for side in self.active_sides)
        if set(sides) != {"left", "right"} or len(sides) != 2:
            raise ValueError("rule-based real control requires both active RM65 arms.")
        if not isinstance(self.collection_pose_frame_contract, PoseFrameContract):
            raise TypeError("collection_pose_frame_contract must be PoseFrameContract.")
        if self.collection_pose_frame_contract.is_raw_storage:
            raise ValueError(
                "rule-based real control requires canonical desk collection storage."
            )
        profile = str(self.tcp_pose_envelope_profile).strip()
        if not profile:
            raise ValueError("tcp_pose_envelope_profile must be non-empty.")
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(self, "tcp_pose_envelope_profile", profile)


def preflight_rule_based_real_teleop_deployment(
    args: object,
) -> RuleBasedRealTeleopDeployment:
    """Fail closed on a conservative first real-robot deployment profile."""

    if set(getattr(args, "active_sides", ())) != {"left", "right"}:
        raise ValueError("rule-based real control requires --active-sides both.")
    if str(getattr(args, "arm_control_mode", "")) != "position":
        raise ValueError("rule-based real control currently requires position mode.")
    if bool(getattr(args, "disable_dual_arm_collision_avoidance", False)):
        raise ValueError("rule-based real control refuses disabled collision avoidance.")
    if not bool(getattr(args, "tcp_pose_envelope_active", False)):
        raise ValueError("rule-based real control requires an Active TCP pose envelope.")
    profile = str(getattr(args, "tcp_pose_envelope_profile", "") or "").strip()
    if not profile:
        raise ValueError("rule-based real control requires a TCP envelope profile.")
    if not bool(getattr(args, "position_force_guard", False)):
        raise ValueError("rule-based real control requires the position force guard.")
    if str(getattr(args, "position_force_guard_source", "tool")) != "tool":
        raise ValueError("rule-based real control requires tool-frame force guarding.")
    if bool(getattr(args, "no_collection", False)):
        raise ValueError("rule-based real control requires formal collection.")
    if any(
        bool(getattr(args, name, False))
        for name in (
            "position_admittance_active",
            "position_admittance_rotation_active",
            "position_admittance_combined_active",
        )
    ):
        raise ValueError(
            "the first rule-based real deployment refuses Active admittance overlays."
        )
    contract = getattr(args, "pose_frame_contract", None)
    return RuleBasedRealTeleopDeployment(
        active_sides=tuple(getattr(args, "active_sides", ())),
        collection_pose_frame_contract=contract,
        tcp_pose_envelope_profile=profile,
    )


@dataclass(frozen=True)
class RuleBasedRealTeleopConfig:
    """Only rule pacing and measured-arrival thresholds live here."""

    action_hz: float | None = None
    position_reference_hz: float = DEFAULT_POSITION_REFERENCE_HZ
    qpos_tolerance_rad: float = math.radians(2.0)
    terminal_translation_tolerance_m: float = 0.006
    terminal_rotation_tolerance_deg: float = 4.0
    settle_consecutive_samples: int = 3
    target_timeout_sec: float = 30.0
    initial_alignment_speed_scale: float = 1.0
    pre_pen_down_replay_normal_episodes: int = 0
    gripper_sides: tuple[str, ...] = ()
    gripper_max_step: float = 0.05
    gripper_qpos_tolerance: float = 0.05
    initial_alignment_hand_start_drift_tolerance: float = 0.02
    manual_gripper_tolerance: float = 0.02
    manual_gripper_endpoint_tolerance: float = 0.002
    manual_gripper_resend_interval_sec: float = 0.25
    manual_gripper_timeout_sec: float = 20.0
    hand_state_max_age_sec: float = 0.5
    arm_feedback_recovery_timeout_sec: float = 0.5
    # Keep this additive field after the established positional fields so
    # compatibility fixtures and external callers retain their bindings.
    manual_gripper_settle_samples: int = 3
    minimum_stage_sample_coverage: float = 0.8
    minimum_stage_sample_count: int = 2

    def __post_init__(self) -> None:
        if self.action_hz is not None and (
            not np.isfinite(float(self.action_hz)) or float(self.action_hz) <= 0.0
        ):
            raise ValueError("action_hz must be finite and positive when provided.")
        for name in (
            "position_reference_hz",
            "qpos_tolerance_rad",
            "terminal_translation_tolerance_m",
            "terminal_rotation_tolerance_deg",
            "target_timeout_sec",
            "initial_alignment_speed_scale",
            "gripper_max_step",
            "gripper_qpos_tolerance",
            "initial_alignment_hand_start_drift_tolerance",
            "manual_gripper_tolerance",
            "manual_gripper_endpoint_tolerance",
            "manual_gripper_resend_interval_sec",
            "manual_gripper_timeout_sec",
            "hand_state_max_age_sec",
            "arm_feedback_recovery_timeout_sec",
            "minimum_stage_sample_coverage",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if float(self.initial_alignment_speed_scale) > 1.0:
            raise ValueError("initial_alignment_speed_scale must be in (0, 1].")
        if float(self.minimum_stage_sample_coverage) > 1.0:
            raise ValueError("minimum_stage_sample_coverage must be in (0, 1].")
        minimum_stage_sample_count = self.minimum_stage_sample_count
        if (
            isinstance(minimum_stage_sample_count, bool)
            or not isinstance(minimum_stage_sample_count, (int, np.integer))
            or int(minimum_stage_sample_count) <= 0
        ):
            raise ValueError("minimum_stage_sample_count must be a positive integer.")
        object.__setattr__(
            self,
            "minimum_stage_sample_count",
            int(minimum_stage_sample_count),
        )
        normal_episodes = self.pre_pen_down_replay_normal_episodes
        if (
            isinstance(normal_episodes, bool)
            or not isinstance(normal_episodes, (int, np.integer))
            or int(normal_episodes) < 0
        ):
            raise ValueError(
                "pre_pen_down_replay_normal_episodes must be a non-negative integer."
            )
        object.__setattr__(
            self,
            "pre_pen_down_replay_normal_episodes",
            int(normal_episodes),
        )
        if float(self.gripper_max_step) > 1.0:
            raise ValueError("gripper_max_step must be in (0, 1].")
        if float(self.manual_gripper_tolerance) > float(self.gripper_max_step):
            raise ValueError(
                "manual_gripper_tolerance must not exceed gripper_max_step."
            )
        if float(self.manual_gripper_endpoint_tolerance) > float(
            self.manual_gripper_tolerance
        ):
            raise ValueError(
                "manual_gripper_endpoint_tolerance must not exceed "
                "manual_gripper_tolerance."
            )
        settle_samples = self.manual_gripper_settle_samples
        if (
            isinstance(settle_samples, bool)
            or not isinstance(settle_samples, (int, np.integer))
            or int(settle_samples) <= 0
        ):
            raise ValueError("manual_gripper_settle_samples must be a positive integer.")
        object.__setattr__(self, "manual_gripper_settle_samples", int(settle_samples))
        if float(self.initial_alignment_hand_start_drift_tolerance) > float(
            self.gripper_max_step
        ):
            raise ValueError(
                "initial_alignment_hand_start_drift_tolerance must not exceed "
                "gripper_max_step."
            )
        gripper_sides = tuple(str(side).strip().lower() for side in self.gripper_sides)
        if any(side not in {"left", "right"} for side in gripper_sides):
            raise ValueError("gripper_sides must contain only left/right.")
        if len(set(gripper_sides)) != len(gripper_sides):
            raise ValueError("gripper_sides must not contain duplicates.")
        object.__setattr__(self, "gripper_sides", gripper_sides)
        if int(self.settle_consecutive_samples) <= 0:
            raise ValueError("settle_consecutive_samples must be positive.")


@dataclass(frozen=True)
class RuleBasedWritingHeightCalibration:
    """One operator-captured right tool-TCP height in canonical desk space."""

    calibrated_z_m: float
    lower_tolerance_m: float
    upper_tolerance_m: float
    captured_at: float
    source_timestamp: float | None
    pose_frame_contract_hash: str
    frame: str = CANONICAL_DESK_ACTION_FRAME
    side: str = "right"
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in (
            "calibrated_z_m",
            "lower_tolerance_m",
            "upper_tolerance_m",
            "captured_at",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.lower_tolerance_m < 0.0001 or self.upper_tolerance_m < 0.0001:
            raise ValueError("height calibration tolerances must be at least 0.1 mm.")
        if self.lower_tolerance_m > 0.02 or self.upper_tolerance_m > 0.02:
            raise ValueError("height calibration tolerances must not exceed 20 mm.")
        if self.source_timestamp is not None and not np.isfinite(
            float(self.source_timestamp)
        ):
            raise ValueError("source_timestamp must be finite when provided.")
        if self.frame != CANONICAL_DESK_ACTION_FRAME or self.side != "right":
            raise ValueError("writing height calibration must bind right canonical_desk TCP.")
        if not str(self.pose_frame_contract_hash).strip():
            raise ValueError("pose_frame_contract_hash must be non-empty.")

    @property
    def lower_bound_z_m(self) -> float:
        return float(self.calibrated_z_m - self.lower_tolerance_m)

    @property
    def upper_bound_z_m(self) -> float:
        return float(self.calibrated_z_m + self.upper_tolerance_m)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "side": self.side,
            "frame": self.frame,
            "calibrated_z_m": float(self.calibrated_z_m),
            "lower_tolerance_m": float(self.lower_tolerance_m),
            "upper_tolerance_m": float(self.upper_tolerance_m),
            "lower_bound_z_m": self.lower_bound_z_m,
            "upper_bound_z_m": self.upper_bound_z_m,
            "captured_at": float(self.captured_at),
            "source_timestamp": (
                None if self.source_timestamp is None else float(self.source_timestamp)
            ),
            "pose_frame_contract_hash": self.pose_frame_contract_hash,
        }


@dataclass(frozen=True)
class RuleBasedRealTeleopStatus:
    started: bool
    armed: bool
    execution_begun: bool
    completed: bool
    aborted: bool
    held: bool
    recording_ready: bool
    waiting_reason: str
    step_id: str | None
    directive_kind: str | None
    sequence_index: int | None
    published_boundaries: tuple[str, ...] = field(default_factory=tuple)
    stage_label: str | None = None
    stage_index: int | None = None
    stage_frame_index: int | None = None
    stage_frame_total: int | None = None
    recorded_sample_count: int = 0
    planned_sample_count: int = 0
    collection_position: str | None = None
    initial_alignment_speed_scale: float = 1.0
    pre_pen_down_replay_active: bool = False
    pre_pen_down_replay_normal_episodes: int = 0
    pre_pen_down_normal_completed_count: int = 0
    gripper_sides: tuple[str, ...] = field(default_factory=tuple)
    control_topology: str = "双臂（无夹爪）"
    gripper_control_available: bool = False
    gripper_action_pending: bool = False
    gripper_action_id: int | None = None
    gripper_action_intent: str = ""
    gripper_action_result: str = "idle"
    rule_start_available: bool = False
    writing_reachability_status: str = "disabled"
    writing_reachability_detail: str = ""
    writing_reachability_skipped_count: int = 0
    gripper_action_status: str = ""
    rule_configuration: Mapping[str, Any] = field(default_factory=dict)
    source_selection: Mapping[str, Any] = field(default_factory=dict)
    rule_configuration_batch: tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple
    )
    batch_index: int = 0
    batch_count: int = 1
    batch_completed_count: int = 0
    batch_cycle_complete: bool = False
    configuration_available: bool = False
    configuration_pending: bool = False
    configuration_error: str = ""
    session_state: str = ""
    readiness_reason: str = ""
    readiness_blockers: Mapping[str, str] = field(default_factory=dict)
    collection_mode: str = "rule_based_writing"
    height_calibration_enabled: bool = False
    height_calibrated_z_m: float | None = None
    height_lower_tolerance_m: float = 0.001
    height_upper_tolerance_m: float = 0.001
    height_lower_bound_z_m: float | None = None
    height_upper_bound_z_m: float | None = None
    height_calibration_available: bool = False
    height_calibration_clear_available: bool = False
    height_calibration_pending: bool = False
    height_calibration_error: str = ""


@dataclass(frozen=True)
class _DeferredProgressArmControlContract:
    target_progress_step_id: str
    target_fixed_sides: tuple[str, ...]
    fixed_scope: str


@dataclass(frozen=True)
class _ReplayReferenceCarrier:
    """Lightweight immutable pose reference materialized at its dispatch cut."""

    resampler: MotionTargetResampler
    reference_sequence: int
    reference_timestamp_sec: float
    parent: RuleBasedCollectionEmission
    projected_parent: RobotAction
    directive: RuleBasedCollectionDirective
    next_offset_sec: float
    left_parent_source_index: int
    right_parent_source_index: int
    captures_source_frame: bool
    keyframe_hz: float
    reference_hz: float
    source_kind: str


class RuleBasedRealTeleopExtension:
    """Exclusive rule-plan source attached to one existing RealTeleop runtime."""

    # The dedicated rule UI must remain usable when startup Home or Session
    # arm admission fails.  Binding/starting the extension is command-inert;
    # the normal Session/Rig admission still owns every motion gate.  The
    # runtime uses this explicit opt-in instead of moving the lifecycle of
    # ordinary teleop/HITL extensions.
    bind_before_startup_admission = True
    startup_home_mode = "robot_state_only"

    def __init__(
        self,
        *,
        deployment: RuleBasedRealTeleopDeployment,
        plan: RuleBasedCollectionPlan,
        trajectory_groups: Mapping[str, ReplayTrajectoryGroup],
        config: RuleBasedRealTeleopConfig | None = None,
        source_selection: Mapping[str, Any] | None = None,
        pre_pen_down_replay_plan: RuleBasedCollectionPlan | None = None,
    ) -> None:
        if not isinstance(deployment, RuleBasedRealTeleopDeployment):
            raise TypeError("deployment must be RuleBasedRealTeleopDeployment.")
        if not isinstance(plan, RuleBasedCollectionPlan):
            raise TypeError("plan must be RuleBasedCollectionPlan.")
        if pre_pen_down_replay_plan is not None and not isinstance(
            pre_pen_down_replay_plan,
            RuleBasedCollectionPlan,
        ):
            raise TypeError(
                "pre_pen_down_replay_plan must be RuleBasedCollectionPlan when provided."
            )
        self.deployment = deployment
        self.trajectory_groups = dict(trajectory_groups)
        self.config = config or RuleBasedRealTeleopConfig()
        self.source_selection = copy.deepcopy(dict(source_selection or {}))
        self._selection_ordinal = int(
            self.source_selection.get("selection_ordinal", 0) or 0
        )
        self._normal_plan = plan
        self._pre_pen_down_replay_plan = pre_pen_down_replay_plan
        self._pre_pen_down_normal_completed_count = 0
        self._pre_pen_down_replay_active = False
        self.plan = self._plan_for_next_attempt()
        self._validate_plan(self._normal_plan)
        if self._pre_pen_down_replay_plan is not None:
            self._validate_plan(self._pre_pen_down_replay_plan)
        self._planned_sample_count = self._count_planned_samples()
        self._stage_total_by_step = self._stage_totals_by_step()
        self.runtime_pose_frame_contract = resolve_runtime_pose_frame_contract(
            deployment.collection_pose_frame_contract
        )
        self._boundaries: dict[str, RuleBasedBoundaryState] = {}
        self._boundary_frame_by_step = self._relative_boundary_frames()
        cursor = RuleBasedCollectionCursor(
            plan,
            self.trajectory_groups,
            boundary_provider=self._boundary_for,
        )
        self.adapter = RuleBasedCollectionAdapter(
            cursor,
            resolver=RealTeleopPoseFrameActionResolver(
                collection_contract=deployment.collection_pose_frame_contract,
                runtime_contract=self.runtime_pose_frame_contract,
            ),
        )
        rates = [
            float(group.action_hz)
            for group in self.trajectory_groups.values()
            if np.isfinite(float(group.action_hz)) and float(group.action_hz) > 0.0
        ]
        self.action_hz = float(
            self.config.action_hz
            if self.config.action_hz is not None
            else np.median(rates) if rates else 15.0
        )
        self._components: object | None = None
        self._clutch_input: object | None = None
        self._dataset_recorder_provider: Callable[[], object | None] | None = None
        self._recording_active: Callable[[], bool] | None = None
        self._collection_transition_active: Callable[[], bool] | None = None
        self._control_ui: object | None = None
        self._legacy_generic_ui = False
        self._initial_alignment_planner: RuleBasedInitialAlignmentPlanner | None = None
        self._initial_alignment_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rule-initial-alignment",
        )
        self._initial_alignment_future: Future[RuleBasedInitialAlignmentPlan] | None = None
        self._initial_alignment_cancel_pending = False
        self._initial_alignment_planning_started_at: float | None = None
        self._initial_alignment_template: RuleBasedCollectionEmission | None = None
        self._initial_alignment_plan: RuleBasedInitialAlignmentPlan | None = None
        self._initial_alignment_waypoint_index = 0
        self._initial_alignment_effective_waypoint_count = 0
        self._initial_alignment_started_at: float | None = None
        self._initial_alignment_last_schedule_lag_sec = 0.0
        self._initial_alignment_cumulative_schedule_shift_sec = 0.0
        self._initial_alignment_skipped_clock_ticks = 0
        self._initial_alignment_reacquire_samples = 0
        self._initial_alignment_hand_start_by_side: dict[str, np.ndarray] = {}
        self._initial_alignment_hand_target_by_side: dict[str, np.ndarray] = {}
        self._coupled_pose_solver_seeded = False
        # Reuse the same source-interval motion classifier as the production
        # MuJoCo/replay position path.  The classifier gives the existing
        # V2-Balanced post-IK smoother physical 15 Hz action-speed evidence;
        # without it every rule target falls back to the sticky 0.14 alpha.
        self._pose_action_motion_intent = {
            side: PoseActionMotionIntentStateMachine()
            for side in ("left", "right")
        }
        self._pose_action_motion_intent_step_id: str | None = None
        self._previous_command_guard: Callable[..., Any] | None = None
        self._deferred_progress_arm_contract: (
            _DeferredProgressArmControlContract | None
        ) = None
        self._deferred_progress_arm_fix_established = False
        self._filtered_initial_alignment_arm_by_side: dict[
            str, EffectorCommand
        ] = {}
        self._filtered_initial_alignment_identity: tuple[str, int, int] | None = None
        # The DMTAC state carrier is a six-slot legacy hand vector, with the
        # active scalar fixed at the backend's configured command dimension
        # (slot 0 for the real DMTAC backend).  Keep this topology from the
        # connected backend rather than trusting stale/incorrect telemetry
        # metadata when normalizing scalar-only snapshots.
        self._hand_command_dim_by_side: dict[str, int] = {}
        self._hand_max_action_by_side: dict[str, float] = {}
        self._current: RuleBasedCollectionEmission | None = None
        # Replay keyframes remain the dataset/provenance cadence.  Arm motion
        # between them is emitted on the shared position-reference clock, using
        # robot_motion's production resampler rather than a rule-only
        # interpolation path.
        self._replay_reference_queue: deque[_ReplayReferenceCarrier] = deque()
        self._replay_reference_stage_started_at: float | None = None
        self._replay_reference_pause_started_at: float | None = None
        self._replay_reference_last_schedule_lag_sec = 0.0
        self._replay_reference_cumulative_schedule_shift_sec = 0.0
        self._replay_reference_rebased_tick_count = 0
        self._frame_emission: RuleBasedCollectionEmission | None = None
        # A writer-pause frame still dispatches a certified arm hold, but it
        # must remain invisible to cursor/arrival/sample advancement.  Keep a
        # separate marker so command guards and dispatch receipts stay
        # fail-closed without treating the hold as a recordable rule frame.
        self._paused_hold_emission: RuleBasedCollectionEmission | None = None
        self._last_dispatched_emission: RuleBasedCollectionEmission | None = None
        self._target_wait_started_at: float | None = None
        self._next_due_at: float | None = None
        self._current_dispatched = False
        self._current_sample_accepted = False
        self._dispatch_retry_started_at: float | None = None
        self._writer_retry_started_at: float | None = None
        self._target_arrival_confirmed = False
        self._target_arrival_snapshots: dict[str, dict[str, Any]] = {}
        self._last_live_dispatch_timestamp: float | None = None
        self._settled_samples = 0
        self._started = False
        self._armed = False
        self._execution_begun = False
        self._completed = False
        self._completion_stop_requested = False
        self._aborted = False
        self._held = False
        self._restart_after_hold_requested = False
        self._closed = False
        self._waiting_reason = "runtime_not_started"
        self._stage_label: str | None = None
        self._stage_index: int | None = None
        self._stage_frame_index: int | None = None
        self._stage_frame_total: int | None = None
        self._collection_position: str | None = None
        self._gripper_action_status = ""
        self._gripper_action_sequence = 0
        self._gripper_action_id: int | None = None
        self._gripper_action_intent = ""
        self._gripper_action_result = "idle"
        self._pending_gripper_action: str | None = None
        self._manual_gripper_started_at: float | None = None
        self._manual_gripper_last_command_by_side: dict[str, float] = {}
        self._manual_gripper_command_baseline_by_side: dict[str, float] = {}
        self._manual_gripper_last_dispatch_at_by_side: dict[str, float] = {}
        self._manual_gripper_last_progress_at_by_side: dict[str, float] = {}
        self._manual_gripper_settle_samples_by_side: dict[str, int] = {}
        self._manual_gripper_last_settle_feedback_token_by_side: dict[
            str, tuple[str, float]
        ] = {}
        self._admitted_capture_sequences: set[int] = set()
        self._accepted_samples_by_detail_stage: dict[str, int] = {}
        # Resampled pose stages execute on the 60 Hz reference clock while
        # formal samples remain owned by the lower-rate dataset clock.  Keep
        # the resulting writer-rate plan per step so stage coverage never
        # compares writer ACKs with the raw keyframe count.
        self._writer_rate_expected_samples_by_step: dict[str, int] = {}
        self._height_calibration: RuleBasedWritingHeightCalibration | None = None
        self._height_calibration_error = ""
        self._height_calibration_pending = False
        self._height_calibration_future: Future[Mapping[str, Any]] | None = None
        self._height_calibration_request: dict[str, float] | None = None
        self._height_lower_tolerance_m = 0.001
        self._height_upper_tolerance_m = 0.001
        self._writing_height_target_z_m: float | None = None
        # Height calibration is deliberately performed after the operator may
        # have moved either arm by hand.  The next initial-alignment planner
        # therefore has to rebase the position transport from the same fresh
        # dual-arm feedback cut it uses as waypoint zero.  This is a one-shot
        # pre-dispatch operation; runtime Home remains the only path allowed to
        # clear an emergency-stop latch.
        self._manual_adjustment_motion_rebase_pending = False
        self._height_descent_planned_count_added = False
        self._height_descent_planned_sample_count = 0
        self._configuration_options: dict[str, Any] = {}
        self._selected_configuration: dict[str, Any] = {}
        self._configuration_batch: list[
            tuple[dict[str, Any], "RuleBasedRealTeleopExtension"]
        ] = []
        self._configuration_batch_index = 0
        self._configuration_batch_completed_count = 0
        self._configuration_batch_cycle_complete = False
        self._configuration_rebuilder: (
            Callable[[Mapping[str, Any]], "RuleBasedRealTeleopExtension"] | None
        ) = None
        self._configuration_on_commit: (
            Callable[
                [Mapping[str, Any], "RuleBasedRealTeleopExtension"],
                None,
            ]
            | None
        ) = None
        self._configuration_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rule-plan-configuration",
        )
        self._configuration_future: Future[
            tuple[tuple[dict[str, Any], "RuleBasedRealTeleopExtension"], ...]
        ] | None = None
        self._configuration_request: dict[str, Any] | None = None
        self._next_attempt_rebuild_future: Future[
            "RuleBasedRealTeleopExtension"
        ] | None = None
        self._next_attempt_rebuild_context: dict[str, Any] | None = None
        self._writing_reachability_future: Future[Mapping[str, Any]] | None = None
        self._writing_reachability_rebuild_future: Future[
            "RuleBasedRealTeleopExtension"
        ] | None = None
        self._writing_reachability_rebuild_ordinal: int | None = None
        self._writing_reachability_fingerprint: tuple[object, ...] | None = None
        self._writing_reachability_status = "disabled"
        self._writing_reachability_detail = ""
        self._writing_reachability_skipped_count = 0
        self._writing_reachability_progress: (
            SimpleQueue[tuple[str, str, int, int, float]] | None
        ) = None
        self._writing_reachability_progress_detail = ""
        self._writing_reachability_started_at: float | None = None
        self._writing_reachability_candidate_label = ""
        self._writing_reachability_preparation_ms: float | None = None
        self._configuration_error = ""
        self._arm_feedback_pause_started_at: float | None = None
        # Readiness reports may perform a bounded serial telemetry read (the
        # DMTAC backend does so to include its vendor error/status fields).
        # ``status()`` is published from the realtime loop, so do not repeat
        # that I/O at 60 Hz while a backend is faulted.  The cache is
        # invalidated explicitly by lifecycle retry/Home edges.
        self._readiness_diagnosis_cache: tuple[
            tuple[dict[str, str], str], tuple[object, ...]
        ] | None = None
        self._readiness_diagnosis_cache_expires_at = 0.0
        self._progress_scope_admission_error = ""
        self._last_status_publish_at: float | None = None
        self._last_status_publish_urgent_signature: tuple[object, ...] | None = None

    def configure_prestart_rebuild(
        self,
        *,
        options: Mapping[str, Any],
        selected: Mapping[str, Any],
        rebuilder: Callable[[Mapping[str, Any]], "RuleBasedRealTeleopExtension"],
        on_commit: (
            Callable[
                [Mapping[str, Any], "RuleBasedRealTeleopExtension"],
                None,
            ]
            | None
        ) = None,
    ) -> None:
        """Attach application choices without transferring runtime authority."""

        if self._components is not None or self._started:
            raise RuntimeError("rule configuration support must be attached before runtime.")
        if not callable(rebuilder):
            raise TypeError("rule configuration rebuilder must be callable.")
        if on_commit is not None and not callable(on_commit):
            raise TypeError("rule configuration commit callback must be callable.")
        self._configuration_options = copy.deepcopy(dict(options))
        self._selected_configuration = copy.deepcopy(dict(selected))
        self._configuration_rebuilder = rebuilder
        self._configuration_on_commit = on_commit

    def _count_planned_samples(self) -> int:
        total = 0
        for step in self.plan.steps:
            if not bool(step.capture.enabled):
                continue
            if isinstance(step, RuleBasedReplay):
                total += max(0, int(step.end_index) - int(step.start_index))
            elif isinstance(step, RuleBasedActionSegment):
                total += int(
                    step.stop_condition.max_steps
                    if step.planned_steps_hint is None
                    else step.planned_steps_hint
                )
            else:
                # Fixed/custom steps are not part of the current writing
                # profile, but retaining a bounded estimate keeps status
                # truthful for future plan variants.
                repeats = int(getattr(step, "repeats", 1))
                total += max(1, repeats)
        return int(total)

    def _stage_totals_by_step(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for step in self.plan.steps:
            if isinstance(step, RuleBasedReplay):
                totals[step.step_id] = max(0, int(step.end_index) - int(step.start_index))
            elif isinstance(step, RuleBasedActionSegment):
                totals[step.step_id] = int(
                    step.stop_condition.max_steps
                    if step.planned_steps_hint is None
                    else step.planned_steps_hint
                )
            else:
                totals[step.step_id] = int(getattr(step, "repeats", 1))
        return totals

    def _expected_samples_by_detail_stage(self) -> dict[str, int]:
        """Return writer-rate expectations for each persisted detail stage."""

        expected: dict[str, int] = {}
        collection_mode = str(
            self.source_selection.get("collection_mode") or "rule_based_writing"
        )
        for step in self.plan.steps:
            if not bool(step.capture.enabled):
                continue
            metadata = self._phase_metadata_for_step(
                step.step_id,
                "replay" if isinstance(step, RuleBasedReplay) else "custom_action",
                collection_mode,
            )
            stage_id = metadata.get("collection_detail_stage_id")
            if stage_id not in {
                "right_write.pen_down",
                "right_write.write",
                "right_write.pen_up",
                "execute_task.full_replay",
            }:
                continue
            if isinstance(step, RuleBasedReplay):
                count = max(0, int(step.end_index) - int(step.start_index))
            elif isinstance(step, RuleBasedActionSegment):
                count = int(
                    step.stop_condition.max_steps
                    if step.planned_steps_hint is None
                    else step.planned_steps_hint
                )
            else:
                count = max(1, int(getattr(step, "repeats", 1)))
            count = self._writer_rate_expected_samples_by_step.get(
                step.step_id,
                count,
            )
            expected[str(stage_id)] = expected.get(str(stage_id), 0) + count
        return expected

    def _stage_sample_coverage(self) -> dict[str, dict[str, int | float | bool]]:
        expected = self._expected_samples_by_detail_stage()
        result: dict[str, dict[str, int | float | bool]] = {}
        for stage_id, planned in expected.items():
            required = min(
                planned,
                max(
                    1,
                    int(self.config.minimum_stage_sample_count),
                    int(math.ceil(planned * self.config.minimum_stage_sample_coverage)),
                ),
            )
            accepted = int(self._accepted_samples_by_detail_stage.get(stage_id, 0))
            result[stage_id] = {
                "planned": int(planned),
                "required": int(required),
                "accepted": accepted,
                "complete": accepted >= required,
            }
        return result

    def _incomplete_detail_stages(self) -> dict[str, dict[str, int | float | bool]]:
        return {
            stage_id: summary
            for stage_id, summary in self._stage_sample_coverage().items()
            if not bool(summary["complete"])
        }

    def _validate_plan(self, plan: RuleBasedCollectionPlan | None = None) -> None:
        steps = tuple((self.plan if plan is None else plan).steps)
        for index, step in enumerate(steps):
            if step.capture.stream_ids:
                raise ValueError(
                    "RealTeleop formal samples are atomic; per-stream rule capture "
                    f"is unsupported for step {step.step_id!r}."
                )
            if isinstance(step, RuleBasedReplay) and step.relative_to is not None:
                if index == 0 or not isinstance(steps[index - 1], RuleBasedActionSegment):
                    raise ValueError(
                        "first real relative replay requires an immediately preceding "
                        "bounded runtime action segment."
                    )
                if step.relative_to != steps[index - 1].step_id:
                    raise ValueError(
                        "relative_to must name the immediately preceding runtime "
                        "segment step_id for measured-boundary publication."
                    )

    def _relative_boundary_frames(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for step in self.plan.steps:
            if not isinstance(step, RuleBasedReplay) or step.relative_to is None:
                continue
            frame = replay_action_pose_frame(self.trajectory_groups[step.group_key])
            previous = result.get(step.relative_to)
            if previous is not None and previous != frame:
                raise ValueError(
                    f"boundary {step.relative_to!r} is requested in conflicting frames."
                )
            result[step.relative_to] = frame
        return result

    def _plan_for_next_attempt(self) -> RuleBasedCollectionPlan:
        interval = int(self.config.pre_pen_down_replay_normal_episodes)
        use_prelude = bool(
            self._pre_pen_down_replay_plan is not None
            and (
                interval == 0
                or self._pre_pen_down_normal_completed_count >= interval
            )
        )
        self._pre_pen_down_replay_active = use_prelude
        return (
            self._pre_pen_down_replay_plan
            if use_prelude and self._pre_pen_down_replay_plan is not None
            else self._normal_plan
        )

    def _advance_pre_pen_down_replay_cadence(self) -> None:
        """Count only a fully committed successful Episode."""

        if self._pre_pen_down_replay_plan is None:
            self._pre_pen_down_normal_completed_count = 0
            return
        if self._pre_pen_down_replay_active:
            self._pre_pen_down_normal_completed_count = 0
        else:
            self._pre_pen_down_normal_completed_count += 1

    def _pre_pen_down_replay_schedule_metadata(self) -> dict[str, Any]:
        return {
            "configured": self._pre_pen_down_replay_plan is not None,
            "active_for_episode": bool(self._pre_pen_down_replay_active),
            "normal_episodes_between_replays": int(
                self.config.pre_pen_down_replay_normal_episodes
            ),
            "normal_successful_episodes_since_replay": int(
                self._pre_pen_down_normal_completed_count
            ),
        }

    def attach_components(self, components: object) -> object:
        if set(getattr(components, "active_sides", ())) != {"left", "right"}:
            raise RuntimeError("rule-based runtime requires coupled dual-arm components.")
        if not bool(getattr(components, "collision_guard_requested", False)):
            raise RuntimeError("rule-based runtime requires collision guarding.")
        retargeter = getattr(components, "dual_arm_retargeter", None)
        if retargeter is None:
            raise RuntimeError("rule-based runtime requires the coupled target solver.")
        target_filter = getattr(retargeter, "pre_ik_target_safety_filter", None)
        if target_filter is None or not bool(
            getattr(target_filter, "apply_to_target", False)
        ):
            raise RuntimeError("rule-based runtime requires an Active final TCP envelope.")
        profile = getattr(target_filter, "profile", None)
        if set(getattr(profile, "enabled_sides", ())) != {"left", "right"}:
            raise RuntimeError("Active TCP envelope must cover both arms.")
        for side in self.config.gripper_sides:
            backend = getattr(components, f"{side}_hand_backend", None)
            if backend is None or not callable(getattr(backend, "send_command", None)):
                raise RuntimeError(
                    f"rule-based gripper replay requires a connected {side} DMTAC backend."
                )
            if not callable(getattr(backend, "get_state", None)):
                raise RuntimeError(
                    f"rule-based gripper replay requires {side} hand telemetry."
                )
            backend_config = getattr(backend, "config", None)
            try:
                command_dim = int(getattr(backend_config, "command_dim", 0))
                max_action = float(getattr(backend_config, "max_action", 1.0))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"rule-based gripper replay requires valid {side} DMTAC configuration."
                ) from exc
            if command_dim < 0 or command_dim >= 6:
                raise RuntimeError(
                    f"rule-based gripper replay has invalid {side} command dimension."
                )
            if command_dim != 0:
                raise RuntimeError(
                    f"rule-based DMTAC replay requires {side} command dimension 0; "
                    f"got {command_dim}."
                )
            if not np.isfinite(max_action) or max_action <= 0.0:
                raise RuntimeError(
                    f"rule-based gripper replay has invalid {side} max action."
                )
            self._hand_command_dim_by_side[side] = command_dim
            self._hand_max_action_by_side[side] = max_action
        session = components.session
        # Rule replay emits canonical ``<side>_hand`` effectors directly; the
        # Session Rig must therefore own the exact backend object even when a
        # compatibility/component-construction path forgot to populate its
        # mutable backend map.  Restore only a missing binding (never replace
        # a different owner) so a transient startup failure remains
        # fail-closed but can be recovered without a dispatch-time KeyError.
        rig = getattr(session, "rig", None)
        rig_backends = getattr(rig, "backends", None)
        if isinstance(rig_backends, dict):
            for side in self.config.gripper_sides:
                effector_name = f"{side}_hand"
                backend = getattr(components, f"{side}_hand_backend", None)
                existing = rig_backends.get(effector_name)
                if existing is None:
                    rig_backends[effector_name] = backend
                elif existing is not backend:
                    raise RuntimeError(
                        f"Session Rig already owns a different backend for "
                        f"{effector_name}"
                    )

        if session.command_override_provider is not None:
            raise RuntimeError("RealTeleop already has a command override provider.")
        if session.human_command_selection_provider is not None:
            raise RuntimeError("rule-based runtime cannot share a source selector.")
        self._initial_alignment_planner = self._make_initial_alignment_planner(components)
        self._previous_command_guard = getattr(session, "command_guard", None)

        def rule_command_guard(command, observation, now):
            if self._previous_command_guard is not None:
                fault = self._previous_command_guard(command, observation, now)
                if fault is not None:
                    return fault
            fault = self._filtered_initial_alignment_command_fault(
                command,
                now=float(now),
            )
            if fault is not None:
                return fault
            assert self._initial_alignment_planner is not None
            fault = rule_based_initial_alignment_command_fault(
                command,
                planner=self._initial_alignment_planner,
                expected_plan_hash=(
                    None
                    if self._initial_alignment_plan is None
                    else self._initial_alignment_plan.semantic_content_hash
                ),
                now=float(now),
            )
            if fault is not None:
                return fault
            fault = rule_based_coupled_pose_command_fault(
                command,
                planner=self._initial_alignment_planner,
                now=float(now),
            )
            if fault is not None:
                return fault
            if self._frame_emission is None and self._paused_hold_emission is None:
                return None
            try:
                validate_collection_command_action(
                    command,
                    allow_backend_target_pose_defer=True,
                )
            except (TypeError, ValueError) as exc:
                return FaultEvent(
                    code="rule_based_collection_action_contract_invalid",
                    message=(
                        "final rule command cannot be represented by the formal "
                        f"collection action contract: {type(exc).__name__}: {exc}"
                    ),
                    severity=FaultSeverity.ERROR,
                    timestamp=float(now),
                    component="rule_based_real_teleop",
                    metadata={"reason": "canonical_action_encoding_failed"},
                )
            return None

        session.command_guard = rule_command_guard
        session.command_override_provider = self.command_for_observation
        self._components = components
        self._invalidate_readiness_diagnosis_cache()
        return components

    def _invalidate_readiness_diagnosis_cache(self) -> None:
        self._readiness_diagnosis_cache = None
        self._readiness_diagnosis_cache_expires_at = 0.0

    def refresh_status(self, *, force: bool = False) -> None:
        """Republish the current rule status after an external lifecycle edge."""

        if force:
            self._invalidate_readiness_diagnosis_cache()
        self._publish_status(force=force)

    def set_progress_scope_admission(self, *, error: str | None = None) -> None:
        """Publish one cached progress/FixArm compatibility result.

        The control loop performs the authoritative complete-profile check at
        its lifecycle boundary.  Keeping the resulting error here makes the
        dedicated rule UI disable ``开始规则采集`` immediately, without
        re-running profile scans or throwing the same mismatch from every
        realtime transition frame.
        """

        normalized = str(error or "").strip()
        if normalized == self._progress_scope_admission_error:
            return
        self._progress_scope_admission_error = normalized
        self._invalidate_readiness_diagnosis_cache()
        self._publish_status()

    def _filtered_initial_alignment_command_fault(
        self,
        command: TeleopCommand,
        *,
        now: float,
    ) -> FaultEvent | None:
        """Reject any downstream rewrite of a certified initialization qref."""

        emission = self._frame_emission or self._paused_hold_emission
        if emission is None or emission.directive.kind != "initialize":
            return None
        expected_metadata = emission.command.metadata
        actual_metadata = (
            command.metadata if isinstance(command.metadata, Mapping) else {}
        )
        try:
            expected_identity = (
                str(expected_metadata["rule_based_initial_alignment_plan_hash"]),
                int(expected_metadata["rule_based_initial_alignment_waypoint_index"]),
                int(expected_metadata["rule_based_initial_alignment_waypoint_count"]),
            )
            actual_identity = (
                str(actual_metadata["rule_based_initial_alignment_plan_hash"]),
                int(actual_metadata["rule_based_initial_alignment_waypoint_index"]),
                int(actual_metadata["rule_based_initial_alignment_waypoint_count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return FaultEvent(
                code="rule_based_initial_alignment_command_rewritten",
                message=(
                    "filtered initialization command lost its certified waypoint "
                    f"identity: {type(exc).__name__}: {exc}"
                ),
                severity=FaultSeverity.ERROR,
                timestamp=float(now),
                component="rule_based_real_teleop",
                metadata={"reason": "certified_waypoint_identity_missing"},
            )
        mismatch: dict[str, object] = {}
        if actual_identity != expected_identity:
            mismatch["identity"] = {
                "expected": list(expected_identity),
                "actual": list(actual_identity),
            }
        template = self._initial_alignment_template
        actively_aligned_sides = set(
            ()
            if template is None
            else template.directive.qpos_by_side
        )
        verified: dict[str, EffectorCommand] = {}
        for side in ("left", "right"):
            name = f"{side}_arm"
            expected = emission.command.commands.get(name)
            actual = command.commands.get(name)
            if expected is None or actual is None:
                mismatch[name] = "missing_expected_or_filtered_command"
                continue
            if (
                actual.effector_name != name
                or actual.side != side
                or actual.command_type != "joint_vector"
            ):
                mismatch[name] = {
                    "reason": "invalid_filtered_effector_contract",
                    "effector_name": actual.effector_name,
                    "side": actual.side,
                    "command_type": actual.command_type,
                }
                continue
            try:
                expected_qpos = np.asarray(expected.value, dtype=float).reshape(-1)
                actual_qpos = np.asarray(actual.value, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                mismatch[name] = "non_numeric_qpos"
                continue
            if (
                expected_qpos.shape != (6,)
                or actual_qpos.shape != (6,)
                or not np.all(np.isfinite(expected_qpos))
                or not np.all(np.isfinite(actual_qpos))
            ):
                mismatch[name] = "non_finite_or_invalid_qpos"
                continue
            max_error = float(np.max(np.abs(actual_qpos - expected_qpos)))
            if max_error > 1e-10:
                mismatch[name] = {
                    "reason": "filtered_qpos_differs_from_certified_waypoint",
                    "max_error_rad": max_error,
                }
                continue
            fix_arm_active = bool(
                getattr(actual, "metadata", {}).get("fix_arm_active", False)
            )
            if fix_arm_active and side in actively_aligned_sides:
                mismatch[name] = "fix_arm_rewrote_initialization_command"
                continue
            verified[side] = actual.copy()
        if mismatch:
            self._filtered_initial_alignment_arm_by_side = {}
            self._filtered_initial_alignment_identity = None
            return FaultEvent(
                code="rule_based_initial_alignment_command_rewritten",
                message=(
                    "a Session command filter changed or replaced the certified "
                    "dual-arm initialization waypoint"
                ),
                severity=FaultSeverity.ERROR,
                timestamp=float(now),
                component="rule_based_real_teleop",
                metadata={
                    "reason": "filtered_command_mismatch",
                    "mismatch": mismatch,
                },
            )
        self._filtered_initial_alignment_arm_by_side = verified
        self._filtered_initial_alignment_identity = expected_identity
        return None

    def bind_runtime(
        self,
        *,
        components: object,
        clutch_input: object,
        camera_sources: Mapping[str, object],
        dataset_recorder: object | None,
        dataset_recorder_provider: Callable[[], object | None],
        episode_recorder: object,
        control_ui: object | None,
        recording_active: Callable[[], bool],
        pause_recording: Callable[..., bool],
        collection_transition_active: Callable[[], bool] | None = None,
    ) -> None:
        del camera_sources, dataset_recorder, episode_recorder, pause_recording
        if components is not self._components:
            raise RuntimeError("RealTeleop did not retain rule-based components.")
        if not callable(dataset_recorder_provider) or not callable(recording_active):
            raise TypeError("rule-based runtime requires recorder state providers.")
        self._clutch_input = clutch_input
        self._dataset_recorder_provider = dataset_recorder_provider
        self._recording_active = recording_active
        self._collection_transition_active = collection_transition_active
        self._control_ui = control_ui
        bind_rule_runtime = getattr(control_ui, "bind_rule_runtime", None)
        if not callable(bind_rule_runtime):
            if not (
                callable(getattr(control_ui, "consume_events", None))
                and callable(getattr(control_ui, "publish", None))
            ):
                raise RuntimeError(
                    "rule-based runtime requires either the dedicated rule UI "
                    "or the established generic real-teleop UI contract."
                )
            # The generic entrypoint/runtime files are owned by the separate
            # episode integration task. Preserve the current local launcher's
            # Episode-driven admission until that owner promotes the dedicated
            # UI hooks; this path still uses the same Session, Rig, writer, and
            # command-override authority.
            self._legacy_generic_ui = True
        else:
            self._legacy_generic_ui = False
            bind_rule_runtime()
        configure_rule_options = getattr(control_ui, "configure_rule_options", None)
        if self._configuration_rebuilder is not None and not self._legacy_generic_ui:
            if not callable(configure_rule_options):
                raise RuntimeError(
                    "rule-based runtime requires a UI plan-configuration port."
                )
            configure_rule_options(
                self._configuration_options,
                self._selected_configuration,
            )

    def start(self, *, components: object) -> None:
        if components is not self._components or self._dataset_recorder_provider is None:
            raise RuntimeError("rule-based runtime is not fully bound.")
        self._started = True
        self._armed = bool(self._legacy_generic_ui)
        self._waiting_reason = (
            "waiting_for_formal_episode"
            if self._legacy_generic_ui
            else "waiting_for_rule_start"
        )
        self._publish_status()

    def allows_pre_episode_armed_warmup(self) -> bool:
        """Allow writer-open readiness before the first rule motion command.

        The Session permits an extension command in ARMED and transitions to
        RUNNING only after that command is dispatched. A rule command remains
        unavailable until the formal Episode is active, so this narrow
        pre-open exception breaks that ordering cycle without authorizing
        motion or samples early.
        """

        return bool(
            self._started
            and self._armed
            and not self._execution_begun
            and not self._completed
            and not self._aborted
            and not self._held
            and self._right_gripper_hold_owner_ready()[0]
        )

    def _right_gripper_hold_owner_ready(self) -> tuple[bool, str]:
        """Return the command-inert receipt required by writing hand holds.

        A writing frame carries a right-hand action for the dataset contract,
        but its physical hand command is deliberately preserved.  The DMTAC
        backend must therefore have already accepted the task-chain close in
        one-shot mode.  This check is deliberately local and has no serial
        side effects.  A backend that cannot provide this receipt is rejected
        for writing rather than being allowed to guess that its physical grip
        is owned.
        """

        if (
            str(self.source_selection.get("collection_mode") or "")
            != "rule_based_writing"
            or "right" not in self.config.gripper_sides
        ):
            return True, ""
        backend = getattr(self._components, "right_hand_backend", None)
        if backend is None:
            return False, "右夹爪 backend 不可用。"
        is_ready = getattr(backend, "is_ready", None)
        if callable(is_ready):
            try:
                if not bool(is_ready()):
                    # The normal readiness diagnosis owns transport/fault
                    # errors.  Do not replace those precise blockers with a
                    # secondary hold-owner message.
                    return True, ""
            except Exception:
                return True, ""
        receipt = getattr(backend, "fixed_closed_one_shot_ready", None)
        if not callable(receipt):
            return False, "右夹爪 backend 未提供固定闭合 owner receipt。"
        try:
            ready = bool(receipt())
        except Exception as exc:  # noqa: BLE001 - admission fails closed.
            return False, f"右夹爪固定闭合 owner 检查失败：{type(exc).__name__}: {exc}。"
        if ready:
            return True, ""
        return (
            False,
            "右夹爪尚未建立书写用的一次性闭合 owner；"
            "请先完成“闭紧右夹爪”并等待“实测已闭紧”。",
        )

    def _rule_frame_requires_right_gripper_hold_owner(
        self,
        emission: RuleBasedCollectionEmission,
    ) -> bool:
        """Return whether this frame would use the command-inert grip hold."""

        return bool(
            str(self.source_selection.get("collection_mode") or "")
            == "rule_based_writing"
            and "right" in self.config.gripper_sides
            and "right_hand" in emission.command.commands
        )

    def collection_startup_blockers(self) -> dict[str, str]:
        """Expose rule-owned readiness before the formal writer is opened.

        The generic device retry runtime cannot infer an exclusive rule plan's
        required hand effectors.  Reuse the existing diagnostic path here so
        a configured right DMTAC fault is rejected before writer preparation,
        while Q6/input remains unrelated when the rule source disabled it.
        """

        status = self.status()
        return {
            str(key): str(value)
            for key, value in status.readiness_blockers.items()
            if str(value).strip()
        }

    def allows_active_episode_bootstrap_without_dispatch(self) -> bool:
        """Identify the bounded pre-first-dispatch phase of an active Episode.

        The production writing plan starts with an asynchronous, command-inert
        initial-alignment certification.  Its first qref is the exact
        command-inert runtime-Home waypoint, and neither planning nor that
        waypoint is allowed to admit a dataset sample.  Once any arm command
        is actually dispatched this exception is permanently disabled.
        """

        provider = self._dataset_recorder_provider
        recorder = None if provider is None else provider()
        first_step = self.plan.steps[0] if self.plan.steps else None
        return bool(
            self._started
            and self._armed
            and not self._execution_begun
            and not self._completed
            and not self._aborted
            and not self._held
            and self._configuration_future is None
            and isinstance(first_step, RuleBasedInitialize)
            and self._recording_active is not None
            and self._recording_active()
            and recorder is not None
            and bool(getattr(recorder, "is_recording", False))
        )

    def defers_progress_entry_arm_control(
        self,
        *,
        target_progress_step_id: str,
        current_fixed_sides: tuple[str, ...],
        target_fixed_sides: tuple[str, ...],
        fixed_scope: str,
    ) -> bool:
        """Delay the rule profile's arm-only Fix until initialization arrives."""

        first_step = self.plan.steps[0] if self.plan.steps else None
        normalized_current = tuple(
            side for side in ("left", "right") if side in set(current_fixed_sides)
        )
        normalized_target = tuple(
            side for side in ("left", "right") if side in set(target_fixed_sides)
        )
        candidate = _DeferredProgressArmControlContract(
            target_progress_step_id=str(target_progress_step_id),
            target_fixed_sides=normalized_target,
            fixed_scope=str(fixed_scope),
        )
        eligible = bool(
            isinstance(first_step, RuleBasedInitialize)
            and len(normalized_target) == 1
            and normalized_target[0] in self.deployment.active_sides
            and str(fixed_scope) == "arm_only"
            and not self._completed
            and not self._aborted
            and not self._held
            and not self._closed
        )
        existing = self._deferred_progress_arm_contract
        if existing is not None and candidate != existing:
            raise RuntimeError(
                "Deferred rule progress arm-control contract changed before "
                f"completion: expected={existing!r} actual={candidate!r}."
            )
        if not eligible:
            return False
        controller = getattr(self._components, "fix_arm_controller", None)
        if controller is None:
            raise RuntimeError(
                "Deferred rule progress arm control requires the FixArm controller."
            )
        scope_for_side = getattr(controller, "scope_for_side", None)
        if not callable(scope_for_side):
            raise RuntimeError("FixArm controller does not expose side scope.")
        actual_scope = str(scope_for_side(normalized_target[0]))
        if actual_scope != candidate.fixed_scope:
            raise RuntimeError(
                "Deferred rule progress Fix scope mismatch: "
                f"expected={candidate.fixed_scope!r} actual={actual_scope!r}."
            )
        if existing is None:
            self._deferred_progress_arm_contract = candidate
        if self._deferred_progress_arm_fix_established:
            return False
        if normalized_current:
            raise RuntimeError(
                "Unexpected FixArm authority appeared during rule initialization: "
                f"fixed_sides={normalized_current!r}."
            )
        current = self._current
        if current is not None and current.directive.kind != "initialize":
            raise RuntimeError(
                "Rule execution reached a recordable phase before the deferred "
                "progress Fix was established."
            )
        return True

    def _abort_pre_dispatch_readiness_loss(self) -> bool:
        """Fail closed when a started rule attempt loses hardware admission.

        The dedicated rule UI intentionally permits a short ARMED window so
        the formal writer can open before the first rule arm command. That
        exception must not leave an attempt armed after a hand/Rig fault: the
        next countdown tick would otherwise report only the generic missing
        dispatch receipt and strand the task chain. Once any command has been
        dispatched, the normal Session/Rule failure path owns the fault.
        """

        if (
            not self._armed
            or self._execution_begun
            or self._completed
            or self._aborted
            or self._held
        ):
            return False
        session = getattr(self._components, "session", None)
        session_state = getattr(getattr(session, "state_machine", None), "state", None)
        session_state_value = str(
            getattr(session_state, "value", session_state) or ""
        ).upper()
        blockers, readiness_reason = self._session_readiness_diagnosis(
            session_state_value=session_state_value,
        )
        if not blockers:
            return False
        detail = "; ".join(
            f"{key}: {value}" for key, value in sorted(blockers.items())
        )
        if not detail:
            detail = readiness_reason or "Rig is not ready"
        message = (
            "规则采集在首条机械臂命令派发前被安全取消："
            + detail
            + "。请先恢复设备并重新完成 runtime Home。"
        )
        self._aborted = True
        self._armed = False
        self._waiting_reason = "session_readiness_failure"
        self._clear_current()
        self._clear_replay_reference_schedule()
        self._cancel_initial_alignment_planning()
        self._establish_hold(message)
        self._request_failure_stop()
        append_message = getattr(self._control_ui, "append_message", None)
        if callable(append_message):
            append_message("RuleBased", message)
        return True

    def after_control_input_poll(self, *, now: float) -> None:
        self._frame_emission = None
        self._paused_hold_emission = None
        self._settle_cancelled_initial_alignment()
        if self._completed:
            self._prepare_next_attempt_after_success(float(now))
        # A Hold may race an asynchronous Episode-open/stop transaction.  The
        # original stop intent is consumed by the collection control plane,
        # but an in-flight open can become active only after that attempt.  Do
        # not strand the cancelled rule attempt in that state: once the
        # transition settles, ask the existing writer owner to discard it.
        # ``request_rule_failure_stop`` is internally one-shot, so this poll
        # remains passive while the request is pending.
        if (
            self._held
            and not self._completed
            and not self._aborted
            and not self._collection_transition_pending()
            and self._recording_active is not None
            and self._recording_active()
        ):
            self._request_failure_stop()
        if self._poll_prestart_configuration(float(now)):
            self._publish_status()
            return
        self._process_height_calibration_requests(float(now))
        self._service_writing_reachability_preflight()
        consume_rule_intents = getattr(self._control_ui, "consume_rule_intents", None)
        if not callable(consume_rule_intents):
            raise RuntimeError("dedicated rule-based UI lost its intent boundary.")
        for intent in consume_rule_intents():
            if intent == "rule_start":
                if (
                    self._held
                    or self._execution_begun
                    or self._completed
                    or self._aborted
                    or self._height_calibration_future is not None
                    or self._height_calibration_pending
                    or (
                        self._writing_reachability_enabled()
                        and not self._writing_reachability_passed_for_current_selection()
                    )
                ):
                    continue
                fix_controller = getattr(self._components, "fix_arm_controller", None)
                fixed_sides = tuple(
                    getattr(fix_controller, "fixed_sides", ()) or ()
                )
                if fixed_sides:
                    self._waiting_reason = "unexpected_preexisting_fix_arm"
                    message = (
                        "规则采集启动被拒绝：初始化需要双臂运动，但当前仍有 "
                        f"FixArm 权限 {fixed_sides!r}。请先安全解除固定后重试。"
                    )
                    append_message = getattr(
                        self._control_ui,
                        "append_message",
                        None,
                    )
                    if callable(append_message):
                        append_message("RuleBased", message)
                    continue
                self._clear_manual_gripper_action()
                if self._configuration_batch_cycle_complete:
                    self._configuration_batch_completed_count = 0
                    self._configuration_batch_cycle_complete = False
                self._last_dispatched_emission = None
                self._armed = True
                self._waiting_reason = "formal_episode_not_active"
                self._gripper_action_status = ""
            elif intent == "rule_stop":
                self._restart_after_hold_requested = bool(
                    self._armed or self._execution_begun
                )
                self._armed = False
                self._waiting_reason = "runtime_home_required_after_hold"
                self._clear_current()
                if self._pending_gripper_action is not None:
                    self._finish_manual_gripper_action(
                        succeeded=False,
                        message="夹爪动作已由任务链取消并进入 Hold。",
                    )
                else:
                    self._clear_manual_gripper_action()
                self._cancel_initial_alignment_planning()
                self._establish_hold("rule-based operator stop")
            elif intent in {
                "release_grippers",
                "release_right_gripper",
                "close_grippers",
                "close_right_gripper",
            }:
                session = getattr(self._components, "session", None)
                state = getattr(
                    getattr(session, "state_machine", None),
                    "state",
                    None,
                )
                state_value = str(getattr(state, "value", state) or "").upper()
                if self._pending_gripper_action is not None:
                    self._gripper_action_status = "已有夹爪动作正在执行；新动作已拒绝。"
                    continue
                self._begin_manual_gripper_action(str(intent))
                if self._height_calibration_future is not None:
                    self._finish_manual_gripper_action(
                        succeeded=False,
                        message="高度标定读取期间夹爪保持不动。",
                    )
                elif not self.config.gripper_sides:
                    self._finish_manual_gripper_action(
                        succeeded=False,
                        message="当前拓扑没有已启用的 DMTAC 夹爪。",
                    )
                elif not self._manual_gripper_lifecycle_available(
                    session_state_value=state_value,
                ):
                    self._finish_manual_gripper_action(
                        succeeded=False,
                        message=(
                            "夹爪独立控制仅在没有活动 Episode 的采集前或轮次间可用。"
                        ),
                    )
                elif (
                    intent in {"release_right_gripper", "close_right_gripper"}
                    and "right" not in self.config.gripper_sides
                ):
                    self._finish_manual_gripper_action(
                        succeeded=False,
                        message="当前拓扑没有已启用的右夹爪。",
                    )
                else:
                    self._manual_gripper_started_at = None
                    self._manual_gripper_last_command_by_side.clear()
                    self._gripper_action_status = (
                        "已排队：松开夹爪"
                        if intent == "release_grippers"
                        else (
                            "已排队：松开右夹爪"
                            if intent == "release_right_gripper"
                            else (
                                "已排队：闭紧右夹爪"
                                if intent == "close_right_gripper"
                                else "已排队：闭紧夹爪"
                            )
                        )
                    )
        if self._abort_pre_dispatch_readiness_loss():
            self._publish_status(force=True)
            return
        self._publish_status()

    def _height_calibration_mutation_idle(self) -> bool:
        if (
            str(self.source_selection.get("collection_mode") or "rule_based_writing")
            != "rule_based_writing"
            or not self._started
            or self._closed
            or self._armed
            or self._execution_begun
            or self._completed
            or self._aborted
            or self._held
            or self._current is not None
            or self._frame_emission is not None
            or self._paused_hold_emission is not None
            or self._replay_reference_queue
            or self._configuration_future is not None
            or self._height_calibration_future is not None
            or self._pending_gripper_action is not None
            or self._initial_alignment_cancel_pending
            or self._collection_transition_pending()
        ):
            return False
        try:
            if self._recording_active is None or self._recording_active():
                return False
        except Exception:
            return False
        return True

    def _process_height_calibration_requests(self, now: float) -> bool:
        if self._poll_height_calibration_future(float(now)):
            return True
        consume = getattr(
            self._control_ui,
            "consume_rule_height_calibration_requests",
            None,
        )
        if not callable(consume):
            return False
        requests = tuple(consume())
        if not requests:
            return False
        publish_result = getattr(
            self._control_ui,
            "publish_rule_height_calibration_result",
            None,
        )
        handled = False
        for request in requests:
            handled = True
            error = ""
            self._height_calibration_pending = True
            async_pending = False
            try:
                if not isinstance(request, Mapping):
                    raise TypeError("height calibration request must be a mapping")
                operation = str(request.get("operation") or "").strip().lower()
                if operation == "clear":
                    if not self._height_calibration_mutation_idle():
                        raise RuntimeError("高度标定只能在数采和规则运动完整停止时清除。")
                    self._height_calibration = None
                    self._invalidate_writing_reachability_preflight()
                    self._writing_height_target_z_m = None
                    self._height_calibration_error = ""
                    append_message = getattr(self._control_ui, "append_message", None)
                    if callable(append_message):
                        append_message(
                            "RuleBasedWriting",
                            "已清除右臂 TCP 高度标定；后续执行完全恢复来源轨迹高度逻辑。",
                        )
                elif operation == "capture":
                    lower = float(request["lower_tolerance_m"])
                    upper = float(request["upper_tolerance_m"])
                    self._start_writing_height_capture(
                        now=float(now),
                        lower_tolerance_m=lower,
                        upper_tolerance_m=upper,
                    )
                    async_pending = True
                else:
                    raise ValueError(f"unsupported height calibration operation {operation!r}")
            except Exception as exc:  # noqa: BLE001 - maintenance request is recoverable.
                error = f"{type(exc).__name__}: {exc}"
                self._height_calibration_error = error
            finally:
                if not async_pending:
                    self._height_calibration_pending = False
                    if callable(publish_result):
                        publish_result(error=error or None)
        return handled

    def _start_writing_height_capture(
        self,
        *,
        now: float,
        lower_tolerance_m: float,
        upper_tolerance_m: float,
    ) -> None:
        if not self._height_calibration_mutation_idle():
            raise RuntimeError("高度标定只能在数采和规则运动完整停止时执行。")
        session = getattr(self._components, "session", None)
        session_state = getattr(getattr(session, "state_machine", None), "state", None)
        session_state_value = str(
            getattr(session_state, "value", session_state) or ""
        ).upper()
        blockers, reason = self._session_readiness_diagnosis(
            session_state_value=session_state_value
        )
        if session_state_value not in {"ARMED", "RUNNING"} or blockers:
            raise RuntimeError(
                "Rig/机械臂尚未安全就绪，不能记录高度："
                + (reason or f"Session state={session_state_value or 'UNKNOWN'}")
            )
        backend = getattr(self._components, "right_state_backend", None)
        refresh = getattr(backend, "refresh_state_from_hardware", None)
        if backend is None or not callable(refresh):
            raise RuntimeError("右臂只读状态后端不可用。")
        is_ready = getattr(backend, "is_ready", None)
        if callable(is_ready) and not bool(is_ready()):
            raise RuntimeError("右臂只读状态后端尚未就绪。")
        for name, value in (
            ("lower_tolerance_m", lower_tolerance_m),
            ("upper_tolerance_m", upper_tolerance_m),
        ):
            normalized = float(value)
            if not np.isfinite(normalized) or not 0.0001 <= normalized <= 0.02:
                raise ValueError(f"{name} must be in [0.0001, 0.02].")
        executor = getattr(self._components, "arm_state_read_executor", None)
        if executor is None:
            executor = self._configuration_executor
        self._height_calibration_request = {
            "requested_at": float(now),
            "lower_tolerance_m": float(lower_tolerance_m),
            "upper_tolerance_m": float(upper_tolerance_m),
        }
        try:
            self._height_calibration_future = executor.submit(
                refresh,
                hardware_read_retries=0,
                hardware_read_retry_delay_sec=0.0,
                include_derived_pose=True,
                include_timing_percentiles=False,
                include_pose_source_attestation=False,
            )
        except Exception:
            self._height_calibration_request = None
            raise

    def _poll_height_calibration_future(self, now: float) -> bool:
        future = self._height_calibration_future
        if future is None:
            return False
        if not future.done():
            self._height_calibration_pending = True
            return True
        request = dict(self._height_calibration_request or {})
        self._height_calibration_future = None
        self._height_calibration_request = None
        error = ""
        try:
            state = future.result()
            if not isinstance(state, Mapping):
                raise RuntimeError("右臂只读状态后端返回了无效结果。")
            if not bool(state.get("hardware_read_success", False)):
                code = state.get("hardware_result_code")
                detail = str(state.get("hardware_result_text") or "unknown read failure")
                raise RuntimeError(
                    "右臂硬件状态读取失败："
                    f"result_code={code!r} detail={detail}"
                )
            self._commit_writing_height(
                state,
                now=float(now),
                requested_at=float(request["requested_at"]),
                lower_tolerance_m=float(request["lower_tolerance_m"]),
                upper_tolerance_m=float(request["upper_tolerance_m"]),
            )
        except Exception as exc:  # noqa: BLE001 - maintenance read is recoverable.
            error = f"{type(exc).__name__}: {exc}"
            self._height_calibration_error = error
            append_message = getattr(self._control_ui, "append_message", None)
            if callable(append_message):
                append_message(
                    "RuleBasedWriting",
                    "右臂 TCP 高度标定失败：" + error,
                )
        finally:
            self._height_calibration_pending = False
            publish_result = getattr(
                self._control_ui,
                "publish_rule_height_calibration_result",
                None,
            )
            if callable(publish_result):
                publish_result(error=error or None)
        return True

    def _commit_writing_height(
        self,
        state: Mapping[str, Any],
        *,
        now: float,
        requested_at: float,
        lower_tolerance_m: float,
        upper_tolerance_m: float,
    ) -> None:
        raw = pose_matrix_to_xyzquat(state.get("current_tcp_pose"))
        if raw is None:
            raise RuntimeError("右臂 TCP 实测姿态无效。")
        canonical = np.asarray(
            self.runtime_pose_frame_contract.storage_pose_vector_to_canonical_desk(
                raw,
                side="right",
            ),
            dtype=float,
        ).reshape(7)
        if not np.all(np.isfinite(canonical)):
            raise RuntimeError("canonical_desk 右臂 TCP 实测姿态包含非有限值。")
        try:
            source_timestamp = float(state.get("measured_timestamp"))
        except (TypeError, ValueError):
            source_timestamp = math.nan
        if not np.isfinite(source_timestamp):
            raise RuntimeError("右臂实测状态缺少采样时间戳。")
        if source_timestamp + 0.05 < float(requested_at):
            raise RuntimeError(
                "右臂实测状态早于本次标定请求，拒绝使用陈旧 TCP。"
            )
        captured = RuleBasedWritingHeightCalibration(
            calibrated_z_m=float(canonical[2]),
            lower_tolerance_m=float(lower_tolerance_m),
            upper_tolerance_m=float(upper_tolerance_m),
            captured_at=float(now),
            source_timestamp=float(source_timestamp),
            pose_frame_contract_hash=self.runtime_pose_frame_contract.contract_hash,
        )
        self._height_calibration = captured
        self._invalidate_writing_reachability_preflight()
        self._height_lower_tolerance_m = float(captured.lower_tolerance_m)
        self._height_upper_tolerance_m = float(captured.upper_tolerance_m)
        self._height_calibration_error = ""
        self._writing_height_target_z_m = None
        self._manual_adjustment_motion_rebase_pending = True
        append_message = getattr(self._control_ui, "append_message", None)
        if callable(append_message):
            append_message(
                "RuleBasedWriting",
                "右臂 TCP 高度标定成功："
                f"canonical_desk Z={captured.calibrated_z_m:.6f}m，"
                f"安全带=[{captured.lower_bound_z_m:.6f}, "
                f"{captured.upper_bound_z_m:.6f}]m。",
            )

    def service_startup_control_plane(self, *, now: float) -> bool:
        """Finish offline configuration work while Rig admission is blocked.

        This deliberately reuses the same request consumer, executor, plan
        adoption, and UI result receipt as the realtime loop.  It does not
        consume rule/motion, manual-gripper, Episode, or Hold intents, and the
        rebuilt extension owns no runtime resources or dispatch authority.
        """

        handled = self._poll_prestart_configuration(float(now))
        if handled:
            self._publish_status()
        return handled

    def _make_initial_alignment_planner(
        self,
        components: object | None = None,
    ) -> RuleBasedInitialAlignmentPlanner:
        components = self._components if components is None else components
        if components is None:
            raise RuntimeError("rule-based components are not attached.")
        retargeter = getattr(components, "dual_arm_retargeter", None)
        target_filter = getattr(retargeter, "pre_ik_target_safety_filter", None)
        if retargeter is None or target_filter is None:
            raise RuntimeError("rule-based initial alignment safety is unavailable.")
        return RuleBasedInitialAlignmentPlanner(
            dual_arm_retargeter=retargeter,
            target_filter=target_filter,
            config=RuleBasedInitialAlignmentConfig(
                speed_scale=float(self.config.initial_alignment_speed_scale),
                reference_hz=float(self.config.position_reference_hz),
            ),
        )

    def _poll_prestart_configuration(self, now: float) -> bool:
        del now
        future = self._configuration_future
        if future is not None:
            if not future.done():
                self._waiting_reason = "configuration_rebuild"
                return True
            request = copy.deepcopy(dict(self._configuration_request or {}))
            self._configuration_future = None
            self._configuration_request = None
            publish_result = getattr(
                self._control_ui,
                "publish_rule_configuration_result",
                None,
            )
            reset_terminal_attempt = bool(self._completed or self._aborted)
            try:
                prepared_batch = future.result()
                if not prepared_batch:
                    raise RuntimeError("rule configuration batch rebuild returned empty")
                request_batch = [
                    copy.deepcopy(dict(configuration))
                    for configuration, _replacement in prepared_batch
                ]
                self._configuration_batch = list(prepared_batch)
                self._configuration_batch_index = 0
                self._configuration_batch_completed_count = 0
                self._configuration_batch_cycle_complete = False
                active_configuration, replacement = self._configuration_batch[0]
                self._adopt_prestart_extension(replacement)
                if self._configuration_on_commit is not None:
                    self._configuration_on_commit(active_configuration, replacement)
            except Exception as exc:  # noqa: BLE001 - invalid plans stay command-inert.
                self._configuration_error = f"{type(exc).__name__}: {exc}"
                self._waiting_reason = "configuration_rejected"
                self._gripper_action_status = ""
                if callable(publish_result):
                    publish_result(selected=None, error=self._configuration_error)
            else:
                if reset_terminal_attempt:
                    # A completed/failed attempt owns terminal cursor and
                    # sample-admission state even after its Episode has been
                    # fully stopped.  Rebuild that attempt around the newly
                    # selected plan, but deliberately retain Hold: changing a
                    # rule combination must never authorize motion or bypass
                    # the existing runtime-Home recovery path.
                    self._reset_attempt_for_runtime_home()
                    self._restart_after_hold_requested = False
                self._selected_configuration = copy.deepcopy(active_configuration)
                self._configuration_error = ""
                self._waiting_reason = (
                    "runtime_home_required_after_hold"
                    if self._held
                    else (
                        "formal_episode_not_active"
                        if self._armed
                        else "waiting_for_rule_start"
                    )
                )
                self._gripper_action_status = ""
                if callable(publish_result):
                    publish_result(
                        selected=(
                            {"batch": request_batch}
                            if isinstance(request.get("batch"), (list, tuple))
                            else copy.deepcopy(active_configuration)
                        ),
                        error=None,
                    )
            return True

        consume = getattr(
            self._control_ui,
            "consume_rule_configuration_requests",
            None,
        )
        if not callable(consume):
            return False
        requests = tuple(consume())
        if not requests:
            return False
        publish_result = getattr(
            self._control_ui,
            "publish_rule_configuration_result",
            None,
        )
        request = copy.deepcopy(dict(requests[-1]))
        if not self._configuration_lifecycle_available():
            self._configuration_error = (
                "规则组合仅可在没有活动 Episode，且停止事务与后台收尾均完成时修改。"
            )
            if callable(publish_result):
                publish_result(selected=None, error=self._configuration_error)
            return True
        self._configuration_request = request
        self._configuration_error = ""
        self._waiting_reason = "configuration_rebuild"
        self._clear_manual_gripper_action()
        self._gripper_action_status = "规则组合验证期间夹爪保持不动。"
        try:
            self._configuration_future = self._configuration_executor.submit(
                self._rebuild_configuration_batch,
                request,
            )
        except RuntimeError as exc:
            # Session teardown may race a final browser request.  Treat a
            # closed executor as a rejected command-inert rebuild, never as an
            # owner-loop exception that can strand the GUI.
            self._configuration_request = None
            self._configuration_error = f"{type(exc).__name__}: {exc}"
            self._waiting_reason = "configuration_rejected"
            self._gripper_action_status = ""
            if callable(publish_result):
                publish_result(selected=None, error=self._configuration_error)
        return True

    def _rebuild_configuration_batch(
        self,
        request: Mapping[str, Any],
    ) -> tuple[tuple[dict[str, Any], "RuleBasedRealTeleopExtension"], ...]:
        rebuilder = self._configuration_rebuilder
        if rebuilder is None:
            raise RuntimeError("rule configuration rebuilder is unavailable")
        raw_batch = request.get("batch")
        configurations = (
            tuple(raw_batch)
            if isinstance(raw_batch, (list, tuple))
            else (request,)
        )
        prepared: list[
            tuple[dict[str, Any], RuleBasedRealTeleopExtension]
        ] = []
        try:
            for raw_configuration in configurations:
                configuration = copy.deepcopy(dict(raw_configuration))
                replacement = rebuilder(configuration)
                if not isinstance(replacement, RuleBasedRealTeleopExtension):
                    raise TypeError(
                        "rule configuration rebuilder returned the wrong type."
                    )
                # ThreadPoolExecutor workers are lazy, but closing these
                # candidate-owned pools now makes the immutable batch cache
                # explicit and prevents future edits from retaining one pool
                # per queued item.
                replacement._initial_alignment_executor.shutdown(
                    wait=False,
                    cancel_futures=True,
                )
                replacement._configuration_executor.shutdown(
                    wait=False,
                    cancel_futures=True,
                )
                prepared.append((configuration, replacement))
        except Exception:
            for _configuration, replacement in prepared:
                replacement._initial_alignment_executor.shutdown(
                    wait=False,
                    cancel_futures=True,
                )
                replacement._configuration_executor.shutdown(
                    wait=False,
                    cancel_futures=True,
                )
            raise
        return tuple(prepared)

    def _adopt_prestart_extension(
        self,
        replacement: "RuleBasedRealTeleopExtension",
    ) -> None:
        if not isinstance(replacement, RuleBasedRealTeleopExtension):
            raise TypeError("rule configuration rebuilder returned the wrong type.")
        if replacement._components is not None or replacement._started:
            raise RuntimeError("rebuilt rule plan unexpectedly owns runtime resources.")
        if tuple(replacement.config.gripper_sides) != tuple(self.config.gripper_sides):
            raise RuntimeError("rule configuration cannot change hardware topology.")
        replacement._initial_alignment_executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
        replacement._configuration_executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
        prior_interval = int(self.config.pre_pen_down_replay_normal_episodes)
        prior_has_prelude = self._pre_pen_down_replay_plan is not None
        prior_normal_completed = self._pre_pen_down_normal_completed_count
        self._normal_plan = replacement._normal_plan
        self._pre_pen_down_replay_plan = replacement._pre_pen_down_replay_plan
        self.trajectory_groups = dict(replacement.trajectory_groups)
        self.config = replacement.config
        self.source_selection = copy.deepcopy(replacement.source_selection)
        preserve_cadence = bool(
            prior_has_prelude
            and self._pre_pen_down_replay_plan is not None
            and prior_interval
            == int(self.config.pre_pen_down_replay_normal_episodes)
        )
        self._pre_pen_down_normal_completed_count = (
            prior_normal_completed if preserve_cadence else 0
        )
        self.plan = self._plan_for_next_attempt()
        self._validate_plan(self._normal_plan)
        if self._pre_pen_down_replay_plan is not None:
            self._validate_plan(self._pre_pen_down_replay_plan)
        self._planned_sample_count = self._count_planned_samples()
        self._stage_total_by_step = self._stage_totals_by_step()
        self._boundaries.clear()
        self._boundary_frame_by_step = self._relative_boundary_frames()
        cursor = RuleBasedCollectionCursor(
            self.plan,
            self.trajectory_groups,
            boundary_provider=self._boundary_for,
        )
        self.adapter = RuleBasedCollectionAdapter(
            cursor,
            resolver=RealTeleopPoseFrameActionResolver(
                collection_contract=self.deployment.collection_pose_frame_contract,
                runtime_contract=self.runtime_pose_frame_contract,
            ),
        )
        rates = [
            float(group.action_hz)
            for group in self.trajectory_groups.values()
            if np.isfinite(float(group.action_hz)) and float(group.action_hz) > 0.0
        ]
        self.action_hz = float(
            self.config.action_hz
            if self.config.action_hz is not None
            else np.median(rates) if rates else 15.0
        )
        self._initial_alignment_planner = self._make_initial_alignment_planner()
        self._reset_pose_action_motion_intent("configuration_rebuild")
        self._initial_alignment_plan = None
        self._initial_alignment_template = None
        self._initial_alignment_waypoint_index = 0
        self._initial_alignment_effective_waypoint_count = 0
        self._initial_alignment_started_at = None
        self._initial_alignment_last_schedule_lag_sec = 0.0
        self._initial_alignment_cumulative_schedule_shift_sec = 0.0
        self._initial_alignment_skipped_clock_ticks = 0
        self._deferred_progress_arm_contract = None
        self._deferred_progress_arm_fix_established = False
        self._filtered_initial_alignment_arm_by_side = {}
        self._filtered_initial_alignment_identity = None
        self._current = None
        self._clear_replay_reference_schedule()
        self._frame_emission = None
        self._paused_hold_emission = None
        self._last_dispatched_emission = None
        self._current_dispatched = False
        self._current_sample_accepted = False
        self._dispatch_retry_started_at = None
        self._writer_retry_started_at = None
        self._target_arrival_confirmed = False
        self._target_arrival_snapshots = {}
        self._settled_samples = 0
        self._stage_label = None
        self._stage_index = None
        self._stage_frame_index = None
        self._stage_frame_total = None
        self._collection_position = None
        self._progress_scope_admission_error = ""
        self._height_descent_planned_count_added = False
        self._writing_height_target_z_m = None

    def command_for_observation(self, observation: object, now: float):
        # This marker describes only the command returned by this invocation.
        # Clearing it here prevents a no-command/completion tick from being
        # attributed to the preceding frame by after_session_step().
        self._frame_emission = None
        self._paused_hold_emission = None
        if self._closed:
            raise RuntimeError("rule-based runtime is closed.")
        if not self._started:
            return TeleopCommandOverride(None)
        if (
            self._completed
            and not self._held
            and self._last_dispatched_emission is not None
        ):
            return self._paused_arm_hold_override(
                reason=(
                    "successful_next_attempt_rebuild"
                    if self._next_attempt_rebuild_future is not None
                    else "successful_episode_finalizing"
                ),
                now=float(now),
            )
        if self._configuration_future is not None:
            return TeleopCommandOverride(None)
        if not self._armed and not self._execution_begun and not self._completed and not self._aborted:
            self._dispatch_pending_gripper_action(observation, float(now))
            return TeleopCommandOverride(None)
        if not self._armed or self._aborted:
            return TeleopCommandOverride(None)
        ready, reason, snapshots = self._runtime_ready(float(now))
        if not ready:
            self._waiting_reason = reason
            recoverable_pause = bool(
                reason == "formal_writer_paused"
                or (
                    reason.startswith("causal_hand_backend_recovery:")
                    and reason.endswith(":recovering")
                )
                or self._recoverable_arm_feedback_pause(
                    reason=reason,
                    now=float(now),
                )
            )
            if self._execution_begun and recoverable_pause:
                if self._replay_reference_pause_started_at is None:
                    self._replay_reference_pause_started_at = float(now)
                return self._paused_arm_hold_override(
                    reason=reason,
                    now=float(now),
                )
            if self._execution_begun:
                raise RuntimeError(
                    "rule-based execution lost formal collection readiness: " + reason
                )
            return TeleopCommandOverride(None)
        self._arm_feedback_pause_started_at = None
        if self._replay_reference_pause_started_at is not None:
            paused_sec = max(
                0.0,
                float(now) - float(self._replay_reference_pause_started_at),
            )
            if self._replay_reference_stage_started_at is not None:
                self._replay_reference_stage_started_at += paused_sec
            if self._initial_alignment_started_at is not None:
                self._initial_alignment_started_at += paused_sec
                self._initial_alignment_cumulative_schedule_shift_sec += paused_sec
            if self._next_due_at is not None:
                self._next_due_at += paused_sec
            # A reversible writer/modality pause is not active motion or an
            # active backend/writer retry.  Keep every bounded deadline on the
            # same pause-adjusted clock instead of letting camera jitter spend
            # the target-arrival budget while only a certified hold is sent.
            for attribute in (
                "_target_wait_started_at",
                "_dispatch_retry_started_at",
                "_writer_retry_started_at",
            ):
                started_at = getattr(self, attribute)
                if started_at is not None:
                    setattr(self, attribute, float(started_at) + paused_sec)
            self._replay_reference_pause_started_at = None
        self._waiting_reason = ""
        if self._current is not None and self._advance_due(
            float(now),
            snapshots=snapshots,
        ):
            self._clear_current()
        if self._aborted:
            self._publish_status(recording_ready=False)
            return TeleopCommandOverride(None)
        if self._current is None and not self._completed:
            self._load_next(observation, float(now), snapshots)
        elif (
            self._current is not None
            and self._current_dispatched
            and self._current.directive.segment_terminal
            and self._current.action.arm_pose_by_side
            and not self._target_reached(self._current, snapshots)
        ):
            # A coupled IK result is a bounded incremental command.  Terminal
            # pose steps must therefore continue solving the same immutable
            # target until causal feedback reaches it; replay pacing alone
            # would otherwise leave a pen_down/pen_up target after one step.
            try:
                bounded = self._bound_full_replay_hand_emission(
                    self._current,
                    snapshots=snapshots,
                )
                self._current = self._coupled_pose_emission(
                    bounded,
                    snapshots=snapshots,
                )
            except Exception as exc:  # noqa: BLE001 - rule motion fails closed.
                self._abort_coupled_pose(
                    "coupled terminal rule pose resolution failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        if self._completed:
            self._publish_status(recording_ready=ready)
            return TeleopCommandOverride(None)
        if self._current is None:
            # Initial-alignment planning is deliberately asynchronous.  The
            # command override remains claimed with no command while the
            # control loop, force guard, and causal state poller keep running.
            self._publish_status(recording_ready=ready)
            return TeleopCommandOverride(None)
        if self._rule_frame_requires_right_gripper_hold_owner(self._current):
            owner_ready, owner_detail = self._right_gripper_hold_owner_ready()
            if not owner_ready:
                self._waiting_reason = "right_gripper_hold_owner_pending"
                if self._execution_begun:
                    raise RuntimeError(
                        "rule-based writing lost its right-gripper hold owner: "
                        + (owner_detail or "owner receipt is no longer valid")
                    )
                # Do not let an arm qref reach Session while the matching
                # command-inert hand hold would be rejected by DMTAC.  Keep
                # the immutable current emission for the next control tick;
                # a successful task-chain close will make it dispatchable.
                self._publish_status(recording_ready=False)
                return TeleopCommandOverride(None)
        self._prepare_live_dispatch(self._current, float(now))
        self._frame_emission = self._current
        self._publish_status(recording_ready=True)
        return TeleopCommandOverride(self._current.command)

    def _prepare_live_dispatch(
        self,
        emission: RuleBasedCollectionEmission,
        now: float,
    ) -> None:
        """Bind every hardware attempt to a fresh, strictly increasing wall time."""

        timestamp = float(now)
        if self._last_live_dispatch_timestamp is not None:
            timestamp = max(
                timestamp,
                float(np.nextafter(self._last_live_dispatch_timestamp, math.inf)),
            )
        self._last_live_dispatch_timestamp = timestamp
        command = emission.command
        command.metadata.setdefault(
            "rule_based_source_action_timestamp",
            emission.action.timestamp,
        )
        source_cut_timestamp = float(
            command.metadata.setdefault(
                "rule_based_source_cut_timestamp",
                command.metadata.get(
                    "rule_based_scheduled_source_cut_timestamp",
                    timestamp,
                ),
            )
        )
        command.metadata["rule_based_dispatch_timestamp"] = timestamp
        fixed_right_grip = bool(
            str(self.source_selection.get("collection_mode") or "")
            == "rule_based_writing"
            and "right" in self.config.gripper_sides
        )
        command.timestamp = timestamp
        for effector in command.commands.values():
            if not isinstance(effector.metadata, dict):
                effector.metadata = dict(effector.metadata or {})
            effector.metadata["_parent_command_timestamp"] = timestamp
            if fixed_right_grip and effector.effector_name == "right_hand":
                # The task chain closes and opens the writing gripper exactly
                # once at its explicit boundaries. Every in-Episode hand
                # command therefore stays in one-shot mode at the fixed
                # closed target. This prevents the replay stream from
                # toggling catch mode or re-driving source hand jitter while
                # preserving one canonical action/command representation for
                # the dataset contract.
                effector.metadata["dmtac_continuous_grasp_override"] = False
                effector.metadata["dmtac_preserve_existing_grip"] = True
                command.metadata[
                    "rule_based_right_gripper_hold_mode"
                ] = "task_chain_fixed_closed_one_shot"
            if effector.effector_name.endswith("_arm"):
                # The strict-timestamp jerk planner must see one source cut
                # per distinct rule qref, not one cut per retry/repeated
                # dispatch of that same target.
                effector.metadata["sample_aligned_source_timestamp"] = (
                    source_cut_timestamp
                )

    def _begin_manual_gripper_action(self, intent: str) -> None:
        self._gripper_action_sequence += 1
        self._gripper_action_id = int(self._gripper_action_sequence)
        self._gripper_action_intent = str(intent)
        self._gripper_action_result = "pending"
        self._pending_gripper_action = str(intent)

    def _finish_manual_gripper_action(self, *, succeeded: bool, message: str) -> None:
        self._gripper_action_status = str(message)
        self._gripper_action_result = "succeeded" if succeeded else "failed"
        self._clear_manual_gripper_action()

    def _clear_manual_gripper_action(self) -> None:
        self._pending_gripper_action = None
        self._manual_gripper_started_at = None
        self._manual_gripper_last_command_by_side.clear()
        self._manual_gripper_command_baseline_by_side.clear()
        self._manual_gripper_last_dispatch_at_by_side.clear()
        self._manual_gripper_last_progress_at_by_side.clear()
        self._manual_gripper_settle_samples_by_side.clear()
        self._manual_gripper_last_settle_feedback_token_by_side.clear()

    def _manual_gripper_state(
        self,
        backend: object,
        *,
        side: str,
        now: float,
    ) -> tuple[dict[str, Any] | None, str]:
        """Read one causal hand snapshot for a manual reset.

        A formal recorder owns a background ``HandStateHistoryPoller``.  The
        realtime control thread must consume that cache instead of calling a
        vendor backend directly; otherwise a gripper reset can contend with
        telemetry/tactile I/O and stall the arm loop.  A present-but-empty
        poller is fail-closed (there is no safe measured position yet).  The
        direct backend path is retained only for compatibility fixtures and
        debug callers that have not constructed a recorder.
        """

        provider = self._dataset_recorder_provider
        recorder = None
        if callable(provider):
            try:
                recorder = provider()
            except Exception as exc:  # noqa: BLE001 - reset fails closed.
                return None, f"正式记录器不可用({type(exc).__name__})"
        poller = getattr(recorder, "hand_state_poller", None)
        lookup = getattr(poller, "latest_at_or_before", None)
        if callable(lookup):
            try:
                try:
                    raw = lookup(
                        float(now),
                        include_tactile_arrays=False,
                    )
                except TypeError:
                    # Compatibility with older injected pollers that accept
                    # only the timestamp.  This still reads a cache and does
                    # not initiate vendor I/O.
                    raw = lookup(float(now))
            except Exception as exc:  # noqa: BLE001 - reset fails closed.
                return None, f"后台遥测缓存不可用({type(exc).__name__})"
            if not isinstance(raw, Mapping):
                return None, "后台遥测缓存格式无效"
            state = raw.get(side)
            if not isinstance(state, Mapping):
                return None, f"等待{side}夹爪后台遥测首帧"
            return dict(state), ""

        get_state = getattr(backend, "get_state", None)
        if not callable(get_state):
            return None, "无反馈接口"
        try:
            state = get_state(
                include_tactile_arrays=False,
                include_tactile_vector=False,
                copy_tactile_arrays=False,
            )
        except TypeError:
            # Older test/debug backends expose a zero-argument get_state.
            try:
                state = get_state()
            except Exception as exc:  # noqa: BLE001 - reset fails closed.
                return None, f"反馈读取失败({type(exc).__name__})"
        except Exception as exc:  # noqa: BLE001 - reset fails closed.
            return None, f"反馈读取失败({type(exc).__name__})"
        if not isinstance(state, Mapping):
            return None, "反馈格式无效"
        return dict(state), ""

    @staticmethod
    def _manual_gripper_feedback_from_state(
        state: Mapping[str, Any] | None,
        state_error: str,
        *,
        side: str,
        now: float,
        command_dim: int,
        max_action: float,
        max_age_sec: float,
    ) -> tuple[np.ndarray | None, str]:
        if state is None:
            return None, state_error or "反馈状态不可用"

        # Manual reset movement is advanced only from measured feedback.  A
        # backend's joint_pos may be its most recent commanded target until
        # telemetry catches up, so it cannot authorize the next increment.
        try:
            measured = np.asarray(state.get("measured_joint_pos"), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None, "缺少实测夹爪位置"
        if measured.shape != (6,) or not np.all(np.isfinite(measured)):
            return None, "实测夹爪位置无效"

        age = state.get("joint_age_sec")
        if age is None:
            timestamp = state.get("joint_timestamp")
            try:
                age = float(now) - float(timestamp)
            except (TypeError, ValueError):
                age = float("inf")
        try:
            age_value = float(age)
        except (TypeError, ValueError):
            age_value = float("inf")
        if not np.isfinite(age_value) or age_value < -0.05:
            return None, "实测夹爪时间戳无效"
        if age_value > float(max_age_sec):
            return None, f"实测夹爪反馈陈旧({age_value:.3f}s)"

        scalar = float(measured[command_dim])
        if not 0.0 <= scalar <= max_action:
            return None, "实测夹爪位置越界"
        return measured.astype(np.float32, copy=True), ""

    @staticmethod
    def _manual_gripper_feedback_timestamp(
        state: Mapping[str, Any],
    ) -> float | None:
        """Return the hardware-side timestamp carried by one hand snapshot."""

        for key in (
            "joint_timestamp",
            "gripper_measurement_timestamp",
            "telemetry_snapshot_source_timestamp",
        ):
            try:
                value = float(state.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
        return None

    @staticmethod
    def _manual_gripper_feedback_token(
        state: Mapping[str, Any],
        *,
        now: float,
    ) -> tuple[str, float]:
        """Identify one distinct cached feedback frame.

        Production hand snapshots carry ``joint_timestamp`` or a telemetry
        sequence.  The control-time fallback keeps compatibility fixtures
        deterministic without pretending that a cached frame is fresh across
        repeated calls with the same timestamp.
        """

        timestamp = RuleBasedRealTeleopExtension._manual_gripper_feedback_timestamp(
            state
        )
        if timestamp is not None:
            return ("hardware_timestamp", timestamp)
        for key in ("telemetry_snapshot_sequence", "sequence"):
            try:
                sequence = float(state.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(sequence):
                return (key, sequence)
        return ("control_timestamp", float(now))

    def _manual_gripper_endpoint_settled(
        self,
        state: Mapping[str, Any],
        *,
        side: str,
        now: float,
        final_target: float,
    ) -> tuple[bool, str]:
        """Require post-command, distinct stable endpoint/contact feedback.

        DMTAC position writes are asynchronous.  Endpoint position alone is
        therefore insufficient: the cached value can be the last register
        read from before the close command, while status code 4 still means
        that the gripper is physically closing/holding.  A short fresh-frame
        barrier preserves the existing command-inert handoff without adding a
        second device loop.
        """

        recovery_phase = str(state.get("gripper_recovery_phase") or "").strip()
        if recovery_phase in {"recovering", "failed"}:
            return False, f"等待夹爪恢复({recovery_phase})"
        raw_error_code = state.get("gripper_error_code", state.get("error_code"))
        try:
            error_code = None if raw_error_code is None else int(raw_error_code)
        except (TypeError, ValueError):
            return False, "夹爪反馈错误码无效"
        if error_code not in (None, 0):
            return False, f"等待夹爪故障清除(error_code={error_code})"
        telemetry_error = str(state.get("telemetry_error") or "").strip()
        if telemetry_error and not telemetry_error.lower().startswith("tactile_"):
            return False, f"等待夹爪传输恢复({telemetry_error})"

        status_code = state.get("gripper_status_code", state.get("status_code"))
        try:
            status_code = None if status_code is None else int(status_code)
        except (TypeError, ValueError):
            return False, "夹爪状态码无效"
        if status_code in {4, 5}:
            direction = "张开" if status_code == 5 else "闭合"
            return False, f"夹爪仍在{direction}运动"
        if bool(state.get("gripper_is_moving")):
            return False, "夹爪反馈仍显示运动中"

        feedback_timestamp = self._manual_gripper_feedback_timestamp(state)
        last_dispatch = self._manual_gripper_last_dispatch_at_by_side.get(side)
        if last_dispatch is None:
            return False, "等待本次夹爪命令后的反馈"
        if (
            feedback_timestamp is not None
            and feedback_timestamp <= float(last_dispatch)
        ):
            return False, "等待夹爪命令后的新鲜反馈"

        token = self._manual_gripper_feedback_token(state, now=float(now))
        previous_token = self._manual_gripper_last_settle_feedback_token_by_side.get(
            side
        )
        if previous_token == token:
            count = self._manual_gripper_settle_samples_by_side.get(side, 0)
            return False, (
                "等待下一帧不同的实测反馈"
                f"({count}/{self.config.manual_gripper_settle_samples})"
            )
        self._manual_gripper_last_settle_feedback_token_by_side[side] = token
        count = self._manual_gripper_settle_samples_by_side.get(side, 0) + 1
        self._manual_gripper_settle_samples_by_side[side] = count
        required = int(self.config.manual_gripper_settle_samples)
        if count < required:
            return False, f"等待端点稳定反馈({count}/{required})"
        return True, ""

    def _manual_gripper_closed_idle_progress(
        self,
        *,
        intent: str,
        side: str,
        status_code: int | None,
        measured_scalar: float,
        last_target: float | None,
        final_target: float,
    ) -> bool:
        """Recognize an object-limited closed-side idle receipt.

        DMTAC may report ``status=2`` (closed-side idle/stop) when the fingers
        contact the pen before reaching the normalized command endpoint.  The
        status alone is ambiguous: an already stalled or disconnected hand
        can produce the same value.  Require a close intent, a command from
        this action, and fresh measured progress from that command's baseline.
        The endpoint settle gate still validates the error, freshness, and
        stable-frame requirements after this evidence is present.
        """

        if (
            intent not in {"close_grippers", "close_right_gripper"}
            or status_code != 2
            or last_target is None
            or side not in self._manual_gripper_last_dispatch_at_by_side
            or abs(float(last_target) - float(final_target))
            > float(self.config.manual_gripper_endpoint_tolerance)
        ):
            return False
        baseline = self._manual_gripper_command_baseline_by_side.get(side)
        if baseline is None:
            return False
        backend = getattr(self._components, f"{side}_hand_backend", None)
        send_threshold = float(
            getattr(getattr(backend, "config", None), "send_threshold", 0.0)
        )
        minimum_progress = max(
            send_threshold,
            float(self.config.manual_gripper_endpoint_tolerance),
            1e-4,
        )
        return float(measured_scalar) - float(baseline) >= minimum_progress

    def _manual_gripper_feedback(
        self,
        backend: object,
        *,
        side: str,
        now: float,
    ) -> tuple[np.ndarray | None, str]:
        command_dim = int(self._hand_command_dim_by_side.get(side, 0))
        max_action = float(self._hand_max_action_by_side.get(side, 1.0))
        state, state_error = self._manual_gripper_state(
            backend,
            side=side,
            now=float(now),
        )
        return self._manual_gripper_feedback_from_state(
            state,
            state_error,
            side=side,
            now=float(now),
            command_dim=command_dim,
            max_action=max_action,
            max_age_sec=float(self.config.hand_state_max_age_sec),
        )

    @staticmethod
    def _manual_gripper_dispatch_outcome(
        backend: object,
        *,
        side: str,
        target: float,
        receipt: object,
        command_dim: int,
        measured_scalar: float,
        state: Mapping[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Classify one DMTAC receipt without turning faults into retries."""

        backend_status = str(getattr(backend, "_last_send_status", "") or "")
        skip_reason = str(
            getattr(backend, "_last_send_skip_reason", "") or ""
        )
        state_provided = isinstance(state, Mapping)
        if not backend_status and state_provided:
            backend_status = str(state.get("gripper_last_send_status", "") or "")
            skip_reason = str(
                state.get("gripper_last_send_skip_reason", "") or ""
            )
        if not backend_status and not state_provided:
            get_state = getattr(backend, "get_state", None)
            if callable(get_state):
                try:
                    # This is a compatibility fallback only.  Production
                    # callers pass the recorder-owned cached state above.
                    state = get_state(
                        include_tactile_arrays=False,
                        include_tactile_vector=False,
                        copy_tactile_arrays=False,
                    )
                except TypeError:
                    state = get_state()
                except Exception:  # noqa: BLE001 - receipt classification fails closed.
                    state = None
                if isinstance(state, Mapping):
                    backend_status = str(
                        state.get("gripper_last_send_status", "") or ""
                    )
                    skip_reason = str(
                        state.get("gripper_last_send_skip_reason", "") or ""
                    )

        try:
            receipt_accepted = bool(receipt)
        except Exception:  # noqa: BLE001 - opaque false receipt stays rejected.
            receipt_accepted = False
        if receipt_accepted:
            return "accepted", backend_status or "accepted"

        if backend_status == "skipped_threshold":
            # With the DMTAC backend contract, a threshold skip is an accepted
            # no-op only when both its command cache and measured feedback
            # already match this target.  Validate the cache here as well so a
            # different backend cannot silently advance the feedback gate.
            last_sent = getattr(backend, "_last_sent_command", None)
            try:
                last_sent_values = np.asarray(last_sent, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                last_sent_values = np.asarray([], dtype=float)
            send_threshold = float(
                getattr(getattr(backend, "config", None), "send_threshold", 0.0)
            )
            cache_matches = (
                last_sent_values.shape == (6,)
                and np.all(np.isfinite(last_sent_values))
                and abs(float(last_sent_values[command_dim]) - float(target))
                <= max(send_threshold, 1e-7)
            )
            feedback_matches = abs(float(measured_scalar) - float(target)) <= max(
                send_threshold,
                1e-7,
            )
            if cache_matches and feedback_matches:
                return "accepted", "skipped_threshold_already_at_target"
            return "failed", f"{side}:阈值跳过但缓存/实测目标不一致"

        transient_labels = {
            "rate_limited": "后端限频",
            "io_busy": "I/O忙",
        }
        if backend_status in transient_labels:
            return "retry", f"{side}:{transient_labels[backend_status]}"

        failure_labels = {
            "send_disabled": "发送开关关闭",
            "hardware_fault": "硬件故障",
            "not_connected": "未连接",
            "send_exception": "发送异常",
        }
        label = failure_labels.get(
            backend_status,
            f"未知拒绝({backend_status or '无后端状态'})",
        )
        suffix = f"/{skip_reason}" if skip_reason else ""
        return "failed", f"{side}:{label}{suffix}"

    def _manual_gripper_action_sides(self, intent: str) -> tuple[str, ...]:
        """Return the hand scope for one maintenance intent.

        Generic GUI gripper actions retain their configured all-side behavior.
        The task chain uses the explicit right-only intent because only the
        right hand owns the pen; the left hand must not receive a close command.
        """

        if intent in {"release_right_gripper", "close_right_gripper"}:
            return ("right",) if "right" in self.config.gripper_sides else ()
        return tuple(self.config.gripper_sides)

    def _dispatch_pending_gripper_action(self, observation: object, now: float) -> None:
        """Advance one feedback-gated reset increment through the Session Rig.

        The rule UI is intentionally not a second device loop. A reset click
        is converted to the same canonical ``RobotAction``/``TeleopCommand``
        carrier used by collection and sent through ``TeleopSession``'s
        existing dispatch seam. Task-chain right-hand actions issue one
        endpoint command and then only observe fresh feedback; generic
        maintenance actions retain bounded incremental motion. No formal
        writer call is made and no arm effector is included.
        """

        intent = self._pending_gripper_action
        if intent is None:
            return
        action_sides = self._manual_gripper_action_sides(intent)
        if not action_sides:
            self._finish_manual_gripper_action(
                succeeded=False,
                message="当前拓扑没有可用的目标夹爪。",
            )
            return
        session = getattr(self._components, "session", None)
        if session is None:
            self._finish_manual_gripper_action(
                succeeded=False,
                message="夹爪控制失败：Session 不可用。",
            )
            return
        state = getattr(getattr(session, "state_machine", None), "state", None)
        state_value = str(getattr(state, "value", state) or "").upper()
        # A successful Episode deliberately leaves the shared Session in
        # RUNNING while this rule-only source is command-inert.  Admit manual
        # gripper maintenance in that inter-Episode state through the same
        # lifecycle predicate used by the GUI; active/finalizing Episodes,
        # configuration changes, alignment and Hold remain fail-closed.
        if not self._manual_gripper_lifecycle_available(
            session_state_value=state_value,
        ):
            self._finish_manual_gripper_action(
                succeeded=False,
                message=(
                    "夹爪控制已拒绝：当前不处于无活动 Episode 的安全维护窗口"
                    f"（Session={state_value or 'UNKNOWN'}）。"
                ),
            )
            return

        if self._manual_gripper_started_at is None:
            self._manual_gripper_started_at = float(now)
            self._manual_gripper_last_progress_at_by_side = {
                side: float(now) for side in action_sides
            }

        hand_qpos: dict[str, np.ndarray] = {}
        measured_scalar_by_side: dict[str, float] = {}
        feedback_state_by_side: dict[str, Mapping[str, Any]] = {}
        unavailable: list[str] = []
        waiting: list[str] = []
        arrived: list[str] = []
        wrong_direction: list[str] = []
        resend_sides: set[str] = set()
        task_chain_right_only = intent in {
            "release_right_gripper",
            "close_right_gripper",
        }
        is_release = intent in {"release_grippers", "release_right_gripper"}
        target_scalar = 0.0 if is_release else 1.0
        direction = -1.0 if is_release else 1.0
        for side in action_sides:
            backend = getattr(self._components, f"{side}_hand_backend", None)
            if backend is None:
                unavailable.append(side)
                continue
            is_ready = getattr(backend, "is_ready", None)
            if callable(is_ready):
                try:
                    if not bool(is_ready()):
                        unavailable.append(side)
                        continue
                except Exception:
                    unavailable.append(side)
                    continue
            feedback_state, feedback_state_error = self._manual_gripper_state(
                backend,
                side=side,
                now=float(now),
            )
            measured_qpos, feedback_error = self._manual_gripper_feedback_from_state(
                feedback_state,
                feedback_state_error,
                side=side,
                now=float(now),
                command_dim=int(self._hand_command_dim_by_side.get(side, 0)),
                max_action=float(self._hand_max_action_by_side.get(side, 1.0)),
                max_age_sec=float(self.config.hand_state_max_age_sec),
            )
            if feedback_state is not None:
                feedback_state_by_side[side] = feedback_state
            if measured_qpos is None:
                waiting.append(f"{side}:{feedback_error}")
                continue
            config = getattr(backend, "config", None)
            command_dim = self._hand_command_dim_by_side.get(
                side,
                int(getattr(config, "command_dim", 0)),
            )
            max_action = self._hand_max_action_by_side.get(
                side,
                float(getattr(config, "max_action", 1.0)),
            )
            if command_dim < 0 or command_dim >= 6 or not np.isfinite(max_action) or max_action <= 0.0:
                unavailable.append(side)
                continue
            measured_scalar = float(measured_qpos[command_dim])
            measured_scalar_by_side[side] = measured_scalar
            final_target = float(np.clip(target_scalar, 0.0, max_action))
            last_target = self._manual_gripper_last_command_by_side.get(side)
            at_endpoint = abs(measured_scalar - final_target) <= float(
                self.config.manual_gripper_endpoint_tolerance
            )
            raw_status_code = (feedback_state or {}).get(
                "gripper_status_code",
                (feedback_state or {}).get("status_code"),
            )
            try:
                status_code = (
                    None if raw_status_code is None else int(raw_status_code)
                )
            except (TypeError, ValueError):
                status_code = None
            force_stopped_grasp = bool(
                intent == "close_right_gripper"
                and side == "right"
                and status_code == 6
                and side in self._manual_gripper_last_dispatch_at_by_side
            )
            closed_idle_progress = self._manual_gripper_closed_idle_progress(
                intent=intent,
                side=side,
                status_code=status_code,
                measured_scalar=measured_scalar,
                last_target=last_target,
                final_target=final_target,
            )
            if at_endpoint or force_stopped_grasp or closed_idle_progress:
                settled, settle_reason = self._manual_gripper_endpoint_settled(
                    feedback_state or {},
                    side=side,
                    now=float(now),
                    final_target=final_target,
                )
                if settled:
                    arrived.append(side)
                    self._manual_gripper_last_progress_at_by_side[side] = float(now)
                else:
                    waiting.append(f"{side}:{settle_reason}")
                    # An explicit maintenance action must have a causal device
                    # command even when the actuator starts at its endpoint.
                    # Reapply the same target once so release disables catch
                    # mode and task-chain close establishes a one-shot closed
                    # state before the writing replay preserves that physical
                    # grip without another serial write.
                    if (
                        at_endpoint
                        and side
                        not in self._manual_gripper_last_dispatch_at_by_side
                    ):
                        last_dispatch_at = (
                            self._manual_gripper_last_dispatch_at_by_side.get(side)
                        )
                        if (
                            last_dispatch_at is None
                            or float(now) - float(last_dispatch_at)
                            >= float(self.config.manual_gripper_resend_interval_sec)
                        ):
                            qpos = np.zeros(6, dtype=np.float32)
                            qpos[command_dim] = final_target
                            hand_qpos[side] = qpos
                            resend_sides.add(side)
                continue
            if last_target is not None:
                command_baseline = self._manual_gripper_command_baseline_by_side.get(side)
                if command_baseline is None:
                    unavailable.append(side)
                    continue
                progress = direction * (measured_scalar - float(command_baseline))
                if progress < -float(self.config.manual_gripper_tolerance):
                    wrong_direction.append(
                        f"{side}={measured_scalar:.3f}（上一实测 {command_baseline:.3f}）"
                    )
                    continue
                send_threshold = float(
                    getattr(getattr(backend, "config", None), "send_threshold", 0.0)
                )
                minimum_progress = max(
                    send_threshold,
                    float(self.config.manual_gripper_endpoint_tolerance),
                    1e-4,
                )
                if task_chain_right_only:
                    if progress >= minimum_progress:
                        # The endpoint command is already owned by firmware.
                        # Fresh forward progress only refreshes the bounded
                        # no-progress watchdog; it must not trigger another
                        # register write.
                        self._manual_gripper_command_baseline_by_side[side] = (
                            measured_scalar
                        )
                        self._manual_gripper_last_progress_at_by_side[side] = (
                            float(now)
                        )
                    waiting.append(
                        f"{side}={measured_scalar:.3f}/{last_target:.3f}"
                    )
                    continue
                if progress < minimum_progress:
                    last_dispatch_at = self._manual_gripper_last_dispatch_at_by_side.get(
                        side,
                        float(now),
                    )
                    if (
                        float(now) - float(last_dispatch_at)
                        < float(self.config.manual_gripper_resend_interval_sec)
                    ):
                        waiting.append(
                            f"{side}={measured_scalar:.3f}/{last_target:.3f}"
                        )
                        continue
                    # A positive Rig/backend receipt proves only that the
                    # register write was accepted; it does not prove physical
                    # motion.  Re-send the same bounded segment when fresh
                    # feedback has made no progress.  Do not advance the target
                    # or reset the measured-progress timeout.
                    qpos = np.zeros(6, dtype=np.float32)
                    qpos[command_dim] = float(last_target)
                    hand_qpos[side] = qpos
                    resend_sides.add(side)
                    continue
                # Real grippers can stop short of, or pass, an intermediate
                # target because their register controller and mechanics are
                # discrete.  Fresh measured progress in the requested
                # direction is sufficient to rebase the next command.  The
                # final target is still feedback-gated and every new command
                # remains bounded from the latest measured position.
            baseline = measured_scalar
            if task_chain_right_only:
                next_target = final_target
            else:
                delta = float(np.clip(
                    final_target - baseline,
                    -float(self.config.gripper_max_step),
                    float(self.config.gripper_max_step),
                ))
                next_target = baseline + delta
            qpos = np.zeros(6, dtype=np.float32)
            qpos[command_dim] = float(np.clip(next_target, 0.0, max_action))
            hand_qpos[side] = qpos
        if wrong_direction:
            self._finish_manual_gripper_action(
                succeeded=False,
                message=(
                    "夹爪控制停止：检测到与请求方向相反的实测运动（"
                    + ",".join(wrong_direction)
                    + "）。"
                ),
            )
            return
        timed_out = [
            side
            for side in action_sides
            if side not in arrived
            and (
                float(now)
                - float(self._manual_gripper_last_progress_at_by_side.get(side, now))
                > float(self.config.manual_gripper_timeout_sec)
            )
        ]
        if timed_out:
            self._finish_manual_gripper_action(
                succeeded=False,
                message=(
                    "夹爪控制超时：以下侧在限时内没有新的实测进展："
                    + ",".join(sorted(timed_out))
                    + "。"
                ),
            )
            return
        if unavailable:
            self._finish_manual_gripper_action(
                succeeded=False,
                message=(
                    "夹爪控制已拒绝：未就绪侧 "
                    + ",".join(sorted(unavailable))
                    + "。"
                ),
            )
            return
        if not hand_qpos:
            if len(arrived) == len(action_sides):
                if intent == "release_right_gripper":
                    message = "右夹爪实测已松开"
                elif is_release:
                    message = "夹爪实测已松开"
                elif intent == "close_right_gripper":
                    message = "右夹爪实测已闭紧"
                else:
                    message = "夹爪实测已闭紧"
                self._finish_manual_gripper_action(
                    succeeded=True,
                    message=message,
                )
            elif waiting:
                if intent == "release_right_gripper":
                    action_label = "松开右夹爪"
                elif is_release:
                    action_label = "松开"
                elif intent == "close_right_gripper":
                    action_label = "闭紧右夹爪"
                else:
                    action_label = "闭紧"
                self._gripper_action_status = (
                    f"正在{action_label}夹爪：等待逐段实测到位（"
                    + ",".join(waiting)
                    + "）。"
                )
            else:
                self._finish_manual_gripper_action(
                    succeeded=False,
                    message="夹爪控制已拒绝：没有可用夹爪。",
                )
            return

        try:
            action = RobotAction.bimanual_qpos(
                left_hand=hand_qpos.get("left"),
                right_hand=hand_qpos.get("right"),
                timestamp=float(now),
                source="rule_based_manual_gripper",
                metadata={
                    "rule_based_manual_gripper_action": intent,
                    "rule_based_manual_gripper_sides": list(action_sides),
                    "rule_based_manual_gripper_max_step": float(
                        self.config.gripper_max_step
                    ),
                    "rule_based_manual_gripper_dispatch_mode": (
                        "single_endpoint_feedback_gated"
                        if task_chain_right_only
                        else "bounded_incremental_feedback_gated"
                    ),
                    "rule_based_capture_enabled": False,
                    "external_gripper_motion": True,
                    "external_non_motion": True,
                    "hand_action_specs_by_side": {
                        side: HandActionSpec.dmtac_gripper(
                            side,
                            gripper_slot=int(self._hand_command_dim_by_side[side]),
                        ).to_metadata()
                        for side in hand_qpos
                    },
                },
            )
            command = self.adapter.resolver.resolve(action)
            # Task-chain right-hand motion and release are one-shot firmware
            # operations. The writing replay keeps the same mode and target;
            # it must not restore continuous catch or re-drive the hand.
            manual_continuous_grasp = intent == "close_grippers"
            manual_motion_mode = (
                "one_shot_open"
                if is_release
                else (
                    "one_shot_close"
                    if task_chain_right_only
                    else "continuous_close"
                )
            )
            for side in hand_qpos:
                effector = command.commands.get(f"{side}_hand")
                if effector is None:
                    raise RuntimeError(
                        f"resolved manual gripper command is missing {side}_hand."
                    )
                if not isinstance(effector.metadata, dict):
                    effector.metadata = dict(effector.metadata or {})
                # Release and the task-chain right-only close are one-shot
                # maintenance moves. The generic all-side close retains its
                # historical continuous-grasp behavior; writing replay keeps
                # the task-chain right hand fixed in the same one-shot mode.
                effector.metadata["dmtac_continuous_grasp_override"] = bool(
                    manual_continuous_grasp
                )
                # Never inherit the writing replay's command-inert hold for a
                # manual release.  The open command must reach DMTAC so the
                # backend can switch back to its explicit one-shot mode even
                # when the target value is already cached at the endpoint.
                effector.metadata["dmtac_preserve_existing_grip"] = False
                effector.metadata["rule_based_manual_gripper_dispatch_mode"] = (
                    "single_endpoint_feedback_gated"
                    if task_chain_right_only
                    else "bounded_incremental_feedback_gated"
                )
                effector.metadata["rule_based_manual_gripper_motion_mode"] = (
                    manual_motion_mode
                )
            command.metadata.update(
                {
                    "rule_based_manual_gripper_action": intent,
                    "rule_based_manual_gripper_sides": list(action_sides),
                    "rule_based_manual_gripper_motion_mode": manual_motion_mode,
                    "rule_based_capture_enabled": False,
                    "rule_based_arm_motion": False,
                    "external_gripper_motion": True,
                    "external_non_motion": True,
                    "dataset_recording_eligible": False,
                }
            )
            dispatch_external = getattr(session, "dispatch_external_command", None)
            if not callable(dispatch_external):
                raise RuntimeError(
                    "Session does not expose the external-command safety seam."
                )
            result = dispatch_external(
                command,
                observation,
                float(now),
                record=False,
                transition_to_running=False,
            )
            if not bool(getattr(result, "dispatched", False)):
                fault = getattr(result, "fault", None) or getattr(
                    result,
                    "guard_fault",
                    None,
                )
                if fault is not None:
                    reason = str(getattr(fault, "message", fault) or fault)
                    message = f"夹爪控制已被安全门禁拒绝：{reason}"
                else:
                    reason = str(
                        getattr(result, "rejection_reason", "dispatch_rejected")
                        or "dispatch_rejected"
                    )
                    message = f"夹爪控制已拒绝：{reason}。"
                self._finish_manual_gripper_action(
                    succeeded=False,
                    message=message,
                )
                return
        except Exception as exc:  # noqa: BLE001 - reset must fail closed.
            message = (
                "夹爪控制发送失败：" f"{type(exc).__name__}: {exc}"
            )
            append_message = getattr(self._control_ui, "append_message", None)
            if callable(append_message):
                append_message("DMTAC", message)
            self._finish_manual_gripper_action(
                succeeded=False,
                message=message,
            )
            return
        receipts = dict(getattr(result, "dispatch_receipts", {}) or {})
        accepted: list[str] = []
        retrying: list[str] = []
        failed: list[str] = []
        for side, qpos in hand_qpos.items():
            receipt = receipts.get(f"{side}_hand")
            command_dim = int(self._hand_command_dim_by_side[side])
            outcome, detail = self._manual_gripper_dispatch_outcome(
                getattr(self._components, f"{side}_hand_backend"),
                side=side,
                target=float(qpos[command_dim]),
                receipt=receipt,
                command_dim=command_dim,
                measured_scalar=measured_scalar_by_side[side],
                state=feedback_state_by_side.get(side),
            )
            if side in resend_sides:
                # Rate-limit every repeated dispatch attempt, including a
                # backend ``rate_limited``/``io_busy`` receipt.  This timestamp
                # is deliberately separate from measured progress and cannot
                # extend the fail-closed no-progress timeout.
                self._manual_gripper_last_dispatch_at_by_side[side] = float(now)
            if outcome == "accepted":
                self._manual_gripper_last_command_by_side[side] = float(
                    qpos[command_dim]
                )
                self._manual_gripper_last_dispatch_at_by_side[side] = float(now)
                # A newly accepted asynchronous command invalidates all
                # endpoint samples collected before it.  The next completion
                # decision must be based on distinct feedback after this
                # command, including when the backend accepts a threshold
                # no-op as already at target.
                self._manual_gripper_settle_samples_by_side[side] = 0
                self._manual_gripper_last_settle_feedback_token_by_side.pop(
                    side,
                    None,
                )
                if side not in resend_sides:
                    self._manual_gripper_command_baseline_by_side[side] = float(
                        measured_scalar_by_side[side]
                    )
                    self._manual_gripper_last_progress_at_by_side[side] = float(now)
                accepted.append(side)
            elif outcome == "retry":
                retrying.append(detail)
            else:
                failed.append(detail)
        if intent == "release_right_gripper":
            action_label = "松开右夹爪"
        elif is_release:
            action_label = "松开"
        elif intent == "close_right_gripper":
            action_label = "闭紧右夹爪"
        else:
            action_label = "闭紧"
        if failed:
            partial = (
                "；部分侧已发送 " + ",".join(sorted(accepted))
                if accepted
                else ""
            )
            message = (
                f"夹爪控制停止：{'，'.join(failed)}{partial}。"
            )
            append_message = getattr(self._control_ui, "append_message", None)
            if callable(append_message):
                append_message("DMTAC", message)
            self._finish_manual_gripper_action(
                succeeded=False,
                message=message,
            )
        elif accepted:
            positions = ", ".join(
                f"{side}={self._manual_gripper_last_command_by_side[side]:.2f}"
                for side in sorted(accepted)
            )
            resend_note = (
                "（无实测进展，重发同一安全分段）"
                if resend_sides.intersection(accepted)
                else ""
            )
            retry_suffix = (
                "；" + "，".join(retrying) + "，等待重试"
                if retrying
                else ""
            )
            self._gripper_action_status = (
                f"正在{action_label}夹爪：已发送受限分段 {positions}{resend_note}"
                f"{retry_suffix}，等待实测到位。"
            )
        elif retrying:
            self._gripper_action_status = (
                f"正在{action_label}夹爪：{'，'.join(retrying)}，等待重试。"
            )

    def _runtime_ready(
        self,
        now: float,
    ) -> tuple[bool, str, dict[str, dict[str, Any]]]:
        provider = self._dataset_recorder_provider
        if provider is None:
            return False, "runtime_not_bound", {}
        recorder = provider()
        if recorder is None:
            return False, "formal_recorder_unavailable", {}
        if self._recording_active is None or not self._recording_active():
            return False, "formal_episode_not_active", {}
        if not bool(getattr(recorder, "is_recording", False)):
            return False, "formal_writer_not_recording", {}
        if not bool(getattr(recorder, "can_record_samples", False)):
            return False, "formal_writer_paused", {}
        poller = getattr(recorder, "arm_state_poller", None)
        if poller is None:
            return False, "causal_arm_feedback_poller_unavailable", {}
        readiness = poller.readiness_by_side(
            required_sides=self.deployment.active_sides,
            sample_timestamp=now,
        )
        blocked = {side: value for side, value in readiness.items() if value}
        if blocked:
            details = ",".join(f"{side}={value}" for side, value in sorted(blocked.items()))
            return False, "causal_arm_feedback_not_ready:" + details, {}
        snapshots = poller.latest_at_or_before(now)
        missing = sorted(set(self.deployment.active_sides) - set(snapshots))
        if missing:
            return False, "causal_arm_feedback_missing:" + ",".join(missing), {}
        snapshots = {side: dict(value) for side, value in snapshots.items()}
        if self.config.gripper_sides:
            hand_poller = getattr(recorder, "hand_state_poller", None)
            if hand_poller is None or not callable(
                getattr(hand_poller, "latest_at_or_before", None)
            ):
                return False, "causal_hand_feedback_poller_unavailable", {}
            hand_states = hand_poller.latest_at_or_before(float(now))
            for side in self.config.gripper_sides:
                state = hand_states.get(side)
                if not isinstance(state, Mapping):
                    return False, f"causal_hand_feedback_missing:{side}", {}
                recovery_phase = str(
                    state.get("gripper_recovery_phase") or ""
                ).strip()
                if recovery_phase in {"recovering", "failed"}:
                    return False, (
                        "causal_hand_backend_recovery:"
                        f"{side}:{recovery_phase}"
                    ), {}
                hand_qpos = self._hand_qpos_from_state(state, side=side)
                if hand_qpos is None:
                    return False, f"causal_hand_feedback_invalid:{side}", {}
                age = state.get("joint_age_sec")
                if age is None:
                    timestamp = state.get("joint_timestamp")
                    try:
                        age = float(now) - float(timestamp)
                    except (TypeError, ValueError):
                        age = float("inf")
                try:
                    age_value = float(age)
                except (TypeError, ValueError):
                    age_value = float("inf")
                if not np.isfinite(age_value) or age_value < -0.05:
                    return False, f"causal_hand_feedback_invalid_timestamp:{side}", {}
                if age_value > float(self.config.hand_state_max_age_sec):
                    return False, (
                        f"causal_hand_feedback_stale:{side}=" f"{age_value:.3f}s"
                    ), {}
                snapshots.setdefault(side, {})["actual_hand_qpos"] = hand_qpos
                snapshots[side]["hand_state"] = dict(state)
        return True, "", snapshots

    def _load_next(
        self,
        observation: object,
        now: float,
        snapshots: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if self._initial_alignment_template is not None:
            if not self._prepare_initial_alignment_waypoint(now, snapshots):
                return
            if self._current is not None:
                return
        runtime_context = {
            "actual_arm_state_by_side": snapshots,
            "runtime_pose_frame_contract": self.runtime_pose_frame_contract,
            "writing_height_calibration": self._height_calibration_metadata(),
            "writing_height_target_z_m": self._writing_height_target_z_m,
        }
        if self._replay_reference_queue:
            emission = self._pop_due_replay_reference(float(now))
        else:
            self._clear_replay_reference_schedule()
            try:
                emission = self.adapter.next_runtime(
                    observation,
                    context=runtime_context,
                )
            except StopIteration:
                incomplete = self._incomplete_detail_stages()
                if incomplete:
                    detail = ", ".join(
                        f"{stage}={values['accepted']}/{values['required']}"
                        for stage, values in sorted(incomplete.items())
                    )
                    self._abort_session_step(
                        "formal dataset stage coverage is incomplete: " + detail
                    )
                    return
                self._completed = True
                self._armed = False
                self._waiting_reason = "completed"
                return
        if emission.directive.kind == "initialize":
            self._start_initial_alignment_planning(
                emission,
                now=now,
                snapshots=snapshots,
            )
            return
        if (
            emission.directive.kind == "replay"
            and emission.action.arm_pose_by_side
            and "rule_based_reference_sequence" not in emission.action.metadata
        ):
            try:
                self._prepare_replay_reference_queue(
                    emission,
                    observation=observation,
                    runtime_context=runtime_context,
                    now=float(now),
                )
                emission = self._pop_due_replay_reference(float(now))
            except Exception as exc:  # noqa: BLE001 - replay planning fails closed.
                self._abort_coupled_pose(
                    "60 Hz replay reference planning failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        elif (
            emission.directive.kind == "custom_action"
            and emission.action.arm_pose_by_side
            and bool(
                emission.action.metadata.get(
                    "rule_based_position_reference_resample",
                    False,
                )
            )
            and "rule_based_reference_sequence" not in emission.action.metadata
        ):
            try:
                self._prepare_custom_action_reference_queue(
                    emission,
                    observation=observation,
                    runtime_context=runtime_context,
                    now=float(now),
                )
                emission = self._pop_due_replay_reference(float(now))
            except Exception as exc:  # noqa: BLE001 - planning fails closed.
                self._abort_coupled_pose(
                    "60 Hz custom-action reference planning failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        if emission.directive.kind == "replay":
            try:
                emission = self._bound_full_replay_hand_emission(
                    emission,
                    snapshots=snapshots,
                )
            except Exception as exc:  # noqa: BLE001 - live target must fail closed.
                self._abort_coupled_pose(
                    "complete replay gripper target bounding failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        if emission.action.arm_pose_by_side:
            try:
                emission = self._coupled_pose_emission(
                    emission,
                    snapshots=snapshots,
                )
            except Exception as exc:  # noqa: BLE001 - rule motion fails closed.
                self._abort_coupled_pose(
                    "coupled rule pose resolution failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        self._adopt_emission(emission, now=now)

    def _height_calibration_metadata(self) -> dict[str, Any]:
        calibration = self._height_calibration
        if calibration is None:
            return {
                "enabled": False,
                "side": "right",
                "frame": CANONICAL_DESK_ACTION_FRAME,
                "lower_tolerance_m": float(self._height_lower_tolerance_m),
                "upper_tolerance_m": float(self._height_upper_tolerance_m),
            }
        return calibration.to_metadata()

    def _project_writing_replay_height(
        self,
        emission: RuleBasedCollectionEmission,
    ) -> RuleBasedCollectionEmission:
        """Apply only the calibrated right-TCP lower floor to writing replay."""

        calibration = self._height_calibration
        if (
            calibration is None
            or not calibration.enabled
            or str(self.source_selection.get("collection_mode") or "rule_based_writing")
            != "rule_based_writing"
            or emission.directive.kind != "replay"
            or emission.directive.step_id
            not in {"right_write.pen_down", "right_write.pen_up"}
            or "right" not in emission.action.arm_pose_by_side
        ):
            return emission
        frame = str(emission.action.frame_by_effector.get("right_arm") or "")
        if frame == CANONICAL_DESK_ACTION_FRAME:
            source_canonical = source_pose = np.asarray(
                emission.action.arm_pose_by_side["right"],
                dtype=float,
            ).reshape(-1)
        elif frame == "right_arm_base":
            source_pose = np.asarray(
                emission.action.arm_pose_by_side["right"],
                dtype=float,
            ).reshape(-1)
            source_canonical = np.asarray(
                self.runtime_pose_frame_contract.storage_pose_vector_to_canonical_desk(
                    source_pose,
                    side="right",
                ),
                dtype=float,
            ).reshape(-1)
        else:
            raise RuntimeError(
                "writing height projection requires canonical_desk or right-arm-base "
                "TCP actions; "
                f"got {frame or 'missing'!r}"
            )
        if (
            source_pose.shape != (7,)
            or source_canonical.shape != (7,)
            or not np.all(np.isfinite(source_pose))
            or not np.all(np.isfinite(source_canonical))
        ):
            raise RuntimeError("writing replay has an invalid right TCP action.")
        target_canonical = source_canonical.copy()
        target_canonical[2] = max(
            float(target_canonical[2]),
            calibration.lower_bound_z_m,
        )
        clipped = not math.isclose(
            float(target_canonical[2]),
            float(source_canonical[2]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        target_pose = (
            target_canonical
            if frame == CANONICAL_DESK_ACTION_FRAME
            else np.asarray(
                self.runtime_pose_frame_contract.canonical_desk_pose_vector_to_storage(
                    target_canonical,
                    side="right",
                ),
                dtype=float,
            ).reshape(7)
        )
        metadata = _mutable_metadata_copy(emission.action.metadata)
        metadata.update(
            {
                "rule_based_height_calibration": calibration.to_metadata(),
                "rule_based_height_projection_applied": True,
                "rule_based_height_projection_policy": "lower_floor_only",
                "rule_based_height_source_z_m": float(source_canonical[2]),
                "rule_based_height_target_z_m": float(target_canonical[2]),
                "rule_based_height_clipped": bool(clipped),
            }
        )
        action = replace(
            emission.action,
            arm_pose_by_side={
                **dict(emission.action.arm_pose_by_side),
                "right": target_pose.astype(np.float32),
            },
            metadata=metadata,
        )
        return replace(emission, action=action)

    def _prepare_replay_reference_queue(
        self,
        first: RuleBasedCollectionEmission,
        *,
        observation: object,
        runtime_context: Mapping[str, Any],
        now: float,
    ) -> None:
        """Expand one replay stage onto the shared 60 Hz reference clock.

        Static replay directives are finite and already bind any relative
        boundary before this method is entered.  Consuming the remainder of
        that one stage here preserves the cursor's sole ownership of source
        indices and provenance while making the complete source-time window
        available to the established offline resampler.
        """

        directive = first.directive
        if directive.group_key is None or directive.replay_frame_index is None:
            raise ValueError("replay directive lacks a source group/frame identity")
        try:
            plan_step_index = int(directive.metadata["rule_based_step_index"])
            step = self.plan.steps[plan_step_index]
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("replay directive lacks a valid plan-step identity") from exc
        if not isinstance(step, RuleBasedReplay):
            raise TypeError("replay directive does not resolve to RuleBasedReplay")
        if step.group_key != directive.group_key:
            raise ValueError("replay directive group differs from its plan step")

        group = self.trajectory_groups[step.group_key]
        reference_hz = float(self.config.position_reference_hz)
        keyframe_hz = float(group.action_hz)
        if reference_hz + 1e-9 < keyframe_hz:
            raise ValueError(
                "position replay reference_hz must be >= source keyframe rate; "
                f"got reference_hz={reference_hz:g}, keyframe_hz={keyframe_hz:g}"
            )

        source_emissions = [first]
        first_frame_index = int(directive.replay_frame_index)
        if first_frame_index != int(step.start_index):
            raise ValueError(
                "replay stage expansion did not begin at its declared start index"
            )
        for expected_frame_index in range(first_frame_index + 1, int(step.end_index)):
            following = self.adapter.next_runtime(
                observation,
                context=runtime_context,
            )
            following_directive = following.directive
            if (
                following_directive.kind != "replay"
                or following_directive.group_key != step.group_key
                or int(following_directive.replay_frame_index or 0)
                != expected_frame_index
                or following_directive.step_id != directive.step_id
            ):
                raise RuntimeError(
                    "rule cursor changed stages during finite replay expansion"
                )
            source_emissions.append(following)

        first_source_time = float(group.relative_time_sec(first_frame_index))
        source_emissions = [
            self._project_writing_replay_height(source_emission)
            for source_emission in source_emissions
        ]
        source_indices = [
            int(source_emission.directive.replay_frame_index or 0)
            for source_emission in source_emissions
        ]
        source_times = [
            float(group.relative_time_sec(source_index)) - first_source_time
            for source_index in source_indices
        ]
        self._install_pose_reference_queue(
            source_emissions,
            source_times=source_times,
            source_indices=source_indices,
            keyframe_hz=keyframe_hz,
            reference_hz=reference_hz,
            source_kind="replay",
            now=float(now),
        )

    def _prepare_custom_action_reference_queue(
        self,
        first: RuleBasedCollectionEmission,
        *,
        observation: object,
        runtime_context: Mapping[str, Any],
        now: float,
    ) -> None:
        """Expand one opted-in finite action segment onto the reference clock."""

        directive = first.directive
        try:
            plan_step_index = int(directive.metadata["rule_based_step_index"])
            step = self.plan.steps[plan_step_index]
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "custom-action directive lacks a valid plan-step identity"
            ) from exc
        if not isinstance(step, RuleBasedActionSegment):
            raise TypeError(
                "resampled custom-action directive does not resolve to an action segment"
            )
        keyframe_hz = float(
            first.action.metadata.get("rule_based_position_keyframe_hz", math.nan)
        )
        reference_hz = float(self.config.position_reference_hz)
        if not np.isfinite(keyframe_hz) or keyframe_hz <= 0.0:
            raise ValueError(
                "resampled custom action requires a finite positive keyframe rate"
            )
        if reference_hz + 1e-9 < keyframe_hz:
            raise ValueError(
                "position reference_hz must be >= custom-action keyframe rate; "
                f"got reference_hz={reference_hz:g}, keyframe_hz={keyframe_hz:g}"
            )

        source_emissions = [first]
        while not source_emissions[-1].directive.segment_terminal:
            following = self.adapter.next_runtime(
                observation,
                context=runtime_context,
            )
            following_directive = following.directive
            following_keyframe_hz = float(
                following.action.metadata.get(
                    "rule_based_position_keyframe_hz",
                    math.nan,
                )
            )
            if (
                following_directive.kind != "custom_action"
                or following_directive.step_id != directive.step_id
                or not bool(
                    following.action.metadata.get(
                        "rule_based_position_reference_resample",
                        False,
                    )
                )
                or not math.isclose(
                    following_keyframe_hz,
                    keyframe_hz,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise RuntimeError(
                    "rule cursor changed stages or reference contract during "
                    "finite custom-action expansion"
                )
            source_emissions.append(following)

        first_source_time = float(source_emissions[0].action.timestamp)
        source_times = [
            float(source_emission.action.timestamp) - first_source_time
            for source_emission in source_emissions
        ]
        source_indices = list(range(len(source_emissions)))
        self._install_pose_reference_queue(
            source_emissions,
            source_times=source_times,
            source_indices=source_indices,
            keyframe_hz=keyframe_hz,
            reference_hz=reference_hz,
            source_kind="custom_action",
            now=float(now),
        )

    def _install_pose_reference_queue(
        self,
        source_emissions: list[RuleBasedCollectionEmission],
        *,
        source_times: list[float],
        source_indices: list[int],
        keyframe_hz: float,
        reference_hz: float,
        source_kind: str,
        now: float,
    ) -> None:
        """Build the shared reference queue without duplicating cursor ownership."""

        if source_kind not in {"replay", "custom_action"}:
            raise ValueError(
                f"unsupported pose reference source kind: {source_kind!r}"
            )
        if (
            not source_emissions
            or len(source_times) != len(source_emissions)
            or len(source_indices) != len(source_emissions)
        ):
            raise ValueError(
                "pose reference source arrays must be non-empty and aligned"
            )
        source_time_array = np.asarray(source_times, dtype=float)
        if (
            not np.all(np.isfinite(source_time_array))
            or not math.isclose(float(source_time_array[0]), 0.0, abs_tol=1e-9)
            or (
                len(source_time_array) > 1
                and np.any(np.diff(source_time_array) <= 0.0)
            )
        ):
            raise ValueError(
                "pose reference source timestamps must start at zero and increase"
            )

        resolver = self.adapter.resolver
        if not isinstance(resolver, RealTeleopPoseFrameActionResolver):
            raise RuntimeError("rule pose-frame resolver was replaced")
        projected_actions: list[RobotAction] = []
        targets: list[TimestampedMotionTarget] = []
        for local_index, source_emission in enumerate(source_emissions):
            projected = resolver.project_action(source_emission.action)
            projected_actions.append(projected)
            targets.append(
                TimestampedMotionTarget(
                    timestamp=float(source_time_array[local_index]),
                    available_timestamp=float(source_time_array[local_index]),
                    source_index=local_index,
                    pose_by_side=projected.arm_pose_by_side,
                    metadata={
                        "rule_based_source_frame_index": int(
                            source_indices[local_index]
                        ),
                        "rule_based_source_sequence_index": int(
                            source_emission.directive.sequence_index
                        ),
                    },
                )
            )

        resampler = MotionTargetResampler(targets)
        reference_timestamps = build_uniform_time_grid(
            resampler.source_timestamps[0],
            resampler.source_timestamps[-1],
            hz=reference_hz,
            include_final=True,
        )
        if not reference_timestamps:
            raise RuntimeError("replay resampler returned no reference timestamps")

        queue: list[_ReplayReferenceCarrier] = []
        captured_source_indices: set[int] = set()
        represented_source_indices: set[int] = set()
        for reference_index, reference_timestamp in enumerate(reference_timestamps):
            right_position = bisect_right(
                resampler.source_timestamps,
                float(reference_timestamp),
            )
            if right_position == 0:
                source_local_index = right_source_local_index = 0
            elif right_position >= len(resampler.samples):
                source_local_index = right_source_local_index = (
                    len(resampler.samples) - 1
                )
            else:
                source_local_index = right_position - 1
                right_source_local_index = (
                    source_local_index
                    if math.isclose(
                        float(reference_timestamp),
                        resampler.source_timestamps[source_local_index],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    else right_position
                )
            if source_local_index < 0 or source_local_index >= len(source_emissions):
                raise RuntimeError("replay resampler returned an invalid source index")
            parent = source_emissions[source_local_index]
            projected_parent = projected_actions[source_local_index]
            # A source frame is represented by either side of the interpolation
            # bracket.  Requiring every frame to become the left/active parent
            # is too strict for jittered or non-integer source timestamps: a
            # frame can be used as the right bracket and still contribute to
            # the emitted pose through MotionTargetResampler.
            represented_source_indices.add(source_local_index)
            represented_source_indices.add(right_source_local_index)
            captures_source_frame = source_local_index not in captured_source_indices
            if captures_source_frame:
                captured_source_indices.add(source_local_index)
            # Every dense reference in a recordable source interval remains
            # eligible for the independent source-cadence sampler.  The
            # former code exposed only the exact source keyframe and then
            # held motion until its asynchronous writer ACK arrived.  That
            # made camera/writer latency an accidental motion clock and
            # produced 20 Hz bursts.
            # Directive sequence de-duplication still commits at most one
            # formal sample per source interval.
            reference_directive = (
                parent.directive
                if captures_source_frame
                else replace(
                    parent.directive,
                    segment_terminal=False,
                    segment_stop_reason=None,
                )
            )

            next_offset = (
                float(reference_timestamps[reference_index + 1])
                if reference_index + 1 < len(reference_timestamps)
                else float(reference_timestamp) + 1.0 / reference_hz
            )
            queue.append(
                _ReplayReferenceCarrier(
                    resampler=resampler,
                    reference_sequence=reference_index,
                    reference_timestamp_sec=float(reference_timestamp),
                    parent=parent,
                    projected_parent=projected_parent,
                    directive=reference_directive,
                    next_offset_sec=next_offset,
                    left_parent_source_index=int(
                        source_indices[source_local_index]
                    ),
                    right_parent_source_index=int(
                        source_indices[right_source_local_index]
                    ),
                    captures_source_frame=captures_source_frame,
                    keyframe_hz=keyframe_hz,
                    reference_hz=reference_hz,
                    source_kind=str(source_kind),
                )
            )

        missing = sorted(
            set(range(len(source_emissions))) - represented_source_indices
        )
        if missing:
            raise RuntimeError(
                "pose reference clock cannot represent source frames: "
                f"unrepresented_local_indices={missing}"
            )
        step_ids = {carrier.directive.step_id for carrier in queue}
        if len(step_ids) != 1:
            raise RuntimeError(
                "pose reference queue crosses rule steps: "
                f"step_ids={sorted(step_ids)!r}"
            )
        capture_sequences = {
            int(carrier.directive.sequence_index)
            for carrier in queue
            if carrier.captures_source_frame and carrier.directive.capture.enabled
        }
        if capture_sequences:
            # The terminal keyframe owns one final keyframe interval.  At the
            # formal writer rate this is the number of samples the stage can
            # intentionally produce before runtime catch-up or a real writer
            # rejection.  Cap by distinct capturable source intervals so a
            # jittered frame represented only as the interpolation right
            # bracket is not counted as an impossible writer sample.
            writer_window_sec = float(reference_timestamps[-1]) + (
                1.0 / float(keyframe_hz)
            )
            writer_rate_count = max(
                1,
                math.ceil(writer_window_sec * float(self.action_hz) - 1e-9),
            )
            expected_count = min(len(capture_sequences), writer_rate_count)
            step_id = next(iter(step_ids))
            previous = self._writer_rate_expected_samples_by_step.get(step_id)
            if previous is not None and previous != expected_count:
                raise RuntimeError(
                    "pose reference writer-rate expectation changed within one "
                    f"attempt: step={step_id!r} previous={previous} "
                    f"current={expected_count}"
                )
            self._writer_rate_expected_samples_by_step[step_id] = expected_count
        self._replay_reference_queue = deque(queue)
        self._replay_reference_stage_started_at = float(now)
        self._replay_reference_pause_started_at = None
        self._replay_reference_last_schedule_lag_sec = 0.0
        self._replay_reference_cumulative_schedule_shift_sec = 0.0
        self._replay_reference_rebased_tick_count = 0

    def _pop_due_replay_reference(self, now: float) -> RuleBasedCollectionEmission:
        """Dispatch one ordered reference and slip a late owner-loop clock.

        The real owner loop may run below the nominal 60 Hz reference rate.
        Skipping every overdue grid point preserves wall-clock duration, but
        turns a smooth resampled path into irregular multi-grid target jumps.
        Keep every certified reference ordered and dispatch it exactly once;
        when this call is late, move the remaining schedule by the observed
        lag so no stale backlog is emitted in a burst.
        """

        if not self._replay_reference_queue:
            raise RuntimeError("replay reference queue is empty")
        stage_started_at = self._replay_reference_stage_started_at
        if stage_started_at is None:
            raise RuntimeError("replay reference stage lacks a monotonic start cut")
        carrier = self._replay_reference_queue.popleft()
        scheduled_cut = (
            float(stage_started_at) + float(carrier.reference_timestamp_sec)
        )
        schedule_lag_sec = max(0.0, float(now) - scheduled_cut)
        self._replay_reference_last_schedule_lag_sec = schedule_lag_sec
        if schedule_lag_sec > 0.0:
            self._replay_reference_stage_started_at = (
                float(stage_started_at) + schedule_lag_sec
            )
            self._replay_reference_cumulative_schedule_shift_sec += schedule_lag_sec
            self._replay_reference_rebased_tick_count += 1

        emission = self._materialize_replay_reference(carrier)
        schedule_metadata: dict[str, Any] = {
            "rule_based_scheduled_source_cut_timestamp": scheduled_cut,
            "rule_based_reference_dispatch_lag_sec": schedule_lag_sec,
            "rule_based_reference_schedule_policy": "ordered_rebase_no_skip",
            "rule_based_reference_schedule_shift_sec": schedule_lag_sec,
            "rule_based_reference_cumulative_schedule_shift_sec": float(
                self._replay_reference_cumulative_schedule_shift_sec
            ),
            "rule_based_reference_rebased_tick_count": int(
                self._replay_reference_rebased_tick_count
            ),
        }
        emission.command.metadata.update(schedule_metadata)
        return replace(
            emission,
            action=replace(
                emission.action,
                metadata={
                    **dict(emission.action.metadata),
                    **schedule_metadata,
                },
            ),
        )

    @staticmethod
    def _materialize_replay_reference(
        carrier: _ReplayReferenceCarrier,
    ) -> RuleBasedCollectionEmission:
        """Build only the one reference that reaches live safety/dispatch."""

        reference = carrier.resampler.resample_at(
            carrier.reference_timestamp_sec,
            sequence=carrier.reference_sequence,
        )
        parent = carrier.parent
        projected_parent = carrier.projected_parent
        parent_source_index = int(carrier.left_parent_source_index)
        pose_vectors: dict[str, np.ndarray] = {}
        for side, pose in reference.pose_by_side.items():
            xyzquat = pose_matrix_to_xyzquat(pose)
            if xyzquat is None:
                raise ValueError(f"{side} replay reference is not a valid rigid pose")
            pose_vectors[str(side)] = np.asarray(
                xyzquat,
                dtype=np.float32,
            ).reshape(7)
        reference_metadata = {
            **dict(projected_parent.metadata),
            "rule_based_reference_clock": "source_time_resampled",
            "rule_based_reference_hz": carrier.reference_hz,
            "rule_based_reference_sequence": int(reference.sequence),
            "rule_based_reference_timestamp_sec": float(reference.timestamp),
            "rule_based_reference_next_offset_sec": carrier.next_offset_sec,
            "rule_based_reference_captures_source_frame": bool(
                carrier.captures_source_frame
            ),
            "rule_based_reference_source_kind": carrier.source_kind,
            "rule_based_reference_left_source_frame_index": parent_source_index,
            "rule_based_reference_right_source_frame_index": (
                carrier.right_parent_source_index
            ),
            "rule_based_parent_source_action_timestamp": parent.action.timestamp,
            "position_pipeline": (
                f"safe_{carrier.keyframe_hz:g}hz_to_reference_"
                f"{carrier.reference_hz:g}hz_to_existing_command_stream"
            ),
            "position_keyframe_hz": carrier.keyframe_hz,
            "position_reference_hz": carrier.reference_hz,
            "position_reference_sequence": int(reference.sequence),
            "left_source_index": parent_source_index,
            "right_source_index": carrier.right_parent_source_index,
            "left_source_timestamp": float(reference.left_source_timestamp),
            "right_source_timestamp": float(reference.right_source_timestamp),
            "interpolation_alpha": float(reference.interpolation_alpha),
            "pose_action_interval_velocity": _mutable_metadata_copy(
                reference.metadata.get("pose_action_interval_velocity", {})
            ),
            "resampling": _mutable_metadata_copy(reference.metadata),
        }
        action = replace(
            projected_parent,
            timestamp=float(reference.timestamp),
            arm_pose_by_side=pose_vectors,
            # Preserve the complete source action throughout its interval so
            # whichever 15 Hz T0 wins admission still records/replays hands.
            hand_qpos_by_side=projected_parent.hand_qpos_by_side,
            end_effector_targets_by_side=(
                projected_parent.end_effector_targets_by_side
            ),
            active_effectors=projected_parent.active_effectors,
            source=(
                "rule_based_replay_position_reference"
                if carrier.source_kind == "replay"
                else "rule_based_custom_action_position_reference"
            ),
            metadata=reference_metadata,
        )
        return RuleBasedCollectionEmission(
            directive=carrier.directive,
            action=action,
            command=TeleopCommand(
                timestamp=float(reference.timestamp),
                metadata=_mutable_metadata_copy(reference_metadata),
            ),
        )

    def _bound_full_replay_hand_emission(
        self,
        emission: RuleBasedCollectionEmission,
        *,
        snapshots: Mapping[str, Mapping[str, Any]],
    ) -> RuleBasedCollectionEmission:
        """Bound one live full-replay hand command from fresh measured state.

        Source Episodes may encode endpoint-like gripper changes between two
        adjacent dataset frames.  Those deltas are not physical dispatch
        steps.  Keep the source target as immutable provenance, but project
        the command emitted on the existing 60 Hz replay clock to at most one
        configured hardware step from the causal measured state.  The arm
        cursor and source-time schedule remain independent of gripper I/O.
        """

        if (
            emission.directive.kind != "replay"
            or str(self.source_selection.get("collection_mode") or "")
            != "full_trajectory_replay"
            or not emission.action.hand_qpos_by_side
        ):
            return emission

        metadata = _mutable_metadata_copy(emission.action.metadata)
        preserved = metadata.get("rule_based_source_hand_qpos_by_side")
        source_values = (
            preserved
            if isinstance(preserved, Mapping)
            else emission.action.hand_qpos_by_side
        )
        expected_sides = set(self.config.gripper_sides)
        actual_sides = {str(side) for side in source_values}
        if actual_sides != expected_sides:
            raise RuntimeError(
                "complete replay hand target topology changed: "
                f"expected={sorted(expected_sides)!r} "
                f"actual={sorted(actual_sides)!r}"
            )

        bounded_by_side: dict[str, np.ndarray] = {}
        source_by_side: dict[str, np.ndarray] = {}
        baseline_by_side: dict[str, np.ndarray] = {}
        limited_by_side: dict[str, bool] = {}
        applied_step_by_side: dict[str, float] = {}
        max_step = float(self.config.gripper_max_step)
        for side in self.config.gripper_sides:
            target = np.asarray(source_values[side], dtype=float).reshape(-1)
            actual = np.asarray(
                snapshots.get(side, {}).get("actual_hand_qpos"),
                dtype=float,
            ).reshape(-1)
            if target.shape != (6,) or not np.all(np.isfinite(target)):
                raise RuntimeError(
                    f"complete replay requires a finite six-slot {side} hand target"
                )
            if actual.shape != (6,) or not np.all(np.isfinite(actual)):
                raise RuntimeError(
                    f"complete replay requires fresh finite causal {side} hand state"
                )
            bounded = actual + np.clip(target - actual, -max_step, max_step)
            applied_step = float(np.max(np.abs(bounded - actual)))
            if applied_step > max_step + 1e-9:
                raise RuntimeError(
                    f"internal {side} hand step bound violation: "
                    f"step={applied_step:.9f} max={max_step:.9f}"
                )
            source_by_side[side] = target.astype(np.float32, copy=True)
            baseline_by_side[side] = actual.astype(np.float32, copy=True)
            bounded_by_side[side] = bounded.astype(np.float32, copy=True)
            limited_by_side[side] = not np.allclose(
                bounded,
                target,
                rtol=0.0,
                atol=1e-9,
            )
            applied_step_by_side[side] = applied_step

        metadata.update(
            {
                "rule_based_source_hand_qpos_by_side": source_by_side,
                "rule_based_hand_dispatch_baseline": "causal_measured_state",
                "rule_based_hand_dispatch_baseline_by_side": baseline_by_side,
                "rule_based_hand_dispatch_target_by_side": bounded_by_side,
                "rule_based_hand_dispatch_max_step": max_step,
                "rule_based_hand_dispatch_applied_step_by_side": (
                    applied_step_by_side
                ),
                "rule_based_hand_dispatch_limited_by_side": limited_by_side,
            }
        )
        return replace(
            emission,
            action=replace(
                emission.action,
                hand_qpos_by_side=bounded_by_side,
                metadata=metadata,
            ),
        )

    def _coupled_pose_emission(
        self,
        emission: RuleBasedCollectionEmission,
        *,
        snapshots: Mapping[str, Mapping[str, Any]],
    ) -> RuleBasedCollectionEmission:
        """Resolve every rule pose through the one live coupled safety owner."""

        components = self._components
        if components is None:
            raise RuntimeError("rule-based components are not attached")
        retargeter = getattr(components, "dual_arm_retargeter", None)
        if retargeter is None:
            raise RuntimeError("coupled target solver is unavailable")
        calibration = getattr(components.session, "calibration", None)
        if calibration is None:
            raise RuntimeError("runtime calibration is unavailable")
        resolver = self.adapter.resolver
        if not isinstance(resolver, RealTeleopPoseFrameActionResolver):
            raise RuntimeError("rule pose-frame resolver was replaced")

        self._ensure_pose_action_motion_intent_step(emission.directive.step_id)
        projected_action = resolver.project_action(emission.action)
        projected_command = resolver.delegate.resolve(projected_action)
        active_sides = tuple(
            side for side in ("left", "right") if side in projected_action.arm_pose_by_side
        )
        if not active_sides:
            raise RuntimeError("rule pose action has no active arm target")

        actual_qpos = self._snapshot_qpos_pair(snapshots)
        if not self._coupled_pose_solver_seeded:
            retargeter.set_current_qpos_pair(
                left_qpos=actual_qpos["left"],
                right_qpos=actual_qpos["right"],
                reset_tracking=True,
            )
            self._coupled_pose_solver_seeded = True

        target_pose_by_side: dict[str, np.ndarray] = {}
        source_by_side: dict[str, BimanualPoseTargetSource | None] = {
            "left": None,
            "right": None,
        }
        desired_base_pose_by_side: dict[str, np.ndarray] = {}
        rigid_projection_by_side: dict[str, dict[str, float | bool]] = {}
        world_to_base_by_side: dict[str, np.ndarray] = {}
        interval_velocity = projected_command.metadata.get(
            "pose_action_interval_velocity",
            {},
        )
        interval_velocity = (
            dict(interval_velocity) if isinstance(interval_velocity, Mapping) else {}
        )
        velocity_by_side = interval_velocity.get("by_side", {})
        velocity_by_side = (
            dict(velocity_by_side) if isinstance(velocity_by_side, Mapping) else {}
        )
        try:
            motion_timestamp = float(emission.action.timestamp)
        except (TypeError, ValueError):
            motion_timestamp = time.monotonic()
        if not np.isfinite(motion_timestamp):
            motion_timestamp = time.monotonic()
        for side in ("left", "right"):
            arm_calibration = calibration.get_arm_for_side(side)
            transform = getattr(arm_calibration, "T_D_B", None)
            if transform is None:
                raise RuntimeError(f"{side} arm calibration lacks T_D_B")
            world_to_base_by_side[side] = np.linalg.inv(
                np.asarray(transform, dtype=float).reshape(4, 4)
            )

            if side in active_sides:
                effector = projected_command.commands.get(f"{side}_arm")
                if effector is None or str(effector.command_type) != "cartesian_pose":
                    raise RuntimeError(
                        f"projected {side} rule target is not a Cartesian arm command"
                    )
                base_pose = np.asarray(effector.value, dtype=float).reshape(4, 4)
            else:
                base_pose = np.asarray(
                    snapshots[side]["actual_tcp_pose"], dtype=float
                ).reshape(4, 4)
            base_pose, projection = project_near_rigid_pose(
                base_pose,
                name=f"rule_base_pose.{side}",
                max_orthogonality_error=_RULE_POSE_RIGID_PROJECTION_LIMIT,
                max_determinant_error=_RULE_POSE_RIGID_PROJECTION_LIMIT,
                max_projection_frobenius=_RULE_POSE_RIGID_PROJECTION_LIMIT,
                homogeneous_atol=1e-9,
            )
            rigid_projection_by_side[side] = projection
            if side in active_sides:
                side_velocity = velocity_by_side.get(side, {})
                side_velocity = (
                    dict(side_velocity)
                    if isinstance(side_velocity, Mapping)
                    else {}
                )
                motion_intent = self._pose_action_motion_intent[side].update(
                    timestamp=motion_timestamp,
                    pose=base_pose,
                    interval_velocity=side_velocity,
                )
                source_by_side[side] = BimanualPoseTargetSource(
                    command_type="cartesian_pose",
                    value=base_pose,
                    frame_name=f"{side}_arm_base",
                    metadata={
                        **dict(effector.metadata),
                        "pose_action_interval_velocity": side_velocity,
                        "pose_action_motion_intent": motion_intent,
                        "rule_based_step_id": emission.directive.step_id,
                        "rule_based_rigid_pose_projection": dict(projection),
                    },
                )
            desired_base_pose_by_side[side] = base_pose.copy()
            tool_tcp_world = calibration.base_to_desk(base_pose, side)
            target_pose_by_side[side] = (
                retargeter.solver_world_target_from_tool_tcp_world_target(
                    side=side,
                    tool_tcp_world_target=tool_tcp_world,
                )
            )

        command = retargeter.build_command_for_target_pair(
            BimanualPoseTarget(
                timestamp=emission.action.timestamp,
                target_pose_by_side=target_pose_by_side,
                active_sides=active_sides,
                hard_hold_inactive_sides=True,
                world_to_base_by_side=world_to_base_by_side,
                source_by_side=source_by_side,
            )
        )
        status = str(command.metadata.get("collision_avoidance_status", ""))
        if status not in {"tracking", "boundary_active"}:
            raise RuntimeError(
                "coupled target solver refused the rule pose: "
                f"status={status or 'missing'} reason="
                f"{command.metadata.get('collision_avoidance_fallback_reason')}"
            )

        command.metadata.update(
            {
                key: _mutable_metadata_copy(value)
                for key, value in projected_command.metadata.items()
                if str(key).startswith(("rule_based_", "position_"))
                or str(key)
                in {
                    "resampling",
                    "left_source_index",
                    "right_source_index",
                    "left_source_timestamp",
                    "right_source_timestamp",
                    "interpolation_alpha",
                }
            }
        )

        # Preserve gripper/hand actions from the generic action resolver while
        # replacing both arm commands atomically with the coupled result.
        for name, effector in projected_command.commands.items():
            if name not in {"left_arm", "right_arm"}:
                command.add_command(effector.copy())
        command.metadata.update(
            {
                "rule_based_action_has_pose": True,
                "rule_based_coupled_pose_safety": True,
                "rule_based_pose_active_sides": list(active_sides),
                "rule_based_target_pose_base_by_side": {
                    side: value.copy()
                    for side, value in desired_base_pose_by_side.items()
                },
                "rule_based_rigid_pose_projection_by_side": _mutable_metadata_copy(
                    rigid_projection_by_side
                ),
                "rule_based_source_action": emission.action.source,
                "rule_based_collection_pose_frame_contract_hash": (
                    resolver.collection_contract.contract_hash
                ),
                "rule_based_runtime_pose_frame_contract_hash": (
                    resolver.runtime_contract.contract_hash
                ),
            }
        )
        return RuleBasedCollectionEmission(
            directive=emission.directive,
            action=emission.action,
            command=command,
        )

    def _reset_pose_action_motion_intent(self, reason: str) -> None:
        for classifier in self._pose_action_motion_intent.values():
            classifier.reset(str(reason))
        self._pose_action_motion_intent_step_id = None

    def _ensure_pose_action_motion_intent_step(self, step_id: str) -> None:
        normalized = str(step_id)
        if normalized == self._pose_action_motion_intent_step_id:
            return
        for classifier in self._pose_action_motion_intent.values():
            classifier.reset(f"rule_step_changed:{normalized}")
        self._pose_action_motion_intent_step_id = normalized

    def _start_initial_alignment_planning(
        self,
        emission: RuleBasedCollectionEmission,
        *,
        now: float,
        snapshots: Mapping[str, Mapping[str, Any]],
    ) -> None:
        planner = self._initial_alignment_planner
        if planner is None:
            raise RuntimeError("initial-alignment planner is not attached.")
        if self._closed:
            # ``close_before_session`` owns executor shutdown.  A late owner
            # loop callback must not turn that normal lifecycle edge into
            # ``cannot schedule new futures after shutdown`` and crash the
            # GUI; it is a terminal, fail-closed runtime state instead.
            self._aborted = True
            self._armed = False
            self._waiting_reason = "runtime_closed"
            self._clear_current()
            return
        if self._initial_alignment_future is not None:
            raise RuntimeError("initial-alignment planning is already active.")
        if self._initial_alignment_template is not None:
            raise RuntimeError("previous initial-alignment execution is still active.")
        # One collection plan may contain more than one smooth transition.  A
        # completed transition intentionally retains its plan long enough for
        # terminal-arrival accounting, so clear that completed cache before
        # submitting the next transition.  Otherwise the second transition
        # could consume the first transition's waypoints.
        self._initial_alignment_plan = None
        self._initial_alignment_waypoint_index = 0
        self._initial_alignment_effective_waypoint_count = 0
        self._initial_alignment_started_at = None
        self._initial_alignment_last_schedule_lag_sec = 0.0
        self._initial_alignment_cumulative_schedule_shift_sec = 0.0
        self._initial_alignment_skipped_clock_ticks = 0
        # RuleBasedInitialAlignmentPlanner owns this pre-dispatch transition.
        # The formal Episode can start many seconds after runtime Home while
        # writer warmup and operator/task-chain steps run.  Its command-inert
        # first waypoint must therefore use the fresh causal hardware qpos
        # from this execution cut, not the older Home admission snapshot.
        # The post-plan start-drift check below still rejects real movement
        # during planning; collision, TCP-envelope, and command guards remain
        # authoritative for every emitted waypoint.
        start = self._initial_alignment_start_qpos_pair(snapshots)
        self._rebase_motion_transport_after_manual_adjustment(start)
        target = {
            side: np.asarray(
                emission.directive.qpos_by_side.get(side, start[side]),
                dtype=float,
            ).copy()
            for side in ("left", "right")
        }
        hand_target = {
            side: np.asarray(values, dtype=float).reshape(-1).copy()
            for side, values in emission.directive.hand_qpos_by_side.items()
        }
        for side, values in hand_target.items():
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise RuleBasedInitialAlignmentPlanningError(
                    f"initial alignment has invalid {side} gripper target."
                )
        hand_start: dict[str, np.ndarray] = {}
        for side in hand_target:
            state = snapshots.get(side, {})
            actual = np.asarray(state.get("actual_hand_qpos"), dtype=float).reshape(-1)
            if actual.shape != (6,) or not np.all(np.isfinite(actual)):
                raise RuleBasedInitialAlignmentPlanningError(
                    f"initial alignment requires finite causal {side} gripper state."
                )
            hand_start[side] = actual.astype(np.float32, copy=True)
        if (
            str(self.source_selection.get("collection_mode") or "")
            == "rule_based_writing"
            and "right" in hand_target
            and not self._writing_right_gripper_hold_admitted(
                target=hand_target["right"],
                snapshots=snapshots,
            )
        ):
            raise RuleBasedInitialAlignmentPlanningError(
                "rule writing requires a feedback-gated right-gripper close "
                "before initial alignment"
            )
        self._initial_alignment_template = emission
        self._initial_alignment_hand_start_by_side = hand_start
        self._initial_alignment_hand_target_by_side = hand_target
        self._initial_alignment_planning_started_at = float(now)
        self._waiting_reason = "initial_alignment_planning"
        try:
            self._initial_alignment_future = self._initial_alignment_executor.submit(
                planner.plan,
                start_qpos_by_side=start,
                target_qpos_by_side=target,
            )
        except RuntimeError as exc:
            # Shutdown can race one final callback during Session teardown.
            # Preserve Hold/fail-closed semantics while keeping the GUI owner
            # loop alive long enough to publish the actionable terminal state.
            self._cancel_initial_alignment_planning()
            self._aborted = True
            self._armed = False
            self._waiting_reason = "initial_alignment_executor_unavailable"
            message = (
                "规则初始化规划器已关闭，未派发机械臂命令；"
                f"{type(exc).__name__}: {exc}"
            )
            self._establish_hold(message)
            self._request_failure_stop()
            append_message = getattr(self._control_ui, "append_message", None)
            if callable(append_message):
                append_message("RuleBased", message)

    def _prepare_initial_alignment_waypoint(
        self,
        now: float,
        snapshots: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        template = self._initial_alignment_template
        future = self._initial_alignment_future
        planner = self._initial_alignment_planner
        if template is None or future is None or planner is None:
            raise RuntimeError("initial-alignment planning state is incomplete.")
        if not future.done():
            started = self._initial_alignment_planning_started_at
            if (
                started is not None
                and float(now) - float(started)
                > float(planner.config.planning_timeout_sec)
            ):
                future.cancel()
                self._abort_initial_alignment(
                    "initial-alignment planning exceeded its bounded timeout"
                )
            self._waiting_reason = "initial_alignment_planning"
            return False
        try:
            plan = future.result()
        except Exception as exc:  # noqa: BLE001 - planning is fail closed.
            self._abort_initial_alignment(
                "initial-alignment planning failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise AssertionError("unreachable") from exc
        if self._initial_alignment_plan is None:
            actual = self._snapshot_qpos_pair(snapshots)
            drift_by_side = {
                side: np.abs(
                    actual[side]
                    - np.asarray(plan.start_qpos_by_side[side], dtype=float)
                )
                for side in ("left", "right")
            }
            drift_side, drift_joint_index = max(
                (
                    (side, int(np.argmax(values)))
                    for side, values in drift_by_side.items()
                ),
                key=lambda item: float(drift_by_side[item[0]][item[1]]),
            )
            drift = float(drift_by_side[drift_side][drift_joint_index])
            if drift > float(planner.start_drift_tolerance_rad):
                self._abort_initial_alignment(
                    "actual arm state drifted while the transition was being planned: "
                    f"side={drift_side} joint={drift_joint_index + 1} "
                    f"max_drift_deg={float(np.rad2deg(drift)):.6f}"
                )
            hand_drift = 0.0
            for side, hand_start in self._initial_alignment_hand_start_by_side.items():
                actual_hand = np.asarray(
                    snapshots.get(side, {}).get("actual_hand_qpos"), dtype=float
                ).reshape(-1)
                if actual_hand.shape != (6,) or not np.all(np.isfinite(actual_hand)):
                    self._abort_initial_alignment(
                        f"causal {side} gripper state was lost while planning"
                    )
                hand_drift = max(
                    hand_drift,
                    float(np.max(np.abs(actual_hand - hand_start))),
                )
            if hand_drift > float(
                self.config.initial_alignment_hand_start_drift_tolerance
            ):
                self._abort_initial_alignment(
                    "actual gripper state drifted while the transition was being "
                    f"planned: max_drift={hand_drift:.6f}"
                )
            self._initial_alignment_plan = plan
            self._initial_alignment_waypoint_index = 0
            self._initial_alignment_reacquire_samples = 0
            self._initial_alignment_started_at = float(now)
            self._initial_alignment_last_schedule_lag_sec = 0.0
            self._initial_alignment_cumulative_schedule_shift_sec = 0.0
            self._initial_alignment_skipped_clock_ticks = 0
            arm_count = max(1, len(plan.waypoints))
            hand_count = 1
            if self._initial_alignment_hand_target_by_side:
                # Count commands, not increments: command zero must reproduce
                # the measured hand state while the arm's zero-motion qref is
                # held by the runtime-Home waypoint.  The remaining commands keep
                # the original gripper_max_step bound and end exactly at the
                # replay target.
                hand_count = 1 + max(
                    0,
                    max(
                        int(
                            math.ceil(
                                float(
                                    np.max(
                                        np.abs(
                                            self._initial_alignment_hand_target_by_side[side]
                                            - self._initial_alignment_hand_start_by_side[side]
                                        )
                                    )
                                )
                                / float(self.config.gripper_max_step)
                            )
                        )
                        for side in self._initial_alignment_hand_target_by_side
                    ),
                )
            hand_duration_sec = (
                max(0, hand_count - 1) * float(planner.config.waypoint_dt_sec)
            )
            arm_duration_sec = float(plan.reference_timestamps_sec[-1])
            effective_duration_sec = max(arm_duration_sec, hand_duration_sec)
            self._initial_alignment_effective_waypoint_count = max(
                arm_count,
                int(
                    math.ceil(
                        effective_duration_sec
                        * float(planner.config.reference_hz)
                        - 1e-12
                    )
                )
                + 1,
            )
            self._waiting_reason = ""

        plan = self._initial_alignment_plan
        assert plan is not None
        effective_count = max(
            1,
            int(self._initial_alignment_effective_waypoint_count),
        )
        if self._initial_alignment_waypoint_index >= effective_count:
            self._initial_alignment_template = None
            self._initial_alignment_future = None
            self._initial_alignment_planning_started_at = None
            self._initial_alignment_started_at = None
            self._initial_alignment_hand_start_by_side = {}
            self._initial_alignment_hand_target_by_side = {}
            return True

        index = int(self._initial_alignment_waypoint_index)
        arm_index = min(index, max(0, len(plan.waypoints) - 1))
        waypoint = (
            plan.waypoints[arm_index]
            if plan.waypoints
            else plan.target_qpos_by_side
        )
        alpha = (
            1.0
            if effective_count <= 1
            else float(index) / float(effective_count - 1)
        )
        reference_timestamp = (
            float(plan.reference_timestamps_sec[arm_index])
            if arm_index < len(plan.reference_timestamps_sec)
            else float(plan.reference_timestamps_sec[-1])
        )
        next_reference_offset = (
            float(plan.reference_timestamps_sec[index + 1])
            if index + 1 < len(plan.reference_timestamps_sec)
            else (
                float(index + 1) / float(planner.config.reference_hz)
                if index + 1 < effective_count
                else None
            )
        )
        hand_waypoint = {
            side: (
                self._initial_alignment_hand_start_by_side[side]
                + alpha
                * (
                    self._initial_alignment_hand_target_by_side[side]
                    - self._initial_alignment_hand_start_by_side[side]
                )
            ).astype(np.float32, copy=True)
            for side in self._initial_alignment_hand_target_by_side
        }
        directive = replace(
            template.directive,
            qpos_by_side={
                side: np.asarray(waypoint[side], dtype=float).copy()
                for side in ("left", "right")
            },
            hand_qpos_by_side=hand_waypoint,
            metadata={
                **dict(template.directive.metadata),
                "initial_alignment_plan_hash": plan.semantic_content_hash,
                "initial_alignment_waypoint_index": index,
                "initial_alignment_waypoint_count": effective_count,
                "initial_alignment_arm_waypoint_count": len(plan.waypoints),
                "initial_alignment_hand_interpolation_alpha": alpha,
                "initial_alignment_dense_check_count": int(plan.dense_check_count),
                "initial_alignment_min_collision_distance_m": float(
                    plan.minimum_collision_distance_m
                ),
                "initial_alignment_reference_timestamp_sec": reference_timestamp,
                "initial_alignment_reference_next_offset_sec": next_reference_offset,
                "initial_alignment_schedule_lag_sec": float(
                    self._initial_alignment_last_schedule_lag_sec
                ),
                "initial_alignment_cumulative_schedule_shift_sec": float(
                    self._initial_alignment_cumulative_schedule_shift_sec
                ),
                "initial_alignment_skipped_clock_ticks": int(
                    self._initial_alignment_skipped_clock_ticks
                ),
            },
        )
        emission = self._emission_for_initial_alignment_directive(directive)
        emission.command.metadata.update(
            {
                "rule_based_initial_alignment_plan_hash": plan.semantic_content_hash,
                "rule_based_initial_alignment_waypoint_index": index,
                "rule_based_initial_alignment_waypoint_count": effective_count,
                "rule_based_initial_alignment_arm_waypoint_count": len(plan.waypoints),
                "rule_based_initial_alignment_hand_interpolation_alpha": alpha,
                "rule_based_initial_alignment_dense_check_count": int(
                    plan.dense_check_count
                ),
                "rule_based_initial_alignment_min_collision_distance_m": float(
                    plan.minimum_collision_distance_m
                ),
                "rule_based_initial_alignment_reference_timestamp_sec": (
                    reference_timestamp
                ),
                "rule_based_initial_alignment_reference_next_offset_sec": (
                    next_reference_offset
                ),
                "rule_based_initial_alignment_schedule_lag_sec": float(
                    self._initial_alignment_last_schedule_lag_sec
                ),
                "rule_based_initial_alignment_cumulative_schedule_shift_sec": float(
                    self._initial_alignment_cumulative_schedule_shift_sec
                ),
                "rule_based_initial_alignment_skipped_clock_ticks": int(
                    self._initial_alignment_skipped_clock_ticks
                ),
                "rule_based_scheduled_source_cut_timestamp": (
                    float(self._initial_alignment_started_at)
                    + reference_timestamp
                    if self._initial_alignment_started_at is not None
                    else float(now)
                ),
            }
        )
        self._initial_alignment_waypoint_index += 1
        self._adopt_emission(emission, now=now)
        return True

    def _emission_for_initial_alignment_directive(
        self,
        directive: RuleBasedCollectionDirective,
    ) -> RuleBasedCollectionEmission:
        action = self.adapter.action_for_directive(directive)
        command = self.adapter.resolver.resolve(action)
        return RuleBasedCollectionEmission(
            directive=directive,
            action=action,
            command=command,
        )

    @staticmethod
    def _snapshot_qpos_pair(
        snapshots: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            try:
                qpos = np.asarray(snapshots[side]["actual_qpos"], dtype=float).reshape(6)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"initial alignment requires finite causal {side} actual_qpos."
                ) from exc
            if not np.all(np.isfinite(qpos)):
                raise RuntimeError(
                    f"initial alignment requires finite causal {side} actual_qpos."
                )
            result[side] = qpos.copy()
        return result

    def _initial_alignment_start_qpos_pair(
        self,
        snapshots: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, np.ndarray]:
        """Select the fresh causal start for one certified transition.

        Runtime Home is the Session-admission anchor, not a motion reference
        that remains current across writer warmup or task-chain interaction.
        Rule motion has no generic StartupHomeContinuity gate and owns a
        bounded post-plan start-drift check, so the safest zero-motion first
        qref is the hardware feedback from the same execution cut that starts
        planning.
        """

        return self._snapshot_qpos_pair(snapshots)

    def _rebase_motion_transport_after_manual_adjustment(
        self,
        start_qpos_by_side: Mapping[str, np.ndarray],
    ) -> None:
        """Align command-planner state with post-calibration arm feedback.

        Manual adjustment changes the physical qpos without passing through
        the command backend.  RM65 position planners otherwise retain their
        last accepted pre-adjustment qcmd and turn the command-inert first
        alignment waypoint into a visible corrective kick.  Rebase all three
        consumers from one causal snapshot before any rule command is built.

        This helper never rearms a stopped backend.  A stopped or incomplete
        transport remains a hard failure that must go through runtime Home.
        """

        if not self._manual_adjustment_motion_rebase_pending:
            return
        components = self._components
        if components is None:
            raise RuntimeError(
                "post-calibration motion rebase requires attached real-arm components."
            )

        prepared: dict[str, tuple[object, object, np.ndarray]] = {}
        for side in ("left", "right"):
            try:
                qpos = np.asarray(start_qpos_by_side[side], dtype=float).reshape(6)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"post-calibration motion rebase requires finite {side} actual_qpos."
                ) from exc
            if not np.all(np.isfinite(qpos)):
                raise RuntimeError(
                    f"post-calibration motion rebase requires finite {side} actual_qpos."
                )
            command_backend = getattr(components, f"{side}_command_backend", None)
            preview_backend = getattr(components, f"{side}_preview_backend", None)
            if not callable(getattr(command_backend, "reset_motion_state", None)):
                raise RuntimeError(
                    f"post-calibration motion rebase requires {side} command backend."
                )
            if not callable(getattr(preview_backend, "set_local_qpos", None)):
                raise RuntimeError(
                    f"post-calibration motion rebase requires {side} preview backend."
                )
            is_stopped = getattr(command_backend, "is_emergency_stopped", None)
            if callable(is_stopped) and bool(is_stopped()):
                raise RuntimeError(
                    f"{side} command backend is stopped; runtime Home is required "
                    "before rule initialization."
                )
            prepared[side] = (command_backend, preview_backend, qpos.copy())

        for command_backend, preview_backend, qpos in prepared.values():
            preview_backend.set_local_qpos(qpos)
            command_backend.reset_motion_state(qpos)
        _sync_dual_arm_retargeter_for_real_control(components)
        self._manual_adjustment_motion_rebase_pending = False

        append_message = getattr(self._control_ui, "append_message", None)
        if callable(append_message):
            append_message(
                "Calibration",
                "已用标定后的双臂新鲜实测姿态重置位控规划锚点；"
                "初始化首帧不会沿用标定前命令状态。",
            )

    def _hand_qpos_from_state(
        self,
        state: Mapping[str, Any],
        *,
        side: str,
    ) -> np.ndarray | None:
        # A present ``joint_pos`` field is authoritative.  If it is malformed
        # (including None, the wrong length, or non-finite values), reject the
        # snapshot rather than silently trusting a scalar field from a
        # different stream.
        if "joint_pos" in state:
            raw = state.get("joint_pos")
            try:
                qpos = np.asarray(raw, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                return None
            if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
                return None
            return qpos.astype(np.float32, copy=True)

        # Scalar-only compatibility snapshots are accepted only when the
        # complete joint vector is absent.  The connected backend configuration
        # is authoritative for the active slot; DMTAC's canonical fallback is
        # slot 0, never the historical slot-1 guess or a stale state field.
        scalar = None
        for key in ("gripper_measured_scalar", "gripper_state_scalar"):
            try:
                value = float(state.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                scalar = value
                break
        if scalar is None:
            return None
        command_dim = int(self._hand_command_dim_by_side.get(side, 0))
        max_action = float(self._hand_max_action_by_side.get(side, 1.0))
        if command_dim < 0 or command_dim >= 6:
            return None
        if not np.isfinite(max_action) or max_action <= 0.0:
            return None
        result = np.zeros(6, dtype=np.float32)
        result[command_dim] = float(np.clip(scalar, 0.0, max_action))
        return result

    def _abort_initial_alignment(self, message: str) -> None:
        self._aborted = True
        self._armed = False
        self._waiting_reason = "initial_alignment_safety_rejected"
        self._clear_current()
        self._establish_hold(str(message))
        self._request_failure_stop()
        raise RuleBasedInitialAlignmentPlanningError(str(message))

    def _abort_coupled_pose(self, message: str) -> None:
        self._aborted = True
        self._armed = False
        self._waiting_reason = "coupled_pose_safety_rejected"
        self._clear_current()
        self._clear_replay_reference_schedule()
        self._establish_hold(str(message))
        self._request_failure_stop()
        raise RuntimeError(str(message))

    def _adopt_emission(
        self,
        emission: RuleBasedCollectionEmission,
        *,
        now: float,
    ) -> None:
        has_pose = bool(emission.action.arm_pose_by_side)
        phase_metadata = self._phase_metadata(emission)
        emission.command.metadata.update(
            {
                "rule_based_real_teleop_schema": RULE_BASED_REAL_TELEOP_SCHEMA_V1,
                "rule_based_step_id": emission.directive.step_id,
                "rule_based_directive_kind": emission.directive.kind,
                "rule_based_sequence_index": int(emission.directive.sequence_index),
                "rule_based_capture_enabled": bool(emission.recordable),
                "rule_based_source_selection": copy.deepcopy(self.source_selection),
                "rule_based_action_has_pose": has_pose,
                "rule_based_height_calibration": self._height_calibration_metadata(),
                **phase_metadata,
            }
        )
        if (
            emission.directive.step_id == _WRITING_HEIGHT_DESCENT_STEP_ID
            and not self._height_descent_planned_count_added
        ):
            try:
                descent_total = int(
                    emission.action.metadata["rule_based_height_descent_total"]
                )
            except (KeyError, TypeError, ValueError):
                descent_total = 0
            if descent_total > 0:
                try:
                    descent_duration_sec = float(
                        emission.action.metadata[
                            "rule_based_height_descent_duration_sec"
                        ]
                    )
                except (KeyError, TypeError, ValueError):
                    descent_duration_sec = math.nan
                if (
                    not np.isfinite(descent_duration_sec)
                    or descent_duration_sec <= 0.0
                ):
                    raise ValueError(
                        "calibrated height descent duration must be finite and positive"
                    )
                # The supplemental path is dispatched on the independent
                # 60 Hz position clock, while the formal writer continues at
                # action_hz.  Keep the stage reference counter at 60 Hz but do
                # not misreport every control reference as one dataset sample.
                descent_planned_samples = max(
                    1,
                    int(math.ceil(descent_duration_sec * self.action_hz)),
                )
                self._planned_sample_count += descent_planned_samples
                self._height_descent_planned_sample_count = descent_planned_samples
                self._stage_total_by_step[_WRITING_HEIGHT_DESCENT_STEP_ID] = descent_total
                self._height_descent_planned_count_added = True
        self._update_stage_display(emission)
        self._current = emission
        self._target_wait_started_at = None
        self._current_dispatched = False
        self._current_sample_accepted = bool(
            emission.recordable
            and int(emission.directive.sequence_index)
            in self._admitted_capture_sequences
        )
        self._target_arrival_confirmed = False
        self._target_arrival_snapshots = {}
        self._settled_samples = 0
        reference_next_offset = emission.action.metadata.get(
            "rule_based_reference_next_offset_sec"
        )
        if reference_next_offset is not None:
            if self._replay_reference_stage_started_at is None:
                raise RuntimeError("replay reference stage lacks a monotonic start cut")
            self._next_due_at = self._replay_reference_stage_started_at + float(
                reference_next_offset
            )
        elif emission.directive.kind == "initialize":
            initial_next_offset = emission.directive.metadata.get(
                "initial_alignment_reference_next_offset_sec"
            )
            started_at = self._initial_alignment_started_at
            self._next_due_at = (
                None
                if initial_next_offset is None or started_at is None
                else float(started_at) + float(initial_next_offset)
            )
        else:
            action_period = emission.action.metadata.get(
                "rule_based_action_period_sec"
            )
            if action_period is None:
                period_sec = self._emission_period(emission)
            else:
                try:
                    period_sec = float(action_period)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "rule action period must be finite and positive"
                    ) from exc
                if not np.isfinite(period_sec) or period_sec <= 0.0:
                    raise ValueError(
                        "rule action period must be finite and positive"
                    )
            self._next_due_at = now + period_sec

    def _emission_period(self, emission: RuleBasedCollectionEmission) -> float:
        reference_timestamp = emission.action.metadata.get(
            "rule_based_reference_timestamp_sec"
        )
        reference_next_offset = emission.action.metadata.get(
            "rule_based_reference_next_offset_sec"
        )
        if reference_timestamp is not None and reference_next_offset is not None:
            interval = float(reference_next_offset) - float(reference_timestamp)
            if np.isfinite(interval) and interval > 0.0:
                return interval
            raise ValueError("replay reference interval must be finite and positive")
        directive = emission.directive
        if directive.kind == "replay" and directive.group_key is not None:
            group = self.trajectory_groups[directive.group_key]
            index = int(directive.replay_frame_index or 0)
            if index + 1 < int(group.frame_count):
                interval = group.relative_time_sec(index + 1) - group.relative_time_sec(index)
                if np.isfinite(interval) and interval > 0.0:
                    return float(interval)
            return 1.0 / float(group.action_hz)
        if (
            directive.kind == "initialize"
            and self._initial_alignment_planner is not None
        ):
            next_offset = directive.metadata.get(
                "initial_alignment_reference_next_offset_sec"
            )
            timestamp = directive.metadata.get(
                "initial_alignment_reference_timestamp_sec"
            )
            if next_offset is not None and timestamp is not None:
                return max(
                    1.0 / float(self._initial_alignment_planner.config.reference_hz),
                    float(next_offset) - float(timestamp),
                )
            return 1.0 / float(self._initial_alignment_planner.config.reference_hz)
        return 1.0 / self.action_hz

    @staticmethod
    def _phase_metadata_for_step(
        step_id: str,
        directive_kind: str,
        collection_mode: str = "rule_based_writing",
    ) -> dict[str, Any]:
        step = str(step_id)
        if str(collection_mode) == "full_trajectory_replay":
            if directive_kind == "initialize":
                return {
                    "collection_task_stage_id": "execute_task",
                    "collection_detail_stage_id": None,
                    "rule_based_phase": "full_replay_initial_alignment",
                    "rule_based_phase_label": "完整轨迹初始化（不计入正式采集）",
                    "rule_based_phase_execution": "initial_alignment",
                }
            return {
                "collection_task_stage_id": "execute_task",
                "collection_detail_stage_id": "execute_task.full_replay",
                "rule_based_phase": "full_trajectory_replay",
                "rule_based_phase_label": "完整轨迹 Replay",
                "rule_based_phase_execution": directive_kind,
            }
        if step == "right_write.pen_down":
            return {
                "collection_task_stage_id": "right_write",
                "collection_detail_stage_id": step,
                "rule_based_phase": "pen_down_replay",
                "rule_based_phase_label": "靠近纸面",
                "rule_based_phase_execution": "replay",
            }
        if step == _WRITING_HEIGHT_DESCENT_STEP_ID:
            return {
                "collection_task_stage_id": "right_write",
                "collection_detail_stage_id": "right_write.pen_down",
                "rule_based_phase": "pen_down_height_descent",
                "rule_based_phase_label": "靠近纸面（高度补偿）",
                "rule_based_phase_execution": "rule-base_z_descent",
            }
        if step == "right_write.write":
            return {
                "collection_task_stage_id": "right_write",
                "collection_detail_stage_id": step,
                "rule_based_phase": "write_rule_base",
                "rule_based_phase_label": "书写",
                "rule_based_phase_execution": "rule-base",
            }
        if step == "right_write.pen_up":
            return {
                "collection_task_stage_id": "right_write",
                "collection_detail_stage_id": step,
                "rule_based_phase": "pen_up_relative_replay",
                "rule_based_phase_label": "离开纸面",
                "rule_based_phase_execution": "relative_replay",
            }
        if directive_kind == "initialize":
            return {
                "collection_task_stage_id": "right_write",
                "collection_detail_stage_id": None,
                "rule_based_phase": "initial_alignment",
                "rule_based_phase_label": "初始化（不计入正式采集）",
                "rule_based_phase_execution": "initial_alignment",
            }
        return {
            "collection_task_stage_id": "right_write",
            "collection_detail_stage_id": step,
            "rule_based_phase": "unknown",
            "rule_based_phase_label": step,
            "rule_based_phase_execution": directive_kind,
        }

    def _phase_metadata(self, emission: RuleBasedCollectionEmission) -> dict[str, Any]:
        metadata = self._phase_metadata_for_step(
            emission.directive.step_id,
            emission.directive.kind,
            str(
                self.source_selection.get("collection_mode")
                or "rule_based_writing"
            ),
        )
        metadata.update(
            {
                "rule_based_capture_enabled": bool(emission.recordable),
                "rule_based_initial_alignment_capture": bool(
                    emission.directive.kind == "initialize"
                ),
            }
        )
        phase_sources = self.source_selection.get("phases", {})
        if isinstance(phase_sources, Mapping):
            stage = metadata.get("rule_based_phase")
            stage_key = {
                "pen_down_replay": "pen_down",
                "pen_down_height_descent": "pen_down",
                "write_rule_base": "write",
                "pen_up_relative_replay": "pen_up",
                "full_trajectory_replay": "full_replay",
            }.get(str(stage))
            if stage_key is not None and isinstance(phase_sources.get(stage_key), Mapping):
                metadata["rule_based_phase_provenance"] = copy.deepcopy(
                    phase_sources[stage_key]
                )
        return metadata

    def _update_stage_display(self, emission: RuleBasedCollectionEmission) -> None:
        directive = emission.directive
        metadata = self._phase_metadata(emission)
        self._stage_label = str(metadata.get("rule_based_phase_label") or directive.step_id)
        if directive.kind == "initialize":
            self._stage_index = None
            waypoint_index = int(
                directive.metadata.get("initial_alignment_waypoint_index", 0)
            )
            waypoint_count = max(
                1,
                int(directive.metadata.get("initial_alignment_waypoint_count", 1)),
            )
            self._stage_frame_index = min(waypoint_count, waypoint_index + 1)
            self._stage_frame_total = waypoint_count
            self._collection_position = (
                "初始化：夹爪与双臂平滑对齐（不计入采集） · 路点 "
                f"{self._stage_frame_index} / {waypoint_count}"
            )
            return
        stage_index = {
            "right_write.pen_down": 1,
            _WRITING_HEIGHT_DESCENT_STEP_ID: 1,
            "right_write.write": 2,
            "right_write.pen_up": 3,
        }.get(directive.step_id)
        full_replay = (
            self.source_selection.get("collection_mode")
            == "full_trajectory_replay"
        )
        if full_replay:
            stage_index = 1
        self._stage_index = stage_index
        if directive.kind == "replay":
            frame_index = int(directive.replay_frame_index or 0)
            total = int(self._stage_total_by_step.get(directive.step_id, 0))
        else:
            frame_index = int(directive.metadata.get("segment_emitted_steps", 1)) - 1
            total = int(
                emission.action.metadata.get(
                    "rule_based_height_descent_total",
                    self._stage_total_by_step.get(directive.step_id, 0),
                )
            )
        self._stage_frame_index = max(1, frame_index + 1)
        self._stage_frame_total = max(1, total)
        if full_replay:
            self._collection_position = (
                f"完整轨迹 · 帧 {self._stage_frame_index} / "
                f"{self._stage_frame_total}"
            )
        else:
            self._collection_position = (
                f"阶段 {stage_index or '?'} / 3 · 帧 {self._stage_frame_index} / "
                f"{self._stage_frame_total}"
            )

    @staticmethod
    def _is_final_initial_alignment_waypoint(
        emission: RuleBasedCollectionEmission,
    ) -> bool:
        if emission.directive.kind != "initialize":
            return False
        metadata = emission.command.metadata
        try:
            index = int(metadata["rule_based_initial_alignment_waypoint_index"])
            count = int(metadata["rule_based_initial_alignment_waypoint_count"])
        except (KeyError, TypeError, ValueError):
            return True
        return count < 1 or index + 1 >= count

    def _initial_alignment_tracking_within_horizon(
        self,
        emission: RuleBasedCollectionEmission,
        snapshots: Mapping[str, Mapping[str, Any]],
        *,
        now: float,
    ) -> bool:
        planner = self._initial_alignment_planner
        if planner is None:
            self._abort_session_step(
                "initial-alignment tracking lost its certified planner"
            )
            return False
        max_lag = 0.0
        for side, target in emission.directive.qpos_by_side.items():
            actual = np.asarray(
                snapshots.get(side, {}).get("actual_qpos"),
                dtype=float,
            ).reshape(-1)
            target_qpos = np.asarray(target, dtype=float).reshape(-1)
            if (
                actual.shape != (6,)
                or target_qpos.shape != (6,)
                or not np.all(np.isfinite(actual))
                or not np.all(np.isfinite(target_qpos))
            ):
                self._abort_session_step(
                    "initial-alignment measured tracking feedback is missing, "
                    f"non-finite, or malformed for {side}"
                )
                return False
            max_lag = max(max_lag, float(np.max(np.abs(actual - target_qpos))))
        if max_lag <= float(planner.streaming_max_lag_rad):
            return True
        self._waiting_reason = "initial_alignment_tracking_lag"
        if (
            self._target_wait_started_at is not None
            and float(now) - float(self._target_wait_started_at)
            > float(self.config.target_timeout_sec)
        ):
            self._abort_session_step(
                "initial-alignment measured tracking lag exceeded its bounded "
                f"horizon for {self.config.target_timeout_sec:.3f}s"
            )
        return False

    def _rebase_late_initial_alignment_schedule(self, *, now: float) -> None:
        """Slip the certified qref clock instead of chasing stale deadlines.

        Every planned qref remains ordered and is still dispatched exactly
        once.  When the owner loop wakes late, move the remaining clock by the
        observed lag so the next certified qref receives a full reference
        interval.  This prevents a late camera/CPU cycle from being followed
        by near-adjacent catch-up commands while preserving the offline C2,
        collision and joint-limit certificate.
        """

        planner = self._initial_alignment_planner
        started_at = self._initial_alignment_started_at
        scheduled_at = self._next_due_at
        if planner is None or started_at is None or scheduled_at is None:
            raise RuntimeError("initial-alignment schedule is incomplete")
        decision = resolve_periodic_tick(
            scheduled_monotonic_time=float(scheduled_at),
            wakeup_monotonic_time=float(now),
            period_sec=1.0 / float(planner.config.reference_hz),
        )
        lag_sec = float(decision.raw_wakeup_lag_sec)
        self._initial_alignment_last_schedule_lag_sec = lag_sec
        self._initial_alignment_skipped_clock_ticks += int(
            decision.skipped_tick_count
        )
        if lag_sec <= 0.0:
            return
        self._initial_alignment_started_at = float(started_at) + lag_sec
        self._initial_alignment_cumulative_schedule_shift_sec += lag_sec

    def _advance_due(
        self,
        now: float,
        *,
        snapshots: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        assert self._current is not None
        if self._current.directive.kind == "initialize":
            if self._is_final_initial_alignment_waypoint(self._current):
                return False
            waypoint_index = int(
                self._current.command.metadata.get(
                    "rule_based_initial_alignment_waypoint_index",
                    0,
                )
            )
            if waypoint_index == 0:
                # The first certified waypoint is the same causal qpos cut
                # used to plan this transition.  Repeat that zero-motion qref
                # for a bounded dwell before advancing, so a backend that
                # entered idle quick-stop between Episodes can reacquire the
                # existing position stream without a jump to waypoint 1.
                planner = self._initial_alignment_planner
                if planner is None:
                    self._abort_session_step(
                        "initial-alignment reacquisition lost its certified planner"
                    )
                    return False
                within_start = True
                for side, target in self._current.directive.qpos_by_side.items():
                    actual = np.asarray(
                        snapshots.get(side, {}).get("actual_qpos"),
                        dtype=float,
                    ).reshape(-1)
                    target_qpos = np.asarray(target, dtype=float).reshape(-1)
                    if (
                        actual.shape != (6,)
                        or target_qpos.shape != (6,)
                        or not np.all(np.isfinite(actual))
                        or not np.all(np.isfinite(target_qpos))
                    ):
                        self._abort_session_step(
                            "initial-alignment zero-motion reacquisition lost "
                            f"finite {side} qpos feedback"
                        )
                        return False
                    within_start = within_start and bool(
                        np.max(np.abs(actual - target_qpos))
                        <= float(planner.start_drift_tolerance_rad)
                    )
                self._initial_alignment_reacquire_samples = (
                    self._initial_alignment_reacquire_samples + 1
                    if self._current_dispatched and within_start
                    else 0
                )
                if self._initial_alignment_reacquire_samples < int(
                    self.config.settle_consecutive_samples
                ):
                    self._waiting_reason = "initial_alignment_reacquire_dwell"
                    if (
                        self._target_wait_started_at is not None
                        and float(now) - float(self._target_wait_started_at)
                        > float(self.config.target_timeout_sec)
                    ):
                        self._abort_session_step(
                            "initial-alignment zero-motion qref reacquisition "
                            f"exceeded {self.config.target_timeout_sec:.3f}s"
                        )
                    return False
            should_advance = bool(
                self._current_dispatched
                and self._next_due_at is not None
                and now >= self._next_due_at
                and self._initial_alignment_tracking_within_horizon(
                    self._current,
                    snapshots,
                    now=now,
                )
            )
            if should_advance:
                self._rebase_late_initial_alignment_schedule(now=float(now))
            return should_advance
        if self._current.directive.segment_terminal:
            return False
        return bool(
            self._current_dispatched
            and self._next_due_at is not None
            and now >= self._next_due_at
        )

    @staticmethod
    def _certified_arm_hold_emission(
        emission: RuleBasedCollectionEmission,
    ) -> RuleBasedCollectionEmission:
        """Snapshot only the accepted arm command needed by pause Hold.

        RobotAction/resampling metadata is intentionally immutable and may
        contain ``mappingproxy`` values.  A generic deepcopy both traversed
        the full debug/provenance graph on every owner-loop frame and crashed
        once the motion-intent metadata became immutable.  The shared runtime
        subset helper already owns the command-safe value copy required here.
        """

        arm_command = command_effector_kind_subset(emission.command, kind="arm")
        if arm_command is None:
            raise RuntimeError("accepted rule emission has no arm command to hold")
        return replace(emission, command=arm_command)

    def _paused_arm_hold_override(
        self,
        *,
        reason: str,
        now: float,
    ) -> TeleopCommandOverride:
        """Keep the last accepted arm qref alive without advancing the rule.

        This is a dispatch-preserving pause, not a second motion controller.
        The canonical Session/Rig path still validates and sends the exact
        accepted arm target, while capture and cursor advancement remain off.
        """

        held = self._last_dispatched_emission
        if held is None:
            raise RuntimeError(
                "rule-based execution has no certified command to hold during "
                "a recoverable pause"
            )
        paused_hold = self._certified_arm_hold_emission(held)
        paused_hold.command.metadata.update(
            {
                "rule_based_pause_hold": True,
                "rule_based_pause_reason": str(reason),
                "rule_based_capture_enabled": False,
                "dataset_recording_eligible": False,
            }
        )
        self._prepare_live_dispatch(paused_hold, float(now))
        self._paused_hold_emission = paused_hold
        self._waiting_reason = str(reason)
        self._publish_status(recording_ready=False)
        return TeleopCommandOverride(paused_hold.command)

    @staticmethod
    def _is_transient_arm_read_failure(reason: str) -> bool:
        prefix = "causal_arm_feedback_not_ready:"
        if not str(reason).startswith(prefix):
            return False
        details = str(reason)[len(prefix) :].split(",")
        return bool(details) and all(
            detail.partition("=")[0] in {"left", "right"}
            and detail.partition("=")[1] == "="
            and detail.partition("=")[2] == "hardware_read_failed"
            for detail in details
        )

    def _recoverable_arm_feedback_pause(self, *, reason: str, now: float) -> bool:
        """Bound one isolated hardware-state read failure without hiding faults."""

        if not self._is_transient_arm_read_failure(reason):
            self._arm_feedback_pause_started_at = None
            return False
        if self._arm_feedback_pause_started_at is None:
            self._arm_feedback_pause_started_at = float(now)
        return (
            float(now) - float(self._arm_feedback_pause_started_at)
            <= float(self.config.arm_feedback_recovery_timeout_sec)
        )

    def after_session_step(self, *, step_result: object, now: float) -> None:
        emission = self._frame_emission
        paused_hold = self._paused_hold_emission
        attempted_emission = emission or paused_hold
        state = str(
            getattr(getattr(step_result, "state", None), "value", getattr(step_result, "state", ""))
            or ""
        ).upper()
        faults = tuple(getattr(step_result, "faults", ()) or ())
        if attempted_emission is not None and (faults or state in {"FAULT", "ESTOP"}):
            detail = str(getattr(faults[0], "message", faults[0])) if faults else state
            self._abort_session_step("session rejected the active rule frame: " + detail)
        elif attempted_emission is not None and not bool(
            getattr(step_result, "command_dispatched", False)
        ):
            # The generic StartupHomeContinuity gate is intentionally not
            # installed for rule-owned motion.  A rule frame that reaches this
            # branch therefore failed a real Session/guard/dispatch boundary;
            # silently replaying it would create an unbounded duplicate-qref
            # loop and could hide a hardware or command-owner fault.
            self._abort_session_step("session did not dispatch the active rule frame")
        elif paused_hold is not None:
            # This dispatch exists only to preserve the certified position
            # stream and supply an arm receipt to collection resume warmup.
            # Validate every backend receipt, but never mutate the rule cursor,
            # arrival state, or sample-admission state from a paused frame.
            outcome, detail = self._formal_dispatch_outcome(
                paused_hold,
                step_result,
                float(now),
            )
            if outcome == "rejected":
                self._abort_session_step(
                    "backend rejected the paused rule hold frame: " + detail
                )
            elif outcome == "accepted":
                self._last_dispatched_emission = self._certified_arm_hold_emission(
                    paused_hold
                )
                self._dispatch_retry_started_at = None
                self._waiting_reason = str(
                    paused_hold.command.metadata.get(
                        "rule_based_pause_reason",
                        "formal_writer_paused",
                    )
                )
        elif emission is not None:
            # This is the irreversible end of the bootstrap exception: the
            # Session has handed an arm command to the Rig.  A later hand
            # retry or backend rejection must never reopen pre-dispatch mode.
            self._execution_begun = True
            outcome, detail = self._formal_dispatch_outcome(emission, step_result, float(now))
            if outcome == "rejected":
                self._abort_session_step("backend rejected the active rule frame: " + detail)
            elif outcome == "retry":
                self._current_dispatched = False
                self._waiting_reason = "backend_dispatch_retry:" + detail
                if self._dispatch_retry_started_at is None:
                    self._dispatch_retry_started_at = float(now)
                elif float(now) - self._dispatch_retry_started_at > float(
                    self.config.target_timeout_sec
                ):
                    self._abort_session_step(
                        "backend retry timed out for the active rule frame: " + detail
                    )
            else:
                self._current_dispatched = True
                self._last_dispatched_emission = self._certified_arm_hold_emission(
                    emission
                )
                self._dispatch_retry_started_at = None
                if self._waiting_reason != "initial_alignment_tracking_lag":
                    self._waiting_reason = ""
                requires_arrival = bool(
                    emission.directive.segment_terminal
                    or self._is_final_initial_alignment_waypoint(emission)
                )
                if emission.directive.kind == "initialize":
                    if self._target_wait_started_at is None:
                        self._target_wait_started_at = float(now)
                if requires_arrival:
                    if self._target_wait_started_at is None:
                        self._target_wait_started_at = float(now)
                    try:
                        self._observe_target_arrival(emission, float(now))
                    except Exception as exc:  # noqa: BLE001 - remain live but fail closed.
                        self._abort_session_step(
                            "active rule target feedback/arrival failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
        if self._completed:
            request_stop = getattr(
                self._control_ui,
                "request_rule_completion_stop",
                None,
            )
            if callable(request_stop) and not self._completion_stop_requested:
                request_stop()
                self._completion_stop_requested = True
        self._publish_status()

    def after_session_error(self, *, error: BaseException, now: float) -> None:
        del now
        if self._armed or self._execution_begun or self._frame_emission is not None:
            self._abort_session_step(
                "Session raised during rule execution: "
                f"{type(error).__name__}: {error}"
            )
            self._publish_status()

    def _abort_session_step(self, message: str) -> None:
        detail = str(message)
        self._aborted = True
        self._armed = False
        self._waiting_reason = "session_dispatch_rejected"
        self._clear_current()
        self._clear_replay_reference_schedule()
        self._establish_hold(detail)
        append_message = getattr(self._control_ui, "append_message", None)
        if callable(append_message):
            append_message("RuleBased", "规则执行已安全终止：" + detail)
        self._request_failure_stop()

    def _request_failure_stop(self) -> None:
        request_stop = getattr(
            self._control_ui,
            "request_rule_failure_stop",
            None,
        )
        if callable(request_stop):
            request_stop()

    def _formal_dispatch_outcome(
        self,
        emission: RuleBasedCollectionEmission,
        step_result: object,
        now: float,
    ) -> tuple[str, str]:
        if not hasattr(step_result, "dispatch_receipts"):
            # Compatibility for older test/application adapters. Production
            # SessionStepResult always carries the per-effector receipt map.
            return "accepted", "legacy_session_result"
        receipts = dict(getattr(step_result, "dispatch_receipts", {}) or {})
        for effector_name, effector_command in emission.command.commands.items():
            if effector_name not in receipts:
                return "rejected", f"missing receipt for {effector_name}"
            receipt = receipts[effector_name]
            explicit_accepted = getattr(receipt, "accepted", None)
            if explicit_accepted is False:
                status = str(getattr(receipt, "status", "rejected") or "rejected")
                if (
                    effector_name.endswith("_hand")
                    and status in DMTAC_GRIPPER_RECOVERABLE_SEND_STATUSES
                ):
                    return "retry", f"{effector_name}:{status}"
                return "rejected", f"{effector_name}:{status}"
            if receipt is not False:
                continue
            if not effector_name.endswith("_hand"):
                return "rejected", f"{effector_name}:explicit false receipt"
            side = effector_name.removesuffix("_hand")
            backend = getattr(self._components, f"{side}_hand_backend", None)
            status = str(getattr(backend, "_last_send_status", "") or "unknown")
            if status in DMTAC_GRIPPER_RECOVERABLE_SEND_STATUSES:
                return "retry", f"{effector_name}:{status}"
            if status == "skipped_threshold":
                ready, reason, snapshots = self._runtime_ready(now)
                measured = (
                    np.asarray(
                        snapshots.get(side, {}).get("actual_hand_qpos"),
                        dtype=float,
                    ).reshape(-1)
                    if ready
                    else np.asarray([], dtype=float)
                )
                values = np.asarray(effector_command.value, dtype=float).reshape(-1)
                dim = int(self._hand_command_dim_by_side.get(side, 0))
                target = (
                    float(values[0])
                    if values.shape == (1,)
                    else float(values[dim])
                    if values.shape == (6,)
                    else math.nan
                )
                if (
                    measured.shape == (6,)
                    and np.all(np.isfinite(measured))
                    and np.isfinite(target)
                    and abs(float(measured[dim]) - target)
                    <= float(self.config.gripper_qpos_tolerance)
                ):
                    continue
                return "retry", (
                    f"{effector_name}:skipped_threshold:"
                    f"{reason or 'causal_state_not_at_target'}"
                )
            return "rejected", f"{effector_name}:{status}"
        return "accepted", "all_effectors_accepted"

    def _observe_target_arrival(
        self,
        emission: RuleBasedCollectionEmission,
        now: float,
    ) -> None:
        ready, reason, snapshots = self._runtime_ready(now)
        if not ready:
            if self._recoverable_arm_feedback_pause(reason=reason, now=float(now)):
                self._waiting_reason = reason
                if self._replay_reference_pause_started_at is None:
                    self._replay_reference_pause_started_at = float(now)
                return
            raise RuntimeError("target arrival feedback unavailable: " + reason)
        self._arm_feedback_pause_started_at = None
        reached = self._target_reached(emission, snapshots)
        self._settled_samples = self._settled_samples + 1 if reached else 0
        if self._settled_samples >= int(self.config.settle_consecutive_samples):
            self._target_arrival_confirmed = True
            self._target_arrival_snapshots = {
                side: {
                    key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
                    for key, value in state.items()
                }
                for side, state in snapshots.items()
            }
            if not emission.recordable or self._current_sample_accepted:
                self._finish_arrived_current(now)
            return
        if (
            self._target_wait_started_at is not None
            and now - self._target_wait_started_at > float(self.config.target_timeout_sec)
        ):
            raise TimeoutError(
                f"rule-based target arrival timed out at step {emission.directive.step_id!r}."
            )

    def _finish_arrived_current(self, now: float) -> None:
        emission = self._current
        if emission is None or not self._target_arrival_confirmed:
            return
        if emission.directive.kind == "initialize":
            self._establish_deferred_progress_arm_fix(emission, now=float(now))
        if emission.directive.segment_terminal:
            self._publish_boundary(
                emission.directive.step_id,
                self._target_arrival_snapshots,
                float(now),
            )
        if emission.directive.step_id == _WRITING_HEIGHT_DESCENT_STEP_ID:
            target_z = emission.action.metadata.get(
                "rule_based_height_descent_target_z_m"
            )
            try:
                target_z_value = float(target_z)
            except (TypeError, ValueError):
                target_z_value = math.nan
            if not np.isfinite(target_z_value):
                raise RuntimeError(
                    "arrived calibrated height descent lacks a finite target Z"
                )
            self._writing_height_target_z_m = target_z_value
        self._clear_current()

    def _establish_deferred_progress_arm_fix(
        self,
        emission: RuleBasedCollectionEmission,
        *,
        now: float,
    ) -> None:
        contract = self._deferred_progress_arm_contract
        if contract is None or self._deferred_progress_arm_fix_established:
            return
        metadata = emission.command.metadata
        try:
            identity = (
                str(metadata["rule_based_initial_alignment_plan_hash"]),
                int(metadata["rule_based_initial_alignment_waypoint_index"]),
                int(metadata["rule_based_initial_alignment_waypoint_count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "final initialization waypoint has no certified identity"
            ) from exc
        _plan_hash, waypoint_index, waypoint_count = identity
        if waypoint_count < 1 or waypoint_index + 1 < waypoint_count:
            return
        if waypoint_index + 1 != waypoint_count:
            raise RuntimeError(
                "final initialization waypoint identity is out of bounds: "
                f"index={waypoint_index} count={waypoint_count}"
            )
        if self._filtered_initial_alignment_identity != identity:
            raise RuntimeError(
                "final initialization arrival has no matching Session-filtered "
                "command receipt"
            )
        controller = getattr(self._components, "fix_arm_controller", None)
        if tuple(getattr(controller, "fixed_sides", ()) or ()):
            raise RuntimeError(
                "FixArm authority changed before final initialization arrival"
            )
        for side in contract.target_fixed_sides:
            command = self._filtered_initial_alignment_arm_by_side.get(side)
            if command is None:
                raise RuntimeError(
                    f"verified final initialization command is missing {side}_arm"
                )
            transition = controller.freeze_arm_only_from_command(
                side=side,
                command=command,
                now=float(now),
                snapshot_origin="rule_initial_alignment_arrived",
            )
            if not transition.ok:
                raise RuntimeError(transition.message)
        actual_fixed = tuple(getattr(controller, "fixed_sides", ()) or ())
        if actual_fixed != contract.target_fixed_sides:
            raise RuntimeError(
                "deferred progress Fix authority mismatch: "
                f"expected={contract.target_fixed_sides!r} actual={actual_fixed!r}"
            )
        self._deferred_progress_arm_fix_established = True
        self._waiting_reason = "progress_entry_arm_fix_established"

    def _writing_right_gripper_force_stop_reached(
        self,
        *,
        side: str,
        target: np.ndarray,
        snapshots: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        """Accept a feedback-gated closed/contact stop as a closed target.

        Status 6 is the explicit force-stop receipt.  Status 2 is also valid
        for a pen-limited close, but only when the hand state proves that the
        causal command was the closed endpoint and the measured hand position
        is no longer at the open endpoint.  This keeps a stalled open hand
        from becoming a write owner merely because it reports closed-side
        idle.
        """

        if (
            side != "right"
            or str(self.source_selection.get("collection_mode") or "")
            != "rule_based_writing"
            or target.shape != (6,)
        ):
            return False
        dim = int(self._hand_command_dim_by_side.get("right", 0))
        max_action = float(self._hand_max_action_by_side.get("right", 1.0))
        if abs(float(target[dim]) - max_action) > float(
            self.config.manual_gripper_endpoint_tolerance
        ):
            return False
        state = snapshots.get("right", {}).get("hand_state")
        if not isinstance(state, Mapping):
            return False
        actual_hand_qpos = np.asarray(
            snapshots.get("right", {}).get("actual_hand_qpos"),
            dtype=float,
        ).reshape(-1)
        raw_status = state.get("gripper_status_code", state.get("status_code"))
        raw_error = state.get("gripper_error_code", state.get("error_code"))
        try:
            status_code = int(raw_status)
            error_code = 0 if raw_error is None else int(raw_error)
        except (TypeError, ValueError):
            return False
        if error_code != 0 or bool(state.get("gripper_is_moving")):
            return False
        telemetry_error = str(state.get("telemetry_error") or "").strip()
        if telemetry_error and not telemetry_error.lower().startswith("tactile_"):
            return False
        if status_code == 6:
            return True
        if status_code != 2:
            return False
        command_value = state.get(
            "gripper_commanded_scalar",
            state.get("gripper_last_send_target_scalar"),
        )
        try:
            command_value = float(command_value)
            actual_value = float(actual_hand_qpos[dim])
        except (TypeError, ValueError, IndexError):
            return False
        if (
            not np.isfinite(command_value)
            or not np.isfinite(actual_value)
            or abs(command_value - max_action)
            > float(self.config.manual_gripper_endpoint_tolerance)
        ):
            return False
        return actual_value > max(
            float(self.config.manual_gripper_endpoint_tolerance),
            1e-4,
        )

    def _writing_right_gripper_hold_admitted(
        self,
        *,
        target: np.ndarray,
        snapshots: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        """Require a stable close/contact receipt before any writing arm move."""

        expected = np.asarray(target, dtype=float).reshape(-1)
        actual = np.asarray(
            snapshots.get("right", {}).get("actual_hand_qpos"),
            dtype=float,
        ).reshape(-1)
        if (
            expected.shape != (6,)
            or actual.shape != (6,)
            or not np.all(np.isfinite(expected))
            or not np.all(np.isfinite(actual))
        ):
            return False
        if self._writing_right_gripper_force_stop_reached(
            side="right",
            target=expected,
            snapshots=snapshots,
        ):
            return True
        dim = int(self._hand_command_dim_by_side.get("right", 0))
        max_action = float(self._hand_max_action_by_side.get("right", 1.0))
        if (
            abs(float(expected[dim]) - max_action)
            > float(self.config.manual_gripper_endpoint_tolerance)
            or abs(float(actual[dim]) - max_action)
            > float(self.config.manual_gripper_endpoint_tolerance)
        ):
            return False
        state = snapshots.get("right", {}).get("hand_state")
        if not isinstance(state, Mapping):
            return False
        raw_status = state.get("gripper_status_code", state.get("status_code"))
        raw_error = state.get("gripper_error_code", state.get("error_code"))
        try:
            status_code = None if raw_status is None else int(raw_status)
            error_code = 0 if raw_error is None else int(raw_error)
        except (TypeError, ValueError):
            return False
        return bool(
            error_code == 0
            and status_code not in {4, 5}
            and not bool(state.get("gripper_is_moving"))
        )

    def _target_reached(
        self,
        emission: RuleBasedCollectionEmission,
        snapshots: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        directive = emission.directive
        if directive.kind == "initialize":
            planner = self._initial_alignment_planner
            tolerance = (
                float(self.config.qpos_tolerance_rad)
                if planner is None
                else float(planner.arrival_tolerance_rad)
            )
            for side, target in directive.qpos_by_side.items():
                actual = np.asarray(
                    snapshots[side].get("actual_qpos"), dtype=float
                ).reshape(-1)
                if actual.shape != (6,) or not np.all(np.isfinite(actual)):
                    return False
                if float(np.max(np.abs(actual - np.asarray(target, dtype=float)))) > float(
                    tolerance
                ):
                    return False
            # The initialization command is atomic across arm and enabled
            # DMTAC hand effectors.  Do not release the first replay frame
            # until every requested gripper has reached its pen-down target;
            # otherwise the arm could begin writing while a hand is still
            # closing/opening toward the source state.
            for side, target_value in directive.hand_qpos_by_side.items():
                target = np.asarray(target_value, dtype=float).reshape(-1)
                actual = np.asarray(
                    snapshots.get(side, {}).get("actual_hand_qpos"), dtype=float
                ).reshape(-1)
                if target.shape != (6,) or actual.shape != (6,):
                    return False
                if not np.all(np.isfinite(target)) or not np.all(np.isfinite(actual)):
                    return False
                if float(np.max(np.abs(actual - target))) > float(
                    self.config.gripper_qpos_tolerance
                ) and not self._writing_right_gripper_force_stop_reached(
                    side=side,
                    target=target,
                    snapshots=snapshots,
                ):
                    return False
            return True
        for side in emission.action.arm_qpos_by_side:
            target = np.asarray(emission.action.arm_qpos_by_side[side], dtype=float)
            actual = np.asarray(
                snapshots[side].get("actual_qpos"), dtype=float
            ).reshape(-1)
            if actual.shape != (6,) or float(np.max(np.abs(actual - target))) > float(
                self.config.qpos_tolerance_rad
            ):
                return False
        for side in emission.action.arm_pose_by_side:
            target_by_side = emission.command.metadata.get(
                "rule_based_target_pose_base_by_side"
            )
            if not isinstance(target_by_side, Mapping) or side not in target_by_side:
                return False
            target = np.asarray(target_by_side[side], dtype=float).reshape(4, 4)
            actual = np.asarray(
                snapshots[side].get("actual_tcp_pose"), dtype=float
            ).reshape(4, 4)
            translation_tolerance = float(
                emission.action.metadata.get(
                    "rule_based_height_arrival_tolerance_m",
                    self.config.terminal_translation_tolerance_m,
                )
            )
            if (
                not np.isfinite(translation_tolerance)
                or translation_tolerance <= 0.0
            ):
                return False
            if float(np.linalg.norm(actual[:3, 3] - target[:3, 3])) > translation_tolerance:
                return False
            relative_rotation = actual[:3, :3].T @ target[:3, :3]
            cosine = float(
                np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
            )
            if math.degrees(math.acos(cosine)) > float(
                self.config.terminal_rotation_tolerance_deg
            ):
                return False
        hand_targets = emission.action.hand_qpos_by_side
        hand_tolerance = float(self.config.gripper_qpos_tolerance)
        if (
            str(self.source_selection.get("collection_mode") or "")
            == "full_trajectory_replay"
            and emission.directive.kind == "replay"
        ):
            preserved = emission.action.metadata.get(
                "rule_based_source_hand_qpos_by_side"
            )
            if isinstance(preserved, Mapping):
                hand_targets = preserved
                # Once a large source delta has been split, reaching the last
                # bounded waypoint is not the same as reaching the source
                # endpoint.  Reuse the existing endpoint tolerance so the
                # terminal frame cannot finish one whole max-step early.
                hand_tolerance = min(
                    hand_tolerance,
                    float(self.config.manual_gripper_endpoint_tolerance),
                )
        for side, target_value in hand_targets.items():
            target = np.asarray(target_value, dtype=float).reshape(-1)
            actual = np.asarray(
                snapshots.get(side, {}).get("actual_hand_qpos"), dtype=float
            ).reshape(-1)
            if target.shape != (6,) or actual.shape != (6,):
                return False
            if not np.all(np.isfinite(target)) or not np.all(np.isfinite(actual)):
                return False
            if (
                float(np.max(np.abs(actual - target))) > hand_tolerance
                and not self._writing_right_gripper_force_stop_reached(
                    side=side,
                    target=target,
                    snapshots=snapshots,
                )
            ):
                return False
        return True

    def _publish_boundary(
        self,
        step_id: str,
        snapshots: Mapping[str, Mapping[str, Any]],
        now: float,
    ) -> None:
        frame = self._boundary_frame_by_step.get(step_id)
        if frame is None:
            return
        poses: dict[str, np.ndarray] = {}
        frames: dict[str, str] = {}
        for side in self.deployment.active_sides:
            raw_pose = pose_matrix_to_xyzquat(snapshots[side]["actual_tcp_pose"])
            if frame == CANONICAL_DESK_ACTION_FRAME:
                poses[side] = (
                    self.runtime_pose_frame_contract.storage_pose_vector_to_canonical_desk(
                        raw_pose,
                        side=side,
                    )
                )
                frames[side] = "canonical_desk"
            elif frame == HARDWARE_ARM_BASE_ACTION_FRAME:
                poses[side] = raw_pose
                frames[side] = f"{side}_arm_base"
            else:
                raise RuntimeError(f"unsupported relative boundary frame {frame!r}.")
        self._boundaries[step_id] = RuleBasedBoundaryState(
            tcp_pose_by_side=poses,
            pose_frame_by_side=frames,
            source="real_teleop_causal_actual_tcp",
            timestamp=now,
            metadata={
                "runtime_pose_frame_contract_hash": (
                    self.runtime_pose_frame_contract.contract_hash
                ),
                "causal_snapshot_contract": (
                    "source_timestamp<=T0_and_available_timestamp<=T0"
                ),
            },
        )

    def _boundary_for(self, boundary_ref: str) -> RuleBasedBoundaryState:
        try:
            return self._boundaries[str(boundary_ref)]
        except KeyError as exc:
            raise RuntimeError(
                f"measured rule-based boundary {boundary_ref!r} is not published."
            ) from exc

    def _clear_current(self) -> None:
        self._current = None
        self._target_wait_started_at = None
        self._next_due_at = None
        self._current_dispatched = False
        self._current_sample_accepted = False
        self._dispatch_retry_started_at = None
        self._writer_retry_started_at = None
        self._target_arrival_confirmed = False
        self._target_arrival_snapshots = {}
        self._settled_samples = 0
        self._filtered_initial_alignment_arm_by_side = {}
        self._filtered_initial_alignment_identity = None

    def _clear_replay_reference_schedule(self) -> None:
        """Invalidate every undispatched replay reference and its live clock."""

        self._replay_reference_queue.clear()
        self._replay_reference_stage_started_at = None
        self._replay_reference_pause_started_at = None
        self._replay_reference_last_schedule_lag_sec = 0.0
        self._replay_reference_cumulative_schedule_shift_sec = 0.0
        self._replay_reference_rebased_tick_count = 0

    def _cancel_initial_alignment_planning(self) -> None:
        future = self._initial_alignment_future
        if future is not None and not future.done():
            self._initial_alignment_cancel_pending = not future.cancel()
        if future is None or future.done() or not self._initial_alignment_cancel_pending:
            self._initial_alignment_future = None
            self._initial_alignment_cancel_pending = False
        self._initial_alignment_planning_started_at = None
        self._initial_alignment_template = None
        self._initial_alignment_plan = None
        self._initial_alignment_waypoint_index = 0
        self._initial_alignment_effective_waypoint_count = 0
        self._initial_alignment_started_at = None
        self._initial_alignment_last_schedule_lag_sec = 0.0
        self._initial_alignment_cumulative_schedule_shift_sec = 0.0
        self._initial_alignment_skipped_clock_ticks = 0
        self._initial_alignment_reacquire_samples = 0
        self._initial_alignment_hand_start_by_side = {}
        self._initial_alignment_hand_target_by_side = {}

    def _settle_cancelled_initial_alignment(self) -> None:
        future = self._initial_alignment_future
        if not self._initial_alignment_cancel_pending or future is None:
            return
        if not future.done():
            return
        try:
            future.result()
        except BaseException:
            # The abandoned result cannot authorize motion.  Its only purpose
            # here is to settle executor ownership before a fresh runtime Home.
            pass
        self._initial_alignment_future = None
        self._initial_alignment_cancel_pending = False
        self._invalidate_readiness_diagnosis_cache()

    def _collection_transition_pending(self) -> bool:
        callback = self._collection_transition_active
        if callback is None:
            return False
        try:
            return bool(callback())
        except Exception:
            return True

    def _manual_gripper_lifecycle_available(
        self,
        *,
        session_state_value: str,
    ) -> bool:
        """Return whether the rule source is in a command-inert maintenance gap.

        ``RUNNING`` alone is not authority to move a hand.  It is accepted only
        after the successful-Episode reset has removed the rule cursor, FixArm
        authority and writer lifecycle.  Intent admission, dispatch and status
        projection share this predicate so the GUI cannot advertise a
        transition that the realtime owner will later reject.
        """

        if (
            not self._started
            or self._closed
            or not self.config.gripper_sides
            or str(session_state_value).upper() not in {"ARMED", "RUNNING"}
            or self._held
            or self._aborted
            or self._completed
            or self._armed
            or self._execution_begun
            or self._current is not None
            or self._frame_emission is not None
            or self._paused_hold_emission is not None
            or self._replay_reference_queue
            or self._initial_alignment_template is not None
            or self._initial_alignment_future is not None
            or self._initial_alignment_cancel_pending
            or self._configuration_future is not None
            or self._next_attempt_rebuild_future is not None
            or self._height_calibration_future is not None
            or self._height_calibration_pending
        ):
            return False
        if self._collection_transition_pending():
            return False
        try:
            if self._recording_active is None or self._recording_active():
                return False
            provider = self._dataset_recorder_provider
            if provider is None:
                return False
            recorder = provider()
            if recorder is not None and (
                bool(getattr(recorder, "is_recording", False))
                or bool(getattr(recorder, "finalizing", False))
            ):
                return False
        except Exception:
            return False
        controller = getattr(self._components, "fix_arm_controller", None)
        if tuple(getattr(controller, "fixed_sides", ()) or ()):
            return False
        return True

    def _configuration_lifecycle_available(self) -> bool:
        """Return whether a plan rebuild is command-inert and collection-idle.

        This is the single runtime authority consumed by both request
        admission and the GUI.  Terminal plan flags are intentionally not a
        permanent lock: after Episode stop/discard *and* recorder finalization
        finish, a completed or aborted attempt may be replaced safely.
        """

        if (
            not self._started
            or self._closed
            or self._configuration_rebuilder is None
            or self._configuration_future is not None
            or self._next_attempt_rebuild_future is not None
            or self._writing_reachability_future is not None
            or self._writing_reachability_rebuild_future is not None
            or self._height_calibration_future is not None
            or self._height_calibration_pending
            or self._pending_gripper_action is not None
            or self._initial_alignment_cancel_pending
        ):
            return False
        if self._collection_transition_pending():
            return False
        try:
            if self._recording_active is None or self._recording_active():
                return False
        except Exception:
            return False
        provider = self._dataset_recorder_provider
        if provider is None:
            return False
        try:
            recorder = provider()
            if recorder is not None and (
                bool(getattr(recorder, "is_recording", False))
                or bool(getattr(recorder, "finalizing", False))
            ):
                return False
        except Exception:
            return False

        terminal_attempt = bool(self._completed or self._aborted)
        if terminal_attempt:
            # Completion and all fail-closed abort paths establish Hold before
            # the operator can submit another request.  Keep this explicit so
            # a stray terminal flag can never make an active motion path look
            # configurable.
            return bool(self._held)
        if self._armed:
            return bool(
                not self._execution_begun
                and self._waiting_reason
                in {
                    "formal_episode_not_active",
                    "formal_writer_not_recording",
                    "formal_writer_paused",
                    "configuration_rejected",
                }
            )
        return bool(
            not self._execution_begun
            and self._current is None
            and self._frame_emission is None
            and self._paused_hold_emission is None
        )

    def _reset_attempt_state(self, *, reason: str) -> None:
        """Rebuild one indivisible rule cursor without changing Session authority."""

        self._abandon_next_attempt_rebuild()
        self._cancel_initial_alignment_planning()
        self._boundaries.clear()
        for step in self.plan.steps:
            if not isinstance(step, RuleBasedActionSegment):
                continue
            reset_source = getattr(step.action_source, "reset", None)
            if callable(reset_source):
                reset_source()
        cursor = RuleBasedCollectionCursor(
            self.plan,
            self.trajectory_groups,
            boundary_provider=self._boundary_for,
        )
        self.adapter = RuleBasedCollectionAdapter(
            cursor,
            resolver=RealTeleopPoseFrameActionResolver(
                collection_contract=self.deployment.collection_pose_frame_contract,
                runtime_contract=self.runtime_pose_frame_contract,
            ),
        )
        self._clear_current()
        self._clear_replay_reference_schedule()
        self._last_dispatched_emission = None
        self._frame_emission = None
        self._paused_hold_emission = None
        self._deferred_progress_arm_contract = None
        self._deferred_progress_arm_fix_established = False
        self._coupled_pose_solver_seeded = False
        self._reset_pose_action_motion_intent(reason)
        self._last_live_dispatch_timestamp = None
        self._admitted_capture_sequences.clear()
        self._accepted_samples_by_detail_stage.clear()
        self._writer_rate_expected_samples_by_step.clear()
        self._planned_sample_count = self._count_planned_samples()
        self._stage_total_by_step = self._stage_totals_by_step()
        self._height_descent_planned_count_added = False
        self._height_descent_planned_sample_count = 0
        self._writing_height_target_z_m = None
        self._stage_label = None
        self._stage_index = None
        self._stage_frame_index = None
        self._stage_frame_total = None
        self._collection_position = None
        self._progress_scope_admission_error = ""
        self._armed = False
        self._execution_begun = False
        self._completed = False
        self._completion_stop_requested = False
        self._aborted = False
        self._arm_feedback_pause_started_at = None
        self._initial_alignment_reacquire_samples = 0
        self._clear_manual_gripper_action()

    def _reset_attempt_for_runtime_home(self) -> None:
        """Rebuild one indivisible rule attempt after authoritative Hold."""

        self._reset_attempt_state(reason="runtime_home_reset")
        self._invalidate_writing_reachability_preflight()
        self._writing_reachability_skipped_count = 0

    def _prepare_next_attempt_after_success(self, now: float) -> None:
        """Return a committed successful run to command-inert next-ready state.

        A normal end is not a safety event.  Wait until the existing writer
        owner has fully closed the Episode, release only the rule profile's
        deferred software FixArm authority, and rebuild the existing cursor.
        No motion starts until the operator explicitly requests the next
        Episode.
        """

        if (
            not self._completed
            or self._held
            or self._collection_transition_pending()
        ):
            return
        try:
            if self._recording_active is None or self._recording_active():
                return
            provider = self._dataset_recorder_provider
            recorder = None if provider is None else provider()
            if recorder is not None and (
                bool(getattr(recorder, "is_recording", False))
                or bool(getattr(recorder, "finalizing", False))
            ):
                return
        except Exception:
            return
        pending = self._next_attempt_rebuild_future
        if pending is not None:
            if not pending.done():
                self._waiting_reason = "successful_next_attempt_rebuild"
                return
            context = dict(self._next_attempt_rebuild_context or {})
            self._next_attempt_rebuild_future = None
            self._next_attempt_rebuild_context = None
            try:
                replacement = pending.result()
                self._activate_successful_next_attempt(
                    replacement=replacement,
                    **context,
                )
            except Exception as exc:  # noqa: BLE001 - activation fails closed.
                self._fail_successful_next_attempt_activation(exc)
            return
        if self._deferred_progress_arm_fix_established:
            controller = getattr(self._components, "fix_arm_controller", None)
            cancel_active = getattr(controller, "cancel_active", None)
            if not callable(cancel_active):
                self._completed = False
                self._aborted = True
                self._waiting_reason = "successful_fix_release_failed"
                self._establish_hold(
                    "completed rule attempt could not release deferred FixArm"
                )
                return
            transition = cancel_active(
                now=float(now),
                reason="rule-based Episode completed",
                clear_snapshots=False,
            )
            if not bool(getattr(transition, "ok", False)):
                self._completed = False
                self._aborted = True
                self._waiting_reason = "successful_fix_release_failed"
                self._establish_hold(
                    "completed rule attempt failed to release deferred FixArm: "
                    + str(getattr(transition, "message", "unknown failure"))
                )
                return
            self._deferred_progress_arm_fix_established = False
        self._advance_pre_pen_down_replay_cadence()
        try:
            next_selection_ordinal = self._selection_ordinal + 1
            if self._configuration_batch:
                count = len(self._configuration_batch)
                completed_index = int(self._configuration_batch_index)
                next_completed_count = min(
                    count,
                    self._configuration_batch_completed_count + 1,
                )
                if completed_index + 1 < count:
                    next_index = completed_index + 1
                    next_cycle_complete = False
                else:
                    next_index = 0
                    next_cycle_complete = True
                configuration, replacement = self._configuration_batch[next_index]
            else:
                next_index = 0
                next_completed_count = 1
                next_cycle_complete = True
                configuration = copy.deepcopy(self._selected_configuration)
                replacement = None
            context = {
                "configuration": copy.deepcopy(dict(configuration)),
                "next_selection_ordinal": int(next_selection_ordinal),
                "next_index": int(next_index),
                "next_completed_count": int(next_completed_count),
                "next_cycle_complete": bool(next_cycle_complete),
            }
            if self._configuration_requires_ordinal_rebuild(configuration):
                self._next_attempt_rebuild_context = context
                self._next_attempt_rebuild_future = (
                    self._configuration_executor.submit(
                        self._rebuild_for_selection_ordinal,
                        configuration,
                        selection_ordinal=next_selection_ordinal,
                    )
                )
                self._waiting_reason = "successful_next_attempt_rebuild"
                return
            self._activate_successful_next_attempt(
                replacement=replacement,
                **context,
            )
        except Exception as exc:  # noqa: BLE001 - batch activation fails closed.
            self._next_attempt_rebuild_context = None
            self._fail_successful_next_attempt_activation(exc)
            return

    def _activate_successful_next_attempt(
        self,
        *,
        configuration: Mapping[str, Any],
        replacement: "RuleBasedRealTeleopExtension" | None,
        next_selection_ordinal: int,
        next_index: int,
        next_completed_count: int,
        next_cycle_complete: bool,
    ) -> None:
        if replacement is not None:
            self._adopt_prestart_extension(replacement)
            if self._configuration_on_commit is not None:
                self._configuration_on_commit(configuration, replacement)
        else:
            self.plan = self._plan_for_next_attempt()
        if self._configuration_batch:
            self._configuration_batch_index = int(next_index)
            self._selected_configuration = copy.deepcopy(dict(configuration))
        self._configuration_batch_completed_count = int(next_completed_count)
        self._configuration_batch_cycle_complete = bool(next_cycle_complete)
        self._selection_ordinal = next_selection_ordinal
        runtime_info = self._configuration_options.get("runtime_info")
        if isinstance(runtime_info, dict) and self._selected_configuration:
            collect_tactile = bool(
                self._selected_configuration.get("collect_tactile", False)
            )
            updates = {
                "tactile_capture_locked": True,
                "tactile_capture_source": "existing_dataset_contract",
                "tactile_capture_mode": (
                    "enabled" if collect_tactile else "disabled"
                ),
            }
            runtime_info.update(updates)
            update_runtime_info = getattr(
                self._control_ui,
                "update_rule_runtime_info",
                None,
            )
            if callable(update_runtime_info):
                update_runtime_info(updates)
        self._reset_attempt_state(reason="successful_episode_reset")
        self._writing_reachability_skipped_count = 0
        self._held = False
        self._armed = bool(self._legacy_generic_ui)
        self._waiting_reason = (
            "waiting_for_formal_episode"
            if self._legacy_generic_ui
            else "waiting_for_rule_start"
        )
        self._invalidate_readiness_diagnosis_cache()

    def _fail_successful_next_attempt_activation(self, exc: BaseException) -> None:
        self._completed = False
        self._aborted = True
        self._waiting_reason = "successful_batch_advance_failed"
        self._establish_hold(
            "completed rule attempt could not activate the next validated "
            f"batch item: {type(exc).__name__}: {exc}"
        )
        append_message = getattr(self._control_ui, "append_message", None)
        if callable(append_message):
            append_message(
                "RuleBased",
                "当前 Episode 已提交，但下一条预验证组合激活失败；"
                "机械臂已进入 Hold，请重新验证批次。",
            )

    @staticmethod
    def _dispose_rebuilt_extension_future(
        future: Future["RuleBasedRealTeleopExtension"],
    ) -> None:
        try:
            replacement = future.result()
        except Exception:
            return
        replacement._initial_alignment_executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
        replacement._configuration_executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

    def _abandon_next_attempt_rebuild(self) -> None:
        pending = self._next_attempt_rebuild_future
        self._next_attempt_rebuild_future = None
        self._next_attempt_rebuild_context = None
        if pending is None:
            return
        if pending.cancel():
            return
        pending.add_done_callback(self._dispose_rebuilt_extension_future)

    @staticmethod
    def _configuration_uses_random_sources(
        configuration: Mapping[str, Any],
    ) -> bool:
        return any(
            str(value) == _RANDOM_SOURCE_SELECTION_ID
            for key, value in configuration.items()
            if str(key).endswith("_candidate_id")
        )

    def _configuration_requires_ordinal_rebuild(
        self,
        configuration: Mapping[str, Any],
    ) -> bool:
        variant = self.source_selection.get("writing_variant_selection", {})
        return bool(
            self._configuration_uses_random_sources(configuration)
            or (
                isinstance(variant, Mapping)
                and bool(variant.get("enabled", False))
            )
        )

    def _rebuild_for_selection_ordinal(
        self,
        configuration: Mapping[str, Any],
        *,
        selection_ordinal: int,
    ) -> "RuleBasedRealTeleopExtension":
        rebuilder = self._configuration_rebuilder
        if rebuilder is None:
            raise RuntimeError("random source selection rebuilder is unavailable")
        request = copy.deepcopy(dict(configuration))
        request["_selection_ordinal"] = int(selection_ordinal)
        replacement = rebuilder(request)
        if not isinstance(replacement, RuleBasedRealTeleopExtension):
            raise TypeError("rule configuration rebuilder returned the wrong type.")
        return replacement

    def _writing_reachability_metadata(self) -> Mapping[str, Any]:
        metadata = self.source_selection.get("writing_reachability", {})
        return metadata if isinstance(metadata, Mapping) else {}

    def _writing_reachability_enabled(self) -> bool:
        return bool(self._writing_reachability_metadata().get("enabled", False))

    def _writing_reachability_current_fingerprint(
        self,
    ) -> tuple[object, ...] | None:
        calibration = self._height_calibration
        if calibration is None or not calibration.enabled:
            return None
        variant = self.source_selection.get("writing_variant_selection", {})
        variant = variant if isinstance(variant, Mapping) else {}
        phases = self.source_selection.get("phases", {})
        phases = phases if isinstance(phases, Mapping) else {}
        pen_down = phases.get("pen_down", {})
        pen_down = pen_down if isinstance(pen_down, Mapping) else {}
        catalog = self.source_selection.get("trajectory_catalog", {})
        catalog = catalog if isinstance(catalog, Mapping) else {}
        return (
            int(
                self.source_selection.get(
                    "selection_ordinal",
                    self._selection_ordinal,
                )
                or 0
            ),
            str(variant.get("selected_trajectory_id") or ""),
            str(variant.get("selected_direction") or ""),
            float(variant.get("selected_size_m") or 0.0),
            str(pen_down.get("candidate_id") or ""),
            str(pen_down.get("source_contract_hash") or ""),
            str(catalog.get("combined_sha256") or ""),
            float(calibration.upper_bound_z_m),
        )

    def _writing_reachability_passed_for_current_selection(self) -> bool:
        return bool(
            not self._writing_reachability_enabled()
            or (
                self._writing_reachability_status == "passed"
                and self._writing_reachability_fingerprint
                == self._writing_reachability_current_fingerprint()
            )
        )

    def _invalidate_writing_reachability_preflight(self) -> None:
        pending = self._writing_reachability_future
        self._writing_reachability_future = None
        if pending is not None:
            pending.cancel()
        rebuild = self._writing_reachability_rebuild_future
        self._writing_reachability_rebuild_future = None
        self._writing_reachability_rebuild_ordinal = None
        if rebuild is not None and not rebuild.cancel():
            rebuild.add_done_callback(self._dispose_rebuilt_extension_future)
        self._writing_reachability_fingerprint = None
        self._writing_reachability_status = (
            "pending" if self._writing_reachability_enabled() else "disabled"
        )
        self._writing_reachability_detail = ""
        self._writing_reachability_progress = None
        self._writing_reachability_progress_detail = ""
        self._writing_reachability_started_at = None
        self._writing_reachability_candidate_label = ""
        self._writing_reachability_preparation_ms = None

    def _writing_reachability_candidate_description(self) -> str:
        variant = self.source_selection.get("writing_variant_selection", {})
        variant = variant if isinstance(variant, Mapping) else {}
        phases = self.source_selection.get("phases", {})
        phases = phases if isinstance(phases, Mapping) else {}
        approach = phases.get("pen_down", {})
        approach = approach if isinstance(approach, Mapping) else {}
        return (
            f"候选 {self._selection_ordinal + 1} · "
            f"{variant.get('selected_trajectory_id') or '未指定轨迹'} / "
            f"{variant.get('selected_direction') or '未指定方向'} / "
            f"{float(variant.get('selected_size_m') or 0.0):.3f}m · "
            f"靠近来源 {approach.get('candidate_id') or '未指定'}"
        )

    def _observe_writing_reachability_progress(self) -> None:
        progress = self._writing_reachability_progress
        if progress is None:
            return
        for _ in range(8):
            try:
                event, name, index, count, elapsed_ms = progress.get_nowait()
            except Empty:
                break
            if event == "start":
                self._writing_reachability_progress_detail = (
                    f"第 {index}/{count} 点 {name} 求解中"
                )
            else:
                self._writing_reachability_progress_detail = (
                    f"第 {index}/{count} 点 {name} 已检查 "
                    f"({elapsed_ms:.0f}ms)"
                )
        if self._writing_reachability_started_at is not None:
            elapsed = max(0.0, time.monotonic() - self._writing_reachability_started_at)
            pending = self._writing_reachability_future
            if pending is not None and not pending.running():
                phase = "等待后台计算资源"
            else:
                phase = self._writing_reachability_progress_detail or "等待第一点检查启动"
            deadline_detail = (
                "；超过预检目标时限，正在等待后台数值调用返回"
                if elapsed > _WRITING_REACHABILITY_MAX_CHECK_SEC
                and pending is not None
                and pending.running()
                else ""
            )
            self._writing_reachability_detail = (
                f"{self._writing_reachability_candidate_label}；"
                f"前台准备 {float(self._writing_reachability_preparation_ms or 0.0):.0f}ms；"
                f"{phase}；已等待 {int(elapsed)}s{deadline_detail}；"
                "尚未启动 Episode。"
            )

    def _writing_reachability_request(self) -> dict[str, Any]:
        metadata = self._writing_reachability_metadata()
        probes = metadata.get("probes")
        if not isinstance(probes, (list, tuple)) or len(probes) != 4:
            raise RuntimeError("writing reachability requires exactly four probes")
        calibration = self._height_calibration
        if calibration is None or not calibration.enabled:
            raise RuntimeError("writing reachability requires calibrated writing height")
        components = self._components
        retargeter = getattr(components, "dual_arm_retargeter", None)
        retargeter_config = getattr(retargeter, "config", None)
        if retargeter is None or retargeter_config is None:
            raise RuntimeError("writing reachability requires the production IK config")
        runtime_calibration = getattr(
            getattr(components, "session", None),
            "calibration",
            None,
        )
        if runtime_calibration is None:
            raise RuntimeError("writing reachability requires runtime calibration")
        resolver = self.adapter.resolver
        if not isinstance(resolver, RealTeleopPoseFrameActionResolver):
            raise RuntimeError("rule pose-frame resolver was replaced")
        pen_down_group = self.trajectory_groups.get("pen_down")
        source_qpos = getattr(
            pen_down_group,
            "arm_observation_actual_qpos",
            None,
        )
        if not isinstance(source_qpos, Mapping):
            raise RuntimeError("writing reachability lacks pen-down source qpos")
        # The live plan moves both arms to frame zero, then holds the left arm
        # while replaying the right-arm pen-down stage.  Mirror that exact
        # boundary here instead of importing incidental left-arm drift from
        # later source observations.
        seed_qpos = {
            "left": np.asarray(source_qpos["left"][0], dtype=float)
            .reshape(6)
            .copy(),
            "right": np.asarray(source_qpos["right"][-1], dtype=float)
            .reshape(6)
            .copy(),
        }
        # The left arm is held at the source qpos throughout writing.  Its
        # solver target must therefore be FK of that exact held state, rather
        # than a recorded action pose that may use a different timestamp or
        # frame.  This is read-only FK and does not change the live IK state.
        seed_solver_targets = retargeter.solver_world_targets_for_qpos_pair(
            left_qpos=seed_qpos["left"],
            right_qpos=seed_qpos["right"],
        )
        left_solver_target = np.asarray(
            seed_solver_targets["left"], dtype=float
        ).reshape(4, 4)
        target_filter = getattr(retargeter, "pre_ik_target_safety_filter", None)
        if not callable(target_filter):
            raise RuntimeError("writing reachability requires the Active TCP envelope")
        world_to_base_by_side: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            arm_calibration = runtime_calibration.get_arm_for_side(side)
            transform = getattr(arm_calibration, "T_D_B", None)
            if transform is None:
                raise RuntimeError(f"{side} calibration lacks T_D_B")
            world_to_base_by_side[side] = np.linalg.inv(
                np.asarray(transform, dtype=float).reshape(4, 4)
            )

        prepared_probes: list[dict[str, Any]] = []
        for raw_probe in probes:
            if not isinstance(raw_probe, Mapping):
                raise RuntimeError("writing reachability probe must be a mapping")
            right_pose = xyzquat_to_pose_matrix(
                np.asarray(
                    raw_probe.get("right_pose_canonical_desk"),
                    dtype=float,
                ).reshape(7)
            )
            right_pose = np.asarray(right_pose, dtype=float).copy()
            right_pose[2, 3] = float(calibration.upper_bound_z_m)
            # Mirror the formal execution path in _coupled_pose_emission:
            # canonical desk -> runtime arm base -> calibrated solver world ->
            # solver endpoint.  Feeding canonical desk directly to the solver
            # can introduce a large translation and a 180-degree rotation.
            right_base_pose = (
                resolver.runtime_contract.canonical_desk_pose_to_storage(
                    right_pose,
                    side="right",
                )
            )
            right_tool_tcp_world = runtime_calibration.base_to_desk(
                right_base_pose,
                "right",
            )
            solver_targets = {
                "left": left_solver_target.copy(),
                "right": retargeter.solver_world_target_from_tool_tcp_world_target(
                    side="right",
                    tool_tcp_world_target=right_tool_tcp_world,
                ),
            }
            effective_targets = target_filter(solver_targets)
            if effective_targets is None:
                raise RuntimeError("Active TCP envelope returned no target")
            modified_sides = tuple(
                side
                for side in ("left", "right")
                if not np.allclose(
                    np.asarray(effective_targets[side], dtype=float),
                    np.asarray(solver_targets[side], dtype=float),
                    rtol=0.0,
                    atol=1e-10,
                )
            )
            prepared_probes.append(
                {
                    "name": str(raw_probe.get("name") or "unknown"),
                    "target_pose_by_side": {
                        side: np.asarray(effective_targets[side], dtype=float).copy()
                        for side in ("left", "right")
                    },
                    "envelope_modified_sides": modified_sides,
                }
            )

        oracle_config = replace(
            retargeter_config,
            left_initial_qpos=seed_qpos["left"],
            right_initial_qpos=seed_qpos["right"],
            enforce_realistic_joint_step=False,
            solve_time_budget_ms=None,
            solve_max_iterations=max(
                32,
                int(getattr(retargeter_config, "solve_max_iterations", 1)),
            ),
            solve_min_iterations=1,
            solve_adaptive_iterations_enabled=False,
            # The oracle has no output smoothing. Keep its fast endpoints
            # consistent with the unsmoothed slow endpoint when cloning the
            # live config; these values do not alter the production solver.
            output_qpos_smoothing_alpha=1.0,
            output_qpos_smoothing_adaptive_enabled=False,
            output_qpos_admittance_fast_path_alpha=1.0,
            output_qpos_smoothing_singularity_alpha=1.0,
            target_pose_smoother=None,
            debug_viz=False,
            debug_profile=False,
        )
        return {
            "oracle_config": oracle_config,
            "seed_qpos": seed_qpos,
            "probes": tuple(prepared_probes),
            "world_to_base_by_side": world_to_base_by_side,
            "pose_projection": {
                "right_probe": (
                    "canonical_desk_to_runtime_arm_base_to_solver_world"
                ),
                "left_hold": "seed_qpos_solver_fk",
                "runtime_pose_frame_contract_hash": (
                    resolver.runtime_contract.contract_hash
                ),
            },
            "translation_tolerance_m": 0.005,
            "rotation_tolerance_deg": 3.0,
        }

    @staticmethod
    def _run_writing_reachability_preflight(
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Check four endpoints without backend, Session, or dispatch access."""

        started_at = time.monotonic()
        deadline = started_at + _WRITING_REACHABILITY_MAX_CHECK_SEC
        report_progress = request.get("progress_callback")
        oracle = DualArmCollisionAwareRetargeter(request["oracle_config"])
        seed_qpos = request["seed_qpos"]
        translation_tolerance = float(request["translation_tolerance_m"])
        rotation_tolerance = float(request["rotation_tolerance_deg"])
        results: list[dict[str, Any]] = []
        probes = request["probes"]
        for index, probe in enumerate(probes, start=1):
            name = str(probe["name"])
            probe_started_at = time.monotonic()
            if callable(report_progress):
                report_progress(("start", name, index, len(probes), 0.0))
            modified_sides = tuple(probe["envelope_modified_sides"])
            # A projected right target would change the written glyph. The
            # left target is only a solver input; its actual qpos remains
            # fixed by FixArm and is collision-checked with the right result.
            if "right" in modified_sides:
                results.append(
                    {
                        "name": str(probe["name"]),
                        "reachable": False,
                        "reason": "tcp_envelope_projection_required",
                        "modified_sides": list(modified_sides),
                        "elapsed_ms": (time.monotonic() - probe_started_at) * 1000.0,
                    }
                )
                if callable(report_progress):
                    report_progress((
                        "finish", name, index, len(probes),
                        results[-1]["elapsed_ms"],
                    ))
                continue
            oracle.set_current_qpos_pair(
                left_qpos=seed_qpos["left"],
                right_qpos=seed_qpos["right"],
                reset_tracking=True,
            )
            final_result = None
            translation_error = float("inf")
            rotation_error = float("inf")
            reachable = False
            reason = "solver_not_converged"
            collision_reason = None
            solve_frames = 0
            for _iteration in range(64):
                if time.monotonic() >= deadline:
                    reason = "solver_budget_exhausted"
                    break
                final_result = oracle.solve_target_pair(
                    BimanualPoseTarget(
                        timestamp=None,
                        target_pose_by_side=probe["target_pose_by_side"],
                        # Fixed writing sends both poses to the live coupled
                        # solver. The downstream FixArm owner holds the left
                        # qpos, checked explicitly below after this solve.
                        active_sides=("left", "right"),
                        hard_hold_inactive_sides=True,
                        world_to_base_by_side=request["world_to_base_by_side"],
                    )
                )
                solve_frames += 1
                right_result = final_result.side_results["right"]
                translation_error, rotation_error = pose_delta_metrics(
                    right_result.solver_pose_world,
                    probe["target_pose_by_side"]["right"],
                )
                if (
                    final_result.status in {"tracking", "boundary_active"}
                    and translation_error <= translation_tolerance
                    and rotation_error <= rotation_tolerance
                ):
                    held_pair = oracle.check_qpos_pair_collision(
                        left_qpos=seed_qpos["left"],
                        right_qpos=right_result.qpos,
                    )
                    reachable = bool(held_pair.safe)
                    reason = (
                        "certified" if reachable else "held_left_collision"
                    )
                    collision_reason = (
                        None if reachable else str(held_pair.reason)
                    )
                    break
                if final_result.status in {"hold_last_safe", "unsafe_candidate"}:
                    reason = (
                        "collision_guard_rejected"
                        if final_result.fallback_reason
                        in {"collision_margin_low", "freeze_on_contact"}
                        else "solver_guard_hold"
                    )
                    break
                # The oracle is a background worker, but yielding between
                # numerical solves keeps the foreground control loop responsive.
                time.sleep(0.001)
            fallback_reason = collision_reason
            if fallback_reason is None and final_result is not None:
                fallback_reason = final_result.fallback_reason
            results.append(
                {
                    "name": str(probe["name"]),
                    "reachable": reachable,
                    "reason": reason,
                    "solver_status": (
                        "not_run" if final_result is None else str(final_result.status)
                    ),
                    "fallback_reason": fallback_reason,
                    "solve_frames": solve_frames,
                    "translation_error_m": translation_error,
                    "rotation_error_deg": rotation_error,
                    "elapsed_ms": (time.monotonic() - probe_started_at) * 1000.0,
                }
            )
            if callable(report_progress):
                report_progress((
                    "finish", name, index, len(probes),
                    results[-1]["elapsed_ms"],
                ))
            if reason == "solver_budget_exhausted":
                break
        return {
            "reachable": all(bool(item["reachable"]) for item in results),
            "probes": results,
            "pose_projection": dict(request.get("pose_projection") or {}),
            "elapsed_ms": (time.monotonic() - started_at) * 1000.0,
        }

    def _service_writing_reachability_preflight(self) -> None:
        if not self._writing_reachability_enabled():
            self._writing_reachability_status = "disabled"
            self._writing_reachability_detail = ""
            return
        if (
            self._closed
            or self._armed
            or self._execution_begun
            or self._completed
            or self._aborted
            or self._held
            or self._height_calibration is None
            or not self._height_calibration.enabled
        ):
            return

        rebuild = self._writing_reachability_rebuild_future
        if rebuild is not None:
            if not rebuild.done():
                self._writing_reachability_status = "skipping_unreachable"
                self._writing_reachability_detail = (
                    f"{self._writing_reachability_candidate_label} 已被安全边界拒绝；"
                    f"累计跳过 {self._writing_reachability_skipped_count} 条，"
                    "正在后台准备下一候选；尚未启动 Episode。"
                )
                return
            ordinal = self._writing_reachability_rebuild_ordinal
            self._writing_reachability_rebuild_future = None
            self._writing_reachability_rebuild_ordinal = None
            try:
                replacement = rebuild.result()
                self._adopt_prestart_extension(replacement)
                if ordinal is None:
                    raise RuntimeError("reachability rebuild lost its selection ordinal")
                self._selection_ordinal = int(ordinal)
                self._reset_attempt_state(reason="unreachable_candidate_skip")
                self._writing_reachability_fingerprint = None
                self._writing_reachability_status = "pending"
                self._writing_reachability_detail = "下一候选已准备，正在重新检查。"
                self._waiting_reason = "writing_reachability_preflight"
            except Exception as exc:  # noqa: BLE001 - command-inert preparation.
                self._writing_reachability_fingerprint = (
                    self._writing_reachability_current_fingerprint()
                )
                self._writing_reachability_status = "error"
                self._writing_reachability_detail = (
                    f"{type(exc).__name__}: {exc}"
                )
                self._waiting_reason = "writing_reachability_error"
            return

        fingerprint = self._writing_reachability_current_fingerprint()
        if fingerprint is None:
            return
        if (
            self._writing_reachability_fingerprint == fingerprint
            and self._writing_reachability_status
            in {"passed", "unreachable", "error"}
        ):
            # A terminal result belongs to this exact candidate and height.
            # Re-submitting it every foreground frame previously spun the IK
            # worker indefinitely after search exhaustion or an exception.
            return

        pending = self._writing_reachability_future
        if pending is None:
            self._writing_reachability_fingerprint = fingerprint
            try:
                self._writing_reachability_candidate_label = (
                    self._writing_reachability_candidate_description()
                )
                preparation_started_at = time.monotonic()
                request = self._writing_reachability_request()
                self._writing_reachability_preparation_ms = (
                    time.monotonic() - preparation_started_at
                ) * 1000.0
                progress: SimpleQueue[tuple[str, str, int, int, float]] = SimpleQueue()
                request["progress_callback"] = progress.put
                self._writing_reachability_progress = progress
                self._writing_reachability_started_at = time.monotonic()
                self._writing_reachability_progress_detail = ""
                self._writing_reachability_future = self._configuration_executor.submit(
                    self._run_writing_reachability_preflight,
                    request,
                )
                self._writing_reachability_status = "pending"
                self._writing_reachability_detail = (
                    f"{self._writing_reachability_candidate_label}；"
                    f"前台准备 {self._writing_reachability_preparation_ms:.0f}ms；"
                    "正在后台检查书写四边界接触点；尚未启动 Episode。"
                )
                self._waiting_reason = "writing_reachability_preflight"
            except Exception as exc:  # noqa: BLE001 - no motion was authorized.
                self._writing_reachability_status = "error"
                self._writing_reachability_detail = (
                    f"{self._writing_reachability_candidate_label}；"
                    f"前台准备失败：{type(exc).__name__}: {exc}"
                )
                self._waiting_reason = "writing_reachability_error"
            return
        self._observe_writing_reachability_progress()
        if not pending.done():
            return
        self._writing_reachability_future = None
        self._writing_reachability_progress = None
        try:
            result = pending.result()
        except Exception as exc:  # noqa: BLE001 - preflight remains command-inert.
            self._writing_reachability_status = "error"
            self._writing_reachability_detail = (
                f"{self._writing_reachability_candidate_label}；"
                f"后台预检失败：{type(exc).__name__}: {exc}"
            )
            self._waiting_reason = "writing_reachability_error"
            return
        elapsed_value = result.get("elapsed_ms")
        elapsed_detail = (
            f"前台准备 {float(self._writing_reachability_preparation_ms or 0.0):.0f}ms；"
            f"预检耗时 {float(elapsed_value):.0f}ms；"
            if elapsed_value is not None
            else f"前台准备 {float(self._writing_reachability_preparation_ms or 0.0):.0f}ms；"
        )
        if bool(result.get("reachable", False)):
            self._writing_reachability_status = "passed"
            probe_timings = ", ".join(
                f"{probe.get('name', 'unknown')} "
                f"{float(probe['elapsed_ms']):.0f}ms/"
                f"{probe.get('solve_frames', 0)}帧"
                for probe in result.get("probes", ())
                if isinstance(probe, Mapping) and probe.get("elapsed_ms") is not None
            )
            self._writing_reachability_detail = (
                f"{self._writing_reachability_candidate_label}；"
                + elapsed_detail
                + (f"各点 {probe_timings}；" if probe_timings else "")
                + "书写四边界接触点 IK/碰撞检查通过。"
            )
            self._waiting_reason = "waiting_for_rule_start"
            append_message = getattr(self._control_ui, "append_message", None)
            if callable(append_message):
                append_message("RuleBasedReachability", self._writing_reachability_detail)
            return

        failed = [
            item
            for item in result.get("probes", ())
            if isinstance(item, Mapping) and not bool(item.get("reachable", False))
        ]
        details = ", ".join(
            f"{item.get('name') or 'unknown'}:{item.get('reason') or 'unknown'}"
            f"/{item.get('solver_status') or 'unknown'}"
            f"/{item.get('fallback_reason') or 'no_fallback'}"
            f"/{item.get('solve_frames', 'n/a')} frames"
            f"/{item.get('translation_error_m', 'n/a')}m"
            f"/{item.get('rotation_error_deg', 'n/a')}deg"
            f"/{item.get('elapsed_ms', 'n/a')}ms"
            for item in failed
        )
        definitive_reasons = {
            "tcp_envelope_projection_required",
            "held_left_collision",
            "collision_guard_rejected",
        }
        if not failed or any(
            str(item.get("reason")) not in definitive_reasons for item in failed
        ):
            self._writing_reachability_status = "error"
            self._writing_reachability_detail = (
                f"{self._writing_reachability_candidate_label}；{elapsed_detail}"
                "书写可达性数值检查未能确认，未判为几何不可达；"
                + (details or "缺少失败点结果")
                + "。请调整书写大小/方向或检查标定后重试。"
            )
            self._waiting_reason = "writing_reachability_unresolved"
            append_message = getattr(self._control_ui, "append_message", None)
            if callable(append_message):
                append_message("RuleBasedReachability", self._writing_reachability_detail)
            return
        self._writing_reachability_skipped_count += 1
        self._writing_reachability_status = "unreachable"
        self._writing_reachability_detail = (
            f"{self._writing_reachability_candidate_label}；{elapsed_detail}"
            "安全边界拒绝：" + details
        )
        append_message = getattr(self._control_ui, "append_message", None)
        if callable(append_message):
            append_message(
                "RuleBasedReachability",
                self._writing_reachability_detail + "；当前候选不启动 Episode。",
            )
        configuration = copy.deepcopy(self._selected_configuration)
        if (
            self._writing_reachability_skipped_count
            >= _WRITING_REACHABILITY_MAX_AUTO_SKIPS
            or not configuration
            or not self._configuration_requires_ordinal_rebuild(configuration)
            or self._configuration_rebuilder is None
        ):
            self._writing_reachability_detail += (
                f"；连续 {_WRITING_REACHABILITY_MAX_AUTO_SKIPS} 条候选未通过，"
                "已停止自动重试。请调整配置后重新开始。"
            )
            self._waiting_reason = "writing_reachability_unreachable"
            return
        next_ordinal = self._selection_ordinal + 1
        self._writing_reachability_rebuild_ordinal = next_ordinal
        self._writing_reachability_rebuild_future = self._configuration_executor.submit(
            self._rebuild_for_selection_ordinal,
            configuration,
            selection_ordinal=next_ordinal,
        )
        self._writing_reachability_status = "skipping_unreachable"
        self._writing_reachability_detail += "；正在跳过并准备下一候选。"
        self._waiting_reason = "writing_reachability_skip"

    def discards_episode_on_stop(self, *, reason: str) -> bool:
        if str(reason) == "rule_based_complete" or self._completed:
            return False
        return bool(
            self._armed
            or self._execution_begun
            or self._aborted
            or self._restart_after_hold_requested
            or self._held
        )

    def restarts_after_hold(self, *, reason: str) -> bool:
        del reason
        return bool(
            not self._completed
            and (
                self._armed
                or self._execution_begun
                or self._aborted
                or self._restart_after_hold_requested
            )
        )

    def after_runtime_hold(
        self,
        *,
        reason: str,
        hardware_hold_confirmed: bool,
    ) -> None:
        if not self.restarts_after_hold(reason=reason):
            return
        self._reset_attempt_for_runtime_home()
        self._restart_after_hold_requested = False
        self._held = True
        self._waiting_reason = "runtime_home_required_after_hold"
        self._invalidate_readiness_diagnosis_cache()
        if (
            not self._collection_transition_pending()
            and self._recording_active is not None
            and self._recording_active()
        ):
            self._request_failure_stop()
        append_message = getattr(self._control_ui, "append_message", None)
        if callable(append_message):
            detail = (
                "本轮规则采集已取消；请确认机械臂安全后点击“重新安全初始化”。"
                if hardware_hold_confirmed
                else "规则运动已禁用，但底层 Hold 回执不完整；请检查设备，禁止重新初始化。"
            )
            append_message("RuleBased", detail)
        self._publish_status()

    def after_runtime_home(self, *, now: float) -> None:
        del now
        self._settle_cancelled_initial_alignment()
        if not self._held or self._aborted or self._completed:
            return
        if self._initial_alignment_cancel_pending:
            raise RuntimeError("abandoned initial-alignment planner is still settling")
        if self._collection_transition_pending():
            raise RuntimeError("cancelled rule Episode lifecycle is still settling")
        if self._recording_active is not None and self._recording_active():
            raise RuntimeError("cancelled rule Episode is still active")
        self._invalidate_writing_reachability_preflight()
        self._held = False
        # Explicit runtime Home already resets preview, command transport and
        # coupled IK from one live snapshot, so no additional rule-owned
        # pre-dispatch rebase is needed.
        self._manual_adjustment_motion_rebase_pending = False
        self._armed = bool(self._legacy_generic_ui)
        self._waiting_reason = (
            "waiting_for_formal_episode"
            if self._legacy_generic_ui
            else "waiting_for_rule_start"
        )
        self._invalidate_readiness_diagnosis_cache()
        self._publish_status()

    def dataset_sample_admission(
        self,
        *,
        step_result: object,
        now: float,
    ) -> RealTeleopDatasetSampleAdmission:
        del now
        emission = self._frame_emission
        allowed = bool(
            emission is not None
            and emission.recordable
            and emission is self._current
            and self._current_dispatched
            and getattr(step_result, "command_dispatched", False)
            and (
                self._deferred_progress_arm_contract is None
                or self._deferred_progress_arm_fix_established
            )
        )
        if emission is None:
            return RealTeleopDatasetSampleAdmission(allowed=False)
        directive = emission.directive
        sequence_index = int(directive.sequence_index)
        if allowed and sequence_index in self._admitted_capture_sequences:
            allowed = False
        phase_metadata = self._phase_metadata(emission)
        return RealTeleopDatasetSampleAdmission(
            allowed=allowed,
            metadata={
                "rule_based_real_teleop_schema": RULE_BASED_REAL_TELEOP_SCHEMA_V1,
                "rule_based_step_id": directive.step_id,
                "rule_based_directive_kind": directive.kind,
                "rule_based_sequence_index": sequence_index,
                "rule_based_capture_enabled": bool(emission.recordable),
                "rule_based_segment_terminal": bool(directive.segment_terminal),
                "rule_based_segment_stop_reason": directive.segment_stop_reason,
                "rule_based_progress_entry_fix_established": bool(
                    self._deferred_progress_arm_fix_established
                ),
                "rule_based_source_selection": copy.deepcopy(self.source_selection),
                "rule_based_pre_pen_down_replay_schedule": (
                    self._pre_pen_down_replay_schedule_metadata()
                ),
                "rule_based_height_calibration": self._height_calibration_metadata(),
                **{
                    str(key): _mutable_metadata_copy(value)
                    for key, value in emission.action.metadata.items()
                    if str(key).startswith("rule_based_height_")
                },
                **phase_metadata,
                "rule_based_stage_frame_index": self._stage_frame_index,
                "rule_based_stage_frame_total": self._stage_frame_total,
                "rule_based_stage_sample_coverage_policy": {
                    "minimum_fraction": float(
                        self.config.minimum_stage_sample_coverage
                    ),
                    "minimum_count": int(self.config.minimum_stage_sample_count),
                },
            },
        )

    def after_dataset_sample_attempt(
        self,
        *,
        admission: RealTeleopDatasetSampleAdmission,
        accepted: bool,
        now: float,
    ) -> None:
        metadata = dict(admission.metadata)
        if not bool(admission.allowed):
            return
        terminal_kind = str(metadata.get("_terminal_kind", "accepted") or "accepted")
        if terminal_kind == "fatal":
            reason = str(
                metadata.get("_terminal_reason")
                or "formal writer reported a fatal sample-processing error"
            )
            self._abort_session_step(
                "formal writer failed while processing the active rule sample: "
                + reason
            )
            self._publish_status()
            return
        try:
            sequence_index = int(metadata["rule_based_sequence_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("rule writer result lacks a valid sequence identity") from exc
        current = self._current
        current_matches = bool(
            current is not None
            and int(current.directive.sequence_index) == sequence_index
        )
        if not accepted:
            if current_matches:
                self._waiting_reason = "formal_writer_rejected_sample"
            if self._writer_retry_started_at is None:
                self._writer_retry_started_at = float(now)
            elif float(now) - self._writer_retry_started_at > float(
                self.config.target_timeout_sec
            ):
                self._abort_session_step(
                    "formal writer repeatedly rejected the active rule sample"
                )
                self._publish_status()
            return
        self._admitted_capture_sequences.add(sequence_index)
        stage_id = metadata.get("collection_detail_stage_id")
        if stage_id:
            stage_key = str(stage_id)
            self._accepted_samples_by_detail_stage[stage_key] = (
                self._accepted_samples_by_detail_stage.get(stage_key, 0) + 1
            )
        # This timer describes writer health, not ownership of whichever
        # 60 Hz reference is active when the detached ACK is drained.  Any
        # successful terminal ACK proves that the writer made fresh progress
        # and therefore closes a preceding rejection interval, even when the
        # motion cursor has already advanced to a newer source sequence.
        self._writer_retry_started_at = None
        # Process-writer ACKs normally arrive after the 60 Hz motion cursor has
        # advanced.  The admission metadata is the causal identity; requiring
        # it to remain the active motion frame coupled the 15 Hz writer back
        # into control and raised a false mismatch on healthy late ACKs.
        if current_matches:
            self._current_sample_accepted = True
            self._waiting_reason = ""
            if self._target_arrival_confirmed:
                self._finish_arrived_current(float(now))

    def _establish_hold(self, reason: str) -> None:
        self._reset_pose_action_motion_intent("runtime_hold")
        if self._held or self._components is None:
            return
        append_message = getattr(self._control_ui, "append_message", None)
        if callable(append_message):
            append_message(
                "RuleBased",
                "规则运行时因以下原因请求 Hold：" + str(reason),
            )
        request_hold = getattr(self._clutch_input, "request_hold", None)
        if callable(request_hold):
            request_hold(reason="rule_based_runtime_hold")
        self._held = True
        try:
            self._components.session.hold_hardware(str(reason))
        except Exception as exc:  # noqa: BLE001 - queued owner-loop Hold remains authoritative.
            if callable(append_message):
                append_message(
                    "RuleBased",
                    "规则运行时直接 Hold 未获得回执；已转交权威控制循环重试："
                    f"{type(exc).__name__}: {exc}",
                )

    def close_before_session(
        self,
        *,
        components: object,
        terminal_error: BaseException | None,
    ) -> None:
        del terminal_error
        if self._closed:
            return
        # Stop every admission predicate before waiting on lower resources.
        # A concurrent late GUI poll then rejects work instead of submitting
        # into an executor while shutdown is in progress.
        self._closed = True
        if components is self._components and self._started:
            self._establish_hold("rule-based runtime closing")
        self._cancel_initial_alignment_planning()
        # This hook runs before the session tears down the shared retargeter.
        # Wait for an in-flight read-only certification to leave that resource
        # rather than racing hardware/session cleanup against its worker.
        self._initial_alignment_executor.shutdown(wait=True, cancel_futures=True)
        self._configuration_executor.shutdown(wait=True, cancel_futures=True)
        pending = self._next_attempt_rebuild_future
        self._next_attempt_rebuild_future = None
        self._next_attempt_rebuild_context = None
        if pending is not None and pending.done() and not pending.cancelled():
            self._dispose_rebuilt_extension_future(pending)
        reachability_rebuild = self._writing_reachability_rebuild_future
        self._writing_reachability_rebuild_future = None
        self._writing_reachability_rebuild_ordinal = None
        if (
            reachability_rebuild is not None
            and reachability_rebuild.done()
            and not reachability_rebuild.cancelled()
        ):
            self._dispose_rebuilt_extension_future(reachability_rebuild)
        self._writing_reachability_future = None

    def _publish_status(
        self,
        *,
        recording_ready: bool | None = None,
        force: bool = False,
    ) -> None:
        publish = getattr(self._control_ui, "publish_rule_status", None)
        if not callable(publish):
            return
        published_recording_ready = (
            bool(recording_ready)
            if recording_ready is not None
            else False
            if self._paused_hold_emission is not None
            else bool(
                self._recording_active is not None and self._recording_active()
            )
        )
        session = getattr(self._components, "session", None)
        session_state = getattr(getattr(session, "state_machine", None), "state", None)
        session_state_value = str(
            getattr(session_state, "value", session_state) or ""
        ).upper()
        directive = None if self._current is None else self._current.directive
        # Lifecycle/safety edges publish immediately.  High-rate progress
        # fields (reference sequence, stage frame, accepted count) are omitted
        # deliberately: they are observational and must not turn JSON/UI work
        # into a second 60 Hz control-loop consumer.
        urgent_signature = (
            self._started,
            self._armed,
            self._execution_begun,
            self._completed,
            self._aborted,
            self._held,
            self._waiting_reason,
            None if directive is None else directive.step_id,
            None if directive is None else directive.kind,
            tuple(sorted(self._boundaries)),
            self._stage_label,
            self._stage_index,
            self._stage_frame_total,
            self._pending_gripper_action,
            self._gripper_action_id,
            self._gripper_action_intent,
            self._gripper_action_result,
            self._gripper_action_status,
            self._configuration_future is not None,
            self._next_attempt_rebuild_future is not None,
            self._configuration_error,
            self._writing_reachability_status,
            self._writing_reachability_detail,
            self._writing_reachability_skipped_count,
            self._configuration_batch_index,
            self._configuration_batch_completed_count,
            self._configuration_batch_cycle_complete,
            self._progress_scope_admission_error,
            tuple(sorted(self._height_calibration_metadata().items())),
            self._height_calibration_pending,
            self._height_calibration_error,
            session_state_value,
            published_recording_ready,
        )
        now = time.monotonic()
        if (
            not force
            and urgent_signature == self._last_status_publish_urgent_signature
            and self._last_status_publish_at is not None
            and now - self._last_status_publish_at < 1.0 / _RULE_UI_STATUS_HZ
        ):
            return
        status = asdict(self.status())
        if recording_ready is not None:
            status["recording_ready"] = bool(recording_ready)
        publish(status)
        self._last_status_publish_at = now
        self._last_status_publish_urgent_signature = urgent_signature

    def _rule_pre_dispatch_session_blocker_is_expected(
        self,
        *,
        session_state_value: str,
        key: str,
        detail: str,
    ) -> bool:
        """Classify the one expected pre-first-command Session diagnostic.

        A dedicated rule start intentionally opens the formal writer before
        the first arm command: the first command is produced by the rule
        initial-alignment plan.  The generic Rig report may still describe
        that ARMED/no-dispatch interval as missing arm admission.  Only this
        exact, command-inert interval is filtered; hand/backend faults and
        every non-ARMED Session state remain blocking.
        """

        if (
            str(key) != "rig.session"
            or str(session_state_value).upper() not in {"ARMED", "RUNNING"}
            or not self._armed
            or self._execution_begun
            or self._completed
            or self._aborted
            or self._held
        ):
            return False
        normalized = " ".join(str(detail or "").lower().split())
        return bool(
            "runtime home" in normalized
            and "hardware arm admission" in normalized
            and (
                "no command" in normalized
                or "not dispatched" in normalized
                or "dispatch" in normalized
                or "before rule collection" in normalized
            )
        )

    def _session_readiness_diagnosis(
        self,
        *,
        session_state_value: str,
    ) -> tuple[dict[str, str], str]:
        """Return the current Rig blockers and a concise operator reason."""

        session = getattr(self._components, "session", None)
        rig = getattr(session, "rig", None)
        cache_key = (
            str(session_state_value or "").upper(),
            bool(self._started),
            bool(self._closed),
            tuple(self.config.gripper_sides),
            id(rig),
            tuple(
                id(getattr(self._components, f"{side}_hand_backend", None))
                for side in self.config.gripper_sides
            ),
        )
        now = time.monotonic()
        cached = self._readiness_diagnosis_cache
        if (
            cached is not None
            and cached[1] == cache_key
            and now < self._readiness_diagnosis_cache_expires_at
        ):
            blockers, reason = cached[0]
            return dict(blockers), str(reason)

        def cache_result(result: tuple[dict[str, str], str]) -> tuple[dict[str, str], str]:
            self._readiness_diagnosis_cache = (
                (dict(result[0]), str(result[1])),
                cache_key,
            )
            # A quarter second is short enough for an operator retry to be
            # reflected promptly, while keeping a serial state read out of
            # every realtime control frame during a persistent fault.
            self._readiness_diagnosis_cache_expires_at = now + 0.25
            return result

        report_fn = getattr(rig, "readiness_report", None)
        report: Mapping[str, object] | None = None
        if callable(report_fn):
            try:
                candidate = report_fn()
            except Exception as exc:  # noqa: BLE001 - UI diagnosis is observational.
                return cache_result((
                    {"rig.session": f"readiness report failed: {type(exc).__name__}: {exc}"},
                    f"Rig readiness report failed: {type(exc).__name__}: {exc}",
                ))
            if isinstance(candidate, Mapping):
                report = candidate

        blockers = readiness_report_blockers(report or {})
        for key, detail in tuple(blockers.items()):
            if self._rule_pre_dispatch_session_blocker_is_expected(
                session_state_value=session_state_value,
                key=key,
                detail=detail,
            ):
                blockers.pop(key, None)
        reported_blockers = set(blockers)

        # The rule plan's replay may contain a configured hand effector even
        # while a legacy/compatibility builder supplied an incomplete Rig map.
        # Check the required binding and backend readiness independently so
        # the UI cannot expose a start button that would later fail with a
        # missing-effector KeyError.  The canonical startup path preserves
        # this binding even when the backend is temporarily unready.
        rig_backends = getattr(rig, "backends", None)
        for side in self.config.gripper_sides:
            effector = f"{side}_hand"
            backend = getattr(self._components, f"{side}_hand_backend", None)
            if backend is None:
                blockers.setdefault(effector, "configured hand backend unavailable")
                continue
            if bool(
                getattr(self._components, f"{side}_hand_control_via_rig", False)
            ) and (
                not isinstance(rig_backends, Mapping)
                or effector not in rig_backends
            ):
                blockers.setdefault(effector, "backend missing from Session Rig")
                continue
            is_ready = getattr(backend, "is_ready", None)
            try:
                backend_ready = bool(callable(is_ready) and is_ready())
            except Exception as exc:  # noqa: BLE001 - readiness is diagnostic only.
                backend_ready = False
                backend_reason = f"readiness check failed: {type(exc).__name__}: {exc}"
            else:
                backend_reason = "backend is not ready"
            if not backend_ready:
                backend_report_fn = getattr(backend, "readiness_report", None)
                if callable(backend_report_fn):
                    try:
                        backend_report = backend_report_fn()
                    except Exception as exc:  # noqa: BLE001 - keep a bounded reason.
                        backend_reason = (
                            f"readiness report failed: {type(exc).__name__}: {exc}"
                        )
                    else:
                        if isinstance(backend_report, Mapping):
                            backend_reason = format_readiness_report(
                                {"backends": {effector: backend_report}}
                            )
                blockers.setdefault(effector, backend_reason)

        if blockers:
            reason_parts: list[str] = []
            if reported_blockers:
                reason_parts.append(format_readiness_report(report or {}))
            reason_parts.extend(
                f"{effector} {reason}"
                for effector, reason in blockers.items()
                if effector not in reported_blockers
            )
            return cache_result((blockers, "; ".join(reason_parts) or "rig is not ready"))
        if self._started and session_state_value not in {"ARMED", "RUNNING"}:
            reason = (
                f"Session state={session_state_value or 'UNKNOWN'}; "
                "runtime Home and hardware arm admission are required before rule collection."
            )
            return cache_result(({"rig.session": reason}, reason))
        return cache_result(({}, ""))

    def status(self) -> RuleBasedRealTeleopStatus:
        directive = None if self._current is None else self._current.directive
        recorded = 0
        provider = self._dataset_recorder_provider
        if provider is not None:
            try:
                recorder = provider()
                recorded = int(getattr(recorder, "frame_count", 0) or 0)
            except Exception:
                recorded = 0
        session = getattr(self._components, "session", None)
        session_state = getattr(getattr(session, "state_machine", None), "state", None)
        session_state_value = str(
            getattr(session_state, "value", session_state) or ""
        ).upper()
        readiness_blockers, readiness_reason = self._session_readiness_diagnosis(
            session_state_value=session_state_value,
        )
        if self._initial_alignment_cancel_pending:
            readiness_blockers = dict(readiness_blockers)
            readiness_blockers.setdefault(
                "rule.initial_alignment",
                "cancelled initial-alignment planner is still settling",
            )
        if self._collection_transition_pending():
            readiness_blockers = dict(readiness_blockers)
            readiness_blockers.setdefault(
                "dataset.lifecycle",
                "cancelled rule Episode lifecycle is still settling",
            )
        elif (
            self._held
            and self._recording_active is not None
            and self._recording_active()
        ):
            readiness_blockers = dict(readiness_blockers)
            readiness_blockers.setdefault(
                "dataset.lifecycle",
                "cancelled rule Episode is still active and awaiting discard",
            )
        if self._progress_scope_admission_error:
            readiness_blockers = dict(readiness_blockers)
            readiness_blockers.setdefault(
                "progress.profile",
                self._progress_scope_admission_error,
            )
            readiness_reason = "; ".join(
                part
                for part in (readiness_reason, self._progress_scope_admission_error)
                if part
            )
        hold_owner_ready, hold_owner_detail = self._right_gripper_hold_owner_ready()
        if not hold_owner_ready:
            readiness_blockers = dict(readiness_blockers)
            readiness_blockers.setdefault(
                "rule.right_hand_hold_owner",
                hold_owner_detail
                or "右夹爪尚未建立书写用的一次性闭合 owner。",
            )
            readiness_reason = "; ".join(
                part
                for part in (readiness_reason, hold_owner_detail)
                if part
            )
        gripper_backends_ready = True
        for side in self.config.gripper_sides:
            backend = getattr(self._components, f"{side}_hand_backend", None)
            is_ready = getattr(backend, "is_ready", None)
            try:
                side_ready = bool(callable(is_ready) and is_ready())
            except Exception:
                side_ready = False
            gripper_backends_ready = gripper_backends_ready and side_ready
        # The UI and dispatch path consume the same command-inert lifecycle
        # gate.  RUNNING is valid only in the inter-Episode state produced by
        # the successful reset above; an active/finalizing writer remains a
        # hard blocker.
        hardware_readiness_blockers = {
            key: value
            for key, value in readiness_blockers.items()
            if key != "rule.right_hand_hold_owner"
        }
        gripper_control_available = bool(
            gripper_backends_ready
            and not hardware_readiness_blockers
            and self._manual_gripper_lifecycle_available(
                session_state_value=session_state_value,
            )
            and self._pending_gripper_action is None
        )
        gripper_action_pending = self._pending_gripper_action is not None
        if (
            self._writing_reachability_enabled()
            and self._height_calibration is not None
            and self._height_calibration.enabled
            and not self._writing_reachability_passed_for_current_selection()
        ):
            readiness_blockers = dict(readiness_blockers)
            reachability_detail = (
                self._writing_reachability_detail
                or "书写四边界接触点尚未完成可达性检查。"
            )
            readiness_blockers.setdefault(
                "rule.writing_reachability",
                reachability_detail,
            )
            readiness_reason = "; ".join(
                part for part in (readiness_reason, reachability_detail) if part
            )
        fix_controller = getattr(self._components, "fix_arm_controller", None)
        fixed_sides = tuple(getattr(fix_controller, "fixed_sides", ()) or ())
        rule_start_available = bool(
            self._started
            and not self._closed
            # A successful rule Episode deliberately does not force Session
            # Hold.  The shared Session therefore remains safely idle in
            # RUNNING after writer close; accepting the next operator START
            # here authorizes a fresh Episode and initial-alignment plan, not
            # an automatic command.
            and session_state_value in {"ARMED", "RUNNING"}
            and not self._held
            and not self._aborted
            and not self._completed
            and not self._armed
            and not self._execution_begun
            and self._configuration_future is None
            and self._next_attempt_rebuild_future is None
            and self._height_calibration_future is None
            and not self._height_calibration_pending
            and not gripper_action_pending
            and not fixed_sides
            and not readiness_blockers
            and hold_owner_ready
        )
        height_mutation_idle = self._height_calibration_mutation_idle()
        height_calibration_available = bool(
            height_mutation_idle
            and session_state_value in {"ARMED", "RUNNING"}
            # The rule-start-only hold-owner receipt must not disable the
            # command-inert height step before the task chain has performed
            # its explicit right-gripper close.  Transport/session faults do
            # remain hard blockers through this filtered view.
            and not hardware_readiness_blockers
        )
        calibration = self._height_calibration
        waiting_reason = self._waiting_reason
        if (
            self._started
            and readiness_blockers
            and not self._armed
            and not self._execution_begun
            and not self._completed
            and not self._aborted
        ):
            if set(readiness_blockers) == {"rule.right_hand_hold_owner"}:
                waiting_reason = "right_gripper_hold_owner_pending"
            elif set(readiness_blockers) == {"rule.writing_reachability"}:
                waiting_reason = str(
                    self._waiting_reason or "writing_reachability_preflight"
                )
            else:
                waiting_reason = "session_readiness_failure"
        side_labels = {"left": "左夹爪", "right": "右夹爪"}
        topology = (
            "双臂（无夹爪）"
            if not self.config.gripper_sides
            else "双臂 + "
            + " + ".join(side_labels[side] for side in self.config.gripper_sides)
        )
        return RuleBasedRealTeleopStatus(
            started=self._started,
            armed=self._armed,
            execution_begun=self._execution_begun,
            completed=self._completed,
            aborted=self._aborted,
            held=self._held,
            recording_ready=bool(
                self._recording_active is not None and self._recording_active()
            ),
            waiting_reason=waiting_reason,
            step_id=None if directive is None else directive.step_id,
            directive_kind=None if directive is None else directive.kind,
            sequence_index=None if directive is None else directive.sequence_index,
            published_boundaries=tuple(sorted(self._boundaries)),
            stage_label=self._stage_label,
            stage_index=self._stage_index,
            stage_frame_index=self._stage_frame_index,
            stage_frame_total=self._stage_frame_total,
            recorded_sample_count=max(0, recorded),
            planned_sample_count=int(self._planned_sample_count),
            collection_position=self._collection_position,
            initial_alignment_speed_scale=float(
                self.config.initial_alignment_speed_scale
            ),
            pre_pen_down_replay_active=bool(self._pre_pen_down_replay_active),
            pre_pen_down_replay_normal_episodes=int(
                self.config.pre_pen_down_replay_normal_episodes
            ),
            pre_pen_down_normal_completed_count=int(
                self._pre_pen_down_normal_completed_count
            ),
            gripper_sides=tuple(self.config.gripper_sides),
            control_topology=topology,
            gripper_control_available=gripper_control_available,
            gripper_action_pending=gripper_action_pending,
            gripper_action_id=self._gripper_action_id,
            gripper_action_intent=str(self._gripper_action_intent),
            gripper_action_result=str(self._gripper_action_result),
            rule_start_available=rule_start_available,
            writing_reachability_status=str(self._writing_reachability_status),
            writing_reachability_detail=str(self._writing_reachability_detail),
            writing_reachability_skipped_count=int(
                self._writing_reachability_skipped_count
            ),
            gripper_action_status=str(self._gripper_action_status),
            rule_configuration=copy.deepcopy(self._selected_configuration),
            source_selection=copy.deepcopy(self.source_selection),
            rule_configuration_batch=tuple(
                copy.deepcopy(configuration)
                for configuration, _replacement in self._configuration_batch
            ),
            batch_index=int(self._configuration_batch_index),
            batch_count=max(1, len(self._configuration_batch)),
            batch_completed_count=int(self._configuration_batch_completed_count),
            batch_cycle_complete=bool(self._configuration_batch_cycle_complete),
            configuration_available=self._configuration_lifecycle_available(),
            configuration_pending=bool(
                self._configuration_future is not None
                or self._next_attempt_rebuild_future is not None
                or self._writing_reachability_future is not None
                or self._writing_reachability_rebuild_future is not None
            ),
            configuration_error=str(self._configuration_error),
            session_state=session_state_value,
            readiness_reason=readiness_reason,
            readiness_blockers=dict(readiness_blockers),
            collection_mode=str(
                self.source_selection.get("collection_mode")
                or "rule_based_writing"
            ),
            height_calibration_enabled=bool(
                calibration is not None and calibration.enabled
            ),
            height_calibrated_z_m=(
                None if calibration is None else float(calibration.calibrated_z_m)
            ),
            height_lower_tolerance_m=float(
                self._height_lower_tolerance_m
                if calibration is None
                else calibration.lower_tolerance_m
            ),
            height_upper_tolerance_m=float(
                self._height_upper_tolerance_m
                if calibration is None
                else calibration.upper_tolerance_m
            ),
            height_lower_bound_z_m=(
                None if calibration is None else calibration.lower_bound_z_m
            ),
            height_upper_bound_z_m=(
                None if calibration is None else calibration.upper_bound_z_m
            ),
            height_calibration_available=height_calibration_available,
            height_calibration_clear_available=bool(
                height_mutation_idle and calibration is not None
            ),
            height_calibration_pending=bool(self._height_calibration_pending),
            height_calibration_error=str(self._height_calibration_error),
        )


__all__ = [
    "RULE_BASED_REAL_TELEOP_SCHEMA_V1",
    "RuleBasedRealTeleopConfig",
    "RuleBasedRealTeleopDeployment",
    "RuleBasedRealTeleopExtension",
    "RuleBasedRealTeleopStatus",
    "RuleBasedWritingHeightCalibration",
    "preflight_rule_based_real_teleop_deployment",
]
