from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from robot_motion.pose import pose_delta_metrics

from ...diagnostics import FaultEvent, FaultSeverity
from ...schemas import EffectorCommand, TeleopCommand
from ...transforms import as_pose_matrix
from ..home_capture_stability import HomeCaptureStabilityState

INIT_ARM_FAULT_CODES = frozenset(
    {
        "missing_channel",
        "invalid_channel",
        "invalid_pose",
        "stale_channel",
        "translation_jump",
        "rotation_jump",
    }
)

# Display-only arrival criteria.  They intentionally match the collection
# audit contract but never gate init-arm motion or home recapture.
INIT_ARM_ARRIVAL_POSITION_TOLERANCE_M = 0.0015
INIT_ARM_ARRIVAL_ROTATION_TOLERANCE_DEG = 0.25
INIT_ARM_ARRIVAL_STILL_TRANSLATION_SPEED_M_S = 0.030
INIT_ARM_ARRIVAL_STILL_ROTATION_SPEED_DEG_S = 5.0
INIT_ARM_ARRIVAL_DWELL_SEC = 2.0
INIT_ARM_ARRIVAL_MAX_SAMPLE_GAP_SEC = 0.5


def normalize_init_arm_side(side: str) -> str:
    normalized = str(side or "").strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"init-arm side must be 'left' or 'right', got {side!r}.")
    return normalized


def _finite_vector(value: Any, *, length: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite {length}D vector, got shape {array.shape}.")
    return array.copy()


def _pose_or_none(value: Any, *, label: str) -> np.ndarray | None:
    if value is None:
        return None
    pose = as_pose_matrix(value)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{label} must be a finite 4x4 pose matrix.")
    return pose.copy()


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, Any] | tuple[None, None]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return key, mapping[key]
    return None, None


def _side_payload(payload: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    candidates: list[Any] = []
    sides = payload.get("sides")
    if isinstance(sides, Mapping):
        candidates.extend([sides.get(side), sides.get(f"{side}_arm")])
    # The repository's default RM65 qpos target predates the generic
    # ``sides`` profile schema and stores the same per-side payload under an
    # ``arms`` object.  Accept that layout as a compatibility alias so the
    # default Init-arm profile is actually loadable.
    arms = payload.get("arms")
    if isinstance(arms, Mapping):
        candidates.extend([arms.get(side), arms.get(f"{side}_arm")])
    candidates.extend(
        [
            payload.get(side),
            payload.get(f"{side}_arm"),
            payload.get(f"{side}_init"),
            payload.get(f"{side}_init_pose"),
        ]
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return candidate
    return {}


@dataclass(frozen=True)
class InitArmResolvedTarget:
    side: str
    tcp_pose_base: np.ndarray
    tcp_pose_desk: np.ndarray | None = None
    solver_pose_desk: np.ndarray | None = None
    arm_qpos: np.ndarray | None = None
    source_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", normalize_init_arm_side(self.side))
        object.__setattr__(self, "tcp_pose_base", as_pose_matrix(self.tcp_pose_base))
        if self.tcp_pose_desk is not None:
            object.__setattr__(self, "tcp_pose_desk", as_pose_matrix(self.tcp_pose_desk))
        if self.solver_pose_desk is not None:
            object.__setattr__(self, "solver_pose_desk", as_pose_matrix(self.solver_pose_desk))
        if self.arm_qpos is not None:
            object.__setattr__(self, "arm_qpos", _finite_vector(self.arm_qpos, length=6, label=f"{self.side}.arm_qpos"))


@dataclass(frozen=True)
class _RawInitArmTarget:
    side: str
    tcp_pose_base: np.ndarray | None = None
    tcp_pose_desk: np.ndarray | None = None
    arm_qpos: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitArmTransition:
    ok: bool
    message: str
    side: str = ""
    mode: str = ""


@dataclass
class _RecalibrationState:
    started_at: float = 0.0
    timeout_sec: float = 0.0
    min_samples: int = 1
    samples: list[np.ndarray] = field(default_factory=list)
    samples_by_side: dict[str, list[np.ndarray]] = field(default_factory=dict)
    stability_by_side: dict[str, HomeCaptureStabilityState] = field(
        default_factory=dict
    )


@dataclass
class _ArrivalState:
    last_actual_pose_base: np.ndarray | None = None
    last_timestamp: float | None = None
    stable_since: float | None = None
    status: str = "inactive"
    reason: str = ""
    position_error_m: float | None = None
    rotation_error_deg: float | None = None
    translation_speed_m_s: float | None = None
    rotation_speed_deg_s: float | None = None
    dwell_sec: float = 0.0


@dataclass
class _QposInitOperation:
    operation_id: str
    side: str
    start_qpos_by_side: dict[str, np.ndarray]
    target_qpos_by_side: dict[str, np.ndarray]
    feedback_provider: Callable[[], Mapping[str, Any]]
    plan_future: Future[Any]
    started_at: float
    max_tracking_error_deg: float
    start_drift_tolerance_deg: float
    arrival_tolerance_deg: float
    arrival_dwell_sec: float
    settle_timeout_sec: float
    tracking_check_every: int
    planning_timeout_sec: float = 30.0
    plan: Any | None = None
    step_cursor: int = 0
    execution_started_at: float | None = None
    last_tracking_check_step_cursor: int = 0
    last_emitted_qpos_by_side: dict[str, np.ndarray] | None = None
    last_feedback_at: float = 0.0
    stable_since: float | None = None
    max_error_deg: float | None = None
    terminal_status: str = ""
    terminal_message: str = ""
    terminal_consumed: bool = False


def _copy_qpos_by_side(value: Mapping[str, Any], *, label: str) -> dict[str, np.ndarray]:
    missing = [side for side in ("left", "right") if side not in value]
    if missing:
        raise ValueError(f"{label} is missing sides {missing}.")
    return {
        side: _finite_vector(value[side], length=6, label=f"{label}.{side}")
        for side in ("left", "right")
    }


def _qpos_from_effector_command(command: EffectorCommand | None) -> np.ndarray | None:
    if command is None:
        return None
    metadata = command.metadata if isinstance(command.metadata, Mapping) else {}
    for key in ("backend_final_target_qpos", "solver_output_qpos"):
        candidate = metadata.get(key)
        if candidate is None:
            continue
        try:
            return _finite_vector(candidate, length=6, label=key)
        except (TypeError, ValueError):
            continue
    if str(command.command_type) == "joint_vector":
        try:
            return _finite_vector(command.value, length=6, label="joint_vector")
        except (TypeError, ValueError):
            return None
    return None


def _parse_raw_side_target(
    payload: Mapping[str, Any],
    side: str,
    *,
    source_path: str,
) -> _RawInitArmTarget:
    side_payload = _side_payload(payload, side)
    metadata = dict(side_payload.get("metadata") or {})
    metadata.setdefault("source_path", source_path)

    q_key, q_value = _first_present(
        side_payload,
        (
            "arm_qpos",
            "qpos",
            "joint_qpos",
            "target_qpos",
            "init_qpos",
            "arm_qpos_rad",
            "qpos_rad",
            "joint_rad",
            "arm_qpos_deg",
            "qpos_deg",
            "joint_deg",
        ),
    )
    qpos = None
    if q_value is not None:
        qpos = _finite_vector(q_value, length=6, label=f"{side}.{q_key}")
        unit = str(side_payload.get("qpos_unit") or payload.get("qpos_unit") or "").strip().lower()
        if q_key and (str(q_key).endswith("_deg") or unit in {"deg", "degree", "degrees"}):
            qpos = np.deg2rad(qpos)

    pose_base = _pose_or_none(
        _first_present(
            side_payload,
            ("tcp_pose_base", "target_pose_base", "pose_base", "init_pose_base"),
        )[1],
        label=f"{side}.tcp_pose_base",
    )
    pose_desk = _pose_or_none(
        _first_present(
            side_payload,
            ("tcp_pose_desk", "target_pose_desk", "pose_desk", "init_pose_desk"),
        )[1],
        label=f"{side}.tcp_pose_desk",
    )
    if qpos is None and pose_base is None and pose_desk is None:
        raise ValueError(
            f"init-arm profile side {side!r} needs one of arm_qpos, tcp_pose_base, or tcp_pose_desk."
        )
    return _RawInitArmTarget(
        side=side,
        tcp_pose_base=pose_base,
        tcp_pose_desk=pose_desk,
        arm_qpos=qpos,
        metadata=metadata,
    )


def _raw_targets_from_profile(path: str | Path) -> dict[str, _RawInitArmTarget]:
    source_path = str(path)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("init-arm profile must be a JSON object.")
    return {
        side: _parse_raw_side_target(payload, side, source_path=source_path)
        for side in ("left", "right")
    }


def _calibration_desk_to_base(calibration: Any, pose_desk: np.ndarray, side: str) -> np.ndarray:
    desk_to_base = getattr(calibration, "desk_to_base", None)
    if not callable(desk_to_base):
        raise ValueError("init-arm tcp_pose_desk requires a TeleopCalibration with desk_to_base().")
    return as_pose_matrix(desk_to_base(pose_desk, side=side))


def _calibration_base_to_desk(calibration: Any, pose_base: np.ndarray, side: str) -> np.ndarray | None:
    get_arm = getattr(calibration, "get_arm_for_side", None)
    if not callable(get_arm):
        return None
    arm = get_arm(side)
    transform = getattr(arm, "T_D_B", None)
    if transform is None:
        return None
    return as_pose_matrix(transform) @ as_pose_matrix(pose_base)


def _dual_kinematics_method(dual_arm_retargeter: Any, name: str):
    method = getattr(dual_arm_retargeter, name, None)
    if not callable(method):
        raise TypeError(
            "dual-arm retargeter does not implement the shared kinematics "
            f"interface method {name}()."
        )
    return method


def load_init_arm_targets(
    path: str | Path,
    *,
    calibration: Any,
    dual_arm_retargeter: Any | None = None,
) -> dict[str, InitArmResolvedTarget]:
    raw_targets = _raw_targets_from_profile(path)
    current_left_qpos: np.ndarray | None = None
    current_right_qpos: np.ndarray | None = None
    solver_targets_for_profile: dict[str, np.ndarray] | None = None
    tcp_targets_for_profile: dict[str, np.ndarray] | None = None
    if dual_arm_retargeter is not None:
        current = _dual_kinematics_method(
            dual_arm_retargeter,
            "current_qpos_by_side",
        )()
        if not isinstance(current, Mapping):
            raise TypeError("current_qpos_by_side() must return a mapping.")
        current_left_qpos = _finite_vector(
            current.get("left"), length=6, label="current left qpos"
        )
        current_right_qpos = _finite_vector(
            current.get("right"), length=6, label="current right qpos"
        )

        if any(target.arm_qpos is not None for target in raw_targets.values()):
            left_profile_qpos = raw_targets["left"].arm_qpos
            right_profile_qpos = raw_targets["right"].arm_qpos
            if left_profile_qpos is None:
                left_profile_qpos = current_left_qpos
            if right_profile_qpos is None:
                right_profile_qpos = current_right_qpos
            if left_profile_qpos is None or right_profile_qpos is None:
                raise ValueError("cannot resolve complete bimanual qpos for init-arm FK.")
            pair_kwargs = {
                "left_qpos": np.asarray(left_profile_qpos, dtype=float).reshape(6),
                "right_qpos": np.asarray(right_profile_qpos, dtype=float).reshape(6),
            }
            solver_targets_for_profile = _dual_kinematics_method(
                dual_arm_retargeter,
                "solver_world_targets_for_qpos_pair",
            )(**pair_kwargs)
            tcp_targets_for_profile = _dual_kinematics_method(
                dual_arm_retargeter,
                "tool_tcp_world_targets_for_qpos_pair",
            )(**pair_kwargs)

    resolved: dict[str, InitArmResolvedTarget] = {}
    for side in ("left", "right"):
        raw = raw_targets[side]
        tcp_pose_base = None if raw.tcp_pose_base is None else as_pose_matrix(raw.tcp_pose_base)
        tcp_pose_desk = None if raw.tcp_pose_desk is None else as_pose_matrix(raw.tcp_pose_desk)
        solver_pose_desk = None

        if raw.arm_qpos is not None:
            if dual_arm_retargeter is None:
                if tcp_pose_base is None and tcp_pose_desk is None:
                    raise ValueError(
                        f"init-arm profile side {side!r} only has arm_qpos, "
                        "but dual-arm FK is unavailable. Add tcp_pose_base/tcp_pose_desk."
                    )
            else:
                if solver_targets_for_profile is None or tcp_targets_for_profile is None:
                    raise RuntimeError("shared dual-arm FK results were not prepared.")
                solver_pose_desk = as_pose_matrix(solver_targets_for_profile[side])
                tcp_pose_desk = as_pose_matrix(tcp_targets_for_profile[side])
                tcp_pose_base = _calibration_desk_to_base(calibration, tcp_pose_desk, side)

        if tcp_pose_base is None and tcp_pose_desk is not None:
            tcp_pose_base = _calibration_desk_to_base(calibration, tcp_pose_desk, side)
        if tcp_pose_desk is None and tcp_pose_base is not None:
            tcp_pose_desk = _calibration_base_to_desk(calibration, tcp_pose_base, side)
        if solver_pose_desk is None and tcp_pose_desk is not None and dual_arm_retargeter is not None:
            solver_pose_desk = as_pose_matrix(
                _dual_kinematics_method(
                    dual_arm_retargeter,
                    "solver_world_target_from_tool_tcp_world_target",
                )(
                    side=side,
                    tool_tcp_world_target=tcp_pose_desk,
                )
            )
        if tcp_pose_base is None:
            raise ValueError(f"init-arm profile side {side!r} could not be resolved to tcp_pose_base.")

        resolved[side] = InitArmResolvedTarget(
            side=side,
            tcp_pose_base=tcp_pose_base,
            tcp_pose_desk=tcp_pose_desk,
            solver_pose_desk=solver_pose_desk,
            arm_qpos=raw.arm_qpos,
            source_path=str(path),
            metadata=dict(raw.metadata),
        )
    return resolved


class InitArmCommandController:
    """Route one selected arm to an init pose without changing dataset schemas."""

    def __init__(
        self,
        *,
        targets_by_side: Mapping[str, InitArmResolvedTarget],
        selected_side: str = "left",
        use_dual_target_override: bool = False,
        qpos_final_collision_checker: Callable[..., Any] | None = None,
    ) -> None:
        self.targets_by_side = {
            normalize_init_arm_side(side): target
            for side, target in targets_by_side.items()
        }
        if set(self.targets_by_side) != {"left", "right"}:
            raise ValueError("init-arm requires resolved left and right targets.")
        self._selected_side = normalize_init_arm_side(selected_side)
        self._mode = "live"
        self._active_side = ""
        self._recalibration = _RecalibrationState()
        self._arrival = _ArrivalState()
        self._status_message = "init-arm ready"
        self._use_dual_target_override = bool(use_dual_target_override)
        self._qpos_final_collision_checker = qpos_final_collision_checker
        self._qpos_operation: _QposInitOperation | None = None
        self._qpos_operation_sequence = 0
        self._qpos_planner_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rm65_gui_qpos_init_plan",
        )

    @property
    def enabled(self) -> bool:
        return True

    @property
    def selected_side(self) -> str:
        return self._selected_side

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def active_side(self) -> str:
        return self._active_side

    @property
    def fixed_side(self) -> str:
        return self._active_side

    @property
    def active(self) -> bool:
        return (
            self._mode in {"moving", "init_fixed", "recapturing"} or self.qpos_active
        ) and bool(self._active_side)

    @property
    def moving(self) -> bool:
        return self._mode in {
            "moving",
            "qpos_planning",
            "qpos_executing",
            "qpos_settling",
        }

    @property
    def fixed(self) -> bool:
        return self._mode in {"init_fixed", "qpos_init_fixed"} and bool(
            self._active_side
        )

    @property
    def fixed_sides(self) -> tuple[str, ...]:
        return (self._active_side,) if self.fixed else ()

    @property
    def fix_hold_sides(self) -> tuple[str, ...]:
        """Sides that still keep Fix-Hold paused until release is complete."""

        return (self._active_side,) if (self.fixed or self.recapturing) else ()

    @property
    def qpos_active(self) -> bool:
        return self._mode.startswith("qpos_") and self._qpos_operation is not None

    @property
    def qpos_operation_id(self) -> str:
        operation = self._qpos_operation
        return "" if operation is None else str(operation.operation_id)

    @property
    def recapturing(self) -> bool:
        return self._mode in {"recapturing", "qpos_recapturing"} and bool(
            self._active_side
        )

    @property
    def recalibration_sides(self) -> tuple[str, ...]:
        if not self.recapturing:
            return ()
        sides = tuple(self._recalibration.samples_by_side)
        return sides or (self._active_side,)

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def sample_count(self) -> int:
        if self._recalibration.samples_by_side:
            return min(
                len(samples)
                for samples in self._recalibration.samples_by_side.values()
            )
        return len(self._recalibration.samples)

    @property
    def min_samples(self) -> int:
        return max(1, int(self._recalibration.min_samples))

    @property
    def recalibration_started_at(self) -> float:
        return float(self._recalibration.started_at)

    @property
    def recalibration_timeout_sec(self) -> float:
        return float(self._recalibration.timeout_sec)

    def recalibration_samples(self, side: str | None = None) -> list[np.ndarray]:
        if self._recalibration.samples_by_side:
            normalized = normalize_init_arm_side(side or self._active_side)
            return [
                sample.copy()
                for sample in self._recalibration.samples_by_side.get(normalized, ())
            ]
        return [sample.copy() for sample in self._recalibration.samples]

    def recalibration_stability_reason(self, side: str | None = None) -> str:
        normalized = normalize_init_arm_side(side or self._active_side)
        state = self._recalibration.stability_by_side.get(normalized)
        return "" if state is None else str(state.reason or "")

    def set_dual_target_override_enabled(self, enabled: bool) -> None:
        self._use_dual_target_override = bool(enabled)

    def close(self) -> None:
        operation = self._qpos_operation
        if operation is not None:
            operation.plan_future.cancel()
        self._qpos_planner_executor.shutdown(wait=False, cancel_futures=True)

    def start_qpos_planning(
        self,
        *,
        start_qpos_by_side: Mapping[str, Any],
        target_qpos_by_side: Mapping[str, Any],
        feedback_provider: Callable[[], Mapping[str, Any]],
        plan_builder: Callable[[], Any],
        now: float,
        max_tracking_error_deg: float = 15.0,
        start_drift_tolerance_deg: float = 1.0,
        arrival_tolerance_deg: float = 0.75,
        arrival_dwell_sec: float = 0.5,
        settle_timeout_sec: float = 12.0,
        tracking_check_every: int = 10,
        planning_timeout_sec: float = 30.0,
    ) -> InitArmTransition:
        if self.active:
            return InitArmTransition(
                ok=False,
                message=f"{self._active_side} is already in init-arm mode",
                side=self._active_side,
                mode=self._mode,
            )
        side = self._selected_side
        target = self.targets_by_side.get(side)
        if target is None or target.arm_qpos is None:
            self._status_message = f"cannot execute exact qpos init for {side}; arm_qpos missing"
            return InitArmTransition(
                ok=False,
                message=self._status_message,
                side=side,
                mode=self._mode,
            )
        if not callable(feedback_provider) or not callable(plan_builder):
            raise TypeError("qpos init feedback_provider and plan_builder must be callable.")
        start = _copy_qpos_by_side(start_qpos_by_side, label="qpos init start")
        target_pair = _copy_qpos_by_side(target_qpos_by_side, label="qpos init target")
        future = self._qpos_planner_executor.submit(plan_builder)
        self._qpos_operation_sequence += 1
        operation_id = f"init-qpos-{self._qpos_operation_sequence}-{int(float(now) * 1_000_000)}"
        self._qpos_operation = _QposInitOperation(
            operation_id=operation_id,
            side=side,
            start_qpos_by_side=start,
            target_qpos_by_side=target_pair,
            feedback_provider=feedback_provider,
            plan_future=future,
            started_at=float(now),
            max_tracking_error_deg=max(0.1, float(max_tracking_error_deg)),
            start_drift_tolerance_deg=max(0.01, float(start_drift_tolerance_deg)),
            arrival_tolerance_deg=max(0.01, float(arrival_tolerance_deg)),
            arrival_dwell_sec=max(0.0, float(arrival_dwell_sec)),
            settle_timeout_sec=max(0.1, float(settle_timeout_sec)),
            tracking_check_every=max(1, int(tracking_check_every)),
            planning_timeout_sec=max(0.1, float(planning_timeout_sec)),
            last_emitted_qpos_by_side={side: qpos.copy() for side, qpos in start.items()},
        )
        self._active_side = side
        self._mode = "qpos_planning"
        self._recalibration = _RecalibrationState()
        self._arrival = _ArrivalState()
        self._status_message = f"{side} Init qpos path planning; both arms are held at measured start"
        return InitArmTransition(ok=True, message=self._status_message, side=side, mode=self._mode)

    def consume_qpos_terminal_event(self) -> dict[str, object] | None:
        operation = self._qpos_operation
        if operation is None or not operation.terminal_status or operation.terminal_consumed:
            return None
        operation.terminal_consumed = True
        return {
            "status": operation.terminal_status,
            "message": operation.terminal_message,
            "side": operation.side,
            "max_error_deg": operation.max_error_deg,
        }

    def qpos_seed_qpos_by_side(self) -> dict[str, np.ndarray]:
        """Return a safe complete command seed for an external qpos executor.

        The normal teleop loop supplies a complete dual-arm command before this
        controller rewrites it.  Startup maintenance intentionally has no
        tracker-derived command, so it uses the last emitted (or measured
        start) pair as a neutral seed and then invokes the same filter path.
        """

        operation = self._qpos_operation
        if operation is None or not self.qpos_active:
            return {}
        return self._operation_hold_qpos(operation)

    def select_side(self, side: str) -> InitArmTransition:
        normalized = normalize_init_arm_side(side)
        if self.active:
            return InitArmTransition(
                ok=False,
                message="init side cannot be changed while init-arm is active",
                side=self._active_side,
                mode=self._mode,
            )
        self._selected_side = normalized
        self._status_message = f"selected {normalized}"
        return InitArmTransition(ok=True, message=self._status_message, side=normalized, mode=self._mode)

    def start_init(self, *, now: float) -> InitArmTransition:
        del now
        if self.active:
            return InitArmTransition(
                ok=False,
                message=f"{self._active_side} is already in init-arm mode",
                side=self._active_side,
                mode=self._mode,
            )
        side = self._selected_side
        if side not in self.targets_by_side:
            self._status_message = f"cannot init {side}; target missing"
            return InitArmTransition(ok=False, message=self._status_message, side=side, mode=self._mode)
        self._active_side = side
        self._mode = "moving"
        self._recalibration = _RecalibrationState()
        self._arrival = _ArrivalState(status="not_reached", reason="waiting_for_actual_feedback")
        self._status_message = f"{side} moving to init pose"
        return InitArmTransition(ok=True, message=self._status_message, side=side, mode=self._mode)

    def start_recalibration(
        self,
        *,
        now: float,
        timeout_sec: float,
        min_samples: int,
    ) -> InitArmTransition:
        if not self.active:
            return InitArmTransition(
                ok=False,
                message="init-arm recalibration requires an active init side",
                side=self._selected_side,
                mode=self._mode,
            )
        if self.qpos_active:
            operation = self._qpos_operation
            assert operation is not None
            operation.plan_future.cancel()
            try:
                operation.last_emitted_qpos_by_side = self._read_qpos_feedback(operation)
            except Exception as exc:  # noqa: BLE001 - retain the last safe command as fallback.
                self._finish_qpos_operation(
                    status="failed",
                    message=(
                        "Init qpos pause could not read fresh dual-arm feedback: "
                        f"{type(exc).__name__}: {exc}; entering Hold"
                    ),
                )
                return InitArmTransition(
                    ok=False,
                    message=self._status_message,
                    side=self._active_side,
                    mode=self._mode,
                )
            support_side = "right" if operation.side == "left" else "left"
            support_drift_deg = float(
                np.max(
                    np.abs(
                        np.rad2deg(
                            operation.last_emitted_qpos_by_side[support_side]
                            - operation.start_qpos_by_side[support_side]
                        )
                    )
                )
            )
            if support_drift_deg > operation.start_drift_tolerance_deg:
                self._finish_qpos_operation(
                    status="failed",
                    message=(
                        f"Cannot pause {operation.side} Init safely: {support_side} support "
                        f"arm drifted {support_drift_deg:.3f}deg (limit "
                        f"{operation.start_drift_tolerance_deg:.3f}deg); entering Hold"
                    ),
                    override_terminal=True,
                )
                return InitArmTransition(
                    ok=False,
                    message=self._status_message,
                    side=self._active_side,
                    mode=self._mode,
                )
            self._mode = "qpos_recapturing"
            recapture_sides = (self._active_side,)
        else:
            self._mode = "recapturing"
            recapture_sides = (self._active_side,)
        self._recalibration = _RecalibrationState(
            started_at=float(now),
            timeout_sec=max(0.1, float(timeout_sec)),
            min_samples=max(1, int(min_samples)),
            samples_by_side={side: [] for side in recapture_sides},
            stability_by_side={
                side: HomeCaptureStabilityState() for side in recapture_sides
            },
        )
        self._status_message = (
            f"paused {self._active_side} exact Init qpos; recapturing target tracker Home only"
            if self._mode == "qpos_recapturing"
            else f"recapturing {self._active_side} home at init pose"
        )
        return InitArmTransition(ok=True, message=self._status_message, side=self._active_side, mode=self._mode)

    def add_recalibration_sample(
        self,
        pose_desk: np.ndarray,
        *,
        side: str | None = None,
    ) -> int:
        if not self.recapturing:
            return self.sample_count
        if self._recalibration.samples_by_side:
            normalized = normalize_init_arm_side(side or self._active_side)
            samples = self._recalibration.samples_by_side.get(normalized)
            if samples is None:
                return self.sample_count
            if len(samples) < self.min_samples:
                samples.append(np.asarray(pose_desk, dtype=float).reshape(4, 4).copy())
            return self.sample_count
        if self.sample_count < self.min_samples:
            self._recalibration.samples.append(
                np.asarray(pose_desk, dtype=float).reshape(4, 4).copy()
            )
        return self.sample_count

    def observe_recalibration_sample(
        self,
        pose_desk: np.ndarray,
        *,
        side: str | None = None,
        source_timestamp: float,
    ) -> int:
        """Accept only fresh, continuously still tracker samples."""

        if not self.recapturing:
            return self.sample_count
        normalized = normalize_init_arm_side(side or self._active_side)
        samples = self._recalibration.samples_by_side.get(normalized)
        if samples is None:
            return self.sample_count
        stability = self._recalibration.stability_by_side.setdefault(
            normalized,
            HomeCaptureStabilityState(),
        )
        decision = stability.observe(
            pose_desk,
            source_timestamp=source_timestamp,
            first_accepted_pose=samples[0] if samples else None,
        )
        if decision.reset_samples:
            samples.clear()
        if decision.accepted:
            self.add_recalibration_sample(pose_desk, side=normalized)
        return self.sample_count

    def complete_recalibration(self, *, now: float, message: str = "") -> InitArmTransition:
        del now
        side = self._active_side
        operation = self._qpos_operation
        if operation is not None:
            operation.plan_future.cancel()
        self._qpos_operation = None
        self._mode = "live"
        self._active_side = ""
        self._recalibration = _RecalibrationState()
        self._arrival = _ArrivalState()
        self._status_message = message or f"{side} home recaptured; init-arm released"
        return InitArmTransition(ok=True, message=self._status_message, side=side, mode=self._mode)

    def cancel_active(self, *, now: float, reason: str = "") -> InitArmTransition:
        del now
        side = self._active_side
        was_active = bool(self.active)
        operation = self._qpos_operation
        if operation is not None:
            operation.plan_future.cancel()
        self._qpos_operation = None
        self._mode = "live"
        self._active_side = ""
        self._recalibration = _RecalibrationState()
        self._arrival = _ArrivalState()
        if was_active:
            suffix = f": {reason}" if reason else ""
            self._status_message = f"{side} init-arm stopped{suffix}"
        else:
            self._status_message = "init-arm ready"
        return InitArmTransition(
            ok=was_active,
            message=self._status_message,
            side=side,
            mode=self._mode,
        )

    def fail_recalibration(self, *, now: float, message: str) -> InitArmTransition:
        del now
        side = self._active_side
        if self._mode == "qpos_recapturing" and self._qpos_operation is not None:
            self._recalibration = _RecalibrationState()
            self._finish_qpos_operation(
                status="failed",
                message=str(message),
                override_terminal=True,
            )
            return InitArmTransition(
                ok=False,
                message=self._status_message,
                side=side,
                mode=self._mode,
            )
        self._mode = "moving" if side else "live"
        self._recalibration = _RecalibrationState()
        self._status_message = str(message)
        return InitArmTransition(ok=False, message=self._status_message, side=side, mode=self._mode)

    def active_target(self) -> InitArmResolvedTarget | None:
        if not self.active or self.qpos_active:
            return None
        return self.targets_by_side.get(self._active_side)

    def observe_actual_feedback(
        self,
        actual_tcp_pose_base: Any | None,
        *,
        timestamp: float,
        valid: bool,
        invalid_reason: str = "",
    ) -> None:
        """Update measured arrival and promote a stable moving arm to Init-fixed.

        Feedback cannot alter the target or start motion.  It may perform the
        one-way ``moving -> init_fixed`` semantic transition after position,
        rotation, speed, continuity, and dwell checks all pass.
        """

        target = self.active_target()
        if target is None:
            self._arrival = _ArrivalState()
            return
        if not valid:
            self._arrival = _ArrivalState(
                status="not_reached",
                reason=str(invalid_reason or "actual_feedback_unavailable"),
            )
            return
        try:
            actual = as_pose_matrix(actual_tcp_pose_base)
            current_time = float(timestamp)
        except (TypeError, ValueError):
            self._arrival = _ArrivalState(
                status="not_reached",
                reason="invalid_actual_feedback",
            )
            return
        if not np.all(np.isfinite(actual)) or not np.isfinite(current_time):
            self._arrival = _ArrivalState(
                status="not_reached",
                reason="invalid_actual_feedback",
            )
            return

        previous = self._arrival
        position_error_m, rotation_error_deg = pose_delta_metrics(
            actual,
            target.tcp_pose_base,
        )
        translation_speed_m_s: float | None = None
        rotation_speed_deg_s: float | None = None
        continuity_valid = False
        if previous.last_actual_pose_base is not None and previous.last_timestamp is not None:
            dt_sec = current_time - float(previous.last_timestamp)
            if 0.0 < dt_sec <= INIT_ARM_ARRIVAL_MAX_SAMPLE_GAP_SEC:
                moved_m, moved_deg = pose_delta_metrics(previous.last_actual_pose_base, actual)
                translation_speed_m_s = float(moved_m / dt_sec)
                rotation_speed_deg_s = float(moved_deg / dt_sec)
                continuity_valid = True

        within_target = bool(
            position_error_m <= INIT_ARM_ARRIVAL_POSITION_TOLERANCE_M
            and rotation_error_deg <= INIT_ARM_ARRIVAL_ROTATION_TOLERANCE_DEG
        )
        still = bool(
            continuity_valid
            and translation_speed_m_s is not None
            and rotation_speed_deg_s is not None
            and translation_speed_m_s <= INIT_ARM_ARRIVAL_STILL_TRANSLATION_SPEED_M_S
            and rotation_speed_deg_s <= INIT_ARM_ARRIVAL_STILL_ROTATION_SPEED_DEG_S
        )
        stable_since = previous.stable_since if within_target and still else None
        if within_target and still and stable_since is None:
            stable_since = current_time
        dwell_sec = 0.0 if stable_since is None else max(0.0, current_time - stable_since)
        if within_target and still and dwell_sec >= INIT_ARM_ARRIVAL_DWELL_SEC:
            status, reason = "stable", "target_reached_and_still"
            if self._mode == "moving":
                self._mode = "init_fixed"
                self._status_message = f"{self._active_side} reached Init pose and is fixed"
        elif not continuity_valid:
            status, reason = "not_reached", "waiting_for_continuous_actual_feedback"
        elif not within_target:
            status, reason = "not_reached", "target_error_above_tolerance"
        elif not still:
            status, reason = "not_reached", "actual_speed_above_stillness_threshold"
        else:
            status, reason = "not_reached", "stability_dwell_in_progress"
        self._arrival = _ArrivalState(
            last_actual_pose_base=actual.copy(),
            last_timestamp=current_time,
            stable_since=stable_since,
            status=status,
            reason=reason,
            position_error_m=float(position_error_m),
            rotation_error_deg=float(rotation_error_deg),
            translation_speed_m_s=translation_speed_m_s,
            rotation_speed_deg_s=rotation_speed_deg_s,
            dwell_sec=float(dwell_sec),
        )

    def filter_dual_arm_targets(
        self,
        targets: Mapping[str, np.ndarray],
        _observation,
        _calibration,
    ) -> dict[str, np.ndarray] | None:
        target = self.active_target()
        if target is None or target.solver_pose_desk is None:
            return None
        filtered = {
            side: as_pose_matrix(pose).copy()
            for side, pose in targets.items()
        }
        filtered[target.side] = target.solver_pose_desk.copy()
        return filtered

    def filter_command(self, command: TeleopCommand, _observation, _now: float) -> TeleopCommand:
        if self.qpos_active:
            return self._filter_qpos_command(command, now=float(_now))
        target = self.active_target()
        if target is None:
            return command
        effector_name = f"{target.side}_arm"
        filtered = TeleopCommand(timestamp=command.timestamp, metadata=dict(command.metadata))
        filtered.metadata["init_arm_active_side"] = target.side
        filtered.metadata["init_arm_mode"] = self._mode
        for name, effector_command in command.commands.items():
            next_command = effector_command.copy()
            if name == effector_name:
                next_command = self._filter_arm_command(next_command, target)
            filtered.commands[name] = next_command
        return filtered

    def filter_faults(
        self,
        faults: Sequence[FaultEvent],
        _channels,
        _now: float,
    ) -> list[FaultEvent]:
        if not self.active:
            return list(faults)
        active_channels = (
            {"left_wrist_pose", "right_wrist_pose"}
            if self.qpos_active
            else {f"{self._active_side}_wrist_pose"}
        )
        filtered: list[FaultEvent] = []
        for fault in faults:
            if fault.channel_name in active_channels and fault.code in INIT_ARM_FAULT_CODES:
                continue
            filtered.append(fault)
        return filtered

    def status_payload(self) -> dict[str, object]:
        active_side = self._active_side if self.active else ""
        arrival = self._arrival
        operation = self._qpos_operation
        qpos_step_count = (
            len(getattr(operation.plan, "steps", ()) or ())
            if operation is not None and operation.plan is not None
            else None
        )
        return {
            "init_arm_available": True,
            "init_arm_selected_side": self._selected_side,
            "init_arm_active_side": active_side,
            "init_arm_active": bool(active_side),
            "init_arm_recalibrating": self.recapturing,
            "init_arm_status": self._status_message,
            "init_arm_sample_count": self.sample_count if self.recapturing else None,
            "init_arm_min_samples": self.min_samples if self.recapturing else None,
            "init_arm_profile_path": next(iter(self.targets_by_side.values())).source_path,
            "init_arm_arrival_status": arrival.status if active_side else "inactive",
            "init_arm_arrival_label": (
                "到位稳定"
                if active_side and arrival.status == "stable"
                else "未到位"
                if active_side
                else ""
            ),
            "init_arm_arrival_reason": arrival.reason if active_side else "",
            "init_arm_target_position_error_m": arrival.position_error_m if active_side else None,
            "init_arm_target_rotation_error_deg": arrival.rotation_error_deg if active_side else None,
            "init_arm_stability_dwell_sec": arrival.dwell_sec if active_side else None,
            "init_arm_qpos_active": bool(self.qpos_active),
            "init_arm_qpos_operation_id": (
                "" if operation is None else str(operation.operation_id)
            ),
            "init_arm_qpos_phase": self._mode if self.qpos_active else "inactive",
            "init_arm_qpos_step_index": (
                int(operation.step_cursor) if operation is not None else None
            ),
            "init_arm_qpos_step_count": qpos_step_count,
            "init_arm_qpos_max_error_deg": (
                operation.max_error_deg if operation is not None else None
            ),
        }

    def qpos_final_collision_fault(
        self,
        command: TeleopCommand,
        *,
        now: float,
    ) -> FaultEvent | None:
        """Fail closed if the final rewritten qpos pair is not collision-safe."""

        if not self.qpos_active:
            return None
        commands = command.commands if isinstance(command.commands, Mapping) else {}
        left_qpos = _qpos_from_effector_command(commands.get("left_arm"))
        right_qpos = _qpos_from_effector_command(commands.get("right_arm"))
        if left_qpos is None or right_qpos is None:
            return self._qpos_collision_fault(
                now=now,
                reason="final qpos pair is incomplete",
                metadata={
                    "missing_sides": [
                        side
                        for side, value in (("left", left_qpos), ("right", right_qpos))
                        if value is None
                    ]
                },
            )
        checker = self._qpos_final_collision_checker
        if not callable(checker):
            return self._qpos_collision_fault(
                now=now,
                reason="final coupled collision checker is unavailable",
            )
        try:
            result = checker(left_qpos=left_qpos, right_qpos=right_qpos)
        except Exception as exc:  # noqa: BLE001 - fail-closed pre-dispatch guard.
            return self._qpos_collision_fault(
                now=now,
                reason=f"collision checker raised {type(exc).__name__}: {exc}",
            )
        if bool(getattr(result, "safe", False)):
            return None
        return self._qpos_collision_fault(
            now=now,
            reason=str(getattr(result, "reason", "collision_check_failed")),
            metadata={
                "collision_distance_m": getattr(result, "min_distance_m", None),
                "collision_pair_count": getattr(result, "collision_pair_count", None),
                "collision_min_distance_m": getattr(result, "required_min_distance_m", None),
                "left_qpos": left_qpos.copy(),
                "right_qpos": right_qpos.copy(),
            },
        )

    def _qpos_collision_fault(
        self,
        *,
        now: float,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> FaultEvent:
        self._finish_qpos_operation(
            status="failed",
            message=f"Init qpos final collision guard rejected command: {reason}",
        )
        return FaultEvent(
            code="init_arm_qpos_final_collision_check_failed",
            message=self._status_message,
            severity=FaultSeverity.ERROR,
            timestamp=float(now),
            component="init_arm_qpos_final_collision_guard",
            metadata=dict(metadata or {}),
        )

    def _filter_qpos_command(self, command: TeleopCommand, *, now: float) -> TeleopCommand:
        operation = self._qpos_operation
        if operation is None:
            return command
        qpos_by_side = self._advance_qpos_operation(operation, now=now)
        recapturing = self._mode == "qpos_recapturing"
        init_fixed = self._mode == "qpos_init_fixed"
        target = self.targets_by_side[operation.side]
        filtered = TeleopCommand(
            timestamp=command.timestamp,
            metadata={
                **dict(command.metadata),
                "init_arm_mode": self._mode,
                "init_arm_qpos_active": True,
                "init_arm_qpos_operation_id": operation.operation_id,
                "init_arm_qpos_step_index": int(operation.step_cursor),
                "init_arm_qpos_step_count": (
                    len(getattr(operation.plan, "steps", ()) or ())
                    if operation.plan is not None
                    else None
                ),
            },
            execution_work_item=command.execution_work_item,
        )
        if recapturing:
            filtered.metadata.update(
                {
                    "arm_mode_stopped_sides": [operation.side],
                    "arm_mode_stop_reason": "init_arm_home_recapture",
                    "init_arm_label_active": False,
                }
            )
        else:
            filtered.metadata.update(
                {
                    "init_arm_active_side": operation.side,
                    "init_arm_label_active": True,
                }
            )
            if not init_fixed:
                support_side = "right" if operation.side == "left" else "left"
                filtered.metadata.update(
                    {
                        "arm_mode_stopped_sides": [support_side],
                        "arm_mode_stop_reason": "init_arm_static_collision_support",
                    }
                )
        for name, effector_command in command.commands.items():
            next_command = effector_command.copy()
            rewrite_arm = name in {"left_arm", "right_arm"} and (
                not (recapturing or init_fixed)
                or name == f"{operation.side}_arm"
            )
            if rewrite_arm:
                side = "left" if name == "left_arm" else "right"
                if side not in qpos_by_side:
                    raise RuntimeError(f"Init qpos command is missing {side} qpos.")
                qpos = qpos_by_side[side].copy()
                metadata = dict(next_command.metadata)
                metadata.update(
                    {
                        "source": "gui_init_qpos_plan",
                        "action_source": "gui_init_qpos_plan",
                        "init_arm_qpos_active": True,
                        "init_arm_qpos_operation_id": operation.operation_id,
                        "init_arm_qpos_selected_side": operation.side,
                        "init_arm_qpos_phase": self._mode,
                        "init_arm_qpos_step_index": int(operation.step_cursor),
                        "solver_output_qpos": qpos.copy(),
                    }
                )
                if side == operation.side:
                    metadata.update(
                        {
                            "init_arm_target_pose_base": target.tcp_pose_base.copy(),
                            "init_arm_target_qpos": target.arm_qpos.copy(),
                        }
                    )
                if recapturing:
                    metadata.update(
                        {
                            "arm_mode_stopped": True,
                            "arm_mode_stop_reason": "init_arm_home_recapture",
                            "init_arm_label_active": False,
                        }
                    )
                elif not init_fixed and side != operation.side:
                    metadata.update(
                        {
                            "arm_mode_stopped": True,
                            "arm_mode_stop_reason": "init_arm_static_collision_support",
                        }
                    )
                next_command.command_type = "joint_vector"
                next_command.value = qpos
                next_command.frame_name = "arm_base"
                next_command.metadata = metadata
            filtered.commands[name] = next_command
        required_sides = (operation.side,) if recapturing or init_fixed else ("left", "right")
        missing_effectors = [
            f"{side}_arm" for side in required_sides if f"{side}_arm" not in filtered.commands
        ]
        if missing_effectors:
            raise RuntimeError(
                "Init qpos requires complete dual-arm commands; missing "
                + ", ".join(missing_effectors)
            )
        return filtered

    def _advance_qpos_operation(
        self,
        operation: _QposInitOperation,
        *,
        now: float,
    ) -> dict[str, np.ndarray]:
        if self._mode in {"qpos_recapturing", "qpos_init_fixed"}:
            return self._operation_hold_qpos(operation)
        if self._mode == "qpos_planning" and operation.plan_future.done():
            try:
                plan = operation.plan_future.result()
                if not bool(getattr(plan, "completed", False)):
                    summary = plan.summary() if hasattr(plan, "summary") else {}
                    raise RuntimeError(f"qpos plan is incomplete: {summary}")
                self._validate_static_support_plan(operation, plan)
                measured = self._read_qpos_feedback(operation)
                drift_deg = self._max_qpos_error_deg(measured, operation.start_qpos_by_side)
                if drift_deg > operation.start_drift_tolerance_deg:
                    raise RuntimeError(
                        "measured qpos moved while planning: "
                        f"max drift {drift_deg:.3f}deg exceeds "
                        f"{operation.start_drift_tolerance_deg:.3f}deg"
                    )
            except Exception as exc:  # noqa: BLE001 - convert planning failure into safe hold.
                self._finish_qpos_operation(
                    status="failed",
                    message=f"Init qpos planning failed before execution: {type(exc).__name__}: {exc}",
                )
                return self._operation_hold_qpos(operation)
            operation.plan = plan
            operation.last_feedback_at = float(now)
            operation.execution_started_at = float(now)
            self._mode = "qpos_executing" if getattr(plan, "steps", ()) else "qpos_settling"
            self._status_message = (
                f"{operation.side} Init qpos collision-safe plan ready; "
                f"executing {len(getattr(plan, 'steps', ()))} steps"
            )

        if (
            self._mode == "qpos_planning"
            and float(now) - operation.started_at > operation.planning_timeout_sec
        ):
            operation.plan_future.cancel()
            self._finish_qpos_operation(
                status="failed",
                message="Init qpos planning timeout; no trajectory was executed and Hold is required",
            )
            return self._operation_hold_qpos(operation)

        if self._mode == "qpos_planning":
            return self._operation_hold_qpos(operation)
        if self._mode in {"qpos_completed", "qpos_failed"}:
            return self._operation_hold_qpos(operation)

        plan = operation.plan
        if plan is None:
            self._finish_qpos_operation(status="failed", message="Init qpos plan disappeared.")
            return self._operation_hold_qpos(operation)
        steps = tuple(getattr(plan, "steps", ()) or ())
        if self._mode == "qpos_executing":
            timed_sample = self._timed_plan_sample(operation, steps=steps, now=now)
            if timed_sample is not None:
                emitted, completed_steps, final_due = timed_sample
                self._check_qpos_execution_feedback(
                    operation,
                    completed_steps=completed_steps,
                )
                if self._mode == "qpos_failed":
                    return self._operation_hold_qpos(operation)
                operation.last_emitted_qpos_by_side = emitted
                operation.step_cursor = int(completed_steps)
                if final_due:
                    operation.step_cursor = len(steps)
                    self._mode = "qpos_settling"
                    self._status_message = (
                        f"{operation.side} Init qpos final command sent; verifying measured arrival"
                    )
                else:
                    self._status_message = (
                        f"{operation.side} Init qpos executing "
                        f"{operation.step_cursor}/{len(steps)}"
                    )
                return {side: qpos.copy() for side, qpos in emitted.items()}

            if (
                operation.step_cursor > 0
                and operation.step_cursor % operation.tracking_check_every == 0
            ):
                self._check_qpos_execution_feedback(
                    operation,
                    completed_steps=operation.step_cursor,
                    force=True,
                )
                if self._mode == "qpos_failed":
                    return self._operation_hold_qpos(operation)
            step = steps[operation.step_cursor]
            emitted = _copy_qpos_by_side(step.qpos_by_side, label="qpos init plan step")
            operation.last_emitted_qpos_by_side = emitted
            operation.step_cursor += 1
            if operation.step_cursor >= len(steps):
                self._mode = "qpos_settling"
                self._status_message = (
                    f"{operation.side} Init qpos final command sent; verifying measured arrival"
                )
            else:
                self._status_message = (
                    f"{operation.side} Init qpos executing "
                    f"{operation.step_cursor}/{len(steps)}"
                )
            return {side: qpos.copy() for side, qpos in emitted.items()}

        final_qpos = _copy_qpos_by_side(
            getattr(plan, "final_qpos_by_side", operation.target_qpos_by_side),
            label="qpos init final target",
        )
        operation.last_emitted_qpos_by_side = final_qpos
        if float(now) - operation.last_feedback_at >= 0.1:
            operation.last_feedback_at = float(now)
            try:
                measured = self._read_qpos_feedback(operation)
                error_deg = self._max_qpos_error_deg(measured, final_qpos)
            except Exception as exc:  # noqa: BLE001 - bounded settle retry below.
                operation.stable_since = None
                self._status_message = (
                    f"Init qpos waiting for fresh final feedback: {type(exc).__name__}: {exc}"
                )
            else:
                operation.max_error_deg = error_deg
                if error_deg <= operation.arrival_tolerance_deg:
                    if operation.stable_since is None:
                        operation.stable_since = float(now)
                    if float(now) - operation.stable_since >= operation.arrival_dwell_sec:
                        self._finish_qpos_operation(
                            status="completed",
                            message=(
                                f"{operation.side} reached exact Init qpos "
                                f"(max error {error_deg:.3f}deg); target arm is now Init-fixed"
                            ),
                        )
                else:
                    operation.stable_since = None
                    self._status_message = (
                        f"{operation.side} Init qpos settling; max error={error_deg:.3f}deg"
                    )
        execution_allowance_sec = max(
            self._plan_duration_sec(plan) * 2.0,
            len(steps) * 0.1,
        )
        if (
            float(now) - operation.started_at
            > operation.planning_timeout_sec + execution_allowance_sec + operation.settle_timeout_sec
        ):
            self._finish_qpos_operation(
                status="failed",
                message="Init qpos measured-arrival timeout; entering Hold",
            )
        return {side: qpos.copy() for side, qpos in final_qpos.items()}

    def _check_qpos_execution_feedback(
        self,
        operation: _QposInitOperation,
        *,
        completed_steps: int,
        force: bool = False,
    ) -> None:
        if not force and (
            int(completed_steps) - int(operation.last_tracking_check_step_cursor)
            < int(operation.tracking_check_every)
        ):
            return
        try:
            measured = self._read_qpos_feedback(operation)
        except Exception as exc:  # noqa: BLE001 - convert feedback loss to Hold.
            self._finish_qpos_operation(
                status="failed",
                message=(
                    "Init qpos lost fresh hardware feedback during execution: "
                    f"{type(exc).__name__}: {exc}; entering Hold"
                ),
            )
            return
        expected = operation.last_emitted_qpos_by_side or operation.start_qpos_by_side
        error_deg = self._max_qpos_error_deg(measured, expected)
        operation.max_error_deg = error_deg
        support_side = "right" if operation.side == "left" else "left"
        support_error_deg = float(
            np.max(
                np.abs(
                    np.rad2deg(
                        measured[support_side]
                        - operation.start_qpos_by_side[support_side]
                    )
                )
            )
        )
        if support_error_deg > operation.start_drift_tolerance_deg:
            self._finish_qpos_operation(
                status="failed",
                message=(
                    f"Init qpos static support arm {support_side} drifted "
                    f"{support_error_deg:.3f}deg; entering Hold"
                ),
            )
            operation.last_emitted_qpos_by_side = measured
            return
        if error_deg > operation.max_tracking_error_deg:
            self._finish_qpos_operation(
                status="failed",
                message=(
                    f"Init qpos tracking error {error_deg:.3f}deg exceeds "
                    f"{operation.max_tracking_error_deg:.3f}deg; entering Hold"
                ),
            )
            operation.last_emitted_qpos_by_side = measured
            return
        operation.last_tracking_check_step_cursor = int(completed_steps)

    @staticmethod
    def _timed_plan_sample(
        operation: _QposInitOperation,
        *,
        steps: tuple[Any, ...],
        now: float,
    ) -> tuple[dict[str, np.ndarray], int, bool] | None:
        """Sample a time-parameterized plan independently of caller loop rate.

        Older compatibility fakes/plans without ``time_sec`` retain the
        historical one-step-per-call behavior. Production Init plans always
        carry explicit timestamps and use this path.
        """

        if not steps or operation.execution_started_at is None:
            return None
        raw_times = [getattr(step, "time_sec", None) for step in steps]
        try:
            times = np.asarray(raw_times, dtype=float)
        except (TypeError, ValueError):
            return None
        if (
            times.shape != (len(steps),)
            or not np.all(np.isfinite(times))
            or np.any(times <= 0.0)
            or np.any(np.diff(times) <= 0.0)
        ):
            return None

        elapsed = max(0.0, float(now) - float(operation.execution_started_at))
        completed_steps = int(np.searchsorted(times, elapsed, side="right"))
        if completed_steps >= len(steps):
            final = _copy_qpos_by_side(
                steps[-1].qpos_by_side,
                label="qpos init timed final step",
            )
            return final, len(steps), True

        upper_index = completed_steps
        upper_time = float(times[upper_index])
        if upper_index == 0:
            lower_time = 0.0
            lower = operation.start_qpos_by_side
        else:
            lower_time = float(times[upper_index - 1])
            lower = _copy_qpos_by_side(
                steps[upper_index - 1].qpos_by_side,
                label="qpos init timed lower step",
            )
        upper = _copy_qpos_by_side(
            steps[upper_index].qpos_by_side,
            label="qpos init timed upper step",
        )
        alpha = min(
            1.0,
            max(0.0, (elapsed - lower_time) / max(upper_time - lower_time, 1e-12)),
        )
        emitted = {
            side: lower[side] + (upper[side] - lower[side]) * alpha
            for side in ("left", "right")
        }
        return emitted, completed_steps, False

    @staticmethod
    def _plan_duration_sec(plan: Any) -> float:
        try:
            return max(0.0, float(plan.summary().get("duration_sec", 0.0)))
        except Exception:
            return 0.0

    @staticmethod
    def _validate_static_support_plan(
        operation: _QposInitOperation,
        plan: Any,
        *,
        tolerance_rad: float = 1e-10,
    ) -> None:
        support_side = "right" if operation.side == "left" else "left"
        expected = operation.start_qpos_by_side[support_side]
        candidates = [
            getattr(step, "qpos_by_side", {}).get(support_side)
            for step in tuple(getattr(plan, "steps", ()) or ())
        ]
        candidates.append(
            getattr(plan, "final_qpos_by_side", {}).get(support_side)
        )
        for index, candidate in enumerate(candidates):
            if candidate is None:
                raise RuntimeError(
                    f"qpos plan is missing static support side {support_side} at step {index}"
                )
            qpos = np.asarray(candidate, dtype=float).reshape(6)
            if float(np.max(np.abs(qpos - expected))) > float(tolerance_rad):
                raise RuntimeError(
                    f"qpos plan moves non-target support arm {support_side} at step {index}"
                )

    @staticmethod
    def _read_qpos_feedback(operation: _QposInitOperation) -> dict[str, np.ndarray]:
        return _copy_qpos_by_side(operation.feedback_provider(), label="qpos init feedback")

    @staticmethod
    def _max_qpos_error_deg(
        actual: Mapping[str, np.ndarray],
        expected: Mapping[str, np.ndarray],
    ) -> float:
        return max(
            float(np.max(np.abs(np.rad2deg(actual[side] - expected[side]))))
            for side in ("left", "right")
        )

    @staticmethod
    def _operation_hold_qpos(operation: _QposInitOperation) -> dict[str, np.ndarray]:
        source = operation.last_emitted_qpos_by_side or operation.start_qpos_by_side
        return {side: qpos.copy() for side, qpos in source.items()}

    def _finish_qpos_operation(
        self,
        *,
        status: str,
        message: str,
        override_terminal: bool = False,
    ) -> None:
        operation = self._qpos_operation
        if operation is None or (operation.terminal_status and not override_terminal):
            return
        normalized = "completed" if str(status) == "completed" else "failed"
        operation.terminal_status = normalized
        operation.terminal_message = str(message)
        if override_terminal:
            operation.terminal_consumed = False
        self._mode = "qpos_init_fixed" if normalized == "completed" else "qpos_failed"
        self._status_message = str(message)

    def _filter_arm_command(
        self,
        command: EffectorCommand,
        target: InitArmResolvedTarget,
    ) -> EffectorCommand:
        metadata = dict(command.metadata)
        metadata.update(
            {
                "init_arm_active": True,
                "init_arm_side": target.side,
                "init_arm_mode": self._mode,
                "init_arm_profile_path": target.source_path,
                "init_arm_target_pose_base": target.tcp_pose_base.copy(),
            }
        )
        if target.tcp_pose_desk is not None:
            metadata["init_arm_target_pose_desk"] = target.tcp_pose_desk.copy()
        if target.arm_qpos is not None:
            metadata["init_arm_target_qpos"] = target.arm_qpos.copy()

        if self._use_dual_target_override and target.solver_pose_desk is not None:
            command.metadata = metadata
            return command

        metadata.update(
            {
                "arm_controlled_frame": "tool_tcp",
                "arm_reference_frame": "arm_base",
                "target_pose_base": target.tcp_pose_base.copy(),
                "target_pose_tcp_base": target.tcp_pose_base.copy(),
            }
        )
        if target.tcp_pose_desk is not None:
            metadata["target_pose_desk"] = target.tcp_pose_desk.copy()
            metadata["target_pose_tcp_desk"] = target.tcp_pose_desk.copy()
        if target.arm_qpos is not None:
            metadata["solver_output_qpos"] = target.arm_qpos.copy()
        command.command_type = "cartesian_pose"
        command.value = target.tcp_pose_base.copy()
        command.frame_name = "arm_base"
        command.metadata = metadata
        return command
