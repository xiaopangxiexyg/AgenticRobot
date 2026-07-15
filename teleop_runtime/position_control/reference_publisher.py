from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from robot_motion.pose import pose_delta_metrics
from robot_motion.position_control import (
    POSITION_TRANSPORT_TARGET_SEQUENCE_KEY,
    POSITION_TRANSPORT_TARGET_SEQUENCES_KEY,
    PositionAuditSink,
    PositionReferenceFrame,
    PositionSideReference,
    ReferenceTimeline,
    position_reference_audit_event,
)

from ..schemas import EffectorCommand, TeleopCommand
from ..session import SessionStepResult


ForwardKinematics = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PositionReferencePublisherConfig:
    active_sides: tuple[str, ...] | Sequence[str]
    source: str = "teleop"
    max_timeline_frames: int = 512
    reference_max_age_sec: float = 0.2
    require_dispatched: bool = True

    def __post_init__(self) -> None:
        sides = tuple(str(side).strip() for side in self.active_sides)
        if not sides or any(not side for side in sides):
            raise ValueError("active_sides must contain at least one non-empty side.")
        if len(set(sides)) != len(sides):
            raise ValueError("active_sides must not contain duplicates.")
        if int(self.max_timeline_frames) <= 0:
            raise ValueError("max_timeline_frames must be positive.")
        max_age = float(self.reference_max_age_sec)
        if not math.isfinite(max_age) or max_age < 0.0:
            raise ValueError("reference_max_age_sec must be finite and >= 0.")
        object.__setattr__(self, "active_sides", sides)


class PositionReferencePublisher:
    """Turn dispatched runtime commands into an atomic causal q-ref timeline.

    The publisher belongs to ``teleop_runtime`` because extracting the accepted
    target and invoking a concrete backend's FK are hardware/runtime concerns.
    The resulting immutable frame and timeline live in ``robot_motion`` and are
    reusable by simulation, replay, and deployment.
    """

    def __init__(
        self,
        *,
        config: PositionReferencePublisherConfig,
        fk_by_side: Mapping[str, ForwardKinematics],
        urdf_root_from_base_by_side: Mapping[str, np.ndarray] | None = None,
        timeline: ReferenceTimeline | None = None,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        epoch: int | None = None,
        audit_observer: PositionAuditSink | None = None,
    ) -> None:
        self.config = config
        self._fk_by_side = dict(fk_by_side)
        missing_fk = set(config.active_sides) - set(self._fk_by_side)
        if missing_fk:
            raise ValueError(f"Missing FK providers for active sides: {sorted(missing_fk)}")
        self._urdf_root_from_base_by_side = _root_transforms_by_side(
            urdf_root_from_base_by_side,
            active_sides=config.active_sides,
        )
        if not callable(wall_clock) or not callable(monotonic_clock):
            raise TypeError("wall_clock and monotonic_clock must be callable.")
        self.timeline = (
            timeline
            if timeline is not None
            else ReferenceTimeline(max_frames=int(config.max_timeline_frames))
        )
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._epoch = self._initial_epoch() if epoch is None else self._validate_epoch(epoch)
        self._next_sequence = 0
        self._audit_observer = audit_observer
        self._audit_observer_error_count = 0
        self._last_audit_observer_error: str | None = None

    @staticmethod
    def _initial_epoch() -> int:
        return max(0, int(time.time_ns()))

    @staticmethod
    def _validate_epoch(value: int) -> int:
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError("epoch must be a non-negative integer.")
        return int(value)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    @property
    def audit_observer_error_count(self) -> int:
        return int(self._audit_observer_error_count)

    @property
    def last_audit_observer_error(self) -> str | None:
        return self._last_audit_observer_error

    def set_audit_observer(self, observer: PositionAuditSink | None) -> None:
        """Dynamically attach/detach a non-critical reference audit sink."""
        if observer is not None and not callable(observer):
            raise TypeError("Position reference audit observer must be callable or None.")
        self._audit_observer = observer

    def reset_epoch(self, epoch: int | None = None) -> int:
        """Start a new control epoch after startup sync or runtime re-calibration."""
        self._epoch = (
            self._epoch + 1
            if epoch is None
            else self._validate_epoch(epoch)
        )
        self._next_sequence = 0
        return self._epoch

    def latest_at_or_before(self, sample_timestamp: float) -> PositionReferenceFrame | None:
        return self.timeline.latest_at_or_before(
            float(sample_timestamp),
            epoch=self._epoch,
            max_age=float(self.config.reference_max_age_sec),
        )

    def reference_for_sample(self, sample_timestamp: float) -> Mapping[str, Any] | None:
        """Select causally by wall time and evaluate freshness on monotonic time."""
        frame = self.timeline.latest_at_or_before(
            float(sample_timestamp),
            epoch=self._epoch,
            max_age=None,
        )
        if frame is None:
            return None
        sample_monotonic = float(self._monotonic_clock())
        if not math.isfinite(sample_monotonic):
            raise RuntimeError("Position reference monotonic clock returned a non-finite value.")
        monotonic_age_sec = max(
            0.0,
            sample_monotonic - float(frame.published_monotonic_time),
        )
        if monotonic_age_sec > float(self.config.reference_max_age_sec):
            return None
        return {
            "epoch": frame.epoch,
            "sequence": frame.sequence,
            "source": frame.source,
            "pipeline_version": frame.pipeline_version,
            "source_wall_time": frame.source_wall_time,
            "generated_wall_time": frame.generated_wall_time,
            "published_wall_time": frame.published_wall_time,
            "generated_monotonic_time": frame.generated_monotonic_time,
            "published_monotonic_time": frame.published_monotonic_time,
            "references": frame.references,
            "valid": frame.valid,
            "hold": frame.hold,
            "selection_age_sec": monotonic_age_sec,
            "selection_age_clock": "monotonic",
        }

    def publish_step(
        self,
        result: SessionStepResult,
        *,
        command: TeleopCommand | None = None,
    ) -> PositionReferenceFrame | None:
        selected_command = result.command if command is None else command
        if selected_command is None:
            return None
        if self.config.require_dispatched and not bool(result.command_dispatched):
            return None
        generated_monotonic = float(self._monotonic_clock())
        generated_wall = float(self._wall_clock())
        references: dict[str, PositionSideReference] = {}
        for side in self.config.active_sides:
            effector = selected_command.commands.get(f"{side}_arm")
            if effector is None:
                return None
            reference = self._side_reference(side, effector)
            if reference is None:
                return None
            references[side] = reference

        published_wall = float(self._wall_clock())
        published_monotonic = float(self._monotonic_clock())
        if not all(
            math.isfinite(value)
            for value in (
                generated_wall,
                generated_monotonic,
                published_wall,
                published_monotonic,
            )
        ):
            raise RuntimeError("Position reference clocks returned non-finite values.")
        published_monotonic = max(generated_monotonic, published_monotonic)
        source_wall_time = _source_wall_time(result, selected_command)
        transport_target_sequences = _transport_target_sequences(
            selected_command,
            sides=self.config.active_sides,
        )
        frame = PositionReferenceFrame(
            epoch=self._epoch,
            sequence=self._next_sequence,
            source_wall_time=source_wall_time,
            generated_wall_time=generated_wall,
            published_wall_time=published_wall,
            generated_monotonic_time=generated_monotonic,
            published_monotonic_time=published_monotonic,
            references=references,
            source=str(self.config.source),
            frame_metadata={
                "session_state": result.state.value,
                "control_hz": float(result.control_hz),
                "command_timestamp": selected_command.timestamp,
                "command_dispatched": bool(result.command_dispatched),
                POSITION_TRANSPORT_TARGET_SEQUENCES_KEY: transport_target_sequences,
            },
        )
        self.timeline.publish(frame)
        self._stamp_command(selected_command, frame)
        self._notify_audit_observer(frame)
        self._next_sequence += 1
        return frame

    def _notify_audit_observer(self, frame: PositionReferenceFrame) -> None:
        observer = self._audit_observer
        if observer is None:
            return
        try:
            observer(position_reference_audit_event(frame))
        except Exception as exc:  # noqa: BLE001 - diagnostics must not break dispatch.
            self._audit_observer_error_count += 1
            self._last_audit_observer_error = f"{type(exc).__name__}: {exc}"

    def _side_reference(
        self,
        side: str,
        command: EffectorCommand,
    ) -> PositionSideReference | None:
        metadata = command.metadata if isinstance(command.metadata, Mapping) else {}
        q_ref = _final_qpos(command, metadata)
        if q_ref is None:
            return None
        q_ik_raw = _raw_ik_qpos(metadata, q_ref.shape)
        try:
            pose_ref = np.asarray(self._fk_by_side[side](q_ref.copy()), dtype=float)
        except Exception:
            return None
        if pose_ref.shape != (4, 4) or not np.all(np.isfinite(pose_ref)):
            return None
        reasons = _modification_reasons(metadata, q_ik_raw=q_ik_raw, q_ref=q_ref)
        frame_metadata: dict[str, object] = {
            "effector_name": command.effector_name,
            "command_type": command.command_type,
            "frame_name": command.frame_name,
            "action_pose_source": metadata.get("action_pose_source"),
            "backend_final_action_source": metadata.get("backend_final_action_source"),
            "source_target_pose_base": _pose_or_none(metadata.get("source_target_pose_base")),
            "target_activity": metadata.get("target_activity"),
            "collision_avoidance_status": metadata.get("collision_avoidance_status"),
            "collision_avoidance_fallback_reason": metadata.get(
                "collision_avoidance_fallback_reason"
            ),
            "ik_solve_pose_error": metadata.get("ik_solve_pose_error"),
            "execution_step_metadata": metadata.get("execution_step_metadata"),
            POSITION_TRANSPORT_TARGET_SEQUENCE_KEY: metadata.get(
                POSITION_TRANSPORT_TARGET_SEQUENCE_KEY
            ),
        }
        urdf_root_from_base = self._urdf_root_from_base_by_side.get(side)
        source_target_pose_base = frame_metadata["source_target_pose_base"]
        if urdf_root_from_base is not None:
            pose_ref_urdf_root = urdf_root_from_base @ pose_ref
            frame_metadata["q_ref_pose_urdf_root"] = pose_ref_urdf_root
            frame_metadata["urdf_root_frame"] = "desk"
            if source_target_pose_base is not None:
                source_target_pose_urdf_root = (
                    urdf_root_from_base @ source_target_pose_base
                )
                source_minus_q_ref_xyz = (
                    source_target_pose_urdf_root[:3, 3] - pose_ref_urdf_root[:3, 3]
                )
                translation_m, rotation_deg = pose_delta_metrics(
                    pose_ref_urdf_root,
                    source_target_pose_urdf_root,
                )
                frame_metadata.update(
                    {
                        "source_target_pose_urdf_root": source_target_pose_urdf_root,
                        "source_target_minus_q_ref_xyz_urdf_root_m": source_minus_q_ref_xyz,
                        "source_target_minus_q_ref_x_urdf_root_m": float(
                            source_minus_q_ref_xyz[0]
                        ),
                        "source_target_to_q_ref_translation_m": float(translation_m),
                        "source_target_to_q_ref_rotation_deg": float(rotation_deg),
                    }
                )
        return PositionSideReference(
            side=side,
            q_ik_raw=q_ik_raw,
            q_ref=q_ref,
            pose_ref=pose_ref,
            valid=True,
            hold=_is_hold(metadata),
            modification_reasons=reasons,
            frame_metadata=frame_metadata,
            profile_metadata={
                "ik_safety_profile": metadata.get("ik_safety_profile"),
                "joint_velocity_profile": metadata.get("joint_velocity_profile"),
            },
        )

    @staticmethod
    def _stamp_command(command: TeleopCommand, frame: PositionReferenceFrame) -> None:
        command.metadata.update(
            {
                "position_reference_epoch": int(frame.epoch),
                "position_reference_sequence": int(frame.sequence),
                "position_reference_published_wall_time": float(frame.published_wall_time),
                "position_reference_pipeline_version": str(frame.pipeline_version),
                POSITION_TRANSPORT_TARGET_SEQUENCES_KEY: dict(
                    frame.frame_metadata.get(POSITION_TRANSPORT_TARGET_SEQUENCES_KEY, {})
                ),
            }
        )
        for side, reference in frame.references.items():
            effector = command.commands.get(f"{side}_arm")
            if effector is None:
                continue
            effector.metadata.update(
                {
                    "position_reference_epoch": int(frame.epoch),
                    "position_reference_sequence": int(frame.sequence),
                    "position_reference_q_ref": np.asarray(reference.q_ref, dtype=float).copy(),
                    "position_reference_pose_ref": np.asarray(reference.pose_ref, dtype=float).copy(),
                }
            )


def _source_wall_time(result: SessionStepResult, command: TeleopCommand) -> float:
    candidates = (
        getattr(command, "timestamp", None),
        getattr(getattr(result, "observation", None), "timestamp", None),
        getattr(result, "timestamp", None),
    )
    for value in candidates:
        if value is None:
            continue
        timestamp = float(value)
        if math.isfinite(timestamp):
            return timestamp
    raise ValueError("Session step has no finite source wall timestamp.")


def _root_transforms_by_side(
    transforms: Mapping[str, np.ndarray] | None,
    *,
    active_sides: Sequence[str],
) -> dict[str, np.ndarray]:
    if transforms is None:
        return {}
    unknown_sides = set(transforms) - set(active_sides)
    if unknown_sides:
        raise ValueError(
            "urdf_root_from_base_by_side contains inactive sides: "
            f"{sorted(unknown_sides)}"
        )
    result: dict[str, np.ndarray] = {}
    for side, transform in transforms.items():
        pose = _pose_or_none(transform)
        if pose is None:
            raise ValueError(
                f"urdf_root_from_base_by_side[{side!r}] must be a finite 4x4 pose."
            )
        result[str(side)] = pose
    return result


def _pose_or_none(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        pose = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    return pose.copy()


def _transport_target_sequences(
    command: TeleopCommand,
    *,
    sides: Sequence[str],
) -> dict[str, int]:
    sequences: dict[str, int] = {}
    for side in sides:
        effector = command.commands.get(f"{side}_arm")
        if effector is None or not isinstance(effector.metadata, Mapping):
            continue
        value = effector.metadata.get(POSITION_TRANSPORT_TARGET_SEQUENCE_KEY)
        if value is None or isinstance(value, bool):
            continue
        try:
            sequence = int(value)
        except (TypeError, ValueError):
            continue
        if sequence < 0 or sequence != value:
            continue
        sequences[str(side)] = sequence
    return sequences


def _finite_qpos(value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        qpos = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if qpos.size == 0 or not np.all(np.isfinite(qpos)):
        return None
    if shape is not None and qpos.shape != shape:
        return None
    return qpos.copy()


def _final_qpos(command: EffectorCommand, metadata: Mapping[str, Any]) -> np.ndarray | None:
    for key in (
        "backend_final_target_qpos",
        "position_reference_q_ref",
        "solver_output_qpos",
    ):
        qpos = _finite_qpos(metadata.get(key))
        if qpos is not None:
            return qpos
    if command.command_type == "joint_vector":
        return _finite_qpos(command.value)
    return None


def _raw_ik_qpos(metadata: Mapping[str, Any], shape: tuple[int, ...]) -> np.ndarray | None:
    for key in (
        "q_ik_raw",
        "raw_ik_qpos",
        "ik_raw_qpos",
        "unlimited_solver_output_qpos",
    ):
        qpos = _finite_qpos(metadata.get(key), shape=shape)
        if qpos is not None:
            return qpos
    return None


def _modification_reasons(
    metadata: Mapping[str, Any],
    *,
    q_ik_raw: np.ndarray | None,
    q_ref: np.ndarray,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if q_ik_raw is not None and not np.allclose(q_ik_raw, q_ref, atol=1e-10, rtol=0.0):
        reasons.append("ik_output_modified")
    for key in (
        "ik_jump_gate_status",
        "singularity_gate_status",
        "collision_avoidance_status",
        "collision_avoidance_fallback_reason",
        "collision_status",
        "collision_fallback",
        "joint_rate_limiter_status",
    ):
        value = metadata.get(key)
        if value in (
            None,
            "",
            "pass",
            "normal",
            "tracking",
            "disabled",
            "not_checked",
        ):
            continue
        reasons.append(f"{key}:{value}")
    execution_step = metadata.get("execution_step_metadata")
    if isinstance(execution_step, Mapping) and bool(execution_step.get("step_limited")):
        reasons.append("execution_joint_step_limited")
    output_smoothing = metadata.get("output_qpos_smoothing")
    if isinstance(output_smoothing, Mapping) and bool(output_smoothing.get("applied")):
        reasons.append("output_qpos_smoothed")
    singularity = metadata.get("singularity_metadata")
    if isinstance(singularity, Mapping):
        value = singularity.get("singularity_gate_status")
        if value not in (None, "", "pass", "normal", "disabled", "not_checked"):
            reasons.append(f"singularity_gate_status:{value}")
    return tuple(dict.fromkeys(reasons))


def _is_hold(metadata: Mapping[str, Any]) -> bool:
    if bool(metadata.get("hold", False)):
        return True
    statuses = (
        metadata.get("ik_jump_gate_status"),
        metadata.get("singularity_gate_status"),
        metadata.get("collision_avoidance_status"),
        metadata.get("collision_avoidance_fallback_reason"),
        metadata.get("collision_status"),
        metadata.get("collision_fallback"),
    )
    singularity = metadata.get("singularity_metadata")
    if isinstance(singularity, Mapping):
        statuses = (*statuses, singularity.get("singularity_gate_status"))
    return any(
        any(token in str(status).lower() for token in ("hold", "freeze", "fallback"))
        for status in statuses
        if status not in (None, "")
    )
