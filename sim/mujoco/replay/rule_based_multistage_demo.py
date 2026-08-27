"""Headless/interactive MuJoCo proof for a composed rule-based collection.

This module is deliberately a simulation-only composition example.  It reuses
the canonical UMI Cartesian action loader, the shared timestamp resampler, the
existing desk-TCP position-control chain, the existing relative replay
boundary function, and the existing simulation dataset writer.  It does not
import ``teleop_runtime`` and it never opens a real hardware resource.

The legacy sequence is:

1. move the simulated arms to the first frame of stage A without recording;
2. record an absolute replay of stage A;
3. record one right-TCP fixed trajectory in the desk XY plane (a circle by
   default, or a catalogued one-stroke digit when ``--fixed-trajectory-id`` is
   supplied);
4. measure the post-fixed-trajectory TCP boundary;
5. rebase stage B on that measured boundary and record the relative replay.

Stage windows use half-open source-frame spans (``start:end``), so the
composition is editable without changing the runtime owners.  The output is a
normal ``SimReplayDatasetWriter`` dataset plus a composition manifest that
labels every recorded row by phase and records the relative-replay boundary.

When ``--detail-annotation-dir`` is supplied, the detail overlay is the
composition boundary:

1. move to ``right_write.pen_down``'s first frame without recording;
2. record absolute ``pen_down`` replay;
3. replace the annotated ``write`` interval with the existing rule-based
   circle or one-stroke trajectory at the selected source-derived speed;
4. measure the actual right TCP boundary and relative-replay ``pen_up``;
5. hold the first ``pen_down`` left pose and gripper target throughout all
   recorded phases.

The detail reader is deliberately read-only and validates the exact sealed
progress Episode, coarse annotation manifest, clock, and successful phase
sequence.  It does not import the collection application or recreate its
annotation lifecycle.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from dataset_interface.progress_detail_annotation_artifacts import (
    LoadedDetailStageAnnotationSet,
    load_detail_stage_annotation_set,
)
from dataset_interface.progress_artifacts import load_sealed_progress_episode
from dataset_interface.progress_stage_annotation_artifacts import (
    LoadedStageAnnotationSet,
    load_stage_annotation_set,
)
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM
from robot_motion.position_control import (
    ResampledMotionReference,
    TimestampedMotionTarget,
    resample_motion_targets,
)
from robot_motion.relative_action import pose_matrix_to_xyzquat
from robot_motion.replay.fixed_trajectory import (
    FIXED_TRAJECTORY_DIRECTIONS,
    build_fixed_trajectory_targets,
)
from robot_motion.replay.rule_based_relative import (
    RuleBasedBoundaryState,
    rebase_replay_action_at_boundary,
)
from robot_motion.replay.trajectory import ReplayTrajectoryGroup
from robot_motion.rm65 import (
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_POSITION_CONTROL_PROFILE_CHOICES,
)
from sim.mujoco.control.desk_tcp_position import (
    DeskTcpPositionControl,
    DeskTcpPositionControlConfig,
)
from sim.mujoco.replay.dataset import read_parquet_rows
from sim.mujoco.replay.dataset_writer import (
    SimReplayDatasetWriter,
    SimReplayDatasetWriterConfig,
)
from sim.mujoco.replay.trajectory import GripperReplaySample, QposSample
from sim.mujoco.replay.umi_dataset import (
    UmiReplayEpisode,
    UmiReplayFrame,
    load_umi_replay_episode,
)
from sim.mujoco.scene import MujocoCameraRenderer
from sim.mujoco.viewer import passive_viewer_handle

from .umi_runner import _advance_to, _control_config

DEFAULT_DATASET_ROOT = Path("remote_data/umi/2026-08-03_dmtac_rm65_bimanual_position_v2_writing_4_stage")
DEFAULT_STAGE_A = "0:120"
DEFAULT_STAGE_B = "250:400"
DEFAULT_CIRCLE_STEPS = 61
DEFAULT_CIRCLE_RADIUS_M = 0.025
DEFAULT_CIRCLE_HZ: float | None = None
DEFAULT_CIRCLE_SPEED_SIDE = "right"
DEFAULT_FIXED_TRAJECTORY_SCALE_M = 0.08
DEFAULT_FIXED_TRAJECTORY_DIRECTION = "up"
DEFAULT_REFERENCE_HZ = 60.0
DETAIL_PROGRESS_STEP_ID = "right_write"
DETAIL_PHASE_IDS = (
    "right_write.pen_down",
    "right_write.write",
    "right_write.pen_up",
)


@dataclass(frozen=True)
class StageSpan:
    """A half-open source frame range used by one composed replay phase."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if int(self.start) < 0 or int(self.end) <= int(self.start):
            raise ValueError(f"invalid half-open stage span: {self.start}:{self.end}")

    @property
    def frame_count(self) -> int:
        return int(self.end) - int(self.start)

    def to_dict(self) -> dict[str, int]:
        return {"start": int(self.start), "end": int(self.end), "frame_count": self.frame_count}


@dataclass(frozen=True)
class DetailPhaseSelection:
    """One immutable detail annotation mapped to source frame indices."""

    stage_id: str
    span: StageSpan
    start_at: float
    end_at: float
    annotation_id: str
    coarse_attempt_id: str
    outcome: str
    clock_domain_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": str(self.stage_id),
            "source_span": self.span.to_dict(),
            "start_at": float(self.start_at),
            "end_at": float(self.end_at),
            "annotation_id": str(self.annotation_id),
            "coarse_attempt_id": str(self.coarse_attempt_id),
            "outcome": str(self.outcome),
            "clock_domain_id": str(self.clock_domain_id),
        }


@dataclass(frozen=True)
class DetailWritingSelection:
    """The validated detail overlay and its three source-frame windows."""

    episode_uuid: str
    annotation_set_id: str
    base_annotation_set_id: str
    target_progress_step_id: str
    bound_clock_domain_id: str
    detail_task_stage_profile_hash: str
    coarse_attempt_id: str
    episode_capture_identity_hash: str
    source_progress_manifest_hash: str
    base_annotation_manifest_hash: str
    detail_annotation_bundle_hash: str
    detail_annotation_manifest_hash: str
    phases: Mapping[str, DetailPhaseSelection]

    def __post_init__(self) -> None:
        phases = dict(self.phases)
        if set(phases) != set(DETAIL_PHASE_IDS):
            raise ValueError(
                "detail writing selection must contain exactly the phases "
                f"{DETAIL_PHASE_IDS!r}"
            )
        object.__setattr__(self, "phases", phases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_uuid": str(self.episode_uuid),
            "annotation_set_id": str(self.annotation_set_id),
            "base_annotation_set_id": str(self.base_annotation_set_id),
            "target_progress_step_id": str(self.target_progress_step_id),
            "bound_clock_domain_id": str(self.bound_clock_domain_id),
            "detail_task_stage_profile_hash": str(
                self.detail_task_stage_profile_hash
            ),
            "coarse_attempt_id": str(self.coarse_attempt_id),
            "episode_capture_identity_hash": str(
                self.episode_capture_identity_hash
            ),
            "source_progress_manifest_hash": str(
                self.source_progress_manifest_hash
            ),
            "base_annotation_manifest_hash": str(
                self.base_annotation_manifest_hash
            ),
            "detail_annotation_bundle_hash": str(
                self.detail_annotation_bundle_hash
            ),
            "detail_annotation_manifest_hash": str(
                self.detail_annotation_manifest_hash
            ),
            "phases": {
                stage_id: self.phases[stage_id].to_dict()
                for stage_id in DETAIL_PHASE_IDS
            },
        }


def parse_stage_span(text: str, *, total_frames: int) -> StageSpan:
    """Parse and validate an editable ``start:end`` source-frame span."""

    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"stage span must be 'start:end', got {text!r}")
    span = StageSpan(start=int(parts[0]), end=int(parts[1]))
    if span.end > int(total_frames):
        raise ValueError(
            f"stage span {text!r} exceeds source frame count {int(total_frames)}"
        )
    return span


def _progress_episode_uuid(episode: UmiReplayEpisode) -> str:
    """Read the immutable progress-Episode reference from the source record."""

    record = episode.episode.record
    if not isinstance(record, Mapping):
        raise ValueError(
            "detail replay requires an episodes.jsonl record containing the "
            "progress_episode reference; an explicit parquet path is not enough"
        )
    candidates: list[object] = [record]
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        candidates.append(metadata)
    for container in candidates:
        reference = container.get("progress_episode")
        if not isinstance(reference, Mapping):
            continue
        value = str(reference.get("episode_uuid", "")).strip()
        if value and Path(value).name == value and value not in {".", ".."}:
            return value
    raise ValueError(
        "detail replay source record is missing metadata.progress_episode.episode_uuid"
    )


def _load_detail_base_annotation_set(
    dataset_root: Path,
    *,
    episode_uuid: str,
    base_annotation_set_id: str,
) -> LoadedStageAnnotationSet:
    """Load the exact coarse annotation set referenced by a detail overlay."""

    if Path(base_annotation_set_id).name != base_annotation_set_id:
        raise ValueError("detail binding contains an unsafe base annotation ID")
    annotation_dir = (
        Path(dataset_root).expanduser().resolve()
        / "meta"
        / "progress"
        / "stage_annotations"
        / str(episode_uuid)
        / str(base_annotation_set_id)
    )
    try:
        return load_stage_annotation_set(annotation_dir)
    except Exception as exc:
        raise ValueError(
            "cannot load the exact coarse annotation set referenced by the "
            f"detail overlay: {annotation_dir}: {exc}"
        ) from exc


def _load_detail_writing_selection(
    episode: UmiReplayEpisode,
    rows: Sequence[Mapping[str, Any]],
    annotation_dir: str | Path,
) -> DetailWritingSelection:
    """Resolve one immutable three-phase detail overlay to half-open rows.

    This adapter intentionally validates the source identity and annotation
    lifecycle seams that the simulation needs.  It does not import the
    collection application or regenerate a detail profile.  The profile hash
    carried by the immutable overlay remains provenance, including when an
    older dataset was annotated with a profile version no longer present in
    the current checkout.
    """

    detail: LoadedDetailStageAnnotationSet = load_detail_stage_annotation_set(
        annotation_dir
    )
    episode_uuid = _progress_episode_uuid(episode)
    sealed = load_sealed_progress_episode(
        Path(episode.episode.dataset_root)
        / "meta"
        / "progress"
        / "episodes"
        / episode_uuid
    )
    binding = detail.binding
    if binding.episode_capture_identity_hash != sealed.binding.episode_capture_identity_hash:
        raise ValueError("detail overlay and source progress Episode identity differ")
    if binding.source_progress_manifest_hash != sealed.manifest.semantic_content_hash:
        raise ValueError("detail overlay does not bind the source progress manifest")
    if binding.bound_clock_domain_id != sealed.binding.bound_clock_domain_id:
        raise ValueError("detail overlay and source Episode use different clocks")
    if binding.target_progress_step_id != DETAIL_PROGRESS_STEP_ID:
        raise ValueError(
            "detail replay currently requires target progress step "
            f"{DETAIL_PROGRESS_STEP_ID!r}, got {binding.target_progress_step_id!r}"
        )

    base = _load_detail_base_annotation_set(
        Path(episode.episode.dataset_root),
        episode_uuid=episode_uuid,
        base_annotation_set_id=binding.base_annotation_set_id,
    )
    if base.binding.annotation_set_id != binding.base_annotation_set_id:
        raise ValueError("detail binding resolved to a different base annotation ID")
    if base.binding.episode_capture_identity_hash != binding.episode_capture_identity_hash:
        raise ValueError("base annotation and detail overlay identity differ")
    if base.binding.source_progress_manifest_hash != binding.source_progress_manifest_hash:
        raise ValueError("base annotation and detail overlay manifest differ")
    if base.binding.task_stage_profile_hash != binding.base_task_stage_profile_hash:
        raise ValueError("detail binding and base annotation profile differ")
    if base.manifest.semantic_content_hash != binding.base_annotation_set_manifest_hash:
        raise ValueError("detail binding does not reference the exact base manifest")
    if base.binding.bound_clock_domain_id != binding.bound_clock_domain_id:
        raise ValueError("base annotation and detail overlay use different clocks")

    occurrences = tuple(
        occurrence
        for occurrence in detail.bundle.occurrences
        if occurrence.progress_step_id == DETAIL_PROGRESS_STEP_ID
    )
    if len(occurrences) != len(detail.bundle.occurrences):
        raise ValueError("detail overlay contains an occurrence outside right_write")
    by_attempt: dict[str, list[Any]] = {}
    for occurrence in occurrences:
        by_attempt.setdefault(occurrence.coarse_attempt_id, []).append(occurrence)
    if len(by_attempt) != 1:
        raise ValueError(
            "detail replay requires exactly one successful right_write attempt; "
            f"got {sorted(by_attempt)}"
        )
    coarse_attempt_id, attempt_occurrences = next(iter(by_attempt.items()))
    if any(occurrence.outcome != "SUCCESS" for occurrence in attempt_occurrences):
        raise ValueError("detail replay requires SUCCESS occurrences for all three phases")
    ordered = tuple(
        sorted(
            attempt_occurrences,
            key=lambda occurrence: (
                int(occurrence.occurrence_index_in_attempt),
                float(occurrence.start_at),
            ),
        )
    )
    if tuple(occurrence.stage_id for occurrence in ordered) != DETAIL_PHASE_IDS:
        raise ValueError(
            "detail overlay must contain ordered phases "
            f"{DETAIL_PHASE_IDS!r}"
        )
    if tuple(int(occurrence.occurrence_index_in_attempt) for occurrence in ordered) != (1, 2, 3):
        raise ValueError("detail phase occurrence indices must be 1, 2, 3")
    if any(
        occurrence.clock_domain_id != binding.bound_clock_domain_id
        for occurrence in ordered
    ):
        raise ValueError("detail phase clock does not match the bound Episode clock")
    if any(
        float(current.start_at) < float(previous.end_at)
        for previous, current in zip(ordered, ordered[1:])
    ):
        raise ValueError("detail phase intervals overlap")

    timestamps = np.asarray(
        [float(frame.timestamp) for frame in episode.frames],
        dtype=float,
    )
    if timestamps.size != len(rows):
        raise ValueError("detail annotation source rows do not match UMI frames")
    phases: dict[str, DetailPhaseSelection] = {}
    for occurrence in ordered:
        start_at = float(occurrence.start_at)
        end_at = float(occurrence.end_at)
        if start_at > float(timestamps[-1]) or end_at < float(timestamps[0]):
            raise ValueError(
                f"detail phase {occurrence.stage_id!r} is outside source timestamps"
            )
        start = int(np.searchsorted(timestamps, start_at, side="left"))
        end = int(np.searchsorted(timestamps, end_at, side="left"))
        end = min(end, int(timestamps.size))
        span = StageSpan(start=start, end=end)
        phases[occurrence.stage_id] = DetailPhaseSelection(
            stage_id=str(occurrence.stage_id),
            span=span,
            start_at=start_at,
            end_at=end_at,
            annotation_id=str(occurrence.annotation_id),
            coarse_attempt_id=str(occurrence.coarse_attempt_id),
            outcome=str(occurrence.outcome),
            clock_domain_id=str(occurrence.clock_domain_id),
        )
    for previous_stage, current_stage in zip(DETAIL_PHASE_IDS, DETAIL_PHASE_IDS[1:]):
        previous_span = phases[previous_stage].span
        current_span = phases[current_stage].span
        if previous_span.end > current_span.start:
            raise ValueError(
                f"mapped detail spans overlap: {previous_stage} and {current_stage}"
            )
    return DetailWritingSelection(
        episode_uuid=episode_uuid,
        annotation_set_id=str(binding.annotation_set_id),
        base_annotation_set_id=str(binding.base_annotation_set_id),
        target_progress_step_id=str(binding.target_progress_step_id),
        bound_clock_domain_id=str(binding.bound_clock_domain_id),
        detail_task_stage_profile_hash=str(binding.detail_task_stage_profile_hash),
        coarse_attempt_id=str(coarse_attempt_id),
        episode_capture_identity_hash=str(binding.episode_capture_identity_hash),
        source_progress_manifest_hash=str(binding.source_progress_manifest_hash),
        base_annotation_manifest_hash=str(binding.base_annotation_set_manifest_hash),
        detail_annotation_bundle_hash=str(detail.bundle.semantic_content_hash),
        detail_annotation_manifest_hash=str(detail.manifest.semantic_content_hash),
        phases=phases,
    )


def _trajectory_tcp_speed_summary(
    frame_spans: Sequence[Sequence[UmiReplayFrame]],
    *,
    side: str,
) -> dict[str, Any]:
    """Measure canonical-desk TCP path speed over selected source spans.

    Spans are measured independently so an editable gap between stage A and
    stage B is not mistaken for robot motion.  The result is a target-speed
    reference for rule-based actions, not a claim about hardware execution.
    """

    side_name = str(side).strip().lower()
    if side_name not in {"left", "right"}:
        raise ValueError(f"unsupported TCP speed side {side!r}")
    path_length_m = 0.0
    duration_sec = 0.0
    interval_count = 0
    for frames in frame_spans:
        if len(frames) < 2:
            continue
        positions = np.stack(
            [
                np.asarray(frame.pose_desk_by_side[side_name], dtype=float)[:3, 3]
                for frame in frames
            ]
        )
        timestamps = np.asarray([float(frame.timestamp) for frame in frames], dtype=float)
        delta_t = np.diff(timestamps)
        if not np.all(np.isfinite(delta_t)) or np.any(delta_t <= 0.0):
            raise ValueError(
                f"{side_name} TCP source timestamps must be finite and strictly increasing"
            )
        path_length_m += float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        duration_sec += float(delta_t.sum())
        interval_count += int(delta_t.size)
    if duration_sec <= 0.0 or interval_count <= 0:
        raise ValueError(
            f"cannot derive {side_name} TCP average speed from selected source spans"
        )
    return {
        "side": side_name,
        "pose_frame": "canonical_desk",
        "path_length_m": float(path_length_m),
        "duration_sec": float(duration_sec),
        "average_linear_speed_m_s": float(path_length_m / duration_sec),
        "interval_count": int(interval_count),
    }


def _require_left_hold_state(
    rows: Sequence[Mapping[str, Any]],
    frames: Sequence[UmiReplayFrame],
    span: StageSpan,
) -> tuple[dict[str, np.ndarray], float, int]:
    """Extract one immutable left-arm hold target for the composed run."""

    selected_rows = tuple(rows[span.start : span.end])
    selected_frames = tuple(frames[span.start : span.end])
    if len(selected_rows) != span.frame_count or len(selected_frames) != span.frame_count:
        raise ValueError("left hold span does not align with source rows and frames")
    labels = [str(row.get("teleop.arm_mode.left.label", "")).strip() for row in selected_rows]
    if any(label != "fixed_hold" for label in labels):
        unexpected = sorted({label for label in labels if label != "fixed_hold"})
        raise ValueError(
            "detail replay requires fixed_hold for the left arm over the entire "
            f"composed interval; got {unexpected or ['missing']}"
        )
    hold_pose = np.asarray(
        selected_frames[0].pose_desk_by_side["left"],
        dtype=float,
    ).reshape(4, 4).copy()
    hold_gripper = float(selected_frames[0].gripper_close_by_side["left"])
    gripper_values = np.asarray(
        [frame.gripper_close_by_side["left"] for frame in selected_frames],
        dtype=float,
    )
    if not np.all(np.isfinite(gripper_values)) or not np.allclose(
        gripper_values,
        hold_gripper,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "detail replay requires a constant left gripper target over the "
            "pen-down/write/pen-up interval"
        )
    return {"left": hold_pose}, hold_gripper, int(selected_frames[0].frame_index)


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _source_arm_action_specs(dataset_root: Path) -> dict[str, dict[str, Any]]:
    """Carry the source dataset's validated arm action specs into the export."""

    info_path = Path(dataset_root) / "meta" / "info.json"
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    raw_specs = payload.get("arm_action_specs_by_side")
    if not isinstance(raw_specs, Mapping):
        raise TypeError(f"Source dataset is missing arm_action_specs_by_side: {info_path}")
    specs: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        spec = raw_specs.get(side)
        if not isinstance(spec, Mapping):
            raise TypeError(f"Source dataset is missing {side} arm action spec: {info_path}")
        specs[side] = dict(spec)
    return specs


def _required_row_pose(row: Mapping[str, Any], column: str) -> np.ndarray:
    value = row.get(column)
    pose = np.asarray(value, dtype=float).reshape(-1)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"row {row.get('frame_index')} {column} must be finite xyzquat")
    if float(np.linalg.norm(pose[3:7])) <= 1e-9:
        raise ValueError(f"row {row.get('frame_index')} {column} has a zero quaternion")
    return pose.copy()


def _required_row_qpos(row: Mapping[str, Any], column: str) -> np.ndarray:
    value = row.get(column)
    qpos = np.asarray(value, dtype=float).reshape(-1)
    if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
        raise ValueError(f"row {row.get('frame_index')} {column} must be finite 6D qpos")
    return qpos.copy()


def _stage_group(
    episode: UmiReplayEpisode,
    rows: Sequence[Mapping[str, Any]],
    span: StageSpan,
    *,
    key: str,
    label: str,
) -> tuple[ReplayTrajectoryGroup, tuple[UmiReplayFrame, ...]]:
    """Build the shared immutable replay group needed by relative rebasing."""

    selected_frames = tuple(episode.frames[span.start : span.end])
    selected_rows = tuple(rows[span.start : span.end])
    if len(selected_frames) != span.frame_count or len(selected_rows) != span.frame_count:
        raise ValueError("stage span does not align with source episode rows")

    actions = np.zeros((span.frame_count, ACTION_DIM), dtype=np.float32)
    arm_joint_targets: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    actual_qpos: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    actual_tcp: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    hand_targets: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    timestamps: list[float] = []
    first_timestamp = float(selected_frames[0].timestamp)

    for local_index, (frame, row) in enumerate(zip(selected_frames, selected_rows)):
        for side in ("left", "right"):
            pose = pose_matrix_to_xyzquat(frame.pose_desk_by_side[side])
            actions[local_index, ACTION_ARM_POSE_SLICES[side]] = pose
            arm_joint_targets[side].append(
                _required_row_qpos(row, f"observation.arm.{side}.actual_qpos")
            )
            actual_qpos[side].append(arm_joint_targets[side][-1].copy())
            actual_tcp[side].append(
                _required_row_pose(row, f"observation.arm.{side}.actual_tcp_pose")
            )
            hand = np.zeros(6, dtype=np.float32)
            hand[0] = float(frame.gripper_close_by_side[side])
            hand_targets[side].append(hand)
            actions[local_index, 14 if side == "left" else 20] = hand[0]
        timestamps.append(float(frame.timestamp) - first_timestamp)

    metadata = {
        "action_metadata": {
            "pose_action_absolute_frame": "canonical_desk",
            "pose_frame_contract": episode.pose_frame_contract.to_dict(),
        },
        "source_metadata": {
            "source_kind": "remote_user_dataset_import",
            "source_episode_id": episode.episode.episode_id,
            "source_parquet_path": str(episode.episode.parquet_path),
            "source_span": span.to_dict(),
            "action_dim": int(episode.action_dim),
        },
    }
    group = ReplayTrajectoryGroup(
        key=str(key),
        label=str(label),
        source_path=str(episode.episode.parquet_path),
        source_episode_index=int(episode.episode.episode_index),
        start_index=int(span.start),
        end_index=int(span.end),
        frame_count=span.frame_count,
        action_hz=float(episode.nominal_hz),
        actions=actions,
        frame_times_sec=tuple(timestamps),
        arm_joint_targets={side: np.stack(values) for side, values in arm_joint_targets.items()},
        arm_observation_current_qpos={side: np.stack(values) for side, values in actual_qpos.items()},
        arm_observation_actual_qpos={side: np.stack(values) for side, values in actual_qpos.items()},
        arm_observation_actual_tcp_pose={side: np.stack(values) for side, values in actual_tcp.items()},
        hand_joint_targets={side: np.stack(values) for side, values in hand_targets.items()},
        metadata=metadata,
    )
    return group, selected_frames


def _stage_targets(
    frames: Sequence[UmiReplayFrame],
    *,
    phase: str,
    hold_pose_by_side: Mapping[str, Any] | None = None,
) -> list[TimestampedMotionTarget]:
    if not frames:
        raise ValueError(f"phase {phase!r} has no source frames")
    first_timestamp = float(frames[0].timestamp)
    hold = {
        str(side): np.asarray(pose, dtype=float).reshape(4, 4).copy()
        for side, pose in (hold_pose_by_side or {}).items()
    }
    targets: list[TimestampedMotionTarget] = []
    for index, frame in enumerate(frames):
        pose_by_side = {
            side: np.asarray(
                hold.get(side, frame.pose_desk_by_side[side]),
                dtype=float,
            ).reshape(4, 4).copy()
            for side in ("left", "right")
        }
        targets.append(
            TimestampedMotionTarget(
                timestamp=float(frame.timestamp) - first_timestamp,
                available_timestamp=float(frame.timestamp) - first_timestamp,
                source_index=index,
                pose_by_side=pose_by_side,
                metadata={
                    "phase": str(phase),
                    "source_frame_index": int(frame.frame_index),
                    "pose_frame": "desk",
                    **(
                        {"held_sides": sorted(hold)}
                        if hold
                        else {}
                    ),
                },
            )
        )
    return targets


def _grippers_for_reference(
    frames: Sequence[UmiReplayFrame],
    reference: ResampledMotionReference,
    *,
    hold_by_side: Mapping[str, float] | None = None,
) -> dict[str, float]:
    left = frames[int(reference.left_source_index)].gripper_close_by_side
    right = frames[int(reference.right_source_index)].gripper_close_by_side
    alpha = float(reference.interpolation_alpha)
    held = {str(side): float(value) for side, value in (hold_by_side or {}).items()}
    return {
        side: held[side]
        if side in held
        else float((1.0 - alpha) * left[side] + alpha * right[side])
        for side in ("left", "right")
    }


def _reference_from_pose(
    pose_by_side: Mapping[str, Any],
    *,
    timestamp: float,
    sequence: int,
    source_index: int,
    metadata: Mapping[str, Any],
) -> ResampledMotionReference:
    return ResampledMotionReference(
        sequence=int(sequence),
        timestamp=float(timestamp),
        left_source_index=int(source_index),
        right_source_index=int(source_index),
        left_source_timestamp=float(timestamp),
        right_source_timestamp=float(timestamp),
        interpolation_alpha=0.0,
        pose_by_side=pose_by_side,
        metadata=dict(metadata),
    )


def _circle_targets(
    boundary_pose_by_side: Mapping[str, np.ndarray],
    *,
    radius_m: float,
    steps: int,
    hz: float | None = None,
    linear_speed_m_s: float | None = None,
) -> list[TimestampedMotionTarget]:
    """Make a continuous one-cycle circle in the desk XY plane.

    The current right TCP is the first and last point on the circumference,
    which makes the rule-based phase continuous with the measured boundary.
    ``z`` and the complete TCP orientation remain constant.  Prefer
    ``linear_speed_m_s`` for a speed-defined action; ``hz`` remains an explicit
    fixed-waypoint-rate compatibility path.
    """

    radius = float(radius_m)
    count = int(steps)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("circle radius must be finite and positive")
    if count < 3:
        raise ValueError("circle steps must be >= 3")
    if hz is not None and linear_speed_m_s is not None:
        raise ValueError("circle action accepts either hz or linear_speed_m_s, not both")
    if hz is None and linear_speed_m_s is None:
        raise ValueError("circle action requires hz or linear_speed_m_s")
    if hz is not None and (not np.isfinite(float(hz)) or float(hz) <= 0.0):
        raise ValueError("circle hz must be finite and positive")
    if linear_speed_m_s is not None and (
        not np.isfinite(float(linear_speed_m_s)) or float(linear_speed_m_s) <= 0.0
    ):
        raise ValueError("circle linear speed must be finite and positive")
    right = np.asarray(boundary_pose_by_side["right"], dtype=float).reshape(4, 4)
    center = right[:3, 3].copy()
    center[0] -= radius
    segment_count = count - 1
    polygon_path_length_m = float(
        segment_count * 2.0 * radius * np.sin(np.pi / float(segment_count))
    )
    if linear_speed_m_s is not None:
        duration_sec = polygon_path_length_m / float(linear_speed_m_s)
        speed_source = "explicit_linear_speed_m_s"
    else:
        duration_sec = float(segment_count) / float(hz)
        speed_source = "explicit_circle_hz"
    target_speed_m_s = polygon_path_length_m / duration_sec
    targets: list[TimestampedMotionTarget] = []
    for index, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, count)):
        pose_by_side = {
            side: np.asarray(pose, dtype=float).reshape(4, 4).copy()
            for side, pose in boundary_pose_by_side.items()
        }
        position = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
        pose_by_side["right"][:3, 3] = position
        targets.append(
            TimestampedMotionTarget(
                timestamp=duration_sec * float(index) / float(segment_count),
                available_timestamp=duration_sec * float(index) / float(segment_count),
                source_index=index,
                pose_by_side=pose_by_side,
                metadata={
                    "phase": "rule_based_circle",
                    "rule": "tcp_circle",
                    "plane": "desk_xy",
                    "constant_axis": "desk_z",
                    "radius_m": radius,
                    "pose_frame": "canonical_desk",
                    "linear_speed_m_s": target_speed_m_s,
                    "path_length_m": polygon_path_length_m,
                    "duration_sec": duration_sec,
                    "speed_source": speed_source,
                },
            )
        )
    return targets


def _actual_tcp_boundary(control: DeskTcpPositionControl) -> dict[str, np.ndarray]:
    qpos = {
        side: control.scene.get_arm_qpos(side).copy() for side in ("left", "right")
    }
    world = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=qpos["left"], right_qpos=qpos["right"]
    )
    desk_from_world = np.linalg.inv(control.world_from_desk)
    return {
        side: desk_from_world @ np.asarray(world[side], dtype=float)
        for side in ("left", "right")
    }


def _require_pose_frame_alignment(
    episode: UmiReplayEpisode,
    control: DeskTcpPositionControl,
) -> dict[str, Any]:
    """Require source canonical desk poses and the sim station frame to match."""

    source_contract = episode.pose_frame_contract
    simulation_contract = control.station_pose_contract
    source_hash = source_contract.contract_hash
    if source_hash != str(episode.pose_frame_contract_hash).strip():
        raise ValueError(
            "source dataset pose_frame_contract_hash does not match its contract payload"
        )
    if source_hash != simulation_contract.contract_hash:
        raise ValueError(
            "source dataset canonical desk pose contract does not match the MuJoCo "
            "station pose contract; refusing implicit sim/real coordinate conversion"
        )
    if source_contract.pose_storage_frame != "desk_absolute":
        raise ValueError("source dataset must use desk_absolute storage for sim/real replay")
    if source_contract.canonical_absolute_frame != "desk":
        raise ValueError("source dataset canonical absolute frame must be 'desk'")
    return {
        "source_pose_frame_contract_hash": source_hash,
        "simulation_station_pose_frame_contract_hash": simulation_contract.contract_hash,
        "contract_match": True,
        "canonical_absolute_frame": "desk",
        "action_pose_frame": "canonical_desk",
        "world_from_desk": np.asarray(control.world_from_desk, dtype=float),
        "mapping_source": "calibrated_station_pose_contract_and_simulation_urdf",
    }


def _writer_gripper_sample(
    control: DeskTcpPositionControl,
    close_by_side: Mapping[str, float],
    *,
    frame_index: int,
    playback_time_sec: float,
    source_by_side: Mapping[str, str] | None = None,
) -> GripperReplaySample:
    close_by_name: dict[str, float] = {}
    source_by_name: dict[str, str] = {}
    resolved_source_by_side = {
        side: str(value)
        for side, value in (source_by_side or {}).items()
    }
    for name, spec in control.scene.gripper_specs.items():
        close_by_name[str(name)] = float(close_by_side[spec.side])
        source_by_name[str(name)] = resolved_source_by_side.get(
            spec.side,
            "composed_rule_based_target",
        )
    for side in ("left", "right"):
        resolved_source_by_side.setdefault(side, "composed_rule_based_target")
    return GripperReplaySample(
        frame_index=int(frame_index),
        timestamp=float(playback_time_sec),
        relative_time_sec=float(playback_time_sec),
        close_by_name=close_by_name,
        close_by_side={side: float(close_by_side[side]) for side in ("left", "right")},
        source_by_name=source_by_name,
        source_by_side=resolved_source_by_side,
    )


def _capture_frame(
    *,
    control: DeskTcpPositionControl,
    renderer: MujocoCameraRenderer,
    writer: SimReplayDatasetWriter,
    camera_names: Sequence[str],
    depth_camera_names: Sequence[str],
    frame_index: int,
    playback_time_sec: float,
    reference: ResampledMotionReference,
    grippers: Mapping[str, float],
    phase: str,
    action_source: str,
    source_frame_indices: Sequence[int],
    action_source_by_side: Mapping[str, str] | None = None,
    source_frame_indices_by_side: Mapping[str, Sequence[int]] | None = None,
) -> None:
    images = renderer.render_all(camera_names)
    depth_images = (
        renderer.render_all_depth(depth_camera_names) if depth_camera_names else {}
    )
    qpos = control.scene.arm_qpos_by_side()
    resolved_action_source_by_side = {
        side: str(value)
        for side, value in (action_source_by_side or {}).items()
    }
    for side in ("left", "right"):
        resolved_action_source_by_side.setdefault(side, str(action_source))
    resolved_source_frame_indices_by_side = {
        side: [int(value) for value in values]
        for side, values in (source_frame_indices_by_side or {}).items()
    }
    if not resolved_source_frame_indices_by_side:
        resolved_source_frame_indices_by_side = {
            side: [int(value) for value in source_frame_indices]
            for side in ("left", "right")
        }
    sample = QposSample(
        frame_index=int(frame_index),
        timestamp=float(playback_time_sec),
        relative_time_sec=float(playback_time_sec),
        qpos_by_side={side: np.asarray(values, dtype=float) for side, values in qpos.items()},
        source_by_side=resolved_action_source_by_side,
    )
    writer.record_frame(
        sample=sample,
        qpos_result=qpos,
        gripper_sample=_writer_gripper_sample(
            control,
            grippers,
            frame_index=int(frame_index),
            playback_time_sec=float(playback_time_sec),
            source_by_side=resolved_action_source_by_side,
        ),
        gripper_result=grippers,
        images=images,
        depth_images=depth_images,
        playback_time_sec=float(playback_time_sec),
        active_sides=("left", "right"),
        action_pose_by_side=reference.pose_by_side,
        frame_metadata={
            "phase": str(phase),
            "action_source": str(action_source),
            "action_source_by_side": resolved_action_source_by_side,
            "source_frame_indices": [int(value) for value in source_frame_indices],
            "source_frame_indices_by_side": resolved_source_frame_indices_by_side,
            "target_pose_desk_by_side": {
                side: pose_matrix_to_xyzquat(reference.pose_by_side[side])
                for side in ("left", "right")
            },
        },
    )


def _control_config_for_args(args: argparse.Namespace) -> DeskTcpPositionControlConfig:
    base = _control_config(args)
    if args.station_pose_contract is not None:
        return DeskTcpPositionControlConfig(
            **{
                **base.__dict__,
                "station_pose_contract_path": args.station_pose_contract,
            }
        )
    return base


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MuJoCo-only rule-based composition demo: initial alignment, absolute "
            "replay, desk-plane fixed trajectory, measured-boundary relative replay, "
            "and dataset/video capture. Use --detail-annotation-dir for the "
            "pen-down -> rule-based write -> pen-up composition."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode", default="0")
    parser.add_argument(
        "--detail-annotation-dir",
        type=Path,
        default=None,
        help=(
            "Immutable detail-stage annotation directory. When supplied, the "
            "source spans are derived from right_write.pen_down/write/pen_up "
            "timestamps; write is replaced by the selected rule-based trajectory, "
            "and pen_up is rebased from its measured endpoint."
        ),
    )
    parser.add_argument("--stage-a", default=DEFAULT_STAGE_A, help="absolute replay span start:end")
    parser.add_argument("--stage-b", default=DEFAULT_STAGE_B, help="relative replay span start:end")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--reference-hz", type=float, default=DEFAULT_REFERENCE_HZ)
    parser.add_argument("--circle-radius-m", type=float, default=DEFAULT_CIRCLE_RADIUS_M)
    parser.add_argument("--circle-steps", type=int, default=DEFAULT_CIRCLE_STEPS)
    parser.add_argument(
        "--circle-speed-m-s",
        type=float,
        default=None,
        help=(
            "Explicit right-TCP linear speed in canonical desk meters per second. "
            "Default: average speed of the selected source replay spans."
        ),
    )
    parser.add_argument(
        "--circle-hz",
        type=float,
        default=DEFAULT_CIRCLE_HZ,
        help=(
            "Explicit fixed circle waypoint rate compatibility override. "
            "Prefer --circle-speed-m-s for real/sim speed matching."
        ),
    )
    parser.add_argument(
        "--fixed-trajectory-id",
        default=None,
        help=(
            "Optional catalogued fixed trajectory, for example digit_1_v1. "
            "When set, it replaces the default circle with one continuous "
            "canonical-desk TCP path."
        ),
    )
    parser.add_argument(
        "--fixed-trajectory-size-m",
        "--fixed-trajectory-scale-m",
        dest="fixed_trajectory_scale_m",
        type=float,
        default=DEFAULT_FIXED_TRAJECTORY_SCALE_M,
        help=(
            "Authored glyph height in canonical-desk meters. The scale alias "
            "is retained for compatibility."
        ),
    )
    parser.add_argument(
        "--fixed-trajectory-direction",
        choices=FIXED_TRAJECTORY_DIRECTIONS,
        default=DEFAULT_FIXED_TRAJECTORY_DIRECTION,
        help=(
            "Direction of the glyph-local +Y axis in canonical_desk: "
            "up, right, down, or left."
        ),
    )
    parser.add_argument(
        "--fixed-trajectory-speed-m-s",
        type=float,
        default=None,
        help=(
            "Explicit catalogued-trajectory speed. Default: average right-TCP "
            "speed of the selected source replay spans."
        ),
    )
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=240)
    parser.add_argument("--record-camera", action="append", dest="record_cameras", default=None)
    parser.add_argument("--record-depth", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument(
        "--station-pose-contract",
        type=Path,
        default=None,
        help=(
            "Optional calibrated station pose contract. It must hash-match the "
            "source dataset contract; no implicit coordinate conversion is applied."
        ),
    )
    parser.add_argument("--left-tool-profile", type=Path, default=None)
    parser.add_argument("--right-tool-profile", type=Path, default=None)
    parser.add_argument(
        "--position-control-profile",
        choices=RM65_POSITION_CONTROL_PROFILE_CHOICES,
        default=RM65_BIMANUAL_POSITION_V1_NAME,
    )
    args = parser.parse_args(argv)
    if args.reference_hz <= 0.0:
        parser.error("--reference-hz must be positive")
    if args.circle_radius_m <= 0.0 or args.circle_steps < 3:
        parser.error("circle radius must be positive and circle-steps >= 3")
    if args.circle_hz is not None and args.circle_hz <= 0.0:
        parser.error("--circle-hz must be positive when explicitly set")
    if args.circle_speed_m_s is not None and args.circle_speed_m_s <= 0.0:
        parser.error("--circle-speed-m-s must be positive when explicitly set")
    if args.circle_hz is not None and args.circle_speed_m_s is not None:
        parser.error("use either --circle-hz or --circle-speed-m-s, not both")
    if args.fixed_trajectory_scale_m <= 0.0:
        parser.error("--fixed-trajectory-size-m must be positive")
    if args.fixed_trajectory_speed_m_s is not None and args.fixed_trajectory_speed_m_s <= 0.0:
        parser.error("--fixed-trajectory-speed-m-s must be positive when explicitly set")
    if args.fixed_trajectory_id is not None and (
        args.circle_hz is not None or args.circle_speed_m_s is not None
    ):
        parser.error(
            "circle speed options cannot be combined with --fixed-trajectory-id; "
            "use --fixed-trajectory-speed-m-s"
        )
    if args.render_width <= 0 or args.render_height <= 0:
        parser.error("render dimensions must be positive")
    if args.settle_seconds < 0.0:
        parser.error("--settle-seconds must be >= 0")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    episode = load_umi_replay_episode(args.dataset_root, args.episode)
    rows = read_parquet_rows(episode.episode.parquet_path)
    if len(rows) != len(episode.frames):
        raise ValueError("canonical UMI frame count does not match source parquet rows")
    detail_selection: DetailWritingSelection | None = None
    if args.detail_annotation_dir is not None:
        detail_selection = _load_detail_writing_selection(
            episode,
            rows,
            args.detail_annotation_dir,
        )
        stage_a_span = detail_selection.phases["right_write.pen_down"].span
        stage_b_span = detail_selection.phases["right_write.pen_up"].span
        detail_hold_span = StageSpan(
            start=stage_a_span.start,
            end=stage_b_span.end,
        )
        left_hold_pose_by_side, left_hold_gripper, left_hold_source_index = (
            _require_left_hold_state(rows, episode.frames, detail_hold_span)
        )
    else:
        stage_a_span = parse_stage_span(args.stage_a, total_frames=len(episode.frames))
        stage_b_span = parse_stage_span(args.stage_b, total_frames=len(episode.frames))
        left_hold_pose_by_side = {}
        left_hold_gripper = None
        left_hold_source_index = None
    _stage_a_group, stage_a_frames = _stage_group(
        episode,
        rows,
        stage_a_span,
        key="a",
        label=(
            "right_write.pen_down"
            if detail_selection is not None
            else "absolute_stage_a"
        ),
    )
    stage_b_group, stage_b_frames = _stage_group(
        episode,
        rows,
        stage_b_span,
        key="b",
        label=(
            "right_write.pen_up"
            if detail_selection is not None
            else "relative_stage_b"
        ),
    )
    speed_reference_by_side = {
        side: _trajectory_tcp_speed_summary(
            (stage_a_frames, stage_b_frames),
            side=side,
        )
        for side in ("left", "right")
    }
    stage_a_refs = resample_motion_targets(
        _stage_targets(
            stage_a_frames,
            phase=(
                "right_write.pen_down"
                if detail_selection is not None
                else "absolute_stage_a"
            ),
            hold_pose_by_side=left_hold_pose_by_side or None,
        ),
        reference_hz=float(args.reference_hz),
        include_final=True,
    )

    control = DeskTcpPositionControl.build(_control_config_for_args(args))
    try:
        pose_frame_alignment = _require_pose_frame_alignment(episode, control)
    except BaseException:
        control.close()
        raise
    renderer: MujocoCameraRenderer | None = None
    writer: SimReplayDatasetWriter | None = None
    control_snapshot: dict[str, Any] | None = None
    capture_frame_index = 0
    playback_time_sec = 0.0
    phase_counts: dict[str, int] = {}
    started = time.monotonic()
    boundary_before_fixed: dict[str, np.ndarray] | None = None
    boundary_after_fixed: dict[str, np.ndarray] | None = None
    fixed_speed_plan: dict[str, Any] | None = None
    fixed_phase = "rule_based_circle"
    fixed_action_source = "rule_based_tcp_circle"
    left_actual_qpos_samples: list[np.ndarray] = []
    viewer_context = (
        passive_viewer_handle(control.scene) if args.viewer else nullcontext(None)
    )
    viewer_handle = viewer_context.__enter__()

    requested_cameras = tuple(args.record_cameras or control.scene.camera_names)
    unknown_cameras = sorted(set(requested_cameras) - set(control.scene.camera_names))
    if unknown_cameras:
        control.close()
        raise ValueError(f"unknown MuJoCo record camera(s): {unknown_cameras}")
    depth_cameras = tuple(
        camera for camera in ("left_wrist", "right_wrist") if camera in requested_cameras
    ) if args.record_depth else ()

    def run_phase(
        phase: str,
        references: Sequence[ResampledMotionReference],
        gripper_fn: Callable[[ResampledMotionReference], dict[str, float]],
        *,
        action_source: str,
        source_index_fn: Callable[[ResampledMotionReference], Sequence[int]],
        action_source_by_side_fn: Callable[
            [ResampledMotionReference], Mapping[str, str]
        ]
        | None = None,
        source_frame_indices_by_side_fn: Callable[
            [ResampledMotionReference], Mapping[str, Sequence[int]]
        ]
        | None = None,
    ) -> None:
        nonlocal capture_frame_index, playback_time_sec
        phase_start = float(control.scene.data.time)
        completed = 0
        for reference in references:
            local_timestamp = float(reference.timestamp)
            target_time = phase_start + local_timestamp
            if not _advance_to(
                control,
                target_time,
                viewer_handle=viewer_handle,
                no_sleep=bool(args.no_sleep),
            ):
                raise RuntimeError(f"MuJoCo viewer closed during phase {phase!r}")
            runtime_reference = replace(
                reference,
                sequence=int(capture_frame_index),
                timestamp=float(target_time),
                metadata={**dict(reference.metadata), "phase": str(phase)},
            )
            grippers = gripper_fn(reference)
            control.solve_and_dispatch(runtime_reference, gripper_close_by_side=grippers)
            control.step_physics()
            control.rig.sync_viewer(viewer_handle)
            capture_time = playback_time_sec + max(0.0, local_timestamp)
            source_frame_indices = tuple(source_index_fn(reference))
            _capture_frame(
                control=control,
                renderer=renderer,
                writer=writer,
                camera_names=requested_cameras,
                depth_camera_names=depth_cameras,
                frame_index=capture_frame_index,
                playback_time_sec=capture_time,
                reference=runtime_reference,
                grippers=grippers,
                phase=phase,
                action_source=action_source,
                source_frame_indices=source_frame_indices,
                action_source_by_side=(
                    None
                    if action_source_by_side_fn is None
                    else action_source_by_side_fn(reference)
                ),
                source_frame_indices_by_side=(
                    None
                    if source_frame_indices_by_side_fn is None
                    else source_frame_indices_by_side_fn(reference)
                ),
            )
            left_actual_qpos_samples.append(
                control.scene.get_arm_qpos("left").copy()
            )
            capture_frame_index += 1
            completed += 1
        playback_time_sec += (
            0.0
            if not references
            else max(0.0, float(references[-1].timestamp))
            + 1.0 / float(args.reference_hz)
        )
        phase_counts[phase] = completed

    try:
        try:
            renderer = MujocoCameraRenderer(
                control.scene, width=int(args.render_width), height=int(args.render_height)
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to create MuJoCo renderer; set MUJOCO_GL=egl or MUJOCO_GL=osmesa "
                "in a headless session."
            ) from exc
        writer = SimReplayDatasetWriter(
            control.scene,
            SimReplayDatasetWriterConfig(
                root_dir=args.output,
                dataset_name=args.output.name,
                task_id=(
                    "rule_based_detail_phase_mujoco_demo"
                    if detail_selection is not None
                    else "rule_based_multistage_mujoco_demo"
                ),
                task_description=(
                    "MuJoCo rule-based detail pen-down replay + fixed writing + "
                    "relative pen-up replay"
                    if detail_selection is not None
                    else "MuJoCo rule-based absolute replay + fixed trajectory + relative replay"
                ),
                nominal_fps=float(args.reference_hz),
                source_dataset_root=episode.episode.dataset_root,
                source_episode_id=episode.episode.episode_id,
                source_parquet_path=episode.episode.parquet_path,
                urdf_path=control.scene.conversion.urdf_path,
                mode="dual",
                qpos_source="mujoco_actual",
                pose_frame_contract=episode.pose_frame_contract.to_dict(),
                pose_frame_contract_hash=episode.pose_frame_contract_hash,
                policy_io_layout_hash=episode.policy_io_layout_hash,
                arm_pose_action_mode="absolute",
                arm_action_specs_by_side=_source_arm_action_specs(
                    episode.episode.dataset_root
                ),
                # The canonical 26D compatibility layout stores each scalar
                # gripper target in the first slot of its fixed q6 block.
                gripper_command_dim=0,
                write_wrist_depth_raw=bool(args.record_depth),
                wrist_depth_camera_names=depth_cameras,
            ),
            camera_names=requested_cameras,
        )
        initial_pose = {
            side: np.asarray(
                left_hold_pose_by_side.get(
                    side,
                    stage_a_frames[0].pose_desk_by_side[side],
                ),
                dtype=float,
            ).reshape(4, 4).copy()
            for side in ("left", "right")
        }
        initial_reference = _reference_from_pose(
            initial_pose,
            timestamp=float(control.scene.data.time),
            sequence=0,
            source_index=0,
            metadata={"phase": "initial_alignment", "capture": False},
        )
        initial_grippers = dict(stage_a_frames[0].gripper_close_by_side)
        if left_hold_gripper is not None:
            initial_grippers["left"] = float(left_hold_gripper)
        control.solve_and_dispatch(initial_reference, gripper_close_by_side=initial_grippers)
        control.step_physics()
        control.rig.sync_viewer(viewer_handle)
        _advance_to(
            control,
            float(control.scene.data.time) + float(args.settle_seconds),
            viewer_handle=viewer_handle,
            no_sleep=bool(args.no_sleep),
        )

        run_phase(
            "right_write.pen_down" if detail_selection is not None else "absolute_stage_a",
            stage_a_refs,
            lambda reference: _grippers_for_reference(
                stage_a_frames,
                reference,
                hold_by_side=(
                    None
                    if left_hold_gripper is None
                    else {"left": float(left_hold_gripper)}
                ),
            ),
            action_source="dataset_absolute_replay",
            source_index_fn=lambda reference: (
                int(stage_a_frames[reference.left_source_index].frame_index),
                int(stage_a_frames[reference.right_source_index].frame_index),
            ),
            action_source_by_side_fn=(
                None
                if detail_selection is None
                else lambda _reference: {
                    "left": "dataset_fixed_hold",
                    "right": "dataset_absolute_replay",
                }
            ),
            source_frame_indices_by_side_fn=(
                None
                if detail_selection is None
                else lambda reference: {
                    "left": [int(left_hold_source_index)],
                    "right": [
                        int(stage_a_frames[reference.right_source_index].frame_index)
                    ],
                }
            ),
        )

        boundary_before_fixed = _actual_tcp_boundary(control)
        fixed_target_boundary = {
            side: np.asarray(pose, dtype=float).reshape(4, 4).copy()
            for side, pose in boundary_before_fixed.items()
        }
        if detail_selection is not None:
            # The measured left TCP is diagnostic only.  The fixed writing
            # rule receives the dataset hold target so it cannot turn the
            # left arm into an accidental second writing source.
            fixed_target_boundary["left"] = left_hold_pose_by_side["left"].copy()
        source_speed = speed_reference_by_side[DEFAULT_CIRCLE_SPEED_SIDE]
        if args.fixed_trajectory_id is not None:
            fixed_phase = (
                "right_write.write"
                if detail_selection is not None
                else "rule_based_fixed_trajectory"
            )
            fixed_action_source = "rule_based_fixed_trajectory"
            fixed_speed = float(
                args.fixed_trajectory_speed_m_s
                if args.fixed_trajectory_speed_m_s is not None
                else source_speed["average_linear_speed_m_s"]
            )
            fixed_targets = build_fixed_trajectory_targets(
                fixed_target_boundary,
                args.fixed_trajectory_id,
                scale_m=float(args.fixed_trajectory_scale_m),
                linear_speed_m_s=fixed_speed,
                direction=args.fixed_trajectory_direction,
            )
            fixed_target_metadata = dict(fixed_targets[0].metadata)
            fixed_speed_plan = {
                "trajectory_id": str(args.fixed_trajectory_id),
                "reference_side": DEFAULT_CIRCLE_SPEED_SIDE,
                "pose_frame": "canonical_desk",
                "source_reference": source_speed,
                "target_linear_speed_m_s": fixed_target_metadata["linear_speed_m_s"],
                "target_path_length_m": fixed_target_metadata["path_length_m"],
                "target_duration_sec": fixed_target_metadata["duration_sec"],
                "size_m": fixed_target_metadata["size_m"],
                "scale_m": fixed_target_metadata["scale_m"],
                "direction": fixed_target_metadata["direction"],
                "direction_yaw_deg": fixed_target_metadata["direction_yaw_deg"],
                "catalog_hash": fixed_target_metadata["catalog_hash"],
                "speed_source": (
                    "explicit_fixed_trajectory_speed_m_s"
                    if args.fixed_trajectory_speed_m_s is not None
                    else "selected_source_right_tcp_average_speed"
                ),
            }
        else:
            if detail_selection is not None:
                fixed_phase = "right_write.write"
                fixed_action_source = "rule_based_tcp_circle"
            circle_kwargs: dict[str, Any] = {
                "radius_m": float(args.circle_radius_m),
                "steps": int(args.circle_steps),
            }
            if args.circle_hz is not None:
                circle_kwargs["hz"] = float(args.circle_hz)
            else:
                circle_kwargs["linear_speed_m_s"] = float(
                    args.circle_speed_m_s
                    if args.circle_speed_m_s is not None
                    else source_speed["average_linear_speed_m_s"]
                )
            fixed_targets = _circle_targets(fixed_target_boundary, **circle_kwargs)
            circle_target_metadata = dict(fixed_targets[0].metadata)
            fixed_speed_plan = {
                "reference_side": DEFAULT_CIRCLE_SPEED_SIDE,
                "pose_frame": "canonical_desk",
                "source_reference": source_speed,
                "target_linear_speed_m_s": circle_target_metadata["linear_speed_m_s"],
                "target_path_length_m": circle_target_metadata["path_length_m"],
                "target_duration_sec": circle_target_metadata["duration_sec"],
                "speed_source": (
                    "explicit_circle_hz"
                    if args.circle_hz is not None
                    else (
                        "explicit_linear_speed_m_s"
                        if args.circle_speed_m_s is not None
                        else "selected_source_right_tcp_average_speed"
                    )
                ),
            }
        fixed_refs = resample_motion_targets(
            fixed_targets,
            reference_hz=float(args.reference_hz),
            include_final=True,
        )
        run_phase(
            fixed_phase,
            fixed_refs,
            lambda _reference: dict(initial_grippers),
            action_source=fixed_action_source,
            source_index_fn=lambda _reference: (-1,),
            action_source_by_side_fn=(
                None
                if detail_selection is None
                else lambda _reference: {
                    "left": "dataset_fixed_hold",
                    "right": fixed_action_source,
                }
            ),
            source_frame_indices_by_side_fn=(
                None
                if detail_selection is None
                else lambda _reference: {
                    "left": [int(left_hold_source_index)],
                    "right": [-1],
                }
            ),
        )
        boundary_after_fixed = _actual_tcp_boundary(control)
        boundary_tcp_pose_by_side = (
            {"right": boundary_after_fixed["right"]}
            if detail_selection is not None
            else boundary_after_fixed
        )
        boundary = RuleBasedBoundaryState(
            tcp_pose_by_side=boundary_tcp_pose_by_side,
            pose_frame_by_side={
                side: "canonical_desk" for side in boundary_tcp_pose_by_side
            },
            source=f"mujoco_actual_tcp_after_{fixed_phase}",
            timestamp=float(control.scene.data.time),
            metadata={
                "phase": f"{fixed_phase}_boundary",
                "plane": "desk_xy",
                "measurement": "retargeter_fk_from_mujoco_qpos",
            },
        )

        rebased_targets: list[TimestampedMotionTarget] = []
        for index in range(stage_b_group.frame_count):
            action = rebase_replay_action_at_boundary(
                stage_b_group,
                ("right",) if detail_selection is not None else ("left", "right"),
                index=index,
                boundary=boundary,
                source_start_index=0,
                include_end_effectors=False,
                source="rule_based_relative_replay",
            )
            rebased_pose_by_side = {
                side: np.asarray(pose, dtype=float).copy()
                for side, pose in action.arm_pose_by_side.items()
            }
            if detail_selection is not None:
                rebased_pose_by_side["left"] = left_hold_pose_by_side["left"].copy()
            rebased_targets.append(
                TimestampedMotionTarget(
                    timestamp=stage_b_group.relative_time_sec(index),
                    available_timestamp=stage_b_group.relative_time_sec(index),
                    source_index=index,
                    pose_by_side=rebased_pose_by_side,
                    metadata={
                        **dict(action.metadata),
                        "phase": (
                            "right_write.pen_up"
                            if detail_selection is not None
                            else "relative_stage_b"
                        ),
                        "source_frame_index": int(stage_b_frames[index].frame_index),
                    },
                )
            )
        relative_refs = resample_motion_targets(
            rebased_targets,
            reference_hz=float(args.reference_hz),
            include_final=True,
        )
        run_phase(
            "right_write.pen_up" if detail_selection is not None else "relative_stage_b",
            relative_refs,
            lambda reference: _grippers_for_reference(
                stage_b_frames,
                reference,
                hold_by_side=(
                    None
                    if left_hold_gripper is None
                    else {"left": float(left_hold_gripper)}
                ),
            ),
            action_source="rule_based_relative_replay",
            source_index_fn=lambda reference: (
                int(stage_b_frames[reference.left_source_index].frame_index),
                int(stage_b_frames[reference.right_source_index].frame_index),
            ),
            action_source_by_side_fn=(
                None
                if detail_selection is None
                else lambda _reference: {
                    "left": "dataset_fixed_hold",
                    "right": "rule_based_relative_replay",
                }
            ),
            source_frame_indices_by_side_fn=(
                None
                if detail_selection is None
                else lambda reference: {
                    "left": [int(left_hold_source_index)],
                    "right": [
                        int(stage_b_frames[reference.right_source_index].frame_index)
                    ],
                }
            ),
        )
        control_snapshot = control.snapshot()
    finally:
        if writer is not None:
            writer.close(
                summary={
                    "schema": "dexteleop.sim.mujoco.rule_based_multistage_writer_close.v1",
                    "phase_counts": dict(phase_counts),
                }
            )
        viewer_context.__exit__(None, None, None)
        if renderer is not None:
            renderer.close()
        control.close()

    left_actual_qpos_span = None
    if left_actual_qpos_samples:
        left_actual_qpos_span = np.ptp(
            np.stack(left_actual_qpos_samples, axis=0),
            axis=0,
        )
    fixed_phase_payload: dict[str, Any] = {
        "capture": True,
        "tcp_side": "right",
        "plane": "desk_xy",
        "constant_axis": "desk_z",
        "trajectory_type": (
            "catalogued_one_stroke"
            if args.fixed_trajectory_id is not None
            else "circle"
        ),
        "fixed_trajectory_id": args.fixed_trajectory_id,
        "radius_m": (
            None
            if args.fixed_trajectory_id is not None
            else float(args.circle_radius_m)
        ),
        "size_m": (
            float(args.fixed_trajectory_scale_m)
            if args.fixed_trajectory_id is not None
            else None
        ),
        "direction": (
            str(args.fixed_trajectory_direction)
            if args.fixed_trajectory_id is not None
            else None
        ),
        "steps": len(fixed_targets),
        "hz": None if args.circle_hz is None else float(args.circle_hz),
        "speed": fixed_speed_plan,
        "boundary_before_fixed": boundary_before_fixed,
        "boundary_after_fixed": boundary_after_fixed,
        **(
            {
                "boundary_before_circle": boundary_before_fixed,
                "boundary_after_circle": boundary_after_fixed,
            }
            if fixed_phase == "rule_based_circle"
            else {}
        ),
    }
    if detail_selection is not None:
        detail_phase_payload = {
            phase_id: detail_selection.phases[phase_id].to_dict()
            for phase_id in DETAIL_PHASE_IDS
        }
        composition: dict[str, Any] = {
            "mode": "detail_pen_down_rule_based_write_relative_pen_up",
            "initial_alignment": {
                "capture": False,
                "target": "right_write.pen_down.first_frame",
                "left_hold": True,
            },
            "right_write.pen_down": {
                **detail_phase_payload["right_write.pen_down"],
                "capture": True,
                "execution": "absolute_dataset_replay",
            },
            "right_write.write": {
                **detail_phase_payload["right_write.write"],
                **fixed_phase_payload,
                "source_replay": "replaced_by_rule_based_trajectory",
            },
            "right_write.pen_up": {
                **detail_phase_payload["right_write.pen_up"],
                "capture": True,
                "execution": "relative_dataset_replay",
                "source_start_index": 0,
                "anchor": f"mujoco_actual_tcp_after_{fixed_phase}",
                "pose_frame": "canonical_desk",
            },
            "left_hold": {
                "capture_active": True,
                "control_source": "dataset_fixed_hold",
                "source_frame_index": left_hold_source_index,
                "target_pose_desk": left_hold_pose_by_side["left"],
                "target_gripper_close": left_hold_gripper,
                "actual_qpos_span_rad": left_actual_qpos_span,
            },
        }
    else:
        composition = {
            "mode": "legacy_stage_a_rule_based_relative_stage_b",
            "initial_alignment": {"capture": False, "target": "stage_a.first_frame"},
            "absolute_stage_a": stage_a_span.to_dict(),
            fixed_phase: fixed_phase_payload,
            "relative_stage_b": {
                **stage_b_span.to_dict(),
                "capture": True,
                "source_start_index": 0,
                "anchor": f"mujoco_actual_tcp_after_{fixed_phase}",
                "pose_frame": "canonical_desk",
            },
        }
    summary: dict[str, Any] = {
        "schema": "dexteleop.sim.mujoco.rule_based_multistage_summary.v2",
        "simulation_only": True,
        "real_hardware_access": False,
        "source": {
            "dataset_root": str(episode.episode.dataset_root),
            "episode_id": episode.episode.episode_id,
            "progress_episode_uuid": (
                None
                if detail_selection is None
                else detail_selection.episode_uuid
            ),
            "source_parquet_path": str(episode.episode.parquet_path),
            "source_frame_count": len(episode.frames),
            "action_dim": int(episode.action_dim),
            "nominal_hz": float(episode.nominal_hz),
            "pose_storage_frame": episode.pose_frame_contract.pose_storage_frame,
            "detail_annotation": (
                None
                if detail_selection is None
                else {
                    "directory": str(args.detail_annotation_dir),
                    "annotation_set_id": detail_selection.annotation_set_id,
                    "base_annotation_set_id": detail_selection.base_annotation_set_id,
                    "base_annotation_manifest_hash": detail_selection.base_annotation_manifest_hash,
                    "detail_annotation_bundle_hash": detail_selection.detail_annotation_bundle_hash,
                    "detail_annotation_manifest_hash": detail_selection.detail_annotation_manifest_hash,
                    "detail_task_stage_profile_hash": detail_selection.detail_task_stage_profile_hash,
                    "resolution": "explicit_immutable_overlay_reader",
                }
            ),
        },
        "pose_frame_alignment": pose_frame_alignment,
        "composition": composition,
        "capture": {
            "output_root": str(args.output),
            "camera_names": list(requested_cameras),
            "depth_camera_names": list(depth_cameras),
            "recorded_frame_count": int(capture_frame_index),
            "phase_counts": dict(phase_counts),
            "recorded_phases": list(phase_counts),
            "reference_hz": float(args.reference_hz),
            "left_hold_actual_qpos_span_rad": left_actual_qpos_span,
        },
        "control": control_snapshot,
        "wall_time_sec": time.monotonic() - started,
    }
    manifest_path = Path(args.output) / "rule_based_composition_manifest.json"
    _write_json(manifest_path, summary)
    summary_path = args.summary_json or (Path(args.output) / "summary.json")
    _write_json(summary_path, summary)
    print(
        "[rule-based-multistage] done "
        f"frames={capture_frame_index} phases={phase_counts} output={args.output}"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_STAGE_A",
    "DEFAULT_STAGE_B",
    "StageSpan",
    "_circle_targets",
    "main",
    "parse_args",
    "parse_stage_span",
    "run",
]
