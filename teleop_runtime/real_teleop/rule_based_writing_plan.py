"""Runtime-level assembly for the real rule-based writing example.

This module is deliberately a thin adapter.  It reads one immutable source
episode and its detail overlay, then composes existing replay, fixed-path, and
RealTeleop extension types.  It does not own hardware, safety, IK, recording,
or device lifecycle.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from dataset_interface.progress_artifacts import load_sealed_progress_episode
from dataset_interface.progress_detail_annotation_artifacts import (
    LoadedDetailStageAnnotationSet,
    load_detail_stage_annotation_set,
)
from dataset_interface.progress_stage_annotation_artifacts import (
    load_stage_annotation_set,
)
from robot_motion.action import BIMANUAL_EE_POSE_V1, RobotAction
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES
from robot_motion.position_control import resample_motion_targets
from robot_motion.relative_action import pose_matrix_to_xyzquat
from robot_motion.replay import (
    CANONICAL_DESK_ACTION_FRAME,
    FIXED_TRAJECTORY_DIRECTIONS,
    ReplayTrajectoryGroup,
    RuleBasedActionSegment,
    RuleBasedCaptureSelection,
    RuleBasedCollectionPlan,
    RuleBasedInitialize,
    RuleBasedReplay,
    RuleBasedSegmentStopCondition,
    available_fixed_trajectory_ids,
    build_fixed_trajectory_targets,
)
from teleop_runtime.deploy.replay import (
    load_contract_replay_actions_from_parquet,
    load_replay_arm_actual_tcp_pose_from_parquet,
    load_replay_arm_observation_qpos_from_parquet,
    resolve_replay_parquet_path,
)
from teleop_runtime.real_teleop.rule_based_runtime import (
    RuleBasedRealTeleopConfig,
    RuleBasedRealTeleopDeployment,
    RuleBasedRealTeleopExtension,
    preflight_rule_based_real_teleop_deployment,
)
from teleop_runtime.replay.dataset import LoadedReplayEpisode, ReplayDatasetIndex

DETAIL_PROGRESS_STEP_ID = "right_write"
DETAIL_PHASE_IDS = (
    "right_write.pen_down",
    "right_write.write",
    "right_write.pen_up",
)


@dataclass(frozen=True)
class StageSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if int(self.start) < 0 or int(self.end) <= int(self.start):
            raise ValueError(f"invalid half-open stage span: {self.start}:{self.end}")

    @property
    def frame_count(self) -> int:
        return int(self.end) - int(self.start)


@dataclass(frozen=True)
class DetailWritingSelection:
    episode_uuid: str
    annotation_set_id: str
    phases: Mapping[str, StageSpan]

    def __post_init__(self) -> None:
        phases = dict(self.phases)
        if tuple(phases) != DETAIL_PHASE_IDS:
            raise ValueError(
                "detail writing phases must be ordered exactly as "
                f"{DETAIL_PHASE_IDS!r}."
            )
        object.__setattr__(self, "phases", phases)


@dataclass(frozen=True)
class RuleBasedWritingBuild:
    extension: RuleBasedRealTeleopExtension
    source_episode: int
    source_contract_hash: str
    detail_annotation_set_id: str
    writing_speed_m_s: float
    writing_action_hz: float


def add_rule_based_writing_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("rule-based writing collection")
    group.add_argument(
        "--rule-source-dataset",
        required=True,
        help="Immutable source dataset root containing the annotated episode.",
    )
    group.add_argument(
        "--rule-source-episode",
        required=True,
        type=int,
        help="Source episode index containing pen_down/write/pen_up detail stages.",
    )
    group.add_argument(
        "--rule-detail-annotation-dir",
        required=True,
        help="Exact immutable detail annotation-set directory for the source episode.",
    )
    group.add_argument(
        "--rule-writing-trajectory",
        default="digit_1_v1",
        choices=available_fixed_trajectory_ids(),
        help="One-stroke fixed trajectory inserted in place of the source write stage.",
    )
    group.add_argument(
        "--rule-writing-size-m",
        type=float,
        default=0.08,
        help="Authored glyph height in metres before direction rotation.",
    )
    group.add_argument(
        "--rule-writing-direction",
        choices=FIXED_TRAJECTORY_DIRECTIONS,
        default="up",
        help="Direction of glyph-local +Y in canonical desk coordinates.",
    )
    group.add_argument(
        "--rule-writing-speed-m-s",
        type=float,
        default=None,
        help="Optional TCP path speed; default is the source replay average.",
    )


def _episode_uuid(episode: LoadedReplayEpisode) -> str:
    record = episode.summary.record
    candidates: list[Mapping[str, Any]] = [record]
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
        "rule source episode is missing metadata.progress_episode.episode_uuid."
    )


def _row_timestamp(row: Mapping[str, Any]) -> float:
    for key in ("timestamp", "observation.timestamp"):
        value = row.get(key)
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(timestamp):
            return timestamp
    raise ValueError("rule source row is missing a finite dataset timestamp.")


def _load_detail_selection(
    episode: LoadedReplayEpisode,
    annotation_dir: str | Path,
) -> DetailWritingSelection:
    detail: LoadedDetailStageAnnotationSet = load_detail_stage_annotation_set(
        annotation_dir
    )
    episode_uuid = _episode_uuid(episode)
    dataset_root = episode.dataset_root
    sealed = load_sealed_progress_episode(
        dataset_root / "meta" / "progress" / "episodes" / episode_uuid
    )
    binding = detail.binding
    if binding.episode_capture_identity_hash != sealed.binding.episode_capture_identity_hash:
        raise ValueError("detail overlay and source progress Episode identity differ.")
    if binding.source_progress_manifest_hash != sealed.manifest.semantic_content_hash:
        raise ValueError("detail overlay does not bind the source progress manifest.")
    if binding.bound_clock_domain_id != sealed.binding.bound_clock_domain_id:
        raise ValueError("detail overlay and source Episode use different clocks.")
    if binding.target_progress_step_id != DETAIL_PROGRESS_STEP_ID:
        raise ValueError(
            "detail overlay must target the right_write progress step, got "
            f"{binding.target_progress_step_id!r}."
        )

    base_id = binding.base_annotation_set_id
    if Path(base_id).name != base_id or base_id in {".", ".."}:
        raise ValueError("detail binding contains an unsafe base annotation-set ID.")
    base = load_stage_annotation_set(
        dataset_root
        / "meta"
        / "progress"
        / "stage_annotations"
        / episode_uuid
        / base_id
    )
    if (
        base.binding.annotation_set_id != base_id
        or base.binding.episode_capture_identity_hash
        != binding.episode_capture_identity_hash
        or base.binding.source_progress_manifest_hash
        != binding.source_progress_manifest_hash
        or base.binding.task_stage_profile_hash
        != binding.base_task_stage_profile_hash
        or base.binding.bound_clock_domain_id != binding.bound_clock_domain_id
        or base.manifest.semantic_content_hash
        != binding.base_annotation_set_manifest_hash
    ):
        raise ValueError("detail overlay does not bind the exact base annotation set.")

    occurrences = tuple(detail.bundle.occurrences)
    if any(item.progress_step_id != DETAIL_PROGRESS_STEP_ID for item in occurrences):
        raise ValueError("detail overlay contains an occurrence outside right_write.")
    attempts = {item.coarse_attempt_id for item in occurrences}
    if len(attempts) != 1:
        raise ValueError("detail overlay must contain exactly one right_write attempt.")
    ordered = tuple(
        sorted(
            occurrences,
            key=lambda item: (
                int(item.occurrence_index_in_attempt),
                float(item.start_at),
            ),
        )
    )
    if (
        tuple(item.stage_id for item in ordered) != DETAIL_PHASE_IDS
        or tuple(int(item.occurrence_index_in_attempt) for item in ordered)
        != (1, 2, 3)
        or any(item.outcome != "SUCCESS" for item in ordered)
    ):
        raise ValueError(
            "detail overlay must contain successful ordered pen_down/write/pen_up phases."
        )
    if any(item.clock_domain_id != binding.bound_clock_domain_id for item in ordered):
        raise ValueError("detail phase clock does not match the bound Episode clock.")
    if any(
        float(current.start_at) < float(previous.end_at)
        for previous, current in pairwise(ordered)
    ):
        raise ValueError("detail phase intervals overlap.")

    timestamps = np.asarray([_row_timestamp(row) for row in episode.rows], dtype=float)
    if timestamps.size < 1 or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("rule source timestamps must be strictly increasing.")
    phases: dict[str, StageSpan] = {}
    for item in ordered:
        start = int(np.searchsorted(timestamps, float(item.start_at), side="left"))
        end = int(np.searchsorted(timestamps, float(item.end_at), side="left"))
        end = min(end, int(timestamps.size))
        phases[item.stage_id] = StageSpan(start=start, end=end)
    for previous, current in pairwise(DETAIL_PHASE_IDS):
        if phases[previous].end > phases[current].start:
            raise ValueError(f"mapped detail phases overlap: {previous} and {current}.")
    return DetailWritingSelection(
        episode_uuid=episode_uuid,
        annotation_set_id=binding.annotation_set_id,
        phases=phases,
    )


def _canonical_actual_tcp(
    raw_by_side: Mapping[str, np.ndarray],
    *,
    source_contract: Any,
) -> dict[str, np.ndarray]:
    converted: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        raw = np.asarray(raw_by_side.get(side), dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 7 or not np.all(np.isfinite(raw)):
            raise ValueError(
                f"relative replay requires finite source actual TCP history for {side}."
            )
        converted[side] = np.stack(
            [
                source_contract.storage_pose_vector_to_canonical_desk(row, side=side)
                for row in raw
            ],
            axis=0,
        ).astype(np.float32)
    return converted


def _slice_group(
    *,
    key: str,
    span: StageSpan,
    episode_index: int,
    source_path: Path,
    action_hz: float,
    frame_times_sec: Sequence[float],
    actions: np.ndarray,
    actual_qpos: Mapping[str, np.ndarray],
    actual_tcp: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    left_hold_pose: np.ndarray,
) -> ReplayTrajectoryGroup:
    selected_actions = np.asarray(actions[span.start : span.end], dtype=np.float32).copy()
    selected_actions[:, ACTION_ARM_POSE_SLICES["left"]] = left_hold_pose
    times = np.asarray(frame_times_sec[span.start : span.end], dtype=float)
    times -= times[0]
    return ReplayTrajectoryGroup(
        key=key,
        label=key,
        source_path=str(source_path),
        source_episode_index=int(episode_index),
        start_index=int(span.start),
        end_index=int(span.end),
        frame_count=span.frame_count,
        action_hz=float(action_hz),
        actions=selected_actions,
        frame_times_sec=tuple(float(value) for value in times),
        arm_observation_actual_qpos={
            side: np.asarray(values[span.start : span.end], dtype=np.float32).copy()
            for side, values in actual_qpos.items()
        },
        arm_observation_actual_tcp_pose={
            side: np.asarray(values[span.start : span.end], dtype=np.float32).copy()
            for side, values in actual_tcp.items()
        },
        metadata=dict(metadata),
    )


def _average_tcp_speed(
    actions: np.ndarray,
    frame_times_sec: Sequence[float],
    spans: Sequence[StageSpan],
) -> float:
    path_length = 0.0
    duration = 0.0
    right_slice = ACTION_ARM_POSE_SLICES["right"]
    times = np.asarray(frame_times_sec, dtype=float)
    for span in spans:
        positions = np.asarray(actions[span.start : span.end, right_slice.start : right_slice.start + 3])
        selected_times = times[span.start : span.end]
        if len(selected_times) < 2:
            continue
        deltas = np.diff(selected_times)
        if np.any(~np.isfinite(deltas)) or np.any(deltas <= 0.0):
            raise ValueError("source replay frame times must be strictly increasing.")
        path_length += float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        duration += float(deltas.sum())
    if duration <= 0.0 or path_length <= 0.0:
        raise ValueError("cannot derive a positive source TCP writing speed.")
    return path_length / duration


class FixedWritingActionSource:
    """Lazily anchor one catalog path at the measured pen-down boundary."""

    def __init__(
        self,
        *,
        trajectory_id: str,
        scale_m: float,
        speed_m_s: float,
        direction: str,
        left_hold_pose: np.ndarray,
        reference_hz: float,
    ) -> None:
        self.trajectory_id = str(trajectory_id)
        self.scale_m = float(scale_m)
        self.speed_m_s = float(speed_m_s)
        self.direction = str(direction)
        self.left_hold_pose = np.asarray(left_hold_pose, dtype=float).reshape(7).copy()
        self.reference_hz = float(reference_hz)
        self._targets: list[Any] | None = None
        self._index = 0

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def produce_action(self, observation: Any, *, context: Any = None) -> RobotAction:
        del observation
        if self._targets is None:
            if not isinstance(context, Mapping):
                raise RuntimeError("fixed writing requires the RealTeleop runtime context.")
            snapshots = context.get("actual_arm_state_by_side")
            contract = context.get("runtime_pose_frame_contract")
            if not isinstance(snapshots, Mapping) or contract is None:
                raise RuntimeError("fixed writing is missing causal TCP or frame context.")
            right_snapshot = snapshots.get("right")
            if not isinstance(right_snapshot, Mapping):
                raise RuntimeError("fixed writing is missing right-arm causal TCP feedback.")
            right_raw = pose_matrix_to_xyzquat(right_snapshot.get("actual_tcp_pose"))
            right_desk = contract.storage_pose_vector_to_canonical_desk(
                right_raw,
                side="right",
            )
            keyframes = build_fixed_trajectory_targets(
                {
                    "left": self.left_hold_pose,
                    "right": right_desk,
                },
                self.trajectory_id,
                scale_m=self.scale_m,
                linear_speed_m_s=self.speed_m_s,
                side="right",
                direction=self.direction,
            )
            self._targets = resample_motion_targets(
                keyframes,
                reference_hz=self.reference_hz,
                include_final=True,
            )
        if self._index >= len(self._targets):
            raise RuntimeError("fixed writing source was consumed past its bounded path.")
        target = self._targets[self._index]
        self._index += 1
        return RobotAction(
            action_space=BIMANUAL_EE_POSE_V1,
            timestamp=float(target.timestamp),
            arm_pose_by_side={
                side: pose_matrix_to_xyzquat(pose)
                for side, pose in target.pose_by_side.items()
            },
            frame_by_effector={
                "left_arm": CANONICAL_DESK_ACTION_FRAME,
                "right_arm": CANONICAL_DESK_ACTION_FRAME,
            },
            controlled_frame_by_effector={
                "left_arm": "tool_tcp",
                "right_arm": "tool_tcp",
            },
            source="rule_based_fixed_writing",
            metadata=dict(target.metadata),
        )


def build_rule_based_writing(
    args: argparse.Namespace,
    *,
    deployment: RuleBasedRealTeleopDeployment | None = None,
) -> RuleBasedWritingBuild:
    deployment = deployment or preflight_rule_based_real_teleop_deployment(args)
    dataset_root = Path(args.rule_source_dataset).expanduser().resolve()
    episode_index = int(args.rule_source_episode)
    episode = ReplayDatasetIndex(dataset_root).load_episode(episode_index)
    selection = _load_detail_selection(
        episode,
        Path(args.rule_detail_annotation_dir).expanduser().resolve(),
    )

    pen_down = selection.phases[DETAIL_PHASE_IDS[0]]
    pen_up = selection.phases[DETAIL_PHASE_IDS[2]]
    hold_rows = episode.rows[pen_down.start : pen_up.end]
    labels = {
        str(row.get("teleop.arm_mode.left.label", "")).strip()
        for row in hold_rows
    }
    if labels != {"fixed_hold"}:
        raise ValueError(
            "rule-based writing requires fixed_hold for the left arm over "
            f"pen_down/write/pen_up; got {sorted(labels) or ['missing']}."
        )

    source_path, resolved_metadata = resolve_replay_parquet_path(
        dataset_root=str(dataset_root),
        parquet_path=None,
        episode_index=episode_index,
    )
    if source_path.resolve() != episode.summary.parquet_path.resolve():
        raise ValueError("source episode index resolved to conflicting parquet paths.")
    actions, action_metadata, source_contract, is_typed = (
        load_contract_replay_actions_from_parquet(
            source_path,
            resolved_metadata=resolved_metadata,
            action_column="action",
            start_index=0,
            end_index=None,
        )
    )
    if source_contract is None:
        raise ValueError(
            "real rule-based replay requires a typed source dataset with an explicit "
            "pose-frame contract."
        )
    if source_contract.contract_hash != deployment.collection_pose_frame_contract.contract_hash:
        raise ValueError(
            "source and real deployment pose-frame contract hashes differ; refusing "
            "to infer coordinate alignment."
        )
    if actions.shape[0] != episode.frame_count:
        raise ValueError("source action rows do not match the indexed episode frame count.")

    actual_qpos = load_replay_arm_observation_qpos_from_parquet(
        source_path,
        active_sides=("left", "right"),
        field_name="actual_hardware_qpos",
        start_index=0,
        end_index=None,
    )
    if set(actual_qpos) != {"left", "right"}:
        raise ValueError("initial alignment requires source actual qpos for both arms.")
    raw_actual_tcp = load_replay_arm_actual_tcp_pose_from_parquet(
        source_path,
        active_sides=("left", "right"),
        start_index=0,
        end_index=None,
    )
    actual_tcp = _canonical_actual_tcp(
        raw_actual_tcp,
        source_contract=source_contract,
    )
    left_hold_pose = np.asarray(
        actions[pen_down.start, ACTION_ARM_POSE_SLICES["left"]],
        dtype=np.float32,
    ).copy()
    if left_hold_pose.shape != (7,) or not np.all(np.isfinite(left_hold_pose)):
        raise ValueError("source pen-down left hold pose is invalid.")

    metadata = {
        "action_metadata": dict(action_metadata),
        "source_metadata": dict(resolved_metadata),
        "typed_action_contract": bool(is_typed),
        "detail_annotation_set_id": selection.annotation_set_id,
        "left_arm_policy": "source_fixed_hold",
    }
    groups = {
        "pen_down": _slice_group(
            key="pen_down",
            span=pen_down,
            episode_index=episode_index,
            source_path=source_path,
            action_hz=episode.nominal_hz,
            frame_times_sec=episode.frame_times_sec,
            actions=actions,
            actual_qpos=actual_qpos,
            actual_tcp=actual_tcp,
            metadata=metadata,
            left_hold_pose=left_hold_pose,
        ),
        "pen_up": _slice_group(
            key="pen_up",
            span=pen_up,
            episode_index=episode_index,
            source_path=source_path,
            action_hz=episode.nominal_hz,
            frame_times_sec=episode.frame_times_sec,
            actions=actions,
            actual_qpos=actual_qpos,
            actual_tcp=actual_tcp,
            metadata=metadata,
            left_hold_pose=left_hold_pose,
        ),
    }

    speed = (
        float(args.rule_writing_speed_m_s)
        if args.rule_writing_speed_m_s is not None
        else _average_tcp_speed(
            actions,
            episode.frame_times_sec,
            (pen_down, pen_up),
        )
    )
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("rule writing speed must be finite and positive.")
    size_m = float(args.rule_writing_size_m)
    if not np.isfinite(size_m) or size_m <= 0.0:
        raise ValueError("rule writing size must be finite and positive.")

    preview_keyframes = build_fixed_trajectory_targets(
        {"left": left_hold_pose, "right": actions[pen_down.end - 1, ACTION_ARM_POSE_SLICES["right"]]},
        args.rule_writing_trajectory,
        scale_m=size_m,
        linear_speed_m_s=speed,
        side="right",
        direction=args.rule_writing_direction,
    )
    preview_targets = resample_motion_targets(
        preview_keyframes,
        reference_hz=float(episode.nominal_hz),
        include_final=True,
    )
    duration = float(preview_targets[-1].timestamp - preview_targets[0].timestamp)
    if duration <= 0.0 or len(preview_targets) < 2:
        raise ValueError("fixed writing trajectory has no positive-duration path.")
    writing_action_hz = float((len(preview_targets) - 1) / duration)
    writing_source = FixedWritingActionSource(
        trajectory_id=args.rule_writing_trajectory,
        scale_m=size_m,
        speed_m_s=speed,
        direction=args.rule_writing_direction,
        left_hold_pose=left_hold_pose,
        reference_hz=float(episode.nominal_hz),
    )
    capture = RuleBasedCaptureSelection(enabled=True)
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedInitialize(
                step_id="initial_alignment",
                group_key="pen_down",
                frame_index=0,
                active_sides=("left", "right"),
                qpos_source="observation_actual_qpos",
                capture=RuleBasedCaptureSelection(enabled=False),
            ),
            RuleBasedReplay(
                step_id=DETAIL_PHASE_IDS[0],
                group_key="pen_down",
                start_index=0,
                end_index=groups["pen_down"].frame_count,
                active_sides=("left", "right"),
                capture=capture,
            ),
            RuleBasedActionSegment(
                step_id=DETAIL_PHASE_IDS[1],
                action_source=writing_source,
                stop_condition=RuleBasedSegmentStopCondition(
                    max_steps=len(preview_targets),
                    min_steps=len(preview_targets),
                    name="fixed_writing_path_complete",
                ),
                capture=capture,
            ),
            RuleBasedReplay(
                step_id=DETAIL_PHASE_IDS[2],
                group_key="pen_up",
                start_index=0,
                end_index=groups["pen_up"].frame_count,
                active_sides=("right",),
                relative_to=DETAIL_PHASE_IDS[1],
                source_start_index=0,
                capture=capture,
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=deployment,
        plan=plan,
        trajectory_groups=groups,
        config=RuleBasedRealTeleopConfig(action_hz=writing_action_hz),
    )
    return RuleBasedWritingBuild(
        extension=extension,
        source_episode=episode_index,
        source_contract_hash=source_contract.contract_hash,
        detail_annotation_set_id=selection.annotation_set_id,
        writing_speed_m_s=speed,
        writing_action_hz=writing_action_hz,
    )


def build_rule_based_writing_extension(
    args: argparse.Namespace,
) -> RuleBasedRealTeleopExtension:
    build = build_rule_based_writing(args)
    print(
        "[RuleBased] prepared "
        f"episode={build.source_episode} "
        f"annotation_set={build.detail_annotation_set_id} "
        f"speed_m_s={build.writing_speed_m_s:.6f} "
        f"action_hz={build.writing_action_hz:.3f} "
        f"pose_contract={build.source_contract_hash}"
    )
    return build.extension


__all__ = [
    "DETAIL_PHASE_IDS",
    "DETAIL_PROGRESS_STEP_ID",
    "DetailWritingSelection",
    "FixedWritingActionSource",
    "RuleBasedWritingBuild",
    "StageSpan",
    "add_rule_based_writing_arguments",
    "build_rule_based_writing",
    "build_rule_based_writing_extension",
]
