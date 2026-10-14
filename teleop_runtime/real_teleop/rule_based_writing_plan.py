"""Runtime-level assembly for rule writing and complete-Episode replay capture.

This module is deliberately a thin adapter.  It either selects immutable
annotated writing subphases or one complete source Episode, then composes the
existing replay, fixed-path, and RealTeleop extension types.  It does not own
hardware, safety, IK, recording, or device lifecycle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from dataset_contracts.capture_modalities import CaptureModalityContract
from dataset_interface.capture_modalities import (
    dataset_is_blank,
    read_capture_modality_contract,
)
from dataset_contracts.rule_based_writing_sources import (
    RULE_BASED_WRITING_SOURCE_PHASES_V1,
    RuleBasedWritingSourceCandidateV1,
)
from dataset_interface.progress_artifacts import load_sealed_progress_episode
from dataset_interface.progress_detail_annotation_artifacts import (
    LoadedDetailStageAnnotationSet,
    load_detail_stage_annotation_set,
)
from dataset_interface.progress_stage_annotation_artifacts import (
    load_stage_annotation_set,
)
from dataset_interface.rule_based_writing_sources import (
    ResolvedRuleBasedWritingSourceCandidate,
    load_rule_based_writing_source_pool,
)
from robot_motion.action import BIMANUAL_EE_POSE_V1, HandActionSpec, RobotAction
from robot_motion.action.hand_spec import hand_action_specs_from_metadata
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_HAND_SLICES
from robot_motion.position_control import (
    MotionTargetResampler,
    TimestampedMotionTarget,
    resample_motion_targets,
)
from robot_motion.relative_action import pose_matrix_to_xyzquat, xyzquat_to_pose_matrix
from robot_motion.replay import (
    CANONICAL_DESK_ACTION_FRAME,
    FIXED_TRAJECTORY_DIRECTIONS,
    ReplayTrajectoryGroup,
    RuleBasedBoundaryState,
    RuleBasedActionSegment,
    RuleBasedCaptureSelection,
    RuleBasedCollectionPlan,
    RuleBasedInitialize,
    RuleBasedReplay,
    RuleBasedSegmentStopCondition,
    available_fixed_trajectory_ids,
    build_fixed_trajectory_targets,
    fixed_trajectory_direction_yaw_rad,
    fixed_trajectory_catalog_hash,
    rebase_replay_action_at_boundary,
    get_fixed_trajectory,
    replay_action_metadata,
    replay_end_effector_targets_at_index,
)
from teleop_runtime.collection.sample_assembler import (
    DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
)
from teleop_runtime.real_teleop.hand_config import (
    HAND_BACKEND_DMTAC_GRIPPER,
    HAND_BACKEND_NONE,
    hand_backend_mode_for_side,
)
from teleop_runtime.real_teleop.control_profile import (
    effective_arm_control_hz,
    effective_outer_control_hz,
    effective_position_reference_hz,
)
from teleop_runtime.deploy.replay import (
    load_contract_replay_actions_from_parquet,
    load_replay_arm_actual_tcp_pose_from_parquet,
    load_replay_arm_observation_qpos_from_parquet,
    load_replay_hand_joint_targets_from_parquet,
    resolve_replay_parquet_path,
)
from teleop_runtime.real_teleop.rule_based_runtime import (
    RuleBasedRealTeleopConfig,
    RuleBasedRealTeleopDeployment,
    RuleBasedRealTeleopExtension,
    preflight_rule_based_real_teleop_deployment,
)
from teleop_runtime.real_teleop.writing_contact_point import (
    contact_height_from_tool_tcp,
    tool_tcp_z_for_contact_height,
)
from teleop_runtime.replay.dataset import LoadedReplayEpisode, ReplayDatasetIndex

DETAIL_PROGRESS_STEP_ID = "right_write"
DETAIL_PHASE_IDS = (
    "right_write.pen_down",
    "right_write.write",
    "right_write.pen_up",
)
HEIGHT_DESCENT_STEP_ID = "right_write.pen_down.height_descent"

# These bounds were introduced on the remote branch as general real-teleop
# defaults.  Keep the rule-writing leaf self-contained until that shared
# defaults change is reviewed independently.
DEFAULT_COLLECTION_FLUSH_TIMEOUT_SEC = 30.0
DEFAULT_COLLECTION_SAMPLE_DRAIN_TIMEOUT_SEC = 30.0
DEFAULT_COLLECTION_TERMINAL_EVENT_PUT_TIMEOUT_SEC = 0.5
DEFAULT_COLLECTION_STORAGE_QUEUE_ENQUEUE_TIMEOUT_SEC = 1.0
DEFAULT_RULE_TACTILE_MIN_FREE_GB = 10.0

RULE_COLLECTION_MODE_WRITING = "rule_based_writing"
RULE_COLLECTION_MODE_FULL_REPLAY = "full_trajectory_replay"
RULE_COLLECTION_MODES = (
    RULE_COLLECTION_MODE_WRITING,
    RULE_COLLECTION_MODE_FULL_REPLAY,
)
# This is a UI-only value.  It is never used as a dataset candidate ID; the
# source resolver turns it into a seeded source-pool choice before loading any
# replay data.  Keeping the sentinel explicit lets the GUI distinguish the
# default random policy from an operator-selected fixed candidate.
RULE_BASED_RANDOM_SOURCE_SELECTION_ID = "__rule_source_pool_random__"
FULL_REPLAY_INITIALIZE_STEP_ID = "execute_task.replay_initial_alignment"
FULL_REPLAY_STEP_ID = "execute_task.manipulate"
PRE_PEN_DOWN_ALIGNMENT_STEP_ID = "right_write.pre_pen_down.alignment"
PRE_PEN_DOWN_REPLAY_STEP_ID = "right_write.pre_pen_down.replay"
PRE_PEN_DOWN_TO_PEN_DOWN_ALIGNMENT_STEP_ID = (
    "right_write.pre_pen_down.to_pen_down_alignment"
)

_RULE_GUI_SHARED_RUNTIME_CONFIGURATION_FIELDS = (
    "collect_tactile",
    "tactile_bundle_root",
    "tactile_bundle_min_free_gb",
    "tactile_compression",
    "writer_workers",
    "writer_queue_size",
    "collection_process_queue_size",
    "storage_queue_size",
    "storage_queue_enqueue_timeout_sec",
    "camera_frame_max_age_sec",
    "collection_warning_hz",
    "collection_flush_timeout_sec",
    "collection_sample_drain_timeout_sec",
    "collection_terminal_event_put_timeout_sec",
    "write_parquet",
    "write_video",
)


def _positive_int(value: str) -> int:
    normalized = int(value)
    if normalized < 1:
        raise argparse.ArgumentTypeError("value must be an integer >= 1")
    return normalized


_RULE_GUI_RUNTIME_CONFIGURATION_FIELDS = (
    "writing_speed_m_s",
    "writing_speed_from_source",
    *_RULE_GUI_SHARED_RUNTIME_CONFIGURATION_FIELDS,
)


def _load_rule_source_pool_cached(
    args: argparse.Namespace,
    manifest: str | Path,
) -> object:
    """Load one immutable source-pool manifest at most once per CLI session.

    Building the initial rule extension needs the pool for deterministic
    selection and again for GUI option construction.  Re-reading and
    re-resolving every dataset directory on both paths made startup look like
    an unexplained second search (and multiplied the cost for a large pool).
    Cache the typed, resolved pool on the short-lived argument namespace while
    retaining a file identity check so an operator edit is never silently
    ignored during a configuration rebuild.
    """

    path = Path(manifest).expanduser().resolve()
    try:
        stat = path.stat()
        cache_key = (str(path), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))
    except OSError:
        cache_key = (str(path), None, None, None)
    cached = getattr(args, "_rule_source_pool_cache", None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == cache_key:
        return cached[1]
    loaded = load_rule_based_writing_source_pool(path)
    setattr(args, "_rule_source_pool_cache", (cache_key, loaded))
    return loaded


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
    detail_annotation_set_id: str | None
    writing_speed_m_s: float | None
    writing_action_hz: float
    source_selection: Mapping[str, Any]
    initial_alignment_speed_scale: float = 1.0
    gripper_sides: tuple[str, ...] = ()
    hand_action_specs_by_side: Mapping[str, HandActionSpec] = field(
        default_factory=dict
    )
    collection_mode: str = RULE_COLLECTION_MODE_WRITING


@dataclass(frozen=True)
class _LoadedWritingSource:
    candidate_id: str
    dataset_root: Path
    detail_annotation_dir: Path
    episode_index: int
    episode: LoadedReplayEpisode
    selection: DetailWritingSelection | None
    source_path: Path
    actions: np.ndarray
    hand_joint_targets: Mapping[str, np.ndarray]
    actual_qpos: Mapping[str, np.ndarray]
    actual_tcp: Mapping[str, np.ndarray]
    source_contract_hash: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RuleTactileCaptureResolution:
    """Effective tactile policy for one rule-based dataset target.

    The dataset-level capture contract is authoritative once a target contains
    data.  This keeps a later run from appending tactile and non-tactile
    episodes to the same dataset even when the operator changes the CLI/GUI
    toggle.
    """

    contract: CaptureModalityContract
    mode: str
    source: str
    locked: bool

    @property
    def collect_tactile(self) -> bool:
        return set(self.contract.ignored_tactile_keys) != {"left", "right"}


def _resolve_rule_tactile_capture(
    args: argparse.Namespace,
    *,
    requested: bool | None = None,
) -> RuleTactileCaptureResolution:
    """Resolve rule tactile capture without mixing a populated dataset.

    A blank target follows the explicit rule toggle, existing ``--ignore-tactile``
    values, or (for backwards compatibility) whether a tactile bundle root was
    supplied. A populated target always follows its persisted contract;
    ambiguous legacy populated targets fail closed instead of guessing.
    """

    dataset_text = str(getattr(args, "dataset_root", "") or "").strip()
    if not dataset_text:
        raise ValueError("rule-based tactile resolution requires --dataset-root.")
    dataset_root = Path(dataset_text).expanduser()
    current_camera_keys = tuple(getattr(args, "ignore_camera", ()) or ())
    existing = read_capture_modality_contract(dataset_root)
    blank = dataset_is_blank(dataset_root)
    if not blank:
        if existing is None:
            raise CaptureModalityBindingError(
                "target dataset contains data but has no capture modality contract; "
                "migrate meta/modality_validity.json explicitly before rule collection"
            )
        contract = existing
        source = "existing_dataset_contract"
        mode = (
            "enabled"
            if not contract.ignored_tactile_keys
            else "disabled"
            if set(contract.ignored_tactile_keys) == {"left", "right"}
            else "existing_partial"
        )
        return RuleTactileCaptureResolution(
            contract=contract,
            mode=mode,
            source=source,
            locked=True,
        )

    if requested is None:
        requested = getattr(args, "rule_collect_tactile", None)
    if requested is not None:
        ignored_tactile_keys = () if bool(requested) else ("left", "right")
        source = "rule_option"
    else:
        current_tactile_keys = tuple(getattr(args, "ignore_tactile", ()) or ())
        if current_tactile_keys:
            ignored_tactile_keys = current_tactile_keys
            source = "cli_ignore_tactile"
        else:
            explicit_dmtac_tactile = getattr(args, "dmtac_tactile", None)
            if explicit_dmtac_tactile is not None:
                ignored_tactile_keys = (
                    () if bool(explicit_dmtac_tactile) else ("left", "right")
                )
                source = "cli_dmtac_tactile"
            else:
                # The previous rule launcher always required a tactile root.
                # Keep that behavior when a root is present, while making a
                # root-less invocation a valid explicit no-tactile collection.
                has_tactile_root = bool(
                    str(getattr(args, "tactile_bundle_root", "") or "").strip()
                )
                ignored_tactile_keys = () if has_tactile_root else ("left", "right")
                source = "default_with_root" if has_tactile_root else "default_without_root"
    contract = CaptureModalityContract(
        ignored_camera_keys=current_camera_keys,
        ignored_tactile_keys=ignored_tactile_keys,
    )
    mode = (
        "enabled"
        if not contract.ignored_tactile_keys
        else "disabled"
        if set(contract.ignored_tactile_keys) == {"left", "right"}
        else "partial"
    )
    return RuleTactileCaptureResolution(
        contract=contract,
        mode=mode,
        source=source,
        locked=False,
    )


def _apply_rule_tactile_capture_resolution(
    args: argparse.Namespace,
    resolution: RuleTactileCaptureResolution,
) -> RuleTactileCaptureResolution:
    """Apply one resolved policy to the existing runtime argument namespace."""

    # The persisted contract covers camera omissions as well.  Restore the
    # complete contract before the recorder binds it again, otherwise an old
    # dataset with a camera ignore rule could fail append admission even when
    # only the tactile setting was changed in the UI.
    args.ignore_camera = list(resolution.contract.ignored_camera_keys)
    args.ignore_tactile = list(resolution.contract.ignored_tactile_keys)
    args.rule_collect_tactile = bool(resolution.collect_tactile)
    args.rule_tactile_capture_mode = resolution.mode
    args.rule_tactile_capture_source = resolution.source
    if not resolution.collect_tactile:
        args.dmtac_tactile = False
        args.tactile_bundle_root = ""
        args.tactile_bundle_min_free_gb = 0.0
        args.tactile_artifact_integrity = False
        args.idle_tactile_compression = False
        return resolution

    tactile_root = str(getattr(args, "tactile_bundle_root", "") or "").strip()
    if not tactile_root:
        raise ValueError(
            "当前规则数据集要求采集触觉，但未配置 --tactile-bundle-root；"
            "请提供已有触觉 Bundle 根目录，不能与无触觉数据混采。"
        )
    args.dmtac_tactile = True
    minimum_source = str(
        getattr(args, "tactile_bundle_min_free_source", "default") or "default"
    )
    minimum_free_gb = float(
        getattr(args, "tactile_bundle_min_free_gb", 0.0) or 0.0
    )
    if minimum_source == "default" and minimum_free_gb <= 0.0:
        args.tactile_bundle_min_free_gb = DEFAULT_RULE_TACTILE_MIN_FREE_GB
    return resolution


def add_rule_based_writing_arguments(parser: argparse.ArgumentParser) -> None:
    # Rule-based collection replays measured DMTAC state and never consumes a
    # human Q6/angle-trigger input stream.  Mark this specialized entrypoint
    # so CLI resolution can default to the explicit no-input topology without
    # changing the ordinary teleop default (automatic Q6 discovery).
    parser.set_defaults(
        _rule_based_writing_entrypoint=True,
        # Rule-based collection keeps the writer/camera/feedback warmup, but
        # has no operator-facing preparation countdown between readiness and
        # the first admitted sample.
        episode_start_countdown_sec=0.0,
    )
    group = parser.add_argument_group("rule-based writing collection")
    group.add_argument(
        "--rule-collection-mode",
        choices=RULE_COLLECTION_MODES,
        default=RULE_COLLECTION_MODE_WRITING,
        help=(
            "Collection profile: the existing three-stage rule-based writing "
            "flow, or a complete source-Episode replay recorded as one new Episode."
        ),
    )
    group.add_argument(
        "--rule-source-dataset",
        default=None,
        help=(
            "Immutable source dataset root. The writing profile requires an "
            "annotated Episode; complete replay consumes the whole Episode."
        ),
    )
    group.add_argument(
        "--rule-source-episode",
        default=None,
        type=int,
        help=(
            "Source Episode index. The writing profile requires pen_down/write/"
            "pen_up detail stages; complete replay does not."
        ),
    )
    group.add_argument(
        "--rule-detail-annotation-dir",
        default=None,
        help=(
            "Exact immutable detail annotation-set directory required only by "
            "the rule-based writing profile."
        ),
    )
    group.add_argument(
        "--rule-source-pool-manifest",
        default=None,
        help=(
            "JSON source pool whose candidates may span datasets; mutually exclusive "
            "with the three legacy single-source arguments."
        ),
    )
    group.add_argument(
        "--rule-source-pool-range",
        default=None,
        metavar="START:END",
        help="Half-open candidate-list range; default selects the complete pool.",
    )
    group.add_argument(
        "--rule-source-selection-seed",
        type=int,
        default=0,
        help=(
            "Deterministic source selection seed: per-subphase for writing or "
            "per-Episode for complete replay (default: 0)."
        ),
    )
    group.add_argument(
        "--rule-full-replay-candidate-id",
        default=None,
        help=(
            "Optional source-pool candidate selected for complete-Episode replay; "
            "the GUI can replace it before collection starts."
        ),
    )
    group.add_argument(
        "--rule-pre-pen-down-replay-candidate-id",
        default=None,
        help=(
            "Optional source-pool candidate replayed before pen-down. The runtime "
            "smoothly aligns to this complete Episode, replays its right-arm motion "
            "without capture, then smoothly aligns to the ordinary pen-down start."
        ),
    )
    group.add_argument(
        "--rule-pre-pen-down-replay-normal-episodes",
        type=int,
        default=0,
        help=(
            "Number of successful normal Episodes between pre-pen-down replays. "
            "Zero preserves the existing behavior and runs the configured replay "
            "before every Episode."
        ),
    )
    group.add_argument(
        "--rule-writing-trajectory-catalog",
        default=None,
        help=(
            "Optional absolute JSON catalog appended to the built-in trajectories. "
            "Ids must be unique and every trajectory must be one stroke."
        ),
    )
    group.add_argument(
        "--rule-writing-trajectory",
        default="digit_1_v1",
        help=(
            "One-stroke fixed trajectory inserted in place of the source write stage; "
            "validated against the composed catalog before device preflight."
        ),
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
        "--rule-writing-random-trajectory-ids",
        default="",
        metavar="IDS",
        help=(
            "Optional comma-separated one-stroke trajectory ids. One id is "
            "selected deterministically for each Episode from the source-selection "
            "seed and ordinal; empty preserves the fixed --rule-writing-trajectory."
        ),
    )
    group.add_argument(
        "--rule-writing-random-directions",
        default="",
        metavar="NAMES",
        help=(
            "Optional comma-separated subset of up,right,down,left selected "
            "deterministically per Episode; empty preserves the fixed direction."
        ),
    )
    group.add_argument(
        "--rule-writing-random-size-min-m",
        type=float,
        default=None,
        help="Optional lower bound for deterministic per-Episode writing size.",
    )
    group.add_argument(
        "--rule-writing-random-direction-deviation-deg",
        type=float,
        default=45.0,
        help="Maximum deterministic deviation around each selected cardinal direction (0..45 degrees).",
    )
    group.add_argument(
        "--rule-writing-random-size-max-m",
        type=float,
        default=None,
        help="Optional upper bound for deterministic per-Episode writing size.",
    )
    group.add_argument(
        "--rule-writing-reachability-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Before each Episode, certify the four centerline/bounding-box contact "
            "points with a command-inert copy of the production IK/collision solver."
        ),
    )
    group.add_argument(
        "--correct-tcp", "--rule-writing-correct-tcp",
        dest="rule_writing_correct_tcp",
        action="store_true",
        help=(
            "Use the measured right fixture tip at flange +Z 0.20 m for writing "
            "height; commands still target the existing 0.14 m tool_tcp. "
            "Default is unchanged."
        ),
    )
    group.add_argument(
        "--rule-writing-speed-m-s",
        type=float,
        default=None,
        help="Optional TCP path speed; default is the selected write-span average.",
    )
    group.add_argument(
        "--rule-initial-alignment-speed-scale",
        type=float,
        default=1.0,
        help=(
            "Initial dual-arm-to-source-start speed multiplier in (0, 1]. "
            "The certified RM65 joint/position limits remain the upper bound."
        ),
    )
    group.add_argument(
        "--rule-collect-tactile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether this rule-based run records DMTAC tactile data. On a "
            "populated dataset, the existing dataset modality contract takes "
            "precedence and prevents mixed tactile/non-tactile Episodes."
        ),
    )
    group.add_argument(
        "--rule-task-chain-foot-pad",
        action="store_true",
        default=False,
        help=(
            "Enable the dedicated two-pedal task-chain mapping: pedal 1 advances "
            "the next task-chain step and pedal 2 requests the existing rule "
            "stop-and-Hold path. This does not change ordinary teleop mappings."
        ),
    )
    group.add_argument(
        "--rule-task-chain-grasp-adjustment-interval",
        type=_positive_int,
        default=10,
        help=(
            "Successful rule Episodes per right-gripper adjustment. The first "
            "Episode always runs release, close, and height calibration; "
            "subsequent Episodes reuse that grasp until this positive interval "
            "is reached (default: 10)."
        ),
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
    hand_joint_targets: Mapping[str, np.ndarray],
    actual_qpos: Mapping[str, np.ndarray],
    actual_tcp: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    left_hold_pose: np.ndarray | None,
) -> ReplayTrajectoryGroup:
    selected_actions = np.asarray(actions[span.start : span.end], dtype=np.float32).copy()
    if left_hold_pose is not None:
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
        hand_joint_targets={
            side: np.asarray(values[span.start : span.end], dtype=np.float32).copy()
            for side, values in hand_joint_targets.items()
        },
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


def _freeze_group_arm_state(
    group: ReplayTrajectoryGroup,
    *,
    side: str,
    action_pose: np.ndarray,
    actual_qpos: np.ndarray,
    actual_tcp: np.ndarray,
) -> ReplayTrajectoryGroup:
    """Hold one arm throughout a replay group at one measured source state."""

    normalized_side = str(side)
    if normalized_side not in ACTION_ARM_POSE_SLICES:
        raise ValueError(f"unsupported held arm side: {normalized_side!r}.")
    pose = np.asarray(action_pose, dtype=np.float32).reshape(-1)
    qpos = np.asarray(actual_qpos, dtype=np.float32).reshape(-1)
    tcp = np.asarray(actual_tcp, dtype=np.float32).reshape(-1)
    if pose.shape != (7,) or tcp.shape != (7,) or qpos.shape != (6,):
        raise ValueError("held arm state must contain pose(7), qpos(6), and TCP(7).")
    if not all(np.all(np.isfinite(values)) for values in (pose, qpos, tcp)):
        raise ValueError("held arm state must be finite.")
    actions = np.asarray(group.actions, dtype=np.float32).copy()
    actions[:, ACTION_ARM_POSE_SLICES[normalized_side]] = pose
    qpos_by_side = {
        key: np.asarray(values, dtype=np.float32).copy()
        for key, values in group.arm_observation_actual_qpos.items()
    }
    tcp_by_side = {
        key: np.asarray(values, dtype=np.float32).copy()
        for key, values in group.arm_observation_actual_tcp_pose.items()
    }
    qpos_by_side[normalized_side] = np.repeat(
        qpos[None, :], int(group.frame_count), axis=0
    )
    tcp_by_side[normalized_side] = np.repeat(
        tcp[None, :], int(group.frame_count), axis=0
    )
    return replace(
        group,
        actions=actions,
        arm_observation_actual_qpos=qpos_by_side,
        arm_observation_actual_tcp_pose=tcp_by_side,
    )


def _dmtac_specs_by_side(sides: Sequence[str]) -> dict[str, HandActionSpec]:
    return {
        str(side): HandActionSpec.dmtac_gripper(str(side), gripper_slot=0)
        for side in sides
    }


def _with_runtime_dmtac_contract(
    group: ReplayTrajectoryGroup,
    *,
    sides: Sequence[str],
) -> ReplayTrajectoryGroup:
    """Resolve source hand targets, then normalize them to the real DMTAC slot.

    Source datasets may carry a typed partial/NaN hand action, a fully resolved
    ``action.replay`` target, or both.  The common replay resolver remains the
    only fallback owner.  This adapter then projects the resolved scalar onto
    the runtime DMTAC six-slot compatibility carrier (slot 0 active, the other
    slots finite zero padding).
    """

    normalized_sides = tuple(str(side) for side in sides)
    if not normalized_sides:
        return group
    metadata = replay_action_metadata(group, active_sides=normalized_sides)
    source_specs = hand_action_specs_from_metadata(metadata)
    targets_by_side: dict[str, np.ndarray] = {}
    for side in normalized_sides:
        rows: list[np.ndarray] = []
        source_spec = source_specs.get(side)
        if source_spec is not None and (
            source_spec.hand_model != "dmtac_gripper"
            or int(source_spec.active_dof) != 1
            or int(source_spec.slot_count) != 6
        ):
            raise ValueError(
                f"source {side} hand contract is not a one-DoF DMTAC gripper."
            )
        for index in range(int(group.frame_count)):
            resolved = replay_end_effector_targets_at_index(
                group,
                (side,),
                index=index,
            )[side]
            if source_spec is None:
                scalar = float(np.asarray(resolved, dtype=float).reshape(6)[0])
                padding = np.asarray(resolved, dtype=float).reshape(6)[1:]
                if not np.all(np.isfinite(padding)) or not np.allclose(
                    padding,
                    0.0,
                    atol=1e-6,
                ):
                    raise ValueError(
                        f"source {side} hand target lacks a DMTAC spec and has "
                        "non-zero legacy padding."
                    )
            else:
                scalar = float(source_spec.active_values(resolved)[0])
            if not np.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
                raise ValueError(
                    f"source {side} DMTAC target at frame {index} is outside [0, 1]."
                )
            target = np.zeros(6, dtype=np.float32)
            target[0] = scalar
            rows.append(target)
        targets_by_side[side] = np.stack(rows, axis=0)

    runtime_specs = _dmtac_specs_by_side(normalized_sides)
    spec_metadata = {
        side: spec.to_metadata() for side, spec in runtime_specs.items()
    }
    active_masks = {
        side: list(spec.active_mask or ()) for side, spec in runtime_specs.items()
    }
    group_metadata = dict(group.metadata)
    source_metadata = dict(group_metadata.get("source_metadata") or {})
    for parent_key in ("dataset_info", "episode_metadata"):
        parent = dict(source_metadata.get(parent_key) or {})
        parent_specs = dict(parent.get("hand_action_specs_by_side") or {})
        parent_specs.update(spec_metadata)
        parent["hand_action_specs_by_side"] = parent_specs
        parent_masks = dict(parent.get("hand_active_mask_by_side") or {})
        parent_masks.update(active_masks)
        parent["hand_active_mask_by_side"] = parent_masks
        source_metadata[parent_key] = parent
    group_metadata["source_metadata"] = source_metadata
    group_metadata["rule_based_hand_contract"] = {
        "mode": "runtime_dmtac_slot0",
        "hand_action_specs_by_side": spec_metadata,
    }
    actions = np.asarray(group.actions, dtype=np.float32).copy()
    hand_joint_targets = {
        side: np.asarray(values, dtype=np.float32).copy()
        for side, values in group.hand_joint_targets.items()
    }
    for side, values in targets_by_side.items():
        actions[:, ACTION_HAND_SLICES[side]] = values
        hand_joint_targets[side] = values.copy()
    return replace(
        group,
        actions=actions,
        hand_joint_targets=hand_joint_targets,
        metadata=group_metadata,
    )


def _rebase_hand_trajectory(
    values: np.ndarray,
    *,
    anchor: np.ndarray,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    anchor_row = np.asarray(anchor, dtype=np.float32).reshape(6)
    if source.ndim != 2 or source.shape[1] != 6 or source.shape[0] < 1:
        raise ValueError("hand trajectory must have finite shape (frames, 6).")
    rebased = anchor_row[None, :] + (source - source[0][None, :])
    if not np.all(np.isfinite(rebased)) or np.any(rebased < -1e-6) or np.any(
        rebased > 1.0 + 1e-6
    ):
        raise ValueError("rebased DMTAC hand trajectory leaves normalized [0, 1].")
    return np.clip(rebased, 0.0, 1.0).astype(np.float32)


def _replace_group_hand_trajectory(
    group: ReplayTrajectoryGroup,
    targets_by_side: Mapping[str, np.ndarray],
) -> ReplayTrajectoryGroup:
    actions = np.asarray(group.actions, dtype=np.float32).copy()
    hand_joint_targets = {
        side: np.asarray(values, dtype=np.float32).copy()
        for side, values in group.hand_joint_targets.items()
    }
    for side, raw in targets_by_side.items():
        values = np.asarray(raw, dtype=np.float32)
        if values.shape != (int(group.frame_count), 6) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                f"replacement {side} hand trajectory must match the replay group."
            )
        actions[:, ACTION_HAND_SLICES[side]] = values
        hand_joint_targets[side] = values.copy()
    return replace(
        group,
        actions=actions,
        hand_joint_targets=hand_joint_targets,
    )


def _fixed_dmtac_trajectory(*, frame_count: int, scalar: float) -> np.ndarray:
    """Return one finite six-slot DMTAC trajectory with a fixed active slot."""

    count = int(frame_count)
    value = float(scalar)
    if count < 1:
        raise ValueError("fixed DMTAC trajectory requires at least one frame.")
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("fixed DMTAC target must be inside normalized [0, 1].")
    target = np.zeros(6, dtype=np.float32)
    target[0] = value
    return np.repeat(target[None, :], count, axis=0)


def _resample_hand_trajectory(
    values: np.ndarray,
    *,
    source_times_sec: Sequence[float],
    target_times_sec: Sequence[float],
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    source_times = np.asarray(source_times_sec, dtype=float).reshape(-1)
    target_times = np.asarray(target_times_sec, dtype=float).reshape(-1)
    if (
        source.ndim != 2
        or source.shape[1] != 6
        or source.shape[0] != source_times.size
        or source_times.size < 1
        or target_times.size < 1
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(source_times))
        or not np.all(np.isfinite(target_times))
    ):
        raise ValueError("hand trajectory resampling inputs are invalid.")
    if source_times.size > 1 and (
        source_times[-1] <= source_times[0] or np.any(np.diff(source_times) <= 0.0)
    ):
        raise ValueError("source hand trajectory times must be strictly increasing.")
    if target_times.size > 1 and (
        target_times[-1] <= target_times[0] or np.any(np.diff(target_times) <= 0.0)
    ):
        raise ValueError("target hand trajectory times must be strictly increasing.")
    source_axis = (
        np.zeros(1, dtype=float)
        if source_times.size == 1
        else (source_times - source_times[0]) / (source_times[-1] - source_times[0])
    )
    target_axis = (
        np.zeros(1, dtype=float)
        if target_times.size == 1
        else (target_times - target_times[0]) / (target_times[-1] - target_times[0])
    )
    if source_times.size == 1:
        return np.repeat(source, target_times.size, axis=0).astype(np.float32)
    return np.stack(
        [np.interp(target_axis, source_axis, source[:, slot]) for slot in range(6)],
        axis=1,
    ).astype(np.float32)


def _validate_hand_step_limit(
    values: np.ndarray,
    *,
    side: str,
    phase: str,
    max_step: float,
) -> None:
    targets = np.asarray(values, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 6 or not np.all(np.isfinite(targets)):
        raise ValueError(f"{phase} {side} hand trajectory is invalid.")
    if targets.shape[0] > 1:
        observed = float(np.max(np.abs(np.diff(targets, axis=0))))
        if observed > float(max_step) + 1e-7:
            raise ValueError(
                f"{phase} {side} gripper step {observed:.6f} exceeds "
                f"the bounded maximum {float(max_step):.6f}."
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
        yaw_rad: float | None = None,
        left_hold_pose: np.ndarray,
        reference_hz: float,
        additional_catalog_path: Path | None = None,
        hand_trajectory_by_side: Mapping[str, np.ndarray] | None = None,
        hand_action_specs_by_side: Mapping[str, HandActionSpec] | None = None,
    ) -> None:
        self.trajectory_id = str(trajectory_id)
        self.scale_m = float(scale_m)
        self.speed_m_s = float(speed_m_s)
        self.direction = str(direction)
        self.yaw_rad = None if yaw_rad is None else float(yaw_rad)
        self.additional_catalog_path = additional_catalog_path
        self.left_hold_pose = np.asarray(left_hold_pose, dtype=float).reshape(7).copy()
        self.hand_trajectory_by_side = {
            str(side): np.asarray(values, dtype=np.float32).copy()
            for side, values in dict(hand_trajectory_by_side or {}).items()
        }
        if any(
            side not in {"left", "right"}
            or values.ndim != 2
            or values.shape[1] != 6
            or values.shape[0] < 1
            or not np.all(np.isfinite(values))
            for side, values in self.hand_trajectory_by_side.items()
        ):
            raise ValueError(
                "hand_trajectory_by_side must contain finite (frames, 6) targets."
            )
        self.hand_action_specs_by_side = {
            str(side): spec
            for side, spec in dict(hand_action_specs_by_side or {}).items()
        }
        if set(self.hand_action_specs_by_side) != set(self.hand_trajectory_by_side):
            raise ValueError(
                "fixed writing requires one HandActionSpec per hand trajectory."
            )
        for side, spec in self.hand_action_specs_by_side.items():
            if not isinstance(spec, HandActionSpec) or spec.side != side:
                raise ValueError("fixed writing HandActionSpec side mismatch.")
        self.reference_hz = float(reference_hz)
        self._targets: list[Any] | None = None
        self._index = 0
        self._height_calibration_metadata: dict[str, Any] = {}

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def reset(self) -> None:
        """Re-anchor the reusable plan from the next attempt's causal boundary."""

        self._targets = None
        self._index = 0
        self._height_calibration_metadata = {}

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
            writing_height = context.get("writing_height_target_z_m")
            height_calibration = context.get("writing_height_calibration")
            if isinstance(height_calibration, Mapping):
                self._height_calibration_metadata = dict(height_calibration)
            correct_tcp = bool(
                isinstance(height_calibration, Mapping)
                and height_calibration.get("contact_offset_tool_tcp_m", 0.0)
            )
            if writing_height is not None:
                writing_height = float(writing_height)
                if not np.isfinite(writing_height):
                    raise RuntimeError("fixed writing received a non-finite height target.")
                right_desk = np.asarray(right_desk, dtype=float).reshape(7).copy()
                right_desk[2] = (
                    tool_tcp_z_for_contact_height(right_desk, writing_height)
                    if correct_tcp else writing_height
                )
            else:
                if isinstance(height_calibration, Mapping) and bool(
                    height_calibration.get("enabled", False)
                ):
                    lower_bound = float(height_calibration["lower_bound_z_m"])
                    if not np.isfinite(lower_bound):
                        raise RuntimeError(
                            "fixed writing received a non-finite calibrated lower bound."
                        )
                    right_desk = np.asarray(right_desk, dtype=float).reshape(7).copy()
                    right_desk[2] += max(
                        0.0,
                        lower_bound - (
                            contact_height_from_tool_tcp(right_desk)
                            if correct_tcp else float(right_desk[2])
                        ),
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
                direction="up" if self.yaw_rad is not None else self.direction,
                yaw_rad=self.yaw_rad,
                additional_catalog_path=self.additional_catalog_path,
            )
            self._targets = resample_motion_targets(
                keyframes,
                reference_hz=self.reference_hz,
                include_final=True,
            )
            if any(
                values.shape[0] != len(self._targets)
                for values in self.hand_trajectory_by_side.values()
            ):
                raise RuntimeError(
                    "fixed writing arm and gripper trajectories have different lengths."
                )
        if self._index >= len(self._targets):
            raise RuntimeError("fixed writing source was consumed past its bounded path.")
        target = self._targets[self._index]
        hand_target = {
            side: values[self._index].copy()
            for side, values in self.hand_trajectory_by_side.items()
        }
        self._index += 1
        return RobotAction(
            action_space=BIMANUAL_EE_POSE_V1,
            timestamp=float(target.timestamp),
            arm_pose_by_side={
                side: pose_matrix_to_xyzquat(pose)
                for side, pose in target.pose_by_side.items()
            },
            hand_qpos_by_side=hand_target,
            frame_by_effector={
                "left_arm": CANONICAL_DESK_ACTION_FRAME,
                "right_arm": CANONICAL_DESK_ACTION_FRAME,
            },
            controlled_frame_by_effector={
                "left_arm": "tool_tcp",
                "right_arm": "tool_tcp",
            },
            source="rule_based_fixed_writing",
            metadata={
                **dict(target.metadata),
                **(
                    {"rule_based_height_calibration": self._height_calibration_metadata}
                    if self._height_calibration_metadata.get("contact_offset_tool_tcp_m")
                    else {}
                ),
                "rule_based_writing_trajectory_id": self.trajectory_id,
                "rule_based_writing_direction": self.direction,
                "rule_based_writing_size_m": self.scale_m,
                # Motion and dataset clocks are intentionally independent.
                # The runtime expands these source-cadence keyframes onto the
                # shared position-reference clock while retaining this
                # segment's cursor sequence as the dataset capture identity.
                "rule_based_position_reference_resample": True,
                "rule_based_position_keyframe_hz": self.reference_hz,
                "hand_action_specs_by_side": {
                    side: spec.to_metadata()
                    for side, spec in self.hand_action_specs_by_side.items()
                },
                "rule_based_hand_trajectory_mode": (
                    "source_write_time_normalized_relative_to_pen_down"
                    if hand_target
                    else "disabled"
                ),
            },
        )


def _height_descent_complete(context: Any) -> bool:
    return bool(
        context.action.metadata.get("rule_based_height_descent_terminal", False)
    )


class CalibratedHeightDescentActionSource:
    """Conditionally add one C2, pure-Z descent after pen-down replay.

    The source is command-inert when no height calibration is active or the
    measured right TCP is already inside/below the upper safety-band edge.
    Every emitted pose remains canonical-desk input to the existing coupled
    IK, TCP-envelope, collision, dispatch, and measured-arrival path.
    """

    name = "calibrated_height_descent"

    def __init__(
        self,
        *,
        reference_hz: float,
        linear_speed_m_s: float,
        hand_action_specs_by_side: Mapping[str, HandActionSpec] | None = None,
        max_acceleration_m_s2: float = 0.15,
        max_descent_m: float = 0.10,
    ) -> None:
        self.reference_hz = float(reference_hz)
        self.linear_speed_m_s = float(
            np.clip(float(linear_speed_m_s), 0.005, 0.05)
        )
        self.max_acceleration_m_s2 = float(max_acceleration_m_s2)
        self.max_descent_m = float(max_descent_m)
        if (
            not np.isfinite(self.reference_hz)
            or self.reference_hz <= 0.0
            or not np.isfinite(self.max_acceleration_m_s2)
            or self.max_acceleration_m_s2 <= 0.0
            or not np.isfinite(self.max_descent_m)
            or self.max_descent_m <= 0.0
        ):
            raise ValueError("height descent limits must be finite and positive.")
        self.hand_action_specs_by_side = {
            str(side): spec
            for side, spec in dict(hand_action_specs_by_side or {}).items()
        }
        self._actions: list[RobotAction] | None = None
        self._index = 0

    @property
    def max_steps(self) -> int:
        duration = max(
            1.875 * self.max_descent_m / self.linear_speed_m_s,
            math.sqrt(
                5.773503 * self.max_descent_m / self.max_acceleration_m_s2
            ),
        )
        return max(1, int(math.ceil(duration * self.reference_hz)))

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def reset(self) -> None:
        self._actions = None
        self._index = 0

    def produce_action(
        self,
        observation: Any,
        *,
        context: Any = None,
    ) -> RobotAction | None:
        del observation
        if self._actions is None:
            self._actions = self._build_actions(context)
        if not self._actions:
            return None
        if self._index >= len(self._actions):
            raise RuntimeError("height descent source exceeded its bounded trajectory.")
        action = self._actions[self._index]
        self._index += 1
        return action

    def _build_actions(self, context: Any) -> list[RobotAction]:
        if not isinstance(context, Mapping):
            raise RuntimeError("height descent requires the RealTeleop runtime context.")
        calibration = context.get("writing_height_calibration")
        if not isinstance(calibration, Mapping) or not bool(
            calibration.get("enabled", False)
        ):
            return []
        snapshots = context.get("actual_arm_state_by_side")
        contract = context.get("runtime_pose_frame_contract")
        if not isinstance(snapshots, Mapping) or contract is None:
            raise RuntimeError("height descent is missing causal TCP/frame context.")

        poses: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            snapshot = snapshots.get(side)
            if not isinstance(snapshot, Mapping):
                raise RuntimeError(f"height descent is missing {side} TCP feedback.")
            raw = pose_matrix_to_xyzquat(snapshot.get("actual_tcp_pose"))
            if raw is None:
                raise RuntimeError(f"height descent received invalid {side} TCP feedback.")
            poses[side] = np.asarray(
                contract.storage_pose_vector_to_canonical_desk(raw, side=side),
                dtype=float,
            ).reshape(7)
            if not np.all(np.isfinite(poses[side])):
                raise RuntimeError(f"height descent received non-finite {side} TCP feedback.")

        upper_z = float(calibration["upper_bound_z_m"])
        correct_tcp = bool(calibration.get("contact_offset_tool_tcp_m", 0.0))
        start_z = (
            contact_height_from_tool_tcp(poses["right"])
            if correct_tcp else float(poses["right"][2])
        )
        if not np.isfinite(upper_z):
            raise RuntimeError("height descent received a non-finite upper bound.")
        distance = start_z - upper_z
        if distance <= 0.0:
            return []
        if distance > self.max_descent_m + 1e-9:
            raise RuntimeError(
                "calibrated height descent exceeds its bounded distance: "
                f"distance={distance:.6f}m max={self.max_descent_m:.6f}m"
            )

        duration = max(
            1.875 * distance / self.linear_speed_m_s,
            math.sqrt(5.773503 * distance / self.max_acceleration_m_s2),
            1.0 / self.reference_hz,
        )
        step_count = max(1, int(math.ceil(duration * self.reference_hz)))
        actual_duration = step_count / self.reference_hz
        hand_targets: dict[str, np.ndarray] = {}
        for side, spec in self.hand_action_specs_by_side.items():
            snapshot = snapshots.get(side)
            values = np.asarray(
                snapshot.get("actual_hand_qpos") if isinstance(snapshot, Mapping) else None,
                dtype=float,
            ).reshape(-1)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"height descent requires fresh finite {side} gripper feedback."
                )
            if not isinstance(spec, HandActionSpec) or spec.side != side:
                raise RuntimeError("height descent HandActionSpec side mismatch.")
            hand_targets[side] = values.astype(np.float32, copy=True)

        calibration_metadata = {
            str(key): value
            for key, value in calibration.items()
            if isinstance(value, (bool, int, float, str)) or value is None
        }
        samples: list[TimestampedMotionTarget] = []
        for index in range(step_count + 1):
            alpha = index / step_count
            blend = 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5
            right = poses["right"].copy()
            desired_z = start_z - distance * blend
            right[2] = (
                tool_tcp_z_for_contact_height(right, desired_z)
                if correct_tcp else desired_z
            )
            samples.append(
                TimestampedMotionTarget(
                    timestamp=index / self.reference_hz,
                    available_timestamp=index / self.reference_hz,
                    source_index=index,
                    pose_by_side={
                        "left": xyzquat_to_pose_matrix(poses["left"]),
                        "right": xyzquat_to_pose_matrix(right),
                    },
                    metadata={"planner": "calibrated_pure_z_c2_quintic"},
                )
            )
        resampler = MotionTargetResampler(samples)
        actions: list[RobotAction] = []
        for index in range(1, step_count + 1):
            reference = resampler.resample_at(
                index / self.reference_hz,
                sequence=index - 1,
            )
            terminal = index == step_count
            actions.append(
                RobotAction(
                    action_space=BIMANUAL_EE_POSE_V1,
                    timestamp=float(reference.timestamp),
                    arm_pose_by_side={
                        side: pose_matrix_to_xyzquat(pose)
                        for side, pose in reference.pose_by_side.items()
                    },
                    hand_qpos_by_side={
                        side: values.copy() for side, values in hand_targets.items()
                    },
                    frame_by_effector={
                        "left_arm": CANONICAL_DESK_ACTION_FRAME,
                        "right_arm": CANONICAL_DESK_ACTION_FRAME,
                    },
                    controlled_frame_by_effector={
                        "left_arm": "tool_tcp",
                        "right_arm": "tool_tcp",
                    },
                    source="rule_based_calibrated_height_descent",
                    metadata={
                        **dict(reference.metadata),
                        "hand_action_specs_by_side": {
                            side: spec.to_metadata()
                            for side, spec in self.hand_action_specs_by_side.items()
                        },
                        "rule_based_height_calibration": calibration_metadata,
                        "rule_based_height_descent": True,
                        "rule_based_height_descent_start_z_m": start_z,
                        "rule_based_height_descent_target_z_m": upper_z,
                        "rule_based_height_descent_distance_m": distance,
                        "rule_based_height_descent_duration_sec": actual_duration,
                        "rule_based_height_descent_index": index,
                        "rule_based_height_descent_total": step_count,
                        "rule_based_height_descent_terminal": terminal,
                        "rule_based_height_arrival_tolerance_m": min(
                            0.001,
                            float(calibration["upper_tolerance_m"]),
                        ),
                        "rule_based_action_period_sec": 1.0 / self.reference_hz,
                        "position_pipeline": (
                            "calibrated_pure_z_c2_quintic_to_existing_"
                            "coupled_ik_command_stream"
                        ),
                        "position_reference_hz": self.reference_hz,
                        "position_interpolation": "quintic_smoothstep_c2",
                        "linear_speed_limit_m_s": self.linear_speed_m_s,
                        "linear_acceleration_limit_m_s2": self.max_acceleration_m_s2,
                    },
                )
            )
        return actions


def _parse_pool_range(value: str | None, *, candidate_count: int) -> tuple[int, int]:
    if value is None:
        return 0, int(candidate_count)
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        raise ValueError("rule source pool range must use START:END syntax.")
    try:
        start = 0 if not parts[0] else int(parts[0])
        end = int(candidate_count) if not parts[1] else int(parts[1])
    except ValueError as exc:
        raise ValueError("rule source pool range bounds must be integers.") from exc
    if start < 0 or end < 0 or start >= end or end > int(candidate_count):
        raise ValueError(
            "rule source pool range must be a non-empty half-open interval within "
            f"0:{candidate_count}; got {start}:{end}."
        )
    return start, end


def _stable_candidate_index(
    *,
    pool_hash: str,
    seed: int,
    phase: str,
    candidate_count: int,
    selection_ordinal: int = 0,
) -> int:
    digest = hashlib.sha256(
        f"{pool_hash}\0{int(seed)}\0{phase}\0{int(selection_ordinal)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % int(candidate_count)


def _comma_separated_unique(value: object, *, name: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in str(value or "").split(",") if part.strip())
    if len(items) != len(set(items)):
        raise ValueError(f"{name} must not contain duplicate values.")
    return items


def _stable_unit_interval(
    *,
    catalog_hash: str,
    seed: int,
    phase: str,
    selection_ordinal: int,
) -> float:
    digest = hashlib.sha256(
        f"{catalog_hash}\0{int(seed)}\0{phase}\0{int(selection_ordinal)}".encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)


def _resolve_writing_variant(
    args: argparse.Namespace,
    *,
    additional_catalog_path: Path | None,
) -> tuple[str, str, float, dict[str, Any]]:
    """Resolve one deterministic writing variant without process RNG state."""

    catalog_hash = fixed_trajectory_catalog_hash(
        additional_catalog_path=additional_catalog_path,
    )
    selection_ordinal = int(getattr(args, "_rule_source_selection_ordinal", 0))
    if selection_ordinal < 0:
        raise ValueError("rule source selection ordinal must be non-negative.")
    seed = int(getattr(args, "rule_source_selection_seed", 0))

    trajectory_ids = _comma_separated_unique(
        getattr(args, "rule_writing_random_trajectory_ids", ""),
        name="rule writing random trajectory ids",
    )
    available_ids = set(
        available_fixed_trajectory_ids(
            additional_catalog_path=additional_catalog_path,
        )
    )
    unknown_ids = tuple(value for value in trajectory_ids if value not in available_ids)
    if unknown_ids:
        raise ValueError(
            "rule writing random trajectory ids are absent from the composed "
            f"one-stroke catalog: {', '.join(unknown_ids)}"
        )
    trajectory_id = str(args.rule_writing_trajectory)
    if trajectory_ids:
        trajectory_id = trajectory_ids[
            _stable_candidate_index(
                pool_hash=catalog_hash,
                seed=seed,
                phase="writing_trajectory",
                candidate_count=len(trajectory_ids),
                selection_ordinal=selection_ordinal,
            )
        ]

    directions = _comma_separated_unique(
        getattr(args, "rule_writing_random_directions", ""),
        name="rule writing random directions",
    )
    invalid_directions = tuple(
        value for value in directions if value not in FIXED_TRAJECTORY_DIRECTIONS
    )
    if invalid_directions:
        raise ValueError(
            "rule writing random directions must be selected from "
            f"{','.join(FIXED_TRAJECTORY_DIRECTIONS)}: {','.join(invalid_directions)}"
        )
    direction = str(args.rule_writing_direction)
    if directions:
        direction = directions[
            _stable_candidate_index(
                pool_hash=catalog_hash,
                seed=seed,
                phase="writing_direction",
                candidate_count=len(directions),
                selection_ordinal=selection_ordinal,
            )
        ]

    minimum = getattr(args, "rule_writing_random_size_min_m", None)
    maximum = getattr(args, "rule_writing_random_size_max_m", None)
    if (minimum is None) != (maximum is None):
        raise ValueError(
            "rule writing random size requires both minimum and maximum bounds."
        )
    size_m = float(args.rule_writing_size_m)
    size_range: list[float] | None = None
    if minimum is not None and maximum is not None:
        lower = float(minimum)
        upper = float(maximum)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0:
            raise ValueError("rule writing random size bounds must be finite and positive.")
        if upper < lower:
            raise ValueError("rule writing random size maximum must be >= minimum.")
        fraction = _stable_unit_interval(
            catalog_hash=catalog_hash,
            seed=seed,
            phase="writing_size",
            selection_ordinal=selection_ordinal,
        )
        size_m = lower + (upper - lower) * fraction
        size_range = [lower, upper]

    max_deviation_deg = float(
        getattr(args, "rule_writing_random_direction_deviation_deg", 45.0)
    )
    if not np.isfinite(max_deviation_deg) or not 0.0 <= max_deviation_deg <= 45.0:
        raise ValueError("random writing direction deviation must be in [0, 45] degrees.")
    deviation_deg = (
        (2.0 * _stable_unit_interval(
            catalog_hash=catalog_hash,
            seed=seed,
            phase="writing_direction_deviation",
            selection_ordinal=selection_ordinal,
        ) - 1.0) * max_deviation_deg
        if directions else 0.0
    )
    yaw_rad = fixed_trajectory_direction_yaw_rad(direction) + math.radians(deviation_deg)
    enabled = bool(trajectory_ids or directions or size_range is not None)
    return trajectory_id, direction, size_m, {
        "schema": "dexteleop.rule_based_writing_variant_selection.v1",
        "enabled": enabled,
        "selection_seed": seed if enabled else None,
        "selection_ordinal": selection_ordinal,
        "trajectory_candidates": list(trajectory_ids),
        "direction_candidates": list(directions),
        "size_range_m": size_range,
        "selected_trajectory_id": trajectory_id,
        "selected_direction": direction,
        "selected_direction_deviation_deg": deviation_deg,
        "selected_yaw_rad": yaw_rad,
        "selected_size_m": size_m,
    }


def _writing_reachability_contact_probes(
    targets: Sequence[TimestampedMotionTarget],
) -> list[dict[str, Any]]:
    """Return the four centerline contacts of a shrinking axis-aligned box."""

    right_poses = [
        np.asarray(target.pose_by_side["right"], dtype=float).reshape(4, 4)
        for target in targets
    ]
    positions = np.stack([pose[:3, 3] for pose in right_poses], axis=0)
    center_xy = 0.5 * (
        np.min(positions[:, :2], axis=0) + np.max(positions[:, :2], axis=0)
    )
    edge_specs = (
        ("left", 0, np.argmin, 1),
        ("right", 0, np.argmax, 1),
        ("bottom", 1, np.argmin, 0),
        ("top", 1, np.argmax, 0),
    )
    probes: list[dict[str, Any]] = []
    for name, axis, reducer, orthogonal_axis in edge_specs:
        extreme_index = int(reducer(positions[:, axis]))
        coordinate = float(positions[extreme_index, axis])
        edge_indices = np.flatnonzero(
            np.isclose(positions[:, axis], coordinate, rtol=0.0, atol=1e-12)
        )
        index = int(
            edge_indices[
                np.argmin(
                    np.abs(
                        positions[edge_indices, orthogonal_axis]
                        - center_xy[orthogonal_axis]
                    )
                )
            ]
        )
        probes.append(
            {
                "name": name,
                "target_index": index,
                "right_pose_canonical_desk": pose_matrix_to_xyzquat(
                    right_poses[index]
                ).tolist(),
            }
        )
    return probes


def _relative_pen_up_reachability_probes(
    group: ReplayTrajectoryGroup,
    *,
    writing_end: TimestampedMotionTarget,
) -> list[dict[str, Any]]:
    """Sample the actual relative replay formula at its temporal midpoint and end.

    The writing endpoint is a planned boundary. Live replay still rebases on
    measured TCP and passes the normal runtime safety/IK gates.
    """

    times = np.asarray(
        [group.relative_time_sec(index) for index in range(group.frame_count)],
        dtype=float,
    )
    middle_time = 0.5 * (float(times[0]) + float(times[-1]))
    middle_index = int(np.argmin(np.abs(times - middle_time)))
    boundary = RuleBasedBoundaryState(
        tcp_pose_by_side={"right": writing_end.pose_by_side["right"]},
        pose_frame_by_side={"right": CANONICAL_DESK_ACTION_FRAME},
        source="planned_writing_endpoint_preflight",
    )
    probes = []
    for name, index in (("pen_up_mid", middle_index), ("pen_up_end", group.frame_count - 1)):
        action = rebase_replay_action_at_boundary(
            group, ("right",), index=index, boundary=boundary,
            source_start_index=0,
        )
        probes.append({
            "name": name,
            "target_index": index,
            "right_pose_canonical_desk": np.asarray(
                action.arm_pose_by_side["right"], dtype=float
            ).reshape(7).tolist(),
            "writing_end_pose_canonical_desk": pose_matrix_to_xyzquat(
                writing_end.pose_by_side["right"]
            ).tolist(),
        })
    return probes


def _resolve_source_specs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, ResolvedRuleBasedWritingSourceCandidate],
    dict[str, Any],
]:
    manifest = getattr(args, "rule_source_pool_manifest", None)
    legacy_values = (
        getattr(args, "rule_source_dataset", None),
        getattr(args, "rule_source_episode", None),
        getattr(args, "rule_detail_annotation_dir", None),
    )
    if manifest is not None and any(value is not None for value in legacy_values):
        raise ValueError(
            "--rule-source-pool-manifest is mutually exclusive with the legacy "
            "single-source arguments."
        )
    if manifest is None:
        if any(value is None for value in legacy_values):
            raise ValueError(
                "provide either --rule-source-pool-manifest or all of "
                "--rule-source-dataset, --rule-source-episode, and "
                "--rule-detail-annotation-dir."
            )
        dataset_root = Path(str(legacy_values[0])).expanduser().resolve()
        detail_dir = Path(str(legacy_values[2])).expanduser().resolve()
        candidate = RuleBasedWritingSourceCandidateV1(
            candidate_id="legacy-single-source",
            dataset_root=str(dataset_root),
            episode_index=int(legacy_values[1]),
            detail_annotation_dir=str(detail_dir),
        )
        resolved = ResolvedRuleBasedWritingSourceCandidate(
            candidate=candidate,
            dataset_root=dataset_root,
            detail_annotation_dir=detail_dir,
        )
        return (
            {phase: resolved for phase in RULE_BASED_WRITING_SOURCE_PHASES_V1},
            {
                "schema": "dexteleop.rule_based_writing_source_selection.v1",
                "mode": "legacy_single_source",
                "pool_id": None,
                "pool_manifest_hash": None,
                "pool_range": None,
                "selection_seed": None,
            },
        )

    loaded_pool = _load_rule_source_pool_cached(args, manifest)
    selection_ordinal = int(getattr(args, "_rule_source_selection_ordinal", 0))
    if selection_ordinal < 0:
        raise ValueError("rule source selection ordinal must be non-negative.")
    start, end = _parse_pool_range(
        getattr(args, "rule_source_pool_range", None),
        candidate_count=len(loaded_pool.candidates),
    )
    ranged = loaded_pool.candidates[start:end]
    requested_ids = getattr(args, "_rule_source_candidate_ids_by_phase", None)
    if requested_ids is not None:
        if not isinstance(requested_ids, Mapping):
            raise ValueError("rule source candidate overrides must be a mapping.")
        normalized_requested_ids = {
            str(phase): value for phase, value in requested_ids.items()
        }
        unknown_phases = set(normalized_requested_ids) - set(
            RULE_BASED_WRITING_SOURCE_PHASES_V1
        )
        if unknown_phases:
            raise ValueError(
                "rule source candidate overrides contain unknown phases: "
                f"{sorted(unknown_phases)!r}."
            )
        by_id = {value.candidate.candidate_id: value for value in ranged}
        selected: dict[str, ResolvedRuleBasedWritingSourceCandidate] = {}
        used: set[str] = set()
        random_phases: list[str] = []
        for phase in RULE_BASED_WRITING_SOURCE_PHASES_V1:
            raw_candidate_id = normalized_requested_ids.get(phase)
            candidate_id = (
                ""
                if raw_candidate_id is None
                else str(raw_candidate_id).strip()
            )
            if not candidate_id or candidate_id == RULE_BASED_RANDOM_SOURCE_SELECTION_ID:
                random_phases.append(phase)
                continue
            spec = by_id.get(candidate_id)
            if spec is None or phase not in spec.candidate.phases:
                raise ValueError(
                    f"candidate {candidate_id!r} is not eligible for phase {phase!r} "
                    "inside the selected source-pool range."
                )
            selected[phase] = spec
            used.add(candidate_id)
        for phase in random_phases:
            eligible = [value for value in ranged if phase in value.candidate.phases]
            if not eligible:
                raise ValueError(
                    f"selected source-pool range has no candidate for phase {phase!r}."
                )
            unused = [
                value
                for value in eligible
                if value.candidate.candidate_id not in used
            ]
            choices = unused or eligible
            index = _stable_candidate_index(
                pool_hash=loaded_pool.contract.semantic_content_hash,
                seed=int(getattr(args, "rule_source_selection_seed", 0)),
                phase=phase,
                candidate_count=len(choices),
                selection_ordinal=selection_ordinal,
            )
            selected[phase] = choices[index]
            used.add(choices[index].candidate.candidate_id)
        selection_seed = (
            None
            if not random_phases
            else int(getattr(args, "rule_source_selection_seed", 0))
        )
        return (
            selected,
            {
                "schema": "dexteleop.rule_based_writing_source_selection.v1",
                "mode": "source_pool_gui_selection",
                "pool_id": loaded_pool.contract.pool_id,
                "pool_manifest_path": str(loaded_pool.manifest_path),
                "pool_manifest_hash": loaded_pool.contract.semantic_content_hash,
                "pool_range": {"start": start, "end": end},
                "selection_seed": selection_seed,
                "selection_ordinal": selection_ordinal,
            },
        )
    selected: dict[str, ResolvedRuleBasedWritingSourceCandidate] = {}
    used: set[str] = set()
    for phase in RULE_BASED_WRITING_SOURCE_PHASES_V1:
        eligible = [value for value in ranged if phase in value.candidate.phases]
        if not eligible:
            raise ValueError(
                f"selected source-pool range has no candidate for phase {phase!r}."
            )
        unused = [value for value in eligible if value.candidate.candidate_id not in used]
        choices = unused or eligible
        index = _stable_candidate_index(
            pool_hash=loaded_pool.contract.semantic_content_hash,
            seed=int(getattr(args, "rule_source_selection_seed", 0)),
            phase=phase,
            candidate_count=len(choices),
            selection_ordinal=selection_ordinal,
        )
        selected[phase] = choices[index]
        used.add(choices[index].candidate.candidate_id)
    return (
        selected,
        {
            "schema": "dexteleop.rule_based_writing_source_selection.v1",
            "mode": "source_pool",
            "pool_id": loaded_pool.contract.pool_id,
            "pool_manifest_path": str(loaded_pool.manifest_path),
            "pool_manifest_hash": loaded_pool.contract.semantic_content_hash,
            "pool_range": {"start": start, "end": end},
            "selection_seed": int(getattr(args, "rule_source_selection_seed", 0)),
            "selection_ordinal": selection_ordinal,
        },
    )


def _source_pool_candidates(
    args: argparse.Namespace,
) -> tuple[ResolvedRuleBasedWritingSourceCandidate, ...]:
    manifest = getattr(args, "rule_source_pool_manifest", None)
    if manifest is None:
        dataset = getattr(args, "rule_source_dataset", None)
        episode = getattr(args, "rule_source_episode", None)
        if dataset is None or episode is None:
            raise ValueError(
                "complete-Episode replay requires --rule-source-pool-manifest or "
                "both --rule-source-dataset and --rule-source-episode."
            )
        dataset_root = Path(str(dataset)).expanduser().resolve()
        detail_value = getattr(args, "rule_detail_annotation_dir", None)
        detail_dir = (
            dataset_root
            if detail_value is None
            else Path(str(detail_value)).expanduser().resolve()
        )
        candidate = RuleBasedWritingSourceCandidateV1(
            candidate_id="legacy-single-source",
            dataset_root=str(dataset_root),
            episode_index=int(episode),
            detail_annotation_dir=str(detail_dir),
        )
        return (
            ResolvedRuleBasedWritingSourceCandidate(
                candidate=candidate,
                dataset_root=dataset_root,
                detail_annotation_dir=detail_dir,
            ),
        )
    if any(
        getattr(args, key, None) is not None
        for key in (
            "rule_source_dataset",
            "rule_source_episode",
            "rule_detail_annotation_dir",
        )
    ):
        raise ValueError(
            "--rule-source-pool-manifest is mutually exclusive with the legacy "
            "single-source arguments."
        )
    loaded_pool = _load_rule_source_pool_cached(args, manifest)
    start, end = _parse_pool_range(
        getattr(args, "rule_source_pool_range", None),
        candidate_count=len(loaded_pool.candidates),
    )
    return tuple(loaded_pool.candidates[start:end])


def _resolve_full_replay_source_spec(
    args: argparse.Namespace,
) -> tuple[ResolvedRuleBasedWritingSourceCandidate, dict[str, Any]]:
    candidates = _source_pool_candidates(args)
    manifest = getattr(args, "rule_source_pool_manifest", None)
    requested = getattr(args, "_rule_full_replay_candidate_id", None)
    if requested is None:
        requested = getattr(args, "rule_full_replay_candidate_id", None)
    if requested is not None:
        requested_id = str(requested)
        by_id = {spec.candidate.candidate_id: spec for spec in candidates}
        try:
            selected = by_id[requested_id]
        except KeyError as exc:
            raise ValueError(
                f"complete-replay candidate {requested_id!r} is outside the selected "
                "source-pool range."
            ) from exc
        selection_mode = "source_pool_gui_selection"
    elif manifest is None:
        selected = candidates[0]
        selection_mode = "legacy_single_source"
    else:
        loaded_pool = _load_rule_source_pool_cached(args, manifest)
        index = _stable_candidate_index(
            pool_hash=loaded_pool.contract.semantic_content_hash,
            seed=int(getattr(args, "rule_source_selection_seed", 0)),
            phase=RULE_COLLECTION_MODE_FULL_REPLAY,
            candidate_count=len(candidates),
        )
        selected = candidates[index]
        selection_mode = "source_pool"
    if manifest is None:
        pool_metadata: dict[str, Any] = {
            "pool_id": None,
            "pool_manifest_hash": None,
            "pool_range": None,
            "selection_seed": None,
        }
    else:
        loaded_pool = _load_rule_source_pool_cached(args, manifest)
        start, end = _parse_pool_range(
            getattr(args, "rule_source_pool_range", None),
            candidate_count=len(loaded_pool.candidates),
        )
        pool_metadata = {
            "pool_id": loaded_pool.contract.pool_id,
            "pool_manifest_path": str(loaded_pool.manifest_path),
            "pool_manifest_hash": loaded_pool.contract.semantic_content_hash,
            "pool_range": {"start": start, "end": end},
            "selection_seed": (
                None
                if selection_mode == "source_pool_gui_selection"
                else int(getattr(args, "rule_source_selection_seed", 0))
            ),
        }
    return (
        selected,
        {
            "schema": "dexteleop.full_trajectory_replay_source_selection.v1",
            "collection_mode": RULE_COLLECTION_MODE_FULL_REPLAY,
            "mode": selection_mode,
            **pool_metadata,
        },
    )


def _requested_pre_pen_down_replay_candidate_id(args: argparse.Namespace) -> str:
    requested = getattr(args, "_rule_pre_pen_down_replay_candidate_id", None)
    if requested is None:
        requested = getattr(args, "rule_pre_pen_down_replay_candidate_id", None)
    return "" if requested is None else str(requested).strip()


def _pre_pen_down_replay_normal_episodes(args: argparse.Namespace) -> int:
    value = getattr(args, "rule_pre_pen_down_replay_normal_episodes", 0)
    if isinstance(value, bool):
        raise ValueError(
            "rule pre-pen-down replay normal Episodes must be a non-negative integer."
        )
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rule pre-pen-down replay normal Episodes must be a non-negative integer."
        ) from exc
    if normalized < 0 or normalized != value:
        raise ValueError(
            "rule pre-pen-down replay normal Episodes must be a non-negative integer."
        )
    return normalized


def _resolve_pre_pen_down_replay_source_spec(
    args: argparse.Namespace,
) -> ResolvedRuleBasedWritingSourceCandidate | None:
    requested_id = _requested_pre_pen_down_replay_candidate_id(args)
    if not requested_id:
        return None
    if getattr(args, "rule_source_pool_manifest", None) is None:
        raise ValueError(
            "pre-pen-down replay requires --rule-source-pool-manifest; legacy "
            "single-source mode has no exact candidate identity."
        )
    candidates = _source_pool_candidates(args)
    by_id = {spec.candidate.candidate_id: spec for spec in candidates}
    try:
        return by_id[requested_id]
    except KeyError as exc:
        raise ValueError(
            f"pre-pen-down replay candidate {requested_id!r} is outside the "
            "selected source-pool range."
        ) from exc


def _configuration_option_entry(
    spec: ResolvedRuleBasedWritingSourceCandidate,
) -> dict[str, str]:
    candidate = spec.candidate
    dataset_label = spec.dataset_root.name or str(spec.dataset_root)
    tags = f" · {','.join(candidate.tags)}" if candidate.tags else ""
    return {
        "value": candidate.candidate_id,
        "label": (
            f"{candidate.candidate_id} · {dataset_label} · "
            f"episode {candidate.episode_index}{tags}"
        ),
    }


def _additional_fixed_trajectory_catalog_path(
    args: argparse.Namespace,
) -> Path | None:
    value = getattr(args, "rule_writing_trajectory_catalog", None)
    if value is None or not str(value).strip():
        return None
    candidate = Path(os.path.expandvars(str(value))).expanduser()
    if not candidate.is_absolute():
        raise ValueError("rule writing trajectory catalog path must be absolute.")
    if candidate.is_symlink():
        raise ValueError("rule writing trajectory catalog must not be a symlink.")
    if not candidate.is_file():
        raise ValueError(
            f"rule writing trajectory catalog does not exist: {candidate}"
        )
    resolved = candidate.resolve(strict=True)

    # Validate the complete overlay, including conflicts with the checked-in
    # catalog, before any option is exposed to the real collection UI.
    available_fixed_trajectory_ids(additional_catalog_path=resolved)
    for trajectory_id in available_fixed_trajectory_ids(catalog_path=resolved):
        spec = get_fixed_trajectory(trajectory_id, catalog_path=resolved)
        if spec.stroke_count != 1:
            raise ValueError(
                f"additional fixed trajectory {trajectory_id!r} is not a one-stroke path"
            )
    return resolved


def _fixed_trajectory_option(
    trajectory_id: str,
    *,
    additional_catalog_path: Path | None = None,
) -> dict[str, str]:
    spec = get_fixed_trajectory(
        trajectory_id,
        additional_catalog_path=additional_catalog_path,
    )
    if spec.family == "digit":
        label = f"数字 {spec.symbol}"
    elif spec.family == "latin":
        label = f"英文字母 {spec.symbol}"
    elif spec.family == "greek":
        latin_name = str(spec.metadata.get("latin_name", "")).strip()
        suffix = f" · {latin_name}" if latin_name else ""
        label = f"希腊字母 {spec.symbol}{suffix}"
    else:
        label = f"{spec.symbol} · {spec.trajectory_id}"
    return {"value": spec.trajectory_id, "label": label}


def _rule_configuration_options(
    args: argparse.Namespace,
    *,
    source_specs: Mapping[str, ResolvedRuleBasedWritingSourceCandidate],
    resolved_writing_speed_m_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tactile_resolution = _resolve_rule_tactile_capture(args)
    additional_catalog_path = _additional_fixed_trajectory_catalog_path(args)
    manifest = getattr(args, "rule_source_pool_manifest", None)
    if manifest is None:
        candidates = tuple({spec.candidate.candidate_id: spec for spec in source_specs.values()}.values())
    else:
        loaded_pool = _load_rule_source_pool_cached(args, manifest)
        start, end = _parse_pool_range(
            getattr(args, "rule_source_pool_range", None),
            candidate_count=len(loaded_pool.candidates),
        )
        candidates = loaded_pool.candidates[start:end]
    source_options = {
        phase: (
            (
                [
                    {
                        "value": RULE_BASED_RANDOM_SOURCE_SELECTION_ID,
                        "label": "随机选择（默认）",
                    }
                ]
                if manifest is not None
                else []
            )
            + [
                _configuration_option_entry(spec)
                for spec in candidates
                if phase in spec.candidate.phases
            ]
        )
        for phase in RULE_BASED_WRITING_SOURCE_PHASES_V1
    }
    tactile_bundle_root = (
        str(getattr(args, "tactile_bundle_root", "") or "").strip()
        if tactile_resolution.collect_tactile
        else ""
    )
    tactile_bundle_free_gb: float | None = None
    if tactile_bundle_root:
        try:
            tactile_bundle_free_gb = float(
                shutil.disk_usage(
                    Path(os.path.expandvars(tactile_bundle_root)).expanduser()
                ).free
            ) / float(1024**3)
        except (OSError, ValueError):
            tactile_bundle_free_gb = None
    current_min_free_gb = float(
        getattr(args, "tactile_bundle_min_free_gb", 0.0) or 0.0
    )
    current_writer_workers = int(getattr(args, "writer_workers", 4) or 4)
    current_writer_queue_size = int(getattr(args, "writer_queue_size", 512) or 512)
    current_process_queue_size = int(
        getattr(args, "collection_process_queue_size", 4) or 4
    )
    current_storage_queue_size = int(getattr(args, "storage_queue_size", 64) or 0)
    current_storage_queue_enqueue_timeout_sec = float(
        getattr(
            args,
            "storage_queue_enqueue_timeout_sec",
            DEFAULT_COLLECTION_STORAGE_QUEUE_ENQUEUE_TIMEOUT_SEC,
        )
        or DEFAULT_COLLECTION_STORAGE_QUEUE_ENQUEUE_TIMEOUT_SEC
    )
    current_camera_frame_max_age_sec = float(
        getattr(
            args,
            "camera_frame_max_age_sec",
            DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
        )
        or DEFAULT_CAMERA_FRAME_MAX_AGE_SEC
    )
    current_warning_hz = float(
        getattr(args, "collection_warning_hz", 15.0) or 15.0
    )
    current_flush_timeout_sec = float(
        getattr(
            args,
            "collection_flush_timeout_sec",
            DEFAULT_COLLECTION_FLUSH_TIMEOUT_SEC,
        )
        or DEFAULT_COLLECTION_FLUSH_TIMEOUT_SEC
    )
    current_sample_drain_timeout_sec = float(
        getattr(
            args,
            "collection_sample_drain_timeout_sec",
            DEFAULT_COLLECTION_SAMPLE_DRAIN_TIMEOUT_SEC,
        )
        or DEFAULT_COLLECTION_SAMPLE_DRAIN_TIMEOUT_SEC
    )
    current_terminal_event_put_timeout_sec = float(
        getattr(
            args,
            "collection_terminal_event_put_timeout_sec",
            DEFAULT_COLLECTION_TERMINAL_EVENT_PUT_TIMEOUT_SEC,
        )
        or DEFAULT_COLLECTION_TERMINAL_EVENT_PUT_TIMEOUT_SEC
    )
    options = {
        "configuration_core_fields": [
            "pre_pen_down_replay_candidate_id",
            "pre_pen_down_replay_normal_episodes",
            "pen_down_candidate_id",
            "write_candidate_id",
            "pen_up_candidate_id",
            "trajectory_id",
            "direction",
            "writing_size_m",
            "initial_alignment_speed_scale",
        ],
        "choice_fields": {
            "pre_pen_down_replay_candidate_id": "pre_pen_down_replay_candidates",
        },
        "pre_pen_down_replay_candidates": [
            {"value": "", "label": "不启用（保持当前流程）"},
            *(
                [_configuration_option_entry(spec) for spec in candidates]
                if manifest is not None
                else []
            ),
        ],
        "source_candidates_by_phase": source_options,
        "trajectory_ids": [
            _fixed_trajectory_option(
                value,
                additional_catalog_path=additional_catalog_path,
            )
            for value in available_fixed_trajectory_ids(
                additional_catalog_path=additional_catalog_path,
            )
        ],
        "directions": [
            {"value": value, "label": {"up": "向上", "right": "向右", "down": "向下", "left": "向左"}[value]}
            for value in FIXED_TRAJECTORY_DIRECTIONS
        ],
        "runtime_configuration_fields": list(
            _RULE_GUI_RUNTIME_CONFIGURATION_FIELDS
        ),
        "integer_fields": [
            "pre_pen_down_replay_normal_episodes",
            "writer_workers",
            "writer_queue_size",
            "collection_process_queue_size",
            "storage_queue_size",
        ],
        "select_fields": {
            "tactile_compression": ["none", "lz4", "zstd"],
        },
        "boolean_fields": [
            "writing_speed_from_source",
            "collect_tactile",
            "write_parquet",
            "write_video",
        ],
        "text_fields": ["tactile_bundle_root"],
        "numeric_bounds": {
            "writing_size_m": [
                min(0.005, float(args.rule_writing_size_m)),
                max(0.2, float(args.rule_writing_size_m)),
            ],
            "initial_alignment_speed_scale": [
                min(
                    0.05,
                    float(getattr(args, "rule_initial_alignment_speed_scale", 1.0)),
                ),
                1.0,
            ],
            "pre_pen_down_replay_normal_episodes": [0, 10000],
            "writing_speed_m_s": [
                min(0.001, float(resolved_writing_speed_m_s)),
                max(0.2, float(resolved_writing_speed_m_s)),
            ],
            "tactile_bundle_min_free_gb": [
                0.0,
                max(4096.0, current_min_free_gb),
            ],
            "writer_workers": [1, max(32, current_writer_workers)],
            "writer_queue_size": [1, max(65536, current_writer_queue_size)],
            "collection_process_queue_size": [
                1,
                max(1024, current_process_queue_size),
            ],
            "storage_queue_size": [0, max(65536, current_storage_queue_size)],
            "storage_queue_enqueue_timeout_sec": [
                0.01,
                max(60.0, current_storage_queue_enqueue_timeout_sec),
            ],
            "camera_frame_max_age_sec": [
                0.05,
                max(0.5, current_camera_frame_max_age_sec),
            ],
            "collection_warning_hz": [
                0.1,
                max(120.0, current_warning_hz),
            ],
            "collection_flush_timeout_sec": [
                0.1,
                max(600.0, current_flush_timeout_sec),
            ],
            "collection_sample_drain_timeout_sec": [
                0.1,
                max(600.0, current_sample_drain_timeout_sec),
            ],
            "collection_terminal_event_put_timeout_sec": [
                0.01,
                max(30.0, current_terminal_event_put_timeout_sec),
            ],
        },
        "runtime_info": {
            "dataset_root": str(getattr(args, "dataset_root", "") or ""),
            "tactile_bundle_free_gb": tactile_bundle_free_gb,
            "tactile_capture_mode": tactile_resolution.mode,
            "tactile_capture_source": tactile_resolution.source,
            "tactile_capture_locked": bool(tactile_resolution.locked),
            "sample_hz": float(getattr(args, "hz", 15.0)),
            "position_reference_hz": effective_position_reference_hz(args),
            "outer_control_hz": effective_outer_control_hz(args),
            "arm_control_hz": effective_arm_control_hz(args),
            "task_chain_grasp_adjustment_interval": int(
                getattr(args, "rule_task_chain_grasp_adjustment_interval", 10)
            ),
        },
    }
    selected = {
        f"{phase}_candidate_id": (
            RULE_BASED_RANDOM_SOURCE_SELECTION_ID
            if manifest is not None
            else source_specs[phase].candidate.candidate_id
        )
        for phase in RULE_BASED_WRITING_SOURCE_PHASES_V1
    }
    selected.update(
        {
            "pre_pen_down_replay_candidate_id": (
                _requested_pre_pen_down_replay_candidate_id(args)
            ),
            "pre_pen_down_replay_normal_episodes": (
                _pre_pen_down_replay_normal_episodes(args)
            ),
            "trajectory_id": str(args.rule_writing_trajectory),
            "direction": str(args.rule_writing_direction),
            "writing_size_m": float(args.rule_writing_size_m),
            "writing_speed_m_s": float(resolved_writing_speed_m_s),
            "writing_speed_from_source": bool(
                getattr(args, "rule_writing_speed_m_s", None) is None
            ),
            "initial_alignment_speed_scale": float(
                getattr(args, "rule_initial_alignment_speed_scale", 1.0)
            ),
            "collect_tactile": bool(tactile_resolution.collect_tactile),
            "tactile_bundle_root": tactile_bundle_root,
            "tactile_bundle_min_free_gb": current_min_free_gb,
            "tactile_compression": str(
                getattr(args, "tactile_compression", "none") or "none"
            ),
            "writer_workers": current_writer_workers,
            "writer_queue_size": current_writer_queue_size,
            "collection_process_queue_size": current_process_queue_size,
            "storage_queue_size": current_storage_queue_size,
            "storage_queue_enqueue_timeout_sec": current_storage_queue_enqueue_timeout_sec,
            "camera_frame_max_age_sec": current_camera_frame_max_age_sec,
            "collection_warning_hz": current_warning_hz,
            "collection_flush_timeout_sec": current_flush_timeout_sec,
            "collection_sample_drain_timeout_sec": current_sample_drain_timeout_sec,
            "collection_terminal_event_put_timeout_sec": current_terminal_event_put_timeout_sec,
            "write_parquet": bool(getattr(args, "write_parquet", True)),
            "write_video": bool(getattr(args, "write_video", True)),
        }
    )
    return options, selected


def _full_replay_configuration_options(
    args: argparse.Namespace,
    *,
    selected_spec: ResolvedRuleBasedWritingSourceCandidate,
    source_specs: Sequence[ResolvedRuleBasedWritingSourceCandidate],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Reuse the existing writer/storage option construction, then expose only
    # the two motion choices that apply to a complete-Episode replay.  This
    # keeps the formal writer contract byte-for-byte aligned with the writing
    # profile without presenting irrelevant glyph/substage controls.
    options, writing_selected = _rule_configuration_options(
        args,
        source_specs={
            phase: selected_spec for phase in RULE_BASED_WRITING_SOURCE_PHASES_V1
        },
        resolved_writing_speed_m_s=max(
            0.001,
            float(getattr(args, "rule_writing_speed_m_s", 0.001) or 0.001),
        ),
    )
    options.update(
        {
            "collection_mode": RULE_COLLECTION_MODE_FULL_REPLAY,
            "configuration_core_fields": [
                "full_replay_candidate_id",
                "initial_alignment_speed_scale",
            ],
            "choice_fields": {
                "full_replay_candidate_id": "full_replay_candidates",
            },
            "full_replay_candidates": [
                _configuration_option_entry(spec) for spec in source_specs
            ],
            "runtime_configuration_fields": list(
                _RULE_GUI_SHARED_RUNTIME_CONFIGURATION_FIELDS
            ),
        }
    )
    selected = {
        "full_replay_candidate_id": selected_spec.candidate.candidate_id,
        "initial_alignment_speed_scale": float(
            getattr(args, "rule_initial_alignment_speed_scale", 1.0)
        ),
    }
    selected.update(
        {
            key: writing_selected[key]
            for key in _RULE_GUI_SHARED_RUNTIME_CONFIGURATION_FIELDS
        }
    )
    return options, selected


def _apply_rule_gui_runtime_configuration(
    target: argparse.Namespace,
    configuration: Mapping[str, Any],
) -> dict[str, object]:
    """Validate and stage GUI-editable rule/writer args without opening devices."""

    def configured_float(name: str, default: float) -> float:
        value = configuration.get(name, getattr(target, name, default))
        return float(value)

    requested_tactile = configuration.get(
        "collect_tactile",
        getattr(target, "rule_collect_tactile", None),
    )
    tactile_resolution = _resolve_rule_tactile_capture(
        target,
        requested=(None if requested_tactile is None else bool(requested_tactile)),
    )
    patch: dict[str, object] = {
        "collect_tactile": bool(tactile_resolution.collect_tactile),
        "ignore_tactile": list(tactile_resolution.contract.ignored_tactile_keys),
        "rule_collect_tactile": bool(tactile_resolution.collect_tactile),
        "rule_tactile_capture_mode": tactile_resolution.mode,
        "rule_tactile_capture_source": tactile_resolution.source,
        "tactile_bundle_min_free_gb": float(
            configuration.get(
                "tactile_bundle_min_free_gb",
                getattr(
                    target,
                    "tactile_bundle_min_free_gb",
                    DEFAULT_RULE_TACTILE_MIN_FREE_GB,
                ),
            )
        ),
        "tactile_compression": str(configuration["tactile_compression"]),
        "writer_workers": int(configuration["writer_workers"]),
        "writer_queue_size": int(configuration["writer_queue_size"]),
        "collection_process_queue_size": int(
            configuration["collection_process_queue_size"]
        ),
        "storage_queue_size": int(configuration["storage_queue_size"]),
        "storage_queue_enqueue_timeout_sec": float(
            configured_float(
                "storage_queue_enqueue_timeout_sec",
                DEFAULT_COLLECTION_STORAGE_QUEUE_ENQUEUE_TIMEOUT_SEC,
            )
        ),
        "camera_frame_max_age_sec": float(
            configured_float(
                "camera_frame_max_age_sec",
                DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
            )
        ),
        "collection_warning_hz": float(configuration["collection_warning_hz"]),
        "collection_flush_timeout_sec": float(
            configured_float(
                "collection_flush_timeout_sec",
                DEFAULT_COLLECTION_FLUSH_TIMEOUT_SEC,
            )
        ),
        "collection_sample_drain_timeout_sec": float(
            configured_float(
                "collection_sample_drain_timeout_sec",
                DEFAULT_COLLECTION_SAMPLE_DRAIN_TIMEOUT_SEC,
            )
        ),
        "collection_terminal_event_put_timeout_sec": float(
            configured_float(
                "collection_terminal_event_put_timeout_sec",
                DEFAULT_COLLECTION_TERMINAL_EVENT_PUT_TIMEOUT_SEC,
            )
        ),
        "write_parquet": bool(configuration["write_parquet"]),
        "write_video": bool(configuration["write_video"]),
    }
    if "writing_speed_m_s" in configuration:
        patch["rule_writing_speed_m_s"] = (
            None
            if bool(configuration.get("writing_speed_from_source", False))
            else float(configuration["writing_speed_m_s"])
        )
    if not tactile_resolution.collect_tactile:
        patch["dmtac_tactile"] = False
        patch["tactile_bundle_root"] = ""
        patch["tactile_bundle_min_free_gb"] = 0.0
        patch["tactile_artifact_integrity"] = False
        patch["idle_tactile_compression"] = False
        for key, value in patch.items():
            setattr(target, key, value)
        return patch

    patch["dmtac_tactile"] = True
    configured_root = str(configuration.get("tactile_bundle_root", "") or "").strip()
    minimum_free_gb = float(patch["tactile_bundle_min_free_gb"])
    if not configured_root:
        if minimum_free_gb > 0.0:
            raise ValueError(
                "触觉最低剩余空间大于 0 时必须配置触觉 Bundle 根目录。"
            )
        patch["tactile_bundle_root"] = ""
        patch["tactile_artifact_integrity"] = bool(
            getattr(target, "tactile_artifact_integrity", False)
        )
    else:
        if bool(getattr(target, "no_collection", False)):
            raise ValueError("触觉 Bundle 根目录要求启用正式采集。")
        if str(getattr(target, "tactile_storage", "external")) != "external":
            raise ValueError("触觉 Bundle 根目录要求 tactile_storage=external。")
        tactile_root = Path(os.path.expandvars(configured_root)).expanduser()
        if tactile_root.is_symlink() or not tactile_root.is_dir():
            raise ValueError(
                "触觉 Bundle 根目录必须是已存在的真实目录（不能是符号链接）："
                f"{tactile_root}"
            )
        tactile_resolved = tactile_root.resolve()
        dataset_root = Path(
            str(getattr(target, "dataset_root", "") or "")
        ).expanduser().resolve()
        if (
            tactile_resolved == dataset_root
            or tactile_resolved.is_relative_to(dataset_root)
            or dataset_root.is_relative_to(tactile_resolved)
        ):
            raise ValueError(
                "触觉 Bundle 根目录与正式数据集根目录必须互不包含："
                f"tactile={tactile_resolved} dataset={dataset_root}"
            )
        free_bytes = int(shutil.disk_usage(tactile_resolved).free)
        required_bytes = int(minimum_free_gb * float(1024**3))
        if free_bytes < required_bytes:
            raise ValueError(
                "触觉 Bundle 根目录剩余空间低于当前 GUI 阈值："
                f"可用 {free_bytes / float(1024**3):.1f} GiB，"
                f"要求 {minimum_free_gb:.1f} GiB。"
            )
        patch["tactile_bundle_root"] = str(tactile_root)
        patch["tactile_artifact_integrity"] = True
    patch["tactile_bundle_min_free_source"] = "rule_gui"
    for key, value in patch.items():
        setattr(target, key, value)
    return patch


def _load_replay_source_episode(
    spec: ResolvedRuleBasedWritingSourceCandidate,
    *,
    deployment: RuleBasedRealTeleopDeployment,
) -> _LoadedWritingSource:
    episode_index = int(spec.candidate.episode_index)
    episode = ReplayDatasetIndex(spec.dataset_root).load_episode(episode_index)
    source_path, resolved_metadata = resolve_replay_parquet_path(
        dataset_root=str(spec.dataset_root),
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
    if (
        source_contract.contract_hash
        != deployment.collection_pose_frame_contract.contract_hash
    ):
        raise ValueError(
            "source and real deployment pose-frame contract hashes differ; refusing "
            "to infer coordinate alignment."
        )
    if actions.shape[0] != episode.frame_count:
        raise ValueError("source action rows do not match the indexed episode frame count.")
    hand_joint_targets = load_replay_hand_joint_targets_from_parquet(
        source_path,
        active_sides=("left", "right"),
        start_index=0,
        end_index=None,
    )
    source_metadata = dict(resolved_metadata)
    dataset_info = dict(source_metadata.get("dataset_info") or {})
    for key in ("hand_action_specs_by_side", "hand_active_mask_by_side"):
        bridged = action_metadata.get(key)
        if key not in dataset_info and isinstance(bridged, Mapping):
            dataset_info[key] = dict(bridged)
    if dataset_info:
        source_metadata["dataset_info"] = dataset_info
    actual_qpos = load_replay_arm_observation_qpos_from_parquet(
        source_path,
        active_sides=("left", "right"),
        field_name="actual_hardware_qpos",
        start_index=0,
        end_index=None,
    )
    if set(actual_qpos) != {"left", "right"}:
        raise ValueError("initial alignment requires source actual qpos for both arms.")
    actual_tcp = _canonical_actual_tcp(
        load_replay_arm_actual_tcp_pose_from_parquet(
            source_path,
            active_sides=("left", "right"),
            start_index=0,
            end_index=None,
        ),
        source_contract=source_contract,
    )
    return _LoadedWritingSource(
        candidate_id=spec.candidate.candidate_id,
        dataset_root=spec.dataset_root,
        detail_annotation_dir=spec.detail_annotation_dir,
        episode_index=episode_index,
        episode=episode,
        selection=None,
        source_path=source_path,
        actions=actions,
        hand_joint_targets=hand_joint_targets,
        actual_qpos=actual_qpos,
        actual_tcp=actual_tcp,
        source_contract_hash=source_contract.contract_hash,
        metadata={
            "action_metadata": dict(action_metadata),
            "source_metadata": source_metadata,
            "typed_action_contract": bool(is_typed),
        },
    )


def _load_writing_source(
    spec: ResolvedRuleBasedWritingSourceCandidate,
    *,
    selected_phases: Sequence[str],
    deployment: RuleBasedRealTeleopDeployment,
) -> _LoadedWritingSource:
    source = _load_replay_source_episode(spec, deployment=deployment)
    selection = _load_detail_selection(source.episode, spec.detail_annotation_dir)
    for phase in selected_phases:
        phase_id = f"{DETAIL_PROGRESS_STEP_ID}.{phase}"
        span = selection.phases[phase_id]
        labels = {
            str(row.get("teleop.arm_mode.left.label", "")).strip()
            for row in source.episode.rows[span.start : span.end]
        }
        if labels != {"fixed_hold"}:
            raise ValueError(
                f"candidate {spec.candidate.candidate_id!r} phase {phase!r} requires "
                f"left fixed_hold; got {sorted(labels) or ['missing']}."
            )
    return replace(
        source,
        selection=selection,
        metadata={
            **dict(source.metadata),
            "detail_annotation_set_id": selection.annotation_set_id,
            "left_arm_policy": "source_fixed_hold",
        },
    )


def _configured_rule_dmtac_sides(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], dict[str, HandActionSpec], float]:
    hand_backend_by_side = {
        side: hand_backend_mode_for_side(args, side) for side in ("left", "right")
    }
    unsupported = {
        side: mode
        for side, mode in hand_backend_by_side.items()
        if mode not in {HAND_BACKEND_NONE, HAND_BACKEND_DMTAC_GRIPPER}
    }
    if unsupported:
        raise ValueError(
            "rule collection supports only DMTAC gripper or none per side; "
            f"got {unsupported}."
        )
    sides = tuple(
        side
        for side in ("left", "right")
        if hand_backend_by_side[side] == HAND_BACKEND_DMTAC_GRIPPER
    )
    if sides and str(getattr(args, "gripper_input_source", "none") or "none") != "none":
        raise ValueError(
            "rule replay requires gripper input source 'none'; Q6/angle-trigger "
            "control would create a second hand command owner."
        )
    return (
        sides,
        _dmtac_specs_by_side(sides),
        float(RuleBasedRealTeleopConfig().gripper_max_step),
    )


def _build_full_trajectory_replay(
    args: argparse.Namespace,
    *,
    deployment: RuleBasedRealTeleopDeployment,
    enable_prestart_configuration: bool,
) -> RuleBasedWritingBuild:
    if getattr(args, "rule_writing_speed_m_s", None) is not None:
        raise ValueError(
            "complete-Episode replay preserves source timestamps; "
            "--rule-writing-speed-m-s is not applicable."
        )
    spec, source_selection = _resolve_full_replay_source_spec(args)
    source = _load_replay_source_episode(spec, deployment=deployment)
    if source.episode.frame_count < 2:
        raise ValueError("complete-Episode replay requires at least two source frames.")
    for side in ("left", "right"):
        poses = np.asarray(source.actions[:, ACTION_ARM_POSE_SLICES[side]], dtype=float)
        if poses.shape != (source.episode.frame_count, 7) or not np.all(
            np.isfinite(poses)
        ):
            raise ValueError(
                f"complete-Episode replay requires finite {side} arm actions for "
                "every source frame."
            )
    (
        configured_gripper_sides,
        runtime_hand_specs,
        gripper_max_step,
    ) = _configured_rule_dmtac_sides(args)
    alignment_speed_scale = float(
        getattr(args, "rule_initial_alignment_speed_scale", 1.0)
    )
    if (
        not np.isfinite(alignment_speed_scale)
        or not 0.0 < alignment_speed_scale <= 1.0
    ):
        raise ValueError("rule initial alignment speed scale must be in (0, 1].")

    provenance = {
        "candidate_id": source.candidate_id,
        "dataset_root": str(source.dataset_root),
        "episode_index": source.episode_index,
        "source_contract_hash": source.source_contract_hash,
        "source_frame_count": source.episode.frame_count,
        "source_timing": source.episode.timing_source,
        "tags": list(spec.candidate.tags),
    }
    source_selection = {
        **source_selection,
        "source": provenance,
        "phases": {"full_replay": provenance},
        "gripper_sides": list(configured_gripper_sides),
        "control_topology": {
            "arms": ["left", "right"],
            "grippers": list(configured_gripper_sides),
        },
        "gripper_trajectory_mode": (
            "source_complete_episode"
            if configured_gripper_sides
            else "disabled"
        ),
        "gripper_dispatch_mode": (
            "causal_measured_step_limited"
            if configured_gripper_sides
            else "disabled"
        ),
        "gripper_max_step": gripper_max_step,
        "initial_alignment_speed_scale": alignment_speed_scale,
    }
    group = _slice_group(
        key="full_episode",
        span=StageSpan(0, source.episode.frame_count),
        episode_index=source.episode_index,
        source_path=source.source_path,
        action_hz=source.episode.nominal_hz,
        frame_times_sec=source.episode.frame_times_sec,
        actions=source.actions,
        hand_joint_targets=source.hand_joint_targets,
        actual_qpos=source.actual_qpos,
        actual_tcp=source.actual_tcp,
        metadata={
            **dict(source.metadata),
            "full_trajectory_replay_source_selection": source_selection,
        },
        left_hold_pose=None,
    )
    if configured_gripper_sides:
        group = _with_runtime_dmtac_contract(
            group,
            sides=configured_gripper_sides,
        )
        # A complete source Episode may legitimately contain an endpoint-like
        # gripper transition between adjacent dataset frames.  That source
        # delta is provenance, not one physical dispatch step.  The real
        # runtime applies ``gripper_max_step`` at the existing live
        # emission/Session boundary against fresh causal feedback, while the
        # arm replay clock continues to follow the source timestamps.
    capture = RuleBasedCaptureSelection(enabled=True)
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedInitialize(
                step_id=FULL_REPLAY_INITIALIZE_STEP_ID,
                group_key=group.key,
                frame_index=0,
                active_sides=("left", "right"),
                qpos_source="observation_actual_qpos",
                include_end_effectors=bool(configured_gripper_sides),
                end_effector_sides=configured_gripper_sides,
                capture=RuleBasedCaptureSelection(enabled=False),
            ),
            RuleBasedReplay(
                step_id=FULL_REPLAY_STEP_ID,
                group_key=group.key,
                start_index=0,
                end_index=group.frame_count,
                active_sides=("left", "right"),
                include_end_effectors=bool(configured_gripper_sides),
                end_effector_sides=configured_gripper_sides,
                wait_for_terminal_arrival=True,
                capture=capture,
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=deployment,
        plan=plan,
        trajectory_groups={group.key: group},
        config=RuleBasedRealTeleopConfig(
            action_hz=float(source.episode.nominal_hz),
            position_reference_hz=effective_position_reference_hz(args),
            initial_alignment_speed_scale=alignment_speed_scale,
            gripper_sides=configured_gripper_sides,
            gripper_max_step=gripper_max_step,
        ),
        source_selection=source_selection,
    )
    if enable_prestart_configuration:
        options, selected_configuration = _full_replay_configuration_options(
            args,
            selected_spec=spec,
            source_specs=_source_pool_candidates(args),
        )
        base_args = copy.copy(args)

        def rebuild_from_configuration(
            configuration: Mapping[str, Any],
        ) -> RuleBasedRealTeleopExtension:
            rebuilt_args = copy.copy(base_args)
            rebuilt_args._rule_source_selection_ordinal = int(
                configuration.get("_selection_ordinal", 0)
            )
            rebuilt_args._rule_full_replay_candidate_id = str(
                configuration["full_replay_candidate_id"]
            )
            rebuilt_args.rule_initial_alignment_speed_scale = float(
                configuration["initial_alignment_speed_scale"]
            )
            runtime_args_patch = _apply_rule_gui_runtime_configuration(
                rebuilt_args,
                configuration,
            )
            replacement = _build_full_trajectory_replay(
                rebuilt_args,
                deployment=deployment,
                enable_prestart_configuration=False,
            ).extension
            replacement._prestart_runtime_args_patch = runtime_args_patch
            return replacement

        def commit_runtime_configuration(
            _configuration: Mapping[str, Any],
            replacement: RuleBasedRealTeleopExtension,
        ) -> None:
            runtime_args_patch = getattr(
                replacement,
                "_prestart_runtime_args_patch",
                None,
            )
            if not isinstance(runtime_args_patch, Mapping):
                raise RuntimeError("重建后的完整 Replay 计划缺少已验证的运行参数。")
            for key, value in runtime_args_patch.items():
                setattr(args, str(key), value)

        extension.configure_prestart_rebuild(
            options=options,
            selected=selected_configuration,
            rebuilder=rebuild_from_configuration,
            on_commit=commit_runtime_configuration,
        )
    return RuleBasedWritingBuild(
        extension=extension,
        source_episode=source.episode_index,
        source_contract_hash=source.source_contract_hash,
        detail_annotation_set_id=None,
        writing_speed_m_s=None,
        writing_action_hz=float(source.episode.nominal_hz),
        source_selection=source_selection,
        initial_alignment_speed_scale=alignment_speed_scale,
        gripper_sides=configured_gripper_sides,
        hand_action_specs_by_side=dict(runtime_hand_specs),
        collection_mode=RULE_COLLECTION_MODE_FULL_REPLAY,
    )


def build_rule_based_writing(
    args: argparse.Namespace,
    *,
    deployment: RuleBasedRealTeleopDeployment | None = None,
    enable_prestart_configuration: bool = True,
) -> RuleBasedWritingBuild:
    tactile_resolution = _resolve_rule_tactile_capture(args)
    _apply_rule_tactile_capture_resolution(args, tactile_resolution)
    print(
        "[RuleBased] tactile_capture="
        f"{tactile_resolution.mode} "
        f"source={tactile_resolution.source} "
        f"locked={str(tactile_resolution.locked).lower()} "
        f"ignored_sides={list(tactile_resolution.contract.ignored_tactile_keys)}"
    )
    deployment = deployment or preflight_rule_based_real_teleop_deployment(args)
    collection_mode = str(
        getattr(args, "rule_collection_mode", RULE_COLLECTION_MODE_WRITING)
        or RULE_COLLECTION_MODE_WRITING
    )
    if collection_mode == RULE_COLLECTION_MODE_FULL_REPLAY:
        if getattr(args, "rule_writing_correct_tcp", False):
            raise ValueError("--correct-tcp is only valid for rule-based writing")
        if getattr(args, "rule_writing_trajectory_catalog", None):
            raise ValueError(
                "rule writing trajectory catalog is not valid in full replay mode."
            )
        if _requested_pre_pen_down_replay_candidate_id(args):
            raise ValueError(
                "pre-pen-down replay is not valid in full replay mode."
            )
        return _build_full_trajectory_replay(
            args,
            deployment=deployment,
            enable_prestart_configuration=enable_prestart_configuration,
        )
    if collection_mode != RULE_COLLECTION_MODE_WRITING:
        raise ValueError(f"unsupported rule collection mode: {collection_mode!r}.")
    additional_catalog_path = _additional_fixed_trajectory_catalog_path(args)
    (
        resolved_trajectory_id,
        resolved_direction,
        resolved_size_m,
        writing_variant_selection,
    ) = _resolve_writing_variant(
        args,
        additional_catalog_path=additional_catalog_path,
    )
    direction_yaw_override = (
        float(writing_variant_selection["selected_yaw_rad"])
        if float(writing_variant_selection["selected_direction_deviation_deg"]) != 0.0
        else None
    )
    source_specs, source_selection = _resolve_source_specs(args)
    pre_pen_down_spec = _resolve_pre_pen_down_replay_source_spec(args)
    requested_by_candidate: dict[str, list[str]] = {}
    spec_by_candidate: dict[str, ResolvedRuleBasedWritingSourceCandidate] = {}
    for phase, spec in source_specs.items():
        requested_by_candidate.setdefault(spec.candidate.candidate_id, []).append(phase)
        spec_by_candidate[spec.candidate.candidate_id] = spec
    loaded_by_candidate = {
        candidate_id: _load_writing_source(
            spec_by_candidate[candidate_id],
            selected_phases=phases,
            deployment=deployment,
        )
        for candidate_id, phases in requested_by_candidate.items()
    }
    selected_sources = {
        phase: loaded_by_candidate[spec.candidate.candidate_id]
        for phase, spec in source_specs.items()
    }
    pre_pen_down_source: _LoadedWritingSource | None = None
    if pre_pen_down_spec is not None:
        pre_pen_down_source = loaded_by_candidate.get(
            pre_pen_down_spec.candidate.candidate_id
        )
        if pre_pen_down_source is None:
            pre_pen_down_source = _load_replay_source_episode(
                pre_pen_down_spec,
                deployment=deployment,
            )
        if pre_pen_down_source.episode.frame_count < 2:
            raise ValueError("pre-pen-down replay requires at least two source frames.")
        right_actions = np.asarray(
            pre_pen_down_source.actions[:, ACTION_ARM_POSE_SLICES["right"]],
            dtype=float,
        )
        if right_actions.shape != (
            pre_pen_down_source.episode.frame_count,
            7,
        ) or not np.all(np.isfinite(right_actions)):
            raise ValueError(
                "pre-pen-down replay requires finite right-arm actions for every "
                "source frame."
            )
    pen_down_source = selected_sources["pen_down"]
    write_source = selected_sources["write"]
    pen_up_source = selected_sources["pen_up"]
    if any(source.selection is None for source in selected_sources.values()):
        raise RuntimeError("annotated writing source lost its detail-stage selection.")
    pen_down = pen_down_source.selection.phases[DETAIL_PHASE_IDS[0]]
    write = write_source.selection.phases[DETAIL_PHASE_IDS[1]]
    pen_up = pen_up_source.selection.phases[DETAIL_PHASE_IDS[2]]
    left_hold_pose = np.asarray(
        pen_down_source.actions[pen_down.start, ACTION_ARM_POSE_SLICES["left"]],
        dtype=np.float32,
    ).copy()
    if left_hold_pose.shape != (7,) or not np.all(np.isfinite(left_hold_pose)):
        raise ValueError("source pen-down left hold pose is invalid.")
    left_hold_qpos = np.asarray(
        pen_down_source.actual_qpos["left"][pen_down.start],
        dtype=np.float32,
    ).copy()
    left_hold_tcp = np.asarray(
        pen_down_source.actual_tcp["left"][pen_down.start],
        dtype=np.float32,
    ).copy()

    (
        configured_gripper_sides,
        runtime_hand_specs,
        gripper_max_step,
    ) = _configured_rule_dmtac_sides(args)

    alignment_speed_scale = float(
        getattr(args, "rule_initial_alignment_speed_scale", 1.0)
    )
    if not np.isfinite(alignment_speed_scale) or not 0.0 < alignment_speed_scale <= 1.0:
        raise ValueError("rule initial alignment speed scale must be in (0, 1].")

    phase_provenance: dict[str, Any] = {}
    for phase in RULE_BASED_WRITING_SOURCE_PHASES_V1:
        source = selected_sources[phase]
        spec = source_specs[phase]
        phase_provenance[phase] = {
            "candidate_id": source.candidate_id,
            "dataset_root": str(source.dataset_root),
            "episode_index": source.episode_index,
            "detail_annotation_dir": str(source.detail_annotation_dir),
            "detail_annotation_set_id": source.selection.annotation_set_id,
            "source_contract_hash": source.source_contract_hash,
            "tags": list(spec.candidate.tags),
        }
    pre_pen_down_provenance: dict[str, Any] | None = None
    pre_pen_down_normal_episodes = _pre_pen_down_replay_normal_episodes(args)
    if pre_pen_down_source is not None and pre_pen_down_spec is not None:
        pre_pen_down_provenance = {
            "candidate_id": pre_pen_down_source.candidate_id,
            "dataset_root": str(pre_pen_down_source.dataset_root),
            "episode_index": pre_pen_down_source.episode_index,
            "source_contract_hash": pre_pen_down_source.source_contract_hash,
            "source_frame_count": pre_pen_down_source.episode.frame_count,
            "source_timing": pre_pen_down_source.episode.timing_source,
            "tags": list(pre_pen_down_spec.candidate.tags),
            "capture_enabled": False,
            "active_arm_sides": ["right"],
            "left_arm_policy": "pen_down_source_fixed_hold",
            "normal_episodes_between_replays": pre_pen_down_normal_episodes,
        }
    source_selection = {
        **source_selection,
        "collection_mode": RULE_COLLECTION_MODE_WRITING,
        "trajectory_catalog": {
            "mode": (
                "default_plus_additional"
                if additional_catalog_path is not None
                else "default"
            ),
            "combined_sha256": fixed_trajectory_catalog_hash(
                additional_catalog_path=additional_catalog_path,
            ),
            "additional_path": (
                str(additional_catalog_path)
                if additional_catalog_path is not None
                else None
            ),
            "additional_sha256": (
                fixed_trajectory_catalog_hash(additional_catalog_path)
                if additional_catalog_path is not None
                else None
            ),
        },
        "phases": phase_provenance,
        **(
            {"pre_pen_down_replay": pre_pen_down_provenance}
            if pre_pen_down_provenance is not None
            else {}
        ),
        "gripper_sides": list(configured_gripper_sides),
        "control_topology": {
            "arms": ["left", "right"],
            "grippers": list(configured_gripper_sides),
        },
        "gripper_trajectory_mode": (
            "right_task_chain_fixed_closed"
            if "right" in configured_gripper_sides
            else "source_phase_relative_continuous_time_normalized_write"
            if configured_gripper_sides
            else "disabled"
        ),
        "gripper_max_step": gripper_max_step,
        "initial_alignment_speed_scale": alignment_speed_scale,
        "writing_variant_selection": writing_variant_selection,
    }

    phase_groups = {
        "pen_down": _slice_group(
            key="pen_down",
            span=pen_down,
            episode_index=pen_down_source.episode_index,
            source_path=pen_down_source.source_path,
            action_hz=pen_down_source.episode.nominal_hz,
            frame_times_sec=pen_down_source.episode.frame_times_sec,
            actions=pen_down_source.actions,
            hand_joint_targets=pen_down_source.hand_joint_targets,
            actual_qpos=pen_down_source.actual_qpos,
            actual_tcp=pen_down_source.actual_tcp,
            metadata={
                **pen_down_source.metadata,
                "rule_based_source_selection": source_selection,
            },
            left_hold_pose=left_hold_pose,
        ),
        "write": _slice_group(
            key="write",
            span=write,
            episode_index=write_source.episode_index,
            source_path=write_source.source_path,
            action_hz=write_source.episode.nominal_hz,
            frame_times_sec=write_source.episode.frame_times_sec,
            actions=write_source.actions,
            hand_joint_targets=write_source.hand_joint_targets,
            actual_qpos=write_source.actual_qpos,
            actual_tcp=write_source.actual_tcp,
            metadata={
                **write_source.metadata,
                "rule_based_source_selection": source_selection,
            },
            left_hold_pose=left_hold_pose,
        ),
        "pen_up": _slice_group(
            key="pen_up",
            span=pen_up,
            episode_index=pen_up_source.episode_index,
            source_path=pen_up_source.source_path,
            action_hz=pen_up_source.episode.nominal_hz,
            frame_times_sec=pen_up_source.episode.frame_times_sec,
            actions=pen_up_source.actions,
            hand_joint_targets=pen_up_source.hand_joint_targets,
            actual_qpos=pen_up_source.actual_qpos,
            actual_tcp=pen_up_source.actual_tcp,
            metadata={
                **pen_up_source.metadata,
                "rule_based_source_selection": source_selection,
            },
            left_hold_pose=left_hold_pose,
        ),
    }
    pre_pen_down_group: ReplayTrajectoryGroup | None = None
    if pre_pen_down_source is not None:
        pre_pen_down_group = _slice_group(
            key="pre_pen_down",
            span=StageSpan(0, pre_pen_down_source.episode.frame_count),
            episode_index=pre_pen_down_source.episode_index,
            source_path=pre_pen_down_source.source_path,
            action_hz=pre_pen_down_source.episode.nominal_hz,
            frame_times_sec=pre_pen_down_source.episode.frame_times_sec,
            actions=pre_pen_down_source.actions,
            hand_joint_targets=pre_pen_down_source.hand_joint_targets,
            actual_qpos=pre_pen_down_source.actual_qpos,
            actual_tcp=pre_pen_down_source.actual_tcp,
            metadata={
                **pre_pen_down_source.metadata,
                "rule_based_source_selection": source_selection,
            },
            left_hold_pose=left_hold_pose,
        )
        pre_pen_down_group = _freeze_group_arm_state(
            pre_pen_down_group,
            side="left",
            action_pose=left_hold_pose,
            actual_qpos=left_hold_qpos,
            actual_tcp=left_hold_tcp,
        )
    if configured_gripper_sides:
        phase_groups = {
            phase: _with_runtime_dmtac_contract(
                group,
                sides=configured_gripper_sides,
            )
            for phase, group in phase_groups.items()
        }
        if pre_pen_down_group is not None:
            pre_pen_down_group = _with_runtime_dmtac_contract(
                pre_pen_down_group,
                sides=configured_gripper_sides,
            )
    if "right" in configured_gripper_sides:
        # The task chain is the sole owner of the writing hand: it closes the
        # right gripper once before this plan starts and opens it once after
        # the Episode. Source hand traces remain represented by source
        # provenance, but must not re-drive the pen grip during approach,
        # writing or departure. Keeping the canonical action fixed at the
        # closed endpoint also keeps recorded action semantics aligned with
        # the physical command without introducing a second gripper loop.
        phase_groups = {
            phase: _replace_group_hand_trajectory(
                group,
                {
                    "right": _fixed_dmtac_trajectory(
                        frame_count=group.frame_count,
                        scalar=1.0,
                    )
                },
            )
            for phase, group in phase_groups.items()
        }
        if pre_pen_down_group is not None:
            pre_pen_down_group = _replace_group_hand_trajectory(
                pre_pen_down_group,
                {
                    "right": _fixed_dmtac_trajectory(
                        frame_count=pre_pen_down_group.frame_count,
                        scalar=1.0,
                    )
                },
            )

    speed = (
        float(args.rule_writing_speed_m_s)
        if args.rule_writing_speed_m_s is not None
        else _average_tcp_speed(
            write_source.actions,
            write_source.episode.frame_times_sec,
            (write,),
        )
    )
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("rule writing speed must be finite and positive.")
    size_m = float(resolved_size_m)
    if not np.isfinite(size_m) or size_m <= 0.0:
        raise ValueError("rule writing size must be finite and positive.")

    preview_keyframes = build_fixed_trajectory_targets(
        {
            "left": left_hold_pose,
            "right": pen_down_source.actions[
                pen_down.end - 1,
                ACTION_ARM_POSE_SLICES["right"],
            ],
        },
        resolved_trajectory_id,
        scale_m=size_m,
        linear_speed_m_s=speed,
        side="right",
        direction="up" if direction_yaw_override is not None else resolved_direction,
        yaw_rad=direction_yaw_override,
        additional_catalog_path=additional_catalog_path,
    )
    preview_targets = resample_motion_targets(
        preview_keyframes,
        reference_hz=float(write_source.episode.nominal_hz),
        include_final=True,
    )
    duration = float(preview_targets[-1].timestamp - preview_targets[0].timestamp)
    if duration <= 0.0 or len(preview_targets) < 2:
        raise ValueError("fixed writing trajectory has no positive-duration path.")
    writing_action_hz = float((len(preview_targets) - 1) / duration)
    writing_reachability_enabled = bool(
        getattr(args, "rule_writing_reachability_check", False)
    )
    reachability_probes = _writing_reachability_contact_probes(preview_keyframes)
    if writing_reachability_enabled:
        reachability_probes.extend(
            _relative_pen_up_reachability_probes(
                phase_groups["pen_up"], writing_end=preview_targets[-1],
            )
        )
    source_selection = {
        **source_selection,
        "writing_reachability": {
            "schema": "dexteleop.rule_based_writing_reachability.v1",
            "enabled": writing_reachability_enabled,
            "probe_definition": (
                "four_writing_bbox_contacts_and_two_relative_pen_up_samples"
                if writing_reachability_enabled
                else "centerline_axis_aligned_bbox_edge_contacts"
            ),
            "probe_count": len(reachability_probes),
            "left_hold_pose_canonical_desk": left_hold_pose.astype(float).tolist(),
            "probes": reachability_probes,
        },
    }
    phase_groups = {
        phase: replace(
            group,
            metadata={
                **dict(group.metadata),
                "rule_based_source_selection": source_selection,
            },
        )
        for phase, group in phase_groups.items()
    }
    if pre_pen_down_group is not None:
        pre_pen_down_group = replace(
            pre_pen_down_group,
            metadata={
                **dict(pre_pen_down_group.metadata),
                "rule_based_source_selection": source_selection,
            },
        )
    writing_hand_trajectory: dict[str, np.ndarray] = {}
    if configured_gripper_sides:
        pen_down_hand = {
            side: np.asarray(
                phase_groups["pen_down"].hand_joint_targets[side],
                dtype=np.float32,
            ).copy()
            for side in configured_gripper_sides
        }
        source_write_hand = {
            side: np.asarray(
                phase_groups["write"].hand_joint_targets[side],
                dtype=np.float32,
            ).copy()
            for side in configured_gripper_sides
        }
        source_pen_up_hand = {
            side: np.asarray(
                phase_groups["pen_up"].hand_joint_targets[side],
                dtype=np.float32,
            ).copy()
            for side in configured_gripper_sides
        }
        preview_times = tuple(float(target.timestamp) for target in preview_targets)
        pen_up_rebased: dict[str, np.ndarray] = {}
        for side in configured_gripper_sides:
            _validate_hand_step_limit(
                pen_down_hand[side],
                side=side,
                phase="pen_down",
                max_step=gripper_max_step,
            )
            if side == "right":
                writing_hand_trajectory[side] = _fixed_dmtac_trajectory(
                    frame_count=len(preview_targets),
                    scalar=1.0,
                )
                pen_up_rebased[side] = _fixed_dmtac_trajectory(
                    frame_count=phase_groups["pen_up"].frame_count,
                    scalar=1.0,
                )
                continue
            anchored_write = _rebase_hand_trajectory(
                source_write_hand[side],
                anchor=pen_down_hand[side][-1],
            )
            writing_hand_trajectory[side] = _resample_hand_trajectory(
                anchored_write,
                source_times_sec=phase_groups["write"].frame_times_sec,
                target_times_sec=preview_times,
            )
            _validate_hand_step_limit(
                writing_hand_trajectory[side],
                side=side,
                phase="write",
                max_step=gripper_max_step,
            )
            pen_up_rebased[side] = _rebase_hand_trajectory(
                source_pen_up_hand[side],
                anchor=writing_hand_trajectory[side][-1],
            )
            _validate_hand_step_limit(
                pen_up_rebased[side],
                side=side,
                phase="pen_up",
                max_step=gripper_max_step,
            )
        for side in configured_gripper_sides:
            if not np.allclose(
                pen_down_hand[side][-1],
                writing_hand_trajectory[side][0],
                atol=1e-7,
            ) or not np.allclose(
                writing_hand_trajectory[side][-1],
                pen_up_rebased[side][0],
                atol=1e-7,
            ):
                raise ValueError(
                    f"{side} gripper phase boundaries are not continuous."
                )
        phase_groups["pen_up"] = _replace_group_hand_trajectory(
            phase_groups["pen_up"],
            pen_up_rebased,
        )
    groups = {
        "pen_down": phase_groups["pen_down"],
        "pen_up": phase_groups["pen_up"],
    }
    if pre_pen_down_group is not None:
        groups[pre_pen_down_group.key] = pre_pen_down_group
    writing_source = FixedWritingActionSource(
        trajectory_id=resolved_trajectory_id,
        scale_m=size_m,
        speed_m_s=speed,
        direction=resolved_direction,
        yaw_rad=direction_yaw_override,
        left_hold_pose=left_hold_pose,
        additional_catalog_path=additional_catalog_path,
        hand_trajectory_by_side=writing_hand_trajectory,
        hand_action_specs_by_side=runtime_hand_specs,
        reference_hz=float(write_source.episode.nominal_hz),
    )
    height_descent_source = CalibratedHeightDescentActionSource(
        reference_hz=effective_position_reference_hz(args),
        linear_speed_m_s=speed,
        hand_action_specs_by_side=runtime_hand_specs,
    )
    capture = RuleBasedCaptureSelection(enabled=True)
    normal_initialize = RuleBasedInitialize(
        step_id="initial_alignment",
        group_key="pen_down",
        frame_index=0,
        active_sides=("left", "right"),
        qpos_source="observation_actual_qpos",
        include_end_effectors=bool(configured_gripper_sides),
        end_effector_sides=configured_gripper_sides,
        capture=RuleBasedCaptureSelection(enabled=False),
    )
    pre_pen_down_steps: tuple[Any, ...] = ()
    if pre_pen_down_group is not None:
        pre_pen_down_steps = (
            RuleBasedInitialize(
                step_id=PRE_PEN_DOWN_ALIGNMENT_STEP_ID,
                group_key=pre_pen_down_group.key,
                frame_index=0,
                active_sides=("left", "right"),
                qpos_source="observation_actual_qpos",
                include_end_effectors=bool(configured_gripper_sides),
                end_effector_sides=configured_gripper_sides,
                capture=RuleBasedCaptureSelection(enabled=False),
            ),
            RuleBasedReplay(
                step_id=PRE_PEN_DOWN_REPLAY_STEP_ID,
                group_key=pre_pen_down_group.key,
                start_index=0,
                end_index=pre_pen_down_group.frame_count,
                active_sides=("right",),
                include_end_effectors=bool(configured_gripper_sides),
                end_effector_sides=configured_gripper_sides,
                wait_for_terminal_arrival=True,
                capture=RuleBasedCaptureSelection(enabled=False),
            ),
            RuleBasedInitialize(
                step_id=PRE_PEN_DOWN_TO_PEN_DOWN_ALIGNMENT_STEP_ID,
                group_key="pen_down",
                frame_index=0,
                # The first alignment already moved the left arm to this
                # Episode's pen-down source state and the runtime FixArm now
                # owns that hold. Only the right arm may move again.
                active_sides=("right",),
                qpos_source="observation_actual_qpos",
                include_end_effectors=bool(configured_gripper_sides),
                end_effector_sides=configured_gripper_sides,
                capture=RuleBasedCaptureSelection(enabled=False),
            ),
        )
    collection_steps: tuple[Any, ...] = (
        RuleBasedReplay(
            step_id=DETAIL_PHASE_IDS[0],
            group_key="pen_down",
            start_index=0,
            end_index=groups["pen_down"].frame_count,
            active_sides=("left", "right"),
            include_end_effectors=bool(configured_gripper_sides),
            end_effector_sides=configured_gripper_sides,
            wait_for_terminal_arrival=True,
            capture=capture,
        ),
        RuleBasedActionSegment(
            step_id=HEIGHT_DESCENT_STEP_ID,
            action_source=height_descent_source,
            stop_condition=RuleBasedSegmentStopCondition(
                max_steps=height_descent_source.max_steps,
                min_steps=1,
                stop_when=_height_descent_complete,
                name="calibrated_height_descent_complete",
            ),
            capture=capture,
            allow_empty=True,
            planned_steps_hint=0,
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
            include_end_effectors=bool(configured_gripper_sides),
            end_effector_sides=configured_gripper_sides,
            wait_for_terminal_arrival=True,
            capture=capture,
        ),
    )
    normal_plan = RuleBasedCollectionPlan(steps=(normal_initialize, *collection_steps))
    pre_pen_down_plan = (
        RuleBasedCollectionPlan(steps=(*pre_pen_down_steps, *collection_steps))
        if pre_pen_down_steps
        else None
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=deployment,
        plan=normal_plan,
        trajectory_groups=groups,
        config=RuleBasedRealTeleopConfig(
            action_hz=writing_action_hz,
            position_reference_hz=effective_position_reference_hz(args),
            initial_alignment_speed_scale=alignment_speed_scale,
            pre_pen_down_replay_normal_episodes=pre_pen_down_normal_episodes,
            gripper_sides=configured_gripper_sides,
            gripper_max_step=gripper_max_step,
            correct_tcp=bool(getattr(args, "rule_writing_correct_tcp", False)),
        ),
        source_selection=source_selection,
        pre_pen_down_replay_plan=pre_pen_down_plan,
    )
    if enable_prestart_configuration:
        options, selected_configuration = _rule_configuration_options(
            args,
            source_specs=source_specs,
            resolved_writing_speed_m_s=speed,
        )
        base_args = copy.copy(args)

        def rebuild_from_configuration(
            configuration: Mapping[str, Any],
        ) -> RuleBasedRealTeleopExtension:
            rebuilt_args = copy.copy(base_args)
            rebuilt_args._rule_source_selection_ordinal = int(
                configuration.get("_selection_ordinal", 0)
            )
            rebuilt_args._rule_source_candidate_ids_by_phase = {
                phase: str(configuration[f"{phase}_candidate_id"])
                for phase in RULE_BASED_WRITING_SOURCE_PHASES_V1
            }
            rebuilt_args._rule_pre_pen_down_replay_candidate_id = str(
                configuration["pre_pen_down_replay_candidate_id"]
            )
            rebuilt_args.rule_pre_pen_down_replay_normal_episodes = int(
                configuration["pre_pen_down_replay_normal_episodes"]
            )
            rebuilt_args.rule_writing_trajectory = str(configuration["trajectory_id"])
            rebuilt_args.rule_writing_direction = str(configuration["direction"])
            rebuilt_args.rule_writing_size_m = float(configuration["writing_size_m"])
            rebuilt_args.rule_initial_alignment_speed_scale = float(
                configuration["initial_alignment_speed_scale"]
            )
            runtime_args_patch = _apply_rule_gui_runtime_configuration(
                rebuilt_args,
                configuration,
            )
            replacement = build_rule_based_writing(
                rebuilt_args,
                deployment=deployment,
                enable_prestart_configuration=False,
            ).extension
            replacement._prestart_runtime_args_patch = runtime_args_patch
            return replacement

        def commit_runtime_configuration(
            _configuration: Mapping[str, Any],
            replacement: RuleBasedRealTeleopExtension,
        ) -> None:
            runtime_args_patch = getattr(
                replacement,
                "_prestart_runtime_args_patch",
                None,
            )
            if not isinstance(runtime_args_patch, Mapping):
                raise RuntimeError("重建后的规则计划缺少已验证的运行参数。")
            for key, value in runtime_args_patch.items():
                setattr(args, str(key), value)

        extension.configure_prestart_rebuild(
            options=options,
            selected=selected_configuration,
            rebuilder=rebuild_from_configuration,
            on_commit=commit_runtime_configuration,
        )
    return RuleBasedWritingBuild(
        extension=extension,
        source_episode=pen_down_source.episode_index,
        source_contract_hash=pen_down_source.source_contract_hash,
        detail_annotation_set_id=pen_down_source.selection.annotation_set_id,
        writing_speed_m_s=speed,
        writing_action_hz=writing_action_hz,
        source_selection=source_selection,
        initial_alignment_speed_scale=alignment_speed_scale,
        gripper_sides=configured_gripper_sides,
        hand_action_specs_by_side=dict(runtime_hand_specs),
        collection_mode=RULE_COLLECTION_MODE_WRITING,
    )


def build_rule_based_writing_extension(
    args: argparse.Namespace,
) -> RuleBasedRealTeleopExtension:
    build = build_rule_based_writing(args)
    # The component builder consumes this private application binding before
    # opening devices.  Rule commands do not use or monitor VIVE wrist sources.
    args._real_teleop_rule_command_source = True
    args._real_teleop_rule_hand_action_specs_by_side = dict(
        build.hand_action_specs_by_side
    )
    selected = build.source_selection.get("phases", {})
    if build.collection_mode == RULE_COLLECTION_MODE_FULL_REPLAY:
        selected_summary = (
            "full_replay="
            f"{selected.get('full_replay', {}).get('candidate_id', '?')}"
        )
        speed_summary = "source_timestamps"
    else:
        selected_summary = ",".join(
            f"{phase}={selected.get(phase, {}).get('candidate_id', '?')}"
            for phase in RULE_BASED_WRITING_SOURCE_PHASES_V1
        )
        prelude = build.source_selection.get("pre_pen_down_replay")
        if isinstance(prelude, Mapping):
            selected_summary = (
                f"pre_pen_down={prelude.get('candidate_id', '?')},"
                f"{selected_summary}"
            )
        speed_summary = f"{build.writing_speed_m_s:.6f}"
    print(
        "[RuleBased] prepared "
        f"mode={build.collection_mode} "
        f"episode={build.source_episode} "
        f"annotation_set={build.detail_annotation_set_id} "
        f"speed_m_s={speed_summary} "
        f"action_hz={build.writing_action_hz:.3f} "
        f"pose_contract={build.source_contract_hash} "
        f"selected_sources={selected_summary}"
        + (" correct_tcp=right_flange_plus_z_0p20m"
           if getattr(args, "rule_writing_correct_tcp", False) else "")
    )
    return build.extension


__all__ = [
    "CalibratedHeightDescentActionSource",
    "DETAIL_PHASE_IDS",
    "DETAIL_PROGRESS_STEP_ID",
    "FULL_REPLAY_INITIALIZE_STEP_ID",
    "FULL_REPLAY_STEP_ID",
    "PRE_PEN_DOWN_ALIGNMENT_STEP_ID",
    "PRE_PEN_DOWN_REPLAY_STEP_ID",
    "PRE_PEN_DOWN_TO_PEN_DOWN_ALIGNMENT_STEP_ID",
    "RULE_BASED_RANDOM_SOURCE_SELECTION_ID",
    "RULE_COLLECTION_MODE_FULL_REPLAY",
    "RULE_COLLECTION_MODE_WRITING",
    "DetailWritingSelection",
    "FixedWritingActionSource",
    "HEIGHT_DESCENT_STEP_ID",
    "RuleBasedWritingBuild",
    "RuleTactileCaptureResolution",
    "StageSpan",
    "add_rule_based_writing_arguments",
    "build_rule_based_writing",
    "build_rule_based_writing_extension",
]
