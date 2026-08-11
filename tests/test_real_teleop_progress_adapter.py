from __future__ import annotations

from collections import deque
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np
import pytest

from control_station.real_teleop.progress import (
    ProgressControlAvailability,
    build_real_teleop_progress_ui_payload,
)
from control_station.real_teleop_application import (
    build_real_teleop_startup_progress_ui_payload,
)
from collection_runtime.progress.profiles import (
    DATASET_PROGRESS_PROFILE_RELATIVE_PATH,
    DatasetProgressProfileBindingError,
    bind_dataset_progress_profile,
    read_dataset_progress_profile,
)
from collection_runtime.sample_acceptance import (
    SampleAcceptanceReceiptBuffer,
)
from collection_runtime.modality_storage import (
    detach_modality,
    plan_modality_detach,
    plan_modality_rehydrate,
    rehydrate_modality,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.pose_frames import PoseFrameContract, PoseSourceProfile
from dataset_contracts.progress_profile import (
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)
from dataset_contracts.v22 import PolicyIOLayout
from dataset_interface.progress_artifacts import load_sealed_progress_episode
from dataset_interface.modality_artifacts import (
    DETACHED_VERIFIED,
    PAYLOAD_ONLY_DETACH,
    WHOLE_PREFIX_DETACH,
    build_modality_artifact_inventory,
    resolve_modality_availability,
)
from dataset_interface.validation import (
    validate_v23_core_dataset,
    validate_v23_dataset,
)
from teleop_runtime.collection.sample_assembler import CollectorSample
from teleop_runtime.collection.writer_v21 import (
    LeRobotV21Writer,
    LeRobotV21WriterConfig,
)
from teleop_runtime.real_teleop.cli import build_real_teleop_argument_parser
from teleop_runtime.real_teleop.dataset_finalize import RealDatasetFinalizeTask
from teleop_runtime.real_teleop.dataset_recorder import RealTeleopDatasetRecorder
from teleop_runtime.real_teleop.progress_adapter import (
    DEFAULT_REAL_TELEOP_MANAGED_PROGRESS_PROFILE_ID,
    RealTeleopProgressEpisode,
    load_real_teleop_managed_progress_profile,
    load_real_teleop_progress_profile,
    resolve_real_teleop_progress_entry,
    resolve_real_teleop_default_progress_profile,
    resolve_real_teleop_task_stage_profile,
    select_real_teleop_managed_progress_profile,
    validate_real_teleop_initial_progress_profile,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FORMAL_PROGRESS_PROFILE = (
    _PROJECT_ROOT
    / "collection_runtime"
    / "progress"
    / "profiles"
    / "linear"
    / "dmtac_rm65_bimanual_position_v1.progress.json"
)


def _profile(*, task_id: str = "task-a") -> TaskProgressProfileV1:
    return TaskProgressProfileV1(
        profile_id="task-a-progress",
        profile_version="1",
        task_id=task_id,
        steps=(
            ProgressStepDefinitionV1(
                progress_step_id="complete_task",
                semantic_type="COMPLETE_TASK",
                display_name="完成任务",
                ordinal=1,
            ),
        ),
        initial_step_id="complete_task",
        terminal_step_ids=("complete_task",),
    )


def _multi_step_profile() -> TaskProgressProfileV1:
    return TaskProgressProfileV1(
        profile_id="pick-place-progress",
        profile_version="1",
        task_id="task-a",
        steps=(
            ProgressStepDefinitionV1(
                progress_step_id="pick",
                semantic_type="PICK",
                display_name="抓取",
                ordinal=1,
            ),
            ProgressStepDefinitionV1(
                progress_step_id="place",
                semantic_type="PLACE",
                display_name="放置",
                ordinal=2,
            ),
        ),
        initial_step_id="pick",
        terminal_step_ids=("place",),
    )


def _identity(episode_uuid: str = "episode-progress-a") -> EpisodeCaptureIdentity:
    return EpisodeCaptureIdentity(
        episode_uuid=episode_uuid,
        legacy_episode_index=0,
        lifecycle_owner_id="fixture",
        prepared_at=1.0,
        prepared_clock_domain_id="identity_clock",
    )


def _translation(x: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[0, 3] = x
    return transform


def _v23_pose_source() -> PoseSourceProfile:
    return PoseSourceProfile(
        source_id="real_teleop_progress_fixture",
        source_family="real_robot",
        source_description="offline RealTeleop progress integration fixture",
        topology="dual_arm",
        raw_absolute_frame_by_side={
            "left": "left_rm65_base",
            "right": "right_rm65_base",
        },
        raw_to_desk_transform_by_side={
            "left": np.eye(4, dtype=float),
            "right": np.eye(4, dtype=float),
        },
    )


def _v23_sample(
    *,
    episode_index: int = 0,
    sample_index: int = 0,
) -> CollectorSample:
    arm_states: dict[str, dict[str, object]] = {}
    observation_state = np.zeros(38, dtype=np.float32)
    action = np.zeros(26, dtype=np.float32)
    action[6] = 1.0
    action[13] = 1.0
    for side, pose_x in (("left", 0.3), ("right", 0.4)):
        pose = _translation(pose_x)
        arm_states[side] = {
            "current_qpos": np.zeros(6, dtype=float),
            "current_tcp_pose": pose,
            "actual_qpos": np.zeros(6, dtype=float),
            "actual_hardware_qpos": np.zeros(6, dtype=float),
            "actual_tcp_pose": pose,
            "actual_tcp_pose_source": "hardware_qpos_fk",
            "actual_state_read_success": True,
            "actual_state_valid": True,
            "actual_state_stale": False,
            "actual_state_sequence": 7,
            "actual_state_timestamp": 9.98,
            "source_timestamp": 9.98,
            "available_timestamp": 9.99,
            "source_clock": "local_wall_time",
            "available_clock": "local_wall_time",
            "causal_measured_state_eligible": True,
            "causal_measured_state_rejection_reason": "accepted",
        }
        state_slice = slice(6, 13) if side == "left" else slice(25, 32)
        pose7 = np.asarray(
            [pose_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        observation_state[state_slice] = pose7
    return CollectorSample(
        index=sample_index,
        episode_index=episode_index,
        frame_index=0,
        timestamp=10.0,
        session_state="running",
        control_hz=15.0,
        action=action,
        observation_state=observation_state,
        observation_last_action=action.copy(),
        arm_state_source="causal_measured_hardware",
        active_sides=("left", "right"),
        arm_states=arm_states,
        hand_states={"left": {}, "right": {}},
    )


def test_profile_loader_accepts_linear_initial_adapter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "progress.json"
    path.write_text(
        json.dumps(_profile().to_payload(), ensure_ascii=False),
        encoding="utf-8",
    )

    loaded = load_real_teleop_progress_profile(str(path))

    assert loaded.semantic_content_hash == _profile().semantic_content_hash
    validate_real_teleop_initial_progress_profile(
        loaded,
        collection_task_id="task-a",
    )
    # The formal dataset task/name is independent from the selected
    # long-horizon profile identity.
    validate_real_teleop_initial_progress_profile(
        loaded,
        collection_task_id="other-task",
    )


def test_profile_loader_accepts_multi_step_linear_profile(
    tmp_path: Path,
) -> None:
    profile = TaskProgressProfileV1(
        profile_id="multi-step",
        profile_version="1",
        task_id="task-a",
        steps=(
            ProgressStepDefinitionV1(
                progress_step_id="pick",
                semantic_type="PICK",
                display_name="抓取",
                ordinal=1,
            ),
            ProgressStepDefinitionV1(
                progress_step_id="place",
                semantic_type="PLACE",
                display_name="放置",
                ordinal=2,
            ),
        ),
        initial_step_id="pick",
        terminal_step_ids=("place",),
    )
    path = tmp_path / "multi_progress.json"
    path.write_text(
        json.dumps(profile.to_payload(), ensure_ascii=False),
        encoding="utf-8",
    )

    loaded = load_real_teleop_progress_profile(str(path))

    assert tuple(step.progress_step_id for step in loaded.steps) == (
        "pick",
        "place",
    )


def test_real_teleop_rejects_rooted_tree_profile() -> None:
    profile = TaskProgressProfileV1(
        profile_id="tree-profile",
        profile_version="1",
        task_id="task-a",
        topology_kind="ROOTED_TREE",
        steps=(
            ProgressStepDefinitionV1(
                progress_step_id="root",
                semantic_type="START",
                display_name="开始",
                ordinal=1,
                allowed_next_step_ids=("left", "right"),
            ),
            ProgressStepDefinitionV1(
                progress_step_id="left",
                semantic_type="PLACE_OBJECT",
                display_name="左分支",
                ordinal=2,
            ),
            ProgressStepDefinitionV1(
                progress_step_id="right",
                semantic_type="PLACE_OBJECT",
                display_name="右分支",
                ordinal=3,
            ),
        ),
        initial_step_id="root",
        terminal_step_ids=("left", "right"),
    )

    with pytest.raises(ValueError, match="only accepts a LINEAR"):
        validate_real_teleop_initial_progress_profile(profile)


def test_formal_profile_is_explicitly_opted_in_by_cli() -> None:
    parser = build_real_teleop_argument_parser()

    defaults = parser.parse_args([])
    enabled = parser.parse_args(
        [
            "--collection-progress-profile",
            str(_FORMAL_PROGRESS_PROFILE),
            "--collection-progress-gui",
        ]
    )
    catalog = parser.parse_args(
        [
            "--collection-progress-profile",
            "catalog:dmtac_rm65_bimanual_position_v1_progress",
        ]
    )

    assert defaults.collection_progress_profile is None
    assert defaults.collection_progress_gui is False
    assert enabled.collection_progress_gui is True
    assert enabled.collection_progress_profile.task_id == (
        "dmtac_rm65_bimanual_position_v1"
    )
    assert enabled.collection_progress_profile.steps[0].progress_step_id == (
        "execute_task"
    )
    assert catalog.collection_progress_profile == (
        enabled.collection_progress_profile
    )


def test_managed_stage_profile_and_recovery_entry_are_explicit() -> None:
    profile = load_real_teleop_progress_profile(
        "catalog:dmtac_rm65_bimanual_position_v1_progress"
    )
    stages = resolve_real_teleop_task_stage_profile(profile)

    assert stages is not None
    assert tuple(
        item.stage_id for item in stages.stages_for_step("execute_task")
    ) == (
        "execute_task.approach",
        "execute_task.manipulate",
        "execute_task.finish",
    )
    with pytest.raises(ValueError, match="RECOVERY_FROM_FAILURE"):
        resolve_real_teleop_progress_entry(
            progress_profile=profile,
            stage_profile=stages,
            entry_step_id="execute_task",
            entry_stage_id="execute_task.manipulate",
            entry_condition="nominal",
        )
    assert resolve_real_teleop_progress_entry(
        progress_profile=profile,
        stage_profile=stages,
        entry_step_id="execute_task",
        entry_stage_id="execute_task.manipulate",
        entry_condition="recovery_from_failure",
    ) == (
        "execute_task",
        "execute_task.manipulate",
        "RECOVERY_FROM_FAILURE",
    )


def test_progress_gui_projection_is_independently_reversible(
    tmp_path: Path,
) -> None:
    profile = _multi_step_profile()
    disabled = build_real_teleop_progress_ui_payload(
        profile=profile,
        episode=None,
        availability=ProgressControlAvailability(),
        gui_enabled=False,
    )
    idle = build_real_teleop_progress_ui_payload(
        profile=profile,
        episode=None,
        availability=ProgressControlAvailability(),
        gui_enabled=True,
    )
    episode = RealTeleopProgressEpisode(
        dataset_root=tmp_path,
        identity=_identity(),
        profile=profile,
    )
    episode.bind(started_at=10.0)
    active = build_real_teleop_progress_ui_payload(
        profile=profile,
        episode=episode,
        availability=ProgressControlAvailability(
            enabled=True,
            recording_active=True,
            phase="ACTIVE_ATTEMPT",
            allowed_next_step_ids=episode.allowed_next_step_ids,
        ),
        gui_enabled=True,
    )

    assert disabled["enabled"] is False
    assert idle["enabled"] is True
    assert idle["phase"] == "IDLE"
    assert idle["active_controls"] == []
    assert active["enabled"] is True
    assert active["phase"] == "ACTIVE_ATTEMPT"
    assert active["current_progress_step_id"] == "pick"
    assert active["next_steps"] == [
        {"progress_step_id": "place", "display_name": "放置"}
    ]
    episode.close_unsealed()


def test_progress_gui_is_visible_and_selectable_before_runtime_home(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        collection_progress_gui=True,
        collection_progress_profile=None,
        collection_task_id="dmtac_rm65_bimanual_position_v1",
        dataset_root=tmp_path,
    )

    connecting = build_real_teleop_startup_progress_ui_payload(
        args=args,
        dataset_recorder=None,
        profile_selection_enabled=False,
    )
    initial = build_real_teleop_startup_progress_ui_payload(
        args=args,
        dataset_recorder=None,
    )

    assert connecting["enabled"] is True
    assert connecting["active_controls"] == []
    assert "设备仍在启动" in str(connecting["message"])
    assert initial["enabled"] is True
    assert initial["configured"] is False
    assert initial["phase"] == "ABSENT"
    assert initial["active_controls"] == [
        "progress_profile_select",
        "progress_arm_control_mode_select",
    ]
    assert [
        item["profile_id"]
        for item in initial["available_profiles"]
    ] == [
        "dmtac_bimanual_writing_acquire_present_window_v1",
        "dmtac_rm65_bimanual_position_v1_progress",
        "dmtac_rm65_bimanual_position_writing_five_stage_progress",
        "dmtac_rm65_bimanual_position_writing_four_stage_progress",
        "dmtac_rm65_bimanual_position_writing_three_stage_progress",
        "example_three_stage_pick_place_progress",
        "test_dmtac_rm65_bimanual_position_v1_three_stage_progress",
    ]

    selected = select_real_teleop_managed_progress_profile(
        args=args,
        dataset_recorder=None,
        profile_id="dmtac_rm65_bimanual_position_v1_progress",
    )
    configured = build_real_teleop_startup_progress_ui_payload(
        args=args,
        dataset_recorder=None,
    )

    assert args.collection_progress_profile == selected
    assert configured["enabled"] is True
    assert configured["configured"] is True
    assert configured["profile_id"] == selected.profile_id
    assert configured["phase"] == "IDLE"


def test_progress_gui_keeps_profile_selection_available_for_any_formal_task(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        collection_progress_gui=True,
        collection_progress_profile=None,
        collection_task_id="dmtac_impedance_15_40hz",
        dataset_root=tmp_path,
    )

    payload = build_real_teleop_startup_progress_ui_payload(
        args=args,
        dataset_recorder=None,
    )

    assert payload["configured"] is False
    assert len(payload["available_profiles"]) == 7
    assert payload["profile_selection_enabled"] is True
    assert "长程任务进度 profile" in str(payload["message"])
    assert "dmtac_impedance_15_40hz" not in str(payload["message"])

    selected = select_real_teleop_managed_progress_profile(
        args=args,
        dataset_recorder=None,
        profile_id="example_three_stage_pick_place_progress",
    )

    assert selected.task_id == "example_three_stage_pick_place"
    assert args.collection_task_id == "dmtac_impedance_15_40hz"
    assert args.collection_progress_profile == selected
    configured = build_real_teleop_startup_progress_ui_payload(
        args=args,
        dataset_recorder=None,
    )
    assert configured["configured"] is True
    assert configured["profile_id"] == selected.profile_id


def test_four_stage_writing_is_default_only_for_blank_gui_dataset(
    tmp_path: Path,
) -> None:
    blank_root = tmp_path / "blank"
    populated_root = tmp_path / "populated_profileless"
    (populated_root / "meta").mkdir(parents=True)
    (populated_root / "meta" / "episodes.jsonl").write_text(
        '{"episode_index":0}\n',
        encoding="utf-8",
    )
    bound_root = tmp_path / "bound"
    bound_profile = load_real_teleop_managed_progress_profile(
        "dmtac_rm65_bimanual_position_v1_progress"
    )
    bind_dataset_progress_profile(bound_root, bound_profile)

    default_profile = resolve_real_teleop_default_progress_profile(
        dataset_root=blank_root,
        requested_profile=None,
        gui_enabled=True,
    )

    assert default_profile is not None
    assert default_profile.profile_id == (
        DEFAULT_REAL_TELEOP_MANAGED_PROGRESS_PROFILE_ID
    )
    assert len(default_profile.steps) == 4
    assert resolve_real_teleop_default_progress_profile(
        dataset_root=blank_root,
        requested_profile=bound_profile,
        gui_enabled=True,
    ) == bound_profile
    assert resolve_real_teleop_default_progress_profile(
        dataset_root=blank_root,
        requested_profile=None,
        gui_enabled=False,
    ) is None
    assert resolve_real_teleop_default_progress_profile(
        dataset_root=populated_root,
        requested_profile=None,
        gui_enabled=True,
    ) is None
    assert resolve_real_teleop_default_progress_profile(
        dataset_root=bound_root,
        requested_profile=None,
        gui_enabled=True,
    ) is None


def test_switching_to_new_dataset_unlocks_profile_without_mutating_old_recorder(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old_dataset"
    new_root = tmp_path / "new_dataset"
    old_profile = load_real_teleop_progress_profile(_FORMAL_PROGRESS_PROFILE)
    old_selection_attempts: list[TaskProgressProfileV1] = []
    old_recorder = SimpleNamespace(
        dataset_root=old_root,
        writer_storage_config=SimpleNamespace(task_id="formal-task"),
        progress_profile=old_profile,
        progress_profile_selection_locked=True,
        progress_profile_binding_source="dataset_binding",
        select_progress_profile=old_selection_attempts.append,
    )
    args = SimpleNamespace(
        collection_progress_gui=True,
        collection_progress_profile=old_profile,
        collection_task_id="formal-task",
        dataset_root=new_root,
    )

    switched = build_real_teleop_startup_progress_ui_payload(
        args=args,
        dataset_recorder=old_recorder,
    )
    selected = select_real_teleop_managed_progress_profile(
        args=args,
        dataset_recorder=old_recorder,
        profile_id="example_three_stage_pick_place_progress",
    )

    assert switched["profile_selection_locked"] is False
    assert "progress_profile_select" in switched["active_controls"]
    assert selected.profile_id == "example_three_stage_pick_place_progress"
    assert args.collection_progress_profile == selected
    assert old_selection_attempts == []


def test_progress_episode_prepare_bind_complete_and_seal(tmp_path: Path) -> None:
    episode = RealTeleopProgressEpisode(
        dataset_root=tmp_path,
        identity=_identity(),
        profile=_profile(),
    )

    assert episode.metadata_reference["manifest_ref"].endswith(
        "/episode-progress-a/manifest.cjson"
    )
    assert episode.store.facts == ()
    episode.bind(started_at=10.0)
    assert episode.controller.state.phase == "ACTIVE_ATTEMPT"

    episode.terminate(reason="manual_key_stop", effective_at=12.0)
    assert episode.controller.state.phase == "TERMINATED"
    report = episode.seal(sealed_at=13.0)

    assert report.manifest.fact_count == 2
    assert episode.store.sealed
    assert (episode.episode_dir / "manifest.cjson").is_file()


def test_manual_stop_before_terminal_step_aborts_instead_of_claiming_success(
    tmp_path: Path,
) -> None:
    profile = TaskProgressProfileV1(
        profile_id="manual-stop-multi",
        profile_version="1",
        task_id="task-a",
        steps=(
            ProgressStepDefinitionV1(
                progress_step_id="pick",
                semantic_type="PICK",
                display_name="抓取",
                ordinal=1,
            ),
            ProgressStepDefinitionV1(
                progress_step_id="place",
                semantic_type="PLACE",
                display_name="放置",
                ordinal=2,
            ),
        ),
        initial_step_id="pick",
        terminal_step_ids=("place",),
    )
    episode = RealTeleopProgressEpisode(
        dataset_root=tmp_path,
        identity=_identity("episode-progress-nonterminal-stop"),
        profile=profile,
    )
    episode.bind(started_at=10.0)

    episode.terminate(reason="manual_key_stop", effective_at=11.0)

    terminal = episode.store.facts[-1].payload
    assert terminal.termination_cause == "MANUAL_STOP"
    assert terminal.active_attempt_disposition == "ABORTED"
    assert terminal.full_task_outcome == "UNKNOWN"
    episode.seal(sealed_at=12.0)


def test_manual_stop_while_failure_paused_closes_progress_without_success(
    tmp_path: Path,
) -> None:
    episode = RealTeleopProgressEpisode(
        dataset_root=tmp_path,
        identity=_identity("episode-progress-failure-stop"),
        profile=_profile(),
    )
    episode.bind(started_at=10.0)
    effect = episode.report_failure(
        intent_id="failure-before-stop",
        effective_at=11.0,
    )
    episode.record_applied_hold_pause(
        effect_request=effect,
        requested_at=11.0,
        application_hold_effective_at=11.1,
        formal_capture_pause_effective_at=11.0,
        available_at=11.2,
    )

    episode.terminate(reason="manual_key_stop", effective_at=12.0)

    terminal = episode.store.facts[-1].payload
    assert terminal.termination_cause == "MANUAL_STOP"
    assert terminal.active_attempt_disposition == "NO_ACTIVE_ATTEMPT"
    assert terminal.full_task_outcome == "UNKNOWN"
    episode.seal(sealed_at=13.0)


@pytest.mark.parametrize(
    "reason",
    ["force_collision", "joint_hard_limit", "runtime_fault", "runtime_estop"],
)
def test_progress_episode_safety_stop_aborts_without_claiming_success(
    tmp_path: Path,
    reason: str,
) -> None:
    episode = RealTeleopProgressEpisode(
        dataset_root=tmp_path,
        identity=_identity("episode-progress-safety"),
        profile=_profile(),
    )
    episode.bind(started_at=10.0)

    episode.terminate(reason=reason, effective_at=11.0)

    terminal = episode.store.facts[-1].payload
    assert terminal.termination_cause == "SAFETY_STOP"
    assert terminal.active_attempt_disposition == "ABORTED"
    assert terminal.full_task_outcome == "UNKNOWN"
    episode.close_unsealed()


@pytest.mark.parametrize("reason", ["quit_requested", "shutdown"])
def test_progress_episode_interruption_is_not_task_success(
    tmp_path: Path,
    reason: str,
) -> None:
    episode = RealTeleopProgressEpisode(
        dataset_root=tmp_path,
        identity=_identity(f"episode-progress-{reason}"),
        profile=_profile(),
    )
    episode.bind(started_at=10.0)

    episode.terminate(reason=reason, effective_at=11.0)

    terminal = episode.store.facts[-1].payload
    assert terminal.termination_cause == "MANUAL_STOP"
    assert terminal.active_attempt_disposition == "ABORTED"
    assert terminal.full_task_outcome == "UNKNOWN"
    episode.close_unsealed()


def test_progress_episode_discard_removes_only_owned_unsealed_directory(
    tmp_path: Path,
) -> None:
    episode = RealTeleopProgressEpisode(
        dataset_root=tmp_path,
        identity=_identity("episode-progress-discard"),
        profile=_profile(),
    )
    sibling = episode.episode_dir.parent / "keep.txt"
    sibling.parent.mkdir(parents=True, exist_ok=True)
    sibling.write_text("keep", encoding="utf-8")

    episode.discard()

    assert not episode.episode_dir.exists()
    assert sibling.read_text(encoding="utf-8") == "keep"


class _Writer:
    next_episode_index = 0

    def __init__(self) -> None:
        self.started: list[dict[str, object]] = []
        self.finalized: list[int] = []
        self.discarded: list[int] = []

    def start_episode(self, *, metadata):
        self.started.append(dict(metadata))
        return 0

    def finalize_episode(self, episode_index: int, *, summary=None):
        self.finalized.append(int(episode_index))
        return None

    def discard_episode(self, episode_index: int) -> bool:
        self.discarded.append(int(episode_index))
        return True


class _WriterLifecycle:
    def flush(self) -> None:
        return None

    def storage_stats(self) -> dict[str, object]:
        return {"enabled": False, "routes": {}}


def _recorder(
    tmp_path: Path,
    *,
    progress: bool,
    episode_uuid: str = "episode-progress-recorder",
) -> RealTeleopDatasetRecorder:
    recorder = RealTeleopDatasetRecorder.__new__(RealTeleopDatasetRecorder)
    recorder.args = SimpleNamespace(
        v25p_native_shadow=False,
        collection_task_id="task-a",
        collection_progress_profile=(_profile() if progress else None),
        collection_progress_gui=progress,
    )
    recorder.dataset_root = tmp_path
    recorder.writer = _Writer()
    recorder.writer_runtime = None
    recorder.writer_observer = _WriterLifecycle()
    recorder.collection_worker_mode = "thread"
    recorder.camera_sources = {}
    recorder.sample_builder = SimpleNamespace(
        frame_index=0,
        set_episode=lambda _episode_index: None,
    )
    recorder.process_worker = None
    recorder.active_episode_index = None
    recorder.active_started_at = None
    recorder.finalize_tasks = []
    recorder.gripper_force_contract = None
    recorder.actual_feedback_warmup_status = lambda _now: (True, "ready")
    recorder.collection_frequency_status = lambda: {"sample_hz": 15.0}
    recorder.pause_summary = lambda: {}
    recorder._record_timestamps = deque()
    recorder._episode_uuid_factory = lambda: episode_uuid
    recorder.control_trace_episode_sink = None
    recorder.v25p_sidecar_store = None
    recorder.v25p_native_shadow = None
    recorder._active_control_trace_preparation = None
    recorder._active_episode_identity = None
    recorder._active_progress_episode = None
    recorder.progress_profile = _profile() if progress else None
    recorder._progress_profile_dataset_locked = False
    recorder._progress_profile_binding_source = "blank_dataset"
    recorder._sample_acceptance_receipt_buffer = (
        SampleAcceptanceReceiptBuffer(capacity=256 if progress else 0)
    )
    recorder._sample_future = None
    recorder._collection_integrity_reasons = []
    recorder._v25p_native_shadow_active = False
    recorder._v25p_native_shadow_error = ""
    recorder.saved_episodes = 0
    recorder.discarded_episodes = 0
    recorder._pause_started_at = None
    recorder._pause_started_monotonic = None
    recorder._pause_reason = None
    recorder._pause_reasons = []
    recorder._pause_started_frame_index = None
    recorder._pause_started_step_index = None
    recorder._pause_started_last_state = None
    recorder._pause_segments = []
    recorder._paused_accumulated_sec = 0.0
    return recorder


def _use_actual_v23_writer(
    recorder: RealTeleopDatasetRecorder,
    *,
    dataset_root: Path,
) -> None:
    recorder.writer = LeRobotV21Writer(
        LeRobotV21WriterConfig(
            root_dir=dataset_root,
            task_id="task-a",
            write_video=False,
            policy_io_layout=PolicyIOLayout(
                arm_reference_frame_by_side={
                    "left": "left_rm65_base",
                    "right": "right_rm65_base",
                }
            ),
            pose_frame_contract=PoseFrameContract.v23_forced_raw(
                _v23_pose_source()
            ),
        )
    )


def _write_and_finalize_actual_v23_episode(
    recorder: RealTeleopDatasetRecorder,
    *,
    reason: str = "manual_key_stop",
) -> RealDatasetFinalizeTask:
    episode_index = recorder.start_episode(
        {"fixture": "actual_v23_writer"},
        now=10.0,
    )
    for job in recorder.writer.plan_sample_jobs(
        _v23_sample(
            episode_index=episode_index,
            sample_index=episode_index,
        )
    ):
        recorder.writer.handle_job(job)
    recorder.sample_builder.frame_index = 1
    task = recorder.stop_episode(
        reason=reason,
        step_index=1,
        last_state="RUNNING",
        defer_finalize_start=True,
    )
    assert task is not None
    task.mark_started()
    recorder._run_finalize_task(task)
    assert task.success
    return task


def _actual_v23_sigint_child(
    dataset_root: str,
    ready_and_result,
) -> None:
    root = Path(dataset_root)
    recorder = _recorder(
        root,
        progress=True,
        episode_uuid="progress-process-sigint",
    )
    _use_actual_v23_writer(recorder, dataset_root=root)
    episode_index = recorder.start_episode(
        {"fixture": "actual_v23_process_sigint"},
        now=10.0,
    )
    for job in recorder.writer.plan_sample_jobs(
        _v23_sample(
            episode_index=episode_index,
            sample_index=episode_index,
        )
    ):
        recorder.writer.handle_job(job)
    recorder.sample_builder.frame_index = 1
    try:
        ready_and_result.send(("ready", episode_index))
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        task = recorder.stop_episode(
            reason="shutdown",
            step_index=1,
            last_state="RUNNING",
            defer_finalize_start=True,
        )
        assert task is not None
        assert task.progress_episode is not None
        task.mark_started()
        recorder._run_finalize_task(task)
        assert task.success
        ready_and_result.send(
            (
                "sealed",
                str(task.progress_episode.episode_dir),
                task.episode_identity.semantic_content_hash,
            )
        )
    finally:
        ready_and_result.close()


def test_recorder_progress_is_additive_and_seals_after_legacy_finalize(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path, progress=True)

    assert recorder.start_episode({"existing": "metadata"}, now=10.0) == 0
    assert read_dataset_progress_profile(tmp_path) == _profile()
    assert recorder.progress_profile_selection_locked
    metadata = recorder.writer.started[0]
    assert metadata["existing"] == "metadata"
    assert metadata["progress_episode"]["episode_uuid"] == (
        "episode-progress-recorder"
    )
    recorder.sample_builder.frame_index = 2

    task = recorder.stop_episode(
        reason="manual_key_stop",
        step_index=20,
        last_state="RUNNING",
        defer_finalize_start=True,
    )

    assert task is not None
    assert task.progress_episode is not None
    assert task.progress_episode.controller.state.phase == "TERMINATED"
    task.mark_started()
    recorder._run_finalize_task(task)
    assert task.success
    assert task.progress_error == ""
    assert task.progress_finalize_report is not None
    assert task.progress_episode.store.sealed
    assert recorder.writer.finalized == [0]


def test_idle_recorder_profile_selection_is_mutable_until_first_episode(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path, progress=True)
    first = _multi_step_profile()
    second = TaskProgressProfileV1(
        profile_id="pick-place-progress-v2",
        profile_version="2",
        task_id="task-a",
        steps=first.steps,
        initial_step_id=first.initial_step_id,
        terminal_step_ids=first.terminal_step_ids,
    )

    recorder.select_progress_profile(first)
    assert recorder.progress_profile == first
    assert recorder.progress_profile_selection_locked is False
    recorder.select_progress_profile(second)
    assert recorder.progress_profile == second

    recorder.start_episode({}, now=10.0)
    assert recorder.progress_profile_selection_locked is True
    recorder.sample_builder.frame_index = 1
    finalize_task = recorder.stop_episode(
        reason="manual_key_stop",
        step_index=1,
        last_state="RUNNING",
        defer_finalize_start=True,
    )
    assert finalize_task is not None
    finalize_task.mark_started()
    recorder._run_finalize_task(finalize_task)
    assert finalize_task.success
    with pytest.raises(
        DatasetProgressProfileBindingError,
        match="differs from the profile already used",
    ):
        recorder.select_progress_profile(first)
    assert (
        tmp_path / DATASET_PROGRESS_PROFILE_RELATIVE_PATH
    ).is_file()


def test_profileless_episode_locks_optional_profile_selector_after_start(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path, progress=False)

    recorder.start_episode({}, now=10.0)

    assert recorder.progress_profile is None
    assert recorder.progress_profile_selection_locked is True
    assert recorder.progress_profile_binding_source == (
        "populated_profileless_dataset"
    )


def test_legacy_finalize_failure_leaves_progress_unsealed(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path, progress=True)
    recorder.start_episode({}, now=10.0)
    recorder.sample_builder.frame_index = 2
    task = recorder.stop_episode(
        reason="manual_key_stop",
        step_index=20,
        last_state="RUNNING",
        defer_finalize_start=True,
    )
    assert task is not None
    assert task.progress_episode is not None

    def fail_finalize(_episode_index: int, *, summary=None):
        del summary
        raise OSError("injected legacy finalize failure")

    recorder.writer.finalize_episode = fail_finalize
    task.mark_started()
    recorder._run_finalize_task(task)

    assert task.done
    assert not task.success
    assert "injected legacy finalize failure" in task.error
    assert not task.progress_episode.store.sealed
    assert not (task.progress_episode.episode_dir / "manifest.cjson").exists()


def test_process_worker_mode_uses_the_same_progress_lifecycle(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path, progress=True)
    recorder.collection_worker_mode = "process"
    recorder.process_worker = SimpleNamespace(
        native_stream_contracts={},
        native_external_stream_bindings={},
        v25p_shadow_config=None,
        flush=lambda: None,
    )

    recorder.start_episode({}, now=10.0)
    recorder._process_episode_submitted_count = 2
    task = recorder.stop_episode(
        reason="manual_key_stop",
        step_index=20,
        last_state="RUNNING",
        defer_finalize_start=True,
    )
    assert task is not None
    task.mark_started()
    recorder._run_finalize_task(task)

    assert task.success
    assert task.progress_finalize_report is not None
    assert task.progress_episode.store.sealed


def test_actual_v23_writer_and_progress_artifacts_share_one_episode_identity(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    recorder = _recorder(tmp_path, progress=True)
    _use_actual_v23_writer(recorder, dataset_root=tmp_path)
    task = _write_and_finalize_actual_v23_episode(recorder)
    assert task.progress_episode is not None
    validation = validate_v23_dataset(tmp_path)
    assert validation.ok, validation.errors
    assert validation.episode_count == 1
    assert validation.frame_count == 1

    episode_record = json.loads(
        (tmp_path / "meta" / "episodes.jsonl").read_text(encoding="utf-8")
    )
    progress_reference = episode_record["metadata"]["progress_episode"]
    assert progress_reference["episode_uuid"] == task.episode_identity.episode_uuid
    assert progress_reference["episode_identity_hash"] == (
        task.episode_identity.semantic_content_hash
    )
    assert (tmp_path / episode_record["parquet_path"]).is_file()

    loaded = load_sealed_progress_episode(
        task.progress_episode.episode_dir,
        expected_episode_identity_hash=progress_reference[
            "episode_identity_hash"
        ],
        expected_profile_hash=progress_reference[
            "task_progress_profile_hash"
        ],
        expected_binding_hash=progress_reference[
            "progress_episode_binding_hash"
        ],
    )
    assert loaded.manifest.fact_count == 2
    assert loaded.timeline.phase == "TERMINATED"
    assert loaded.binding.bound_clock_domain_id == "local_wall_time"
    assert all(
        fact.effective_clock_domain_id == "local_wall_time"
        for fact in loaded.facts
    )
    assert all(
        previous.effective_at <= current.effective_at
        for previous, current in zip(loaded.facts, loaded.facts[1:])
    )
    assert loaded.manifest.sealed_clock_domain_id == "local_wall_time"


@pytest.mark.parametrize(
    "detachment_scope",
    (PAYLOAD_ONLY_DETACH, WHOLE_PREFIX_DETACH),
)
def test_actual_v23_core_validation_accepts_only_verified_detached_tactile(
    tmp_path: Path,
    detachment_scope: str,
) -> None:
    pytest.importorskip("pyarrow")
    recorder = _recorder(tmp_path, progress=True)
    _use_actual_v23_writer(recorder, dataset_root=tmp_path)
    _write_and_finalize_actual_v23_episode(recorder)

    manifest_relative = (
        "tactile/observation.hand.left.tactile/chunk-000/"
        "episode_000000/manifest.json"
    )
    payload_relative = (
        "tactile/observation.hand.left.tactile/chunk-000/"
        "episode_000000/L_RAW.uint8.bin"
    )
    payload_path = tmp_path / payload_relative
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(b"fixture")
    (tmp_path / manifest_relative).write_text(
        json.dumps(
            {
                "format": "binary_stream_v1",
                "side": "left",
                "frame_count": 1,
                "sensor_keys": ["L_RAW"],
                "streams": {
                    "L_RAW": {
                        "path": payload_relative,
                        "dtype": "uint8",
                        "shape": [7],
                        "frame_count": 1,
                        "encoding": "raw_c_order",
                        "compression": "none",
                    }
                },
                "frames": [
                    {
                        "sample_frame_index": 0,
                        "tactile_frame_index": 0,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    catalog_path = tmp_path / "meta" / "episodes.jsonl"
    episode_record = json.loads(catalog_path.read_text(encoding="utf-8"))
    episode_record["tactile_paths"] = {"left": manifest_relative}
    catalog_path.write_text(
        json.dumps(episode_record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inventory = build_modality_artifact_inventory(tmp_path)
    if detachment_scope == WHOLE_PREFIX_DETACH:
        unreceipted_tree = tmp_path.parent / f"{tmp_path.name}-unreceipted-tactile"
        os.replace(tmp_path / "tactile", unreceipted_tree)
        unreceipted_core = validate_v23_core_dataset(
            tmp_path,
            verified_inventories=(inventory,),
            verified_detachment_scopes={"tactile": WHOLE_PREFIX_DETACH},
        )
        assert not unreceipted_core.ok
        assert any(
            "durable retirement receipt" in error
            for error in unreceipted_core.errors
        )
        os.replace(unreceipted_tree, tmp_path / "tactile")
    cold_root = tmp_path.parent / f"{tmp_path.name}-cold"
    cold_root.mkdir()
    detach_plan = plan_modality_detach(
        tmp_path,
        cold_root,
        detachment_scope=detachment_scope,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    assert detach_plan.ready, detach_plan.errors
    detached_result = detach_modality(
        tmp_path,
        cold_root,
        expected_plan_hash=detach_plan.plan_hash,
        confirm_dataset_fingerprint=detach_plan.dataset_fingerprint,
        detachment_scope=detachment_scope,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )
    assert detached_result.inventory_hash == inventory.semantic_content_hash

    availability = resolve_modality_availability(
        tmp_path,
        inventory,
        detachment_scope=detachment_scope,
    )
    assert availability.status == DETACHED_VERIFIED
    assert availability.core_training_eligible is True
    assert availability.tactile_read_eligible is False
    strict = validate_v23_dataset(tmp_path)
    assert not strict.ok
    assert any("missing" in error for error in strict.errors)
    core = validate_v23_core_dataset(
        tmp_path,
        verified_inventories=(inventory,),
        verified_detachment_scopes={"tactile": detachment_scope},
    )
    assert core.ok, core.errors
    assert any("DETACHED_VERIFIED" in warning for warning in core.warnings)

    rehydrate_plan = plan_modality_rehydrate(
        tmp_path,
        cold_root,
        inventory.semantic_content_hash,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    assert rehydrate_plan.ready, rehydrate_plan.errors
    restored_result = rehydrate_modality(
        tmp_path,
        cold_root,
        inventory.semantic_content_hash,
        expected_plan_hash=rehydrate_plan.plan_hash,
        confirm_dataset_fingerprint=rehydrate_plan.dataset_fingerprint,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )
    assert restored_result.final_status == "SELF_CONTAINED"
    restored_validation = validate_v23_dataset(tmp_path)
    assert restored_validation.ok, restored_validation.errors


def test_actual_v23_writer_disabled_path_emits_no_progress_artifacts(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    recorder = _recorder(tmp_path, progress=False)
    _use_actual_v23_writer(recorder, dataset_root=tmp_path)

    _write_and_finalize_actual_v23_episode(recorder)

    validation = validate_v23_dataset(tmp_path)
    assert validation.ok, validation.errors
    episode_record = json.loads(
        (tmp_path / "meta" / "episodes.jsonl").read_text(encoding="utf-8")
    )
    assert "progress_episode" not in episode_record["metadata"]
    assert not (tmp_path / "meta" / "progress").exists()


def test_actual_v23_writer_shutdown_seals_an_interrupted_progress_attempt(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    recorder = _recorder(tmp_path, progress=True)
    _use_actual_v23_writer(recorder, dataset_root=tmp_path)

    task = _write_and_finalize_actual_v23_episode(
        recorder,
        reason="shutdown",
    )

    assert task.progress_episode is not None
    loaded = load_sealed_progress_episode(
        task.progress_episode.episode_dir,
        expected_episode_identity_hash=task.episode_identity.semantic_content_hash,
    )
    assert loaded.timeline.termination_cause == "MANUAL_STOP"
    assert loaded.timeline.full_task_outcome == "UNKNOWN"
    assert loaded.timeline.attempts[0].outcome == "ABORTED"
    validation = validate_v23_dataset(tmp_path)
    assert validation.ok, validation.errors


def test_actual_v23_writer_restart_continues_with_a_new_bound_identity(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    first = _recorder(
        tmp_path,
        progress=True,
        episode_uuid="progress-before-restart",
    )
    _use_actual_v23_writer(first, dataset_root=tmp_path)
    first_task = _write_and_finalize_actual_v23_episode(first)

    restarted = _recorder(
        tmp_path,
        progress=True,
        episode_uuid="progress-after-restart",
    )
    _use_actual_v23_writer(restarted, dataset_root=tmp_path)
    second_task = _write_and_finalize_actual_v23_episode(restarted)

    validation = validate_v23_dataset(tmp_path)
    assert validation.ok, validation.errors
    assert validation.episode_count == 2
    assert validation.frame_count == 2
    assert first_task.episode_index == 0
    assert second_task.episode_index == 1
    assert first_task.episode_identity != second_task.episode_identity
    for task in (first_task, second_task):
        assert task.progress_episode is not None
        loaded = load_sealed_progress_episode(
            task.progress_episode.episode_dir,
            expected_episode_identity_hash=(
                task.episode_identity.semantic_content_hash
            ),
        )
        assert loaded.timeline.full_task_outcome == "SUCCESS"

    episode_rows = [
        json.loads(line)
        for line in (tmp_path / "meta" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["episode_index"] for row in episode_rows] == [0, 1]
    assert {
        row["metadata"]["progress_episode"]["episode_identity_hash"]
        for row in episode_rows
    } == {
        first_task.episode_identity.semantic_content_hash,
        second_task.episode_identity.semantic_content_hash,
    }


def test_actual_v23_writer_process_sigint_seals_before_exit(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_actual_v23_sigint_child,
        args=(str(tmp_path), send),
        name="actual-v23-sigint-fixture",
    )
    process.start()
    send.close()
    try:
        assert receive.poll(15.0), "child did not prepare the Episode in time"
        assert receive.recv() == ("ready", 0)
        os.kill(process.pid, signal.SIGINT)
        assert receive.poll(30.0), "child did not seal after SIGINT"
        status, progress_dir, identity_hash = receive.recv()
        assert status == "sealed"
        process.join(timeout=10.0)
        assert process.exitcode == 0

        loaded = load_sealed_progress_episode(
            progress_dir,
            expected_episode_identity_hash=identity_hash,
        )
        assert loaded.timeline.termination_cause == "MANUAL_STOP"
        assert loaded.timeline.attempts[0].outcome == "ABORTED"
        validation = validate_v23_dataset(tmp_path)
        assert validation.ok, validation.errors
        assert validation.episode_count == 1
        assert validation.frame_count == 1
    finally:
        receive.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)


def test_recorder_disabled_path_emits_no_progress_metadata_or_files(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path, progress=False)

    recorder.start_episode({"existing": "metadata"}, now=10.0)

    assert "progress_episode" not in recorder.writer.started[0]
    assert not (tmp_path / "meta" / "progress").exists()


def test_zero_frame_episode_discards_progress_with_legacy_episode(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path, progress=True)
    recorder.start_episode({}, now=10.0)
    progress_dir = (
        tmp_path
        / "meta"
        / "progress"
        / "episodes"
        / "episode-progress-recorder"
    )
    assert progress_dir.is_dir()

    task = recorder.stop_episode(
        reason="manual_key_stop",
        step_index=0,
        last_state="RUNNING",
    )

    assert task is None
    assert recorder.writer.discarded == [0]
    assert not progress_dir.exists()


def test_episode_compaction_fails_closed_when_progress_exists(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path, progress=True)
    progress_root = tmp_path / "meta" / "progress" / "episodes" / "episode-a"
    progress_root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="deletion/compaction is disabled"):
        recorder.delete_episode(0)
