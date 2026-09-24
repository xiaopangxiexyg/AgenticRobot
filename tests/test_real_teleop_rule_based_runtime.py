from __future__ import annotations

import pickle
import threading
import time
from concurrent.futures import Future
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_MIXED_V1,
    HandActionSpec,
    RobotAction,
)
from robot_motion.action.sources import CallableActionSource
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
)
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_collection_pose_frame_contract,
)
from robot_motion.replay import (
    ReplayTrajectoryGroup,
    RuleBasedActionSegment,
    RuleBasedCaptureSelection,
    RuleBasedCollectionCursor,
    RuleBasedCollectionPlan,
    RuleBasedCustomAction,
    RuleBasedInitialize,
    RuleBasedReplay,
    RuleBasedSegmentStopCondition,
)
from robot_motion.rm65 import DualArmCollisionAwareRetargeterConfig
from teleop_runtime.command_override import TeleopCommandOverride
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.backends.grippers.dmtac import (
    DMTacGripperBackend,
    DMTacGripperBackendConfig,
)
from teleop_runtime.backends.grippers import dmtac as dmtac_module
from teleop_runtime.collection.action_record import validate_collection_command_action
from teleop_runtime.schemas import EffectorCommand, TeleopCommand
from teleop_runtime.real_teleop.rule_based_runtime import (
    RuleBasedRealTeleopConfig,
    RuleBasedRealTeleopDeployment,
    RuleBasedRealTeleopExtension,
    RuleBasedWritingHeightCalibration,
    preflight_rule_based_real_teleop_deployment,
)
from teleop_runtime.real_teleop.rule_based_initial_alignment import (
    RuleBasedInitialAlignmentPlanningError,
)
from teleop_runtime.real_teleop.rule_based_writing_plan import (
    CalibratedHeightDescentActionSource,
)
from teleop_runtime.real_teleop import rule_based_runtime as rule_runtime_module
from teleop_runtime.real_teleop.runtime_extension import (
    RealTeleopDatasetSampleAdmission,
    runtime_extension_allows_active_episode_bootstrap_without_dispatch,
    runtime_extension_allows_pre_episode_armed_warmup,
    runtime_extension_dataset_sample_admission,
    runtime_extension_service_startup_control_plane,
)
from teleop_runtime.real_teleop import control_loop
from teleop_runtime.special_tasks.fix_arm import FixArmCommandController


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.5) -> np.ndarray:
    return np.asarray([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _float32_pose_with_nontrivial_rotation() -> np.ndarray:
    pose = np.asarray(
        [0.4, -0.1, 0.3, 0.1, 0.2, 0.3, 0.9],
        dtype=np.float32,
    )
    pose[3:] /= np.linalg.norm(pose[3:])
    return pose


def _matrix(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = pose[:3]
    return matrix


def _group() -> ReplayTrajectoryGroup:
    actions = np.zeros((2, ACTION_DIM), dtype=np.float32)
    for side in ("left", "right"):
        actions[0, ACTION_ARM_POSE_SLICES[side]] = _pose(0.0)
        actions[1, ACTION_ARM_POSE_SLICES[side]] = _pose(0.02)
    qpos = np.stack((np.zeros(6), np.full(6, 0.1)), axis=0)
    return ReplayTrajectoryGroup(
        key="source",
        label="source",
        source_path="fixture",
        source_episode_index=0,
        start_index=0,
        end_index=2,
        frame_count=2,
        action_hz=10.0,
        actions=actions,
        arm_joint_targets={"left": qpos, "right": qpos},
        arm_observation_actual_tcp_pose={
            "left": np.stack((_pose(0.0), _pose(0.01))),
            "right": np.stack((_pose(0.0), _pose(0.01))),
        },
    )


def _float32_rotated_group() -> ReplayTrajectoryGroup:
    group = _group()
    actions = np.asarray(group.actions, dtype=np.float32).copy()
    for side in ("left", "right"):
        actions[:, ACTION_ARM_POSE_SLICES[side]] = (
            _float32_pose_with_nontrivial_rotation()
        )
    return replace(group, actions=actions)


def _dual_hand_group() -> ReplayTrajectoryGroup:
    group = _group()
    actions = np.asarray(group.actions, dtype=np.float32).copy()
    targets: dict[str, np.ndarray] = {}
    specs = {}
    for side in ("left", "right"):
        values = np.zeros((2, 6), dtype=np.float32)
        values[:, 0] = 0.10
        targets[side] = values
        actions[:, ACTION_HAND_SLICES[side]] = values
        specs[side] = HandActionSpec.dmtac_gripper(side).to_metadata()
    return replace(
        group,
        actions=actions,
        hand_joint_targets=targets,
        metadata={
            "source_metadata": {
                "dataset_info": {"hand_action_specs_by_side": specs}
            }
        },
    )


def _full_replay_jump_hand_group() -> ReplayTrajectoryGroup:
    group = _dual_hand_group()
    actions = np.asarray(group.actions, dtype=np.float32).copy()
    targets = {
        side: np.asarray(values, dtype=np.float32).copy()
        for side, values in group.hand_joint_targets.items()
    }
    for side in ("left", "right"):
        actions[:, ACTION_ARM_POSE_SLICES[side]] = _pose(0.0)
        targets[side][:, 0] = 0.0
        actions[:, ACTION_HAND_SLICES[side]] = targets[side]
    targets["right"][1, 0] = 0.70
    actions[1, ACTION_HAND_SLICES["right"]] = targets["right"][1]
    return replace(
        group,
        actions=actions,
        hand_joint_targets=targets,
        arm_observation_actual_tcp_pose={
            "left": np.stack((_pose(0.0), _pose(0.0))),
            "right": np.stack((_pose(0.0), _pose(0.0))),
        },
    )


def _deployment() -> RuleBasedRealTeleopDeployment:
    contract = default_rm65_workspace_near_boundary_collection_pose_frame_contract(
        ("left", "right")
    )
    assert contract is not None
    return RuleBasedRealTeleopDeployment(
        active_sides=("left", "right"),
        collection_pose_frame_contract=contract,
        tcp_pose_envelope_profile="fixture-envelope.json",
    )


def test_writing_stage_coverage_requires_each_persisted_detail_stage() -> None:
    group = _group()
    capture = RuleBasedCaptureSelection(enabled=True)
    steps = tuple(
        RuleBasedReplay(
            step_id=f"right_write.{phase}",
            group_key=phase,
            start_index=0,
            end_index=2,
            active_sides=("right",),
            capture=capture,
        )
        for phase in ("pen_down", "write", "pen_up")
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(steps=steps),
        trajectory_groups={phase: replace(group, key=phase) for phase in (
            "pen_down",
            "write",
            "pen_up",
        )},
        config=RuleBasedRealTeleopConfig(
            minimum_stage_sample_coverage=0.8,
            minimum_stage_sample_count=2,
        ),
    )

    assert set(extension._incomplete_detail_stages()) == {
        "right_write.pen_down",
        "right_write.write",
        "right_write.pen_up",
    }
    extension._accepted_samples_by_detail_stage.update(
        {
            "right_write.pen_down": 2,
            "right_write.write": 1,
            "right_write.pen_up": 2,
        }
    )
    assert set(extension._incomplete_detail_stages()) == {"right_write.write"}
    extension._accepted_samples_by_detail_stage["right_write.write"] = 2
    assert extension._incomplete_detail_stages() == {}


def test_successful_random_source_attempt_advances_selection_ordinal() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(
                    left_arm=np.zeros(6),
                    right_arm=np.zeros(6),
                ),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
        source_selection={"selection_ordinal": 0},
    )
    rebuilt_ordinals: list[int] = []
    rebuild_started = threading.Event()
    release_rebuild = threading.Event()

    def rebuild(configuration):
        ordinal = int(configuration["_selection_ordinal"])
        rebuilt_ordinals.append(ordinal)
        rebuild_started.set()
        assert release_rebuild.wait(timeout=2.0)
        return RuleBasedRealTeleopExtension(
            deployment=_deployment(),
            plan=plan,
            trajectory_groups={},
            source_selection={"selection_ordinal": ordinal},
        )

    extension._configuration_rebuilder = rebuild
    extension._selected_configuration = {
        "pen_down_candidate_id": "__rule_source_pool_random__",
    }
    extension._configuration_options = {"runtime_info": {}}
    extension._make_initial_alignment_planner = lambda: None
    extension._recording_active = lambda: False
    extension._dataset_recorder_provider = lambda: None
    extension._control_ui = SimpleNamespace(
        update_rule_runtime_info=lambda _values: None
    )
    extension._completed = True

    extension._prepare_next_attempt_after_success(now=1.0)

    assert rebuild_started.wait(timeout=1.0)
    assert rebuilt_ordinals == [1]
    assert extension._selection_ordinal == 0
    assert extension._next_attempt_rebuild_future is not None

    release_rebuild.set()
    deadline = time.monotonic() + 2.0
    while extension._next_attempt_rebuild_future is not None:
        extension._prepare_next_attempt_after_success(now=1.1)
        if time.monotonic() >= deadline:
            raise AssertionError("next-attempt rebuild did not settle")
        time.sleep(0.001)

    assert extension._selection_ordinal == 1
    assert extension.source_selection["selection_ordinal"] == 1


def test_writing_reachability_blocks_start_until_background_check_passes() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        source_selection={
            "collection_mode": "rule_based_writing",
            "selection_ordinal": 0,
            "writing_reachability": {"enabled": True},
            "writing_variant_selection": {
                "enabled": True,
                "selected_trajectory_id": "latin_a_v1",
                "selected_direction": "up",
                "selected_size_m": 0.06,
            },
        },
    )
    recorder = _Recorder(_Poller())
    _session, _clutch, _ui = _bind(extension, recorder, arm=False)
    extension._height_calibration = RuleBasedWritingHeightCalibration(
        calibrated_z_m=0.3,
        lower_tolerance_m=0.001,
        upper_tolerance_m=0.001,
        captured_at=1.0,
        source_timestamp=1.0,
        pose_frame_contract_hash=extension.runtime_pose_frame_contract.contract_hash,
    )
    extension._writing_reachability_request = lambda: {}
    extension._run_writing_reachability_preflight = lambda _request: {
        "reachable": True,
        "probes": (),
    }
    extension._submit_writing_reachability_preflight = lambda request: (
        extension._configuration_executor.submit(
            extension._run_writing_reachability_preflight,
            request,
        )
    )

    extension.after_control_input_poll(now=1.0)
    assert extension.status().rule_start_available is False
    deadline = time.monotonic() + 2.0
    while extension.status().writing_reachability_status != "passed":
        extension.after_control_input_poll(now=1.1)
        if time.monotonic() >= deadline:
            raise AssertionError("writing reachability preflight did not settle")
        time.sleep(0.001)

    status = extension.status()
    assert status.writing_reachability_status == "passed"
    assert status.rule_start_available is True


def test_writing_reachability_submission_uses_process_safe_payload() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    captured: dict[str, object] = {}

    class Executor:
        def submit(self, fn, request):
            captured["fn"] = fn
            captured["request"] = request
            future = Future()
            future.set_result({"reachable": True, "probes": ()})
            return future

    extension._reachability_executor = Executor()
    callback = lambda _event: None
    future = extension._submit_writing_reachability_preflight(
        {"value": 1, "progress_callback": callback}
    )

    assert future.result()["reachable"] is True
    assert captured["fn"].__name__ == (
        "_run_writing_reachability_preflight_process"
    )
    assert captured["request"] == {"value": 1}


def test_writing_reachability_spawn_worker_round_trip() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    projected_probe = {
        "name": "right",
        "target_pose_by_side": {
            "left": np.eye(4),
            "right": np.eye(4),
        },
        "envelope_modified_sides": ("right",),
    }
    try:
        future = extension._submit_writing_reachability_preflight(
            {
                "oracle_config": object(),
                "seed_qpos": {"left": np.zeros(6), "right": np.zeros(6)},
                "probes": tuple(projected_probe for _ in range(4)),
                "world_to_base_by_side": {
                    "left": np.eye(4),
                    "right": np.eye(4),
                },
                "translation_tolerance_m": 0.005,
                "rotation_tolerance_deg": 3.0,
                "progress_callback": lambda _event: None,
            }
        )
        result = future.result(timeout=15.0)
    finally:
        extension._shutdown_writing_reachability_executor(wait=True)

    assert result["reachable"] is False
    assert len(result["probes"]) == 4
    assert all(
        probe["reason"] == "tcp_envelope_projection_required"
        for probe in result["probes"]
    )


def test_writing_reachability_replaces_crashed_process_pool_on_retry() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )

    class BrokenExecutor:
        shutdown_called = False

        def submit(self, *_args, **_kwargs):
            raise RuntimeError("spawn worker crashed")

        def shutdown(self, *, wait, cancel_futures):
            assert wait is False
            assert cancel_futures is True
            self.shutdown_called = True

    broken = BrokenExecutor()
    extension._reachability_executor = broken

    with pytest.raises(RuntimeError, match="spawn worker crashed"):
        extension._submit_writing_reachability_preflight({"value": 1})

    assert broken.shutdown_called is True
    assert extension._reachability_executor is None


def test_writing_reachability_shows_candidate_probe_and_slow_solve() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        source_selection={
            "selection_ordinal": 2,
            "writing_reachability": {"enabled": True},
            "writing_variant_selection": {
                "selected_trajectory_id": "latin_a_v1",
                "selected_direction": "right",
                "selected_size_m": 0.07,
            },
            "phases": {"pen_down": {"candidate_id": "approach-17"}},
        },
    )
    recorder = _Recorder(_Poller())
    _session, _clutch, ui = _bind(extension, recorder, arm=False)
    extension._height_calibration = RuleBasedWritingHeightCalibration(
        calibrated_z_m=0.3,
        lower_tolerance_m=0.001,
        upper_tolerance_m=0.001,
        captured_at=1.0,
        source_timestamp=1.0,
        pose_frame_contract_hash=extension.runtime_pose_frame_contract.contract_hash,
    )
    extension._writing_reachability_request = lambda: {}
    worker_busy = threading.Event()
    free_worker = threading.Event()
    extension._configuration_executor.submit(
        lambda: (worker_busy.set(), free_worker.wait(timeout=2.0))
    )
    assert worker_busy.wait(timeout=1.0)
    started = threading.Event()
    release = threading.Event()

    def check(request):
        request["progress_callback"](("start", "top", 1, 4, 0.0))
        started.set()
        assert release.wait(timeout=2.0)
        request["progress_callback"](("finish", "top", 1, 4, 1200.0))
        return {"reachable": True, "probes": (), "elapsed_ms": 1200.0}

    extension._run_writing_reachability_preflight = check
    extension._submit_writing_reachability_preflight = lambda request: (
        extension._configuration_executor.submit(
            extension._run_writing_reachability_preflight,
            request,
        )
    )
    try:
        extension.after_control_input_poll(now=1.0)
        extension.after_control_input_poll(now=1.05)
        assert "等待后台计算资源" in extension.status().writing_reachability_detail
        assert "前台准备" in extension.status().writing_reachability_detail
        free_worker.set()
        assert started.wait(timeout=1.0)
        extension._writing_reachability_started_at -= 5.0
        extension.after_control_input_poll(now=1.1)
        status = extension.status()
        assert "候选 3" in status.writing_reachability_detail
        assert "latin_a_v1 / right / 0.070m" in status.writing_reachability_detail
        assert "靠近来源 approach-17" in status.writing_reachability_detail
        assert "第 1/4 点 top 求解中" in status.writing_reachability_detail
        assert "超过预检目标时限" in status.writing_reachability_detail
        assert status.readiness_blockers["rule.writing_reachability"] == (
            status.writing_reachability_detail
        )
        assert recorder.is_recording is False
        release.set()
        deadline = time.monotonic() + 2.0
        while extension.status().writing_reachability_status != "passed":
            extension.after_control_input_poll(now=1.2)
            if time.monotonic() >= deadline:
                raise AssertionError("writing reachability preflight did not settle")
            time.sleep(0.001)
        assert "预检耗时 1200ms" in extension.status().writing_reachability_detail
        assert ui.statuses
        assert any(
            tag == "RuleBasedReachability" and "预检耗时 1200ms" in message
            for tag, message in ui.messages
        )
    finally:
        free_worker.set()
        release.set()


def test_writing_reachability_receipt_is_bound_to_pen_down_source() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        source_selection={
            "selection_ordinal": 0,
            "phases": {
                "pen_down": {
                    "candidate_id": "source-a",
                    "source_contract_hash": "contract-a",
                }
            },
            "trajectory_catalog": {"combined_sha256": "catalog-a"},
            "writing_variant_selection": {
                "selected_trajectory_id": "digit_1_v1",
                "selected_direction": "up",
                "selected_size_m": 0.06,
            },
        },
    )
    extension._height_calibration = RuleBasedWritingHeightCalibration(
        calibrated_z_m=0.3,
        lower_tolerance_m=0.001,
        upper_tolerance_m=0.001,
        captured_at=1.0,
        source_timestamp=1.0,
        pose_frame_contract_hash=extension.runtime_pose_frame_contract.contract_hash,
    )
    original = extension._writing_reachability_current_fingerprint()

    extension.source_selection["phases"]["pen_down"]["candidate_id"] = "source-b"

    assert extension._writing_reachability_current_fingerprint() != original


def test_runtime_home_invalidates_terminal_writing_reachability_result() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        source_selection={"writing_reachability": {"enabled": True}},
    )
    _bind(extension, _Recorder(_Poller()), arm=False)
    extension._height_calibration = RuleBasedWritingHeightCalibration(
        calibrated_z_m=0.3,
        lower_tolerance_m=0.001,
        upper_tolerance_m=0.001,
        captured_at=1.0,
        source_timestamp=1.0,
        pose_frame_contract_hash=extension.runtime_pose_frame_contract.contract_hash,
    )
    extension._writing_reachability_fingerprint = (
        extension._writing_reachability_current_fingerprint()
    )
    extension._writing_reachability_status = "error"
    extension._writing_reachability_detail = "old solver error"
    extension._held = True

    extension.after_runtime_home(now=2.0)

    assert extension._writing_reachability_fingerprint is None
    assert extension.status().writing_reachability_status == "pending"
    assert extension.status().writing_reachability_detail == ""


def test_writing_reachability_oracle_accepts_production_fast_path_config() -> None:
    source = _group()
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="pen_down"),)
        ),
        trajectory_groups={
            "pen_down": replace(
                source,
                arm_observation_actual_qpos={
                    side: np.zeros((2, 6)) for side in ("left", "right")
                },
            )
        },
        source_selection={
            "writing_reachability": {
                "enabled": True,
                "left_hold_pose_canonical_desk": _pose().tolist(),
                "probes": [
                    {"name": name, "right_pose_canonical_desk": _pose().tolist()}
                    for name in (
                        "left", "right", "top", "bottom", "pen_up_mid", "pen_up_end"
                    )
                ],
            }
        },
    )
    extension._height_calibration = RuleBasedWritingHeightCalibration(
        calibrated_z_m=0.3,
        lower_tolerance_m=0.001,
        upper_tolerance_m=0.001,
        captured_at=1.0,
        source_timestamp=1.0,
        pose_frame_contract_hash=extension.runtime_pose_frame_contract.contract_hash,
    )
    production_config = DualArmCollisionAwareRetargeterConfig(
        output_qpos_smoothing_alpha=0.14,
        output_qpos_admittance_fast_path_alpha=0.85,
        output_qpos_smoothing_singularity_alpha=0.45,
    )
    retargeter = _Retargeter()
    retargeter.config = production_config
    def project_left(targets):
        effective = {
            side: np.asarray(pose, dtype=float).copy()
            for side, pose in targets.items()
        }
        effective["left"][0, 3] += 0.01
        return effective

    retargeter.pre_ik_target_safety_filter = project_left
    extension._components = SimpleNamespace(
        dual_arm_retargeter=retargeter,
        session=_Session(),
    )

    request = extension._writing_reachability_request()

    assert pickle.loads(pickle.dumps(request))["oracle_config"] is not None
    assert len(request["probes"]) == 6
    assert request["probes"][0]["envelope_modified_sides"] == ("left",)
    assert request["probes"][4]["active_sides"] == ("right",)
    assert request["probes"][5]["active_sides"] == ("right",)
    assert np.isclose(request["probes"][0]["target_pose_by_side"]["left"][0, 3], 0.01)
    assert request["oracle_config"].output_qpos_smoothing_alpha == 1.0
    assert request["oracle_config"].output_qpos_admittance_fast_path_alpha == 1.0
    assert request["oracle_config"].output_qpos_smoothing_singularity_alpha == 1.0
    assert production_config.output_qpos_admittance_fast_path_alpha == 0.85


def test_writing_reachability_uses_live_frame_projection_and_held_qpos_fk() -> None:
    source = _group()
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="pen_down"),)
        ),
        trajectory_groups={
            "pen_down": replace(
                source,
                arm_observation_actual_qpos={
                    "left": np.stack((np.full(6, 0.2), np.full(6, 0.3))),
                    "right": np.stack((np.full(6, 0.4), np.full(6, 0.5))),
                },
            )
        },
        source_selection={
            "writing_reachability": {
                "enabled": True,
                # Deliberately unrelated to held-qpos FK.  The preflight must
                # not use this recorded action pose as the fixed-arm target.
                "left_hold_pose_canonical_desk": _pose(9.0, 8.0, 7.0).tolist(),
                "probes": [
                    {
                        "name": name,
                        "right_pose_canonical_desk": _pose(
                            0.10 + index * 0.01,
                            -0.20,
                            0.50,
                        ).tolist(),
                    }
                    for index, name in enumerate(("left", "right", "top", "bottom"))
                ],
            }
        },
    )
    extension._height_calibration = RuleBasedWritingHeightCalibration(
        calibrated_z_m=0.3,
        lower_tolerance_m=0.001,
        upper_tolerance_m=0.001,
        captured_at=1.0,
        source_timestamp=1.0,
        pose_frame_contract_hash=extension.runtime_pose_frame_contract.contract_hash,
    )

    transforms = {
        "left": np.asarray(
            [
                [0.0, -1.0, 0.0, 0.15],
                [1.0, 0.0, 0.0, -0.05],
                [0.0, 0.0, 1.0, 0.02],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "right": np.asarray(
            [
                [0.0, 1.0, 0.0, -0.12],
                [-1.0, 0.0, 0.0, 0.08],
                [0.0, 0.0, 1.0, 0.03],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    }

    class Calibration:
        def get_arm_for_side(self, side):
            return SimpleNamespace(T_D_B=transforms[side])

        def base_to_desk(self, pose, side):
            return transforms[side] @ np.asarray(pose, dtype=float)

    held_left_solver_target = np.eye(4)
    held_left_solver_target[:3, 3] = [0.7, -0.4, 0.25]

    class RecordingRetargeter(_Retargeter):
        def __init__(self):
            super().__init__()
            self.config = DualArmCollisionAwareRetargeterConfig()
            self.seed_fk_calls = []
            self.right_tool_targets = []

        def solver_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
            self.seed_fk_calls.append(
                {
                    "left": np.asarray(left_qpos, dtype=float).copy(),
                    "right": np.asarray(right_qpos, dtype=float).copy(),
                }
            )
            return {"left": held_left_solver_target, "right": np.eye(4)}

        def solver_world_target_from_tool_tcp_world_target(
            self, *, side, tool_tcp_world_target
        ):
            assert side == "right"
            value = np.asarray(tool_tcp_world_target, dtype=float).copy()
            self.right_tool_targets.append(value)
            return value

    retargeter = RecordingRetargeter()
    extension._components = SimpleNamespace(
        dual_arm_retargeter=retargeter,
        session=SimpleNamespace(calibration=Calibration()),
    )

    request = extension._writing_reachability_request()

    assert len(retargeter.seed_fk_calls) == 1
    assert np.array_equal(
        retargeter.seed_fk_calls[0]["left"], np.full(6, 0.2)
    )
    assert np.array_equal(
        retargeter.seed_fk_calls[0]["right"], np.full(6, 0.5)
    )
    assert request["pose_projection"]["left_hold"] == "seed_qpos_solver_fk"
    assert request["pose_projection"]["runtime_pose_frame_contract_hash"] == (
        extension.runtime_pose_frame_contract.contract_hash
    )
    for index, probe in enumerate(request["probes"]):
        canonical_pose = _matrix(_pose(0.10 + index * 0.01, -0.20, 0.301))
        expected_base = (
            extension.runtime_pose_frame_contract.canonical_desk_pose_to_storage(
                canonical_pose,
                side="right",
            )
        )
        expected_world = transforms["right"] @ expected_base
        assert np.allclose(
            probe["target_pose_by_side"]["right"],
            expected_world,
        )
        assert not np.allclose(
            probe["target_pose_by_side"]["right"],
            canonical_pose,
        )
        assert np.array_equal(
            probe["target_pose_by_side"]["left"],
            held_left_solver_target,
        )
    assert len(retargeter.right_tool_targets) == 4
    assert np.allclose(
        request["world_to_base_by_side"]["right"],
        np.linalg.inv(transforms["right"]),
    )


@pytest.mark.parametrize("held_pair_safe", [True, False])
@pytest.mark.parametrize("left_projected", [True, False])
def test_writing_reachability_matches_live_coupled_solver_and_fixed_left_collision(
    monkeypatch, held_pair_safe, left_projected,
) -> None:
    calls = []

    class Oracle:
        def __init__(self, _config):
            pass

        def set_current_qpos_pair(self, **kwargs):
            calls.append(("seed", kwargs))

        def solve_target_pair(self, target):
            calls.append(("solve", target.active_sides))
            return SimpleNamespace(
                status="tracking",
                fallback_reason=None,
                side_results={
                    "right": SimpleNamespace(
                        solver_pose_world=target.target_pose_by_side["right"],
                        qpos=np.ones(6),
                    )
                },
            )

        def check_qpos_pair_collision(self, **kwargs):
            calls.append(("collision", kwargs))
            return SimpleNamespace(
                safe=held_pair_safe,
                reason="" if held_pair_safe else "collision_margin_low",
            )

    monkeypatch.setattr(rule_runtime_module, "DualArmCollisionAwareRetargeter", Oracle)
    target = np.eye(4)
    target[0, 3] = 0.02
    request = {
        "oracle_config": object(),
        "seed_qpos": {"left": np.zeros(6), "right": np.zeros(6)},
        "probes": tuple(
            {
                "name": name,
                "target_pose_by_side": {"left": np.eye(4), "right": target},
                "envelope_modified_sides": ("left",) if left_projected else (),
            }
            for name in ("left", "right", "bottom", "top")
        ),
        "world_to_base_by_side": {"left": np.eye(4), "right": np.eye(4)},
        "translation_tolerance_m": 0.005,
        "rotation_tolerance_deg": 3.0,
    }
    progress = []
    request["progress_callback"] = progress.append

    result = RuleBasedRealTeleopExtension._run_writing_reachability_preflight(
        request
    )

    assert result["reachable"] is held_pair_safe
    assert result["elapsed_ms"] >= 0.0
    assert [event[:4] for event in progress] == [
        (phase, name, index, 4)
        for index, name in enumerate(("left", "right", "bottom", "top"), start=1)
        for phase in ("start", "finish")
    ]
    assert all(event[4] >= 0.0 for event in progress)
    if not held_pair_safe:
        assert {probe["reason"] for probe in result["probes"]} == {
            "held_left_collision"
        }
    assert [value for kind, value in calls if kind == "solve"] == [
        ("left", "right")
    ] * 4
    held_pairs = [value for kind, value in calls if kind == "collision"]
    assert len(held_pairs) == 4
    assert all(np.array_equal(pair["left_qpos"], np.zeros(6)) for pair in held_pairs)
    assert all(np.array_equal(pair["right_qpos"], np.ones(6)) for pair in held_pairs)


def test_writing_reachability_rejects_projected_right_glyph_target(
    monkeypatch,
) -> None:
    class Oracle:
        def __init__(self, _config):
            pass

        def solve_target_pair(self, _target):
            raise AssertionError("projected right glyph must not enter IK")

    monkeypatch.setattr(rule_runtime_module, "DualArmCollisionAwareRetargeter", Oracle)
    result = RuleBasedRealTeleopExtension._run_writing_reachability_preflight(
        {
            "oracle_config": object(),
            "seed_qpos": {"left": np.zeros(6), "right": np.zeros(6)},
            "probes": (
                {
                    "name": "right",
                    "target_pose_by_side": {
                        "left": np.eye(4),
                        "right": np.eye(4),
                    },
                    "envelope_modified_sides": ("left", "right"),
                },
            ),
            "world_to_base_by_side": {"left": np.eye(4), "right": np.eye(4)},
            "translation_tolerance_m": 0.005,
            "rotation_tolerance_deg": 3.0,
        }
    )

    assert result["reachable"] is False
    assert result["probes"][0]["reason"] == "tcp_envelope_projection_required"


def test_writing_reachability_budget_exhaustion_is_not_unreachable(
    monkeypatch,
) -> None:
    class Oracle:
        def __init__(self, _config):
            pass

        def set_current_qpos_pair(self, **_kwargs):
            pass

        def solve_target_pair(self, _target):
            raise AssertionError("expired preflight must not enter IK")

    monkeypatch.setattr(rule_runtime_module, "DualArmCollisionAwareRetargeter", Oracle)
    monkeypatch.setattr(
        rule_runtime_module, "_WRITING_REACHABILITY_MAX_CHECK_SEC", 0.0
    )
    result = RuleBasedRealTeleopExtension._run_writing_reachability_preflight(
        {
            "oracle_config": object(),
            "seed_qpos": {"left": np.zeros(6), "right": np.zeros(6)},
            "probes": (
                {
                    "name": "right",
                    "target_pose_by_side": {
                        "left": np.eye(4),
                        "right": np.eye(4),
                    },
                    "envelope_modified_sides": (),
                },
            ),
            "world_to_base_by_side": {"left": np.eye(4), "right": np.eye(4)},
            "translation_tolerance_m": 0.005,
            "rotation_tolerance_deg": 3.0,
        }
    )

    assert result["reachable"] is False
    assert result["probes"][0]["reason"] == "solver_budget_exhausted"


def test_writing_reachability_cannot_pass_without_all_replay_probes(monkeypatch) -> None:
    clock = [0.0]

    class Oracle:
        def __init__(self, _config):
            pass

        def set_current_qpos_pair(self, **_kwargs):
            pass

        def solve_target_pair(self, target):
            return SimpleNamespace(
                status="tracking",
                fallback_reason=None,
                side_results={
                    "right": SimpleNamespace(
                        solver_pose_world=target.target_pose_by_side["right"],
                        qpos=np.zeros(6),
                    )
                },
            )

        def check_qpos_pair_collision(self, **_kwargs):
            clock[0] = 5.0
            return SimpleNamespace(safe=True, reason="")

    monkeypatch.setattr(rule_runtime_module, "DualArmCollisionAwareRetargeter", Oracle)
    monkeypatch.setattr(rule_runtime_module.time, "monotonic", lambda: clock[0])
    request = {
        "oracle_config": object(),
        "seed_qpos": {"left": np.zeros(6), "right": np.zeros(6)},
        "probes": tuple(
            {
                "name": name,
                "target_pose_by_side": {"left": np.eye(4), "right": np.eye(4)},
                "envelope_modified_sides": (),
            }
            for name in ("write_left", "pen_up_mid", "pen_up_end")
        ),
        "world_to_base_by_side": {"left": np.eye(4), "right": np.eye(4)},
        "translation_tolerance_m": 0.005,
        "rotation_tolerance_deg": 3.0,
    }

    result = RuleBasedRealTeleopExtension._run_writing_reachability_preflight(request)
    assert result["reachable"] is False
    assert [probe["reason"] for probe in result["probes"]] == [
        "certified", "solver_budget_exhausted", "solver_budget_exhausted"
    ]


@pytest.mark.parametrize(
    ("reason", "expected_status"),
    [
        ("solver_not_converged", "uncertified"),
        ("tcp_envelope_projection_required", "unreachable"),
        ("internal_protocol_error", "error"),
    ],
)
def test_terminal_writing_reachability_result_is_not_resubmitted(
    reason, expected_status,
) -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        source_selection={
            "selection_ordinal": 0,
            "writing_reachability": {"enabled": True},
            "writing_variant_selection": {
                "selected_trajectory_id": "latin_a_v1",
                "selected_direction": "up",
                "selected_size_m": 0.06,
            },
        },
    )
    _session, _clutch, ui = _bind(extension, _Recorder(_Poller()), arm=False)
    extension._height_calibration = RuleBasedWritingHeightCalibration(
        calibrated_z_m=0.3,
        lower_tolerance_m=0.001,
        upper_tolerance_m=0.001,
        captured_at=1.0,
        source_timestamp=1.0,
        pose_frame_contract_hash=extension.runtime_pose_frame_contract.contract_hash,
    )
    extension._writing_reachability_request = lambda: {}
    extension._writing_reachability_skipped_count = 31
    checks = []

    def check(_request):
        checks.append(1)
        return {
            "reachable": False,
            "probes": ({"name": "right", "reachable": False, "reason": reason},),
        }

    extension._run_writing_reachability_preflight = check
    extension._submit_writing_reachability_preflight = lambda request: (
        extension._configuration_executor.submit(
            extension._run_writing_reachability_preflight,
            request,
        )
    )
    deadline = time.monotonic() + 2.0
    while extension.status().writing_reachability_status != expected_status:
        extension.after_control_input_poll(now=1.0)
        if time.monotonic() >= deadline:
            raise AssertionError("terminal preflight did not settle")
        time.sleep(0.001)

    for _ in range(20):
        extension.after_control_input_poll(now=1.1)
    assert checks == [1]
    status = extension.status()
    assert status.rule_start_available is False
    assert "候选 1" in status.writing_reachability_detail
    assert reason in status.writing_reachability_detail
    assert any(
        tag == "RuleBasedReachability" and reason in message
        for tag, message in ui.messages
    )


@pytest.mark.parametrize(
    ("first_reason", "expected_summary"),
    [
        ("tcp_envelope_projection_required", "安全边界拒绝"),
        ("solver_guard_hold", "数值 IK 未能认证"),
    ],
)
def test_uncertified_writing_candidate_is_skipped_without_hold_or_episode(
    first_reason: str,
    expected_summary: str,
) -> None:
    plan = RuleBasedCollectionPlan(
        steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
    )

    def selection(ordinal: int) -> dict[str, object]:
        return {
            "collection_mode": "rule_based_writing",
            "selection_ordinal": ordinal,
            "writing_reachability": {"enabled": True},
            "writing_variant_selection": {
                "enabled": True,
                "selected_trajectory_id": "latin_a_v1",
                "selected_direction": "up",
                "selected_size_m": 0.06,
            },
        }

    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={"source": _group()},
        source_selection=selection(0),
    )
    recorder = _Recorder(_Poller())
    _session, _clutch, ui = _bind(extension, recorder, arm=False)
    extension._height_calibration = RuleBasedWritingHeightCalibration(
        calibrated_z_m=0.3,
        lower_tolerance_m=0.001,
        upper_tolerance_m=0.001,
        captured_at=1.0,
        source_timestamp=1.0,
        pose_frame_contract_hash=extension.runtime_pose_frame_contract.contract_hash,
    )
    extension._selected_configuration = {
        "pen_down_candidate_id": "__rule_source_pool_random__",
    }
    extension._configuration_options = {"runtime_info": {}}
    extension._configuration_rebuilder = lambda request: RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={"source": _group()},
        source_selection=selection(int(request["_selection_ordinal"])),
    )
    extension._writing_reachability_request = lambda: {}
    check_count = 0

    def check(_request):
        nonlocal check_count
        check_count += 1
        if check_count == 1:
            return {
                "reachable": False,
                "probes": (
                    {
                        "name": "right",
                        "reachable": False,
                        "reason": first_reason,
                    },
                ),
            }
        return {"reachable": True, "probes": ()}

    extension._run_writing_reachability_preflight = check
    extension._submit_writing_reachability_preflight = lambda request: (
        extension._configuration_executor.submit(
            extension._run_writing_reachability_preflight,
            request,
        )
    )

    deadline = time.monotonic() + 2.0
    while extension.status().writing_reachability_status != "passed":
        extension.after_control_input_poll(now=1.0)
        if time.monotonic() >= deadline:
            raise AssertionError("unreachable candidate skip did not settle")
        time.sleep(0.001)

    assert extension._selection_ordinal == 1
    assert extension.status().writing_reachability_skipped_count == 1
    assert extension.status().held is False
    assert extension.status().aborted is False
    assert recorder.is_recording is False
    assert any(
        tag == "RuleBasedReachability" and expected_summary in message
        for tag, message in ui.messages
    )


def test_successful_next_attempt_rebuild_keeps_certified_arm_hold_stream() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(
                    left_arm=np.zeros(6),
                    right_arm=np.zeros(6),
                ),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
        source_selection={"selection_ordinal": 0},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    active = _wait_for_command(session, now=2.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=2.0,
    )

    rebuild_started = threading.Event()
    release_rebuild = threading.Event()

    def rebuild(configuration):
        rebuild_started.set()
        assert release_rebuild.wait(timeout=2.0)
        return RuleBasedRealTeleopExtension(
            deployment=_deployment(),
            plan=plan,
            trajectory_groups={},
            source_selection={
                "selection_ordinal": int(configuration["_selection_ordinal"])
            },
        )

    extension._configuration_rebuilder = rebuild
    extension._selected_configuration = {
        "pen_down_candidate_id": "__rule_source_pool_random__",
    }
    extension._configuration_options = {"runtime_info": {}}
    extension._completed = True
    extension._armed = False
    recorder.is_recording = False

    extension._prepare_next_attempt_after_success(now=2.1)
    assert rebuild_started.wait(timeout=1.0)
    held = session.command_override_provider(SimpleNamespace(), 2.11)

    assert set(held.command.commands) == {"left_arm", "right_arm"}
    assert held.command.metadata["rule_based_pause_hold"] is True
    assert (
        held.command.metadata["rule_based_pause_reason"]
        == "successful_next_attempt_rebuild"
    )
    assert held.command.metadata["dataset_recording_eligible"] is False

    release_rebuild.set()
    deadline = time.monotonic() + 2.0
    while extension._next_attempt_rebuild_future is not None:
        extension._prepare_next_attempt_after_success(now=2.2)
        if time.monotonic() >= deadline:
            raise AssertionError("next-attempt rebuild did not settle")
        time.sleep(0.001)


def test_completed_episode_stop_handoff_does_not_hold_without_recorder() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6), right_arm=np.zeros(6)
                    ),
                ),
            )
        ),
        trajectory_groups={},
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    active = _wait_for_command(session, now=2.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=2.0,
    )
    extension._clear_current()
    extension._load_next(SimpleNamespace(), 2.01, poller.latest_at_or_before(2.01))
    assert extension.status().completed is True
    extension._completion_stop_requested = True

    # The real stop worker detaches the recorder before publishing completion.
    # This gap must neither resend a qref without feedback nor become a robot
    # Hold, which would prevent the next successful attempt from starting.
    transition_pending = True
    extension._dataset_recorder_provider = lambda: None if transition_pending else recorder
    extension._collection_transition_active = lambda: transition_pending
    recorder.is_recording = False
    extension.after_control_input_poll(now=2.02)
    during_stop = session.command_override_provider(SimpleNamespace(), 2.02)
    assert during_stop.command is None
    assert extension._completed_hold_stream_suspended is True
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=None,
            command_dispatched=False,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=2.02,
    )
    assert extension.status().completed is True
    assert extension.status().held is False
    assert extension.status().aborted is False
    assert session.hold_reasons == []

    transition_pending = False
    # Recovery of the recorder must not restart the old terminal qref after
    # the backend has been allowed to expire that target.
    recovered = session.command_override_provider(SimpleNamespace(), 2.025)
    assert recovered.command is None
    extension.after_control_input_poll(now=2.03)
    assert extension.status().completed is False
    assert extension.status().held is False
    assert extension.status().rule_start_available is True
    assert session.hold_reasons == []
    # The following transition must use post-stop measured state, not the
    # successful Episode's last accepted command.
    current_after_stop = {
        side: {"actual_qpos": np.full(6, 0.1, dtype=float)}
        for side in ("left", "right")
    }
    selected = extension._initial_alignment_start_qpos_pair(current_after_stop)
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            selected[side], current_after_stop[side]["actual_qpos"]
        )


@pytest.mark.parametrize(
    "blocked",
    [
        {"left": "hardware_read_failed", "right": "feedback_stale"},
        {"right": "hardware_read_failed"},
        {"left": "feedback_stale"},
    ],
)
def test_completed_episode_feedback_failure_releases_old_stream_without_hold(
    blocked: dict[str, str],
) -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6), right_arm=np.zeros(6)
                    ),
                ),
            )
        ),
        trajectory_groups={},
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, ui = _bind(extension, recorder)
    active = _wait_for_command(session, now=2.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=2.0,
    )
    extension._clear_current()
    extension._load_next(SimpleNamespace(), 2.01, poller.latest_at_or_before(2.01))
    assert extension.status().completed is True
    extension._completion_stop_requested = True
    recorder.is_recording = False
    poller.blocked.update(blocked)

    after_stop = session.command_override_provider(SimpleNamespace(), 2.02)
    assert after_stop.command is None
    assert extension._completed_hold_stream_suspended is True
    assert extension.status().held is False
    assert extension.status().aborted is False
    assert session.hold_reasons == []

    poller.blocked.clear()
    # A late read cannot resurrect the terminal qref, including during a
    # next-selection rebuild. Motion only resumes from the next measured start.
    assert session.command_override_provider(SimpleNamespace(), 2.03).command is None
    assert sum("本条采集已完成；末端保持命令" in message for _, message in ui.messages) == 1
    extension.after_control_input_poll(now=2.04)
    assert extension._completed_hold_stream_suspended is False
    assert extension.status().rule_start_available is True
    assert session.hold_reasons == []


@pytest.mark.parametrize("feedback_mode", ["poller_unavailable", "mixed_not_ready"])
def test_missing_arm_feedback_outside_completed_stop_handoff_still_holds(
    feedback_mode: str,
) -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6), right_arm=np.zeros(6)
                    ),
                ),
            )
        ),
        trajectory_groups={},
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    active = _wait_for_command(session, now=2.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=2.0,
    )
    extension._completed = True
    extension._armed = False
    if feedback_mode == "poller_unavailable":
        extension._dataset_recorder_provider = lambda: None
    else:
        poller.blocked.update(
            {"left": "hardware_read_failed", "right": "feedback_stale"}
        )
    extension._collection_transition_active = lambda: False

    missing = session.command_override_provider(SimpleNamespace(), 2.01)
    assert missing.command is None
    assert extension.status().held is True
    assert extension.status().aborted is True
    assert (
        "causal_arm_feedback_poller_unavailable"
        if feedback_mode == "poller_unavailable"
        else "causal_arm_feedback_not_ready:left=hardware_read_failed,right=feedback_stale"
    ) in session.hold_reasons[-1]


@pytest.mark.parametrize("moved_away", [False, True])
def test_successful_episode_finalization_distinguishes_lag_from_manual_motion(
    moved_away: bool,
) -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6), right_arm=np.zeros(6)
                    ),
                ),
            )
        ),
        trajectory_groups={},
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    active = _wait_for_command(session, now=2.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=2.0,
    )
    # The terminal target can be accepted with up to 2 degrees of joint
    # tracking error. This is not evidence that someone moved the arm.
    poller.qpos["right"][2] = np.deg2rad(1.0)
    extension._clear_current()
    extension._load_next(SimpleNamespace(), 2.01, poller.latest_at_or_before(2.01))
    assert extension.status().completed is True

    finalizing = session.command_override_provider(SimpleNamespace(), 2.02)

    assert finalizing.command is not None
    assert finalizing.command.metadata["rule_based_pause_hold"] is True
    assert extension.status().aborted is False
    assert extension.status().held is False
    assert session.hold_reasons == []

    # Servo convergence cannot invalidate a completed Episode either.
    poller.qpos["right"][2] = np.deg2rad(0.4)
    converging = session.command_override_provider(SimpleNamespace(), 2.03)
    assert converging.command is not None
    assert extension.status().held is False

    if moved_away:
        # A subsequent move *away* from the accepted target is still caught
        # under the unchanged 0.75-degree movement limit.
        poller.qpos["right"][2] = np.deg2rad(1.3)
        moved = session.command_override_provider(SimpleNamespace(), 2.04)
        assert moved.command is None
        assert extension.status().aborted is True
        assert extension.status().held is True
        assert "error increased by" in session.hold_reasons[-1]
    else:
        recorder.is_recording = False
        extension.after_control_input_poll(now=2.04)
        assert extension.status().completed is False
        assert extension.status().held is False
        assert extension.status().rule_start_available is True
        assert session.hold_reasons == []


def test_successful_episode_reset_uses_current_qpos() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(
                    left_arm=np.zeros(6),
                    right_arm=np.zeros(6),
                ),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
    )
    extension._make_initial_alignment_planner = lambda: None
    extension._recording_active = lambda: False
    extension._dataset_recorder_provider = lambda: None
    extension._control_ui = SimpleNamespace(
        update_rule_runtime_info=lambda _values: None
    )
    extension._completed = True

    extension._prepare_next_attempt_after_success(now=1.0)

    current_after_episode = {
        side: {"actual_qpos": np.full(6, 0.1, dtype=float)}
        for side in ("left", "right")
    }
    selected = extension._initial_alignment_start_qpos_pair(current_after_episode)
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            selected[side], current_after_episode[side]["actual_qpos"]
        )



class _Poller:
    def __init__(self) -> None:
        self.qpos = {"left": np.zeros(6), "right": np.zeros(6)}
        self.tcp = {"left": np.eye(4), "right": np.eye(4)}
        self.blocked: dict[str, str] = {}

    def readiness_by_side(self, *, required_sides, sample_timestamp):
        del sample_timestamp
        return {side: self.blocked.get(side) for side in required_sides}

    def latest_at_or_before(self, now):
        return {
            side: {
                "actual_qpos": self.qpos[side].copy(),
                "actual_tcp_pose": self.tcp[side].copy(),
            }
            for side in ("left", "right")
        }


class _Recorder:
    def __init__(self, poller: _Poller) -> None:
        self.arm_state_poller = poller
        self.is_recording = False
        self.is_paused = False
        self.finalizing = False

    @property
    def can_record_samples(self) -> bool:
        return self.is_recording and not self.is_paused


class _Session:
    def __init__(self) -> None:
        self.command_override_provider = None
        self.human_command_selection_provider = None
        self.command_guard = None
        self.hold_reasons: list[str] = []
        self.calibration = _Calibration()
        self.state_machine = SimpleNamespace(state=SimpleNamespace(value="ARMED"))
        self.external_commands: list[TeleopCommand] = []

    def hold_hardware(self, reason: str) -> bool:
        self.hold_reasons.append(reason)
        return True

    def dispatch_external_command(
        self,
        command,
        observation,
        now,
        *,
        record,
        transition_to_running,
    ):
        del observation, now
        assert record is False
        assert transition_to_running is False
        self.external_commands.append(command)
        return SimpleNamespace(
            dispatched=True,
            dispatch_receipts={name: True for name in command.commands},
        )


class _HandBackend:
    def __init__(self) -> None:
        self.config = SimpleNamespace(command_dim=0, max_action=1.0)
        self.measured_joint_pos = np.zeros(6, dtype=np.float32)
        self.joint_age_sec = 0.0
        self.fixed_closed_one_shot_owner = True

    @staticmethod
    def send_command(_values) -> None:
        return None

    def get_state(self):
        return {
            "joint_pos": self.measured_joint_pos.copy(),
            "measured_joint_pos": self.measured_joint_pos.copy(),
            "joint_age_sec": float(self.joint_age_sec),
        }

    def set_measured_scalar(self, value: float) -> None:
        self.measured_joint_pos[:] = 0.0
        self.measured_joint_pos[0] = float(value)

    @staticmethod
    def is_ready() -> bool:
        return True

    def fixed_closed_one_shot_ready(self) -> bool:
        return bool(self.fixed_closed_one_shot_owner)


class _HandPoller:
    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=np.float32),
            "right": np.zeros(6, dtype=np.float32),
        }
        self.recovery_phase: dict[str, str] = {}

    def latest_at_or_before(self, _now):
        return {
            side: {
                "joint_pos": values.copy(),
                "joint_age_sec": 0.0,
                **(
                    {"gripper_recovery_phase": self.recovery_phase[side]}
                    if side in self.recovery_phase
                    else {}
                ),
            }
            for side, values in self.qpos.items()
        }


class _Clutch:
    def __init__(self) -> None:
        self.reasons: list[str] = []

    def request_hold(self, *, reason: str) -> None:
        self.reasons.append(reason)


class _RuleUi:
    def __init__(self) -> None:
        self.bound = False
        self.intents: list[str] = []
        self.statuses: list[dict[str, object]] = []
        self.completion_stop_requests = 0
        self.failure_stop_requests = 0
        self.configuration_requests: list[dict[str, object]] = []
        self.configuration_results: list[dict[str, object]] = []
        self.height_calibration_requests: list[dict[str, object]] = []
        self.height_calibration_results: list[str | None] = []
        self.configuration_options: dict[str, object] = {}
        self.selected_configuration: dict[str, object] = {}
        self.messages: list[tuple[str, str]] = []

    def bind_rule_runtime(self) -> None:
        self.bound = True

    def consume_rule_intents(self) -> tuple[str, ...]:
        result = tuple(self.intents)
        self.intents.clear()
        return result

    def publish_rule_status(self, status) -> None:
        self.statuses.append(dict(status))

    def request_rule_completion_stop(self) -> None:
        self.completion_stop_requests += 1

    def request_rule_failure_stop(self) -> None:
        self.failure_stop_requests += 1

    def append_message(self, tag: str, message: str) -> None:
        self.messages.append((str(tag), str(message)))

    def request_start(self) -> None:
        self.intents.append("rule_start")

    def configure_rule_options(self, options, selected) -> None:
        self.configuration_options = dict(options)
        self.selected_configuration = dict(selected)

    def consume_rule_configuration_requests(self):
        result = tuple(self.configuration_requests)
        self.configuration_requests.clear()
        return result

    def publish_rule_configuration_result(self, *, selected, error) -> None:
        self.configuration_results.append({"selected": selected, "error": error})

    def consume_rule_height_calibration_requests(self):
        result = tuple(self.height_calibration_requests)
        self.height_calibration_requests.clear()
        return result

    def publish_rule_height_calibration_result(self, *, error) -> None:
        self.height_calibration_results.append(error)


class _TargetFilter:
    apply_to_target = True
    profile = SimpleNamespace(enabled_sides=("left", "right"))

    def __call__(self, targets):
        normalized = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in targets.items()
        }
        for side, pose in normalized.items():
            rotation = pose[:3, :3]
            if not np.allclose(
                rotation.T @ rotation,
                np.eye(3),
                atol=1e-9,
                rtol=0.0,
            ):
                raise ValueError(f"tool_tcp_target.{side} rotation must be orthonormal")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9, rtol=0.0):
                raise ValueError(f"tool_tcp_target.{side} determinant must be +1")
        return normalized


class _CollisionCheck:
    safe = True
    reason = ""
    min_distance_m = 0.1
    required_min_distance_m = 0.01
    collision_pair_count = 1


class _ArmCalibration:
    T_D_B = np.eye(4)


class _Calibration:
    @staticmethod
    def get_arm_for_side(_side):
        return _ArmCalibration()

    @staticmethod
    def base_to_desk(pose, _side):
        return np.asarray(pose, dtype=float).copy()


class _Retargeter:
    def __init__(self) -> None:
        self.pre_ik_target_safety_filter = _TargetFilter()
        self.current_qpos = {"left": np.zeros(6), "right": np.zeros(6)}
        self.target_pairs = []
        self.command_status = "tracking"

    def check_qpos_pair_collision(self, **_kwargs):
        return _CollisionCheck()

    @staticmethod
    def solver_world_targets_for_qpos_pair(**_kwargs):
        return {"left": np.eye(4), "right": np.eye(4)}

    @staticmethod
    def tool_tcp_world_targets_for_qpos_pair(**_kwargs):
        return {"left": np.eye(4), "right": np.eye(4)}

    def set_current_qpos_pair(self, *, left_qpos, right_qpos, reset_tracking):
        assert reset_tracking is True
        self.current_qpos = {
            "left": np.asarray(left_qpos, dtype=float).copy(),
            "right": np.asarray(right_qpos, dtype=float).copy(),
        }

    @staticmethod
    def solver_world_target_from_tool_tcp_world_target(
        *, side, tool_tcp_world_target
    ):
        del side
        return np.asarray(tool_tcp_world_target, dtype=float).copy()

    def build_command_for_target_pair(self, target):
        self.target_pairs.append(target)
        command = TeleopCommand(
            timestamp=target.timestamp,
            metadata={
                "collision_avoidance_status": self.command_status,
                "collision_avoidance_fallback_reason": None,
            },
        )
        for side in ("left", "right"):
            command.add_command(
                EffectorCommand(
                    effector_name=f"{side}_arm",
                    side=side,
                    command_type="joint_vector",
                    value=self.current_qpos[side].copy(),
                    metadata={"solver_output_qpos": self.current_qpos[side].copy()},
                )
            )
        return command


def _components(session: _Session, *, hand_sides: tuple[str, ...] = ()):
    class CommandBackend:
        def __init__(self) -> None:
            self.resets: list[np.ndarray] = []

        @staticmethod
        def is_emergency_stopped() -> bool:
            return False

        @staticmethod
        def normalize_local_qpos_observation(qpos, *, label):
            del label
            return np.asarray(qpos, dtype=float).copy(), {"status": "exact"}

        def reset_motion_state(self, qpos) -> None:
            self.resets.append(np.asarray(qpos, dtype=float).copy())

    class PreviewBackend:
        def __init__(self) -> None:
            self.qpos = np.zeros(6, dtype=float)

        def set_local_qpos(self, qpos) -> None:
            self.qpos = np.asarray(qpos, dtype=float).copy()

        def get_state(self):
            return {"current_qpos": self.qpos.copy()}

    components = SimpleNamespace(
        session=session,
        active_sides=("left", "right"),
        collision_guard_requested=True,
        dual_arm_retargeter=_Retargeter(),
    )
    for side in ("left", "right"):
        setattr(components, f"{side}_command_backend", CommandBackend())
        setattr(components, f"{side}_preview_backend", PreviewBackend())
        setattr(
            components,
            f"{side}_hand_backend",
            _HandBackend() if side in hand_sides else None,
        )
    return components


def _bind(
    extension: RuleBasedRealTeleopExtension,
    recorder: _Recorder,
    *,
    arm: bool = True,
    hand_sides: tuple[str, ...] = (),
) -> tuple[_Session, _Clutch, _RuleUi]:
    session = _Session()
    components = _components(session, hand_sides=hand_sides)
    clutch = _Clutch()
    ui = _RuleUi()
    extension.attach_components(components)
    extension.bind_runtime(
        components=components,
        clutch_input=clutch,
        camera_sources={},
        dataset_recorder=recorder,
        dataset_recorder_provider=lambda: recorder,
        episode_recorder=SimpleNamespace(),
        control_ui=ui,
        recording_active=lambda: recorder.is_recording,
        pause_recording=lambda *_args, **_kwargs: True,
    )
    extension.start(components=components)
    if arm:
        ui.request_start()
        extension.after_control_input_poll(now=0.0)
    return session, clutch, ui


def _install_arm_only_fix_controller(
    extension: RuleBasedRealTeleopExtension,
) -> FixArmCommandController:
    controller = FixArmCommandController(
        active_sides=("left", "right"),
        required_effectors_by_side={
            "left": ("left_arm",),
            "right": ("right_arm",),
        },
    )
    extension._components.fix_arm_controller = controller
    return controller


def _wait_for_command(session: _Session, *, now: float):
    for _ in range(200):
        override = session.command_override_provider(SimpleNamespace(), now)
        if override.command is not None:
            return override
        time.sleep(0.001)
    raise AssertionError("rule command did not become ready")


def test_writing_height_capture_uses_fresh_async_right_arm_read_and_is_recoverable() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedReplay(
                    step_id="right_write.pen_down",
                    group_key="source",
                    start_index=0,
                    end_index=1,
                ),
            )
        ),
        trajectory_groups={"source": _group()},
        source_selection={"collection_mode": "rule_based_writing"},
    )
    poller = _Poller()
    canonical = _pose(0.2, -0.1, 0.321)
    raw = extension.runtime_pose_frame_contract.canonical_desk_pose_vector_to_storage(
        canonical,
        side="right",
    )
    poller.tcp["right"] = _matrix(raw)
    recorder = _Recorder(poller)
    # Production defers the formal recorder until Episode start.  Height
    # capture must therefore not depend on its optional polling resource.
    recorder.arm_state_poller = None
    session, _clutch, ui = _bind(extension, recorder, arm=False)

    class RightStateBackend:
        def __init__(self) -> None:
            self.success = True
            self.timestamp = 10.0

        @staticmethod
        def is_ready() -> bool:
            return True

        def refresh_state_from_hardware(self, **_kwargs):
            return {
                "hardware_read_success": self.success,
                "hardware_result_code": 0 if self.success else -1,
                "hardware_result_text": "ok" if self.success else "fixture failure",
                "current_tcp_pose": poller.tcp["right"].copy(),
                "measured_timestamp": self.timestamp,
            }

    state_backend = RightStateBackend()
    extension._components.right_state_backend = state_backend

    def settle_capture(now: float) -> None:
        for _ in range(200):
            extension.after_control_input_poll(now=now)
            if not extension.status().height_calibration_pending:
                return
            time.sleep(0.001)
        raise AssertionError("height calibration did not finish")

    assert extension.status().height_calibration_available is True
    ui.height_calibration_requests.append(
        {
            "operation": "capture",
            "lower_tolerance_m": 0.001,
            "upper_tolerance_m": 0.002,
        }
    )
    extension.after_control_input_poll(now=10.0)
    assert extension.status().height_calibration_pending is True
    assert ui.height_calibration_results == []
    settle_capture(10.0)

    captured = extension._height_calibration
    assert isinstance(captured, RuleBasedWritingHeightCalibration)
    assert captured.calibrated_z_m == pytest.approx(0.321)
    assert captured.lower_bound_z_m == pytest.approx(0.320)
    assert captured.upper_bound_z_m == pytest.approx(0.323)
    assert ui.height_calibration_results[-1] is None

    recorder.is_recording = True
    ui.height_calibration_requests.append({"operation": "clear"})
    extension.after_control_input_poll(now=10.5)
    assert extension._height_calibration is captured
    assert "完整停止" in str(ui.height_calibration_results[-1])
    recorder.is_recording = False

    state_backend.success = False
    state_backend.timestamp = 11.0
    ui.height_calibration_requests.append(
        {
            "operation": "capture",
            "lower_tolerance_m": 0.003,
            "upper_tolerance_m": 0.003,
        }
    )
    settle_capture(11.0)
    assert extension._height_calibration is captured
    assert "fixture failure" in str(ui.height_calibration_results[-1])
    assert "fixture failure" in ui.messages[-1][1]

    state_backend.success = True
    session.state_machine.state = SimpleNamespace(value="DEBUG_ONLY")
    ui.height_calibration_requests.append(
        {
            "operation": "capture",
            "lower_tolerance_m": 0.001,
            "upper_tolerance_m": 0.001,
        }
    )
    extension.after_control_input_poll(now=11.5)
    assert extension._height_calibration is captured
    assert "Rig/机械臂尚未安全就绪" in str(ui.height_calibration_results[-1])
    session.state_machine.state = SimpleNamespace(value="ARMED")

    ui.height_calibration_requests.append({"operation": "clear"})
    extension.after_control_input_poll(now=12.0)
    assert extension._height_calibration is None
    assert extension.status().height_calibration_enabled is False


def test_each_alignment_rebases_command_transport_from_measured_snapshot() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        source_selection={"collection_mode": "rule_based_writing"},
    )
    components = _components(_Session())
    extension.attach_components(components)

    class CommandBackend:
        def __init__(self) -> None:
            self.resets: list[np.ndarray] = []
            self.stopped = False

        def is_emergency_stopped(self) -> bool:
            return self.stopped

        @staticmethod
        def normalize_local_qpos_observation(qpos, *, label):
            del label
            return np.asarray(qpos, dtype=float).copy(), {"status": "exact"}

        def reset_motion_state(self, qpos) -> None:
            self.resets.append(np.asarray(qpos, dtype=float).copy())

    class PreviewBackend:
        def __init__(self) -> None:
            self.qpos = np.zeros(6, dtype=float)

        def set_local_qpos(self, qpos) -> None:
            self.qpos = np.asarray(qpos, dtype=float).copy()

        def get_state(self):
            return {"current_qpos": self.qpos.copy()}

    command_backends = {
        side: CommandBackend() for side in ("left", "right")
    }
    preview_backends = {
        side: PreviewBackend() for side in ("left", "right")
    }
    for side in ("left", "right"):
        setattr(components, f"{side}_command_backend", command_backends[side])
        setattr(components, f"{side}_preview_backend", preview_backends[side])
    components.position_execution_qvel_feedback_enabled = False

    canonical = _pose(0.2, -0.1, 0.321)
    raw = extension.runtime_pose_frame_contract.canonical_desk_pose_vector_to_storage(
        canonical,
        side="right",
    )
    extension._commit_writing_height(
        {
            "current_tcp_pose": _matrix(raw),
            "measured_timestamp": 10.0,
        },
        now=10.0,
        requested_at=9.9,
        lower_tolerance_m=0.001,
        upper_tolerance_m=0.001,
    )
    start = {
        "left": np.linspace(-0.4, 0.1, 6),
        "right": np.linspace(0.2, -0.3, 6),
    }

    extension._rebase_motion_transport_from_measured_start(start)

    for side in ("left", "right"):
        assert len(command_backends[side].resets) == 1
        np.testing.assert_allclose(command_backends[side].resets[0], start[side])
        np.testing.assert_allclose(preview_backends[side].qpos, start[side])
        np.testing.assert_allclose(
            components.dual_arm_retargeter.current_qpos[side],
            start[side],
        )

    # A later Episode without a new height calibration must rebase again;
    # the call site is one per planned transition, never each planner poll.
    next_start = {side: value + 0.05 for side, value in start.items()}
    extension._rebase_motion_transport_from_measured_start(next_start)
    for side in ("left", "right"):
        assert len(command_backends[side].resets) == 2
        np.testing.assert_allclose(command_backends[side].resets[1], next_start[side])
        np.testing.assert_allclose(preview_backends[side].qpos, next_start[side])


def test_alignment_rebase_normalizes_only_tiny_measured_limit_roundoff() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    components = _components(_Session())
    for side in ("left", "right"):
        setattr(
            components,
            f"{side}_command_backend",
            RM65ArmBackend(
                RM65ArmBackendConfig(
                    arm_ip="192.0.2.1",
                    side=side,
                    connect_hardware=False,
                    initial_qpos=(0.0,) * 6,
                    singularity_protection_enabled=False,
                )
            ),
        )
    extension.attach_components(components)
    measured = {"left": np.zeros(6), "right": np.zeros(6)}
    measured["right"][2] = 2.355008

    anchor = extension._rebase_motion_transport_from_measured_start(measured)

    assert measured["right"][2] == pytest.approx(2.355008)
    assert anchor["right"][2] == pytest.approx(2.355)
    assert components.right_preview_backend.qpos[2] == pytest.approx(2.355)
    assert components.right_command_backend.get_state()["current_qpos"][2] == pytest.approx(2.355)

    outside = {side: qpos.copy() for side, qpos in measured.items()}
    outside["right"][2] = 2.355 + np.deg2rad(0.02)
    with pytest.raises(RuleBasedInitialAlignmentPlanningError, match="local-observation tolerance"):
        extension._rebase_motion_transport_from_measured_start(outside)
    assert components.right_preview_backend.qpos[2] == pytest.approx(2.355)


def test_coupled_pose_solver_reseeds_only_at_new_rule_step() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    extension._ensure_pose_action_motion_intent_step("right_write.pen_down")
    extension._coupled_pose_solver_seeded = True
    extension._ensure_pose_action_motion_intent_step("right_write.pen_down")
    assert extension._coupled_pose_solver_seeded is True

    extension._ensure_pose_action_motion_intent_step("right_write.write")
    assert extension._coupled_pose_solver_seeded is False


def test_height_calibration_rebase_never_rearms_stopped_arm() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    components = _components(_Session())
    extension.attach_components(components)

    class StoppedBackend:
        @staticmethod
        def is_emergency_stopped() -> bool:
            return True

        @staticmethod
        def reset_motion_state(_qpos) -> None:
            raise AssertionError("stopped backend must not be reset")

    class PreviewBackend:
        @staticmethod
        def set_local_qpos(_qpos) -> None:
            raise AssertionError("preview must not change after failed preflight")

    for side in ("left", "right"):
        setattr(components, f"{side}_command_backend", StoppedBackend())
        setattr(components, f"{side}_preview_backend", PreviewBackend())
    with pytest.raises(RuntimeError, match="runtime Home is required"):
        extension._rebase_motion_transport_from_measured_start(
            {"left": np.zeros(6), "right": np.zeros(6)}
        )


def test_writing_replay_height_projection_is_lower_floor_only_and_opt_in() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedReplay(
                    step_id="right_write.pen_down",
                    group_key="source",
                    start_index=0,
                    end_index=1,
                ),
            )
        ),
        trajectory_groups={"source": _group()},
        source_selection={"collection_mode": "rule_based_writing"},
    )
    original = extension.adapter.next_runtime(None)
    assert extension._project_writing_replay_height(original) is original
    extension._height_calibration = RuleBasedWritingHeightCalibration(
        calibrated_z_m=0.600,
        lower_tolerance_m=0.001,
        upper_tolerance_m=0.001,
        captured_at=1.0,
        source_timestamp=None,
        pose_frame_contract_hash=extension.runtime_pose_frame_contract.contract_hash,
    )

    projected = extension._project_writing_replay_height(original)
    projected_canonical = (
        extension.runtime_pose_frame_contract.storage_pose_vector_to_canonical_desk(
            projected.action.arm_pose_by_side["right"],
            side="right",
        )
    )
    original_canonical = (
        extension.runtime_pose_frame_contract.storage_pose_vector_to_canonical_desk(
            original.action.arm_pose_by_side["right"],
            side="right",
        )
    )
    assert projected_canonical[2] == pytest.approx(0.599)
    assert np.allclose(
        projected_canonical[[0, 1, 3, 4, 5, 6]],
        original_canonical[[0, 1, 3, 4, 5, 6]],
    )
    assert np.array_equal(
        projected.action.arm_pose_by_side["left"],
        original.action.arm_pose_by_side["left"],
    )
    assert projected.action.metadata["rule_based_height_clipped"] is True

    pen_up = replace(
        original,
        directive=replace(original.directive, step_id="right_write.pen_up"),
        action=replace(
            original.action,
            arm_pose_by_side={
                **dict(original.action.arm_pose_by_side),
                "right": extension.runtime_pose_frame_contract.canonical_desk_pose_vector_to_storage(
                    _pose(0.0, 0.0, 0.650),
                    side="right",
                ),
            },
        ),
    )
    unbounded_above = extension._project_writing_replay_height(pen_up)
    assert np.array_equal(
        unbounded_above.action.arm_pose_by_side["right"],
        pen_up.action.arm_pose_by_side["right"],
    )
    assert unbounded_above.action.metadata["rule_based_height_clipped"] is False


def test_calibrated_height_descent_uses_existing_coupled_safety_and_admission() -> None:
    source = CalibratedHeightDescentActionSource(
        reference_hz=60.0,
        linear_speed_m_s=0.05,
    )
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedActionSegment(
                step_id="right_write.pen_down.height_descent",
                action_source=source,
                stop_condition=RuleBasedSegmentStopCondition(
                    max_steps=source.max_steps,
                    min_steps=1,
                    stop_when=lambda context: bool(
                        context.action.metadata.get(
                            "rule_based_height_descent_terminal",
                            False,
                        )
                    ),
                ),
                capture=RuleBasedCaptureSelection(enabled=True),
                allow_empty=True,
                planned_steps_hint=0,
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
        source_selection={"collection_mode": "rule_based_writing"},
        config=RuleBasedRealTeleopConfig(
            position_reference_hz=60.0,
            settle_consecutive_samples=1,
        ),
    )
    poller = _Poller()
    for side, canonical in {
        "left": _pose(0.0, 0.0, 0.45),
        "right": _pose(0.2, -0.1, 0.303),
    }.items():
        raw = extension.runtime_pose_frame_contract.canonical_desk_pose_vector_to_storage(
            canonical,
            side=side,
        )
        poller.tcp[side] = _matrix(raw)
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, ui = _bind(extension, recorder, arm=False)
    extension._height_calibration = RuleBasedWritingHeightCalibration(
        calibrated_z_m=0.300,
        lower_tolerance_m=0.001,
        upper_tolerance_m=0.001,
        captured_at=1.0,
        source_timestamp=1.0,
        pose_frame_contract_hash=extension.runtime_pose_frame_contract.contract_hash,
    )
    ui.request_start()
    extension.after_control_input_poll(now=2.0)

    override = _wait_for_command(session, now=2.1)
    command = override.command
    assert command.metadata["rule_based_step_id"] == (
        "right_write.pen_down.height_descent"
    )
    assert command.metadata["rule_based_coupled_pose_safety"] is True
    assert command.metadata["rule_based_height_descent"] is True
    assert command.metadata["rule_based_action_period_sec"] == pytest.approx(1.0 / 60.0)
    assert len(extension._components.dual_arm_retargeter.target_pairs) == 1
    assert 0 < extension.status().planned_sample_count < int(
        command.metadata["rule_based_height_descent_total"]
    )

    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=2.1,
    )
    admission = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True),
        now=2.1,
    )
    assert admission.allowed is True
    assert admission.metadata["rule_based_height_descent"] is True
    assert admission.metadata["rule_based_height_calibration"]["enabled"] is True


def test_rule_start_allows_only_pre_episode_armed_warmup_without_motion() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    recorder = _Recorder(_Poller())
    session, _clutch, _ui = _bind(extension, recorder)

    assert runtime_extension_allows_pre_episode_armed_warmup(extension)
    assert session.command_override_provider(
        SimpleNamespace(),
        1.0,
    ).command is None
    assert not extension.status().execution_begun

    extension._held = True
    assert not runtime_extension_allows_pre_episode_armed_warmup(extension)


def test_rule_start_waits_for_right_one_shot_grip_owner_without_disabling_release() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedInitialize(step_id="init", group_key="source"),
            )
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
        source_selection={"collection_mode": "rule_based_writing"},
    )
    recorder = _Recorder(_Poller())
    session, _clutch, _ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )
    extension._components.right_hand_backend.fixed_closed_one_shot_ready = (
        lambda: False
    )

    status = extension.status()
    assert status.rule_start_available is False
    assert status.gripper_control_available is True
    assert status.readiness_blockers["rule.right_hand_hold_owner"]
    assert extension.allows_pre_episode_armed_warmup() is False
    assert session.external_commands == []


def test_rule_runtime_exposes_configured_hand_fault_before_writer_open() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                    ),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, _ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )

    class _UnreadyHandBackend(_HandBackend):
        @staticmethod
        def is_ready() -> bool:
            return False

        @staticmethod
        def readiness_report() -> dict[str, object]:
            return {"ready": False, "reason": "DMTAC fault"}

    extension._components.right_hand_backend = _UnreadyHandBackend()
    blockers = extension.collection_startup_blockers()

    assert blockers == {"right_hand": "right_hand DMTAC fault"}
    assert session.state_machine.state.value == "ARMED"


def test_pre_dispatch_hand_fault_cancels_armed_attempt_instead_of_stranding_countdown() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                    ),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())
    session, clutch, ui = _bind(
        extension,
        recorder,
        arm=True,
        hand_sides=("right",),
    )
    session.rig = SimpleNamespace(
        readiness_report=lambda: {
            "ready": False,
            "reason": "unready backends: right_hand",
            "backends": {
                "right_hand": {
                    "ready": False,
                    "reason": "telemetry_failed: Write timeout",
                }
            },
        }
    )
    right_backend = extension._components.right_hand_backend
    right_backend.is_ready = lambda: False
    right_backend.readiness_report = lambda: {
        "ready": False,
        "reason": "telemetry_failed: Write timeout",
    }

    extension.after_control_input_poll(now=1.0)

    status = extension.status()
    assert status.armed is False
    assert status.aborted is True
    assert status.held is True
    assert status.waiting_reason == "session_readiness_failure"
    assert session.hold_reasons
    assert clutch.reasons == ["rule_based_runtime_hold"]
    assert ui.failure_stop_requests == 1
    assert any("Write timeout" in message for _tag, message in ui.messages)
    assert any(
        tag == "RuleBased"
        and "规则运行时因以下原因请求 Hold" in message
        and "Write timeout" in message
        for tag, message in ui.messages
    )


def test_rule_status_never_claims_start_ready_while_session_is_debug_only() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                    ),
                ),
            )
        ),
        trajectory_groups={},
        source_selection={
            "selection_ordinal": 3,
            "phases": {
                "pen_down": {"candidate_id": "approach-3"},
                "write": {"candidate_id": "write-3"},
                "pen_up": {"candidate_id": "depart-3"},
            },
        },
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(extension, recorder, arm=False)

    session.state_machine.state = SimpleNamespace(value="DEBUG_ONLY")
    status = extension.status()

    assert status.rule_start_available is False
    assert status.readiness_blockers == {
        "rig.session": (
            "Session state=DEBUG_ONLY; runtime Home and hardware arm admission "
            "are required before rule collection."
        )
    }
    assert status.waiting_reason == "session_readiness_failure"
    publish_count = len(ui.statuses)
    extension._publish_status()
    assert len(ui.statuses) == publish_count + 1
    assert ui.statuses[-1]["rule_start_available"] is not True
    assert ui.statuses[-1]["source_selection"]["selection_ordinal"] == 3
    assert ui.statuses[-1]["source_selection"]["phases"]["pen_down"] == {
        "candidate_id": "approach-3"
    }
    extension._publish_status()
    assert len(ui.statuses) == publish_count + 1


def test_rule_status_throttles_high_rate_progress_but_not_lifecycle_edges(
    monkeypatch,
) -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                    ),
                ),
            )
        ),
        trajectory_groups={},
    )
    recorder = _Recorder(_Poller())
    _session, _clutch, ui = _bind(extension, recorder)
    clock = [100.0]
    monkeypatch.setattr(rule_runtime_module.time, "monotonic", lambda: clock[0])

    extension._publish_status(force=True)
    published = len(ui.statuses)
    extension._stage_frame_index = 12
    extension._collection_position = "阶段 1 / 3 · 帧 12 / 100"
    extension._admitted_capture_sequences.add(4)
    extension._publish_status()
    assert len(ui.statuses) == published

    extension._waiting_reason = "formal_writer_paused"
    extension._publish_status()
    assert len(ui.statuses) == published + 1


def test_operator_hold_resets_attempt_and_home_reopens_rule_start() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, ui = _bind(extension, recorder, arm=True)
    extension._execution_begun = True
    extension._admitted_capture_sequences.update({3, 4})
    extension._boundaries["old"] = SimpleNamespace()
    extension._stage_label = "write"
    extension._collection_position = "old position"
    transition_pending = [True]
    extension._collection_transition_active = lambda: transition_pending[0]

    ui.intents.append("rule_stop")
    extension.after_control_input_poll(now=1.0)

    assert extension.status().aborted is False
    assert extension.restarts_after_hold(reason="rule_based_runtime_hold") is True
    assert extension.discards_episode_on_stop(reason="rule_based_hold_cancel") is True

    extension.after_runtime_hold(
        reason="rule_based_runtime_hold",
        hardware_hold_confirmed=True,
    )
    held = extension.status()
    assert held.held is True
    assert held.aborted is False
    assert held.execution_begun is False
    assert held.published_boundaries == ()
    assert held.stage_label is None
    assert held.collection_position is None
    assert extension._admitted_capture_sequences == set()
    assert held.readiness_blockers["dataset.lifecycle"] == (
        "cancelled rule Episode lifecycle is still settling"
    )

    recorder.is_recording = False
    with pytest.raises(RuntimeError, match="Episode lifecycle is still settling"):
        extension.after_runtime_home(now=1.5)
    transition_pending[0] = False
    recorder.is_recording = True  # The raced Episode-open landed after Hold.
    extension.after_control_input_poll(now=1.75)
    assert ui.failure_stop_requests == 1
    assert extension.status().readiness_blockers["dataset.lifecycle"] == (
        "cancelled rule Episode is still active and awaiting discard"
    )
    recorder.is_recording = False  # The retried discard has now completed.
    extension.after_runtime_home(now=2.0)
    recovered = extension.status()
    assert recovered.held is False
    assert recovered.rule_start_available is True

    ui.request_start()
    extension.after_control_input_poll(now=2.1)
    assert extension.status().armed is True


def test_failed_rule_attempt_becomes_runtime_home_retryable_after_confirmed_hold() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(extension, recorder, arm=True)
    extension._execution_begun = True
    extension._abort_session_step("fixture dispatch rejection")
    assert extension.status().aborted is True
    assert extension.restarts_after_hold(reason="rule_based_runtime_hold") is True

    session.state_machine.state = SimpleNamespace(value="DEBUG_ONLY")
    extension.after_runtime_hold(
        reason="rule_based_runtime_hold",
        hardware_hold_confirmed=True,
    )

    status = extension.status()
    assert status.aborted is False
    assert status.execution_begun is False
    assert status.held is True
    assert status.waiting_reason == "session_readiness_failure"
    assert "rig.session" in status.readiness_blockers
    assert ui.failure_stop_requests == 1


def test_external_hold_resets_armed_attempt_but_completion_still_finalizes() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                    ),
                ),
            )
        ),
        trajectory_groups={},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    _session, _clutch, _ui = _bind(extension, recorder, arm=True)

    assert extension.restarts_after_hold(reason="position_force_guard_collision")
    extension.after_runtime_hold(
        reason="position_force_guard_collision",
        hardware_hold_confirmed=True,
    )

    assert extension.status().held is True
    assert extension.status().aborted is False
    assert extension.discards_episode_on_stop(reason="force_collision") is True
    extension._completed = True
    assert extension.discards_episode_on_stop(reason="rule_based_complete") is False


def test_rule_readiness_diagnosis_is_bounded_and_force_refreshable() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                    ),
                ),
            )
        ),
        trajectory_groups={},
    )
    recorder = _Recorder(_Poller())
    session, _clutch, _ui = _bind(extension, recorder, arm=False)
    session.state_machine.state = SimpleNamespace(value="DEBUG_ONLY")
    calls = 0

    class Rig:
        def readiness_report(self):
            nonlocal calls
            calls += 1
            return {
                "ready": False,
                "reason": "unready backends: right_arm",
                "backends": {
                    "right_arm": {"ready": False, "reason": "offline"},
                },
            }

    session.rig = Rig()
    extension.status()
    extension.status()
    assert calls == 1
    extension.refresh_status(force=True)
    assert calls == 2


def test_rule_readiness_diagnosis_caches_observational_failures() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                    ),
                ),
            )
        ),
        trajectory_groups={},
    )
    recorder = _Recorder(_Poller())
    session, _clutch, _ui = _bind(extension, recorder, arm=False)
    session.state_machine.state = SimpleNamespace(value="DEBUG_ONLY")
    calls = 0

    class BrokenRig:
        def readiness_report(self):
            nonlocal calls
            calls += 1
            raise RuntimeError("serial diagnostic unavailable")

    session.rig = BrokenRig()
    first = extension.status()
    second = extension.status()
    assert calls == 1
    assert first.readiness_blockers == second.readiness_blockers == {
        "rig.session": (
            "readiness report failed: RuntimeError: serial diagnostic unavailable"
        )
    }
    extension.refresh_status(force=True)
    assert calls == 2


def test_rule_status_blocks_when_configured_hand_was_removed_from_rig() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                    ),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, _ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )
    components = extension._components
    components.right_hand_control_via_rig = True
    session.rig = SimpleNamespace(
        backends={"left_arm": object(), "right_arm": object()},
        readiness_report=lambda: {"ready": True, "backends": {}},
    )

    status = extension.status()

    assert status.rule_start_available is False
    assert status.readiness_blockers == {
        "right_hand": "backend missing from Session Rig",
    }
    assert status.waiting_reason == "session_readiness_failure"


def test_rule_attachment_restores_missing_hand_backend_binding() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                    ),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    session = _Session()
    components = _components(session, hand_sides=("right",))
    session.rig = SimpleNamespace(backends={"left_arm": object(), "right_arm": object()})

    extension.attach_components(components)

    assert session.rig.backends["right_hand"] is components.right_hand_backend


def test_active_episode_async_alignment_bootstrap_does_not_pause_or_admit_samples() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedInitialize(step_id="init", group_key="source"),
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                    ),
                    capture=RuleBasedCaptureSelection(enabled=True),
                ),
            )
        ),
        trajectory_groups={"source": _group()},
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(extension, recorder, arm=False)
    ui.request_start()
    extension.after_control_input_poll(now=1.0)

    assert not runtime_extension_allows_active_episode_bootstrap_without_dispatch(
        extension
    )
    recorder.is_recording = True
    planning = session.command_override_provider(SimpleNamespace(), 1.1)

    assert planning.command is None
    assert runtime_extension_allows_active_episode_bootstrap_without_dispatch(
        extension
    )
    assert not control_loop._recording_requires_arm_dispatch_pause(
        arm_dispatch_confirmed=False,
        recording_active=True,
        recording_paused=False,
        runtime_extension=extension,
    )
    admission = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=False),
        now=1.1,
    )
    assert admission.allowed is False

    # A transient modality pause must retain the bootstrap identity so the
    # existing episode-resume warmup can run in ARMED and unblock the writer.
    recorder.is_paused = True
    assert runtime_extension_allows_active_episode_bootstrap_without_dispatch(
        extension
    )
    paused = session.command_override_provider(SimpleNamespace(), 1.2)
    assert paused.command is None
    assert extension.status().waiting_reason == "formal_writer_paused"


def test_progress_fix_is_deferred_until_final_dual_arm_alignment_arrives() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedInitialize(step_id="init", group_key="source"),
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                    ),
                    capture=RuleBasedCaptureSelection(enabled=True),
                ),
            )
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=1),
    )
    poller = _Poller()
    for side in ("left", "right"):
        poller.qpos[side] = np.full(6, -0.02)
    poller.qpos = {
        "left": np.full(6, -0.02),
        "right": np.full(6, -0.02),
    }
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, ui = _bind(extension, recorder, arm=False)
    controller = _install_arm_only_fix_controller(extension)

    assert extension.defers_progress_entry_arm_control(
        target_progress_step_id="right_write",
        current_fixed_sides=(),
        target_fixed_sides=("left",),
        fixed_scope="arm_only",
    )
    assert controller.fixed_sides == ()
    ui.request_start()
    extension.after_control_input_poll(now=10.0)

    now = 10.1
    current = _wait_for_command(session, now=now)
    initialization_count = 0
    while current.command.metadata["rule_based_directive_kind"] == "initialize":
        initialization_count += 1
        filtered = controller.filter_command(
            current.command,
            SimpleNamespace(),
            now,
        )
        assert session.command_guard(filtered, SimpleNamespace(), now) is None
        for side in ("left", "right"):
            poller.qpos[side] = np.asarray(
                filtered.commands[f"{side}_arm"].value,
                dtype=float,
            ).copy()
        metadata = filtered.metadata
        final_waypoint = (
            int(metadata["rule_based_initial_alignment_waypoint_index"]) + 1
            == int(metadata["rule_based_initial_alignment_waypoint_count"])
        )
        assert (
            f"路点 {int(metadata['rule_based_initial_alignment_waypoint_index']) + 1} "
            f"/ {int(metadata['rule_based_initial_alignment_waypoint_count'])}"
            in str(extension.status().collection_position)
        )
        extension.after_session_step(
            step_result=SimpleNamespace(
                command=filtered,
                command_dispatched=True,
            ),
            now=now,
        )
        assert controller.fixed_sides == (("left",) if final_waypoint else ())
        now += 0.051
        current = _wait_for_command(session, now=now)

    assert initialization_count >= 2
    assert extension._deferred_progress_arm_fix_established is True
    collect = controller.filter_command(
        current.command,
        SimpleNamespace(),
        now,
    )
    assert collect.commands["left_arm"].metadata["fix_arm_active"] is True
    assert (
        collect.commands["left_arm"].metadata["fix_arm_snapshot_origin"]
        == "rule_initial_alignment_arrived"
    )
    extension.after_session_step(
        step_result=SimpleNamespace(command=collect, command_dispatched=True),
        now=now,
    )
    admission = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True),
        now=now,
    )
    assert admission.allowed is True
    assert admission.metadata["rule_based_progress_entry_fix_established"] is True


def test_first_float32_pen_down_pose_runs_after_alignment_and_deferred_fix() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedInitialize(step_id="init", group_key="source"),
                RuleBasedReplay(
                    step_id="right_write.pen_down",
                    group_key="source",
                    start_index=0,
                    end_index=1,
                    capture=RuleBasedCaptureSelection(enabled=True),
                ),
            )
        ),
        trajectory_groups={"source": _float32_rotated_group()},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=1),
    )
    poller = _Poller()
    for side in ("left", "right"):
        poller.qpos[side] = np.full(6, -0.02)
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, ui = _bind(extension, recorder, arm=False)
    controller = _install_arm_only_fix_controller(extension)
    assert extension.defers_progress_entry_arm_control(
        target_progress_step_id="right_write",
        current_fixed_sides=(),
        target_fixed_sides=("left",),
        fixed_scope="arm_only",
    )
    ui.request_start()
    extension.after_control_input_poll(now=10.0)

    now = 10.1
    current = _wait_for_command(session, now=now)
    while current.command.metadata["rule_based_directive_kind"] == "initialize":
        filtered = controller.filter_command(current.command, SimpleNamespace(), now)
        assert session.command_guard(filtered, SimpleNamespace(), now) is None
        for side in ("left", "right"):
            poller.qpos[side] = np.asarray(
                filtered.commands[f"{side}_arm"].value,
                dtype=float,
            ).copy()
        extension.after_session_step(
            step_result=SimpleNamespace(
                command=filtered,
                command_dispatched=True,
            ),
            now=now,
        )
        now += 0.051
        next_override = session.command_override_provider(SimpleNamespace(), now)
        assert next_override.command is not None, extension.status()
        current = next_override

    assert current.command.metadata["rule_based_step_id"] == "right_write.pen_down"
    assert current.command.metadata["rule_based_coupled_pose_safety"] is True
    assert extension.status().aborted is False
    diagnostics = current.command.metadata[
        "rule_based_rigid_pose_projection_by_side"
    ]
    assert diagnostics["left"]["rotation_projected"] is True
    assert diagnostics["right"]["rotation_projected"] is True
    filtered = controller.filter_command(current.command, SimpleNamespace(), now)
    # The real Session guard runs after right-arm IK has attached its accepted
    # TCP target, while the left FixArm command intentionally remains qpos.
    # Reproduce that exact final-command boundary in this lightweight fixture.
    filtered.commands["right_arm"].metadata["target_pose_base"] = _matrix(
        _float32_pose_with_nontrivial_rotation()
    )
    assert session.command_guard(filtered, SimpleNamespace(), now) is None
    canonical_action = validate_collection_command_action(
        filtered,
        allow_backend_target_pose_defer=True,
    )
    assert canonical_action.action_space == BIMANUAL_MIXED_V1
    extension.after_session_step(
        step_result=SimpleNamespace(command=filtered, command_dispatched=True),
        now=now,
    )
    admission = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True),
        now=now,
    )
    assert admission.allowed is True


def test_filtered_initialization_rewrite_fails_closed_before_dispatch() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=1),
    )
    poller = _Poller()
    poller.qpos = {
        "left": np.full(6, -0.02),
        "right": np.full(6, -0.02),
    }
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, clutch, _ui = _bind(extension, recorder, arm=False)
    controller = _install_arm_only_fix_controller(extension)
    assert extension.defers_progress_entry_arm_control(
        target_progress_step_id="right_write",
        current_fixed_sides=(),
        target_fixed_sides=("left",),
        fixed_scope="arm_only",
    )
    extension._control_ui.request_start()
    extension.after_control_input_poll(now=20.0)
    current = _wait_for_command(session, now=20.1)
    rewritten = TeleopCommand(
        commands={
            name: command.copy()
            for name, command in current.command.commands.items()
        },
        timestamp=current.command.timestamp,
        metadata=dict(current.command.metadata),
    )
    rewritten.commands["left_arm"].value = (
        np.asarray(rewritten.commands["left_arm"].value, dtype=float) + 0.01
    )

    fault = session.command_guard(rewritten, SimpleNamespace(), 20.1)

    assert fault is not None
    assert fault.code == "rule_based_initial_alignment_command_rewritten"
    assert "left_arm" in fault.metadata["mismatch"]
    assert controller.fixed_sides == ()
    extension.after_session_step(
        step_result=SimpleNamespace(
            state=SimpleNamespace(value="FAULT"),
            faults=(fault,),
            command=rewritten,
            command_dispatched=False,
        ),
        now=20.1,
    )
    assert extension.status().aborted is True
    assert extension.status().held is True
    assert clutch.reasons == ["rule_based_runtime_hold"]


def test_right_only_second_alignment_preserves_established_left_fix() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedInitialize(
                    step_id="prelude_alignment",
                    group_key="source",
                    active_sides=("left", "right"),
                ),
                RuleBasedInitialize(
                    step_id="pen_down_alignment",
                    group_key="source",
                    frame_index=1,
                    active_sides=("right",),
                ),
            )
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=1),
    )
    poller = _Poller()
    poller.qpos = {
        "left": np.full(6, -0.02),
        "right": np.full(6, -0.02),
    }
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, ui = _bind(extension, recorder, arm=False)
    controller = _install_arm_only_fix_controller(extension)
    assert extension.defers_progress_entry_arm_control(
        target_progress_step_id="right_write",
        current_fixed_sides=(),
        target_fixed_sides=("left",),
        fixed_scope="arm_only",
    )
    ui.request_start()
    extension.after_control_input_poll(now=30.0)

    now = 30.1
    current = _wait_for_command(session, now=now)
    while current.command.metadata["rule_based_step_id"] == "prelude_alignment":
        filtered = controller.filter_command(current.command, SimpleNamespace(), now)
        assert session.command_guard(filtered, SimpleNamespace(), now) is None
        for side in ("left", "right"):
            poller.qpos[side] = np.asarray(
                filtered.commands[f"{side}_arm"].value,
                dtype=float,
            ).copy()
        extension.after_session_step(
            step_result=SimpleNamespace(
                command=filtered,
                command_dispatched=True,
            ),
            now=now,
        )
        now += 0.051
        current = _wait_for_command(session, now=now)

    assert current.command.metadata["rule_based_step_id"] == "pen_down_alignment"
    assert controller.fixed_sides == ("left",)
    filtered = controller.filter_command(current.command, SimpleNamespace(), now)
    assert filtered.commands["left_arm"].metadata["fix_arm_active"] is True
    np.testing.assert_allclose(
        filtered.commands["left_arm"].value,
        current.command.commands["left_arm"].value,
        atol=1e-10,
        rtol=0.0,
    )
    assert session.command_guard(filtered, SimpleNamespace(), now) is None
    assert extension.status().aborted is False


def test_deferred_fix_establishment_failure_holds_and_reclaims_episode() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=1),
    )
    poller = _Poller()
    poller.qpos = {
        "left": np.full(6, -0.02),
        "right": np.full(6, -0.02),
    }
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, clutch, ui = _bind(extension, recorder, arm=False)
    controller = _install_arm_only_fix_controller(extension)
    controller.freeze_arm_only_from_command = lambda **_kwargs: SimpleNamespace(
        ok=False,
        message="injected Fix failure",
    )
    assert extension.defers_progress_entry_arm_control(
        target_progress_step_id="right_write",
        current_fixed_sides=(),
        target_fixed_sides=("left",),
        fixed_scope="arm_only",
    )
    ui.request_start()
    extension.after_control_input_poll(now=30.0)

    now = 30.1
    for _ in range(500):
        current = _wait_for_command(session, now=now)
        filtered = controller.filter_command(current.command, SimpleNamespace(), now)
        assert session.command_guard(filtered, SimpleNamespace(), now) is None
        for side in ("left", "right"):
            poller.qpos[side] = np.asarray(
                filtered.commands[f"{side}_arm"].value,
                dtype=float,
            ).copy()
        extension.after_session_step(
            step_result=SimpleNamespace(
                state=SimpleNamespace(value="RUNNING"),
                faults=(),
                command=filtered,
                command_dispatched=True,
            ),
            now=now,
        )
        if extension.status().aborted:
            break
        now += 1.0 / 60.0

    assert extension.status().aborted is True
    assert extension.status().held is True
    assert extension._deferred_progress_arm_fix_established is False
    assert controller.fixed_sides == ()
    assert clutch.reasons == ["rule_based_runtime_hold"]
    assert ui.failure_stop_requests == 1
    assert any(
        tag == "RuleBased" and "injected Fix failure" in message
        for tag, message in ui.messages
    )


def test_rule_start_is_unavailable_with_preexisting_fix_authority() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(extension, recorder, arm=False)
    controller = _install_arm_only_fix_controller(extension)
    transition = controller.freeze_arm_only_from_command(
        side="left",
        command=EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.zeros(6),
        ),
        now=1.0,
        snapshot_origin="test_preexisting_fix",
    )
    assert transition.ok is True
    assert controller.fixed_sides == ("left",)
    assert extension.status().rule_start_available is False

    ui.request_start()
    extension.after_control_input_poll(now=1.1)

    assert extension.status().armed is False
    assert extension.status().waiting_reason == "unexpected_preexisting_fix_arm"
    assert session.command_override_provider(SimpleNamespace(), 1.2).command is None


def test_first_initial_alignment_uses_fresh_causal_state_after_runtime_home() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    later_causal = {
        side: {"actual_qpos": np.zeros(6, dtype=float)}
        for side in ("left", "right")
    }

    selected = extension._initial_alignment_start_qpos_pair(later_causal)

    np.testing.assert_array_equal(
        selected["left"], later_causal["left"]["actual_qpos"]
    )
    np.testing.assert_array_equal(
        selected["right"], later_causal["right"]["actual_qpos"]
    )

    extension._execution_begun = True
    current_after_dispatch = extension._initial_alignment_start_qpos_pair(
        later_causal
    )
    np.testing.assert_array_equal(
        current_after_dispatch["left"], later_causal["left"]["actual_qpos"]
    )


def test_initial_alignment_rejects_actual_arm_drift_during_background_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    planner = extension._initial_alignment_planner
    assert planner is not None
    original_plan = planner.plan
    entered = threading.Event()
    release = threading.Event()

    def blocked_plan(**kwargs):
        entered.set()
        if not release.wait(timeout=2.0):
            raise AssertionError("test did not release the planning worker")
        return original_plan(**kwargs)

    monkeypatch.setattr(planner, "plan", blocked_plan)
    assert session.command_override_provider(SimpleNamespace(), 2.0).command is None
    assert entered.wait(timeout=2.0)
    recorder.arm_state_poller.qpos["right"][2] = np.deg2rad(0.5)
    release.set()
    for _ in range(200):
        if extension._initial_alignment_future.done():
            break
        time.sleep(0.001)

    with pytest.raises(
        RuntimeError,
        match=(
            "actual arm state drifted while the transition was being planned: "
            "side=right joint=3"
        ),
    ):
        session.command_override_provider(SimpleNamespace(), 2.1)
    assert extension.status().aborted is True
    assert extension.status().held is True


def test_non_dispatched_rule_frame_holds_and_automatically_reclaims_episode() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, clutch, ui = _bind(extension, recorder)
    first = _wait_for_command(session, now=3.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=first.command,
            command_dispatched=False,
            state=SimpleNamespace(value="ARMED"),
            faults=(),
        ),
        now=3.0,
    )

    assert extension.status().aborted is True
    assert extension.status().held is True
    assert session.hold_reasons == ["session did not dispatch the active rule frame"]
    assert clutch.reasons == ["rule_based_runtime_hold"]
    assert ui.failure_stop_requests == 1
    assert not runtime_extension_allows_active_episode_bootstrap_without_dispatch(
        extension
    )


def test_initial_alignment_planning_timeout_holds_and_requests_episode_reclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, clutch, ui = _bind(extension, recorder)
    planner = extension._initial_alignment_planner
    assert planner is not None
    planner.config = replace(planner.config, planning_timeout_sec=0.01)
    original_plan = planner.plan
    entered = threading.Event()
    release = threading.Event()

    def blocked_plan(**kwargs):
        entered.set()
        if not release.wait(timeout=2.0):
            raise AssertionError("test did not release the planning worker")
        return original_plan(**kwargs)

    monkeypatch.setattr(planner, "plan", blocked_plan)
    assert session.command_override_provider(SimpleNamespace(), 4.0).command is None
    assert entered.wait(timeout=1.0)
    future = extension._initial_alignment_future
    assert future is not None
    try:
        with pytest.raises(
            RuleBasedInitialAlignmentPlanningError,
            match="exceeded its bounded timeout",
        ):
            session.command_override_provider(SimpleNamespace(), 4.02)
    finally:
        release.set()
        future.result(timeout=2.0)

    assert extension.status().aborted is True
    assert extension.status().held is True
    assert clutch.reasons == ["rule_based_runtime_hold"]
    assert ui.failure_stop_requests == 1


@pytest.mark.parametrize(
    "blocked_reason", ("joint_limit_hold_margin", "collision_soft")
)
def test_geometric_alignment_rejection_holds_without_faulting_command_provider(
    monkeypatch: pytest.MonkeyPatch,
    blocked_reason: str,
) -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, clutch, ui = _bind(extension, recorder)
    planner = extension._initial_alignment_planner
    assert planner is not None

    def reject_transition(**_kwargs):
        raise RuleBasedInitialAlignmentPlanningError(
            f"status=hold_current reason={blocked_reason}",
            status="hold_current",
            blocked_reason=blocked_reason,
        )

    monkeypatch.setattr(planner, "plan", reject_transition)

    def confirmed_hold(reason: str) -> bool:
        session.hold_reasons.append(reason)
        session.state_machine.state = SimpleNamespace(value="DEBUG_ONLY")
        return True

    monkeypatch.setattr(session, "hold_hardware", confirmed_hold)
    assert session.command_override_provider(SimpleNamespace(), 4.0).command is None
    future = extension._initial_alignment_future
    assert future is not None
    for _ in range(200):
        if future.done():
            break
        time.sleep(0.001)
    assert future.done()

    assert session.command_override_provider(SimpleNamespace(), 4.1).command is None
    assert extension.status().aborted is True
    assert extension.status().held is True
    assert blocked_reason in session.hold_reasons[0]
    assert len(session.hold_reasons) == 1
    assert clutch.reasons == ["rule_based_runtime_hold"]
    assert ui.failure_stop_requests == 1
    assert session.command_override_provider(SimpleNamespace(), 4.2).command is None

    # The normal Hold owner then retires the failed Episode before Home.  A
    # planning rejection must not leave an aborted cursor blocking that path.
    recorder.is_recording = False
    extension.after_runtime_hold(
        reason="rule_based_runtime_hold", hardware_hold_confirmed=True
    )
    assert extension.status().aborted is False
    extension.after_runtime_home(now=4.3)
    assert extension.status().held is False


@pytest.mark.parametrize(
    "blocked_reason", ("joint_limit_hold_margin", "collision_soft")
)
def test_geometric_alignment_rejection_faults_if_hold_is_unconfirmed(
    monkeypatch: pytest.MonkeyPatch,
    blocked_reason: str,
) -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, ui = _bind(extension, recorder)
    planner = extension._initial_alignment_planner
    assert planner is not None

    def reject_transition(**_kwargs):
        raise RuleBasedInitialAlignmentPlanningError(
            f"status=hold_current reason={blocked_reason}",
            status="hold_current",
            blocked_reason=blocked_reason,
        )

    monkeypatch.setattr(planner, "plan", reject_transition)
    monkeypatch.setattr(session, "hold_hardware", lambda _reason: False)
    assert session.command_override_provider(SimpleNamespace(), 4.0).command is None
    future = extension._initial_alignment_future
    assert future is not None
    for _ in range(200):
        if future.done():
            break
        time.sleep(0.001)
    assert future.done()

    with pytest.raises(RuleBasedInitialAlignmentPlanningError):
        session.command_override_provider(SimpleNamespace(), 4.1)
    assert extension.status().aborted is True
    assert ui.failure_stop_requests == 1


@pytest.mark.parametrize("session_state", ("DEBUG_ONLY", "FAULT"))
def test_unarmed_session_does_not_advertise_reachability_work_in_progress(
    session_state: str,
) -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        source_selection={
            "collection_mode": "rule_based_writing",
            "writing_reachability": {"enabled": True},
        },
    )
    recorder = _Recorder(_Poller())
    session, _clutch, _ui = _bind(extension, recorder, arm=False)
    extension._height_calibration = RuleBasedWritingHeightCalibration(
        calibrated_z_m=0.3,
        lower_tolerance_m=0.001,
        upper_tolerance_m=0.001,
        captured_at=1.0,
        source_timestamp=1.0,
        pose_frame_contract_hash=extension.runtime_pose_frame_contract.contract_hash,
    )
    session.state_machine.state = SimpleNamespace(value=session_state)

    status = extension.status()

    assert "rig.session" in status.readiness_blockers
    assert "rule.writing_reachability" not in status.readiness_blockers
    assert status.rule_start_available is False


def test_consecutive_initializations_plan_from_their_own_targets() -> None:
    first_group = _group()
    second_qpos = np.full((2, 6), 0.08, dtype=np.float32)
    second_group = replace(
        _group(),
        key="second",
        label="second",
        arm_joint_targets={
            "left": second_qpos.copy(),
            "right": second_qpos.copy(),
        },
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedInitialize(step_id="first_init", group_key="source"),
                RuleBasedInitialize(step_id="second_init", group_key="second"),
            )
        ),
        trajectory_groups={"source": first_group, "second": second_group},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=1),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    now = 6.0
    first_plan_hash: str | None = None
    for _ in range(500):
        current = _wait_for_command(session, now=now)
        metadata = current.command.metadata
        step_id = str(metadata["rule_based_step_id"])
        if step_id == "first_init":
            first_plan_hash = str(metadata["rule_based_initial_alignment_plan_hash"])
        elif step_id == "second_init":
            assert first_plan_hash is not None
            assert metadata["rule_based_initial_alignment_plan_hash"] != first_plan_hash
            assert extension._initial_alignment_plan is not None
            for side in ("left", "right"):
                np.testing.assert_allclose(
                    extension._initial_alignment_plan.target_qpos_by_side[side],
                    np.full(6, 0.08),
                )
            break
        else:
            raise AssertionError(f"unexpected rule step {step_id!r}")
        for side in ("left", "right"):
            recorder.arm_state_poller.qpos[side] = np.asarray(
                current.command.commands[f"{side}_arm"].value,
                dtype=float,
            ).copy()
        extension.after_session_step(
            step_result=SimpleNamespace(
                command=current.command,
                command_dispatched=True,
                state=SimpleNamespace(value="RUNNING"),
                faults=(),
            ),
            now=now,
        )
        now += 0.051
    else:
        raise AssertionError("second initial alignment did not start")


def test_target_arrival_timeout_is_contained_in_hold_and_episode_reclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, clutch, ui = _bind(extension, recorder)
    current = _wait_for_command(session, now=5.0)
    now = 5.0
    while not extension._is_final_initial_alignment_waypoint(extension._current):
        for side in ("left", "right"):
            recorder.arm_state_poller.qpos[side] = np.asarray(
                current.command.commands[f"{side}_arm"].value,
                dtype=float,
            ).copy()
        extension.after_session_step(
            step_result=SimpleNamespace(
                command=current.command,
                command_dispatched=True,
                state=SimpleNamespace(value="RUNNING"),
                faults=(),
            ),
            now=now,
        )
        now += 1.0 / 60.0 + 1e-4
        current = _wait_for_command(session, now=now)

    def reject_arrival(*_args, **_kwargs):
        raise TimeoutError("synthetic causal target timeout")

    monkeypatch.setattr(extension, "_observe_target_arrival", reject_arrival)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=current.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=now,
    )

    assert extension.status().aborted is True
    assert extension.status().held is True
    assert clutch.reasons == ["rule_based_runtime_hold"]
    assert ui.failure_stop_requests == 1


def test_ordinary_active_recording_without_arm_dispatch_still_pauses() -> None:
    assert control_loop._recording_requires_arm_dispatch_pause(
        arm_dispatch_confirmed=False,
        recording_active=True,
        recording_paused=False,
        runtime_extension=None,
    )


def test_deployment_preflight_requires_existing_safety_and_collection() -> None:
    contract = _deployment().collection_pose_frame_contract
    args = SimpleNamespace(
        active_sides=("left", "right"),
        arm_control_mode="position",
        disable_dual_arm_collision_avoidance=False,
        tcp_pose_envelope_active=True,
        tcp_pose_envelope_profile="envelope.json",
        position_force_guard=True,
        position_force_guard_source="tool",
        no_collection=False,
        position_admittance_active=False,
        position_admittance_rotation_active=False,
        position_admittance_combined_active=False,
        pose_frame_contract=contract,
    )

    deployment = preflight_rule_based_real_teleop_deployment(args)
    assert deployment.collection_pose_frame_contract is contract

    args.position_force_guard = False
    with pytest.raises(ValueError, match="force guard"):
        preflight_rule_based_real_teleop_deployment(args)


def test_real_plan_rejects_partial_stream_selection() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="partial",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                capture=RuleBasedCaptureSelection(stream_ids=("action.left",)),
            ),
        )
    )

    with pytest.raises(ValueError, match="formal samples are atomic"):
        RuleBasedRealTeleopExtension(
            deployment=_deployment(),
            plan=plan,
            trajectory_groups={},
        )


def test_runtime_waits_for_formal_episode_and_holds_after_completion() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedInitialize(step_id="init", group_key="source"),
            RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(
                    left_arm=np.full(6, 0.2),
                    right_arm=np.full(6, 0.2),
                    source="rule",
                ),
                capture=RuleBasedCaptureSelection(enabled=True),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(
            action_hz=10.0,
            settle_consecutive_samples=2,
        ),
        source_selection={"schema": "fixture", "phases": {"write": "source-b"}},
    )
    poller = _Poller()
    poller.qpos = {
        "left": np.full(6, -0.02),
        "right": np.full(6, -0.02),
    }
    recorder = _Recorder(poller)
    session, clutch, ui = _bind(extension, recorder, arm=False)

    waiting = session.command_override_provider(SimpleNamespace(), 10.0)
    assert isinstance(waiting, TeleopCommandOverride)
    assert waiting.command is None
    assert extension.status().execution_begun is False

    recorder.is_recording = True
    ordinary_episode_only = session.command_override_provider(SimpleNamespace(), 10.05)
    assert ordinary_episode_only.command is None
    assert extension.status().armed is False

    ui.request_start()
    extension.after_control_input_poll(now=10.06)
    init = _wait_for_command(session, now=10.1)
    assert init.command.commands["left_arm"].command_type == "joint_vector"
    assert init.command.metadata["rule_based_initial_alignment_plan_hash"].startswith(
        "sha256:"
    )
    current = init
    tick = 10.1
    while current.command.metadata["rule_based_directive_kind"] == "initialize":
        for side in ("left", "right"):
            poller.qpos[side] = np.asarray(
                current.command.commands[f"{side}_arm"].value,
                dtype=float,
            ).copy()
        for _ in range(2):
            extension.after_session_step(
                step_result=SimpleNamespace(command_dispatched=True),
                now=tick,
            )
            tick += 0.01
        current = _wait_for_command(session, now=tick)

    collect = current
    assert collect.command.metadata["rule_based_step_id"] == "collect"
    assert collect.command.metadata["rule_based_source_selection"]["schema"] == "fixture"
    collect_now = tick
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=collect_now,
    )
    admission = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True),
        now=collect_now,
    )
    assert admission.allowed is True
    assert admission.metadata["rule_based_source_selection"]["phases"] == {
        "write": "source-b"
    }
    duplicate = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True),
        now=collect_now + 0.005,
    )
    assert duplicate.allowed is True
    extension.after_dataset_sample_attempt(
        admission=admission,
        accepted=True,
        now=collect_now + 0.01,
    )
    committed_duplicate = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True),
        now=collect_now + 0.015,
    )
    assert committed_duplicate.allowed is False
    completed = session.command_override_provider(SimpleNamespace(), collect_now + 0.11)
    assert completed.command is None
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=False),
        now=collect_now + 0.11,
    )

    assert extension.status().completed is True
    assert session.hold_reasons == []
    assert clutch.reasons == []
    assert ui.completion_stop_requests == 1
    recorder.is_recording = False
    extension.after_control_input_poll(now=collect_now + 0.2)
    status = extension.status()
    assert status.completed is False
    assert status.execution_begun is False
    assert status.rule_start_available is True


def test_dual_gripper_initial_alignment_is_interpolated_and_gates_first_replay() -> None:
    group = _dual_hand_group()
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedInitialize(
                    step_id="initial_alignment",
                    group_key="source",
                    include_end_effectors=True,
                    end_effector_sides=("left", "right"),
                    capture=RuleBasedCaptureSelection(enabled=False),
                ),
                RuleBasedReplay(
                    step_id="right_write.pen_down",
                    group_key="source",
                    start_index=0,
                    end_index=1,
                    include_end_effectors=True,
                    end_effector_sides=("left", "right"),
                    capture=RuleBasedCaptureSelection(enabled=True),
                    wait_for_terminal_arrival=True,
                ),
            )
        ),
        trajectory_groups={"source": group},
        config=RuleBasedRealTeleopConfig(
            action_hz=10.0,
            settle_consecutive_samples=1,
            gripper_sides=("left", "right"),
            gripper_max_step=0.05,
        ),
    )
    arm_poller = _Poller()
    hand_poller = _HandPoller()
    recorder = _Recorder(arm_poller)
    recorder.hand_state_poller = hand_poller
    recorder.is_recording = True
    session, _clutch, _ui = _bind(
        extension,
        recorder,
        hand_sides=("left", "right"),
    )

    current = _wait_for_command(session, now=40.0)
    previous = {"left": 0.0, "right": 0.0}
    initialization_commands = 0
    now = 40.0
    while current.command.metadata["rule_based_directive_kind"] == "initialize":
        initialization_commands += 1
        assert session.command_guard(current.command, SimpleNamespace(), now) is None
        if initialization_commands == 1:
            hand = current.command.commands["right_hand"]
            original_full = np.asarray(
                hand.metadata["hand_qpos_full"],
                dtype=np.float32,
            ).copy()
            mismatched_full = original_full.copy()
            mismatched_full[0] += 0.01
            hand.metadata["hand_qpos_full"] = mismatched_full
            fault = session.command_guard(current.command, SimpleNamespace(), now)
            assert fault is not None
            assert fault.code == "rule_based_collection_action_contract_invalid"
            hand.metadata["hand_qpos_full"] = original_full
        for side in ("left", "right"):
            command = current.command.commands[f"{side}_hand"]
            assert command.value.shape == (1,)
            if initialization_commands == 1:
                assert command.value[0] == pytest.approx(previous[side])
            else:
                assert 0.0 < command.value[0] - previous[side] <= 0.050001
            previous[side] = float(command.value[0])
            full = np.asarray(command.metadata["hand_qpos_full"], dtype=np.float32)
            assert np.allclose(full[1:], 0.0)
            hand_poller.qpos[side] = full
            arm_poller.qpos[side] = np.asarray(
                current.command.commands[f"{side}_arm"].value, dtype=float
            )
        extension.after_session_step(
            step_result=SimpleNamespace(command_dispatched=True),
            now=now,
        )
        now += 0.051
        current = _wait_for_command(session, now=now)

    assert initialization_commands >= 2
    assert previous == pytest.approx({"left": 0.10, "right": 0.10})
    replay = current
    assert replay.command.metadata["rule_based_step_id"] == "right_write.pen_down"
    assert set(replay.command.commands) == {
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    }
    assert replay.command.metadata["rule_based_capture_enabled"] is True
    assert session.command_guard(replay.command, SimpleNamespace(), now) is None
    canonical_action = validate_collection_command_action(replay.command)
    for side in ("left", "right"):
        assert canonical_action.hand_qpos_by_side[side].shape == (6,)
        np.testing.assert_array_equal(
            canonical_action.hand_qpos_by_side[side][1:],
            np.zeros(5, dtype=np.float32),
        )


def test_initial_alignment_rejects_gripper_drift_during_background_planning() -> None:
    group = _dual_hand_group()
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedInitialize(
                    step_id="initial_alignment",
                    group_key="source",
                    include_end_effectors=True,
                    end_effector_sides=("left", "right"),
                ),
            )
        ),
        trajectory_groups={"source": group},
        config=RuleBasedRealTeleopConfig(
            gripper_sides=("left", "right"),
            initial_alignment_hand_start_drift_tolerance=0.01,
        ),
    )
    recorder = _Recorder(_Poller())
    recorder.hand_state_poller = _HandPoller()
    recorder.is_recording = True
    session, _clutch, _ui = _bind(
        extension, recorder, hand_sides=("left", "right")
    )
    assert session.command_override_provider(SimpleNamespace(), 50.0).command is None
    for _ in range(200):
        if extension._initial_alignment_future.done():
            break
        time.sleep(0.001)
    recorder.hand_state_poller.qpos["right"][0] = 0.05

    with pytest.raises(
        RuntimeError,
        match="actual gripper state drifted while the transition was being planned",
    ):
        session.command_override_provider(SimpleNamespace(), 50.1)
    assert extension.status().aborted is True
    assert extension.status().held is True


def test_manual_dual_gripper_buttons_use_typed_non_recorded_dispatch_and_lock_after_start() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("left", "right")),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("left", "right"),
    )
    backends = {
        side: getattr(extension._components, f"{side}_hand_backend")
        for side in ("left", "right")
    }
    for backend in backends.values():
        backend.set_measured_scalar(0.5)

    ui.intents.append("release_grippers")
    extension.after_control_input_poll(now=50.0)
    previous = {"left": 0.5, "right": 0.5}
    now = 50.0
    while extension._pending_gripper_action is not None:
        command_count = len(session.external_commands)
        override = session.command_override_provider(SimpleNamespace(), now)
        assert override.command is None
        if len(session.external_commands) > command_count:
            release = session.external_commands[-1]
            assert set(release.commands) == {"left_hand", "right_hand"}
            assert release.metadata["dataset_recording_eligible"] is False
            assert release.metadata["external_gripper_motion"] is True
            for side in ("left", "right"):
                command = release.commands[f"{side}_hand"]
                assert command.value.shape == (1,)
                assert command.metadata["dmtac_continuous_grasp_override"] is False
                assert abs(float(command.value[0]) - previous[side]) <= 0.05 + 1e-7
                assert command.metadata["hand_action_spec"]["hand_model"] == "dmtac_gripper"
                previous[side] = float(command.value[0])
                backends[side].set_measured_scalar(previous[side])
        now += 0.05
        assert now < 55.0
    assert previous == pytest.approx({"left": 0.0, "right": 0.0})
    assert extension.status().gripper_action_status == "夹爪实测已松开"

    ui.intents.append("close_grippers")
    extension.after_control_input_poll(now=now)
    while extension._pending_gripper_action is not None:
        command_count = len(session.external_commands)
        session.command_override_provider(SimpleNamespace(), now)
        if len(session.external_commands) > command_count:
            close = session.external_commands[-1]
            for side in ("left", "right"):
                command = close.commands[f"{side}_hand"]
                assert command.metadata["dmtac_continuous_grasp_override"] is True
                assert abs(float(command.value[0]) - previous[side]) <= 0.05 + 1e-7
                previous[side] = float(command.value[0])
                backends[side].set_measured_scalar(previous[side])
        now += 0.05
        assert now < 60.0
    assert previous == pytest.approx({"left": 1.0, "right": 1.0})
    assert extension.status().gripper_action_status == "夹爪实测已闭紧"
    assert extension.status().control_topology == "双臂 + 左夹爪 + 右夹爪"
    assert extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True),
        now=now,
    ).allowed is False

    recorder.is_recording = True
    ui.request_start()
    extension.after_control_input_poll(now=now)
    ui.intents.append("release_grippers")
    extension.after_control_input_poll(now=now + 0.01)
    command_count = len(session.external_commands)
    session.command_override_provider(SimpleNamespace(), now + 0.01)
    assert len(session.external_commands) == command_count
    assert "没有活动 Episode" in extension.status().gripper_action_status


def test_task_chain_close_right_gripper_only_dispatches_right_hand() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("left", "right")),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("left", "right"),
    )
    backends = {
        side: getattr(extension._components, f"{side}_hand_backend")
        for side in ("left", "right")
    }
    for backend in backends.values():
        backend.set_measured_scalar(0.5)

    ui.intents.append("close_right_gripper")
    extension.after_control_input_poll(now=10.0)
    now = 10.0
    while extension._pending_gripper_action is not None:
        command_count = len(session.external_commands)
        session.command_override_provider(SimpleNamespace(), now)
        if len(session.external_commands) > command_count:
            close = session.external_commands[-1]
            assert set(close.commands) == {"right_hand"}
            assert close.metadata["rule_based_manual_gripper_sides"] == ["right"]
            command = close.commands["right_hand"]
            # Admission closes in one-shot mode so a real idle/force-stop
            # receipt is required. Writing retains this same fixed mode.
            assert command.metadata["dmtac_continuous_grasp_override"] is False
            assert command.metadata["dmtac_preserve_existing_grip"] is False
            assert command.metadata["rule_based_manual_gripper_dispatch_mode"] == (
                "single_endpoint_feedback_gated"
            )
            assert command.metadata["rule_based_manual_gripper_motion_mode"] == (
                "one_shot_close"
            )
            assert float(command.value[0]) == pytest.approx(1.0)
            backends["right"].set_measured_scalar(float(command.value[0]))
            assert float(backends["left"].measured_joint_pos[0]) == pytest.approx(0.5)
        now += 0.05
        assert now < 20.0

    assert float(backends["left"].measured_joint_pos[0]) == pytest.approx(0.5)
    assert float(backends["right"].measured_joint_pos[0]) == pytest.approx(1.0)
    assert len(session.external_commands) == 1
    assert extension.status().gripper_action_status == "右夹爪实测已闭紧"
    assert extension.status().gripper_action_intent == "close_right_gripper"
    assert extension.status().gripper_action_result == "succeeded"
    assert extension.status().gripper_action_id == 1


def test_task_chain_release_right_gripper_preserves_left_hand() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("left", "right")),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("left", "right"),
    )
    left = extension._components.left_hand_backend
    right = extension._components.right_hand_backend
    left.set_measured_scalar(0.5)
    right.set_measured_scalar(0.5)

    ui.intents.append("release_right_gripper")
    extension.after_control_input_poll(now=10.0)
    now = 10.0
    while extension._pending_gripper_action is not None:
        command_count = len(session.external_commands)
        session.command_override_provider(SimpleNamespace(), now)
        if len(session.external_commands) > command_count:
            release = session.external_commands[-1]
            assert set(release.commands) == {"right_hand"}
            command = release.commands["right_hand"]
            assert command.metadata["dmtac_continuous_grasp_override"] is False
            assert command.metadata["dmtac_preserve_existing_grip"] is False
            assert command.metadata["rule_based_manual_gripper_dispatch_mode"] == (
                "single_endpoint_feedback_gated"
            )
            assert command.metadata["rule_based_manual_gripper_motion_mode"] == (
                "one_shot_open"
            )
            assert release.metadata["rule_based_manual_gripper_motion_mode"] == (
                "one_shot_open"
            )
            assert float(command.value[0]) == pytest.approx(0.0)
            right.set_measured_scalar(float(command.value[0]))
        now += 0.05
        assert now < 20.0

    assert float(left.measured_joint_pos[0]) == pytest.approx(0.5)
    assert float(right.measured_joint_pos[0]) == pytest.approx(0.0)
    assert len(session.external_commands) == 1
    status = extension.status()
    assert status.gripper_action_status == "右夹爪实测已松开"
    assert status.gripper_action_intent == "release_right_gripper"
    assert status.gripper_action_result == "succeeded"


def test_task_chain_close_right_waits_for_idle_after_closed_endpoint() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())

    class _AlreadyClosedCatchPoller:
        def __init__(self) -> None:
            self.status_code = 4
            self.is_moving = True
            self.timestamp = 9.0

        def latest_at_or_before(self, _now):
            measured = np.zeros(6, dtype=np.float32)
            measured[0] = 1.0
            return {
                "right": {
                    "measured_joint_pos": measured,
                    "joint_age_sec": 0.0,
                    "joint_timestamp": self.timestamp,
                    "gripper_status_code": self.status_code,
                    "gripper_is_moving": self.is_moving,
                }
            }

    hand_poller = _AlreadyClosedCatchPoller()
    recorder.hand_state_poller = hand_poller
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )

    ui.intents.append("close_right_gripper")
    extension.after_control_input_poll(now=10.0)
    session.command_override_provider(SimpleNamespace(), 10.0)
    assert extension._pending_gripper_action == "close_right_gripper"
    assert len(session.external_commands) == 1
    command = session.external_commands[-1]
    assert set(command.commands) == {"right_hand"}
    assert float(command.commands["right_hand"].value[0]) == pytest.approx(1.0)
    assert command.commands["right_hand"].metadata[
        "dmtac_continuous_grasp_override"
    ] is False

    # Position=1 is not a completion receipt while firmware still reports
    # closing/catching. The one-shot transition must reach a stable idle
    # status before the task chain can enter height calibration.
    hand_poller.timestamp = 10.1
    for now in (10.1, 10.2):
        session.command_override_provider(SimpleNamespace(), now)
        assert extension._pending_gripper_action == "close_right_gripper"
        hand_poller.timestamp += 0.1
    assert "闭合运动" in extension.status().gripper_action_status

    hand_poller.status_code = 2
    hand_poller.is_moving = False
    for now in (10.3, 10.4):
        session.command_override_provider(SimpleNamespace(), now)
        assert extension._pending_gripper_action == "close_right_gripper"
        hand_poller.timestamp += 0.1
    session.command_override_provider(SimpleNamespace(), 10.5)
    assert extension._pending_gripper_action is None
    assert extension.status().gripper_action_status == "右夹爪实测已闭紧"


def test_writing_keeps_right_gripper_in_fixed_one_shot_mode() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
        source_selection={"collection_mode": "rule_based_writing"},
    )
    action = SimpleNamespace(
        timestamp=1.0,
        metadata={"collection_detail_stage_id": "right_write.pen_up"},
    )
    command = TeleopCommand(
        commands={
            "right_hand": EffectorCommand(
                effector_name="right_hand",
                side="right",
                command_type="joint_vector",
                value=np.full(6, 0.8),
                metadata={},
            )
        },
        timestamp=1.0,
        metadata={},
    )
    emission = SimpleNamespace(
        action=action,
        command=command,
        directive=SimpleNamespace(segment_terminal=True),
    )

    extension._prepare_live_dispatch(emission, 2.0)

    hand_command = command.commands["right_hand"]
    assert np.asarray(hand_command.value, dtype=float) == pytest.approx(
        np.full(6, 0.8)
    )
    assert hand_command.metadata["dmtac_continuous_grasp_override"] is False
    assert hand_command.metadata["dmtac_preserve_existing_grip"] is True
    assert command.metadata[
        "rule_based_right_gripper_hold_mode"
    ] == "task_chain_fixed_closed_one_shot"


def test_nonterminal_writing_frame_uses_same_fixed_one_shot_grip_mode() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
        source_selection={"collection_mode": "rule_based_writing"},
    )
    command = TeleopCommand(
        commands={
            "right_hand": EffectorCommand(
                effector_name="right_hand",
                side="right",
                command_type="joint_vector",
                value=np.full(6, 0.8),
                metadata={},
            )
        },
        timestamp=1.0,
        metadata={"collection_detail_stage_id": "right_write.pen_up"},
    )
    emission = SimpleNamespace(
        action=SimpleNamespace(timestamp=1.0, metadata={}),
        command=command,
        directive=SimpleNamespace(segment_terminal=False),
    )

    extension._prepare_live_dispatch(emission, 2.0)

    assert command.commands["right_hand"].metadata[
        "dmtac_continuous_grasp_override"
    ] is False
    assert command.commands["right_hand"].metadata[
        "dmtac_preserve_existing_grip"
    ] is True
    assert command.metadata["rule_based_right_gripper_hold_mode"] == (
        "task_chain_fixed_closed_one_shot"
    )


def test_task_chain_close_right_accepts_stable_force_stopped_grasp() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
        source_selection={"collection_mode": "rule_based_writing"},
    )
    recorder = _Recorder(_Poller())

    class _ForceStoppedPoller:
        def __init__(self) -> None:
            self.value = 0.5
            self.timestamp = 9.0
            self.status_code = 2

        def latest_at_or_before(self, _now):
            measured = np.zeros(6, dtype=np.float32)
            measured[0] = self.value
            return {
                "right": {
                    "measured_joint_pos": measured,
                    "joint_age_sec": 0.0,
                    "joint_timestamp": self.timestamp,
                    "gripper_status_code": self.status_code,
                    "gripper_is_moving": False,
                }
            }

    hand_poller = _ForceStoppedPoller()
    recorder.hand_state_poller = hand_poller
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )

    ui.intents.append("close_right_gripper")
    extension.after_control_input_poll(now=10.0)
    session.command_override_provider(SimpleNamespace(), 10.0)
    assert extension._pending_gripper_action == "close_right_gripper"
    assert len(session.external_commands) == 1

    # A closed-side idle sample without measured progress is ambiguous: it can
    # be a stall and must not be promoted to a successful grasp.
    hand_poller.status_code = 2
    hand_poller.timestamp = 10.1
    session.command_override_provider(SimpleNamespace(), 10.1)
    assert extension._pending_gripper_action == "close_right_gripper"
    assert len(session.external_commands) == 1

    # A pen/object can stop the hand below the numeric endpoint.  Once the
    # causal close has produced forward measured progress, the same stable
    # endpoint gate accepts status=2 as a valid closed-side idle receipt.
    hand_poller.value = 0.53
    for timestamp, now in ((10.2, 10.2), (10.3, 10.3), (10.4, 10.4)):
        hand_poller.timestamp = timestamp
        session.command_override_provider(SimpleNamespace(), now)
        if now < 10.4:
            assert extension._pending_gripper_action == "close_right_gripper"

    status = extension.status()
    assert extension._pending_gripper_action is None
    assert len(session.external_commands) == 1
    assert status.gripper_action_status == "右夹爪实测已闭紧"
    assert status.gripper_action_result == "succeeded"

    nominal = np.zeros(6, dtype=np.float32)
    nominal[0] = 1.0
    assert extension._writing_right_gripper_force_stop_reached(
        side="right",
        target=nominal,
        snapshots={
            "right": {
                "hand_state": {
                    "gripper_status_code": 6,
                    "gripper_error_code": 0,
                    "gripper_is_moving": False,
                },
            }
        },
    )
    partial_idle_state = {
        "gripper_status_code": 2,
        "gripper_error_code": 0,
        "gripper_is_moving": False,
        "gripper_commanded_scalar": 1.0,
    }
    assert extension._writing_right_gripper_force_stop_reached(
        side="right",
        target=nominal,
        snapshots={
            "right": {
                "actual_hand_qpos": np.asarray(
                    [0.53, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                "hand_state": partial_idle_state,
            }
        },
    )
    assert extension._writing_right_gripper_hold_admitted(
        target=nominal,
        snapshots={
            "right": {
                "actual_hand_qpos": np.asarray(
                    [0.53, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                "hand_state": partial_idle_state,
            }
        },
    )
    assert extension._writing_right_gripper_hold_admitted(
        target=nominal,
        snapshots={
            "right": {
                "actual_hand_qpos": np.asarray(
                    [0.53, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                "hand_state": {
                    "gripper_status_code": 6,
                    "gripper_error_code": 0,
                    "gripper_is_moving": False,
                },
            }
        },
    )


@pytest.mark.parametrize("closed_status", (2, 6))
def test_writing_gui_standalone_close_accepts_stable_contact_feedback(
    closed_status: int,
) -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
        source_selection={"collection_mode": "rule_based_writing"},
    )
    recorder = _Recorder(_Poller())

    class ContactPoller:
        value = 0.5
        timestamp = 9.0
        status_code = 2

        def latest_at_or_before(self, _now):
            measured = np.zeros(6, dtype=np.float32)
            measured[0] = self.value
            return {
                "right": {
                    "measured_joint_pos": measured,
                    "joint_age_sec": 0.0,
                    "joint_timestamp": self.timestamp,
                    "gripper_status_code": self.status_code,
                    "gripper_is_moving": self.status_code == 4,
                    "gripper_error_code": 0,
                }
            }

    poller = ContactPoller()
    recorder.hand_state_poller = poller
    session, _clutch, ui = _bind(
        extension, recorder, arm=False, hand_sides=("right",)
    )

    # The standalone GUI emits the generic intent, not the task-chain intent.
    ui.intents.append("close_grippers")
    extension.after_control_input_poll(now=10.0)
    session.command_override_provider(SimpleNamespace(), 10.0)
    assert len(session.external_commands) == 1
    command = session.external_commands[0].commands["right_hand"]
    assert float(command.value[0]) == pytest.approx(1.0)
    assert command.metadata["dmtac_continuous_grasp_override"] is False
    assert command.metadata["rule_based_manual_gripper_dispatch_mode"] == (
        "single_endpoint_feedback_gated"
    )

    poller.timestamp = 10.0
    poller.status_code = closed_status
    poller.value = 0.53 if closed_status == 6 else 0.5
    session.command_override_provider(SimpleNamespace(), 10.1)
    if closed_status == 6:
        assert "新鲜反馈" in extension.status().gripper_action_status
    else:
        assert extension._pending_gripper_action == "close_grippers"

    if closed_status == 2:
        # A pre-existing closed-side idle code without any motion from this
        # command must not be mistaken for a successful grasp.
        poller.timestamp = 10.05
        session.command_override_provider(SimpleNamespace(), 10.15)
        assert extension._pending_gripper_action == "close_grippers"
        poller.value = 0.53

    poller.status_code = 4
    for timestamp, now in ((10.1, 10.2), (10.2, 10.3)):
        poller.timestamp = timestamp
        session.command_override_provider(SimpleNamespace(), now)
        assert extension._pending_gripper_action == "close_grippers"

    poller.status_code = closed_status
    for timestamp, now in ((10.3, 10.4), (10.4, 10.5), (10.5, 10.6)):
        poller.timestamp = timestamp
        session.command_override_provider(SimpleNamespace(), now)
        if now < 10.6:
            assert extension._pending_gripper_action == "close_grippers"
    assert len(session.external_commands) == 1
    assert extension.status().gripper_action_result == "succeeded"
    assert extension.status().gripper_action_status == "夹爪实测已闭紧"


def test_writing_gui_standalone_release_uses_one_shot_open_feedback() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
        source_selection={"collection_mode": "rule_based_writing"},
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(
        extension, recorder, arm=False, hand_sides=("right",)
    )
    backend = extension._components.right_hand_backend
    backend.set_measured_scalar(0.5)

    ui.intents.append("release_grippers")
    extension.after_control_input_poll(now=10.0)
    session.command_override_provider(SimpleNamespace(), 10.0)
    assert len(session.external_commands) == 1
    command = session.external_commands[0].commands["right_hand"]
    assert float(command.value[0]) == pytest.approx(0.0)
    assert command.metadata["dmtac_continuous_grasp_override"] is False
    assert command.metadata["rule_based_manual_gripper_motion_mode"] == (
        "one_shot_open"
    )

    backend.set_measured_scalar(0.0)
    for now in (10.1, 10.2, 10.3):
        session.command_override_provider(SimpleNamespace(), now)
    assert len(session.external_commands) == 1
    assert extension.status().gripper_action_result == "succeeded"
    assert extension.status().gripper_action_status == "夹爪实测已松开"


def test_manual_gripper_reset_uses_recorder_hand_history_without_backend_read() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())

    class CachedHandPoller:
        def __init__(self) -> None:
            self.calls = 0

        def latest_at_or_before(self, now, *, include_tactile_arrays=True):
            del now, include_tactile_arrays
            self.calls += 1
            measured = np.zeros(6, dtype=np.float32)
            measured[0] = 0.5
            return {
                "right": {
                    "measured_joint_pos": measured,
                    "joint_age_sec": 0.0,
                }
            }

    poller = CachedHandPoller()
    recorder.hand_state_poller = poller
    session, _clutch, _ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )
    backend = extension._components.right_hand_backend

    def fail_if_read():
        raise AssertionError("manual reset must not synchronously read backend state")

    backend.get_state = fail_if_read
    state, error = extension._manual_gripper_state(
        backend,
        side="right",
        now=1.0,
    )
    assert error == ""
    assert state is not None
    assert float(np.asarray(state["measured_joint_pos"])[0]) == pytest.approx(0.5)
    assert poller.calls == 1

def test_manual_gripper_requires_fresh_measured_feedback_and_times_out() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(
            gripper_sides=("right",),
            manual_gripper_timeout_sec=0.1,
        ),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )
    backend = extension._components.right_hand_backend
    backend.set_measured_scalar(0.5)
    backend.joint_age_sec = 1.0

    ui.intents.append("release_grippers")
    extension.after_control_input_poll(now=10.0)
    session.command_override_provider(SimpleNamespace(), 10.0)
    assert not session.external_commands
    assert "反馈陈旧" in extension.status().gripper_action_status

    session.command_override_provider(SimpleNamespace(), 10.11)
    assert extension._pending_gripper_action is None
    assert "控制超时" in extension.status().gripper_action_status


def test_manual_gripper_rebases_on_measured_partial_progress_until_open() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(
            gripper_sides=("right",),
            gripper_max_step=0.05,
            manual_gripper_tolerance=0.02,
        ),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )
    backend = extension._components.right_hand_backend
    backend.set_measured_scalar(0.5)

    ui.intents.append("release_grippers")
    extension.after_control_input_poll(now=10.0)
    now = 10.0
    measured = 0.5
    sent_segments: list[tuple[float, float]] = []
    while extension._pending_gripper_action is not None:
        before = len(session.external_commands)
        session.command_override_provider(SimpleNamespace(), now)
        if len(session.external_commands) > before:
            target = float(session.external_commands[-1].commands["right_hand"].value[0])
            sent_segments.append((measured, target))
            # Reproduce the real failure: the actuator advances only halfway
            # to each intermediate command, outside the old +/-0.02 gate.
            measured += 0.5 * (target - measured)
            backend.set_measured_scalar(measured)
        now += 0.05
        assert now < 15.0

    assert len(sent_segments) > 1
    assert all(
        abs(target - measured_before) <= 0.050001
        for measured_before, target in sent_segments
    )
    assert measured <= 0.02
    assert extension.status().gripper_action_status == "夹爪实测已松开"


def test_manual_gripper_resends_same_segment_until_measured_progress() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(
            gripper_sides=("right",),
            gripper_max_step=0.05,
            manual_gripper_resend_interval_sec=0.25,
        ),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )
    backend = extension._components.right_hand_backend
    backend.set_measured_scalar(1.0)

    ui.intents.append("release_grippers")
    extension.after_control_input_poll(now=10.0)
    session.command_override_provider(SimpleNamespace(), 10.0)
    assert len(session.external_commands) == 1
    first_target = float(
        session.external_commands[-1].commands["right_hand"].value[0]
    )
    assert first_target == pytest.approx(0.95)

    session.command_override_provider(SimpleNamespace(), 10.24)
    assert len(session.external_commands) == 1
    assert "right=1.000/0.950" in extension.status().gripper_action_status

    session.command_override_provider(SimpleNamespace(), 10.251)
    assert len(session.external_commands) == 2
    resent_target = float(
        session.external_commands[-1].commands["right_hand"].value[0]
    )
    assert resent_target == pytest.approx(first_target)
    assert "重发同一安全分段" in extension.status().gripper_action_status
    assert extension._manual_gripper_last_progress_at_by_side["right"] == pytest.approx(
        10.0
    )

    measured = resent_target
    backend.set_measured_scalar(measured)
    now = 10.30
    bounded_segments: list[tuple[float, float]] = []
    while extension._pending_gripper_action is not None:
        before = len(session.external_commands)
        session.command_override_provider(SimpleNamespace(), now)
        if len(session.external_commands) > before:
            target = float(
                session.external_commands[-1].commands["right_hand"].value[0]
            )
            bounded_segments.append((measured, target))
            measured = target
            backend.set_measured_scalar(measured)
        now += 0.05
        assert now < 15.0

    assert bounded_segments
    assert all(
        abs(target - measured_before) <= 0.050001
        for measured_before, target in bounded_segments
    )
    assert extension.status().gripper_action_status == "夹爪实测已松开"


def test_manual_gripper_resends_do_not_extend_measured_progress_timeout() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(
            gripper_sides=("right",),
            manual_gripper_resend_interval_sec=0.1,
            manual_gripper_timeout_sec=0.5,
        ),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )
    extension._components.right_hand_backend.set_measured_scalar(1.0)

    ui.intents.append("release_grippers")
    extension.after_control_input_poll(now=10.0)
    for now in (10.0, 10.11, 10.22, 10.33, 10.44):
        session.command_override_provider(SimpleNamespace(), now)
        assert extension._pending_gripper_action == "release_grippers"
    assert len(session.external_commands) == 5
    assert all(
        float(command.commands["right_hand"].value[0]) == pytest.approx(0.95)
        for command in session.external_commands
    )

    session.command_override_provider(SimpleNamespace(), 10.51)

    assert extension._pending_gripper_action is None
    assert "控制超时" in extension.status().gripper_action_status


def test_manual_gripper_close_requires_strict_endpoint_and_sends_final_target() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )
    backend = extension._components.right_hand_backend
    backend.set_measured_scalar(0.981)

    ui.intents.append("close_grippers")
    extension.after_control_input_poll(now=10.0)
    session.command_override_provider(SimpleNamespace(), 10.0)

    assert extension._pending_gripper_action == "close_grippers"
    command = session.external_commands[-1].commands["right_hand"]
    assert float(command.value[0]) == pytest.approx(1.0)
    assert command.metadata["dmtac_continuous_grasp_override"] is True

    backend.set_measured_scalar(1.0)
    for now in (10.1, 10.2):
        session.command_override_provider(SimpleNamespace(), now)
        assert extension._pending_gripper_action == "close_grippers"
    session.command_override_provider(SimpleNamespace(), 10.3)
    assert extension._pending_gripper_action is None
    assert extension.status().gripper_action_status == "夹爪实测已闭紧"


def test_manual_gripper_endpoint_waits_for_post_command_fresh_frames() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())

    class TimestampedHandPoller:
        def __init__(self) -> None:
            self.value = 0.981
            self.timestamp = 9.0
            self.status_code = 2
            self.is_moving = False

        def latest_at_or_before(self, _now, **_kwargs):
            measured = np.zeros(6, dtype=np.float32)
            measured[0] = self.value
            return {
                "right": {
                    "measured_joint_pos": measured,
                    "joint_age_sec": 0.0,
                    "joint_timestamp": self.timestamp,
                    "gripper_status_code": self.status_code,
                    "gripper_is_moving": self.is_moving,
                }
            }

    hand_poller = TimestampedHandPoller()
    recorder.hand_state_poller = hand_poller
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )

    ui.intents.append("close_grippers")
    extension.after_control_input_poll(now=10.0)
    session.command_override_provider(SimpleNamespace(), 10.0)

    hand_poller.value = 1.0
    hand_poller.timestamp = 10.0  # The pre-dispatch cached frame is rejected.
    session.command_override_provider(SimpleNamespace(), 10.1)
    assert extension._pending_gripper_action == "close_grippers"
    assert "新鲜反馈" in extension.status().gripper_action_status

    hand_poller.status_code = 4
    hand_poller.is_moving = True
    for timestamp, now in ((10.1, 10.2), (10.2, 10.3)):
        hand_poller.timestamp = timestamp
        session.command_override_provider(SimpleNamespace(), now)
        assert extension._pending_gripper_action == "close_grippers"
    assert "闭合运动" in extension.status().gripper_action_status

    # Position=1 with status 4 is still the user's observed failure mode:
    # the gripper is closing, not settled.  Only an idle closed status may
    # contribute endpoint samples.
    hand_poller.status_code = 2
    hand_poller.is_moving = False
    for timestamp, now in ((10.3, 10.4), (10.4, 10.5)):
        hand_poller.timestamp = timestamp
        session.command_override_provider(SimpleNamespace(), now)
        assert extension._pending_gripper_action == "close_grippers"
    hand_poller.timestamp = 10.5
    session.command_override_provider(SimpleNamespace(), 10.6)
    assert extension._pending_gripper_action is None
    assert extension.status().gripper_action_status == "夹爪实测已闭紧"


def test_manual_gripper_stops_on_measured_reverse_motion() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )
    backend = extension._components.right_hand_backend
    backend.set_measured_scalar(0.5)
    ui.intents.append("release_grippers")
    extension.after_control_input_poll(now=10.0)
    session.command_override_provider(SimpleNamespace(), 10.0)
    backend.set_measured_scalar(0.53)
    session.command_override_provider(SimpleNamespace(), 10.1)

    assert extension._pending_gripper_action is None
    assert "相反" in extension.status().gripper_action_status


@pytest.mark.parametrize("writing_mode", (False, True))
def test_manual_gripper_close_then_release_recovers_from_real_backend_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
    writing_mode: bool,
) -> None:
    class _Clock:
        def __init__(self) -> None:
            self.now = 1000.0

        def time(self) -> float:
            return self.now

        def perf_counter(self) -> float:
            return self.now

    clock = _Clock()
    monkeypatch.setattr(dmtac_module, "time", clock)
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
        source_selection=(
            {"collection_mode": "rule_based_writing"} if writing_mode else {}
        ),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )
    backend = DMTacGripperBackend(
        DMTacGripperBackendConfig(
            side="right",
            connect_hardware=False,
            connect_tactile=False,
            enable_telemetry=False,
            command_hz=20.0,
            send_threshold=0.002,
        )
    )
    extension._components.right_hand_backend = backend

    def set_measured(value: float) -> None:
        measured = np.zeros(6, dtype=np.float32)
        measured[0] = float(value)
        backend._latest_measured_joint_pos = measured
        backend._latest_joint_timestamp = clock.now

    set_measured(0.0)

    def dispatch(
        command,
        observation,
        now,
        *,
        record,
        transition_to_running,
    ):
        del observation, now
        assert record is False
        assert transition_to_running is False
        session.external_commands.append(command)
        receipts = {}
        for name, effector_command in command.commands.items():
            sent = backend.send_command(effector_command)
            receipts[name] = sent
            if sent:
                set_measured(float(backend._last_send_target_scalar))
        return SimpleNamespace(dispatched=True, dispatch_receipts=receipts)

    session.dispatch_external_command = dispatch

    def run_until_complete(*, deadline: float) -> None:
        while extension._pending_gripper_action is not None:
            # With telemetry disabled this fixture has to advance the same
            # hardware timestamp that production HandStateHistoryPoller
            # advances after each fresh register read.
            backend._latest_joint_timestamp = clock.now
            session.command_override_provider(SimpleNamespace(), clock.now)
            clock.now += 0.01
            assert clock.now < deadline

    ui.intents.append("close_grippers")
    extension.after_control_input_poll(now=clock.now)
    run_until_complete(deadline=1005.0)
    assert extension.status().gripper_action_status == "夹爪实测已闭紧"
    assert float(backend._latest_measured_joint_pos[0]) == pytest.approx(1.0)

    ui.intents.append("release_grippers")
    extension.after_control_input_poll(now=clock.now)
    session.command_override_provider(SimpleNamespace(), clock.now)
    assert backend._last_send_status == "rate_limited"
    assert "后端限频" in extension.status().gripper_action_status
    clock.now += 0.01
    run_until_complete(deadline=1010.0)
    assert extension.status().gripper_action_status == "夹爪实测已松开"
    assert float(backend._latest_measured_joint_pos[0]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("backend_status", "expected"),
    (
        ("send_disabled", "发送开关关闭"),
        ("hardware_fault", "硬件故障"),
        ("not_connected", "未连接"),
        ("send_exception", "发送异常"),
    ),
)
def test_manual_gripper_terminal_backend_rejection_stops_instead_of_retrying(
    backend_status: str,
    expected: str,
) -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )
    backend = extension._components.right_hand_backend
    backend.set_measured_scalar(0.5)
    backend._last_send_status = backend_status
    backend._last_send_skip_reason = "fixture"

    def reject(
        command,
        observation,
        now,
        *,
        record,
        transition_to_running,
    ):
        del observation, now, record, transition_to_running
        session.external_commands.append(command)
        return SimpleNamespace(
            dispatched=True,
            dispatch_receipts={"right_hand": False},
        )

    session.dispatch_external_command = reject
    ui.intents.append("release_grippers")
    extension.after_control_input_poll(now=10.0)
    session.command_override_provider(SimpleNamespace(), 10.0)

    assert extension._pending_gripper_action is None
    assert expected in extension.status().gripper_action_status
    assert "等待重试" not in extension.status().gripper_action_status


def test_prestart_configuration_rebuild_updates_numeric_alignment_speed_off_loop() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(initial_alignment_speed_scale=1.0),
    )
    replacement = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(initial_alignment_speed_scale=0.25),
    )
    selected = {"initial_alignment_speed_scale": 1.0}
    commits: list[tuple[dict[str, object], object]] = []
    extension.configure_prestart_rebuild(
        options={"fixture": True},
        selected=selected,
        rebuilder=lambda request: replacement,
        on_commit=lambda request, adopted: commits.append((dict(request), adopted)),
    )
    calibration = RuleBasedWritingHeightCalibration(
        calibrated_z_m=0.321,
        lower_tolerance_m=0.001,
        upper_tolerance_m=0.001,
        captured_at=0.5,
        source_timestamp=0.5,
        pose_frame_contract_hash=extension.runtime_pose_frame_contract.contract_hash,
    )
    extension._height_calibration = calibration
    recorder = _Recorder(_Poller())
    _session, _clutch, ui = _bind(extension, recorder, arm=False)
    assert ui.selected_configuration == selected
    request = {"initial_alignment_speed_scale": 0.25}
    ui.configuration_requests.append(request)

    extension.after_control_input_poll(now=1.0)
    assert extension.status().configuration_pending is True
    assert extension.command_for_observation(SimpleNamespace(), 1.0).command is None
    for index in range(200):
        extension.after_control_input_poll(now=1.0 + index * 0.001)
        if not extension.status().configuration_pending:
            break
        time.sleep(0.001)

    assert extension.status().configuration_pending is False
    assert extension.config.initial_alignment_speed_scale == pytest.approx(0.25)
    assert extension._initial_alignment_planner.config.speed_scale == pytest.approx(0.25)
    assert extension._height_calibration is calibration
    assert ui.configuration_results[-1] == {"selected": request, "error": None}
    assert commits == [(request, replacement)]


def test_prestart_configuration_rejects_closed_executor_without_loop_crash() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
    )
    extension.configure_prestart_rebuild(
        options={"fixture": True},
        selected={"fixture": "old"},
        rebuilder=lambda _request: extension,
    )
    recorder = _Recorder(_Poller())
    _session, _clutch, ui = _bind(extension, recorder, arm=False)
    extension._configuration_executor.shutdown(wait=True, cancel_futures=True)
    ui.configuration_requests.append({"fixture": "new"})

    extension.after_control_input_poll(now=1.0)

    status = extension.status()
    assert status.configuration_pending is False
    assert status.waiting_reason == "configuration_rejected"
    assert "cannot schedule new futures after shutdown" in status.configuration_error
    assert ui.configuration_results[-1] == {
        "selected": None,
        "error": status.configuration_error,
    }


def test_prestart_configuration_completes_while_rig_fault_blocks_runtime_home() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(initial_alignment_speed_scale=1.0),
    )
    replacement = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(initial_alignment_speed_scale=0.5),
    )
    extension.configure_prestart_rebuild(
        options={"fixture": True},
        selected={"initial_alignment_speed_scale": 1.0},
        rebuilder=lambda _request: replacement,
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(extension, recorder, arm=False)
    session.state_machine.state = SimpleNamespace(value="DEBUG_ONLY")
    session.rig = SimpleNamespace(
        readiness_report=lambda: {
            "ready": False,
            "reason": "unready backends: right_hand",
            "backends": {
                "right_hand": {
                    "ready": False,
                    "reason": "runtime_fault_recovery_failed",
                }
            },
        }
    )
    request = {"initial_alignment_speed_scale": 0.5}
    ui.configuration_requests.append(request)

    assert runtime_extension_service_startup_control_plane(
        extension,
        now=1.0,
    )
    assert extension.status().configuration_pending is True
    assert extension.status().readiness_blockers["right_hand"] == (
        "runtime_fault_recovery_failed"
    )
    for index in range(200):
        runtime_extension_service_startup_control_plane(
            extension,
            now=1.0 + index * 0.001,
        )
        if not extension.status().configuration_pending:
            break
        time.sleep(0.001)

    status = extension.status()
    assert status.configuration_pending is False
    assert status.configuration_available is True
    assert status.rule_start_available is False
    assert status.readiness_blockers["right_hand"] == (
        "runtime_fault_recovery_failed"
    )
    assert extension.config.initial_alignment_speed_scale == pytest.approx(0.5)
    assert ui.configuration_results[-1] == {"selected": request, "error": None}


def test_prevalidated_batch_advances_only_after_successful_writer_close() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
    )

    def rebuild(request):
        return RuleBasedRealTeleopExtension(
            deployment=_deployment(),
            plan=plan,
            trajectory_groups={},
            config=RuleBasedRealTeleopConfig(
                initial_alignment_speed_scale=float(request["speed"])
            ),
        )

    commits = []
    extension.configure_prestart_rebuild(
        options={"fixture": True},
        selected={"speed": 1.0},
        rebuilder=rebuild,
        on_commit=lambda request, _replacement: commits.append(dict(request)),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(extension, recorder, arm=False)
    request = {"batch": [{"speed": 0.5}, {"speed": 0.25}]}
    ui.configuration_requests.append(request)
    extension.after_control_input_poll(now=1.0)
    for index in range(200):
        extension.after_control_input_poll(now=1.0 + index * 0.001)
        if not extension.status().configuration_pending:
            break
        time.sleep(0.001)

    status = extension.status()
    assert status.batch_count == 2
    assert status.batch_index == 0
    assert status.rule_configuration == {"speed": 0.5}
    assert status.rule_configuration_batch == ({"speed": 0.5}, {"speed": 0.25})
    assert commits == [{"speed": 0.5}]

    recorder.is_recording = True
    extension._completed = True
    extension._execution_begun = True
    extension.after_control_input_poll(now=2.0)
    assert extension.status().completed is True
    assert extension.status().batch_index == 0

    recorder.is_recording = False
    session.state_machine.state = SimpleNamespace(value="RUNNING")
    extension.after_control_input_poll(now=2.1)
    status = extension.status()
    assert status.completed is False
    assert status.batch_index == 1
    assert status.batch_completed_count == 1
    assert status.rule_configuration == {"speed": 0.25}
    assert status.rule_start_available is True
    assert commits == [{"speed": 0.5}, {"speed": 0.25}]

    extension._completed = True
    extension._execution_begun = True
    extension.after_control_input_poll(now=3.0)
    status = extension.status()
    assert status.batch_index == 0
    assert status.batch_completed_count == 2
    assert status.batch_cycle_complete is True
    assert status.rule_configuration == {"speed": 0.5}
    assert session.external_commands == []


def test_pre_pen_down_replay_cadence_counts_only_committed_normal_episodes() -> None:
    normal_plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="normal",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
            ),
        )
    )
    prelude_plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="with_prelude",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=normal_plan,
        pre_pen_down_replay_plan=prelude_plan,
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(
            pre_pen_down_replay_normal_episodes=2,
        ),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, _ui = _bind(extension, recorder, arm=False)
    session.state_machine.state = SimpleNamespace(value="RUNNING")

    assert extension.plan is normal_plan
    assert extension.status().pre_pen_down_replay_active is False

    # An aborted or held attempt never reaches the successful writer-close
    # transition and therefore cannot consume the cadence.
    extension._aborted = True
    extension._held = True
    extension._execution_begun = True
    extension.after_control_input_poll(now=1.0)
    assert extension.status().pre_pen_down_normal_completed_count == 0
    extension._aborted = False
    extension._held = False
    extension._execution_begun = False

    for expected_count, now in ((1, 2.0), (2, 3.0)):
        extension._completed = True
        extension._execution_begun = True
        extension.after_control_input_poll(now=now)
        status = extension.status()
        assert status.pre_pen_down_normal_completed_count == expected_count

    assert extension.plan is prelude_plan
    assert extension.status().pre_pen_down_replay_active is True
    assert extension._pre_pen_down_replay_schedule_metadata() == {
        "configured": True,
        "active_for_episode": True,
        "normal_episodes_between_replays": 2,
        "normal_successful_episodes_since_replay": 2,
    }

    extension._completed = True
    extension._execution_begun = True
    extension.after_control_input_poll(now=4.0)
    status = extension.status()
    assert status.pre_pen_down_normal_completed_count == 0
    assert status.pre_pen_down_replay_active is False
    assert extension.plan is normal_plan


def test_batch_adoption_changes_left_source_target_and_preserves_cadence() -> None:
    def rebuilt(request):
        left_qpos = float(request["left_qpos"])
        group = replace(
            _group(),
            arm_observation_actual_qpos={
                "left": np.full((2, 6), left_qpos, dtype=np.float32),
                "right": np.zeros((2, 6), dtype=np.float32),
            },
        )
        normal_plan = RuleBasedCollectionPlan(
            steps=(
                RuleBasedInitialize(
                    step_id=f"normal_{request['name']}",
                    group_key="source",
                    qpos_source="observation_actual_qpos",
                ),
            )
        )
        prelude_plan = RuleBasedCollectionPlan(
            steps=(
                RuleBasedInitialize(
                    step_id=f"prelude_{request['name']}",
                    group_key="source",
                    qpos_source="observation_actual_qpos",
                ),
            )
        )
        return RuleBasedRealTeleopExtension(
            deployment=_deployment(),
            plan=normal_plan,
            pre_pen_down_replay_plan=prelude_plan,
            trajectory_groups={"source": group},
            config=RuleBasedRealTeleopConfig(
                pre_pen_down_replay_normal_episodes=2,
            ),
        )

    first = {"name": "first", "left_qpos": 0.1}
    second = {"name": "second", "left_qpos": 0.2}
    extension = rebuilt(first)
    extension.configure_prestart_rebuild(
        options={"fixture": True},
        selected=first,
        rebuilder=rebuilt,
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(extension, recorder, arm=False)
    session.state_machine.state = SimpleNamespace(value="RUNNING")
    ui.configuration_requests.append({"batch": [first, second]})
    extension.after_control_input_poll(now=1.0)
    for index in range(200):
        extension.after_control_input_poll(now=1.0 + index * 0.001)
        if not extension.status().configuration_pending:
            break
        time.sleep(0.001)

    first_directive = next(
        RuleBasedCollectionCursor(
            extension.plan,
            extension.trajectory_groups,
        )
    )
    np.testing.assert_allclose(first_directive.qpos_by_side["left"], 0.1)
    assert extension.status().pre_pen_down_replay_active is False

    extension._completed = True
    extension._execution_begun = True
    extension.after_control_input_poll(now=2.0)
    second_directive = next(
        RuleBasedCollectionCursor(
            extension.plan,
            extension.trajectory_groups,
        )
    )
    np.testing.assert_allclose(second_directive.qpos_by_side["left"], 0.2)
    assert extension.status().pre_pen_down_normal_completed_count == 1
    assert extension.status().pre_pen_down_replay_active is False

    extension._completed = True
    extension._execution_begun = True
    extension.after_control_input_poll(now=3.0)
    wrapped_directive = next(
        RuleBasedCollectionCursor(
            extension.plan,
            extension.trajectory_groups,
        )
    )
    np.testing.assert_allclose(wrapped_directive.qpos_by_side["left"], 0.1)
    assert wrapped_directive.step_id == "prelude_first"
    assert extension.status().pre_pen_down_normal_completed_count == 2
    assert extension.status().pre_pen_down_replay_active is True


def test_zero_pre_pen_down_interval_preserves_replay_every_episode() -> None:
    normal_plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="normal",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
            ),
        )
    )
    prelude_plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="with_prelude",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=normal_plan,
        pre_pen_down_replay_plan=prelude_plan,
        trajectory_groups={},
    )
    assert extension.plan is prelude_plan
    assert extension.status().pre_pen_down_replay_active is True

    with pytest.raises(ValueError, match="non-negative integer"):
        RuleBasedRealTeleopConfig(pre_pen_down_replay_normal_episodes=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        RuleBasedRealTeleopConfig(pre_pen_down_replay_normal_episodes=True)


def test_successful_attempt_waits_for_finalize_then_releases_rule_fix_without_hold() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
    )
    recorder = _Recorder(_Poller())
    session, clutch, _ui = _bind(extension, recorder, arm=False)

    class Controller:
        fixed_sides = ("left",)
        calls = []

        def cancel_active(self, *, now, reason, clear_snapshots):
            self.calls.append((now, reason, clear_snapshots))
            self.fixed_sides = ()
            return SimpleNamespace(ok=True, message="released")

    controller = Controller()
    extension._components.fix_arm_controller = controller
    extension._deferred_progress_arm_fix_established = True
    extension._completed = True
    extension._execution_begun = True
    recorder.finalizing = True
    session.state_machine.state = SimpleNamespace(value="RUNNING")

    extension.after_control_input_poll(now=4.0)
    assert extension.status().completed is True
    assert controller.fixed_sides == ("left",)

    recorder.finalizing = False
    extension.after_control_input_poll(now=4.1)
    status = extension.status()
    assert status.completed is False
    assert status.execution_begun is False
    assert status.rule_start_available is True
    assert controller.fixed_sides == ()
    assert controller.calls == [(4.1, "rule-based Episode completed", False)]
    assert clutch.reasons == []

    # The shared Session intentionally remains RUNNING between successful
    # attempts.  A fresh START must still reopen the narrow command-inert
    # writer bootstrap used before the first rule qref.
    _ui.request_start()
    extension.after_control_input_poll(now=4.2)
    assert extension.status().armed is True
    assert extension.status().execution_begun is False
    assert extension.allows_pre_episode_armed_warmup() is True


def test_successful_attempt_opens_inter_episode_gripper_maintenance_window() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(
        extension,
        recorder,
        arm=False,
        hand_sides=("right",),
    )
    backend = extension._components.right_hand_backend
    backend.set_measured_scalar(0.5)
    session.state_machine.state = SimpleNamespace(value="RUNNING")
    extension._completed = True
    extension._execution_begun = True
    recorder.finalizing = True

    extension.after_control_input_poll(now=4.0)
    assert extension.status().completed is True
    assert extension.status().gripper_control_available is False

    recorder.finalizing = False
    extension.after_control_input_poll(now=4.1)
    ready = extension.status()
    assert ready.completed is False
    assert ready.execution_begun is False
    assert ready.gripper_control_available is True
    assert ready.rule_start_available is True

    ui.intents.append("release_grippers")
    extension.after_control_input_poll(now=4.2)
    assert extension.status().gripper_action_pending is True
    assert extension.status().rule_start_available is False
    session.command_override_provider(SimpleNamespace(), 4.2)

    assert len(session.external_commands) == 1
    command = session.external_commands[-1]
    assert set(command.commands) == {"right_hand"}
    assert command.metadata["dataset_recording_eligible"] is False
    assert float(command.commands["right_hand"].value[0]) == pytest.approx(0.45)

    recorder.is_recording = True
    extension._clear_manual_gripper_action()
    assert extension.status().gripper_control_available is False


def test_writer_blocked_armed_runtime_accepts_reconfiguration_without_motion() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
    )
    replacement = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(initial_alignment_speed_scale=0.4),
    )
    commits: list[dict[str, object]] = []
    rebuild_attempts = 0

    def rebuild(request):
        nonlocal rebuild_attempts
        rebuild_attempts += 1
        if request["initial_alignment_speed_scale"] == 0.3:
            raise ValueError("invalid writer reserve")
        return replacement

    extension.configure_prestart_rebuild(
        options={"fixture": True},
        selected={"initial_alignment_speed_scale": 1.0},
        rebuilder=rebuild,
        on_commit=lambda request, _replacement: commits.append(dict(request)),
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(extension, recorder, arm=True)
    assert extension.status().armed is True
    assert extension.status().waiting_reason == "formal_episode_not_active"

    ui.configuration_requests.append({"initial_alignment_speed_scale": 0.3})
    extension.after_control_input_poll(now=0.5)
    for index in range(200):
        extension.after_control_input_poll(now=0.5 + index * 0.001)
        if not extension.status().configuration_pending:
            break
        time.sleep(0.001)
    assert extension.status().waiting_reason == "configuration_rejected"
    assert "invalid writer reserve" in ui.configuration_results[-1]["error"]

    request = {"initial_alignment_speed_scale": 0.4}
    ui.configuration_requests.append(request)
    extension.after_control_input_poll(now=1.0)
    assert extension.command_for_observation(SimpleNamespace(), 1.0).command is None
    for index in range(200):
        extension.after_control_input_poll(now=1.0 + index * 0.001)
        if not extension.status().configuration_pending:
            break
        time.sleep(0.001)

    status = extension.status()
    assert status.armed is True
    assert status.execution_begun is False
    assert status.waiting_reason == "formal_episode_not_active"
    assert extension.config.initial_alignment_speed_scale == pytest.approx(0.4)
    assert commits == [request]
    assert rebuild_attempts == 2
    assert session.external_commands == []

    extension._execution_begun = True
    ui.configuration_requests.append({"initial_alignment_speed_scale": 0.3})
    extension.after_control_input_poll(now=2.0)
    assert "没有活动 Episode" in ui.configuration_results[-1]["error"]


@pytest.mark.parametrize("terminal_flag", ("_completed", "_aborted"))
def test_stopped_terminal_attempt_can_reconfigure_only_after_finalize(
    terminal_flag: str,
) -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
    )
    replacement = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(initial_alignment_speed_scale=0.5),
    )
    extension.configure_prestart_rebuild(
        options={"fixture": True},
        selected={"initial_alignment_speed_scale": 1.0},
        rebuilder=lambda _request: replacement,
    )
    recorder = _Recorder(_Poller())
    session, _clutch, ui = _bind(extension, recorder, arm=False)
    setattr(extension, terminal_flag, True)
    extension._execution_begun = True
    extension._held = True
    extension._admitted_capture_sequences.update({4, 5})
    transition_pending = [True]
    extension._collection_transition_active = lambda: transition_pending[0]

    assert extension.status().configuration_available is False
    transition_pending[0] = False
    recorder.finalizing = True
    assert extension.status().configuration_available is False
    recorder.finalizing = False
    assert extension.status().configuration_available is True

    request = {"initial_alignment_speed_scale": 0.5}
    ui.configuration_requests.append(request)
    extension.after_control_input_poll(now=3.0)
    for index in range(200):
        extension.after_control_input_poll(now=3.0 + index * 0.001)
        if not extension.status().configuration_pending:
            break
        time.sleep(0.001)

    status = extension.status()
    assert status.configuration_available is True
    assert status.completed is False
    assert status.aborted is False
    assert status.execution_begun is False
    assert status.held is True
    assert status.waiting_reason == "runtime_home_required_after_hold"
    assert extension._admitted_capture_sequences == set()
    assert extension.config.initial_alignment_speed_scale == pytest.approx(0.5)
    assert session.external_commands == []


def test_failed_terminal_reconfiguration_keeps_attempt_retryable() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={},
    )
    extension.configure_prestart_rebuild(
        options={"fixture": True},
        selected={"fixture": "old"},
        rebuilder=lambda _request: (_ for _ in ()).throw(ValueError("bad selection")),
    )
    recorder = _Recorder(_Poller())
    _session, _clutch, ui = _bind(extension, recorder, arm=False)
    extension._aborted = True
    extension._execution_begun = True
    extension._held = True

    ui.configuration_requests.append({"fixture": "bad"})
    extension.after_control_input_poll(now=4.0)
    for index in range(200):
        extension.after_control_input_poll(now=4.0 + index * 0.001)
        if not extension.status().configuration_pending:
            break
        time.sleep(0.001)

    status = extension.status()
    assert status.aborted is True
    assert status.execution_begun is True
    assert status.held is True
    assert status.configuration_available is True
    assert status.waiting_reason == "configuration_rejected"
    assert "bad selection" in ui.configuration_results[-1]["error"]


def test_rule_phase_metadata_matches_detailed_writing_profile() -> None:
    expected = {
        "right_write.pen_down": ("pen_down_replay", "replay"),
        "right_write.write": ("write_rule_base", "rule-base"),
        "right_write.pen_up": ("pen_up_relative_replay", "relative_replay"),
    }
    for step_id, (phase, execution) in expected.items():
        metadata = RuleBasedRealTeleopExtension._phase_metadata_for_step(
            step_id,
            "replay",
        )
        assert metadata["collection_task_stage_id"] == "right_write"
        assert metadata["collection_detail_stage_id"] == step_id
        assert metadata["rule_based_phase"] == phase
        assert metadata["rule_based_phase_execution"] == execution


def test_full_replay_phase_metadata_uses_single_complete_trajectory_stage() -> None:
    initialize = RuleBasedRealTeleopExtension._phase_metadata_for_step(
        "execute_task.replay_initial_alignment",
        "initialize",
        "full_trajectory_replay",
    )
    replay = RuleBasedRealTeleopExtension._phase_metadata_for_step(
        "execute_task.manipulate",
        "replay",
        "full_trajectory_replay",
    )

    assert initialize["collection_task_stage_id"] == "execute_task"
    assert initialize["collection_detail_stage_id"] is None
    assert initialize["rule_based_phase"] == "full_replay_initial_alignment"
    assert replay["collection_task_stage_id"] == "execute_task"
    assert replay["collection_detail_stage_id"] == "execute_task.full_replay"
    assert replay["rule_based_phase"] == "full_trajectory_replay"
    assert replay["rule_based_phase_execution"] == "replay"


def test_runtime_holds_only_last_certified_arm_qref_while_writer_pause_recovers() -> None:
    right_hand = np.zeros(6, dtype=np.float32)
    right_hand[0] = 0.1
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                        right_hand=right_hand,
                        metadata={
                            "hand_action_specs_by_side": {
                                "right": HandActionSpec.dmtac_gripper(
                                    "right"
                                ).to_metadata()
                            }
                        },
                    ),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    recorder.hand_state_poller = _HandPoller()
    session, _clutch, _ui = _bind(
        extension,
        recorder,
        hand_sides=("right",),
    )
    active = session.command_override_provider(SimpleNamespace(), 15.0)
    assert active.command is not None
    assert "right_hand" in active.command.commands
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=15.0,
    )

    recorder.is_paused = True
    paused = session.command_override_provider(SimpleNamespace(), 15.01)
    assert paused.command is not None
    assert set(paused.command.commands) == {"left_arm", "right_arm"}
    assert paused.command.metadata["rule_based_pause_hold"] is True
    assert paused.command.metadata["rule_based_capture_enabled"] is False
    assert paused.command.metadata["rule_based_sequence_index"] == (
        active.command.metadata["rule_based_sequence_index"]
    )
    assert extension._frame_emission is None
    assert extension._paused_hold_emission is not None
    assert extension.status().waiting_reason == "formal_writer_paused"
    assert extension.status().aborted is False
    admission = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True),
        now=15.01,
    )
    assert admission.allowed is False

    # The held command supplies the ordinary resume warmup with a real arm
    # dispatch receipt, but cannot advance the rule cursor while paused.
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=paused.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=15.01,
    )
    assert extension._current is not None
    assert extension._current.directive.sequence_index == (
        active.command.metadata["rule_based_sequence_index"]
    )
    recorder.is_paused = False
    resumed = session.command_override_provider(SimpleNamespace(), 15.02)
    assert resumed.command is not None
    assert "right_hand" in resumed.command.commands
    assert extension.status().aborted is False


def test_runtime_holds_only_arms_during_hand_backend_recovery_race() -> None:
    right_hand = np.zeros(6, dtype=np.float32)
    right_hand[0] = 0.1
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                        right_hand=right_hand,
                        metadata={
                            "hand_action_specs_by_side": {
                                "right": HandActionSpec.dmtac_gripper(
                                    "right"
                                ).to_metadata()
                            }
                        },
                    ),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    hand_poller = _HandPoller()
    recorder.hand_state_poller = hand_poller
    session, _clutch, _ui = _bind(
        extension,
        recorder,
        hand_sides=("right",),
    )
    active = session.command_override_provider(SimpleNamespace(), 20.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=20.0,
    )

    hand_poller.recovery_phase["right"] = "recovering"
    held = session.command_override_provider(SimpleNamespace(), 20.01)

    assert held.command is not None
    assert set(held.command.commands) == {"left_arm", "right_arm"}
    assert held.command.metadata["rule_based_pause_reason"] == (
        "causal_hand_backend_recovery:right:recovering"
    )
    assert extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True),
        now=20.01,
    ).allowed is False
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=held.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=20.01,
    )
    assert extension.status().aborted is False
    assert extension.status().waiting_reason == (
        "causal_hand_backend_recovery:right:recovering"
    )

    hand_poller.recovery_phase["right"] = "ready"
    resumed = session.command_override_provider(SimpleNamespace(), 20.02)
    assert resumed.command is not None
    assert "right_hand" in resumed.command.commands


def test_runtime_failed_hand_recovery_is_terminal_not_a_resumable_pause() -> None:
    right_hand = np.zeros(6, dtype=np.float32)
    right_hand[0] = 0.1
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                        right_hand=right_hand,
                        metadata={
                            "hand_action_specs_by_side": {
                                "right": HandActionSpec.dmtac_gripper(
                                    "right"
                                ).to_metadata()
                            }
                        },
                    ),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    hand_poller = _HandPoller()
    recorder.hand_state_poller = hand_poller
    session, _clutch, _ui = _bind(
        extension,
        recorder,
        hand_sides=("right",),
    )
    active = session.command_override_provider(SimpleNamespace(), 20.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=20.0,
    )

    hand_poller.recovery_phase["right"] = "failed"

    with pytest.raises(
        RuntimeError,
        match=r"lost formal collection readiness:.*right:failed",
    ):
        session.command_override_provider(SimpleNamespace(), 20.01)


def test_writer_pause_hold_keeps_initial_alignment_rewrite_guard_active() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=1),
    )
    poller = _Poller()
    for side in ("left", "right"):
        poller.qpos[side] = np.full(6, -0.02)
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    active = _wait_for_command(session, now=30.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=30.0,
    )
    recorder.is_paused = True
    paused = session.command_override_provider(SimpleNamespace(), 30.01)
    assert paused.command is not None
    rewritten = TeleopCommand(
        commands={
            name: command.copy()
            for name, command in paused.command.commands.items()
        },
        timestamp=paused.command.timestamp,
        metadata=dict(paused.command.metadata),
    )
    rewritten.commands["left_arm"].value = (
        np.asarray(rewritten.commands["left_arm"].value, dtype=float) + 0.01
    )

    fault = session.command_guard(rewritten, SimpleNamespace(), 30.01)

    assert fault is not None
    assert fault.code == "rule_based_initial_alignment_command_rewritten"


def test_transient_arm_read_failure_holds_then_resumes_without_cursor_advance() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(
            settle_consecutive_samples=1,
            arm_feedback_recovery_timeout_sec=0.5,
        ),
    )
    poller = _Poller()
    for side in ("left", "right"):
        poller.qpos[side] = np.full(6, -0.02)
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    active = _wait_for_command(session, now=32.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=32.0,
    )
    current = extension._current

    poller.blocked["right"] = "hardware_read_failed"
    paused = session.command_override_provider(SimpleNamespace(), 32.1)
    assert paused.command is None
    assert extension._current is current
    assert extension.status().aborted is False

    poller.blocked.clear()
    resumed = session.command_override_provider(SimpleNamespace(), 32.2)
    assert resumed.command is not None
    assert resumed.command.metadata.get("rule_based_pause_hold") is not True
    assert extension.status().aborted is False


def test_transient_read_pause_allows_normal_tracking_lag_to_converge() -> None:
    target = np.zeros(6)
    target[2] = 0.03  # More than the 0.75 degree pause-motion tolerance.
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(
                    left_arm=np.zeros(6), right_arm=target,
                ),
            ),)
        ),
        trajectory_groups={},
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    active = _wait_for_command(session, now=32.0)
    assert active.command.commands["right_arm"].value[2] == pytest.approx(0.03)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=32.0,
    )
    poller.blocked["right"] = "hardware_read_failed"
    assert session.command_override_provider(SimpleNamespace(), 32.1).command is None
    assert extension.status().aborted is False

    poller.blocked.clear()
    poller.qpos["right"][2] = 0.005
    session.command_override_provider(SimpleNamespace(), 32.2)

    assert extension.status().aborted is False, session.hold_reasons
    assert session.hold_reasons == []
    assert extension._replay_reference_pause_started_at is None
    assert extension._paused_arm_start_qpos_by_side == {}


def test_writer_pause_keeps_certified_qref_with_ordinary_tracking_lag() -> None:
    target = np.zeros(6)
    target[2] = 0.03
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(
                    left_arm=np.zeros(6), right_arm=target,
                ),
            ),)
        ),
        trajectory_groups={},
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    active = _wait_for_command(session, now=35.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=35.0,
    )

    recorder.is_paused = True
    paused = session.command_override_provider(SimpleNamespace(), 35.01)

    assert paused.command is not None
    assert paused.command.metadata["rule_based_pause_hold"] is True
    assert extension.status().aborted is False
    assert session.hold_reasons == []


def test_transient_read_pause_rejects_motion_away_from_tracking_corridor() -> None:
    target = np.zeros(6)
    target[2] = 0.03
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(
                    left_arm=np.zeros(6), right_arm=target,
                ),
            ),)
        ),
        trajectory_groups={},
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    active = _wait_for_command(session, now=33.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=33.0,
    )
    poller.blocked["right"] = "hardware_read_failed"
    assert session.command_override_provider(SimpleNamespace(), 33.1).command is None
    poller.blocked.clear()
    poller.qpos["right"][2] = -0.02

    rejected = session.command_override_provider(SimpleNamespace(), 33.2)

    assert rejected.command is None
    assert extension.status().aborted is True
    assert "right joint 3 moved outside" in session.hold_reasons[-1]


def test_sustained_arm_read_failure_remains_fail_closed_after_grace() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6),
                        right_arm=np.zeros(6),
                    ),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(
            settle_consecutive_samples=1,
            arm_feedback_recovery_timeout_sec=0.25,
        ),
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    active = _wait_for_command(session, now=34.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=34.0,
    )
    poller.blocked["left"] = "hardware_read_failed"
    assert session.command_override_provider(SimpleNamespace(), 34.1).command is None
    with pytest.raises(RuntimeError, match="lost formal collection readiness"):
        session.command_override_provider(SimpleNamespace(), 34.36)


def test_writer_pause_shifts_initial_alignment_clock_without_resume_catchup() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=1),
    )
    poller = _Poller()
    for side in ("left", "right"):
        poller.qpos[side] = np.full(6, -0.02)
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    first_time = 35.0
    active = _wait_for_command(session, now=first_time)
    for side in ("left", "right"):
        backend = getattr(extension._components, f"{side}_command_backend")
        assert len(backend.resets) == 1
        np.testing.assert_allclose(backend.resets[0], poller.qpos[side])
    active_index = int(
        active.command.metadata["rule_based_initial_alignment_waypoint_index"]
    )
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=first_time,
    )

    recorder.is_paused = True
    paused_at = first_time + 0.01
    paused = session.command_override_provider(SimpleNamespace(), paused_at)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=paused.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=paused_at,
    )

    resumed_at = first_time + 0.11
    recorder.is_paused = False
    resumed = session.command_override_provider(SimpleNamespace(), resumed_at)
    assert int(
        resumed.command.metadata["rule_based_initial_alignment_waypoint_index"]
    ) == active_index
    shifted_due = float(extension._next_due_at)
    assert shifted_due > resumed_at

    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=resumed_at,
    )
    advanced_at = shifted_due + 0.001
    advanced = session.command_override_provider(SimpleNamespace(), advanced_at)
    assert int(
        advanced.command.metadata["rule_based_initial_alignment_waypoint_index"]
    ) == active_index + 1
    assert advanced.command.metadata[
        "rule_based_scheduled_source_cut_timestamp"
    ] == pytest.approx(advanced_at)
    assert float(extension._next_due_at) - advanced_at >= 1.0 / 60.0 - 1e-9


def test_writer_pause_rejects_operator_moved_arm_before_old_qref_dispatch() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(
                    left_arm=np.zeros(6), right_arm=np.zeros(6)
                ),
            ),)
        ),
        trajectory_groups={},
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    active = _wait_for_command(session, now=40.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=40.0,
    )
    recorder.is_paused = True
    poller.qpos["right"][2] = 0.1

    paused = session.command_override_provider(SimpleNamespace(), 40.01)

    assert paused.command is None
    assert extension.status().aborted is True
    assert session.hold_reasons
    assert "right joint 3" in session.hold_reasons[-1]


def test_writer_resume_rechecks_measured_arm_after_safe_pause_hold() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(
                    left_arm=np.zeros(6), right_arm=np.zeros(6)
                ),
            ),)
        ),
        trajectory_groups={},
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    active = _wait_for_command(session, now=41.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=41.0,
    )
    recorder.is_paused = True
    safe_hold = session.command_override_provider(SimpleNamespace(), 41.01)
    assert safe_hold.command is not None
    recorder.is_paused = False
    poller.qpos["left"][0] = 0.1

    resumed = session.command_override_provider(SimpleNamespace(), 41.02)

    assert resumed.command is None
    assert extension.status().aborted is True
    assert session.hold_reasons


def test_writer_pause_hold_backend_rejection_fails_closed() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, clutch, _ui = _bind(extension, recorder)
    active = session.command_override_provider(SimpleNamespace(), 40.0)
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=active.command,
            command_dispatched=True,
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=40.0,
    )
    recorder.is_paused = True
    paused = session.command_override_provider(SimpleNamespace(), 40.01)

    extension.after_session_step(
        step_result=SimpleNamespace(
            command=paused.command,
            command_dispatched=True,
            dispatch_receipts={"left_arm": False},
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=40.01,
    )

    assert extension.status().aborted is True
    assert extension.status().held is True
    assert clutch.reasons == ["rule_based_runtime_hold"]


def test_initial_alignment_streams_qrefs_and_bounds_measured_tracking_lag() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=1),
    )
    poller = _Poller()
    for side in ("left", "right"):
        poller.qpos[side] = np.full(6, -0.12)
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    now = 50.0
    current = _wait_for_command(session, now=now)
    first_index = int(
        current.command.metadata["rule_based_initial_alignment_waypoint_index"]
    )
    assert first_index == 0
    for side in ("left", "right"):
        np.testing.assert_allclose(
            current.command.commands[f"{side}_arm"].value,
            poller.qpos[side],
            rtol=0.0,
            atol=1e-7,
        )
    first_source_cut = float(
        current.command.commands["left_arm"].metadata[
            "sample_aligned_source_timestamp"
        ]
    )
    first_parent = float(
        current.command.commands["left_arm"].metadata["_parent_command_timestamp"]
    )
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=now,
    )

    repeated = session.command_override_provider(SimpleNamespace(), now + 0.01)
    assert int(
        repeated.command.metadata["rule_based_initial_alignment_waypoint_index"]
    ) == first_index
    assert repeated.command.commands["left_arm"].metadata[
        "sample_aligned_source_timestamp"
    ] == pytest.approx(first_source_cut)
    assert repeated.command.commands["left_arm"].metadata[
        "_parent_command_timestamp"
    ] > first_parent

    # Causal state need not settle at the intermediate reference. Once the
    # shared 60 Hz period elapses, the next C2 qref is admitted.
    current = session.command_override_provider(SimpleNamespace(), now + 0.017)
    assert int(
        current.command.metadata["rule_based_initial_alignment_waypoint_index"]
    ) == first_index + 1
    assert current.command.commands["left_arm"].metadata[
        "sample_aligned_source_timestamp"
    ] > first_source_cut

    # If the real arm does not follow, the four-waypoint lead bound freezes
    # source advancement without faulting or issuing a farther qref.
    now += 0.017
    held_index = first_index + 1
    for _ in range(240):
        extension.after_session_step(
            step_result=SimpleNamespace(command_dispatched=True),
            now=now,
        )
        candidate = session.command_override_provider(
            SimpleNamespace(),
            now + 0.017,
        )
        candidate_index = int(
            candidate.command.metadata[
                "rule_based_initial_alignment_waypoint_index"
            ]
        )
        if candidate_index == held_index:
            break
        held_index = candidate_index
        current = candidate
        now += 0.017
    else:
        raise AssertionError("initial alignment did not enforce tracking lag bound")
    assert extension.status().waiting_reason == "initial_alignment_tracking_lag"
    assert extension.status().aborted is False


def test_initial_alignment_reacquires_zero_motion_qref_before_second_waypoint() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=3),
    )
    poller = _Poller()
    for side in ("left", "right"):
        poller.qpos[side] = np.full(6, -0.12)
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    now = 60.0
    first = _wait_for_command(session, now=now)
    assert first.command.metadata["rule_based_initial_alignment_waypoint_index"] == 0
    for sample_index in range(2):
        extension.after_session_step(
            step_result=SimpleNamespace(command_dispatched=True),
            now=now,
        )
        now += 0.02
        repeated = session.command_override_provider(SimpleNamespace(), now)
        assert (
            repeated.command.metadata[
                "rule_based_initial_alignment_waypoint_index"
            ]
            == 0
        )

    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=now,
    )
    advanced = session.command_override_provider(SimpleNamespace(), now + 0.02)
    assert advanced.command.metadata["rule_based_initial_alignment_waypoint_index"] == 1


def test_initial_alignment_slips_late_clock_without_catchup_burst() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=1),
    )
    poller = _Poller()
    for side in ("left", "right"):
        poller.qpos[side] = np.full(6, -0.12)
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    first_time = 70.0
    first = _wait_for_command(session, now=first_time)
    first_index = int(
        first.command.metadata["rule_based_initial_alignment_waypoint_index"]
    )
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=first_time,
    )

    late_time = first_time + 0.085
    second = session.command_override_provider(SimpleNamespace(), late_time)
    assert int(
        second.command.metadata["rule_based_initial_alignment_waypoint_index"]
    ) == first_index + 1
    lag = float(
        second.command.metadata["rule_based_initial_alignment_schedule_lag_sec"]
    )
    assert lag > 0.06
    assert second.command.metadata[
        "rule_based_initial_alignment_skipped_clock_ticks"
    ] >= 3
    assert second.command.metadata[
        "rule_based_scheduled_source_cut_timestamp"
    ] == pytest.approx(late_time)
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=late_time,
    )

    # The next call is well after the old absolute deadline but before one
    # full post-wakeup reference interval.  It must hold the same certified
    # qref instead of chasing the stale backlog.
    held = session.command_override_provider(SimpleNamespace(), late_time + 0.005)
    assert int(
        held.command.metadata["rule_based_initial_alignment_waypoint_index"]
    ) == first_index + 1
    for side in ("left", "right"):
        np.testing.assert_allclose(
            held.command.commands[f"{side}_arm"].value,
            second.command.commands[f"{side}_arm"].value,
            rtol=0.0,
            atol=0.0,
        )

    next_due = float(extension._next_due_at)
    assert next_due - late_time >= 1.0 / 60.0 - 1e-9
    third = session.command_override_provider(SimpleNamespace(), next_due + 1e-6)
    assert int(
        third.command.metadata["rule_based_initial_alignment_waypoint_index"]
    ) == first_index + 2


def test_initial_alignment_invalid_tracking_feedback_fails_closed() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedInitialize(step_id="init", group_key="source"),)
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=1),
    )
    poller = _Poller()
    for side in ("left", "right"):
        poller.qpos[side] = np.full(6, -0.12)
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, clutch, _ui = _bind(extension, recorder)

    current = _wait_for_command(session, now=60.0)
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=60.0,
    )
    poller.qpos["left"][0] = np.nan

    rejected = session.command_override_provider(SimpleNamespace(), 60.051)

    assert rejected.command is None
    assert extension.status().aborted is True
    assert extension.status().held is True
    assert clutch.reasons == ["rule_based_runtime_hold"]
    assert current.command is not None


def test_attachment_refuses_inactive_final_tcp_envelope() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
    )
    components = _components(_Session())
    components.dual_arm_retargeter.pre_ik_target_safety_filter.apply_to_target = False

    with pytest.raises(RuntimeError, match="Active final TCP envelope"):
        extension.attach_components(components)


def test_single_arm_canonical_pose_uses_checked_runtime_contract() -> None:
    action = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={"right": _pose(0.4, -0.1, 0.3)},
        active_effectors=("right_arm",),
        frame_by_effector={"right_arm": "canonical_desk"},
        controlled_frame_by_effector={"right_arm": "tool_tcp"},
        source="canonical_rule",
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedCustomAction(step_id="pose", action=action),)
        ),
        trajectory_groups={},
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    override = session.command_override_provider(SimpleNamespace(), 20.0)
    expected = extension.runtime_pose_frame_contract.canonical_desk_pose_vector_to_storage(
        action.arm_pose_by_side["right"],
        side="right",
    )

    assert override.command.commands["right_arm"].command_type == "joint_vector"
    assert override.command.metadata["rule_based_coupled_pose_safety"] is True
    target_pair = extension._components.dual_arm_retargeter.target_pairs[-1]
    assert target_pair.active_sides == ("right",)
    assert target_pair.hard_hold_inactive_sides is True
    np.testing.assert_allclose(
        override.command.metadata["rule_based_target_pose_base_by_side"]["right"][:3, 3],
        expected[:3],
    )
    assert session.command_guard(override.command, SimpleNamespace(), 20.0) is None


def test_recordable_rule_dmtac_command_is_collection_checked_before_dispatch() -> None:
    spec = HandActionSpec.dmtac_gripper("right")
    full = np.asarray([0.65, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="right_write.write",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6, dtype=np.float32),
                        right_arm=np.zeros(6, dtype=np.float32),
                        right_hand=full,
                        metadata={
                            "hand_action_specs_by_side": {
                                "right": spec.to_metadata()
                            }
                        },
                    ),
                    capture=RuleBasedCaptureSelection(enabled=True),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())
    recorder.hand_state_poller = _HandPoller()
    recorder.is_recording = True
    session, _clutch, _ui = _bind(
        extension,
        recorder,
        hand_sides=("right",),
    )
    override = _wait_for_command(session, now=20.0)
    hand_command = override.command.commands["right_hand"]
    assert np.asarray(hand_command.value).shape == (1,)

    assert session.command_guard(override.command, SimpleNamespace(), 20.0) is None

    hand_command.metadata["hand_qpos_full"] = np.asarray(
        [0.60, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    fault = session.command_guard(override.command, SimpleNamespace(), 20.01)
    assert fault is not None
    assert fault.code == "rule_based_collection_action_contract_invalid"
    assert extension._current_dispatched is False


def test_float32_rule_pose_drift_is_bounded_projected_before_coupled_safety() -> None:
    action = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={"right": _float32_pose_with_nontrivial_rotation()},
        active_effectors=("right_arm",),
        frame_by_effector={"right_arm": "canonical_desk"},
        controlled_frame_by_effector={"right_arm": "tool_tcp"},
        source="canonical_rule_float32",
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedCustomAction(step_id="pose", action=action),)
        ),
        trajectory_groups={},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    override = session.command_override_provider(SimpleNamespace(), 20.0)

    assert override.command is not None
    target = extension._components.dual_arm_retargeter.target_pairs[-1]
    right_rotation = target.target_pose_by_side["right"][:3, :3]
    np.testing.assert_allclose(
        right_rotation.T @ right_rotation,
        np.eye(3),
        atol=1e-9,
        rtol=0.0,
    )
    diagnostics = override.command.metadata[
        "rule_based_rigid_pose_projection_by_side"
    ]
    assert diagnostics["right"]["rotation_projected"] is True
    assert 1e-9 < diagnostics["right"]["rotation_orthogonality_error"] < 1e-5
    source = target.source_by_side["right"]
    assert source is not None
    assert source.metadata["rule_based_rigid_pose_projection"] == diagnostics["right"]


@pytest.mark.parametrize("bad_rotation", ["scaled", "reflection"])
def test_materially_invalid_inactive_pose_holds_before_coupled_ik(
    bad_rotation: str,
) -> None:
    action = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={"right": _pose(0.4, -0.1, 0.3)},
        active_effectors=("right_arm",),
        frame_by_effector={"right_arm": "canonical_desk"},
        controlled_frame_by_effector={"right_arm": "tool_tcp"},
        source="canonical_rule",
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedCustomAction(step_id="pose", action=action),)
        ),
        trajectory_groups={},
    )
    poller = _Poller()
    if bad_rotation == "scaled":
        poller.tcp["left"][0, 0] = 1.01
    else:
        poller.tcp["left"][0, 0] = -1.0
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, clutch, _ui = _bind(extension, recorder)

    with pytest.raises(RuntimeError, match="coupled rule pose resolution failed"):
        session.command_override_provider(SimpleNamespace(), 20.0)

    assert extension.status().aborted is True
    assert extension.status().held is True
    assert clutch.reasons == ["rule_based_runtime_hold"]
    assert extension._components.dual_arm_retargeter.target_pairs == []


def test_pose_solver_rejection_holds_before_dispatch() -> None:
    action = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={"right": _pose(0.4, -0.1, 0.3)},
        active_effectors=("right_arm",),
        frame_by_effector={"right_arm": "canonical_desk"},
        controlled_frame_by_effector={"right_arm": "tool_tcp"},
        source="canonical_rule",
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedCustomAction(step_id="pose", action=action),)
        ),
        trajectory_groups={},
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    extension._components.dual_arm_retargeter.command_status = "hold_last_safe"

    with pytest.raises(RuntimeError, match="refused the rule pose"):
        session.command_override_provider(SimpleNamespace(), 21.0)

    assert extension.status().aborted is True
    assert extension.status().held is True
    assert session.hold_reasons


def test_terminal_runtime_segment_publishes_measured_relative_boundary() -> None:
    source = CallableActionSource(
        lambda _observation: RobotAction.bimanual_qpos(
            left_arm=np.zeros(6),
            right_arm=np.zeros(6),
            source="bounded_rule",
        )
    )
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedActionSegment(
                step_id="collect",
                action_source=source,
                stop_condition=RuleBasedSegmentStopCondition(max_steps=1),
            ),
            RuleBasedReplay(
                step_id="relative",
                group_key="source",
                start_index=1,
                end_index=2,
                relative_to="collect",
                source_start_index=0,
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=1),
    )
    poller = _Poller()
    poller.tcp["left"][:3, 3] = [0.5, 0.0, 0.5]
    poller.tcp["right"][:3, 3] = [0.7, 0.0, 0.5]
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    terminal = session.command_override_provider(SimpleNamespace(), 30.0)
    assert terminal.command is not None
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=30.0,
    )
    admission = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True),
        now=30.0,
    )
    extension.after_dataset_sample_attempt(
        admission=admission,
        accepted=True,
        now=30.005,
    )
    relative = session.command_override_provider(SimpleNamespace(), 30.01)

    assert extension.status().published_boundaries == ("collect",)
    assert relative.command.metadata["rule_based_step_id"] == "relative"
    left_target = relative.command.metadata["rule_based_target_pose_base_by_side"][
        "left"
    ]
    assert left_target[0, 3] == pytest.approx(0.52)


def test_formal_sample_gate_defaults_open_and_rejects_bad_hook_type() -> None:
    default = runtime_extension_dataset_sample_admission(
        None,
        step_result=SimpleNamespace(),
        now=1.0,
    )
    assert default == RealTeleopDatasetSampleAdmission()

    bad = SimpleNamespace(dataset_sample_admission=lambda **_kwargs: True)
    with pytest.raises(TypeError, match="RealTeleopDatasetSampleAdmission"):
        runtime_extension_dataset_sample_admission(
            bad,
            step_result=SimpleNamespace(),
            now=1.0,
        )


def test_late_writer_ack_does_not_block_replay_and_live_time_is_rebound() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedReplay(
                    step_id="collect",
                    group_key="source",
                    start_index=0,
                    end_index=2,
                ),
            )
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(action_hz=10.0),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    first = session.command_override_provider(SimpleNamespace(), 2000.0).command
    sequence = first.metadata["rule_based_sequence_index"]
    first_timestamp = float(first.timestamp)
    assert first_timestamp == pytest.approx(2000.0)
    assert all(
        item.metadata["_parent_command_timestamp"] == pytest.approx(2000.0)
        for item in first.commands.values()
    )
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True), now=2000.0
    )
    admission = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True), now=2000.0
    )
    # The process writer ACK is deliberately late.  Motion must still advance
    # on its 60 Hz reference clock instead of waiting on the 15 Hz collector.
    advanced = session.command_override_provider(SimpleNamespace(), 2000.02).command
    assert advanced.metadata["rule_based_sequence_index"] == sequence
    assert advanced.metadata["rule_based_reference_sequence"] == 1
    assert advanced.metadata["rule_based_capture_enabled"] is True
    assert advanced.timestamp > first_timestamp
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True), now=2000.02
    )
    # Its causal admission identity remains valid after the motion cursor has
    # moved to a later reference.
    extension.after_dataset_sample_attempt(
        admission=admission, accepted=True, now=2000.03
    )
    assert sequence in extension._admitted_capture_sequences


def test_late_successful_writer_ack_clears_a_prior_rejection_interval() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedReplay(
                    step_id="collect",
                    group_key="source",
                    start_index=0,
                    end_index=2,
                ),
            )
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(action_hz=10.0),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    first = session.command_override_provider(SimpleNamespace(), 2100.0).command
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True), now=2100.0
    )
    rejected = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True), now=2100.0
    )
    extension.after_dataset_sample_attempt(
        admission=rejected, accepted=False, now=2100.001
    )
    assert extension._writer_retry_started_at == pytest.approx(2100.001)

    session.command_override_provider(SimpleNamespace(), 2100.02)
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True), now=2100.02
    )
    accepted = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True), now=2100.02
    )
    session.command_override_provider(SimpleNamespace(), 2100.12)
    extension.after_dataset_sample_attempt(
        admission=accepted, accepted=True, now=2100.13
    )

    assert extension._writer_retry_started_at is None
    assert int(first.metadata["rule_based_sequence_index"]) in (
        extension._admitted_capture_sequences
    )


def test_replay_uses_source_duration_with_dense_60hz_coupled_references() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedReplay(
                    step_id="collect",
                    group_key="source",
                    start_index=0,
                    end_index=2,
                ),
            )
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(
            action_hz=10.0,
            position_reference_hz=60.0,
        ),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    references = []
    for index in range(7):
        now = 3000.0 + index / 60.0 + 1e-6
        command = session.command_override_provider(SimpleNamespace(), now).command
        assert command is not None
        references.append(command)
        extension.after_session_step(
            step_result=SimpleNamespace(command_dispatched=True),
            now=now,
        )
        admission = extension.dataset_sample_admission(
            step_result=SimpleNamespace(command_dispatched=True),
            now=now,
        )
        if admission.allowed:
            extension.after_dataset_sample_attempt(
                admission=admission,
                accepted=True,
                now=now,
            )

    assert [item.metadata["rule_based_reference_sequence"] for item in references] == list(
        range(7)
    )
    assert references[0].metadata["rule_based_reference_timestamp_sec"] == pytest.approx(0.0)
    assert references[-1].metadata["rule_based_reference_timestamp_sec"] == pytest.approx(0.1)
    assert all(item.metadata["position_reference_hz"] == 60.0 for item in references)
    assert all(item.metadata["rule_based_coupled_pose_safety"] for item in references)
    target_sources = extension._components.dual_arm_retargeter.target_pairs[0].source_by_side
    for side in ("left", "right"):
        source = target_sources[side]
        assert source is not None
        intent = source.metadata["pose_action_motion_intent"]
        assert intent["source"] == "pose_action_interval_velocity_state_machine"
        assert intent["state"] == "ACTIVE"
        assert intent["evidence"]["quality_valid"] is True
        assert intent["evidence"]["linear_speed_p90_m_s"] == pytest.approx(0.2)
    # The 15 Hz dataset deadline may land on any of the four 60 Hz references
    # in a source interval.  All stay eligible, while directive-sequence
    # de-duplication commits at most one sample for each source frame.
    assert all(
        bool(item.metadata["rule_based_capture_enabled"]) for item in references
    )
    assert {
        item.metadata["rule_based_sequence_index"]
        for item in references
    } == {0, 1}


def test_opted_in_custom_segment_uses_dense_reference_clock_and_source_capture_ids() -> None:
    source_actions = iter(
        (
            RobotAction(
                action_space=BIMANUAL_EE_POSE_V1,
                timestamp=0.0,
                arm_pose_by_side={
                    "left": _pose(0.0),
                    "right": _pose(0.0),
                },
                frame_by_effector={
                    "left_arm": "canonical_desk",
                    "right_arm": "canonical_desk",
                },
                controlled_frame_by_effector={
                    "left_arm": "tool_tcp",
                    "right_arm": "tool_tcp",
                },
                source="writing_keyframe",
                metadata={
                    "rule_based_position_reference_resample": True,
                    "rule_based_position_keyframe_hz": 15.0,
                },
            ),
            RobotAction(
                action_space=BIMANUAL_EE_POSE_V1,
                timestamp=1.0 / 15.0,
                arm_pose_by_side={
                    "left": _pose(0.0),
                    "right": _pose(0.02),
                },
                frame_by_effector={
                    "left_arm": "canonical_desk",
                    "right_arm": "canonical_desk",
                },
                controlled_frame_by_effector={
                    "left_arm": "tool_tcp",
                    "right_arm": "tool_tcp",
                },
                source="writing_keyframe",
                metadata={
                    "rule_based_position_reference_resample": True,
                    "rule_based_position_keyframe_hz": 15.0,
                },
            ),
        )
    )
    source_calls = 0

    def produce(_observation):
        nonlocal source_calls
        source_calls += 1
        return next(source_actions)

    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedActionSegment(
                    step_id="right_write.write",
                    action_source=CallableActionSource(produce),
                    stop_condition=RuleBasedSegmentStopCondition(max_steps=2),
                    capture=RuleBasedCaptureSelection(enabled=True),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(
            action_hz=15.0,
            position_reference_hz=60.0,
        ),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    references = []
    admitted_sequences = []
    for index in range(5):
        now = 3200.0 + index / 60.0 + 1e-6
        command = session.command_override_provider(SimpleNamespace(), now).command
        assert command is not None
        references.append(command)
        extension.after_session_step(
            step_result=SimpleNamespace(command_dispatched=True),
            now=now,
        )
        admission = extension.dataset_sample_admission(
            step_result=SimpleNamespace(command_dispatched=True),
            now=now,
        )
        if admission.allowed:
            admitted_sequences.append(
                int(admission.metadata["rule_based_sequence_index"])
            )
            extension.after_dataset_sample_attempt(
                admission=admission,
                accepted=True,
                now=now,
            )

    assert source_calls == 2
    assert [
        item.metadata["rule_based_reference_sequence"] for item in references
    ] == list(range(5))
    assert references[0].metadata["rule_based_reference_timestamp_sec"] == pytest.approx(
        0.0
    )
    assert references[-1].metadata[
        "rule_based_reference_timestamp_sec"
    ] == pytest.approx(1.0 / 15.0)
    assert all(
        item.metadata["rule_based_reference_source_kind"] == "custom_action"
        for item in references
    )
    assert all(
        item.metadata["rule_based_phase"] == "write_rule_base"
        for item in references
    )
    assert all(
        item.metadata["rule_based_coupled_pose_safety"] for item in references
    )
    right_positions = np.stack(
        [
            item.metadata["rule_based_target_pose_base_by_side"]["right"][:3, 3]
            for item in references
        ]
    )
    left_positions = np.stack(
        [
            item.metadata["rule_based_target_pose_base_by_side"]["left"][:3, 3]
            for item in references
        ]
    )
    assert np.max(np.linalg.norm(np.diff(right_positions, axis=0), axis=1)) <= 0.005001
    assert np.allclose(np.diff(left_positions, axis=0), 0.0)
    assert {
        item.metadata["rule_based_sequence_index"] for item in references
    } == {0, 1}
    assert admitted_sequences == [0, 1]


def test_custom_action_reference_accepts_source_frame_used_as_right_bracket() -> None:
    """Jittered source times must not fail when interpolation still represents a frame."""

    timestamps = (0.0, 0.018, 0.019, 0.04)
    source_actions = iter(
        RobotAction(
            action_space=BIMANUAL_EE_POSE_V1,
            timestamp=timestamp,
            arm_pose_by_side={
                "left": _pose(0.0),
                "right": _pose(index * 0.01),
            },
            frame_by_effector={
                "left_arm": "canonical_desk",
                "right_arm": "canonical_desk",
            },
            controlled_frame_by_effector={
                "left_arm": "tool_tcp",
                "right_arm": "tool_tcp",
            },
            source="writing_keyframe",
            metadata={
                "rule_based_position_reference_resample": True,
                "rule_based_position_keyframe_hz": 30.0,
            },
        )
        for index, timestamp in enumerate(timestamps)
    )

    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedActionSegment(
                    step_id="right_write.write",
                    action_source=CallableActionSource(
                        lambda _observation: next(source_actions)
                    ),
                    stop_condition=RuleBasedSegmentStopCondition(
                        max_steps=len(timestamps),
                        min_steps=len(timestamps),
                        name="jittered_source_complete",
                    ),
                    capture=RuleBasedCaptureSelection(enabled=True),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(
            action_hz=30.0,
            position_reference_hz=60.0,
        ),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    first = session.command_override_provider(SimpleNamespace(), 3300.0).command
    assert first is not None
    assert any(
        carrier.right_parent_source_index == 1
        for carrier in extension._replay_reference_queue
    )
    second = session.command_override_provider(
        SimpleNamespace(), 3300.0 + (1.0 / 60.0) + 1e-6
    ).command
    assert second is not None
    # Source frame 1 is not the left parent of any reference in this jittered
    # timeline, but it is retained as the right interpolation bracket.
    assert second.metadata["rule_based_reference_source_kind"] == "custom_action"
    coverage = extension._stage_sample_coverage()["right_write.write"]
    assert coverage == {
        "planned": 3,
        "required": 3,
        "accepted": 0,
        "complete": False,
    }


def test_stage_coverage_uses_writer_rate_for_dense_reference_plan() -> None:
    source_actions = iter(
        RobotAction(
            action_space=BIMANUAL_EE_POSE_V1,
            timestamp=index / 60.0,
            arm_pose_by_side={
                "left": _pose(0.0),
                "right": _pose(index * 0.001),
            },
            frame_by_effector={
                "left_arm": "canonical_desk",
                "right_arm": "canonical_desk",
            },
            controlled_frame_by_effector={
                "left_arm": "tool_tcp",
                "right_arm": "tool_tcp",
            },
            source="writing_keyframe",
            metadata={
                "rule_based_position_reference_resample": True,
                "rule_based_position_keyframe_hz": 60.0,
            },
        )
        for index in range(12)
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedActionSegment(
                    step_id="right_write.write",
                    action_source=CallableActionSource(
                        lambda _observation: next(source_actions)
                    ),
                    stop_condition=RuleBasedSegmentStopCondition(
                        max_steps=12,
                        min_steps=12,
                        name="dense_source_complete",
                    ),
                    capture=RuleBasedCaptureSelection(enabled=True),
                ),
            )
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(
            action_hz=15.0,
            position_reference_hz=60.0,
        ),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    first = session.command_override_provider(SimpleNamespace(), 3400.0).command

    assert first is not None
    assert extension._writer_rate_expected_samples_by_step == {
        "right_write.write": 3
    }
    extension._accepted_samples_by_detail_stage["right_write.write"] = 2
    assert set(extension._incomplete_detail_stages()) == {"right_write.write"}
    extension._accepted_samples_by_detail_stage["right_write.write"] = 3
    assert extension._incomplete_detail_stages() == {}


def test_replay_queue_resolves_only_source_keyframes_and_live_references() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedReplay(
                    step_id="collect",
                    group_key="source",
                    start_index=0,
                    end_index=2,
                ),
            )
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(
            action_hz=10.0,
            position_reference_hz=60.0,
        ),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    resolver = extension.adapter.resolver
    resolve_calls = 0
    original_resolve = resolver.resolve

    def counted_resolve(action, **kwargs):
        nonlocal resolve_calls
        resolve_calls += 1
        return original_resolve(action, **kwargs)

    resolver.resolve = counted_resolve

    first = session.command_override_provider(SimpleNamespace(), 3500.0).command

    assert first is not None
    assert first.metadata["rule_based_reference_sequence"] == 0
    # The cursor resolves each of the two 10 Hz source actions once.  The
    # seven 60 Hz references are no longer all resolved synchronously during
    # first-frame queue construction; only the emitted reference goes through
    # the live coupled resolver.
    assert resolve_calls == 2


def test_full_replay_large_hand_source_delta_is_bounded_only_at_live_dispatch() -> None:
    group = _full_replay_jump_hand_group()
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedReplay(
                    step_id="execute_task.manipulate",
                    group_key="source",
                    start_index=0,
                    end_index=2,
                    include_end_effectors=True,
                    end_effector_sides=("left", "right"),
                    wait_for_terminal_arrival=True,
                    capture=RuleBasedCaptureSelection(enabled=True),
                ),
            )
        ),
        trajectory_groups={"source": group},
        config=RuleBasedRealTeleopConfig(
            action_hz=10.0,
            position_reference_hz=60.0,
            settle_consecutive_samples=1,
            gripper_sides=("left", "right"),
            gripper_max_step=0.05,
        ),
        source_selection={"collection_mode": "full_trajectory_replay"},
    )
    arm_poller = _Poller()
    for side in ("left", "right"):
        arm_poller.tcp[side] = _matrix(_pose(0.0))
    hand_poller = _HandPoller()
    recorder = _Recorder(arm_poller)
    recorder.hand_state_poller = hand_poller
    recorder.is_recording = True
    session, _clutch, _ui = _bind(
        extension,
        recorder,
        hand_sides=("left", "right"),
    )

    first = session.command_override_provider(SimpleNamespace(), 7000.0).command
    assert first is not None
    assert float(first.commands["right_hand"].value[0]) == pytest.approx(0.0)
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=7000.0,
    )
    first_admission = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True),
        now=7000.0,
    )
    assert first_admission.allowed
    extension.after_dataset_sample_attempt(
        admission=first_admission,
        accepted=True,
        now=7000.0,
    )

    now = 7000.100001
    measured = 0.0
    dispatched: list[tuple[float, float]] = []
    saw_large_source_target = False
    while extension._current is not None or not dispatched:
        command = session.command_override_provider(SimpleNamespace(), now).command
        assert command is not None
        if not dispatched:
            assert command.metadata["rule_based_reference_sequence"] == 1
            assert command.metadata["right_source_index"] == 1
        target = float(command.commands["right_hand"].value[0])
        dispatched.append((measured, target))
        assert abs(target - measured) <= 0.050001
        assert command.metadata["rule_based_hand_dispatch_baseline"] == (
            "causal_measured_state"
        )
        source_target = float(
            command.metadata["rule_based_source_hand_qpos_by_side"]["right"][0]
        )
        if source_target > 0.6:
            saw_large_source_target = True
            assert source_target == pytest.approx(0.70)
        extension.after_session_step(
            step_result=SimpleNamespace(command_dispatched=True),
            now=now,
        )
        admission = extension.dataset_sample_admission(
            step_result=SimpleNamespace(command_dispatched=True),
            now=now,
        )
        if admission.allowed:
            extension.after_dataset_sample_attempt(
                admission=admission,
                accepted=True,
                now=now,
            )
        measured = target
        hand_poller.qpos["right"][0] = measured
        now += 0.01
        assert len(dispatched) < 30

    assert measured == pytest.approx(0.70)
    assert saw_large_source_target is True
    assert len(dispatched) > 2
    assert extension.status().completed is False
    completed = session.command_override_provider(SimpleNamespace(), now)
    assert completed.command is None
    assert extension.status().completed is True


def test_replay_rebases_late_clock_without_skipping_or_bursting_references() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedReplay(
                    step_id="collect",
                    group_key="source",
                    start_index=0,
                    end_index=2,
                ),
            )
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(
            action_hz=10.0,
            position_reference_hz=60.0,
        ),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)

    first = session.command_override_provider(SimpleNamespace(), 5000.0).command
    assert first.metadata["rule_based_reference_sequence"] == 0
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True), now=5000.0
    )

    # A 55 ms owner-loop delay must keep the next 16.7 ms reference instead of
    # jumping directly to 50 ms.  The remaining clock slips by the measured
    # lag, so neither a multi-grid target jump nor a fast stale backlog burst
    # can reach the physical command path.
    current = session.command_override_provider(SimpleNamespace(), 5000.055).command
    assert current.metadata["rule_based_reference_sequence"] == 1
    assert current.metadata["rule_based_reference_schedule_policy"] == (
        "ordered_rebase_no_skip"
    )
    assert current.metadata["rule_based_reference_schedule_shift_sec"] == pytest.approx(
        0.055 - (1.0 / 60.0)
    )
    assert current.metadata["rule_based_reference_rebased_tick_count"] == 1
    assert current.metadata["rule_based_scheduled_source_cut_timestamp"] == pytest.approx(
        5000.0 + (1.0 / 60.0)
    )
    assert current.metadata["rule_based_source_cut_timestamp"] == pytest.approx(
        5000.0 + (1.0 / 60.0)
    )
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True), now=5000.055
    )

    # The rebased next deadline is 16.7 ms after this delayed dispatch.  An
    # immediate owner-loop iteration therefore holds sequence 1; once due,
    # sequence 2 is emitted exactly once.
    held = session.command_override_provider(SimpleNamespace(), 5000.060).command
    assert held.metadata["rule_based_reference_sequence"] == 1
    # Continue at a sustained 40 Hz owner cadence.  Every reference remains
    # ordered even though each tick is later than the 60 Hz nominal clock;
    # the scheduler must not skip a point or emit two points on one tick.
    observed_sequences: list[int] = []
    observed_rebase_counts: list[int] = []
    for tick in (5000.080, 5000.105, 5000.130):
        next_reference = session.command_override_provider(
            SimpleNamespace(), tick
        ).command
        observed_sequences.append(
            int(next_reference.metadata["rule_based_reference_sequence"])
        )
        observed_rebase_counts.append(
            int(next_reference.metadata["rule_based_reference_rebased_tick_count"])
        )
        extension.after_session_step(
            step_result=SimpleNamespace(command_dispatched=True), now=tick
        )

    assert observed_sequences == [2, 3, 4]
    assert observed_rebase_counts == [2, 3, 4]


def test_accepted_rule_hold_snapshot_supports_immutable_motion_metadata() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedReplay(
                    step_id="collect",
                    group_key="source",
                    start_index=0,
                    end_index=2,
                ),
            )
        ),
        trajectory_groups={"source": _group()},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    command = session.command_override_provider(SimpleNamespace(), 6000.0).command
    command.metadata["immutable_motion_intent"] = MappingProxyType(
        {"nested": MappingProxyType({"state": "ACTIVE"})}
    )

    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=6000.0,
    )

    held = extension._last_dispatched_emission
    assert held is not None
    assert set(held.command.commands) == {"left_arm", "right_arm"}
    assert held.command.metadata["immutable_motion_intent"]["nested"]["state"] == (
        "ACTIVE"
    )


def test_replay_reference_clock_freezes_while_formal_writer_is_paused() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedReplay(
                    step_id="collect",
                    group_key="source",
                    start_index=0,
                    end_index=2,
                ),
            )
        ),
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(
            action_hz=10.0,
            position_reference_hz=60.0,
        ),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, ui = _bind(extension, recorder)

    session.command_override_provider(SimpleNamespace(), 4000.0)
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=4000.0,
    )
    admission = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True),
        now=4000.0,
    )
    extension.after_dataset_sample_attempt(
        admission=admission,
        accepted=True,
        now=4000.0,
    )
    extension._target_wait_started_at = 3999.0
    extension._dispatch_retry_started_at = 3999.25
    extension._writer_retry_started_at = 3999.5

    recorder.is_paused = True
    held = session.command_override_provider(SimpleNamespace(), 4000.005).command
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=4000.005,
    )
    assert held.metadata["rule_based_reference_sequence"] == 0
    assert ui.statuses[-1]["recording_ready"] is False

    recorder.is_paused = False
    resumed = session.command_override_provider(SimpleNamespace(), 4005.0).command
    assert resumed.metadata["rule_based_reference_sequence"] == 0
    assert extension._target_wait_started_at == pytest.approx(4003.995)
    # A successful paused-hold receipt clears the dispatch retry because the
    # backend did accept that hold.  The target and writer clocks still prove
    # that the 4.995 s pause was excluded from their bounded deadlines.
    assert extension._dispatch_retry_started_at is None
    assert extension._writer_retry_started_at == pytest.approx(4004.495)
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=4005.0,
    )

    advanced = session.command_override_provider(SimpleNamespace(), 4005.02).command
    assert advanced.metadata["rule_based_reference_sequence"] == 1
    assert advanced.metadata["rule_based_reference_timestamp_sec"] == pytest.approx(
        1.0 / 60.0
    )


@pytest.mark.parametrize(
    "backend_status",
    (
        "rate_limited",
        "io_busy",
        "hardware_fault",
        "not_connected",
        "send_exception",
    ),
)
def test_formal_dmtac_recoverable_false_receipt_retries_without_sampling(
    backend_status: str,
) -> None:
    right_hand = np.zeros(6, dtype=np.float32)
    right_hand[0] = 0.1
    action = RobotAction.bimanual_qpos(
        left_arm=np.zeros(6),
        right_arm=np.zeros(6),
        right_hand=right_hand,
        metadata={
            "hand_action_specs_by_side": {
                "right": HandActionSpec.dmtac_gripper("right").to_metadata()
            }
        },
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedCustomAction(step_id="collect", action=action),)
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(action_hz=10.0, gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    recorder.hand_state_poller = _HandPoller()
    session, _clutch, _ui = _bind(extension, recorder, hand_sides=("right",))
    first = session.command_override_provider(SimpleNamespace(), 1000.0).command
    sequence = first.metadata["rule_based_sequence_index"]
    extension._components.right_hand_backend._last_send_status = backend_status
    extension.after_session_step(
        step_result=SimpleNamespace(
            command_dispatched=True,
            dispatch_receipts={
                "left_arm": True,
                "right_arm": True,
                "right_hand": False,
            },
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=1000.0,
    )
    assert extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True), now=1000.0
    ).allowed is False
    retry = session.command_override_provider(SimpleNamespace(), 1000.2).command
    assert retry.metadata["rule_based_sequence_index"] == sequence
    assert extension.status().aborted is False
    extension.after_session_step(
        step_result=SimpleNamespace(
            command_dispatched=True,
            dispatch_receipts={
                "left_arm": True,
                "right_arm": True,
                "right_hand": False,
            },
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=1031.0,
    )
    assert extension.status().aborted is True
    assert extension.status().held is True


def test_formal_dmtac_recoverable_typed_receipt_retries_without_sampling() -> None:
    right_hand = np.zeros(6, dtype=np.float32)
    right_hand[0] = 0.1
    action = RobotAction.bimanual_qpos(
        left_arm=np.zeros(6),
        right_arm=np.zeros(6),
        right_hand=right_hand,
        metadata={
            "hand_action_specs_by_side": {
                "right": HandActionSpec.dmtac_gripper("right").to_metadata()
            }
        },
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedCustomAction(step_id="collect", action=action),)
        ),
        trajectory_groups={},
        config=RuleBasedRealTeleopConfig(action_hz=10.0, gripper_sides=("right",)),
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    recorder.hand_state_poller = _HandPoller()
    session, _clutch, _ui = _bind(extension, recorder, hand_sides=("right",))
    first = session.command_override_provider(SimpleNamespace(), 1000.0).command
    sequence = first.metadata["rule_based_sequence_index"]

    extension.after_session_step(
        step_result=SimpleNamespace(
            command_dispatched=True,
            dispatch_receipts={
                "left_arm": True,
                "right_arm": True,
                "right_hand": SimpleNamespace(
                    accepted=False,
                    status="hardware_fault",
                ),
            },
            state=SimpleNamespace(value="RUNNING"),
            faults=(),
        ),
        now=1000.0,
    )

    assert extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True), now=1000.0
    ).allowed is False
    retry = session.command_override_provider(SimpleNamespace(), 1000.2).command
    assert retry.metadata["rule_based_sequence_index"] == sequence
    assert extension.status().aborted is False


def test_session_fault_aborts_rule_and_establishes_hold() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(
                        left_arm=np.zeros(6), right_arm=np.zeros(6)
                    ),
                ),
            )
        ),
        trajectory_groups={},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, clutch, _ui = _bind(extension, recorder)
    assert session.command_override_provider(SimpleNamespace(), 3.0).command is not None
    extension.after_session_step(
        step_result=SimpleNamespace(
            command_dispatched=False,
            state=SimpleNamespace(value="FAULT"),
            faults=(SimpleNamespace(message="guard rejected"),),
        ),
        now=3.0,
    )
    assert extension.status().aborted is True
    assert extension.status().held is True
    assert session.hold_reasons == ["session rejected the active rule frame: guard rejected"]
    assert clutch.reasons == ["rule_based_runtime_hold"]


def test_raised_session_dispatch_error_also_aborts_rule_state() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch, _ui = _bind(extension, recorder)
    assert session.command_override_provider(SimpleNamespace(), 4.0).command is not None

    extension.after_session_error(error=RuntimeError("rig dispatch failed"), now=4.0)

    assert extension.status().aborted is True
    assert extension.status().held is True
    assert "rig dispatch failed" in session.hold_reasons[0]
