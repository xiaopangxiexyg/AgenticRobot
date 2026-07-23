from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace
from unittest.mock import call, patch

import numpy as np
import pytest

from dataset_contracts.pose_frames import PoseFrameContract
from dataset_contracts.pose_training import PoseTrainingDataContract
from robot_motion.rm65 import RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S
from teleop_runtime.collection import run_collect
from teleop_runtime.collection.writer_v21 import LeRobotV21WriterConfig
from teleop_runtime.backends.arms import (
    RM65ArmBackend,
    RM65ArmBackendConfig,
    RM65ImpedanceArmBackend,
)
from teleop_runtime.debug import run_vive_rm65_calibration_debug as calibration_debug
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop
from teleop_runtime.debug import run_vive_sapien_virtual as virtual_teleop
from teleop_runtime.defaults import (
    DEFAULT_IMPEDANCE_TELEOP_CONTROL_HZ,
    DEFAULT_RM65_ARM_CONTROL_HZ,
    DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
    DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    DEFAULT_TELEOP_SAMPLE_HZ,
)
from teleop_runtime.deploy import run_policy
from teleop_runtime.schemas import EffectorCommand, TeleopCommand


def _parse_collect_args(argv: list[str]):
    with patch.object(sys, "argv", ["prog", *argv]):
        return run_collect.parse_args()


def _parse_deploy_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    return parser.parse_args(argv)


def _parse_real_teleop_args(argv: list[str]):
    with patch.object(sys, "argv", ["prog", "--no-device-preflight", *argv]):
        return real_teleop.parse_args()


def _parse_calibration_debug_args(argv: list[str]):
    with patch.object(sys, "argv", ["prog", *argv]):
        return calibration_debug.parse_args()


def _parse_virtual_teleop_args(argv: list[str]):
    with patch.object(sys, "argv", ["prog", *argv]):
        return virtual_teleop.parse_args()


def test_collect_defaults_to_15hz_for_bimanual() -> None:
    args = _parse_collect_args(["--active-sides", "both", "--no-camera", "--no-video"])

    assert args.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.position_reference_hz == DEFAULT_TELEOP_POSITION_REFERENCE_HZ
    assert args.dual_arm_ik_hz == DEFAULT_TELEOP_POSITION_REFERENCE_HZ
    assert args.nominal_fps == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.arm_control_hz == DEFAULT_RM65_POSITION_ARM_CONTROL_HZ
    assert args.tracker_auto_hold_sec == pytest.approx(0.1)
    assert args.tracker_manual_hold_sec == pytest.approx(1.0)


def test_vive_clock_sync_cli_defaults_to_new_protocol_and_allows_legacy() -> None:
    default_collect = _parse_collect_args(
        ["--active-sides", "right", "--no-camera", "--no-video"]
    )
    default_real = _parse_real_teleop_args([])
    collect = _parse_collect_args(
        [
            "--active-sides",
            "right",
            "--no-camera",
            "--no-video",
            "--vive-clock-sync-port",
            "10001",
            "--vive-clock-sync-expected-sender-host",
            "192.168.0.10",
            "--vive-clock-sync-required",
        ]
    )
    real = _parse_real_teleop_args(
        [
            "--vive-clock-sync-port",
            "10001",
            "--vive-clock-sync-required",
        ]
    )

    assert collect.vive_clock_sync_port == 10001
    assert collect.vive_clock_sync_required is True
    assert collect.vive_clock_sync_expected_sender_host == "192.168.0.10"
    assert real.vive_clock_sync_port == 10001
    assert real.vive_clock_sync_required is True
    assert default_collect.vive_clock_sync_port == 10001
    assert default_collect.vive_clock_sync_required is False
    assert default_real.vive_clock_sync_port == 10001
    assert default_real.vive_clock_sync_required is False

    legacy_collect = _parse_collect_args(
        [
            "--active-sides",
            "right",
            "--no-camera",
            "--no-video",
            "--vive-clock-sync-port",
            "0",
        ]
    )
    legacy_real = _parse_real_teleop_args(["--vive-clock-sync-port", "0"])
    assert legacy_collect.vive_clock_sync_port == 0
    assert legacy_real.vive_clock_sync_port == 0


def test_vive_debug_receivers_default_to_new_protocol() -> None:
    calibration = _parse_calibration_debug_args([])
    virtual = _parse_virtual_teleop_args([])

    assert calibration.vive_clock_sync_port == 10001
    assert virtual.vive_clock_sync_port == 10001


def test_real_teleop_position_command_step_envelope_cli() -> None:
    args = _parse_real_teleop_args(
        [
            "--arm-execution-profile",
            "position",
            "--position-command-planner",
            "acceleration_limited",
            "--position-max-command-step-deg",
            "0.675",
        ]
    )

    assert args.position_max_command_step_deg == pytest.approx(0.675)
    assert args.position_target_velocity_filter_tau_sec == pytest.approx(0.03)


def test_real_teleop_position_target_velocity_filter_can_be_bypassed() -> None:
    args = _parse_real_teleop_args(
        [
            "--arm-execution-profile",
            "position",
            "--position-command-planner",
            "acceleration_limited",
            "--position-target-velocity-filter-tau-sec",
            "0",
        ]
    )

    assert args.position_target_velocity_filter_tau_sec == pytest.approx(0.0)


def test_real_teleop_position_command_step_envelope_rejects_impedance() -> None:
    with pytest.raises(SystemExit):
        _parse_real_teleop_args(
            [
                "--arm-execution-profile",
                "impedance_40hz",
                "--position-max-command-step-deg",
                "0.675",
            ]
        )


def test_collect_defaults_to_15hz_for_single_arm() -> None:
    args = _parse_collect_args(["--active-sides", "right", "--no-camera", "--no-video"])

    assert args.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.position_reference_hz == DEFAULT_TELEOP_POSITION_REFERENCE_HZ
    assert args.dual_arm_ik_hz == DEFAULT_TELEOP_POSITION_REFERENCE_HZ
    assert args.nominal_fps == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.arm_control_hz == DEFAULT_RM65_POSITION_ARM_CONTROL_HZ


def test_collect_nominal_fps_follows_explicit_hz_when_omitted() -> None:
    args = _parse_collect_args(["--active-sides", "right", "--no-camera", "--no-video", "--hz", "12"])

    assert args.hz == 12.0
    assert args.nominal_fps == 12.0


def test_lerobot_writer_default_nominal_fps_is_15hz(tmp_path) -> None:
    config = LeRobotV21WriterConfig(root_dir=tmp_path)

    assert config.nominal_fps == DEFAULT_TELEOP_SAMPLE_HZ


def test_dual_rm65_collection_defaults_to_workspace_near_boundary_desk() -> None:
    args = _parse_collect_args(
        ["--active-sides", "both", "--no-camera", "--no-video"]
    )

    assert args.pose_frame_contract is not None
    assert args.pose_frame_contract.pose_storage_frame == "desk_absolute"
    assert (
        args.pose_frame_contract.source.source_id
        == "rm65_real_dual_arm_workspace_near_boundary_v1"
    )
    assert args.policy_io_layout == "auto"
    assert args.pose_frame_contract_source == "default_rm65_dual_arm_workspace_near_boundary_v1"


def test_collection_can_explicitly_disable_default_pose_contract() -> None:
    args = _parse_collect_args(
        [
            "--active-sides",
            "both",
            "--no-camera",
            "--no-video",
            "--no-pose-frame-contract",
        ]
    )

    assert args.pose_frame_contract is None
    assert args.policy_io_layout == ""
    assert args.pose_frame_contract_source == "disabled"


def test_real_teleop_custom_station_geometry_requires_explicit_pose_choice() -> None:
    with pytest.raises(SystemExit):
        _parse_real_teleop_args(["--left-right-base-distance", "0.6"])

    args = _parse_real_teleop_args(
        ["--left-right-base-distance", "0.6", "--no-pose-frame-contract"]
    )
    assert args.pose_frame_contract is None


def test_single_rm65_collection_does_not_reuse_dual_station_extrinsics() -> None:
    args = _parse_collect_args(
        ["--active-sides", "right", "--no-camera", "--no-video"]
    )

    assert args.pose_frame_contract is None


def test_collection_explicit_pose_contract_is_loaded_before_runtime(tmp_path) -> None:
    contract = PoseFrameContract.legacy_v22_raw(active_sides=("left", "right"))
    path = tmp_path / "pose_contract.json"
    path.write_text(
        json.dumps({"pose_frame_contract": contract.to_dict()}),
        encoding="utf-8",
    )
    args = _parse_collect_args(
        [
            "--active-sides",
            "both",
            "--no-camera",
            "--no-video",
            "--policy-io-layout",
            "auto",
            "--pose-frame-contract",
            str(path),
        ]
    )
    assert args.pose_frame_contract.to_dict() == contract.to_dict()
    real_args = _parse_real_teleop_args(
        [
            "--policy-io-layout",
            "auto",
            "--pose-frame-contract",
            str(path),
        ]
    )
    assert real_args.pose_frame_contract.to_dict() == contract.to_dict()


def test_native_v23_deploy_defaults_to_rm65_workspace_near_boundary_runtime_contract() -> None:
    artifact = SimpleNamespace(pose_training_data=PoseTrainingDataContract.native_v23())

    contract, source = run_policy._resolve_runtime_pose_frame_contract(
        None,
        policy_artifact=artifact,
        policy_metadata={},
    )

    assert contract is not None
    assert contract.pose_storage_frame == "raw_absolute"
    assert contract.source.source_id == "rm65_real_dual_arm_workspace_near_boundary_v1"
    assert source == "default_rm65_dual_arm_workspace_near_boundary_v1"


def test_v22_identity_deploy_does_not_receive_workspace_near_boundary_default() -> None:
    artifact = SimpleNamespace(
        pose_training_data=PoseTrainingDataContract.v22_raw_identity(
            active_sides=("left", "right")
        )
    )

    contract, source = run_policy._resolve_runtime_pose_frame_contract(
        None,
        policy_artifact=artifact,
        policy_metadata={},
    )

    assert contract is None
    assert source == "v22_identity_compatibility"


def test_collection_pose_contract_requires_typed_layout(tmp_path) -> None:
    contract = PoseFrameContract.legacy_v22_raw(active_sides=("left", "right"))
    path = tmp_path / "pose_contract.json"
    path.write_text(json.dumps(contract.to_dict()), encoding="utf-8")
    with pytest.raises(SystemExit):
        _parse_collect_args(
            [
                "--active-sides",
                "both",
                "--no-camera",
                "--no-video",
                "--pose-frame-contract",
                str(path),
            ]
        )


def test_v23_collection_contract_cannot_disable_measured_arm_feedback(tmp_path) -> None:
    contract = PoseFrameContract.legacy_v22_raw(active_sides=("left", "right"))
    path = tmp_path / "pose_contract.json"
    path.write_text(json.dumps(contract.to_dict()), encoding="utf-8")

    with pytest.raises(SystemExit):
        _parse_collect_args(
            [
                "--active-sides",
                "both",
                "--no-camera",
                "--no-video",
                "--policy-io-layout",
                "auto",
                "--pose-frame-contract",
                str(path),
                "--no-record-actual-wrist-camera-pose",
            ]
        )
    with pytest.raises(SystemExit):
        _parse_real_teleop_args(
            [
                "--policy-io-layout",
                "auto",
                "--pose-frame-contract",
                str(path),
                "--no-record-actual-wrist-camera-pose",
            ]
        )


def test_collect_actual_wrist_camera_pose_recording_defaults_on() -> None:
    args = _parse_collect_args(["--active-sides", "right", "--no-camera", "--no-video"])

    assert args.record_actual_wrist_camera_pose is True


def test_collect_actual_wrist_camera_pose_recording_can_be_disabled() -> None:
    args = _parse_collect_args(
        [
            "--active-sides",
            "right",
            "--no-camera",
            "--no-video",
            "--no-record-actual-wrist-camera-pose",
        ]
    )

    assert args.record_actual_wrist_camera_pose is False


def test_real_teleop_actual_wrist_camera_pose_recording_defaults_on() -> None:
    args = _parse_real_teleop_args([])

    assert args.record_actual_wrist_camera_pose is True


def test_real_teleop_actual_wrist_camera_pose_recording_can_be_disabled() -> None:
    args = _parse_real_teleop_args(
        ["--no-pose-frame-contract", "--no-record-actual-wrist-camera-pose"]
    )

    assert args.record_actual_wrist_camera_pose is False


def test_real_teleop_without_collection_does_not_bind_dataset_pose_contract() -> None:
    args = _parse_real_teleop_args(
        ["--no-collection", "--no-record-actual-wrist-camera-pose"]
    )

    assert args.pose_frame_contract is None
    assert args.record_actual_wrist_camera_pose is False


def test_real_teleop_legacy_session_recording_defaults_off() -> None:
    args = _parse_real_teleop_args([])

    assert args.no_record is True


def test_real_teleop_legacy_session_recording_can_be_enabled() -> None:
    args = _parse_real_teleop_args(["--record"])

    assert args.no_record is False


def test_real_teleop_console_heartbeats_and_orientation_debug_are_opt_in() -> None:
    defaults = _parse_real_teleop_args([])
    debug = _parse_real_teleop_args(
        ["--status-sec", "2.5", "--debug-orientation-console"]
    )

    assert defaults.status_sec == 0.0
    assert defaults.debug_orientation_console is False
    assert debug.status_sec == 2.5
    assert debug.debug_orientation_console is True


def test_real_teleop_gripper_health_signature_ignores_live_command_values() -> None:
    first = (
        "left=connected/q6/cmd=0.100/meas=0.200 input=q6 valid=ok dim=2 src=0.10 "
        "target=0.20 filt=0.19 vals=[0.10,0.20] send=sent/1 send_ms=0.4",
    )
    second = (
        "left=connected/q6/cmd=0.900/meas=0.800 input=q6 valid=ok dim=2 src=0.90 "
        "target=0.80 filt=0.81 vals=[0.90,0.80] send=sent/1 send_ms=1.1",
    )
    faulted = (
        "left=disconnected/q6/cmd=-/meas=- input=q6 valid=invalid dim=2 src=- "
        "target=- filt=- vals=- send=failed/0 send_ms=- q6_error=serial lost",
    )

    assert real_teleop._gripper_runtime_health_signature(first) == (
        "left=connected/q6 valid=ok unhealthy=0",
    )
    first_health = real_teleop._gripper_runtime_health_signature(first)
    assert first_health == real_teleop._gripper_runtime_health_signature(second)
    assert real_teleop._gripper_runtime_health_signature(faulted) == (
        "left=disconnected/q6 valid=invalid unhealthy=1",
    )


def test_real_teleop_status_health_signatures_ignore_live_control_values() -> None:
    normal = {
        "send_result_code": 0,
        "background_streaming": {
            "position_command_planner": "acceleration_limited",
            "trajectory_policy_mode": "normal",
            "quick_stop_active": False,
            "last_trajectory_debug": {"trajectory_qd_command_rad_s": [0.1]},
        },
    }
    changed_command = {
        **normal,
        "background_streaming": {
            **normal["background_streaming"],
            "last_trajectory_debug": {"trajectory_qd_command_rad_s": [1.0]},
        },
    }

    assert real_teleop._command_backend_health_signature(
        normal
    ) == real_teleop._command_backend_health_signature(changed_command)
    first_safety = {
        "enabled": True,
        "limited": False,
        "effectors": {
            "left_arm": {
                "command_type": "cartesian_pose",
                "limited": False,
                "raw_translation_step_m": 0.001,
            }
        },
    }
    second_safety = {
        **first_safety,
        "effectors": {
            "left_arm": {
                **first_safety["effectors"]["left_arm"],
                "raw_translation_step_m": 0.02,
            }
        },
    }

    assert real_teleop._temporary_safety_health_signature(
        first_safety
    ) == real_teleop._temporary_safety_health_signature(second_safety)


def test_real_teleop_debug_action_options_parse() -> None:
    args = _parse_real_teleop_args(
        [
            "--debug-action",
            "--debug-action-log-root",
            "teleop_records/custom_action_debug",
            "--debug-action-flush-every",
            "3",
            "--debug-action-queue-size",
            "64",
            "--debug-action-sample-hz",
            "15",
            "--debug-timing-log-sample-hz",
            "5",
            "--no-control-debug-episode-sidecar",
        ]
    )

    assert args.debug_action is True
    assert args.debug_action_log_root == "teleop_records/custom_action_debug"
    assert args.debug_action_flush_every == 3
    assert args.debug_action_queue_size == 64
    assert args.debug_action_sample_hz == 15.0
    assert args.debug_timing_log_sample_hz == 5.0
    assert args.control_debug_episode_sidecar is False


def test_real_teleop_debug_action_can_be_explicitly_disabled() -> None:
    args = _parse_real_teleop_args(["--debug-action", "--no-debug-action"])

    assert args.debug_action is False


def test_real_teleop_arm_singularity_protection_can_be_disabled() -> None:
    default_args = _parse_real_teleop_args([])
    disabled_args = _parse_real_teleop_args(["--no-arm-singularity-protection"])
    reenabled_args = _parse_real_teleop_args(
        ["--no-arm-singularity-protection", "--arm-singularity-protection"]
    )

    assert default_args.arm_singularity_protection is True
    assert disabled_args.arm_singularity_protection is False
    assert reenabled_args.arm_singularity_protection is True


def test_action_debug_log_recorder_writes_without_dataset(tmp_path) -> None:
    recorder = real_teleop.ActionDebugLogRecorder(root_dir=tmp_path, flush_every=1)

    path = recorder.start(now=1.25)
    recorder.record_frame(
        {
            "schema": "dexteleop_action_debug_v1",
            "event": "frame",
            "step_index": 7,
            "jitter_focus_by_side": {"left": {"available": True}},
        }
    )
    stopped = recorder.stop(reason="test", now=2.0, join_timeout_sec=1.0)

    assert stopped == path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["start", "frame", "stop"]
    assert rows[1]["step_index"] == 7
    assert rows[2]["payload"]["frame_count"] == 1


def test_action_debug_jitter_focus_includes_force_admittance_fields() -> None:
    focus = real_teleop._action_debug_jitter_focus(
        {
            "left": {
                "available": True,
                "latest_target_tcp_pose6": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                "backend_state_compact": {
                    "last_impedance_record": {
                        "reference": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                        "offset": [0.0, 0.001, 0.0, 0.0, 0.0, 0.0],
                        "applied_offset": [0.0, 0.001, 0.0, 0.0, 0.0, 0.0],
                        "target_offset": [0.0, 0.002, 0.0, 0.0, 0.0, 0.0],
                        "selected_force_wrench": [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                        "admittance_wrench": [0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                        "offset_step_after_velocity_limit": [
                            0.0,
                            0.0005,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        "applied_offset_source": "final_command_pose",
                        "admittance_offset_antiwindup_applied": True,
                        "linear_velocity_limit_applied": False,
                        "angular_velocity_limit_applied": True,
                        "joint_rate_limited": True,
                    }
                },
            }
        }
    )

    left = focus["left"]
    assert left["reference"] == [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
    assert left["offset"] == [0.0, 0.001, 0.0, 0.0, 0.0, 0.0]
    assert left["applied_offset_source"] == "final_command_pose"
    assert left["admittance_offset_antiwindup_applied"] is True
    assert left["angular_velocity_limit_applied"] is True
    assert left["joint_rate_limited"] is True


def test_action_debug_jitter_focus_includes_position_motion_stream_fields() -> None:
    focus = real_teleop._action_debug_jitter_focus(
        {
            "left": {
                "available": True,
                "backend_state_compact": {
                    "background_streaming": {
                        "enabled": True,
                        "hz": 150.0,
                        "target_sequence": 17,
                        "target_age_sec": 0.012,
                        "segment_duration_sec": 1.0 / 45.0,
                        "actual_hz": 149.8,
                        "last_sent_step_deg": 0.21,
                        "deadline_miss_count": 2,
                        "healthy": True,
                    }
                },
            }
        }
    )

    stream = focus["left"]["position_background_streaming"]
    assert stream["enabled"] is True
    assert stream["hz"] == 150.0
    assert stream["target_sequence"] == 17
    assert stream["segment_duration_sec"] == 1.0 / 45.0
    assert stream["actual_hz"] == 149.8
    assert stream["last_sent_step_deg"] == 0.21
    assert stream["healthy"] is True


def test_real_teleop_canfd_follow_options_parse() -> None:
    args = _parse_real_teleop_args([
        "--arm-canfd-follow",
        "--position-post-ik-qpos-smoothing-alpha",
        "0.8",
        "--arm-follow-qpos-alpha",
        "0.1",
        "--arm-canfd-follow-stream-hz",
        "120",
        "--arm-canfd-follow-warning-ratio",
        "0.85",
        "--arm-canfd-trajectory-mode",
        "1",
        "--arm-canfd-radio",
        "50",
        "--arm-state-read-every",
        "6",
        "--arm-disable-force-admittance",
    ])

    assert args.arm_canfd_follow is True
    assert args.position_post_ik_qpos_smoothing_alpha == 0.8
    assert args.arm_follow_qpos_alpha == 0.1
    assert args.arm_canfd_follow_stream_hz == 120.0
    assert args.arm_canfd_follow_warning_ratio == 0.85
    assert args.arm_canfd_trajectory_mode == 1
    assert args.arm_canfd_radio == 50
    assert args.arm_state_read_every == 6
    assert args.arm_disable_force_admittance is True


def test_real_teleop_canfd_follow_defaults_to_backend_or_profile() -> None:
    args = _parse_real_teleop_args([])

    assert args.arm_canfd_follow is None
    assert args.arm_control_hz is None
    assert args.arm_follow_qpos_alpha is None
    assert args.position_post_ik_qpos_smoothing_alpha == pytest.approx(0.14)
    assert args.position_post_ik_qpos_smoothing_fast_alpha == pytest.approx(0.45)
    assert args.position_post_ik_qpos_smoothing_adaptive is True
    assert args.arm_canfd_follow_stream_hz is None
    assert args.arm_canfd_follow_warning_ratio is None
    assert args.arm_canfd_trajectory_mode is None
    assert args.arm_canfd_radio is None
    assert args.arm_state_read_every is None
    assert args.arm_disable_force_admittance is False
    assert real_teleop._effective_arm_control_hz(args) == DEFAULT_RM65_ARM_CONTROL_HZ
    assert real_teleop._effective_arm_canfd_follow(args) is False


def test_real_teleop_position_defaults_to_formal_120hz_profile() -> None:
    args = _parse_real_teleop_args(["--arm-control-mode", "position"])

    assert args.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.position_reference_hz == DEFAULT_TELEOP_POSITION_REFERENCE_HZ
    assert real_teleop._effective_outer_control_hz(args) == DEFAULT_TELEOP_POSITION_REFERENCE_HZ
    assert real_teleop._effective_dual_arm_ik_hz(args) == DEFAULT_TELEOP_POSITION_REFERENCE_HZ
    assert args.arm_control_hz == 120.0
    assert args.arm_canfd_follow is True
    assert args.position_control_profile_resolved == "rm65_bimanual_position_v1"
    assert args.position_control_profile_metadata["source_shaping_applied"] is True
    assert args.position_reference_max_age_sec == pytest.approx(0.2)
    assert args.arm_canfd_trajectory_mode == 0
    assert args.arm_canfd_radio == 0
    assert args.position_post_ik_qpos_smoothing_alpha == pytest.approx(0.14)
    assert args.arm_execution_config.position_post_ik_qpos_smoothing_alpha == pytest.approx(
        0.14
    )
    assert real_teleop._effective_position_post_ik_qpos_smoothing_alpha(
        args
    ) == pytest.approx(0.14)
    assert args.position_post_ik_qpos_smoothing_fast_alpha == pytest.approx(0.45)
    assert args.position_post_ik_qpos_smoothing_adaptive is True
    assert (
        args.position_post_ik_qpos_smoothing_fast_alpha
        >= real_teleop._effective_position_post_ik_qpos_smoothing_alpha(args)
    )
    assert real_teleop._effective_arm_control_hz(args) == DEFAULT_RM65_POSITION_ARM_CONTROL_HZ
    assert real_teleop._effective_arm_canfd_follow(args) is True
    canfd_config = real_teleop._canfd_follow_runtime_config(args)
    assert canfd_config["position_background_streaming_enabled"] is True
    assert canfd_config["position_background_streaming_hz"] == DEFAULT_RM65_POSITION_ARM_CONTROL_HZ
    assert canfd_config["canfd_follow_warning_ratio"] == 0.9
    assert canfd_config["canfd_trajectory_mode"] == 0
    assert canfd_config["canfd_radio"] == 0
    temp_safety = real_teleop._effective_dual_arm_temporary_safety(
        args,
        position_canfd_passthrough=bool(
            canfd_config["position_background_streaming_enabled"]
        ),
    )
    assert temp_safety["requested"] is True
    assert temp_safety["enabled"] is False
    assert (
        temp_safety["disabled_reason"]
        == "position_canfd_passthrough_interpolation_only"
    )


def test_position_canfd_keeps_live_retargeter_stability_enabled() -> None:
    args = _parse_real_teleop_args(["--arm-control-mode", "position"])

    switches = real_teleop._position_retargeter_stability_switches(
        args,
        position_canfd_passthrough=True,
    )

    assert switches == {
        "target_pose_smoothing_enabled": True,
        "local_step_candidate_selection_enabled": True,
        "inactive_side_hold_enabled": True,
        "inactive_side_hold_execution_cap_enabled": True,
        "hold_disabled": False,
    }


def test_position_safe_debug_disable_hold_only_disables_hold_features() -> None:
    args = _parse_real_teleop_args(
        ["--arm-control-mode", "position", "--safe-debug", "--disable-hold"]
    )

    switches = real_teleop._position_retargeter_stability_switches(
        args,
        position_canfd_passthrough=True,
    )

    assert switches["target_pose_smoothing_enabled"] is True
    assert switches["local_step_candidate_selection_enabled"] is True
    assert switches["inactive_side_hold_enabled"] is False
    assert switches["inactive_side_hold_execution_cap_enabled"] is False
    assert switches["hold_disabled"] is True


def test_real_teleop_impedance_decouples_15hz_collection_from_30hz_outer_defaults() -> None:
    args = _parse_real_teleop_args([])

    assert args.arm_control_mode == "impedance"
    assert args.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert real_teleop._effective_outer_control_hz(args) == DEFAULT_IMPEDANCE_TELEOP_CONTROL_HZ
    assert real_teleop._effective_dual_arm_ik_hz(args) == DEFAULT_IMPEDANCE_TELEOP_CONTROL_HZ
    assert real_teleop._effective_arm_control_hz(args) == DEFAULT_RM65_ARM_CONTROL_HZ
    assert args.arm_step_dt_max_scale == 3.0


def test_real_teleop_impedance_control_rate_is_independent_from_dataset_hz() -> None:
    explicit = _parse_real_teleop_args(["--impedance-teleop-control-hz", "36"])
    legacy = _parse_real_teleop_args(["--hz", "20"])

    assert explicit.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert real_teleop._effective_outer_control_hz(explicit) == 36.0
    assert real_teleop._effective_dual_arm_ik_hz(explicit) == 36.0
    assert legacy.hz == 20.0
    assert real_teleop._effective_outer_control_hz(legacy) == DEFAULT_IMPEDANCE_TELEOP_CONTROL_HZ


def test_real_teleop_position_dt_scaling_uses_reference_rate_not_sample_rate() -> None:
    args = _parse_real_teleop_args(["--arm-control-mode", "position"])

    expected = max(
        3.0,
        1.1
        * DEFAULT_RM65_POSITION_ARM_CONTROL_HZ
        / DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    )
    assert args.arm_step_dt_max_scale == expected


def test_real_teleop_impedance_uses_pre_ik_safety_instead_of_temporary_output_limiter() -> None:
    args = _parse_real_teleop_args([])

    temp_safety = real_teleop._effective_dual_arm_temporary_safety(
        args,
        position_canfd_passthrough=False,
    )

    assert temp_safety["requested"] is True
    assert temp_safety["enabled"] is False
    assert temp_safety["disabled_reason"] == "impedance_uses_pre_ik_coupled_safety_bounds"


def test_real_teleop_position_can_override_follow_and_control_hz() -> None:
    args = _parse_real_teleop_args(
        [
            "--arm-control-mode",
            "position",
            "--no-arm-canfd-follow",
            "--arm-control-hz",
            "80",
        ]
    )

    assert real_teleop._effective_arm_control_hz(args) == 80.0
    assert real_teleop._effective_arm_canfd_follow(args) is False
    canfd_config = real_teleop._canfd_follow_runtime_config(args)
    assert canfd_config["position_background_streaming_enabled"] is False
    assert canfd_config["position_background_streaming_hz"] == 80.0


def test_real_teleop_position_can_disable_follow_streaming_with_zero_hz() -> None:
    args = _parse_real_teleop_args(
        [
            "--arm-control-mode",
            "position",
            "--arm-control-hz",
            "150",
            "--arm-canfd-follow-stream-hz",
            "0",
        ]
    )

    canfd_config = real_teleop._canfd_follow_runtime_config(args)
    assert canfd_config["position_background_streaming_enabled"] is False
    assert canfd_config["position_background_streaming_hz"] == 150.0


def test_real_teleop_disable_force_admittance_keeps_impedance_follow_disabled() -> None:
    args = _parse_real_teleop_args(["--arm-disable-force-admittance"])

    assert args.arm_canfd_follow is None
    assert real_teleop._effective_arm_control_hz(args) == DEFAULT_RM65_ARM_CONTROL_HZ
    assert real_teleop._effective_arm_canfd_follow(args) is False


def test_real_teleop_step_limit_sync_prefers_explicit_backend_steps_over_profile() -> None:
    calls: list[tuple[str, object, object]] = []

    class _Retargeter:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                enforce_realistic_joint_step=True,
                left_execution_max_joint_step_rad=np.ones(6),
                right_execution_max_joint_step_rad=np.ones(6),
                joint_velocity_profile=object(),
            )

        def set_execution_max_joint_step(self, *, left_step_rad, right_step_rad) -> None:
            calls.append(("explicit", np.asarray(left_step_rad), np.asarray(right_step_rad)))

        def set_execution_dt_sec(self, dt_sec: float) -> None:
            calls.append(("profile", float(dt_sec), None))

    left_config = SimpleNamespace(
        control_dt=0.025,
        action_hz=15.0,
        actual_send_dt_stall_reset_sec=0.5,
        actual_send_dt_alpha=1.0,
        actual_send_dt_min_scale=0.1,
        actual_send_dt_max_scale=10.0,
        max_angvel=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    )
    right_config = SimpleNamespace(
        **{
            **left_config.__dict__,
            "max_angvel": (6.0, 5.0, 4.0, 3.0, 2.0, 1.0),
        }
    )
    components = SimpleNamespace(
        dual_arm_retargeter=_Retargeter(),
        left_command_backend=SimpleNamespace(config=left_config),
        right_command_backend=SimpleNamespace(config=right_config),
        arm_loop_step_dt_tracker={"last_loop_time": 10.0, "smoothed_loop_dt": 0.025},
    )

    real_teleop._sync_real_execution_step_limits(components, now=10.05)

    assert calls and calls[0][0] == "explicit"
    assert np.allclose(calls[0][1], np.asarray(left_config.max_angvel) * 0.05)
    assert np.allclose(calls[0][2], np.asarray(right_config.max_angvel) * 0.05)


def test_real_teleop_impedance_step_limit_sync_never_catches_up_after_slow_frame() -> None:
    calls: list[tuple[str, object, object]] = []

    class _Retargeter:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                enforce_realistic_joint_step=True,
                left_execution_max_joint_step_rad=np.ones(6),
                right_execution_max_joint_step_rad=np.ones(6),
                joint_velocity_profile=object(),
            )

        def set_execution_max_joint_step(self, *, left_step_rad, right_step_rad) -> None:
            calls.append(("explicit", np.asarray(left_step_rad), np.asarray(right_step_rad)))

        def set_execution_dt_sec(self, dt_sec: float) -> None:
            calls.append(("profile", float(dt_sec), None))

    left_config = SimpleNamespace(
        control_dt=0.025,
        action_hz=15.0,
        actual_send_dt_stall_reset_sec=0.5,
        actual_send_dt_alpha=1.0,
        actual_send_dt_min_scale=1.0,
        actual_send_dt_max_scale=1.0,
        scale_outer_execution_step_by_actual_dt=False,
        max_angvel=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    )
    right_config = SimpleNamespace(
        **{
            **left_config.__dict__,
            "max_angvel": (6.0, 5.0, 4.0, 3.0, 2.0, 1.0),
        }
    )
    components = SimpleNamespace(
        dual_arm_retargeter=_Retargeter(),
        left_command_backend=SimpleNamespace(config=left_config),
        right_command_backend=SimpleNamespace(config=right_config),
        arm_loop_step_dt_tracker={"last_loop_time": 10.0, "smoothed_loop_dt": 0.025},
    )

    real_teleop._sync_real_execution_step_limits(components, now=10.20)

    assert calls and calls[0][0] == "explicit"
    assert np.allclose(calls[0][1], np.asarray(left_config.max_angvel) / 15.0)
    assert np.allclose(calls[0][2], np.asarray(right_config.max_angvel) / 15.0)
    assert components.arm_loop_step_dt_tracker["smoothed_loop_dt"] == pytest.approx(1.0 / 15.0)


def test_real_teleop_wrist_depth_resolution_also_sets_wrist_rgb_resolution(tmp_path) -> None:
    camera_config = tmp_path / "realsense.json"
    camera_config.write_text(
        json.dumps(
            {
                "front": {"serial": "F", "width": 640, "height": 480, "fps": 15},
                "left_wrist": {"serial": "L", "width": 640, "height": 480, "fps": 15},
                "right_wrist": {"serial": "R", "width": 640, "height": 480, "fps": 15},
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        no_camera=False,
        camera_config=str(camera_config),
        vis_cam=False,
        wrist_depth_mask=True,
        wrist_depth_raw=False,
        wrist_cam_width=424,
        wrist_cam_height=240,
        wrist_depth_stream_fps=15,
        wrist_depth_align_to_color=False,
    )

    sources = real_teleop._build_camera_sources(args)

    front_config = sources["front"].config
    left_config = sources["left_wrist"].config
    right_config = sources["right_wrist"].config
    assert front_config.enable_depth is False
    assert front_config.width == 640
    assert front_config.height == 480
    assert left_config.width == 424
    assert left_config.height == 240
    assert left_config.depth_width == 424
    assert left_config.depth_height == 240
    assert left_config.depth_fps == 15
    assert left_config.align_depth_to_color is False
    assert left_config.metadata["wrist_cam_original_rgb_width"] == 640
    assert left_config.metadata["wrist_cam_original_rgb_height"] == 480
    assert left_config.metadata["wrist_cam_matched_rgb_depth_resolution"] is True
    assert right_config.width == 424
    assert right_config.height == 240
    assert right_config.depth_width == 424
    assert right_config.depth_height == 240


def test_real_teleop_wrist_cam_resolution_args_parse_to_wrist_cam_fields() -> None:
    args = _parse_real_teleop_args(
        [
            "--wrist-depth-mask",
            "--wrist-cam-width",
            "480",
            "--wrist-cam-height",
            "270",
        ]
    )

    assert args.wrist_cam_width == 480
    assert args.wrist_cam_height == 270


def test_real_teleop_legacy_wrist_depth_resolution_args_remain_aliases() -> None:
    args = _parse_real_teleop_args(
        [
            "--wrist-depth-mask",
            "--wrist-depth-width",
            "480",
            "--wrist-depth-height",
            "270",
        ]
    )

    assert args.wrist_cam_width == 480
    assert args.wrist_cam_height == 270


def test_real_teleop_camera_sources_start_staggered() -> None:
    started: list[str] = []

    class _Source:
        def __init__(self, name: str) -> None:
            self.name = name

        def start(self) -> None:
            started.append(self.name)

    sources = {
        "left_wrist": _Source("left_wrist"),
        "right_wrist": _Source("right_wrist"),
    }

    with patch.object(real_teleop.time, "sleep") as sleep_mock:
        real_teleop._start_camera_sources(sources, stagger_sec=0.5)

    assert started == ["left_wrist", "right_wrist"]
    sleep_mock.assert_has_calls([call(0.5)])


def test_deploy_default_fallback_action_hz_is_15hz_for_bimanual() -> None:
    args = _parse_deploy_args(["--active-sides", "both"])
    active_sides = run_policy.validate_args(args)
    hz, info = run_policy.resolve_effective_action_hz(args, policy=None)

    assert active_sides == ("left", "right")
    assert args.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.position_reference_hz == 60.0
    assert args.arm_control_hz == 120.0
    assert args.position_chunk_lookahead is True
    assert args.position_ik_solve_min_iterations == 2
    assert args.position_ik_solve_max_iterations == 4
    assert args.position_ik_damping == pytest.approx(0.03)
    assert args.position_ik_adaptive_iterations is True
    assert args.position_control_profile_resolved == "rm65_bimanual_position_v1"
    assert args.position_control_profile_metadata["source_shaping_applied"] is False
    assert args.position_control_profile_metadata["source_shaping_reapplied"] is False
    assert args.position_ik_iteration_reserve_ms == pytest.approx(5.0)
    assert hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert info["source"] == "cli.hz"


def test_deploy_position_chunk_lookahead_has_explicit_rollback_switch() -> None:
    args = _parse_deploy_args(
        ["--active-sides", "both", "--no-position-chunk-lookahead"]
    )
    run_policy.validate_args(args)

    assert args.position_chunk_lookahead is False


def test_deploy_position_ik_iteration_range_is_validated() -> None:
    args = _parse_deploy_args(
        [
            "--active-sides",
            "both",
            "--position-ik-solve-min-iterations",
            "3",
            "--position-ik-solve-max-iterations",
            "2",
        ]
    )

    with pytest.raises(ValueError, match="solve-min-iterations"):
        run_policy.validate_args(args)


def test_deploy_default_fallback_action_hz_is_15hz_for_single_arm() -> None:
    args = _parse_deploy_args(["--active-sides", "right"])
    active_sides = run_policy.validate_args(args)
    hz, info = run_policy.resolve_effective_action_hz(args, policy=None)

    assert active_sides == ("right",)
    assert args.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.position_reference_hz == 60.0
    assert args.arm_control_hz == 120.0
    assert hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert info["source"] == "cli.hz"


def test_deploy_dry_run_arm_backend_uses_position_control_default() -> None:
    args = _parse_deploy_args(["--active-sides", "right"])
    active_sides = run_policy.validate_args(args)

    backends = run_policy.build_arm_backends(args, active_sides, control_hz=args.arm_control_hz)

    assert args.arm_control_mode == "position"
    assert backends["right"].config.control_dt == 1.0 / 60.0
    assert backends["right"].config.background_streaming_hz == 120.0
    assert backends["right"].config.canfd_follow is True


def test_collect_position_backend_uses_reference_dt_and_stream_control_hz() -> None:
    args = SimpleNamespace(
        hz=DEFAULT_TELEOP_SAMPLE_HZ,
        position_reference_hz=DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
        arm_control_hz=DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
        right_arm_ip="192.0.2.20",
        arm_max_angvel=(0.35, 0.35, 0.35, 0.5, 0.5, 0.5),
        with_tactile=False,
        hand_tactile_regions="",
        hand_backend=run_collect.HAND_BACKEND_NONE,
        right_tool_profile="",
        rm65_ik_framework="robot_motion",
    )

    with patch.object(run_collect, "require_asset_path", return_value="rm65_arm.urdf"), patch.object(
        run_collect,
        "RM65ArmBackend",
        side_effect=lambda config: SimpleNamespace(config=config),
    ):
        all_backends, _rig_backends = run_collect._build_backends(args, ("right",))

    backend = all_backends["right_arm"]
    assert backend.config.control_dt == 1.0 / DEFAULT_TELEOP_POSITION_REFERENCE_HZ
    assert backend.config.background_streaming_hz == DEFAULT_RM65_POSITION_ARM_CONTROL_HZ


def test_deploy_impedance_arm_backend_uses_40hz_no_follow_default() -> None:
    args = _parse_deploy_args(["--active-sides", "right", "--arm-control-mode", "impedance"])
    active_sides = run_policy.validate_args(args)

    backends = run_policy.build_arm_backends(args, active_sides, control_hz=args.arm_control_hz)

    assert args.arm_control_hz == DEFAULT_RM65_ARM_CONTROL_HZ
    assert args.arm_canfd_follow is False
    assert isinstance(backends["right"], RM65ImpedanceArmBackend)
    assert backends["right"].config.control_dt == 1.0 / DEFAULT_RM65_ARM_CONTROL_HZ
    assert backends["right"].config.impedance_overrides["canfd_follow"] is False
    overrides = backends["right"].config.impedance_overrides
    assert overrides["joint_rate_limit_use_nominal_dt"] is True
    assert overrides["admittance_use_nominal_dt"] is True
    assert overrides["joint_operating_margin_deg"] == 3.0
    assert overrides["ik_hard_deadline_enabled"] is True
    assert overrides["deployment_dual_rate_ik_enabled"] is True
    assert overrides["deployment_full_ik_hz"] == 15.0
    assert overrides["deployment_full_ik_deadline_ms"] == pytest.approx(1000.0 / 15.0)
    assert overrides["deployment_local_ik_deadline_ms"] == 6.0
    assert overrides["deployment_full_ik_preempts_local"] is True
    assert overrides["ik_resolver_candidate_provider_names"] == ["rm65_pinocchio"]
    assert overrides["ik_resolver_local_step_max_position_worsen_m"] == 0.003
    assert overrides["ik_resolver_local_step_max_orientation_worsen_deg"] == 0.5
    assert overrides["ik_resolver_local_step_dynamic_worsen_enabled"] is False
    assert overrides["direct_joint_hint_max_linear_offset_m"] == 0.003
    assert overrides["direct_joint_hint_max_angular_offset_rad"] == pytest.approx(np.deg2rad(0.5))


def test_deploy_named_execution_profiles_use_shared_real_config_resolver() -> None:
    impedance_argv = [
        "--active-sides",
        "right",
        "--arm-execution-profile",
        "impedance_40hz",
    ]
    impedance_args = _parse_deploy_args(impedance_argv)
    run_policy.validate_args(impedance_args, raw_argv=impedance_argv)
    assert impedance_args.arm_control_mode == "impedance"
    assert impedance_args.arm_control_hz == 40.0
    assert impedance_args.arm_canfd_follow is False
    assert (
        impedance_args._real_arm_execution_config.resolved_profile
        == "impedance_40hz"
    )

    position_argv = [
        "--active-sides",
        "both",
        "--arm-execution-profile",
        "position",
    ]
    position_args = _parse_deploy_args(position_argv)
    run_policy.validate_args(position_args, raw_argv=position_argv)
    assert position_args.arm_control_mode == "position"
    assert position_args.position_reference_hz == 60.0
    assert position_args.arm_control_hz == 120.0
    assert position_args.arm_canfd_follow is True
    assert (
        position_args._real_arm_execution_config.resolved_profile
        == "position"
    )


def test_deploy_named_profile_rejects_conflicting_explicit_mode() -> None:
    argv = [
        "--arm-execution-profile",
        "impedance_40hz",
        "--arm-control-mode",
        "position",
    ]
    args = _parse_deploy_args(argv)
    with pytest.raises(ValueError, match="requires --arm-control-mode impedance"):
        run_policy.validate_args(args, raw_argv=argv)


def test_deploy_position_can_disable_canfd_follow() -> None:
    args = _parse_deploy_args(
        [
            "--active-sides",
            "right",
            "--arm-control-mode",
            "position",
            "--no-arm-canfd-follow",
        ]
    )
    active_sides = run_policy.validate_args(args)

    backends = run_policy.build_arm_backends(args, active_sides, control_hz=args.arm_control_hz)

    assert args.arm_control_hz == 120.0
    assert args.arm_canfd_follow is False
    assert backends["right"].config.canfd_follow is False
    assert backends["right"].config.background_streaming_enabled is False


def test_deploy_arm_limits_match_real_teleop_defaults() -> None:
    args = _parse_deploy_args(["--active-sides", "right"])
    active_sides = run_policy.validate_args(args)

    backends = run_policy.build_arm_backends(args, active_sides, control_hz=args.arm_control_hz)

    assert args.arm_max_angvel == RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S
    assert args.arm_step_dt_scaling is True
    assert args.arm_step_dt_alpha == 0.25
    assert args.arm_step_dt_stall_reset_sec == 0.25
    assert np.isclose(
        args.arm_step_dt_max_scale,
        max(
            run_policy.DEPLOY_ARM_STEP_DT_MAX_SCALE,
            120.0
            / 60.0
            * 1.1,
        ),
    )
    assert args.arm_step_dt_max_scale_auto is True
    assert np.isclose(args.max_arm_joint_step_rad, np.deg2rad(6.0))
    assert (
        backends["right"].config.max_angvel
        == RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S
    )
    assert backends["right"].config.position_command_planner == "acceleration_limited"
    assert backends["right"].config.position_target_velocity_filter_tau_sec == 0.03
    assert backends["right"].config.position_max_command_step_deg == 0.675
    assert backends["right"].config.arm_follow_qpos_alpha == 0.9
    assert backends["right"].config.use_actual_send_dt_scaling is True
    assert backends["right"].config.actual_send_dt_max_scale == args.arm_step_dt_max_scale


def test_deploy_arm_step_dt_max_scale_auto_follows_position_reference_rate() -> None:
    args = _parse_deploy_args(
        [
            "--active-sides",
            "right",
            "--position-reference-hz",
            "25",
            "--arm-control-hz",
            "50",
        ]
    )
    active_sides = run_policy.validate_args(args)

    backends = run_policy.build_arm_backends(args, active_sides, control_hz=args.arm_control_hz)

    assert args.arm_step_dt_max_scale == run_policy.DEPLOY_ARM_STEP_DT_MAX_SCALE
    assert args.arm_step_dt_max_scale_auto is True
    assert backends["right"].config.actual_send_dt_max_scale == args.arm_step_dt_max_scale


def test_deploy_arm_step_dt_max_scale_keeps_explicit_value() -> None:
    args = _parse_deploy_args(
        [
            "--active-sides",
            "right",
            "--arm-step-dt-max-scale",
            "2.5",
        ]
    )
    active_sides = run_policy.validate_args(args)

    run_policy.build_arm_backends(args, active_sides, control_hz=args.arm_control_hz)

    assert args.arm_step_dt_max_scale == 2.5
    assert args.arm_step_dt_max_scale_auto is False


def test_deploy_policy_arm_command_limit_uses_action_dt() -> None:
    args = _parse_deploy_args(["--active-sides", "right", "--hz", "15"])
    active_sides = run_policy.validate_args(args)
    run_policy.apply_arm_step_dt_defaults(args, control_hz=40.0, action_hz=15.0)
    command = TeleopCommand(
        commands={
            "right_arm": EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="cartesian_pose",
                value=np.eye(4),
                metadata={},
            )
        }
    )

    run_policy.annotate_deploy_arm_command_limits(
        command,
        args=args,
        active_sides=active_sides,
        control_hz=40.0,
        action_hz=15.0,
    )

    expected = np.asarray(RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S) / 15.0
    actual = command.commands["right_arm"].metadata["max_joint_step_rad"]
    assert np.allclose(actual, expected)
    assert command.commands["right_arm"].metadata["deploy_action_step_dt_sec"] == 1.0 / 15.0
    assert command.commands["right_arm"].metadata["deploy_backend_control_dt_sec"] == (
        1.0 / 60.0
    )


def test_rm65_backend_uses_command_max_joint_step_metadata() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            connect_hardware=False,
            control_dt=0.01,
            max_angvel=(0.1,) * 6,
            dry_run_fast=False,
            initial_qpos=(0.0,) * 6,
        )
    )
    command = EffectorCommand(
        effector_name="right_arm",
        side="right",
        command_type="joint_vector",
        value=np.ones(6, dtype=float),
        metadata={"max_joint_step_rad": np.full(6, 0.5, dtype=float)},
    )

    backend.send_command(command)

    state = backend.get_state()
    assert np.allclose(state["current_qpos"], np.full(6, 0.5, dtype=float))


def test_deploy_arm_backend_accepts_qpos_init_singularity_slowdown() -> None:
    args = _parse_deploy_args(
        [
            "--active-sides",
            "right",
            "--singularity-qpos-init-slowdown",
            "0.35",
        ]
    )
    active_sides = run_policy.validate_args(args)

    backends = run_policy.build_arm_backends(args, active_sides, control_hz=args.arm_control_hz)

    assert backends["right"].config.singularity_qpos_init_slowdown == 0.35
