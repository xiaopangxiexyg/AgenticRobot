from __future__ import annotations

import argparse
import json
import sys
from unittest.mock import call, patch

import numpy as np

from teleop_runtime.collection import run_collect
from teleop_runtime.collection.writer_v21 import LeRobotV21WriterConfig
from teleop_runtime.backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop
from teleop_runtime.defaults import DEFAULT_RM65_ARM_CONTROL_HZ, DEFAULT_TELEOP_SAMPLE_HZ
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


def test_collect_defaults_to_15hz_for_bimanual() -> None:
    args = _parse_collect_args(["--active-sides", "both", "--no-camera", "--no-video"])

    assert args.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.nominal_fps == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.arm_control_hz == DEFAULT_RM65_ARM_CONTROL_HZ


def test_collect_defaults_to_15hz_for_single_arm() -> None:
    args = _parse_collect_args(["--active-sides", "right", "--no-camera", "--no-video"])

    assert args.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.nominal_fps == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.arm_control_hz == DEFAULT_RM65_ARM_CONTROL_HZ


def test_collect_nominal_fps_follows_explicit_hz_when_omitted() -> None:
    args = _parse_collect_args(["--active-sides", "right", "--no-camera", "--no-video", "--hz", "12"])

    assert args.hz == 12.0
    assert args.nominal_fps == 12.0


def test_lerobot_writer_default_nominal_fps_is_15hz(tmp_path) -> None:
    config = LeRobotV21WriterConfig(root_dir=tmp_path)

    assert config.nominal_fps == DEFAULT_TELEOP_SAMPLE_HZ


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
    args = _parse_real_teleop_args(["--no-record-actual-wrist-camera-pose"])

    assert args.record_actual_wrist_camera_pose is False


def test_real_teleop_legacy_session_recording_defaults_off() -> None:
    args = _parse_real_teleop_args([])

    assert args.no_record is True


def test_real_teleop_legacy_session_recording_can_be_enabled() -> None:
    args = _parse_real_teleop_args(["--record"])

    assert args.no_record is False


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
    assert args.arm_control_hz == DEFAULT_RM65_ARM_CONTROL_HZ
    assert hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert info["source"] == "cli.hz"


def test_deploy_default_fallback_action_hz_is_15hz_for_single_arm() -> None:
    args = _parse_deploy_args(["--active-sides", "right"])
    active_sides = run_policy.validate_args(args)
    hz, info = run_policy.resolve_effective_action_hz(args, policy=None)

    assert active_sides == ("right",)
    assert args.hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert args.arm_control_hz == DEFAULT_RM65_ARM_CONTROL_HZ
    assert hz == DEFAULT_TELEOP_SAMPLE_HZ
    assert info["source"] == "cli.hz"


def test_deploy_dry_run_arm_backend_uses_40hz_control_default() -> None:
    args = _parse_deploy_args(["--active-sides", "right"])
    active_sides = run_policy.validate_args(args)

    backends = run_policy.build_arm_backends(args, active_sides, control_hz=args.arm_control_hz)

    assert backends["right"].config.control_dt == 1.0 / DEFAULT_RM65_ARM_CONTROL_HZ


def test_deploy_arm_limits_match_real_teleop_defaults() -> None:
    args = _parse_deploy_args(["--active-sides", "right"])
    active_sides = run_policy.validate_args(args)

    backends = run_policy.build_arm_backends(args, active_sides, control_hz=args.arm_control_hz)

    assert args.arm_max_angvel == (0.7, 0.7, 0.9, 1.5, 1.5, 2.0)
    assert args.arm_step_dt_scaling is True
    assert args.arm_step_dt_alpha == 0.25
    assert args.arm_step_dt_stall_reset_sec == 0.25
    assert args.arm_step_dt_max_scale == 3.0
    assert args.arm_step_dt_max_scale_auto is True
    assert np.isclose(args.max_arm_joint_step_rad, np.deg2rad(6.0))
    assert backends["right"].config.max_angvel == (0.7, 0.7, 0.9, 1.5, 1.5, 2.0)
    assert backends["right"].config.use_actual_send_dt_scaling is True
    assert backends["right"].config.actual_send_dt_max_scale == 3.0


def test_deploy_arm_step_dt_max_scale_auto_follows_action_rate() -> None:
    args = _parse_deploy_args(["--active-sides", "right", "--hz", "15", "--arm-control-hz", "50"])
    active_sides = run_policy.validate_args(args)

    backends = run_policy.build_arm_backends(args, active_sides, control_hz=args.arm_control_hz)

    assert args.arm_step_dt_max_scale == 50.0 / 15.0 * 1.1
    assert args.arm_step_dt_max_scale_auto is True
    assert backends["right"].config.actual_send_dt_max_scale == 50.0 / 15.0 * 1.1


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

    expected = np.asarray((0.7, 0.7, 0.9, 1.5, 1.5, 2.0), dtype=float) / 15.0
    actual = command.commands["right_arm"].metadata["max_joint_step_rad"]
    assert np.allclose(actual, expected)
    assert command.commands["right_arm"].metadata["deploy_action_step_dt_sec"] == 1.0 / 15.0


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
