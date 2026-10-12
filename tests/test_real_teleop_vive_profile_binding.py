from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop_runtime.configs import build_annotated_dual_rm65_calibration
from teleop_runtime.hand_tcp_alignment import tracker_hand_to_rm65_tcp_rotation
from teleop_runtime.real_teleop import vive_calibration_runtime as workflow
from teleop_runtime.real_teleop.side_mapping import build_vive_tracker_configs_for_args
from teleop_runtime.real_teleop.vive_calibration_profiles import (
    _load_gui_vive_calibration_profile,
    _save_gui_vive_calibration_profile,
    load_vive_calibration_profile,
    vive_configuration_metadata,
)
from teleop_runtime.retargeters.pose_to_arm import (
    PoseToArmRetargeter,
    PoseToArmRetargeterConfig,
)
from teleop_runtime.schemas import TeleopObservation
from teleop_runtime.sources.pose import VivePoseSource
from teleop_runtime.tracker_hand import canonical_hand_rpy_default


def _pose(rotation: np.ndarray | None = None) -> np.ndarray:
    pose = np.eye(4)
    if rotation is not None:
        pose[:3, :3] = rotation
    return pose


def _packet(poses: dict[str, np.ndarray], seq: int) -> bytes:
    return json.dumps(
        {
            "protocol": "teleop_runtime.vive/v1",
            "seq": seq,
            "sender_timestamp": float(seq),
            "trackers": [
                {
                    "tracker_name": "tracker_" + str(index + 1),
                    "tracker_role": role,
                    "valid": True,
                    "pose_matrix_4x4": pose.tolist(),
                }
                for index, (role, pose) in enumerate(poses.items())
            ],
        }
    ).encode()


def _calibrate(tmp_path, monkeypatch, swap, yaw, active_sides=("left", "right")):
    args = SimpleNamespace(
        swap_operator_sides=swap,
        swap_arm_sides=False,
        left_tracker_name="tracker_1",
        right_tracker_name="tracker_2",
        aux_tracker_name="",
        virtual_desk_post_yaw_deg=yaw,
        movement_multiple=1.35,
        translation_scale=1.35,
        load_profile="",
    )
    source = VivePoseSource(
        tracker_configs=build_vive_tracker_configs_for_args(args, active_sides),
        clock_sync_port=0,
    )
    # Use real v1 parsing/routing, with no socket, OpenVR, or robot backend.
    physical = {
        "left_wrist": _pose(Rotation.from_euler("z", 20, degrees=True).as_matrix()),
        "right_wrist": _pose(Rotation.from_euler("x", -35, degrees=True).as_matrix()),
    }
    source._process_received_batch([(_packet(physical, 1), 1000.0, 20.0)])
    calibration = build_annotated_dual_rm65_calibration()
    for role in ("left_wrist", "right_wrist"):
        calibration.set_tracker_offset(role, _pose())
    components = SimpleNamespace(
        session=SimpleNamespace(calibration=calibration, sources=[source]),
        active_sides=active_sides,
        left_retargeter=SimpleNamespace(clear_home_anchors=lambda: None),
        right_retargeter=SimpleNamespace(clear_home_anchors=lambda: None),
    )

    def sweep(**kw):
        if kw["stage_name"] == "WristOffset":
            channel = source.get_latest_channels()[kw["channel_name"]]
            pose = kw["components"].calibration.world_to_desk(channel.value)
            return [pose] * 3
        axis = {"VirtualDeskX": 0, "VirtualDeskY": 1, "VirtualDeskZ": 2}[
            kw["stage_name"]
        ]
        poses = [_pose() for _ in range(4)]
        for pose, distance in zip(poses, (0.0, 0.1, 0.2, 0.3)):
            pose[axis, 3] = distance
        return poses

    def wrists(**kw):
        channels = source.get_latest_channels()
        return tuple(
            [
                kw["components"].calibration.world_to_desk(
                    channels[f"{side}_wrist_pose"].value
                )
            ]
            * 3
            for side in components.active_sides
        )

    monkeypatch.setattr(
        workflow,
        "_select_gui_vive_coordinate_sweep_hand",
        lambda **kw: ("right", "right_wrist_pose"),
    )
    monkeypatch.setattr(workflow, "_capture_single_vive_channel_from_gui", sweep)
    monkeypatch.setattr(workflow, "_capture_dual_wrist_offsets_from_gui", wrists)
    ui = SimpleNamespace(append_message=lambda *args, **kwargs: None)
    path = workflow.calibrate_gui_vive_coordinate_profile(
        args,
        components,
        ui,
        save_profile=lambda args, components: _save_gui_vive_calibration_profile(
            args,
            components,
            profile_dir=tmp_path,
        ),
    )
    return args, source, physical, components, path


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("yaw", [0.0, 180.0])
def test_real_gui_swap_calibrate_reload_home_and_rotation_are_consistent(
    tmp_path, monkeypatch, swap, yaw
):
    args, source, physical, components, path = _calibrate(
        tmp_path, monkeypatch, swap, yaw
    )
    loaded = load_vive_calibration_profile(path)
    assert loaded.metadata["tracker_offset_role_binding"] == "source_role"
    assert loaded.captured_home_qpos == {}
    assert all(arm.home_qpos is None for arm in loaded.arms.values())
    assert (
        loaded.metadata["runtime_vive_profile"]["sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    components.session.calibration = loaded
    _load_gui_vive_calibration_profile(args, components)
    assert args.swap_operator_sides is swap
    assert args.movement_multiple == 1.35
    canonical = Rotation.from_euler(
        "xyz", canonical_hand_rpy_default(), degrees=True
    ).as_matrix()
    channels = source.get_latest_channels()
    a = tracker_hand_to_rm65_tcp_rotation()
    for config in source.tracker_configs.values():
        assert (
            loaded.metadata["tracker_offset_calibration"][config.role][
                "capture_channel"
            ]
            == config.channel_name
        )
        neutral = loaded.world_to_desk(physical[config.role], tracker_role=config.role)
        np.testing.assert_allclose(neutral[:3, :3], canonical, atol=1e-12)
        retargeter = PoseToArmRetargeter(
            PoseToArmRetargeterConfig(
                side=config.side,
                input_channel_name=config.channel_name,
                effector_name=config.side + "_arm",
                translation_scale=1.35,
                workspace_clip=False,
                input_local_translation_offset=np.array([0.0, 0.0, -0.04]),
                input_local_rotation_offset=a,
            )
        )
        robot_home = _pose(
            Rotation.from_euler("xyz", [10, -20, 5], degrees=True).as_matrix()
        )
        robot_home[:3, 3] = [0.4, 0.1, 0.2]
        human_home = retargeter.resolve_runtime_home_input_pose_desk(
            channel=channels[config.channel_name],
            calibration=loaded,
        )
        retargeter.set_home_anchors(
            human_pose_desk=human_home, robot_pose_desk=robot_home
        )
        initial = retargeter.retarget(
            TeleopObservation(channels=channels, timestamp=1000.0), loaded
        )
        # PoseToArm emits base-frame commands; Home anchors are desk-frame.
        np.testing.assert_allclose(
            loaded.base_to_desk(
                initial.commands[config.side + "_arm"].value,
                config.side,
            ),
            robot_home,
            atol=1e-12,
        )
        # Continuous multi-axis movement after Home must use the same frame,
        # not a crossed offset hidden by the initial relative anchor.
        for tick in range(1, 61):
            delta = Rotation.from_euler(
                "xyz", [tick / 20.0, -tick / 30.0, tick / 40.0], degrees=True
            ).as_matrix()
            desk_pose = _pose(canonical @ delta)
            desk_pose[:3, 3] = neutral[:3, 3] + np.array([tick / 10000.0, 0.0, 0.0])
            raw = (
                loaded.world_to_desk_pose()
                @ desk_pose
                @ np.linalg.inv(loaded.tracker_offsets[config.role])
            )
            packet_poses = dict(physical, **{config.role: raw})
            source._process_received_batch(
                [(_packet(packet_poses, tick + 1), 1000.0 + tick, 20.0 + tick)]
            )
            observation = TeleopObservation(
                channels=source.get_latest_channels(), timestamp=1000.0 + tick
            )
            base_output = (
                retargeter.retarget(observation, loaded)
                .commands[config.side + "_arm"]
                .value
            )
            output = loaded.base_to_desk(base_output, config.side)
            expected = robot_home.copy()
            expected[:3, :3] = robot_home[:3, :3] @ a.T @ delta @ a
            expected[:3, 3] += 1.35 * (
                desk_pose[:3, 3]
                - neutral[:3, 3]
                + (canonical @ delta - canonical) @ np.array([0.0, 0.0, -0.04])
            )
            np.testing.assert_allclose(output, expected, atol=1e-11)


def test_reload_keeps_explicit_mapping_and_multiplier_instead_of_saved_values(
    tmp_path, monkeypatch
):
    args, _source, _physical, components, _path = _calibrate(
        tmp_path, monkeypatch, False, 180.0
    )
    args.swap_operator_sides = True
    args.movement_multiple = args.translation_scale = 1.0
    _load_gui_vive_calibration_profile(args, components)
    state = vive_configuration_metadata(
        args, components.session.calibration, components.active_sides
    )
    assert state["tracker_input_mapping"] == {"left": "right", "right": "left"}
    assert state["movement_multiple"] == 1.0
    assert state["profile"]["saved_tracker_input_mapping"] == {
        "left": "left",
        "right": "right",
    }
    assert state["profile"]["saved_movement_multiple"] == 1.35


def test_source_mapping_change_during_calibration_refuses_to_save(
    tmp_path, monkeypatch
):
    # The wrapped capture is installed by _calibrate; wrap assignment so the
    # real source changes only after samples have been returned.
    original_setattr = monkeypatch.setattr

    def set_attribute(obj, name, value, **kwargs):
        if obj is workflow and name == "_capture_dual_wrist_offsets_from_gui":

            def capture(**kw):
                poses = value(**kw)
                kw["source"].swap_left_right_source_roles()
                return poses

            value_to_set = capture
        else:
            value_to_set = value
        original_setattr(obj, name, value_to_set, **kwargs)

    original_setattr(monkeypatch, "setattr", set_attribute)
    with pytest.raises(RuntimeError, match="mapping changed"):
        _calibrate(tmp_path, monkeypatch, False, 0.0)
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize("failure", ["capture", "cancel", "save"])
def test_incomplete_calibration_preserves_installed_frame_and_home(
    monkeypatch, failure
):
    calibration = build_annotated_dual_rm65_calibration()
    calibration.set_world_to_desk(_pose())
    calibration.arms["right_arm"].home_qpos = np.ones(6)
    before = calibration.to_dict()
    source = VivePoseSource(
        tracker_configs=build_vive_tracker_configs_for_args(
            SimpleNamespace(
                swap_operator_sides=False,
                left_tracker_name="tracker_1",
                right_tracker_name="tracker_2",
                aux_tracker_name="",
            ),
            ("left", "right"),
        ),
        clock_sync_port=0,
    )
    components = SimpleNamespace(
        session=SimpleNamespace(calibration=calibration, sources=[source]),
        active_sides=("left", "right"),
        left_retargeter=SimpleNamespace(clear_home_anchors=Mock()),
        right_retargeter=SimpleNamespace(clear_home_anchors=Mock()),
    )

    def capture(args, context, ui, **kw):
        changed = _pose()
        changed[0, 3] = 0.25
        context.calibration.set_world_to_desk(changed)
        context.calibration.clear_home_state()
        if failure == "capture":
            raise RuntimeError("synthetic capture error")
        if failure == "cancel":
            raise KeyboardInterrupt

    monkeypatch.setattr(workflow, "calibrate_vive_coordinate_profile", capture)
    save = Mock(side_effect=OSError("synthetic save failure"))
    with pytest.raises(
        {"capture": RuntimeError, "cancel": KeyboardInterrupt, "save": OSError}[failure]
    ):
        workflow.calibrate_gui_vive_coordinate_profile(
            SimpleNamespace(), components, SimpleNamespace(), save_profile=save
        )
    assert components.session.calibration is calibration
    assert calibration.to_dict() == before
    components.left_retargeter.clear_home_anchors.assert_not_called()
    components.right_retargeter.clear_home_anchors.assert_not_called()
    assert save.call_count == (1 if failure == "save" else 0)


def test_runtime_load_rechecks_digest_after_external_file_change(tmp_path, monkeypatch):
    args, _source, _physical, _components, path = _calibrate(
        tmp_path, monkeypatch, False, 0.0
    )
    approved = args.expected_vive_profile_sha256
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 differs"):
        load_vive_calibration_profile(path, expected_sha256=approved)


def test_legacy_unmarked_profile_keeps_numeric_transforms_and_operator_settings(
    tmp_path, monkeypatch
):
    args, _source, _physical, _components, path = _calibrate(
        tmp_path, monkeypatch, False, 180.0
    )
    data = json.loads(path.read_text())
    data["metadata"].pop("tracker_offset_role_binding")
    path.write_text(json.dumps(data), encoding="utf-8")
    from teleop_runtime.calibration import TeleopCalibration

    legacy = TeleopCalibration.load_json(path)
    loaded = load_vive_calibration_profile(path)
    assert (
        loaded.metadata["runtime_vive_profile"]["offset_role_binding"]
        == "legacy_unmarked"
    )
    for role in legacy.tracker_offsets:
        np.testing.assert_array_equal(
            loaded.tracker_offsets[role], legacy.tracker_offsets[role]
        )
    np.testing.assert_array_equal(loaded.T_W_O, legacy.T_W_O)
    np.testing.assert_array_equal(loaded.T_O_D, legacy.T_O_D)
    assert args.swap_operator_sides is False
    assert args.movement_multiple == 1.35


def test_marked_logical_coordinate_profile_is_not_silently_used_as_real_source_offsets(
    tmp_path, monkeypatch
):
    _args, _source, _physical, _components, path = _calibrate(
        tmp_path, monkeypatch, False, 0.0
    )
    data = json.loads(path.read_text())
    data["metadata"]["tracker_offset_role_binding"] = "logical_channel"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="source-role"):
        load_vive_calibration_profile(path)


def test_single_right_gui_readiness_uses_the_mapped_left_source_offset():
    from teleop_runtime.real_teleop.gui_vive_calibration import (
        vive_calibration_profile_ui_state,
    )

    calibration = build_annotated_dual_rm65_calibration()
    calibration.set_world_to_desk(_pose())
    calibration.set_tracker_offset("left_wrist", _pose())
    args = SimpleNamespace(
        swap_operator_sides=True,
        left_tracker_name="tracker_1",
        right_tracker_name="tracker_2",
        aux_tracker_name="",
        load_profile="profile.json",
    )
    source = VivePoseSource(
        tracker_configs=build_vive_tracker_configs_for_args(args, ("right",)),
        clock_sync_port=0,
    )
    components = SimpleNamespace(
        session=SimpleNamespace(calibration=calibration, sources=[source]),
        active_sides=("right",),
    )
    state = vive_calibration_profile_ui_state(args, components)
    assert state["vive_calibration_profile_ready"] is True
    assert state["vive_calibration_profile_status"] == "desk=ok tracker_offsets=ok"
    source.swap_left_right_source_roles()
    state = vive_calibration_profile_ui_state(args, components)
    assert state["vive_calibration_profile_ready"] is False
    assert "missing right_wrist" in state["vive_calibration_profile_status"]


def _canonical_args(path, monkeypatch, expected=None):
    from control_station import real_teleop_entrypoint as frontend

    digest = (
        hashlib.sha256(path.read_bytes()).hexdigest() if expected is None else expected
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test",
            "--load-profile",
            str(path),
            "--expected-vive-profile-sha256",
            digest,
            "--active-sides",
            "both",
            "--left-arm-ip",
            "192.0.2.11",
            "--right-arm-ip",
            "192.0.2.12",
            "--hand-backend",
            "none",
            "--no-device-preflight",
            "--no-q6-default-config",
            "--no-dmtac-default-config",
            "--no-camera",
            "--no-collection",
            "--no-arm-ip-save-latest",
            "--arm-control-mode",
            "position",
            "--movement-multiple",
            "1.0",
        ],
    )
    return frontend.parse_args()


def test_startup_verifies_the_actual_digest_before_constructing_arm_backends(
    tmp_path, monkeypatch
):
    from teleop_runtime.real_teleop import component_builder

    _args, _source, _physical, _components, path = _calibrate(
        tmp_path, monkeypatch, False, 0.0
    )
    args = _canonical_args(path, monkeypatch, expected="0" * 64)
    arm = Mock(side_effect=AssertionError("arm construction must not happen"))
    monkeypatch.setattr(component_builder, "RM65ArmBackend", arm)
    monkeypatch.setattr(component_builder, "RM65ImpedanceArmBackend", arm)
    with pytest.raises(ValueError, match="SHA256 differs"):
        component_builder.build_real_teleop_components(args)
    arm.assert_not_called()


def test_actuator_startup_snapshot_records_file_and_actual_source_mapping(
    tmp_path, monkeypatch
):
    from teleop_runtime.real_teleop.diagnostic_setup import (
        _new_actuator_command_trace_recorder,
    )

    _args, source, _physical, components, path = _calibrate(
        tmp_path, monkeypatch, True, 180.0
    )
    args = _canonical_args(path, monkeypatch)
    components.logical_arm_ip_mapping = {"left": "192.0.2.11", "right": "192.0.2.12"}
    recorder = _new_actuator_command_trace_recorder(
        args, components=components, output_dir=tmp_path / "trace"
    )
    state = recorder.config.metadata["vive_startup_configuration"]
    assert state["profile"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        state["source_role_by_output_channel"] == source.source_role_by_output_channel
    )
    assert state["movement_multiple"] == 1.0
    assert list((tmp_path / "trace").iterdir()) == []
    recorder.close()
    manifest = json.loads((tmp_path / "trace" / "manifest.json").read_text())
    assert manifest["runtime_metadata"]["vive_startup_configuration"] == state
    assert manifest["result"]["rows_by_side"] == {"left": 0, "right": 0}


def test_single_swapped_calibration_discards_offsets_from_the_old_global_frame(
    tmp_path, monkeypatch
):
    _args, _source, _physical, components, path = _calibrate(
        tmp_path, monkeypatch, True, 180.0, ("right",)
    )
    assert set(components.session.calibration.tracker_offsets) == {"left_wrist"}
    loaded = load_vive_calibration_profile(path)
    assert set(loaded.tracker_offsets) == {"left_wrist"}
    assert (
        loaded.metadata["tracker_offset_calibration"]["left_wrist"]["capture_channel"]
        == "right_wrist_pose"
    )
