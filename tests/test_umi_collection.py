from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from teleop_runtime.collection.operator_labels import (
    LABEL_FIX_ARM,
    LABEL_INIT_ARM,
    LABEL_TELEOP,
    OperatorLabelStateMachine,
)
from teleop_runtime.collection.pose_storage import (
    bind_collection_row_pose_storage,
    validate_v23_collection_row_actual_tcp,
)
from teleop_runtime.collection.policy_io import load_policy_io_layout
from teleop_runtime.collection.umi.calibration import UmiCalibrationBinding
from teleop_runtime.collection.umi.cli import main as umi_cli_main
from teleop_runtime.collection.umi.config import load_umi_collection_config
from teleop_runtime.collection.umi.pose_adapter import UmiPoseAdapter
from teleop_runtime.collection.umi.row_encoder import UmiV23RowEncoder
from teleop_runtime.collection.umi.recorder import UmiDatasetRecorder
from teleop_runtime.collection.umi.types import UmiPoseMeasurement, UmiSample
from teleop_runtime.gui.calibration_web_ui import _render_initial_html


ROOT = Path(__file__).parents[1]


def test_default_config_and_layout_are_umi_specific():
    config = load_umi_collection_config(
        ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    assert config.left.camera_serial == "261122273924"
    assert config.right.camera_serial == "260522278055"
    assert config.left.uses_identity_tcp_placeholder
    layout = load_policy_io_layout(config.policy_io_layout)
    assert layout is not None
    assert layout.action_dim == 16
    assert layout.state_dim == 16
    assert layout.arm_reference_frame_by_side == {"left": "desk", "right": "desk"}


def test_pose_adapter_uses_explicit_tcp_tracker_direction():
    T_vive_to_desk = np.eye(4)
    T_vive_to_desk[:3, 3] = [1.0, 0.0, 0.0]
    T_tcp_to_tracker = np.eye(4)
    T_tcp_to_tracker[:3, 3] = [0.0, 0.2, 0.0]
    adapter = UmiPoseAdapter(
        T_vive_world_to_desk=T_vive_to_desk,
        T_tcp_to_tracker_by_side={
            "left": T_tcp_to_tracker,
            "right": np.eye(4),
        },
    )
    T_tracker_to_vive = np.eye(4)
    T_tracker_to_vive[:3, 3] = [0.0, 0.0, 0.3]
    assert np.allclose(adapter.raw_pose7("left", T_tracker_to_vive)[:3], [0.0, 0.2, 0.3])
    assert np.allclose(adapter.desk_pose7("left", T_tracker_to_vive)[:3], [1.0, 0.2, 0.3])


def test_operator_labels_are_latched_exclusive_and_annotation_only():
    labels = OperatorLabelStateMachine()
    labels.toggle("left", "fix_arm", source="foot_pad", timestamp=1.0)
    assert labels.snapshot().labels_by_side["left"] == LABEL_FIX_ARM
    labels.toggle("left", "init_arm", source="gui", timestamp=2.0)
    snapshot = labels.snapshot()
    assert snapshot.labels_by_side["left"] == LABEL_INIT_ARM
    labels.toggle("left", "init_arm", source="gui", timestamp=3.0)
    assert labels.snapshot().labels_by_side["left"] == LABEL_TELEOP
    payload = snapshot.arm_mode_compatibility_payload()
    assert payload["sides"]["left"]["detail"]["control_effect"] is False
    assert payload["sides"]["left"]["audit"]["init_phase_valid"] is False


def _calibration_payload(now: datetime) -> dict:
    return {
        "schema": "dexteleop.umi_hand_eye.result.v1",
        "created_at_utc": "2026-07-17T00:00:00Z",
        "expires_at_utc": "2026-07-17T01:00:00Z",
        "desk_definition": "calibration_board_frame",
        "side": "left",
        "camera_serial": "261122273924",
        "tracker_name": "tracker_1",
        "tracker_role": "left_wrist",
        "T_vive_world_to_desk": np.eye(4).tolist(),
    }


def test_calibration_binding_checks_expiry_and_device(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_calibration_payload(datetime.now(timezone.utc))))
    binding = UmiCalibrationBinding.load(path)
    config = load_umi_collection_config(
        ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    binding.require_valid(
        config,
        when=datetime(2026, 7, 17, 0, 30, tzinfo=timezone.utc),
    )
    with pytest.raises(RuntimeError, match="expired"):
        binding.require_valid(
            config,
            when=datetime(2026, 7, 17, 2, 0, tzinfo=timezone.utc),
        )


def test_native_umi_row_is_validated_then_stored_in_desk(tmp_path):
    layout = load_policy_io_layout(
        ROOT / "teleop_runtime/configs/policy_io_layouts/dual_umi_dmtac_v23.json"
    )
    assert layout is not None
    labels = OperatorLabelStateMachine()
    poses = {}
    for index, side in enumerate(("left", "right"), start=1):
        poses[side] = UmiPoseMeasurement(
            side=side,
            raw_tcp_pose7=np.array([float(index), 0, 0, 0, 0, 0, 1]),
            desk_tcp_pose7=np.array([float(index + 10), 0, 0, 0, 0, 0, 1]),
            source_timestamp=99.9,
            available_timestamp=99.95,
            source_clock="vive_sender_monotonic_calibrated_to_local_wall",
            available_clock="local_wall_time",
            sequence=index,
            tracker_name=f"tracker_{index}",
            tracker_role=f"{side}_wrist",
            age_sec=0.1,
        )
    sample = UmiSample(
        episode_index=0,
        frame_index=0,
        timestamp=100.0,
        poses=poses,
        gripper_commands={"left": 0.2, "right": 0.8},
        hand_states={
            "left": {"gripper_measured_scalar": 0.25, "joint_timestamp": 99.9},
            "right": {"gripper_measured_scalar": 0.75, "joint_timestamp": 99.9},
        },
        operator_labels=labels.snapshot(),
    )
    row, expectations = UmiV23RowEncoder().encode_row(
        sample,
        layout=layout,
        camera_video_paths=None,
        tactile_storage_refs=None,
        inline_tactile=False,
    )
    assert expectations == {}
    assert len(row["action"]) == 16
    assert len(row["observation.state"]) == 16

    contract_payload = {
        "schema": "dexteleop.umi_hand_eye.result.v1",
        "created_at_utc": "2026-07-17T00:00:00Z",
        "expires_at_utc": "2026-07-17T01:00:00Z",
        "desk_definition": "calibration_board_frame",
        "side": "left",
        "camera_serial": "261122273924",
        "tracker_name": "tracker_1",
        "tracker_role": "left_wrist",
        "T_vive_world_to_desk": np.array(
            [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        ).tolist(),
    }
    temp = tmp_path / "calibration.json"
    temp.write_text(json.dumps(contract_payload))
    contract = UmiCalibrationBinding.load(temp).pose_frame_contract()
    validate_v23_collection_row_actual_tcp(row, contract=contract)
    stored = bind_collection_row_pose_storage(row, contract=contract)
    assert stored["observation.left_arm.pose"][0] == pytest.approx(11.0)
    assert stored["observation.right_arm.pose"][0] == pytest.approx(12.0)


def test_umi_recorder_uses_native_encoder_without_arm_sample(tmp_path):
    config = load_umi_collection_config(
        ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(_calibration_payload(datetime.now(timezone.utc))))
    binding = UmiCalibrationBinding.load(calibration_path)
    layout = load_policy_io_layout(config.policy_io_layout)
    assert layout is not None
    from dataclasses import replace

    config = replace(config, dataset_root=tmp_path / "dataset")
    recorder = UmiDatasetRecorder(config=config, calibration=binding, policy_io_layout=layout)
    # The minimal test environment intentionally omits the optional pyarrow export dependency.
    recorder.writer.config.write_parquet = False
    episode = recorder.start_episode({"test": True})
    poses = {
        side: UmiPoseMeasurement(
            side=side,
            raw_tcp_pose7=np.array([float(index), 0, 0, 0, 0, 0, 1]),
            desk_tcp_pose7=np.array([float(index), 0, 0, 0, 0, 0, 1]),
            source_timestamp=9.9,
            available_timestamp=9.95,
            source_clock="vive_receiver_wall_fallback",
            available_clock="local_wall_time",
            sequence=index,
            tracker_name=f"tracker_{index}",
            tracker_role=f"{side}_wrist",
            age_sec=0.1,
        )
        for index, side in enumerate(("left", "right"), start=1)
    }
    sample = UmiSample(
        episode_index=episode,
        frame_index=0,
        timestamp=10.0,
        poses=poses,
        gripper_commands={"left": 0.1, "right": 0.9},
        hand_states={
            "left": {"gripper_measured_scalar": 0.2, "joint_timestamp": 9.9},
            "right": {"gripper_measured_scalar": 0.8, "joint_timestamp": 9.9},
        },
        operator_labels=OperatorLabelStateMachine().snapshot(),
    )
    recorder.submit(sample)
    recorder.stop_episode(reason="test")
    recorder.close()
    assert (tmp_path / "dataset/meta/info.json").exists()
    info = json.loads((tmp_path / "dataset/meta/info.json").read_text())
    assert info["embodiment"] == "umi_dual_wrist"
    assert info["embodiment_has_arm"] is False
    assert (tmp_path / "dataset/meta/episodes.jsonl").read_text().strip()


def test_umi_components_do_not_reference_arm_backends():
    source = (ROOT / "teleop_runtime/collection/umi/components.py").read_text()
    assert "RM65" not in source
    assert "backends.arms" not in source
    assert "FixArmCommandController" not in source
    assert "InitArmCommandController" not in source


def test_existing_formal_gui_renders_umi_title_without_second_web_server():
    html = _render_initial_html({"ui_mode": "umi_collection", "stage": "ready"})
    assert "DexTeleop UMI 数采" in html
    assert "collection_web_ui" not in (
        ROOT / "teleop_runtime/collection/umi/controller.py"
    ).read_text()


def test_validate_only_cli_does_not_initialize_devices(tmp_path, capsys):
    now = datetime.now(timezone.utc)
    calibration = _calibration_payload(now)
    calibration["created_at_utc"] = now.isoformat()
    calibration["expires_at_utc"] = (now + timedelta(minutes=30)).isoformat()
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    config_path = tmp_path / "umi.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_root": str(tmp_path / "dataset"),
                "calibration_result": str(calibration_path),
                "policy_io_layout": str(
                    ROOT
                    / "teleop_runtime/configs/policy_io_layouts/dual_umi_dmtac_v23.json"
                ),
                "allow_identity_tcp_placeholder": True,
                "sides": {
                    "left": {"camera_serial": "261122273924", "tracker_name": "tracker_1"},
                    "right": {"camera_serial": "260522278055", "tracker_name": "tracker_2"},
                },
            }
        )
    )
    assert umi_cli_main(["--config", str(config_path), "--validate-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["action_dim"] == 16
    assert not (tmp_path / "dataset").exists()
