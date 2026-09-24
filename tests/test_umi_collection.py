from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import inspect
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pytest

from collection_runtime.progress.profiles import (
    bind_dataset_progress_profile,
    load_task_progress_profile_file,
)
from control_station.umi.application_policy import (
    DEFAULT_STATION_UMI_OPERATOR_POLICY,
)
from control_station.umi_application import (
    _run_collection_session,
    run_umi_collection as run_station_umi_collection,
)
from dataset_contracts.v23_validation import validate_v23_row_causal_snapshots
from dataset_contracts.v25p_native_storage import (
    V25P_PARQUET_CHUNK_LAYOUT_V1,
    V25P_RECORD_CJSON_LAYOUT_V1,
)
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
from teleop_runtime.collection.umi.calibration import UmiCalibrationBinding, UmiCalibrationSet
from teleop_runtime.collection.umi.cli import build_parser as build_umi_parser
from teleop_runtime.collection.umi.cli import main as umi_cli_main
from teleop_runtime.collection.umi.config import load_umi_collection_config
from teleop_runtime.collection.umi.controller import UmiCollectionController
from teleop_runtime.collection.umi.components import (
    UmiComponents,
    UmiStartupResource,
    _configure_dmtac_backend_configs,
)
from teleop_runtime.collection.umi.app import run_umi_collection
from teleop_runtime.collection.umi.angle_trigger_calibration import (
    ensure_default_command_input_from_gui,
)
from teleop_runtime.collection.umi.application_ports import (
    UmiOperatorPolicyPort,
)
from teleop_runtime.collection.umi.default_components import (
    DEFAULT_UMI_HARDWARE_PRESENTATION,
    build_default_umi_components,
    prepare_default_umi_hardware,
)
from teleop_runtime.collection.umi.pose_adapter import (
    UmiPoseAdapter,
    as_rigid_transform,
    normalize_vive_tracker_transform,
)
from teleop_runtime.collection.umi.row_encoder import UmiV23RowEncoder
from teleop_runtime.collection.umi.recorder import (
    UmiDatasetRecorder,
    _validate_umi_dataset_admission,
)
from teleop_runtime.collection.umi.snapshot import _gripper_measurement_timing
from teleop_runtime.collection.umi.recalibration import (
    config_for_calibration_dataset,
    validate_hand_eye_config_binding,
)
from teleop_runtime.collection.umi.types import (
    SnapshotRejection,
    UmiPoseMeasurement,
    UmiSample,
)
from control_station.web.calibration_web_ui import _render_initial_html


ROOT = Path(__file__).parents[1]


def test_umi_runtime_factories_are_injectable_without_changing_cli_defaults():
    assert run_umi_collection is run_station_umi_collection
    signature = inspect.signature(run_umi_collection)
    assert (
        signature.parameters["hardware_preparer"].default
        is prepare_default_umi_hardware
    )
    assert (
        signature.parameters["component_builder"].default
        is build_default_umi_components
    )
    assert (
        signature.parameters["hardware_presentation"].default
        is DEFAULT_UMI_HARDWARE_PRESENTATION
    )
    assert signature.parameters["pose_adapter_builder"].default is UmiPoseAdapter
    assert (
        signature.parameters["command_input_initializer"].default
        is ensure_default_command_input_from_gui
    )
    assert (
        signature.parameters["operator_policy"].default
        is DEFAULT_STATION_UMI_OPERATOR_POLICY
    )
    assert isinstance(
        DEFAULT_STATION_UMI_OPERATOR_POLICY,
        UmiOperatorPolicyPort,
    )


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize(
    "storage_layout", (V25P_RECORD_CJSON_LAYOUT_V1, V25P_PARQUET_CHUNK_LAYOUT_V1)
)
def test_umi_application_binds_shadow_only_when_enabled(
    tmp_path, monkeypatch, enabled, storage_layout
) -> None:
    from dataset_contracts.v22 import PolicyIOLayout
    from teleop_runtime.collection.umi.v25p_producer import UmiV25pProducer

    config = SimpleNamespace(
        dataset_root=tmp_path,
        task_id="umi-shadow-binding",
        v25p_native_shadow=enabled,
        v25p_native_storage_layout=storage_layout,
    )
    calibrations = SimpleNamespace(
        desk=SimpleNamespace(
            T_vive_world_to_desk=np.eye(4),
            pose_frame_contract=lambda: SimpleNamespace(contract_hash="a" * 64),
        ),
        T_tcp_to_tracker_by_side={},
        by_side={},
    )
    received = {}
    reached_recorder = RuntimeError("stop before device construction")

    def capture_recorder(**kwargs):
        received.update(kwargs)
        raise reached_recorder

    monkeypatch.setattr("control_station.umi_application.UmiDatasetRecorder", capture_recorder)
    with pytest.raises(RuntimeError) as raised:
        _run_collection_session(
            config=config,
            calibrations=calibrations,
            layout=PolicyIOLayout(name="umi_shadow_binding"),
            ui=None,
            prepared_hardware=None,
            component_builder=lambda *_args, **_kwargs: pytest.fail("device construction"),
            pose_adapter_builder=lambda **_kwargs: object(),
        )
    assert raised.value is reached_recorder
    shadow = received["v25p_native_shadow"]
    if enabled:
        assert isinstance(shadow.producer, UmiV25pProducer)
        assert shadow.store.dataset_root == tmp_path
        assert shadow.store.dataset_id == "umi:umi-shadow-binding"
        assert shadow.store.physical_storage_layout == storage_layout
        assert shadow.active_identity is None
    else:
        assert shadow is None
        assert not (tmp_path / "meta").exists()


def test_umi_component_bundle_exposes_backend_neutral_ports():
    pose = object()
    cameras = {"wrist": object()}
    command_input = object()
    end_effectors = {"tool": object()}
    rig = object()
    retargeters = {"tool": object()}
    state_history = object()
    components = UmiComponents(
        vive=pose,
        cameras=cameras,
        angle_trigger=command_input,
        grippers=end_effectors,
        rig=rig,
        retargeters=retargeters,
        hand_state_poller=state_history,
    )

    assert components.pose_source is pose
    assert components.camera_sources is cameras
    assert components.command_input is command_input
    assert components.end_effectors is end_effectors
    assert components.command_rig is rig
    assert components.command_retargeters is retargeters
    assert components.state_history is state_history
    assert components.command_input_name == "command_input"


def test_umi_camera_start_action_is_used_without_replacing_frame_source():
    events: list[str] = []

    class Camera:
        def start(self) -> None:
            events.append("direct-start")

        def close(self) -> None:
            events.append("close")

    camera = Camera()
    resource = UmiStartupResource(
        key="camera.left_wrist",
        component=camera,
        label="camera",
        requirement="collection",
        strict_start_message="starting camera",
        start_action=lambda: events.append("bounded-start"),
    )
    components = UmiComponents(
        vive=object(),
        cameras={"left_wrist": camera},
        angle_trigger=object(),
        grippers={},
        rig=object(),
        retargeters={},
        hand_state_poller=object(),
        startup_resources=(resource,),
    )

    components.start()
    assert events == ["bounded-start"]
    assert components.camera_sources["left_wrist"] is camera
    components.close()
    assert events == ["bounded-start", "close"]


def test_umi_optional_camera_native_start_timeout_aborts_next_resource():
    events: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def start(self) -> None:
            events.append(self.name)

        def close(self) -> None:
            events.append(f"close:{self.name}")

    camera = Resource("camera")
    next_resource = Resource("next")
    components = UmiComponents(
        vive=object(),
        cameras={"left_wrist": camera},
        angle_trigger=object(),
        grippers={},
        rig=object(),
        retargeters={},
        hand_state_poller=object(),
        startup_resources=(
            UmiStartupResource(
                key="camera.left_wrist",
                component=camera,
                label="camera",
                requirement="collection",
                strict_start_message="starting camera",
                start_action=lambda: (_ for _ in ()).throw(TimeoutError("native start")),
            ),
            UmiStartupResource(
                key="next",
                component=next_resource,
                label="next",
                requirement="collection_core",
                strict_start_message="starting next",
            ),
        ),
    )

    with pytest.raises(TimeoutError, match="native start"):
        components.start()
    assert events == []
    assert "native start" in components.startup_errors["camera.left_wrist"]


def test_umi_component_bundle_uses_injected_diagnostics_without_backend_config():
    backend = object()
    components = UmiComponents(
        vive=object(),
        cameras={},
        angle_trigger=object(),
        grippers={"tool": backend},
        rig=object(),
        retargeters={},
        hand_state_poller=object(),
        end_effector_diagnostics_provider=lambda side, candidate, _config: {
            "component_id": side,
            "same_backend": candidate is backend,
            "is_connected": True,
        },
    )

    assert components.end_effector_diagnostics(
        "tool",
        SimpleNamespace(),
    ) == {
        "component_id": "tool",
        "same_backend": True,
        "is_connected": True,
    }


def test_umi_strict_start_rolls_back_resource_when_readiness_check_fails():
    events: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def start(self) -> None:
            events.append(f"start:{self.name}")

        def close(self) -> None:
            events.append(f"close:{self.name}")

    pose = Resource("pose")
    tool = Resource("tool")
    components = UmiComponents(
        vive=pose,
        cameras={},
        angle_trigger=Resource("input"),
        grippers={"tool": object()},
        rig=tool,
        retargeters={},
        hand_state_poller=Resource("history"),
        strict_startup=True,
        startup_resources=(
            UmiStartupResource(
                key="pose",
                component=pose,
                label="pose",
                requirement="collection_core",
                strict_start_message="start pose",
            ),
            UmiStartupResource(
                key="tool",
                component=tool,
                label="tool",
                requirement="collection_core",
                strict_start_message="start tool",
                require_end_effector_readiness=True,
            ),
        ),
        end_effector_readiness_check=lambda: (_ for _ in ()).throw(
            RuntimeError("not ready")
        ),
    )

    with pytest.raises(RuntimeError, match="not ready"):
        components.start()

    assert events == [
        "start:pose",
        "start:tool",
        "close:tool",
        "close:pose",
    ]
    assert components._started is False


def test_umi_custom_startup_resource_keys_drive_status_without_device_names():
    class Resource:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

    source = Resource()
    components = UmiComponents(
        vive=source,
        cameras={},
        angle_trigger=source,
        grippers={},
        rig=source,
        retargeters={},
        hand_state_poller=source,
        startup_resources=(
            UmiStartupResource(
                key="custom.pose",
                component=source,
                label="custom pose",
                requirement="collection_core",
                strict_start_message="start custom pose",
            ),
        ),
        startup_errors={"custom.pose": "offline"},
    )

    assert components.hardware_status() == {
        "custom.pose": {
            "online": False,
            "health": "error",
            "status_level": "error",
            "status_label": "连接失败",
            "requirement": "collection_core",
            "error": "offline",
        }
    }


def test_umi_components_reject_dmtac_start_that_swallowed_connection_failure():
    backend = SimpleNamespace(
        is_ready=lambda: False,
        get_state=lambda **_kwargs: {"telemetry_error": "connect_failed: serial"},
    )
    components = UmiComponents(
        vive=SimpleNamespace(),
        cameras={},
        angle_trigger=SimpleNamespace(),
        grippers={"left": backend},
        rig=SimpleNamespace(),
        retargeters={},
        hand_state_poller=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="left=connect_failed: serial"):
        components._require_dmtac_ready()


def test_default_config_and_layout_are_umi_specific():
    config = load_umi_collection_config(
        ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    assert config.left.camera_serial == "260522278055"
    assert config.right.camera_serial == "261122273924"
    assert not config.left.uses_identity_tcp_placeholder
    assert not config.right.uses_identity_tcp_placeholder
    assert config.left.tcp_profile == "dmgripper"
    assert config.right.tcp_profile == "dmgripper"
    assert config.left.hand_eye_result.name == "latest_left.json"
    assert config.right.hand_eye_result.name == "latest_right.json"
    assert config.calibration_warning_sec == 600.0
    assert config.calibration_critical_sec == 300.0
    assert config.hand_eye_config.name == "rm65_dual_wrist_umi_v1.json"
    assert config.hand_eye_output_dir.name == "umi_hand_eye"
    assert config.calibration_validity_sec == pytest.approx(3.0 * 3600.0)
    assert config.dmtac_tactile_read_hz == pytest.approx(15.0)
    assert config.dmtac_gripper_command_hz == pytest.approx(20.0)
    assert config.dmtac_gripper_telemetry_hz == pytest.approx(20.0)
    assert config.dmtac_gripper_telemetry_command_grace_s == pytest.approx(0.02)
    assert config.foot_pad_enabled is True
    assert config.v25p_native_shadow is False
    assert config.v25p_native_storage_layout == "record_cjson_v1"
    validate_hand_eye_config_binding(config)
    layout = load_policy_io_layout(config.policy_io_layout)
    assert layout is not None
    assert layout.action_dim == 16
    assert layout.state_dim == 16
    assert layout.arm_reference_frame_by_side == {"left": "desk", "right": "desk"}


def test_umi_v25p_physical_layout_is_explicit_and_shadow_gated() -> None:
    config = load_umi_collection_config(
        ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    with pytest.raises(ValueError, match="requires v25p_native_shadow"):
        replace(
            config,
            v25p_native_storage_layout="parquet_stream_chunks_v1",
        )
    enabled = replace(
        config,
        v25p_native_shadow=True,
        v25p_native_storage_layout="parquet_stream_chunks_v1",
    )
    assert enabled.v25p_native_storage_layout == "parquet_stream_chunks_v1"
    args = build_umi_parser().parse_args(
        [
            "--v25p-native-shadow",
            "--v25p-native-storage-layout",
            "parquet_stream_chunks_v1",
        ]
    )
    assert args.v25p_native_shadow is True
    assert args.v25p_native_storage_layout == "parquet_stream_chunks_v1"


def test_umi_ignore_policy_is_dataset_scoped_and_normalized():
    config = replace(
        load_umi_collection_config(
            ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
        ),
        ignored_camera_keys=("right_wrist", "left_wrist", "right_wrist"),
        ignored_tactile_keys=("right",),
    )

    contract = config.capture_modality_contract
    assert contract.ignored_camera_keys == ("left_wrist", "right_wrist")
    assert contract.ignored_tactile_keys == ("right",)
    assert contract.to_dict()["scope"] == "dataset"

    with pytest.raises(ValueError, match="unknown ignored UMI camera"):
        replace(config, ignored_camera_keys=("front",))


def test_umi_overrides_dmtac_tactile_rate_for_collection():
    from dataclasses import replace

    config = replace(
        load_umi_collection_config(
            ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
        ),
        dmtac_tactile_read_hz=12.5,
        dmtac_gripper_command_hz=18.0,
        dmtac_gripper_telemetry_hz=25.0,
        dmtac_gripper_telemetry_command_grace_s=0.01,
        send_gripper_commands=False,
    )
    backend_configs = {
        side: SimpleNamespace(tactile_read_hz=1.0, send_commands=True)
        for side in ("left", "right")
    }

    _configure_dmtac_backend_configs(config, backend_configs)

    assert {item.tactile_read_hz for item in backend_configs.values()} == {12.5}
    assert {item.command_hz for item in backend_configs.values()} == {18.0}
    assert {item.telemetry_hz for item in backend_configs.values()} == {25.0}
    assert {
        item.gripper_telemetry_command_grace_s for item in backend_configs.values()
    } == {0.01}
    assert not any(item.send_commands for item in backend_configs.values())

    with pytest.raises(ValueError, match="freshness window"):
        replace(config, dmtac_tactile_read_hz=1.0)


def test_umi_gripper_freshness_uses_joint_measurement_timestamp() -> None:
    state = {
        "joint_timestamp": 9.4,
        # A newer worker-loop timestamp must not refresh cached position data.
        "telemetry_timestamp": 10.0,
        "telemetry_snapshot_age_sec": 0.01,
    }
    age, rejection = _gripper_measurement_timing(
        state,
        cutoff_timestamp=10.0,
        max_age_sec=0.5,
    )
    assert age == pytest.approx(0.6)
    assert rejection == "stale_gripper_measurement:0.600"

    age, rejection = _gripper_measurement_timing(
        {"joint_timestamp": 10.1},
        cutoff_timestamp=10.0,
        max_age_sec=0.5,
    )
    assert age == pytest.approx(-0.1)
    assert rejection == "future_gripper_measurement:-0.100"


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


def test_umi_vive_pose_projects_small_float_drift_onto_so3():
    pose = np.eye(4)
    pose[:3, :3] = np.array(
        [
            [0.99996, -0.00801, 0.0],
            [0.00800, 0.99995, 0.00002],
            [0.0, -0.00002, 1.00001],
        ]
    )

    normalized, diagnostics = normalize_vive_tracker_transform(pose)

    np.testing.assert_allclose(
        normalized[:3, :3].T @ normalized[:3, :3], np.eye(3), atol=1e-10
    )
    assert np.linalg.det(normalized[:3, :3]) == pytest.approx(1.0, abs=1e-10)
    assert diagnostics["rotation_projected"] is True
    assert diagnostics["rotation_projection_frobenius"] > 0.0


def test_valid_static_transform_is_projected_before_transform_composition():
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [1.0, 4.0e-6, 0.0],
            [-4.0e-6, 1.0, 3.0e-6],
            [0.0, -3.0e-6, 1.0],
        ]
    )

    normalized = as_rigid_transform(transform, name="static_calibration")

    np.testing.assert_allclose(
        normalized[:3, :3].T @ normalized[:3, :3], np.eye(3), atol=1e-12
    )
    assert np.linalg.det(normalized[:3, :3]) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "rotation",
    [
        np.diag([-1.0, 1.0, 1.0]),
        np.diag([1.08, 1.0, 1.0]),
    ],
)
def test_umi_vive_pose_rejects_reflection_or_large_rotation_distortion(rotation):
    pose = np.eye(4)
    pose[:3, :3] = rotation

    with pytest.raises(ValueError):
        normalize_vive_tracker_transform(pose)


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
        "camera_serial": "260522278055",
        "tracker_name": "tracker_1",
        "tracker_role": "left_wrist",
        "T_vive_world_to_desk": np.eye(4).tolist(),
    }


def _calibration_payload_with_tcp(*, side: str, now: datetime, tx: float) -> dict:
    index = 1 if side == "left" else 2
    payload = _calibration_payload(now)
    payload.update(
        {
            "created_at_utc": now.isoformat(),
            "expires_at_utc": (now + timedelta(minutes=30)).isoformat(),
            "side": side,
            "camera_serial": "260522278055" if side == "left" else "261122273924",
            "tracker_name": f"tracker_{index}",
            "tracker_role": f"{side}_wrist",
            "tcp_extrinsic": {
                "schema": "dexteleop.umi_tcp_extrinsic.v1",
                "side": side,
                "tcp_profile": "dmgripper",
                "T_tcp_to_tracker": np.array(
                    [[1, 0, 0, tx], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    dtype=float,
                ).tolist(),
            },
        }
    )
    return payload


def test_calibration_set_requires_and_loads_independent_left_right_tcp_results(tmp_path):
    now = datetime.now(timezone.utc)
    result_paths = {}
    for side, tx in (("left", 0.11), ("right", -0.22)):
        result_paths[side] = tmp_path / f"{side}.json"
        result_paths[side].write_text(
            json.dumps(_calibration_payload_with_tcp(side=side, now=now, tx=tx))
        )
    config_path = tmp_path / "umi.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_root": str(tmp_path / "dataset"),
                "policy_io_layout": str(
                    ROOT / "teleop_runtime/configs/policy_io_layouts/dual_umi_dmtac_v23.json"
                ),
                "desk_calibration_side": "left",
                "allow_identity_tcp_placeholder": False,
                "sides": {
                    "left": {
                        "camera_serial": "260522278055",
                        "tracker_name": "tracker_1",
                        "hand_eye_result": str(result_paths["left"]),
                        "tcp_profile": "dmgripper",
                    },
                    "right": {
                        "camera_serial": "261122273924",
                        "tracker_name": "tracker_2",
                        "hand_eye_result": str(result_paths["right"]),
                        "tcp_profile": "dmgripper",
                    },
                },
            }
        )
    )
    config = load_umi_collection_config(config_path)
    calibrations = UmiCalibrationSet.load(config)
    assert calibrations.desk.side == "left"
    assert calibrations.by_side["left"].path == result_paths["left"]
    assert calibrations.by_side["right"].path == result_paths["right"]
    assert calibrations.T_tcp_to_tracker_by_side["left"][0, 3] == pytest.approx(0.11)
    assert calibrations.T_tcp_to_tracker_by_side["right"][0, 3] == pytest.approx(-0.22)

    mismatched = json.loads(result_paths["right"].read_text())
    mismatched["tcp_extrinsic"]["tcp_profile"] = "another_tool"
    result_paths["right"].write_text(json.dumps(mismatched))
    with pytest.raises(RuntimeError, match="TCP profile mismatch"):
        UmiCalibrationSet.load(config)


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
            side: {
                "gripper_measured_scalar": 0.25 if side == "left" else 0.75,
                "joint_timestamp": 99.9,
                "joint_age_sec": 0.1,
                "gripper_measurement_timestamp": 99.9,
                "last_sent_command_timestamp": 99.8,
                "last_sent_command_completed_timestamp": 99.82,
                "gripper_read_skipped": True,
                "gripper_read_skip_reason": "recent_command",
                "gripper_last_send_status": "sent",
                "telemetry_snapshot_sequence": index,
                "telemetry_snapshot_source_timestamp": 99.9,
                "telemetry_snapshot_available_timestamp": 99.95,
                "telemetry_snapshot_age_sec": 0.1,
                "telemetry_snapshot_source_age_sec": 0.1,
                "telemetry_snapshot_available_age_sec": 0.05,
                "telemetry_snapshot_source_clock": "local_wall_time",
                "telemetry_snapshot_available_clock": "local_wall_time",
                "tactile_snapshot_sequence": index,
                "tactile_snapshot_source_timestamp": 99.9,
                "tactile_snapshot_available_timestamp": 99.95,
                "tactile_snapshot_age_sec": 0.1,
                "tactile_snapshot_source_age_sec": 0.1,
                "tactile_snapshot_available_age_sec": 0.05,
                "tactile_snapshot_source_clock": "local_wall_time",
                "tactile_snapshot_available_clock": "local_wall_time",
                "causal_snapshot_cutoff_timestamp": 100.0,
                "causal_snapshot_contract": (
                    "source_timestamp<=T0_and_available_timestamp<=T0"
                ),
            }
            for index, side in enumerate(("left", "right"), start=1)
        },
        operator_labels=labels.snapshot(),
    )
    next_poses = {
        side: replace(
            pose,
            raw_tcp_pose7=pose.raw_tcp_pose7 + np.array([0.5, 0, 0, 0, 0, 0, 0]),
            desk_tcp_pose7=pose.desk_tcp_pose7 + np.array([0.5, 0, 0, 0, 0, 0, 0]),
            source_timestamp=100.05,
            available_timestamp=100.08,
            sequence=pose.sequence + 10,
        )
        for side, pose in poses.items()
    }
    aligned_sample = replace(
        sample,
        arm_action_poses=next_poses,
        arm_action_source_frame_index=1,
        arm_action_source_timestamp=100.1,
    )
    encoder = UmiV23RowEncoder()
    row, expectations = encoder.encode_row(
        aligned_sample,
        layout=layout,
        camera_video_paths=None,
        tactile_storage_refs=None,
        inline_tactile=False,
    )
    assert expectations == {}
    assert len(row["action"]) == 16
    assert len(row["observation.state"]) == 16
    assert row["action.left_arm.pose"][0] == pytest.approx(1.5)
    assert row["action.right_arm.pose"][0] == pytest.approx(2.5)
    assert row["observation.left_arm.pose"][0] == pytest.approx(1.0)
    assert row["observation.right_arm.pose"][0] == pytest.approx(2.0)
    assert row["action"][-2:] == pytest.approx([0.2, 0.8])
    assert row["observation.state"][-2:] == pytest.approx([0.25, 0.75])
    assert row["observation.last_action"][:14] == pytest.approx(
        row["observation.state"][:14]
    )
    assert row["action.arm.temporal_alignment"] == "next_accepted_observation"
    assert row["action.arm.source_frame_index"] == 1
    assert row["observation.hand.left.telemetry_snapshot_sequence"] == 1
    assert row["observation.hand.right.tactile_snapshot_sequence"] == 2
    assert row["observation.hand.left.joint_age_sec"] == pytest.approx(0.1)
    assert row["observation.hand.left.gripper_measurement_timestamp"] == pytest.approx(
        99.9
    )
    assert row[
        "observation.hand.left.last_sent_command_completed_timestamp"
    ] == pytest.approx(99.82)
    assert row["observation.hand.left.gripper_read_skipped"] is True
    assert row["observation.hand.left.gripper_read_skip_reason"] == "recent_command"
    following_poses = {
        side: replace(
            pose,
            raw_tcp_pose7=pose.raw_tcp_pose7
            + np.array([0.75, 0, 0, 0, 0, 0, 0]),
            desk_tcp_pose7=pose.desk_tcp_pose7
            + np.array([0.75, 0, 0, 0, 0, 0, 0]),
            source_timestamp=100.15,
            available_timestamp=100.18,
            sequence=pose.sequence + 20,
        )
        for side, pose in poses.items()
    }
    second_row, _ = encoder.encode_row(
        replace(
            sample,
            frame_index=1,
            timestamp=100.1,
            poses=next_poses,
            gripper_commands={"left": 0.3, "right": 0.7},
            arm_action_poses=following_poses,
            arm_action_source_frame_index=2,
            arm_action_source_timestamp=100.2,
        ),
        layout=layout,
        camera_video_paths=None,
        tactile_storage_refs=None,
        inline_tactile=False,
    )
    assert second_row["observation.last_action"] == pytest.approx(row["action"])
    assert second_row["action"][-2:] == pytest.approx([0.3, 0.7])
    assert second_row["action"][:14] != pytest.approx(
        second_row["observation.state"][:14]
    )
    validate_v23_row_causal_snapshots(
        row,
        arm_source_clocks=(
            "vive_sender_monotonic_calibrated_to_local_wall",
            "vive_receiver_wall_fallback",
        ),
    )

    contract_payload = {
        "schema": "dexteleop.umi_hand_eye.result.v1",
        "created_at_utc": "2026-07-17T00:00:00Z",
        "expires_at_utc": "2026-07-17T01:00:00Z",
        "desk_definition": "calibration_board_frame",
        "side": "left",
        "camera_serial": "260522278055",
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
    assert (
        contract.source.metadata["canonical_desk_urdf_frame"]
        == "workspace_near_boundary_frame"
    )
    validate_v23_collection_row_actual_tcp(row, contract=contract)
    stored = bind_collection_row_pose_storage(row, contract=contract)
    assert stored["observation.left_arm.pose"][0] == pytest.approx(11.0)
    assert stored["observation.right_arm.pose"][0] == pytest.approx(12.0)
    assert stored["action.left_arm.pose"][0] == pytest.approx(11.5)
    assert stored["action.right_arm.pose"][0] == pytest.approx(12.5)


def test_umi_recorder_uses_native_encoder_without_arm_sample(tmp_path):
    config = load_umi_collection_config(
        ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(_calibration_payload(datetime.now(timezone.utc))))
    binding = UmiCalibrationBinding.load(calibration_path)
    layout = load_policy_io_layout(config.policy_io_layout)
    assert layout is not None
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
    recorder.note_snapshot_attempt(("right:missing_or_stale_tactile",))
    recorder.note_snapshot_attempt()
    recorder.submit(sample)
    next_poses = {
        side: replace(
            pose,
            raw_tcp_pose7=pose.raw_tcp_pose7 + np.array([0.1, 0, 0, 0, 0, 0, 0]),
            desk_tcp_pose7=pose.desk_tcp_pose7 + np.array([0.1, 0, 0, 0, 0, 0, 0]),
            source_timestamp=10.05,
            available_timestamp=10.08,
            sequence=pose.sequence + 10,
        )
        for side, pose in poses.items()
    }
    recorder.note_snapshot_attempt()
    recorder.submit(
        replace(
            sample,
            frame_index=1,
            timestamp=10.1,
            poses=next_poses,
        )
    )
    recorder.stop_episode(reason="test")
    assert (tmp_path / "dataset/meta/info.json").exists()
    info = json.loads((tmp_path / "dataset/meta/info.json").read_text())
    assert info["embodiment"] == "umi_dual_wrist"
    assert info["embodiment_has_arm"] is False
    episode_record = json.loads(
        (tmp_path / "dataset/meta/episodes.jsonl").read_text().strip()
    )
    assert episode_record["summary"]["sample_attempt_count"] == 3
    assert episode_record["summary"]["sample_rejection_count"] == 1
    assert episode_record["summary"]["source_sample_count"] == 2
    assert episode_record["summary"]["terminal_source_sample_dropped"] is True
    assert episode_record["frame_count"] == 1
    assert episode_record["summary"]["sample_rejection_counts_by_reason"] == {
        "right:missing_or_stale_tactile": 1
    }

    shutil.rmtree(tmp_path / "dataset")
    assert recorder.start_episode({"after_external_delete": True}) == 0
    recorder.stop_episode(reason="empty_after_external_delete")
    recorder.close()
    assert (tmp_path / "dataset/meta/info.json").is_file()


def test_umi_recorder_isolates_direct_stale_measured_state(tmp_path):
    config = load_umi_collection_config(
        ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(_calibration_payload(datetime.now(timezone.utc)))
    )
    binding = UmiCalibrationBinding.load(calibration_path)
    layout = load_policy_io_layout(config.policy_io_layout)
    assert layout is not None
    config = replace(config, dataset_root=tmp_path / "stale_state_dataset")
    recorder = UmiDatasetRecorder(
        config=config,
        calibration=binding,
        policy_io_layout=layout,
    )
    recorder.writer.config.write_parquet = False
    episode = recorder.start_episode({"test": "stale_state_isolation"})

    def sample(frame_index: int, timestamp: float, joint_timestamp: float) -> UmiSample:
        poses = {
            side: UmiPoseMeasurement(
                side=side,
                raw_tcp_pose7=np.array(
                    [timestamp, 0, 0, 0, 0, 0, 1], dtype=float
                ),
                desk_tcp_pose7=np.array(
                    [timestamp, 0, 0, 0, 0, 0, 1], dtype=float
                ),
                source_timestamp=timestamp,
                available_timestamp=timestamp,
                source_clock="vive_receiver_wall_fallback",
                available_clock="local_wall_time",
                sequence=frame_index + 1,
                tracker_name=f"tracker_{side}",
                tracker_role=f"{side}_wrist",
                age_sec=0.0,
            )
            for side in ("left", "right")
        }
        return UmiSample(
            episode_index=episode,
            frame_index=frame_index,
            timestamp=timestamp,
            poses=poses,
            gripper_commands={"left": 0.1, "right": 0.9},
            hand_states={
                "left": {
                    "gripper_measured_scalar": 0.1,
                    "joint_timestamp": joint_timestamp,
                },
                "right": {
                    "gripper_measured_scalar": 0.9,
                    "joint_timestamp": joint_timestamp,
                },
            },
            operator_labels=OperatorLabelStateMachine().snapshot(),
        )

    assert recorder.submit(sample(0, 10.0, 9.0)) is False
    assert recorder.recording
    assert recorder.frame_count == 0
    assert recorder.next_frame_index == 0
    assert recorder.stats()["sample_attempt_count"] == 1
    assert recorder.stats()["sample_rejection_count"] == 1
    assert recorder.stats()["sample_rejection_counts_by_reason"] == {
        "left:stale_gripper_measurement:1.000": 1
    }

    assert recorder.submit(sample(0, 10.0, 9.9)) is True
    assert recorder.submit(sample(1, 10.1, 10.0)) is True
    recorder.stop_episode(reason="test")
    assert recorder.last_outcome == "saved"
    recorder.close()


def test_umi_recorder_discards_empty_episode(tmp_path):
    config = load_umi_collection_config(
        ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(_calibration_payload(datetime.now(timezone.utc))))
    binding = UmiCalibrationBinding.load(calibration_path)
    layout = load_policy_io_layout(config.policy_io_layout)
    assert layout is not None
    from dataclasses import replace

    config = replace(config, dataset_root=tmp_path / "empty_dataset")
    recorder = UmiDatasetRecorder(config=config, calibration=binding, policy_io_layout=layout)
    recorder.writer.config.write_parquet = False
    episode = recorder.start_episode()
    assert recorder.stop_episode(reason="no_ready_samples") == episode
    assert recorder.last_outcome == "discarded"
    assert recorder.last_frame_count == 0
    recorder.close()
    episodes_path = tmp_path / "empty_dataset/meta/episodes.jsonl"
    assert not episodes_path.exists() or not episodes_path.read_text().strip()


def test_umi_recorder_does_not_align_arm_action_across_rejected_gap(tmp_path):
    config = load_umi_collection_config(
        ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(_calibration_payload(datetime.now(timezone.utc)))
    )
    binding = UmiCalibrationBinding.load(calibration_path)
    layout = load_policy_io_layout(config.policy_io_layout)
    assert layout is not None
    config = replace(config, dataset_root=tmp_path / "gap_dataset")
    recorder = UmiDatasetRecorder(
        config=config,
        calibration=binding,
        policy_io_layout=layout,
    )
    recorder.writer.config.write_parquet = False
    episode = recorder.start_episode({"test": "continuity_gap"})

    def sample(frame_index: int, timestamp: float, x: float) -> UmiSample:
        poses = {
            side: UmiPoseMeasurement(
                side=side,
                raw_tcp_pose7=np.array([x, 0, 0, 0, 0, 0, 1], dtype=float),
                desk_tcp_pose7=np.array([x, 0, 0, 0, 0, 0, 1], dtype=float),
                source_timestamp=timestamp,
                available_timestamp=timestamp,
                source_clock="vive_receiver_wall_fallback",
                available_clock="local_wall_time",
                sequence=frame_index + 1,
                tracker_name=f"tracker_{side}",
                tracker_role=f"{side}_wrist",
                age_sec=0.0,
            )
            for side in ("left", "right")
        }
        return UmiSample(
            episode_index=episode,
            frame_index=frame_index,
            timestamp=timestamp,
            poses=poses,
            gripper_commands={"left": 0.1, "right": 0.9},
            hand_states={
                "left": {"gripper_measured_scalar": 0.1, "joint_timestamp": timestamp},
                "right": {"gripper_measured_scalar": 0.9, "joint_timestamp": timestamp},
            },
            operator_labels=OperatorLabelStateMachine().snapshot(),
        )

    recorder.submit(sample(0, 10.0, 0.0))
    assert recorder.break_action_continuity(reason="snapshot_rejected:test") is True
    # The first source after recovery reuses output frame 0.  It must not be
    # paired with the pre-gap source observation.
    recorder.submit(sample(recorder.next_frame_index, 11.0, 1.0))
    recorder.submit(sample(recorder.next_frame_index, 11.1, 1.1))
    recorder.stop_episode(reason="test")

    episode_record = json.loads(
        (tmp_path / "gap_dataset/meta/episodes.jsonl").read_text().strip()
    )
    assert episode_record["frame_count"] == 1
    assert episode_record["summary"]["source_sample_count"] == 3
    assert episode_record["summary"]["action_continuity_break_count"] == 1
    assert episode_record["summary"]["continuity_dropped_source_sample_count"] == 1
    assert episode_record["summary"]["action_continuity_break_reasons"] == {
        "snapshot_rejected:test": 1
    }
    recorder.close()


def test_umi_recorder_does_not_align_arm_action_across_long_accepted_gap(tmp_path):
    config = load_umi_collection_config(
        ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(_calibration_payload(datetime.now(timezone.utc)))
    )
    binding = UmiCalibrationBinding.load(calibration_path)
    layout = load_policy_io_layout(config.policy_io_layout)
    assert layout is not None
    config = replace(config, dataset_root=tmp_path / "long_gap_dataset")
    recorder = UmiDatasetRecorder(
        config=config,
        calibration=binding,
        policy_io_layout=layout,
    )
    recorder.writer.config.write_parquet = False
    episode = recorder.start_episode()

    def sample(frame_index: int, timestamp: float) -> UmiSample:
        poses = {
            side: UmiPoseMeasurement(
                side=side,
                raw_tcp_pose7=np.array([timestamp, 0, 0, 0, 0, 0, 1], dtype=float),
                desk_tcp_pose7=np.array([timestamp, 0, 0, 0, 0, 0, 1], dtype=float),
                source_timestamp=timestamp,
                available_timestamp=timestamp,
                source_clock="vive_receiver_wall_fallback",
                available_clock="local_wall_time",
                sequence=max(1, int(timestamp * 10)),
                tracker_name=f"tracker_{side}",
                tracker_role=f"{side}_wrist",
                age_sec=0.0,
            )
            for side in ("left", "right")
        }
        return UmiSample(
            episode_index=episode,
            frame_index=frame_index,
            timestamp=timestamp,
            poses=poses,
            gripper_commands={"left": 0.1, "right": 0.9},
            hand_states={
                "left": {"gripper_measured_scalar": 0.1, "joint_timestamp": timestamp},
                "right": {"gripper_measured_scalar": 0.9, "joint_timestamp": timestamp},
            },
            operator_labels=OperatorLabelStateMachine().snapshot(),
        )

    recorder.submit(sample(0, 10.0))
    recorder.submit(sample(1, 11.0))
    assert recorder.next_frame_index == 1
    recorder.submit(sample(1, 11.1))
    recorder.stop_episode(reason="test")

    episode_record = json.loads(
        (tmp_path / "long_gap_dataset/meta/episodes.jsonl").read_text().strip()
    )
    assert episode_record["frame_count"] == 1
    assert episode_record["summary"]["action_continuity_break_reasons"] == {
        "accepted_sample_gap_exceeded": 1
    }
    recorder.close()


def _controller_for_logic_tests(*, ui, recorder, assembler):
    timing = SimpleNamespace(sample_hz=15.0, control_hz=30.0)
    return UmiCollectionController(
        config=SimpleNamespace(timing=timing),
        calibration=SimpleNamespace(),
        components=SimpleNamespace(foot_pad=None),
        recorder=recorder,
        ui=ui,
        operator_policy=DEFAULT_STATION_UMI_OPERATOR_POLICY,
        labels=OperatorLabelStateMachine(),
        snapshot_assembler=assembler,
    )


def test_umi_foot_pad_test_mode_publishes_mapping_without_mutating_labels():
    published = []
    messages = []

    class FakeUi:
        def publish(self, **updates):
            published.append(updates)

        def append_message(self, tag, message):
            messages.append((tag, message))

    foot_pad = SimpleNamespace(is_online=True, foot_pad_count=4)
    controller = _controller_for_logic_tests(
        ui=FakeUi(),
        recorder=SimpleNamespace(recording=False),
        assembler=SimpleNamespace(),
    )
    controller.components.foot_pad = foot_pad
    controller._last_foot_pad_online = True

    controller._publish_foot_pad_test_state()
    assert published[-1]["special_arm_control_mode"] == "foot-pad"
    assert published[-1]["foot_pad_available"] is True
    assert published[-1]["foot_pad_count"] == 4
    assert "foot_pad_test_start" in published[-1]["active_controls"]
    assert "fix_arm_toggle" in published[-1]["active_controls"]
    assert "init_arm_toggle" in published[-1]["active_controls"]

    controller._handle_events(
        [("foot_pad_test_start", "web_gui", 1.0)],
        gripper_ready=True,
        gripper_reason="",
    )
    assert controller._foot_pad_test_active is True
    assert "foot_pad_test_stop" in published[-1]["active_controls"]
    assert "fix_arm_toggle" in published[-1]["active_controls"]
    assert "init_arm_toggle" in published[-1]["active_controls"]

    controller._handle_events(
        [("init_arm_right", "foot_pad", 2.0)],
        gripper_ready=True,
        gripper_reason="",
    )
    assert controller._foot_pad_test_last_event == "踏板 4 / F16 → 右臂归位"
    assert controller.labels.snapshot().labels_by_side["right"] == LABEL_TELEOP
    assert messages and "未写入 UMI 标注" in messages[-1][1]

    controller._handle_events(
        [("foot_pad_test_stop", "web_gui", 3.0)],
        gripper_ready=True,
        gripper_reason="",
    )
    assert controller._foot_pad_test_active is False
    assert "foot_pad_test_start" in published[-1]["active_controls"]


def test_umi_two_pedal_mode_keeps_manual_init_controls_and_reconnects_to_manual():
    published = []

    class FakeUi:
        def consume_events(self):
            return []

        def publish(self, **updates):
            published.append(updates)

    foot_pad = SimpleNamespace(is_online=True, foot_pad_count=2, consume_events=lambda: [])
    controller = _controller_for_logic_tests(
        ui=FakeUi(),
        recorder=SimpleNamespace(recording=False),
        assembler=SimpleNamespace(),
    )
    controller.components.foot_pad = foot_pad
    controller._last_foot_pad_online = True
    controls = controller._active_controls()
    assert "g" in controls
    assert "init_arm_toggle" in controls
    assert "c" in controls
    assert "fix_arm_toggle" in controls

    controller._start_foot_pad_test()
    assert controller._foot_pad_test_active is True
    foot_pad.is_online = False
    controller._events()
    assert controller._foot_pad_test_active is False
    assert published[-1]["special_arm_control_mode"] == "manual"
    assert published[-1]["foot_pad_available"] is False
    assert "fix_arm_toggle" in published[-1]["active_controls"]


def test_umi_quit_event_does_not_run_an_extra_control_cycle():
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(consume_events=lambda: ["q"]),
        recorder=SimpleNamespace(),
        assembler=SimpleNamespace(),
    )

    def unexpected_update(_now):
        raise AssertionError("quit must be handled before gripper dispatch")

    controller._update_grippers = unexpected_update
    controller.run()
    assert controller.should_quit


def test_umi_device_retry_does_not_sample_or_republish_after_acceptance():
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(consume_events=lambda: ["device_retry"]),
        recorder=SimpleNamespace(recording=False),
        assembler=SimpleNamespace(),
    )
    controller._update_grippers = lambda _now: (True, "")
    controller.calibration.require_valid = lambda _config: None
    controller._sample_if_due = lambda _now: (_ for _ in ()).throw(
        AssertionError("device retry must transfer lifecycle ownership before sampling")
    )
    controller._publish = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("device retry must not republish ordinary controls")
    )

    outcome = controller.run()

    assert outcome == "retry_devices"
    assert controller.should_quit


def test_umi_calibration_warning_has_friendly_warning_critical_and_expired_levels():
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(),
        recorder=SimpleNamespace(),
        assembler=SimpleNamespace(),
    )
    controller.config.calibration_warning_sec = 600.0
    controller.config.calibration_critical_sec = 300.0
    now = datetime(2026, 7, 19, 10, 0, tzinfo=timezone.utc)

    controller.calibration = SimpleNamespace(expires_at=now + timedelta(seconds=601))
    assert controller._calibration_validity_warning(now=now)["active"] is False

    controller.calibration = SimpleNamespace(expires_at=now + timedelta(seconds=600))
    warning = controller._calibration_validity_warning(now=now)
    assert warning["level"] == "warning"
    assert warning["countdown"] == "10:00"

    controller.calibration = SimpleNamespace(expires_at=now + timedelta(seconds=300))
    critical = controller._calibration_validity_warning(now=now)
    assert critical["level"] == "critical"
    assert critical["countdown"] == "05:00"

    controller.calibration = SimpleNamespace(expires_at=now)
    expired = controller._calibration_validity_warning(now=now)
    assert expired["level"] == "expired"
    assert "重新标定左右 UMI" in expired["message"]


def test_umi_expiry_finalizes_episode_and_keeps_ui_alive_for_message():
    stopped = []
    published = []
    recorder = SimpleNamespace(recording=True)

    def stop_episode(*, reason):
        stopped.append(reason)
        recorder.recording = False

    recorder.stop_episode = stop_episode
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(consume_events=lambda: []),
        recorder=recorder,
        assembler=SimpleNamespace(),
    )
    controller.calibration = SimpleNamespace(
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        require_valid=lambda _config: (_ for _ in ()).throw(RuntimeError("expired")),
    )

    def publish_once(*_args, **_kwargs):
        published.append(controller._event_message)
        controller.should_quit = True

    controller._publish = publish_once
    controller.run()
    assert stopped == ["calibration_expired"]
    assert controller._calibration_expired is True
    assert published and "标定已过期" in published[0]
    assert controller._event_status == "error"


def test_umi_start_is_rejected_before_opening_episode_when_snapshot_not_ready():
    starts = []
    recorder = SimpleNamespace(
        recording=False,
        start_episode=lambda metadata: starts.append(metadata),
    )
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(),
        recorder=recorder,
        assembler=SimpleNamespace(
            assemble=lambda **_kwargs: SnapshotRejection(
                timestamp=1.0,
                reasons=("left:missing_vive",),
            )
        ),
    )
    controller.calibration = SimpleNamespace(
        require_valid=lambda _config: None,
    )
    controller._handle_events(
        [("episode_start", "web_gui", 1.0)],
        gripper_ready=True,
        gripper_reason="",
    )
    assert starts == []
    assert controller.last_rejection == ("left:missing_vive",)
    assert controller._event_status == "ready"


def test_umi_start_request_reports_gripper_blocker_instead_of_being_disabled():
    starts = []
    recorder = SimpleNamespace(
        recording=False,
        start_episode=lambda metadata: starts.append(metadata),
    )
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(),
        recorder=recorder,
        assembler=SimpleNamespace(),
    )

    controller._handle_events(
        [("episode_start", "web_gui", 1.0)],
        gripper_ready=False,
        gripper_reason="missing_angle_trigger",
    )

    assert starts == []
    assert controller._event_message == "cannot start: missing_angle_trigger"


def test_umi_start_runtime_error_is_reported_without_killing_gui():
    config = load_umi_collection_config(
        ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    messages = []
    recorder = SimpleNamespace(
        recording=False,
        start_episode=lambda _metadata: (_ for _ in ()).throw(
            OSError("simulated disk admission failure")
        ),
    )
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(
            append_message=lambda tag, message: messages.append((tag, message))
        ),
        recorder=recorder,
        assembler=SimpleNamespace(assemble=lambda **_kwargs: object(), config=config),
    )
    controller.config = config
    controller.calibration = SimpleNamespace(
        require_valid=lambda _config: None,
        sha256="sha256:test",
        T_tcp_to_tracker_by_side={
            "left": np.eye(4),
            "right": np.eye(4),
        },
        by_side={},
    )

    controller._handle_events(
        [("episode_start", "web_gui", 1.0)],
        gripper_ready=True,
        gripper_reason="",
    )

    assert recorder.recording is False
    assert controller._event_status == "error"
    assert controller.last_rejection == ("episode_start_failed:OSError",)
    assert "simulated disk admission failure" in controller._event_message
    assert messages[-1][0] == "Dataset"


def test_umi_finalize_runtime_error_keeps_episode_open_and_allows_retry():
    messages = []
    recorder = SimpleNamespace(
        recording=True,
        stop_episode=lambda **_kwargs: (_ for _ in ()).throw(
            OSError("simulated finalize failure")
        ),
    )
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(
            publish=lambda **_kwargs: None,
            append_message=lambda tag, message: messages.append((tag, message)),
        ),
        recorder=recorder,
        assembler=SimpleNamespace(),
    )

    controller._handle_events(
        [("episode_stop", "web_gui", 1.0)],
        gripper_ready=True,
        gripper_reason="",
    )

    assert recorder.recording is True
    assert controller._event_status == "error"
    assert controller.last_rejection == ("episode_finalize_failed:OSError",)
    assert "可重试停止" in controller._event_message
    assert messages[-1][0] == "Dataset"


def test_umi_delete_request_during_recording_is_rejected_explicitly():
    delete_calls = []
    recorder = SimpleNamespace(
        recording=True,
        delete_episode=lambda episode_index: delete_calls.append(episode_index),
    )
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(),
        recorder=recorder,
        assembler=SimpleNamespace(),
    )

    controller._handle_events(
        [("episode_delete:3", "web_gui", 1.0)],
        gripper_ready=True,
        gripper_reason="",
    )

    assert delete_calls == []
    assert controller._event_status == "error"
    assert "当前仍在录制" in controller._event_message


def test_umi_episode_start_payload_switches_idle_dataset_root(tmp_path):
    config = replace(
        load_umi_collection_config(
            ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
        ),
        dataset_root=tmp_path / "before",
    )
    selected = tmp_path / "selected"
    switched = []
    recorder = SimpleNamespace(
        recording=False,
        reconfigure_dataset_root=lambda root: switched.append(Path(root)),
    )
    assembler = SimpleNamespace(config=config)
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(), recorder=recorder, assembler=assembler
    )
    controller.config = config

    controller._handle_events(
        [(
            "episode_start:"
            + json.dumps({"dataset_base_root": str(selected)}),
            "web_gui",
            1.0,
        )],
        gripper_ready=False,
        gripper_reason="test_not_ready",
    )

    assert switched == [selected]
    assert controller.config.dataset_root == selected
    assert assembler.config.dataset_root == selected


def test_umi_task_select_switches_idle_dataset_root_without_starting(tmp_path):
    config = replace(
        load_umi_collection_config(
            ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
        ),
        dataset_root=tmp_path / "before",
    )
    selected = tmp_path / "selected"
    switched = []
    recorder = SimpleNamespace(
        recording=False,
        reconfigure_config=lambda next_config: switched.append(next_config),
    )
    assembler = SimpleNamespace(config=config)
    ui_messages = []
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(
            append_message=lambda tag, message: ui_messages.append((tag, message))
        ),
        recorder=recorder,
        assembler=assembler,
    )
    controller.config = config

    controller._handle_events(
        [(
            "task_select:"
            + json.dumps(
                {
                    "dataset_base_root": str(selected),
                    # UMI task semantics are runtime-owned and these GUI
                    # fields must not be able to replace them.
                    "task_key": "another_task",
                    "task_english_name": "Another Task",
                }
            ),
            "web_gui",
            1.0,
        )],
        gripper_ready=True,
        gripper_reason="",
    )

    assert len(switched) == 1
    assert switched[0].dataset_root == selected
    assert switched[0].task_id == config.task_id
    assert controller.config.dataset_root == selected
    assert assembler.config.dataset_root == selected
    assert controller._event_status == "ready"
    assert ui_messages[-1][0] == "Dataset"


def test_umi_task_select_fails_closed_while_recording(tmp_path):
    config = replace(
        load_umi_collection_config(
            ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
        ),
        dataset_root=tmp_path / "before",
    )
    recorder = SimpleNamespace(recording=True)
    assembler = SimpleNamespace(config=config)
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(),
        recorder=recorder,
        assembler=assembler,
    )
    controller.config = config

    controller._handle_events(
        [(
            "task_select:"
            + json.dumps({"dataset_base_root": str(tmp_path / "selected")}),
            "web_gui",
            1.0,
        )],
        gripper_ready=True,
        gripper_reason="",
    )

    assert controller.config.dataset_root == tmp_path / "before"
    assert assembler.config.dataset_root == tmp_path / "before"
    assert controller._event_status == "error"
    assert "while an episode is recording" in controller._event_message


def test_umi_native_shadow_disables_root_switch_but_allows_same_binding(tmp_path):
    config = replace(
        load_umi_collection_config(
            ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
        ),
        dataset_root=tmp_path / "current",
    )
    applied = []
    recorder = SimpleNamespace(
        recording=False,
        v25p_sidecar_store=None,
        v25p_native_shadow=object(),
        reconfigure_config=lambda next_config: applied.append(next_config),
    )
    assembler = SimpleNamespace(config=config)
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(),
        recorder=recorder,
        assembler=assembler,
    )
    controller.config = config

    assert "task_select" not in controller._active_controls()
    controller._apply_dataset_configuration(
        {"dataset_base_root": str(config.dataset_root)},
        include_modality_selection=False,
    )
    assert len(applied) == 1

    with pytest.raises(RuntimeError, match="native shadow"):
        controller._apply_dataset_configuration(
            {"dataset_base_root": str(tmp_path / "other")},
            include_modality_selection=False,
        )


def test_umi_expired_state_executes_admitted_dataset_rebind(tmp_path):
    config = replace(
        load_umi_collection_config(
            ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
        ),
        dataset_root=tmp_path / "before",
    )
    selected = tmp_path / "selected"
    applied = []
    event_batches = iter(
        [[
            "task_select:"
            + json.dumps({"dataset_base_root": str(selected)})
        ], []]
    )
    recorder = SimpleNamespace(
        recording=False,
        last_outcome="ready",
        reconfigure_config=lambda next_config: applied.append(next_config),
    )
    assembler = SimpleNamespace(config=config)
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(
            consume_events=lambda: next(event_batches, []),
            append_message=lambda *_args: None,
        ),
        recorder=recorder,
        assembler=assembler,
    )
    controller.config = config
    controller.calibration = SimpleNamespace(
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        require_valid=lambda _config: (_ for _ in ()).throw(RuntimeError("expired")),
    )

    def publish_once(*_args, **_kwargs):
        controller.should_quit = True

    controller._publish = publish_once
    controller.run()

    assert len(applied) == 1
    assert applied[0].dataset_root == selected
    assert controller.config.dataset_root == selected
    assert assembler.config.dataset_root == selected


def test_umi_dataset_switch_rejects_progress_profiled_target(tmp_path):
    profile = load_task_progress_profile_file(
        ROOT
        / "collection_runtime/progress/profiles/linear/"
        "example_three_stage_pick_place.progress.json"
    )
    bind_dataset_progress_profile(tmp_path, profile)

    with pytest.raises(ValueError, match="collection progress profile"):
        _validate_umi_dataset_admission(tmp_path, task_id="umi_collection")


def test_umi_dataset_switch_rejects_populated_different_task(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "episode_count": 1,
                "frame_count": 10,
                "task_id": "another_task",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="task binding differs"):
        _validate_umi_dataset_admission(tmp_path, task_id="umi_collection")


def test_umi_recalibration_event_exits_session_without_open_episode():
    recorder = SimpleNamespace(recording=False)
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(consume_events=lambda: ["umi_recalibrate"]),
        recorder=recorder,
        assembler=SimpleNamespace(),
    )
    assert controller.run() == "recalibrate"
    assert controller.recalibration_requested is True


def test_angle_recalibration_event_exits_session_into_gui_angle_flow():
    recorder = SimpleNamespace(recording=False)
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(consume_events=lambda: ["angle_recalibrate"]),
        recorder=recorder,
        assembler=SimpleNamespace(),
    )
    assert controller.run() == "recalibrate_angle"
    assert controller.angle_recalibration_requested is True


def test_umi_sample_scheduler_never_emits_idle_catch_up_burst(monkeypatch):
    submitted = []
    recorder = SimpleNamespace(
        recording=True,
        episode_index=3,
        frame_count=0,
        submit=lambda sample: submitted.append(sample),
    )
    assembler = SimpleNamespace(assemble=lambda **kwargs: kwargs)
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(), recorder=recorder, assembler=assembler
    )
    controller._next_sample_at = 0.0
    clock = iter((100.0, 100.01))
    monkeypatch.setattr(
        "teleop_runtime.collection.umi.controller.time.perf_counter",
        lambda: next(clock),
    )
    controller._sample_if_due(200.0)
    controller._sample_if_due(200.01)
    assert len(submitted) == 1
    assert controller._next_sample_at == pytest.approx(100.0 + 1.0 / 15.0)


def test_umi_sample_rejections_are_counted_without_closing_episode(monkeypatch):
    attempts = []
    recorder = SimpleNamespace(
        recording=True,
        episode_index=3,
        frame_count=0,
        note_snapshot_attempt=lambda reasons=(): attempts.append(tuple(reasons)),
        submit=lambda _sample: pytest.fail("rejected sample must not be submitted"),
    )
    rejection = SnapshotRejection(
        timestamp=200.0,
        reasons=("right:missing_or_stale_tactile",),
    )
    controller = _controller_for_logic_tests(
        ui=SimpleNamespace(),
        recorder=recorder,
        assembler=SimpleNamespace(assemble=lambda **_kwargs: rejection),
    )
    controller._next_sample_at = 0.0
    monkeypatch.setattr(
        "teleop_runtime.collection.umi.controller.time.perf_counter",
        lambda: 100.0,
    )

    controller._sample_if_due(200.0)

    assert attempts == [("right:missing_or_stale_tactile",)]
    assert controller.last_rejection == ("right:missing_or_stale_tactile",)


def test_umi_gui_reports_selected_label_side_and_no_send_truthfully():
    from dataclasses import replace

    config = replace(
        load_umi_collection_config(
            ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
        ),
        send_gripper_commands=False,
    )
    published = {}
    gripper = SimpleNamespace(
        is_ready=lambda: True,
        config=SimpleNamespace(speed=1.0, torque=2.0),
    )
    recorder = SimpleNamespace(
        recording=False,
        frame_count=0,
        episode_index=None,
        last_episode_index=4,
        last_frame_count=12,
        last_message="saved",
        effective_sample_hz=lambda now: None,
        stats=lambda: {},
    )
    resource = SimpleNamespace(start=lambda: None, close=lambda: None)
    components = UmiComponents(
        foot_pad=None,
        vive=SimpleNamespace(
            start=lambda: None,
            close=lambda: None,
            get_latest_channels_at_or_before=lambda _now: {},
        ),
        hand_state_poller=SimpleNamespace(
            start=lambda: None,
            close=lambda: None,
            latest_at_or_before=lambda _now: {},
        ),
        angle_trigger=SimpleNamespace(
            start=lambda: None,
            close=lambda: None,
            get_latest_channels=lambda: {},
        ),
        cameras={},
        grippers={"left": gripper, "right": gripper},
        rig=resource,
        retargeters={},
    )
    labels = OperatorLabelStateMachine()
    labels.set("left", "fixed_hold", source="test", timestamp=1.0)
    controller = UmiCollectionController(
        config=config,
        calibration=SimpleNamespace(
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            path=Path("calibration.json"),
            sha256="sha256:test",
        ),
        components=components,
        recorder=recorder,
        ui=SimpleNamespace(publish=lambda **kwargs: published.update(kwargs)),
        operator_policy=DEFAULT_STATION_UMI_OPERATOR_POLICY,
        labels=labels,
        snapshot_assembler=SimpleNamespace(),
    )
    controller._selected_label_side["fix"] = "right"
    controller._publish(time.time(), gripper_ready=True, gripper_reason="")
    assert published["fix_arm_selected_side"] == "right"
    assert published["fix_arm_fixed_side"] == ""
    assert published["fix_arm_active"] is False
    assert published["gripper_control_enabled"] is False
    assert published["gripper_status"]["left"]["q6_command"]["sent"] is False
    assert published["episode_last_episode_index"] == 4
    assert published["dataset_root"] == str(config.dataset_root)
    assert "angle_recalibrate" in published["active_controls"]
    controller._publish(
        time.time(),
        gripper_ready=False,
        gripper_reason="invalid_or_stale_angle_trigger:left",
    )
    assert "episode_start" in published["active_controls"]
    assert "episode_delete" in published["active_controls"]
    assert published["episode_event_message"] == "invalid_or_stale_angle_trigger:left"


def test_umi_components_do_not_reference_arm_backends():
    source = (ROOT / "teleop_runtime/collection/umi/components.py").read_text()
    assert "RM65" not in source
    assert "backends.arms" not in source
    assert "FixArmCommandController" not in source
    assert "InitArmCommandController" not in source


def test_existing_formal_gui_renders_umi_title_without_second_web_server():
    html = _render_initial_html({"ui_mode": "umi_collection", "stage": "ready"})
    packaged_html = (
        ROOT / "control_station/web/static/calibration_web_ui.html"
    ).read_text(encoding="utf-8")
    packaged_fallback = (
        ROOT / "control_station/web/static/calibration_web_ui_fallback.js"
    ).read_text(encoding="utf-8")
    assert "DexTeleop UMI 数采" in html
    assert "collection_web_ui" not in (
        ROOT / "teleop_runtime/collection/umi/controller.py"
    ).read_text()
    assert 'state.ui_mode || "") === "umi_collection"' in (
        packaged_html + packaged_fallback
    )
    assert 'id="episodeWarningTitle"' in html
    assert "calibration_validity_warning" in html
    assert "calibration-warning.critical" in html
    assert 'id="umiRecalibrateButton"' in html
    assert 'data-event="umi_recalibrate"' in html
    assert 'data-event="angle_recalibrate"' in html
    assert 'data-event="angle_calibration_next"' in html
    assert "等待有效快照" in html
    assert "采样尝试/接受/拒绝" in html
    assert "episode_attempted_sample_count" in (
        ROOT / "control_station/web/calibration_web_ui.py"
    ).read_text()


def test_recalibration_rotates_populated_dataset_with_old_calibration(tmp_path):
    config = load_umi_collection_config(
        ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    from dataclasses import replace

    root = tmp_path / "dataset"
    config = replace(config, dataset_root=root)
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "calibration_sha256": "sha256:old",
                "tcp_extrinsic_by_side": {
                    "left": {"hand_eye_result_sha256": "sha256:old-left"},
                    "right": {"hand_eye_result_sha256": "sha256:old-right"},
                },
            }
        )
    )
    calibrations = SimpleNamespace(
        sha256="sha256:new-left",
        created_at=datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc),
        by_side={
            "left": SimpleNamespace(sha256="sha256:new-left"),
            "right": SimpleNamespace(sha256="sha256:new-right"),
        },
    )
    resolved = config_for_calibration_dataset(config, calibrations)
    assert resolved.dataset_root.parent == root.parent
    assert resolved.dataset_root.name.startswith("dataset__calib_20260719T120000Z_")
    assert resolved.dataset_root != root


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
                    "left": {"camera_serial": "260522278055", "tracker_name": "tracker_1"},
                    "right": {"camera_serial": "261122273924", "tracker_name": "tracker_2"},
                },
            }
        )
    )
    assert umi_cli_main(["--config", str(config_path), "--validate-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["action_dim"] == 16
    assert not (tmp_path / "dataset").exists()


def test_runtime_cli_defers_expired_calibration_to_gui_flow(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "teleop_runtime.collection.umi.calibration.UmiCalibrationSet.load",
        lambda _config: (_ for _ in ()).throw(AssertionError("eager calibration load")),
    )
    monkeypatch.setattr(
        "teleop_runtime.collection.umi.app.run_umi_collection",
        lambda config: calls.append(config) or 0,
    )
    assert umi_cli_main(
        [
            "--config",
            str(ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"),
            "--no-open-browser",
        ]
    ) == 0
    assert len(calls) == 1


def test_runtime_cli_can_override_new_calibration_validity_in_hours(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "teleop_runtime.collection.umi.app.run_umi_collection",
        lambda config: calls.append(config) or 0,
    )

    assert umi_cli_main(
        [
            "--config",
            str(ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"),
            "--calibration-validity-hours",
            "5.5",
            "--no-open-browser",
        ]
    ) == 0
    assert calls[0].calibration_validity_sec == pytest.approx(5.5 * 3600.0)


def test_runtime_cli_treats_keyboard_interrupt_as_clean_operator_cancel(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        "teleop_runtime.collection.umi.app.run_umi_collection",
        lambda _config: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    result = umi_cli_main(
        [
            "--config",
            str(ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"),
            "--no-open-browser",
        ]
    )

    assert result == 130
    assert "Ctrl+C" in capsys.readouterr().out


def test_umi_config_rejects_ambiguous_calibration_validity_units(tmp_path):
    payload = json.loads(
        (ROOT / "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json").read_text()
    )
    payload["calibration_validity_sec"] = 3600.0
    config_path = tmp_path / "ambiguous.json"
    config_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="configure only one"):
        load_umi_collection_config(config_path)
