"""Formal UMI collection application assembly."""

from __future__ import annotations

import time

from dataset_contracts.v25p_native_storage import V25P_RECORD_CJSON_LAYOUT_V1
from robot_motion.runtime.lifecycle import run_best_effort_cleanup

from ...configs.dmtac_hardware import prepare_dmtac_hardware_assignment_before_gui
from ...gui.calibration_web_ui import CalibrationWebUi
from ...gui.startup_progress import GuiStartupProgress
from ..operator_labels import OperatorLabelStateMachine
from ..policy_io import load_policy_io_layout
from .calibration import UmiCalibrationSet
from .angle_trigger_calibration import ensure_angle_trigger_ready
from .components import build_umi_components
from .config import UmiCollectionConfig
from .controller import UmiCollectionController
from .pose_adapter import UmiPoseAdapter
from .recorder import UmiDatasetRecorder
from .recalibration import (
    config_for_calibration_dataset,
    run_default_dual_calibration,
    validate_hand_eye_config_binding,
)
from .snapshot import UmiSnapshotAssembler


def run_umi_collection(config: UmiCollectionConfig) -> int:
    layout = load_policy_io_layout(config.policy_io_layout)
    if layout is None:
        raise RuntimeError("UMI collection requires a policy I/O layout")
    ui = CalibrationWebUi(
        host=config.gui_host,
        port=config.gui_port,
        open_browser=config.open_browser,
    )
    try:
        ui.publish(
            ui_mode="umi_collection",
            stage="UMI 设备连接中",
            control_hint="正在尝试全部配置设备；单项失败不会阻止进入 GUI。",
        )
        ui.start()
        assignment_progress = GuiStartupProgress(
            ui,
            stage_prefix="UMI 设备连接中",
        )
        assignment_progress.start()
        assignment_progress.begin(
            "dmtac.assignment",
            label="DMTac USB 左右绑定",
            message="正在扫描并校验触觉夹爪设备",
            requirement="collection_core",
        )
        dmtac_assignment = None
        dmtac_startup_error: Exception | None = None
        try:
            dmtac_assignment = prepare_dmtac_hardware_assignment_before_gui(
                config.dmtac_assignment
            )
        except (OSError, ValueError, RuntimeError) as exc:
            dmtac_startup_error = exc
            assignment_progress.fail(
                "dmtac.assignment",
                f"{type(exc).__name__}: {exc}",
            )
            ui.publish(
                hardware_status={
                    "dmtac.assignment": {
                        "online": False,
                        "health": "error",
                        "status_level": "error",
                        "status_label": "连接失败",
                        "requirement": "collection_core",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                }
            )
        else:
            assignment_progress.succeed(
                "dmtac.assignment",
                message="DMTac USB 左右绑定已确认",
            )
        finally:
            assignment_progress.close()
        if not _ensure_angle_trigger_from_gui(ui, config):
            return 0
        validate_hand_eye_config_binding(config)
        calibration_required_reason: str | None = None
        while True:
            if calibration_required_reason is not None:
                action = _wait_for_calibration(ui, reason=calibration_required_reason)
                if action == "quit":
                    return 0
                if not _run_calibration_from_gui(ui, config):
                    calibration_required_reason = "上次标定未完成，请重试。"
                    continue
                calibration_required_reason = None
            try:
                calibrations = UmiCalibrationSet.load(config)
            except (OSError, ValueError, RuntimeError) as exc:
                calibration_required_reason = str(exc)
                continue

            runtime_config = config_for_calibration_dataset(config, calibrations)
            if dmtac_assignment is None:
                try:
                    dmtac_assignment = prepare_dmtac_hardware_assignment_before_gui(
                        runtime_config.dmtac_assignment
                    )
                    dmtac_startup_error = None
                except (OSError, ValueError, RuntimeError) as exc:
                    dmtac_startup_error = exc
                if dmtac_startup_error is not None:
                    outcome = _wait_for_device_initialization_recovery(
                        ui, dmtac_startup_error
                    )
                    if outcome == "retry_devices":
                        continue
                    if outcome == "recalibrate_angle":
                        if not _ensure_angle_trigger_from_gui(ui, config, force=True):
                            return 0
                        continue
                    return 0
            assert dmtac_assignment is not None
            outcome = _run_collection_session(
                config=runtime_config,
                calibrations=calibrations,
                layout=layout,
                ui=ui,
                dmtac_assignment=dmtac_assignment,
            )
            if outcome == "recalibrate_angle":
                if not _ensure_angle_trigger_from_gui(ui, config, force=True):
                    return 0
                continue
            if outcome == "retry_devices":
                dmtac_assignment = None
                continue
            if outcome != "recalibrate":
                return 0
            calibration_required_reason = "操作员请求重新标定左右 UMI。"
    finally:
        ui.close()


def _ensure_angle_trigger_from_gui(ui, config, *, force: bool = False) -> bool:
    retry_force = bool(force)
    while True:
        try:
            return ensure_angle_trigger_ready(
                ui,
                config_root=config.angle_trigger_config_root,
                rate_hz=config.angle_trigger_rate_hz,
                force=retry_force,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            ui.publish(
                ui_mode="umi_collection",
                stage="AngleTrigger 标定未完成",
                teleop_running=False,
                episode_recording_available=False,
                episode_event_status="error",
                episode_event_message=str(exc),
                active_controls=["angle_recalibrate", "q"],
                control_hint="检查两个 Angle 传感器连接后点击“重新标定 Angle”。",
            )
            while True:
                events = set(ui.consume_events())
                if "q" in events:
                    return False
                if "angle_recalibrate" in events:
                    retry_force = True
                    break
                time.sleep(0.1)


def _run_collection_session(
    *, config, calibrations, layout, ui, dmtac_assignment
) -> str:
    calibration = calibrations.desk
    pose_adapter = UmiPoseAdapter(
        T_vive_world_to_desk=calibration.T_vive_world_to_desk,
        T_tcp_to_tracker_by_side=calibrations.T_tcp_to_tracker_by_side,
    )
    labels = OperatorLabelStateMachine()
    v25p_native_shadow = None
    if config.v25p_native_shadow:
        from ..v25p_capture_profiles import (
            V25P_UMI_SOURCE,
            build_policy_io_v25p_capture_profiles,
        )
        from ..v25p_dual_write_shadow import V25pDualWriteShadow
        from ..v25p_native_store import V25pNativeEpisodeStore
        from .v25p_producer import UmiV25pProducer

        bundle = build_policy_io_v25p_capture_profiles(
            source=V25P_UMI_SOURCE,
            layout=layout,
            pose_frame_contract_hash=calibration.pose_frame_contract().contract_hash,
        )
        store = V25pNativeEpisodeStore(
            dataset_root=config.dataset_root,
            dataset_id=f"umi:{config.task_id}",
            embodiment_profile=bundle.embodiment,
            component_stream_profile=bundle.streams,
            compatibility_projection_profiles=(bundle.v23_projection,),
            physical_storage_layout=config.v25p_native_storage_layout,
            defer_sample_file_writes=(
                config.v25p_native_storage_layout
                == V25P_RECORD_CJSON_LAYOUT_V1
            ),
        )
        v25p_native_shadow = V25pDualWriteShadow(
            store=store,
            bundle=bundle,
            producer=UmiV25pProducer(
                stream_profile=bundle.streams,
                bindings=bundle.producer_bindings,
                component_action_schema=bundle.component_action_schema,
                component_action_projection=bundle.component_action_projection,
            ),
        )
    recorder = UmiDatasetRecorder(
        config=config,
        calibration=calibration,
        policy_io_layout=layout,
        T_tcp_to_tracker_by_side=calibrations.T_tcp_to_tracker_by_side,
        calibration_by_side=calibrations.by_side,
        v25p_native_shadow=v25p_native_shadow,
    )
    components = None
    startup_progress = GuiStartupProgress(
        ui,
        stage_prefix="UMI 设备连接中",
    )
    startup_progress.start()
    try:
        ui.publish(
            stage="UMI 设备初始化中",
            control_hint="正在初始化 Vive、双腕相机、AngleTrigger 与 DMTac。",
            active_controls=[],
        )
        startup_progress.begin(
            "umi.components",
            label="UMI 运行时组件",
            message="正在构建 VIVE、双腕相机、AngleTrigger 与 DMTac 组件",
            requirement="collection_core",
        )
        try:
            components = build_umi_components(
                config,
                dmtac_assignment=dmtac_assignment,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            startup_progress.fail(
                "umi.components",
                f"{type(exc).__name__}: {exc}",
                message="UMI 运行时组件构建失败",
            )
            startup_progress.close()
            return _wait_for_device_initialization_recovery(ui, exc)
        startup_progress.succeed(
            "umi.components",
            message="UMI 运行时组件已构建，尚未启动设备",
        )
        last_commands = {"left": 0.0, "right": 0.0}
        assembler = UmiSnapshotAssembler(
            config=config,
            pose_adapter=pose_adapter,
            vive_source=components.vive,
            camera_selectors=components.camera_selectors(),
            hand_state_poller=components.hand_state_poller,
            gripper_command_provider=lambda: dict(last_commands),
            labels=labels,
        )
        controller = UmiCollectionController(
            config=config,
            calibration=calibrations,
            components=components,
            recorder=recorder,
            ui=ui,
            labels=labels,
            snapshot_assembler=assembler,
            last_commands=last_commands,
        )
        components.start(startup_progress=startup_progress)
        startup_progress.close()
        ui.publish(stage="UMI 就绪", control_hint=f"UMI GUI: {ui.url}")
        return controller.run()
    finally:
        startup_progress.close()
        steps = []
        if components is not None:
            steps.append(("components", components.close))
        steps.append(("recorder", recorder.close))
        run_best_effort_cleanup("umi_collection_session.close", steps)


def _wait_for_device_initialization_recovery(ui, error: Exception) -> str:
    message = str(error)
    if "DMTac" in message or "DMTAC" in message or "Configured DMTac" in message:
        message = (
            "AngleTrigger 配置已通过；当前失败来自 DMTac USB 左右绑定。"
            "旧配置设备路径与当前扫描结果不一致，程序无法安全猜测左右侧。 "
            + message
        )
    ui.publish(
        ui_mode="umi_collection",
        stage="UMI 设备加载失败",
        teleop_running=False,
        episode_recording_available=False,
        episode_event_status="error",
        episode_event_message=message,
        active_controls=["device_retry", "angle_recalibrate", "q"],
        control_hint="修复设备连接/配置后点击“重试设备初始化”。",
    )
    while True:
        events = set(ui.consume_events())
        if "q" in events:
            return "quit"
        if "angle_recalibrate" in events:
            return "recalibrate_angle"
        if "device_retry" in events:
            return "retry_devices"
        time.sleep(0.1)


def _wait_for_calibration(ui: CalibrationWebUi, *, reason: str) -> str:
    ui.publish(
        ui_mode="umi_collection",
        stage="需要 UMI 标定",
        teleop_running=False,
        episode_recording=False,
        episode_recording_available=False,
        episode_event_status="unavailable",
        episode_event_message=reason,
        calibration_validity_warning={
            "active": True,
            "level": "expired",
            "title": "需要重新标定左右 UMI",
            "message": "点击下方按钮；标定完成后设备会自动恢复并可继续数采。",
            "remaining_sec": 0.0,
            "countdown": "00:00",
        },
        active_controls=["umi_recalibrate", "q"],
        control_hint=reason,
    )
    while True:
        for event in ui.consume_events():
            if event == "q":
                return "quit"
            if event == "umi_recalibrate":
                return "recalibrate"
        time.sleep(0.1)


def _run_calibration_from_gui(ui: CalibrationWebUi, config: UmiCollectionConfig) -> bool:
    ui.publish(
        stage="左右 UMI 标定中",
        teleop_running=False,
        episode_recording_available=False,
        episode_event_status="calibrating",
        episode_event_message="请在弹出的双 UMI 标定窗口中移动两套 UMI；满足覆盖后按 c 求解。",
        calibration_validity_warning={
            "active": True,
            "level": "warning",
            "title": "双 UMI 标定进行中",
            "message": "自动采样；左右均满足覆盖条件后，在标定窗口按 c。按 q 会取消。",
        },
        active_controls=[],
    )
    try:
        run_default_dual_calibration(config)
    except (OSError, ValueError, RuntimeError) as exc:
        ui.publish(
            stage="UMI 标定未完成",
            episode_event_status="error",
            episode_event_message=str(exc),
            control_hint=str(exc),
        )
        return False
    ui.publish(
        stage="UMI 标定完成",
        episode_event_status="ready",
        episode_event_message="左右结果已校验，正在恢复数采设备。",
        calibration_validity_warning={"active": False},
        active_controls=[],
    )
    return True


__all__ = ["run_umi_collection"]
