"""Concrete Control Station application binding for formal UMI collection.

UMI device production, calibration, recording, and collection-loop ownership
remain in :mod:`teleop_runtime.collection.umi`.  This application leaf binds
those lower ports to the Station-owned Web UI and operator policy.
"""

from __future__ import annotations

import time
from typing import Callable

from dataset_interface.policy_io_layout import load_policy_io_layout
from robot_motion.runtime.lifecycle import run_best_effort_cleanup

from .umi.application_policy import (
    DEFAULT_STATION_UMI_OPERATOR_POLICY,
)
from .web.calibration_web_ui import CalibrationWebUi
from .web.startup_progress import GuiStartupProgress
from teleop_runtime.collection.operator_labels import OperatorLabelStateMachine
from teleop_runtime.collection.umi.angle_trigger_calibration import (
    ensure_default_command_input_from_gui,
)
from teleop_runtime.collection.umi.application_ports import UmiOperatorPolicyPort
from teleop_runtime.collection.umi.calibration import UmiCalibrationSet
from teleop_runtime.collection.umi.components import (
    UmiComponentBuilder,
    UmiHardwarePreparer,
    UmiHardwarePresentation,
)
from teleop_runtime.collection.umi.config import UmiCollectionConfig
from teleop_runtime.collection.umi.controller import UmiCollectionController
from teleop_runtime.collection.umi.default_components import (
    DEFAULT_UMI_HARDWARE_PRESENTATION,
    build_default_umi_components,
    prepare_default_umi_hardware,
)
from teleop_runtime.collection.umi.pose_adapter import (
    UmiPoseAdapter,
    UmiPoseAdapterPort,
)
from teleop_runtime.collection.umi.recorder import UmiDatasetRecorder
from teleop_runtime.collection.umi.recalibration import (
    config_for_calibration_dataset,
    run_default_dual_calibration,
    validate_hand_eye_config_binding,
)
from teleop_runtime.collection.umi.snapshot import UmiSnapshotAssembler


UmiPoseAdapterBuilder = Callable[..., UmiPoseAdapterPort]
UmiCommandInputInitializer = Callable[..., bool]


def run_umi_collection(
    config: UmiCollectionConfig,
    *,
    hardware_preparer: UmiHardwarePreparer = prepare_default_umi_hardware,
    hardware_presentation: UmiHardwarePresentation = (
        DEFAULT_UMI_HARDWARE_PRESENTATION
    ),
    component_builder: UmiComponentBuilder = build_default_umi_components,
    pose_adapter_builder: UmiPoseAdapterBuilder = UmiPoseAdapter,
    command_input_initializer: UmiCommandInputInitializer = (
        ensure_default_command_input_from_gui
    ),
    operator_policy: UmiOperatorPolicyPort = (
        DEFAULT_STATION_UMI_OPERATOR_POLICY
    ),
) -> int:
    initialize_command_input = command_input_initializer
    layout = load_policy_io_layout(config.policy_io_layout)
    if layout is None:
        raise RuntimeError("UMI collection requires a policy I/O layout")
    ui = CalibrationWebUi(
        host=config.gui_host,
        port=config.gui_port,
        open_browser=config.open_browser,
        # Keep application assembly compatible with injected/older config
        # objects.  These are presentation-only capabilities and therefore
        # must not prevent device recovery or raw collection startup when an
        # otherwise valid compatibility config predates the fields.
        resizable_sidebars=bool(
            getattr(config, "gui_resizable_sidebars", True)
        ),
        input_activity_indicators=bool(
            getattr(config, "gui_input_activity_indicators", True)
        ),
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
            hardware_presentation.preflight_key,
            label=hardware_presentation.preflight_label,
            message=hardware_presentation.preflight_message,
            requirement="collection_core",
        )
        prepared_hardware = None
        hardware_startup_error: Exception | None = None
        try:
            prepared_hardware = hardware_preparer(config)
        except (OSError, ValueError, RuntimeError) as exc:
            hardware_startup_error = exc
            assignment_progress.fail(
                hardware_presentation.preflight_key,
                f"{type(exc).__name__}: {exc}",
            )
            ui.publish(
                hardware_status={
                    hardware_presentation.preflight_key: {
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
                hardware_presentation.preflight_key,
                message=hardware_presentation.preflight_success_message,
            )
        finally:
            assignment_progress.close()
        if not initialize_command_input(ui, config):
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
            if prepared_hardware is None:
                try:
                    prepared_hardware = hardware_preparer(runtime_config)
                    hardware_startup_error = None
                except (OSError, ValueError, RuntimeError) as exc:
                    hardware_startup_error = exc
                if hardware_startup_error is not None:
                    outcome = _wait_for_device_initialization_recovery(
                        ui,
                        hardware_startup_error,
                        hardware_presentation=hardware_presentation,
                    )
                    if outcome == "retry_devices":
                        continue
                    if outcome == "recalibrate_angle":
                        if not initialize_command_input(ui, config, force=True):
                            return 0
                        continue
                    return 0
            assert prepared_hardware is not None
            outcome = _run_collection_session(
                config=runtime_config,
                calibrations=calibrations,
                layout=layout,
                ui=ui,
                prepared_hardware=prepared_hardware,
                component_builder=component_builder,
                pose_adapter_builder=pose_adapter_builder,
                hardware_presentation=hardware_presentation,
                operator_policy=operator_policy,
            )
            if outcome == "recalibrate_angle":
                if not initialize_command_input(ui, config, force=True):
                    return 0
                continue
            if outcome == "retry_devices":
                prepared_hardware = None
                continue
            if outcome != "recalibrate":
                return 0
            calibration_required_reason = "操作员请求重新标定左右 UMI。"
    finally:
        ui.close()


def _run_collection_session(
    *,
    config,
    calibrations,
    layout,
    ui,
    prepared_hardware,
    component_builder: UmiComponentBuilder = build_default_umi_components,
    pose_adapter_builder: UmiPoseAdapterBuilder = UmiPoseAdapter,
    hardware_presentation: UmiHardwarePresentation = (
        DEFAULT_UMI_HARDWARE_PRESENTATION
    ),
    operator_policy: UmiOperatorPolicyPort = (
        DEFAULT_STATION_UMI_OPERATOR_POLICY
    ),
) -> str:
    calibration = calibrations.desk
    pose_adapter = pose_adapter_builder(
        T_vive_world_to_desk=calibration.T_vive_world_to_desk,
        T_tcp_to_tracker_by_side=calibrations.T_tcp_to_tracker_by_side,
    )
    labels = OperatorLabelStateMachine()
    v25p_native_shadow = None
    if config.v25p_native_shadow:
        from teleop_runtime.collection.v25p_capture_profiles import V25P_UMI_SOURCE
        from teleop_runtime.collection.v25p_dual_write_shadow import (
            build_policy_io_v25p_shadow,
        )

        v25p_native_shadow = build_policy_io_v25p_shadow(
            source=V25P_UMI_SOURCE,
            layout=layout,
            dataset_root=config.dataset_root,
            dataset_id=f"umi:{config.task_id}",
            pose_frame_contract_hash=calibration.pose_frame_contract().contract_hash,
            physical_storage_layout=config.v25p_native_storage_layout,
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
            control_hint=hardware_presentation.component_control_hint,
            active_controls=[],
        )
        startup_progress.begin(
            "umi.components",
            label="UMI 运行时组件",
            message=hardware_presentation.component_build_message,
            requirement="collection_core",
        )
        try:
            components = component_builder(
                config,
                prepared_hardware=prepared_hardware,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            startup_progress.fail(
                "umi.components",
                f"{type(exc).__name__}: {exc}",
                message="UMI 运行时组件构建失败",
            )
            startup_progress.close()
            return _wait_for_device_initialization_recovery(
                ui,
                exc,
                hardware_presentation=hardware_presentation,
            )
        startup_progress.succeed(
            "umi.components",
            message="UMI 运行时组件已构建，尚未启动设备",
        )
        last_commands = {"left": 0.0, "right": 0.0}
        assembler = UmiSnapshotAssembler(
            config=config,
            pose_adapter=pose_adapter,
            vive_source=components.pose_source,
            camera_selectors=components.camera_selectors(),
            hand_state_poller=components.state_history,
            gripper_command_provider=lambda: dict(last_commands),
            labels=labels,
        )
        controller = UmiCollectionController(
            config=config,
            calibration=calibrations,
            components=components,
            recorder=recorder,
            ui=ui,
            operator_policy=operator_policy,
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


def _wait_for_device_initialization_recovery(
    ui,
    error: Exception,
    *,
    hardware_presentation: UmiHardwarePresentation = (
        DEFAULT_UMI_HARDWARE_PRESENTATION
    ),
) -> str:
    message = hardware_presentation.recovery_message(error)
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
