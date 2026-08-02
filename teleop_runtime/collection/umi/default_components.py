"""Default VIVE + RealSense + AngleTrigger + DMTac UMI hardware factory."""

from __future__ import annotations

from typing import Any

from hardware.angle_trigger.angle_trigger import load_configured_reader

from ...backends.grippers.dmtac import DMTacGripperBackend
from ...configs.dmtac_hardware import (
    dmtac_backend_configs_from_assignment,
    load_dmtac_hardware_assignment,
    prepare_dmtac_hardware_assignment_before_gui,
)
from ...real_teleop.foot_pad_controls import (
    FootPadSpecialArmInput,
    FootPadUnavailableError,
)
from ...retargeters.scalar_to_gripper import (
    ScalarToGripperRetargeter,
    ScalarToGripperRetargeterConfig,
)
from ...rigs.hand_only import HandOnlyRig
from ...sources.camera.realsense_rgb import (
    RealSenseRGBCameraConfig,
    RealSenseRGBSource,
)
from ...sources.hand.angle_trigger import AngleTriggerSource
from ...sources.pose.vive import VivePoseSource, ViveTrackerConfig
from ..hand_state_polling import HandStateHistoryPoller
from .components import (
    UmiComponents,
    UmiHardwarePresentation,
    UmiStartupResource,
)
from .config import UmiCollectionConfig
from .topology import (
    CAMERA_KEY_BY_SIDE,
    DEFAULT_COMMAND_CHANNEL_BY_SIDE,
    HAND_EFFECTOR_BY_SIDE,
    SIDES,
)


def configure_dmtac_backend_configs(
    config: UmiCollectionConfig,
    backend_configs: dict[str, Any],
) -> None:
    for backend_config in backend_configs.values():
        backend_config.send_commands = bool(config.send_gripper_commands)
        backend_config.tactile_read_hz = float(config.dmtac_tactile_read_hz)
        backend_config.command_hz = float(config.dmtac_gripper_command_hz)
        backend_config.telemetry_hz = float(config.dmtac_gripper_telemetry_hz)
        backend_config.gripper_telemetry_command_grace_s = float(
            config.dmtac_gripper_telemetry_command_grace_s
        )


def prepare_default_umi_hardware(config: UmiCollectionConfig) -> dict[str, Any]:
    """Prepare the default end-effector assignment without starting devices."""

    return prepare_dmtac_hardware_assignment_before_gui(config.dmtac_assignment)


def _format_default_hardware_recovery_error(error: Exception) -> str:
    message = str(error)
    if "DMTac" in message or "DMTAC" in message or "Configured DMTac" in message:
        return (
            "AngleTrigger 配置已通过；当前失败来自 DMTac USB 左右绑定。"
            "旧配置设备路径与当前扫描结果不一致，程序无法安全猜测左右侧。 "
            + message
        )
    return message


DEFAULT_UMI_HARDWARE_PRESENTATION = UmiHardwarePresentation(
    preflight_key="dmtac.assignment",
    preflight_label="DMTac USB 左右绑定",
    preflight_message="正在扫描并校验触觉夹爪设备",
    preflight_success_message="DMTac USB 左右绑定已确认",
    component_build_message="正在构建 VIVE、双腕相机、AngleTrigger 与 DMTac 组件",
    component_control_hint="正在初始化 Vive、双腕相机、AngleTrigger 与 DMTac。",
    recovery_error_formatter=_format_default_hardware_recovery_error,
)


def _require_default_end_effectors_ready(
    grippers: dict[str, DMTacGripperBackend],
) -> None:
    unavailable: list[str] = []
    for side, backend in sorted(grippers.items()):
        if backend.is_ready():
            continue
        try:
            state = backend.get_state(include_tactile_arrays=False)
        except TypeError:
            state = backend.get_state()
        detail = str(state.get("telemetry_error") or "backend is not ready")
        unavailable.append(f"{side}={detail}")
    if unavailable:
        raise RuntimeError("DMTAC startup incomplete: " + "; ".join(unavailable))


def _default_end_effector_diagnostics(
    _side: str,
    backend: DMTacGripperBackend,
    config: UmiCollectionConfig,
) -> dict[str, Any]:
    backend_config = backend.config
    return {
        "configured": True,
        "is_connected": backend.is_ready(),
        "max_action": 1.0,
        "speed": backend_config.speed,
        "torque": backend_config.torque,
        "connect_tactile": bool(backend_config.connect_tactile),
        "tactile_process_read": bool(backend_config.tactile_process_read),
        "tactile_read_hz": float(
            getattr(
                backend_config,
                "tactile_read_hz",
                config.dmtac_tactile_read_hz,
            )
        ),
        "tactile_max_fps": int(backend_config.tactile_max_fps),
    }


def build_default_umi_components(
    config: UmiCollectionConfig,
    *,
    prepared_hardware: Any = None,
    dmtac_assignment: dict[str, Any] | None = None,
) -> UmiComponents:
    tracker_configs = [
        ViveTrackerConfig(
            name=config.side(side).tracker_name,
            side=side,
            role=config.side(side).tracker_role,
            channel_name=config.side(side).tracker_channel,
            legacy_index=1 if side == "left" else 2,
            critical=True,
        )
        for side in SIDES
    ]
    vive = VivePoseSource(
        tracker_configs,
        host=config.vive_host,
        port=config.vive_port,
        clock_sync_port=config.vive_clock_sync_port,
        name="vive_umi_pose",
    )
    cameras = {
        CAMERA_KEY_BY_SIDE[side]: RealSenseRGBSource(
            RealSenseRGBCameraConfig(
                serial=config.side(side).camera_serial,
                width=640,
                height=480,
                fps=15,
                camera_key=CAMERA_KEY_BY_SIDE[side],
                name=f"umi_{side}_wrist_camera",
                keep_last=8,
                color_space="rgb",
                enable_depth=False,
            )
        )
        for side in SIDES
    }
    angle_reader = load_configured_reader(config_root=config.angle_trigger_config_root)
    angle_trigger = AngleTriggerSource(
        angle_reader,
        rate_hz=config.angle_trigger_rate_hz,
        config_root=config.angle_trigger_config_root,
    )

    if prepared_hardware is not None and dmtac_assignment is not None:
        raise ValueError(
            "Pass either prepared_hardware or dmtac_assignment, not both."
        )
    supplied_assignment = (
        prepared_hardware
        if prepared_hardware is not None
        else dmtac_assignment
    )
    assignment = (
        supplied_assignment
        if supplied_assignment is not None
        else load_dmtac_hardware_assignment(
            config.dmtac_assignment,
            validate_current_usb=True,
        )
    )
    backend_configs = dmtac_backend_configs_from_assignment(
        assignment,
        active_sides=SIDES,
    )
    missing = sorted(set(SIDES) - set(backend_configs))
    if missing:
        raise RuntimeError(f"DMTac assignment is missing UMI side(s): {missing}")
    configure_dmtac_backend_configs(config, backend_configs)
    grippers = {
        side: DMTacGripperBackend(backend_configs[side])
        for side in SIDES
    }
    rig = HandOnlyRig(
        {HAND_EFFECTOR_BY_SIDE[side]: backend for side, backend in grippers.items()},
        name="umi_hand_only_rig",
        parallel_dispatch=True,
        fail_stop_on_dispatch_error=True,
    )
    retargeters = {
        side: ScalarToGripperRetargeter(
            ScalarToGripperRetargeterConfig(
                side=side,
                input_channel_name=DEFAULT_COMMAND_CHANNEL_BY_SIDE[side],
                effector_name=HAND_EFFECTOR_BY_SIDE[side],
                source_dim=None,
            )
        )
        for side in SIDES
    }
    poller = HandStateHistoryPoller(backends_by_side=grippers)
    foot_pad = None
    foot_pad_error = ""
    if config.foot_pad_enabled:
        try:
            foot_pad = FootPadSpecialArmInput.open_online()
        except (FootPadUnavailableError, OSError, RuntimeError, ValueError) as exc:
            # UMI remains usable with the legacy GUI label controls when the
            # optional virtual keyboard is absent.
            foot_pad_error = str(exc)
    startup_resources = (
        UmiStartupResource(
            key="dmtac.rig",
            component=rig,
            label="DMTac 双夹爪",
            requirement="collection_core",
            strict_start_message="正在启动双夹爪 Rig",
            cleanup_name="rig",
            require_end_effector_readiness=True,
        ),
        UmiStartupResource(
            key="dmtac.state_poller",
            component=poller,
            label="DMTac 状态采样",
            requirement="collection_core",
            strict_start_message="正在启动状态采样",
            cleanup_name="hand_state_poller",
        ),
        UmiStartupResource(
            key="vive",
            component=vive,
            label="VIVE tracker",
            requirement="collection_core",
            strict_start_message="正在启动 tracker 接收",
            cleanup_name="vive",
        ),
        *(
            UmiStartupResource(
                key=f"camera.{key}",
                component=camera,
                label=f"腕部相机 {key}",
                requirement="collection",
                strict_start_message="正在启动视频流",
                strict_success_message="视频流已启动",
                cleanup_name=f"camera[{key}]",
            )
            for key, camera in cameras.items()
        ),
        UmiStartupResource(
            key="angle_trigger",
            component=angle_trigger,
            label="AngleTrigger",
            requirement="collection_core",
            strict_start_message="正在启动串口输入",
            cleanup_name="angle_trigger",
        ),
    )
    return UmiComponents(
        vive=vive,
        cameras=cameras,
        angle_trigger=angle_trigger,
        grippers=grippers,
        rig=rig,
        retargeters=retargeters,
        hand_state_poller=poller,
        foot_pad=foot_pad,
        foot_pad_error=foot_pad_error,
        strict_startup=config.hardware_admission_mode == "strict",
        startup_resources=startup_resources,
        end_effector_readiness_check=lambda: _require_default_end_effectors_ready(
            grippers
        ),
        end_effector_diagnostics_provider=_default_end_effector_diagnostics,
        command_channel_by_side=dict(DEFAULT_COMMAND_CHANNEL_BY_SIDE),
        command_input_name="angle_trigger",
    )


__all__ = [
    "build_default_umi_components",
    "configure_dmtac_backend_configs",
    "DEFAULT_UMI_HARDWARE_PRESENTATION",
    "prepare_default_umi_hardware",
]
