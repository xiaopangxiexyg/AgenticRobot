from __future__ import annotations

import numpy as np

from ..backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from ..backends.grippers import DMTacGripperBackend, DMTacGripperBackendConfig
from ..backends.hands import InspireTacHandBackend, InspireTacHandBackendConfig
from ..calibration import TeleopCalibration
from ..hand_tcp_alignment import (
    DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
    tracker_hand_to_rm65_tcp_rotation,
    vive_tracker_height_correction_offset,
)
from ..rigs import BimanualRig
from ..session import TeleopSession, TeleopSessionConfig
from ..sources.hand import HandQ6Source
from ..sources.misc import KeyboardButtonSource
from ..sources.pose import VivePoseSource, ViveTrackerConfig
from ..retargeters import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    JointsToDexHandRetargeter,
    JointsToDexHandRetargeterConfig,
    PoseToArmRetargeter,
    PoseToArmRetargeterConfig,
    ScalarToGripperRetargeter,
    ScalarToGripperRetargeterConfig,
)
from ..interfaces import EffectorBackend
from .q6_hardware import load_latest_q6_ports
from .pose_smoothing import build_bimanual_pose_smoother, load_default_pose_smoothing_config


def default_vive_tracker_configs(
    *,
    left_tracker_name: str = "tracker_1",
    right_tracker_name: str = "tracker_2",
    auxiliary_tracker_name: str | None = None,
) -> list[ViveTrackerConfig]:
    configs = [
        ViveTrackerConfig(
            name=left_tracker_name,
            side="left",
            role="left_wrist",
            channel_name="left_wrist_pose",
            legacy_index=0,
        ),
        ViveTrackerConfig(
            name=right_tracker_name,
            side="right",
            role="right_wrist",
            channel_name="right_wrist_pose",
            legacy_index=1,
        ),
    ]
    if auxiliary_tracker_name is not None:
        configs.append(
            ViveTrackerConfig(
                name=auxiliary_tracker_name,
                side=None,
                role="auxiliary",
                channel_name="auxiliary_tracker_pose",
                legacy_index=2,
                critical=False,
            )
        )
    return configs


HandBackendConfig = InspireTacHandBackendConfig | DMTacGripperBackendConfig


def _build_hand_backend(config: HandBackendConfig) -> EffectorBackend:
    if isinstance(config, DMTacGripperBackendConfig):
        return DMTacGripperBackend(config)
    if isinstance(config, InspireTacHandBackendConfig):
        return InspireTacHandBackend(config)
    raise TypeError(f"Unsupported hand backend config: {type(config).__name__}")


def _hand_effector_name(
    config: HandBackendConfig | None,
    *,
    side: str,
) -> str:
    if config is not None and config.effector_name is not None:
        return str(config.effector_name)
    return f"{side}_hand"


def _build_hand_retargeter(
    config: HandBackendConfig | None,
    *,
    side: str,
    input_channel_name: str,
):
    effector_name = _hand_effector_name(config, side=side)
    if isinstance(config, DMTacGripperBackendConfig):
        return ScalarToGripperRetargeter(
            ScalarToGripperRetargeterConfig(
                side=side,
                input_channel_name=input_channel_name,
                effector_name=effector_name,
                source_dim=2,
                output_dim=0,
                output_max=float(config.max_action),
            )
        )
    return JointsToDexHandRetargeter(
        JointsToDexHandRetargeterConfig(
            side=side,
            input_channel_name=input_channel_name,
            effector_name=effector_name,
        )
    )


def build_bimanual_session(
    *,
    calibration: TeleopCalibration,
    session_config: TeleopSessionConfig | None = None,
    vive_port: int = 9999,
    vive_clock_sync_port: int | None = None,
    vive_tracker_configs: list[ViveTrackerConfig] | None = None,
    left_hand_serial_port: str | None = None,
    right_hand_serial_port: str | None = None,
    left_arm_backend_config: RM65ArmBackendConfig | None = None,
    right_arm_backend_config: RM65ArmBackendConfig | None = None,
    left_hand_backend_config: HandBackendConfig | None = None,
    right_hand_backend_config: HandBackendConfig | None = None,
    hand_serial_config: str = "latest",
    load_default_hand_serial_ports: bool = True,
    keyboard_source: KeyboardButtonSource | None = None,
    enable_dual_arm_collision_avoidance: bool = True,
    dual_arm_collision_config: DualArmCollisionAwareRetargeterConfig | None = None,
    rm65_ik_framework: str = "robot_motion",
    vive_tracker_height_correction: bool = (
        DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG.height_correction_enabled
    ),
    vive_tracker_height_correction_m: float = (
        DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG.height_correction_m
    ),
) -> TeleopSession:
    sources = []
    retargeters = []
    backends = {}

    if bool(load_default_hand_serial_ports):
        missing_hand_sides = []
        if left_hand_serial_port is None and left_hand_backend_config is not None:
            missing_hand_sides.append("left")
        if right_hand_serial_port is None and right_hand_backend_config is not None:
            missing_hand_sides.append("right")
        if missing_hand_sides:
            default_ports = load_latest_q6_ports(
                active_sides=tuple(missing_hand_sides),
                config=hand_serial_config,
                allow_missing=True,
            )
            left_hand_serial_port = left_hand_serial_port or default_ports.get("left")
            right_hand_serial_port = right_hand_serial_port or default_ports.get("right")

    vive_source = VivePoseSource(
        tracker_configs=vive_tracker_configs or default_vive_tracker_configs(),
        port=vive_port,
        clock_sync_port=vive_clock_sync_port,
    )
    sources.append(vive_source)

    if keyboard_source is not None:
        sources.append(keyboard_source)

    both_arm_backends_present = (
        left_arm_backend_config is not None and right_arm_backend_config is not None
    )
    if float(vive_tracker_height_correction_m) < 0.0:
        raise ValueError("vive_tracker_height_correction_m must be >= 0.")
    tracker_translation_offset = (
        vive_tracker_height_correction_offset(vive_tracker_height_correction_m)
        if bool(vive_tracker_height_correction)
        else None
    )
    left_pose_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="left",
            input_channel_name="left_wrist_pose",
            effector_name="left_arm",
            input_local_translation_offset=tracker_translation_offset,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
            tracker_motion_intent_enabled=True,
        )
    )
    right_pose_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="right",
            input_channel_name="right_wrist_pose",
            effector_name="right_arm",
            input_local_translation_offset=tracker_translation_offset,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
            tracker_motion_intent_enabled=True,
        )
    )
    if enable_dual_arm_collision_avoidance and both_arm_backends_present:
        left_execution_step = tuple(
            (
                np.asarray(left_arm_backend_config.max_angvel, dtype=float)
                * float(left_arm_backend_config.control_dt)
            ).tolist()
        )
        right_execution_step = tuple(
            (
                np.asarray(right_arm_backend_config.max_angvel, dtype=float)
                * float(right_arm_backend_config.control_dt)
            ).tolist()
        )
        enforce_realistic_joint_step = bool(
            left_arm_backend_config.connect_hardware
            or right_arm_backend_config.connect_hardware
            or (not left_arm_backend_config.dry_run_fast)
            or (not right_arm_backend_config.dry_run_fast)
        )
        collision_config = (
            dual_arm_collision_config
            if dual_arm_collision_config is not None
            else DualArmCollisionAwareRetargeterConfig(
                enforce_realistic_joint_step=enforce_realistic_joint_step,
                left_execution_max_joint_step_rad=left_execution_step,
                right_execution_max_joint_step_rad=right_execution_step,
                left_initial_qpos=(
                    None
                    if left_arm_backend_config is None
                    else left_arm_backend_config.initial_qpos
                ),
                right_initial_qpos=(
                    None
                    if right_arm_backend_config is None
                    else right_arm_backend_config.initial_qpos
                ),
                left_tool_tcp_pose_flange_m_rad=(
                    None
                    if left_arm_backend_config is None
                    else left_arm_backend_config.tcp_pose_flange_m_rad
                ),
                right_tool_tcp_pose_flange_m_rad=(
                    None
                    if right_arm_backend_config is None
                    else right_arm_backend_config.tcp_pose_flange_m_rad
                ),
                ik_framework=str(rm65_ik_framework),
                target_pose_smoother=build_bimanual_pose_smoother(
                    load_default_pose_smoothing_config()
                ),
            )
        )
        retargeters.append(
            DualArmCollisionAwareRetargeter(
                collision_config,
                left_pose_retargeter=left_pose_retargeter,
                right_pose_retargeter=right_pose_retargeter,
            )
        )
    else:
        retargeters.extend((left_pose_retargeter, right_pose_retargeter))

    if left_hand_serial_port is not None or right_hand_serial_port is not None:
        sources.append(
            HandQ6Source.from_bimanual(
                left_port=left_hand_serial_port,
                right_port=right_hand_serial_port,
            )
        )

    if left_hand_serial_port is not None:
        retargeters.append(
            _build_hand_retargeter(
                left_hand_backend_config,
                side="left",
                input_channel_name="left_hand_q6",
            )
        )
    if right_hand_serial_port is not None:
        retargeters.append(
            _build_hand_retargeter(
                right_hand_backend_config,
                side="right",
                input_channel_name="right_hand_q6",
            )
        )

    if left_arm_backend_config is not None:
        backends["left_arm"] = RM65ArmBackend(left_arm_backend_config)
    if right_arm_backend_config is not None:
        backends["right_arm"] = RM65ArmBackend(right_arm_backend_config)
    if left_hand_backend_config is not None:
        left_backend = _build_hand_backend(left_hand_backend_config)
        backends[left_backend.name] = left_backend
    if right_hand_backend_config is not None:
        right_backend = _build_hand_backend(right_hand_backend_config)
        backends[right_backend.name] = right_backend

    rig = BimanualRig(backends=backends)
    return TeleopSession(
        config=session_config or TeleopSessionConfig(mode="bimanual"),
        sources=sources,
        retargeters=retargeters,
        rig=rig,
        calibration=calibration,
    )
