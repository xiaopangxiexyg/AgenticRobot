from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from ..backends.hands import InspireTacHandBackend, InspireTacHandBackendConfig
from ..calibration import TeleopCalibration
from ..hand_tcp_alignment import tracker_hand_to_rm65_tcp_rotation
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
)


def default_vive_tracker_configs(
    *,
    left_tracker_name: str = "tracker_1",
    right_tracker_name: str = "tracker_2",
    auxiliary_tracker_name: str | None = "tracker_3",
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


def build_bimanual_session(
    *,
    calibration: TeleopCalibration,
    session_config: TeleopSessionConfig | None = None,
    vive_port: int = 9999,
    vive_tracker_configs: list[ViveTrackerConfig] | None = None,
    left_hand_serial_port: str | None = None,
    right_hand_serial_port: str | None = None,
    left_arm_backend_config: RM65ArmBackendConfig | None = None,
    right_arm_backend_config: RM65ArmBackendConfig | None = None,
    left_hand_backend_config: InspireTacHandBackendConfig | None = None,
    right_hand_backend_config: InspireTacHandBackendConfig | None = None,
    keyboard_source: KeyboardButtonSource | None = None,
    enable_dual_arm_collision_avoidance: bool = True,
    dual_arm_collision_config: DualArmCollisionAwareRetargeterConfig | None = None,
) -> TeleopSession:
    sources = []
    retargeters = []
    backends = {}

    vive_source = VivePoseSource(
        tracker_configs=vive_tracker_configs or default_vive_tracker_configs(),
        port=vive_port,
    )
    sources.append(vive_source)

    if keyboard_source is not None:
        sources.append(keyboard_source)

    both_arm_backends_present = (
        left_arm_backend_config is not None and right_arm_backend_config is not None
    )
    left_pose_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="left",
            input_channel_name="left_wrist_pose",
            effector_name="left_arm",
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
        )
    )
    right_pose_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="right",
            input_channel_name="right_wrist_pose",
            effector_name="right_arm",
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
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
            JointsToDexHandRetargeter(
                JointsToDexHandRetargeterConfig(
                    side="left",
                    input_channel_name="left_hand_q6",
                    effector_name="left_hand",
                )
            )
        )
    if right_hand_serial_port is not None:
        retargeters.append(
            JointsToDexHandRetargeter(
                JointsToDexHandRetargeterConfig(
                    side="right",
                    input_channel_name="right_hand_q6",
                    effector_name="right_hand",
                )
            )
        )

    if left_arm_backend_config is not None:
        backends["left_arm"] = RM65ArmBackend(left_arm_backend_config)
    if right_arm_backend_config is not None:
        backends["right_arm"] = RM65ArmBackend(right_arm_backend_config)
    if left_hand_backend_config is not None:
        backends["left_hand"] = InspireTacHandBackend(left_hand_backend_config)
    if right_hand_backend_config is not None:
        backends["right_hand"] = InspireTacHandBackend(right_hand_backend_config)

    rig = BimanualRig(backends=backends)
    return TeleopSession(
        config=session_config or TeleopSessionConfig(mode="bimanual"),
        sources=sources,
        retargeters=retargeters,
        rig=rig,
        calibration=calibration,
    )
