from __future__ import annotations

from collections.abc import Mapping

from ..backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from ..backends.hands import InspireTacHandBackend, InspireTacHandBackendConfig
from ..calibration import TeleopCalibration
from ..rigs import BimanualRig
from ..session import TeleopSession, TeleopSessionConfig
from ..sources.hand import HandQ6Source
from ..sources.misc import KeyboardButtonSource
from ..sources.pose import VivePoseSource, ViveTrackerConfig
from ..retargeters import (
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

    for side in ("left", "right"):
        retargeters.append(
            PoseToArmRetargeter(
                PoseToArmRetargeterConfig(
                    side=side,
                    input_channel_name=f"{side}_wrist_pose",
                    effector_name=f"{side}_arm",
                )
            )
        )

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
