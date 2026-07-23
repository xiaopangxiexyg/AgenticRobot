from __future__ import annotations

import pytest

from teleop_runtime.backends.grippers import DMTacGripperBackendConfig
from teleop_runtime.configs import (
    build_annotated_dual_rm65_calibration,
    build_bimanual_passive_session,
)
from teleop_runtime.retargeters import ScalarToGripperRetargeter
from teleop_runtime.session import TeleopSessionConfig


def test_collection_profile_projection_has_no_rig_or_backend() -> None:
    session = build_bimanual_passive_session(
        calibration=build_annotated_dual_rm65_calibration(),
        load_default_hand_serial_ports=False,
    )
    try:
        assert session.rig is None
        assert session.recorder is None
        assert session.config.mode == "bimanual_passive_pose"
        assert session.config.record_enabled is False
        assert session.config.allow_hardware_after_debug_pass is False
        assert [source.name for source in session.sources] == ["vive_pose"]
        assert [retargeter.name for retargeter in session.retargeters] == [
            "pose_to_arm_left",
            "pose_to_arm_right",
        ]
    finally:
        session.close()


def test_collection_profile_projection_reuses_hand_retargeter_kind_without_backend() -> None:
    config = DMTacGripperBackendConfig(
        side="left",
        gripper_port="/dev/null",
        connect_hardware=False,
        connect_tactile=False,
    )
    session = build_bimanual_passive_session(
        calibration=build_annotated_dual_rm65_calibration(),
        left_hand_serial_port="/dev/null",
        left_hand_backend_config=config,
        load_default_hand_serial_ports=False,
    )
    try:
        assert session.rig is None
        assert any(isinstance(item, ScalarToGripperRetargeter) for item in session.retargeters)
        assert [source.name for source in session.sources] == ["vive_pose", "hand_q6_bimanual"]
    finally:
        session.close()


@pytest.mark.parametrize(
    "config",
    (
        TeleopSessionConfig(record_enabled=True, allow_hardware_after_debug_pass=False),
        TeleopSessionConfig(record_enabled=False, allow_hardware_after_debug_pass=True),
    ),
)
def test_collection_profile_projection_rejects_active_session_capabilities(
    config: TeleopSessionConfig,
) -> None:
    with pytest.raises(ValueError):
        build_bimanual_passive_session(
            calibration=build_annotated_dual_rm65_calibration(),
            session_config=config,
            load_default_hand_serial_ports=False,
        )
