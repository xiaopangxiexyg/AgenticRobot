from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop
from teleop_runtime.schemas import ObservationChannel


def test_action_debug_records_bounded_pre_retarget_tracker_observation() -> None:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.12, -0.34, 0.56]
    motion_window = {
        "schema": "teleop_runtime.vive_motion_window/v1",
        "sample_count": 30,
        "window_span_sec": 0.1,
        "linear_speed_p90_m_s": 0.012,
        "angular_speed_p90_rad_s": 0.034,
        "translation_span_m": 0.0008,
        "rotation_span_rad": 0.0012,
        "quality_valid": True,
        "quality_reasons": [],
    }
    channel = ObservationChannel(
        channel_name="left_wrist_pose",
        capability="pose6d",
        side="left",
        value=pose,
        frame_name="vive_world",
        timestamp=10.0,
        source_name="vive_pose",
        metadata={
            "tracker_name": "tracker_1",
            "tracker_role": "left_wrist",
            "seq": 123,
            "linear_velocity_m_s": [0.01, 0.0, 0.0],
            "angular_velocity_rad_s": [0.0, 0.02, 0.0],
            "velocity_frame_name": "vive_world",
            "velocity_source": "openvr",
            "velocity_valid": True,
            "tracker_motion_window": motion_window,
        },
    )
    step_result = SimpleNamespace(
        timestamp=10.0,
        command=None,
        command_dispatched=False,
        channels={"left_wrist_pose": channel},
        faults=(),
        state=SimpleNamespace(value="running"),
    )
    components = SimpleNamespace(
        active_sides=("left",),
        left_command_backend=None,
        left_state_backend=None,
        fix_arm_controller=SimpleNamespace(
            status_payload=lambda: {
                "fix_arm_available": True,
                "fix_arm_selected_side": "left",
                "fix_arm_fixed_side": "left",
                "fix_arm_active": True,
                "fix_arm_recalibrating": False,
                "fix_arm_status": "fixed",
                "fix_arm_sample_count": None,
                "fix_arm_min_samples": None,
            }
        ),
    )

    payload = real_teleop._build_action_debug_metadata(
        step_result,
        components=components,
        record_wall_time=10.005,
    )

    tracker = payload["tracker_channels"]["left_wrist_pose"]
    assert payload["fix_arm"] == {
        "available": True,
        "selected_side": "left",
        "fixed_side": "left",
        "active": True,
        "recalibrating": False,
        "status": "fixed",
        "sample_count": None,
        "min_samples": None,
    }
    assert tracker["pose_stage"] == "observation_channel_before_retarget"
    assert tracker["pose"]["xyz_m"] == [0.12, -0.34, 0.56]
    assert tracker["pose"]["rotation_matrix"] == np.eye(3).tolist()
    assert tracker["motion_window"]["translation_span_m"] == 0.0008
    assert tracker["motion_window"]["quality_valid"] is True
    assert tracker["motion"] == {
        "linear_velocity_m_s": [0.01, 0.0, 0.0],
        "angular_velocity_rad_s": [0.0, 0.02, 0.0],
        "velocity_frame_name": "vive_world",
        "velocity_source": "openvr",
        "velocity_valid": True,
    }
