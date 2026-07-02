from __future__ import annotations

import pytest

from hardware.rm65.rm_api2_adapter import Arm
from hardware.rm65.safety_diagnostics import check_rm65_joint_safety


class FakeCore:
    def __init__(self, *, self_collision_code: int = 0, position_limit_code: int = 0):
        self.self_collision_code = self_collision_code
        self.position_limit_code = position_limit_code

    def rm_algo_safety_robot_self_collision_detection(self, joint):
        return self.self_collision_code

    def rm_algo_ikine_check_joint_position_limit(self, joint):
        return self.position_limit_code


class FakeRobot:
    def __init__(self, *, self_collision_code: int = 0, position_limit_code: int = 0):
        self._robot = FakeCore(
            self_collision_code=self_collision_code,
            position_limit_code=position_limit_code,
        )


def test_rm65_joint_safety_report_ok_status_contains_available_checks():
    report = check_rm65_joint_safety(
        FakeRobot(),
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        side="right",
        label="12",
    )

    assert report.ok
    assert report.status_text == "position_limit=ok self_collision=ok"
    assert report.position_limit_code == 0
    assert report.self_collision_code == 0


def test_rm65_joint_safety_error_includes_rate_limited_context():
    report = check_rm65_joint_safety(
        FakeRobot(self_collision_code=1),
        [0.0, 0.0, 0.0, 35.0, 3.0, -120.0],
        side="right",
        label="37.rate12/56",
        previous_joint_deg=[0.0, 0.0, 0.0, 34.0, 3.0, -119.0],
        start_joint_deg=[-82.0, 9.0, 100.0, 31.0, 4.0, -108.0],
        context={
            "phase": "rate_limited_joint_transition",
            "joint_rate_limited": True,
            "joint_rate_limit_index": 12,
            "joint_rate_limit_count": 56,
            "joint_velocity_limit_profile": "rm65",
            "joint_velocity_limit_delta_deg": [0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
            "joint_velocity_limit_max_step_deg": [0.5, 0.75, 0.75, 1.0, 1.0, 1.0],
        },
    )

    assert not report.ok
    with pytest.raises(RuntimeError) as exc_info:
        report.raise_if_unsafe()
    message = str(exc_info.value)
    assert "right: RM65 joint safety check failed at step=37.rate12/56" in message
    assert "phase=rate_limited_joint_transition" in message
    assert "self_collision_code=1" in message
    assert "joint_deg=[0.000000, 0.000000, 0.000000, 35.000000, 3.000000, -120.000000]" in message
    assert "previous_joint_deg=[0.000000, 0.000000, 0.000000, 34.000000, 3.000000, -119.000000]" in message
    assert "joint_rate_limit_index=12" in message
    assert "joint_rate_limit_count=56" in message
    assert "joint_velocity_limit_profile=rm65" in message
    assert "rm_self_collision_code_note=0 means clear; 1 means collision or joint-position-limit" in message


def test_rm65_joint_safety_error_reports_position_limit_code():
    report = check_rm65_joint_safety(
        FakeRobot(position_limit_code=4),
        [0.0, 0.0, 0.0, 181.0, 0.0, 0.0],
        side="right",
        label="target",
        check_self_collision=False,
    )

    assert not report.ok
    assert "joint_position_limit_code=4" in report.message
    assert "position_limit_code=4" in report.message


def test_rm65_adapter_position_limit_checker_falls_back_to_joint_limits():
    class Core:
        arm_dof = 6

        def rm_algo_ikine_check_joint_position_limit(self, joint):
            raise TypeError("SDK wrapper did not convert list to c_float pointer")

        def rm_algo_get_joint_min_limit(self):
            return [-180.0] * 6

        def rm_algo_get_joint_max_limit(self):
            return [180.0] * 6

    arm = object.__new__(Arm)
    arm.dev_mode = 65
    arm._robot = Core()

    assert arm.Algo_IKine_Check_Joint_Position_Limit([0.0, 0.0, 0.0, 178.0, 32.0, -206.0]) == 6
    assert arm.Algo_IKine_Check_Joint_Position_Limit([0.0, 0.0, 0.0, 178.0, 32.0, -120.0]) == 0
