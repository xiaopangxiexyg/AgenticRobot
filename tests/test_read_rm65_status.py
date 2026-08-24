from __future__ import annotations

from pathlib import Path

from scripts import read_rm65_status as status


class FakeRobot:
    def __init__(self, *, failed_query: str | None = None) -> None:
        self.failed_query = failed_query
        self.calls: list[str] = []

    def _reply(self, name: str, data):
        self.calls.append(name)
        return (7 if self.failed_query == name else 0), data

    def rm_get_robot_info(self):
        return self._reply("robot_info", {"arm_model": 65, "arm_dof": 6})

    def rm_get_arm_software_info(self):
        return self._reply("arm_software_info", {"product_version": "test"})

    def rm_get_current_arm_state(self):
        return self._reply("current_arm_state", {"joint": [0.0] * 6, "pose": [0.0] * 6})

    def rm_get_joint_en_state(self):
        return self._reply("joint_enable_state", [1] * 6)

    def rm_get_current_joint_temperature(self):
        return self._reply("joint_temperature_c", [30.0] * 6)

    def rm_get_current_joint_current(self):
        return self._reply("joint_current_ma", [100.0] * 6)

    def rm_get_current_joint_voltage(self):
        return self._reply("joint_voltage_v", [48.0] * 6)

    def rm_get_joint_err_flag(self):
        self.calls.append("joint_error_and_brake_state")
        return {"return_code": 0, "err_flag": [0] * 6, "brake_state": [0] * 6}


class FakeArm:
    def __init__(self, *, failed_query: str | None = None) -> None:
        self._robot = FakeRobot(failed_query=failed_query)
        self.closed = False

    def API_Version(self):
        return b"1.2.3"

    def close(self):
        self.closed = True
        return 0


def test_defaults_select_all_from_assignment() -> None:
    args = status.build_parser().parse_args([])

    config_path, endpoints = status.resolve_endpoints(
        args,
        assignment_loader=lambda value: (
            Path("/tmp/arms.json"),
            {
                "assignments": {
                    "left": {"arm_ip": "192.0.2.1"},
                    "right": {"arm_ip": "192.0.2.2"},
                }
            },
        ),
    )

    assert config_path == Path("/tmp/arms.json")
    assert endpoints == [
        status.ArmEndpoint("left", "192.0.2.1"),
        status.ArmEndpoint("right", "192.0.2.2"),
    ]


def test_explicit_side_and_ip_need_no_config() -> None:
    args = status.build_parser().parse_args(
        ["--config", "none", "--side", "right", "--right-ip", "192.0.2.9"]
    )

    config_path, endpoints = status.resolve_endpoints(args)

    assert config_path is None
    assert endpoints == [status.ArmEndpoint("right", "192.0.2.9")]


def test_status_reader_uses_only_read_queries_and_closes_connection() -> None:
    fake = FakeArm()

    result = status.read_arm_status(
        status.ArmEndpoint("left", "192.0.2.1"),
        arm_factory=lambda endpoint: fake,
    )

    assert result["ok"] is True
    assert result["mode"] == "read_only"
    assert result["api_version"] == "1.2.3"
    assert set(fake._robot.calls) == {
        "robot_info",
        "arm_software_info",
        "current_arm_state",
        "joint_enable_state",
        "joint_temperature_c",
        "joint_current_ma",
        "joint_voltage_v",
        "joint_error_and_brake_state",
    }
    assert fake.closed is True


def test_nonzero_query_code_is_reported_without_hiding_other_status() -> None:
    fake = FakeArm(failed_query="current_arm_state")

    result = status.read_arm_status(
        status.ArmEndpoint("right", "192.0.2.2"),
        arm_factory=lambda endpoint: fake,
    )

    assert result["ok"] is False
    assert result["queries"]["current_arm_state"]["return_code"] == 7
    assert "current_arm_state return_code=7" in result["errors"]
    assert "joint_voltage_v" in result["queries"]
    assert fake.closed is True
