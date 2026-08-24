from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import check_dmtac_gripper_health as health


def _state(position: float, *, error_code: int = 0, moving: bool = False):
    payload = {
        "joint_pos": position,
        "is_moving": moving,
        "error_code": error_code,
        "status_code": 3,
    }
    return SimpleNamespace(
        joint_pos=position,
        is_moving=moving,
        error_code=error_code,
        status_code=3,
        to_dict=lambda: dict(payload),
    )


class FakeGripper:
    def __init__(
        self,
        position: float = 0.5,
        *,
        error_code: int = 0,
        stuck: bool = False,
    ) -> None:
        self.position = position
        self.error_code = error_code
        self.stuck = stuck
        self.calls: list[tuple] = []
        self.closed = False
        self.stopped = False

    def get_state(self):
        return _state(self.position, error_code=self.error_code)

    def get_speed(self):
        return [0.7]

    def get_torque(self):
        return [0.3]

    def get_continuous_grasp(self):
        return True

    def set_joint_pos(self, position, **kwargs):
        self.calls.append(("move", position, kwargs))
        if not self.stuck:
            self.position = float(position)

    def wait_until_idle(self, timeout):
        self.calls.append(("wait", timeout))
        return self.get_state()

    def set_speed(self, value):
        self.calls.append(("speed", value))

    def set_torque(self, value):
        self.calls.append(("force", value))

    def set_continuous_grasp(self, value):
        self.calls.append(("continuous", value))

    def estop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def _args(*extra: str):
    return health.build_parser().parse_args(
        [
            "--config",
            "none",
            "--left-port",
            "/dev/left",
            "--right-port",
            "/dev/right",
            *extra,
        ]
    )


def test_defaults_select_both_with_normal_bounded_settings() -> None:
    _, plans = health.build_plans(_args())

    assert [plan.endpoint.side for plan in plans] == ["left", "right"]
    assert all(plan.force == pytest.approx(0.2) for plan in plans)
    assert all(plan.speed == pytest.approx(0.2) for plan in plans)
    assert all(plan.amplitude == pytest.approx(0.04) for plan in plans)


def test_execute_requires_exact_confirmation() -> None:
    with pytest.raises(ValueError, match=health.EXECUTE_CONFIRMATION):
        health.build_plans(_args("--side", "left", "--execute"))


def test_execute_moves_low_high_restore_and_restores_settings() -> None:
    fake = FakeGripper(position=0.5)
    plan = health.GripperHealthPlan(
        endpoint=health.GripperEndpoint("left", "/dev/left"),
        amplitude=0.04,
        force=0.2,
        speed=0.2,
        wait_timeout=2.0,
    )

    result = health.execute_plan(plan, gripper_factory=lambda endpoint: fake)

    moves = [call for call in fake.calls if call[0] == "move"]
    assert [call[1] for call in moves] == pytest.approx([0.48, 0.52, 0.5])
    assert all(call[2]["force_send"] is True for call in moves)
    assert all(call[2]["continuous"] is False for call in moves)
    assert ("speed", 0.7) in fake.calls
    assert ("force", 0.3) in fake.calls
    assert ("continuous", True) in fake.calls
    assert result["ok"] is True
    assert result["settings_restored"] is True
    assert fake.closed is True
    assert fake.stopped is False


def test_preflight_fault_refuses_motion_without_clearing_fault() -> None:
    fake = FakeGripper(error_code=8)
    plan = health.GripperHealthPlan(
        endpoint=health.GripperEndpoint("right", "/dev/right"),
        amplitude=0.04,
        force=0.2,
        speed=0.2,
        wait_timeout=2.0,
    )

    with pytest.raises(RuntimeError, match="error_code=8"):
        health.execute_plan(plan, gripper_factory=lambda endpoint: fake)

    assert not [call for call in fake.calls if call[0] == "move"]
    assert fake.stopped is False
    assert fake.closed is True


def test_motion_window_keeps_full_small_span_at_limits() -> None:
    assert health.motion_window(0.0, 0.04) == pytest.approx((0.0, 0.04))
    assert health.motion_window(1.0, 0.04) == pytest.approx((0.96, 1.0))
    with pytest.raises(ValueError, match="amplitude"):
        health.motion_window(0.5, health.MAX_AMPLITUDE + 0.01)


def test_stuck_gripper_fails_measured_target_check_and_requests_stop() -> None:
    fake = FakeGripper(position=0.5, stuck=True)
    plan = health.GripperHealthPlan(
        endpoint=health.GripperEndpoint("left", "/dev/left"),
        amplitude=0.04,
        force=0.2,
        speed=0.2,
        wait_timeout=2.0,
    )

    with pytest.raises(RuntimeError, match="did not reach low target"):
        health.execute_plan(plan, gripper_factory=lambda endpoint: fake)

    assert fake.stopped is True
    assert fake.closed is True
