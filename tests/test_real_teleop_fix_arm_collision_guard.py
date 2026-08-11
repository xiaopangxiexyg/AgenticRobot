from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from teleop_runtime.real_teleop.fix_arm_collision_guard import (
    fixed_arm_final_collision_fault,
)
from teleop_runtime.real_teleop.command_filters import (
    _compose_real_command_filter,
    _compose_real_command_guard,
)
from teleop_runtime.real_teleop.control_loop import (
    _request_position_admittance_endpoint_hold,
)
from teleop_runtime.schemas import EffectorCommand, TeleopCommand


def _command() -> TeleopCommand:
    command = TeleopCommand(timestamp=1.0)
    for side, value in (("left", 0.1), ("right", -0.2)):
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=np.full(6, value),
                metadata={"solver_output_qpos": np.full(6, value)},
            )
        )
    return command


def test_fixed_arm_guard_allows_safe_final_pair() -> None:
    checker_calls = []

    def check_qpos_pair_collision(**kwargs):
        checker_calls.append(kwargs)
        return SimpleNamespace(
            safe=True,
            collision_pair_count=12,
            min_distance_m=0.12,
            required_min_distance_m=0.03,
        )

    fault = fixed_arm_final_collision_fault(
        _command(),
        fix_arm_controller=SimpleNamespace(fixed=True, fixed_side="left"),
        dual_arm_retargeter=SimpleNamespace(check_qpos_pair_collision=check_qpos_pair_collision),
        now=2.0,
    )

    assert fault is None
    assert len(checker_calls) == 1
    np.testing.assert_allclose(checker_calls[0]["left_qpos"], np.full(6, 0.1))
    np.testing.assert_allclose(checker_calls[0]["right_qpos"], np.full(6, -0.2))


def test_fixed_arm_guard_fails_closed_when_final_pair_is_unsafe() -> None:
    fault = fixed_arm_final_collision_fault(
        _command(),
        fix_arm_controller=SimpleNamespace(fixed=True, fixed_side="right"),
        dual_arm_retargeter=SimpleNamespace(
            check_qpos_pair_collision=lambda **_kwargs: SimpleNamespace(
                safe=False,
                reason="collision_margin_low",
                collision_pair_count=12,
                min_distance_m=0.01,
                required_min_distance_m=0.03,
            )
        ),
        now=2.0,
    )

    assert fault is not None
    assert fault.code == "fix_arm_final_collision_check_failed"
    assert fault.metadata["fixed_side"] == "right"
    assert fault.metadata["collision_pair_count"] == 12
    assert fault.metadata["reason"] == "collision_margin_low"


def test_endpoint_hold_filter_runs_after_fix_arm_filter() -> None:
    events: list[str] = []

    class _EndpointGuard:
        def filter_command(self, command: TeleopCommand, *, now: float):
            del now
            events.append("endpoint")
            return command

    class _FixController:
        @staticmethod
        def filter_command(command, observation, now):
            del observation, now
            events.append("fix")
            return command

    command_filter = _compose_real_command_filter(
        temporary_safety_limiter=None,
        fix_arm_controller=_FixController(),
        init_arm_controller=None,
        position_admittance_endpoint_guard=_EndpointGuard(),
    )

    assert command_filter is not None
    command_filter(_command(), None, 2.0)
    assert events == ["fix", "endpoint"]


def test_composed_guard_keeps_final_fix_arm_collision_fault() -> None:

    retargeter = SimpleNamespace(
        check_qpos_pair_collision=lambda **_kwargs: SimpleNamespace(
            safe=True,
            collision_pair_count=12,
            min_distance_m=0.12,
            required_min_distance_m=0.03,
        )
    )
    guard = _compose_real_command_guard(
        fix_arm_controller=SimpleNamespace(fixed=True, fixed_side="right"),
        init_arm_controller=None,
        dual_arm_retargeter=retargeter,
    )

    assert guard is not None
    assert guard(_command(), None, 2.0) is None

    retargeter.check_qpos_pair_collision = lambda **_kwargs: SimpleNamespace(
        safe=False,
        reason="collision_margin_low",
        collision_pair_count=12,
        min_distance_m=0.01,
        required_min_distance_m=0.03,
    )
    fault = guard(_command(), None, 2.1)

    assert fault is not None
    assert fault.code == "fix_arm_final_collision_check_failed"


def test_endpoint_hold_receipt_requests_standard_gui_hold_without_fault() -> None:
    command = _command()
    command.metadata["position_admittance_endpoint_hold"] = {
        "code": "position_admittance_endpoint_excursion_hold",
        "reason": "left endpoint reached",
        "requires_runtime_home": True,
    }
    requested: list[str] = []
    logged: list[tuple[str, str]] = []
    events: list[tuple[str, dict[str, object]]] = []

    clutch = SimpleNamespace(
        request_hold=lambda *, reason: requested.append(reason)
    )
    recorder = SimpleNamespace(
        active=True,
        write_event=lambda name, payload, **_kwargs: events.append((name, payload)),
    )

    consumed = _request_position_admittance_endpoint_hold(
        command,
        clutch_input=clutch,
        action_debug_log_recorder=recorder,
        log=lambda tag, message, **_kwargs: logged.append((tag, message)),
        now=2.0,
    )

    assert consumed is True
    assert requested == ["position_admittance_endpoint_limit"]
    assert logged and logged[0][0] == "PositionAdmittance"
    assert events[0][0] == "position_admittance_endpoint_hold"
