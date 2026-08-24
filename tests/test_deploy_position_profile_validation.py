from __future__ import annotations

from types import SimpleNamespace

import pytest

from teleop_runtime.deploy.position_profile_validation import (
    validate_position_worker_args,
)


def _args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "position_continuous_waypoint_worker_isolation": "thread",
        "position_continuous_waypoint_async_precompute": False,
        "position_reference_worker_cpu": None,
        "position_continuous_waypoint_worker_cpu": None,
        "position_worker_cpu_reservation": False,
        "position_continuous_waypoint_experimental_admission": False,
        "send_actions": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _validate(args: SimpleNamespace, *, allowed: tuple[int, ...] = (2, 3, 7)) -> None:
    validate_position_worker_args(
        args,
        available_cpu_ids=lambda: allowed,
        cpu_thread_sibling_ids=lambda cpu: (cpu - 1, cpu, cpu + 4),
        continuous_worker_thread="thread",
        continuous_worker_process="process",
    )


def test_defaults_clear_worker_affinity_and_reservation_outputs() -> None:
    args = _args()

    _validate(args)

    assert args.position_worker_cpu_affinity_allowed is None
    assert args.position_worker_cpu_reservation_cpu_ids == ()


def test_cpu_affinity_and_reservation_use_supplied_host_helpers() -> None:
    args = _args(
        position_continuous_waypoint_worker_isolation="process",
        position_continuous_waypoint_async_precompute=True,
        position_reference_worker_cpu=3,
        position_continuous_waypoint_worker_cpu=7,
        position_worker_cpu_reservation=True,
    )

    _validate(args, allowed=(2, 3, 7))

    assert args.position_worker_cpu_affinity_allowed == (2, 3, 7)
    assert args.position_worker_cpu_reservation_cpu_ids == (2, 3, 7)


def test_worker_cpu_outside_affinity_fails_closed() -> None:
    args = _args(position_reference_worker_cpu=11)

    with pytest.raises(ValueError, match="outside this process affinity mask"):
        _validate(args, allowed=(2, 3, 7))


def test_real_dispatch_rejects_continuous_c2_admission() -> None:
    args = _args(
        position_continuous_waypoint_experimental_admission=True,
        send_actions=True,
    )

    with pytest.raises(ValueError, match="not authorized for real hardware dispatch"):
        _validate(args)


def test_process_isolation_requires_async_precompute() -> None:
    args = _args(position_continuous_waypoint_worker_isolation="process")

    with pytest.raises(ValueError, match="requires async_c2 precompute"):
        _validate(args)
