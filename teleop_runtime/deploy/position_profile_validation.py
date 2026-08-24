"""Pure position-profile admission checks used by policy deployment.

This module validates worker-isolation, CPU-affinity, reservation, and
hardware-dispatch admission flags.  It only mutates the parsed argument
namespace; it does not create workers, inspect devices, or dispatch actions.
The deployment facade supplies its CPU helpers so existing monkeypatch seams
and host-affinity behavior remain unchanged.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence


def validate_position_worker_args(
    args: argparse.Namespace,
    *,
    available_cpu_ids: Callable[[], Sequence[int]],
    cpu_thread_sibling_ids: Callable[[int], Sequence[int]],
    continuous_worker_thread: str,
    continuous_worker_process: str,
) -> None:
    """Validate and resolve position worker affinity without starting runtime state."""

    continuous_worker_isolation = (
        str(args.position_continuous_waypoint_worker_isolation).strip().lower()
    )
    if continuous_worker_isolation != continuous_worker_thread and not bool(
        args.position_continuous_waypoint_async_precompute
    ):
        raise ValueError(
            "--position-continuous-waypoint-worker-isolation process requires "
            "async_c2 precompute"
        )
    reference_worker_cpu = args.position_reference_worker_cpu
    continuous_worker_cpu = args.position_continuous_waypoint_worker_cpu
    if continuous_worker_cpu is not None and reference_worker_cpu is None:
        raise ValueError(
            "--position-continuous-waypoint-worker-cpu requires "
            "--position-reference-worker-cpu"
        )
    if bool(args.position_continuous_waypoint_async_precompute) and (
        (reference_worker_cpu is None) != (continuous_worker_cpu is None)
    ):
        raise ValueError(
            "async_c2 CPU isolation requires both "
            "--position-reference-worker-cpu and "
            "--position-continuous-waypoint-worker-cpu"
        )
    args.position_worker_cpu_affinity_allowed = None
    if reference_worker_cpu is not None:
        if continuous_worker_cpu is not None and not bool(
            args.position_continuous_waypoint_async_precompute
        ):
            raise ValueError(
                "--position-continuous-waypoint-worker-cpu requires async_c2 "
                "precompute"
            )
        if continuous_worker_cpu is not None and int(reference_worker_cpu) == int(
            continuous_worker_cpu
        ):
            raise ValueError(
                "position reference and continuous C2 workers require distinct CPUs"
            )
        allowed_cpus = available_cpu_ids()
        missing_cpus = sorted(
            (
                {int(reference_worker_cpu)}
                if continuous_worker_cpu is None
                else {
                    int(reference_worker_cpu),
                    int(continuous_worker_cpu),
                }
            ).difference(allowed_cpus)
        )
        if missing_cpus:
            raise ValueError(
                "requested position worker CPUs are outside this process affinity "
                f"mask: missing={missing_cpus} allowed={list(allowed_cpus)}"
            )
        args.position_worker_cpu_affinity_allowed = allowed_cpus
    args.position_worker_cpu_reservation_cpu_ids = ()
    if bool(args.position_worker_cpu_reservation):
        reservation_violations: list[str] = []
        if reference_worker_cpu is None:
            reservation_violations.append("--position-reference-worker-cpu is required")
        if bool(args.position_continuous_waypoint_async_precompute):
            if continuous_worker_cpu is None:
                reservation_violations.append(
                    "--position-continuous-waypoint-worker-cpu is required"
                )
            if continuous_worker_isolation != continuous_worker_process:
                reservation_violations.append(
                    "--position-continuous-waypoint-worker-isolation process is required"
                )
        if reservation_violations:
            raise ValueError(
                "--position-worker-cpu-reservation requirements are not met: "
                + "; ".join(reservation_violations)
            )
        assert args.position_worker_cpu_affinity_allowed is not None
        reservation_cpu_ids: set[int] = set()
        for worker_cpu in (reference_worker_cpu, continuous_worker_cpu):
            if worker_cpu is None:
                continue
            reservation_cpu_ids.update(
                int(cpu)
                for cpu in cpu_thread_sibling_ids(int(worker_cpu))
                if int(cpu) in args.position_worker_cpu_affinity_allowed
            )
        args.position_worker_cpu_reservation_cpu_ids = tuple(
            sorted(reservation_cpu_ids)
        )
    if bool(args.send_actions) and bool(
        args.position_continuous_waypoint_experimental_admission
    ):
        raise ValueError(
            "continuous C2 waypoint admission is not authorized for real "
            "hardware dispatch: the 2026-08-08 full-runtime trace exceeded "
            "the authoritative 60 Hz deadline. Use formal_third_order for "
            "--send-actions; re-authorize C2 only after isolated real-executor "
            "deadline soak passes."
        )


__all__ = ["validate_position_worker_args"]
