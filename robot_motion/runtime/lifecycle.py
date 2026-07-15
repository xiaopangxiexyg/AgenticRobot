from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Iterable


CleanupStep = tuple[str, Callable[[], None]]


@dataclass(frozen=True)
class LifecycleFailure:
    """One failed step from a best-effort lifecycle operation."""

    label: str
    error: BaseException


class LifecycleCleanupError(RuntimeError):
    """Aggregates cleanup failures without replacing an active primary error.

    ``primary_error`` is populated when cleanup runs from an exception's
    ``finally``/``__exit__`` path.  In that case it is also the explicit cause;
    individual cleanup errors remain available in ``failures``.
    """

    def __init__(
        self,
        operation: str,
        failures: Iterable[LifecycleFailure],
        *,
        primary_error: BaseException | None = None,
    ) -> None:
        self.operation = str(operation)
        self.failures = tuple(failures)
        self.primary_error = primary_error
        if not self.failures:
            raise ValueError("LifecycleCleanupError requires at least one failure.")
        details = "; ".join(
            f"{failure.label}: {type(failure.error).__name__}: {failure.error}"
            for failure in self.failures
        )
        message = f"{self.operation} failed in {len(self.failures)} step(s): {details}"
        if self.primary_error is not None:
            message = (
                f"{self.operation} cleanup failed while handling primary "
                f"{type(self.primary_error).__name__}: {self.primary_error}; "
                f"cleanup failed in {len(self.failures)} step(s): {details}"
            )
        super().__init__(message)


class LifecycleStartupError(RuntimeError):
    """Reports a startup failure together with any incomplete rollback."""

    def __init__(
        self,
        operation: str,
        start_error: BaseException,
        rollback_failures: Iterable[LifecycleFailure],
    ) -> None:
        self.operation = str(operation)
        self.start_error = start_error
        self.rollback_failures = tuple(rollback_failures)
        details = "; ".join(
            f"{failure.label}: {type(failure.error).__name__}: {failure.error}"
            for failure in self.rollback_failures
        )
        super().__init__(
            f"{self.operation} failed: {type(start_error).__name__}: {start_error}; "
            f"rollback failed in {len(self.rollback_failures)} step(s): {details}"
        )


def collect_lifecycle_failures(
    steps: Iterable[CleanupStep],
) -> tuple[LifecycleFailure, ...]:
    """Run all steps in order and return every failure without short-circuiting.

    ``BaseException`` is intentionally captured here: teardown must still reach
    later hardware transports and writer threads even when a cleanup hook raises
    an interrupt-like exception.  The aggregate is raised only after all steps
    have received a cleanup attempt.
    """

    failures: list[LifecycleFailure] = []
    for label, callback in steps:
        try:
            callback()
        except BaseException as error:  # noqa: BLE001 - lifecycle must not short-circuit.
            failures.append(LifecycleFailure(label=str(label), error=error))
    return tuple(failures)


def raise_lifecycle_failures(
    operation: str,
    failures: Iterable[LifecycleFailure],
    *,
    primary_error: BaseException | None = None,
) -> None:
    resolved = tuple(failures)
    if not resolved:
        return
    active_primary = primary_error
    if active_primary is None:
        active_primary = sys.exc_info()[1]
    error = LifecycleCleanupError(
        operation,
        resolved,
        primary_error=active_primary,
    )
    raise error from (
        active_primary if active_primary is not None else resolved[0].error
    )


def run_best_effort_cleanup(
    operation: str,
    steps: Iterable[CleanupStep],
) -> None:
    active_primary = sys.exc_info()[1]
    failures = collect_lifecycle_failures(steps)
    raise_lifecycle_failures(
        operation,
        failures,
        primary_error=active_primary,
    )


__all__ = [
    "CleanupStep",
    "LifecycleCleanupError",
    "LifecycleFailure",
    "LifecycleStartupError",
    "collect_lifecycle_failures",
    "raise_lifecycle_failures",
    "run_best_effort_cleanup",
]
