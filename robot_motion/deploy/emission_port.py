"""Backend-neutral ports for an already-decoded policy emission stream.

These contracts describe the boundary consumed by composition applications.
They do not own model construction, observation capture, prediction queues,
composition, safety, execution, or backend resources.  A synchronous policy,
an asynchronous chunk worker, and a remote model adapter can therefore expose
the same immutable :class:`PolicyActionEmission` stream.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Protocol, runtime_checkable

from .execution_feedback import (
    PolicyExecutionFeedback,
    PolicyExecutionFeedbackApplication,
)
from .runtime import PolicyActionEmission


@runtime_checkable
class PolicyEmissionSourcePort(Protocol):
    """Lifecycle and polling port for one lower-owned policy source."""

    @property
    def ready(self) -> bool: ...

    @property
    def last_error(self) -> str | None: ...

    @property
    def maximum_hold_sec(self) -> float: ...

    def start(self) -> None: ...

    def reset_for_activation(self) -> None: ...

    def step(
        self,
        *,
        now: float | None = None,
        step_index: int | None = None,
    ) -> PolicyActionEmission | None: ...

    def status(self, *, now: float | None = None) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


@runtime_checkable
class PolicyExecutionFeedbackPort(Protocol):
    """Optional passive feedback sink implemented by some policy sources."""

    def observe_execution_feedback(
        self,
        feedback: PolicyExecutionFeedback,
    ) -> PolicyExecutionFeedbackApplication: ...


@runtime_checkable
class PassivePolicyEmissionObserverPort(Protocol):
    """Best-effort bounded tap for an already-created formal emission.

    This port is observation-only.  Its return value and failures must never
    decide source selection, consume a policy queue, or gate execution.
    """

    nonblocking_bounded: bool

    def put_nowait(self, emission: PolicyActionEmission) -> None: ...


def require_policy_emission_source(value: Any) -> PolicyEmissionSourcePort:
    """Validate the common immutable-emission port without starting it."""

    if not isinstance(value, PolicyEmissionSourcePort):
        raise TypeError(
            "Policy source must implement the typed PolicyActionEmission port."
        )
    maximum_hold = float(value.maximum_hold_sec)
    if not math.isfinite(maximum_hold) or maximum_hold <= 0.0:
        raise ValueError(
            "Policy source maximum_hold_sec must be finite and positive."
        )
    return value


def require_passive_policy_emission_observer(
    value: Any,
) -> PassivePolicyEmissionObserverPort:
    """Reject observers that do not explicitly promise bounded non-blocking IO."""

    if not isinstance(value, PassivePolicyEmissionObserverPort):
        raise TypeError(
            "Policy emission observer must implement the passive bounded port."
        )
    if getattr(value, "nonblocking_bounded", None) is not True:
        raise ValueError(
            "Policy emission observer must declare nonblocking_bounded=True."
        )
    return value


__all__ = [
    "PolicyEmissionSourcePort",
    "PolicyExecutionFeedbackPort",
    "PassivePolicyEmissionObserverPort",
    "require_passive_policy_emission_observer",
    "require_policy_emission_source",
]
