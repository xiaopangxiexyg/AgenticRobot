from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class CommandDispatchReceipt:
    """Transport-neutral result for one effector command dispatch.

    Backends that can determine transport acceptance should return this receipt.
    Legacy backends may continue returning ``None`` or backend-specific values;
    an explicit rejection must be raised as :class:`BackendDispatchError` so a
    caller cannot mistake the command for a successful dispatch.
    """

    effector_name: str
    accepted: bool
    operation: str
    status: str
    timestamp: float = field(default_factory=time.time)
    result_code: int | None = None
    result_text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "effector_name", str(self.effector_name))
        object.__setattr__(self, "operation", str(self.operation))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "result_text", str(self.result_text))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def require_accepted(self) -> "CommandDispatchReceipt":
        if not self.accepted:
            raise BackendDispatchError(self)
        return self


class BackendDispatchError(RuntimeError):
    """A backend transport explicitly rejected or could not confirm a command."""

    def __init__(self, receipt: CommandDispatchReceipt):
        self.receipt = receipt
        result = (
            "unknown"
            if receipt.result_code is None
            else str(int(receipt.result_code))
        )
        detail = receipt.result_text or receipt.status
        super().__init__(
            f"{receipt.effector_name} {receipt.operation} was not accepted "
            f"(result_code={result}, status={receipt.status}): {detail}"
        )


class RigDispatchError(RuntimeError):
    """One or more effectors failed during a coordinated rig dispatch."""

    def __init__(
        self,
        *,
        rig_name: str,
        failures: Mapping[str, Exception],
        hold_error: Exception | None = None,
    ):
        self.rig_name = str(rig_name)
        self.failures = dict(failures)
        self.hold_error = hold_error
        failure_text = "; ".join(
            f"{name}: {type(error).__name__}: {error}"
            for name, error in self.failures.items()
        )
        message = (
            f"{self.rig_name} dispatch failed; group fail-stop requested: {failure_text}"
        )
        if hold_error is not None:
            message += f"; fail-stop error: {type(hold_error).__name__}: {hold_error}"
        super().__init__(message)


__all__ = [
    "BackendDispatchError",
    "CommandDispatchReceipt",
    "RigDispatchError",
]
