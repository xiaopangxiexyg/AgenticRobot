"""CS-0b operation-adapter protocols and deterministic no-I/O fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .contracts import StationIntent, StationOperationResultDraft


class StationOperationAdapter(Protocol):
    """Application adapter for one source slot or execution target.

    Implementations return an operation result draft.  They do not mutate the
    station state directly, and an adapter result is not a control permit or a
    backend dispatch receipt.
    """

    @property
    def adapter_id(self) -> str:
        ...

    @property
    def subject_id(self) -> str:
        ...

    @property
    def supported_operations(self) -> tuple[str, ...]:
        ...

    def perform(self, intent: StationIntent) -> StationOperationResultDraft:
        ...


@dataclass
class DeterministicNoIoStationAdapter:
    """A deterministic adapter for contract tests and the null composition.

    The adapter intentionally has no action method and no backend handle.  It
    only acknowledges station-level connect/preflight/shadow operations.
    """

    adapter_id: str
    subject_id: str
    supported_operations: tuple[str, ...]
    clock: Callable[[], float]
    clock_domain_id: str = "station-adapter-monotonic"
    fail_operations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.adapter_id = str(self.adapter_id).strip()
        self.subject_id = str(self.subject_id).strip()
        self.clock_domain_id = str(self.clock_domain_id).strip()
        if not self.adapter_id or not self.subject_id or not self.clock_domain_id:
            raise ValueError("Adapter identity/clock must be non-empty.")
        if not callable(self.clock):
            raise TypeError("clock must be callable.")
        operations = tuple(sorted(str(value).strip() for value in self.supported_operations))
        if not operations or any(not value for value in operations):
            raise ValueError("supported_operations must be non-empty.")
        if len(set(operations)) != len(operations):
            raise ValueError("supported_operations cannot contain duplicates.")
        failures = tuple(sorted(str(value).strip() for value in self.fail_operations))
        if any(not value for value in failures) or not set(failures).issubset(operations):
            raise ValueError("fail_operations must be a supported-operation subset.")
        self.supported_operations = operations
        self.fail_operations = failures
        self.perform_count = 0

    def perform(self, intent: StationIntent) -> StationOperationResultDraft:
        if not isinstance(intent, StationIntent):
            raise TypeError("intent must be a StationIntent.")
        if intent.subject_id != self.subject_id:
            raise ValueError("Station operation belongs to another adapter subject.")
        if intent.intent_kind not in self.supported_operations:
            raise ValueError("Station operation is not supported by this adapter.")
        self.perform_count += 1
        failed = intent.intent_kind in self.fail_operations
        return StationOperationResultDraft(
            source_intent_id=intent.intent_id,
            operation_kind=intent.intent_kind,
            subject_id=self.subject_id,
            succeeded=not failed,
            completed_at=float(self.clock()),
            completed_clock_domain_id=self.clock_domain_id,
            status="configured_failure" if failed else "no_io_completed",
            evidence_refs=(
                f"no-io-adapter:{self.adapter_id}:operation:{self.perform_count}",
            ),
        )


class NullExecutionTargetAdapter(DeterministicNoIoStationAdapter):
    """Target adapter that can preflight but can never dispatch motion."""

    def __init__(
        self,
        *,
        adapter_id: str,
        target_binding_id: str,
        clock: Callable[[], float],
        clock_domain_id: str = "station-adapter-monotonic",
        fail_operations: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            adapter_id=adapter_id,
            subject_id=target_binding_id,
            supported_operations=(
                "connect_target",
                "disconnect_target",
                "run_preflight",
            ),
            clock=clock,
            clock_domain_id=clock_domain_id,
            fail_operations=fail_operations,
        )


class FakeSourceSlotAdapter(DeterministicNoIoStationAdapter):
    """No-I/O source adapter used before human/replay/policy bindings exist."""

    def __init__(
        self,
        *,
        adapter_id: str,
        source_slot_id: str,
        clock: Callable[[], float],
        clock_domain_id: str = "station-adapter-monotonic",
        fail_operations: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            adapter_id=adapter_id,
            subject_id=source_slot_id,
            supported_operations=(
                "connect_source",
                "disconnect_source",
                "run_preflight",
                "start_shadow",
                "stop_shadow",
            ),
            clock=clock,
            clock_domain_id=clock_domain_id,
            fail_operations=fail_operations,
        )


__all__ = [
    "DeterministicNoIoStationAdapter",
    "FakeSourceSlotAdapter",
    "NullExecutionTargetAdapter",
    "StationOperationAdapter",
]
