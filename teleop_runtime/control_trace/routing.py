"""Recorder-owned ControlTrace routing without control-path integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from dataset_contracts.control_trace import ControlTraceFact
from dataset_contracts.control_trace_capture import (
    CONTROL_TRACE_RETENTION_CLASSES_V1,
    ControlTraceCaptureRoutingPolicy,
)

from .runtime import ControlTraceRuntime, TraceAdmissionReceipt


CONTROL_TRACE_CAPTURE_DISPOSITIONS_V1 = frozenset(
    {"durable_journal", "transient_sink", "omitted"}
)


@dataclass(frozen=True)
class ControlTraceCaptureSelection:
    """Select persistence behavior for semantic retention classes.

    Causal core facts are always durable.  Remaining classes may be durable,
    transient, or omitted.  The selection is external recorder policy and
    never changes fact identity.
    """

    durable_retention_classes: frozenset[str] = frozenset({"causal_core"})
    transient_retention_classes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        durable = frozenset(
            str(value).strip() for value in self.durable_retention_classes
        )
        transient = frozenset(
            str(value).strip() for value in self.transient_retention_classes
        )
        unknown = (
            durable | transient
        ) - CONTROL_TRACE_RETENTION_CLASSES_V1
        if unknown:
            raise ValueError(
                "Capture selection contains unsupported retention classes: "
                f"{sorted(unknown)!r}."
            )
        if durable & transient:
            raise ValueError(
                "Durable and transient retention classes must be disjoint."
            )
        if "causal_core" not in durable:
            raise ValueError("causal_core must use the durable journal.")
        if "causal_core" in transient:
            raise ValueError("causal_core cannot be transient.")
        if "live_telemetry" in durable:
            raise ValueError(
                "live_telemetry cannot be selected for durable capture."
            )
        object.__setattr__(
            self,
            "durable_retention_classes",
            durable,
        )
        object.__setattr__(
            self,
            "transient_retention_classes",
            transient,
        )

    @property
    def omitted_retention_classes(self) -> frozenset[str]:
        return CONTROL_TRACE_RETENTION_CLASSES_V1 - (
            self.durable_retention_classes
            | self.transient_retention_classes
        )

    def disposition_for(self, retention_class: str) -> str:
        requested = str(retention_class).strip()
        if requested in self.durable_retention_classes:
            return "durable_journal"
        if requested in self.transient_retention_classes:
            return "transient_sink"
        if requested in CONTROL_TRACE_RETENTION_CLASSES_V1:
            return "omitted"
        raise ValueError(
            f"Unsupported ControlTrace retention class: {requested!r}."
        )


@dataclass(frozen=True)
class ControlTraceCaptureRoutingReceipt:
    fact_id: str
    event_type: str
    retention_class: str
    disposition: str
    trace_admission_receipt: TraceAdmissionReceipt | None
    transient_delivered: bool


class ControlTraceCaptureRouter:
    """Route facts into a journal, transient observer, or explicit omission.

    This wrapper is deliberately not installed by any production frontend.
    Existing callers continue to call ``ControlTraceRuntime.issue`` directly.
    """

    def __init__(
        self,
        *,
        runtime: ControlTraceRuntime,
        routing_policy: ControlTraceCaptureRoutingPolicy,
        selection: ControlTraceCaptureSelection,
        transient_sink: Callable[[ControlTraceFact], object] | None = None,
    ) -> None:
        if not isinstance(runtime, ControlTraceRuntime):
            raise TypeError("runtime must be a ControlTraceRuntime.")
        if not isinstance(
            routing_policy,
            ControlTraceCaptureRoutingPolicy,
        ):
            raise TypeError(
                "routing_policy must be a "
                "ControlTraceCaptureRoutingPolicy."
            )
        if not isinstance(selection, ControlTraceCaptureSelection):
            raise TypeError(
                "selection must be a ControlTraceCaptureSelection."
            )
        if transient_sink is not None and not callable(transient_sink):
            raise TypeError("transient_sink must be callable or None.")
        if (
            selection.transient_retention_classes
            and transient_sink is None
        ):
            raise ValueError(
                "A transient sink is required when transient retention "
                "classes are enabled."
            )
        required_not_durable = tuple(
            sorted(
                event_type
                for event_type in runtime.profile.required_event_types
                if selection.disposition_for(
                    routing_policy.retention_class_for(event_type)
                )
                != "durable_journal"
            )
        )
        if required_not_durable:
            raise ValueError(
                "Trace-required event types must route to the durable "
                f"journal: {required_not_durable!r}."
            )
        self.runtime = runtime
        self.routing_policy = routing_policy
        self.selection = selection
        self.transient_sink = transient_sink

    def route(
        self,
        fact: ControlTraceFact,
    ) -> ControlTraceCaptureRoutingReceipt:
        if not isinstance(fact, ControlTraceFact):
            raise TypeError("route requires a ControlTraceFact.")
        retention_class = self.routing_policy.retention_class_for(
            fact.event_type
        )
        disposition = self.selection.disposition_for(retention_class)
        admission_receipt: TraceAdmissionReceipt | None = None
        transient_delivered = False
        if disposition == "durable_journal":
            admission_receipt = self.runtime.issue(fact)
        elif disposition == "transient_sink":
            assert self.transient_sink is not None
            self.transient_sink(fact)
            transient_delivered = True
        return ControlTraceCaptureRoutingReceipt(
            fact_id=fact.fact_id,
            event_type=fact.event_type,
            retention_class=retention_class,
            disposition=disposition,
            trace_admission_receipt=admission_receipt,
            transient_delivered=transient_delivered,
        )


__all__ = [
    "CONTROL_TRACE_CAPTURE_DISPOSITIONS_V1",
    "ControlTraceCaptureRouter",
    "ControlTraceCaptureRoutingReceipt",
    "ControlTraceCaptureSelection",
]
