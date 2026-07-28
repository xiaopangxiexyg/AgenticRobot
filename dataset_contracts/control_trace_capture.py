"""Pure recorder-owned routing contracts for ControlTrace facts.

This module deliberately does not alter ``ControlTraceFact`` or decide which
facts a runtime must produce.  ``TraceRequirementProfile`` owns completeness;
this contract classifies already-produced facts for an external recorder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .canonical_encoding import canonical_content_hash
from .control_trace import CONTROL_TRACE_EVENT_TYPES_V24


CONTROL_TRACE_CAPTURE_ROUTING_POLICY_SCHEMA_V1 = (
    "dexteleop.control_trace_capture_routing_policy.v1"
)
CONTROL_TRACE_CAPTURE_ROUTING_POLICY_HASH_DOMAIN_V1 = (
    "dexteleop.control_trace_capture_routing_policy.semantic.v1"
)
CONTROL_TRACE_RETENTION_CLASSES_V1 = frozenset(
    {
        "causal_core",
        "declared_observation",
        "diagnostic_debug",
        "live_telemetry",
    }
)


def _required(value: object, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


@dataclass(frozen=True)
class ControlTraceCaptureRoute:
    """Recorder classification for one registered ControlTrace event type."""

    event_type: str
    retention_class: str

    def __post_init__(self) -> None:
        event_type = _required(self.event_type, name="event_type")
        if event_type not in CONTROL_TRACE_EVENT_TYPES_V24:
            raise ValueError(
                f"Unknown ControlTrace event type: {event_type!r}."
            )
        retention_class = _required(
            self.retention_class,
            name="retention_class",
        )
        if retention_class not in CONTROL_TRACE_RETENTION_CLASSES_V1:
            raise ValueError(
                f"Unsupported ControlTrace retention class: "
                f"{retention_class!r}."
            )
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "retention_class", retention_class)

    def semantic_payload(self) -> dict[str, str]:
        return {
            "event_type": self.event_type,
            "retention_class": self.retention_class,
        }


@dataclass(frozen=True)
class ControlTraceCaptureRoutingPolicy:
    """Complete, immutable classification owned by an external recorder.

    Every currently registered event type must be classified exactly once.
    This intentionally fails when a selective policy is not updated for a new
    event type instead of silently retaining or dropping that new fact.
    """

    policy_id: str
    routes: tuple[ControlTraceCaptureRoute, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=CONTROL_TRACE_CAPTURE_ROUTING_POLICY_SCHEMA_V1,
        init=False,
    )
    _route_by_event_type: Mapping[str, ControlTraceCaptureRoute] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        policy_id = _required(self.policy_id, name="policy_id")
        routes = tuple(self.routes)
        if any(
            not isinstance(route, ControlTraceCaptureRoute)
            for route in routes
        ):
            raise TypeError(
                "routes must contain ControlTraceCaptureRoute values."
            )
        event_types = tuple(route.event_type for route in routes)
        if len(event_types) != len(set(event_types)):
            raise ValueError(
                "ControlTrace capture routes must classify each event exactly "
                "once."
            )
        covered = frozenset(event_types)
        missing = CONTROL_TRACE_EVENT_TYPES_V24 - covered
        if missing:
            raise ValueError(
                "ControlTrace capture routing policy must classify every "
                f"registered event type; missing {sorted(missing)!r}."
            )
        extra = covered - CONTROL_TRACE_EVENT_TYPES_V24
        if extra:
            raise ValueError(
                "ControlTrace capture routing policy contains unknown events: "
                f"{sorted(extra)!r}."
            )
        ordered = tuple(sorted(routes, key=lambda route: route.event_type))
        route_by_event_type = MappingProxyType(
            {route.event_type: route for route in ordered}
        )
        semantic_payload = {
            "schema": self.schema,
            "policy_id": policy_id,
            "routes": tuple(route.semantic_payload() for route in ordered),
        }
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "routes", ordered)
        object.__setattr__(self, "_route_by_event_type", route_by_event_type)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                semantic_payload,
                domain=(
                    CONTROL_TRACE_CAPTURE_ROUTING_POLICY_HASH_DOMAIN_V1
                ),
            ),
        )

    def route_for_event_type(
        self,
        event_type: str,
    ) -> ControlTraceCaptureRoute:
        requested = _required(event_type, name="event_type")
        try:
            return self._route_by_event_type[requested]
        except KeyError as exc:
            raise ValueError(
                f"Event type {requested!r} is not classified by "
                f"{self.policy_id!r}."
            ) from exc

    def retention_class_for(self, event_type: str) -> str:
        return self.route_for_event_type(event_type).retention_class

    def event_types_for(
        self,
        retention_class: str,
    ) -> frozenset[str]:
        requested = _required(
            retention_class,
            name="retention_class",
        )
        if requested not in CONTROL_TRACE_RETENTION_CLASSES_V1:
            raise ValueError(
                f"Unsupported ControlTrace retention class: {requested!r}."
            )
        return frozenset(
            route.event_type
            for route in self.routes
            if route.retention_class == requested
        )

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "routes": tuple(
                route.semantic_payload() for route in self.routes
            ),
            "semantic_content_hash": self.semantic_content_hash,
        }


def build_preserve_all_capture_routing_policy(
    *,
    policy_id: str = "control-trace-preserve-all-compat-v1",
) -> ControlTraceCaptureRoutingPolicy:
    """Return an explicit compatibility policy that drops nothing.

    ``causal_core`` here means only that every fact is retained by the legacy
    compatibility path.  It is not a semantic claim that every event is
    required by every replay capability.
    """

    return ControlTraceCaptureRoutingPolicy(
        policy_id=policy_id,
        routes=tuple(
            ControlTraceCaptureRoute(
                event_type=event_type,
                retention_class="causal_core",
            )
            for event_type in sorted(CONTROL_TRACE_EVENT_TYPES_V24)
        ),
    )


__all__ = [
    "CONTROL_TRACE_CAPTURE_ROUTING_POLICY_HASH_DOMAIN_V1",
    "CONTROL_TRACE_CAPTURE_ROUTING_POLICY_SCHEMA_V1",
    "CONTROL_TRACE_RETENTION_CLASSES_V1",
    "ControlTraceCaptureRoute",
    "ControlTraceCaptureRoutingPolicy",
    "build_preserve_all_capture_routing_policy",
]
