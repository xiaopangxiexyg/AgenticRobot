from __future__ import annotations

from dataclasses import replace

import pytest

from dataset_contracts.control_trace import (
    CONTROL_TRACE_EVENT_TYPES_V24,
    ControlTraceFact,
)
from dataset_contracts.control_trace_capture import (
    ControlTraceCaptureRoute,
    ControlTraceCaptureRoutingPolicy,
    build_preserve_all_capture_routing_policy,
)
from teleop_runtime.control_trace.lifecycle import RuntimeTraceSession
from teleop_runtime.control_trace.routing import (
    ControlTraceCaptureRouter,
    ControlTraceCaptureSelection,
)
from teleop_runtime.control_trace.runtime import ControlTraceRuntime


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _policy(
    *,
    policy_id: str = "test-semantic-routing-v1",
) -> ControlTraceCaptureRoutingPolicy:
    declared_observation = {
        "source_difference_observation",
        "component_mixing_shadow",
    }
    diagnostic_debug = {"arbiter_shadow_status"}
    live_telemetry = {"arbitration_computation"}
    return ControlTraceCaptureRoutingPolicy(
        policy_id=policy_id,
        routes=tuple(
            ControlTraceCaptureRoute(
                event_type=event_type,
                retention_class=(
                    "declared_observation"
                    if event_type in declared_observation
                    else "diagnostic_debug"
                    if event_type in diagnostic_debug
                    else "live_telemetry"
                    if event_type in live_telemetry
                    else "causal_core"
                ),
            )
            for event_type in reversed(sorted(CONTROL_TRACE_EVENT_TYPES_V24))
        ),
    )


def _terminal_fact() -> ControlTraceFact:
    return ControlTraceFact.create(
        fact_id="terminal-1",
        event_type="runtime_trace_terminal",
        runtime_session_id="runtime-1",
        control_domain_id="robot-1",
        embodiment_instance_id="virtual-1",
        occurred_at=1.0,
        clock_domain_id="control.monotonic",
        payload={
            "terminal_status": "completed",
            "reason": "test_complete",
            "occurred_at": 1.0,
        },
    )


def _source_validation_fact() -> ControlTraceFact:
    return ControlTraceFact.create(
        fact_id="source-validation-1",
        event_type="source_validation",
        runtime_session_id="runtime-1",
        control_domain_id="robot-1",
        embodiment_instance_id="virtual-1",
        occurred_at=0.5,
        clock_domain_id="control.monotonic",
        payload={
            "validation_id": "source-validation-1",
            "source_event_id": "source-1",
            "accepted": True,
            "reasons": (),
        },
    )


def _runtime() -> ControlTraceRuntime:
    return ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="trace-1",
            runtime_session_id="runtime-1",
            control_domain_id="robot-1",
            profile_id="plain_dry_run_v1",
            started_at=0.0,
            clock_domain_id="control.monotonic",
        )
    )


def test_policy_classifies_every_event_exactly_once_and_is_order_stable() -> None:
    policy = _policy()
    reordered = ControlTraceCaptureRoutingPolicy(
        policy_id=policy.policy_id,
        routes=tuple(reversed(policy.routes)),
    )

    assert policy.semantic_content_hash == reordered.semantic_content_hash
    assert policy.retention_class_for("runtime_trace_terminal") == "causal_core"
    assert (
        policy.retention_class_for("source_difference_observation")
        == "declared_observation"
    )
    assert (
        policy.retention_class_for("arbiter_shadow_status")
        == "diagnostic_debug"
    )
    assert (
        policy.retention_class_for("arbitration_computation")
        == "live_telemetry"
    )
    assert set().union(
        *(
            policy.event_types_for(retention_class)
            for retention_class in (
                "causal_core",
                "declared_observation",
                "diagnostic_debug",
                "live_telemetry",
            )
        )
    ) == set(CONTROL_TRACE_EVENT_TYPES_V24)


def test_policy_rejects_missing_duplicate_and_unsupported_routes() -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="must classify every"):
        ControlTraceCaptureRoutingPolicy(
            policy_id="missing-v1",
            routes=policy.routes[:-1],
        )
    with pytest.raises(ValueError, match="exactly once"):
        ControlTraceCaptureRoutingPolicy(
            policy_id="duplicate-v1",
            routes=(*policy.routes, policy.routes[0]),
        )
    with pytest.raises(ValueError, match="retention class"):
        replace(policy.routes[0], retention_class="unknown")
    with pytest.raises(ValueError, match="Unknown ControlTrace"):
        ControlTraceCaptureRoute(
            event_type="future_unregistered_event",
            retention_class="causal_core",
        )


def test_routing_lookup_does_not_change_fact_identity_or_payload() -> None:
    fact = _terminal_fact()
    payload_before = fact.payload
    hash_before = fact.semantic_content_hash

    assert _policy().retention_class_for(fact.event_type) == "causal_core"
    assert fact.payload is payload_before
    assert fact.semantic_content_hash == hash_before


def test_preserve_all_policy_is_explicit_no_drop_compatibility() -> None:
    policy = build_preserve_all_capture_routing_policy()

    assert policy.event_types_for("causal_core") == (
        CONTROL_TRACE_EVENT_TYPES_V24
    )
    assert policy.event_types_for("declared_observation") == frozenset()
    assert policy.event_types_for("diagnostic_debug") == frozenset()
    assert policy.event_types_for("live_telemetry") == frozenset()


def test_router_omits_debug_without_consuming_journal_sequence() -> None:
    policy = _policy()
    source_route = policy.route_for_event_type("source_validation")
    policy = ControlTraceCaptureRoutingPolicy(
        policy_id="source-validation-debug-v1",
        routes=tuple(
            replace(route, retention_class="diagnostic_debug")
            if route.event_type == source_route.event_type
            else route
            for route in policy.routes
        ),
    )
    runtime = _runtime()
    router = ControlTraceCaptureRouter(
        runtime=runtime,
        routing_policy=policy,
        selection=ControlTraceCaptureSelection(),
    )

    omitted = router.route(_source_validation_fact())
    admitted = router.route(_terminal_fact())

    assert omitted.disposition == "omitted"
    assert omitted.trace_admission_receipt is None
    assert admitted.disposition == "durable_journal"
    assert admitted.trace_admission_receipt is not None
    assert admitted.trace_admission_receipt.accepted
    assert runtime.snapshot().journal_issued_sequence == 1


def test_router_rejects_selection_that_omits_required_trace_fact() -> None:
    policy = _policy()
    policy = ControlTraceCaptureRoutingPolicy(
        policy_id="terminal-debug-v1",
        routes=tuple(
            replace(route, retention_class="diagnostic_debug")
            if route.event_type == "runtime_trace_terminal"
            else route
            for route in policy.routes
        ),
    )
    with pytest.raises(ValueError, match="Trace-required event types"):
        ControlTraceCaptureRouter(
            runtime=_runtime(),
            routing_policy=policy,
            selection=ControlTraceCaptureSelection(),
        )


def test_router_can_deliver_live_fact_without_journal_admission() -> None:
    delivered: list[ControlTraceFact] = []
    runtime = _runtime()
    policy = _policy()
    policy = ControlTraceCaptureRoutingPolicy(
        policy_id="source-validation-live-v1",
        routes=tuple(
            replace(route, retention_class="live_telemetry")
            if route.event_type == "source_validation"
            else route
            for route in policy.routes
        ),
    )
    router = ControlTraceCaptureRouter(
        runtime=runtime,
        routing_policy=policy,
        selection=ControlTraceCaptureSelection(
            transient_retention_classes=frozenset({"live_telemetry"})
        ),
        transient_sink=delivered.append,
    )
    live_fact = _source_validation_fact()

    receipt = router.route(live_fact)

    assert receipt.disposition == "transient_sink"
    assert receipt.transient_delivered
    assert delivered == [live_fact]
    assert runtime.snapshot().journal_issued_sequence == 0
