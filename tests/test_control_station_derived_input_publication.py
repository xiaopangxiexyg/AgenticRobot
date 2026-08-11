from __future__ import annotations

import pytest

from control_composition import build_composition_activation_scope
from control_composition.composition import (
    DerivedCompositionInputRoleBinding,
    build_same_source_composition_input_batch,
)
from control_station import (
    StationOperationResultDraft,
    StationPublicationBarrier,
    admit_station_operation_result,
)
from control_station.composition import (
    StationDerivedInputPublicationPort,
    StationDerivedInputPublicationProfile,
)
from tests.test_composition_input_batch_admission import _fixture


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _activation():
    return build_composition_activation_scope(
        operation_spec_id="derived-input-test",
        operation_spec_hash="derived-input-test-hash",
        session_epoch=1,
        activated_at=9.9,
        clock_domain_id="control.monotonic",
        input_barrier_at=9.9,
        required_roles=("policy", "human_residual", "human_absolute"),
    )


def _barrier() -> StationPublicationBarrier:
    barrier = StationPublicationBarrier(source_slot_id="composer")
    result = admit_station_operation_result(
        StationOperationResultDraft(
            source_intent_id="intent:start-derived-input-test",
            operation_kind="start_composition_session",
            subject_id="composer",
            succeeded=True,
            completed_at=9.9,
            completed_clock_domain_id="control.monotonic",
            status="composition_session_ready",
            evidence_refs=("virtual-derived-input-test",),
        ),
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        available_at=9.9,
        available_clock_domain_id="control.monotonic",
        admission_sequence=0,
    )
    barrier.enable_from_composition_session_result(result)
    return barrier


def _batch(action, activation):
    return build_same_source_composition_input_batch(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        role_bindings=(
            DerivedCompositionInputRoleBinding(
                input_role="human_absolute",
                producer_id="human-absolute",
                producer_epoch=1,
                action_schema_hash=action.action_schema.semantic_content_hash,
            ),
            DerivedCompositionInputRoleBinding(
                input_role="human_residual",
                producer_id="human-residual",
                producer_epoch=1,
                action_schema_hash=action.action_schema.semantic_content_hash,
            ),
        ),
        action_by_role={
            "human_absolute": action,
            "human_residual": action,
        },
        source_sequence=7,
        produced_at=9.95,
        produced_clock_domain_id="human-device.monotonic",
        effective_from=10.0,
        effective_until=10.5,
        effective_clock_domain_id="control.monotonic",
        source_fact_ids=("human-sample:7",),
        source_fact_hashes=("human-sample-hash:7",),
        activation_scope=activation,
        observation_cut_id="human-sample:7",
    )


def test_default_off_port_does_not_fetch_batch() -> None:
    port = StationDerivedInputPublicationPort(
        profile=StationDerivedInputPublicationProfile()
    )

    cycle = port.publish(
        lambda: pytest.fail("disabled publication fetched its batch")
    )

    assert cycle.status == "disabled"
    assert cycle.batch is None
    assert cycle.receipt is None


def test_enabled_port_admits_exact_batch_under_session_barrier() -> None:
    _clock, _coordinator, store, action = _fixture()
    activation = _activation()
    barrier = _barrier()
    port = StationDerivedInputPublicationPort(
        profile=StationDerivedInputPublicationProfile(
            profile_id="test-derived-input-publication",
            enabled=True,
        ),
        activation_scope=activation,
        publication_barrier=barrier,
        admit_batch=lambda batch: store.admit_batch(
            batch,
            activation_scope=activation,
        ),
    )

    cycle = port.publish(lambda: _batch(action, activation))

    assert cycle.status == "admitted"
    assert cycle.receipt.accepted
    assert cycle.receipt.batch_id == cycle.batch.batch_id
    assert len(store) == 2


def test_closed_barrier_blocks_before_batch_factory() -> None:
    _clock, _coordinator, store, action = _fixture()
    activation = _activation()
    barrier = _barrier()
    port = StationDerivedInputPublicationPort(
        profile=StationDerivedInputPublicationProfile(
            profile_id="test-derived-input-publication",
            enabled=True,
        ),
        activation_scope=activation,
        publication_barrier=barrier,
        admit_batch=lambda batch: store.admit_batch(
            batch,
            activation_scope=activation,
        ),
    )
    barrier.disable(reason="test_hold")
    calls = []

    with pytest.raises(RuntimeError, match="barrier is disabled"):
        port.publish(lambda: (calls.append(True) or _batch(action, activation)))

    assert calls == []
    assert len(store) == 0
