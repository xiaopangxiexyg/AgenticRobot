from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from control_composition.composition import (
    CompositionInputBatchDraft,
    CompositionInputDraft,
    CompositionInputStore,
    DerivedCompositionInputRoleBinding,
    build_same_source_composition_input_batch,
    composition_input_semantic_payload,
)
from control_composition import build_composition_activation_scope
from control_composition.events import CompositionDomainEvent
from control_composition.retention import TimeWindowRetentionSpec
from dataset_interface.composition_control_trace import (
    project_composition_domain_event,
)
from robot_motion.action import (
    BIMANUAL_QPOS_V1,
    ActionComponentCatalog,
    ActionComponentSpec,
    RobotAction,
    build_robot_action_v1_runtime_schema,
    project_robot_action_v1_to_component_action,
)
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _fixture():
    clock = _Clock()
    coordinator = ControlAdmissionCoordinator(
        "robot-0",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="control.monotonic",
            runtime_session_id="session-0",
            boot_id="boot-0",
            reset_epoch=0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="session-0",
        coordinator=coordinator,
    )
    for producer_id in ("human-absolute", "human-residual"):
        epochs.activate(
            producer_id=producer_id,
            producer_epoch=1,
            produced_at=9.9,
            produced_clock_domain_id=f"{producer_id}.monotonic",
            reason="test_startup",
        )
    schema = build_robot_action_v1_runtime_schema(
        component_catalog=ActionComponentCatalog(
            catalog_id="batch-input-test",
            action_space=BIMANUAL_QPOS_V1,
            components=(
                ActionComponentSpec("left_arm", "left_arm", "left_arm"),
            ),
        )
    )
    store = CompositionInputStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        retention_spec=TimeWindowRetentionSpec(
            spec_id="batch-input-window",
            history_window_sec=1.0,
            max_future_lead_sec=1.0,
            max_effective_duration_sec=1.0,
        ),
        action_schema=schema,
        input_action_schema_by_role={
            "human_absolute": schema,
            "human_residual": schema,
        },
    )
    action = project_robot_action_v1_to_component_action(
        RobotAction(
            action_space=BIMANUAL_QPOS_V1,
            arm_qpos_by_side={
                "left": np.zeros(6, dtype=np.float32),
            },
            source="batch-input-test",
        ),
        action_schema=schema,
    )
    return clock, coordinator, store, action


def _draft(*, role: str, producer_id: str, action, sequence: int = 0):
    return CompositionInputDraft(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        producer_id=producer_id,
        producer_epoch=1,
        sequence=sequence,
        input_role=role,
        action=action,
        produced_at=9.95,
        produced_clock_domain_id="human-device.monotonic",
        effective_from=10.0,
        effective_until=10.5,
        effective_clock_domain_id="control.monotonic",
        observation_cut_id="human-sample:7",
        source_fact_ids=("human-sample:7",),
        source_fact_hashes=("human-sample-hash:7",),
    )


def test_multi_role_batch_has_one_availability_stamp_and_atomic_cut() -> None:
    clock, coordinator, store, action = _fixture()

    receipt = store.admit_batch(
        CompositionInputBatchDraft(
            inputs=(
                _draft(
                    role="human_absolute",
                    producer_id="human-absolute",
                    action=action,
                ),
                _draft(
                    role="human_residual",
                    producer_id="human-residual",
                    action=action,
                ),
            )
        )
    )

    assert receipt.accepted
    assert len(receipt.input_receipts) == 2
    assert receipt.first_control_admission_sequence + 1 == (
        receipt.last_control_admission_sequence
    )
    assert {
        item.composition_input.available_at for item in receipt.input_receipts
    } == {10.0}
    assert coordinator.committed_high_watermark == (
        receipt.last_control_admission_sequence
    )
    cut = coordinator.capture_cut(cutoff_at=clock.value).cut
    assert {
        record.input_role for record in store.records_at_or_before(cut)
    } == {"human_absolute", "human_residual"}


def test_batch_admission_projects_atomic_receipt_without_action_copies() -> None:
    clock, _coordinator, store, action = _fixture()
    receipt = store.admit_batch(
        CompositionInputBatchDraft(
            inputs=(
                _draft(
                    role="human_absolute",
                    producer_id="human-absolute",
                    action=action,
                ),
                _draft(
                    role="human_residual",
                    producer_id="human-residual",
                    action=action,
                ),
            )
        )
    )

    fact = project_composition_domain_event(
        AuthoritativeTraceContext(
            runtime_session_id="session-0",
            control_domain_id="robot-0",
            embodiment_instance_id="virtual-robot",
            clock_domain_id="control.monotonic",
        ),
        CompositionDomainEvent(
            event_kind="composition_input_batch_admission",
            subject=receipt,
            occurred_at=clock.value,
        ),
    )

    assert fact.event_type == "composition_input_batch_admission"
    assert fact.payload["batch_id"] == receipt.batch_id
    assert fact.payload["accepted"] is True
    assert fact.payload["input_ids"] == tuple(
        item.composition_input_id for item in receipt.input_receipts
    )
    assert "action" not in fact.payload


def test_batch_prevalidation_failure_admits_nothing() -> None:
    _clock, coordinator, store, action = _fixture()
    first = _draft(
        role="human_absolute",
        producer_id="human-absolute",
        action=action,
    )
    invalid = replace(
        _draft(
            role="human_residual",
            producer_id="human-residual",
            action=action,
        ),
        runtime_session_id="another-session",
    )

    receipt = store.admit_batch(
        CompositionInputBatchDraft(inputs=(first, invalid))
    )

    assert receipt.accepted is False
    assert receipt.rejection_code == "session_mismatch"
    assert len(store) == 0
    assert coordinator.committed_high_watermark == 2


def test_batch_insert_failure_rolls_back_every_role() -> None:
    _clock, coordinator, store, action = _fixture()
    previous = store.admit(
        _draft(
            role="human_absolute",
            producer_id="human-absolute",
            action=action,
            sequence=1,
        )
    )
    assert previous.accepted
    before = coordinator.committed_high_watermark

    receipt = store.admit_batch(
        CompositionInputBatchDraft(
            inputs=(
                _draft(
                    role="human_residual",
                    producer_id="human-residual",
                    action=action,
                    sequence=0,
                ),
                _draft(
                    role="human_absolute",
                    producer_id="human-absolute",
                    action=action,
                    sequence=0,
                ),
            )
        )
    )

    assert receipt.accepted is False
    assert receipt.rejection_code == "sequence_rollback"
    assert len(store) == 1
    assert coordinator.committed_high_watermark == before


def test_batch_rejects_mixed_source_lineage() -> None:
    _clock, coordinator, store, action = _fixture()
    first = _draft(
        role="human_absolute",
        producer_id="human-absolute",
        action=action,
    )
    other_source = replace(
        _draft(
            role="human_residual",
            producer_id="human-residual",
            action=action,
        ),
        source_fact_ids=("human-sample:8",),
        source_fact_hashes=("human-sample-hash:8",),
    )

    receipt = store.admit_batch(
        CompositionInputBatchDraft(inputs=(first, other_source))
    )

    assert receipt.accepted is False
    assert receipt.rejection_code == "batch_causal_mismatch"
    assert len(store) == 0
    assert coordinator.committed_high_watermark == 2


def test_batch_without_policy_output_keeps_legacy_identity() -> None:
    _clock, _coordinator, _store, action = _fixture()
    batch = CompositionInputBatchDraft(
        inputs=(
            _draft(
                role="human_absolute",
                producer_id="human-absolute",
                action=action,
            ),
            _draft(
                role="human_residual",
                producer_id="human-residual",
                action=action,
            ),
        )
    )

    legacy_hash = "a3ebe9a68f3f143406552a43367989d3d92ea30ade67ae4a3be1839a2a9134ec"
    assert batch.semantic_content_hash == legacy_hash
    assert batch.batch_id == f"composition-input-batch:{legacy_hash}"


def test_batch_identity_includes_each_policy_output_id() -> None:
    _clock, _coordinator, _store, action = _fixture()
    first = replace(
        _draft(
            role="human_absolute",
            producer_id="human-absolute",
            action=action,
        ),
        policy_output_id="policy-output:7",
    )
    second = replace(
        _draft(
            role="human_residual",
            producer_id="human-residual",
            action=action,
        ),
        policy_output_id="policy-output:7",
    )
    original = CompositionInputBatchDraft(inputs=(first, second))
    changed = CompositionInputBatchDraft(
        inputs=(
            first,
            replace(second, policy_output_id="policy-output:8"),
        )
    )

    assert changed.semantic_content_hash != original.semantic_content_hash
    assert changed.batch_id != original.batch_id


def test_batch_rejects_mixed_policy_output_lineage() -> None:
    _clock, coordinator, store, action = _fixture()
    first = replace(
        _draft(
            role="human_absolute",
            producer_id="human-absolute",
            action=action,
        ),
        policy_output_id="policy-output:7",
    )
    other_output = replace(
        _draft(
            role="human_residual",
            producer_id="human-residual",
            action=action,
        ),
        policy_output_id="policy-output:8",
    )

    receipt = store.admit_batch(
        CompositionInputBatchDraft(inputs=(first, other_output))
    )

    assert receipt.accepted is False
    assert receipt.rejection_code == "batch_causal_mismatch"
    assert len(store) == 0
    assert coordinator.committed_high_watermark == 2


def test_input_rejects_incomplete_semantic_binding_identity() -> None:
    _clock, coordinator, store, action = _fixture()
    draft = replace(
        _draft(
            role="human_absolute",
            producer_id="human-absolute",
            action=action,
        ),
        semantic_binding_id_by_kind={"residual_anchor": "anchor:0"},
    )

    receipt = store.admit(draft)

    assert receipt.accepted is False
    assert receipt.rejection_code == "invalid_semantic_binding"
    assert len(store) == 0
    assert coordinator.committed_high_watermark == 2


def test_unbound_input_keeps_legacy_semantic_payload_shape() -> None:
    _clock, _coordinator, store, action = _fixture()

    receipt = store.admit(
        _draft(
            role="human_absolute",
            producer_id="human-absolute",
            action=action,
        )
    )

    assert receipt.accepted
    record = receipt.composition_input
    assert record is not None
    payload = composition_input_semantic_payload(record)
    assert "semantic_binding_id_by_kind" not in payload
    assert "semantic_binding_hash_by_kind" not in payload


def test_generic_builder_binds_fixed_roles_and_exact_source_lineage() -> None:
    _clock, _coordinator, _store, action = _fixture()
    activation = build_composition_activation_scope(
        operation_spec_id="test-allocation-profile",
        operation_spec_hash="test-allocation-profile-hash",
        session_epoch=1,
        activated_at=9.9,
        clock_domain_id="control.monotonic",
        input_barrier_at=9.9,
        required_roles=("policy", "human_residual", "human_absolute"),
    )

    batch = build_same_source_composition_input_batch(
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

    assert tuple(item.input_role for item in batch.inputs) == (
        "human_absolute",
        "human_residual",
    )
    assert {
        item.composition_activation_id for item in batch.inputs
    } == {activation.activation_id}
    assert {
        item.source_fact_ids for item in batch.inputs
    } == {("human-sample:7",)}
    assert all(
        item.audit_metadata["same_source_derived_batch"]
        for item in batch.inputs
    )
