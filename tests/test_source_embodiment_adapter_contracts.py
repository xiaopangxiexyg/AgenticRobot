from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    BIMANUAL_QPOS_V1,
    ActionComponentCatalog,
    ActionComponentSpec,
    ActionProposalDraft,
    ProposalBuffer,
    RobotAction,
    build_robot_action_v1_runtime_schema,
    materialize_robot_action_v1,
    project_robot_action_v1_to_component_action,
)
from robot_motion.action.proposal import robot_action_semantic_payload
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


@dataclass(frozen=True)
class FakeSourceReceipt:
    adapter_id: str
    raw_schema_id: str
    canonical_action_hash: str


class VectorSourceAdapter:
    adapter_id = "fake-vector-controller-v1"
    raw_schema_id = "xyz-controller-vector-v1"

    def adapt(self, payload) -> tuple[RobotAction, FakeSourceReceipt]:
        values = np.asarray(payload, dtype=np.float32).reshape(6)
        action = RobotAction(
            action_space=BIMANUAL_QPOS_V1,
            arm_qpos_by_side={"left": values},
            source="human",
        )
        return action, FakeSourceReceipt(
            adapter_id=self.adapter_id,
            raw_schema_id=self.raw_schema_id,
            canonical_action_hash=canonical_content_hash(
                robot_action_semantic_payload(action),
                domain="dexteleop.fake_source_canonical_action.v1",
            ),
        )


class NamedSourceAdapter:
    adapter_id = "fake-named-controller-v1"
    raw_schema_id = "named-joint-controller-v1"

    def adapt(self, payload) -> tuple[RobotAction, FakeSourceReceipt]:
        names = tuple(payload["names"])
        if names != tuple(f"joint_{index}" for index in range(6)):
            raise ValueError("named source joint contract mismatch")
        values = np.asarray(payload["positions"], dtype=np.float32).reshape(6)
        action = RobotAction(
            action_space=BIMANUAL_QPOS_V1,
            arm_qpos_by_side={"left": values},
            source="human",
        )
        return action, FakeSourceReceipt(
            adapter_id=self.adapter_id,
            raw_schema_id=self.raw_schema_id,
            canonical_action_hash=canonical_content_hash(
                robot_action_semantic_payload(action),
                domain="dexteleop.fake_source_canonical_action.v1",
            ),
        )


@dataclass(frozen=True)
class FakeEmbodimentReceipt:
    sink_id: str
    command_schema_id: str
    command_payload: tuple[float, ...]


class FakeEmbodimentSink:
    def __init__(
        self,
        *,
        sink_id: str,
        command_schema_id: str,
        action_space: str,
        action_schema_hash: str,
        arm_dof: int,
    ) -> None:
        self.sink_id = sink_id
        self.command_schema_id = command_schema_id
        self.action_space = action_space
        self.action_schema_hash = action_schema_hash
        self.arm_dof = arm_dof
        self.dispatch_count = 0

    def preflight(self, proposal, *, action_schema_hash: str) -> None:
        action = materialize_robot_action_v1(proposal.action)
        if action.action_space != self.action_space:
            raise ValueError("embodiment action-space mismatch")
        if action_schema_hash != self.action_schema_hash:
            raise ValueError("embodiment action-schema mismatch")
        qpos = action.arm_qpos_by_side.get("left")
        if qpos is None or qpos.shape != (self.arm_dof,):
            raise ValueError("embodiment DOF mismatch")

    def dispatch(self, proposal, *, action_schema_hash: str) -> FakeEmbodimentReceipt:
        self.preflight(proposal, action_schema_hash=action_schema_hash)
        self.dispatch_count += 1
        qpos = materialize_robot_action_v1(
            proposal.action
        ).arm_qpos_by_side["left"]
        if self.command_schema_id == "real-servo-deg-v1":
            payload = tuple(float(value) for value in np.rad2deg(qpos))
        else:
            payload = tuple(float(value) for value in qpos)
        return FakeEmbodimentReceipt(
            sink_id=self.sink_id,
            command_schema_id=self.command_schema_id,
            command_payload=payload,
        )


def proposal_runtime(catalog: ActionComponentCatalog):
    coordinator = ControlAdmissionCoordinator(
        "robot-0",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="control.monotonic",
            runtime_session_id="session-0",
            boot_id="boot-0",
            reset_epoch=0,
        ),
        clock=lambda: 10.0,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="session-0", coordinator=coordinator
    )
    epochs.activate(
        producer_id="human-canonical",
        producer_epoch=1,
        produced_at=9.0,
        produced_clock_domain_id="control.monotonic",
        reason="startup",
    )
    return ProposalBuffer(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=4,
    )


def test_two_raw_source_schemas_produce_same_canonical_action_semantics() -> None:
    values = np.linspace(0.0, 0.5, 6, dtype=np.float32)
    vector_action, vector_receipt = VectorSourceAdapter().adapt(values)
    named_action, named_receipt = NamedSourceAdapter().adapt(
        {
            "names": tuple(f"joint_{index}" for index in range(6)),
            "positions": values.tolist(),
        }
    )
    assert vector_receipt.adapter_id != named_receipt.adapter_id
    assert vector_receipt.raw_schema_id != named_receipt.raw_schema_id
    assert vector_receipt.canonical_action_hash == named_receipt.canonical_action_hash
    np.testing.assert_array_equal(
        vector_action.arm_qpos_by_side["left"], named_action.arm_qpos_by_side["left"]
    )


def test_compatible_embodiment_sinks_keep_proposal_fact_but_change_command_schema() -> None:
    catalog = ActionComponentCatalog(
        catalog_id="left-arm-qpos-v1",
        action_space=BIMANUAL_QPOS_V1,
        components=(ActionComponentSpec("left_arm", "left_arm", "left_arm"),),
    )
    action, _ = VectorSourceAdapter().adapt(np.linspace(0.0, 0.5, 6))
    action_schema = build_robot_action_v1_runtime_schema(
        component_catalog=catalog,
    )
    proposal = proposal_runtime(catalog).admit(
        ActionProposalDraft(
            runtime_session_id="session-0",
            control_domain_id="robot-0",
            embodiment_instance_id="compatible-left-arm",
            producer_kind="human",
            producer_id="human-canonical",
            producer_epoch=1,
            sequence=0,
            action=project_robot_action_v1_to_component_action(
                action,
                action_schema=action_schema,
            ),
            produced_at=9.9,
            produced_clock_domain_id="control.monotonic",
            effective_from=10.0,
            effective_until=11.0,
            target_effective_at=10.5,
            effective_clock_domain_id="control.monotonic",
        )
    ).proposal
    assert proposal is not None
    real = FakeEmbodimentSink(
        sink_id="fake-real-rm65",
        command_schema_id="real-servo-deg-v1",
        action_space=BIMANUAL_QPOS_V1,
        action_schema_hash=action_schema.semantic_content_hash,
        arm_dof=6,
    )
    sim = FakeEmbodimentSink(
        sink_id="fake-sim-rm65",
        command_schema_id="sim-qpos-rad-v1",
        action_space=BIMANUAL_QPOS_V1,
        action_schema_hash=action_schema.semantic_content_hash,
        arm_dof=6,
    )
    real_receipt = real.dispatch(
        proposal, action_schema_hash=action_schema.semantic_content_hash
    )
    sim_receipt = sim.dispatch(
        proposal, action_schema_hash=action_schema.semantic_content_hash
    )
    assert real_receipt.command_schema_id != sim_receipt.command_schema_id
    assert real_receipt.command_payload != sim_receipt.command_payload
    assert proposal.semantic_content_hash


def test_incompatible_embodiment_fails_before_dispatch() -> None:
    catalog = ActionComponentCatalog(
        catalog_id="left-arm-qpos-v1",
        action_space=BIMANUAL_QPOS_V1,
        components=(ActionComponentSpec("left_arm", "left_arm", "left_arm"),),
    )
    action, _ = VectorSourceAdapter().adapt(np.zeros(6))
    action_schema = build_robot_action_v1_runtime_schema(
        component_catalog=catalog,
    )
    proposal = proposal_runtime(catalog).admit(
        ActionProposalDraft(
            runtime_session_id="session-0",
            control_domain_id="robot-0",
            embodiment_instance_id="incompatible-arm",
            producer_kind="human",
            producer_id="human-canonical",
            producer_epoch=1,
            sequence=0,
            action=project_robot_action_v1_to_component_action(
                action,
                action_schema=action_schema,
            ),
            produced_at=9.9,
            produced_clock_domain_id="control.monotonic",
            effective_from=10.0,
            effective_until=11.0,
            target_effective_at=10.5,
            effective_clock_domain_id="control.monotonic",
        )
    ).proposal
    assert proposal is not None
    incompatible = FakeEmbodimentSink(
        sink_id="fake-7dof",
        command_schema_id="sim-qpos-rad-v1",
        action_space=BIMANUAL_QPOS_V1,
        action_schema_hash=action_schema.semantic_content_hash,
        arm_dof=7,
    )
    with pytest.raises(ValueError, match="DOF mismatch"):
        incompatible.dispatch(
            proposal, action_schema_hash=action_schema.semantic_content_hash
        )
    assert incompatible.dispatch_count == 0
