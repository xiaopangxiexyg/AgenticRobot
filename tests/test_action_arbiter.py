from __future__ import annotations

import inspect
import numpy as np

from robot_motion.action import BIMANUAL_EE_POSE_V1, RobotAction
from robot_motion.action.arbiter import ActionArbiter
from robot_motion.action.arbitration_input import capture_arbitration_input
from robot_motion.action.arbitration_runtime import DecisionScheduleSpec, DecisionScheduler
from robot_motion.action.authority import (
    AuthorityEventDraft,
    AuthorityPolicySpec,
    AuthorityStore,
)
from robot_motion.action.components import ActionComponentCatalog, ActionComponentSpec
from robot_motion.action.proposal import ActionProposalDraft, ProposalBuffer
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.execution_gate import (
    ExecutionGateEventDraft,
    ExecutionGateStore,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


class Clock:
    value = 5.0

    def __call__(self):
        return float(self.value)


def setup_runtime(*, coupled: bool = False):
    clock = Clock()
    coordinator = ControlAdmissionCoordinator(
        "robot-0",
        clock_domain=ClockDomainDescriptor(
            "control.monotonic", "session-0", "boot-0", 0
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(runtime_session_id="session-0", coordinator=coordinator)
    for producer in ("policy-main", "human-main"):
        epochs.activate(
            producer_id=producer,
            producer_epoch=1,
            produced_at=1.0,
            produced_clock_domain_id=f"{producer}.clock",
            reason="startup",
        )
    components = [ActionComponentSpec("left_arm", "arms", "left_arm", coupled=coupled)]
    if coupled:
        components.append(ActionComponentSpec("right_arm", "arms", "right_arm", coupled=True))
    catalog = ActionComponentCatalog("arms-v1", BIMANUAL_EE_POSE_V1, tuple(components))
    proposals = ProposalBuffer(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=32,
    )
    policy = AuthorityPolicySpec(
        policy_id="human-over-policy-v1",
        priority_by_group={"arms": {"kind:policy": 10, "kind:human": 20}},
        fallback_by_group={"arms": "no_action"},
        takeover_allowed_producers=("human-main", "policy-main"),
    )
    authority = AuthorityStore(
        runtime_session_id="session-0", coordinator=coordinator, policy=policy
    )
    gate = ExecutionGateStore(runtime_session_id="session-0", coordinator=coordinator)
    return clock, coordinator, epochs, catalog, proposals, authority, gate


def proposal(catalog, *, producer: str, kind: str, sequence: int, coverage=("left_arm",)):
    return ActionProposalDraft(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        producer_kind=kind,
        producer_id=producer,
        producer_epoch=1,
        sequence=sequence,
        action=RobotAction.bimanual_ee_pose(
            left_arm=np.asarray([0.3, 0.0, 0.4, 0, 0, 0, 1], dtype=np.float32),
            right_arm=(
                np.asarray([0.3, 0.0, 0.4, 0, 0, 0, 1], dtype=np.float32)
                if "right_arm" in coverage
                else None
            ),
            source=kind,
        ),
        produced_at=4.0,
        produced_clock_domain_id=f"{producer}.clock",
        effective_from=10.0,
        effective_until=11.0,
        target_effective_at=10.5,
        effective_clock_domain_id="control.monotonic",
        component_schema_hash=catalog.semantic_content_hash,
        component_coverage=coverage,
        selection_groups=("arms",),
    )


def capture(runtime):
    clock, coordinator, epochs, catalog, proposals, authority, gate = runtime
    clock.value = 10.5
    return capture_arbitration_input(
        coordinator=coordinator,
        proposal_buffer=proposals,
        producer_epochs=epochs,
        authority_store=authority,
        execution_gate_store=gate,
        scheduler=DecisionScheduler(
            DecisionScheduleSpec(spec_id="30hz", decision_hz=30), clock=clock
        ),
        embodiment_instance_id="rm65-pair-0",
        component_catalog=catalog,
        active_selection_groups=("arms",),
    )


def test_same_complete_snapshot_produces_identical_result_hash() -> None:
    runtime = setup_runtime()
    runtime[4].admit(proposal(runtime[3], producer="policy-main", kind="policy", sequence=0))
    snapshot = capture(runtime)
    assert snapshot is not None

    first = ActionArbiter().arbitrate(snapshot)
    second = ActionArbiter().arbitrate(snapshot)

    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.output_status == "selected"
    assert first.selected_by_group["arms"].endswith("policy-main:1:0")
    assert first.composed_action is not None


def test_pure_arbiter_accepts_only_complete_snapshot_and_reads_no_store_or_clock() -> None:
    parameters = tuple(inspect.signature(ActionArbiter.arbitrate).parameters)
    source = inspect.getsource(ActionArbiter.arbitrate)

    assert parameters == ("self", "snapshot")
    assert "time.monotonic" not in source
    assert "os.environ" not in source
    assert "ProposalBuffer" not in source
    assert "AuthorityStore" not in source


def test_authority_priority_and_snapshot_isolation_are_explicit() -> None:
    runtime = setup_runtime()
    runtime[4].admit(proposal(runtime[3], producer="policy-main", kind="policy", sequence=0))
    runtime[4].admit(proposal(runtime[3], producer="human-main", kind="human", sequence=0))
    snapshot = capture(runtime)
    assert snapshot is not None
    before = ActionArbiter().arbitrate(snapshot)

    runtime[5].admit(
        AuthorityEventDraft(
            "session-0",
            "robot-0",
            0,
            "takeover",
            10.6,
            "control.monotonic",
            selection_group_id="arms",
            owner_producer_id="human-main",
            reason="late event",
        )
    )
    runtime[6].admit(
        ExecutionGateEventDraft(
            "session-0",
            "robot-0",
            0,
            "hold",
            10.6,
            "control.monotonic",
            "late hold",
        )
    )
    runtime[2].activate(
        producer_id="policy-main",
        producer_epoch=2,
        produced_at=10.6,
        produced_clock_domain_id="policy-main.clock",
        reason="late restart",
    )
    after = ActionArbiter().arbitrate(snapshot)

    assert before.semantic_content_hash == after.semantic_content_hash
    assert before.selected_by_group["arms"].endswith("human-main:1:0")


def test_coupled_group_rejects_incomplete_proposal() -> None:
    runtime = setup_runtime(coupled=True)
    runtime[4].admit(
        proposal(
            runtime[3],
            producer="policy-main",
            kind="policy",
            sequence=0,
            coverage=("left_arm",),
        )
    )
    snapshot = capture(runtime)
    assert snapshot is not None

    result = ActionArbiter().arbitrate(snapshot)

    assert result.output_status == "no_action"
    assert any(item.reason == "coupled_group_incomplete" for item in result.rejections)


def test_gate_hold_is_not_an_authoritative_selected_action() -> None:
    runtime = setup_runtime()
    runtime[4].admit(proposal(runtime[3], producer="policy-main", kind="policy", sequence=0))
    runtime[6].admit(
        ExecutionGateEventDraft(
            "session-0", "robot-0", 0, "hold", 5.0, "control.monotonic", "operator"
        )
    )
    snapshot = capture(runtime)
    assert snapshot is not None

    result = ActionArbiter().arbitrate(snapshot)

    assert result.output_status == "hold"
    assert result.composed_action is None
    assert not result.selected_by_group


def test_authority_lease_overrides_priority_and_expiry_restores_policy_rules() -> None:
    active_runtime = setup_runtime()
    active_runtime[4].admit(
        proposal(active_runtime[3], producer="policy-main", kind="policy", sequence=0)
    )
    active_runtime[4].admit(
        proposal(active_runtime[3], producer="human-main", kind="human", sequence=0)
    )
    active_runtime[5].admit(
        AuthorityEventDraft(
            "session-0",
            "robot-0",
            0,
            "takeover",
            5.0,
            "control.monotonic",
            selection_group_id="arms",
            owner_producer_id="policy-main",
            expires_at=11.0,
            expires_clock_domain_id="control.monotonic",
            reason="active policy lease",
        )
    )
    active_snapshot = capture(active_runtime)
    assert active_snapshot is not None
    active_result = ActionArbiter().arbitrate(active_snapshot)
    assert active_result.selected_by_group["arms"].endswith("policy-main:1:0")

    runtime = setup_runtime()
    runtime[4].admit(proposal(runtime[3], producer="policy-main", kind="policy", sequence=0))
    runtime[4].admit(proposal(runtime[3], producer="human-main", kind="human", sequence=0))
    runtime[5].admit(
        AuthorityEventDraft(
            "session-0",
            "robot-0",
            0,
            "takeover",
            5.0,
            "control.monotonic",
            selection_group_id="arms",
            owner_producer_id="policy-main",
            expires_at=10.4,
            expires_clock_domain_id="control.monotonic",
            reason="temporary policy lease",
        )
    )
    snapshot = capture(runtime)
    assert snapshot is not None

    result = ActionArbiter().arbitrate(snapshot)

    assert "arms" not in snapshot.authority_state_snapshot.active_leases
    assert result.selected_by_group["arms"].endswith("human-main:1:0")
