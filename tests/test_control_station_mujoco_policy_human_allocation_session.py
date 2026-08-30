from __future__ import annotations

from types import SimpleNamespace

import pytest

from control_composition.lifecycle import CompositionSessionOperationSpec
from control_station import (
    CompositionSessionOperationEvidence,
    StationOperationResultDraft,
    StationPublicationBarrier,
    admit_station_operation_result,
)
from control_station.composition import MujocoPolicyHumanAllocationSession
from control_station.composition.mujoco_policy_human_allocation_lifecycle import (
    MUJOCO_ALLOCATION_SESSION_REARM_STATUS,
    MUJOCO_ALLOCATION_SESSION_START_STATUS,
)
from control_station.composition.mujoco_sim_hil import (
    MUJOCO_SIM_HIL_READINESS_FIELDS_V1,
    MujocoSimHILReadinessEvidence,
)
from robot_motion.action import ActionComponentCatalog, ActionComponentSpec
from robot_motion.runtime import LifecycleStartupError
from tests.test_control_station_mujoco_policy_human_allocation_input import _bundle


pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _VirtualAllocationRuntime:
    def __init__(self, bundle) -> None:
        self.now = 10.0
        self.events: list[str] = []
        self.started = False
        self.gate_status = "ready"
        self.config = SimpleNamespace(
            runtime_session_id="session-allocation",
            control_domain_id="domain-arms",
            clock_domain_id="control.monotonic",
            enable_composer_authority=True,
            enable_primary_policy_authority=False,
            enable_replay_authority=False,
            producer_id="virtual-policy",
            producer_epoch=1,
            composer_producer_id=bundle.authority_profile.producer_id,
        )
        self.pose_execution_enabled = True
        self.explicit_shared_safety = True
        self.composition_strategy = bundle.composition_contracts.strategy
        self.catalog = ActionComponentCatalog(
            catalog_id="virtual-allocation-catalog",
            action_space="component-keyed",
            components=tuple(
                ActionComponentSpec(
                    component_id=component_id,
                    selection_group_id=group_id,
                    effector_name=f"{component_id}-effector",
                )
                for component_id, group_id in zip(
                    bundle.authority_profile.component_coverage,
                    bundle.authority_profile.selection_groups,
                    strict=True,
                )
            ),
        )

    def clock(self) -> float:
        return self.now

    def start(self) -> None:
        self.started = True
        self.events.append("runtime.start")

    def close(self) -> None:
        self.started = False
        self.events.append("runtime.close")

    def activate_policy_node_producer(
        self,
        producer_id: str,
        *,
        producer_epoch: int,
        reason: str,
    ):
        self.events.append(f"producer:{producer_id}:{producer_epoch}")
        return SimpleNamespace(
            activation_id=f"producer-activation:{producer_id}:{producer_epoch}",
            semantic_content_hash=f"producer-hash:{producer_id}:{producer_epoch}",
            available_at=self.now,
            available_clock_domain_id=self.config.clock_domain_id,
        )

    def capture_execution_gate_snapshot(self, *, now: float):
        assert now == pytest.approx(self.now)
        return SimpleNamespace(
            status=self.gate_status,
            cutoff_at=self.now,
            semantic_content_hash=(f"gate-snapshot:{self.gate_status}:{self.now}"),
        )

    def transition_execution_gate(
        self,
        status: str,
        *,
        reason: str,
        source_event_ids=(),
        force_cancellation: bool = False,
    ):
        self.gate_status = status
        self.events.append(
            f"gate:{status}:{reason}:{force_cancellation}:"
            + ",".join(source_event_ids)
        )
        snapshot = SimpleNamespace(
            status=status,
            cutoff_at=self.now,
            semantic_content_hash=f"gate-snapshot:{status}:{self.now}",
        )
        return SimpleNamespace(
            lower_event_ids=(f"gate-event:{status}:{len(self.events)}",),
            event=SimpleNamespace(
                available_at=self.now,
                available_clock_domain_id=self.config.clock_domain_id,
            ),
            snapshot=snapshot,
        )


def _session_fixture(
    *,
    omit_start_evidence: bool = False,
    omit_rearm_evidence: bool = False,
    predate_start_evidence: bool = False,
):
    bundle = _bundle(enabled=True)
    runtime = _VirtualAllocationRuntime(bundle)
    barrier = StationPublicationBarrier(source_slot_id="allocation-composer")
    binding_factory = SimpleNamespace(
        runtime=runtime,
        bundle=bundle,
        publication_barrier=barrier,
        policy_input_adapter=SimpleNamespace(
            binding=SimpleNamespace(
                source_slot_id="policy-source",
                runtime_session_id=runtime.config.runtime_session_id,
                control_domain_id=runtime.config.control_domain_id,
                producer_id=runtime.config.producer_id,
                producer_epoch=runtime.config.producer_epoch,
            )
        ),
        human_source=SimpleNamespace(
            binding_receipt=SimpleNamespace(rig_absent=True, authoritative=False)
        ),
        required_producer_epochs=lambda scope: {
            "human-absolute": scope.session_epoch,
            "human-residual": scope.session_epoch,
            "human-anchor": scope.session_epoch,
            "allocation-context": scope.session_epoch,
        },
    )
    spec = CompositionSessionOperationSpec(
        session_key="virtual-allocation-session",
        composer_source_slot_id=barrier.source_slot_id,
        contributor_source_slot_ids=("policy-source", "human-source"),
        composition_spec_hash=(
            bundle.composition_contracts.composition_spec.semantic_content_hash
        ),
        authority_gate_spec_hash="virtual-allocation-authority-gate",
        execution_binding_hash="virtual-mujoco-execution-binding",
        capture_boundary_id="virtual-episode-boundary",
    )
    readiness = SimpleNamespace(
        snapshot=lambda *, now: MujocoSimHILReadinessEvidence.all_ready()
    )
    evidence_seen = []
    observed_activations = []
    result_sequence = 0

    def result_provider(evidence):
        nonlocal result_sequence
        evidence_seen.append(evidence)
        refs = evidence.required_evidence_refs
        if omit_start_evidence and evidence.operation_kind == "start_composition_session":
            refs = refs[:-1]
        if omit_rearm_evidence and evidence.operation_kind == "rearm_composition_session":
            refs = refs[:-1]
        status = (
            MUJOCO_ALLOCATION_SESSION_START_STATUS
            if evidence.operation_kind == "start_composition_session"
            else MUJOCO_ALLOCATION_SESSION_REARM_STATUS
        )
        result_at = (
            runtime.now - 0.1
            if predate_start_evidence
            and evidence.operation_kind == "start_composition_session"
            else runtime.now
        )
        result = admit_station_operation_result(
            StationOperationResultDraft(
                source_intent_id=(
                    f"station-intent:{evidence.operation_kind}:"
                    f"{evidence.activation_scope.session_epoch}"
                ),
                operation_kind=evidence.operation_kind,
                subject_id=barrier.source_slot_id,
                succeeded=True,
                completed_at=result_at,
                completed_clock_domain_id=runtime.config.clock_domain_id,
                status=status,
                evidence_refs=tuple(refs),
            ),
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            available_at=result_at,
            available_clock_domain_id=runtime.config.clock_domain_id,
            admission_sequence=result_sequence,
        )
        result_sequence += 1
        return result

    session = MujocoPolicyHumanAllocationSession(
        runtime=runtime,
        bundle=bundle,
        binding_factory=binding_factory,
        operation_spec=spec,
        readiness_provider=readiness,
        activation_observer=observed_activations.append,
        start_result_provider=result_provider,
        rearm_result_provider=result_provider,
    )
    return session, runtime, barrier, evidence_seen, observed_activations


def test_allocation_session_start_hold_rearm_uses_exact_typed_barriers() -> None:
    session, runtime, barrier, evidence_seen, observed_activations = (
        _session_fixture()
    )

    report = session.start()

    assert report.passed
    assert session.activation_scope.session_epoch == 1
    assert barrier.snapshot().transition_reason == (
        "typed_composition_session_completion"
    )
    assert len(evidence_seen) == 1
    assert isinstance(evidence_seen[0], CompositionSessionOperationEvidence)
    assert set(evidence_seen[0].required_evidence_refs).issubset(
        set(barrier.snapshot().authority_evidence_refs)
    )
    assert observed_activations[-1] is session.activation_scope

    session.request_hold(reason="operator_pause")
    assert barrier.snapshot().enabled is False
    assert runtime.gate_status == "hold"

    runtime.now = 10.1
    rearm = session.rearm(now=runtime.now)

    assert rearm.passed
    assert session.activation_scope.session_epoch == 2
    assert evidence_seen[-1].prior_activation_scope.session_epoch == 1
    assert tuple(scope.session_epoch for scope in observed_activations) == (1, 2)
    assert barrier.snapshot().transition_reason == (
        "typed_composition_session_rearm_completion"
    )
    assert runtime.gate_status == "ready"
    assert "producer:human-residual:1" in runtime.events
    assert "producer:human-residual:2" in runtime.events

    session.close()
    assert barrier.snapshot().enabled is False
    assert runtime.started is False


def test_allocation_session_missing_evidence_never_opens_publication() -> None:
    session, runtime, barrier, _evidence_seen, _observed = _session_fixture(
        omit_start_evidence=True
    )

    with pytest.raises(ValueError, match="misses activation evidence"):
        session.start()

    assert barrier.snapshot().enabled is False
    assert runtime.started is False
    assert runtime.events[-1] == "runtime.close"


def test_allocation_session_rejects_completion_before_receiver_readiness() -> None:
    session, runtime, barrier, _seen, _observed = _session_fixture(
        predate_start_evidence=True
    )

    with pytest.raises(ValueError, match="predates activation readiness"):
        session.start()

    assert barrier.snapshot().enabled is False
    assert runtime.started is False


def test_allocation_session_start_requires_lower_ready_gate() -> None:
    session, runtime, barrier, seen, _observed = _session_fixture()
    runtime.gate_status = "hold"

    with pytest.raises(RuntimeError, match="requires lower execution READY"):
        session.start()

    assert seen == []
    assert barrier.snapshot().enabled is False
    assert runtime.started is False


def test_allocation_session_failed_rearm_returns_lower_gate_to_hold() -> None:
    session, runtime, barrier, _seen, _observed = _session_fixture(
        omit_rearm_evidence=True
    )
    session.start()
    session.request_hold(reason="operator_pause")
    runtime.now = 10.1

    with pytest.raises(ValueError, match="misses activation evidence"):
        session.rearm(now=runtime.now)

    assert barrier.snapshot().enabled is False
    assert runtime.gate_status == "hold"
    with pytest.raises(RuntimeError, match="requires application rebuild"):
        session.rearm(now=runtime.now)
    session.close()


def test_allocation_session_requires_every_readiness_field() -> None:
    session, _runtime, _barrier, _evidence_seen, _observed = _session_fixture()
    session.readiness_provider = SimpleNamespace(
        snapshot=lambda *, now: MujocoSimHILReadinessEvidence(
            policy_source_ready=True,
            human_device_ready=True,
            calibration_ready=True,
            home_ready=True,
            rebase_ready=False,
            deadman_ready=True,
            policy_chunk_ttl_ready=True,
            measured_state_ready=True,
        )
    )

    report = session.preflight(now=10.0)

    assert not report.passed
    assert session.required_readiness_checks() == (
        MUJOCO_SIM_HIL_READINESS_FIELDS_V1
    )
    assert session.readiness_failures(report.readiness) == ("rebase_ready",)


def test_allocation_session_close_preserves_runtime_ownership_for_retry() -> None:
    session, runtime, _barrier, _evidence_seen, _observed = _session_fixture()
    session.start()
    activation = session.activation_scope
    close_attempts = 0
    original_close = runtime.close

    def close_runtime() -> None:
        nonlocal close_attempts
        close_attempts += 1
        if close_attempts == 1:
            raise RuntimeError("synthetic runtime close failure")
        original_close()

    runtime.close = close_runtime

    with pytest.raises(RuntimeError, match="synthetic runtime close failure"):
        session.close()

    assert session.started is True
    assert session.activation_scope is activation

    session.close()

    assert close_attempts == 2
    assert session.started is False
    assert session.activation_scope is None
    assert runtime.started is False


def test_allocation_start_rollback_failure_remains_closeable() -> None:
    session, runtime, barrier, _evidence_seen, _observed = _session_fixture(
        omit_start_evidence=True
    )
    close_attempts = 0
    original_close = runtime.close

    def close_runtime() -> None:
        nonlocal close_attempts
        close_attempts += 1
        if close_attempts == 1:
            raise RuntimeError("synthetic rollback close failure")
        original_close()

    runtime.close = close_runtime

    with pytest.raises(LifecycleStartupError) as caught:
        session.start()

    assert isinstance(caught.value.start_error, ValueError)
    assert barrier.snapshot().enabled is False
    assert session.started is False

    session.close()

    assert close_attempts == 2
    assert runtime.started is False
