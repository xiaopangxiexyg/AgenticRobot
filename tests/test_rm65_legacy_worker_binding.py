from __future__ import annotations

import argparse
import threading

import numpy as np
import pytest

from robot_motion.runtime import ExecutionCancellationRequest, WorkerEpochDrainReceipt
from teleop_runtime.action import LegacyControlOwnerStateTransitionRequest
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.backends.arms.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
)
from teleop_runtime.backends.arms.rm65_execution_evidence import (
    build_rm65_execution_evidence_bundle,
)
from teleop_runtime.backends.arms.rm65_worker_drain import (
    drain_rm65_arms_only_workers,
)
from teleop_runtime.deploy import run_policy
from teleop_runtime.deploy.rm65_legacy_hold_convergence import (
    RM65LegacyBackendHoldReceipt,
    converge_rm65_legacy_control_stop,
)
from teleop_runtime.deploy.rm65_legacy_worker_binding import (
    build_fake_only_rm65_legacy_worker_binding,
)
from teleop_runtime.position_control.deploy_executor import (
    PositionDeployExecutor,
    PositionDeployExecutorConfig,
)
from tests.test_legacy_execution_context import ManualClock, _capability
from tests.test_position_deploy_executor import (
    _filtered_qpos,
    _observation,
    _pipeline,
    _wait_until,
)


def _binding():
    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    evidence = build_rm65_execution_evidence_bundle(manifest)
    return build_fake_only_rm65_legacy_worker_binding(
        evidence_bundle=evidence,
        runtime_session_id="rm65-binding-fixture",
    )


def test_one_context_injects_same_guard_into_all_three_required_workers() -> None:
    binding = _binding()
    position_kwargs = binding.position_executor_kwargs()
    left_kwargs = binding.arm_backend_kwargs("left", connect_hardware=False)
    right_kwargs = binding.arm_backend_kwargs("right", connect_hardware=False)

    assert position_kwargs["execution_epoch_guard"] is binding.context.execution_guard
    assert left_kwargs["execution_epoch_guard"] is binding.context.execution_guard
    assert right_kwargs["execution_epoch_guard"] is binding.context.execution_guard
    participant_ids = {
        position_kwargs["worker_drain_participant"].participant_id,
        left_kwargs["worker_drain_participant"].participant_id,
        right_kwargs["worker_drain_participant"].participant_id,
    }
    assert participant_ids == {
        "position-deploy-reference-worker",
        "rm65-left-background-stream-worker",
        "rm65-right-background-stream-worker",
    }
    assert not binding.production_binding_enabled
    assert not binding.confers_execution_authorization


def test_fake_binding_constructs_real_worker_classes_without_hardware() -> None:
    binding = _binding()
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
        ),
        canonical_pipeline=_pipeline(("left", "right")),
        **binding.position_executor_kwargs(),
    )
    left = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
        ),
        **binding.arm_backend_kwargs("left", connect_hardware=False),
    )
    right = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
        ),
        **binding.arm_backend_kwargs("right", connect_hardware=False),
    )

    assert executor.snapshot()["execution_epoch_binding_enabled"] is True
    assert left.get_state()["background_streaming"]["execution_epoch_binding_enabled"]
    assert right.get_state()["background_streaming"]["execution_epoch_binding_enabled"]


def test_reference_work_derives_through_position_and_backend_workers() -> None:
    binding = _binding()
    binding.context.transition_owner_state(
        LegacyControlOwnerStateTransitionRequest(
            target_lifecycle_state="shadow_ready",
            target_execution_gate="hold",
            reason="fixture_shadow_ready",
        )
    )
    binding.context.transition_owner_state(
        LegacyControlOwnerStateTransitionRequest(
            target_lifecycle_state="running",
            target_execution_gate="ready",
            reason="fixture_running",
        )
    )
    reference_work = binding.context.create_reference_segment_work_item(
        source_decision_id="fixture-policy-decision",
        source_reference_segment_id="fixture-reference-segment",
    )
    emitted_commands = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
            stale_after_sec=0.5,
        ),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=emitted_commands.append,
        **binding.position_executor_kwargs(),
    )
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
        ),
        **binding.arm_backend_kwargs("left", connect_hardware=False),
    )

    executor.start()
    try:
        executor.submit(
            _filtered_qpos("left", np.full(6, 0.01, dtype=float)),
            _observation(sides=("left",)),
            "fixture_policy",
            execution_work_item=reference_work,
        )
        _wait_until(lambda: bool(emitted_commands))
        position_command = emitted_commands[0]
        position_work = position_command.execution_work_item

        assert position_work is not None
        assert position_work.work_kind == "position_command"
        assert position_work.control_domain_id == reference_work.control_domain_id
        assert position_work.authority_epoch == reference_work.authority_epoch
        assert position_work.execution_epoch == reference_work.execution_epoch
        assert position_work.segment_generation_by_group == (
            reference_work.segment_generation_by_group
        )
        assert position_command.commands["left_arm"].execution_work_item == position_work

        class RecordingArm:
            def __init__(self) -> None:
                self.calls = []

            def Movej_CANFD(self, joint_deg, **_kwargs):
                self.calls.append(tuple(float(value) for value in joint_deg))
                return 0

        recording_arm = RecordingArm()
        backend.arm = recording_arm
        backend._stream_stop_event.clear()
        backend_worker = threading.Thread(
            target=backend._background_stream_loop_impl,
            daemon=True,
        )
        backend_worker.start()
        backend.send_command(position_command.commands["left_arm"])
        _wait_until(lambda: bool(recording_arm.calls))
        backend_work = backend._last_completed_execution_work_item
        assert backend_work is not None
        assert backend_work.work_kind == "backend_command"
        assert backend_work.control_domain_id == reference_work.control_domain_id
        assert backend_work.authority_epoch == reference_work.authority_epoch
        assert backend_work.execution_epoch == reference_work.execution_epoch
        assert backend_work.segment_generation_by_group == (
            reference_work.segment_generation_by_group
        )
        assert backend_work.source_decision_id == reference_work.source_decision_id
        assert binding.context.execution_guard.check_at_dequeue(backend_work).accepted
        backend._stream_stop_event.set()
        backend_worker.join(timeout=1.0)
        assert not backend_worker.is_alive()
    finally:
        backend._stream_stop_event.set()
        executor.close()


def test_fake_binding_refuses_hardware_and_unknown_participant_scope() -> None:
    binding = _binding()

    with pytest.raises(RuntimeError, match="cannot be attached to hardware"):
        binding.arm_backend_kwargs("left", connect_hardware=True)
    with pytest.raises(ValueError, match="left or right"):
        binding.arm_backend_kwargs("center", connect_hardware=False)


def test_canonical_policy_backend_builder_has_default_off_fake_binding_seam() -> None:
    binding = _binding()
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    args = parser.parse_args(["--active-sides", "both"])
    active_sides = run_policy.validate_args(args)
    args._rm65_legacy_worker_binding = binding

    backends = run_policy.build_arm_backends(
        args,
        active_sides,
        control_hz=args.arm_control_hz,
    )

    assert tuple(backends) == ("left", "right")
    assert backends["left"]._execution_epoch_guard is binding.context.execution_guard
    assert backends["right"]._execution_epoch_guard is binding.context.execution_guard

    args.left_arm_ip = "192.0.2.1"
    args.right_arm_ip = "192.0.2.2"
    with pytest.raises(RuntimeError, match="cannot be attached to hardware"):
        run_policy.build_arm_backends(
            args,
            active_sides,
            control_hz=args.arm_control_hz,
        )


def test_run_policy_entrypoint_exposes_only_programmatic_default_off_binding() -> None:
    import inspect

    signature = inspect.signature(run_policy.main)
    parameter = signature.parameters["legacy_worker_binding"]

    assert parameter.default is None
    assert "--legacy-worker-binding" not in inspect.getsource(run_policy.add_args)
    source = inspect.getsource(run_policy.main)
    assert "legacy_worker_binding.context.transition_owner_state" in source
    assert "legacy_owner_desired_state" in source


def test_run_policy_default_off_dry_run_uses_and_stops_the_unique_owner() -> None:
    binding = _binding()

    exit_code = run_policy.main(
        ["--max-steps", "1", "--status-sec", "0.01"],
        legacy_worker_binding=binding,
    )
    snapshot = binding.context.snapshot()

    assert exit_code == 0
    assert snapshot.lifecycle_state == "stopped"
    assert snapshot.execution_gate == "hold"
    assert snapshot.owner_state_revision >= 2
    assert snapshot.production_binding_enabled is False


def test_production_hold_cannot_be_claimed_from_untyped_boolean_acknowledgements() -> None:
    binding = _binding()
    cancellation = binding.context.cancel_execution(
        ExecutionCancellationRequest(
            reason="hold",
            authority_epoch=0,
            requested_scope="control_domain",
            source_event_ids=("fixture-untyped-production-hold",),
        ),
        capability=_capability(),
    )

    with pytest.raises(ValueError, match="typed lower stop receipts"):
        RM65LegacyBackendHoldReceipt.evaluate(
            cancellation=cancellation,
            acknowledgement_by_backend_id={"left": True, "right": True},
            acknowledged_at=cancellation.effective_at,
            clock_domain_id=binding.context.clock_domain_id,
            evidence_refs=("untyped-left", "untyped-right"),
            production_binding_enabled=True,
        )


def test_typed_drain_and_backend_hold_converge_the_same_owner_to_hold() -> None:
    clock = ManualClock()
    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    evidence = build_rm65_execution_evidence_bundle(manifest)
    binding = build_fake_only_rm65_legacy_worker_binding(
        evidence_bundle=evidence,
        runtime_session_id="rm65-hold-convergence-fixture",
        clock=clock,
    )
    cancellation = binding.context.cancel_execution(
        ExecutionCancellationRequest(
            reason="hold",
            authority_epoch=0,
            requested_scope="control_domain",
            source_event_ids=("fixture-hold-request",),
        ),
        capability=_capability(),
    )
    clock.value = cancellation.effective_at + 0.01
    participant_by_id = {
        item.participant.participant_id: item.participant
        for item in binding.worker_drain_plan.participant_bindings
    }

    class Worker:
        def __init__(self, participant_id: str) -> None:
            self.participant_id = participant_id

        def drain_execution_cancellation(self, event):
            return WorkerEpochDrainReceipt.evaluate(
                cancellation=event,
                participant=participant_by_id[self.participant_id],
                observed_execution_epoch=event.execution_epoch,
                discarded_work_ids=(),
                last_old_epoch_invocation_started_at=None,
                last_old_epoch_invocation_completed_at=None,
                in_flight_old_epoch_count=0,
                no_future_old_epoch_calls=True,
                acknowledged_at=clock.value,
            )

    worker_drain = drain_rm65_arms_only_workers(
        plan=binding.worker_drain_plan,
        cancellation=cancellation,
        position_reference_worker=Worker("position-deploy-reference-worker"),
        left_background_worker=Worker("rm65-left-background-stream-worker"),
        right_background_worker=Worker("rm65-right-background-stream-worker"),
        clock=clock,
    )
    backend_hold = RM65LegacyBackendHoldReceipt.evaluate(
        cancellation=cancellation,
        acknowledgement_by_backend_id={"left": True, "right": True},
        acknowledged_at=clock.value,
        clock_domain_id=binding.context.clock_domain_id,
        evidence_refs=("fake-left-hold", "fake-right-hold"),
    )

    receipt = converge_rm65_legacy_control_stop(
        binding=binding,
        cancellation=cancellation,
        worker_drain=worker_drain,
        backend_hold=backend_hold,
    )

    assert receipt.complete
    assert receipt.status == "hold_converged"
    assert receipt.lifecycle_state == "hold"
    assert receipt.execution_gate == "hold"
    assert receipt.production_binding_enabled is False
    assert binding.context.snapshot().semantic_content_hash == receipt.owner_state_hash


def test_rejected_backend_hold_faults_instead_of_claiming_hold() -> None:
    clock = ManualClock()
    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    evidence = build_rm65_execution_evidence_bundle(manifest)
    binding = build_fake_only_rm65_legacy_worker_binding(
        evidence_bundle=evidence,
        runtime_session_id="rm65-hold-rejected-fixture",
        clock=clock,
    )
    cancellation = binding.context.cancel_execution(
        ExecutionCancellationRequest(
            reason="hold",
            authority_epoch=0,
            requested_scope="control_domain",
            source_event_ids=("fixture-hold-rejected",),
        ),
        capability=_capability(),
    )
    clock.value = cancellation.effective_at + 0.01
    participant_by_id = {
        item.participant.participant_id: item.participant
        for item in binding.worker_drain_plan.participant_bindings
    }

    class Worker:
        def __init__(self, participant_id: str) -> None:
            self.participant_id = participant_id

        def drain_execution_cancellation(self, event):
            return WorkerEpochDrainReceipt.evaluate(
                cancellation=event,
                participant=participant_by_id[self.participant_id],
                observed_execution_epoch=event.execution_epoch,
                discarded_work_ids=(),
                last_old_epoch_invocation_started_at=None,
                last_old_epoch_invocation_completed_at=None,
                in_flight_old_epoch_count=0,
                no_future_old_epoch_calls=True,
                acknowledged_at=clock.value,
            )

    worker_drain = drain_rm65_arms_only_workers(
        plan=binding.worker_drain_plan,
        cancellation=cancellation,
        position_reference_worker=Worker("position-deploy-reference-worker"),
        left_background_worker=Worker("rm65-left-background-stream-worker"),
        right_background_worker=Worker("rm65-right-background-stream-worker"),
        clock=clock,
    )
    rejected_hold = RM65LegacyBackendHoldReceipt.evaluate(
        cancellation=cancellation,
        acknowledgement_by_backend_id={"left": True, "right": False},
        acknowledged_at=clock.value,
        clock_domain_id=binding.context.clock_domain_id,
        evidence_refs=("fake-left-hold", "fake-right-hold-rejected"),
    )

    receipt = converge_rm65_legacy_control_stop(
        binding=binding,
        cancellation=cancellation,
        worker_drain=worker_drain,
        backend_hold=rejected_hold,
    )

    assert not receipt.complete
    assert receipt.status == "stop_evidence_incomplete_faulted"
    assert receipt.lifecycle_state == "fault"
    assert receipt.execution_gate == "fault"
