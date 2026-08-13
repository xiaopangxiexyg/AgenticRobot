from __future__ import annotations

from dataclasses import replace
import time
from pathlib import Path
import json
import sys

import numpy as np
import pytest

from dataset_contracts.completeness import trace_requirement_profile
from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action.authoritative_trace import (
    AuthoritativeTraceContext,
    project_execution_cancellation,
    project_execution_cancellation_ack,
)
from robot_motion.deploy import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
)
from robot_motion.runtime import (
    CancellationAcknowledgementTracker,
    ExecutionCancellationRequest,
    acknowledge_worker_epoch_drain,
)
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.backends.arms import rm65 as rm65_module
from teleop_runtime.backends.arms.rm65_dispatch_call import (
    RM65JointTargetDispatchContext,
)
from teleop_runtime.backends.arms.rm65_idle_worker_preflight_binding import (
    RM65IdleWorkerDrainHardwareAuthorization,
    RM65IdleWorkerDrainPreflightBinding,
    RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1,
)
from teleop_runtime.backends.arms.rm65_worker_drain import (
    drain_rm65_arms_only_workers,
)
from teleop_runtime.control_trace import (
    ControlTraceRuntime,
    ControlTraceSpool,
    ControlTraceSpoolSink,
    ControlTraceWriterHealthLedger,
    RuntimeTraceSession,
)
from teleop_runtime.control_trace.rm65_worker_drain import (
    admit_rm65_worker_drain_trace,
)
from teleop_runtime.backends.arms.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_capability,
    build_rm65_policy_arm_qpos_candidate_manifest,
)
from teleop_runtime.backends.arms.rm65_execution_evidence import (
    build_rm65_execution_evidence_bundle,
)
from teleop_runtime.deploy.rm65_legacy_hold_convergence import (
    compile_rm65_idle_worker_backend_hold_receipt,
    converge_rm65_legacy_control_stop,
)
from teleop_runtime.position_control.deploy_executor import (
    PositionDeployExecutor,
    PositionDeployExecutorConfig,
)


def _binding():
    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    evidence = build_rm65_execution_evidence_bundle(manifest)
    now = time.monotonic()
    authorization = RM65IdleWorkerDrainHardwareAuthorization.create(
        operator_authorization_id="operator-test-only",
        execution_manifest=manifest,
        dispatch_inventory=evidence.dispatch_inventory,
        worker_drain_plan=evidence.worker_drain_plan,
        control_domain_id="rm65-arms",
        embodiment_instance_id="rm65-bimanual-0",
        allowed_sides=("left", "right"),
        issued_at=now - 1.0,
        expires_at=now + 60.0,
    )
    binding = RM65IdleWorkerDrainPreflightBinding(
        runtime_session_id="rm65-idle-worker-test",
        owner_runtime_id="legacy-idle-worker-test",
        evidence_bundle=evidence,
        authorization=authorization,
    )
    return manifest, evidence, authorization, binding


def _pipeline() -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(active_sides=("left", "right"))
        ),
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=False,
    )


def test_narrow_authorization_does_not_change_candidate_manifest_or_confer_motion() -> None:
    before = build_rm65_policy_arm_qpos_candidate_manifest()
    manifest, evidence, authorization, binding = _binding()
    after = build_rm65_policy_arm_qpos_candidate_manifest()

    assert manifest.semantic_content_hash == before.semantic_content_hash
    assert after.semantic_content_hash == before.semantic_content_hash
    assert authorization.profile_id == RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1
    assert not authorization.motion_authorized
    assert not authorization.lifecycle_authorized
    assert binding.production_binding_enabled
    assert not binding.confers_motion_authorization
    assert binding.context.snapshot().composition_mode == (
        "hardware_idle_drain_stop_only_preflight"
    )
    assert evidence.integration_status == "evidence_only_unbound"
    assert not evidence.production_binding_enabled


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"connect_hardware": False}, "requires real hardware"),
        ({"background_streaming_enabled": False}, "idle background worker"),
        ({"force_position_control": True}, "forbids force-position"),
        ({"configure_singularity_thresholds": True}, "startup SDK configuration"),
        ({"initial_qpos": np.zeros(6)}, "fresh hardware state"),
    ],
)
def test_backend_binding_rejects_any_broader_or_ambiguous_config(
    overrides: dict[str, object],
    message: str,
) -> None:
    _manifest, _evidence, _authorization, binding = _binding()
    backend_binding = binding.backend_binding("left")
    values = {
        "side": "left",
        "connect_hardware": True,
        "background_streaming_enabled": True,
        "force_position_control": False,
        "configure_singularity_thresholds": False,
        "initial_qpos": None,
        "now": time.monotonic(),
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        backend_binding.validate_config(**values)


def test_real_config_can_only_bind_through_the_exact_isolated_object_without_starting_sdk() -> None:
    _manifest, _evidence, _authorization, binding = _binding()
    backend_binding = binding.backend_binding("left")
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.1",
            side="left",
            connect_hardware=True,
            initial_qpos=None,
            configure_singularity_thresholds=False,
            force_position_control=False,
            background_streaming_enabled=True,
        ),
        hardware_idle_worker_preflight_binding=backend_binding,
    )
    try:
        stream = backend.get_state()["background_streaming"]
        assert stream["execution_epoch_binding_enabled"] is True
        assert (
            stream["hardware_idle_worker_preflight_binding_id"]
            == backend_binding.binding_id
        )
        assert backend.arm is None
    finally:
        backend.close()

    with pytest.raises(ValueError, match="explicit duplicates are forbidden"):
        RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="192.0.2.1",
                side="left",
                connect_hardware=True,
                initial_qpos=None,
                configure_singularity_thresholds=False,
                force_position_control=False,
                background_streaming_enabled=True,
            ),
            pre_sdk_dispatch_adapter=binding.motion_adapter,
            hardware_idle_worker_preflight_binding=backend_binding,
        )


def test_narrow_backend_binding_is_instance_local_and_does_not_change_normal_defaults() -> None:
    _manifest, _evidence, _authorization, binding = _binding()
    preflight = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.1",
            side="left",
            connect_hardware=True,
            initial_qpos=None,
            configure_singularity_thresholds=False,
            force_position_control=False,
            background_streaming_enabled=True,
        ),
        hardware_idle_worker_preflight_binding=binding.backend_binding("left"),
    )
    preflight.close()

    default_config = RM65ArmBackendConfig(
        arm_ip="192.0.2.2",
        side="right",
        connect_hardware=False,
        initial_qpos=(0.0,) * 6,
    )
    assert default_config.configure_singularity_thresholds is True
    assert default_config.background_streaming_enabled is False
    assert default_config.force_position_control is False

    normal = RM65ArmBackend(default_config)
    try:
        stream = normal.get_state()["background_streaming"]
        assert stream["execution_epoch_binding_enabled"] is False
        assert stream["hardware_idle_worker_preflight_binding_id"] is None
    finally:
        normal.close()


def test_motion_adapter_is_hard_denied_even_with_preflight_binding() -> None:
    _manifest, _evidence, _authorization, binding = _binding()
    context = RM65JointTargetDispatchContext.create(
        side="left",
        qpos_rad=np.zeros(6),
        command_source="test",
        call_path="background_worker",
        quick_stop_active=False,
        lifecycle_home=False,
    )
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="rejects every RM65 joint-target"):
        binding.motion_adapter.invoke(
            context=context,
            backend_call=lambda: calls.append("forbidden"),
        )
    assert calls == []


def test_one_registry_drains_exact_three_real_worker_classes_without_hardware_start() -> None:
    _manifest, _evidence, _authorization, binding = _binding()
    capability = build_rm65_policy_arm_qpos_candidate_capability()
    position = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=30.0,
            reference_hz=capability.reference_resample_hz,
            command_hz=capability.command_hz,
        ),
        canonical_pipeline=_pipeline(),
        **binding.position_executor_kwargs(),
    )
    left = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.1",
            side="left",
            connect_hardware=True,
            initial_qpos=None,
            configure_singularity_thresholds=False,
            force_position_control=False,
            background_streaming_enabled=True,
        ),
        hardware_idle_worker_preflight_binding=binding.backend_binding("left"),
    )
    right = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.2",
            side="right",
            connect_hardware=True,
            initial_qpos=None,
            configure_singularity_thresholds=False,
            force_position_control=False,
            background_streaming_enabled=True,
        ),
        hardware_idle_worker_preflight_binding=binding.backend_binding("right"),
    )
    try:
        calls: list[str] = []

        class _StopOnlyArm:
            def Arm_Slow_Stop(self):
                calls.append("Arm_Slow_Stop")
                return 0

        left.arm = _StopOnlyArm()
        with pytest.raises(RuntimeError, match="before the exact worker drain"):
            left.hold_motion("m5_8b1_idle_worker_drain:operator_confirmed")
        assert calls == []

        cancellation = binding.context.execution_epochs.cancel(
            ExecutionCancellationRequest(
                reason="hold",
                authority_epoch=0,
                requested_scope="control_domain",
                source_event_ids=("operator-test",),
            ),
            capability=capability,
        )
        binding.context.observe_execution_cancellation(cancellation)
        result = drain_rm65_arms_only_workers(
            plan=binding.worker_drain_plan,
            cancellation=cancellation,
            position_reference_worker=position,
            left_background_worker=left,
            right_background_worker=right,
        )
        binding.accept_completed_worker_drain(
            result,
            cancellation=cancellation,
        )
        left.hold_motion("m5_8b1_idle_worker_drain:operator_confirmed")

        assert result.aggregate.complete
        assert result.aggregate.accepted_participant_ids == (
            "position-deploy-reference-worker",
            "rm65-left-background-stream-worker",
            "rm65-right-background-stream-worker",
        )
        assert binding.snapshot().execution_epoch == 1
        assert binding.snapshot().completed_drain_cancellation_id == (
            cancellation.cancellation_id
        )
        assert not binding.snapshot().motion_authorized
        assert calls == ["Arm_Slow_Stop"]
        with pytest.raises(RuntimeError, match="already consumed"):
            left.hold_motion("m5_8b1_idle_worker_drain:operator_confirmed")
    finally:
        left.arm = None
        position.close()
        left.close()
        right.close()


def test_dual_successful_stop_receipts_converge_the_same_owner_to_hold() -> None:
    _manifest, _evidence, _authorization, binding = _binding()
    capability = build_rm65_policy_arm_qpos_candidate_capability()
    position = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=30.0,
            reference_hz=capability.reference_resample_hz,
            command_hz=capability.command_hz,
        ),
        canonical_pipeline=_pipeline(),
        **binding.position_executor_kwargs(),
    )
    left = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.1",
            side="left",
            connect_hardware=True,
            initial_qpos=None,
            configure_singularity_thresholds=False,
            force_position_control=False,
            background_streaming_enabled=True,
        ),
        hardware_idle_worker_preflight_binding=binding.backend_binding("left"),
    )
    right = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.2",
            side="right",
            connect_hardware=True,
            initial_qpos=None,
            configure_singularity_thresholds=False,
            force_position_control=False,
            background_streaming_enabled=True,
        ),
        hardware_idle_worker_preflight_binding=binding.backend_binding("right"),
    )

    class StopOnlyArm:
        def Arm_Slow_Stop(self):
            return 0

    left.arm = StopOnlyArm()
    right.arm = StopOnlyArm()
    try:
        cancellation = binding.context.cancel_execution(
            ExecutionCancellationRequest(
                reason="hold",
                authority_epoch=0,
                requested_scope="control_domain",
                source_event_ids=("fixture-stop-convergence",),
            ),
            capability=capability,
        )
        worker_drain = drain_rm65_arms_only_workers(
            plan=binding.worker_drain_plan,
            cancellation=cancellation,
            position_reference_worker=position,
            left_background_worker=left,
            right_background_worker=right,
        )
        binding.accept_completed_worker_drain(
            worker_drain,
            cancellation=cancellation,
        )
        left.hold_motion("m5_8b1_idle_worker_drain:fixture")
        right.hold_motion("m5_8b1_idle_worker_drain:fixture")
        failed_right = replace(
            binding.stop_receipts[1],
            sdk_result_code=-1,
            sdk_result_text="-1",
            sdk_succeeded=False,
            sdk_result_repr="-1",
        )
        successful_stop_receipts = binding.stop_receipts
        binding._stop_receipts = [successful_stop_receipts[0], failed_right]
        rejected_hold = compile_rm65_idle_worker_backend_hold_receipt(
            binding=binding,
            cancellation=cancellation,
            acknowledged_at=time.monotonic(),
            clock_domain_id=binding.context.clock_domain_id,
        )
        assert not rejected_hold.complete
        assert rejected_hold.rejected_backend_ids == ("right",)
        binding._stop_receipts = list(successful_stop_receipts)

        backend_hold = compile_rm65_idle_worker_backend_hold_receipt(
            binding=binding,
            cancellation=cancellation,
            acknowledged_at=time.monotonic(),
            clock_domain_id=binding.context.clock_domain_id,
        )

        receipt = converge_rm65_legacy_control_stop(
            binding=binding,
            cancellation=cancellation,
            worker_drain=worker_drain,
            backend_hold=backend_hold,
        )

        assert backend_hold.complete
        assert backend_hold.production_binding_enabled
        assert receipt.complete
        assert receipt.production_binding_enabled
        assert receipt.lifecycle_state == "hold"
        assert receipt.execution_gate == "hold"
        assert binding.context.snapshot().semantic_content_hash == (
            receipt.owner_state_hash
        )
    finally:
        left.arm = None
        right.arm = None
        position.close()
        left.close()
        right.close()


def test_preflight_trace_profile_is_isolated_runtime_diagnostic() -> None:
    profile = trace_requirement_profile(RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1)

    assert profile.runtime_mode == "rm65_hardware_idle_worker_drain_preflight"
    assert profile.profile_scope == "runtime_diagnostic"
    assert profile.implementation_status == "runtime_integrated"
    assert profile.required_event_types == {
        "hardware_preflight_authorization",
        "execution_cancellation",
        "execution_cancellation_ack",
        "execution_worker_drain",
        "execution_worker_drain_aggregate",
        "runtime_trace_terminal",
    }


def test_preflight_trace_sequence_is_durable_without_becoming_capture_commit(
    tmp_path: Path,
) -> None:
    _manifest, _evidence, authorization, binding = _binding()
    capability = build_rm65_policy_arm_qpos_candidate_capability()
    position = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=30.0,
            reference_hz=capability.reference_resample_hz,
            command_hz=capability.command_hz,
        ),
        canonical_pipeline=_pipeline(),
        **binding.position_executor_kwargs(),
    )
    left = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.1",
            side="left",
            connect_hardware=True,
            initial_qpos=None,
            configure_singularity_thresholds=False,
            force_position_control=False,
            background_streaming_enabled=True,
        ),
        hardware_idle_worker_preflight_binding=binding.backend_binding("left"),
    )
    right = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.2",
            side="right",
            connect_hardware=True,
            initial_qpos=None,
            configure_singularity_thresholds=False,
            force_position_control=False,
            background_streaming_enabled=True,
        ),
        hardware_idle_worker_preflight_binding=binding.backend_binding("right"),
    )
    profile = trace_requirement_profile(RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1)
    runtime = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="trace-idle-worker-test",
            runtime_session_id=binding.runtime_session_id,
            control_domain_id=authorization.control_domain_id,
            profile_id=profile.profile_id,
            started_at=time.monotonic(),
            clock_domain_id="control-monotonic",
        ),
        profile=profile,
    )
    spool = ControlTraceSpool(
        tmp_path / "control_trace.wal",
        trace_session_id=runtime.session.trace_session_id,
        max_bytes=1 << 20,
        physical_preallocation=False,
    )
    health = ControlTraceWriterHealthLedger(
        tmp_path / "writer_health.ledger",
        trace_session_id=runtime.session.trace_session_id,
    )
    sink = ControlTraceSpoolSink(
        sink_id="idle-worker-test",
        runtime=runtime,
        spool=spool,
        health_ledger=health,
    )
    trace_context = AuthoritativeTraceContext(
        runtime_session_id=binding.runtime_session_id,
        control_domain_id=authorization.control_domain_id,
        embodiment_instance_id=authorization.embodiment_instance_id,
        clock_domain_id="control-monotonic",
    )

    def admit(fact: ControlTraceFact) -> None:
        assert runtime.issue(fact).accepted

    try:
        admit(binding.project_authorization_fact(trace_context))
        cancellation = binding.context.execution_epochs.cancel(
            ExecutionCancellationRequest(
                reason="hold",
                authority_epoch=0,
                requested_scope="control_domain",
                source_event_ids=(authorization.authorization_id,),
            ),
            capability=capability,
        )
        binding.context.observe_execution_cancellation(cancellation)
        admit(project_execution_cancellation(trace_context, cancellation))
        result = drain_rm65_arms_only_workers(
            plan=binding.worker_drain_plan,
            cancellation=cancellation,
            position_reference_worker=position,
            left_background_worker=left,
            right_background_worker=right,
        )
        binding.accept_completed_worker_drain(
            result,
            cancellation=cancellation,
        )
        report = admit_rm65_worker_drain_trace(
            result,
            cancellation=cancellation,
            embodiment_instance_id=authorization.embodiment_instance_id,
            fact_sink=admit,
        )
        assert report.complete
        tracker = CancellationAcknowledgementTracker()
        tracker.register(cancellation)
        acknowledgement = acknowledge_worker_epoch_drain(
            result.aggregate,
            tracker=tracker,
            clock_domain_id="control-monotonic",
        )
        admit(project_execution_cancellation_ack(trace_context, acknowledgement))
        # The diagnostic uses the same cancellation as the sole backend-hold
        # evidence.  This test does not call a hardware SDK, so mint/validate
        # the exact permit through a fake arm before projecting it.
        calls: list[str] = []

        class _StopOnlyArm:
            def Arm_Slow_Stop(self):
                calls.append("Arm_Slow_Stop")
                return 0

        left.arm = _StopOnlyArm()
        right.arm = _StopOnlyArm()
        left.hold_motion("m5_8b1_idle_worker_drain:operator_confirmed")
        right.hold_motion("m5_8b1_idle_worker_drain:operator_confirmed")
        assert calls == ["Arm_Slow_Stop", "Arm_Slow_Stop"]
        for fact in binding.project_stop_dispatch_facts(trace_context):
            admit(fact)
        terminal_at = time.monotonic()
        admit(
            ControlTraceFact.create(
                fact_id="idle-worker-terminal",
                event_type="runtime_trace_terminal",
                runtime_session_id=binding.runtime_session_id,
                control_domain_id=authorization.control_domain_id,
                embodiment_instance_id=authorization.embodiment_instance_id,
                occurred_at=terminal_at,
                clock_domain_id="control-monotonic",
                payload={
                    "terminal_status": "completed",
                    "reason": "test_complete",
                    "occurred_at": terminal_at,
                },
            )
        )
        assert sink.flush_pending().failure_code is None

        completeness = runtime.completeness()
        assert completeness.in_memory_complete, completeness.lineage_issues
        assert completeness.durable_complete
        assert not completeness.capture_commit_eligible
    finally:
        left.arm = None
        right.arm = None
        position.close()
        left.close()
        right.close()
        spool.close()
        health.close()


def test_debug_cli_wires_real_worker_classes_without_any_joint_target_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from teleop_runtime.debug.rm65_real_debug import (
        run_rm65_idle_worker_drain_preflight as cli,
    )

    class _FakeHardwareArm:
        def __init__(self) -> None:
            self.slow_stop_count = 0
            self.movej_count = 0
            self.uninit_count = 0
            self.socket_close_count = 0

        def Get_Current_Arm_State(self):
            return 0, [0.0] * 6, [0.0] * 6, 0, 0

        def Arm_Slow_Stop(self):
            self.slow_stop_count += 1
            return 0

        def Movej_CANFD(self, *_args, **_kwargs):
            self.movej_count += 1
            raise AssertionError("M5.8-b1 idle preflight must never call Movej_CANFD")

        def RM_API_UnInit(self):
            self.uninit_count += 1

        def Arm_Socket_Close(self):
            self.socket_close_count += 1

    arms = {
        "192.0.2.1": _FakeHardwareArm(),
        "192.0.2.2": _FakeHardwareArm(),
    }
    monkeypatch.setattr(
        rm65_module,
        "_load_rm65_sdk",
        lambda: (lambda _model, ip: arms[str(ip)], object()),
    )
    monkeypatch.setattr(
        RM65ArmBackend,
        "_auto_load_active_tool_profile",
        lambda self: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_rm65_idle_worker_drain_preflight",
            "--left-arm-ip",
            "192.0.2.1",
            "--right-arm-ip",
            "192.0.2.2",
            "--operator-authorization-id",
            "operator-cli-fixture",
            "--execute-idle-worker-drain-stop-only",
            "--confirm-arms-stationary",
            "--confirm-estop-ready",
            "--idle-observation-sec",
            "0.03",
            "--state-interval-sec",
            "0.01",
            "--evidence-root",
            str(tmp_path),
        ],
    )

    cli.main()

    assert all(arm.slow_stop_count == 1 for arm in arms.values())
    assert all(arm.movej_count == 0 for arm in arms.values())
    assert all(arm.uninit_count == 1 for arm in arms.values())
    assert all(arm.socket_close_count == 1 for arm in arms.values())
    evidence_files = tuple(tmp_path.glob("*/evidence.json"))
    assert len(evidence_files) == 1
    payload = json.loads(evidence_files[0].read_text(encoding="utf-8"))
    assert payload["status"] == "idle_worker_drain_stop_only_completed"
    assert payload["canonical_defaults_modified"] is False
    assert payload["motion_authorized"] is False
    assert payload["trace"]["durable_complete"] is True
    assert payload["trace"]["capture_commit_eligible"] is False
    assert len(payload["stop_receipts"]) == 2
    assert payload["artifact_storage"]["selection_source"] == "explicit"
    assert payload["artifact_storage"]["resolved_layout"] == "explicit"


def test_debug_cli_evidence_layout_is_legacy_by_default_and_opt_in_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from teleop_runtime.debug.rm65_real_debug import (
        run_rm65_idle_worker_drain_preflight as cli,
    )

    required = [
        "--left-arm-ip",
        "192.0.2.1",
        "--right-arm-ip",
        "192.0.2.2",
        "--operator-authorization-id",
        "operator-path-test",
    ]
    legacy_args = cli._parse_args(required)
    legacy = cli.resolve_preflight_evidence_root(legacy_args)
    assert legacy.path == cli.legacy_preflight_evidence_root()
    assert legacy.source == "legacy"

    canonical = tmp_path / "canonical-preflight"
    monkeypatch.setattr(
        cli,
        "canonical_preflight_evidence_root",
        lambda: canonical,
    )
    canonical_args = cli._parse_args(
        required + ["--artifact-layout", "runtime-artifacts-v1"]
    )
    selected = cli.resolve_preflight_evidence_root(canonical_args)
    assert selected.path == canonical
    assert selected.source == "canonical"
    assert not canonical.exists()

    explicit = tmp_path / "exact"
    explicit_args = cli._parse_args(
        required
        + [
            "--artifact-layout",
            "runtime-artifacts-v1",
            "--evidence-root",
            str(explicit),
        ]
    )
    selected = cli.resolve_preflight_evidence_root(explicit_args)
    assert selected.path == explicit
    assert selected.source == "explicit"
