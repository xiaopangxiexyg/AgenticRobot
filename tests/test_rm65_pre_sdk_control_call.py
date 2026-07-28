from __future__ import annotations

import numpy as np
import pytest

from robot_motion.runtime import (
    AdmittedSafetyControlEvidence,
    DispatchOwnerRegistry,
    DispatchPermitInvocation,
    LifecycleControlAuthorization,
)
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.backends.arms.rm65_control_dispatch_call import (
    RM65ControlDispatchContext,
    RM65PermittedPreSdkControlAdapter,
)


class _ControlFakeArm:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def Arm_Slow_Stop(self):
        self.calls.append("Arm_Slow_Stop")
        return 0

    def Arm_Stop(self):
        self.calls.append("Arm_Stop")
        return 0

    def Clear_Force_Data(self, _block):
        self.calls.append("Clear_Force_Data")
        return 0

    def Start_Force_Position_Move(self, _block):
        self.calls.append("Start_Force_Position_Move")
        return 0

    def Stop_Force_Position_Move(self, _block):
        self.calls.append("Stop_Force_Position_Move")
        return 0

    def RM_API_UnInit(self):
        self.calls.append("RM_API_UnInit")
        return 0

    def Arm_Socket_Close(self):
        self.calls.append("Arm_Socket_Close")
        return 0


class _TransientStopFakeArm(_ControlFakeArm):
    def __init__(self, *, force_stop_raises: bool = False) -> None:
        super().__init__()
        self.force_stop_raises = bool(force_stop_raises)
        self.stop_results = [-2, 0]

    def Stop_Force_Position_Move(self, _block):
        self.calls.append("Stop_Force_Position_Move")
        if self.force_stop_raises:
            raise RuntimeError("force stop response timeout")
        return 0

    def Arm_Stop(self):
        self.calls.append("Arm_Stop")
        return self.stop_results.pop(0) if self.stop_results else 0


class _RecordingControlAdapter:
    def __init__(self, *, invoke_backend: bool = True) -> None:
        self.invoke_backend = bool(invoke_backend)
        self.contexts: list[RM65ControlDispatchContext] = []

    def invoke(self, *, context, backend_call):
        self.contexts.append(context)
        if not self.invoke_backend:
            raise RuntimeError("control call rejected by test adapter")
        return backend_call()


def _backend(adapter=None) -> tuple[RM65ArmBackend, _ControlFakeArm]:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
        ),
        pre_sdk_control_adapter=adapter,
    )
    arm = _ControlFakeArm()
    backend.arm = arm
    return backend, arm


def test_default_off_hold_keeps_the_existing_single_sdk_call() -> None:
    backend, arm = _backend()

    backend.hold_motion("test")

    assert arm.calls == ["Arm_Slow_Stop"]


def test_explicit_hold_adapter_rejects_before_the_sdk_call() -> None:
    adapter = _RecordingControlAdapter(invoke_backend=False)
    backend, arm = _backend(adapter)

    with pytest.raises(RuntimeError, match="control call rejected"):
        backend.hold_motion("test")

    assert arm.calls == []
    assert len(adapter.contexts) == 1
    assert adapter.contexts[0].seam_id == "rm65.safety.hold.direct_stop"
    assert adapter.contexts[0].permit_kind == "safety_control"
    assert adapter.contexts[0].operation == "hold"


def test_emergency_stop_exposes_two_distinct_safety_calls() -> None:
    adapter = _RecordingControlAdapter()
    backend, arm = _backend(adapter)
    backend._force_position_active = True

    backend.emergency_stop()

    assert arm.calls == ["Stop_Force_Position_Move", "Arm_Stop"]
    assert [context.attempt_sequence for context in adapter.contexts] == [1, 2]
    assert [context.sdk_method for context in adapter.contexts] == [
        "Stop_Force_Position_Move",
        "Arm_Stop",
    ]
    assert all(
        context.seam_id == "rm65.safety.emergency_stop.force_and_arm"
        for context in adapter.contexts
    )


def test_emergency_stop_retries_receive_timeout_and_continues_after_force_stop_error() -> None:
    backend, _ = _backend()
    transient_arm = _TransientStopFakeArm(force_stop_raises=True)
    backend.arm = transient_arm
    backend._force_position_active = True

    backend.emergency_stop()

    assert transient_arm.calls == [
        "Stop_Force_Position_Move",
        "Arm_Stop",
        "Arm_Stop",
    ]
    assert backend.get_state()["send_result_text"] == "emergency_stop:Arm_Stop:0"


def test_manual_rearm_clears_emergency_latch_without_repeating_sdk_stop() -> None:
    backend, arm = _backend()
    backend._hardware_ready = True
    backend._emergency_stopped = True

    result = backend.manual_rearm_after_emergency_stop(np.zeros(6))

    assert result["rearmed"] is True
    assert backend.is_emergency_stopped() is False
    assert arm.calls == []
    assert backend.get_state()["send_result_text"] == "manual_rearm_ready"


def test_hold_after_emergency_latch_does_not_issue_duplicate_sdk_stop() -> None:
    backend, arm = _backend()
    backend._emergency_stopped = True

    backend.hold_motion("collision")
    backend.disengage_position_command_stream("collision")

    assert arm.calls == []
    assert "emergency_latched" in str(backend.get_state()["send_result_text"])


def test_force_position_start_uses_one_lifecycle_permit_per_sdk_call() -> None:
    registry = DispatchOwnerRegistry(
        runtime_session_id="session-rm65-control",
        control_domain_id="rm65-arms",
        initial_authority_epoch=0,
        initial_execution_epoch=3,
    )
    issued: list[str] = []

    def provider(context: RM65ControlDispatchContext):
        permit = registry.issue_lifecycle_control_permit(
            authorization=LifecycleControlAuthorization(
                authorization_event_id=(
                    f"startup-auth-{context.attempt_sequence}-{context.sdk_method}"
                ),
                runtime_session_id="session-rm65-control",
                control_domain_id="rm65-arms",
                lifecycle_state_machine_id="rm65-startup",
                requester_kind="lifecycle_state_machine",
                operation="startup",
                target_execution_epoch=3,
                operator_or_safety_policy_id="startup-profile-v1",
            )
        )
        issued.append(permit.permit_id)
        return DispatchPermitInvocation(
            permit=permit,
            current_authority_epoch=0,
            current_execution_epoch=3,
        )

    contexts: list[RM65ControlDispatchContext] = []
    adapter = RM65PermittedPreSdkControlAdapter(
        registry=registry,
        invocation_provider=provider,
        receipt_sink=lambda context, _result: contexts.append(context),
    )
    backend, arm = _backend(adapter)

    backend._start_force_position_control()

    assert arm.calls == ["Clear_Force_Data", "Start_Force_Position_Move"]
    assert len(issued) == 2
    assert len(set(issued)) == 2
    assert [context.attempt_sequence for context in contexts] == [1, 2]


def test_permitted_emergency_stop_uses_distinct_safety_evidence_and_permits() -> None:
    registry = DispatchOwnerRegistry(
        runtime_session_id="session-rm65-stop",
        control_domain_id="rm65-arms",
        initial_authority_epoch=0,
        initial_execution_epoch=5,
    )
    sequence = 0
    permits: list[str] = []

    def provider(context: RM65ControlDispatchContext):
        nonlocal sequence
        sequence += 1
        permit = registry.issue_safety_control_permit(
            operation="stop",
            evidence=AdmittedSafetyControlEvidence(
                event_id=f"estop-evidence-{sequence}",
                runtime_session_id="session-rm65-stop",
                control_domain_id="rm65-arms",
                execution_epoch=5,
                evidence_kind="safety_event",
                safety_source="runtime_supervisor",
                control_admission_sequence=sequence,
            ),
            safety_source="runtime_supervisor",
        )
        permits.append(permit.permit_id)
        return DispatchPermitInvocation(
            permit=permit,
            current_authority_epoch=0,
            current_execution_epoch=5,
        )

    adapter = RM65PermittedPreSdkControlAdapter(
        registry=registry,
        invocation_provider=provider,
    )
    backend, arm = _backend(adapter)
    backend._force_position_active = True

    backend.emergency_stop()

    assert arm.calls == ["Stop_Force_Position_Move", "Arm_Stop"]
    assert len(permits) == 2
    assert len(set(permits)) == 2


def test_disengage_and_fault_fallback_report_their_exact_seams() -> None:
    adapter = _RecordingControlAdapter()
    disengage_backend, disengage_arm = _backend(adapter)

    disengage_backend.disengage_position_command_stream("operator hold")

    assert disengage_arm.calls == ["Arm_Slow_Stop"]
    assert adapter.contexts[-1].seam_id == (
        "rm65.safety.position_stream_disengage.stop"
    )
    assert adapter.contexts[-1].operation == "disable_motion"

    fault_adapter = _RecordingControlAdapter()
    fault_backend, fault_arm = _backend(fault_adapter)
    fault_backend._attempt_background_fault_stop()

    assert fault_arm.calls == ["Arm_Slow_Stop"]
    assert fault_adapter.contexts[-1].seam_id == (
        "rm65.safety.background_fault.stop_fallback_loop"
    )
    assert fault_adapter.contexts[-1].operation == "stop"


def test_startup_rollback_wraps_force_stop_but_not_disconnect_calls() -> None:
    adapter = _RecordingControlAdapter()
    backend, arm = _backend(adapter)
    backend._force_position_active = True

    backend._rollback_failed_start()

    assert [context.seam_id for context in adapter.contexts] == [
        "rm65.cleanup.startup_rollback.force_stop"
    ]
    assert arm.calls == [
        "Stop_Force_Position_Move",
        "RM_API_UnInit",
        "Arm_Socket_Close",
    ]
    assert backend.arm is None


def test_close_wraps_each_stop_call_and_keeps_disconnect_outside_motion_permits() -> None:
    adapter = _RecordingControlAdapter()
    backend, arm = _backend(adapter)
    backend._force_position_active = True
    backend._last_attempted_sequence = 1

    backend.close()

    assert [context.sdk_method for context in adapter.contexts] == [
        "Stop_Force_Position_Move",
        "Arm_Slow_Stop",
    ]
    assert [context.attempt_sequence for context in adapter.contexts] == [1, 2]
    assert all(
        context.seam_id == "rm65.cleanup.close.stop_before_disconnect"
        for context in adapter.contexts
    )
    assert arm.calls == [
        "Stop_Force_Position_Move",
        "Arm_Slow_Stop",
        "RM_API_UnInit",
        "Arm_Socket_Close",
    ]
