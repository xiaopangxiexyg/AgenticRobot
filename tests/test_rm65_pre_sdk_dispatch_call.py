from __future__ import annotations

import numpy as np
import pytest

from robot_motion.runtime import (
    AdmittedSafetyControlEvidence,
    DispatchOwnerRegistry,
    DispatchPermitInvocation,
    EffectorCommand,
    LifecycleControlAuthorization,
)
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.backends.arms.rm65_dispatch_call import (
    RM65JointTargetDispatchContext,
    RM65PermittedPreSdkDispatchAdapter,
)


class _FakeArm:
    def __init__(self) -> None:
        self.calls: list[tuple[float, ...]] = []

    def Movej_CANFD(self, joint_deg, **_kwargs):
        self.calls.append(tuple(float(value) for value in joint_deg))
        return 0


class _RecordingAdapter:
    def __init__(self, *, invoke_backend: bool = True) -> None:
        self.invoke_backend = bool(invoke_backend)
        self.contexts: list[RM65JointTargetDispatchContext] = []

    def invoke(self, *, context, backend_call):
        self.contexts.append(context)
        if not self.invoke_backend:
            raise RuntimeError("dispatch rejected by test adapter")
        return backend_call()


def _backend(adapter=None) -> tuple[RM65ArmBackend, _FakeArm]:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
        ),
        pre_sdk_dispatch_adapter=adapter,
    )
    arm = _FakeArm()
    backend.arm = arm
    return backend, arm


def test_default_off_path_preserves_legacy_private_call_without_context() -> None:
    backend, arm = _backend()

    assert backend._send_joint_target(np.zeros(6)) == 0

    assert arm.calls == [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]


def test_explicit_adapter_fails_closed_when_call_context_is_missing() -> None:
    adapter = _RecordingAdapter()
    backend, arm = _backend(adapter)

    with pytest.raises(RuntimeError, match="explicit command source"):
        backend._send_joint_target(np.zeros(6))

    assert adapter.contexts == []
    assert arm.calls == []


def test_typed_context_distinguishes_motion_home_and_quick_stop() -> None:
    adapter = _RecordingAdapter()
    backend, arm = _backend(adapter)

    backend._send_joint_target(
        np.zeros(6),
        command_source="policy",
        call_path="direct",
    )
    backend._send_joint_target(
        np.zeros(6),
        command_source="replay_qpos_init",
        call_path="direct",
    )
    backend._send_joint_target(
        np.zeros(6),
        command_source="policy:quick_stop",
        call_path="background_worker",
        quick_stop_active=True,
        target_sequence=7,
    )

    assert len(arm.calls) == 3
    assert [context.seam_id for context in adapter.contexts] == [
        "rm65.motion.pre_sdk.send_joint_target",
        "rm65.lifecycle.home.pre_sdk.send_joint_target",
        "rm65.safety.quick_stop.tick.pre_sdk",
    ]
    assert [context.permit_kind for context in adapter.contexts] == [
        "motion",
        "lifecycle_control",
        "safety_control",
    ]
    assert [context.operation for context in adapter.contexts] == [
        None,
        "home",
        "rm65_quick_stop_tick",
    ]
    assert adapter.contexts[-1].target_sequence == 7


def test_adapter_rejection_happens_before_the_sdk_call() -> None:
    adapter = _RecordingAdapter(invoke_backend=False)
    backend, arm = _backend(adapter)

    with pytest.raises(RuntimeError, match="dispatch rejected"):
        backend._send_joint_target(
            np.zeros(6),
            command_source="policy",
            call_path="direct",
        )

    assert len(adapter.contexts) == 1
    assert arm.calls == []


def test_qpos_init_send_command_reaches_lifecycle_typed_context() -> None:
    adapter = _RecordingAdapter()
    backend, arm = _backend(adapter)

    receipt = backend.send_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.zeros(6),
            frame_name=None,
            metadata={"source": "replay_qpos_init"},
        )
    )

    assert receipt.accepted
    assert len(arm.calls) == 1
    assert len(adapter.contexts) == 1
    assert adapter.contexts[0].permit_kind == "lifecycle_control"
    assert adapter.contexts[0].operation == "home"


def test_permitted_adapter_mints_nothing_and_consumes_one_new_permit_per_call() -> None:
    registry = DispatchOwnerRegistry(
        runtime_session_id="session-rm65-pre-sdk",
        control_domain_id="rm65-arms",
        initial_authority_epoch=2,
        initial_execution_epoch=4,
    )
    token = registry.grant_initial_owner(
        owner_runtime_id="legacy-policy-runtime",
        owner_kind="legacy_control_path",
    )
    issued_permit_ids: list[str] = []
    validation_statuses: list[str] = []
    safety_sequence = 0

    def invocation_provider(context: RM65JointTargetDispatchContext):
        nonlocal safety_sequence
        if context.permit_kind == "motion":
            permit = registry.issue_motion_permit(
                token=token,
                authority_epoch=2,
                execution_epoch=4,
                command_lineage_id=context.semantic_content_hash,
                directive_lineage_id="directive-motion",
            )
        elif context.permit_kind == "lifecycle_control":
            permit = registry.issue_lifecycle_control_permit(
                authorization=LifecycleControlAuthorization(
                    authorization_event_id="home-authorization",
                    runtime_session_id="session-rm65-pre-sdk",
                    control_domain_id="rm65-arms",
                    lifecycle_state_machine_id="policy-qpos-init",
                        requester_kind="lifecycle_state_machine",
                    operation="home",
                    target_execution_epoch=4,
                    operator_or_safety_policy_id="operator-0",
                )
            )
        else:
            safety_sequence += 1
            permit = registry.issue_safety_control_permit(
                operation="rm65_quick_stop_tick",
                evidence=AdmittedSafetyControlEvidence(
                    event_id=f"quick-stop-tick-{safety_sequence}",
                    runtime_session_id="session-rm65-pre-sdk",
                    control_domain_id="rm65-arms",
                    execution_epoch=4,
                    evidence_kind="safety_operation_tick",
                    safety_source="shared_safety_guard",
                    control_admission_sequence=safety_sequence,
                ),
                safety_source="shared_safety_guard",
            )
        issued_permit_ids.append(permit.permit_id)
        return DispatchPermitInvocation(
            permit=permit,
            current_authority_epoch=2,
            current_execution_epoch=4,
        )

    adapter = RM65PermittedPreSdkDispatchAdapter(
        registry=registry,
        invocation_provider=invocation_provider,
        receipt_sink=lambda _context, result: validation_statuses.append(
            result.validation.status
        ),
    )
    backend, arm = _backend(adapter)

    for source, quick_stop in (
        ("policy", False),
        ("policy", False),
        ("replay_qpos_init", False),
        ("policy:quick_stop", True),
    ):
        backend._send_joint_target(
            np.zeros(6),
            command_source=source,
            call_path="background_worker" if quick_stop else "direct",
            quick_stop_active=quick_stop,
        )

    assert len(arm.calls) == 4
    assert len(issued_permit_ids) == 4
    assert len(set(issued_permit_ids)) == 4
    assert validation_statuses == ["accepted"] * 4
