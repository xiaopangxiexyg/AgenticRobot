from __future__ import annotations

import numpy as np
import pytest

from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.backends.arms.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
)
from teleop_runtime.backends.arms.rm65_control_dispatch_call import (
    RM65ControlDispatchContext,
)
from teleop_runtime.backends.arms.rm65_dispatch_call import (
    RM65JointTargetDispatchContext,
)
from teleop_runtime.backends.arms.rm65_dispatch_inventory import (
    build_rm65_arms_only_dispatch_inventory,
)
from teleop_runtime.backends.arms.rm65_stop_only_binding import (
    RM65StopOnlyHardwareAuthorization,
    RM65StopOnlyLegacyBinding,
)


class ManualClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class FakeArm:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def Arm_Slow_Stop(self):
        self.calls.append("Arm_Slow_Stop")
        return 0


def _binding(clock: ManualClock | None = None):
    clock = clock or ManualClock()
    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    inventory = build_rm65_arms_only_dispatch_inventory(manifest)
    authorization = RM65StopOnlyHardwareAuthorization.create(
        operator_authorization_id="operator-approved-m5.8b0",
        execution_manifest=manifest,
        dispatch_inventory=inventory,
        control_domain_id="rm65-arms",
        embodiment_instance_id="rm65-bimanual-0",
        allowed_sides=("left", "right"),
        issued_at=clock.value,
        expires_at=clock.value + 30.0,
    )
    return RM65StopOnlyLegacyBinding(
        runtime_session_id="session-rm65-stop-only",
        owner_runtime_id="legacy-stop-only-preflight",
        authorization=authorization,
        clock=clock,
    )


def _backend(binding: RM65StopOnlyLegacyBinding, side: str):
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side=side,
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            configure_singularity_thresholds=False,
            force_position_control=False,
            background_streaming_enabled=False,
        ),
        pre_sdk_dispatch_adapter=binding.motion_adapter,
        pre_sdk_control_adapter=binding.control_adapter,
    )
    arm = FakeArm()
    backend.arm = arm
    return backend, arm


def test_stop_only_binding_allows_one_exact_slow_stop_per_authorized_side() -> None:
    binding = _binding()
    left, left_arm = _backend(binding, "left")
    right, right_arm = _backend(binding, "right")

    left.hold_motion("m5_8b0_stop_only:operator_confirmed")
    right.hold_motion("m5_8b0_stop_only:operator_confirmed")

    assert left_arm.calls == ["Arm_Slow_Stop"]
    assert right_arm.calls == ["Arm_Slow_Stop"]
    assert [receipt.side for receipt in binding.receipts] == ["left", "right"]
    assert all(receipt.dispatch_accepted for receipt in binding.receipts)
    assert all(receipt.dispatch_status == "accepted" for receipt in binding.receipts)
    assert all(
        receipt.seam_id == "rm65.safety.hold.direct_stop"
        for receipt in binding.receipts
    )
    assert all(receipt.sdk_method == "Arm_Slow_Stop" for receipt in binding.receipts)
    assert len({receipt.permit_id for receipt in binding.receipts}) == 2
    assert all(receipt.owner_validation_receipt_id for receipt in binding.receipts)
    snapshot = binding.snapshot()
    assert snapshot.production_binding_enabled
    assert not snapshot.motion_authorized
    assert snapshot.attempted_sides == ("left", "right")
    assert snapshot.receipt_count == 2

    with pytest.raises(RuntimeError, match="already consumed"):
        left.hold_motion("m5_8b0_stop_only:second-attempt")
    assert left_arm.calls == ["Arm_Slow_Stop"]


def test_stop_only_binding_rejects_motion_and_lifecycle_before_callback() -> None:
    binding = _binding()
    calls: list[str] = []
    qpos_context = RM65JointTargetDispatchContext.create(
        side="left",
        qpos_rad=np.zeros(6),
        command_source="normal_policy",
        call_path="direct",
        quick_stop_active=False,
        lifecycle_home=False,
    )
    with pytest.raises(RuntimeError, match="rejects every RM65 joint-target"):
        binding.motion_adapter.invoke(
            context=qpos_context,
            backend_call=lambda: calls.append("motion"),
        )

    lifecycle_context = RM65ControlDispatchContext.create(
        side="left",
        seam_id="rm65.lifecycle.force_position.start",
        permit_kind="lifecycle_control",
        operation="startup",
        source="force_position_startup",
        sdk_method="Start_Force_Position_Move",
    )
    with pytest.raises(RuntimeError, match="permits only"):
        binding.control_adapter.invoke(
            context=lifecycle_context,
            backend_call=lambda: calls.append("lifecycle"),
        )
    assert calls == []


def test_stop_only_binding_expires_and_fails_before_sdk_call() -> None:
    clock = ManualClock()
    binding = _binding(clock)
    backend, arm = _backend(binding, "left")
    clock.value += 31.0

    with pytest.raises(RuntimeError, match="authorization expired"):
        backend.hold_motion("m5_8b0_stop_only:operator_confirmed")

    assert arm.calls == []
    assert binding.receipts == ()
