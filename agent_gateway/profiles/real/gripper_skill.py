"""Real DMTAC adapter and contract builder for atomic gripper control."""

from __future__ import annotations

import time
from typing import Any

from agent_gateway.skills.gripper.deployment import (
    GripperSkillContractBundle,
    GripperSkillDeploymentBinding,
    build_gripper_skill_contract_bundle,
)
from agent_gateway.skills.gripper.runtime import (
    GripperAgentGatewaySession,
    GripperCommandReceipt,
    GripperHoldReceipt,
    GripperReadiness,
    GripperStateSample,
    assemble_gripper_agent_gateway_session,
)
from dataset_contracts.canonical_encoding import canonical_content_hash


class RealDmtacGripperControlPort:
    """Adapt an already-started DMTAC backend without owning its lifecycle."""

    component_id = "right_gripper"
    target_kind = "real"

    def __init__(self, *, backend: Any, clock=time.monotonic) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable.")
        for method_name in (
            "is_ready",
            "readiness_report",
            "send_fixed_joint_pos",
            "read_gripper_state_now",
            "get_state",
            "hold_gripper_motion",
        ):
            if not callable(getattr(backend, method_name, None)):
                raise TypeError(f"backend must expose {method_name}().")
        self.backend = backend
        self.clock = clock
        self._sequence = 0
        config = getattr(backend, "config", None)
        if bool(getattr(config, "observation_only", False)):
            raise ValueError(
                "real gripper Skill cannot use an observation-only backend."
            )
        if getattr(config, "connect_hardware", True) is not True:
            raise ValueError(
                "real gripper Skill requires a hardware-connected backend."
            )
        if getattr(config, "disconnected_state_scalar", None) is not None:
            raise ValueError("real gripper Skill rejects assumed disconnected state.")
        identity = {
            "adapter": "real_dmtac_gripper_control.v1",
            "component_id": self.component_id,
            "side": str(getattr(config, "side", "right")),
            "gripper_port": str(getattr(config, "gripper_port", "bound")),
            "gripper_id": int(getattr(config, "gripper_id", 1)),
        }
        self.port_binding_hash = canonical_content_hash(
            identity,
            domain="dexteleop.agent_skill.gripper_port_binding.v1",
        )

    def _ref(self, kind: str, payload: dict[str, object]) -> str:
        self._sequence += 1
        digest = canonical_content_hash(
            {"kind": kind, "sequence": self._sequence, **payload},
            domain="dexteleop.real.gripper_skill_evidence.v1",
        )
        return f"dmtac-gripper-{kind}:{digest}"

    def readiness(self) -> GripperReadiness:
        report = self.backend.readiness_report()
        ready = bool(self.backend.is_ready()) and bool(report.get("ready"))
        ref = self._ref(
            "readiness",
            {"ready": ready, "reason": str(report.get("reason", ""))},
        )
        return GripperReadiness(
            ready=ready,
            lower_operation_id="dmtac-gripper:live-session",
            evidence_refs=(ref,),
        )

    def command_close(self, target_close: float) -> GripperCommandReceipt:
        target = float(target_close)
        accepted = bool(self.backend.send_fixed_joint_pos(target))
        return GripperCommandReceipt(
            accepted=accepted,
            lower_operation_id=f"dmtac-gripper-command:{self._sequence + 1}",
            evidence_refs=(
                self._ref("command", {"target_close": target, "accepted": accepted}),
            ),
        )

    def observe_state(self) -> GripperStateSample:
        self.backend.read_gripper_state_now()
        state = self.backend.get_state(
            include_tactile_arrays=False,
            include_tactile_vector=False,
        )
        measured = state.get("gripper_measured_scalar")
        if measured is None:
            raise RuntimeError("DMTAC backend has no measured gripper scalar.")
        error_code = state.get("gripper_error_code")
        fault_code = (
            None if error_code in (None, 0) else f"dmtac_error_{int(error_code)}"
        )
        moving_raw = state.get("gripper_is_moving")
        moving = None if moving_raw is None else bool(moving_raw)
        observed_at = float(self.clock())
        return GripperStateSample(
            measured_close=float(measured),
            moving=moving,
            fault_code=fault_code,
            observed_at=observed_at,
            evidence_ref=self._ref(
                "sample",
                {
                    "measured_close": float(measured),
                    "moving": moving,
                    "fault_code": fault_code,
                    "observed_at": observed_at,
                },
            ),
        )

    def poll_state(self) -> GripperStateSample:
        return self.observe_state()

    def hold(self, *, reason: str) -> GripperHoldReceipt:
        succeeded = bool(self.backend.hold_gripper_motion(reason))
        return GripperHoldReceipt(
            succeeded=succeeded,
            evidence_refs=(
                self._ref("hold", {"reason": reason, "succeeded": succeeded}),
            ),
        )


def build_real_dmtac_gripper_skill_contract_bundle(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    port: RealDmtacGripperControlPort,
) -> GripperSkillContractBundle:
    if not isinstance(port, RealDmtacGripperControlPort):
        raise TypeError("port must be a RealDmtacGripperControlPort.")
    return build_gripper_skill_contract_bundle(
        GripperSkillDeploymentBinding(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            component_id=port.component_id,
            target_binding_id=target_binding_id,
            target_kind="real",
            current_mode="real_active",
            executor_id="control-station.real-dmtac-gripper-control.v1",
            port_binding_hash=port.port_binding_hash,
            station_manifest_hash=station_manifest_hash,
            execution_manifest_hash=execution_manifest_hash,
        )
    )


def assemble_real_dmtac_gripper_agent_gateway_session(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    port: RealDmtacGripperControlPort,
    allowed_client_ids: tuple[str, ...],
    current_station_revision,
    clock_domain_id: str,
    convergence=None,
) -> GripperAgentGatewaySession:
    bundle = build_real_dmtac_gripper_skill_contract_bundle(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        port=port,
    )
    return assemble_gripper_agent_gateway_session(
        contract_bundle=bundle,
        port=port,
        allowed_client_ids=allowed_client_ids,
        current_station_revision=current_station_revision,
        clock_domain_id=clock_domain_id,
        convergence=convergence,
        clock=port.clock,
    )


__all__ = [
    "RealDmtacGripperControlPort",
    "assemble_real_dmtac_gripper_agent_gateway_session",
    "build_real_dmtac_gripper_skill_contract_bundle",
]
