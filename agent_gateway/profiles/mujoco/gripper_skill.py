"""MuJoCo adapter and contract builder for atomic gripper control."""

from __future__ import annotations

import time
from collections.abc import Callable
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


class MujocoGripperControlPort:
    """Drive one gripper through the rig's gripper-only lower boundary."""

    target_kind = "sim"

    def __init__(
        self,
        *,
        rig: Any,
        gripper_name: str,
        component_id: str | None = None,
        clock=time.monotonic,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable.")
        for method_name in (
            "is_ready",
            "queue_gripper_close_targets",
            "gripper_execution_snapshot",
            "hold_gripper_motion",
            "step_physics",
        ):
            if not callable(getattr(rig, method_name, None)):
                raise TypeError(f"rig must expose {method_name}().")
        if bool(getattr(getattr(rig, "config", None), "realtime_worker", True)):
            raise ValueError(
                "atomic MuJoCo gripper Skill requires deterministic physics ownership."
            )
        self.rig = rig
        explicit_component = component_id is not None
        component_id = "right_gripper" if component_id is None else component_id
        if component_id not in {"left_gripper", "right_gripper"}:
            raise ValueError("component_id must identify a left or right gripper.")
        self.component_id = component_id
        self.gripper_name = str(gripper_name).strip()
        if not self.gripper_name:
            raise ValueError("gripper_name must be non-empty.")
        profile = getattr(rig.config, "gripper_actuation_profile", None)
        side_binding = getattr(profile, "gripper_for_side", None)
        if explicit_component or profile is not None:
            if not callable(side_binding):
                raise ValueError("Explicit gripper components require the lower side binding.")
            if side_binding(component_id.removesuffix("_gripper")) != self.gripper_name:
                raise ValueError("Selected component does not match the lower physical gripper.")
        self.clock = clock
        self._sequence = 0
        self.port_binding_hash = canonical_content_hash(
            {
                "adapter": "mujoco_gripper_control.v1",
                "component_id": self.component_id,
                "gripper_name": self.gripper_name,
            },
            domain="dexteleop.agent_skill.gripper_port_binding.v1",
        )

    def _ref(self, kind: str, payload: dict[str, object]) -> str:
        self._sequence += 1
        digest = canonical_content_hash(
            {
                "kind": kind,
                "sequence": self._sequence,
                "component_id": self.component_id,
                **payload,
            },
            domain="dexteleop.sim.gripper_skill_evidence.v1",
        )
        return f"mujoco-gripper-{kind}:{digest}"

    def readiness(self) -> GripperReadiness:
        ready = bool(self.rig.is_ready())
        snapshot = self.rig.gripper_execution_snapshot()
        configured = self.gripper_name in snapshot["measured_close_by_gripper"]
        ref = self._ref("readiness", {"ready": ready, "configured": configured})
        return GripperReadiness(
            ready=ready and configured,
            lower_operation_id=f"mujoco-gripper:{self.gripper_name}",
            evidence_refs=(ref,),
        )

    def command_close(
        self, target_close: float, *, cancel_requested: Callable[[], bool] | None = None,
    ) -> GripperCommandReceipt:
        target = float(target_close)
        if cancel_requested is not None and cancel_requested():
            return GripperCommandReceipt(
                accepted=False,
                lower_operation_id=f"mujoco-gripper:{self.gripper_name}:preempted",
                evidence_refs=(self._ref("command_preempted", {"target_close": target}),),
            )
        sequence = self.rig.queue_gripper_close_targets({self.gripper_name: target})
        return GripperCommandReceipt(
            accepted=True,
            lower_operation_id=f"mujoco-gripper-command:{sequence}",
            evidence_refs=(
                self._ref("command", {"target_close": target, "sequence": sequence}),
            ),
        )

    def observe_state(self) -> GripperStateSample:
        snapshot = self.rig.gripper_execution_snapshot()
        measured = float(snapshot["measured_close_by_gripper"][self.gripper_name])
        rate = float(snapshot["actuator_target_rate_by_gripper"][self.gripper_name])
        pending = self.gripper_name in snapshot["pending_close_by_gripper"]
        moving = bool(pending or abs(rate) > 1e-5)
        observed_at = float(self.clock())
        return GripperStateSample(
            measured_close=measured,
            moving=moving,
            fault_code=None,
            observed_at=observed_at,
            evidence_ref=self._ref(
                "sample",
                {
                    "measured_close": measured,
                    "moving": moving,
                    "observed_at": observed_at,
                },
            ),
        )

    def poll_state(self) -> GripperStateSample:
        self.rig.step_physics()
        return self.observe_state()

    def hold(self, *, reason: str) -> GripperHoldReceipt:
        self.rig.hold_gripper_motion(reason)
        return GripperHoldReceipt(
            succeeded=True,
            evidence_refs=(self._ref("hold", {"reason": reason}),),
        )


def build_mujoco_gripper_skill_contract_bundle(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    port: MujocoGripperControlPort,
) -> GripperSkillContractBundle:
    if not isinstance(port, MujocoGripperControlPort):
        raise TypeError("port must be a MujocoGripperControlPort.")
    return build_gripper_skill_contract_bundle(
        GripperSkillDeploymentBinding(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            component_id=port.component_id,
            target_binding_id=target_binding_id,
            target_kind="sim",
            current_mode="sim_active",
            executor_id="control-station.mujoco-gripper-control.v1",
            port_binding_hash=port.port_binding_hash,
            station_manifest_hash=station_manifest_hash,
            execution_manifest_hash=execution_manifest_hash,
        )
    )


def assemble_mujoco_gripper_agent_gateway_session(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    port: MujocoGripperControlPort,
    allowed_client_ids: tuple[str, ...],
    current_station_revision,
    clock_domain_id: str,
    convergence=None,
) -> GripperAgentGatewaySession:
    bundle = build_mujoco_gripper_skill_contract_bundle(
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
    "MujocoGripperControlPort",
    "assemble_mujoco_gripper_agent_gateway_session",
    "build_mujoco_gripper_skill_contract_bundle",
]
