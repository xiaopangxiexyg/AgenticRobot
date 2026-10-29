"""MuJoCo adapter and contract builder for atomic gripper control."""

from __future__ import annotations

import math
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
from sim.mujoco.deploy.grasp_contact import MujocoGraspContactBinding

# Named-support release: the 0.6 s release window plus margin for the fingers
# to clear and the object to settle; checked every 10 ms of simulated time.
NAMED_RELEASE_MAXIMUM_OBSERVATION_SEC = 1.5
NAMED_RELEASE_CHECK_INTERVAL_SEC = 0.01


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
        arm_held: Callable[[], bool] | None = None,
        stationary_joint_speed_rad_sec: float = 0.02,
        target_contact_binding: MujocoGraspContactBinding | None = None,
        release_support_body_names: tuple[str, ...] = (),
        held_transaction: Callable[[Callable[[], Any]], Any] | None = None,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable.")
        if arm_held is not None and not callable(arm_held):
            raise TypeError("arm_held must be callable or None.")
        stationary_joint_speed_rad_sec = float(stationary_joint_speed_rad_sec)
        if not math.isfinite(stationary_joint_speed_rad_sec) or stationary_joint_speed_rad_sec <= 0:
            raise ValueError("stationary_joint_speed_rad_sec must be finite and positive.")
        self._stationary_joint_speed_rad_sec = stationary_joint_speed_rad_sec
        self._arm_held = arm_held
        self._requires_arm_hold = arm_held is not None
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
        if target_contact_binding is not None:
            if (not isinstance(target_contact_binding, MujocoGraspContactBinding)
                    or component_id != "right_gripper"
                    or len(target_contact_binding.sensor_geom_names) != 2):
                raise ValueError("Target contact must bind the right gripper and two native sensors.")
            rig.capture_scene(lambda scene: target_contact_binding.pair_clearances(scene))
        self.target_contact_binding = target_contact_binding
        if isinstance(release_support_body_names, str):
            raise ValueError("Named release supports require explicit object identities.")
        release_support_body_names = tuple(release_support_body_names)
        if release_support_body_names and (
                component_id != "right_gripper" or target_contact_binding is None
                or any(not isinstance(name, str) or not name or name != name.strip()
                       for name in release_support_body_names)
                or len(set(release_support_body_names)) != len(release_support_body_names)):
            raise ValueError("Named release requires the target-bound right gripper and support IDs.")
        self.release_support_body_names = release_support_body_names
        self._release_observer = None
        if release_support_body_names and not callable(held_transaction):
            raise TypeError("Named release requires the application's ordered HOLD transaction.")
        self._held_transaction = held_transaction
        if release_support_body_names and not callable(getattr(
                rig, "queue_gripper_close_targets_after_scene_check", None)):
            raise TypeError("Named release requires the rig's atomic gripper precondition port.")
        self._contact_scratch = None
        self._contact_model = None
        profile = getattr(rig.config, "gripper_actuation_profile", None)
        side_binding = getattr(profile, "gripper_for_side", None)
        if explicit_component or profile is not None:
            if not callable(side_binding):
                raise ValueError("Explicit gripper components require the lower side binding.")
            if side_binding(component_id.removesuffix("_gripper")) != self.gripper_name:
                raise ValueError("Selected component does not match the lower physical gripper.")
        self.clock = clock
        self._sequence = 0
        actuator_profile = getattr(rig.config, "gripper_actuation_profile", None)
        self.port_binding_hash = canonical_content_hash(
            {
                "adapter": "mujoco_gripper_control.v1",
                "component_id": self.component_id,
                "gripper_name": self.gripper_name,
                "actuation_profile": None if actuator_profile is None else actuator_profile.to_metadata(),
                "requires_arm_hold": arm_held is not None,
                "stationary_joint_speed_rad_sec": stationary_joint_speed_rad_sec,
                **({} if target_contact_binding is None else {
                    "target_contact_binding_hash": target_contact_binding.semantic_content_hash,
                }),
                **({} if not release_support_body_names else {
                    "release_support_body_names": release_support_body_names,
                    "held_transaction": "application_lock_before_rig_lock.v1",
                }),
            },
            domain="dexteleop.agent_skill.gripper_port_binding.v1",
        )

    @property
    def requires_arm_hold(self) -> bool:
        return self._requires_arm_hold

    def bind_motion_guard(self, guard: Callable[[], bool]) -> None:
        if not callable(guard):
            raise TypeError("motion guard must be callable.")
        if not self.requires_arm_hold:
            raise ValueError("Composed grippers require an arm-HOLD-bound port.")
        self._arm_held = guard

    def bind_release_observer(self, observer) -> None:
        """Attach the existing rig-tick observer; never create a second owner."""
        from sim.mujoco.deploy.grasped_object import MujocoGraspedObjectObserver

        binding = self.target_contact_binding
        if (not self.release_support_body_names or self._release_observer is not None
                or not isinstance(observer, MujocoGraspedObjectObserver)
                or observer.rig is not self.rig or binding is None
                or observer.object_body_name != binding.target_body_name
                or observer.sensor_geom_names != binding.sensor_geom_names
                or observer.gripper_name != self.gripper_name
                or tuple(observer.support_body_names[1:]) != self.release_support_body_names):
            raise ValueError("Named release observer differs from the bound rig/object/supports.")
        self._release_observer = observer

    def command_open_to_support(
        self, support_body_name: str, *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> GripperCommandReceipt:
        if support_body_name not in self.release_support_body_names:
            raise ValueError("Named release support differs from the deployment allowlist.")
        if self._release_observer is None:
            raise RuntimeError("Named release requires the bound measured object observer.")
        if (cancel_requested is not None and cancel_requested()) or (
                self._arm_held is not None and not self._arm_held()):
            return GripperCommandReceipt(
                accepted=False,
                lower_operation_id=f"mujoco-gripper:{self.gripper_name}:preempted",
                evidence_refs=(self._ref("command_preempted", {
                    "target_close": 0., "support_body_name": support_body_name}),),
            )

        def ordered_transition():
            previous_support = []

            def check_scene(scene):
                previous_support.append(self._release_observer.active_support_body_name)
                return self._release_observer.check_named_release_precondition(
                    scene, support_body_name)

            return self.rig.queue_gripper_close_targets_after_scene_check(
                {self.gripper_name: 0.},
                scene_check=check_scene,
                cancelled=cancel_requested,
                on_queued=lambda _scene, _cut:
                    self._release_observer.commit_named_release_support(support_body_name),
                on_queue_failed=lambda _scene, _cut:
                    self._release_observer.commit_named_release_support(previous_support[0]))

        sequence, cut = self._held_transaction(ordered_transition)
        if sequence is None:
            return GripperCommandReceipt(
                accepted=False,
                lower_operation_id=f"mujoco-gripper:{self.gripper_name}:preempted",
                evidence_refs=(self._ref("command_preempted", {
                    "target_close": 0., "support_body_name": support_body_name}),),
            )
        return GripperCommandReceipt(
            accepted=True,
            lower_operation_id=f"mujoco-gripper-command:{sequence}",
            evidence_refs=(self._ref("command", {
                "target_close": 0., "sequence": sequence,
                "support_body_name": support_body_name,
                "measured_held": True,
                "measured_at": cut[0]}),),
        )

    def observe_named_release(
        self, *, cancel_requested: Callable[[], bool],
        maximum_observation_sec: float = NAMED_RELEASE_MAXIMUM_OBSERVATION_SEC,
        check_interval_sec: float = NAMED_RELEASE_CHECK_INTERVAL_SEC,
    ) -> dict[str, Any] | None:
        """Keep the open command and let simulated time pass for the release window.

        Without a realtime worker, physics advances only inside an operation.
        After a named-support open reaches its target, the same gripper owner
        keeps stepping its rig (arm HOLD, gripper open) until the bound
        observer reports a stable supported release or the bound elapses.
        This adds measured samples only; it does not judge or latch success.
        """
        observer = self._release_observer
        if observer is None:
            raise RuntimeError("Named release observation requires the bound measured object observer.")
        timestep = float(self.rig.config.timing.physics_timestep_sec)
        if not (math.isfinite(timestep) and timestep > 0.):
            raise RuntimeError("Named release observation needs a positive physics timestep.")
        steps = max(1, int(math.ceil(float(maximum_observation_sec) / timestep)))
        stride = max(1, int(round(float(check_interval_sec) / timestep)))
        snapshot = observer.release_snapshot()
        for index in range(steps):
            if snapshot is not None and snapshot.get("stable_supported_release") is True:
                break
            if cancel_requested():
                break
            self.rig.step_physics()
            if (index + 1) % stride == 0 or index + 1 == steps:
                snapshot = observer.release_snapshot()
        return snapshot

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
        ready = bool(self.rig.is_ready()) and (self._arm_held is None or self._arm_held())
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
        if (cancel_requested is not None and cancel_requested()) or (
            self._arm_held is not None and not self._arm_held()
        ):
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
        joint_speed = float(snapshot["measured_max_joint_speed_rad_sec_by_gripper"][self.gripper_name])
        if not math.isfinite(joint_speed) or joint_speed < 0:
            raise RuntimeError("MuJoCo measured gripper joint speed is invalid.")
        # A settled interpolator is not proof that the physical hand stopped.
        moving = bool(pending or abs(rate) > 1e-5
                      or joint_speed > self._stationary_joint_speed_rad_sec)
        contact = None
        if self.target_contact_binding is not None:
            contact = bool(
                not moving and self.target_command_settled(snapshot)
                and self.rig.capture_scene(self._target_bilateral_contact)
            )
        observed_at = float(self.clock())
        return GripperStateSample(
            measured_close=measured,
            moving=moving,
            target_contact_confirmed=contact,
            fault_code=None,
            observed_at=observed_at,
            evidence_ref=self._ref(
                "sample",
                {
                    "measured_close": measured,
                    "moving": moving,
                    "target_contact_confirmed": contact,
                    "max_joint_speed_rad_sec": joint_speed,
                    "observed_at": observed_at,
                },
            ),
        )

    def target_command_settled(self, snapshot=None) -> bool:
        """Read the existing actuator and joint cut; no new motion or owner."""
        if snapshot is None:
            snapshot = self.rig.gripper_execution_snapshot()
        name = self.gripper_name
        return bool(
            name not in snapshot["pending_close_by_gripper"]
            and float(snapshot["accepted_close_by_gripper"][name]) == 1.0
            and float(snapshot["actuator_target_close_by_gripper"][name]) >= 1.0 - 1e-8
            and abs(float(snapshot["actuator_target_rate_by_gripper"][name])) <= 1e-5
            and float(snapshot["measured_max_joint_speed_rad_sec_by_gripper"][name])
            <= self._stationary_joint_speed_rad_sec
        )

    def _target_bilateral_contact(self, scene) -> bool:
        binding = self.target_contact_binding
        if binding is None:
            return False
        model, mj = scene.model, scene.mujoco
        if self._contact_scratch is None:
            self._contact_scratch, self._contact_model = mj.MjData(model), model
        if self._contact_model is not model:
            raise ValueError("Target contact model changed.")
        data = self._contact_scratch
        mj.mj_copyData(data, model, scene.data)
        mj.mj_kinematics(model, data)
        mj.mj_collision(model, data)
        target = int(model.body(binding.target_body_name).id)
        target_geoms = {int(g) for g in range(model.ngeom)
                        if int(model.geom_bodyid[g]) == target and model.geom_contype[g] == 8}
        sensors = {int(model.geom(name).id) for name in binding.sensor_geom_names}
        touching = set()
        for item in data.contact[:int(data.ncon)]:
            if item.dist > 0:
                continue
            a, b = int(item.geom1), int(item.geom2)
            if a in target_geoms and b in sensors:
                touching.add(b)
            elif b in target_geoms and a in sensors:
                touching.add(a)
        return touching == sensors

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
