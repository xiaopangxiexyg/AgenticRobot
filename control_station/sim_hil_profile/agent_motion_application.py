"""No-model Station lifecycle around one lower MuJoCo motion runtime.

The application serializes operation lifecycle, proposal submission, stepping
and passive capture.  The lower runtime remains the only Arbiter, canonical
safety, coupled-IK, reference and dispatch owner.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from robot_motion.action import ComponentAction
from robot_motion.deploy.schema import pose_matrix_to_xyzquat
from sim.mujoco.deploy.sim_hil_target import (
    MujocoAgentSkillMotionTarget,
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
)

from .application import MujocoSimHILPassiveApplicationCut


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


@dataclass(frozen=True)
class MujocoAgentSkillMotionStep:
    proposal_admission: Any
    authoritative_step: Any


class MujocoAgentSkillMotionApplication:
    """Own one reusable no-model Agent motion session, default-off."""

    def __init__(
        self,
        *,
        target: MujocoAgentSkillMotionTarget,
        calibration_hash: str,
    ) -> None:
        if not isinstance(target, MujocoAgentSkillMotionTarget):
            raise TypeError("target must be a MujocoAgentSkillMotionTarget.")
        runtime = target.runtime
        direct = getattr(
            getattr(runtime, "config", None),
            "direct_component_proposal_source",
            None,
        )
        if (
            getattr(direct, "producer_kind", None) != "agent_skill"
            or getattr(runtime.config, "initial_control_mode", None)
            != "agent_skill"
            or tuple(getattr(runtime.action_schema, "component_ids", ()))
            != ("right_arm",)
        ):
            raise ValueError(
                "Motion application requires the exact right-arm Agent Skill runtime."
            )
        self.target = target
        self.task_config = target.task_config
        self.scene = target.scene
        self.runtime = runtime
        self.target_frame_profile = target.target_frame_profile
        self.tool_frame_profile = target.tool_frame_profile
        self.task_initialization_info = dict(target.task_initialization_info)
        self.calibration_hash = _required(
            calibration_hash,
            field_name="calibration_hash",
        )
        self._lock = threading.RLock()
        self._started = False
        self._held = False
        self._closed = False
        self._active_operation_id: str | None = None

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    @property
    def held(self) -> bool:
        with self._lock:
            return self._held

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def active_operation_id(self) -> str | None:
        with self._lock:
            return self._active_operation_id

    def activate_operation(self, *, operation_id: str) -> Any:
        """Start once and explicitly open READY for one fresh operation."""

        normalized_id = _required(operation_id, field_name="operation_id")
        with self._lock:
            if self._closed:
                raise RuntimeError("Agent motion application is closed.")
            if self._active_operation_id not in {None, normalized_id}:
                raise RuntimeError("Another Agent motion operation is active.")
            if not self._started:
                self.runtime.start()
                self._started = True
            transition = self.runtime.transition_execution_gate(
                "ready",
                reason="agent_skill_operation_activation",
                source_event_ids=(normalized_id,),
            )
            self._active_operation_id = normalized_id
            self._held = False
            return transition

    def submit_and_step(
        self,
        *,
        operation_id: str,
        action: ComponentAction,
        source_event_id: str,
        observation_cut_id: str | None = None,
        context_ids: tuple[str, ...] = (),
        physics_ticks_after_dispatch: int = 0,
    ) -> MujocoAgentSkillMotionStep:
        normalized_id = _required(operation_id, field_name="operation_id")
        with self._lock:
            if not self._started or self._closed:
                raise RuntimeError("Agent motion application is not running.")
            if self._active_operation_id != normalized_id:
                raise ValueError("Operation does not own the Agent motion session.")
            now = float(self.runtime.clock())
            target_at = (
                now
                + self.runtime.core.config.decision_schedule.target_lead_time_sec
            )
            admission = self.runtime.submit_component_action(
                action,
                source_event_id=source_event_id,
                produced_at=now,
                target_effective_at=target_at,
                observation_cut_id=observation_cut_id,
                context_ids=(normalized_id, *tuple(context_ids)),
            )
            step = (
                None
                if not admission.accepted
                else self.runtime.step(
                    now=target_at,
                    physics_ticks_after_dispatch=physics_ticks_after_dispatch,
                )
            )
            return MujocoAgentSkillMotionStep(admission, step)

    def capture_tcp_pose_by_component(self) -> Mapping[str, tuple[float, ...]]:
        """Capture measured TCP state under the application/rig lock."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Agent motion application is closed.")

            def capture(scene: Any) -> Mapping[str, tuple[float, ...]]:
                if not bool(scene.state_is_finite()):
                    raise RuntimeError("MuJoCo scene state is non-finite.")
                poses = capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                    scene,
                    self.task_config,
                )
                vector = pose_matrix_to_xyzquat(poses["right"])
                if vector is None:
                    raise RuntimeError("right_arm TCP pose is not representable.")
                values = tuple(float(item) for item in vector)
                if len(values) != 7 or not all(math.isfinite(item) for item in values):
                    raise RuntimeError("right_arm TCP pose is invalid.")
                return {"right_arm": values}

            return self.runtime.rig.capture_scene(capture)

    def request_hold(
        self,
        *,
        reason: str,
        operation_id: str | None = None,
    ) -> Any | None:
        normalized_reason = _required(reason, field_name="reason")
        with self._lock:
            if not self._started:
                return None
            if operation_id is not None and self._active_operation_id not in {
                None,
                operation_id,
            }:
                raise ValueError("HOLD operation identity mismatch.")
            transition = self.runtime.transition_execution_gate(
                "hold",
                reason=normalized_reason,
                source_event_ids=(
                    ()
                    if operation_id is None
                    else (_required(operation_id, field_name="operation_id"),)
                ),
                force_cancellation=True,
            )
            self._held = True
            return transition

    def release_operation(self, *, operation_id: str) -> None:
        normalized_id = _required(operation_id, field_name="operation_id")
        with self._lock:
            if self._active_operation_id not in {None, normalized_id}:
                raise ValueError("Operation release identity mismatch.")
            self._active_operation_id = None

    def capture_passive_application_cut(
        self,
        *,
        scene_projector: Callable[[Any], Any],
    ) -> MujocoSimHILPassiveApplicationCut:
        if not callable(scene_projector):
            raise TypeError("scene_projector must be callable.")
        with self._lock:
            captured_at = float(self.runtime.clock())
            receiver = self.runtime.core.capture_receiver_state()
            owner_generation = receiver.current_owner_generation
            if owner_generation is None:
                raise RuntimeError("Passive cut requires the lower dispatch owner.")
            if self._closed:
                lifecycle_state = "closed"
                gate_status = "closed"
                gate_epoch = 0
                projection = None
            else:
                lifecycle_state = (
                    "held" if self._held else "running" if self._started else "configured"
                )
                if self._started:
                    gate = self.runtime.capture_execution_gate_snapshot(
                        now=captured_at
                    )
                    gate_status = str(gate.status)
                    gate_epoch = int(gate.gate_epoch)
                else:
                    gate_status = "not_started"
                    gate_epoch = 0
                projection = self.runtime.rig.capture_scene(scene_projector)
            coordinator = self.runtime.core.coordinator
            control_cut = coordinator.capture_cut(
                cutoff_at=float(self.runtime.clock())
            ).cut
            return MujocoSimHILPassiveApplicationCut(
                captured_at=captured_at,
                authority_epoch=int(receiver.current_authority_epoch),
                execution_epoch=int(receiver.current_execution_epoch),
                dispatch_owner_generation=int(owner_generation),
                lifecycle_state=lifecycle_state,
                execution_gate=gate_status,
                execution_gate_epoch=gate_epoch,
                fault_class=None,
                fault_rearmable=False,
                scene_projection=projection,
                control_cut=control_cut,
            )

    def close_after_hold(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._started and not self._held:
                raise RuntimeError("Closing a started motion application requires HOLD.")
            if self._started:
                self.runtime.close()
            self._closed = True
            self._active_operation_id = None


__all__ = [
    "MujocoAgentSkillMotionApplication",
    "MujocoAgentSkillMotionStep",
]
