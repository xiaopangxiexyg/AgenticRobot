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
from robot_motion.target_safety import (
    TCPPoseEnvelopeActivationReceipt,
    TCPPoseEnvelopeLifecycleError,
    TCPPoseEnvelopeSafetyRuntime,
    TCPPoseEnvelopeSelection,
)
from sim.mujoco.deploy.agent_skill_target import MujocoAgentSkillMotionTarget
from sim.mujoco.deploy.target_measurement import (
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
)
from sim.mujoco.rgb_snapshot import (
    EncodedMujocoRgbFrame,
    MujocoRgbSnapshotSource,
)

from ..application import MujocoSimHILPassiveApplicationCut


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


@dataclass(frozen=True)
class MujocoAgentSkillActivation:
    """READY transition plus the lower envelope activation proof."""

    transition: Any
    safety_activation_receipt: TCPPoseEnvelopeActivationReceipt | None

    @property
    def snapshot(self) -> Any:
        return self.transition.snapshot

    @property
    def lower_event_ids(self) -> tuple[str, ...]:
        refs = tuple(getattr(self.transition, "lower_event_ids", ()))
        if self.safety_activation_receipt is None:
            return refs
        return tuple(
            dict.fromkeys(
                (
                    self.safety_activation_receipt.semantic_content_hash,
                    *refs,
                )
            )
        )


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
        safety_runtime = target.tcp_pose_envelope_runtime
        if safety_runtime is not None and not isinstance(
            safety_runtime,
            TCPPoseEnvelopeSafetyRuntime,
        ):
            raise TypeError("target TCP envelope runtime has an invalid type.")
        self.tcp_pose_envelope_runtime = safety_runtime
        self.calibration_hash = _required(
            calibration_hash,
            field_name="calibration_hash",
        )
        self._lock = threading.RLock()
        self._started = False
        self._held = False
        self._closed = False
        self._active_operation_id: str | None = None
        self._visual_snapshot_source = MujocoRgbSnapshotSource(
            capture_scene=self._capture_visual_scene,
        )

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

    def activate_operation(
        self,
        *,
        operation_id: str,
        route_id: str | None = None,
        safety_selection: TCPPoseEnvelopeSelection | None = None,
    ) -> Any:
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
            safety_activation = None
            if self.tcp_pose_envelope_runtime is not None:
                if route_id is None or safety_selection is None:
                    raise TCPPoseEnvelopeLifecycleError(
                        "Agent motion requires an exact lower safety selection"
                    )
                gate = self.runtime.capture_execution_gate_snapshot(
                    now=float(self.runtime.clock())
                )
                activation_boundary_refs: tuple[str, ...] = ()
                if str(gate.status) != "hold":
                    boundary = self.runtime.transition_execution_gate(
                        "hold",
                        reason="agent_skill_safety_profile_activation_boundary",
                        source_event_ids=(normalized_id,),
                        force_cancellation=True,
                    )
                    gate = boundary.snapshot
                    activation_boundary_refs = tuple(
                        getattr(boundary, "lower_event_ids", ())
                    )
                    if (
                        str(gate.status) != "hold"
                        or not activation_boundary_refs
                    ):
                        raise TCPPoseEnvelopeLifecycleError(
                            "lower did not establish HOLD before safety activation"
                        )
                    self._held = True
                if str(gate.status) != "hold":
                    raise TCPPoseEnvelopeLifecycleError(
                        "lower execution gate must be HOLD before safety activation"
                    )
                receiver = self.runtime.authoritative_port.capture_receiver_state()
                execution_epoch = int(receiver.current_execution_epoch)
                safety_activation = self.tcp_pose_envelope_runtime.activate(
                    selection=safety_selection,
                    operation_id=normalized_id,
                    route_id=_required(route_id, field_name="route_id"),
                    execution_epoch=execution_epoch,
                    source_event_ids=tuple(
                        dict.fromkeys(
                            (
                                normalized_id,
                                f"lower-execution-epoch:{execution_epoch}",
                                *activation_boundary_refs,
                            )
                        )
                    ),
                    lower_state="hold",
                )
            try:
                transition = self.runtime.transition_execution_gate(
                    "ready",
                    reason="agent_skill_operation_activation",
                    source_event_ids=(
                        normalized_id,
                        *(
                            ()
                            if safety_activation is None
                            else (safety_activation.semantic_content_hash,)
                        ),
                    ),
                )
            except Exception:
                if safety_activation is not None:
                    receiver = self.runtime.authoritative_port.capture_receiver_state()
                    self.tcp_pose_envelope_runtime.deactivate_after_hold(
                        operation_id=normalized_id,
                        next_execution_epoch=int(receiver.current_execution_epoch),
                        hold_evidence_ids=(
                            "agent-skill-ready-transition-failed",
                        ),
                    )
                raise
            self._active_operation_id = normalized_id
            self._held = False
            if safety_activation is None:
                return transition
            return MujocoAgentSkillActivation(
                transition=transition,
                safety_activation_receipt=safety_activation,
            )

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
            profile = self.runtime.authoritative_port.profile
            # A direct proposal is not visible to a control cut captured at
            # the timestamp sampled immediately before its admission.  The
            # immediate-reference schedule has zero configured target lead,
            # so submitting and stepping at ``now`` deterministically omits
            # the just-admitted proposal and produces ``no_action``.  Give
            # that proposal one decision period to cross the admission cut;
            # scheduled-reference profiles already provide a larger lead and
            # retain their exact configured timing.
            target_at = (
                now
                + max(
                    float(profile.target_lead_time_sec),
                    float(profile.decision_period_sec),
                )
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

    def visual_source_catalog(self) -> tuple[Mapping[str, object], ...]:
        """Describe every current RGB source without rendering a frame."""

        return self._visual_snapshot_source.source_catalog()

    def capture_visual_jpeg_frames(
        self,
        *,
        source_ids: tuple[str, ...],
        width: int,
        height: int,
    ) -> Mapping[str, EncodedMujocoRgbFrame]:
        """Capture validated JPEG frames through the lower MuJoCo source."""

        return self._visual_snapshot_source.capture_jpeg_frames(
            source_ids=source_ids,
            width=width,
            height=height,
        )

    def _capture_visual_scene(self, capture: Callable[[Any], Any]) -> Any:
        """Keep renderer access serialized with motion and lifecycle changes."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Agent motion application is closed.")
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
            if self.tcp_pose_envelope_runtime is not None:
                receiver = self.runtime.authoritative_port.capture_receiver_state()
                hold_refs = tuple(getattr(transition, "lower_event_ids", ()))
                if not hold_refs:
                    raise TCPPoseEnvelopeLifecycleError(
                        "lower HOLD did not provide envelope deactivation evidence"
                    )
                self.tcp_pose_envelope_runtime.deactivate_after_hold(
                    operation_id=operation_id,
                    next_execution_epoch=int(receiver.current_execution_epoch),
                    hold_evidence_ids=hold_refs,
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
            receiver = self.runtime.authoritative_port.capture_receiver_state()
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
            control_cut = self.runtime.capture_control_cut(
                cutoff_at=float(self.runtime.clock())
            )
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
        visual_source: MujocoRgbSnapshotSource | None = None
        with self._lock:
            if self._closed:
                return
            if self._started and not self._held:
                raise RuntimeError("Closing a started motion application requires HOLD.")
            if self._started:
                self.runtime.close()
            visual_source = getattr(self, "_visual_snapshot_source", None)
            self._closed = True
            self._active_operation_id = None
        # Do not invert the source-lock -> application-lock order used during
        # capture.  Once _closed is visible, no new scene capture can begin.
        if visual_source is not None:
            visual_source.close()


__all__ = [
    "MujocoAgentSkillActivation",
    "MujocoAgentSkillMotionApplication",
    "MujocoAgentSkillMotionStep",
]
