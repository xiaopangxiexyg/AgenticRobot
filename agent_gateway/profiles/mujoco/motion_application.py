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

from control_station.sim_hil_profile.application import (
    MujocoSimHILPassiveApplicationCut,
)
from robot_motion.action import ComponentAction
from robot_motion.deploy.schema import pose_matrix_to_xyzquat, xyzquat_to_pose_matrix
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
from sim.mujoco.depth_snapshot import (
    EncodedMujocoDepthFrame,
    MujocoDepthSnapshotSource,
)
from sim.mujoco.rgb_snapshot import EncodedMujocoRgbSnapshot, MujocoRgbSnapshotSource

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
            != ("right_arm", "right_gripper")
        ):
            raise ValueError(
                "Motion application requires the exact right-arm/right-gripper "
                "Agent Skill runtime."
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
        self._active_tcp_safety_operation_id: str | None = None
        self._visual_snapshot_source = MujocoRgbSnapshotSource(
            capture_scene=self._capture_visual_scene,
            clock=self.runtime.clock,
        )
        self._depth_snapshot_source = MujocoDepthSnapshotSource(
            capture_scene=self._capture_visual_scene,
            clock=self.runtime.clock,
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

    def start_observation_ready(self) -> Any:
        """Start MuJoCo and prove HOLD before exposing passive reads.

        Construction remains default-off.  The dedicated lower launcher calls
        this explicit lifecycle boundary before it exposes the Gateway, so
        observation and optional history capture see one running lower runtime
        while motion admission remains closed.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("Agent motion application is closed.")
            if self._active_operation_id is not None:
                raise RuntimeError(
                    "Observation-ready startup requires no active operation."
                )
            if not self._started:
                self.runtime.start()
                self._started = True
            transition = self.runtime.transition_execution_gate(
                "hold",
                reason="agent_gateway_observation_ready_startup",
                source_event_ids=("agent-gateway-observation-ready",),
                force_cancellation=True,
            )
            lower_event_ids = tuple(getattr(transition, "lower_event_ids", ()))
            if (
                str(getattr(transition.snapshot, "status", "")) != "hold"
                or not lower_event_ids
            ):
                raise RuntimeError(
                    "Lower runtime did not prove HOLD during Gateway startup."
                )
            self._held = True
            return transition

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
            self._active_tcp_safety_operation_id = (
                normalized_id if safety_activation is not None else None
            )
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

    def _capture_arm_gripper_state_from_scene(
        self,
        scene: Any,
    ) -> Mapping[str, object]:
        if not bool(scene.state_is_finite()):
            raise RuntimeError("MuJoCo scene state is non-finite.")
        poses = capture_mujoco_sim_hil_tool_tcp_pose_by_side(
            scene,
            self.task_config,
        )
        vector = pose_matrix_to_xyzquat(poses["right"])
        if vector is None:
            raise RuntimeError("right_arm TCP pose is not representable.")
        arm = tuple(float(item) for item in vector)
        if len(arm) != 7 or not all(math.isfinite(item) for item in arm):
            raise RuntimeError("right_arm TCP pose is invalid.")
        name = str(self.task_config.right_gripper_name)
        qpos = tuple(float(value) for value in scene.get_gripper_qpos(name))
        close = float(scene.get_gripper_close(name))
        if not qpos or not all(math.isfinite(value) for value in qpos):
            raise RuntimeError("right_gripper qpos is invalid.")
        if not math.isfinite(close) or not 0.0 <= close <= 1.0:
            raise RuntimeError("right_gripper close fraction is invalid.")
        return {
            "right_arm": arm,
            "right_gripper": {
                "qpos": qpos,
                "close_fraction": close,
                "measured_at": float(self.runtime.clock()),
                "state_source": "mujoco_measured_joint_state",
            },
        }

    def capture_tcp_pose_by_component(self) -> Mapping[str, tuple[float, ...]]:
        """Capture measured TCP state under the application/rig lock."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Agent motion application is closed.")
            captured = self.runtime.rig.capture_scene(
                self._capture_arm_gripper_state_from_scene
            )
            return {"right_arm": tuple(captured["right_arm"])}

    def capture_gripper_state_by_component(self) -> Mapping[str, Mapping[str, object]]:
        """Capture physical gripper qpos and normalized close under the rig lock."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Agent motion application is closed.")
            captured = self.runtime.rig.capture_scene(
                self._capture_arm_gripper_state_from_scene
            )
            return {"right_gripper": dict(captured["right_gripper"])}

    def gripper_qpos_for_close(self, close_fraction: float) -> tuple[float, ...]:
        """Resolve a normalized endpoint through the bound scene geometry."""

        close = float(close_fraction)
        if not math.isfinite(close) or not 0.0 <= close <= 1.0:
            raise ValueError("close_fraction must lie in [0, 1].")
        with self._lock:
            if self._closed:
                raise RuntimeError("Agent motion application is closed.")

            def capture(scene: Any) -> tuple[float, ...]:
                values = tuple(
                    float(value)
                    for value in scene.gripper_qpos_from_close(
                        str(self.task_config.right_gripper_name),
                        close,
                    )
                )
                if not values or not all(math.isfinite(value) for value in values):
                    raise RuntimeError("Resolved right_gripper qpos is invalid.")
                return values

            return self.runtime.rig.capture_scene(capture)

    def complete_component_action(
        self,
        values_by_component: Mapping[str, object],
    ) -> ComponentAction:
        """Fill unchanged support components from one application-locked cut."""

        provided = dict(values_by_component)
        runtime_components = tuple(self.runtime.action_schema.component_ids)
        unknown = set(provided) - set(runtime_components)
        if unknown:
            raise ValueError(f"Unknown Agent component values: {sorted(unknown)}.")
        if "right_gripper" not in runtime_components:
            return ComponentAction(
                action_schema=self.runtime.action_schema,
                values_by_component=provided,
            )
        with self._lock:
            captured = self.runtime.rig.capture_scene(
                self._capture_arm_gripper_state_from_scene
            )
            gripper = captured["right_gripper"]
            complete = {
                "right_arm": captured["right_arm"],
                "right_gripper": gripper["qpos"],
                **provided,
            }
            return ComponentAction(
                action_schema=self.runtime.action_schema,
                values_by_component=complete,
            )

    def transform_tcp_pose_xyzquat(
        self,
        pose: object,
        *,
        source_frame_id: str,
        target_frame_id: str,
    ) -> tuple[float, ...]:
        """Transform one tool pose through the deployment-bound frame graph.

        MuJoCo's lower action and FK owners keep using their component arm-base
        frame.  Agent-visible state and absolute targets use the canonical
        Station frame, so the concrete application profile is the sole seam
        that may translate between those two identities.
        """

        source = _required(source_frame_id, field_name="source_frame_id")
        target = _required(target_frame_id, field_name="target_frame_id")
        matrix = xyzquat_to_pose_matrix(pose)
        if matrix is None:
            raise ValueError("TCP pose must be finite normalized xyz+quat_xyzw.")
        if source == target:
            vector = pose_matrix_to_xyzquat(matrix)
            if vector is None:  # pragma: no cover - validated above
                raise RuntimeError("TCP pose is not representable.")
            return tuple(float(item) for item in vector)
        transform = self.target_frame_profile.frame_graph.resolve(source, target)
        transformed = transform.matrix_target_source @ matrix
        vector = pose_matrix_to_xyzquat(transformed)
        if vector is None:
            raise RuntimeError("Transformed TCP pose is not representable.")
        values = tuple(float(item) for item in vector)
        if len(values) != 7 or not all(math.isfinite(item) for item in values):
            raise RuntimeError("Transformed TCP pose is invalid.")
        return values

    def visual_source_catalog(self) -> tuple[Mapping[str, object], ...]:
        """Describe every current RGB source without rendering a frame."""

        return self._visual_snapshot_source.source_catalog()

    def capture_visual_jpeg_snapshot(
        self,
        *,
        source_ids: tuple[str, ...],
        width: int,
        height: int,
    ) -> EncodedMujocoRgbSnapshot:
        """Capture one timestamped RGB scene cut through the lower source."""

        return self._visual_snapshot_source.capture_jpeg_snapshot(
            source_ids=source_ids,
            width=width,
            height=height,
        )

    def depth_source_catalog(self) -> tuple[Mapping[str, object], ...]:
        """Describe deployment-registered native depth views."""

        return self._depth_snapshot_source.source_catalog()

    def capture_depth_npy_frame(
        self,
        *,
        view_id: str,
    ) -> EncodedMujocoDepthFrame:
        """Capture one passive native-resolution metric depth Artifact."""

        return self._depth_snapshot_source.capture_npy_frame(view_id=view_id)

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
            if self._active_tcp_safety_operation_id is not None:
                safety_operation_id = self._active_tcp_safety_operation_id
                if operation_id is not None and operation_id != safety_operation_id:
                    raise TCPPoseEnvelopeLifecycleError(
                        "TCP safety operation identity mismatch during HOLD"
                    )
                receiver = self.runtime.authoritative_port.capture_receiver_state()
                hold_refs = tuple(getattr(transition, "lower_event_ids", ()))
                if not hold_refs:
                    raise TCPPoseEnvelopeLifecycleError(
                        "lower HOLD did not provide envelope deactivation evidence"
                    )
                self.tcp_pose_envelope_runtime.deactivate_after_hold(
                    operation_id=safety_operation_id,
                    next_execution_epoch=int(receiver.current_execution_epoch),
                    hold_evidence_ids=hold_refs,
                )
                self._active_tcp_safety_operation_id = None
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
        snapshot_sources: tuple[object, ...] = ()
        with self._lock:
            if self._closed:
                return
            if self._started and not self._held:
                raise RuntimeError("Closing a started motion application requires HOLD.")
            if self._started:
                self.runtime.close()
            snapshot_sources = tuple(
                source
                for source in (
                    getattr(self, "_visual_snapshot_source", None),
                    getattr(self, "_depth_snapshot_source", None),
                )
                if source is not None
            )
            self._closed = True
            self._active_operation_id = None
        # Do not invert the source-lock -> application-lock order used during
        # capture.  Once _closed is visible, no new scene capture can begin.
        first_error: Exception | None = None
        for source in snapshot_sources:
            try:
                source.close()
            except Exception as exc:  # pragma: no cover - backend cleanup
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


__all__ = [
    "MujocoAgentSkillActivation",
    "MujocoAgentSkillMotionApplication",
    "MujocoAgentSkillMotionStep",
]
