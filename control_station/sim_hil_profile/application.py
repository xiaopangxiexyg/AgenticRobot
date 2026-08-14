"""Lifecycle owner for one concrete default-off MuJoCo Sim-HIL application.

The class coordinates already-constructed lower-owned ports.  It does not
construct devices, policies, MuJoCo targets, composition algorithms, safety,
IK, dispatch, or dataset writers.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from robot_motion.runtime.control_admission import ControlCut
from control_composition.activation import CompositionActivationScope
from robot_motion.deploy import (
    PolicyActionEmission,
    PolicyExecutionFeedbackApplication,
    project_policy_execution_feedback,
)

from .application_human import _SimHILApplicationHumanMixin
from .application_status import _SimHILApplicationStatusMixin
from .application_transition import _SimHILApplicationTransitionMixin
from control_station.hitl import (
    HITLActivationPrimeEvidence,
    build_hitl_activation_prime_evidence,
)
from .config import (
    SIM_HIL_BLEND_FAILURE_FALLBACKS,
    SIM_HIL_HUMAN_TIMEOUT_FALLBACKS,
    MujocoSimHILApplicationConfig,
)
from .control_cycle import (
    SIM_HIL_ACTIVATION_PRIMING_APPLICATION_COMPATIBILITY,
    SIM_HIL_ACTIVATION_PRIMING_CONTROL_CYCLE_PORT,
    SIM_HIL_POLICY_SOURCE_ALWAYS,
    SIM_HIL_POLICY_SOURCE_CONTROL_CYCLE,
    FixedAlphaCompatibilityControlCyclePort,
    SimHILControlCyclePort,
    SimHILControlCyclePortSpec,
)
from .human import _SimHILHumanAttachment
from .policy_feedback import project_sim_hil_policy_execution_feedback
from .policy_source import SimHILPolicyExecutionFeedbackPort
from .profile_selection import SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL
from .validation import _finite, _required


def _advance_control_deadline(
    *,
    previous_deadline: float,
    completed_at: float,
    period: float,
) -> tuple[float, int]:
    """Advance one periodic Station deadline without burst catch-up."""

    previous = _finite(previous_deadline, name="previous_deadline")
    completed = _finite(completed_at, name="completed_at")
    cadence = _finite(period, name="period")
    if cadence <= 0.0:
        raise ValueError("period must be positive.")
    candidate = previous + cadence
    skipped = 0
    if candidate <= completed:
        skipped = int(math.floor((completed - candidate) / cadence)) + 1
        candidate += skipped * cadence
    return candidate, skipped


@dataclass(frozen=True)
class MujocoSimHILPassiveApplicationCut:
    """One application-locked planning cut with no execution side effect.

    ``scene_projection`` is produced under the same application/rig lock as
    the lifecycle and lower control facts.  The cut is suitable for a
    low-rate planning projection, never for pre-dispatch safety.
    """

    captured_at: float
    authority_epoch: int
    execution_epoch: int
    dispatch_owner_generation: int
    lifecycle_state: str
    execution_gate: str
    execution_gate_epoch: int
    fault_class: str | None
    fault_rearmable: bool
    scene_projection: Any | None
    control_cut: ControlCut | None = None


class MujocoSimHILApplication(
    _SimHILApplicationHumanMixin,
    _SimHILApplicationTransitionMixin,
    _SimHILApplicationStatusMixin,
):
    """Own only the lifecycle of one Station-selected Sim-HIL composition."""

    def __init__(
        self,
        *,
        config: MujocoSimHILApplicationConfig,
        task_config: Any,
        scene: Any,
        runtime: Any,
        composition: Any,
        policy_source: Any,
        human_source: Any,
        readiness_provider: Any,
        control_cycle_port: SimHILControlCyclePort | None = None,
        execution_difference_observer: Any | None = None,
        target_frame_profile: Any | None = None,
        tool_frame_profile: Any | None = None,
        gripper_input_selection: Any | None = None,
        calibration_binding: dict[str, Any] | None = None,
        task_initialization_info: dict[str, object] | None = None,
        viewer: Any | None = None,
        runtime_home_pose_desk_by_side_capture: (
            Callable[[], Mapping[str, Any]] | None
        ) = None,
        human_attachment_factory: (
            Callable[[Path, str], _SimHILHumanAttachment] | None
        ) = None,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self.config = config
        self.task_config = task_config
        self.scene = scene
        self.runtime = runtime
        self.composition = composition
        self.policy_source = policy_source
        self.human_source = human_source
        self.readiness_provider = readiness_provider
        cycle_port = (
            FixedAlphaCompatibilityControlCyclePort(composition)
            if control_cycle_port is None
            else control_cycle_port
        )
        if not isinstance(
            getattr(cycle_port, "spec", None),
            SimHILControlCyclePortSpec,
        ):
            raise TypeError("control_cycle_port must expose a typed port spec.")
        for method_name in ("activate", "deactivate", "run_cycle", "close"):
            if not callable(getattr(cycle_port, method_name, None)):
                raise TypeError(
                    "control_cycle_port is missing required method " f"{method_name}()."
                )
        if (
            cycle_port.spec.activation_priming_owner
            == SIM_HIL_ACTIVATION_PRIMING_CONTROL_CYCLE_PORT
            and not callable(getattr(cycle_port, "prime_activation", None))
        ):
            raise TypeError(
                "Port-owned activation priming requires prime_activation()."
            )
        self.control_cycle_port = cycle_port
        self.execution_difference_observer = execution_difference_observer
        # Immutable lower-owned geometry identities.  They are optional for
        # compatibility fixtures, but production Agent Skill binding requires
        # both and rejects a graph that cannot prove its exact frames/tools.
        self.target_frame_profile = target_frame_profile
        self.tool_frame_profile = tool_frame_profile
        self.gripper_input_selection = gripper_input_selection
        self.calibration_binding = dict(calibration_binding or {})
        self.task_initialization_info = dict(task_initialization_info or {})
        self.viewer = viewer
        self._runtime_home_pose_desk_by_side_capture = (
            runtime_home_pose_desk_by_side_capture
        )
        self._human_attachment_factory = human_attachment_factory
        self.clock = clock
        self.sleep = sleep
        self._lock = threading.RLock()
        self._started = False
        self._sources_started = False
        self._last_preflight: Any | None = None
        self._last_cycle: Any | None = None
        self._policy_emission_by_id: OrderedDict[
            str, PolicyActionEmission
        ] = OrderedDict()
        self._last_policy_feedback_status: dict[str, object] | None = None
        self._last_human_input_check_step: Any | None = None
        self._human_input_check_report: dict[str, object] | None = None
        self._human_input_check_active = False
        self._runtime_home_capture: dict[str, object] | None = None
        self._last_passive_observation_error: str | None = None
        self._last_error: str | None = None
        self._first_control_fault: str | None = None
        self._held = False
        self._episode_ended = False
        self._episode_end_reason: str | None = None
        self._last_sim_contact_observation: dict[str, object] | None = None
        self._last_hold_outcome: Any | None = None
        self._last_activation_prime_evidence: (
            HITLActivationPrimeEvidence | None
        ) = None
        self._next_decision_at: float | None = None
        self._overrun_dropped_cycles = 0
        self._last_schedule_overrun: dict[str, object] | None = None
        self._last_cycle_timing: dict[str, float] | None = None
        self._blend_failure_fallback = config.blend_failure_fallback
        self._blend_failure_grace_sec = config.blend_failure_grace_sec
        self._blend_failure_window_started_at: float | None = None
        self._blend_failure_consecutive_count = 0
        self._blend_failure_fallback_latched_source: str | None = None
        self._last_blend_failure_fallback: dict[str, object] | None = None
        self._human_timeout_fallback = config.human_timeout_fallback
        self._human_timeout_fallback_latched = False
        self._last_human_timeout_fallback: dict[str, object] | None = None
        self._vive_source_swap_count = 0
        self._startup_failure: str | None = None
        self._restart_required = False
        self._closed = False
        self._viewer_exit_requested = False
        # A demonstration session may open the passive viewer before any
        # Human device, source, composition, control cycle, or trace episode
        # is activated.  This flag is deliberately application-local: the
        # viewer remains observational and owns no execution authority.
        self._viewer_preview_started = False

    @property
    def human_input_attached(self) -> bool:
        return bool(getattr(self.human_source, "attached", True))

    @property
    def viewer_exit_requested(self) -> bool:
        with self._lock:
            return bool(self._viewer_exit_requested)

    @property
    def viewer_preview_started(self) -> bool:
        with self._lock:
            return bool(self._viewer_preview_started and not self._started)

    @property
    def started(self) -> bool:
        return self._started

    @property
    def held(self) -> bool:
        with self._lock:
            return bool(self._held)

    @property
    def next_decision_at(self) -> float | None:
        """Read-only Station pacing cut; it grants no execution authority."""

        with self._lock:
            return self._next_decision_at

    @property
    def last_hold_outcome(self) -> Any | None:
        """Latest lower-owned HOLD outcome; read-only lifecycle evidence."""

        with self._lock:
            return self._last_hold_outcome

    @property
    def last_activation_prime_evidence(
        self,
    ) -> HITLActivationPrimeEvidence | None:
        """Fresh source identities admitted after the latest activation."""

        with self._lock:
            return self._last_activation_prime_evidence

    @property
    def episode_ended(self) -> bool:
        with self._lock:
            return bool(self._episode_ended)

    @property
    def restart_required(self) -> bool:
        return self._restart_required

    @property
    def closed(self) -> bool:
        return self._closed

    def capture_passive_application_cut(
        self,
        *,
        scene_projector: Callable[[Any], Any],
    ) -> MujocoSimHILPassiveApplicationCut:
        """Capture lifecycle, lower epochs/gate, and a passive scene view.

        This method neither starts the application nor advances physics.  The
        application lock prevents a Station control cycle or lifecycle call
        from interleaving with the scene projection.  A closed or
        restart-required application intentionally returns no scene view.
        """

        if not callable(scene_projector):
            raise TypeError("scene_projector must be callable.")
        with self._lock:
            captured_at = _finite(
                self.runtime.clock(),
                name="passive_application_cut.captured_at",
            )
            receiver = self.runtime.core.capture_receiver_state()
            runtime_config = self.runtime.config
            if (
                receiver.runtime_session_id != runtime_config.runtime_session_id
                or receiver.control_domain_id != runtime_config.control_domain_id
            ):
                raise RuntimeError(
                    "Passive application cut crossed runtime identity."
                )
            owner_generation = receiver.current_owner_generation
            if owner_generation is None:
                raise RuntimeError(
                    "Passive application cut requires the existing dispatch owner."
                )

            if self._restart_required:
                lifecycle_state = "restart_required"
                execution_gate = "fault"
                gate_epoch = 0
                fault_class = "sim_hil_restart_required"
            elif self._closed:
                lifecycle_state = "closed"
                execution_gate = "closed"
                gate_epoch = 0
                fault_class = None
            elif self._episode_ended:
                lifecycle_state = "episode_ended"
                gate = self.runtime.capture_execution_gate_snapshot(
                    now=captured_at
                )
                execution_gate = str(gate.status)
                gate_epoch = int(gate.gate_epoch)
                fault_class = None
            elif self._held:
                lifecycle_state = "held"
                gate = self.runtime.capture_execution_gate_snapshot(
                    now=captured_at
                )
                execution_gate = str(gate.status)
                gate_epoch = int(gate.gate_epoch)
                fault_class = (
                    None
                    if self._first_control_fault is None
                    else "sim_hil_control_fault"
                )
            elif self._started:
                lifecycle_state = "running"
                gate = self.runtime.capture_execution_gate_snapshot(
                    now=captured_at
                )
                execution_gate = str(gate.status)
                gate_epoch = int(gate.gate_epoch)
                fault_class = None
            elif self._viewer_preview_started:
                lifecycle_state = "viewer_preview"
                execution_gate = "not_started"
                gate_epoch = 0
                fault_class = None
            else:
                lifecycle_state = "configured"
                execution_gate = "not_started"
                gate_epoch = 0
                fault_class = None

            scene_projection = (
                None
                if self._closed or self._restart_required
                else self.runtime.rig.capture_scene(scene_projector)
            )
            coordinator = getattr(self.runtime.core, "coordinator", None)
            capture_control_cut = getattr(coordinator, "capture_cut", None)
            control_cut = (
                None
                if not callable(capture_control_cut)
                else capture_control_cut(
                    cutoff_at=_finite(
                        self.runtime.clock(),
                        name="passive_application_cut.cutoff_at",
                    )
                ).cut
            )
            if control_cut is not None and (
                control_cut.runtime_session_id != runtime_config.runtime_session_id
                or control_cut.control_domain_id != runtime_config.control_domain_id
                or control_cut.cutoff_at < captured_at
            ):
                raise RuntimeError(
                    "Passive application control cut crossed its measurement."
                )
            return MujocoSimHILPassiveApplicationCut(
                captured_at=captured_at,
                authority_epoch=int(receiver.current_authority_epoch),
                execution_epoch=int(receiver.current_execution_epoch),
                dispatch_owner_generation=int(owner_generation),
                lifecycle_state=lifecycle_state,
                execution_gate=execution_gate,
                execution_gate_epoch=gate_epoch,
                fault_class=fault_class,
                fault_rearmable=False,
                scene_projection=scene_projection,
                control_cut=control_cut,
            )

    @staticmethod
    def _raise_if_cancelled(
        cancel_requested: Callable[[], bool] | None,
        *,
        phase: str,
    ) -> None:
        if cancel_requested is not None and bool(cancel_requested()):
            raise InterruptedError(f"Sim-HIL startup cancelled during {phase}.")

    def _policy_contributor_active(self) -> bool:
        """Return whether the current composition needs policy values.

        A human-only acceptance stage must not synchronously run a potentially
        cold PPO inference just to discover that its result is unused.  The
        source is still started during application startup so a later stage
        can enable it without rebuilding the application.
        """

        cycle_port = getattr(self, "control_cycle_port", None)
        spec = None if cycle_port is None else cycle_port.spec
        if (
            spec is not None
            and spec.policy_source_requirement
            == SIM_HIL_POLICY_SOURCE_CONTROL_CYCLE
        ):
            required = getattr(cycle_port, "policy_emission_required", None)
            if not callable(required):
                raise TypeError(
                    "Control-cycle-owned policy selection requires "
                    "policy_emission_required()."
                )
            value = required()
            if not isinstance(value, bool):
                raise TypeError("policy_emission_required() must return bool.")
            return value
        if (
            spec is not None
            and spec.policy_source_requirement == SIM_HIL_POLICY_SOURCE_ALWAYS
        ):
            return True
        values = getattr(self.composition, "human_alpha_by_component", None)
        if values is None:
            # Keep lightweight/fake application bindings conservative: older
            # callers without the typed alpha view retain the old behavior.
            return True
        return any(float(value) < 1.0 - 1e-12 for value in dict(values).values())

    def _warmup_sources(
        self,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._raise_if_cancelled(
            cancel_requested,
            phase="source_start",
        )
        self.policy_source.start()
        self.human_source.start()
        # The application now owns a live lower source even if the following
        # calibration transition raises.
        self._sources_started = True
        self.human_source.set_calibration_ready()
        self._wait_for_source_readiness(
            cancel_requested=cancel_requested,
            phase="source_warmup",
        )

    def _wait_for_source_readiness(
        self,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        phase: str,
    ) -> None:
        """Acquire genuinely current contributor emissions and verify them."""

        deadline = self.clock() + self.config.warmup_timeout_sec
        while True:
            self._raise_if_cancelled(
                cancel_requested,
                phase=phase,
            )
            now = _finite(self.clock(), name="warmup.now")
            if self._policy_contributor_active():
                self.policy_source.step(now=now)
            self.human_source.step(now=now)
            # Source emissions timestamp themselves with the local monotonic
            # clock during ``step``.  Observe after both calls so a freshly
            # produced emission is not compared against the pre-step cut and
            # reported as ``emission_in_future``.
            observed_at = max(
                now,
                _finite(self.clock(), name="warmup.observed_at"),
            )
            evidence = self.readiness_provider.snapshot(now=observed_at)
            readiness_failures = self.composition.readiness_failures(
                evidence,
                target_effective_at=observed_at,
            )
            if not readiness_failures:
                return
            if observed_at >= deadline:
                raise TimeoutError(
                    f"Sim-HIL source readiness timed out during {phase}: "
                    + ";".join(
                        f"{name}={evidence.detail_by_check.get(name, '')}"
                        for name in readiness_failures
                    )
                )
            self.sleep(0.005)

    def _prime_activation_inputs(
        self,
    ) -> HITLActivationPrimeEvidence | None:
        """Admit exactly the fresh contributors required by this activation."""

        requested_at = _finite(self.clock(), name="activation.requested_at")
        alpha = dict(self.composition.human_alpha_by_component)
        human_active = any(value > 1e-12 for value in alpha.values())
        policy_active = any(value < 1.0 - 1e-12 for value in alpha.values())
        policy_reset_at = None
        if policy_active:
            self.policy_source.reset_for_activation()
            policy_reset_at = max(
                requested_at,
                _finite(self.clock(), name="activation.policy_reset_at"),
            )
        human_step, human_emission = self._acquire_fresh_human_activation_emission(
            requested_at=requested_at,
            required=human_active,
        )
        # A required asynchronous human source may take longer than one policy
        # validity interval to produce its first activation-local emission.
        # Produce policy only after that wait so its immutable interval is not
        # already expired when the receiver admits the contributor pair.
        policy_emission = None
        if policy_active:
            policy_requested_at = max(
                requested_at,
                _finite(self.clock(), name="activation.policy_requested_at"),
            )
            policy_emission = self._acquire_fresh_policy_activation_emission(
                requested_at=policy_requested_at,
            )
        observed_at = max(
            requested_at,
            _finite(self.runtime.clock(), name="activation.observed_at"),
            _finite(self.clock(), name="activation.station_observed_at"),
        )
        if policy_emission is None and policy_active:
            raise RuntimeError("Policy source produced no fresh activation emission.")
        self._remember_policy_emission(policy_emission)
        policy_submission = None
        if policy_emission is not None:
            policy_submission = self.composition.submit_policy_emission(
                policy_emission,
                observed_at=observed_at,
            )
            if not policy_submission.accepted and policy_active:
                raise RuntimeError(
                    "Fresh policy activation emission was rejected: "
                    f"{policy_submission.error}"
                )
        if human_emission is None and human_active:
            raise RuntimeError(
                "Human source produced no fresh activation emission within "
                f"{self.config.human_emission_timeout_sec:.3f}s."
            )
        human_submission = None
        if human_emission is not None:
            human_submission = self.composition.submit_human_emission(
                human_emission,
                observed_at=observed_at,
            )
            if human_active and (
                human_submission is None or not human_submission.accepted
            ):
                raise RuntimeError(
                    "Fresh human activation emission was rejected: "
                    f"{None if human_submission is None else human_submission.error}"
                )
        scope = getattr(self.composition, "activation_scope", None)
        if not isinstance(scope, CompositionActivationScope):
            return None
        return build_hitl_activation_prime_evidence(
            activation_scope=scope,
            primed_at=max(
                observed_at,
                _finite(self.clock(), name="activation.primed_at"),
            ),
            clock_domain_id=scope.clock_domain_id,
            policy_required=policy_active,
            human_required=human_active,
            policy_reset_performed=policy_active,
            policy_reset_at=policy_reset_at,
            policy_emission=policy_emission,
            human_emission=human_emission,
            policy_submission=policy_submission,
            human_submission=human_submission,
            strict_source_identity=(
                self.config.control_profile_id
                == SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL
            ),
        )

    def _acquire_fresh_human_activation_emission(
        self,
        *,
        requested_at: float,
        required: bool,
    ) -> tuple[Any, Any | None]:
        """Wait only for an activation-local emission, never a cached pose."""

        human_step = self.human_source.step(now=requested_at)
        human_emission = getattr(human_step, "emission", None)
        deadline = requested_at + self.config.human_emission_timeout_sec
        while required and human_emission is None:
            if _finite(self.clock(), name="activation.wait_now") >= deadline:
                break
            self.sleep(0.005)
            poll_at = _finite(self.clock(), name="activation.poll_at")
            human_step = self.human_source.step(now=poll_at)
            human_emission = getattr(human_step, "emission", None)
        return human_step, human_emission

    def _acquire_fresh_policy_activation_emission(
        self,
        *,
        requested_at: float,
    ) -> Any:
        """Wait for one activation-local emission from sync or threaded sources."""

        deadline = requested_at + self.config.warmup_timeout_sec
        emission = self.policy_source.step(now=requested_at)
        while emission is None:
            now = _finite(self.clock(), name="activation.policy_wait_now")
            if now >= deadline:
                detail = str(
                    getattr(self.policy_source, "last_error", None)
                    or "no policy emission"
                )
                raise TimeoutError(
                    "Policy source produced no fresh activation emission within "
                    f"{self.config.warmup_timeout_sec:.3f}s: {detail}"
                )
            self.sleep(0.005)
            emission = self.policy_source.step(
                now=_finite(
                    self.clock(),
                    name="activation.policy_poll_at",
                )
            )
        return emission

    def _activate_and_prime_control_cycle(
        self,
        *,
        human_emission_required: bool | None = None,
    ) -> None:
        """Prime through the selected profile without sharing source semantics."""

        self._policy_emission_by_id.clear()
        self._last_policy_feedback_status = None
        self._last_activation_prime_evidence = None
        activation_scope = getattr(
            self.composition,
            "activation_scope",
            None,
        )
        owner = self.control_cycle_port.spec.activation_priming_owner
        if owner == SIM_HIL_ACTIVATION_PRIMING_APPLICATION_COMPATIBILITY:
            # Preserve the established fixed-alpha priming owner and exact
            # submission path. New profiles must not inherit its alpha-derived
            # roles.
            self._last_activation_prime_evidence = (
                self._prime_activation_inputs()
            )
            self.control_cycle_port.activate(
                activation_scope=activation_scope,
            )
            return
        if owner != SIM_HIL_ACTIVATION_PRIMING_CONTROL_CYCLE_PORT:
            raise RuntimeError("Unknown control-cycle activation priming owner.")
        self.control_cycle_port.activate(
            activation_scope=activation_scope,
        )
        requested_at = _finite(
            self.clock(),
            name="control_cycle_prime.requested_at",
        )
        policy_active = self._policy_contributor_active()
        policy_reset_at = None
        if policy_active:
            self.policy_source.reset_for_activation()
            policy_reset_at = max(
                requested_at,
                _finite(
                    self.clock(),
                    name="control_cycle_prime.policy_reset_at",
                ),
            )
        human_required = bool(
            self.control_cycle_port.spec.activation_human_emission_required
            if human_emission_required is None
            else human_emission_required
        )
        _human_step, human_emission = (
            self._acquire_fresh_human_activation_emission(
                requested_at=requested_at,
                required=human_required,
            )
        )
        if human_emission is None and human_required:
            raise RuntimeError(
                "Human source produced no fresh activation emission within "
                f"{self.config.human_emission_timeout_sec:.3f}s."
            )
        # Keep the first policy interval causal to the completed human wait.
        # The activation-local human emission still precedes policy inference,
        # while no composition proposal or lower execution can run here.
        policy_emission = None
        if policy_active:
            policy_requested_at = max(
                requested_at,
                _finite(
                    self.clock(),
                    name="control_cycle_prime.policy_requested_at",
                ),
            )
            policy_emission = self._acquire_fresh_policy_activation_emission(
                requested_at=policy_requested_at,
            )
        self._remember_policy_emission(policy_emission)
        observed_at = max(
            requested_at,
            _finite(
                self.runtime.clock(),
                name="control_cycle_prime.runtime_observed_at",
            ),
            _finite(
                self.clock(),
                name="control_cycle_prime.station_observed_at",
            ),
        )
        activation_spec = self.runtime.reference_activation_spec
        target_effective_at = (
            observed_at
            if activation_spec.activation_mode == "immediate_target"
            else observed_at + activation_spec.target_lead_time_sec
        )
        prime = self.control_cycle_port.prime_activation(
            now=observed_at,
            policy_emission=policy_emission,
            human_emission=human_emission,
            poll_human=False,
            target_effective_at=target_effective_at,
        )
        if isinstance(activation_scope, CompositionActivationScope):
            self._last_activation_prime_evidence = (
                build_hitl_activation_prime_evidence(
                    activation_scope=activation_scope,
                    primed_at=max(
                        observed_at,
                        _finite(
                            self.clock(),
                            name="control_cycle_prime.primed_at",
                        ),
                    ),
                    clock_domain_id=activation_scope.clock_domain_id,
                    policy_required=policy_active,
                    human_required=human_required,
                    policy_reset_performed=policy_active,
                    policy_reset_at=policy_reset_at,
                    policy_emission=policy_emission,
                    human_emission=human_emission,
                    policy_submission=getattr(prime, "policy_submission", None),
                    human_submission=getattr(prime, "human_submission", None),
                    strict_source_identity=(
                        self.config.control_profile_id
                        == SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL
                    ),
                )
            )

    def set_human_blend_alpha(
        self,
        alpha: float | Mapping[str, float],
        *,
        effective_from: float | None = None,
    ) -> Any:
        """Switch or proportionally blend human input without changing owners."""

        with self._lock:
            if not self._started:
                raise RuntimeError("MujocoSimHILApplication.start() is required.")
            if not (
                self.control_cycle_port.spec.application_human_blend_control_enabled
            ):
                raise RuntimeError(
                    "The selected Sim-HIL control-cycle profile does not use "
                    "fixed-alpha human blend control."
                )
            receipt = self.composition.set_human_blend_alpha(
                alpha,
                effective_from=effective_from,
            )
            if not receipt.accepted:
                raise RuntimeError(
                    "Sim-HIL blend update was rejected: "
                    f"{receipt.rejection_code}:{receipt.message}"
                )
            self._blend_failure_fallback_latched_source = None
            self._reset_blend_failure_window()
            if isinstance(alpha, Mapping):
                human_enabled = any(float(value) > 1e-12 for value in alpha.values())
            else:
                human_enabled = float(alpha) > 1e-12
            if human_enabled:
                self._human_timeout_fallback_latched = False
            return receipt

    def submit_allocation_button_operation(
        self,
        *,
        signal_name: str,
        event_id: str,
        source_fact_id: str,
        source_fact_hash: str,
        occurred_at: float,
    ) -> Any:
        """Submit one typed UI/device event to the selected allocation profile."""

        with self._lock:
            if not self._started:
                raise RuntimeError("MujocoSimHILApplication.start() is required.")
            if self._held:
                raise RuntimeError("Allocation operations require a running session.")
            if self._episode_ended:
                raise RuntimeError("The Sim-HIL episode has ended.")
            return self._submit_allocation_operation_locked(
                signal_name=signal_name,
                event_id=event_id,
                source_fact_id=source_fact_id,
                source_fact_hash=source_fact_hash,
                occurred_at=occurred_at,
            )

    def _submit_allocation_operation_locked(
        self,
        *,
        signal_name: str,
        event_id: str,
        source_fact_id: str,
        source_fact_hash: str,
        occurred_at: float,
    ) -> Any:
        submit = getattr(
            self.control_cycle_port,
            "submit_operation_trigger",
            None,
        )
        if not callable(submit):
            raise RuntimeError(
                "The selected Sim-HIL profile has no allocation operation port."
            )
        return submit(
            signal_name=signal_name,
            event_id=event_id,
            source_fact_id=source_fact_id,
            source_fact_hash=source_fact_hash,
            occurred_at=float(occurred_at),
        )

    def _reset_blend_failure_window(self) -> None:
        self._blend_failure_window_started_at = None
        self._blend_failure_consecutive_count = 0

    def set_blend_failure_fallback(
        self,
        source: str,
        *,
        grace_sec: float | None = None,
    ) -> dict[str, object]:
        """Select the Station response to an infeasible blended target.

        This changes no target, safety, IK, or dispatch state.  A human/policy
        choice is applied only after a typed resolution failure and then uses
        the existing formal composition-parameter admission path.
        """

        normalized = str(source).strip().lower()
        if normalized not in SIM_HIL_BLEND_FAILURE_FALLBACKS:
            raise ValueError("blend failure fallback must be hold, human, or policy.")
        grace = (
            float(getattr(self, "_blend_failure_grace_sec", 0.0))
            if grace_sec is None
            else _finite(grace_sec, name="blend_failure.grace_sec")
        )
        if grace < 0.0:
            raise ValueError("blend failure grace_sec must be non-negative.")
        with self._lock:
            self._blend_failure_fallback = normalized
            self._blend_failure_grace_sec = grace
            self._reset_blend_failure_window()
            return {
                "status": "selected",
                "source": normalized,
                "grace_sec": grace,
                "latched_source": self._blend_failure_fallback_latched_source,
            }

    def set_human_timeout_fallback(self, source: str) -> dict[str, object]:
        """Select what to do after the Sim-HIL human input window expires.

        ``hold`` leaves the lower composition gate responsible for stopping
        publication.  ``policy`` is an explicit, latched source handoff; it
        never republishes the last human pose and never switches back
        automatically when VIVE packets resume.
        """

        normalized = str(source).strip().lower()
        if normalized not in SIM_HIL_HUMAN_TIMEOUT_FALLBACKS:
            raise ValueError("human timeout fallback must be hold or policy.")
        with self._lock:
            self._human_timeout_fallback = normalized
            return {
                "status": "selected",
                "source": normalized,
                "latched": self._human_timeout_fallback_latched,
            }

    @staticmethod
    def _resolution_failure_status(failure: Any | None) -> dict[str, object] | None:
        if failure is None:
            return None
        return {
            "reason_code": getattr(failure, "reason_code", None),
            "resolver_status": getattr(failure, "resolver_status", None),
            "message": getattr(failure, "message", None),
            "diagnostics": dict(getattr(failure, "diagnostics", {}) or {}),
        }

    @staticmethod
    def _collision_stop_failure(cycle: Any) -> Any | None:
        """Return the exact lower collision rejection, if this cycle has one."""

        execution = getattr(cycle, "execution", None)
        if getattr(execution, "status", None) != (
            "compound_resolution_failure_hold"
        ):
            return None
        failure = getattr(execution, "resolution_failure", None)
        return (
            failure
            if getattr(failure, "reason_code", None) == "collision_margin_low"
            else None
        )

    def _terminate_episode_for_collision_stop(self, failure: Any) -> None:
        """Promote one lower collision rejection to a terminal Sim episode."""

        reason_code = str(getattr(failure, "reason_code", "") or "").strip()
        if reason_code != "collision_margin_low":
            raise ValueError(
                "Sim collision termination requires collision_margin_low."
            )
        self._last_error = f"SimCollisionStop:{reason_code}"
        if self._first_control_fault is None:
            self._first_control_fault = self._last_error
        self.end_episode(
            reason=f"sim_collision_stop:{reason_code}",
            force_cancellation=True,
        )

    def _capture_sim_contact_observation(self) -> dict[str, object]:
        """Read lower-owned MuJoCo contact evidence without changing physics."""

        rig = getattr(getattr(self, "runtime", None), "rig", None)
        capture = getattr(rig, "capture_active_contact_observation", None)
        if not callable(capture):
            observation = {
                "available": False,
                "safe": True,
                "semantics": "mujoco_active_forbidden_contacts",
            }
        else:
            observation = dict(capture())
        self._last_sim_contact_observation = observation
        return observation

    def _terminate_episode_for_active_forbidden_contact(
        self,
        observation: Mapping[str, object],
    ) -> None:
        """Promote a lower-owned hard-contact verdict to terminal Sim HOLD."""

        evidence = dict(observation)
        if not bool(evidence.get("available")) or bool(evidence.get("safe", True)):
            raise ValueError(
                "Active-contact termination requires an available unsafe verdict."
            )
        pair = " ↔ ".join(
            str(value)
            for value in (evidence.get("geom1"), evidence.get("geom2"))
            if value
        )
        self._last_error = "SimCollisionStop:active_forbidden_contact"
        if pair:
            self._last_error += f":{pair}"
        if self._first_control_fault is None:
            self._first_control_fault = self._last_error
        record_guard = getattr(
            self.runtime,
            "record_active_contact_guard_result",
            None,
        )
        if not callable(record_guard):
            raise TypeError(
                "Sim runtime must expose record_active_contact_guard_result()."
            )
        record_guard(evidence)
        self.end_episode(
            reason="sim_collision_stop:active_forbidden_contact",
            force_cancellation=True,
        )

    def _apply_blend_failure_fallback(
        self,
        cycle: Any,
        *,
        observed_at: float,
    ) -> None:
        cycle_port = getattr(self, "control_cycle_port", None)
        if (
            cycle_port is not None
            and not cycle_port.spec.application_blend_failure_fallback_enabled
        ):
            return
        execution = getattr(cycle, "execution", None)
        execution_status = getattr(execution, "status", None)
        if execution_status != "compound_resolution_failure_hold":
            if execution_status not in {None, "not_due"}:
                self._reset_blend_failure_window()
            return
        failure = getattr(execution, "resolution_failure", None)
        if failure is None:
            return
        alpha_by_component = dict(
            getattr(self.composition, "human_alpha_by_component", {}) or {}
        )
        if not alpha_by_component or not all(
            1e-12 < float(value) < 1.0 - 1e-12 for value in alpha_by_component.values()
        ):
            self._reset_blend_failure_window()
            return
        selected = self._blend_failure_fallback
        grace_sec = float(getattr(self, "_blend_failure_grace_sec", 0.0))
        window_started_at = getattr(
            self,
            "_blend_failure_window_started_at",
            None,
        )
        if window_started_at is None:
            window_started_at = float(observed_at)
            self._blend_failure_window_started_at = window_started_at
            self._blend_failure_consecutive_count = 0
        self._blend_failure_consecutive_count = int(
            getattr(self, "_blend_failure_consecutive_count", 0)
        ) + 1
        failure_elapsed_sec = max(
            0.0,
            float(observed_at) - float(window_started_at),
        )
        event: dict[str, object] = {
            "status": "hold_selected" if selected == "hold" else "latched",
            "selected_source": selected,
            "triggered_at": observed_at,
            "cycle_index": getattr(cycle, "cycle_index", None),
            "alpha_before": alpha_by_component,
            "failure": self._resolution_failure_status(failure),
            "grace_sec": grace_sec,
            "failure_window_started_at": window_started_at,
            "failure_elapsed_sec": failure_elapsed_sec,
            "consecutive_failure_count": self._blend_failure_consecutive_count,
            "parameter_receipt_id": None,
        }
        if selected != "hold" and failure_elapsed_sec < grace_sec:
            event["status"] = "grace_active"
            self._last_blend_failure_fallback = event
            return
        if selected != "hold":
            target_alpha = 1.0 if selected == "human" else 0.0
            receipt = self.composition.set_human_blend_alpha(
                target_alpha,
                effective_from=max(
                    observed_at,
                    _finite(
                        self.runtime.clock(),
                        name="blend_fallback.effective_from",
                    ),
                ),
            )
            if not receipt.accepted:
                raise RuntimeError(
                    "Sim-HIL blend failure fallback was rejected: "
                    f"{receipt.rejection_code}:{receipt.message}"
                )
            self._blend_failure_fallback_latched_source = selected
            event["alpha_after"] = {
                component_id: target_alpha for component_id in alpha_by_component
            }
            event["parameter_receipt_id"] = getattr(
                receipt,
                "parameter_snapshot_id",
                getattr(receipt, "semantic_content_hash", None),
            )
            self._reset_blend_failure_window()
        self._last_blend_failure_fallback = event

    def _apply_human_timeout_fallback(
        self,
        *,
        observed_at: float,
        policy_emission: Any | None,
    ) -> Any | None:
        """Apply the configured network-gap fallback without stale-pose reuse."""

        cycle_port = getattr(self, "control_cycle_port", None)
        if (
            cycle_port is not None
            and not cycle_port.spec.application_human_timeout_fallback_enabled
        ):
            return policy_emission
        if self._human_timeout_fallback_latched:
            return policy_emission
        alpha_by_component = dict(
            getattr(self.composition, "human_alpha_by_component", {}) or {}
        )
        if not alpha_by_component or not any(
            float(value) > 1e-12 for value in alpha_by_component.values()
        ):
            return policy_emission
        snapshot = getattr(self.readiness_provider, "snapshot", None)
        if not callable(snapshot):
            # Lightweight test/compatibility bindings may not expose the
            # optional readiness observer; retain their previous behavior.
            return policy_emission
        readiness = snapshot(now=observed_at)
        if bool(getattr(readiness, "human_device_ready", True)):
            return policy_emission
        event: dict[str, object] = {
            "status": "hold_selected",
            "selected_source": self._human_timeout_fallback,
            "triggered_at": observed_at,
            "human_detail": getattr(
                readiness,
                "detail_by_check",
                {},
            ).get("human_device_ready"),
            "parameter_receipt_id": None,
        }
        if self._human_timeout_fallback == "policy":
            if policy_emission is None:
                try:
                    policy_emission = self.policy_source.step(now=observed_at)
                except Exception as error:  # noqa: BLE001 - lower gate still holds.
                    event["status"] = "policy_fallback_failed"
                    event["error"] = f"{type(error).__name__}:{error}"
                    self._last_human_timeout_fallback = event
                    return None
            if policy_emission is None:
                event["status"] = "policy_fallback_unavailable"
                self._last_human_timeout_fallback = event
                return None
            receipt = self.composition.set_human_blend_alpha(
                0.0,
                effective_from=observed_at,
            )
            if not receipt.accepted:
                event["status"] = "policy_fallback_rejected"
                event["error"] = f"{receipt.rejection_code}:{receipt.message}"
                self._last_human_timeout_fallback = event
                return policy_emission
            event["status"] = "policy_latched"
            event["parameter_receipt_id"] = getattr(
                receipt,
                "parameter_snapshot_id",
                getattr(receipt, "semantic_content_hash", None),
            )
            self._human_timeout_fallback_latched = True
            self._last_human_timeout_fallback = event
        return policy_emission

    def start(
        self,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> Any:
        with self._lock:
            if self._started:
                return self._last_preflight
            if self._restart_required:
                raise RuntimeError(
                    "The Sim-HIL application was closed after a failed startup "
                    "and must be rebuilt. First startup failure: "
                    f"{self._startup_failure or 'unknown'}"
                )
            if self._closed:
                raise RuntimeError(
                    "The Sim-HIL application is closed and must be rebuilt."
                )
            if self._human_input_check_active:
                raise RuntimeError(
                    "Sim-HIL cannot start while passive human input "
                    "acceptance is running."
                )
            try:
                self._raise_if_cancelled(
                    cancel_requested,
                    phase="runtime_start",
                )
                self._warmup_sources(
                    cancel_requested=cancel_requested,
                )
                self._raise_if_cancelled(
                    cancel_requested,
                    phase="viewer_start",
                )
                # Open (or reuse) the native viewer before starting the
                # realtime physics worker. ``launch_passive`` and the worker
                # both touch the same MjModel/MjData; starting them
                # concurrently has caused native crashes on some GLFW/GLX
                # builds. A Human-demonstration preview may already own this
                # viewer, but it has not advanced physics or opened authority.
                if self.viewer is not None:
                    self.viewer.start()
                    # Native window construction may exceed the contributor
                    # TTL even though the receiver remains healthy.  Re-poll
                    # the real sources after that boundary instead of passing
                    # the pre-viewer emission to composition preflight.
                    self._wait_for_source_readiness(
                        cancel_requested=cancel_requested,
                        phase="post_viewer_source_refresh",
                    )
                report = self.composition.start()
                # The warm-up emissions predate the composition activation barrier.
                # Acquire and admit one fresh pair after that barrier; no
                # second queue or authority is introduced.
                self._activate_and_prime_control_cycle()
                self._last_preflight = report
                self._started = True
                self._viewer_preview_started = False
                self._held = False
                self._episode_ended = False
                self._episode_end_reason = None
                self._next_decision_at = _finite(
                    self.runtime.clock(),
                    name="start.next_decision_at",
                )
                self._overrun_dropped_cycles = 0
                self._last_schedule_overrun = None
                self._last_error = None
                return report
            except Exception as error:
                primary_error = f"{type(error).__name__}:{error}"
                self._startup_failure = primary_error
                self._restart_required = True
                cleanup_error: str | None = None
                try:
                    self.close()
                except (
                    Exception
                ) as close_error:  # noqa: BLE001 - preserve primary cause.
                    cleanup_error = f"{type(close_error).__name__}:{close_error}"
                self._last_error = (
                    primary_error
                    if cleanup_error is None
                    else f"{primary_error}; cleanup_error={cleanup_error}"
                )
                raise

    def start_viewer_preview(self) -> dict[str, object]:
        """Open only the passive MuJoCo viewer for scene/camera adjustment.

        This boundary does not start sources, composition, the control-cycle
        port, the physics worker, or episode recording.  The existing runtime
        rig remains the sole scene owner and viewer synchronization is passive.
        """

        with self._lock:
            if self._started:
                raise RuntimeError(
                    "Viewer preview must be opened before Sim-HIL starts."
                )
            if self._restart_required:
                raise RuntimeError(
                    "The Sim-HIL application requires rebuild before preview."
                )
            if self._closed:
                raise RuntimeError(
                    "The Sim-HIL application is closed and must be rebuilt."
                )
            if self.viewer is None:
                raise RuntimeError(
                    "Human demonstration preview requires the MuJoCo viewer."
                )
            viewer_status = dict(self.viewer.start())
            self._viewer_preview_started = True
            self._viewer_exit_requested = False
            return {
                "status": "viewer_preview",
                "command_dispatched": False,
                "authoritative": False,
                "sources_started": False,
                "composition_started": False,
                "viewer": viewer_status,
            }

    def sync_viewer_preview(self) -> bool:
        """Render one preview frame without stepping sources or simulation."""

        with self._lock:
            if not self._viewer_preview_started or self._started:
                return False
            if self.viewer is None:
                return False
            synchronized = bool(self.viewer.sync())
            if not synchronized:
                viewer_state = str(
                    dict(self.viewer.status()).get("state") or ""
                )
                if viewer_state in {"closed_by_user", "faulted"}:
                    self._viewer_exit_requested = True
            return synchronized

    def _skip_missed_compose_cuts(
        self,
        *,
        decision_at: float,
        observed_at: float,
        period: float,
        phase: str,
    ) -> float:
        aligned_decision_at, skipped = _advance_control_deadline(
            previous_deadline=decision_at - period,
            completed_at=observed_at,
            period=period,
        )
        self._overrun_dropped_cycles += skipped
        self._last_schedule_overrun = {
            "observed_at": observed_at,
            "late_by_sec": observed_at - decision_at,
            "skipped_cycle_count": skipped,
            "previous_decision_at": decision_at,
            "resumed_decision_at": aligned_decision_at,
            "resolution": "skip_to_next_compose_grid",
            "phase": _required(phase, name="overrun.phase"),
        }
        return aligned_decision_at

    def _remember_policy_emission(
        self,
        emission: PolicyActionEmission | None,
    ) -> None:
        if emission is None:
            return
        if not isinstance(emission, PolicyActionEmission):
            # Production graphs validate the typed source port in the factory.
            # Lightweight lifecycle fixtures may use identity-only stand-ins;
            # those cannot participate in the causal feedback registry.
            return
        self._policy_emission_by_id[emission.emission_id] = emission
        self._policy_emission_by_id.move_to_end(emission.emission_id)
        while len(self._policy_emission_by_id) > 64:
            self._policy_emission_by_id.popitem(last=False)

    def _record_policy_execution_feedback(
        self,
        *,
        cycle: Any,
        cycle_emission: PolicyActionEmission | None,
        observed_at: float,
    ) -> None:
        projection = project_sim_hil_policy_execution_feedback(
            cycle=cycle,
            cycle_emission=cycle_emission,
            emission_by_id=self._policy_emission_by_id,
            execution_difference_observer=self.execution_difference_observer,
            observed_at=observed_at,
            clock_domain_id=self.runtime.config.clock_domain_id,
            policy_role=str(
                getattr(
                    getattr(self.composition, "config", None),
                    "policy_role",
                    "policy",
                )
            ),
        )
        if projection.feedback is None:
            self._last_policy_feedback_status = {
                "status": projection.status,
                "detail": projection.detail,
            }
            return
        feedback = projection.feedback
        if isinstance(self.policy_source, SimHILPolicyExecutionFeedbackPort):
            try:
                application = self.policy_source.observe_execution_feedback(feedback)
                source_error = None
            except Exception as error:  # noqa: BLE001 - passive feedback cannot control motion.
                source_error = f"{type(error).__name__}:{error}"
                application = PolicyExecutionFeedbackApplication(
                    feedback_id=feedback.feedback_id,
                    feedback_hash=feedback.semantic_content_hash,
                    decision="source_consumer_error_no_state_mutation",
                    reason=source_error,
                )
        else:
            source_error = None
            application = PolicyExecutionFeedbackApplication(
                feedback_id=feedback.feedback_id,
                feedback_hash=feedback.semantic_content_hash,
                decision="recorded_by_application_only",
                reason="policy source does not implement the optional feedback port",
            )
        fact = project_policy_execution_feedback(
            feedback,
            application,
            runtime_session_id=self.runtime.config.runtime_session_id,
            control_domain_id=self.runtime.config.control_domain_id,
            embodiment_instance_id=self.runtime.config.embodiment_instance_id,
        )
        self.runtime.record_external_facts((fact,))
        self._last_policy_feedback_status = {
            "status": projection.status,
            "feedback_id": feedback.feedback_id,
            "relation": feedback.policy_output_relation,
            "association_status": feedback.association_status,
            "application_decision": application.decision,
            "source_error": source_error,
        }

    def step(self, *, now: float | None = None) -> Any:
        with self._lock:
            cycle_started_at = time.perf_counter()
            if not self._started:
                raise RuntimeError("MujocoSimHILApplication.start() is required.")
            if self._held:
                raise RuntimeError(
                    "MuJoCo Sim-HIL is held; explicit rearm is required."
                )
            timestamp = _finite(
                self.clock() if now is None else now,
                name="step.now",
            )
            decision_at = (
                timestamp if self._next_decision_at is None else self._next_decision_at
            )
            period = 1.0 / float(self.runtime.config.decision_hz)
            lateness = timestamp - decision_at
            maximum_lateness = max(0.02, period)
            if lateness > maximum_lateness:
                # Station composition is not the canonical execution watchdog.
                # Skip missed upstream compose cuts without burst catch-up;
                # the lower target gate remains authoritative for whether the
                # newly scheduled target can still be admitted and executed.
                decision_at = self._skip_missed_compose_cuts(
                    decision_at=decision_at,
                    observed_at=timestamp,
                    period=period,
                    phase="step_entry",
                )
            try:
                policy_started_at = time.perf_counter()
                policy_emission = None
                if self._policy_contributor_active():
                    policy_emission = self.policy_source.step(now=timestamp)
                policy_elapsed_sec = time.perf_counter() - policy_started_at
                observed_at = max(
                    timestamp,
                    _finite(self.runtime.clock(), name="step.observed_at"),
                )
                policy_emission = self._apply_human_timeout_fallback(
                    observed_at=observed_at,
                    policy_emission=policy_emission,
                )
                self._remember_policy_emission(policy_emission)
                # A policy fallback may have produced a synchronous emission
                # after the first observation cut; close the cut again before
                # admitting it so clock ordering cannot reject a valid fresh
                # fallback as ``emission_in_future``.
                observed_at = max(
                    observed_at,
                    _finite(
                        self.runtime.clock(),
                        name="step.fallback_observed_at",
                    ),
                    _finite(
                        self.clock(),
                        name="step.fallback_station_observed_at",
                    ),
                )
                if observed_at - decision_at > maximum_lateness:
                    # Synchronous policy work can consume the original target
                    # lead after entry-time alignment. Move the upstream cut
                    # forward before publication; the target still owns final
                    # late admission and HOLD.
                    decision_at = self._skip_missed_compose_cuts(
                        decision_at=decision_at,
                        observed_at=observed_at,
                        period=period,
                        phase="post_policy_source",
                    )
                composition_started_at = time.perf_counter()
                self._last_cycle = self.control_cycle_port.run_cycle(
                    now=observed_at,
                    policy_emission=policy_emission,
                    poll_human=True,
                    target_effective_at=(
                        observed_at
                        if getattr(
                            self.runtime.reference_activation_spec,
                            "activation_mode",
                            None,
                        )
                        == "immediate_target"
                        else (
                            decision_at
                            + self.runtime.reference_activation_spec.target_lead_time_sec
                        )
                    ),
                )
                collision_stop_failure = (
                    self._collision_stop_failure(self._last_cycle)
                    if getattr(
                        self.config,
                        "sim_collision_stop_enabled",
                        True,
                    )
                    else None
                )
                active_contact = self._capture_sim_contact_observation()
                active_forbidden_contact = bool(
                    getattr(
                        self.config,
                        "sim_collision_stop_enabled",
                        True,
                    )
                    and active_contact.get("available")
                    and not active_contact.get("safe", True)
                )
                if collision_stop_failure is None:
                    self._apply_blend_failure_fallback(
                        self._last_cycle,
                        observed_at=observed_at,
                    )
                feedback_observed_at = max(
                    observed_at,
                    _finite(
                        self.runtime.clock(),
                        name="step.policy_feedback_observed_at",
                    ),
                )
                try:
                    self._record_policy_execution_feedback(
                        cycle=self._last_cycle,
                        cycle_emission=policy_emission,
                        observed_at=feedback_observed_at,
                    )
                except Exception as error:  # noqa: BLE001 - passive evidence cannot control motion.
                    self._last_policy_feedback_status = {
                        "status": "feedback_projection_error",
                        "detail": f"{type(error).__name__}:{error}",
                    }
                composition_elapsed_sec = time.perf_counter() - composition_started_at
                nominal_next = decision_at + period
                lower_due = getattr(
                    self.runtime,
                    "next_decision_due_at",
                    None,
                )
                if (
                    getattr(
                        getattr(self._last_cycle, "execution", None),
                        "status",
                        None,
                    )
                    == "not_due"
                    and lower_due is not None
                    and lower_due > observed_at
                ):
                    # A small phase offset is not a missed compose cycle. Retry
                    # at the lower scheduler's exact cut instead of discarding
                    # a complete input period.
                    self._next_decision_at = lower_due
                else:
                    # Never let a stale lower deadline move Station backwards.
                    self._next_decision_at = (
                        nominal_next
                        if lower_due is None
                        else max(nominal_next, lower_due)
                    )
                gate = self.runtime.capture_execution_gate_snapshot(
                    now=max(
                        observed_at,
                        _finite(
                            self.runtime.clock(),
                            name="step.gate_observed_at",
                        ),
                    )
                )
                if gate.status != "ready":
                    self._held = True
                    gate_reason = str(getattr(gate, "reason", "") or "unspecified")
                    self._last_error = (
                        "ExecutionGateNotReady:" f"{gate.status}:{gate_reason}"
                    )
                    if self._first_control_fault is None:
                        self._first_control_fault = self._last_error
                if active_forbidden_contact:
                    self._terminate_episode_for_active_forbidden_contact(
                        active_contact
                    )
                elif collision_stop_failure is not None:
                    self._terminate_episode_for_collision_stop(
                        collision_stop_failure
                    )
                if self.viewer is not None and not self.viewer.sync():
                    viewer_state = str(dict(self.viewer.status()).get("state") or "")
                    if viewer_state in {"closed_by_user", "faulted"}:
                        # Closing the native window is an operator exit
                        # request, not merely a missing visualization frame.
                        # Establish lower HOLD now; the Station loop performs
                        # the one authoritative application teardown.
                        self._viewer_exit_requested = True
                        self.hold(reason=f"mujoco_viewer_{viewer_state}")
                self._last_cycle_timing = {
                    "total_sec": time.perf_counter() - cycle_started_at,
                    "policy_source_sec": policy_elapsed_sec,
                    "composition_and_execution_sec": composition_elapsed_sec,
                    "configured_period_sec": period,
                }
                if gate.status == "ready":
                    self._last_error = None
                return self._last_cycle
            except Exception as error:
                self._last_cycle_timing = {
                    "total_sec": time.perf_counter() - cycle_started_at,
                    "configured_period_sec": period,
                }
                self._last_error = f"{type(error).__name__}:{error}"
                if self._first_control_fault is None:
                    self._first_control_fault = self._last_error
                raise

    def hold(
        self,
        *,
        reason: str,
        force_cancellation: bool = False,
    ) -> Any | None:
        """Pause publication through the lower execution gate."""

        if not isinstance(force_cancellation, bool):
            raise TypeError("force_cancellation must be a bool.")
        with self._lock:
            # A startup failure can leave the lower composition active before
            # this application publishes ``started=True``.  That partial state
            # still requires a lower HOLD before cleanup.
            if not self._started and not bool(
                getattr(self.composition, "started", False)
            ):
                return None
            normalized_reason = _required(reason, name="reason")
            outcome = self._deactivate_cycle_and_request_lower_hold(
                reason=f"sim_hil_application:{normalized_reason}",
                force_cancellation=force_cancellation,
            )
            self._last_hold_outcome = outcome
            return outcome

    def _deactivate_cycle_and_request_lower_hold(
        self,
        *,
        reason: str,
        force_cancellation: bool = False,
    ) -> Any:
        """Fence publication and establish lower HOLD even if fencing faults."""

        normalized_reason = _required(reason, name="reason")
        if not isinstance(force_cancellation, bool):
            raise TypeError("force_cancellation must be a bool.")
        deactivation_error: Exception | None = None
        try:
            self.control_cycle_port.deactivate(reason=normalized_reason)
        except Exception as error:  # noqa: BLE001 - lower HOLD still required.
            deactivation_error = error
        try:
            if force_cancellation:
                outcome = self.composition.request_hold(
                    reason=normalized_reason,
                    force_cancellation=True,
                )
            else:
                outcome = self.composition.request_hold(reason=normalized_reason)
        except Exception as hold_error:  # noqa: BLE001 - preserve both failures.
            self._held = True
            details = f"{type(hold_error).__name__}:{hold_error}"
            if deactivation_error is not None:
                details = (
                    "control_cycle_fence_failed:"
                    f"{type(deactivation_error).__name__}:{deactivation_error};"
                    f"lower_hold_failed:{details}"
                )
            self._last_error = details
            raise
        self._held = True
        reset_pose_state = getattr(
            self.runtime,
            "reset_pose_resolution_temporal_state",
            None,
        )
        if callable(reset_pose_state):
            try:
                reset_pose_state()
            except Exception as reset_error:
                self._last_error = (
                    "pose_resolution_temporal_reset_failed_after_lower_hold:"
                    f"{type(reset_error).__name__}:{reset_error}"
                )
                raise RuntimeError(self._last_error) from reset_error
        if deactivation_error is not None:
            self._last_error = (
                "control_cycle_fence_failed_after_lower_hold:"
                f"{type(deactivation_error).__name__}:{deactivation_error}"
            )
            raise RuntimeError(self._last_error) from deactivation_error
        return outcome

    def end_episode(
        self,
        *,
        reason: str = "episode_end",
        force_cancellation: bool = False,
    ) -> Any | None:
        """Establish terminal HOLD while passive inputs remain connected.

        Reset remains owned by the simulator task.  A subsequent episode must
        close and rebuild this target instance instead of mutating the scene
        behind a live execution rig.
        """

        if not isinstance(force_cancellation, bool):
            raise TypeError("force_cancellation must be a bool.")
        with self._lock:
            if not self._started:
                return None
            self._episode_ended = True
            self._episode_end_reason = _required(reason, name="reason")
            return self.hold(
                reason=f"episode_end:{self._episode_end_reason}",
                force_cancellation=force_cancellation,
            )

    def rearm(
        self,
        *,
        now: float | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> Any:
        """Refresh both contributors around a new composition activation."""

        with self._lock:
            report, _receipt = self._rearm_locked(
                now=now,
                cancel_requested=cancel_requested,
            )
            return report

    def rearm_with_allocation_operation(
        self,
        *,
        signal_name: str,
        event_id: str,
        source_fact_id: str,
        source_fact_hash: str,
        occurred_at: float,
        human_emission_required: bool,
        now: float | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> tuple[Any, Any]:
        """Rearm and select a streaming allocation mode before cycles reopen.

        This is a downstream transaction seam, not another composer.  The
        selected allocation binding is activated and source-primed first; the
        typed operation is then admitted while the application still holds
        its cycle lock and before ``_held`` is cleared, so no target cycle can
        observe the new activation with the old/default mode.
        """

        if not isinstance(human_emission_required, bool):
            raise TypeError("human_emission_required must be bool.")
        operation = {
            "signal_name": signal_name,
            "event_id": event_id,
            "source_fact_id": source_fact_id,
            "source_fact_hash": source_fact_hash,
            "occurred_at": float(occurred_at),
        }
        with self._lock:
            report, receipt = self._rearm_locked(
                now=now,
                cancel_requested=cancel_requested,
                human_emission_required=human_emission_required,
                activation_operation=operation,
            )
            assert receipt is not None
            return report, receipt

    def _rearm_locked(
        self,
        *,
        now: float | None,
        cancel_requested: Callable[[], bool] | None,
        human_emission_required: bool | None = None,
        activation_operation: dict[str, Any] | None = None,
    ) -> tuple[Any, Any | None]:
        """Locked implementation shared by ordinary and mode-selecting rearm."""

        if self._episode_ended:
            raise RuntimeError(
                "An ended Sim-HIL episode requires target rebuild, not rearm."
            )
        if not self._started or not self._held:
            raise RuntimeError(
                "MuJoCo Sim-HIL rearm requires a started held application."
            )
        timestamp = _finite(
            self.clock() if now is None else now,
            name="rearm.now",
        )
        # Establish the authoritative lower HOLD before resetting or waiting
        # on contributors. Application-local ``_held`` is only a latch and
        # cannot protect the target while lower READY is still current.
        self._deactivate_cycle_and_request_lower_hold(
            reason="sim_hil_application:rearm_boundary",
        )
        hold_observed_at = _finite(
            self.runtime.clock(),
            name="rearm.hold_observed_at",
        )
        if self._policy_contributor_active():
            self.policy_source.reset_for_activation()
        self._wait_for_source_readiness(
            cancel_requested=cancel_requested,
            phase="rearm_source_refresh",
        )
        rearm_at = max(
            timestamp,
            hold_observed_at,
            _finite(self.runtime.clock(), name="rearm.observed_at"),
        )
        report = self.composition.rearm(now=rearm_at)
        try:
            self._activate_and_prime_control_cycle(
                human_emission_required=human_emission_required,
            )
            operation_receipt = None
            if activation_operation is not None:
                operation_receipt = self._submit_allocation_operation_locked(
                    **activation_operation,
                )
                # Human absolute ownership needs a concrete allocation context
                # before the first target cycle. Policy/base-only is the
                # context-free fallback by contract and must remain resumable
                # even when the passive Human source is unavailable.
                if human_emission_required:
                    synchronized_at = max(
                        rearm_at,
                        _finite(
                            self.runtime.clock(),
                            name="rearm.operation_context_observed_at",
                        ),
                    )
                    activation_spec = self.runtime.reference_activation_spec
                    target_effective_at = (
                        synchronized_at
                        if activation_spec.activation_mode == "immediate_target"
                        else synchronized_at + activation_spec.target_lead_time_sec
                    )
                    self.control_cycle_port.synchronize_operation_context(
                        now=synchronized_at,
                        target_effective_at=target_effective_at,
                        timeout_sec=self.config.warmup_timeout_sec,
                    )
        except BaseException:
            # The lower session rearm has already moved its gate to READY.
            # Keep a failed activation/selection transaction physically held,
            # even though the application-local held latch never opened.
            self._deactivate_cycle_and_request_lower_hold(
                reason="sim_hil_application:rearm_operation_failed",
                force_cancellation=True,
            )
            raise
        self._next_decision_at = _finite(
            self.runtime.clock(),
            name="rearm.next_decision_at",
        )
        self._last_schedule_overrun = None
        self._held = False
        self._last_error = None
        return report, operation_receipt

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            errors: list[BaseException] = []
            if self.composition.started:
                try:
                    self.hold(reason="shutdown")
                except BaseException as error:  # noqa: BLE001 - cleanup continues.
                    errors.append(error)
                try:
                    self.control_cycle_port.close()
                except BaseException as error:  # noqa: BLE001 - cleanup continues.
                    errors.append(error)
                try:
                    self.composition.close()
                except BaseException as error:  # noqa: BLE001 - continue lower cleanup.
                    errors.append(error)
                if self.control_cycle_port.spec.application_owns_human_source_close:
                    try:
                        self.human_source.close()
                    except BaseException as error:  # noqa: BLE001 - cleanup continues.
                        errors.append(error)
            elif self._sources_started:
                try:
                    self.control_cycle_port.close()
                except BaseException as error:  # noqa: BLE001 - cleanup continues.
                    errors.append(error)
                try:
                    self.human_source.close()
                except BaseException as error:  # noqa: BLE001 - continue lower cleanup.
                    errors.append(error)
            else:
                try:
                    self.control_cycle_port.close()
                except BaseException as error:  # noqa: BLE001 - cleanup continues.
                    errors.append(error)
            try:
                self.policy_source.close()
            except BaseException as error:  # noqa: BLE001 - continue lower cleanup.
                errors.append(error)
            # The composition/runtime close above joins the physics worker.
            # Only then may the native viewer tear down the shared MjData.
            if self.viewer is not None:
                try:
                    self.viewer.close()
                except BaseException as error:  # noqa: BLE001 - cleanup continues.
                    errors.append(error)
            self._started = False
            self._viewer_preview_started = False
            self._sources_started = False
            self._held = False
            if errors:
                self._last_error = ";".join(
                    f"{type(error).__name__}:{error}" for error in errors
                )
                raise RuntimeError(
                    "Sim-HIL close encountered lower-runtime errors: "
                    + self._last_error
                ) from errors[0]


__all__ = ["MujocoSimHILApplication"]
