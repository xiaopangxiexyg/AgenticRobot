"""Lower RealTeleop consumer for exclusive Human/Policy command selection."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from deployments.openpi.client import (
    OpenPIEmissionSourceConfig,
    build_openpi_policy_emission_source,
)
from deployments.policy_artifact import load_policy_artifact
from robot_motion.action import RobotActionResolver
from robot_motion.deploy import (
    PolicyActionEmission,
    PolicyEmissionSourcePort,
    require_policy_emission_source,
)

from ..command_override import TeleopCommandOverride
from ..schemas import TeleopCommand, TeleopObservation
from .policy_observation import RealTeleopPolicyObservationCapture
from .single_source_composition import ExclusiveSingleSourceComposer


HITL_POLICY_MODE_NORMAL = "normal"
HITL_POLICY_MODE_RTC = "rtc"
HITL_POLICY_MODES = (HITL_POLICY_MODE_NORMAL, HITL_POLICY_MODE_RTC)


def preflight_normal_hitl_policy_source(
    args: object,
) -> OpenPIEmissionSourceConfig | None:
    """Validate the existing OpenPI binding before RealTeleop opens devices."""

    if getattr(args, "_hitl_policy_emission_source", None) is not None:
        return None
    artifact_path = getattr(args, "hitl_policy_artifact_json", None)
    if artifact_path is None or not str(artifact_path).strip():
        raise ValueError(
            "Normal HITL policy mode requires --hitl-policy-artifact-json."
        )
    if bool(getattr(args, "no_collection", False)):
        raise ValueError(
            "Normal HITL requires formal collection; remove --no-collection."
        )
    pose_frame_contract = getattr(args, "pose_frame_contract", None)
    if pose_frame_contract is None:
        raise ValueError(
            "Normal HITL requires the existing pose-frame contract."
        )
    artifact = load_policy_artifact(Path(artifact_path))
    if (
        artifact.camera_input is not None
        and artifact.camera_input.required
        and bool(getattr(args, "no_camera", False))
    ):
        raise ValueError(
            "The selected HITL policy artifact requires cameras; "
            "remove --no-camera."
        )
    requested_sides = tuple(
        str(side) for side in getattr(args, "active_sides", ())
    )
    if not requested_sides:
        raise ValueError(
            "Normal HITL requires the resolved RealTeleop active sides."
        )
    execution_steps = int(
        getattr(args, "hitl_policy_chunk_execution_steps", 0)
    )
    return OpenPIEmissionSourceConfig(
        policy_artifact=artifact,
        runtime_pose_frame_contract=pose_frame_contract,
        output_active_sides=requested_sides,
        model_observation_sides=requested_sides,
        host=str(getattr(args, "hitl_openpi_host", "127.0.0.1")),
        port=int(getattr(args, "hitl_openpi_port", 8000)),
        prompt=(
            str(getattr(args, "hitl_policy_prompt", "") or "").strip()
            or None
        ),
        action_hz=float(getattr(args, "hitl_policy_action_hz", 15.0)),
        policy_hz=float(getattr(args, "hitl_policy_inference_hz", 15.0)),
        action_chunk_execution_steps=(
            None if execution_steps <= 0 else execution_steps
        ),
        prediction_request_queue_threshold=int(
            getattr(args, "hitl_policy_prefetch_threshold", 4)
        ),
    )


def build_normal_hitl_policy_source(
    *,
    args: object,
    config: OpenPIEmissionSourceConfig | None,
    components: object,
    dataset_recorder: object | None,
    dataset_recorder_provider: Callable[[], object | None],
    camera_sources: Mapping[str, Any],
) -> tuple[object, RealTeleopPolicyObservationCapture | None]:
    """Bind the existing OpenPI source to collection-owned causal snapshots."""

    injected = getattr(args, "_hitl_policy_emission_source", None)
    if injected is not None:
        return injected, None
    if config is None:
        raise RuntimeError("Normal HITL policy preflight was not completed.")
    if dataset_recorder is None and dataset_recorder_provider() is None:
        raise RuntimeError(
            "Normal HITL collection requires the formal dataset recorder "
            "so policy inference can reuse its causal state pollers."
        )
    if tuple(config.output_active_sides) != tuple(components.active_sides):
        raise RuntimeError(
            "RealTeleop active sides changed after HITL policy preflight."
        )
    artifact = config.policy_artifact
    capture = RealTeleopPolicyObservationCapture(
        components=components,
        dataset_recorder=dataset_recorder_provider,
        camera_sources=camera_sources,
        require_images=bool(
            artifact.camera_input is not None
            and artifact.camera_input.required
        ),
        pose_frame_contract=config.runtime_pose_frame_contract,
    )
    source = build_openpi_policy_emission_source(
        config,
        observation_capture=capture,
        observation_reset=capture.reset,
        last_action_sink=capture.observe_last_action,
    )
    return source, capture


@dataclass(frozen=True)
class HITLPolicySelectionStatus:
    selected_source: str
    policy_mode: str
    policy_activation_pending: bool
    fresh_policy_emission_ready: bool
    last_emission_id: str | None
    last_error: str | None
    rtc_no_action: bool


class ExclusiveHITLPolicyCommandSelector:
    """Select one source before the existing RealTeleop safety/dispatch path.

    Normal policy emissions are identity-composed through
    :mod:`control_composition` and resolved to the common ``TeleopCommand``.
    RTC consumes only publisher-validated committed emissions and remains
    deliberately no-action in this revision.
    """

    def __init__(
        self,
        *,
        policy_mode: str,
        initial_source: str,
        normal_policy_source: PolicyEmissionSourcePort | None = None,
        rtc_committed_emissions: queue.Queue[PolicyActionEmission] | None = None,
        runtime_session_id: str = "real-teleop-hitl",
        control_domain_id: str = "real-teleop",
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        mode = str(policy_mode).strip().lower()
        if mode not in HITL_POLICY_MODES:
            raise ValueError("policy_mode must be 'normal' or 'rtc'.")
        source = str(initial_source).strip().lower()
        if source not in {"human", "policy"}:
            raise ValueError("initial_source must be 'human' or 'policy'.")
        if mode == HITL_POLICY_MODE_NORMAL:
            if normal_policy_source is None:
                raise ValueError("normal policy mode requires a policy source.")
            normal_policy_source = require_policy_emission_source(
                normal_policy_source
            )
            if rtc_committed_emissions is not None:
                raise ValueError("normal policy mode cannot bind an RTC inbox.")
        else:
            if normal_policy_source is not None:
                raise ValueError("RTC policy mode cannot own a normal source.")
            if not isinstance(rtc_committed_emissions, queue.Queue):
                raise TypeError("RTC policy mode requires a bounded queue.Queue.")
            if int(rtc_committed_emissions.maxsize) <= 0:
                raise ValueError("RTC committed-emission inbox must be bounded.")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable.")
        self.policy_mode = mode
        self.selected_source = source
        self.normal_policy_source = normal_policy_source
        self.rtc_committed_emissions = rtc_committed_emissions
        self.runtime_session_id = str(runtime_session_id).strip()
        self.control_domain_id = str(control_domain_id).strip()
        if not self.runtime_session_id or not self.control_domain_id:
            raise ValueError("runtime session and control domain must be non-empty.")
        self.clock = monotonic_clock
        self._composer = ExclusiveSingleSourceComposer()
        self._resolver = RobotActionResolver()
        self._started = False
        self._closed = False
        self._step_index = 0
        self._policy_activation_pending = False
        self._activation_barrier_at: float | None = None
        self._cached_emission: PolicyActionEmission | None = None
        self._last_emission: PolicyActionEmission | None = None
        self._last_error: str | None = None
        self._switch_id: str | None = None
        self._intent_id: str | None = None
        self._intent_hash: str | None = None

    @property
    def rtc_no_action(self) -> bool:
        return self.policy_mode == HITL_POLICY_MODE_RTC

    @property
    def fresh_policy_emission_ready(self) -> bool:
        return self._cached_emission is not None

    @property
    def cached_emission(self) -> PolicyActionEmission | None:
        return self._cached_emission

    def status(self) -> HITLPolicySelectionStatus:
        source_error = (
            None
            if self.normal_policy_source is None
            else self.normal_policy_source.last_error
        )
        return HITLPolicySelectionStatus(
            selected_source=self.selected_source,
            policy_mode=self.policy_mode,
            policy_activation_pending=self._policy_activation_pending,
            fresh_policy_emission_ready=self.fresh_policy_emission_ready,
            last_emission_id=(
                None if self._last_emission is None else self._last_emission.emission_id
            ),
            last_error=self._last_error or source_error,
            rtc_no_action=self.rtc_no_action,
        )

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("HITL policy selector is closed.")
        if self._started:
            return
        if self.normal_policy_source is not None:
            self.normal_policy_source.start()
        self._started = True

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.normal_policy_source is not None:
                self.normal_policy_source.close()
        finally:
            self._started = False
            self._closed = True
            self._cached_emission = None

    def select_human(
        self,
        *,
        switch_id: str | None,
        intent_id: str | None,
        intent_hash: str | None,
    ) -> None:
        self.selected_source = "human"
        self._policy_activation_pending = False
        self._cached_emission = None
        self._bind_switch_lineage(
            switch_id=switch_id,
            intent_id=intent_id,
            intent_hash=intent_hash,
        )

    def reset_policy_for_activation(
        self,
        *,
        barrier_at: float,
        switch_id: str | None,
        intent_id: str | None,
        intent_hash: str | None,
    ) -> None:
        if not self._started:
            raise RuntimeError("HITL policy selector must be started before reset.")
        self._composer.reset_activation()
        self._cached_emission = None
        self._last_error = None
        self._policy_activation_pending = True
        self._activation_barrier_at = float(barrier_at)
        self._bind_switch_lineage(
            switch_id=switch_id,
            intent_id=intent_id,
            intent_hash=intent_hash,
        )
        if self.normal_policy_source is not None:
            self.normal_policy_source.reset_for_activation()
        else:
            assert self.rtc_committed_emissions is not None
            while True:
                try:
                    self.rtc_committed_emissions.get_nowait()
                except queue.Empty:
                    break

    def commit_policy_selection(self) -> None:
        if not self._policy_activation_pending or self._cached_emission is None:
            raise RuntimeError("A fresh policy emission is required before selection.")
        if self.rtc_no_action:
            raise RuntimeError(
                "RTC committed-emission integration is no-action in this revision."
            )
        self.selected_source = "policy"
        self._policy_activation_pending = False

    def _bind_switch_lineage(
        self,
        *,
        switch_id: str | None,
        intent_id: str | None,
        intent_hash: str | None,
    ) -> None:
        self._switch_id = None if switch_id is None else str(switch_id)
        self._intent_id = None if intent_id is None else str(intent_id)
        self._intent_hash = None if intent_hash is None else str(intent_hash)

    def _next_emission(self) -> PolicyActionEmission | None:
        if self.normal_policy_source is not None:
            emission = self.normal_policy_source.step(
                now=self.clock(),
                step_index=self._step_index,
            )
            self._step_index += 1
            return emission
        assert self.rtc_committed_emissions is not None
        latest = None
        while True:
            try:
                latest = self.rtc_committed_emissions.get_nowait()
            except queue.Empty:
                return latest

    def _poll_fresh_policy_emission(self) -> PolicyActionEmission | None:
        emission = self._next_emission()
        if emission is None:
            source_error = (
                None
                if self.normal_policy_source is None
                else self.normal_policy_source.last_error
            )
            if source_error:
                raise RuntimeError(
                    "HITL policy emission source failed: " + str(source_error)
                )
            return None
        barrier = self._activation_barrier_at
        if barrier is not None and float(emission.emitted_at) < barrier:
            return None
        if emission.pre_safety_robot_action is None or emission.pre_safety_faults:
            raise ValueError("Policy emission has no accepted pre-safety RobotAction.")
        self._last_emission = emission
        if self._policy_activation_pending:
            self._cached_emission = emission
        return emission

    def _annotate_common(self, command: TeleopCommand, *, source: str) -> None:
        command.metadata["control_source"] = source
        command.metadata["hitl_exclusive_source"] = source
        command.metadata["hitl_policy_mode"] = self.policy_mode
        command.metadata["hitl_rtc_no_action"] = self.rtc_no_action
        if self._switch_id is not None:
            command.metadata["hitl_switch_id"] = self._switch_id
        if self._intent_id is not None:
            command.metadata["hitl_intent_id"] = self._intent_id
        if self._intent_hash is not None:
            command.metadata["hitl_intent_hash"] = self._intent_hash

    def _policy_command(
        self,
        emission: PolicyActionEmission,
        *,
        human_candidate: TeleopCommand,
    ) -> TeleopCommand:
        action = emission.pre_safety_robot_action
        if action is None:
            raise ValueError("Policy emission has no pre-safety RobotAction.")
        composition = self._composer.compose(
            action,
            runtime_session_id=self.runtime_session_id,
            control_domain_id=self.control_domain_id,
            producer_id=emission.policy_node_id,
            producer_epoch=1,
            policy_output_id=emission.policy_output_id,
            produced_at=emission.produced_at,
            produced_clock_domain_id=emission.produced_clock_domain_id,
            available_at=emission.emitted_at,
            available_clock_domain_id=emission.effective_clock_domain_id,
            effective_from=emission.effective_from,
            effective_until=emission.effective_until,
            effective_clock_domain_id=emission.effective_clock_domain_id,
            target_effective_at=emission.target_effective_at,
            context_ids=tuple(
                value
                for value in (self._switch_id, self._intent_id)
                if value is not None
            ),
        )
        command = self._resolver.resolve(composition.action)
        command.timestamp = human_candidate.timestamp
        self._annotate_common(command, source="policy")
        command.metadata.update(
            {
                "policy_emission_id": emission.emission_id,
                "policy_emission_hash": emission.semantic_content_hash,
                "policy_output_id": emission.policy_output_id,
                "policy_inference_id": emission.inference_id,
                "policy_target_effective_at": emission.target_effective_at,
                "policy_effective_until": emission.effective_until,
                "hitl_composition_input_id": composition.composition_input_id,
                "hitl_composition_input_hash": composition.composition_input_hash,
                "hitl_composition_evaluation_id": (
                    composition.composition_evaluation_id
                ),
                "hitl_composition_evaluation_hash": (
                    composition.composition_evaluation_hash
                ),
                "hitl_composition_spec_hash": composition.composition_spec_hash,
                "hitl_composed_candidate_hash": composition.composed_candidate_hash,
            }
        )
        return command

    def __call__(
        self,
        observation: TeleopObservation,
        human_candidate: TeleopCommand,
        now: float,
    ) -> TeleopCommandOverride | None:
        del observation, now
        if not self._started:
            raise RuntimeError("HITL policy selector is not started.")
        if self._policy_activation_pending:
            if self._cached_emission is None:
                self._poll_fresh_policy_emission()
            return TeleopCommandOverride(None)
        if self.selected_source == "human":
            self._annotate_common(human_candidate, source="human")
            return None
        if self.rtc_no_action:
            self._poll_fresh_policy_emission()
            return TeleopCommandOverride(None)
        emission = self._cached_emission
        self._cached_emission = None
        if emission is None:
            emission = self._poll_fresh_policy_emission()
        if emission is None:
            return TeleopCommandOverride(None)
        try:
            return TeleopCommandOverride(
                self._policy_command(emission, human_candidate=human_candidate)
            )
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}:{exc}"
            raise


__all__ = [
    "HITL_POLICY_MODE_NORMAL",
    "HITL_POLICY_MODE_RTC",
    "HITL_POLICY_MODES",
    "ExclusiveHITLPolicyCommandSelector",
    "HITLPolicySelectionStatus",
    "build_normal_hitl_policy_source",
    "preflight_normal_hitl_policy_source",
]
