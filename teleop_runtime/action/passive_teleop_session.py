"""Observe-only Human session runtime over an existing TeleopSession.

The caller may provide VIVE, another tracker, or a virtual source, but the
wrapped session must have ``rig=None`` and remain outside ARMED/RUNNING. It
therefore cannot dispatch to either a simulator or a real robot.

The wrapper owns no worker.  The application calls :meth:`step`; the existing
TeleopSession polls and retargets once, while this module validates and fans out
the immutable ``HumanCommandEmission`` produced by its observer seam.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ArmActionSpec, HandActionSpec, split_effector_name
from robot_motion.diagnostics import TeleopRunState
from teleop_runtime.calibration import TeleopCalibration
from teleop_runtime.interfaces import ObservationSource, Retargeter
from teleop_runtime.safety import TeleopSafetyConfig
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanEffectorCommandSnapshot,
)
from teleop_runtime.session import TeleopSession, TeleopSessionConfig
from .human_observer import (
    PassiveHumanCommandEmissionIngress,
    PassiveHumanCommandObserverFanout,
)


from .passive_teleop_contracts import (
    PASSIVE_TELEOP_COMPONENT_BINDING_HASH_DOMAIN_V1,
    PASSIVE_TELEOP_POSE_BINDING_HASH_DOMAIN_V1,
    PassiveTeleopComponentBindingReceipt,
    PassiveTeleopComponentSourceSpec,
    PassiveTeleopComponentSpec,
    PassiveTeleopComponentStep,
    PassiveTeleopPoseBindingReceipt,
    PassiveTeleopPoseSourceSpec,
    PassiveTeleopPoseStep,
    _required,
)


class _PassiveTeleopSessionSource:
    """Shared non-dispatching TeleopSession lifecycle and observer fanout."""

    _source_label = "teleop"
    _spec_type: type[object]
    _binding_receipt_type: type[object]
    _step_type: type[object]
    _binding_hash_domain: str

    def __init__(
        self,
        session: TeleopSession,
        spec: object,
        *,
        queue_capacity: int = 8,
        passive_subscribers: Mapping[
            str,
            Callable[[HumanCommandEmission], object],
        ]
        | None = None,
    ) -> None:
        if not isinstance(session, TeleopSession):
            raise TypeError("session must be a TeleopSession.")
        if not isinstance(spec, self._spec_type):
            raise TypeError(f"spec must be a {self._spec_type.__name__}.")
        if session.rig is not None:
            raise ValueError(
                f"Observe-only teleop {self._source_label} session requires rig=None."
            )
        if session.human_command_observer is not None:
            raise ValueError(
                "Observe-only wrapper requires exclusive observer injection at build time."
            )
        if session.recorder is not None:
            raise ValueError(
                "Observe-only source cannot own a TeleopSession recorder/episode sink."
            )
        self.spec = spec
        self._session = session
        self.ingress = PassiveHumanCommandEmissionIngress(
            spec.observer_binding_id,
            queue_capacity=queue_capacity,
        )
        subscribers = {"station_ingress": self.ingress.observe_emission}
        for subscriber_id, callback in dict(passive_subscribers or {}).items():
            name = _required(subscriber_id, name="subscriber_id")
            if name in subscribers:
                raise ValueError(f"Reserved passive subscriber ID {name!r}.")
            subscribers[name] = callback
        self.fanout = PassiveHumanCommandObserverFanout(subscribers)
        self._lock = threading.RLock()
        self._last_observed_sequence: int | None = None
        self._last_accepted_emission: HumanCommandEmission | None = None
        self._last_rejection_reason: str | None = None
        self._started = False
        self._closed = False
        session.human_command_observer = self._observe_emission
        payload = {
            "source_contract_hash": spec.semantic_content_hash,
            "observer_binding_id": spec.observer_binding_id,
            "session_mode": session.config.mode,
            "rig_absent": True,
            "observer_exclusive": True,
            "dispatch_forbidden_states": (
                TeleopRunState.ARMED.value,
                TeleopRunState.RUNNING.value,
            ),
            "authoritative": False,
        }
        self.binding_receipt = self._binding_receipt_type(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=self._binding_hash_domain,
            ),
        )

    @property
    def session_state(self) -> TeleopRunState:
        return self._session.state_machine.state

    @property
    def started(self) -> bool:
        """Whether this passive source has started its lower TeleopSession."""

        with self._lock:
            return bool(self._started and not self._closed)

    @property
    def session(self) -> TeleopSession:
        """Expose the lower-owned session for read-only station readiness.

        The returned object remains owned by ``teleop_runtime``.  This accessor
        is intentionally observation-only at the Station boundary; callers may
        inspect state, but source lifecycle transitions still go through this
        passive source's typed methods.
        """

        return self._session

    @property
    def last_emission(self) -> HumanCommandEmission | None:
        """Return the latest accepted passive emission, if any."""

        with self._lock:
            return self._last_accepted_emission

    @property
    def last_observed_emission_sequence(self) -> int | None:
        """Return the newest emission sequence seen before validation."""

        with self._lock:
            return self._last_observed_sequence

    @property
    def last_rejection_reason(self) -> str | None:
        """Return the latest validation rejection, if any."""

        with self._lock:
            return self._last_rejection_reason

    def start(self) -> object:
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    f"Observe-only teleop {self._source_label} source is closed."
                )
            self._assert_non_dispatch_state()
            self._session.start()
            self._assert_non_dispatch_state()
            self._started = True
            return self.binding_receipt

    def set_calibration_ready(self) -> None:
        """Expose only the non-motion calibration transition needed for polling."""

        with self._lock:
            self._assert_non_dispatch_state()
            self._session.set_calibration_ready()
            self._assert_non_dispatch_state()

    def step(self, *, now: float | None = None) -> object:
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    f"Observe-only teleop {self._source_label} source is closed."
                )
            self._assert_non_dispatch_state()
            before = self._last_observed_sequence
            result = self._session.run_step(now=now)
            if result.command_dispatched:
                raise RuntimeError(
                    f"Observe-only teleop {self._source_label} session "
                    "unexpectedly dispatched a command."
                )
            self._assert_non_dispatch_state()
            emission = (
                self._last_accepted_emission
                if self._last_observed_sequence != before
                else None
            )
            rejection = (
                self._last_rejection_reason
                if self._last_observed_sequence != before and emission is None
                else None
            )
            return self._step_type(
                session_step=result,
                emission=emission,
                emission_valid=emission is not None,
                rejection_reason=rejection,
                command_dispatched=False,
            )

    def _observe_emission(self, emission: HumanCommandEmission) -> None:
        with self._lock:
            self._last_observed_sequence = emission.emission_sequence
            self._last_accepted_emission = None
            try:
                self._assert_non_dispatch_state()
                self._validate_emission(emission)
            except (TypeError, ValueError, RuntimeError) as error:
                self._last_rejection_reason = f"{type(error).__name__}:{error}"
                return
            self._last_rejection_reason = None
            self._last_accepted_emission = emission
        self.fanout.observe_emission(emission)

    def _validate_emission(self, emission: HumanCommandEmission) -> None:
        raise NotImplementedError

    def _validate_common_emission(
        self,
        emission: HumanCommandEmission,
    ) -> dict[str, HumanEffectorCommandSnapshot]:
        if not isinstance(emission, HumanCommandEmission):
            raise TypeError("Observer received a non-HumanCommandEmission value.")
        commands = {item.effector_name: item for item in emission.merged_commands}
        if len(commands) != len(emission.merged_commands):
            raise ValueError("Emission contains duplicate effector commands.")
        return commands

    def _validate_pose_emission(self, emission: HumanCommandEmission) -> None:
        commands = self._validate_common_emission(emission)
        expected_frames = dict(self.spec.reference_frame_by_effector)
        if set(commands) != set(expected_frames):
            unexpected = sorted(set(commands) - set(expected_frames))
            missing = sorted(set(expected_frames) - set(commands))
            detail = []
            if missing:
                detail.append("missing=" + ",".join(missing))
            if unexpected:
                detail.append("unexpected=" + ",".join(unexpected))
            raise ValueError(
                "Pose emission must contain exactly the declared pose effectors ("
                + "; ".join(detail)
                + "). Hand/qpos components require a separate typed component source."
            )
        for effector, expected_frame in expected_frames.items():
            command = commands[effector]
            if command.command_type != "cartesian_pose":
                raise ValueError(f"{effector} is not a cartesian_pose command.")
            if command.frame_name != expected_frame:
                raise ValueError(
                    f"{effector} frame {command.frame_name!r} does not match "
                    f"{expected_frame!r}."
                )
            pose = np.asarray(command.value, dtype=float)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                raise ValueError(f"{effector} pose must be a finite 4x4 matrix.")
            if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
                raise ValueError(f"{effector} pose is not homogeneous.")
            rotation = pose[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
                raise ValueError(f"{effector} pose rotation is not orthonormal.")
            if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
                raise ValueError(f"{effector} pose rotation determinant is not +1.")

    def _assert_non_dispatch_state(self) -> None:
        state = self._session.state_machine.state
        if state in {TeleopRunState.ARMED, TeleopRunState.RUNNING}:
            raise RuntimeError(
                f"Observe-only teleop {self._source_label} session entered "
                f"forbidden state {state.value}."
            )
        if self._session.rig is not None:
            raise RuntimeError(
                f"Observe-only teleop {self._source_label} session acquired a rig."
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._started = False
        self._session.close()


class PassiveTeleopPoseSessionSource(_PassiveTeleopSessionSource):
    """Run one existing TeleopSession only as a passive pose producer."""

    _source_label = "pose"
    _spec_type = PassiveTeleopPoseSourceSpec
    _binding_receipt_type = PassiveTeleopPoseBindingReceipt
    _step_type = PassiveTeleopPoseStep
    _binding_hash_domain = PASSIVE_TELEOP_POSE_BINDING_HASH_DOMAIN_V1

    def _validate_emission(self, emission: HumanCommandEmission) -> None:
        self._validate_pose_emission(emission)


class PassiveTeleopComponentSessionSource(_PassiveTeleopSessionSource):
    """Run one TeleopSession as a typed pose + hand-qpos producer."""

    _source_label = "component"
    _spec_type = PassiveTeleopComponentSourceSpec
    _binding_receipt_type = PassiveTeleopComponentBindingReceipt
    _step_type = PassiveTeleopComponentStep
    _binding_hash_domain = PASSIVE_TELEOP_COMPONENT_BINDING_HASH_DOMAIN_V1

    def _validate_emission(self, emission: HumanCommandEmission) -> None:
        commands = self._validate_common_emission(emission)
        components = {
            item.effector_name: item for item in self.spec.components
        }
        if set(commands) != set(components):
            unexpected = sorted(set(commands) - set(components))
            missing = sorted(set(components) - set(commands))
            detail = []
            if missing:
                detail.append("missing=" + ",".join(missing))
            if unexpected:
                detail.append("unexpected=" + ",".join(unexpected))
            raise ValueError(
                "Component emission must exactly match the declared effectors ("
                + "; ".join(detail)
                + ")."
            )
        for effector, component in components.items():
            command = commands[effector]
            side, _ = split_effector_name(effector)
            if command.command_name != effector:
                raise ValueError(f"{effector} command key does not match its effector.")
            if command.side != side:
                raise ValueError(f"{effector} side does not match its effector.")
            if command.command_type != component.command_type:
                raise ValueError(
                    f"{effector} command_type {command.command_type!r} does not "
                    f"match {component.command_type!r}."
                )
            if component.command_type == "cartesian_pose":
                self._validate_component_pose(command, component)
            else:
                self._validate_component_hand_qpos(command, component)

    @staticmethod
    def _validate_component_pose(
        command: HumanEffectorCommandSnapshot,
        component: PassiveTeleopComponentSpec,
    ) -> None:
        arm_spec = component.arm_action_spec
        if command.frame_name != arm_spec.reference_frame:
            raise ValueError(
                f"{component.effector_name} frame {command.frame_name!r} does "
                f"not match {arm_spec.reference_frame!r}."
            )
        pose = np.asarray(command.value, dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError(
                f"{component.effector_name} pose must be a finite 4x4 matrix."
            )
        if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
            raise ValueError(f"{component.effector_name} pose is not homogeneous.")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError(
                f"{component.effector_name} pose rotation is not orthonormal."
            )
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
            raise ValueError(
                f"{component.effector_name} pose rotation determinant is not +1."
            )
        payload = command.metadata.get("arm_action_spec")
        if payload is not None:
            try:
                observed = (
                    payload
                    if isinstance(payload, ArmActionSpec)
                    else ArmActionSpec.from_metadata(payload)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{component.effector_name} has invalid arm_action_spec metadata."
                ) from exc
            expected_metadata = arm_spec.to_metadata()
            observed_metadata = observed.to_metadata()
            if observed_metadata != expected_metadata:
                differing_fields = tuple(
                    field
                    for field in sorted(
                        set(expected_metadata) | set(observed_metadata)
                    )
                    if observed_metadata.get(field) != expected_metadata.get(field)
                )
                raise ValueError(
                    f"{component.effector_name} arm_action_spec does not match "
                    "the bound component contract; differing_fields="
                    f"{','.join(differing_fields) or 'unknown'}."
                )

    @staticmethod
    def _validate_component_hand_qpos(
        command: HumanEffectorCommandSnapshot,
        component: PassiveTeleopComponentSpec,
    ) -> None:
        hand_spec = component.hand_action_spec
        values = np.asarray(command.value, dtype=float).reshape(-1)
        if values.shape != (hand_spec.slot_count,):
            raise ValueError(
                f"{component.effector_name} joint_vector must have canonical "
                f"slot shape ({hand_spec.slot_count},), got {values.shape}."
            )
        if np.any(np.isinf(values)):
            raise ValueError(
                f"{component.effector_name} joint_vector contains infinite values."
            )
        try:
            hand_spec.validate_vector(values)
        except ValueError as exc:
            raise ValueError(f"{component.effector_name} {exc}") from exc
        payload = command.metadata.get("hand_action_spec")
        if payload is not None:
            try:
                observed = (
                    payload
                    if isinstance(payload, HandActionSpec)
                    else HandActionSpec.from_metadata(payload)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{component.effector_name} has invalid hand_action_spec metadata."
                ) from exc
            if observed.to_metadata() != hand_spec.to_metadata():
                raise ValueError(
                    f"{component.effector_name} hand_action_spec does not match "
                    "the bound component contract."
                )


def build_passive_teleop_pose_session_source(
    spec: PassiveTeleopPoseSourceSpec,
    *,
    sources: Sequence[ObservationSource],
    retargeters: Sequence[Retargeter],
    calibration: TeleopCalibration | None = None,
    safety_config: TeleopSafetyConfig | None = None,
    session_config: TeleopSessionConfig | None = None,
    queue_capacity: int = 8,
    passive_subscribers: Mapping[
        str,
        Callable[[HumanCommandEmission], object],
    ]
    | None = None,
) -> PassiveTeleopPoseSessionSource:
    """Compose an arbitrary teleop device as a non-dispatching pose source.

    Device construction remains with the application profile.  This helper
    only creates the common ``TeleopSession(rig=None)`` boundary, so VIVE,
    another tracker, and virtual sources share the same lifecycle and pose
    validation without importing a concrete device here.
    """

    source_items = tuple(sources)
    retargeter_items = tuple(retargeters)
    if not source_items:
        raise ValueError("Passive teleop pose composition requires a source.")
    if not retargeter_items:
        raise ValueError("Passive teleop pose composition requires retargeters.")
    config = session_config or TeleopSessionConfig(
        mode="control-station-passive-pose",
        record_enabled=False,
        allow_hardware_after_debug_pass=False,
    )
    if config.record_enabled:
        raise ValueError("Passive teleop pose composition forbids session recording.")
    if config.allow_hardware_after_debug_pass:
        raise ValueError(
            "Passive teleop pose composition forbids hardware-arm eligibility."
        )
    session = TeleopSession(
        config,
        sources=source_items,
        retargeters=retargeter_items,
        rig=None,
        calibration=calibration,
        safety_config=safety_config,
        recorder=None,
    )
    return PassiveTeleopPoseSessionSource(
        session,
        spec,
        queue_capacity=queue_capacity,
        passive_subscribers=passive_subscribers,
    )


def build_passive_teleop_component_session_source(
    spec: PassiveTeleopComponentSourceSpec,
    *,
    sources: Sequence[ObservationSource],
    retargeters: Sequence[Retargeter],
    calibration: TeleopCalibration | None = None,
    safety_config: TeleopSafetyConfig | None = None,
    session_config: TeleopSessionConfig | None = None,
    queue_capacity: int = 8,
    passive_subscribers: Mapping[
        str,
        Callable[[HumanCommandEmission], object],
    ]
    | None = None,
) -> PassiveTeleopComponentSessionSource:
    """Compose a device as a non-dispatching pose/hand-qpos source.

    Device polling and retargeting remain lower-level seams.  The component
    contract only validates their canonical target emissions and never owns a
    rig, dispatcher, recorder, authority store, or episode lifecycle.
    """

    source_items = tuple(sources)
    retargeter_items = tuple(retargeters)
    if not source_items:
        raise ValueError("Passive teleop component composition requires a source.")
    if not retargeter_items:
        raise ValueError(
            "Passive teleop component composition requires retargeters."
        )
    config = session_config or TeleopSessionConfig(
        mode="control-station-passive-components",
        record_enabled=False,
        allow_hardware_after_debug_pass=False,
    )
    if config.record_enabled:
        raise ValueError(
            "Passive teleop component composition forbids session recording."
        )
    if config.allow_hardware_after_debug_pass:
        raise ValueError(
            "Passive teleop component composition forbids hardware-arm eligibility."
        )
    session = TeleopSession(
        config,
        sources=source_items,
        retargeters=retargeter_items,
        rig=None,
        calibration=calibration,
        safety_config=safety_config,
        recorder=None,
    )
    return PassiveTeleopComponentSessionSource(
        session,
        spec,
        queue_capacity=queue_capacity,
        passive_subscribers=passive_subscribers,
    )


class PassiveTeleopComponentClockBridge:
    """Adapt monotonic composition ticks to the session's wall-clock poll API."""

    def __init__(self, source: PassiveTeleopComponentSessionSource, *, wall_clock=time.time) -> None:
        if not isinstance(source, PassiveTeleopComponentSessionSource):
            raise TypeError(
                "source must be a PassiveTeleopComponentSessionSource."
            )
        if not callable(wall_clock):
            raise TypeError("wall_clock must be callable.")
        self._source = source
        self._wall_clock = wall_clock

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    @property
    def binding_receipt(self) -> PassiveTeleopComponentBindingReceipt:
        return self._source.binding_receipt

    @property
    def spec(self) -> PassiveTeleopComponentSourceSpec:
        return self._source.spec

    @property
    def session(self) -> TeleopSession:
        return self._source.session

    @property
    def session_state(self) -> TeleopRunState:
        return self._source.session_state

    @property
    def started(self) -> bool:
        return self._source.started

    @property
    def last_emission(self) -> HumanCommandEmission | None:
        return self._source.last_emission

    def start(self) -> object:
        return self._source.start()

    def set_calibration_ready(self) -> None:
        self._source.set_calibration_ready()

    def step(self, *, now: float | None = None) -> PassiveTeleopComponentStep:
        del now
        return self._source.step(now=float(self._wall_clock()))

    def close(self) -> None:
        self._source.close()

__all__ = [
    "PassiveTeleopComponentClockBridge",
    "PassiveTeleopComponentSessionSource",
    "PassiveTeleopPoseSessionSource",
    "build_passive_teleop_component_session_source",
    "build_passive_teleop_pose_session_source",
]
