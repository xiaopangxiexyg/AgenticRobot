"""Observe-only pose emissions from an already-composed TeleopSession.

This application-layer wrapper is deliberately device neutral.  The caller
may provide VIVE, another tracker, or a virtual source, but the wrapped session
must have ``rig=None`` and must remain outside ARMED/RUNNING.  It therefore
cannot dispatch to either a simulator or a real robot.

The wrapper owns no worker.  The application calls :meth:`step`; the existing
TeleopSession polls and retargets once, while this module validates and fans out
the immutable ``HumanCommandEmission`` produced by its observer seam.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

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
from teleop_runtime.session import (
    SessionStepResult,
    TeleopSession,
    TeleopSessionConfig,
)

from .human import (
    PassiveHumanCommandEmissionIngress,
    PassiveHumanCommandObserverFanout,
)


PASSIVE_TELEOP_POSE_SOURCE_CONTRACT_ID_V1 = (
    "control_station_passive_teleop_bimanual_pose_v1"
)
# The historical constant above is kept as the compatibility identifier for
# the default RM65 bilateral tuple.  New cardinalities use this generic
# contract identifier; the actual effector/frame/spec tuple is always part of
# the semantic hash, so a single-arm source cannot accidentally bind to the
# bilateral profile.
PASSIVE_TELEOP_POSE_GENERIC_SOURCE_CONTRACT_ID_V1 = (
    "control_station_passive_teleop_pose_v1"
)
PASSIVE_TELEOP_POSE_SOURCE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.passive_teleop_pose_source.v1"
)
PASSIVE_TELEOP_POSE_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.passive_teleop_pose_binding.v1"
)
PASSIVE_TELEOP_COMPONENT_SOURCE_CONTRACT_ID_V1 = (
    "control_station_passive_teleop_components_v1"
)
PASSIVE_TELEOP_COMPONENT_SOURCE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.passive_teleop_component_source.v1"
)
PASSIVE_TELEOP_COMPONENT_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.passive_teleop_component_binding.v1"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


@dataclass(frozen=True)
class PassiveTeleopPoseSourceSpec:
    """Device-neutral pose source contract with declared effector cardinality.

    The tuple fields are intentionally data driven.  The default retains the
    historical RM65 bilateral contract, while ``for_effectors``/``for_arm``
    make a single-arm or another compatible topology explicit without adding a
    second HIL lifecycle.  Target-side components still have to match this
    declaration during adapter binding.
    """

    source_id: str = "passive-teleop-pose"
    observer_binding_id: str = "passive-teleop-pose-observer-v1"
    source_profile_id: str = "device-neutral-pose-session-v1"
    reference_frame_by_effector: tuple[tuple[str, str], ...] = (
        ("left_arm", "left_arm_base"),
        ("right_arm", "right_arm_base"),
    )
    pose_frame_contract_hash: str = "rm65-arm-base-tool-tcp-pose-frame-v1"
    action_spec_hash_by_effector: tuple[tuple[str, str], ...] = (
        ("left_arm", "rm65-tool-tcp-pose-v1"),
        ("right_arm", "rm65-tool-tcp-pose-v1"),
    )
    units_convention_id: str = "si_m_rad_xyzw_v1"

    @classmethod
    def for_effectors(
        cls,
        reference_frame_by_effector: Mapping[str, str]
        | Sequence[tuple[str, str]],
        *,
        action_spec_hash_by_effector: Mapping[str, str]
        | Sequence[tuple[str, str]],
        **kwargs: object,
    ) -> "PassiveTeleopPoseSourceSpec":
        """Construct one contract for any non-empty pose-effector set.

        Mappings are sorted by effector name to make the resulting contract
        hash independent of dictionary insertion order.  Tuple callers may
        retain an intentional order when they need legacy ordering evidence.
        """

        def _pairs(
            value: Mapping[str, str] | Sequence[tuple[str, str]],
            *,
            name: str,
        ) -> tuple[tuple[str, str], ...]:
            if isinstance(value, Mapping):
                return tuple(sorted((str(k), str(v)) for k, v in value.items()))
            try:
                return tuple((str(k), str(v)) for k, v in value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be a mapping or pair sequence.") from exc

        return cls(
            reference_frame_by_effector=_pairs(
                reference_frame_by_effector,
                name="reference_frame_by_effector",
            ),
            action_spec_hash_by_effector=_pairs(
                action_spec_hash_by_effector,
                name="action_spec_hash_by_effector",
            ),
            **kwargs,
        )

    @classmethod
    def for_arm(
        cls,
        effector_name: str,
        *,
        reference_frame: str,
        action_spec_hash: str,
        **kwargs: object,
    ) -> "PassiveTeleopPoseSourceSpec":
        """Convenience constructor for a one-arm or one-effector profile."""

        return cls.for_effectors(
            ((effector_name, reference_frame),),
            action_spec_hash_by_effector=((effector_name, action_spec_hash),),
            **kwargs,
        )

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "observer_binding_id",
            "source_profile_id",
            "pose_frame_contract_hash",
            "units_convention_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        frames = tuple(
            (_required(effector, name="effector"), _required(frame, name="frame"))
            for effector, frame in self.reference_frame_by_effector
        )
        if not frames:
            raise ValueError(
                "Passive teleop pose must declare at least one pose effector."
            )
        frame_effectors = tuple(effector for effector, _ in frames)
        if len(set(frame_effectors)) != len(frame_effectors):
            raise ValueError("Passive teleop pose effector names must be unique.")
        specs = tuple(
            (
                _required(effector, name="action spec effector"),
                _required(spec_hash, name="action spec hash"),
            )
            for effector, spec_hash in self.action_spec_hash_by_effector
        )
        spec_effectors = tuple(effector for effector, _ in specs)
        if len(set(spec_effectors)) != len(spec_effectors):
            raise ValueError("Pose action-spec effector names must be unique.")
        if set(spec_effectors) != set(frame_effectors):
            raise ValueError("Pose action specs must exactly cover the pose frames.")
        object.__setattr__(self, "reference_frame_by_effector", frames)
        object.__setattr__(self, "action_spec_hash_by_effector", specs)

    @property
    def effector_names(self) -> tuple[str, ...]:
        """Declared pose effectors in source-contract order."""

        return tuple(effector for effector, _ in self.reference_frame_by_effector)

    @property
    def contract_id(self) -> str:
        """Return a compatibility ID without hiding non-bilateral cardinality."""

        if self.effector_names == ("left_arm", "right_arm"):
            return PASSIVE_TELEOP_POSE_SOURCE_CONTRACT_ID_V1
        return PASSIVE_TELEOP_POSE_GENERIC_SOURCE_CONTRACT_ID_V1

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "contract_id": self.contract_id,
                "source_id": self.source_id,
                "observer_binding_id": self.observer_binding_id,
                "source_profile_id": self.source_profile_id,
                "reference_frame_by_effector": self.reference_frame_by_effector,
                "pose_frame_contract_hash": self.pose_frame_contract_hash,
                "action_spec_hash_by_effector": self.action_spec_hash_by_effector,
                "units_convention_id": self.units_convention_id,
            },
            domain=PASSIVE_TELEOP_POSE_SOURCE_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class PassiveTeleopComponentSpec:
    """One canonical target component emitted by a passive teleop session.

    This first component contract intentionally covers only the active HIL
    mainline: arm targets are Cartesian poses and hand/gripper targets are
    typed joint vectors.  Arm joint targets remain outside the mainline.
    """

    component_id: str
    effector_name: str
    command_type: str
    action_spec_hash: str
    arm_action_spec: ArmActionSpec | Mapping[str, object] | None = None
    hand_action_spec: HandActionSpec | Mapping[str, object] | None = None

    @classmethod
    def arm_pose(
        cls,
        component_id: str,
        effector_name: str,
        *,
        action_spec_hash: str,
        arm_action_spec: ArmActionSpec | Mapping[str, object],
    ) -> "PassiveTeleopComponentSpec":
        return cls(
            component_id=component_id,
            effector_name=effector_name,
            command_type="cartesian_pose",
            action_spec_hash=action_spec_hash,
            arm_action_spec=arm_action_spec,
        )

    @classmethod
    def hand_qpos(
        cls,
        component_id: str,
        effector_name: str,
        *,
        action_spec_hash: str,
        hand_action_spec: HandActionSpec | Mapping[str, object],
    ) -> "PassiveTeleopComponentSpec":
        return cls(
            component_id=component_id,
            effector_name=effector_name,
            command_type="joint_vector",
            action_spec_hash=action_spec_hash,
            hand_action_spec=hand_action_spec,
        )

    def __post_init__(self) -> None:
        for name in ("component_id", "effector_name", "command_type", "action_spec_hash"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        try:
            side, kind = split_effector_name(self.effector_name)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported passive teleop effector {self.effector_name!r}."
            ) from exc

        arm_spec = self.arm_action_spec
        if isinstance(arm_spec, Mapping):
            arm_spec = ArmActionSpec.from_metadata(arm_spec)
        hand_spec = self.hand_action_spec
        if isinstance(hand_spec, Mapping):
            hand_spec = HandActionSpec.from_metadata(hand_spec)

        if self.command_type == "cartesian_pose":
            if kind != "arm":
                raise ValueError("cartesian_pose components must target an arm effector.")
            if not isinstance(arm_spec, ArmActionSpec):
                raise ValueError(
                    "cartesian_pose components require an explicit ArmActionSpec."
                )
            if arm_spec.side != side:
                raise ValueError("ArmActionSpec side must match the component effector.")
            if hand_spec is not None:
                raise ValueError("cartesian_pose components cannot declare HandActionSpec.")
        elif self.command_type == "joint_vector":
            if kind != "hand":
                raise ValueError(
                    "The v1 HIL joint_vector component is hand/gripper only; "
                    "arm qpos mixing is not a mainline representation."
                )
            if not isinstance(hand_spec, HandActionSpec):
                raise ValueError(
                    "hand joint_vector components require an explicit HandActionSpec."
                )
            if hand_spec.side != side:
                raise ValueError("HandActionSpec side must match the component effector.")
            if arm_spec is not None:
                raise ValueError("hand joint_vector components cannot declare ArmActionSpec.")
        else:
            raise ValueError(
                "Passive teleop component command_type must be cartesian_pose "
                "or joint_vector."
            )
        object.__setattr__(self, "arm_action_spec", arm_spec)
        object.__setattr__(self, "hand_action_spec", hand_spec)

    @property
    def semantic_payload(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "effector_name": self.effector_name,
            "command_type": self.command_type,
            "action_spec_hash": self.action_spec_hash,
            "arm_action_spec": (
                None
                if self.arm_action_spec is None
                else self.arm_action_spec.to_metadata()
            ),
            "hand_action_spec": (
                None
                if self.hand_action_spec is None
                else self.hand_action_spec.to_metadata()
            ),
        }


@dataclass(frozen=True)
class PassiveTeleopComponentSourceSpec:
    """Device- and backend-neutral binding for the current action-v1 topology."""

    components: tuple[PassiveTeleopComponentSpec, ...]
    source_id: str = "passive-teleop-components"
    observer_binding_id: str = "passive-teleop-components-observer-v1"
    source_profile_id: str = "device-neutral-component-session-v1"
    units_convention_id: str = "si_m_rad_xyzw_v1"

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "observer_binding_id",
            "source_profile_id",
            "units_convention_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        components = tuple(self.components)
        if not components:
            raise ValueError("Passive teleop source must declare at least one component.")
        if not all(isinstance(item, PassiveTeleopComponentSpec) for item in components):
            raise TypeError("components must contain PassiveTeleopComponentSpec values.")
        component_ids = tuple(item.component_id for item in components)
        effectors = tuple(item.effector_name for item in components)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("Passive teleop component IDs must be unique.")
        if len(set(effectors)) != len(effectors):
            raise ValueError("Passive teleop component effectors must be unique.")
        object.__setattr__(self, "components", components)

    @property
    def effector_names(self) -> tuple[str, ...]:
        return tuple(item.effector_name for item in self.components)

    @property
    def hand_action_specs_by_side(self) -> dict[str, HandActionSpec]:
        result: dict[str, HandActionSpec] = {}
        for item in self.components:
            if item.hand_action_spec is not None:
                result[item.hand_action_spec.side] = item.hand_action_spec
        return result

    @property
    def arm_action_specs_by_side(self) -> dict[str, ArmActionSpec]:
        result: dict[str, ArmActionSpec] = {}
        for item in self.components:
            if item.arm_action_spec is not None:
                result[item.arm_action_spec.side] = item.arm_action_spec
        return result

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "contract_id": PASSIVE_TELEOP_COMPONENT_SOURCE_CONTRACT_ID_V1,
                "source_id": self.source_id,
                "observer_binding_id": self.observer_binding_id,
                "source_profile_id": self.source_profile_id,
                "components": tuple(
                    item.semantic_payload for item in self.components
                ),
                "units_convention_id": self.units_convention_id,
            },
            domain=PASSIVE_TELEOP_COMPONENT_SOURCE_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class PassiveTeleopPoseBindingReceipt:
    source_contract_hash: str
    observer_binding_id: str
    session_mode: str
    rig_absent: bool
    observer_exclusive: bool
    dispatch_forbidden_states: tuple[str, ...]
    authoritative: bool
    semantic_content_hash: str


@dataclass(frozen=True)
class PassiveTeleopPoseStep:
    session_step: SessionStepResult
    emission: HumanCommandEmission | None
    emission_valid: bool
    rejection_reason: str | None
    command_dispatched: bool
    authoritative: bool = False


@dataclass(frozen=True)
class PassiveTeleopComponentBindingReceipt:
    source_contract_hash: str
    observer_binding_id: str
    session_mode: str
    rig_absent: bool
    observer_exclusive: bool
    dispatch_forbidden_states: tuple[str, ...]
    authoritative: bool
    semantic_content_hash: str


@dataclass(frozen=True)
class PassiveTeleopComponentStep:
    session_step: SessionStepResult
    emission: HumanCommandEmission | None
    emission_valid: bool
    rejection_reason: str | None
    command_dispatched: bool
    authoritative: bool = False


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

    def start(self) -> object:
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    f"Observe-only teleop {self._source_label} source is closed."
                )
            self._assert_non_dispatch_state()
            self._session.start()
            self._assert_non_dispatch_state()
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
            if observed.to_metadata() != arm_spec.to_metadata():
                raise ValueError(
                    f"{component.effector_name} arm_action_spec does not match "
                    "the bound component contract."
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


__all__ = [
    "PASSIVE_TELEOP_COMPONENT_SOURCE_CONTRACT_ID_V1",
    "PASSIVE_TELEOP_POSE_GENERIC_SOURCE_CONTRACT_ID_V1",
    "PASSIVE_TELEOP_POSE_SOURCE_CONTRACT_ID_V1",
    "PassiveTeleopComponentBindingReceipt",
    "PassiveTeleopComponentSessionSource",
    "PassiveTeleopComponentSourceSpec",
    "PassiveTeleopComponentSpec",
    "PassiveTeleopComponentStep",
    "PassiveTeleopPoseBindingReceipt",
    "PassiveTeleopPoseSessionSource",
    "PassiveTeleopPoseSourceSpec",
    "PassiveTeleopPoseStep",
    "build_passive_teleop_component_session_source",
    "build_passive_teleop_pose_session_source",
]
