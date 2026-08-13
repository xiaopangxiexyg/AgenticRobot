"""Typed contracts for passive, non-dispatching TeleopSession sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ArmActionSpec, HandActionSpec, split_effector_name
from teleop_runtime.action.human_proposal import HumanCommandEmission
from teleop_runtime.session import SessionStepResult


PASSIVE_TELEOP_POSE_SOURCE_CONTRACT_ID_V1 = (
    "control_station_passive_teleop_bimanual_pose_v1"
)
# Preserve the historical ID for the default bilateral tuple. Other
# cardinalities use the generic ID and remain distinguished by their complete
# effector/frame/spec tuple in the semantic hash.
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
    """Device-neutral pose contract with declared effector cardinality."""

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
        """Construct one contract for any non-empty pose-effector set."""

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
        return tuple(effector for effector, _ in self.reference_frame_by_effector)

    @property
    def contract_id(self) -> str:
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
    """One canonical target component emitted by a passive session.

    The active HIL mainline uses Cartesian pose for arms and typed joint
    vectors for hands/grippers. Arm joint targets remain outside this profile.
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
    """Device- and backend-neutral binding for the current action topology."""

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


__all__ = [
    "PASSIVE_TELEOP_COMPONENT_BINDING_HASH_DOMAIN_V1",
    "PASSIVE_TELEOP_COMPONENT_SOURCE_CONTRACT_ID_V1",
    "PASSIVE_TELEOP_COMPONENT_SOURCE_HASH_DOMAIN_V1",
    "PASSIVE_TELEOP_POSE_BINDING_HASH_DOMAIN_V1",
    "PASSIVE_TELEOP_POSE_GENERIC_SOURCE_CONTRACT_ID_V1",
    "PASSIVE_TELEOP_POSE_SOURCE_CONTRACT_ID_V1",
    "PASSIVE_TELEOP_POSE_SOURCE_HASH_DOMAIN_V1",
    "PassiveTeleopComponentBindingReceipt",
    "PassiveTeleopComponentSourceSpec",
    "PassiveTeleopComponentSpec",
    "PassiveTeleopComponentStep",
    "PassiveTeleopPoseBindingReceipt",
    "PassiveTeleopPoseSourceSpec",
    "PassiveTeleopPoseStep",
]
