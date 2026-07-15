"""Canonical RobotAction schema and per-effector DOF constants.

Arm pose vectors are always ``xyz + quat_xyzw`` with 7 values. RM65 arm qpos
vectors are 6 values. The default hand contract is q6; hand16 action spaces
use 16 slots and may contain NaN padding for inactive hand slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from robot_motion.frames import TOOL_TCP_FRAME, canonical_arm_controlled_frame
from robot_motion.runtime import TeleopCommand


VALID_SIDES: tuple[str, ...] = ("left", "right")
ARM_POSE_DOF = 7
RM65_ARM_DOF = 6
INSPIRE_HAND_DOF = 6
HAND16_SLOT_DOF = 16

BIMANUAL_EE_POSE_V1 = "bimanual_ee_pose_v1"
BIMANUAL_QPOS_V1 = "bimanual_qpos_v1"
BIMANUAL_MIXED_V1 = "bimanual_mixed_v1"
BIMANUAL_EE_POSE_HAND16_V1 = "bimanual_ee_pose_hand16_v1"
BIMANUAL_QPOS_HAND16_V1 = "bimanual_qpos_hand16_v1"
BIMANUAL_MIXED_HAND16_V1 = "bimanual_mixed_hand16_v1"

HAND16_ACTION_SPACES = frozenset(
    {
        BIMANUAL_EE_POSE_HAND16_V1,
        BIMANUAL_QPOS_HAND16_V1,
        BIMANUAL_MIXED_HAND16_V1,
    }
)


def normalize_side(side: str) -> str:
    value = str(side)
    if value not in VALID_SIDES:
        raise ValueError(f"Unsupported side '{side}'. Expected one of {VALID_SIDES}.")
    return value


def effector_name(side: str, kind: str) -> str:
    side = normalize_side(side)
    if kind not in {"arm", "hand"}:
        raise ValueError(f"Unsupported effector kind '{kind}'.")
    return f"{side}_{kind}"


def split_effector_name(name: str) -> tuple[str, str]:
    value = str(name)
    for side in VALID_SIDES:
        for kind in ("arm", "hand"):
            if value == effector_name(side, kind):
                return side, kind
    raise ValueError(f"Unsupported effector name '{name}'.")


def default_arm_frame(side: str) -> str:
    return f"{normalize_side(side)}_arm_base"


def default_arm_controlled_frame(side: str | None = None) -> str:
    if side is not None:
        normalize_side(side)
    return TOOL_TCP_FRAME


def hand_dof_for_action_space(action_space: str) -> int:
    if str(action_space) in HAND16_ACTION_SPACES:
        return HAND16_SLOT_DOF
    return INSPIRE_HAND_DOF


def _vector(value: Any, *, length: int, name: str, allow_nan: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (int(length),):
        raise ValueError(f"{name} must have shape ({int(length)},), got {array.shape}.")
    if allow_nan:
        if np.any(np.isinf(array)):
            raise ValueError(f"{name} contains infinite values.")
        if not np.any(np.isfinite(array)):
            raise ValueError(f"{name} must contain at least one finite active slot.")
    elif not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array.astype(np.float32, copy=True)


def _normalize_side_vectors(
    values: Mapping[str, Any] | None,
    *,
    length: int,
    label: str,
    allow_nan: bool = False,
) -> dict[str, np.ndarray]:
    if not values:
        return {}
    normalized: dict[str, np.ndarray] = {}
    for side, value in values.items():
        normalized_side = normalize_side(side)
        normalized[normalized_side] = _vector(
            value,
            length=length,
            name=f"{label}.{normalized_side}",
            allow_nan=allow_nan,
        )
    return normalized


def _normalize_effectors(values: tuple[str, ...] | list[str] | set[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    normalized = []
    for value in values:
        side, kind = split_effector_name(str(value))
        normalized.append(effector_name(side, kind))
    return tuple(dict.fromkeys(normalized))


def _derive_active_effectors(
    *,
    arm_pose_by_side: Mapping[str, np.ndarray],
    arm_qpos_by_side: Mapping[str, np.ndarray],
    hand_qpos_by_side: Mapping[str, np.ndarray],
    end_effector_targets_by_side: Mapping[str, "EndEffectorTarget"],
) -> tuple[str, ...]:
    effectors: list[str] = []
    for side in VALID_SIDES:
        if side in arm_pose_by_side or side in arm_qpos_by_side:
            effectors.append(effector_name(side, "arm"))
        if side in hand_qpos_by_side or side in end_effector_targets_by_side:
            effectors.append(effector_name(side, "hand"))
    return tuple(effectors)


@dataclass(frozen=True)
class EndEffectorTarget:
    """Typed, adapter-bound native end-effector target for v2.2 actions."""

    value_type: str
    values: Any
    adapter: str
    names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        value_type = str(self.value_type).strip().lower()
        if value_type not in {"continuous", "discrete"}:
            raise ValueError("EndEffectorTarget.value_type must be continuous or discrete.")
        adapter = str(self.adapter).strip()
        if not adapter:
            raise ValueError("EndEffectorTarget.adapter must be non-empty.")
        if value_type == "continuous":
            values = np.asarray(self.values, dtype=np.float32).reshape(-1)
            if values.size <= 0 or not np.all(np.isfinite(values)):
                raise ValueError("Continuous end-effector targets must be non-empty and finite.")
        else:
            raw = np.asarray(self.values).reshape(-1)
            if raw.size <= 0:
                raise ValueError("Discrete end-effector targets must be non-empty.")
            try:
                values = raw.astype(np.int64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Discrete end-effector targets must contain int64 tokens.") from exc
            if not np.array_equal(raw, values):
                raise ValueError("Discrete end-effector targets must contain exact integer tokens.")
        names = tuple(str(value).strip() for value in self.names)
        if names and (len(names) != values.size or any(not value for value in names)):
            raise ValueError("EndEffectorTarget.names must match the target dimension.")
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EndEffectorTarget":
        return cls(
            value_type=str(payload["value_type"]),
            values=payload["values"],
            adapter=str(payload["adapter"]),
            names=tuple(payload.get("names", ())),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class RobotAction:
    """Canonical action shared by collection, replay, policy deploy, sim, and real."""

    action_space: str
    timestamp: float | None = None
    arm_pose_by_side: Mapping[str, Any] = field(default_factory=dict)
    arm_qpos_by_side: Mapping[str, Any] = field(default_factory=dict)
    hand_qpos_by_side: Mapping[str, Any] = field(default_factory=dict)
    end_effector_targets_by_side: Mapping[
        str, EndEffectorTarget | Mapping[str, Any]
    ] = field(default_factory=dict)
    active_effectors: tuple[str, ...] = ()
    frame_by_effector: Mapping[str, str] = field(default_factory=dict)
    controlled_frame_by_effector: Mapping[str, str] = field(default_factory=dict)
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action_space = str(self.action_space)
        if not action_space:
            raise ValueError("action_space must be non-empty.")
        hand_dof = hand_dof_for_action_space(action_space)
        arm_pose = _normalize_side_vectors(
            self.arm_pose_by_side,
            length=ARM_POSE_DOF,
            label="arm_pose",
        )
        arm_qpos = _normalize_side_vectors(
            self.arm_qpos_by_side,
            length=RM65_ARM_DOF,
            label="arm_qpos",
        )
        hand_qpos = _normalize_side_vectors(
            self.hand_qpos_by_side,
            length=hand_dof,
            label="hand_qpos",
            allow_nan=hand_dof == HAND16_SLOT_DOF,
        )
        end_effector_targets: dict[str, EndEffectorTarget] = {}
        for raw_side, raw_target in dict(self.end_effector_targets_by_side).items():
            side = normalize_side(str(raw_side))
            if isinstance(raw_target, EndEffectorTarget):
                target = raw_target
            elif isinstance(raw_target, Mapping):
                target = EndEffectorTarget.from_dict(raw_target)
            else:
                raise TypeError(
                    "end_effector_targets_by_side values must be EndEffectorTarget or mappings."
                )
            end_effector_targets[side] = target
        ambiguous_end_effectors = sorted(set(hand_qpos) & set(end_effector_targets))
        if ambiguous_end_effectors:
            raise ValueError(
                "An end effector cannot provide both legacy hand_qpos and a native target: "
                + ",".join(ambiguous_end_effectors)
            )
        ambiguous_sides = sorted(set(arm_pose) & set(arm_qpos))
        if ambiguous_sides:
            raise ValueError(
                "An arm action cannot provide both pose and qpos for the same side: "
                + ",".join(ambiguous_sides)
            )

        active = _normalize_effectors(self.active_effectors)
        if not active:
            active = _derive_active_effectors(
                arm_pose_by_side=arm_pose,
                arm_qpos_by_side=arm_qpos,
                hand_qpos_by_side=hand_qpos,
                end_effector_targets_by_side=end_effector_targets,
            )

        for effector in active:
            side, kind = split_effector_name(effector)
            if kind == "arm" and side not in arm_pose and side not in arm_qpos:
                raise ValueError(f"Active effector '{effector}' has no arm action value.")
            if kind == "hand" and side not in hand_qpos and side not in end_effector_targets:
                raise ValueError(f"Active effector '{effector}' has no hand action value.")

        frames = {str(key): str(value) for key, value in dict(self.frame_by_effector).items()}
        for side in arm_pose:
            frames.setdefault(effector_name(side, "arm"), default_arm_frame(side))
        controlled_frames = {
            str(key): canonical_arm_controlled_frame(value)
            for key, value in dict(self.controlled_frame_by_effector).items()
        }
        metadata = dict(self.metadata)
        arm_specs_payload = metadata.get("arm_action_specs_by_side")
        if isinstance(arm_specs_payload, Mapping):
            for side, payload in arm_specs_payload.items():
                normalized_side = normalize_side(str(side))
                if not isinstance(payload, Mapping):
                    continue
                frame = payload.get("controlled_frame")
                if frame is not None:
                    controlled_frames.setdefault(
                        effector_name(normalized_side, "arm"),
                        canonical_arm_controlled_frame(str(frame)),
                    )
                reference_frame = payload.get("reference_frame")
                if reference_frame is not None:
                    frames.setdefault(effector_name(normalized_side, "arm"), str(reference_frame))
        for side in set(arm_pose) | set(arm_qpos):
            controlled_frames.setdefault(
                effector_name(side, "arm"),
                default_arm_controlled_frame(side),
            )

        object.__setattr__(self, "action_space", action_space)
        object.__setattr__(self, "arm_pose_by_side", arm_pose)
        object.__setattr__(self, "arm_qpos_by_side", arm_qpos)
        object.__setattr__(self, "hand_qpos_by_side", hand_qpos)
        object.__setattr__(self, "end_effector_targets_by_side", end_effector_targets)
        object.__setattr__(self, "active_effectors", active)
        object.__setattr__(self, "frame_by_effector", frames)
        object.__setattr__(self, "controlled_frame_by_effector", controlled_frames)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "metadata", metadata)

    @property
    def active_sides(self) -> tuple[str, ...]:
        sides = [split_effector_name(effector)[0] for effector in self.active_effectors]
        return tuple(side for side in VALID_SIDES if side in set(sides))

    def is_active(self, effector: str) -> bool:
        return str(effector) in set(self.active_effectors)

    def arm_frame(self, side: str) -> str:
        return self.frame_by_effector.get(effector_name(side, "arm"), default_arm_frame(side))

    def arm_controlled_frame(self, side: str) -> str:
        return self.controlled_frame_by_effector.get(
            effector_name(side, "arm"),
            default_arm_controlled_frame(side),
        )

    @classmethod
    def bimanual_ee_pose(
        cls,
        *,
        left_arm: Any | None = None,
        right_arm: Any | None = None,
        left_hand: Any | None = None,
        right_hand: Any | None = None,
        active_effectors: tuple[str, ...] = (),
        timestamp: float | None = None,
        source: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "RobotAction":
        arm_pose = {}
        hand_qpos = {}
        if left_arm is not None:
            arm_pose["left"] = left_arm
        if right_arm is not None:
            arm_pose["right"] = right_arm
        if left_hand is not None:
            hand_qpos["left"] = left_hand
        if right_hand is not None:
            hand_qpos["right"] = right_hand
        return cls(
            action_space=BIMANUAL_EE_POSE_V1,
            timestamp=timestamp,
            arm_pose_by_side=arm_pose,
            hand_qpos_by_side=hand_qpos,
            active_effectors=active_effectors,
            source=source,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def bimanual_ee_pose_hand16(
        cls,
        *,
        left_arm: Any | None = None,
        right_arm: Any | None = None,
        left_hand: Any | None = None,
        right_hand: Any | None = None,
        active_effectors: tuple[str, ...] = (),
        timestamp: float | None = None,
        source: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "RobotAction":
        arm_pose = {}
        hand_qpos = {}
        if left_arm is not None:
            arm_pose["left"] = left_arm
        if right_arm is not None:
            arm_pose["right"] = right_arm
        if left_hand is not None:
            hand_qpos["left"] = left_hand
        if right_hand is not None:
            hand_qpos["right"] = right_hand
        return cls(
            action_space=BIMANUAL_EE_POSE_HAND16_V1,
            timestamp=timestamp,
            arm_pose_by_side=arm_pose,
            hand_qpos_by_side=hand_qpos,
            active_effectors=active_effectors,
            source=source,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def bimanual_qpos(
        cls,
        *,
        left_arm: Any | None = None,
        right_arm: Any | None = None,
        left_hand: Any | None = None,
        right_hand: Any | None = None,
        active_effectors: tuple[str, ...] = (),
        timestamp: float | None = None,
        source: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "RobotAction":
        arm_qpos = {}
        hand_qpos = {}
        if left_arm is not None:
            arm_qpos["left"] = left_arm
        if right_arm is not None:
            arm_qpos["right"] = right_arm
        if left_hand is not None:
            hand_qpos["left"] = left_hand
        if right_hand is not None:
            hand_qpos["right"] = right_hand
        return cls(
            action_space=BIMANUAL_QPOS_V1,
            timestamp=timestamp,
            arm_qpos_by_side=arm_qpos,
            hand_qpos_by_side=hand_qpos,
            active_effectors=active_effectors,
            source=source,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def bimanual_qpos_hand16(
        cls,
        *,
        left_arm: Any | None = None,
        right_arm: Any | None = None,
        left_hand: Any | None = None,
        right_hand: Any | None = None,
        active_effectors: tuple[str, ...] = (),
        timestamp: float | None = None,
        source: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "RobotAction":
        arm_qpos = {}
        hand_qpos = {}
        if left_arm is not None:
            arm_qpos["left"] = left_arm
        if right_arm is not None:
            arm_qpos["right"] = right_arm
        if left_hand is not None:
            hand_qpos["left"] = left_hand
        if right_hand is not None:
            hand_qpos["right"] = right_hand
        return cls(
            action_space=BIMANUAL_QPOS_HAND16_V1,
            timestamp=timestamp,
            arm_qpos_by_side=arm_qpos,
            hand_qpos_by_side=hand_qpos,
            active_effectors=active_effectors,
            source=source,
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True)
class ActionExecutionResult:
    timestamp: float
    command: TeleopCommand | None = None
    sent: bool = False
    faults: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "faults", tuple(str(fault) for fault in self.faults))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ActionRecord:
    timestamp: float
    observation: Any
    action: RobotAction | None
    command: TeleopCommand | None
    execution: ActionExecutionResult | None = None
    phases: tuple[tuple[str, float], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phases = tuple((str(name), float(elapsed)) for name, elapsed in self.phases)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "metadata", dict(self.metadata))


__all__ = [
    "ARM_POSE_DOF",
    "BIMANUAL_EE_POSE_HAND16_V1",
    "BIMANUAL_EE_POSE_V1",
    "BIMANUAL_MIXED_HAND16_V1",
    "BIMANUAL_MIXED_V1",
    "BIMANUAL_QPOS_HAND16_V1",
    "BIMANUAL_QPOS_V1",
    "HAND16_ACTION_SPACES",
    "HAND16_SLOT_DOF",
    "INSPIRE_HAND_DOF",
    "RM65_ARM_DOF",
    "VALID_SIDES",
    "ActionExecutionResult",
    "ActionRecord",
    "RobotAction",
    "default_arm_controlled_frame",
    "default_arm_frame",
    "effector_name",
    "hand_dof_for_action_space",
    "normalize_side",
    "split_effector_name",
]
