"""Explicit-anchor shared keyframe safety for canonical ``RobotAction``.

The numeric bounds reuse ``robot_motion.position_control.safety``.  This
adapter does not choose what "last action" means: the caller must provide a
typed anchor and remains responsible for proving that anchor policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.position_control.safety import (
    PositionKeyframeSafetyConfig,
    limit_joint_keyframes,
    limit_pose_keyframes,
)
from robot_motion.runtime.immutable import deep_freeze

from .safety import SharedCanonicalSafetyResult
from .schema import EndEffectorTarget, RobotAction, effector_name
from .hand_spec import HandActionSpec, hand_action_specs_from_metadata


KEYFRAME_SAFETY_PROFILE_HASH_DOMAIN_V1 = "dexteleop.keyframe_safety_profile.v1"
KEYFRAME_SAFETY_ANCHOR_HASH_DOMAIN_V1 = "dexteleop.keyframe_safety_anchor.v1"
KEYFRAME_SAFETY_PROFILE_HASH_DOMAIN_V2 = "dexteleop.keyframe_safety_profile.v2"
KEYFRAME_SAFETY_ANCHOR_HASH_DOMAIN_V2 = "dexteleop.keyframe_safety_anchor.v2"
KEYFRAME_SAFETY_ANCHOR_KINDS_V1 = frozenset(
    {
        "previous_selected",
        "previous_safe",
        "previous_safe_committed",
        "previous_commanded",
        "measured_state",
    }
)


@dataclass(frozen=True)
class HandQposSafetyLimits:
    """Canonical active-slot limits for one hand/gripper component.

    The limits use the command units declared by ``HandActionSpec``.  This
    mirrors the existing position-teleop behavior: finite typed targets are
    clamped to an absolute actuator envelope, with an optional per-keyframe
    step bound.  Backend send thresholds and command-rate limits remain
    transport concerns and are deliberately not duplicated here.
    """

    hand_spec: HandActionSpec
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    max_step: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.hand_spec, HandActionSpec):
            raise TypeError("hand_spec must be a HandActionSpec.")
        active_dof = int(self.hand_spec.active_dof)
        lower = tuple(float(value) for value in self.lower)
        upper = tuple(float(value) for value in self.upper)
        if len(lower) != active_dof or len(upper) != active_dof:
            raise ValueError(
                "Hand qpos lower/upper bounds must match hand_spec.active_dof."
            )
        if (
            not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(np.asarray(lower) > np.asarray(upper))
        ):
            raise ValueError("Hand qpos bounds must be finite and ordered.")
        if self.max_step is None:
            max_step = None
        else:
            max_step = tuple(float(value) for value in self.max_step)
            if len(max_step) != active_dof:
                raise ValueError(
                    "Hand qpos max_step must match hand_spec.active_dof."
                )
            if not np.all(np.isfinite(max_step)) or np.any(
                np.asarray(max_step) < 0.0
            ):
                raise ValueError("Hand qpos max_step must be finite and >= 0.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "max_step", max_step)

    @classmethod
    def uniform(
        cls,
        hand_spec: HandActionSpec,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        max_step: float | None = None,
    ) -> "HandQposSafetyLimits":
        active_dof = int(hand_spec.active_dof)
        return cls(
            hand_spec=hand_spec,
            lower=tuple(float(lower) for _ in range(active_dof)),
            upper=tuple(float(upper) for _ in range(active_dof)),
            max_step=(
                None
                if max_step is None
                else tuple(float(max_step) for _ in range(active_dof))
            ),
        )

    @property
    def semantic_payload(self) -> dict[str, object]:
        return {
            "hand_spec": self.hand_spec.to_metadata(),
            "lower": self.lower,
            "upper": self.upper,
            "max_step": self.max_step,
        }


@dataclass(frozen=True)
class EndEffectorSafetyCapability:
    """Adapter-bound native end-effector limits in canonical command units."""

    adapter: str
    value_type: str
    names: tuple[str, ...]
    units: tuple[str, ...]
    lower: tuple[float, ...] = ()
    upper: tuple[float, ...] = ()
    max_step: tuple[float, ...] | None = None
    max_rate_per_sec: tuple[float, ...] | None = None
    maximum_rate_interval_sec: float | None = None
    allowed_tokens_by_component: tuple[tuple[int, ...], ...] = ()
    allowed_transitions_by_component: tuple[
        tuple[tuple[int, int], ...], ...
    ] = ()

    def __post_init__(self) -> None:
        adapter = str(self.adapter).strip()
        if not adapter:
            raise ValueError("End-effector safety adapter must be non-empty.")
        value_type = str(self.value_type).strip().lower()
        if value_type not in {"continuous", "discrete"}:
            raise ValueError(
                "End-effector safety value_type must be continuous or discrete."
            )
        names = tuple(str(value).strip() for value in self.names)
        units = tuple(str(value).strip() for value in self.units)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not value for value in names)
        ):
            raise ValueError(
                "End-effector safety names must be non-empty and unique."
            )
        if len(units) != len(names) or any(not value for value in units):
            raise ValueError(
                "End-effector safety units must match the target dimension."
            )
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "units", units)

        dimension = len(names)
        lower = tuple(float(value) for value in self.lower)
        upper = tuple(float(value) for value in self.upper)
        max_step = (
            None
            if self.max_step is None
            else tuple(float(value) for value in self.max_step)
        )
        max_rate = (
            None
            if self.max_rate_per_sec is None
            else tuple(float(value) for value in self.max_rate_per_sec)
        )
        allowed_tokens = tuple(
            tuple(int(token) for token in component)
            for component in self.allowed_tokens_by_component
        )
        allowed_transitions = tuple(
            tuple((int(start), int(end)) for start, end in component)
            for component in self.allowed_transitions_by_component
        )
        if value_type == "continuous":
            if len(lower) != dimension or len(upper) != dimension:
                raise ValueError(
                    "Continuous end-effector bounds must match target dimension."
                )
            if (
                not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(np.asarray(lower) > np.asarray(upper))
            ):
                raise ValueError(
                    "Continuous end-effector bounds must be finite and ordered."
                )
            for label, values in (
                ("max_step", max_step),
                ("max_rate_per_sec", max_rate),
            ):
                if values is not None and (
                    len(values) != dimension
                    or not np.all(np.isfinite(values))
                    or np.any(np.asarray(values) < 0.0)
                ):
                    raise ValueError(
                        f"Continuous end-effector {label} must be finite, "
                        "non-negative, and match target dimension."
                    )
            interval = self.maximum_rate_interval_sec
            if max_rate is None:
                if interval is not None:
                    raise ValueError(
                        "maximum_rate_interval_sec requires max_rate_per_sec."
                    )
            else:
                if interval is None or not math.isfinite(float(interval)) or float(interval) <= 0:
                    raise ValueError(
                        "Rate-limited end-effector safety requires a positive "
                        "maximum_rate_interval_sec."
                    )
                object.__setattr__(
                    self,
                    "maximum_rate_interval_sec",
                    float(interval),
                )
            if allowed_tokens or allowed_transitions:
                raise ValueError(
                    "Continuous end-effector safety cannot declare discrete tokens."
                )
        else:
            if lower or upper or max_step is not None or max_rate is not None:
                raise ValueError(
                    "Discrete end-effector safety cannot declare continuous limits."
                )
            if self.maximum_rate_interval_sec is not None:
                raise ValueError(
                    "Discrete end-effector safety cannot declare a rate interval."
                )
            if len(allowed_tokens) != dimension or any(
                not component for component in allowed_tokens
            ):
                raise ValueError(
                    "Discrete end-effector safety requires an allowlist per component."
                )
            if allowed_transitions and len(allowed_transitions) != dimension:
                raise ValueError(
                    "Discrete transition lists must match target dimension."
                )
            for index, transitions in enumerate(allowed_transitions):
                allowed = set(allowed_tokens[index])
                if any(
                    start not in allowed or end not in allowed
                    for start, end in transitions
                ):
                    raise ValueError(
                        "Discrete transitions must reference allowlisted tokens."
                    )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "max_step", max_step)
        object.__setattr__(self, "max_rate_per_sec", max_rate)
        object.__setattr__(
            self,
            "allowed_tokens_by_component",
            allowed_tokens,
        )
        object.__setattr__(
            self,
            "allowed_transitions_by_component",
            allowed_transitions,
        )

    @property
    def semantic_payload(self) -> dict[str, object]:
        return {
            "adapter": self.adapter,
            "value_type": self.value_type,
            "names": self.names,
            "units": self.units,
            "lower": self.lower,
            "upper": self.upper,
            "max_step": self.max_step,
            "max_rate_per_sec": self.max_rate_per_sec,
            "maximum_rate_interval_sec": self.maximum_rate_interval_sec,
            "allowed_tokens_by_component": self.allowed_tokens_by_component,
            "allowed_transitions_by_component": (
                self.allowed_transitions_by_component
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "EndEffectorSafetyCapability":
        return cls(
            adapter=str(payload["adapter"]),
            value_type=str(payload["value_type"]),
            names=tuple(payload["names"]),  # type: ignore[arg-type]
            units=tuple(payload["units"]),  # type: ignore[arg-type]
            lower=tuple(payload.get("lower", ())),  # type: ignore[arg-type]
            upper=tuple(payload.get("upper", ())),  # type: ignore[arg-type]
            max_step=(
                None
                if payload.get("max_step") is None
                else tuple(payload["max_step"])  # type: ignore[arg-type]
            ),
            max_rate_per_sec=(
                None
                if payload.get("max_rate_per_sec") is None
                else tuple(payload["max_rate_per_sec"])  # type: ignore[arg-type]
            ),
            maximum_rate_interval_sec=(
                None
                if payload.get("maximum_rate_interval_sec") is None
                else float(payload["maximum_rate_interval_sec"])
            ),
            allowed_tokens_by_component=tuple(
                tuple(component)
                for component in payload.get(  # type: ignore[union-attr]
                    "allowed_tokens_by_component",
                    (),
                )
            ),
            allowed_transitions_by_component=tuple(
                tuple(tuple(pair) for pair in component)
                for component in payload.get(  # type: ignore[union-attr]
                    "allowed_transitions_by_component",
                    (),
                )
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return self.semantic_payload


@dataclass(frozen=True)
class CanonicalKeyframeSafetyProfile:
    profile_id: str
    limits: PositionKeyframeSafetyConfig = PositionKeyframeSafetyConfig()
    hand_limits_by_side: Mapping[str, HandQposSafetyLimits] = field(
        default_factory=dict
    )
    end_effector_capabilities_by_side: Mapping[
        str, EndEffectorSafetyCapability
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.profile_id).strip():
            raise ValueError("profile_id must be non-empty.")
        hand_limits: dict[str, HandQposSafetyLimits] = {}
        for raw_side, value in dict(self.hand_limits_by_side).items():
            side = str(raw_side)
            if not isinstance(value, HandQposSafetyLimits):
                raise TypeError(
                    f"hand_limits_by_side[{side!r}] must be HandQposSafetyLimits."
                )
            if value.hand_spec.side != side:
                raise ValueError(
                    "Hand safety profile key must match hand_spec.side."
                )
            hand_limits[side] = value
        object.__setattr__(
            self,
            "hand_limits_by_side",
            MappingProxyType(hand_limits),
        )
        end_effector_capabilities: dict[str, EndEffectorSafetyCapability] = {}
        for raw_side, value in dict(
            self.end_effector_capabilities_by_side
        ).items():
            side = str(raw_side)
            if side not in {"left", "right"}:
                raise ValueError(
                    "End-effector safety profile side must be left or right."
                )
            if not isinstance(value, EndEffectorSafetyCapability):
                raise TypeError(
                    "end_effector_capabilities_by_side values must be "
                    "EndEffectorSafetyCapability."
                )
            end_effector_capabilities[side] = value
        object.__setattr__(
            self,
            "end_effector_capabilities_by_side",
            MappingProxyType(end_effector_capabilities),
        )

    @property
    def semantic_content_hash(self) -> str:
        limits = self.limits
        payload = {
            "profile_id": self.profile_id,
            "workspace_min": limits.workspace_min,
            "workspace_max": limits.workspace_max,
            "workspace_soft_margin_m": limits.workspace_soft_margin_m,
            "max_translation_step_m": limits.max_translation_step_m,
            "max_rotation_step_deg": limits.max_rotation_step_deg,
            "max_joint_step_rad": limits.max_joint_step_rad,
            "joint_lower_rad": limits.joint_lower_rad,
            "joint_upper_rad": limits.joint_upper_rad,
        }
        # Missing in the original v1 payload means enabled.  Preserve hashes
        # for that default while making the new opt-out semantically explicit.
        if not limits.workspace_enabled:
            payload["workspace_enabled"] = False
        if self.hand_limits_by_side:
            payload["hand_limits_by_side"] = {
                side: value.semantic_payload
                for side, value in sorted(self.hand_limits_by_side.items())
            }
        if self.end_effector_capabilities_by_side:
            payload["end_effector_capabilities_by_side"] = {
                side: value.semantic_payload
                for side, value in sorted(
                    self.end_effector_capabilities_by_side.items()
                )
            }
        return canonical_content_hash(
            payload,
            domain=(
                KEYFRAME_SAFETY_PROFILE_HASH_DOMAIN_V2
                if self.end_effector_capabilities_by_side
                else KEYFRAME_SAFETY_PROFILE_HASH_DOMAIN_V1
            ),
        )


@dataclass(frozen=True)
class CanonicalActionSafetyAnchor:
    anchor_id: str
    anchor_kind: str
    action_space: str
    arm_pose_by_side: Mapping[str, np.ndarray] = field(default_factory=dict)
    arm_qpos_by_side: Mapping[str, np.ndarray] = field(default_factory=dict)
    hand_qpos_by_side: Mapping[str, np.ndarray] = field(default_factory=dict)
    hand_action_specs_by_side: Mapping[str, HandActionSpec] = field(
        default_factory=dict
    )
    end_effector_targets_by_side: Mapping[str, EndEffectorTarget] = field(
        default_factory=dict
    )
    frame_by_side: Mapping[str, str] = field(default_factory=dict)
    observed_at: float | None = None

    def __post_init__(self) -> None:
        if not str(self.anchor_id).strip() or not str(self.action_space).strip():
            raise ValueError("anchor_id/action_space must be non-empty.")
        kind = str(self.anchor_kind).strip()
        if kind not in KEYFRAME_SAFETY_ANCHOR_KINDS_V1:
            raise ValueError(f"Unsupported keyframe safety anchor kind {kind!r}.")
        poses = {
            str(side): np.asarray(value, dtype=float).reshape(7)
            for side, value in dict(self.arm_pose_by_side).items()
        }
        qpos = {
            str(side): np.asarray(value, dtype=float).reshape(6)
            for side, value in dict(self.arm_qpos_by_side).items()
        }
        hand_qpos = {
            str(side): np.asarray(value, dtype=np.float32).reshape(-1)
            for side, value in dict(self.hand_qpos_by_side).items()
        }
        hand_specs: dict[str, HandActionSpec] = {}
        for raw_side, payload in dict(self.hand_action_specs_by_side).items():
            side = str(raw_side)
            spec = (
                payload
                if isinstance(payload, HandActionSpec)
                else HandActionSpec.from_metadata(payload)
            )
            if spec.side != side:
                raise ValueError("Hand anchor spec key must match spec.side.")
            hand_specs[side] = spec
        end_effector_targets: dict[str, EndEffectorTarget] = {}
        for raw_side, target in dict(
            self.end_effector_targets_by_side
        ).items():
            side = str(raw_side)
            if side not in {"left", "right"}:
                raise ValueError(
                    "End-effector safety anchor side must be left or right."
                )
            if not isinstance(target, EndEffectorTarget):
                raise TypeError(
                    "end_effector_targets_by_side values must be "
                    "EndEffectorTarget."
                )
            frozen = EndEffectorTarget(
                value_type=target.value_type,
                values=target.values,
                adapter=target.adapter,
                names=target.names,
                metadata=target.metadata,
            )
            object.__setattr__(frozen, "values", deep_freeze(frozen.values))
            object.__setattr__(frozen, "metadata", deep_freeze(frozen.metadata))
            end_effector_targets[side] = frozen
        if poses and qpos:
            raise ValueError("A safety anchor cannot mix pose and qpos modalities.")
        if set(hand_qpos) & set(end_effector_targets):
            raise ValueError(
                "A safety anchor cannot mix hand qpos and native "
                "end-effector targets on one side."
            )
        if not poses and not qpos and not hand_qpos and not end_effector_targets:
            raise ValueError(
                "A safety anchor must contain arm pose, arm qpos, hand qpos, "
                "or native end-effector values."
            )
        if any(not np.all(np.isfinite(value)) for value in (*poses.values(), *qpos.values())):
            raise ValueError("Safety anchor values must be finite.")
        if set(hand_qpos) != set(hand_specs):
            raise ValueError(
                "Hand safety anchors require one explicit HandActionSpec per side."
            )
        for side, values in hand_qpos.items():
            hand_specs[side].validate_vector(values)
        frames = {str(side): str(value) for side, value in dict(self.frame_by_side).items()}
        if poses and set(frames) != set(poses):
            raise ValueError("Pose anchors require one explicit frame per side.")
        if any(not value.strip() for value in frames.values()):
            raise ValueError("Safety anchor frames must be non-empty.")
        if self.observed_at is not None and not math.isfinite(float(self.observed_at)):
            raise ValueError("observed_at must be finite when provided.")
        object.__setattr__(self, "anchor_kind", kind)
        object.__setattr__(self, "arm_pose_by_side", deep_freeze(poses))
        object.__setattr__(self, "arm_qpos_by_side", deep_freeze(qpos))
        object.__setattr__(self, "hand_qpos_by_side", deep_freeze(hand_qpos))
        object.__setattr__(
            self,
            "hand_action_specs_by_side",
            MappingProxyType(hand_specs),
        )
        object.__setattr__(
            self,
            "end_effector_targets_by_side",
            MappingProxyType(end_effector_targets),
        )
        object.__setattr__(self, "frame_by_side", MappingProxyType(frames))

    @property
    def semantic_content_hash(self) -> str:
        payload = {
                "anchor_id": self.anchor_id,
                "anchor_kind": self.anchor_kind,
                "action_space": self.action_space,
                "arm_pose_by_side": self.arm_pose_by_side,
                "arm_qpos_by_side": self.arm_qpos_by_side,
                "frame_by_side": self.frame_by_side,
                "observed_at": self.observed_at,
            }
        if self.hand_qpos_by_side:
            payload["hand_qpos_by_side"] = self.hand_qpos_by_side
            payload["hand_action_specs_by_side"] = {
                side: spec.to_metadata()
                for side, spec in sorted(
                    self.hand_action_specs_by_side.items()
                )
            }
        if self.end_effector_targets_by_side:
            payload["end_effector_targets_by_side"] = {
                side: {
                    "value_type": target.value_type,
                    "values": target.values,
                    "adapter": target.adapter,
                    "names": target.names,
                    "metadata": target.metadata,
                }
                for side, target in sorted(
                    self.end_effector_targets_by_side.items()
                )
            }
        return canonical_content_hash(
            payload,
            domain=(
                KEYFRAME_SAFETY_ANCHOR_HASH_DOMAIN_V2
                if self.end_effector_targets_by_side
                else KEYFRAME_SAFETY_ANCHOR_HASH_DOMAIN_V1
            ),
        )


def _limit_hand_qpos(
    values_by_side: Mapping[str, Sequence[float] | np.ndarray],
    *,
    reference_by_side: Mapping[str, Sequence[float] | np.ndarray],
    limits_by_side: Mapping[str, HandQposSafetyLimits],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, bool],
    dict[str, tuple[str, ...]],
    dict[str, Mapping[str, object]],
]:
    output: dict[str, np.ndarray] = {}
    modified: dict[str, bool] = {}
    reasons: dict[str, tuple[str, ...]] = {}
    metadata: dict[str, Mapping[str, object]] = {}
    sides = set(values_by_side)
    if sides != set(reference_by_side) or sides != set(limits_by_side):
        raise ValueError(
            "Hand target, anchor, and safety-profile sides must match exactly."
        )
    for side in sorted(sides):
        limits = limits_by_side[side]
        spec = limits.hand_spec
        target = spec.validate_vector(values_by_side[side])
        reference = spec.validate_vector(reference_by_side[side])
        mask = np.asarray(spec.active_mask, dtype=bool)
        active_target = target[mask].astype(np.float64, copy=True)
        active_reference = reference[mask].astype(np.float64, copy=True)
        lower = np.asarray(limits.lower, dtype=np.float64)
        upper = np.asarray(limits.upper, dtype=np.float64)
        if np.any(active_reference < lower) or np.any(active_reference > upper):
            raise ValueError(f"{side} hand safety anchor is outside absolute bounds.")
        original_active = active_target.copy()
        active_target = np.clip(active_target, lower, upper)
        side_reasons: list[str] = []
        if not np.array_equal(active_target, original_active):
            side_reasons.append("absolute_hand_qpos_limit")
        raw_max_step = float(
            np.max(np.abs(original_active - active_reference))
        ) if original_active.size else 0.0
        if limits.max_step is not None:
            max_step = np.asarray(limits.max_step, dtype=np.float64)
            stepped = active_reference + np.clip(
                active_target - active_reference,
                -max_step,
                max_step,
            )
            if not np.array_equal(stepped, active_target):
                side_reasons.append("hand_qpos_step_limit")
            active_target = stepped
        target[mask] = active_target.astype(np.float32)
        output[side] = target
        modified[side] = not np.array_equal(
            active_target,
            original_active,
        )
        reasons[side] = tuple(side_reasons)
        metadata[side] = {
            "command_space": spec.command_space,
            "active_dof": int(spec.active_dof),
            "active_mask": tuple(bool(value) for value in spec.active_mask),
            "raw_max_step": raw_max_step,
            "applied_max_step": (
                float(np.max(np.abs(active_target - active_reference)))
                if active_target.size
                else 0.0
            ),
            "lower": limits.lower,
            "upper": limits.upper,
            "max_step": limits.max_step,
        }
    return output, modified, reasons, metadata


def _target_units(target: EndEffectorTarget) -> tuple[str, ...]:
    raw_units = target.metadata.get("units")
    if not isinstance(raw_units, (tuple, list)):
        raise ValueError(
            "Native end-effector targets require ordered metadata.units."
        )
    units = tuple(str(value).strip() for value in raw_units)
    if len(units) != target.values.size or any(not value for value in units):
        raise ValueError(
            "Native end-effector target units must match target dimension."
        )
    return units


def _validate_end_effector_identity(
    target: EndEffectorTarget,
    capability: EndEffectorSafetyCapability,
) -> None:
    if target.adapter != capability.adapter:
        raise ValueError("Native end-effector adapter does not match capability.")
    if target.value_type != capability.value_type:
        raise ValueError("Native end-effector value_type does not match capability.")
    if target.names != capability.names:
        raise ValueError("Native end-effector names do not match capability.")
    if _target_units(target) != capability.units:
        raise ValueError("Native end-effector units do not match capability.")


def _limit_native_end_effectors(
    values_by_side: Mapping[str, EndEffectorTarget],
    *,
    reference_by_side: Mapping[str, EndEffectorTarget],
    capabilities_by_side: Mapping[str, EndEffectorSafetyCapability],
    anchor_observed_at: float | None,
    target_effective_at: float | None,
) -> tuple[
    dict[str, EndEffectorTarget],
    dict[str, bool],
    dict[str, tuple[str, ...]],
    dict[str, Mapping[str, object]],
]:
    output: dict[str, EndEffectorTarget] = {}
    modified: dict[str, bool] = {}
    reasons: dict[str, tuple[str, ...]] = {}
    metadata: dict[str, Mapping[str, object]] = {}
    sides = set(values_by_side)
    if sides != set(reference_by_side) or sides != set(capabilities_by_side):
        raise ValueError(
            "Native end-effector target, anchor, and safety-profile sides "
            "must match exactly."
        )
    for side in sorted(sides):
        target = values_by_side[side]
        reference = reference_by_side[side]
        capability = capabilities_by_side[side]
        _validate_end_effector_identity(target, capability)
        _validate_end_effector_identity(reference, capability)
        side_reasons: list[str] = []
        side_metadata: dict[str, object] = {
            "adapter": capability.adapter,
            "value_type": capability.value_type,
            "names": capability.names,
            "units": capability.units,
        }
        if capability.value_type == "continuous":
            original = np.asarray(
                target.values,
                dtype=np.float64,
            ).reshape(-1)
            anchor_values = np.asarray(
                reference.values,
                dtype=np.float64,
            ).reshape(-1)
            lower = np.asarray(capability.lower, dtype=np.float64)
            upper = np.asarray(capability.upper, dtype=np.float64)
            if np.any(anchor_values < lower) or np.any(anchor_values > upper):
                raise ValueError(
                    f"{side} native end-effector anchor is outside bounds."
                )
            bounded = np.clip(original, lower, upper)
            if not np.array_equal(bounded, original):
                side_reasons.append("absolute_end_effector_limit")

            allowed_delta: np.ndarray | None = None
            if capability.max_step is not None:
                step_limit = np.asarray(
                    capability.max_step,
                    dtype=np.float64,
                )
                allowed_delta = step_limit
                if np.any(np.abs(bounded - anchor_values) > step_limit):
                    side_reasons.append("end_effector_step_limit")
                side_metadata["max_step"] = capability.max_step
            if capability.max_rate_per_sec is not None:
                if anchor_observed_at is None or target_effective_at is None:
                    raise ValueError(
                        "Rate-limited native end-effector safety requires "
                        "anchor and target effective times."
                    )
                elapsed = float(target_effective_at) - float(anchor_observed_at)
                if not math.isfinite(elapsed) or elapsed < 0.0:
                    raise ValueError(
                        "Native end-effector safety elapsed time must be "
                        "finite and non-negative."
                    )
                assert capability.maximum_rate_interval_sec is not None
                bounded_elapsed = min(
                    elapsed,
                    capability.maximum_rate_interval_sec,
                )
                rate_delta = np.asarray(
                    capability.max_rate_per_sec,
                    dtype=np.float64,
                ) * bounded_elapsed
                if np.any(np.abs(bounded - anchor_values) > rate_delta):
                    side_reasons.append("end_effector_rate_limit")
                allowed_delta = (
                    rate_delta
                    if allowed_delta is None
                    else np.minimum(allowed_delta, rate_delta)
                )
                side_metadata.update(
                    {
                        "max_rate_per_sec": capability.max_rate_per_sec,
                        "elapsed_sec": elapsed,
                        "bounded_elapsed_sec": bounded_elapsed,
                        "maximum_rate_interval_sec": (
                            capability.maximum_rate_interval_sec
                        ),
                    }
                )
            if allowed_delta is not None:
                bounded = anchor_values + np.clip(
                    bounded - anchor_values,
                    -allowed_delta,
                    allowed_delta,
                )
            values = bounded.astype(np.float32)
            side_metadata.update(
                {
                    "lower": capability.lower,
                    "upper": capability.upper,
                    "raw_max_delta": float(
                        np.max(np.abs(original - anchor_values))
                    ),
                    "applied_max_delta": float(
                        np.max(np.abs(bounded - anchor_values))
                    ),
                }
            )
        else:
            values = np.asarray(target.values, dtype=np.int64).reshape(-1)
            anchor_values = np.asarray(
                reference.values,
                dtype=np.int64,
            ).reshape(-1)
            for index, (candidate, anchor_value) in enumerate(
                zip(values, anchor_values, strict=True)
            ):
                allowed = set(
                    capability.allowed_tokens_by_component[index]
                )
                if int(candidate) not in allowed:
                    raise ValueError(
                        f"{side} native end-effector token is not allowlisted."
                    )
                if int(anchor_value) not in allowed:
                    raise ValueError(
                        f"{side} native end-effector anchor token is not "
                        "allowlisted."
                    )
                if capability.allowed_transitions_by_component:
                    transitions = set(
                        capability.allowed_transitions_by_component[index]
                    )
                    if (int(anchor_value), int(candidate)) not in transitions:
                        raise ValueError(
                            f"{side} native end-effector transition is not "
                            "allowed."
                        )
            side_metadata["allowed_tokens_by_component"] = (
                capability.allowed_tokens_by_component
            )
            side_metadata["allowed_transitions_by_component"] = (
                capability.allowed_transitions_by_component
            )
        safe_target = EndEffectorTarget(
            value_type=target.value_type,
            values=values,
            adapter=target.adapter,
            names=target.names,
            metadata=target.metadata,
        )
        output[side] = safe_target
        modified[side] = not np.array_equal(values, target.values)
        reasons[side] = tuple(dict.fromkeys(side_reasons))
        metadata[side] = side_metadata
    return output, modified, reasons, metadata


def _fault(
    input_action_id: str,
    *reasons: str,
    anchor: CanonicalActionSafetyAnchor,
    profile: CanonicalKeyframeSafetyProfile,
) -> SharedCanonicalSafetyResult:
    return SharedCanonicalSafetyResult(
        input_action_id=input_action_id,
        safe_action=None,
        reasons=tuple(reasons),
        status="fault",
        safety_contract_id=profile.profile_id,
        safety_contract_hash=profile.semantic_content_hash,
        anchor_id=anchor.anchor_id,
        anchor_kind=anchor.anchor_kind,
        anchor_hash=anchor.semantic_content_hash,
    )


def apply_canonical_keyframe_safety(
    action: RobotAction,
    input_action_id: str,
    *,
    anchor: CanonicalActionSafetyAnchor,
    profile: CanonicalKeyframeSafetyProfile,
    target_effective_at: float | None = None,
) -> SharedCanonicalSafetyResult:
    """Apply explicit keyframe bounds without selecting an anchor implicitly."""

    if not isinstance(action, RobotAction):
        raise TypeError("action must be a RobotAction.")
    if action.action_space != anchor.action_space:
        return _fault(
            input_action_id,
            "anchor_action_space_mismatch",
            anchor=anchor,
            profile=profile,
        )
    if action.hand_qpos_by_side and not profile.hand_limits_by_side:
        return _fault(
            input_action_id,
            "keyframe_safety_v1_hand_unsupported",
            anchor=anchor,
            profile=profile,
        )
    if action.arm_pose_by_side and action.arm_qpos_by_side:
        return _fault(
            input_action_id,
            "keyframe_safety_v1_mixed_arm_modalities",
            anchor=anchor,
            profile=profile,
        )
    if (
        not action.arm_pose_by_side
        and not action.arm_qpos_by_side
        and not action.hand_qpos_by_side
        and not action.end_effector_targets_by_side
    ):
        return _fault(
            input_action_id,
            "keyframe_safety_v1_no_action",
            anchor=anchor,
            profile=profile,
        )

    replacement_pose: Mapping[str, np.ndarray] = {}
    replacement_qpos: Mapping[str, np.ndarray] = {}
    replacement_hand: Mapping[str, np.ndarray] = {}
    arm_modified: Mapping[str, bool] = {}
    arm_reasons: Mapping[str, tuple[str, ...]] = {}
    arm_metadata: Mapping[str, Mapping[str, object]] = {}
    hand_modified: Mapping[str, bool] = {}
    hand_reasons: Mapping[str, tuple[str, ...]] = {}
    hand_metadata: Mapping[str, Mapping[str, object]] = {}
    replacement_end_effector: Mapping[str, EndEffectorTarget] = {}
    end_effector_modified: Mapping[str, bool] = {}
    end_effector_reasons: Mapping[str, tuple[str, ...]] = {}
    end_effector_metadata: Mapping[str, Mapping[str, object]] = {}
    try:
        if action.arm_pose_by_side:
            sides = set(action.arm_pose_by_side)
            if sides != set(anchor.arm_pose_by_side):
                return _fault(
                    input_action_id,
                    "pose_anchor_component_mismatch",
                    anchor=anchor,
                    profile=profile,
                )
            for side in sides:
                if action.arm_frame(side) != anchor.frame_by_side.get(side):
                    return _fault(
                        input_action_id,
                        f"{side}_pose_anchor_frame_mismatch",
                        anchor=anchor,
                        profile=profile,
                    )
            bounded = limit_pose_keyframes(
                action.arm_pose_by_side,
                reference_by_side=anchor.arm_pose_by_side,
                config=profile.limits,
            )
            replacement_pose = bounded.values_by_side
            arm_modified = bounded.modified_by_side
            arm_reasons = bounded.reasons_by_side
            arm_metadata = bounded.metadata_by_side
            if not bounded.valid:
                return _fault(
                    input_action_id,
                    *bounded.faults,
                    anchor=anchor,
                    profile=profile,
                )
        elif action.arm_qpos_by_side:
            sides = set(action.arm_qpos_by_side)
            if sides != set(anchor.arm_qpos_by_side):
                return _fault(
                    input_action_id,
                    "qpos_anchor_component_mismatch",
                    anchor=anchor,
                    profile=profile,
                )
            bounded = limit_joint_keyframes(
                action.arm_qpos_by_side,
                reference_by_side=anchor.arm_qpos_by_side,
                config=profile.limits,
            )
            replacement_qpos = bounded.values_by_side
            arm_modified = bounded.modified_by_side
            arm_reasons = bounded.reasons_by_side
            arm_metadata = bounded.metadata_by_side
            if not bounded.valid:
                return _fault(
                    input_action_id,
                    *bounded.faults,
                    anchor=anchor,
                    profile=profile,
                )

        if action.hand_qpos_by_side:
            action_hand_specs = hand_action_specs_from_metadata(action.metadata)
            hand_sides = set(action.hand_qpos_by_side)
            if hand_sides != set(action_hand_specs):
                return _fault(
                    input_action_id,
                    "hand_action_spec_component_mismatch",
                    anchor=anchor,
                    profile=profile,
                )
            if hand_sides != set(anchor.hand_qpos_by_side):
                return _fault(
                    input_action_id,
                    "hand_anchor_component_mismatch",
                    anchor=anchor,
                    profile=profile,
                )
            if hand_sides != set(profile.hand_limits_by_side):
                return _fault(
                    input_action_id,
                    "hand_safety_profile_component_mismatch",
                    anchor=anchor,
                    profile=profile,
                )
            for side in sorted(hand_sides):
                expected = profile.hand_limits_by_side[side].hand_spec
                if (
                    action_hand_specs[side].to_metadata()
                    != expected.to_metadata()
                    or anchor.hand_action_specs_by_side[side].to_metadata()
                    != expected.to_metadata()
                ):
                    return _fault(
                        input_action_id,
                        f"{side}_hand_action_spec_mismatch",
                        anchor=anchor,
                        profile=profile,
                    )
            (
                replacement_hand,
                hand_modified,
                hand_reasons,
                hand_metadata,
            ) = _limit_hand_qpos(
                action.hand_qpos_by_side,
                reference_by_side=anchor.hand_qpos_by_side,
                limits_by_side=profile.hand_limits_by_side,
            )

        if action.end_effector_targets_by_side:
            target_sides = set(action.end_effector_targets_by_side)
            if target_sides != set(anchor.end_effector_targets_by_side):
                return _fault(
                    input_action_id,
                    "end_effector_anchor_component_mismatch",
                    anchor=anchor,
                    profile=profile,
                )
            if target_sides != set(
                profile.end_effector_capabilities_by_side
            ):
                return _fault(
                    input_action_id,
                    "end_effector_safety_profile_component_mismatch",
                    anchor=anchor,
                    profile=profile,
                )
            (
                replacement_end_effector,
                end_effector_modified,
                end_effector_reasons,
                end_effector_metadata,
            ) = _limit_native_end_effectors(
                action.end_effector_targets_by_side,
                reference_by_side=anchor.end_effector_targets_by_side,
                capabilities_by_side=(
                    profile.end_effector_capabilities_by_side
                ),
                anchor_observed_at=anchor.observed_at,
                target_effective_at=target_effective_at,
            )
    except (TypeError, ValueError) as exc:
        return _fault(
            input_action_id,
            f"keyframe_safety_input_invalid:{type(exc).__name__}:{exc}",
            anchor=anchor,
            profile=profile,
        )

    modified_components = tuple(
        effector_name(side, "arm")
        for side, modified in sorted(arm_modified.items())
        if modified
    ) + tuple(
        effector_name(side, "hand")
        for side, modified in sorted(hand_modified.items())
        if modified
    ) + tuple(
        effector_name(side, "hand")
        for side, modified in sorted(end_effector_modified.items())
        if modified
    )
    reasons = tuple(
        f"{side}:{reason}"
        for side in sorted(arm_reasons)
        for reason in arm_reasons[side]
    ) + tuple(
        f"{side}_hand:{reason}"
        for side in sorted(hand_reasons)
        for reason in hand_reasons[side]
    ) + tuple(
        f"{side}_hand:{reason}"
        for side in sorted(end_effector_reasons)
        for reason in end_effector_reasons[side]
    )
    safe_action = RobotAction(
        action_space=action.action_space,
        timestamp=action.timestamp,
        arm_pose_by_side=replacement_pose,
        arm_qpos_by_side=replacement_qpos,
        hand_qpos_by_side=replacement_hand,
        end_effector_targets_by_side=replacement_end_effector,
        active_effectors=action.active_effectors,
        frame_by_effector=action.frame_by_effector,
        controlled_frame_by_effector=action.controlled_frame_by_effector,
        source=action.source,
        metadata=action.metadata,
    )
    return SharedCanonicalSafetyResult(
        input_action_id=input_action_id,
        safe_action=safe_action,
        modified_components=modified_components,
        reasons=reasons,
        status="modified" if modified_components else "safe",
        safety_contract_id=profile.profile_id,
        safety_contract_hash=profile.semantic_content_hash,
        anchor_id=anchor.anchor_id,
        anchor_kind=anchor.anchor_kind,
        anchor_hash=anchor.semantic_content_hash,
        details={
            "reasons_by_side": arm_reasons,
            "metadata_by_side": arm_metadata,
            "arm_reasons_by_side": arm_reasons,
            "arm_metadata_by_side": arm_metadata,
            "hand_reasons_by_side": hand_reasons,
            "hand_metadata_by_side": hand_metadata,
            "end_effector_reasons_by_side": end_effector_reasons,
            "end_effector_metadata_by_side": end_effector_metadata,
        },
    )


__all__ = [
    "CanonicalActionSafetyAnchor",
    "CanonicalKeyframeSafetyProfile",
    "EndEffectorSafetyCapability",
    "HandQposSafetyLimits",
    "KEYFRAME_SAFETY_ANCHOR_KINDS_V1",
    "apply_canonical_keyframe_safety",
]
