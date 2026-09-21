"""Optional B601 MuJoCo gripper geometry and normalized target adapter.

The existing B601 scene intentionally keeps the gripper open and outside the
arm driver.  This module provides an explicit opt-in scene spec for mapping
tests.  It is a kinematic visualization adapter only; it is not a RobStride
J7 actuator model and is never used by the real hardware owner.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from robot_motion.rebot_b601_rs.gripper_mapping import (
    B601_RS_GRIPPER_SOURCE_V1,
    GRIPPER_CLOCK_DOMAIN_LOCAL_WALL_TIME,
    RM65_DMTAC_GRIPPER_SOURCE_V1,
    B601RM65GripperMappingProfile,
    NormalizedGripperSample,
)
from sim.mujoco.config import GripperSpec
from sim.mujoco.scene import MujocoScene

B601_RS_GRIPPER_NAME = "b601_gripper"
B601_RS_GRIPPER_JOINT_NAMES = ("joint_left", "joint_right")
B601_RS_GRIPPER_CLOSED_QPOS = {
    "joint_left": 0.0,
    "joint_right": 0.0,
}


def b601_gripper_spec(*, name: str = B601_RS_GRIPPER_NAME) -> GripperSpec:
    """Return the opt-in normalized B601 finger mapping.

    The upstream URDF ranges are ``joint_left=[0, .05]`` and
    ``joint_right=[0, .0715]``; the existing B601 scene names the upper end
    the open position.  This is a geometry mapping for MuJoCo visualization,
    not a claim about the RobStride motor's rotary calibration.
    """

    return GripperSpec(
        name=name,
        joint_names=B601_RS_GRIPPER_JOINT_NAMES,
        open_qpos=(0.05, 0.0715),
        closed_qpos=(0.0, 0.0),
        effector_name="b601_gripper_end",
        command_name="gripper_close",
        side="right",
    )


def _close(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("close must be finite and in [0, 1].")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("close must be finite and in [0, 1].")
    return normalized


def _positive_age(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return normalized


def _non_negative(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative finite number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return normalized


class B601GripperSimAdapter:
    """Apply canonical close targets to an opt-in B601 MuJoCo scene."""

    execution_mode = "b601_mujoco_gripper_kinematic_mapping_only"

    def __init__(
        self,
        scene: MujocoScene,
        *,
        gripper_name: str = B601_RS_GRIPPER_NAME,
    ) -> None:
        if not isinstance(scene, MujocoScene):
            raise TypeError("scene must be a MujocoScene.")
        name = str(gripper_name).strip()
        if name not in scene.gripper_specs:
            raise KeyError(
                f"scene has no opt-in B601 gripper spec {name!r}; "
                "build it with enable_gripper=True."
            )
        self.scene = scene
        self.gripper_name = name
        self._target_close = float(scene.get_gripper_close(name))

    @property
    def target_close(self) -> float:
        return self._target_close

    @property
    def measured_close(self) -> float:
        return float(self.scene.get_gripper_close(self.gripper_name))

    def set_close(self, close: float) -> float:
        target = _close(close)
        self.scene.set_gripper_close(self.gripper_name, target)
        self._target_close = target
        return target

    def metadata(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode,
            "gripper_name": self.gripper_name,
            "joint_names": list(self.scene.gripper_specs[self.gripper_name].joint_names),
            "target_close": self.target_close,
            "measured_close": self.measured_close,
            "hardware_fidelity": "not_claimed",
            "default_scene_behavior_preserved": True,
        }


@dataclass(frozen=True)
class GripperBridgeStep:
    """One fail-closed normalized source-to-MuJoCo target update."""

    accepted: bool
    source_sequence: int
    source_close: float | None
    target_close: float | None
    rejection_reason: str | None = None


class NormalizedGripperSimBridge:
    """Map one canonical scalar into an existing MuJoCo gripper spec.

    This bridge is deliberately a target adapter rather than a second control
    loop.  It rejects invalid or out-of-order samples and leaves the last
    target untouched; the caller remains the single simulation step owner.
    """

    def __init__(
        self,
        scene: MujocoScene,
        mapping: B601RM65GripperMappingProfile,
        *,
        direction: Literal["b601_to_rm65", "rm65_to_b601"],
        target_gripper_name: str,
        expected_source_id: str | None = None,
        maximum_source_age_sec: float = 0.25,
        maximum_receive_age_sec: float = 0.25,
        maximum_future_skew_sec: float = 0.05,
        expected_clock_domain_id: str = GRIPPER_CLOCK_DOMAIN_LOCAL_WALL_TIME,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(scene, MujocoScene):
            raise TypeError("scene must be a MujocoScene.")
        if not isinstance(mapping, B601RM65GripperMappingProfile):
            raise TypeError("mapping must be a B601RM65GripperMappingProfile.")
        if direction not in {"b601_to_rm65", "rm65_to_b601"}:
            raise ValueError(f"unsupported direction: {direction!r}")
        name = scene._resolve_gripper_name(str(target_gripper_name).strip())
        default_source_id = (
            B601_RS_GRIPPER_SOURCE_V1
            if direction == "b601_to_rm65"
            else RM65_DMTAC_GRIPPER_SOURCE_V1
        )
        source_id = (
            default_source_id
            if expected_source_id is None
            else str(expected_source_id).strip()
        )
        if not source_id:
            raise ValueError("expected_source_id must be non-empty.")
        clock_domain_id = str(expected_clock_domain_id).strip()
        if not clock_domain_id:
            raise ValueError("expected_clock_domain_id must be non-empty.")
        self.maximum_source_age_sec = _positive_age(
            maximum_source_age_sec,
            name="maximum_source_age_sec",
        )
        self.maximum_receive_age_sec = _positive_age(
            maximum_receive_age_sec,
            name="maximum_receive_age_sec",
        )
        self.maximum_future_skew_sec = _non_negative(
            maximum_future_skew_sec,
            name="maximum_future_skew_sec",
        )
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable or None.")
        self.scene = scene
        self.mapping = mapping
        self.direction = direction
        self.target_gripper_name = name
        self.expected_source_id = source_id
        self.expected_clock_domain_id = clock_domain_id
        self.clock = time.time if clock is None else clock
        self._last_sequence = -1

    @property
    def last_sequence(self) -> int:
        return self._last_sequence

    def apply(
        self,
        sample: NormalizedGripperSample,
        *,
        now_timestamp_sec: float | None = None,
    ) -> GripperBridgeStep:
        if not isinstance(sample, NormalizedGripperSample):
            raise TypeError("sample must be a NormalizedGripperSample.")
        if not sample.valid:
            return GripperBridgeStep(
                accepted=False,
                source_sequence=sample.sequence,
                source_close=sample.close,
                target_close=None,
                rejection_reason="invalid_sample",
            )
        if sample.source_id != self.expected_source_id:
            return GripperBridgeStep(
                accepted=False,
                source_sequence=sample.sequence,
                source_close=sample.close,
                target_close=None,
                rejection_reason="source_id_mismatch",
            )
        if sample.clock_domain_id != self.expected_clock_domain_id:
            return GripperBridgeStep(
                accepted=False,
                source_sequence=sample.sequence,
                source_close=sample.close,
                target_close=None,
                rejection_reason="clock_domain_mismatch",
            )
        now = float(self.clock() if now_timestamp_sec is None else now_timestamp_sec)
        if not math.isfinite(now):
            raise ValueError("now_timestamp_sec must be finite.")
        source_age = now - sample.source_timestamp_sec
        receive_age = now - sample.receive_timestamp_sec
        if source_age < -self.maximum_future_skew_sec:
            reason = "source_timestamp_in_future"
        elif receive_age < -self.maximum_future_skew_sec:
            reason = "receive_timestamp_in_future"
        elif source_age > self.maximum_source_age_sec:
            reason = "source_sample_stale"
        elif receive_age > self.maximum_receive_age_sec:
            reason = "source_receipt_stale"
        else:
            reason = None
        if reason is not None:
            return GripperBridgeStep(
                accepted=False,
                source_sequence=sample.sequence,
                source_close=sample.close,
                target_close=None,
                rejection_reason=reason,
            )
        if sample.sequence <= self._last_sequence:
            return GripperBridgeStep(
                accepted=False,
                source_sequence=sample.sequence,
                source_close=sample.close,
                target_close=None,
                rejection_reason="stale_sequence",
            )
        target_close = self.mapping.map_normalized(
            sample.close,
            direction=self.direction,
        )
        self.scene.set_gripper_close(self.target_gripper_name, target_close)
        self._last_sequence = sample.sequence
        return GripperBridgeStep(
            accepted=True,
            source_sequence=sample.sequence,
            source_close=sample.close,
            target_close=target_close,
        )


__all__ = [
    "B601_RS_GRIPPER_CLOSED_QPOS",
    "B601_RS_GRIPPER_JOINT_NAMES",
    "B601_RS_GRIPPER_NAME",
    "B601GripperSimAdapter",
    "GripperBridgeStep",
    "NormalizedGripperSimBridge",
    "b601_gripper_spec",
]
