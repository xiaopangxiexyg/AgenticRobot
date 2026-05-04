from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..common import EffectorCommand, nan_vector, vector_or_none
from ..utils.writer import ACTION_DIM, ARM_DOF, HAND_DOF, HAND_VECTOR_ORDER
from .observation import DeploymentObservation


_ARM_SLICE = slice(0, ARM_DOF)
_RIGHT_HAND_SLICE = slice(ARM_DOF, ARM_DOF + HAND_DOF)
_LEFT_HAND_SLICE = slice(ARM_DOF + HAND_DOF, ACTION_DIM)
_HAND_SLICES = {
    "right": _RIGHT_HAND_SLICE,
    "left": _LEFT_HAND_SLICE,
}


@dataclass(frozen=True)
class DeploymentAction:
    vector: np.ndarray
    commands: dict[str, EffectorCommand] = field(default_factory=dict)
    faults: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _base_action(observation: DeploymentObservation) -> np.ndarray:
    base = observation.hold_action()
    if base.shape != (ACTION_DIM,):
        base = nan_vector(ACTION_DIM)
    return base.astype(np.float32, copy=True)


def _replace_vector_slice(
    vector: np.ndarray,
    target_slice: slice,
    value: Any,
    *,
    expected_dim: int,
) -> None:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (expected_dim,):
        raise ValueError(f"Expected action component dim {expected_dim}, got {arr.shape}.")
    vector[target_slice] = arr


def _compact_vector_to_action(
    value: Any,
    observation: DeploymentObservation,
    *,
    active_sides: tuple[str, ...],
    arm_enabled: bool,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    base = _base_action(observation)
    if arr.shape == (ACTION_DIM,):
        return arr.astype(np.float32, copy=True)
    if arr.shape == (HAND_DOF * 2,):
        base[_RIGHT_HAND_SLICE] = arr[:HAND_DOF]
        base[_LEFT_HAND_SLICE] = arr[HAND_DOF:]
        return base
    if arm_enabled and arr.shape == (ARM_DOF,):
        base[_ARM_SLICE] = arr
        return base
    ordered_active_sides = tuple(side for side in HAND_VECTOR_ORDER if side in set(active_sides))
    if arr.shape == (HAND_DOF * len(ordered_active_sides),) and ordered_active_sides:
        offset = 0
        for side in ordered_active_sides:
            base[_HAND_SLICES[side]] = arr[offset : offset + HAND_DOF]
            offset += HAND_DOF
        return base
    if arr.shape == (HAND_DOF,) and len(ordered_active_sides) == 1:
        base[_HAND_SLICES[ordered_active_sides[0]]] = arr
        return base
    raise ValueError(
        "Unsupported action vector shape "
        f"{arr.shape}; expected {ACTION_DIM}, {HAND_DOF * 2}, "
        f"{ARM_DOF} arm-only, or active hand dims."
    )


def _array_to_action_chunk(
    value: Any,
    observation: DeploymentObservation,
    *,
    active_sides: tuple[str, ...],
    arm_enabled: bool,
) -> list[np.ndarray]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return [
            _compact_vector_to_action(
                arr,
                observation,
                active_sides=active_sides,
                arm_enabled=arm_enabled,
            )
        ]
    if arr.ndim == 2:
        return [
            _compact_vector_to_action(
                arr[index],
                observation,
                active_sides=active_sides,
                arm_enabled=arm_enabled,
            )
            for index in range(arr.shape[0])
        ]
    raise ValueError(f"Unsupported action array ndim {arr.ndim}; expected 1 or 2.")


def _mapping_value(mapping: Mapping[str, Any], aliases: Sequence[str]) -> Any | None:
    for key in aliases:
        if key in mapping:
            return mapping[key]
    return None


def _component_horizon(value: Any) -> int:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return 1
    if arr.ndim == 2:
        return int(arr.shape[0])
    raise ValueError(f"Unsupported component ndim {arr.ndim}; expected 1 or 2.")


def _component_at(value: Any, index: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1)
    return arr[int(index)].reshape(-1)


def _mapping_to_action_chunk(
    mapping: Mapping[str, Any],
    observation: DeploymentObservation,
    *,
    active_sides: tuple[str, ...],
    arm_enabled: bool,
) -> list[np.ndarray]:
    for key in ("action_chunk", "actions", "action"):
        if key in mapping:
            return normalize_policy_output(
                mapping[key],
                observation,
                active_sides=active_sides,
                arm_enabled=arm_enabled,
            )

    components: dict[str, Any] = {}
    arm_value = _mapping_value(mapping, ("arm", "arm_joint_target", "action.arm.joint_target"))
    if arm_value is not None:
        components["arm"] = arm_value
    for side in HAND_VECTOR_ORDER:
        value = _mapping_value(
            mapping,
            (
                side,
                f"{side}_hand",
                f"{side}_hand_joint_target",
                f"action.{side}_hand.joint_target",
            ),
        )
        if value is not None:
            components[side] = value
    if not components:
        raise ValueError("Policy mapping did not contain action/action_chunk or effector keys.")

    horizon = max(_component_horizon(value) for value in components.values())
    chunk: list[np.ndarray] = []
    for index in range(horizon):
        vector = _base_action(observation)
        if "arm" in components:
            _replace_vector_slice(
                vector,
                _ARM_SLICE,
                _component_at(components["arm"], min(index, _component_horizon(components["arm"]) - 1)),
                expected_dim=ARM_DOF,
            )
        for side in HAND_VECTOR_ORDER:
            if side not in components:
                continue
            _replace_vector_slice(
                vector,
                _HAND_SLICES[side],
                _component_at(components[side], min(index, _component_horizon(components[side]) - 1)),
                expected_dim=HAND_DOF,
            )
        chunk.append(vector)
    return chunk


def normalize_policy_output(
    output: Any,
    observation: DeploymentObservation,
    *,
    active_sides: tuple[str, ...] | None = None,
    arm_enabled: bool | None = None,
) -> list[np.ndarray]:
    """Normalize common model outputs to full absolute action vectors.

    The full vector layout matches collection:
    [arm(7), right_hand(6), left_hand(6)].
    """

    active = tuple(observation.active_sides if active_sides is None else active_sides)
    use_arm = bool(observation.arm_enabled if arm_enabled is None else arm_enabled)
    if output is None:
        return []
    for attr in ("action_chunk", "actions", "action"):
        if hasattr(output, attr):
            return normalize_policy_output(
                getattr(output, attr),
                observation,
                active_sides=active,
                arm_enabled=use_arm,
            )
    if isinstance(output, Mapping):
        return _mapping_to_action_chunk(
            output,
            observation,
            active_sides=active,
            arm_enabled=use_arm,
        )
    if isinstance(output, (list, tuple)) and not output:
        return []
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], Mapping):
        chunk: list[np.ndarray] = []
        for item in output:
            chunk.extend(
                normalize_policy_output(
                    item,
                    observation,
                    active_sides=active,
                    arm_enabled=use_arm,
                )
            )
        return chunk
    return _array_to_action_chunk(
        output,
        observation,
        active_sides=active,
        arm_enabled=use_arm,
    )


class ActionChunkBuffer:
    def __init__(self):
        self._queue: deque[np.ndarray] = deque()
        self.replaced_count = 0
        self.popped_count = 0

    def __len__(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()

    def replace(self, actions: Sequence[np.ndarray]) -> None:
        self._queue.clear()
        self._queue.extend(np.asarray(action, dtype=np.float32).reshape(ACTION_DIM) for action in actions)
        self.replaced_count += 1

    def append(self, actions: Sequence[np.ndarray]) -> None:
        self._queue.extend(np.asarray(action, dtype=np.float32).reshape(ACTION_DIM) for action in actions)

    def pop(self, default: np.ndarray | None = None) -> np.ndarray | None:
        if self._queue:
            self.popped_count += 1
            return self._queue.popleft().astype(np.float32, copy=True)
        if default is None:
            return None
        return np.asarray(default, dtype=np.float32).reshape(ACTION_DIM).copy()


@dataclass(frozen=True)
class ActionSafetyFilterConfig:
    active_sides: tuple[str, ...]
    arm_enabled: bool = False
    arm_shrink_deg: float = 10.0
    max_arm_step_rad: float = 0.10
    max_hand_step: float = 0.0


class ActionSafetyFilter:
    def __init__(self, config: ActionSafetyFilterConfig):
        self.config = config

    @staticmethod
    def _rate_limit(
        target: np.ndarray,
        reference: np.ndarray,
        *,
        max_step: float,
    ) -> tuple[np.ndarray, bool]:
        if float(max_step) <= 0.0:
            return target.astype(np.float32, copy=True), False
        if reference.shape != target.shape or not np.all(np.isfinite(reference)):
            return target.astype(np.float32, copy=True), False
        delta = target - reference
        clipped_delta = np.clip(delta, -float(max_step), float(max_step))
        limited = reference + clipped_delta
        limited_any = bool(np.any(np.abs(clipped_delta - delta) > 1e-12))
        return limited.astype(np.float32, copy=True), limited_any

    @staticmethod
    def _reference_for_slice(
        observation: DeploymentObservation,
        target_slice: slice,
    ) -> np.ndarray:
        last = np.asarray(observation.last_action, dtype=np.float32).reshape(-1)
        if last.shape == (ACTION_DIM,) and np.all(np.isfinite(last[target_slice])):
            return last[target_slice].copy()
        state = np.asarray(observation.state, dtype=np.float32).reshape(-1)
        if state.shape == (ACTION_DIM,):
            return state[target_slice].copy()
        return nan_vector(target_slice.stop - target_slice.start)

    @staticmethod
    def _safe_clip_arm(target: np.ndarray, *, shrink_deg: float) -> tuple[np.ndarray, bool]:
        try:
            from ..arm.retargeter import safe_clip

            clipped, was_clipped, _ = safe_clip(target, shrink_deg=float(shrink_deg))
            return clipped.astype(np.float32, copy=True), bool(was_clipped)
        except Exception:
            # Keep deployment usable in environments where exo_teleop is not
            # importable, while still applying the FR3 limits used by collection.
            q_low = np.deg2rad(np.array([-166, -105, -166, -176, -165, 25, -175], dtype=np.float32))
            q_high = np.deg2rad(np.array([166, 105, 166, -7, 165, 265, 175], dtype=np.float32))
            margin = np.deg2rad(float(shrink_deg))
            clipped = np.clip(target, q_low + margin, q_high - margin)
            return clipped.astype(np.float32, copy=True), bool(np.any(np.abs(clipped - target) > 1e-12))

    def filter(
        self,
        raw_action: Any,
        observation: DeploymentObservation,
        *,
        timestamp: float,
        policy_name: str = "",
    ) -> DeploymentAction:
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if raw.shape != (ACTION_DIM,):
            raise ValueError(f"Expected full action shape {(ACTION_DIM,)}, got {raw.shape}.")

        filtered = raw.astype(np.float32, copy=True)
        faults: list[str] = []
        commands: dict[str, EffectorCommand] = {}
        startup_mode = policy_name == "startup_init"
        max_arm_step_rad = 0.0 if startup_mode else float(self.config.max_arm_step_rad)
        safety_metadata: dict[str, Any] = {
            "policy_name": policy_name,
            "arm_clipped": False,
            "arm_rate_limited": False,
            "hand_rate_limited": {},
        }

        if self.config.arm_enabled:
            arm_target = vector_or_none(filtered[_ARM_SLICE], length=ARM_DOF)
            if arm_target is None:
                faults.append("arm_action_invalid")
            else:
                arm_target, arm_clipped = self._safe_clip_arm(
                    arm_target,
                    shrink_deg=float(self.config.arm_shrink_deg),
                )
                reference = self._reference_for_slice(observation, _ARM_SLICE)
                if (
                    float(max_arm_step_rad) > 0.0
                    and not np.all(np.isfinite(reference))
                ):
                    faults.append("arm_reference_unavailable")
                else:
                    arm_target, arm_limited = self._rate_limit(
                        arm_target,
                        reference,
                        max_step=float(max_arm_step_rad),
                    )
                    filtered[_ARM_SLICE] = arm_target
                    safety_metadata["arm_clipped"] = bool(arm_clipped)
                    safety_metadata["arm_rate_limited"] = bool(arm_limited)
                    commands["arm"] = EffectorCommand(
                        effector_name="arm",
                        side=str(observation.arm_state.get("arm_side") or "right"),
                        value=arm_target,
                        timestamp=float(timestamp),
                        metadata={
                            "source": "policy",
                            "policy_name": policy_name,
                            "raw_action": raw[_ARM_SLICE].copy(),
                            "safety": {
                                "safe_clip": bool(arm_clipped),
                                "rate_limited": bool(arm_limited),
                                "max_step_rad": float(max_arm_step_rad),
                                "shrink_deg": float(self.config.arm_shrink_deg),
                            },
                        },
                    )

        active = set(self.config.active_sides)
        for side in HAND_VECTOR_ORDER:
            if side not in active:
                continue
            target_slice = _HAND_SLICES[side]
            hand_target = vector_or_none(filtered[target_slice], length=HAND_DOF)
            if hand_target is None:
                faults.append(f"{side}_hand_action_invalid")
                continue
            clipped = np.clip(hand_target, 0.0, 1.0)
            filtered[target_slice] = clipped
            safety_metadata["hand_rate_limited"][side] = False
            commands[f"{side}_hand"] = EffectorCommand(
                effector_name=f"{side}_hand",
                side=side,
                value=clipped,
                timestamp=float(timestamp),
                metadata={
                    "source": "policy",
                    "policy_name": policy_name,
                    "raw_action": raw[target_slice].copy(),
                    "safety": {
                        "clip_range": [0.0, 1.0],
                        "rate_limited": False,
                        "max_step": 0.0,
                    },
                },
            )

        return DeploymentAction(
            vector=filtered,
            commands=commands,
            faults=faults,
            metadata=safety_metadata,
        )
