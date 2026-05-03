from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PolicyDeploy.Model_FR3_deploy.policy import ManipRepFR3Policy


REALMAN_STATE_DIM = 38
REALMAN_ACTION_DIM = 26
ARM_POSE_DOF = 7
ARM_JOINT_DOF = 6
HAND_DOF = 6
ALL_SIDES: tuple[str, ...] = ("left", "right")

ACTION_ARM_POSE_SLICES: dict[str, slice] = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}
ACTION_HAND_SLICES: dict[str, slice] = {
    "left": slice(14, 20),
    "right": slice(20, 26),
}
STATE_ARM_QPOS_SLICES: dict[str, slice] = {
    "left": slice(0, 6),
    "right": slice(19, 25),
}
STATE_ARM_POSE_SLICES: dict[str, slice] = {
    "left": slice(6, 13),
    "right": slice(25, 32),
}
STATE_HAND_SLICES: dict[str, slice] = {
    "left": slice(13, 19),
    "right": slice(32, 38),
}


_ACTION_LAYOUT_DIMS: dict[str, int] = {
    "realman_full": REALMAN_ACTION_DIM,
    "right_arm_pose": ARM_POSE_DOF,
    "right_arm_qpos": ARM_JOINT_DOF,
    "right_hand": HAND_DOF,
    "right_arm_pose_hand": ARM_POSE_DOF + HAND_DOF,
    "right_arm_qpos_hand": ARM_JOINT_DOF + HAND_DOF,
    "bimanual_arm_pose": ARM_POSE_DOF * 2,
    "bimanual_arm_qpos": ARM_JOINT_DOF * 2,
    "bimanual_hand": HAND_DOF * 2,
    "bimanual_arm_pose_hand": ARM_POSE_DOF * 2 + HAND_DOF * 2,
}

_ACTION_LAYOUT_ALIASES: dict[str, str] = {
    "full": "realman_full",
    "realman": "realman_full",
    "realman_full": "realman_full",
    "26": "realman_full",
    "right_pose": "right_arm_pose",
    "right_arm": "right_arm_pose",
    "right_arm_pose": "right_arm_pose",
    "right_qpos": "right_arm_qpos",
    "right_arm_qpos": "right_arm_qpos",
    "right_hand": "right_hand",
    "right_arm_hand": "right_arm_pose_hand",
    "right_arm_pose_hand": "right_arm_pose_hand",
    "right_pose_hand": "right_arm_pose_hand",
    "right_arm_qpos_hand": "right_arm_qpos_hand",
    "both_arm_pose": "bimanual_arm_pose",
    "bimanual_arm_pose": "bimanual_arm_pose",
    "both_arm_qpos": "bimanual_arm_qpos",
    "bimanual_arm_qpos": "bimanual_arm_qpos",
    "both_hand": "bimanual_hand",
    "both_hands": "bimanual_hand",
    "bimanual_hand": "bimanual_hand",
    "bimanual_hands": "bimanual_hand",
    "both_arm_pose_hand": "bimanual_arm_pose_hand",
    "bimanual_arm_pose_hand": "bimanual_arm_pose_hand",
}


def _normalize_active_sides(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("right",)
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "both":
            return ("left", "right")
        parts = [part.strip().lower() for part in text.split(",") if part.strip()]
    else:
        parts = [str(part).strip().lower() for part in value if str(part).strip()]
    invalid = sorted(set(parts) - set(ALL_SIDES))
    if invalid:
        raise ValueError(f"Unsupported active_sides={value!r}; invalid sides={invalid}.")
    if not parts:
        raise ValueError("active_sides must not be empty.")
    return tuple(side for side in ALL_SIDES if side in set(parts))


def _auto_action_layout(action_dim: int, active_sides: tuple[str, ...]) -> str:
    if int(action_dim) == REALMAN_ACTION_DIM:
        return "realman_full"
    if active_sides == ("right",):
        if int(action_dim) == ARM_POSE_DOF:
            return "right_arm_pose"
        if int(action_dim) == ARM_POSE_DOF + HAND_DOF:
            return "right_arm_pose_hand"
        if int(action_dim) == HAND_DOF:
            raise ValueError(
                "Cannot infer a 6-D right-side Realman action layout. "
                "Set ACTION_LAYOUT=right_hand or ACTION_LAYOUT=right_arm_qpos explicitly."
            )
    if active_sides == ("left", "right"):
        if int(action_dim) == ARM_POSE_DOF * 2:
            return "bimanual_arm_pose"
        if int(action_dim) == ARM_POSE_DOF * 2 + HAND_DOF * 2:
            return "bimanual_arm_pose_hand"
        if int(action_dim) == ARM_JOINT_DOF * 2:
            raise ValueError(
                "Cannot infer a 12-D bimanual Realman action layout. "
                "Set ACTION_LAYOUT=bimanual_hand or ACTION_LAYOUT=bimanual_arm_qpos explicitly."
            )
    raise ValueError(
        f"Unsupported Realman action_dim={action_dim} for active_sides={active_sides}. "
        "Expected 26 for the native Realman schema, or set ACTION_LAYOUT for a compact checkpoint."
    )


def _resolve_action_layout(value: Any, *, action_dim: int, active_sides: tuple[str, ...]) -> str:
    text = "auto" if value is None else str(value).strip().lower()
    if text in {"", "auto", "infer"}:
        layout = _auto_action_layout(action_dim, active_sides)
    else:
        layout = _ACTION_LAYOUT_ALIASES.get(text)
        if layout is None:
            valid = sorted(set(_ACTION_LAYOUT_ALIASES.values()))
            raise ValueError(f"Unsupported ACTION_LAYOUT={value!r}; expected auto or one of {valid}.")
    expected_dim = int(_ACTION_LAYOUT_DIMS[layout])
    if int(action_dim) != expected_dim:
        raise ValueError(
            f"ACTION_LAYOUT={layout} expects action_dim={expected_dim}, "
            f"but checkpoint action_dim={action_dim}."
        )
    if layout.startswith("right_") and active_sides != ("right",):
        raise ValueError(f"ACTION_LAYOUT={layout} requires ACTIVE_SIDES=right.")
    if layout.startswith("bimanual_") and active_sides != ("left", "right"):
        raise ValueError(f"ACTION_LAYOUT={layout} requires ACTIVE_SIDES=both.")
    return layout


def _safe_vector(value: Any, *, length: int) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        arr = np.zeros((0,), dtype=np.float32)
    if arr.shape != (int(length),):
        return np.zeros((int(length),), dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _hold_action(observation: Any) -> np.ndarray:
    hold = getattr(observation, "hold_action", None)
    if callable(hold):
        return _safe_vector(hold(), length=REALMAN_ACTION_DIM)
    return np.zeros((REALMAN_ACTION_DIM,), dtype=np.float32)


def _hold_component(observation: Any, layout: str) -> np.ndarray:
    hold = _hold_action(observation)
    if layout == "realman_full":
        return hold
    if layout == "right_arm_pose":
        return hold[ACTION_ARM_POSE_SLICES["right"]].copy()
    if layout == "right_hand":
        return hold[ACTION_HAND_SLICES["right"]].copy()
    if layout == "right_arm_pose_hand":
        return np.concatenate(
            (
                hold[ACTION_ARM_POSE_SLICES["right"]],
                hold[ACTION_HAND_SLICES["right"]],
            )
        ).astype(np.float32, copy=False)
    if layout == "bimanual_arm_pose":
        return np.concatenate(
            (
                hold[ACTION_ARM_POSE_SLICES["left"]],
                hold[ACTION_ARM_POSE_SLICES["right"]],
            )
        ).astype(np.float32, copy=False)
    if layout == "bimanual_hand":
        return np.concatenate(
            (
                hold[ACTION_HAND_SLICES["left"]],
                hold[ACTION_HAND_SLICES["right"]],
            )
        ).astype(np.float32, copy=False)
    if layout == "bimanual_arm_pose_hand":
        return np.concatenate(
            (
                hold[ACTION_ARM_POSE_SLICES["left"]],
                hold[ACTION_ARM_POSE_SLICES["right"]],
                hold[ACTION_HAND_SLICES["left"]],
                hold[ACTION_HAND_SLICES["right"]],
            )
        ).astype(np.float32, copy=False)

    state = _safe_vector(getattr(observation, "state", None), length=REALMAN_STATE_DIM)
    if layout == "right_arm_qpos":
        return state[STATE_ARM_QPOS_SLICES["right"]].copy()
    if layout == "right_arm_qpos_hand":
        return np.concatenate(
            (
                state[STATE_ARM_QPOS_SLICES["right"]],
                hold[ACTION_HAND_SLICES["right"]],
            )
        ).astype(np.float32, copy=False)
    if layout == "bimanual_arm_qpos":
        return np.concatenate(
            (
                state[STATE_ARM_QPOS_SLICES["left"]],
                state[STATE_ARM_QPOS_SLICES["right"]],
            )
        ).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported action layout={layout!r}.")


class ManipRepRealmanPolicy(ManipRepFR3Policy):
    """ManipRep adapter for teleop_runtime Realman/RM65 deployment.

    Native Realman checkpoints use:
    state  = [left_qpos6, left_pose7, left_hand6, right_qpos6, right_pose7, right_hand6]
    action = [left_pose7, right_pose7, left_hand6, right_hand6]
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        camera_key: str | None = None,
        tactile_side: str | None = None,
        require_image: bool = True,
        require_tactile: bool = True,
        strict_state_dim: bool = True,
        action_clip_nan_to_hold: bool = True,
        action_chunk_size: int | str | None = None,
        active_sides: str | tuple[str, ...] = "right",
        action_layout: str = "auto",
        strict_realman_schema: bool = True,
    ) -> None:
        self.active_sides = _normalize_active_sides(active_sides)
        self.strict_realman_schema = bool(strict_realman_schema)
        super().__init__(
            checkpoint=checkpoint,
            device=device,
            camera_key=camera_key,
            tactile_side=tactile_side,
            require_image=require_image,
            require_tactile=require_tactile,
            strict_state_dim=strict_state_dim,
            action_clip_nan_to_hold=action_clip_nan_to_hold,
            action_chunk_size=action_chunk_size,
        )
        if self.strict_realman_schema and int(self.state_dim) != REALMAN_STATE_DIM:
            raise ValueError(
                f"Realman deployment requires checkpoint state_dim={REALMAN_STATE_DIM}, "
                f"got state_dim={self.state_dim}. A 19-D checkpoint is an FR3-style model "
                "and must not be sent to Realman without retraining or an explicit adapter."
            )
        self.action_layout = _resolve_action_layout(
            action_layout,
            action_dim=int(self.action_dim),
            active_sides=self.active_sides,
        )

    @property
    def status_text(self) -> str:
        mode = self._mode_label()
        chunk = "all" if self.action_chunk_size is None else f"first{self.action_chunk_size}"
        active = "both" if self.active_sides == ("left", "right") else ",".join(self.active_sides)
        return (
            f"maniprep_realman={mode}:state={self.state_dim}:action={self.action_dim}"
            f":layout={self.action_layout}:active={active}:ckpt={self.checkpoint_path.name}:chunk={chunk}"
        )

    def _prepare_state(self, observation: Any) -> torch.Tensor:
        state = np.asarray(getattr(observation, "state"), dtype=np.float32).reshape(-1)
        if self.strict_realman_schema and state.shape != (REALMAN_STATE_DIM,):
            raise ValueError(f"Expected Realman deployment state dim={REALMAN_STATE_DIM}, got {state.shape}.")
        if self.strict_state_dim and state.shape != (self.state_dim,):
            raise ValueError(f"Expected checkpoint state dim={self.state_dim}, got deployment state shape={state.shape}.")
        if state.shape[0] < self.state_dim:
            padded = np.zeros((self.state_dim,), dtype=np.float32)
            padded[: state.shape[0]] = state
            state = padded
        elif state.shape[0] > self.state_dim:
            state = state[: self.state_dim]
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return torch.from_numpy(state).unsqueeze(0).to(device=self.device)

    def _extract_model_action(self, output: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
        action = output["action"].detach().to("cpu").numpy().astype(np.float32, copy=False)
        if action.ndim == 3:
            if int(action.shape[0]) != 1:
                raise ValueError(f"Expected batch size 1 action output, got {action.shape}.")
            action = action[0]
        elif action.ndim == 2:
            pass
        elif action.ndim == 1:
            action = action.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported action output shape {action.shape}.")
        if int(action.shape[-1]) != self.action_dim:
            raise ValueError(f"Expected checkpoint action dim={self.action_dim}, got shape={action.shape}.")
        return action, list(action.shape)

    def _replace_nonfinite_action(self, action: np.ndarray, observation: Any) -> np.ndarray:
        if np.all(np.isfinite(action)):
            return action
        fallback = _hold_component(observation, self.action_layout).reshape(1, -1)
        if fallback.shape[-1] != action.shape[-1]:
            fallback = np.zeros((1, action.shape[-1]), dtype=np.float32)
        return np.where(np.isfinite(action), action, fallback).astype(np.float32, copy=False)

    def _format_action_output(self, action: np.ndarray) -> dict[str, Any]:
        if self.action_layout == "realman_full":
            return {"action_chunk": action}
        if self.action_layout == "right_arm_pose":
            return {"right_arm_pose": action}
        if self.action_layout == "right_arm_qpos":
            return {"right_arm_qpos": action}
        if self.action_layout == "right_hand":
            return {"right_hand": action}
        if self.action_layout == "right_arm_pose_hand":
            return {
                "right_arm_pose": action[:, :ARM_POSE_DOF],
                "right_hand": action[:, ARM_POSE_DOF : ARM_POSE_DOF + HAND_DOF],
            }
        if self.action_layout == "right_arm_qpos_hand":
            return {
                "right_arm_qpos": action[:, :ARM_JOINT_DOF],
                "right_hand": action[:, ARM_JOINT_DOF : ARM_JOINT_DOF + HAND_DOF],
            }
        if self.action_layout == "bimanual_arm_pose":
            return {
                "left_arm_pose": action[:, :ARM_POSE_DOF],
                "right_arm_pose": action[:, ARM_POSE_DOF : ARM_POSE_DOF * 2],
            }
        if self.action_layout == "bimanual_arm_qpos":
            return {
                "left_arm_qpos": action[:, :ARM_JOINT_DOF],
                "right_arm_qpos": action[:, ARM_JOINT_DOF : ARM_JOINT_DOF * 2],
            }
        if self.action_layout == "bimanual_hand":
            return {
                "left_hand": action[:, :HAND_DOF],
                "right_hand": action[:, HAND_DOF : HAND_DOF * 2],
            }
        if self.action_layout == "bimanual_arm_pose_hand":
            return {
                "left_arm_pose": action[:, :ARM_POSE_DOF],
                "right_arm_pose": action[:, ARM_POSE_DOF : ARM_POSE_DOF * 2],
                "left_hand": action[:, ARM_POSE_DOF * 2 : ARM_POSE_DOF * 2 + HAND_DOF],
                "right_hand": action[:, ARM_POSE_DOF * 2 + HAND_DOF : ARM_POSE_DOF * 2 + HAND_DOF * 2],
            }
        raise ValueError(f"Unsupported action layout={self.action_layout!r}.")

    def predict(self, observation: Any) -> dict[str, Any]:
        inputs = self._model_inputs(observation)
        with torch.inference_mode():
            output = self.model(**inputs)
        action, model_action_shape = self._extract_model_action(output)
        if self.action_chunk_size is not None:
            action = action[: self.action_chunk_size]
            if int(action.shape[0]) <= 0:
                raise ValueError(
                    f"action_chunk_size={self.action_chunk_size} produced an empty action chunk "
                    f"from model output shape {model_action_shape}."
                )
        if self.action_clip_nan_to_hold:
            action = self._replace_nonfinite_action(action, observation)
        if not np.all(np.isfinite(action)):
            raise ValueError("Model produced non-finite action values.")
        payload = self._format_action_output(action)
        self._last_metadata = {
            "checkpoint": str(self.checkpoint_path),
            "experiment": self.experiment_name,
            "mode": self._mode_label(),
            "active_sides": list(self.active_sides),
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "action_layout": self.action_layout,
            "action_shape": list(action.shape),
            "model_action_shape": model_action_shape,
            "action_chunk_size": self.action_chunk_size,
        }
        payload["metadata"] = dict(self._last_metadata)
        return payload


Policy = ManipRepRealmanPolicy


def create_policy(**kwargs: Any) -> ManipRepRealmanPolicy:
    return ManipRepRealmanPolicy(**kwargs)
