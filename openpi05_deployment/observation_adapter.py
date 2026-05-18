from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from openpi05_deployment.errors import ObservationAdapterError
from openpi05_deployment.metadata import ServerContract
from teleop_runtime.deploy.observation import DeploymentObservation
from teleop_runtime.deploy.schema import STATE_ARM_POSE_SLICES, STATE_HAND_SLICES


@dataclass(frozen=True)
class AdaptedObservation:
    payload: dict[str, Any]
    warnings: tuple[str, ...] = ()


def _required_vector(
    observation: DeploymentObservation,
    target_slice: slice,
    *,
    label: str,
) -> np.ndarray:
    vector = np.asarray(observation.state[target_slice], dtype=np.float32).reshape(-1)
    valid = np.asarray(observation.state_valid[target_slice], dtype=bool).reshape(-1)
    if not np.all(valid) or not np.all(np.isfinite(vector)):
        raise ObservationAdapterError(f"Required observation component {label} is unavailable.")
    return vector.astype(np.float32, copy=True)


def _build_state_vector(
    observation: DeploymentObservation,
    contract: ServerContract,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for side in contract.embodiment.active_sides:
        parts.append(_required_vector(observation, STATE_ARM_POSE_SLICES[side], label=f"{side}_arm_pose"))
    for side in contract.embodiment.active_sides:
        parts.append(_required_vector(observation, STATE_HAND_SLICES[side], label=f"{side}_hand"))
    state = np.concatenate(parts).astype(np.float32, copy=False)
    if state.shape != (contract.embodiment.state_dim,):
        raise ObservationAdapterError(
            f"OpenPI state shape mismatch: expected {(contract.embodiment.state_dim,)}, got {state.shape}."
        )
    return state


def _adapt_tactile(
    observation: DeploymentObservation,
    contract: ServerContract,
    *,
    allow_missing_tactile: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []
    tactile_rows: list[np.ndarray] = []
    tactile_mask: list[bool] = []
    for side in contract.embodiment.tactile_token_order:
        raw = observation.tactile_by_side.get(side)
        vector = np.asarray(raw, dtype=np.float32).reshape(-1) if raw is not None else np.zeros((0,), dtype=np.float32)
        valid = bool(observation.tactile_valid_by_side.get(side, False))
        valid = bool(valid and vector.shape == (contract.tactile_dim,) and np.all(np.isfinite(vector)))
        if not valid:
            if not allow_missing_tactile:
                raise ObservationAdapterError(
                    "Required tactile token is unavailable for "
                    f"{side}_hand; expected shape {(contract.tactile_dim,)}."
                )
            vector = np.zeros((contract.tactile_dim,), dtype=np.float32)
            warnings.append(f"missing_tactile:{side}")
        tactile_rows.append(vector.astype(np.float32, copy=True))
        tactile_mask.append(valid)
    tactile = np.stack(tactile_rows, axis=0).astype(np.float32, copy=False)
    mask = np.asarray(tactile_mask, dtype=bool)
    return tactile, mask, warnings


def build_openpi_observation(
    observation: DeploymentObservation,
    contract: ServerContract,
    *,
    camera_key: str = "front",
    prompt: str | None = None,
    allow_missing_tactile: bool = False,
) -> AdaptedObservation:
    if tuple(observation.active_sides) != contract.embodiment.active_sides:
        raise ObservationAdapterError(
            "Runtime active_sides mismatch: "
            f"expected {contract.embodiment.active_sides}, got {tuple(observation.active_sides)}."
        )
    if not observation.arms_enabled:
        raise ObservationAdapterError("OpenPI deployment requires arms_enabled=True.")
    if not observation.hands_enabled:
        raise ObservationAdapterError("OpenPI deployment requires hands_enabled=True.")

    image_key = str(camera_key or "front")
    image = observation.images.get(image_key)
    if image is None:
        raise ObservationAdapterError(f"Missing required camera frame {image_key!r}.")
    image_array = np.asarray(image)
    if image_array.ndim != 3 or (image_array.shape[-1] != 3 and image_array.shape[0] != 3):
        raise ObservationAdapterError(
            f"Camera frame {image_key!r} must be RGB with shape HxWx3 or 3xHxW; got {image_array.shape}."
        )

    payload: dict[str, Any] = {
        "observation/image": np.asarray(image_array).copy(),
        "observation/state": _build_state_vector(observation, contract),
    }
    if prompt is not None:
        payload["prompt"] = str(prompt)

    warnings: list[str] = []
    if contract.tactile_required:
        tactile, tactile_mask, warnings = _adapt_tactile(
            observation,
            contract,
            allow_missing_tactile=bool(allow_missing_tactile),
        )
        payload["observation/tactile"] = tactile
        payload["observation/tactile_mask"] = tactile_mask

    return AdaptedObservation(payload=payload, warnings=tuple(warnings))
