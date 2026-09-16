"""Typed DexTeleop-v2.2 policy I/O adapters for deployment.

This module is deliberately hardware-free.  It turns a v2.2 model observation
into typed float/token inputs and, only when an explicit legacy-q6 mapping is
declared, adapts a typed model action into the existing safety-gated q6 path.
It never sends commands or opens a device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from dataset_contracts.v22 import (
    ARM_ACTION_DIM,
    DatasetLayoutBindingError,
    DatasetLayoutError,
    EndEffectorChannelSpec,
    INVALID_DISCRETE_TOKEN,
    PolicyIOLayout,
    SIDES,
)
from dataset_contracts.state_input import (
    STATE_INPUT_MODE_RAW,
    StateInputMode,
    deployment_state_input,
    normalize_state_input_mode,
    state_input_dim,
)
from robot_motion.action import EndEffectorTarget, HandActionSpec, RobotAction

from .actions import NormalizedAction
from .action_semantics import ArmHoldKeyframePrediction
from .observation import DeploymentObservation
from .schema import ACTION_ARM_POSE_SLICES, ACTION_DIM, ACTION_HAND_SLICES, STATE_ARM_POSE_SLICES, STATE_HAND_SLICES


class V22DeploymentError(DatasetLayoutError):
    """Raised before a v2.2 policy value could reach the legacy safety path."""


def _float_matrix(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise V22DeploymentError(f"{name} must be a numeric vector or matrix.") from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise V22DeploymentError(f"{name} must be 1D or 2D [horizon, dim], got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise V22DeploymentError(f"{name} contains non-finite values.")
    return array.astype(np.float32, copy=True)


def _token_matrix(value: Any, *, name: str, horizon: int | None = None) -> np.ndarray:
    if value is None:
        if horizon is None:
            return np.zeros((1, 0), dtype=np.int64)
        return np.zeros((int(horizon), 0), dtype=np.int64)
    raw = np.asarray(value)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim != 2:
        raise V22DeploymentError(f"{name} must be 1D or 2D [horizon, token_dim], got {raw.shape}.")
    if horizon is not None and raw.shape[0] != int(horizon):
        raise V22DeploymentError(
            f"{name} horizon {raw.shape[0]} differs from float action horizon {int(horizon)}."
        )
    tokens = np.empty(raw.shape, dtype=np.int64)
    for index, value_at_index in np.ndenumerate(raw):
        if isinstance(value_at_index, (bool, np.bool_)):
            raise V22DeploymentError(f"{name}{index} must be an integer token, not a boolean.")
        try:
            token = int(value_at_index)
            if float(value_at_index) != float(token):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise V22DeploymentError(f"{name}{index} must be an exact integer token.") from exc
        tokens[index] = token
    return tokens


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    current: Any = mapping
    for part in str(key).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first_mapping_value(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _mapping_value(mapping, key)
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class V22PolicyActionChunk:
    """Model output with independent float32 action and int64 token matrices."""

    actions: np.ndarray
    discrete_actions: np.ndarray = field(default_factory=lambda: np.zeros((1, 0), dtype=np.int64))
    layout_hash: str | None = None
    arm_hold_probabilities: np.ndarray | None = None
    arm_hold_contract_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = _float_matrix(self.actions, name="v2.2 actions")
        tokens = _token_matrix(
            self.discrete_actions,
            name="v2.2 discrete actions",
            horizon=actions.shape[0],
        )
        layout_hash = None if self.layout_hash is None else str(self.layout_hash).strip()
        hold = self.arm_hold_probabilities
        hold_hash = self.arm_hold_contract_hash
        if (hold is None) != (hold_hash is None):
            raise V22DeploymentError(
                "arm-hold probabilities and contract hash must be supplied together."
            )
        if hold is not None:
            hold = np.asarray(hold, dtype=np.float32)
            if hold.ndim == 1:
                hold = hold.reshape(1, -1)
            if hold.shape != (actions.shape[0], 2):
                raise V22DeploymentError(
                    "arm-hold probabilities must align with the action horizon."
                )
            if (
                not np.all(np.isfinite(hold))
                or np.any(hold < 0.0)
                or np.any(hold > 1.0)
            ):
                raise V22DeploymentError(
                    "arm-hold probabilities must be finite values in [0, 1]."
                )
            hold_hash = str(hold_hash).strip()
            if not hold_hash:
                raise V22DeploymentError(
                    "arm-hold contract hash must be non-empty."
                )
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "discrete_actions", tokens)
        object.__setattr__(self, "layout_hash", layout_hash or None)
        object.__setattr__(self, "arm_hold_probabilities", hold)
        object.__setattr__(self, "arm_hold_contract_hash", hold_hash)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class V22DeploymentObservation:
    """Model-facing deployment observation with typed state tokens."""

    state: np.ndarray
    state_discrete: np.ndarray
    policy_io_layout_hash: str
    timestamp: float
    step_index: int
    images: Mapping[str, np.ndarray] = field(default_factory=dict)
    active_sides: tuple[str, ...] = SIDES
    metadata: Mapping[str, Any] = field(default_factory=dict)
    state_input_mode: StateInputMode | str = STATE_INPUT_MODE_RAW
    base_state_dim: int | None = None

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32).reshape(-1)
        tokens = _token_matrix(self.state_discrete, name="v2.2 observation.state.discrete", horizon=1)
        state_input_mode = normalize_state_input_mode(self.state_input_mode)
        if self.base_state_dim is None:
            if state_input_mode == STATE_INPUT_MODE_RAW:
                base_state_dim = int(state.size)
            else:
                if state.size % 2:
                    raise V22DeploymentError(
                        "masked v2.2 observation state must have an even concatenated dimension."
                    )
                base_state_dim = int(state.size // 2)
        else:
            base_state_dim = int(self.base_state_dim)
        expected_dim = state_input_dim(base_state_dim, state_input_mode)
        if state.shape != (expected_dim,):
            raise V22DeploymentError(
                f"v2.2 observation state has shape {state.shape}; expected ({expected_dim},) for "
                f"{state_input_mode!r}."
            )
        if not np.all(np.isfinite(state)):
            raise V22DeploymentError("v2.2 observation state must be finite.")
        object.__setattr__(self, "state", state.copy())
        object.__setattr__(self, "state_discrete", tokens[0].copy())
        object.__setattr__(self, "policy_io_layout_hash", str(self.policy_io_layout_hash))
        object.__setattr__(self, "images", {str(key): np.asarray(value).copy() for key, value in self.images.items()})
        object.__setattr__(self, "active_sides", tuple(str(side) for side in self.active_sides))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "state_input_mode", state_input_mode)
        object.__setattr__(self, "base_state_dim", base_state_dim)

    def model_input(self) -> dict[str, Any]:
        """Canonical typed input names for custom policy/training adapters."""

        return {
            "state": self.state.copy(),
            "state_discrete": self.state_discrete.copy(),
            "state_input_mode": self.state_input_mode,
            "base_state_dim": int(self.base_state_dim),
            "observation.state": self.state.copy(),
            "observation.state.discrete": self.state_discrete.copy(),
            "policy_io_layout_hash": self.policy_io_layout_hash,
            "images": {key: value.copy() for key, value in self.images.items()},
            "timestamp": float(self.timestamp),
            "step_index": int(self.step_index),
            "active_sides": tuple(self.active_sides),
            "metadata": dict(self.metadata),
        }


def _select_numeric(channel: EndEffectorChannelSpec, raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    try:
        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if channel.source_indices is not None:
        indices = np.asarray(channel.source_indices, dtype=int)
        if indices.shape != (channel.dim,) or np.any(indices >= vector.size):
            return None
        vector = vector[indices]
    if vector.shape != (channel.dim,) or not np.all(np.isfinite(vector)):
        return None
    return vector.astype(np.float32, copy=True)


def _channel_state_value(
    *,
    observation: DeploymentObservation,
    layout: PolicyIOLayout,
    side: str,
) -> Any:
    channel = layout.channel("state", side)
    if channel.value_type == "none":
        return None
    spec = layout.end_effector(side)
    hand_state = dict(observation.hand_states.get(side) or {})
    raw = _mapping_value(hand_state, spec.state_source_field)
    if raw is None and spec.state_source_field in {"gripper_state", "gripper_state_scalar", "state"}:
        raw = _first_mapping_value(
            hand_state,
            ("gripper_state", "gripper_state_scalar", "state", "joint_pos"),
        )
    if raw is None:
        raw = observation.state[STATE_HAND_SLICES[side]]
    if channel.value_type == "continuous":
        return _select_numeric(channel, raw)

    if channel.source_indices is not None:
        numeric = _select_numeric(channel, raw)
        if numeric is None:
            return None
        rounded = np.rint(numeric)
        if not np.allclose(numeric, rounded):
            return None
        return [int(value) for value in rounded]

    values: list[Any] = []
    raw_vector = None if raw is None else np.asarray(raw, dtype=object).reshape(-1)
    for index, token in enumerate(channel.tokens):
        value = None
        if token.source_metadata_key:
            value = _mapping_value(hand_state, token.source_metadata_key)
            if value is None:
                value = _mapping_value(observation.metadata, token.source_metadata_key)
        elif raw_vector is not None and raw_vector.shape == (channel.dim,):
            value = raw_vector[index]
        elif channel.dim == 1 and raw is not None:
            value = raw
        if value is None:
            return None
        values.append(value)
    return values


def _chunk_layout_hash(metadata: Mapping[str, Any]) -> str | None:
    value = _first_mapping_value(
        metadata,
        (
            "policy_io_layout_hash",
            "remote_policy.server.policy_io_layout_hash",
            "remote_policy.server.dataset_adapter.policy_io_layout_hash",
            "remote_policy.server_summary.policy_io_layout_hash",
        ),
    )
    return None if value is None else str(value).strip() or None


def _server_artifact_declares_layout(metadata: Mapping[str, Any], layout: PolicyIOLayout) -> bool:
    """Accept an unhashed remote action only with the server's actual layout.

    The OpenPI response carries no per-action layout hash. Its transport adds
    the negotiated server artifact to action metadata, so compare that layout
    directly instead of requiring a redundant hash on every response.
    """

    advertised = _mapping_value(
        metadata, "remote_policy.server.policy_artifact.policy_io_layout"
    )
    return isinstance(advertised, Mapping) and dict(advertised) == layout.to_dict()


def _action_end_effector_component(
    layout: PolicyIOLayout,
    *,
    side: str,
    value_type: str,
):
    """Select the configured target component, never the same-side arm pose."""

    expected_name = f"{side}.end_effector.target"
    matches = tuple(
        component
        for component in layout.components("action", value_type)
        if component.name == expected_name
    )
    if len(matches) != 1:
        raise V22DeploymentError(
            f"Expected exactly one v2.2 action component {expected_name!r}, "
            f"found {len(matches)}."
        )
    return matches[0]


def coerce_v22_policy_action_chunk(value: Any) -> V22PolicyActionChunk:
    """Coerce a local/remote policy response without converting tokens to float."""

    if isinstance(value, V22PolicyActionChunk):
        return value
    if isinstance(value, Mapping):
        metadata = value.get("metadata") if isinstance(value.get("metadata"), Mapping) else {}
        actions = _first_mapping_value(value, ("actions", "action", "action.continuous"))
        tokens = _first_mapping_value(
            value,
            ("discrete_actions", "action.discrete", "actions_discrete", "action_discrete"),
        )
        layout_hash = value.get("policy_io_layout_hash") or _chunk_layout_hash(metadata)
        if actions is None:
            raise V22DeploymentError("v2.2 policy mapping is missing action float values.")
        return V22PolicyActionChunk(
            actions=actions,
            discrete_actions=tokens,
            layout_hash=None if layout_hash is None else str(layout_hash),
            arm_hold_probabilities=value.get("arm_hold_probabilities"),
            arm_hold_contract_hash=value.get("arm_hold_contract_hash"),
            metadata=metadata,
        )
    if hasattr(value, "actions"):
        metadata = getattr(value, "metadata", {})
        metadata = metadata if isinstance(metadata, Mapping) else {}
        layout_hash = getattr(value, "layout_hash", None) or _chunk_layout_hash(metadata)
        return V22PolicyActionChunk(
            actions=getattr(value, "actions"),
            discrete_actions=getattr(value, "discrete_actions", None),
            layout_hash=None if layout_hash is None else str(layout_hash),
            arm_hold_probabilities=getattr(
                value,
                "arm_hold_probabilities",
                None,
            ),
            arm_hold_contract_hash=getattr(
                value,
                "arm_hold_contract_hash",
                None,
            ),
            metadata=metadata,
        )
    return V22PolicyActionChunk(actions=value)


class V22DeploymentAdapter:
    """Bind v2.2 policy inputs/outputs to an explicitly configured layout.

    ``to_legacy_normalized_actions`` is a compatibility bridge for current q6
    safety/dispatch code.  It requires a per-channel ``legacy_q6_slot_indices``
    declaration and fails closed for an embodiment that needs a native adapter.
    """

    def __init__(
        self,
        layout: PolicyIOLayout,
        *,
        require_action_layout_hash: bool = True,
        state_input_mode: StateInputMode | str = STATE_INPUT_MODE_RAW,
    ) -> None:
        self.layout = layout
        self.require_action_layout_hash = bool(require_action_layout_hash)
        self.state_input_mode = normalize_state_input_mode(state_input_mode)

    @property
    def end_effector_sides(self) -> tuple[str, ...]:
        return tuple(side for side in SIDES if self.layout.end_effector(side).present)

    def observation_from_deployment(self, observation: DeploymentObservation) -> V22DeploymentObservation:
        arm_poses: dict[str, np.ndarray] = {}
        for side in SIDES:
            pose = np.asarray(observation.state[STATE_ARM_POSE_SLICES[side]], dtype=np.float32).reshape(-1)
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                raise V22DeploymentError(f"v2.2 deployment observation lacks a finite {side} arm pose.")
            arm_poses[side] = pose
        end_effector_values = {
            side: _channel_state_value(observation=observation, layout=self.layout, side=side)
            for side in SIDES
            if self.layout.end_effector(side).present
        }
        try:
            packed = self.layout.pack_observation_state(
                arm_pose_by_side=arm_poses,
                end_effector_by_side=end_effector_values,
                strict=True,
            )
        except DatasetLayoutError as exc:
            raise V22DeploymentError(f"Could not build typed v2.2 state: {exc}") from exc
        try:
            model_state = deployment_state_input(packed.continuous, self.state_input_mode)
        except ValueError as exc:
            raise V22DeploymentError(f"Could not prepare v2.2 model state input: {exc}") from exc
        return V22DeploymentObservation(
            state=model_state,
            state_discrete=packed.discrete,
            policy_io_layout_hash=self.layout.layout_hash,
            timestamp=observation.timestamp,
            step_index=observation.step_index,
            images=observation.images,
            active_sides=observation.active_sides,
            metadata={
                **dict(observation.metadata),
                "policy_io_layout_hash": self.layout.layout_hash,
                "v22_end_effector_sides": list(self.end_effector_sides),
                "state_input_mode": self.state_input_mode,
                "base_state_dim": int(self.layout.state_dim),
                "model_state_dim": int(model_state.size),
            },
            state_input_mode=self.state_input_mode,
            base_state_dim=self.layout.state_dim,
        )

    def _validate_chunk(self, chunk: V22PolicyActionChunk) -> None:
        if chunk.actions.shape[1] != self.layout.action_dim:
            raise V22DeploymentError(
                f"v2.2 action float dim mismatch: expected {self.layout.action_dim}, "
                f"got {chunk.actions.shape[1]}."
            )
        if chunk.discrete_actions.shape[1] != self.layout.action_discrete_dim:
            raise V22DeploymentError(
                "v2.2 action token dim mismatch: expected "
                f"{self.layout.action_discrete_dim}, got {chunk.discrete_actions.shape[1]}."
            )
        if self.require_action_layout_hash and (
            (chunk.layout_hash is not None and chunk.layout_hash != self.layout.layout_hash)
            or (
                chunk.layout_hash is None
                and not _server_artifact_declares_layout(chunk.metadata, self.layout)
            )
        ):
            raise DatasetLayoutBindingError(
                "v2.2 policy action layout mismatch: "
                f"expected={self.layout.layout_hash!r}, actual={chunk.layout_hash!r}."
            )
        for component in self.layout.components("action", "discrete"):
            assert component.side is not None
            channel = self.layout.channel("action", component.side)
            for local_index, token_spec in enumerate(channel.tokens):
                values = chunk.discrete_actions[:, component.offset + local_index]
                allowed = token_spec.category_ids
                if any(int(value) not in allowed for value in values):
                    raise V22DeploymentError(
                        f"v2.2 action token {token_spec.name!r} contains an undeclared category."
                    )

    def _legacy_hand_vector(
        self,
        *,
        side: str,
        continuous_action: np.ndarray,
        discrete_action: np.ndarray,
    ) -> np.ndarray | None:
        channel = self.layout.channel("action", side)
        if channel.value_type == "none":
            return None
        slots = channel.legacy_q6_slot_indices
        if slots is None:
            raise V22DeploymentError(
                f"{side} end effector {self.layout.end_effector(side).adapter!r} has no "
                "legacy_q6_slot_indices; use a native adapter instead of guessing a q6 bridge."
            )
        target = np.zeros((6,), dtype=np.float32)
        component = _action_end_effector_component(
            self.layout,
            side=side,
            value_type=channel.value_type,
        )
        if channel.value_type == "continuous":
            values = continuous_action[component.offset : component.stop]
            if not np.all(np.isfinite(values)):
                raise V22DeploymentError(f"{side} v2.2 continuous end-effector action is non-finite.")
            target[np.asarray(slots, dtype=int)] = values
            return target
        for local_index, token_spec in enumerate(channel.tokens):
            token = int(discrete_action[component.offset + local_index])
            category = next(category for category in token_spec.categories if category.id == token)
            backend_value = category.backend_value
            if isinstance(backend_value, (bool, np.bool_)) or backend_value is None:
                raise V22DeploymentError(
                    f"Discrete token {token_spec.name!r} category {category.label!r} needs a numeric "
                    "backend_value for the explicit legacy-q6 bridge."
                )
            try:
                numeric = float(backend_value)
            except (TypeError, ValueError) as exc:
                raise V22DeploymentError(
                    f"Discrete token {token_spec.name!r} backend_value must be scalar for legacy q6."
                ) from exc
            if not np.isfinite(numeric):
                raise V22DeploymentError(
                    f"Discrete token {token_spec.name!r} backend_value must be finite for legacy q6."
                )
            target[int(slots[local_index])] = np.float32(numeric)
        return target

    def _hand_specs_metadata(self) -> dict[str, dict[str, Any]]:
        specs: dict[str, dict[str, Any]] = {}
        for side in self.end_effector_sides:
            channel = self.layout.channel("action", side)
            if channel.legacy_q6_slot_indices is None:
                raise V22DeploymentError(
                    f"{side} end effector has no legacy_q6_slot_indices; "
                    "it cannot use the q6 compatibility bridge."
                )
            active_mask = tuple(index in set(channel.legacy_q6_slot_indices) for index in range(6))
            spec = HandActionSpec(
                side=side,
                hand_model=self.layout.end_effector(side).adapter,
                slot_count=6,
                active_dof=channel.dim,
                active_mask=active_mask,
                joint_names=channel.names,
                command_space=str(
                    self.layout.end_effector(side).metadata.get(
                        "legacy_q6_command_space", "normalized_0_1"
                    )
                ),
                slot_map={name: int(slot) for name, slot in zip(channel.names, channel.legacy_q6_slot_indices)},
                invalid_slot_value="zero_padding",
                metadata={
                    "policy_io_layout_hash": self.layout.layout_hash,
                    "v22_adapter": self.layout.end_effector(side).adapter,
                },
            )
            specs[side] = spec.to_metadata()
        return specs

    def to_legacy_normalized_actions(
        self,
        output: Any,
        *,
        source: str = "v22_policy_io_adapter",
    ) -> list[NormalizedAction]:
        """Return old 26D normalized actions for the existing safety gate.

        No side without a configured end effector is included in
        ``end_effector_enabled_sides``.  The common action pipeline consumes
        this metadata so a left-arm-only installation never manufactures a
        left-hand command from zero padding.
        """

        chunk = coerce_v22_policy_action_chunk(output)
        self._validate_chunk(chunk)
        hand_specs = self._hand_specs_metadata()
        normalized: list[NormalizedAction] = []
        for index in range(chunk.horizon):
            float_action = chunk.actions[index]
            token_action = chunk.discrete_actions[index]
            vector = np.zeros((ACTION_DIM,), dtype=np.float32)
            vector[ACTION_ARM_POSE_SLICES["left"]] = float_action[:7]
            vector[ACTION_ARM_POSE_SLICES["right"]] = float_action[7:ARM_ACTION_DIM]
            for side in self.end_effector_sides:
                hand_target = self._legacy_hand_vector(
                    side=side,
                    continuous_action=float_action,
                    discrete_action=token_action,
                )
                assert hand_target is not None
                vector[ACTION_HAND_SLICES[side]] = hand_target
            metadata = {
                **dict(chunk.metadata),
                "source": str(source),
                "policy_io_layout_hash": self.layout.layout_hash,
                "end_effector_enabled_sides": list(self.end_effector_sides),
                "hand_action_specs_by_side": hand_specs,
                "hand_active_mask_by_side": {
                    side: list(spec["active_mask"]) for side, spec in hand_specs.items()
                },
                "v22_policy_action_index": int(index),
            }
            hold_prediction = None
            if chunk.arm_hold_probabilities is not None:
                assert chunk.arm_hold_contract_hash is not None
                row = chunk.arm_hold_probabilities[index]
                hold_prediction = ArmHoldKeyframePrediction(
                    probabilities=(
                        float(row[0]),
                        float(row[1]),
                    ),
                    contract_hash=chunk.arm_hold_contract_hash,
                )
            normalized.append(
                NormalizedAction(
                    vector=vector,
                    metadata=metadata,
                    arm_hold_prediction=hold_prediction,
                )
            )
        return normalized

    def to_native_robot_actions(
        self,
        output: Any,
        *,
        source: str = "v22_native_policy_io_adapter",
    ) -> list[RobotAction]:
        """Lift typed policy outputs into RobotAction without a q6 projection.

        These actions still require the normal safety gate and a runtime
        end-effector adapter before dispatch. This method only preserves typed
        semantic values across the canonical action boundary.
        """

        chunk = coerce_v22_policy_action_chunk(output)
        self._validate_chunk(chunk)
        actions: list[RobotAction] = []
        for index in range(chunk.horizon):
            float_action = chunk.actions[index]
            token_action = chunk.discrete_actions[index]
            targets: dict[str, EndEffectorTarget] = {}
            for side in self.end_effector_sides:
                channel = self.layout.channel("action", side)
                component = _action_end_effector_component(
                    self.layout,
                    side=side,
                    value_type=channel.value_type,
                )
                values = (
                    float_action[component.offset : component.stop]
                    if channel.value_type == "continuous"
                    else token_action[component.offset : component.stop]
                )
                targets[side] = EndEffectorTarget(
                    value_type=channel.value_type,
                    values=values,
                    adapter=self.layout.end_effector(side).adapter,
                    names=channel.names,
                    metadata={
                        **(
                            {
                                "units": tuple(
                                    str(channel.units)
                                    for _ in range(channel.dim)
                                )
                            }
                            if channel.units is not None
                            else (
                                {"units": ("token",) * channel.dim}
                                if channel.value_type == "discrete"
                                else {}
                            )
                        ),
                    },
                )
            actions.append(
                RobotAction(
                    action_space="dexteleop_policy_io_v2.2",
                    arm_pose_by_side={
                        "left": float_action[:7],
                        "right": float_action[7:ARM_ACTION_DIM],
                    },
                    end_effector_targets_by_side=targets,
                    frame_by_effector={
                        f"{side}_arm": self.layout.arm_reference_frame_by_side[side]
                        for side in SIDES
                    },
                    controlled_frame_by_effector={
                        f"{side}_arm": self.layout.arm_controlled_frame_by_side[side]
                        for side in SIDES
                    },
                    source=str(source),
                    metadata={},
                )
            )
        return actions


__all__ = [
    "V22DeploymentAdapter",
    "V22DeploymentError",
    "V22DeploymentObservation",
    "V22PolicyActionChunk",
    "coerce_v22_policy_action_chunk",
]
