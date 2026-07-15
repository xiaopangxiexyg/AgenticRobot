"""Opt-in DexTeleop v2.2 rows on top of LeRobot-v2.1-style storage.

The legacy collector continues to produce its stable 26D/38D working vectors.
This adapter derives the v2.2 typed policy-I/O columns from that raw capture
record, so enabling a v2.2 writer does not alter the live teleoperation command
path.  A future native v2.2 assembler can populate the same canonical columns
without changing reader, training, replay, or writer contracts.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from dataset_contracts.v22 import (
    ARM_ACTION_DIM,
    DEXTELEOP_DATASET_SCHEMA_V22,
    EndEffectorChannelSpec,
    PolicyIOLayout,
    SIDES,
)
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_HAND_SLICES,
    ACTION_DIM,
    HAND_DOF,
    OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
    STATE_HAND_SLICES,
)

from .sample_assembler import CollectorSample
from .schema_v21 import collector_sample_to_row


def _vector_or_none(value: Any, *, length: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if length is not None and vector.shape != (int(length),):
        return None
    return vector.astype(np.float32, copy=True)


def _legacy_action(sample: CollectorSample, *, last: bool = False) -> np.ndarray | None:
    raw = sample.observation_last_action if last else sample.action
    return _vector_or_none(raw, length=ACTION_DIM)


def _legacy_state(sample: CollectorSample) -> np.ndarray | None:
    return _vector_or_none(sample.observation_state, length=OBSERVATION_STATE_DIM)


def _select_numeric_channel(
    channel: EndEffectorChannelSpec,
    raw: Any,
) -> np.ndarray | None:
    vector = _vector_or_none(raw)
    if vector is None:
        return None
    if channel.source_indices is not None:
        indices = np.asarray(channel.source_indices, dtype=int)
        if indices.size != channel.dim or np.any(indices >= vector.size):
            return None
        return vector[indices].astype(np.float32, copy=True)
    if vector.shape == (channel.dim,):
        return vector
    return None


def _lookup_metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata[key]
    current: Any = metadata
    for part in str(key).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _select_discrete_channel(
    channel: EndEffectorChannelSpec,
    *,
    raw: Any,
    metadata: Mapping[str, Any],
) -> list[Any] | None:
    if channel.source_indices is not None:
        selected = _select_numeric_channel(channel, raw)
        if selected is None:
            return None
        return [int(value) for value in selected]
    values: list[Any] = []
    for token in channel.tokens:
        if token.source_metadata_key is None:
            return None
        value = _lookup_metadata_value(metadata, token.source_metadata_key)
        if value is None:
            return None
        values.append(value)
    return values


def _command_metadata(sample: CollectorSample, side: str) -> Mapping[str, Any]:
    root = sample.metadata.get("command_metadata_by_effector")
    if not isinstance(root, Mapping):
        return {}
    payload = root.get(f"{side}_hand")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _eef_values_from_sample(
    sample: CollectorSample,
    layout: PolicyIOLayout,
    *,
    role: str,
    last_action: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    action = _legacy_action(sample, last=last_action)
    for side in SIDES:
        channel = layout.channel("action" if role == "action" else "state", side)
        if channel.value_type == "none":
            continue
        if role == "action":
            raw = None if action is None else action[ACTION_HAND_SLICES[side]]
            metadata = {} if last_action else _command_metadata(sample, side)
        else:
            state = sample.hand_states.get(side) or {}
            end_effector = layout.end_effector(side)
            state_source_field = end_effector.state_source_field
            raw = state.get(state_source_field)
            if strict:
                if raw is None:
                    raise ValueError(
                        f"v2.2 {side} end-effector state is missing required measured field "
                        f"{state_source_field!r}."
                    )
                state_metadata = end_effector.metadata
                timestamp_field = str(
                    state_metadata.get("collection_state_timestamp_field", "joint_timestamp")
                )
                timestamp = state.get(timestamp_field)
                try:
                    timestamp_value = float(timestamp)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"v2.2 {side} end-effector state is missing timestamp "
                        f"{timestamp_field!r}."
                    ) from exc
                max_age_sec = float(state_metadata.get("collection_state_max_age_sec", 0.5))
                age_sec = float(sample.timestamp) - timestamp_value
                if not np.isfinite(timestamp_value) or age_sec < -0.05 or age_sec > max_age_sec:
                    raise ValueError(
                        f"v2.2 {side} end-effector measured state is stale "
                        f"(age_sec={age_sec:.3f}, max_age_sec={max_age_sec:.3f})."
                    )
            if raw is None:
                legacy_state = _legacy_state(sample)
                # v2.1 state always contains six raw hand slots.  It is only a
                # fallback for the v2.2 adapter, never a declaration of v2.2
                # model-state semantics.
                raw = None if legacy_state is None else legacy_state[STATE_HAND_SLICES[side]]
            metadata = state if isinstance(state, Mapping) else {}
        if channel.value_type == "continuous":
            selected = _select_numeric_channel(channel, raw)
        else:
            selected = (
                None
                if last_action and role == "action" and channel.source_indices is None
                else _select_discrete_channel(channel, raw=raw, metadata=metadata)
            )
        if selected is not None:
            values[side] = selected
    return values


def _require_finite_model_vector(value: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"v2.2 strict collection requires finite {name} values.")


def _require_valid_discrete_tokens(value: np.ndarray, *, name: str) -> None:
    if np.any(value < 0):
        raise ValueError(f"v2.2 strict collection requires valid {name} tokens.")


def _arm_poses_from_action(sample: CollectorSample, *, last_action: bool = False) -> dict[str, np.ndarray]:
    action = _legacy_action(sample, last=last_action)
    if action is None:
        return {}
    return {
        side: action[ACTION_ARM_POSE_SLICES[side]].astype(np.float32, copy=True)
        for side in SIDES
        if np.all(np.isfinite(action[ACTION_ARM_POSE_SLICES[side]]))
    }


def _arm_poses_from_state(sample: CollectorSample) -> dict[str, np.ndarray]:
    state = _legacy_state(sample)
    if state is None:
        return {}
    return {
        side: state[STATE_ARM_POSE_SLICES[side]].astype(np.float32, copy=True)
        for side in SIDES
        if np.all(np.isfinite(state[STATE_ARM_POSE_SLICES[side]]))
    }


def _arm_qpos_from_state(sample: CollectorSample) -> dict[str, np.ndarray]:
    state = _legacy_state(sample)
    if state is None:
        return {}
    return {
        side: state[STATE_ARM_QPOS_SLICES[side]].astype(np.float32, copy=True)
        for side in SIDES
        if np.all(np.isfinite(state[STATE_ARM_QPOS_SLICES[side]]))
    }


def _write_end_effector_component_columns(
    row: dict[str, Any],
    layout: PolicyIOLayout,
    *,
    role: str,
    continuous: np.ndarray,
    discrete: np.ndarray,
) -> None:
    layout_role = "action" if role == "action" else "state"
    prefix = "action" if role == "action" else "observation"
    suffix = "target" if role == "action" else "state"
    for component in layout.components(layout_role, "continuous"):
        if component.name.endswith(".arm.pose"):
            continue
        assert component.side is not None
        row[f"{prefix}.end_effector.{component.side}.{suffix}"] = (
            continuous[component.offset : component.stop].astype(np.float32, copy=False).tolist()
        )
    for component in layout.components(layout_role, "discrete"):
        assert component.side is not None
        row[f"{prefix}.end_effector.{component.side}.{suffix}_token"] = (
            discrete[component.offset : component.stop].astype(np.int64, copy=False).tolist()
        )


def collector_sample_to_v22_row(
    sample: CollectorSample,
    *,
    layout: PolicyIOLayout,
    camera_video_paths: Mapping[str, str] | None = None,
    tactile_storage_refs: Mapping[str, Mapping[str, Any]] | None = None,
    inline_tactile: bool = True,
    strict: bool = False,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Serialize one legacy working sample into canonical v2.2 policy-I/O rows.

    Existing v2.1 audit, media, and replay columns remain in the row.  The
    canonical ``action`` and ``observation.state`` fields are replaced by the
    configuration-derived v2.2 values, while arm qpos is written only to its
    dedicated audit field.
    """

    row, camera_expectations = collector_sample_to_row(
        sample,
        camera_video_paths=camera_video_paths,
        tactile_storage_refs=tactile_storage_refs,
        inline_tactile=inline_tactile,
    )
    action = layout.pack_action(
        arm_pose_by_side=_arm_poses_from_action(sample),
        end_effector_by_side=_eef_values_from_sample(
            sample,
            layout,
            role="action",
            strict=bool(strict),
        ),
        strict=bool(strict),
    )
    state = layout.pack_observation_state(
        arm_pose_by_side=_arm_poses_from_state(sample),
        end_effector_by_side=_eef_values_from_sample(
            sample,
            layout,
            role="state",
            strict=bool(strict),
        ),
        strict=bool(strict),
    )
    last_action = layout.pack_action(
        arm_pose_by_side=_arm_poses_from_action(sample, last_action=True),
        end_effector_by_side=_eef_values_from_sample(
            sample,
            layout,
            role="action",
            last_action=True,
            strict=bool(strict),
        ),
        strict=bool(strict),
    )
    arm_qpos = layout.pack_arm_qpos_audit(_arm_qpos_from_state(sample))
    if strict:
        if str(sample.arm_state_source) != "causal_measured_hardware":
            raise ValueError(
                "v2.2 strict collection requires causal measured arm feedback; "
                f"got arm_state_source={sample.arm_state_source!r}."
            )
        _require_finite_model_vector(action.continuous, name="action")
        _require_finite_model_vector(state.continuous, name="observation.state")
        _require_finite_model_vector(last_action.continuous, name="observation.last_action")
        _require_finite_model_vector(arm_qpos, name="observation.arm.qpos audit")
        _require_valid_discrete_tokens(action.discrete, name="action.discrete")
        _require_valid_discrete_tokens(
            state.discrete,
            name="observation.state.discrete",
        )
        _require_valid_discrete_tokens(
            last_action.discrete,
            name="observation.last_action.discrete",
        )

    row["dexteleop.dataset_schema"] = DEXTELEOP_DATASET_SCHEMA_V22
    row["policy_io_layout_hash"] = layout.layout_hash
    row["action"] = action.continuous.tolist()
    row["action.arm.pose"] = action.continuous[:ARM_ACTION_DIM].tolist()
    row["observation.state"] = state.continuous.tolist()
    row["observation.arm.pose"] = state.continuous[:ARM_ACTION_DIM].tolist()
    row["observation.arm.qpos"] = arm_qpos.tolist()
    row["observation.arm.qpos_valid"] = np.isfinite(arm_qpos).tolist()
    row["observation.last_action"] = last_action.continuous.tolist()
    if layout.action_discrete_dim:
        row["action.discrete"] = action.discrete.astype(np.int64, copy=False).tolist()
        row["observation.last_action.discrete"] = last_action.discrete.astype(
            np.int64, copy=False
        ).tolist()
    if layout.state_discrete_dim:
        row["observation.state.discrete"] = state.discrete.astype(np.int64, copy=False).tolist()
    _write_end_effector_component_columns(
        row,
        layout,
        role="action",
        continuous=action.continuous,
        discrete=action.discrete,
    )
    _write_end_effector_component_columns(
        row,
        layout,
        role="state",
        continuous=state.continuous,
        discrete=state.discrete,
    )
    return row, camera_expectations


__all__ = ["collector_sample_to_v22_row"]
