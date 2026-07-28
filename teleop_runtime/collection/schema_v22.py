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
    DatasetLayoutError,
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


LAST_ACTION_INITIALIZATION_METADATA_KEY = "last_action_initialization"
STRICT_LAST_ACTION_INITIALIZATION_METADATA = {
    "schema": "dexteleop_last_action_initialization_v1",
    "scope": "episode_and_continuity_segment_start",
    "reset_sentinel": "legacy_observation_last_action_all_non_finite",
    "steady_state": "action[t-1]",
    "arm_at_reset": "observation.arm.pose[t]",
    "end_effector_at_reset": "action.end_effector.target[t]",
    "partial_non_finite_policy": "reject",
    "future_arm_action_used_at_reset": False,
}


class StrictV22SampleRejection(ValueError):
    """A sample-level v2.2 contract rejection, not a writer failure."""

    def __init__(
        self,
        message: str,
        *,
        side: str,
        reason: str,
        age_sec: float | None = None,
    ) -> None:
        super().__init__(str(message))
        self.side = str(side)
        self.reason = str(reason)
        self.age_sec = None if age_sec is None else float(age_sec)


def validate_strict_v22_measured_state(
    hand_states: Mapping[str, Mapping[str, Any]],
    *,
    layout: PolicyIOLayout,
    sample_timestamp: float,
    future_tolerance_sec: float = 0.05,
) -> None:
    """Validate measured end-effector feedback before writer admission.

    This is intentionally the same check used by the strict row adapter.  The
    real-teleop parent can reject a stale snapshot before it enters a process
    worker, while direct writer callers retain the same fail-closed contract.
    """

    sample_time = float(sample_timestamp)
    future_tolerance = max(0.0, float(future_tolerance_sec))
    for side in SIDES:
        channel = layout.channel("state", side)
        if channel.value_type == "none":
            continue
        raw_state = hand_states.get(side)
        state = raw_state if isinstance(raw_state, Mapping) else {}
        end_effector = layout.end_effector(side)
        state_source_field = end_effector.state_source_field
        raw = state.get(state_source_field)
        if raw is None:
            raise StrictV22SampleRejection(
                f"v2.2 {side} end-effector state is missing required measured field "
                f"{state_source_field!r}.",
                side=side,
                reason="missing_measured_field",
            )
        if channel.value_type == "continuous":
            selected = _select_numeric_channel(channel, raw)
            if selected is None:
                raise StrictV22SampleRejection(
                    f"v2.2 {side} end-effector measured field "
                    f"{state_source_field!r} is invalid.",
                    side=side,
                    reason="invalid_measured_field",
                )
            if not np.all(np.isfinite(selected)):
                raise StrictV22SampleRejection(
                    f"v2.2 {side} end-effector measured field "
                    f"{state_source_field!r} is non-finite.",
                    side=side,
                    reason="nonfinite_measured_field",
                )
        elif channel.value_type == "discrete":
            try:
                selected_discrete = _select_discrete_channel(
                    channel,
                    raw=raw,
                    metadata=state,
                )
                if selected_discrete is None:
                    raise ValueError("no discrete state value")
                for token, value in zip(channel.tokens, selected_discrete):
                    token.resolve(value)
            except (DatasetLayoutError, TypeError, ValueError) as exc:
                raise StrictV22SampleRejection(
                    f"v2.2 {side} end-effector measured field "
                    f"{state_source_field!r} is invalid: {exc}",
                    side=side,
                    reason="invalid_measured_field",
                ) from exc
        state_metadata = end_effector.metadata
        timestamp_field = str(
            state_metadata.get("collection_state_timestamp_field", "joint_timestamp")
        )
        timestamp = state.get(timestamp_field)
        try:
            timestamp_value = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise StrictV22SampleRejection(
                f"v2.2 {side} end-effector state is missing timestamp "
                f"{timestamp_field!r}.",
                side=side,
                reason="missing_measured_timestamp",
            ) from exc
        max_age_sec = float(state_metadata.get("collection_state_max_age_sec", 0.5))
        age_sec = sample_time - timestamp_value
        if (
            not np.isfinite(timestamp_value)
            or age_sec < -future_tolerance
            or age_sec > max_age_sec
        ):
            raise StrictV22SampleRejection(
                f"v2.2 {side} end-effector measured state is stale "
                f"(age_sec={age_sec:.3f}, max_age_sec={max_age_sec:.3f}).",
                side=side,
                reason="stale_measured_state",
                age_sec=age_sec,
            )


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


def _last_action_is_temporal_reset_sentinel(sample: CollectorSample) -> bool:
    """Return whether the legacy assembler explicitly reset temporal history.

    ``SampleAssembler`` represents an episode/continuity-boundary reset with an
    entirely non-finite legacy last-action vector.  A partially invalid vector
    is data corruption and must not receive the causal initialization below.
    """

    last_action = _legacy_action(sample, last=True)
    return bool(
        last_action is not None
        and last_action.size > 0
        and not np.any(np.isfinite(last_action))
    )


def _pack_with_context(label: str, packer: Any, **kwargs: Any):
    try:
        return packer(**kwargs)
    except DatasetLayoutError as exc:
        raise DatasetLayoutError(f"{label}: {exc}") from exc


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
    if role == "state" and strict:
        validate_strict_v22_measured_state(
            sample.hand_states,
            layout=layout,
            sample_timestamp=float(sample.timestamp),
        )
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


def extract_collector_policy_io_component_values(
    sample: CollectorSample,
    *,
    layout: PolicyIOLayout,
    strict: bool = False,
) -> dict[str, np.ndarray]:
    """Extract typed current action/state components from one source sample.

    This is the shared source-production boundary for row and native shadow
    encoders.  It does not create a dataset row and therefore does not assign
    compatibility field names or own persistence.
    """

    action_poses = _arm_poses_from_action(sample)
    state_poses = _arm_poses_from_state(sample)
    state_qpos = _arm_qpos_from_state(sample)
    action_eef = _eef_values_from_sample(
        sample,
        layout,
        role="action",
        strict=bool(strict),
    )
    state_eef = _eef_values_from_sample(
        sample,
        layout,
        role="state",
        strict=bool(strict),
    )
    values: dict[str, np.ndarray] = {}
    for side in SIDES:
        if side in action_poses:
            values[f"action.pose.{side}"] = action_poses[side]
        if side in state_poses:
            values[f"state.pose.{side}"] = state_poses[side]
        if side in state_qpos:
            values[f"state.qpos.{side}"] = state_qpos[side]
        for role, source in (("action", action_eef), ("state", state_eef)):
            if side not in source:
                continue
            channel = layout.channel(role, side)
            dtype = np.int64 if channel.value_type == "discrete" else np.float32
            values[f"{role}.end_effector.{side}"] = np.asarray(
                source[side], dtype=dtype
            )
    return values


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
    action = _pack_with_context(
        "action",
        layout.pack_action,
        arm_pose_by_side=_arm_poses_from_action(sample),
        end_effector_by_side=_eef_values_from_sample(
            sample,
            layout,
            role="action",
            strict=bool(strict),
        ),
        strict=bool(strict),
    )
    state = _pack_with_context(
        "observation.state",
        layout.pack_observation_state,
        arm_pose_by_side=_arm_poses_from_state(sample),
        end_effector_by_side=_eef_values_from_sample(
            sample,
            layout,
            role="state",
            strict=bool(strict),
        ),
        strict=bool(strict),
    )
    temporal_reset = bool(strict) and _last_action_is_temporal_reset_sentinel(sample)
    last_action_arm_poses = (
        _arm_poses_from_state(sample)
        if temporal_reset
        else _arm_poses_from_action(sample, last_action=True)
    )
    last_action_end_effectors = _eef_values_from_sample(
        sample,
        layout,
        role="action",
        # At an explicit temporal reset there is no previous command.  Seed
        # only the end-effector part from the final command accepted at T0;
        # the arm part above comes from measured state, never action[t].
        last_action=not temporal_reset,
        strict=bool(strict),
    )
    last_action = _pack_with_context(
        "observation.last_action",
        layout.pack_action,
        arm_pose_by_side=last_action_arm_poses,
        end_effector_by_side=last_action_end_effectors,
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


__all__ = [
    "StrictV22SampleRejection",
    "collector_sample_to_v22_row",
    "extract_collector_policy_io_component_values",
    "validate_strict_v22_measured_state",
]
