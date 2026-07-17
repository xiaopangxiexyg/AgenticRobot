"""Pure deploy-session contract resolution shared by real and sim launchers.

This module performs no device construction and no dispatch.  It merges the
model artifact/layout identity with runtime capabilities into one immutable
session description so launchers can fail before creating a real backend.
Legacy 26D/38D policies remain an explicit, unchanged contract generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from dataset_contracts.policy_artifact import (
    DEPLOY_SCHEMA_V2,
    POLICY_ARTIFACT_METADATA_KEY,
    PolicyArtifactContract,
    policy_artifact_from_metadata,
)
from dataset_contracts.pose_frames import (
    POSE_STORAGE_FRAME_RAW,
    PoseFrameContract,
)
from dataset_contracts.pose_training import (
    POSE_TRAINING_ORIGIN_NATIVE_V23,
    POSE_TRAINING_ORIGIN_V22_RAW_IDENTITY,
)
from dataset_contracts.state_input import STATE_INPUT_MODE_RAW, normalize_state_input_mode
from dataset_contracts.v22 import (
    DEXTELEOP_DATASET_SCHEMA_V22,
    POLICY_IO_LAYOUT_HASH_METADATA_KEY,
    POLICY_IO_LAYOUT_METADATA_KEY,
    DatasetLayoutBindingError,
    PolicyIOLayout,
    SIDES,
)
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23

from .schema import ACTION_DIM, OBSERVATION_STATE_DIM


DEPLOY_SCHEMA_V1 = "dexteleop_deploy_v1"
POLICY_CONTRACT_LEGACY_V1 = "legacy_v1"
POLICY_CONTRACT_V22_Q6_BRIDGE = "v22_q6_bridge"
POLICY_CONTRACT_V22_NATIVE = "v22_native"
POLICY_CONTRACT_MODES = (
    POLICY_CONTRACT_LEGACY_V1,
    POLICY_CONTRACT_V22_Q6_BRIDGE,
    POLICY_CONTRACT_V22_NATIVE,
)
POLICY_ABSOLUTE_FRAME_RUNTIME_RAW = "runtime_raw"
POLICY_ABSOLUTE_FRAME_CANONICAL_DESK = "canonical_desk"
POLICY_ABSOLUTE_FRAMES = (
    POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
    POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
)
POLICY_ACTION_POSE_MODES = ("absolute", "relative")

RequestedExecutionMode = Literal["auto", "legacy_v1", "v22_q6_bridge", "v22_native"]


def resolve_policy_action_pose_mode_from_artifact(
    policy_artifact: PolicyArtifactContract | Mapping[str, Any] | Any | None,
    *,
    configured_mode: str,
    explicit: bool = False,
) -> str:
    """Resolve the runtime action pose mode from a typed policy artifact.

    The helper intentionally accepts only contract/config values.  CLI token
    parsing remains in the frontend, while the agreement checks are shared by
    every deployment frontend.
    """

    configured = str(configured_mode).strip().lower()
    if configured not in POLICY_ACTION_POSE_MODES:
        raise ValueError(
            "configured_mode must be one of "
            f"{POLICY_ACTION_POSE_MODES!r}, got {configured_mode!r}."
        )
    if policy_artifact is None:
        return configured

    artifact: Any = policy_artifact
    if isinstance(policy_artifact, Mapping):
        artifact = policy_artifact_from_metadata(policy_artifact)

    structured = getattr(artifact, "pose_action_chunk", None)

    def _structured_value(name: str) -> Any:
        if isinstance(structured, Mapping):
            return structured.get(name)
        return getattr(structured, name, None) if structured is not None else None

    structured_mode_value = _structured_value("action_pose_mode")
    structured_last_mode_value = _structured_value("last_action_pose_mode")
    structured_mode = (
        None
        if structured_mode_value is None
        else str(structured_mode_value).strip().lower()
    )
    structured_last_mode = (
        None
        if structured_last_mode_value is None
        else str(structured_last_mode_value).strip().lower()
    )
    if (
        structured_mode is not None
        and structured_last_mode is not None
        and structured_mode != structured_last_mode
    ):
        raise ValueError(
            "Current deploy runtime requires pose_action_chunk_contract action_pose_mode "
            "and last_action_pose_mode to match."
        )

    metadata = getattr(artifact, "metadata", {})
    representation = (
        metadata.get("pose_representation")
        if isinstance(metadata, Mapping)
        else None
    )
    legacy_declared = (
        representation.get("action", representation.get("action_pose_mode"))
        if isinstance(representation, Mapping)
        else None
    )
    legacy_mode = (
        None if legacy_declared is None else str(legacy_declared).strip().lower()
    )
    if (
        structured_mode is not None
        and legacy_mode is not None
        and structured_mode != legacy_mode
    ):
        raise ValueError(
            "policy artifact pose_action_chunk_contract disagrees with legacy "
            f"pose_representation.action: {structured_mode!r} != {legacy_mode!r}."
        )

    declared_mode = structured_mode or legacy_mode
    if declared_mode is None:
        return configured
    if declared_mode not in POLICY_ACTION_POSE_MODES:
        raise ValueError(
            "policy artifact pose_representation.action must be 'absolute' or 'relative', "
            f"got {declared_mode!r}."
        )
    if not explicit:
        return declared_mode
    if configured != declared_mode:
        raise ValueError(
            "CLI policy action pose mode disagrees with policy artifact: "
            f"{configured!r} != {declared_mode!r}"
        )
    return configured


def _normalize_sides(values: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(value).strip().lower() for value in values)
    invalid = sorted(set(requested) - set(SIDES))
    if invalid:
        raise DatasetLayoutBindingError(f"Unsupported deploy side(s): {invalid}.")
    if len(set(requested)) != len(requested):
        raise DatasetLayoutBindingError("Deploy active_sides must not contain duplicates.")
    return tuple(side for side in SIDES if side in set(requested))


def _nested_get(mapping: Mapping[str, Any], key: str) -> Any:
    current: Any = mapping
    for part in str(key).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = _nested_get(mapping, key)
        if value is not None:
            return value
    return None


def _model_layout_hash(metadata: Mapping[str, Any]) -> str | None:
    value = _first_present(
        metadata,
        (
            POLICY_IO_LAYOUT_HASH_METADATA_KEY,
            "dataset_adapter.policy_io_layout_hash",
            "model.policy_io_layout_hash",
            "deployment.policy_io_layout_hash",
            "policy_artifact.policy_io_layout_hash",
        ),
    )
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _metadata_v22_signal(metadata: Mapping[str, Any]) -> bool:
    return bool(
        metadata.get(POLICY_IO_LAYOUT_METADATA_KEY) is not None
        or metadata.get(POLICY_ARTIFACT_METADATA_KEY) is not None
        or _model_layout_hash(metadata) is not None
        or str(metadata.get("dexteleop_dataset_schema") or "")
        == DEXTELEOP_DATASET_SCHEMA_V22
        or str(
            _first_present(
                metadata,
                ("deploy_schema_version", "schema_version", "deployment.deploy_schema_version"),
            )
            or ""
        )
        == DEPLOY_SCHEMA_V2
    )


def _metadata_v23_pose_signal(metadata: Mapping[str, Any]) -> bool:
    return any(
        str(value or "") == DEXTELEOP_DATASET_SCHEMA_V23
        for value in (
            metadata.get("dexteleop_dataset_schema"),
            _nested_get(metadata, "dataset_info.dexteleop_dataset_schema"),
        )
    )


@dataclass(frozen=True)
class RuntimeEndEffectorCapability:
    """Resolved real/sim runtime implementation for one logical side."""

    side: str
    adapter: str
    backend: str
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        side = str(self.side).strip().lower()
        if side not in SIDES:
            raise DatasetLayoutBindingError(f"Unsupported end-effector side {self.side!r}.")
        adapter = str(self.adapter).strip()
        backend = str(self.backend).strip()
        if bool(self.enabled) and (not adapter or not backend or backend == "none"):
            raise DatasetLayoutBindingError(
                f"Enabled {side} end effector requires non-empty adapter and backend."
            )
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "adapter", adapter or "none_v1")
        object.__setattr__(self, "backend", backend or "none")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeEndEffectorCapability":
        return cls(
            side=str(payload["side"]),
            adapter=str(payload.get("adapter", "none_v1")),
            backend=str(payload.get("backend", "none")),
            enabled=bool(payload.get("enabled", True)),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "adapter": self.adapter,
            "backend": self.backend,
            "enabled": bool(self.enabled),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResolvedDeploySessionContract:
    """Immutable result of model/layout/runtime compatibility resolution."""

    deploy_schema_version: str
    policy_contract_mode: str
    active_sides: tuple[str, ...]
    action_dim: int
    action_discrete_dim: int
    state_dim: int
    state_discrete_dim: int
    state_input_mode: str = STATE_INPUT_MODE_RAW
    policy_absolute_frame: str = POLICY_ABSOLUTE_FRAME_RUNTIME_RAW
    runtime_pose_frame_contract: PoseFrameContract | None = None
    policy_io_layout: PolicyIOLayout | None = None
    runtime_end_effectors: Mapping[str, RuntimeEndEffectorCapability] = field(
        default_factory=dict
    )
    q6_bridge_compatible: bool = False
    hardware_output: bool = False
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.policy_contract_mode not in POLICY_CONTRACT_MODES:
            raise DatasetLayoutBindingError(
                f"Unsupported policy contract mode {self.policy_contract_mode!r}."
            )
        object.__setattr__(self, "active_sides", _normalize_sides(self.active_sides))
        policy_absolute_frame = str(self.policy_absolute_frame).strip().lower()
        if policy_absolute_frame not in POLICY_ABSOLUTE_FRAMES:
            raise DatasetLayoutBindingError(
                f"Unsupported policy_absolute_frame {self.policy_absolute_frame!r}."
            )
        runtime_pose_frames = self.runtime_pose_frame_contract
        if runtime_pose_frames is not None and not isinstance(
            runtime_pose_frames, PoseFrameContract
        ):
            raise TypeError("runtime_pose_frame_contract must be a PoseFrameContract or None.")
        if (
            policy_absolute_frame == POLICY_ABSOLUTE_FRAME_CANONICAL_DESK
            and runtime_pose_frames is None
        ):
            raise DatasetLayoutBindingError(
                "canonical_desk policy view requires a runtime_pose_frame_contract."
            )
        object.__setattr__(self, "policy_absolute_frame", policy_absolute_frame)
        object.__setattr__(self, "state_input_mode", normalize_state_input_mode(self.state_input_mode))
        object.__setattr__(
            self,
            "runtime_end_effectors",
            {str(side): value for side, value in dict(self.runtime_end_effectors).items()},
        )
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def layout_hash(self) -> str | None:
        return None if self.policy_io_layout is None else self.policy_io_layout.layout_hash

    @property
    def end_effector_sides(self) -> tuple[str, ...]:
        if self.policy_io_layout is None:
            return tuple(
                side
                for side in SIDES
                if side in self.runtime_end_effectors
                and self.runtime_end_effectors[side].enabled
            )
        return tuple(
            side for side in SIDES if self.policy_io_layout.end_effector(side).present
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "deploy_schema_version": self.deploy_schema_version,
            "policy_contract_mode": self.policy_contract_mode,
            "active_sides": list(self.active_sides),
            "action_dim": int(self.action_dim),
            "action_discrete_dim": int(self.action_discrete_dim),
            "state_dim": int(self.state_dim),
            "state_discrete_dim": int(self.state_discrete_dim),
            "state_input_mode": self.state_input_mode,
            "policy_absolute_frame": self.policy_absolute_frame,
            "runtime_pose_frame_contract": (
                None
                if self.runtime_pose_frame_contract is None
                else self.runtime_pose_frame_contract.to_dict()
            ),
            "runtime_pose_frame_contract_hash": (
                None
                if self.runtime_pose_frame_contract is None
                else self.runtime_pose_frame_contract.contract_hash
            ),
            "policy_io_layout_hash": self.layout_hash,
            "end_effector_sides": list(self.end_effector_sides),
            "runtime_end_effectors": {
                side: capability.to_dict()
                for side, capability in self.runtime_end_effectors.items()
            },
            "q6_bridge_compatible": bool(self.q6_bridge_compatible),
            "hardware_output": bool(self.hardware_output),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def _coerce_runtime_capabilities(
    values: Mapping[str, RuntimeEndEffectorCapability | Mapping[str, Any]] | None,
) -> dict[str, RuntimeEndEffectorCapability]:
    if values is None:
        return {}
    output: dict[str, RuntimeEndEffectorCapability] = {}
    for raw_side, raw_value in values.items():
        side = str(raw_side).strip().lower()
        if isinstance(raw_value, RuntimeEndEffectorCapability):
            value = raw_value
        elif isinstance(raw_value, Mapping):
            payload = dict(raw_value)
            payload.setdefault("side", side)
            value = RuntimeEndEffectorCapability.from_dict(payload)
        else:
            raise TypeError(
                "runtime_end_effectors values must be RuntimeEndEffectorCapability or mappings."
            )
        if value.side != side:
            raise DatasetLayoutBindingError(
                f"Runtime end-effector key {side!r} disagrees with capability side {value.side!r}."
            )
        output[side] = value
    return output


def _layout_q6_bridge_compatible(layout: PolicyIOLayout) -> bool:
    for side in SIDES:
        channel = layout.channel("action", side)
        if channel.value_type == "none":
            continue
        if channel.legacy_q6_slot_indices is None:
            return False
    return True


def _validate_v22_identity_runtime_pose_binding(
    *,
    runtime_pose_frames: PoseFrameContract,
    active_sides: Sequence[str],
    expected_runtime_source_id: str,
) -> None:
    """Fail before hardware construction when legacy raw state would be remapped."""

    actual_source_id = str(runtime_pose_frames.source_id)
    if actual_source_id != str(expected_runtime_source_id):
        raise DatasetLayoutBindingError(
            "v2.2 raw identity compatibility policy requires runtime pose source_id "
            f"{expected_runtime_source_id!r}, got {actual_source_id!r}. This checkpoint "
            "is not a native v2.3 canonical-desk policy."
        )
    identity = np.eye(4, dtype=float)
    nonidentity_sides = [
        str(side)
        for side in active_sides
        if not np.allclose(
            runtime_pose_frames.source.effective_raw_to_canonical_desk_transform(side),
            identity,
            atol=1e-9,
            rtol=0.0,
        )
    ]
    if nonidentity_sides:
        raise DatasetLayoutBindingError(
            "v2.2 raw identity compatibility policy cannot use a non-identity runtime "
            "raw->canonical-desk transform for side(s): "
            f"{nonidentity_sides}. Relative actions are world-transform invariant, but "
            "the model's absolute state distribution is not. Use the matching identity "
            "runtime profile or retrain with native v2.3 canonical-desk state."
        )


def resolve_deploy_session_contract(
    *,
    active_sides: Sequence[str] = SIDES,
    model_metadata: Mapping[str, Any] | None = None,
    policy_artifact: PolicyArtifactContract | Mapping[str, Any] | None = None,
    policy_io_layout: PolicyIOLayout | None = None,
    runtime_pose_frame_contract: PoseFrameContract | Mapping[str, Any] | None = None,
    runtime_end_effectors: Mapping[
        str, RuntimeEndEffectorCapability | Mapping[str, Any]
    ]
    | None = None,
    requested_mode: RequestedExecutionMode = "auto",
    hardware_output: bool = False,
    allow_missing_end_effectors: bool = False,
    require_model_layout_hash: bool = True,
    require_fixed_dual_arm_v22: bool = True,
) -> ResolvedDeploySessionContract:
    """Resolve a legacy or v2.2 deploy session without touching hardware."""

    active = _normalize_sides(tuple(active_sides))
    metadata = {} if model_metadata is None else dict(model_metadata)
    artifact: PolicyArtifactContract | None
    if policy_artifact is None:
        raw_artifact = metadata.get(POLICY_ARTIFACT_METADATA_KEY)
        artifact = (
            None
            if raw_artifact is None
            else policy_artifact_from_metadata({POLICY_ARTIFACT_METADATA_KEY: raw_artifact})
        )
    elif isinstance(policy_artifact, PolicyArtifactContract):
        artifact = policy_artifact
    elif isinstance(policy_artifact, Mapping):
        artifact = policy_artifact_from_metadata(policy_artifact)
    else:
        raise TypeError("policy_artifact must be a PolicyArtifactContract, mapping, or None.")

    runtime_pose_frames: PoseFrameContract | None
    if isinstance(runtime_pose_frame_contract, Mapping):
        runtime_pose_frames = PoseFrameContract.from_dict(runtime_pose_frame_contract)
    elif runtime_pose_frame_contract is None or isinstance(
        runtime_pose_frame_contract, PoseFrameContract
    ):
        runtime_pose_frames = runtime_pose_frame_contract
    else:
        raise TypeError(
            "runtime_pose_frame_contract must be a PoseFrameContract, mapping, or None."
        )
    v23_pose_view = bool(
        (artifact is not None and artifact.pose_action_chunk is not None)
        or _metadata_v23_pose_signal(metadata)
    )
    pose_training_data = None if artifact is None else artifact.pose_training_data
    pose_training_origin = (
        None if pose_training_data is None else pose_training_data.origin
    )
    runtime_pose_binding_was_explicit = runtime_pose_frames is not None
    if (
        pose_training_origin == POSE_TRAINING_ORIGIN_NATIVE_V23
        and runtime_pose_frames is None
    ):
        raise DatasetLayoutBindingError(
            "A native v2.3 pose policy requires an explicit "
            "runtime_pose_frame_contract. Identity fallback is reserved for the "
            "separately declared v2.2 raw identity compatibility path."
        )
    if runtime_pose_frames is not None:
        if runtime_pose_frames.pose_storage_frame != POSE_STORAGE_FRAME_RAW:
            raise DatasetLayoutBindingError(
                "runtime_pose_frame_contract must describe runtime raw_absolute poses."
            )
        missing_pose_sides = sorted(
            set(active) - set(runtime_pose_frames.source.active_sides)
        )
        if missing_pose_sides:
            raise DatasetLayoutBindingError(
                "runtime_pose_frame_contract is missing active side(s): "
                f"{missing_pose_sides}."
            )
    runtime_pose_identity_compatibility = bool(
        v23_pose_view and runtime_pose_frames is None
    )
    if v23_pose_view and runtime_pose_frames is None:
        runtime_pose_frames = PoseFrameContract.legacy_v22_raw(active_sides=active)
    if not v23_pose_view and runtime_pose_frames is not None:
        raise DatasetLayoutBindingError(
            "runtime_pose_frame_contract requires a structured v2.3 policy artifact "
            "or v2.3 debug replay dataset."
        )
    policy_absolute_frame = (
        POLICY_ABSOLUTE_FRAME_CANONICAL_DESK
        if v23_pose_view
        else POLICY_ABSOLUTE_FRAME_RUNTIME_RAW
    )
    if pose_training_origin == POSE_TRAINING_ORIGIN_V22_RAW_IDENTITY:
        assert runtime_pose_frames is not None
        assert pose_training_data is not None
        assert pose_training_data.expected_runtime_source_id is not None
        _validate_v22_identity_runtime_pose_binding(
            runtime_pose_frames=runtime_pose_frames,
            active_sides=active,
            expected_runtime_source_id=pose_training_data.expected_runtime_source_id,
        )

    if artifact is not None:
        if policy_io_layout is not None and artifact.layout_hash != policy_io_layout.layout_hash:
            raise DatasetLayoutBindingError(
                "Explicit policy_io_layout differs from the policy artifact layout."
            )
        policy_io_layout = artifact.policy_io_layout

    v22 = policy_io_layout is not None or artifact is not None or _metadata_v22_signal(metadata)
    mode = str(requested_mode).strip().lower()
    if mode not in {"auto", *POLICY_CONTRACT_MODES}:
        raise DatasetLayoutBindingError(f"Unsupported requested deploy mode {requested_mode!r}.")
    runtime = _coerce_runtime_capabilities(runtime_end_effectors)
    warnings: list[str] = []
    if runtime_pose_identity_compatibility:
        warnings.append("v23_runtime_pose_identity_compatibility")
    if pose_training_origin == POSE_TRAINING_ORIGIN_V22_RAW_IDENTITY:
        warnings.append("v22_raw_identity_compatibility_policy_state_binding")

    if not v22:
        if mode not in {"auto", POLICY_CONTRACT_LEGACY_V1}:
            raise DatasetLayoutBindingError(
                f"Deploy mode {mode!r} requires a v2.2 policy I/O layout."
            )
        warnings.append("legacy_v1_implicit")
        return ResolvedDeploySessionContract(
            deploy_schema_version=DEPLOY_SCHEMA_V1,
            policy_contract_mode=POLICY_CONTRACT_LEGACY_V1,
            active_sides=active,
            action_dim=ACTION_DIM,
            action_discrete_dim=0,
            state_dim=OBSERVATION_STATE_DIM,
            state_discrete_dim=0,
            state_input_mode=STATE_INPUT_MODE_RAW,
            policy_absolute_frame=policy_absolute_frame,
            runtime_pose_frame_contract=runtime_pose_frames,
            runtime_end_effectors=runtime,
            q6_bridge_compatible=True,
            hardware_output=bool(hardware_output),
            warnings=tuple(warnings),
        )

    if policy_io_layout is None:
        raise DatasetLayoutBindingError(
            "A v2.2 deploy signal was declared without an embedded or explicit policy_io_layout."
        )
    if require_fixed_dual_arm_v22 and active != SIDES:
        raise DatasetLayoutBindingError(
            "v2.2 fixed dual-arm pose core requires active_sides=('left', 'right')."
        )
    declared_hash = _model_layout_hash(metadata)
    if declared_hash is None and artifact is not None:
        declared_hash = artifact.layout_hash
    if require_model_layout_hash and declared_hash is None:
        raise DatasetLayoutBindingError(
            "v2.2 model metadata is missing policy_io_layout_hash."
        )
    if declared_hash is not None and declared_hash != policy_io_layout.layout_hash:
        raise DatasetLayoutBindingError(
            "v2.2 model/runtime layout mismatch: "
            f"model={declared_hash!r}, runtime={policy_io_layout.layout_hash!r}."
        )

    expected_sides = tuple(
        side for side in SIDES if policy_io_layout.end_effector(side).present
    )
    if runtime_end_effectors is not None:
        actual_sides = tuple(
            side for side in SIDES if side in runtime and runtime[side].enabled
        )
        extra = sorted(set(actual_sides) - set(expected_sides))
        missing = sorted(set(expected_sides) - set(actual_sides))
        if extra:
            raise DatasetLayoutBindingError(
                f"Runtime enables end effector(s) absent from layout: {extra}."
            )
        if missing and hardware_output and not bool(allow_missing_end_effectors):
            raise DatasetLayoutBindingError(
                f"Real v2.2 runtime is missing configured end effector(s): {missing}."
            )
        if missing:
            warning = (
                "hardware_end_effector_omitted_arm_only:"
                if hardware_output and bool(allow_missing_end_effectors)
                else "dry_run_end_effector_missing:"
            )
            warnings.append(warning + ",".join(missing))
        for side in set(expected_sides) & set(actual_sides):
            expected_adapter = policy_io_layout.end_effector(side).adapter
            if runtime[side].adapter != expected_adapter:
                raise DatasetLayoutBindingError(
                    f"{side} runtime adapter {runtime[side].adapter!r} differs from "
                    f"layout adapter {expected_adapter!r}."
                )

    bridge_compatible = _layout_q6_bridge_compatible(policy_io_layout)
    if mode == "auto":
        resolved_mode = (
            POLICY_CONTRACT_V22_Q6_BRIDGE
            if bridge_compatible
            else POLICY_CONTRACT_V22_NATIVE
        )
    else:
        resolved_mode = mode
    if resolved_mode == POLICY_CONTRACT_LEGACY_V1:
        raise DatasetLayoutBindingError(
            "A v2.2 policy layout cannot be executed as implicit legacy_v1."
        )
    if resolved_mode == POLICY_CONTRACT_V22_Q6_BRIDGE and not bridge_compatible:
        raise DatasetLayoutBindingError(
            "v22_q6_bridge requires legacy_q6_slot_indices for every configured action channel."
        )

    return ResolvedDeploySessionContract(
        deploy_schema_version=DEPLOY_SCHEMA_V2,
        policy_contract_mode=resolved_mode,
        active_sides=active,
        action_dim=policy_io_layout.action_dim,
        action_discrete_dim=policy_io_layout.action_discrete_dim,
        state_dim=(policy_io_layout.state_dim if artifact is None else artifact.state_input_dim),
        state_discrete_dim=policy_io_layout.state_discrete_dim,
        state_input_mode=(STATE_INPUT_MODE_RAW if artifact is None else artifact.state_input_mode),
        policy_absolute_frame=policy_absolute_frame,
        runtime_pose_frame_contract=runtime_pose_frames,
        policy_io_layout=policy_io_layout,
        runtime_end_effectors=runtime,
        q6_bridge_compatible=bridge_compatible,
        hardware_output=bool(hardware_output),
        warnings=tuple(warnings),
        metadata={
            "model_layout_hash_declared": declared_hash,
            "policy_artifact_schema": None if artifact is None else artifact.schema,
            "state_input_mode": STATE_INPUT_MODE_RAW if artifact is None else artifact.state_input_mode,
            "state_augmentation": {} if artifact is None else dict(artifact.state_augmentation),
            "runtime_pose_identity_compatibility": bool(
                runtime_pose_identity_compatibility
            ),
            "runtime_pose_binding_was_explicit": bool(
                runtime_pose_binding_was_explicit
            ),
            "pose_training_data_origin": pose_training_origin,
            "pose_training_runtime_binding": (
                None
                if pose_training_data is None
                else pose_training_data.runtime_pose_binding
            ),
            "pose_training_expected_runtime_source_id": (
                None
                if pose_training_data is None
                else pose_training_data.expected_runtime_source_id
            ),
        },
    )


__all__ = [
    "DEPLOY_SCHEMA_V1",
    "DEPLOY_SCHEMA_V2",
    "POLICY_CONTRACT_LEGACY_V1",
    "POLICY_CONTRACT_MODES",
    "POLICY_CONTRACT_V22_NATIVE",
    "POLICY_CONTRACT_V22_Q6_BRIDGE",
    "POLICY_ABSOLUTE_FRAME_CANONICAL_DESK",
    "POLICY_ABSOLUTE_FRAME_RUNTIME_RAW",
    "POLICY_ABSOLUTE_FRAMES",
    "ResolvedDeploySessionContract",
    "RuntimeEndEffectorCapability",
    "resolve_deploy_session_contract",
]
