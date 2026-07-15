"""Pure deploy-session contract resolution shared by real and sim launchers.

This module performs no device construction and no dispatch.  It merges the
model artifact/layout identity with runtime capabilities into one immutable
session description so launchers can fail before creating a real backend.
Legacy 26D/38D policies remain an explicit, unchanged contract generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from dataset_contracts.policy_artifact import (
    DEPLOY_SCHEMA_V2,
    POLICY_ARTIFACT_METADATA_KEY,
    PolicyArtifactContract,
    policy_artifact_from_metadata,
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

RequestedExecutionMode = Literal["auto", "legacy_v1", "v22_q6_bridge", "v22_native"]


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


def resolve_deploy_session_contract(
    *,
    active_sides: Sequence[str] = SIDES,
    model_metadata: Mapping[str, Any] | None = None,
    policy_artifact: PolicyArtifactContract | Mapping[str, Any] | None = None,
    policy_io_layout: PolicyIOLayout | None = None,
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
        },
    )


__all__ = [
    "DEPLOY_SCHEMA_V1",
    "DEPLOY_SCHEMA_V2",
    "POLICY_CONTRACT_LEGACY_V1",
    "POLICY_CONTRACT_MODES",
    "POLICY_CONTRACT_V22_NATIVE",
    "POLICY_CONTRACT_V22_Q6_BRIDGE",
    "ResolvedDeploySessionContract",
    "RuntimeEndEffectorCapability",
    "resolve_deploy_session_contract",
]
