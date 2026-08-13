"""Model-neutral policy adapter identity and execution configuration.

The deployment stack has four representations that must agree before a model
output can reach a target:

* the action codec/layout and its normalization contract;
* the returned chunk horizon and the executable prefix;
* absolute/relative pose semantics and the runtime frame contract; and
* the artifact identity that produced the output.

This module is intentionally only a typed configuration/identity boundary. It
does not call a model, own an action queue, perform IK, or dispatch a command.
The existing :class:`PolicyDeployRuntime` remains the implementation of the
shared action/pose representation boundary; numeric normalization is declared
and hash-checked here, while model-family adapters own any non-identity
client/server transform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.pose_frames import PoseFrameContract, PoseFrameContractError

from .chunk import ActionChunkExecutionPolicy
from .normalization import (
    NORMALIZATION_CONTRACT_HASH_DOMAIN_V1,
    NormalizationContract,
    NormalizationContractError,
    explicit_identity_normalization_contract,
    normalization_contract_hash,
)
from .schema import DEPLOY_ACTION_VECTOR_ACTION_SPACE, normalize_active_sides
from .session_contract import (
    POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
    POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
    POLICY_ABSOLUTE_FRAMES,
    POLICY_ACTION_POSE_MODES,
)


POLICY_ADAPTER_CONFIG_SCHEMA_V1 = "dexteleop_policy_adapter_config_v1"
POLICY_ADAPTER_CONFIG_HASH_DOMAIN_V1 = "dexteleop.policy_adapter_config.v1"
POLICY_ACTION_CODEC_HASH_DOMAIN_V1 = "dexteleop.policy_action_codec.v1"
# Keep the historical public constant name, but use the same domain as the
# typed contract so the advertised hash and the inner contract hash cannot
# diverge.
POLICY_NORMALIZATION_HASH_DOMAIN_V1 = NORMALIZATION_CONTRACT_HASH_DOMAIN_V1


def _mapping(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return {str(key): value[key] for key in value}


def _non_empty_identity(value: Any, *, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty identity.")
    return text


def _normalize_pose_contract(
    value: PoseFrameContract | Mapping[str, Any] | None,
) -> PoseFrameContract | None:
    if value is None:
        return None
    if isinstance(value, PoseFrameContract):
        return value
    if isinstance(value, Mapping):
        try:
            return PoseFrameContract.from_dict(value)
        except PoseFrameContractError as exc:
            raise ValueError(f"invalid runtime_pose_frame_contract: {exc}") from exc
    raise TypeError(
        "runtime_pose_frame_contract must be a PoseFrameContract, mapping, or None."
    )


@dataclass(frozen=True)
class PolicyAdapterConfig:
    """Immutable, model-neutral representation/execution contract.

    ``normalization_contract`` describes the numeric convention expected by
    the shared decoder (normally normalized pose/gripper values). Its hash is
    derived when a caller does not provide one. The four explicit identity
    fields are kept alongside the payload so a server artifact can bind the
    exact layout without making the generic runtime know a model family.
    """

    adapter_id: str
    active_sides: tuple[str, ...] = ("left", "right")
    model_observation_sides: tuple[str, ...] | None = None
    arms_enabled: bool = True
    hands_enabled: bool = True
    action_space: str = DEPLOY_ACTION_VECTOR_ACTION_SPACE
    action_dim: int = 26
    # Model/server output dimensions may differ from the shared runtime vector
    # (for example OpenPI v2.2 emits a 16D typed action that is bridged to the
    # runtime's 26D pose+q6 vector).
    model_action_dim: int | None = None
    model_layout_hash: str | None = None
    model_action_codec_hash: str | None = None
    action_codec_contract: Mapping[str, Any] = field(
        default_factory=lambda: {
            "schema": "dexteleop_policy_action_codec_v1",
            "layout": "bimanual_ee_pose_q6_v1",
        }
    )
    action_codec_hash: str | None = None
    # This is an explicit declaration, not a promise that the generic runtime
    # will guess/apply a transform.  Legacy payloads remain accepted as opaque
    # compatibility metadata and are never interpreted numerically.
    normalization_contract: Mapping[str, Any] = field(
        default_factory=explicit_identity_normalization_contract
    )
    normalization_hash: str | None = None
    action_pose_mode: str = "absolute"
    policy_absolute_frame: str = POLICY_ABSOLUTE_FRAME_RUNTIME_RAW
    runtime_pose_frame_contract: PoseFrameContract | Mapping[str, Any] | None = None
    artifact_hash: str | None = None
    prediction_horizon_steps: int = 1
    minimum_execution_horizon_steps: int = 1
    action_period_sec: float = 1.0 / 15.0
    action_chunk_execution_steps: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        adapter_id = _non_empty_identity(self.adapter_id, name="adapter_id")
        sides = normalize_active_sides(self.active_sides)
        if not sides:
            raise ValueError("active_sides must not be empty.")
        observation_sides = (
            sides
            if self.model_observation_sides is None
            else normalize_active_sides(self.model_observation_sides)
        )
        if not observation_sides:
            raise ValueError("model_observation_sides must not be empty.")
        if not set(sides).issubset(set(observation_sides)):
            raise ValueError(
                "model_observation_sides must cover every active output side."
            )
        action_space = _non_empty_identity(self.action_space, name="action_space")
        action_dim = int(self.action_dim)
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        model_action_dim = action_dim if self.model_action_dim is None else int(self.model_action_dim)
        if model_action_dim <= 0:
            raise ValueError("model_action_dim must be positive when supplied.")

        pose_mode = str(self.action_pose_mode).strip().lower()
        if pose_mode not in POLICY_ACTION_POSE_MODES:
            raise ValueError(
                "action_pose_mode must be one of "
                f"{POLICY_ACTION_POSE_MODES!r}."
            )
        absolute_frame = str(self.policy_absolute_frame).strip().lower()
        if absolute_frame not in POLICY_ABSOLUTE_FRAMES:
            raise ValueError(
                "policy_absolute_frame must be one of "
                f"{POLICY_ABSOLUTE_FRAMES!r}."
            )
        pose_contract = _normalize_pose_contract(self.runtime_pose_frame_contract)
        if absolute_frame == POLICY_ABSOLUTE_FRAME_CANONICAL_DESK and pose_contract is None:
            raise ValueError(
                "canonical_desk policy actions require runtime_pose_frame_contract."
            )

        action_codec = _mapping(self.action_codec_contract, name="action_codec_contract")
        normalization = (
            explicit_identity_normalization_contract()
            if self.normalization_contract is None
            else _mapping(self.normalization_contract, name="normalization_contract")
        )
        if not action_codec:
            raise ValueError("action_codec_contract must not be empty.")
        if not normalization:
            raise ValueError("normalization_contract must not be empty.")
        try:
            explicit_normalization = NormalizationContract.from_optional_mapping(
                normalization,
                allow_legacy=True,
            )
        except NormalizationContractError as exc:
            raise ValueError(str(exc)) from exc
        if explicit_normalization is not None:
            # Store the canonical form so field order/omitted identity values
            # cannot produce two semantic contracts for one policy.
            normalization = explicit_normalization.to_dict()
        action_codec_hash = self.action_codec_hash
        if action_codec_hash is None:
            action_codec_hash = canonical_content_hash(
                {
                    "action_space": action_space,
                    "action_dim": action_dim,
                    "contract": action_codec,
                },
                domain=POLICY_ACTION_CODEC_HASH_DOMAIN_V1,
            )
        normalization_hash = self.normalization_hash
        computed_normalization_hash = canonical_content_hash(
            normalization,
            domain=POLICY_NORMALIZATION_HASH_DOMAIN_V1,
        )
        if normalization_hash is None:
            normalization_hash = computed_normalization_hash
        elif explicit_normalization is not None and str(normalization_hash) != computed_normalization_hash:
            raise ValueError(
                "normalization_hash does not match normalization_contract contents."
            )
        for name, value in (
            ("action_codec_hash", action_codec_hash),
            ("normalization_hash", normalization_hash),
        ):
            _non_empty_identity(value, name=name)
        artifact_hash = None if self.artifact_hash is None else _non_empty_identity(
            self.artifact_hash,
            name="artifact_hash",
        )
        model_layout_hash = (
            None
            if self.model_layout_hash is None
            else _non_empty_identity(self.model_layout_hash, name="model_layout_hash")
        )
        model_action_codec_hash = (
            None
            if self.model_action_codec_hash is None
            else _non_empty_identity(
                self.model_action_codec_hash,
                name="model_action_codec_hash",
            )
        )

        prediction_horizon = int(self.prediction_horizon_steps)
        minimum_horizon = int(self.minimum_execution_horizon_steps)
        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon_steps must be positive.")
        if minimum_horizon <= 0:
            raise ValueError("minimum_execution_horizon_steps must be positive.")
        if minimum_horizon > prediction_horizon:
            raise ValueError(
                "minimum_execution_horizon_steps cannot exceed prediction_horizon_steps."
            )
        period = float(self.action_period_sec)
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("action_period_sec must be finite and positive.")
        execution_policy = ActionChunkExecutionPolicy(self.action_chunk_execution_steps)
        if (
            execution_policy.execution_steps is not None
            and execution_policy.execution_steps > prediction_horizon
        ):
            raise ValueError(
                "action_chunk_execution_steps cannot exceed prediction_horizon_steps."
            )

        object.__setattr__(self, "adapter_id", adapter_id)
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(self, "model_observation_sides", observation_sides)
        object.__setattr__(self, "arms_enabled", bool(self.arms_enabled))
        object.__setattr__(self, "hands_enabled", bool(self.hands_enabled))
        object.__setattr__(self, "action_space", action_space)
        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "model_action_dim", model_action_dim)
        object.__setattr__(self, "model_layout_hash", model_layout_hash)
        object.__setattr__(self, "model_action_codec_hash", model_action_codec_hash)
        object.__setattr__(self, "action_codec_contract", action_codec)
        object.__setattr__(self, "action_codec_hash", str(action_codec_hash))
        object.__setattr__(self, "normalization_contract", normalization)
        object.__setattr__(self, "normalization_hash", str(normalization_hash))
        object.__setattr__(self, "action_pose_mode", pose_mode)
        object.__setattr__(self, "policy_absolute_frame", absolute_frame)
        object.__setattr__(self, "runtime_pose_frame_contract", pose_contract)
        object.__setattr__(self, "artifact_hash", artifact_hash)
        object.__setattr__(self, "prediction_horizon_steps", prediction_horizon)
        object.__setattr__(self, "minimum_execution_horizon_steps", minimum_horizon)
        object.__setattr__(self, "action_period_sec", period)
        object.__setattr__(
            self,
            "action_chunk_execution_steps",
            execution_policy.execution_steps,
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, name="metadata"))

    @property
    def execution_policy(self) -> ActionChunkExecutionPolicy:
        return ActionChunkExecutionPolicy(self.action_chunk_execution_steps)

    @property
    def frame_contract_hash(self) -> str | None:
        contract = self.runtime_pose_frame_contract
        return None if contract is None else contract.contract_hash

    @property
    def explicit_normalization_contract(self) -> NormalizationContract | None:
        """Return the typed contract, or ``None`` for opaque legacy metadata."""

        return NormalizationContract.from_optional_mapping(
            self.normalization_contract,
            allow_legacy=True,
        )

    @property
    def normalization_contract_hash(self) -> str:
        """Return the inner contract hash when explicit, otherwise an opaque hash."""

        return normalization_contract_hash(self.normalization_contract)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POLICY_ADAPTER_CONFIG_SCHEMA_V1,
            "adapter_id": self.adapter_id,
            "active_sides": list(self.active_sides),
            "model_observation_sides": list(self.model_observation_sides),
            "arms_enabled": self.arms_enabled,
            "hands_enabled": self.hands_enabled,
            "action_space": self.action_space,
            "action_dim": self.action_dim,
            "model_action_dim": self.model_action_dim,
            "model_layout_hash": self.model_layout_hash,
            "model_action_codec_hash": self.model_action_codec_hash,
            "action_codec_contract": dict(self.action_codec_contract),
            "action_codec_hash": self.action_codec_hash,
            "normalization_contract": dict(self.normalization_contract),
            "normalization_hash": self.normalization_hash,
            "action_pose_mode": self.action_pose_mode,
            "policy_absolute_frame": self.policy_absolute_frame,
            "runtime_pose_frame_contract": (
                None
                if self.runtime_pose_frame_contract is None
                else self.runtime_pose_frame_contract.to_dict()
            ),
            "frame_contract_hash": self.frame_contract_hash,
            "artifact_hash": self.artifact_hash,
            "prediction_horizon_steps": self.prediction_horizon_steps,
            "minimum_execution_horizon_steps": self.minimum_execution_horizon_steps,
            "action_period_sec": self.action_period_sec,
            "action_chunk_execution_steps": self.action_chunk_execution_steps,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyAdapterConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("PolicyAdapterConfig payload must be a mapping.")
        schema = str(payload.get("schema", POLICY_ADAPTER_CONFIG_SCHEMA_V1))
        if schema != POLICY_ADAPTER_CONFIG_SCHEMA_V1:
            raise ValueError(f"Unsupported policy adapter config schema {schema!r}.")
        contract = payload.get("runtime_pose_frame_contract")
        return cls(
            adapter_id=str(payload["adapter_id"]),
            active_sides=tuple(payload.get("active_sides", ("left", "right"))),
            model_observation_sides=(
                None
                if payload.get("model_observation_sides") is None
                else tuple(payload["model_observation_sides"])
            ),
            arms_enabled=bool(payload.get("arms_enabled", True)),
            hands_enabled=bool(payload.get("hands_enabled", True)),
            action_space=str(payload.get("action_space", DEPLOY_ACTION_VECTOR_ACTION_SPACE)),
            action_dim=int(payload.get("action_dim", 26)),
            model_action_dim=(
                None
                if payload.get("model_action_dim") is None
                else int(payload["model_action_dim"])
            ),
            model_layout_hash=payload.get("model_layout_hash"),
            model_action_codec_hash=payload.get("model_action_codec_hash"),
            action_codec_contract=_mapping(
                payload.get("action_codec_contract"),
                name="action_codec_contract",
            ),
            action_codec_hash=payload.get("action_codec_hash"),
            normalization_contract=_mapping(
                payload.get("normalization_contract"),
                name="normalization_contract",
            ),
            normalization_hash=payload.get("normalization_hash"),
            action_pose_mode=str(payload.get("action_pose_mode", "absolute")),
            policy_absolute_frame=str(
                payload.get("policy_absolute_frame", POLICY_ABSOLUTE_FRAME_RUNTIME_RAW)
            ),
            runtime_pose_frame_contract=contract,
            artifact_hash=payload.get("artifact_hash"),
            prediction_horizon_steps=int(payload.get("prediction_horizon_steps", 1)),
            minimum_execution_horizon_steps=int(
                payload.get("minimum_execution_horizon_steps", 1)
            ),
            action_period_sec=float(payload.get("action_period_sec", 1.0 / 15.0)),
            action_chunk_execution_steps=payload.get("action_chunk_execution_steps"),
            metadata=_mapping(payload.get("metadata"), name="metadata"),
        )

    @property
    def semantic_content_hash(self) -> str:
        """Hash representation semantics, excluding non-semantic metadata."""

        payload = self.to_dict()
        payload.pop("metadata", None)
        return canonical_content_hash(payload, domain=POLICY_ADAPTER_CONFIG_HASH_DOMAIN_V1)

    def to_runtime_config(self):
        """Build the shared decoder config while preserving normalization identity."""

        # Imported lazily so adapter_config remains a leaf next to runtime.py.
        from .runtime import PolicyDeployRuntimeConfig

        return PolicyDeployRuntimeConfig(
            active_sides=self.active_sides,
            model_observation_sides=self.model_observation_sides,
            arms_enabled=self.arms_enabled,
            hands_enabled=self.hands_enabled,
            policy_action_pose_mode=self.action_pose_mode,
            policy_absolute_frame=self.policy_absolute_frame,
            runtime_pose_frame_contract=self.runtime_pose_frame_contract,
            normalization_contract=self.normalization_contract,
            normalization_hash=self.normalization_hash,
            action_keyframe_period_sec=self.action_period_sec,
            action_maximum_hold_sec=self.action_period_sec,
            action_chunk_execution_steps=self.action_chunk_execution_steps,
        )


__all__ = [
    "POLICY_ADAPTER_CONFIG_HASH_DOMAIN_V1",
    "POLICY_ADAPTER_CONFIG_SCHEMA_V1",
    "POLICY_ACTION_CODEC_HASH_DOMAIN_V1",
    "POLICY_NORMALIZATION_HASH_DOMAIN_V1",
    "PolicyAdapterConfig",
]
