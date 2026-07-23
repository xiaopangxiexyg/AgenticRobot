"""OpenPI model adapter for a simulation deployment observation.

Only the model-family boundary lives here: this class binds an artifact to the
OpenPI websocket client and hands the result to the shared deploy decoder. It
does not know how MuJoCo steps, owns no target authority, and never dispatches
an action. A simulator composition layer can consume the returned immutable
normalized chunk in exactly the same way as a real frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.policy_artifact import (
    PolicyArtifactContract,
    policy_artifact_from_metadata,
)
from robot_motion.deploy import (
    NormalizedAction,
    NormalizationContract,
    NormalizationContractError,
    PolicyAdapterConfig,
    PolicyDeployRuntime,
    normalization_contract_hash,
    resolve_policy_action_pose_mode_from_artifact,
)
from robot_motion.deploy.observation import DeploymentObservation

from .policy import OpenPIClientPolicy, OpenPIClientPolicyConfig


OPENPI_SIM_ADAPTER_SCHEMA_V1 = "openpi_sim_policy_adapter_v1"
# Reuse the existing artifact identity domain so a loaded checkpoint can bind
# raw-RTC and sim shadow paths without producing two hashes for one sidecar.
OPENPI_SIM_ARTIFACT_HASH_DOMAIN_V1 = "dexteleop.openpi_rtc_artifact.v1"
OPENPI_SIM_CODEC_HASH_DOMAIN_V1 = "dexteleop.openpi_rtc_v22_codec.v1"


class OpenPIModelPolicy(Protocol):
    """Minimal policy surface accepted by the simulation adapter."""

    metadata: Mapping[str, Any]

    def predict(self, observation: Any) -> Any:
        ...

    def reset(self) -> None:
        ...


def _artifact_hash(artifact: PolicyArtifactContract) -> str:
    return canonical_content_hash(
        artifact.to_dict(),
        domain=OPENPI_SIM_ARTIFACT_HASH_DOMAIN_V1,
    )


def _model_action_codec_hash(
    artifact: PolicyArtifactContract,
    *,
    action_pose_mode: str,
) -> str:
    return canonical_content_hash(
        {
            "adapter": "V22DeploymentAdapter.to_legacy_normalized_actions",
            "adapter_version": "v1",
            "layout_hash": artifact.layout_hash,
            "action_pose_mode": str(action_pose_mode),
            "output_type": "NormalizedAction",
        },
        domain=OPENPI_SIM_CODEC_HASH_DOMAIN_V1,
    )


@dataclass(frozen=True)
class OpenPISimAdapterConfig:
    """OpenPI-only server binding plus the model-neutral adapter contract."""

    adapter: PolicyAdapterConfig
    policy_artifact: PolicyArtifactContract | Mapping[str, Any]
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    prompt: str | None = None
    include_debug_metadata: bool = True
    retry_interval_s: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, PolicyAdapterConfig):
            raise TypeError("adapter must be a PolicyAdapterConfig.")
        artifact = self.policy_artifact
        if isinstance(artifact, Mapping):
            artifact = policy_artifact_from_metadata(artifact)
        if not isinstance(artifact, PolicyArtifactContract):
            raise TypeError(
                "policy_artifact must be a PolicyArtifactContract or mapping."
            )
        artifact_normalization = None
        if isinstance(artifact.metadata, Mapping):
            artifact_normalization = artifact.metadata.get("normalization_contract")
        if artifact_normalization is not None:
            try:
                parsed_normalization = NormalizationContract.from_optional_mapping(
                    artifact_normalization,
                    allow_legacy=True,
                )
            except NormalizationContractError as exc:
                raise ValueError(
                    "policy artifact normalization_contract is invalid: " + str(exc)
                ) from exc
            artifact_normalization_hash = normalization_contract_hash(artifact_normalization)
            if self.adapter.normalization_contract_hash != artifact_normalization_hash:
                raise ValueError(
                    "adapter normalization contract disagrees with policy artifact "
                    "normalization_contract."
                )
            if parsed_normalization is not None and parsed_normalization.has_client_owned_transform:
                raise ValueError(
                    "OpenPI sim adapter cannot execute client-owned normalization before "
                    "the model-specific adapter is bound."
                )
        if int(self.adapter.action_dim) != 26:
            raise ValueError(
                "OpenPI legacy normalized bridge currently targets the shared 26D "
                "runtime action vector; use a native adapter for another target layout."
            )
        if int(self.adapter.model_action_dim or 0) != int(artifact.policy_io_layout.action_dim):
            raise ValueError(
                "adapter.model_action_dim disagrees with policy artifact action_dim: "
                f"{self.adapter.model_action_dim} != {artifact.policy_io_layout.action_dim}."
            )
        if (
            self.adapter.model_layout_hash is not None
            and self.adapter.model_layout_hash != artifact.layout_hash
        ):
            raise ValueError(
                "adapter.model_layout_hash disagrees with policy artifact layout_hash."
            )
        resolved_pose_mode = resolve_policy_action_pose_mode_from_artifact(
            artifact,
            configured_mode=self.adapter.action_pose_mode,
            explicit=True,
        )
        if resolved_pose_mode != self.adapter.action_pose_mode:
            raise ValueError(
                "adapter action_pose_mode disagrees with OpenPI policy artifact."
            )
        expected_model_codec_hash = _model_action_codec_hash(
            artifact,
            action_pose_mode=resolved_pose_mode,
        )
        if (
            self.adapter.model_action_codec_hash is not None
            and self.adapter.model_action_codec_hash != expected_model_codec_hash
        ):
            raise ValueError(
                "adapter.model_action_codec_hash disagrees with the OpenPI v2.2 bridge."
            )
        if self.adapter.artifact_hash is not None and self.adapter.artifact_hash != _artifact_hash(artifact):
            raise ValueError("adapter.artifact_hash disagrees with policy artifact contents.")
        port = int(self.port)
        if port <= 0 or port > 65535:
            raise ValueError("OpenPI port must be in [1, 65535].")
        host = str(self.host).strip()
        if not host:
            raise ValueError("OpenPI host must be non-empty.")
        retry_interval = float(self.retry_interval_s)
        if retry_interval <= 0:
            raise ValueError("retry_interval_s must be positive.")
        object.__setattr__(self, "policy_artifact", artifact)
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "include_debug_metadata", bool(self.include_debug_metadata))
        object.__setattr__(self, "retry_interval_s", retry_interval)
        if self.prompt is not None:
            object.__setattr__(self, "prompt", str(self.prompt))

    @property
    def artifact_hash(self) -> str:
        return _artifact_hash(self.policy_artifact)


@dataclass(frozen=True)
class OpenPISimPrediction:
    """Decoded model output; no execution or target authority is included."""

    actions: tuple[NormalizedAction, ...]
    source_chunk_size: int
    executed_chunk_size: int
    adapter_config_hash: str
    artifact_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        source_size = int(self.source_chunk_size)
        executed_size = int(self.executed_chunk_size)
        if source_size < 0 or executed_size < 0 or executed_size > source_size:
            raise ValueError("OpenPI prediction chunk sizes are inconsistent.")
        if executed_size != len(actions):
            raise ValueError("executed_chunk_size must equal len(actions).")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "source_chunk_size", source_size)
        object.__setattr__(self, "executed_chunk_size", executed_size)
        object.__setattr__(self, "adapter_config_hash", str(self.adapter_config_hash))
        object.__setattr__(self, "artifact_hash", str(self.artifact_hash))
        object.__setattr__(self, "metadata", dict(self.metadata))


class OpenPISimAdapter:
    """Bind OpenPI inference to the generic representation boundary."""

    def __init__(
        self,
        config: OpenPISimAdapterConfig,
        *,
        policy: OpenPIModelPolicy | None = None,
    ) -> None:
        if not isinstance(config, OpenPISimAdapterConfig):
            raise TypeError("config must be an OpenPISimAdapterConfig.")
        self.config = config
        self.runtime = PolicyDeployRuntime(config.adapter.to_runtime_config())
        if policy is None:
            camera_input = config.policy_artifact.camera_input
            policy = OpenPIClientPolicy(
                OpenPIClientPolicyConfig(
                    host=config.host,
                    port=config.port,
                    api_key=config.api_key,
                    prompt=config.prompt,
                    active_sides=config.adapter.active_sides,
                    include_debug_metadata=config.include_debug_metadata,
                    retry_interval_s=config.retry_interval_s,
                    required_camera_slots=(
                        ()
                        if camera_input is None
                        else camera_input.server_camera_slots
                    ),
                    required_image_keys=(
                        ()
                        if camera_input is None or not camera_input.required
                        else camera_input.source_keys
                    ),
                    image_key_map=(
                        None if camera_input is None else camera_input.image_key_map
                    ),
                    policy_artifact=config.policy_artifact,
                )
            )
        if not hasattr(policy, "predict") or not callable(policy.predict):
            raise TypeError("policy must expose a callable predict(observation) method.")
        self.policy = policy

    @property
    def metadata(self) -> dict[str, Any]:
        policy_metadata = getattr(self.policy, "metadata", {})
        return {
            "adapter_schema": OPENPI_SIM_ADAPTER_SCHEMA_V1,
            "adapter_config_hash": self.config.adapter.semantic_content_hash,
            "artifact_hash": self.config.artifact_hash,
            "model_id": self.config.policy_artifact.model_id,
            "checkpoint_step": self.config.policy_artifact.checkpoint_step,
            "policy": dict(policy_metadata) if isinstance(policy_metadata, Mapping) else {},
        }

    def predict(self, observation: DeploymentObservation) -> OpenPISimPrediction:
        """Infer once and return a decoded executable prefix for a simulator."""

        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation must be a DeploymentObservation.")
        model_output = self.policy.predict(observation)
        decoded = self.runtime.normalize_and_decode(model_output, observation)
        source_size = len(decoded)
        if source_size < self.config.adapter.minimum_execution_horizon_steps:
            raise ValueError(
                "OpenPI action horizon is shorter than the configured minimum: "
                f"{source_size} < {self.config.adapter.minimum_execution_horizon_steps}."
            )
        selected = self.config.adapter.execution_policy.select(decoded)
        metadata = self.metadata
        metadata.update(
            {
                "source_chunk_size": source_size,
                "executed_chunk_size": len(selected),
                "action_pose_mode": self.config.adapter.action_pose_mode,
                "policy_absolute_frame": self.config.adapter.policy_absolute_frame,
                "normalization_hash": self.config.adapter.normalization_hash,
                "action_codec_hash": self.config.adapter.action_codec_hash,
                "model_action_codec_hash": _model_action_codec_hash(
                    self.config.policy_artifact,
                    action_pose_mode=self.config.adapter.action_pose_mode,
                ),
                "model_layout_hash": self.config.policy_artifact.layout_hash,
            }
        )
        return OpenPISimPrediction(
            actions=tuple(selected),
            source_chunk_size=source_size,
            executed_chunk_size=len(selected),
            adapter_config_hash=self.config.adapter.semantic_content_hash,
            artifact_hash=self.config.artifact_hash,
            metadata=metadata,
        )

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()


__all__ = [
    "OPENPI_SIM_ADAPTER_SCHEMA_V1",
    "OPENPI_SIM_CODEC_HASH_DOMAIN_V1",
    "OpenPIModelPolicy",
    "OpenPISimAdapter",
    "OpenPISimAdapterConfig",
    "OpenPISimPrediction",
]
