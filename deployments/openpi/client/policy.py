from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from deployments.client import (
    DEXTELEOP_DEPLOY_SCHEMA_VERSION,
    DEXTELEOP_DEPLOY_SCHEMA_V2,
    DeployPolicyContract,
    RemoteModelPolicy,
    RemotePolicyConfig,
    validate_server_metadata,
)
from dataset_contracts.v22 import PolicyIOLayout
from dataset_contracts.policy_artifact import PolicyArtifactContract, policy_artifact_from_metadata
from robot_motion.deploy.v22 import V22DeploymentAdapter

from .observation_adapter import OpenPIObservationAdapterConfig, deployment_observation_to_openpi
from .transport import OpenPIWebsocketTransport


OPENPI_LOCAL_PROTOCOL_VERSION = "openpi.local_policy.websocket.v1"


@dataclass(frozen=True)
class OpenPIClientPolicyConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    prompt: str | None = None
    expected_model_id: str | None = None
    expected_checkpoint_step: int | str | None = None
    deploy_schema_version: str | None = DEXTELEOP_DEPLOY_SCHEMA_VERSION
    active_sides: tuple[str, ...] = ("left", "right")
    require_active_sides_match: bool = True
    required_camera_slots: tuple[str, ...] | None = None
    required_image_keys: tuple[str, ...] | None = None
    action_key: str = "actions"
    action_dim: int = 26
    action_discrete_key: str | None = None
    action_discrete_dim: int = 0
    policy_io_layout_hash: str | None = None
    min_action_horizon: int = 1
    hand_layout: str | None = "q6_slot0_gripper"
    image_key_map: Mapping[str, str] | None = None
    include_debug_metadata: bool = True
    retry_interval_s: float = 2.0
    policy_io_layout: PolicyIOLayout | Mapping[str, Any] | None = None
    policy_artifact: PolicyArtifactContract | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", int(self.port))
        artifact = self.policy_artifact
        if isinstance(artifact, Mapping):
            artifact = policy_artifact_from_metadata(artifact)
        elif artifact is not None and not isinstance(artifact, PolicyArtifactContract):
            raise TypeError("policy_artifact must be a PolicyArtifactContract, mapping, or None.")
        layout = self.policy_io_layout
        if isinstance(layout, Mapping):
            layout = PolicyIOLayout.from_dict(layout)
        elif layout is not None and not isinstance(layout, PolicyIOLayout):
            raise TypeError("policy_io_layout must be a PolicyIOLayout, mapping, or None.")
        if artifact is not None:
            if layout is not None and layout.layout_hash != artifact.layout_hash:
                raise ValueError("policy_io_layout disagrees with policy_artifact layout.")
            if self.expected_model_id not in (None, "", artifact.model_id):
                raise ValueError("expected_model_id disagrees with policy_artifact model_id.")
            layout = artifact.policy_io_layout
        if layout is None:
            action_dim = int(self.action_dim)
            action_discrete_dim = int(self.action_discrete_dim)
            layout_hash = None if self.policy_io_layout_hash is None else str(self.policy_io_layout_hash)
            action_discrete_key = self.action_discrete_key
            hand_layout = self.hand_layout
        else:
            requested_action_dim = int(self.action_dim)
            if requested_action_dim not in (26, layout.action_dim):
                raise ValueError(
                    "action_dim disagrees with policy_io_layout: "
                    f"expected {layout.action_dim}, got {requested_action_dim}."
                )
            requested_token_dim = int(self.action_discrete_dim)
            if requested_token_dim not in (0, layout.action_discrete_dim):
                raise ValueError(
                    "action_discrete_dim disagrees with policy_io_layout: "
                    f"expected {layout.action_discrete_dim}, got {requested_token_dim}."
                )
            if self.policy_io_layout_hash not in (None, "", layout.layout_hash):
                raise ValueError("policy_io_layout_hash disagrees with policy_io_layout contents.")
            action_dim = layout.action_dim
            action_discrete_dim = layout.action_discrete_dim
            layout_hash = layout.layout_hash
            action_discrete_key = (
                self.action_discrete_key
                if self.action_discrete_key is not None
                else (f"{self.action_key}_discrete" if action_discrete_dim else None)
            )
            # The v2.2 layout, rather than a legacy q6 label, describes the model I/O.
            hand_layout = None
        if action_discrete_dim < 0:
            raise ValueError("action_discrete_dim must be non-negative.")
        camera_input = None if artifact is None else artifact.camera_input
        if camera_input is not None:
            artifact_slots = camera_input.server_camera_slots
            artifact_keys = camera_input.source_keys
            artifact_map = camera_input.image_key_map
            if self.required_camera_slots is not None and tuple(
                str(slot) for slot in self.required_camera_slots
            ) != artifact_slots:
                raise ValueError(
                    "required_camera_slots disagrees with policy_artifact camera_input."
                )
            if self.required_image_keys is not None and tuple(
                str(key) for key in self.required_image_keys
            ) != artifact_keys:
                raise ValueError(
                    "required_image_keys disagrees with policy_artifact camera_input."
                )
            if self.image_key_map is not None and {
                str(key): str(value) for key, value in self.image_key_map.items()
            } != artifact_map:
                raise ValueError(
                    "image_key_map disagrees with policy_artifact camera_input."
                )
            required_camera_slots = artifact_slots
            required_image_keys = artifact_keys if camera_input.required else ()
            image_key_map = artifact_map
        else:
            required_camera_slots = (
                ("base_0_rgb",)
                if self.required_camera_slots is None
                else tuple(str(slot) for slot in self.required_camera_slots)
            )
            required_image_keys = (
                None
                if self.required_image_keys is None
                else tuple(str(key) for key in self.required_image_keys)
            )
            image_key_map = (
                None
                if self.image_key_map is None
                else {str(key): str(value) for key, value in self.image_key_map.items()}
            )
        object.__setattr__(self, "policy_io_layout", layout)
        object.__setattr__(self, "policy_artifact", artifact)
        if artifact is not None:
            object.__setattr__(self, "deploy_schema_version", artifact.deploy_schema_version)
            object.__setattr__(self, "expected_model_id", artifact.model_id)
            object.__setattr__(self, "expected_checkpoint_step", artifact.checkpoint_step)
            object.__setattr__(self, "action_key", artifact.action_key)
            object.__setattr__(self, "action_discrete_key", artifact.action_discrete_key)
        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "action_discrete_dim", action_discrete_dim)
        object.__setattr__(self, "action_discrete_key", action_discrete_key)
        object.__setattr__(self, "policy_io_layout_hash", layout_hash)
        object.__setattr__(self, "hand_layout", hand_layout)
        object.__setattr__(self, "min_action_horizon", int(self.min_action_horizon))
        object.__setattr__(self, "active_sides", tuple(str(side) for side in self.active_sides))
        object.__setattr__(self, "required_camera_slots", required_camera_slots)
        object.__setattr__(self, "required_image_keys", required_image_keys)
        object.__setattr__(self, "image_key_map", image_key_map)
        object.__setattr__(self, "retry_interval_s", float(self.retry_interval_s))


class OpenPIClientPolicy:
    """OpenPI-specific policy adapter returning unified action chunks."""

    def __init__(
        self,
        config: OpenPIClientPolicyConfig | None = None,
        *,
        remote_policy: RemoteModelPolicy | None = None,
    ) -> None:
        self.config = OpenPIClientPolicyConfig() if config is None else config
        strict_v22 = (
            self.config.policy_io_layout is not None
            and self.config.deploy_schema_version == DEXTELEOP_DEPLOY_SCHEMA_V2
        )
        artifact = self.config.policy_artifact
        contract = DeployPolicyContract(
            protocol_version=OPENPI_LOCAL_PROTOCOL_VERSION,
            deploy_schema_version=self.config.deploy_schema_version,
            model_id=self.config.expected_model_id,
            checkpoint_step=self.config.expected_checkpoint_step,
            state_key=(artifact.state_key if artifact is not None else "state") if strict_v22 else None,
            state_dim=(
                artifact.state_input_dim
                if artifact is not None
                else self.config.policy_io_layout.state_dim
            )
            if strict_v22
            else None,
            state_input_mode=(artifact.state_input_mode if artifact is not None else None)
            if strict_v22
            else None,
            state_discrete_key=(
                artifact.state_discrete_key if artifact is not None else "state_discrete"
            ) if strict_v22 else None,
            state_discrete_dim=(
                self.config.policy_io_layout.state_discrete_dim if strict_v22 else None
            ),
            action_key=self.config.action_key,
            action_dim=self.config.action_dim,
            action_discrete_dim=(
                int(self.config.action_discrete_dim)
                if int(self.config.action_discrete_dim) > 0
                else None
            ),
            policy_io_layout_hash=self.config.policy_io_layout_hash,
            min_action_horizon=self.config.min_action_horizon,
            active_sides=self.config.active_sides,
            require_active_sides_match=self.config.require_active_sides_match,
            required_camera_slots=self.config.required_camera_slots,
            hand_layout=self.config.hand_layout,
        )
        if remote_policy is None:
            observation_config = OpenPIObservationAdapterConfig(
                prompt=self.config.prompt,
                image_key_map=self.config.image_key_map,
                required_image_keys=self.config.required_image_keys,
                include_debug_metadata=self.config.include_debug_metadata,
                state_key=(artifact.state_key if artifact is not None else "state"),
                state_discrete_key=(
                    artifact.state_discrete_key if artifact is not None else "state_discrete"
                ),
            )
            transport = OpenPIWebsocketTransport(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key,
                retry_interval_s=self.config.retry_interval_s,
            )
            remote_policy = RemoteModelPolicy(
                transport=transport,
                observation_adapter=lambda observation: deployment_observation_to_openpi(
                    observation,
                    observation_config,
                ),
                config=RemotePolicyConfig(
                    action_key=self.config.action_key,
                    action_dim=self.config.action_dim,
                    action_discrete_key=self.config.action_discrete_key,
                    action_discrete_dim=self.config.action_discrete_dim,
                    contract=contract,
                    source="openpi_remote_policy",
                ),
            )
        else:
            validate_server_metadata(remote_policy.server_metadata, contract)
        self.remote_policy = remote_policy
        self.v22_adapter = (
            None
            if self.config.policy_io_layout is None
            else V22DeploymentAdapter(
                self.config.policy_io_layout,
                state_input_mode=(
                    artifact.state_input_mode if artifact is not None else "raw_state_v1"
                ),
            )
        )
        self.name = _policy_name(self.remote_policy.server_metadata)
        self.metadata = dict(getattr(self.remote_policy, "metadata", {}) or {})
        if self.v22_adapter is not None:
            self.metadata["policy_io_layout_hash"] = self.v22_adapter.layout.layout_hash
            self.metadata["policy_io_layout"] = self.v22_adapter.layout.to_dict()

    def reset(self) -> None:
        self.remote_policy.reset()

    def predict(self, observation: Any):
        if self.v22_adapter is None:
            return self.remote_policy.predict(observation)
        typed_observation = self.v22_adapter.observation_from_deployment(observation)
        output = self.remote_policy.predict(typed_observation)
        return self.v22_adapter.to_legacy_normalized_actions(output, source="openpi_v22_policy")


def create_policy(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str | None = None,
    prompt: str | None = None,
    expected_model_id: str | None = None,
    expected_checkpoint_step: int | str | None = None,
    deploy_schema_version: str | None = DEXTELEOP_DEPLOY_SCHEMA_VERSION,
    active_sides: tuple[str, ...] | list[str] = ("left", "right"),
    require_active_sides_match: bool = True,
    required_camera_slots: tuple[str, ...] | list[str] | None = None,
    required_image_keys: tuple[str, ...] | list[str] | None = None,
    action_key: str = "actions",
    action_dim: int = 26,
    action_discrete_key: str | None = None,
    action_discrete_dim: int = 0,
    policy_io_layout_hash: str | None = None,
    min_action_horizon: int = 1,
    hand_layout: str | None = "q6_slot0_gripper",
    image_key_map: Mapping[str, str] | None = None,
    include_debug_metadata: bool = True,
    retry_interval_s: float = 2.0,
    policy_io_layout: PolicyIOLayout | Mapping[str, Any] | None = None,
    policy_artifact: PolicyArtifactContract | Mapping[str, Any] | None = None,
) -> OpenPIClientPolicy:
    return OpenPIClientPolicy(
        OpenPIClientPolicyConfig(
            host=host,
            port=int(port),
            api_key=api_key,
            prompt=prompt,
            expected_model_id=expected_model_id,
            expected_checkpoint_step=expected_checkpoint_step,
            deploy_schema_version=deploy_schema_version,
            active_sides=tuple(str(side) for side in active_sides),
            require_active_sides_match=bool(require_active_sides_match),
            required_camera_slots=(
                None
                if required_camera_slots is None
                else tuple(str(slot) for slot in required_camera_slots)
            ),
            required_image_keys=(
                None
                if required_image_keys is None
                else tuple(str(key) for key in required_image_keys)
            ),
            action_key=str(action_key),
            action_dim=int(action_dim),
            action_discrete_key=action_discrete_key,
            action_discrete_dim=int(action_discrete_dim),
            policy_io_layout_hash=policy_io_layout_hash,
            min_action_horizon=int(min_action_horizon),
            hand_layout=hand_layout,
            image_key_map=image_key_map,
            include_debug_metadata=bool(include_debug_metadata),
            retry_interval_s=float(retry_interval_s),
            policy_io_layout=policy_io_layout,
            policy_artifact=policy_artifact,
        )
    )


def _policy_name(metadata: dict[str, Any]) -> str:
    deployment = metadata.get("deployment") if isinstance(metadata.get("deployment"), dict) else {}
    for key in ("model_id", "config_name"):
        value = metadata.get(key) or deployment.get(key)
        if value:
            return f"openpi:{value}"
    return "openpi_remote"
