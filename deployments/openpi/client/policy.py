from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deployments.client import (
    DEXTELEOP_DEPLOY_SCHEMA_VERSION,
    DeployPolicyContract,
    RemoteModelPolicy,
    RemotePolicyConfig,
)

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
    required_camera_slots: tuple[str, ...] = ("base_0_rgb",)
    required_image_keys: tuple[str, ...] = ("front",)
    action_key: str = "actions"
    action_dim: int = 26
    min_action_horizon: int = 1
    hand_layout: str | None = "q6_slot0_gripper"
    image_key_map: dict[str, str] | None = None
    include_debug_metadata: bool = True
    retry_interval_s: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", int(self.port))
        object.__setattr__(self, "action_dim", int(self.action_dim))
        object.__setattr__(self, "min_action_horizon", int(self.min_action_horizon))
        object.__setattr__(self, "active_sides", tuple(str(side) for side in self.active_sides))
        object.__setattr__(self, "required_camera_slots", tuple(str(slot) for slot in self.required_camera_slots))
        object.__setattr__(self, "required_image_keys", tuple(str(key) for key in self.required_image_keys))
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
        if remote_policy is None:
            contract = DeployPolicyContract(
                protocol_version=OPENPI_LOCAL_PROTOCOL_VERSION,
                deploy_schema_version=self.config.deploy_schema_version,
                model_id=self.config.expected_model_id,
                checkpoint_step=self.config.expected_checkpoint_step,
                action_key=self.config.action_key,
                action_dim=self.config.action_dim,
                min_action_horizon=self.config.min_action_horizon,
                active_sides=self.config.active_sides,
                require_active_sides_match=self.config.require_active_sides_match,
                required_camera_slots=self.config.required_camera_slots,
                hand_layout=self.config.hand_layout,
            )
            observation_config = OpenPIObservationAdapterConfig(
                prompt=self.config.prompt,
                image_key_map=self.config.image_key_map,
                required_image_keys=self.config.required_image_keys,
                include_debug_metadata=self.config.include_debug_metadata,
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
                    contract=contract,
                    source="openpi_remote_policy",
                ),
            )
        self.remote_policy = remote_policy
        self.name = _policy_name(self.remote_policy.server_metadata)
        self.metadata = dict(getattr(self.remote_policy, "metadata", {}) or {})

    def reset(self) -> None:
        self.remote_policy.reset()

    def predict(self, observation: Any):
        return self.remote_policy.predict(observation)


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
    required_camera_slots: tuple[str, ...] | list[str] = ("base_0_rgb",),
    required_image_keys: tuple[str, ...] | list[str] = ("front",),
    action_key: str = "actions",
    action_dim: int = 26,
    min_action_horizon: int = 1,
    hand_layout: str | None = "q6_slot0_gripper",
    image_key_map: dict[str, str] | None = None,
    include_debug_metadata: bool = True,
    retry_interval_s: float = 2.0,
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
            required_camera_slots=tuple(str(slot) for slot in required_camera_slots),
            required_image_keys=tuple(str(key) for key in required_image_keys),
            action_key=str(action_key),
            action_dim=int(action_dim),
            min_action_horizon=int(min_action_horizon),
            hand_layout=hand_layout,
            image_key_map=image_key_map,
            include_debug_metadata=bool(include_debug_metadata),
            retry_interval_s=float(retry_interval_s),
        )
    )


def _policy_name(metadata: dict[str, Any]) -> str:
    deployment = metadata.get("deployment") if isinstance(metadata.get("deployment"), dict) else {}
    for key in ("model_id", "config_name"):
        value = metadata.get(key) or deployment.get(key)
        if value:
            return f"openpi:{value}"
    return "openpi_remote"

