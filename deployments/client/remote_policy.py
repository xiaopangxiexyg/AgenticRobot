from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .action import UnifiedActionChunk
from .contracts import (
    DEFAULT_ACTION_KEY,
    DeployPolicyContract,
    MetadataValidationResult,
    validate_server_metadata,
)


class RemoteTransport(Protocol):
    @property
    def metadata(self) -> dict[str, Any]:
        ...

    def infer(self, observation: dict[str, Any]) -> Mapping[str, Any]:
        ...

    def reset(self) -> None:
        ...


ObservationAdapter = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class RemotePolicyConfig:
    action_key: str = DEFAULT_ACTION_KEY
    action_dim: int = 26
    action_discrete_key: str | None = None
    action_discrete_dim: int = 0
    contract: DeployPolicyContract = field(default_factory=DeployPolicyContract)
    source: str = "remote_model_policy"

    def __post_init__(self) -> None:
        action_key = str(self.action_key or DEFAULT_ACTION_KEY)
        action_dim = int(self.action_dim)
        action_discrete_dim = int(self.action_discrete_dim)
        if action_discrete_dim < 0:
            raise ValueError("action_discrete_dim must be non-negative.")
        action_discrete_key = None if self.action_discrete_key is None else str(self.action_discrete_key).strip()
        if action_discrete_dim and not action_discrete_key:
            raise ValueError("action_discrete_key is required when action_discrete_dim is non-zero.")
        expected_discrete_dim = (
            action_discrete_dim
            if action_discrete_dim > 0 or self.contract.action_discrete_dim is not None
            else None
        )
        object.__setattr__(self, "action_key", action_key)
        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "action_discrete_dim", action_discrete_dim)
        object.__setattr__(self, "action_discrete_key", action_discrete_key or None)
        object.__setattr__(self, "source", str(self.source or "remote_model_policy"))
        if (
            self.contract.action_key != action_key
            or self.contract.action_dim != action_dim
            or self.contract.action_discrete_dim != expected_discrete_dim
        ):
            object.__setattr__(
                self,
                "contract",
                DeployPolicyContract(
                    protocol_version=self.contract.protocol_version,
                    deploy_schema_version=self.contract.deploy_schema_version,
                    model_id=self.contract.model_id,
                    checkpoint_step=self.contract.checkpoint_step,
                    state_key=self.contract.state_key,
                    state_dim=self.contract.state_dim,
                    state_discrete_key=self.contract.state_discrete_key,
                    state_discrete_dim=self.contract.state_discrete_dim,
                    action_key=action_key,
                    action_dim=action_dim,
                    action_discrete_dim=expected_discrete_dim,
                    policy_io_layout_hash=self.contract.policy_io_layout_hash,
                    min_action_horizon=self.contract.min_action_horizon,
                    active_sides=self.contract.active_sides,
                    require_active_sides_match=self.contract.require_active_sides_match,
                    required_camera_slots=self.contract.required_camera_slots,
                    hand_layout=self.contract.hand_layout,
                    metadata_required=self.contract.metadata_required,
                    extra=self.contract.extra,
                ),
            )


class RemoteModelPolicy:
    """Model-agnostic remote policy client.

    Transport and observation field names are supplied by model-family packages;
    this class owns metadata validation and conversion of server responses into
    the unified dense action chunk.
    """

    def __init__(
        self,
        *,
        transport: RemoteTransport,
        observation_adapter: ObservationAdapter,
        config: RemotePolicyConfig | None = None,
    ) -> None:
        self.config = RemotePolicyConfig() if config is None else config
        self.transport = transport
        self.observation_adapter = observation_adapter
        self.server_metadata = dict(self.transport.metadata)
        self.validation = validate_server_metadata(self.server_metadata, self.config.contract)
        self.metadata = {
            "policy": self.config.source,
            "server": dict(self.server_metadata),
            "validation": dict(self.validation.server_summary),
        }

    def predict(self, observation: Any) -> UnifiedActionChunk:
        return self.predict_chunk(observation)

    def predict_chunk(self, observation: Any) -> UnifiedActionChunk:
        request = self.observation_adapter(observation)
        response = dict(self.transport.infer(request))
        if self.config.action_key not in response:
            raise ValueError(
                f"Remote response missing action key {self.config.action_key!r}; "
                f"available keys={sorted(str(key) for key in response)}."
            )
        actions = np.asarray(response[self.config.action_key], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2:
            raise ValueError(f"Remote actions must be 2D [horizon, dim], got shape {actions.shape}.")
        if actions.shape[1] != int(self.config.action_dim):
            raise ValueError(
                f"Remote action dim mismatch: expected {self.config.action_dim}, got {actions.shape[1]}."
            )
        discrete_actions = None
        if self.config.action_discrete_dim:
            assert self.config.action_discrete_key is not None
            if self.config.action_discrete_key not in response:
                raise ValueError(
                    "Remote response missing discrete action key "
                    f"{self.config.action_discrete_key!r}; "
                    f"available keys={sorted(str(key) for key in response)}."
                )
            discrete_actions = np.asarray(response[self.config.action_discrete_key])
            if discrete_actions.ndim == 1:
                discrete_actions = discrete_actions.reshape(1, -1)
            if discrete_actions.ndim != 2:
                raise ValueError(
                    "Remote discrete actions must be 2D [horizon, token_dim], "
                    f"got shape {discrete_actions.shape}."
                )
            if discrete_actions.shape != (
                actions.shape[0],
                int(self.config.action_discrete_dim),
            ):
                raise ValueError(
                    "Remote discrete action shape mismatch: expected "
                    f"({actions.shape[0]}, {int(self.config.action_discrete_dim)}), "
                    f"got {discrete_actions.shape}."
                )
        return UnifiedActionChunk(
            actions=actions,
            discrete_actions=discrete_actions,
            metadata=_action_metadata(
                source=self.config.source,
                response=response,
                server_metadata=self.server_metadata,
                validation=self.validation,
                action_key=self.config.action_key,
                action_discrete_key=self.config.action_discrete_key,
            ),
        )

    def reset(self) -> None:
        self.transport.reset()


def _action_metadata(
    *,
    source: str,
    response: Mapping[str, Any],
    server_metadata: Mapping[str, Any],
    validation: MetadataValidationResult,
    action_key: str,
    action_discrete_key: str | None,
) -> dict[str, Any]:
    timing = {}
    for key in ("server_timing", "policy_timing"):
        value = response.get(key)
        if isinstance(value, Mapping):
            timing[key] = dict(value)
    return {
        "source": str(source),
        "remote_policy": {
            "action_key": str(action_key),
            "action_discrete_key": action_discrete_key,
            "server": dict(server_metadata),
            "server_summary": dict(validation.server_summary),
            "validation_warnings": list(validation.warnings),
            "response_keys": sorted(str(key) for key in response.keys()),
            **timing,
        },
    }
