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
    contract: DeployPolicyContract = field(default_factory=DeployPolicyContract)
    source: str = "remote_model_policy"

    def __post_init__(self) -> None:
        action_key = str(self.action_key or DEFAULT_ACTION_KEY)
        action_dim = int(self.action_dim)
        object.__setattr__(self, "action_key", action_key)
        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "source", str(self.source or "remote_model_policy"))
        if self.contract.action_key != action_key or self.contract.action_dim != action_dim:
            object.__setattr__(
                self,
                "contract",
                DeployPolicyContract(
                    protocol_version=self.contract.protocol_version,
                    deploy_schema_version=self.contract.deploy_schema_version,
                    model_id=self.contract.model_id,
                    checkpoint_step=self.contract.checkpoint_step,
                    action_key=action_key,
                    action_dim=action_dim,
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
        return UnifiedActionChunk(
            actions=actions,
            metadata=_action_metadata(
                source=self.config.source,
                response=response,
                server_metadata=self.server_metadata,
                validation=self.validation,
                action_key=self.config.action_key,
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
            "server": dict(server_metadata),
            "server_summary": dict(validation.server_summary),
            "validation_warnings": list(validation.warnings),
            "response_keys": sorted(str(key) for key in response.keys()),
            **timing,
        },
    }

