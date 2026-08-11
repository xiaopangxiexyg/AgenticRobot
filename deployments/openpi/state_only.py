"""State-only observation bridge for an otherwise ordinary OpenPI policy.

The current upstream Pi0/Pi0.5 model contract requires at least one internal
image slot.  A state-only DexTeleop policy therefore keeps that internal slot
but binds it to a constant zero image with a false image mask.  Source images
are never decoded and external policy requests do not carry camera inputs.

This is an input adapter, not a model-RTC feature.  It leaves action chunk
generation, checkpointing, and OpenPI's ordinary inference path unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.policy_artifact import PolicyCameraInputContract


OPENPI_STATE_ONLY_PROFILE_V1 = "state_only_masked_dummy_v1"
OPENPI_STATE_ONLY_CONTRACT_SCHEMA_V1 = (
    "dexteleop.openpi_state_only_input_bridge.v1"
)
OPENPI_STATE_ONLY_CONTRACT_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_state_only_input_bridge.v1"
)
OPENPI_STATE_ONLY_CAPABILITY_KEY_V1 = "dexteleop_state_only_input_v1"


class OpenPIStateOnlyError(ValueError):
    """Raised when a state-only OpenPI bridge cannot be honored exactly."""


@dataclass(frozen=True)
class OpenPIStateOnlyInputBridge:
    """Bind external state-only observations to OpenPI internal image slots."""

    image_keys: tuple[str, ...]
    image_shape: tuple[int, int, int] = (224, 224, 3)
    profile: str = OPENPI_STATE_ONLY_PROFILE_V1

    def __post_init__(self) -> None:
        profile = str(self.profile).strip()
        if profile != OPENPI_STATE_ONLY_PROFILE_V1:
            raise OpenPIStateOnlyError(
                f"Unsupported OpenPI state-only profile: {profile!r}."
            )
        keys = tuple(str(value).strip() for value in self.image_keys)
        if not keys or any(not value for value in keys):
            raise OpenPIStateOnlyError(
                "The compatibility bridge requires at least one internal image key."
            )
        if len(keys) != len(set(keys)):
            raise OpenPIStateOnlyError(
                "OpenPI state-only internal image keys must be unique."
            )
        shape = tuple(int(value) for value in self.image_shape)
        if len(shape) != 3 or shape[0] <= 0 or shape[1] <= 0 or shape[2] != 3:
            raise OpenPIStateOnlyError(
                "OpenPI state-only image_shape must be positive [H, W, 3]."
            )
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "image_keys", keys)
        object.__setattr__(self, "image_shape", shape)

    @classmethod
    def from_profile_value(
        cls,
        value: object,
        *,
        image_keys: Sequence[str],
    ) -> "OpenPIStateOnlyInputBridge | None":
        if value is None:
            return None
        profile = str(value).strip()
        if profile != OPENPI_STATE_ONLY_PROFILE_V1:
            raise OpenPIStateOnlyError(
                f"Unsupported openpi_observation_mode: {profile!r}."
            )
        return cls(image_keys=tuple(str(key) for key in image_keys))

    @property
    def external_camera_contract(self) -> PolicyCameraInputContract:
        return PolicyCameraInputContract(inputs=(), required=False)

    @property
    def contract(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": OPENPI_STATE_ONLY_CONTRACT_SCHEMA_V1,
            "profile": self.profile,
            "external_camera_required": False,
            "external_camera_slots": [],
            "source_media_decoded": False,
            "internal_image_keys": list(self.image_keys),
            "internal_image_shape": list(self.image_shape),
            "internal_image_dtype": "uint8",
            "internal_image_value": "zero",
            "internal_image_mask": False,
        }
        payload["contract_hash"] = "sha256:" + canonical_content_hash(
            payload,
            domain=OPENPI_STATE_ONLY_CONTRACT_HASH_DOMAIN_V1,
        )
        return payload

    def apply(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(sample, Mapping):
            raise TypeError("OpenPI state-only input must be a mapping.")
        if sample.get("state", sample.get("observation.state")) is None:
            raise OpenPIStateOnlyError("OpenPI state-only input is missing state.")
        output = dict(sample)
        output["image"] = {
            key: np.zeros(self.image_shape, dtype=np.uint8)
            for key in self.image_keys
        }
        output["image_mask"] = dict.fromkeys(self.image_keys, np.False_)
        return output

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        return self.apply(sample)

    def apply_with_identity(
        self,
        sample: Mapping[str, Any],
        *,
        identity: tuple[int, int],
    ) -> dict[str, Any]:
        del identity
        return self.apply(sample)


def bind_state_only_policy_metadata(
    metadata: Mapping[str, Any],
    bridge: OpenPIStateOnlyInputBridge,
) -> dict[str, Any]:
    """Return metadata whose external artifact contract is camera-free."""

    output = dict(metadata)
    contract = bridge.contract
    output[OPENPI_STATE_ONLY_CAPABILITY_KEY_V1] = contract
    artifact_value = output.get("policy_artifact")
    if isinstance(artifact_value, Mapping):
        artifact = dict(artifact_value)
        artifact["camera_input"] = bridge.external_camera_contract.to_dict()
        artifact_metadata = dict(artifact.get("metadata") or {})
        artifact_metadata[OPENPI_STATE_ONLY_CAPABILITY_KEY_V1] = contract
        artifact["metadata"] = artifact_metadata
        output["policy_artifact"] = artifact
    return output


def _copy_request_observation(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = dict(request)
    if "__openpi_request__" not in root:
        return root, root
    observation_value = root.get("observation")
    if not isinstance(observation_value, Mapping):
        raise OpenPIStateOnlyError(
            "Conditioned OpenPI state-only request observation must be a mapping."
        )
    observation = dict(observation_value)
    root["observation"] = observation
    return root, observation


@contextmanager
def install_openpi_state_only_server_adapter(
    websocket_policy_server_module: Any,
    bridge: OpenPIStateOnlyInputBridge | None,
) -> Iterator[None]:
    """Inject the constant masked image inside one prepared server process."""

    if bridge is None:
        yield
        return
    if not isinstance(bridge, OpenPIStateOnlyInputBridge):
        raise TypeError("bridge must be an OpenPIStateOnlyInputBridge or None.")
    original = websocket_policy_server_module.WebsocketPolicyServer

    class StateOnlyWebsocketPolicyServer(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            existing = self._metadata.get(OPENPI_STATE_ONLY_CAPABILITY_KEY_V1)
            contract = bridge.contract
            if existing is not None and existing != contract:
                raise OpenPIStateOnlyError(
                    "OpenPI server already advertises a different state-only contract."
                )
            self._metadata[OPENPI_STATE_ONLY_CAPABILITY_KEY_V1] = contract

        def _infer(self, request: Any) -> Any:
            if not isinstance(request, Mapping):
                raise OpenPIStateOnlyError(
                    "OpenPI state-only server request must be a mapping."
                )
            adapted, observation = _copy_request_observation(request)
            observation.update(bridge.apply(observation))
            return super()._infer(adapted)

    StateOnlyWebsocketPolicyServer.__name__ = original.__name__
    StateOnlyWebsocketPolicyServer.__qualname__ = original.__qualname__
    websocket_policy_server_module.WebsocketPolicyServer = (
        StateOnlyWebsocketPolicyServer
    )
    try:
        yield
    finally:
        if (
            websocket_policy_server_module.WebsocketPolicyServer
            is StateOnlyWebsocketPolicyServer
        ):
            websocket_policy_server_module.WebsocketPolicyServer = original


__all__ = [
    "OPENPI_STATE_ONLY_CAPABILITY_KEY_V1",
    "OPENPI_STATE_ONLY_CONTRACT_SCHEMA_V1",
    "OPENPI_STATE_ONLY_PROFILE_V1",
    "OpenPIStateOnlyError",
    "OpenPIStateOnlyInputBridge",
    "bind_state_only_policy_metadata",
    "install_openpi_state_only_server_adapter",
]
