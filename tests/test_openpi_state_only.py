from __future__ import annotations

import types

import numpy as np
import pytest

from deployments.openpi.state_only import (
    OPENPI_STATE_ONLY_CAPABILITY_KEY_V1,
    OPENPI_STATE_ONLY_PROFILE_V1,
    OpenPIStateOnlyError,
    OpenPIStateOnlyInputBridge,
    bind_state_only_policy_metadata,
    install_openpi_state_only_server_adapter,
)


def test_state_only_training_bridge_emits_only_masked_constant_images() -> None:
    bridge = OpenPIStateOnlyInputBridge(image_keys=("base_0_rgb",))
    state = np.arange(16, dtype=np.float32)
    actions = np.zeros((16, 16), dtype=np.float32)
    output = bridge({"state": state, "actions": actions})

    assert output["state"] is state
    assert output["actions"] is actions
    assert tuple(output["image"]) == ("base_0_rgb",)
    assert output["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert output["image"]["base_0_rgb"].dtype == np.uint8
    assert not output["image"]["base_0_rgb"].any()
    assert output["image_mask"] == {"base_0_rgb": np.False_}
    assert bridge.contract["source_media_decoded"] is False
    assert bridge.external_camera_contract.required is False
    assert bridge.external_camera_contract.inputs == ()


def test_state_only_bridge_rejects_missing_state_and_unknown_profile() -> None:
    bridge = OpenPIStateOnlyInputBridge(image_keys=("base_0_rgb",))
    with pytest.raises(OpenPIStateOnlyError, match="missing state"):
        bridge({"actions": np.zeros((2, 16), dtype=np.float32)})
    with pytest.raises(OpenPIStateOnlyError, match="Unsupported"):
        OpenPIStateOnlyInputBridge.from_profile_value(
            "not-a-profile",
            image_keys=("base_0_rgb",),
        )


def test_state_only_metadata_replaces_external_camera_contract() -> None:
    bridge = OpenPIStateOnlyInputBridge(image_keys=("base_0_rgb",))
    metadata = bind_state_only_policy_metadata(
        {
            "policy_artifact": {
                "camera_input": {
                    "inputs": [{"server_camera_slot": "base_0_rgb"}],
                    "required": True,
                },
                "metadata": {"existing": True},
            }
        },
        bridge,
    )

    assert metadata[OPENPI_STATE_ONLY_CAPABILITY_KEY_V1] == bridge.contract
    artifact = metadata["policy_artifact"]
    assert artifact["camera_input"]["inputs"] == []
    assert artifact["camera_input"]["required"] is False
    assert artifact["metadata"]["existing"] is True
    assert (
        artifact["metadata"][OPENPI_STATE_ONLY_CAPABILITY_KEY_V1]
        == bridge.contract
    )


def test_state_only_server_adapter_injects_bridge_for_plain_and_conditioned_requests() -> None:
    seen: list[dict[str, object]] = []

    class FakeServer:
        def __init__(self) -> None:
            self._metadata: dict[str, object] = {}

        def _infer(self, request: dict[str, object]) -> dict[str, object]:
            seen.append(request)
            return request

    module = types.SimpleNamespace(WebsocketPolicyServer=FakeServer)
    bridge = OpenPIStateOnlyInputBridge(image_keys=("base_0_rgb",))
    with install_openpi_state_only_server_adapter(module, bridge):
        server = module.WebsocketPolicyServer()
        plain = server._infer({"state": np.zeros(16, dtype=np.float32)})
        conditioned = server._infer(
            {
                "__openpi_request__": True,
                "observation": {"state": np.zeros(16, dtype=np.float32)},
            }
        )
        assert server._metadata[OPENPI_STATE_ONLY_CAPABILITY_KEY_V1] == (
            bridge.contract
        )

    assert module.WebsocketPolicyServer is FakeServer
    assert plain["image_mask"] == {"base_0_rgb": np.False_}
    assert conditioned["observation"]["image_mask"] == {
        "base_0_rgb": np.False_
    }
    assert len(seen) == 2


def test_state_only_profile_value_is_additive_and_default_off() -> None:
    assert (
        OpenPIStateOnlyInputBridge.from_profile_value(
            None,
            image_keys=("base_0_rgb",),
        )
        is None
    )
    bridge = OpenPIStateOnlyInputBridge.from_profile_value(
        OPENPI_STATE_ONLY_PROFILE_V1,
        image_keys=("base_0_rgb",),
    )
    assert bridge is not None
