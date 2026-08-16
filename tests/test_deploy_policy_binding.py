from __future__ import annotations

from types import SimpleNamespace

import pytest

from teleop_runtime.deploy import policy_binding, run_policy


class _Layout:
    layout_hash = "layout-hash"

    def end_effector(self, side: str) -> SimpleNamespace:
        return SimpleNamespace(
            present=side == "left",
            state_source_field=f"{side}_state",
        )

    def channel(self, kind: str, side: str) -> SimpleNamespace:
        assert kind == "state"
        return SimpleNamespace(
            tokens=(
                SimpleNamespace(source_metadata_key=f"{side}_token"),
                SimpleNamespace(source_metadata_key=None),
            )
        )


class _Artifact:
    def to_dict(self) -> dict[str, object]:
        return {"artifact": "unit"}


def test_run_policy_preserves_policy_binding_compatibility_exports() -> None:
    assert (
        run_policy._bind_policy_pose_mode_from_artifact
        is policy_binding.bind_policy_pose_mode_from_artifact
    )
    assert (
        run_policy._policy_observation_contract
        is policy_binding.build_policy_observation_contract
    )
    assert run_policy.parse_policy_kwargs is policy_binding.parse_policy_kwargs


def test_policy_kwargs_parser_preserves_object_only_contract() -> None:
    assert policy_binding.parse_policy_kwargs("") == {}
    assert policy_binding.parse_policy_kwargs('{"host": "127.0.0.1"}') == {
        "host": "127.0.0.1"
    }
    with pytest.raises(
        TypeError,
        match="--policy-kwargs-json must decode to a JSON object",
    ):
        policy_binding.parse_policy_kwargs("[]")


def test_observation_binding_prefers_artifact_identity_and_collects_layout_keys() -> None:
    contract = policy_binding.build_policy_observation_contract(
        policy=SimpleNamespace(name="unit-policy"),
        policy_artifact=_Artifact(),
        deploy_layout=_Layout(),
        pose_mode="relative",
    )

    assert contract.contract_id.startswith("dexteleop.deploy.actor.v1:artifact=")
    assert contract.last_action_contract is not None
    assert contract.last_action_contract.pose_mode == "relative"
    assert "left_state" in contract.compatibility_hand_state_keys
    assert "left_token" in contract.compatibility_hand_state_keys
    assert "right_token" in contract.compatibility_hand_state_keys


def test_observation_binding_uses_loaded_policy_layout_then_policy_identity() -> None:
    layout = _Layout()
    policy = SimpleNamespace(
        name="unit-policy",
        policy=SimpleNamespace(
            config=SimpleNamespace(policy_io_layout=layout),
        ),
    )

    layout_contract = policy_binding.build_policy_observation_contract(
        policy=policy,
        policy_artifact=None,
        deploy_layout=None,
        pose_mode="absolute",
    )
    assert layout_contract.contract_id.endswith(
        "layout=layout-hash;model=unversioned"
    )

    policy_contract = policy_binding.build_policy_observation_contract(
        policy=SimpleNamespace(name="unit-policy"),
        policy_artifact=None,
        deploy_layout=None,
        pose_mode="absolute",
    )
    assert policy_contract.contract_id.endswith("policy=unit-policy")
