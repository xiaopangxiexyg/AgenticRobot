from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np

from dataset_contracts import PolicyArtifactContract
from dataset_contracts.v22 import PolicyIOLayout
from deployments.openpi.server import build_runtime_registry_entry
from deployments.openpi.client import (
    OpenPIObservationAdapterConfig,
    OpenPIClientPolicyConfig,
    deployment_observation_to_openpi,
    make_openpi_quantile_normalization_contract,
)
from deployments.client import DeployPolicyContract, validate_server_metadata
from robot_motion.deploy import (
    NORMALIZATION_CONTRACT_SCHEMA_V1,
    NormalizationContract,
    NormalizationContractError,
    NormalizationDomainSpec,
    PolicyAdapterConfig,
    PolicyDeployRuntimeConfig,
    normalization_stats_hash,
)


def test_identity_contract_is_explicit_for_all_three_domains() -> None:
    contract = NormalizationContract.identity()

    payload = contract.to_dict()
    assert payload["schema"] == NORMALIZATION_CONTRACT_SCHEMA_V1
    assert set(payload["domains"]) == {"state", "action", "image"}
    assert all(
        domain["owner"] == "none" and domain["method"] == "identity"
        for domain in payload["domains"].values()
    )
    assert NormalizationContract.from_dict(payload) == contract
    assert contract.contract_hash


def test_non_identity_contract_requires_an_executor_and_space_names() -> None:
    with pytest.raises(NormalizationContractError, match="owner='client' or owner='server'"):
        NormalizationDomainSpec(
            owner="none",
            method="zscore",
            input_space="physical",
            output_space="model",
            stats_ref="normalization.npz::state",
            stats_hash="sha256:state",
        )

    with pytest.raises(NormalizationContractError, match="stats_hash"):
        NormalizationDomainSpec(
            owner="server",
            method="zscore",
            input_space="physical",
            output_space="model",
            stats_ref="normalization.npz::state",
        )

    with pytest.raises(NormalizationContractError, match="distinct input_space"):
        NormalizationDomainSpec(
            owner="client",
            method="affine",
            input_space="raw",
            output_space="raw",
            parameters={"scale": 1.0, "offset": 0.0},
        )

    with pytest.raises(NormalizationContractError, match="operation='normalize'"):
        NormalizationDomainSpec(
            owner="server",
            method="quantile_range",
            operation="identity",
            input_space="raw",
            output_space="model",
            stats_ref="norm_stats.json#state",
            stats_hash="sha256:" + "a" * 64,
        )


def test_openpi_quantile_contract_is_directional_and_server_owned() -> None:
    stats_bytes = b'{"norm_stats":{"state":{},"actions":{}}}'
    stats_hash = normalization_stats_hash(stats_bytes)
    contract = make_openpi_quantile_normalization_contract(
        stats_ref="checkpoint://assets/dual_rm65_dmgripper_v23_relative/norm_stats.json",
        stats_hash=stats_hash,
    )

    parsed = NormalizationContract.from_dict(contract)
    assert parsed.state.method == "quantile_range"
    assert parsed.state.operation == "normalize"
    assert parsed.state.owner == "server"
    assert parsed.action.method == "quantile_range"
    assert parsed.action.operation == "unnormalize"
    assert parsed.action.owner == "server"
    assert parsed.state.stats_ref.endswith("#state")
    assert parsed.action.stats_ref.endswith("#actions")
    assert parsed.image.method == "identity"

    with pytest.raises(NormalizationContractError, match="canonical sha256"):
        make_openpi_quantile_normalization_contract(
            stats_ref="norm_stats.json",
            stats_hash="sha256:state",
        )


def test_range_contract_rejects_reversed_ranges() -> None:
    with pytest.raises(NormalizationContractError, match="max > min"):
        NormalizationDomainSpec(
            owner="client",
            method="range",
            input_space="policy_unit",
            output_space="backend_unit",
            parameters={
                "input_min": 1.0,
                "input_max": 0.0,
                "output_min": 0.0,
                "output_max": 0.9,
            },
        )


def test_legacy_payload_is_opaque_and_never_interpreted() -> None:
    payload = {"schema": "legacy_normalized_v1", "range": "unit"}

    assert NormalizationContract.from_optional_mapping(payload) is None
    with pytest.raises(NormalizationContractError, match="not executable"):
        NormalizationContract.from_optional_mapping(payload, allow_legacy=False)

    config = PolicyAdapterConfig(
        adapter_id="legacy-compat",
        normalization_contract=payload,
    )
    assert config.explicit_normalization_contract is None
    assert config.normalization_contract == payload


def test_adapter_and_runtime_preserve_explicit_contract_and_hash() -> None:
    contract = NormalizationContract(
        state=NormalizationDomainSpec(
            owner="server",
            method="zscore",
            input_space="physical_state",
            output_space="model_state",
            stats_ref="normalization.npz::state",
            stats_hash="sha256:state",
        )
    )
    config = PolicyAdapterConfig(
        adapter_id="explicit-server-normalization",
        normalization_contract=contract.to_dict(),
    )

    runtime = config.to_runtime_config()
    assert runtime.normalization_contract == contract.to_dict()
    assert runtime.normalization_hash == config.normalization_hash
    assert runtime.normalization_contract["domains"]["state"]["owner"] == "server"

    with pytest.raises(ValueError, match="does not match"):
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            normalization_contract=contract.to_dict(),
            normalization_hash="wrong-hash",
        )


def test_openpi_observation_metadata_carries_contract_without_reapplying_it() -> None:
    contract = NormalizationContract(
        state=NormalizationDomainSpec(
            owner="server",
            method="zscore",
            input_space="physical_state",
            output_space="model_state",
            stats_ref="normalization.npz::state",
            stats_hash="sha256:state",
        )
    )
    config = OpenPIObservationAdapterConfig(
        image_key_map={},
        required_image_keys=(),
        normalization_contract=contract.to_dict(),
    )
    observation = type(
        "Observation",
        (),
        {
            "images": {},
            "state": np.asarray([1.0], dtype=np.float32),
            "timestamp": 1.0,
            "step_index": 2,
            "active_sides": ("right",),
            "arms_enabled": True,
            "hands_enabled": True,
            "metadata": {},
        },
    )()

    request = deployment_observation_to_openpi(observation, config)

    # The local adapter only declares the server-owned transform; it does not
    # alter the raw state or perform a second z-score.
    np.testing.assert_array_equal(request["state"], np.asarray([1.0], dtype=np.float32))
    assert request["dexteleop"]["normalization_hash"] == contract.contract_hash
    assert request["dexteleop"]["normalization_contract_status"] == "explicit"


def test_server_metadata_hash_is_checked_only_when_explicitly_required() -> None:
    contract = DeployPolicyContract(
        action_dim=26,
        hand_layout=None,
        require_active_sides_match=False,
        metadata_required=False,
        normalization_hash="expected-normalization-hash",
    )
    mismatch = validate_server_metadata(
        {
            "local_robot": {"output_action_dim": 26},
            "normalization_hash": "wrong-normalization-hash",
        },
        contract,
        raise_on_error=False,
    )
    assert any("normalization_hash_mismatch" in error for error in mismatch.errors)

    legacy = validate_server_metadata(
        {"local_robot": {"output_action_dim": 26}},
        DeployPolicyContract(
            action_dim=26,
            hand_layout=None,
            require_active_sides_match=False,
            metadata_required=False,
        ),
        raise_on_error=False,
    )
    assert not any("normalization_hash" in error for error in legacy.errors)


def test_current_runtime_rejects_unimplemented_client_owned_transform() -> None:
    contract = NormalizationContract(
        state=NormalizationDomainSpec(
            owner="client",
            method="range",
            input_space="physical_state",
            output_space="model_state",
            parameters={
                "input_min": -1.0,
                "input_max": 1.0,
                "output_min": 0.0,
                "output_max": 1.0,
            },
        )
    )
    with pytest.raises(ValueError, match="does not execute client-owned"):
        PolicyAdapterConfig(
            adapter_id="client-normalization-not-yet-bound",
            normalization_contract=contract.to_dict(),
        ).to_runtime_config()

    with pytest.raises(ValueError, match="does not execute client-owned"):
        OpenPIObservationAdapterConfig(
            normalization_contract=contract.to_dict(),
        )


@pytest.mark.integration
def test_openpi_runtime_registry_publishes_explicit_artifact_hash(tmp_path: Path) -> None:
    contract = NormalizationContract(
        state=NormalizationDomainSpec(
            owner="server",
            method="zscore",
            input_space="physical_state",
            output_space="model_state",
            stats_ref="normalization.npz::state",
            stats_hash="sha256:state",
        )
    )
    checkpoint = tmp_path / "checkpoints" / "policy" / "10"
    checkpoint.mkdir(parents=True)
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id="policy",
        checkpoint_step=10,
        metadata={"normalization_contract": contract.to_dict()},
    )
    (checkpoint / "policy_artifact.json").write_text(
        json.dumps(artifact.to_dict()),
        encoding="utf-8",
    )

    _, entry = build_runtime_registry_entry(checkpoint)

    assert entry["normalization_contract_status"] == "explicit"
    assert entry["normalization_hash"] == contract.contract_hash
    assert entry["normalization_contract"] == contract.to_dict()

    policy_config = OpenPIClientPolicyConfig(policy_artifact=artifact)
    assert policy_config.normalization_contract == contract.to_dict()
    with pytest.raises(ValueError, match="disagrees with policy_artifact"):
        OpenPIClientPolicyConfig(
            policy_artifact=artifact,
            normalization_contract=NormalizationContract.identity().to_dict(),
        )


@pytest.mark.integration
def test_openpi_runtime_registry_distinguishes_missing_from_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "legacy-policy" / "11"
    checkpoint.mkdir(parents=True)
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id="legacy-policy",
        checkpoint_step=11,
    )
    (checkpoint / "policy_artifact.json").write_text(
        json.dumps(artifact.to_dict()),
        encoding="utf-8",
    )

    _, entry = build_runtime_registry_entry(checkpoint)

    assert entry["normalization_contract_status"] == "missing"
    assert "normalization_hash" not in entry
