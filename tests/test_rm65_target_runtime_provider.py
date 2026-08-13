from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from robot_motion.target_runtime import (
    TargetRuntimeProviderRegistry,
    TargetRuntimeProviderRegistryEntry,
)
from teleop_runtime.deploy.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
)
from teleop_runtime.deploy.rm65_execution_evidence import (
    build_rm65_execution_evidence_bundle,
)
from teleop_runtime.deploy.rm65_target_runtime_provider import (
    RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_ID,
    RM65PolicyArmQposCandidateTargetConfig,
    build_rm65_policy_arm_qpos_candidate_provider_entry,
    preflight_rm65_policy_arm_qpos_candidate_target,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "rm65_policy_arm_qpos_candidate_v1.json"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _characterize_current_candidate() -> dict[str, object]:
    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    evidence = build_rm65_execution_evidence_bundle(manifest)
    return {
        "schema_version": (
            "rm65_policy_arm_qpos_candidate_characterization.v1"
        ),
        "manifest": {
            "manifest_id": manifest.manifest_id,
            "manifest_version": manifest.manifest_version,
            "semantic_content_hash": manifest.semantic_content_hash,
            "action_space": manifest.action_space,
            "action_schema_hash": manifest.action_schema_hash,
            "component_target_capability_hash": (
                manifest.component_target_capability_hash
            ),
            "component_catalog_hash": manifest.component_catalog_hash,
            "controlled_components": list(manifest.controlled_components),
            "selection_groups": list(manifest.selection_groups),
            "motion_authority_sources": list(
                manifest.motion_authority_sources
            ),
            "human_motion_proposal_status": (
                manifest.human_motion_proposal_status
            ),
            "safety_control_requests": list(
                manifest.safety_control_requests
            ),
            "lifecycle_control_requests": list(
                manifest.lifecycle_control_requests
            ),
            "actual_reference_frame_producer": (
                manifest.actual_reference_frame_producer
            ),
            "exclusion_hash_by_capability": {
                item.capability_id: item.semantic_content_hash
                for item in manifest.exclusions
            },
        },
        "evidence": {
            "bundle_hash": evidence.semantic_content_hash,
            "integration_status": evidence.integration_status,
            "production_binding_enabled": (
                evidence.production_binding_enabled
            ),
            "confers_execution_authorization": (
                evidence.confers_execution_authorization
            ),
            "dispatch_inventory_hash": (
                evidence.dispatch_inventory.semantic_content_hash
            ),
            "dispatch_seam_ids": [
                item.seam_id for item in evidence.dispatch_inventory.entries
            ],
            "worker_drain_plan_hash": (
                evidence.worker_drain_plan.semantic_content_hash
            ),
            "worker_participant_ids": [
                item.participant_id
                for item in evidence.worker_drain_plan.participants
            ],
            "safety_migration_matrix_hash": (
                evidence.safety_migration_matrix.semantic_content_hash
            ),
            "inventory_gate_ready": evidence.inventory_gate.ready,
            "legacy_mediation_gate_ready": (
                evidence.legacy_mediation_gate.ready
            ),
            "shared_authority_gate_ready": (
                evidence.shared_authority_gate.ready
            ),
        },
    }


def _config(
    *,
    manifest: object | None = None,
) -> RM65PolicyArmQposCandidateTargetConfig:
    return RM65PolicyArmQposCandidateTargetConfig(
        control_execution_manifest=(
            build_rm65_policy_arm_qpos_candidate_manifest()
            if manifest is None
            else manifest
        ),
        embodiment_instance_id="lab-a-rm65-pair-01",
        embodiment_instance_hash="instance-profile-hash:lab-a-rm65-pair-01",
        clock_domain_id="control.monotonic",
    )


def test_frozen_candidate_characterization_fixture_has_not_drifted() -> None:
    assert _characterize_current_candidate() == _fixture()


def test_preflight_executes_without_loading_backend_or_vendor_sdk() -> None:
    script = """
import json
import sys
from teleop_runtime.deploy.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
)
from teleop_runtime.deploy.rm65_target_runtime_provider import (
    RM65PolicyArmQposCandidateTargetConfig,
    preflight_rm65_policy_arm_qpos_candidate_target,
)

result = preflight_rm65_policy_arm_qpos_candidate_target(
    RM65PolicyArmQposCandidateTargetConfig(
        control_execution_manifest=(
            build_rm65_policy_arm_qpos_candidate_manifest()
        ),
        embodiment_instance_id="lab-a-rm65-pair-01",
        embodiment_instance_hash="instance-profile-hash",
        clock_domain_id="control.monotonic",
    ),
    runtime_session_id="session-preflight-subprocess",
    control_domain_id="rm65-arms",
    bound_at=100.0,
)

forbidden_modules = {
    "teleop_runtime.backends.arms.rm65",
    "teleop_runtime.backends.arms.rm65_impedance",
}
forbidden_vendor_prefixes = (
    "hardware.rm65.Robotic_Arm",
    "hardware.rm65.rm_api2_adapter",
)
print(json.dumps({
    "forbidden_modules": sorted(
        name
        for name in sys.modules
        if name in forbidden_modules
        or name.startswith(forbidden_vendor_prefixes)
    ),
    "runtime_handle_created": result.runtime_handle_created,
    "confers_execution_authorization": (
        result.confers_execution_authorization
    ),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "confers_execution_authorization": False,
        "forbidden_modules": [],
        "runtime_handle_created": False,
    }


def test_preflight_projects_exact_legacy_contract_without_initializing_backend(
) -> None:
    result = preflight_rm65_policy_arm_qpos_candidate_target(
        _config(),
        runtime_session_id="session-preflight-1",
        control_domain_id="rm65-arms",
        bound_at=100.0,
    )
    manifest = result.resolved_manifest
    legacy = _fixture()["manifest"]
    evidence = _fixture()["evidence"]

    assert manifest.provider.runtime_handle_mode == "preflight_only"
    assert manifest.action_schema_hash == legacy["action_schema_hash"]
    assert manifest.component_ids == tuple(legacy["controlled_components"])
    assert manifest.execution_observation_capability is None
    assert manifest.operation_capabilities == ()
    assert result.binding_receipt.bound_component_ids == manifest.component_ids
    assert result.legacy_execution_manifest_hash == (
        legacy["semantic_content_hash"]
    )
    assert result.execution_evidence_bundle_hash == evidence["bundle_hash"]
    assert result.integration_status == "evidence_only_unbound"
    assert not result.runtime_handle_created
    assert not result.confers_execution_authorization
    assert {
        item.asset_id: item.content_hash for item in manifest.asset_references
    } == {
        "legacy-control-execution-manifest": legacy[
            "semantic_content_hash"
        ],
        "legacy-dispatch-inventory": evidence["dispatch_inventory_hash"],
        "legacy-safety-migration-matrix": evidence[
            "safety_migration_matrix_hash"
        ],
        "legacy-worker-drain-plan": evidence["worker_drain_plan_hash"],
    }


def test_preflight_session_identity_changes_without_mutating_legacy_contract() -> None:
    config = _config()
    first = preflight_rm65_policy_arm_qpos_candidate_target(
        config,
        runtime_session_id="session-preflight-1",
        control_domain_id="rm65-arms",
        bound_at=100.0,
    )
    second = preflight_rm65_policy_arm_qpos_candidate_target(
        config,
        runtime_session_id="session-preflight-2",
        control_domain_id="rm65-arms",
        bound_at=100.0,
    )

    assert (
        first.resolved_manifest.semantic_content_hash
        != second.resolved_manifest.semantic_content_hash
    )
    assert (
        first.legacy_execution_manifest_hash
        == second.legacy_execution_manifest_hash
    )
    assert (
        first.resolved_manifest.embodiment_definition_hash
        == second.resolved_manifest.embodiment_definition_hash
    )
    assert first.resolved_manifest.asset_references == (
        second.resolved_manifest.asset_references
    )


def test_preflight_only_registry_rejects_runtime_handle_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    preflight = preflight_rm65_policy_arm_qpos_candidate_target(
        config,
        runtime_session_id="session-preflight-1",
        control_domain_id="rm65-arms",
        bound_at=100.0,
    )
    entry = build_rm65_policy_arm_qpos_candidate_provider_entry()
    def reject_provider_handle(*args: object, **kwargs: object) -> None:
        raise AssertionError("registry must reject before provider handle call")

    provider = entry.loader()
    monkeypatch.setattr(
        provider,
        "create_target_runtime_handle",
        reject_provider_handle,
    )
    load_count = 0

    def load_provider() -> object:
        nonlocal load_count
        load_count += 1
        return provider

    registry = TargetRuntimeProviderRegistry()
    registry.register(
        TargetRuntimeProviderRegistryEntry(
            provider_identity=entry.provider_identity,
            loader=load_provider,
        )
    )

    with pytest.raises(RuntimeError, match="preflight-only"):
        registry.create_handle(
            RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_ID,
            config,
            manifest=preflight.resolved_manifest,
            binding_receipt=preflight.binding_receipt,
        )
    assert load_count == 0


@pytest.mark.parametrize(
    ("changes", "label"),
    (
        (
            {"controlled_components": ("left_arm",)},
            "single-arm topology",
        ),
        (
            {"motion_authority_sources": ("policy-v2",)},
            "authority source",
        ),
        (
            {"actual_reference_frame_producer": "fabricated-reference"},
            "reference producer",
        ),
        (
            {"semantic_content_hash": "tampered-manifest-hash"},
            "manifest hash",
        ),
    ),
)
def test_provider_fails_closed_on_candidate_feature_drift(
    changes: dict[str, object],
    label: str,
) -> None:
    candidate = build_rm65_policy_arm_qpos_candidate_manifest()
    drifted = replace(candidate, **changes)

    with pytest.raises(ValueError, match="exact frozen"):
        preflight_rm65_policy_arm_qpos_candidate_target(
            _config(manifest=drifted),
            runtime_session_id="session-preflight-1",
            control_domain_id="rm65-arms",
            bound_at=100.0,
        )


@pytest.mark.parametrize(
    ("field", "missing_value"),
    tuple(
        (field, missing_value)
        for field in (
            "embodiment_instance_id",
            "embodiment_instance_hash",
            "clock_domain_id",
        )
        for missing_value in (" ", None)
    ),
)
def test_provider_config_rejects_missing_instance_or_clock_identity(
    field: str,
    missing_value: object,
) -> None:
    values = {
        "control_execution_manifest": (
            build_rm65_policy_arm_qpos_candidate_manifest()
        ),
        "embodiment_instance_id": "lab-a-rm65-pair-01",
        "embodiment_instance_hash": "instance-profile-hash",
        "clock_domain_id": "control.monotonic",
    }
    values[field] = missing_value
    with pytest.raises(ValueError, match=field):
        RM65PolicyArmQposCandidateTargetConfig(**values)
