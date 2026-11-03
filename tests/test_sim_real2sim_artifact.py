from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from sim.contracts import SimulationContract
from sim.rm65.real2sim import (
    ArtifactRef,
    FidelityClaim,
    Real2SimArtifactManifest,
    ResolvedTiming,
    ValidationEvidence,
)

CONTRACT_PATH = Path("sim/configs/rm65_bimanual_physics_v1.json")
ARM_PROFILE_PATH = Path("sim/rm65/actuation/profile.py")
ARM_CORE_PATH = Path("sim/rm65/actuation/core.py")
ARM_MUJOCO_ADAPTER_PATH = Path("sim/mujoco/actuation/executor.py")
ARM_BASELINE_PATH = Path(
    "sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4.json"
)
ARM_RUNTIME_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261103_mujoco_oracle_array_v1.json"
)
MUJOCO_G4_RELEASE_LEVELING_V1_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261103_mujoco_g4_release_leveling_v1.json"
)
COMMAND_REFUSAL_V1_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261103_command_refusal_v1.json"
)
MUJOCO_G2_QUERIES_V1_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261103_mujoco_g2_queries_v1.json"
)
PARAMETER_FEEDBACK_V1_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261103_parameter_feedback_v1.json"
)
MUJOCO_CLOSURE_ENTRY_V1_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261103_mujoco_closure_entry_v1.json"
)
MUJOCO_PLACE_SLOPE_V1_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261101_mujoco_place_slope_v1.json"
)
REAL_HARDWARE_CONNECTION_V1_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261101_real_hardware_connection_v1.json"
)
REAL_HELD_CLOSURE_V1_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261101_real_held_closure_v1.json"
)
REAL_SMALL_SUPPORT_V1_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261101_real_small_support_v1.json"
)
REAL_SUPPORT_RELEASE_V1_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261101_real_support_release_v1.json"
)
REAL_RM65_6F_V1_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261031_real_rm65_6f_v1.json"
)
REAL_HELD_OWN_WRIST_V1_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261031_real_held_own_wrist_v1.json"
)
MUJOCO_AGENT_PLACE_V3_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261031_mujoco_agent_place_v3.json"
)
MUJOCO_AGENT_PLACE_V2_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261030_mujoco_agent_place_v2.json"
)
REAL_PLACE_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261030_real_place_v1.json"
)
MUJOCO_AGENT_PLACE_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261030_mujoco_agent_place_v1.json"
)
GATEWAY_UNIFY_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261029_skill_gateway_unify_v1.json"
)
SKILL_RELATIONS_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261029_skill_relations_v1.json"
)
OBSERVED_CACHE_ORDER_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261029_observed_cache_order_v1.json"
)
HOVER_VISUAL_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261029_skill_hover_visual_v1.json"
)
SKILL_REGION_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_skill_region_checked_v2.json"
)
UNQUALIFIED_REGION_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_skill_region_v1.json"
)
EXPLICIT_CLEARANCE_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_explicit_clearance_v1.json"
)
REAL_OBSERVED_MOTION_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_real_observed_motion_v1.json"
)
NAMED_RELEASE_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_named_release_v1.json"
)
AGENT_PROPOSAL_CUT_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_agent_proposal_cut_v1.json"
)
SAM3_PLACEMENT_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_sam3_placement_v1.json"
)
SAM3_CANDIDATES_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261028_sam3_candidates_v1.json"
)
MOTION_GRIPPER_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261027_motion_gripper_v1.json"
)
SAM3_DEFAULT_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261027_sam3_default_v1.json"
)
READONLY_DEFAULTS_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261027_readonly_defaults_v1.json"
)
CHECKPOINT_INTEGRATION_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261026_checkpoint_integration_v1.json"
)
ARM_RUNTIME_BINDING_V25_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261026_v25.json"
)
CHECKPOINT_OBSERVER_BINDING_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261026_checkpoint_observer_v1.json"
)
ARM_RUNTIME_BINDING_V24_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v24.json'
)
ARM_RUNTIME_BINDING_V23_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v23.json'
)
ARM_RUNTIME_BINDING_V22_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v22.json'
)
ARM_RUNTIME_BINDING_V21_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v21.json'
)
ARM_RUNTIME_BINDING_V20_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v20.json'
)
ARM_RUNTIME_BINDING_V19_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v19.json'
)
ARM_RUNTIME_BINDING_V18_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v18.json'
)
ARM_RUNTIME_BINDING_V17_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v17.json'
)
ARM_RUNTIME_BINDING_V16_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v16.json'
)
ARM_RUNTIME_BINDING_V15_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261024_v15.json'
)
ARM_RUNTIME_BINDING_V14_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261023_v14.json'
)
ARM_RUNTIME_BINDING_V13_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261023_v13.json'
)
ARM_RUNTIME_BINDING_V12_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261023_v12.json'
)
ARM_RUNTIME_BINDING_V10_PATH = Path(
    'sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261023_v10.json'
)
ARM_RUNTIME_BINDING_V9_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261022_v9.json"
)
ARM_RUNTIME_BINDING_V8_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261018_v8.json"
)
ARM_RUNTIME_BINDING_V7_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261017_v7.json"
)
ARM_RUNTIME_BINDING_V5_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20260922_v5.json"
)
ARM_RUNTIME_BINDING_V4_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20260829_v4.json"
)
ARM_RUNTIME_BINDING_V3_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20260822_v3.json"
)
ARM_RUNTIME_BINDING_V2_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20260811_v2.json"
)
ARM_RUNTIME_BINDING_V1_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20260724_v1.json"
)
ARM_V3_BASELINE_PATH = Path(
    "sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v3.json"
)
ARM_V2_BASELINE_PATH = Path(
    "sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v2.json"
)
ARM_V1_BASELINE_PATH = Path(
    "sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v1.json"
)
CAMERA_REGISTRY_PATH = Path("robot_motion/configs/rm65_camera_registry.json")
CAMERA_REGISTRY_V1_ARCHIVE_PATH = Path(
    "sim/rm65/real2sim/artifacts/rm65_camera_registry_runtime_20260724_v1.json"
)
CONTRACT_V2_ARCHIVE_PATH = Path(
    "sim/rm65/real2sim/artifacts/"
    "rm65_bimanual_physics_runtime_20260811_v2.json"
)

_CAD_SCENE_URDF = (
    "assets/rm65_inspire_gripper_v3/"
    "rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf"
)
_CAMERA_FRAME_BY_ROLE = {
    "front": ("head_camera_frame", "left_arm_base_link"),
    "left_wrist": ("left_camera_rgb", "left_arm_link6"),
    "right_wrist": ("right_camera_rgb", "right_arm_link6"),
}


def _ref(artifact_id: str, artifact_kind: str, digit: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        sha256=digit * 64,
        locator=f"artifact://{artifact_id}",
    )


def _manifest() -> Real2SimArtifactManifest:
    timing = ResolvedTiming.from_timing_spec(SimulationContract.load(CONTRACT_PATH).timing)
    contract = _ref("sim-contract", "simulation_contract", "0")
    arm_trace = _ref("arm-step-heldout", "real_trace", "1")
    position_profile = _ref("arm-position-profile", "actuation_profile", "2")
    camera_calibration = _ref("wrist-camera-calibration", "calibration", "3")
    return Real2SimArtifactManifest(
        name="rm65-bimanual-real2sim-test",
        simulation_contract=contract,
        resolved_timing=timing,
        dependency_artifacts=(arm_trace, position_profile, camera_calibration),
        fidelity_claims=(
            FidelityClaim(
                component="arm.position_response",
                status="identified",
                evidence_artifact_ids=("arm-step-heldout", "arm-position-profile"),
            ),
            FidelityClaim(
                component="camera.photometry",
                status="provisional",
                evidence_artifact_ids=("wrist-camera-calibration",),
                limitations=("Exposure and white balance have not been identified.",),
            ),
        ),
        validation_evidence=(
            ValidationEvidence(
                validation_id="arm-step-heldout-v1",
                split="held_out",
                status="passed",
                metrics={"joint_rmse_rad": 0.012, "p95_delay_error_sec": 0.004},
                evidence_artifact_ids=("arm-step-heldout",),
            ),
        ),
        code_revision="test-revision",
        environment_fingerprint={"python": "3.10", "mujoco": "test-version"},
        metadata={"owner": "sim", "tags": ["unit", "real2sim"]},
    )


def test_real2sim_manifest_round_trip_is_content_addressed_and_immutable() -> None:
    manifest = _manifest()
    payload = manifest.to_dict()

    restored = Real2SimArtifactManifest.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.artifact_sha256 == payload["artifact_sha256"]
    assert restored.resolved_timing.command_hz == 120.0
    with pytest.raises(TypeError):
        restored.environment_fingerprint["python"] = "changed"
    with pytest.raises(TypeError):
        restored.metadata["owner"] = "mainline"


def test_real2sim_manifest_rejects_tampering_and_unknown_evidence() -> None:
    payload = _manifest().to_dict()
    payload["code_revision"] = "tampered"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        Real2SimArtifactManifest.from_dict(payload)

    with pytest.raises(ValueError, match="unknown artifact IDs"):
        Real2SimArtifactManifest(
            name="unknown-evidence",
            simulation_contract=_ref("sim-contract", "simulation_contract", "0"),
            resolved_timing=_manifest().resolved_timing,
            dependency_artifacts=(),
            fidelity_claims=(
                FidelityClaim(
                    component="contact.friction",
                    status="identified",
                    evidence_artifact_ids=("missing-trace",),
                ),
            ),
            validation_evidence=(),
            code_revision="test-revision",
            environment_fingerprint={"python": "3.10"},
        )


def test_fidelity_claims_cannot_silently_promote_provisional_models() -> None:
    with pytest.raises(ValueError, match="require evidence_artifact_ids"):
        FidelityClaim(component="arm.position_response", status="identified")
    with pytest.raises(ValueError, match="must state at least one limitation"):
        FidelityClaim(component="contact.friction", status="provisional")


def test_fit_and_held_out_validation_cannot_share_a_real_trace() -> None:
    timing = _manifest().resolved_timing
    trace = _ref("shared-real-trace", "real_trace", "4")
    validation_kwargs = {
        "status": "passed",
        "metrics": {"rmse": 0.1},
        "evidence_artifact_ids": (trace.artifact_id,),
    }
    with pytest.raises(ValueError, match="different real_trace artifacts"):
        Real2SimArtifactManifest(
            name="leaked-validation",
            simulation_contract=_ref("sim-contract", "simulation_contract", "0"),
            resolved_timing=timing,
            dependency_artifacts=(trace,),
            fidelity_claims=(
                FidelityClaim(
                    component="arm.position_response",
                    status="identified",
                    evidence_artifact_ids=(trace.artifact_id,),
                ),
            ),
            validation_evidence=(
                ValidationEvidence(
                    validation_id="fit",
                    split="fit",
                    **validation_kwargs,
                ),
                ValidationEvidence(
                    validation_id="held-out",
                    split="held_out",
                    **validation_kwargs,
                ),
            ),
            code_revision="test-revision",
            environment_fingerprint={"python": "3.10"},
        )


def test_artifact_ref_hashes_the_exact_file_content() -> None:
    artifact = ArtifactRef.from_file(
        CONTRACT_PATH,
        artifact_id="sim-contract-file",
        artifact_kind="simulation_contract",
    )
    assert len(artifact.sha256) == 64
    assert artifact.locator == str(CONTRACT_PATH)
    assert (
        Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
        == _manifest().resolved_timing
    )


def test_v4_arm_fit_baseline_remains_frozen_and_provisional() -> None:
    manifest = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    artifacts = {
        artifact.artifact_id: artifact
        for artifact in manifest.dependency_artifacts
    }

    assert manifest.simulation_contract.sha256 == (
        "baa5efcd490f27b0d2e261ac2e5c91181a0618de75596017a05714a02333a0df"
    )
    assert manifest.artifact_sha256 == (
        "501702e3b741096fc3038b082405073bad51f63626b7296b75d2847e85df778d"
    )
    assert artifacts["rm65-arm-profile-source-20260721-v4"].sha256 == (
        "551b8a832f97d6e4fdfaef96b0cd3ff8b7dee46f4610ac948e23ff13602c48de"
    )
    assert artifacts["rm65-arm-actuator-core-source-20260721-v4"].sha256 == (
        "f37393bd9d0b46ca2810967a0ccdda717feb47a558219da0ccfa5371b1090f16"
    )
    assert artifacts["rm65-arm-mujoco-adapter-source-20260721-v4"].sha256 == (
        "214d47466cb37bf90fbfd9fe95c2ae46002e38afda3d16fde8356ad29308edd0"
    )
    assert artifacts[
        "rm65-arm-all-valid-fit-source-index-20260721-v4"
    ].sha256 == (
        "d52c53279d8b48bbae3413fbea0e5294971b30493dabe579d4a27dcf791e3cdf"
    )
    assert manifest.fidelity_claims[0].status == "provisional"
    held_out = next(
        evidence
        for evidence in manifest.validation_evidence
        if evidence.split == "held_out"
    )
    assert held_out.status == "not_run"
    fit = next(
        evidence for evidence in manifest.validation_evidence if evidence.split == "fit"
    )
    assert fit.metrics["session_count"] == 14.0
    assert fit.metrics["v4_weighted_mean_per_joint_qpos_rmse_deg"] < (
        fit.metrics["v3_weighted_mean_per_joint_qpos_rmse_deg"]
    )
    assert fit.metrics["v4_max_joint_qpos_rmse_deg"] > (
        fit.metrics["v3_max_joint_qpos_rmse_deg"]
    )


def test_current_arm_runtime_binding_uses_exact_current_inputs() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_PATH)
    previous_path = Path(
        "sim/rm65/real2sim/artifacts/"
        "rm65_arm_free_motion_20260721_v4_runtime_binding_20261103_real_failure_diagnostics_v1.json"
    )
    previous = Real2SimArtifactManifest.load(previous_path)
    source = "12de49816b45fee70d8671b47092099789d71f79"
    source_tree = "241427e48bb58ab3708413505e01a9f8e5953cb0"
    tag, old_tag = "20261103-mujoco-oracle-array-v1", "20261103-real-failure-diagnostics-v1"
    current = {r.locator: r for r in binding.dependency_artifacts if "-runtime-" + tag in r.artifact_id}
    inherited = {r.locator: r for r in previous.dependency_artifacts if "-runtime-" + old_tag in r.artifact_id}
    assert len(current) == len(inherited) == 205 and set(current) == set(inherited)
    assert sum(r.locator.endswith(".py") for r in current.values()) == 200
    for path, ref in current.items():
        assert ref.sha256 == ArtifactRef.from_file(path, artifact_id="current-check", artifact_kind=ref.artifact_kind).sha256
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
        assert ref.artifact_kind == inherited[path].artifact_kind
    changed = {p for p in inherited if current[p].sha256 != inherited[p].sha256}
    assert changed == {'sim/mujoco/collision_safety/oracle.py'}
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["new_runtime_sources"]) == ()
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == ()
    assert binding.code_revision == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_commit"] == source
    assert binding.metadata["offline_validation_source_tree"] == binding.environment_fingerprint["source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    refs = {r.artifact_id: r for r in binding.dependency_artifacts}
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    parent = refs["rm65-arm-runtime-binding-real-failure-diagnostics-20261103-v1"]
    assert parent.locator == str(previous_path)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"] == hashlib.sha256(previous_path.read_bytes()).hexdigest()
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.resolved_timing == previous.resolved_timing
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    for artifact_id in binding.metadata["offline_query_input_artifact_ids"]:
        ref = refs[artifact_id]
        assert ref.sha256 == ArtifactRef.from_file(ref.locator, artifact_id="offline-check", artifact_kind=ref.artifact_kind).sha256
    assert binding.metadata["migration"] == "namespaced_mujoco_oracle_array_binding_v1"
    assert binding.metadata["runtime_behavior_change_scope"] == "equivalent_collision_oracle_array_selection_and_layout_only"
    for key in ("runtime_behavior_change", "arm_profile_refit", "new_physical_hardware_validation",
                "runtime_control_semantics_changed", "wire_schema_version_changed", "automatic_retry_authorization"):
        assert binding.metadata[key] is False
    evidence = next(e for e in binding.validation_evidence if "passed_cases" in e.metrics)
    assert evidence.metrics["passed_cases"] == 47
    assert evidence.metrics["failed_cases"] == evidence.metrics["error_cases"] == 0
    assert len(binding.metadata["offline_validation_report_artifact_ids"]) == 5
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"



def test_mujoco_closure_entry_binding_remains_frozen_after_parameter_feedback() -> None:
    binding = Real2SimArtifactManifest.load(MUJOCO_CLOSURE_ENTRY_V1_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(MUJOCO_PLACE_SLOPE_V1_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261103-mujoco-closure-entry-v1", "20261101-mujoco-place-slope-v1"
    source = "7da5e97728f055f5d0a9de0a55eb1b169a58a3dd"
    source_tree = "692f45f3355be0dd22e4f91192ca3717ff6c75c8"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert extras == ()
    assert len(inherited) == 201 and len(current) == 201
    assert set(current) == set(inherited) | set(extras)
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"mujoco-place-slope-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == "runtime_source"
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {'agent_gateway/profiles/mujoco/grasp_approach_query.py'}
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == ()
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["mainline_input_commit"] == "ca545b971204e4d7d2e284dfc246481918a0fdea"
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        MUJOCO_PLACE_SLOPE_V1_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    parent = refs["rm65-arm-runtime-binding-mujoco-place-slope-20261101-v1"]
    assert parent.locator == str(MUJOCO_PLACE_SLOPE_V1_BINDING_PATH)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    # Every non-runtime predecessor reference is inherited unchanged.
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    assert binding.resolved_timing == previous.resolved_timing
    assert binding.simulation_contract.sha256 == hashlib.sha256(
        subprocess.check_output(["git", "show", source + ":" + binding.simulation_contract.locator])).hexdigest()
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    # Pre-contact closure veto; inherited slope behavior remains unchanged.
    assert binding.metadata['runtime_behavior_change'] is True
    assert binding.metadata['runtime_behavior_change_scope'] == 'default_off_mujoco_grasp_contact_planning_and_precontact_closure_obstacle_veto'
    assert binding.metadata['migration'] == 'namespaced_mujoco_closure_entry_binding_v1'
    assert tuple(binding.metadata['grasp_approach_query_candidates']) == ('top_down', 'top_down_turned_left', 'top_down_turned_right', 'side', 'front', 'top_down_half_turn')
    assert binding.metadata['grasp_approach_query_pitched_face_grasps'] is False
    assert binding.metadata['grasp_approach_query_front_candidate'] == 'level_front_approach_face_grasp_after_side_before_half_turn'
    assert binding.metadata['mujoco_support_release_v1_changed'] is False
    assert binding.metadata['mujoco_support_release_query_version'] == '1'
    assert binding.metadata['support_release_inclined_measurement'] == 'depth_only_certified_inclined_rectangle_v1'
    assert binding.metadata['support_release_inclined_measurement_opt_in'] == 'measure_support_tops_allow_inclined_default_false'
    assert binding.metadata['support_release_inclined_maximum_tilt_deg'] == 30.0
    assert binding.metadata['support_release_inclined_plane_fit'] == 'whole_mask_trimmed_least_squares_three_refits_nearest_70_percent_inliers_within_2mm_at_least_60_percent'
    assert binding.metadata['support_release_inclined_top_height'] == 'plane_height_at_region_center_plus_2mm_residual'
    assert binding.metadata['support_release_leveled_pass'] == 'first_on_an_inclined_region_held_yaw_samples_never_leaned'
    assert binding.metadata['support_release_leveling_maximum_rad'] == 0.6981317007977318
    assert binding.metadata['support_surface_observation_default_maximum_tilt_deg'] == 1.0
    assert binding.metadata['support_surface_observation_maximum_tilt_deg'] == 30.0
    assert binding.metadata['scene_contract_fixed_objects'] == 'free_body_object_spec_fixed_flag_serialized_only_when_true_world_welded_without_free_joint'
    assert binding.metadata['mujoco_place_slope_evidence'] == 'S5 native slope chains 12/12 placed stably on the final suites (toward 6, side 6); S7 DeepSeek Agent cohorts on 116 at lower 833835014: toward 18/18 and side 18/18 placed on the board, 28 finish(success) admitted, 0 admitted without a graded success'
    assert binding.metadata['predecessor_qualification'] == 'development ca545b971204e4d7d2e284dfc246481918a0fdea contains the frozen slope binding and the core-check retirement. This binding adds only the current MuJoCo closure-entry source change and its own offline evidence; inherited Agent cohort reports do not qualify this candidate.'
    # Gates and the real place path stay as they were.
    assert (binding.metadata["hard_penetration_gate_changed"], binding.metadata["observation_gap_gate_changed"],
            binding.metadata["release_window_changed"], binding.metadata["noncontact_clearance_gate_changed"]) == (
        False, False, False, False)
    assert binding.metadata["mujoco_object_motion_slip_gate_reference"] == "frozen_lift_relation_3mm_2deg_unchanged"
    assert binding.metadata["support_release_top_height"] == previous.metadata["support_release_top_height"]
    assert binding.metadata["support_release_measurement"] == previous.metadata["support_release_measurement"]
    assert binding.metadata["mujoco_agent_place_skills_default_enabled"] is False
    for key in ("real_support_release_query_version", "real_support_release_maximum_tilt_deg",
                "real_support_release_default_enabled", "real_held_object_gripper_geometry",
                "real_held_object_motion_default_enabled", "new_physical_hardware_validation"):
        assert binding.metadata[key] == previous.metadata[key]
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    for index, path in enumerate(expected_offline):
        ref = refs[f"mujoco-closure-entry-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path and old_ref.artifact_kind == ref.artifact_kind
        assert (old_ref.sha256 != ref.sha256) is (path in binding.metadata["changed_previous_offline_inputs"])
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"mujoco-closure-entry-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    reports = {
        'mujoco-closure-entry-focused-report-20261103-mujoco-closure-entry-v1': '36e5a36a5676beda940a3f6aa721296cdcfdb08b2e21fee03ceb250d1598ce18',
        'mujoco-closure-entry-focused-binding-20261103-mujoco-closure-entry-v1': '61dc89e49a10e5c2f1492c81b5e3005b7e950b3546c969d64bd91ce718d12d57',
        'mujoco-closure-entry-native-report-20261103-mujoco-closure-entry-v1': 'b74f520d30eaae36dcb4eea8381cb0e1004d0e247188b3c3a5a5278a6384dc7c',
        'mujoco-closure-entry-review-report-20261103-mujoco-closure-entry-v1': 'f85c6c945430bb9ed8ba1ad2c019f1052ab2bd5aa7688502cb8d9c563eafe5cd',
    }
    assert tuple(binding.metadata["offline_validation_report_artifact_ids"]) == tuple(reports)
    for artifact_id, digest in reports.items():
        assert refs[artifact_id].sha256 == digest and refs[artifact_id].artifact_kind == "offline_validation_report"
    assert binding.metadata["offline_validation_report_artifact_id"] == "mujoco-closure-entry-focused-report-" + tag
    offline = next(e for e in binding.validation_evidence if e.validation_id == "mujoco-closure-entry-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {'passed_cases': 72.0, 'skipped_cases': 0.0, 'failed_cases': 0.0, 'error_cases': 0.0}
    integrity = next(e for e in binding.validation_evidence
                     if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert dict(integrity.metrics) == {'bound_runtime_input_file_count': 201.0, 'bound_runtime_source_file_count': 196.0, 'changed_runtime_source_file_count_from_previous_binding': 1.0, 'new_runtime_source_file_count': 0.0, 'new_runtime_configuration_file_count': 0.0, 'arm_profile_refit_count': 0.0, 'mutated_historical_artifact_count': 0.0}
    assert binding.artifact_sha256 == "1b272b35f106ba5fcce47325a099d05cdd1ed7828911342087e0fc2f848f3022"
    assert ArtifactRef.from_file(MUJOCO_CLOSURE_ENTRY_V1_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "68802b54b941802fe741991400acd4bc28a8546c8a6b03233c5c0ee7eb7e25ee"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_real_hardware_connection_v1_binding_remains_frozen_after_mujoco_place_slope() -> None:
    binding = Real2SimArtifactManifest.load(REAL_HARDWARE_CONNECTION_V1_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(REAL_HELD_CLOSURE_V1_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261101-real-hardware-connection-v1", "20261101-real-held-closure-v1"
    source = "f99ab7c8a3f3975f90219f0fad17cec85aaed22b"
    source_tree = "0e826b1ec0965026032af2ae3f6df3b77d17ea07"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert extras == (
        "agent_gateway/profiles/real/hardware_connection.py",
    )
    assert len(current) == 199 and len(inherited) == 198 and set(current) == set(inherited) | set(extras)
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"real-hardware-connection-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == ("runtime_source" if path.endswith(".py") else "runtime_configuration")
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {
        "agent_gateway/profiles/real/gateway_cli.py",
        "agent_gateway/profiles/real/realsense_perception.py",
        "agent_gateway/profiles/real/rm65_composition.py",
    }
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == ()
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["mainline_input_commit"] == "d92d01f3e287803163e8060ae11bec8d7148fdfc"
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        REAL_HELD_CLOSURE_V1_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    parent = refs["rm65-arm-runtime-binding-real-held-closure-20261101-v1"]
    assert parent.locator == str(REAL_HELD_CLOSURE_V1_BINDING_PATH)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    # Every non-runtime predecessor reference is inherited unchanged.
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "real_operator_hardware_connection_and_gripper_control"
    assert binding.metadata["migration"] == "namespaced_real_hardware_connection_binding_v1"
    # Operator-only hardware generations and gripper open/close, off unless the CLI flag is given.
    assert binding.metadata["real_hardware_connection_control_flag"] == "--enable-hardware-connection-control"
    assert binding.metadata["real_hardware_connection_default_enabled"] is False
    assert binding.metadata["real_hardware_connection_schema"] == "agent_gateway.hardware_connection.v1"
    assert tuple(binding.metadata["real_hardware_connection_components"]) == (
        "left_arm", "right_arm", "left_gripper", "right_gripper")
    assert tuple(binding.metadata["real_hardware_connection_retained_resources"]) == ("cameras", "perception_models")
    assert binding.metadata["real_operator_gripper_schema"] == "agent_gateway.operator_gripper.v1"
    assert binding.metadata["real_operator_gripper_timeout_sec"] == 5.0
    assert binding.metadata["real_hardware_connection_source_lineage"] == (
        "agent-safe-move-hardware-connection-production-final-20261024 da450b2a6 998359d5c 36a39440e")
    # Inherited unchanged.
    assert binding.metadata["real_held_object_primary_view"] == "holding_arm_wrist_camera"
    assert binding.metadata["real_held_object_grip_gate"] == previous.metadata["real_held_object_grip_gate"]
    assert binding.metadata["real_held_object_gripper_geometry"] == previous.metadata["real_held_object_gripper_geometry"]
    assert binding.metadata["held_object_measured_box_semantics"] == "agenticrobot.held_object.measured_box.v2"
    assert binding.metadata["real_held_object_motion_default_enabled"] is False
    assert binding.metadata["real_support_release_query_version"] == "2"
    assert binding.metadata["real_support_release_default_enabled"] is False
    assert tuple(binding.metadata["rm65_6f_scene_profiles"]) == (
        "dual_arm_hand_proxy_dmtac_strict_rm65_6f", "dual_arm_hand_proxy_dmtac_grasp_rm65_6f")
    registry = "robot_motion/configs/rm65_camera_registry_v5_rm65_6f_20261031.json"
    assert binding.metadata["camera_registry_v5_sha256"] == current[registry].sha256 == inherited[registry].sha256
    assert binding.metadata["head_camera_depth_scale_correction"] == 1.01406
    assert binding.metadata["new_physical_hardware_validation"] is False
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    for index, path in enumerate(expected_offline):
        ref = refs[f"real-hardware-connection-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path and old_ref.artifact_kind == ref.artifact_kind
        assert (old_ref.sha256 != ref.sha256) is (path in binding.metadata["changed_previous_offline_inputs"])
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"real-hardware-connection-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    reports = {
        "real-hardware-connection-focused-report-" + tag: "8e84d282d6686452c9a4c60f749da32f785b40f239d600c657f9cf433c11604f",
        "real-hardware-connection-focused-binding-" + tag: "98bb7dcbe7ed3d7fbd431e7b658aaba1b60e5e0eead82e22e94f2bc4d5831971",
        "real-hardware-connection-design-report-" + tag: "180690cfebcfdbc2b1a92e8eef93870c0a10aed774a4522a3f9ad48d9ddb07b3",
    }
    assert tuple(binding.metadata["offline_validation_report_artifact_ids"]) == tuple(reports)
    for artifact_id, digest in reports.items():
        assert refs[artifact_id].sha256 == digest and refs[artifact_id].artifact_kind == "offline_validation_report"
    assert binding.metadata["offline_validation_report_artifact_id"] == "real-hardware-connection-focused-report-" + tag
    offline = next(e for e in binding.validation_evidence if e.validation_id == "real-hardware-connection-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 1353.0, "skipped_cases": 1.0, "failed_cases": 0., "error_cases": 0.}
    integrity = next(e for e in binding.validation_evidence
                     if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert dict(integrity.metrics) == {
        "bound_runtime_input_file_count": 199.0, "bound_runtime_source_file_count": 194.0,
        "changed_runtime_source_file_count_from_previous_binding": 3.0, "new_runtime_source_file_count": 1.0,
        "new_runtime_configuration_file_count": 0.0, "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    assert binding.artifact_sha256 == "46e1ee6dde46ba5ed2bb5353bbe81aab7da65cbfee4e3b3ea95c10653f941e41"
    assert ArtifactRef.from_file(REAL_HARDWARE_CONNECTION_V1_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "bbad6ea8aec425a8790d4d440d7c31c0e411800f5a6aafaf53808ff8934cecf8"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_real_held_closure_v1_binding_remains_frozen_after_real_hardware_connection() -> None:
    binding = Real2SimArtifactManifest.load(REAL_HELD_CLOSURE_V1_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(REAL_SMALL_SUPPORT_V1_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261101-real-held-closure-v1", "20261101-real-small-support-v1"
    source = "1664dba7fd0bc41c8a1c0dc583e55b28d072eb48"
    source_tree = "865bb89cb6f65094ab190733f7385447c5e69eea"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    assert tuple(binding.metadata["new_runtime_sources"]) == ()
    assert len(current) == len(inherited) == 198 and set(current) == set(inherited)
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
        assert ref.artifact_kind == inherited[path].artifact_kind
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {
        "agent_gateway/profiles/real/held_object.py",
        "robot_motion/held_object_measurement.py",
    }
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == ()
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["mainline_input_commit"] == "d95a133742209a71c4b214814a1e1408ac4636a5"
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        REAL_SMALL_SUPPORT_V1_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    parent = refs["rm65-arm-runtime-binding-real-small-support-20261101-v1"]
    assert parent.locator == str(REAL_SMALL_SUPPORT_V1_BINDING_PATH)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    # Every non-runtime predecessor reference is inherited unchanged.
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "real_held_object_gripper_geometry_at_measured_dmtac_closure"
    assert binding.metadata["migration"] == "namespaced_real_held_closure_binding_v1"
    # Held-object measurement with the fingers at the measured DMTAC closure.
    assert binding.metadata["real_held_object_gripper_geometry"] == (
        "fingers_contact_solids_and_closing_region_at_measured_dmtac_opening")
    assert binding.metadata["real_held_object_opening_model"] == "linear_from_maximum_opening_at_closure_0_to_closed_at_closure_1"
    assert (binding.metadata["real_held_object_maximum_opening_m"], binding.metadata["real_held_object_minimum_opening_m"]) == (
        0.0705, 0.001)
    assert binding.metadata["real_held_object_without_gripper_reader"] == "open_envelope_geometry_unchanged"
    assert binding.metadata["real_held_object_gripper_sample_order"] == "read_before_views"
    # Inherited unchanged.
    assert binding.metadata["real_held_object_primary_view"] == "holding_arm_wrist_camera"
    assert binding.metadata["real_held_object_grip_gate"] == previous.metadata["real_held_object_grip_gate"]
    assert binding.metadata["held_object_measured_box_semantics"] == "agenticrobot.held_object.measured_box.v2"
    assert binding.metadata["real_held_object_motion_default_enabled"] is False
    assert binding.metadata["real_support_release_query_version"] == "2"
    assert binding.metadata["real_support_release_default_enabled"] is False
    assert tuple(binding.metadata["rm65_6f_scene_profiles"]) == (
        "dual_arm_hand_proxy_dmtac_strict_rm65_6f", "dual_arm_hand_proxy_dmtac_grasp_rm65_6f")
    registry = "robot_motion/configs/rm65_camera_registry_v5_rm65_6f_20261031.json"
    assert binding.metadata["camera_registry_v5_sha256"] == current[registry].sha256 == inherited[registry].sha256
    assert binding.metadata["head_camera_depth_scale_correction"] == 1.01406
    assert binding.metadata["new_physical_hardware_validation"] is False
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    for index, path in enumerate(expected_offline):
        ref = refs[f"real-held-closure-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path and old_ref.artifact_kind == ref.artifact_kind
        assert (old_ref.sha256 != ref.sha256) is (path in binding.metadata["changed_previous_offline_inputs"])
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"real-held-closure-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    reports = {
        "real-held-closure-focused-report-" + tag: "7393edabe47bfc6d639aa7546b52495feadbbaa398e7162a8c0580bba95555e6",
        "real-held-closure-focused-binding-" + tag: "b9b78812f1689cb9471f38ed21b21c11141dcd453dc1fedc9e2bf37caa810e39",
        "real-held-closure-offline-samples-report-" + tag: "8818f523b7c0c1c179ad6aaf084d7b27c62ccdccd654b4ad16af5ab9be93556d",
        "real-held-closure-held-sample-states-" + tag: "ba469bb83071f0eff3cae4e566a2637609ac2e353157fbcec31b967b1bee16d8",
        "real-held-closure-held-sample-states-before-" + tag: "47582601bc464d55978d3886afdffd2ffedb26011e73dd334556a7a7a8c9840c",
        "real-held-closure-saved-scene-regression-" + tag: "eb1f2a1829a711ea69640439d55a3f5acd4031ccc56cc17e51a961cf5a096358",
    }
    assert tuple(binding.metadata["offline_validation_report_artifact_ids"]) == tuple(reports)
    for artifact_id, digest in reports.items():
        assert refs[artifact_id].sha256 == digest and refs[artifact_id].artifact_kind == "offline_validation_report"
    assert binding.metadata["offline_validation_report_artifact_id"] == "real-held-closure-focused-report-" + tag
    offline = next(e for e in binding.validation_evidence if e.validation_id == "real-held-closure-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 1412.0, "skipped_cases": 0.0, "failed_cases": 0., "error_cases": 0.}
    integrity = next(e for e in binding.validation_evidence
                     if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert dict(integrity.metrics) == {
        "bound_runtime_input_file_count": 198.0, "bound_runtime_source_file_count": 193.0,
        "changed_runtime_source_file_count_from_previous_binding": 2.0, "new_runtime_source_file_count": 0.,
        "new_runtime_configuration_file_count": 0., "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    assert binding.artifact_sha256 == "ca751d10cb36d55b8689fd5d28b9583e3b3def67e29df190d05ef60b19c5ba34"
    assert ArtifactRef.from_file(REAL_HELD_CLOSURE_V1_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "d4303a983dc13c9698f812d680f386c0fc9d54c246804a305240fff297ed85bf"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_real_small_support_v1_binding_remains_frozen_after_real_held_closure() -> None:
    binding = Real2SimArtifactManifest.load(REAL_SMALL_SUPPORT_V1_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(REAL_SUPPORT_RELEASE_V1_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261101-real-small-support-v1", "20261101-real-support-release-v1"
    source = "43f134f953396ba755ebe37402dba2d1da5f9d32"
    source_tree = "998efc7bc4a8d20afcf7a4c6d58bcd3a48aff29e"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert extras == ()
    assert len(current) == 198 and len(inherited) == 198 and set(current) == set(inherited) | set(extras)
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"real-small-support-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == ("runtime_source" if path.endswith(".py") else "runtime_configuration")
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {
        "agent_gateway/profiles/real/object_motion.py",
        "agent_gateway/profiles/real/support_release_query.py",
    }
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == ()
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["mainline_input_commit"] == "dfbd54b9896eaa93798b069764d64778ce75082d"
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        REAL_SUPPORT_RELEASE_V1_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    parent = refs["rm65-arm-runtime-binding-real-support-release-20261101-v1"]
    assert parent.locator == str(REAL_SUPPORT_RELEASE_V1_BINDING_PATH)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    # Every non-runtime predecessor reference is inherited unchanged.
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "real_place_small_support_window"
    assert binding.metadata["migration"] == "namespaced_real_small_support_binding_v1"
    # The real place's support windows and the release query's matching margin.
    assert tuple(binding.metadata["real_place_support_windows_m"]) == (0.04, 0.01)
    assert binding.metadata["real_place_support_window_order"] == "footprint_plus_4cm_then_footprint_plus_1cm"
    assert binding.metadata["real_place_support_dropoff_m"] == 0.015
    assert tuple(binding.metadata["real_place_support_dropoff_allowance"]) == (20, 0.01)
    assert binding.metadata["real_place_gap_check_changed"] is False
    assert binding.metadata["real_place_planner_support_allowance_changed"] is False
    assert binding.metadata["real_lift_completion_changed"] is False
    assert binding.metadata["real_support_release_lateral_margin_m"] == 0.0075
    # Inherited from real-support-release v1 unchanged.
    assert binding.metadata["real_support_release_query_version"] == "2"
    assert binding.metadata["real_support_release_default_enabled"] is False
    assert binding.metadata["real_support_release_dispatches_motion"] is False
    assert tuple(binding.metadata["rm65_6f_scene_profiles"]) == (
        "dual_arm_hand_proxy_dmtac_strict_rm65_6f", "dual_arm_hand_proxy_dmtac_grasp_rm65_6f")
    assert binding.metadata["camera_registry_v5_sha256"] == previous.metadata["camera_registry_v5_sha256"]
    assert binding.metadata["mujoco_agent_place_skills_default_enabled"] is False
    assert binding.metadata["real_held_object_motion_default_enabled"] is False
    assert binding.metadata["new_physical_hardware_validation"] is False
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    for index, path in enumerate(expected_offline):
        ref = refs[f"real-small-support-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path and old_ref.artifact_kind == ref.artifact_kind
        assert (old_ref.sha256 != ref.sha256) is (path in binding.metadata["changed_previous_offline_inputs"])
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"real-small-support-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    reports = {
        "real-small-support-focused-report-" + tag: "403d8bdc584a544b613a7291ebe287a7de2c7ef7f43d1064726418117cc73cd9",
        "real-small-support-focused-binding-" + tag: "3558058787dafb407a88f414a038b93da0992ed1503dc893f9e5fff49e07b017",
        "real-small-support-design-report-" + tag: "99c831938177203fc90636a2d1d6bc493b35399eff1f0c9a9e5047dc657bae9b",
        "real-small-support-place-lifecycle-smoke-" + tag: "b2fd84b6bcb04bdf7609894b81baf2f356e10a72660662dba6314df5505abc33",
    }
    assert tuple(binding.metadata["offline_validation_report_artifact_ids"]) == tuple(reports)
    for artifact_id, digest in reports.items():
        assert refs[artifact_id].sha256 == digest and refs[artifact_id].artifact_kind == "offline_validation_report"
    assert binding.metadata["offline_validation_report_artifact_id"] == "real-small-support-focused-report-" + tag
    offline = next(e for e in binding.validation_evidence if e.validation_id == "real-small-support-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 1168.0, "skipped_cases": 0.0, "failed_cases": 0., "error_cases": 0.}
    integrity = next(e for e in binding.validation_evidence
                     if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert dict(integrity.metrics) == {
        "bound_runtime_input_file_count": 198.0, "bound_runtime_source_file_count": 193.0,
        "changed_runtime_source_file_count_from_previous_binding": 2.0, "new_runtime_source_file_count": 0.0,
        "new_runtime_configuration_file_count": 0.0, "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    assert binding.artifact_sha256 == "3161b284eccf2a0017dd7fd7a94a71cf5b45d6b3ead8634fdfab246abdea036d"
    assert ArtifactRef.from_file(REAL_SMALL_SUPPORT_V1_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "ecf08c532e2aa11a76140821eff253f93a6938e2304997af99941d611bf97907"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_real_support_release_v1_binding_remains_frozen_after_real_small_support() -> None:
    binding = Real2SimArtifactManifest.load(REAL_SUPPORT_RELEASE_V1_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(REAL_RM65_6F_V1_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261101-real-support-release-v1", "20261031-real-rm65-6f-v1"
    source = "b9091bf5ecc8f4024f8a09034de717e92dae84ba"
    source_tree = "9ce324cf9650c7ed1cde31c37c81c8a38b15e191"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert extras == (
        "agent_gateway/profiles/real/support_release_query.py",
        "agent_gateway/skills/perception/support_release_query_v2.py",
    )
    assert len(current) == 198 and len(inherited) == 196 and set(current) == set(inherited) | set(extras)
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"real-support-release-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == ("runtime_source" if path.endswith(".py") else "runtime_configuration")
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {
        "agent_gateway/profiles/real/exclusive_factory.py",
        "agent_gateway/profiles/real/gateway_cli.py",
        "agent_gateway/profiles/real/motion_gateway_session.py",
        "agent_gateway/profiles/real/motion_primitive_profiles.py",
        "agent_gateway/profiles/real/object_motion.py",
        "agent_gateway/profiles/real/rm65_composition.py",
        "agent_gateway/profiles/real/tool_entity.py",
    }
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == ()
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["mainline_input_commit"] == "be42c02d9af76f2b3eab0582b28d9e8c951563e4"
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        REAL_RM65_6F_V1_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    parent = refs["rm65-arm-runtime-binding-real-rm65-6f-20261031-v1"]
    assert parent.locator == str(REAL_RM65_6F_V1_BINDING_PATH)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    # Every non-runtime predecessor reference is inherited unchanged.
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "real_support_release_query_v2_for_named_supports"
    assert binding.metadata["migration"] == "namespaced_real_support_release_binding_v1"
    # The real release-pose query: read-only, opt-in, RM65-6F only.
    assert binding.metadata["real_support_release_query_version"] == "2"
    assert binding.metadata["real_support_release_semantics"] == "agenticrobot.held_object.support_release_pose.v2"
    assert binding.metadata["real_support_release_default_enabled"] is False
    assert binding.metadata["real_support_release_dispatches_motion"] is False
    assert tuple(binding.metadata["real_support_release_requires"]) == (
        "real_object_motion", "rm65_6f_scene_profile", "camera_registry_v5", "holding_arm_wrist_camera")
    assert binding.metadata["real_support_release_held_object"] == "verified_grip_retained_since_its_lift"
    assert binding.metadata["real_support_release_measurement"] == "most_populated_height_band_seed_grown_within_residual"
    assert (binding.metadata["real_support_release_plane_residual_m"], binding.metadata["real_support_release_maximum_tilt_deg"],
            binding.metadata["real_support_release_grid_m"]) == (0.006, 5.0, 0.004)
    assert binding.metadata["real_support_release_preplace_height_m"] == 0.05
    assert tuple(binding.metadata["real_support_release_clearance_band_m"]) == (0.005, 0.015)
    assert binding.metadata["real_support_release_check_order"] == "held_yaw_on_every_region_then_turned_yaws_then_lean"
    assert binding.metadata["real_support_release_place_check_prediction_radius_m"] == 0.03
    assert binding.metadata["real_support_release_max_checked_candidates"] == 24
    assert binding.metadata["real_support_release_max_operation_ms"] == 120000
    assert binding.metadata["mujoco_support_release_v1_changed"] is False
    # Inherited from real-rm65-6f v1 unchanged.
    assert tuple(binding.metadata["rm65_6f_scene_profiles"]) == (
        "dual_arm_hand_proxy_dmtac_strict_rm65_6f", "dual_arm_hand_proxy_dmtac_grasp_rm65_6f")
    assert binding.metadata["rm65_6f_planning_model_sha256"] == previous.metadata["rm65_6f_planning_model_sha256"]
    assert binding.metadata["rm65_6f_station_model_sha256"] == previous.metadata["rm65_6f_station_model_sha256"]
    registry = "robot_motion/configs/rm65_camera_registry_v5_rm65_6f_20261031.json"
    assert binding.metadata["camera_registry_v5_sha256"] == current[registry].sha256 == inherited[registry].sha256
    assert binding.metadata["head_camera_depth_scale_correction"] == 1.01406
    assert binding.metadata["table_hand_clearance_default"] == "fixed_scene_margin_unchanged"
    assert binding.metadata["real_held_object_primary_view"] == "holding_arm_wrist_camera"
    assert binding.metadata["support_release_measurement"] == "depth_only_certified_top_rectangle_v2"
    assert binding.metadata["mujoco_agent_place_skills_default_enabled"] is False
    assert binding.metadata["real_held_object_motion_default_enabled"] is False
    assert binding.metadata["new_physical_hardware_validation"] is False
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    for index, path in enumerate(expected_offline):
        ref = refs[f"real-support-release-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path and old_ref.artifact_kind == ref.artifact_kind
        assert (old_ref.sha256 != ref.sha256) is (path in binding.metadata["changed_previous_offline_inputs"])
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"real-support-release-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    reports = {
        "real-support-release-focused-report-" + tag: "cd37a149d0edd0f4b3c398da4e28c5d88d5e3f122569c612eb4c7cfc83bd064d",
        "real-support-release-focused-binding-" + tag: "6ddb068d94d136ef68cada8e5e270ac5fb5fb8c220a70de2ebea0b78b8fa5ca4",
        "real-support-release-offline-samples-report-" + tag: "149f93725f62763612781b3ce022c95e8283c14eb4c366d8f24761fc1ae65c3f",
        "real-support-release-prompt-survey-" + tag: "61e1c8e26a725a4c9d038f646b68d176514e08ea00567de1643ea519144b6fae",
    }
    assert tuple(binding.metadata["offline_validation_report_artifact_ids"]) == tuple(reports)
    for artifact_id, digest in reports.items():
        assert refs[artifact_id].sha256 == digest and refs[artifact_id].artifact_kind == "offline_validation_report"
    assert binding.metadata["offline_validation_report_artifact_id"] == "real-support-release-focused-report-" + tag
    offline = next(e for e in binding.validation_evidence if e.validation_id == "real-support-release-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 1278.0, "skipped_cases": 0.0, "failed_cases": 0., "error_cases": 0.}
    integrity = next(e for e in binding.validation_evidence
                     if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert dict(integrity.metrics) == {
        "bound_runtime_input_file_count": 198.0, "bound_runtime_source_file_count": 193.0,
        "changed_runtime_source_file_count_from_previous_binding": 7.0, "new_runtime_source_file_count": 2.0,
        "new_runtime_configuration_file_count": 0.0, "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    assert binding.artifact_sha256 == "940da5b0e006e771d75904cc18ccc6e45e5d5b079ce5d020b82a129964859fcb"
    assert ArtifactRef.from_file(REAL_SUPPORT_RELEASE_V1_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "649c88a34a62baca9d1157bd2cf71087f5b2d6f297bc651f5e483c34eafb283b"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_real_rm65_6f_v1_binding_remains_frozen_after_real_support_release() -> None:
    binding = Real2SimArtifactManifest.load(REAL_RM65_6F_V1_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(REAL_HELD_OWN_WRIST_V1_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261031-real-rm65-6f-v1", "20261031-real-held-own-wrist-v1"
    source = "790182d3aca8046ec2c75cfd1ead9b618ffac265"
    source_tree = "9d92ef6dd177fcc24635d2c91a8340323bb4443a"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert extras == (
        "agent_gateway/profiles/real/realsense_perception.py",
        "robot_motion/rm65/scene_profiles.py",
        "robot_motion/scene_obstacles.py",
        "robot_motion/configs/rm65_camera_registry_v5_rm65_6f_20261031.json",
    )
    assert len(current) == 196 and len(inherited) == 192 and set(current) == set(inherited) | set(extras)
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"real-rm65-6f-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == ("runtime_source" if path.endswith(".py") else "runtime_configuration")
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {
        "agent_gateway/profiles/real/gateway_cli.py",
        "robot_motion/rm65/obstacle_collision.py",
        "teleop_runtime/skill_deploy/rm65_grasp_trajectory.py",
    }
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == ()
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["mainline_input_commit"] == "eed84fb8f61e63fecc8a9d661ec64d796a9e6fb0"
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        REAL_HELD_OWN_WRIST_V1_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    parent = refs["rm65-arm-runtime-binding-real-held-own-wrist-20261031-v1"]
    assert parent.locator == str(REAL_HELD_OWN_WRIST_V1_BINDING_PATH)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    # Every non-runtime predecessor reference is inherited unchanged.
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == (
        "real_rm65_6f_models_registry_v5_depth_correction_and_optional_table_hand_clearance")
    assert binding.metadata["migration"] == "namespaced_real_rm65_6f_binding_v1"
    # RM65-6F motion and Station models (asset store, not versioned: bound by content hash here).
    assert tuple(binding.metadata["rm65_6f_scene_profiles"]) == (
        "dual_arm_hand_proxy_dmtac_strict_rm65_6f", "dual_arm_hand_proxy_dmtac_grasp_rm65_6f")
    assert binding.metadata["rm65_6f_flange_extension_m"] == 0.0172
    assert binding.metadata["rm65_default_scene_profiles_changed"] is False
    assert binding.metadata["rm65_6f_planning_model_sha256"] == (
        "a5ec008c89f454c862e857cc9122887cfa4fa2d07b8a05661c5574fc1e9208d5")
    assert binding.metadata["rm65_6f_station_model_sha256"] == (
        "dfed5e8441507e3690596e0b26d77e09ee48491928f4fdc359d240eb74d3d12c")
    assert binding.metadata["rm65_6f_planning_model_source_sha256"] == (
        "4d96173e88bd28de6f8e358e114d2b5b0d94d0e724176aef0dceca37e9622e0c")
    assert binding.metadata["rm65_6f_station_model_source_sha256"] == (
        "310c3e440d4725ab2511b7bf84b32cf8ad6cefbb90586fbac50893f95c4a1411")
    registry = "robot_motion/configs/rm65_camera_registry_v5_rm65_6f_20261031.json"
    assert binding.metadata["camera_registry_v5_sha256"] == current[registry].sha256
    assert binding.metadata["head_camera_depth_scale_correction"] == 1.01406
    assert binding.metadata["wrist_camera_depth_scale_corrections"] == "none_until_board_measurement"
    assert tuple(binding.metadata["depth_scale_correction_range"]) == (0.95, 1.05)
    assert binding.metadata["table_hand_clearance_default"] == "fixed_scene_margin_unchanged"
    assert tuple(binding.metadata["table_hand_clearance_cli_range_m"]) == (0.0, 0.02)
    assert binding.metadata["table_hand_clearance_requires_rm65_6f"] is True
    assert binding.metadata["table_hand_clearance_contact_blocked"] is True
    assert binding.metadata["observed_obstacle_block_attribution"] == "observed_voxels_named_when_blocking"
    # Inherited from real-held-own-wrist v1 unchanged.
    assert binding.metadata["real_held_object_primary_view"] == "holding_arm_wrist_camera"
    assert binding.metadata["point_prompt_mask_miss"] == "per_view_no_mask"
    assert binding.metadata["grasp_fusion_auxiliary_point_mask_miss"] == "excluded_view_target_point_mask_unavailable"
    assert binding.metadata["real_object_motion_support_view"] == (
        "holding_arm_wrist_camera_other_views_as_consistency_evidence")
    assert binding.metadata["ordinary_clearance_m"] == 0.02
    assert binding.metadata["mujoco_held_object_speed_scale"] == 0.25
    assert binding.metadata["support_release_measurement"] == "depth_only_certified_top_rectangle_v2"
    assert binding.metadata["mujoco_agent_place_skills_default_enabled"] is False
    assert binding.metadata["real_gateway_held_object_motion"] is True
    assert binding.metadata["real_held_object_motion_default_enabled"] is False
    assert binding.metadata["new_physical_hardware_validation"] is False
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    for index, path in enumerate(expected_offline):
        ref = refs[f"real-rm65-6f-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path and old_ref.artifact_kind == ref.artifact_kind
        assert (old_ref.sha256 != ref.sha256) is (path in binding.metadata["changed_previous_offline_inputs"])
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"real-rm65-6f-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    reports = {
        "real-rm65-6f-focused-report-" + tag: "89ede6359f494a65faf07f3648b356f994114cf3ac8f0d0ca0103114f6bfe998",
        "real-rm65-6f-focused-binding-" + tag: "8f1ba9b0f6fd93ec83372a29baa466bea6e8fa967fbfa565f3d84bf682daa68a",
        "real-rm65-6f-model-check-" + tag: "861f7690ad4634934af878ee0f5562ac88b0b906f373a1598e6969cea2ab16aa",
        "real-rm65-6f-table-clearance-check-" + tag: "819d1f6c90529054a850a68dbcf9db4107f4e4e1f1151457c7c42cfabbf5adb1",
        "real-rm65-6f-asset-check-" + tag: "dd6f25fdb91a780f769a056c78a02d9b9fcdc7a7771f347cb67ad60359e0f5e0",
        "real-rm65-6f-offline-samples-report-" + tag: "e2f2442d874bb3d81a9c25d5822d2df75b150b92531529a75e6b87e0a173ac4c",
    }
    assert tuple(binding.metadata["offline_validation_report_artifact_ids"]) == tuple(reports)
    for artifact_id, digest in reports.items():
        assert refs[artifact_id].sha256 == digest and refs[artifact_id].artifact_kind == "offline_validation_report"
    assert binding.metadata["offline_validation_report_artifact_id"] == "real-rm65-6f-focused-report-" + tag
    offline = next(e for e in binding.validation_evidence if e.validation_id == "real-rm65-6f-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 3683.0, "skipped_cases": 1.0, "failed_cases": 0., "error_cases": 0.}
    integrity = next(e for e in binding.validation_evidence
                     if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert dict(integrity.metrics) == {
        "bound_runtime_input_file_count": 196.0, "bound_runtime_source_file_count": 191.0,
        "changed_runtime_source_file_count_from_previous_binding": 3.0, "new_runtime_source_file_count": 3.0,
        "new_runtime_configuration_file_count": 1.0, "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    assert binding.artifact_sha256 == "042030cee98722fbbb683378a2ef3e71d8e091aa301adfe8cbd1dd54794c037b"
    assert ArtifactRef.from_file(REAL_RM65_6F_V1_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "d43886c9eb4e8c1d71fd583ab88f948c52853722625b4f35ea156c5b8b4b3104"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_real_held_own_wrist_v1_binding_remains_frozen_after_real_rm65_6f() -> None:
    binding = Real2SimArtifactManifest.load(REAL_HELD_OWN_WRIST_V1_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(MUJOCO_AGENT_PLACE_V3_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261031-real-held-own-wrist-v1", "20261031-mujoco-agent-place-v3"
    source = "b278f4532de3414a3558f0a96e622534cde93bf8"
    source_tree = "977805e0570eefcefbbc3291400bed57de287c20"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert extras == ()
    assert len(current) == 192 and len(inherited) == 192 and set(current) == set(inherited) | set(extras)
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"real-held-own-wrist-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == "runtime_source"
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {
        "agent_gateway/profiles/real/held_object.py",
        "agent_gateway/profiles/real/object_motion.py",
        "agent_gateway/profiles/segmentation.py",
        "deployments/perception/sam3_service.py",
        "robot_motion/held_object_measurement.py",
    }
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == ()
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["mainline_input_commit"] == "e00a9d9d3b32f0fbc9c79b5a89798bccf7c08694"
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        MUJOCO_AGENT_PLACE_V3_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    parent = refs["rm65-arm-runtime-binding-mujoco-agent-place-20261031-v3"]
    assert parent.locator == str(MUJOCO_AGENT_PLACE_V3_BINDING_PATH)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    # Every non-runtime predecessor reference is inherited unchanged.
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == (
        "real_held_object_own_wrist_measurement_and_point_prompt_miss_as_per_view_no_mask")
    assert binding.metadata["migration"] == "namespaced_real_held_own_wrist_binding_v1"
    # Real held object measured by the holding arm's own wrist camera; point-prompt misses.
    assert binding.metadata["real_held_object_observation_semantics"] == "agenticrobot.real.held_object_observation.v2"
    assert binding.metadata["held_object_measured_box_semantics"] == "agenticrobot.held_object.measured_box.v2"
    assert binding.metadata["real_held_object_primary_view"] == "holding_arm_wrist_camera"
    assert binding.metadata["real_held_object_seed_gripper_solid_clearance_m"] == 0.005
    assert binding.metadata["real_held_object_seed_minimum_pixels"] == 15
    assert binding.metadata["real_held_object_extension_view_depth_agreement_m"] == 0.03
    assert binding.metadata["real_held_object_extension_per_side_m"] == 0.03
    assert binding.metadata["real_object_motion_support_view"] == (
        "holding_arm_wrist_camera_other_views_as_consistency_evidence")
    assert binding.metadata["point_prompt_mask_miss"] == "per_view_no_mask"
    assert binding.metadata["grasp_fusion_auxiliary_point_mask_miss"] == "excluded_view_target_point_mask_unavailable"
    # Inherited from mujoco-agent-place v3 unchanged.
    assert binding.metadata["mujoco_held_object_speed_scale"] == 0.25
    assert binding.metadata["support_release_measurement"] == "depth_only_certified_top_rectangle_v2"
    assert (binding.metadata["hard_penetration_gate_changed"], binding.metadata["observation_gap_gate_changed"],
            binding.metadata["release_window_changed"], binding.metadata["noncontact_clearance_gate_changed"]) == (
        False, False, False, False)
    assert binding.metadata["mujoco_agent_place_skills_default_enabled"] is False
    assert binding.metadata["real_gateway_held_object_motion"] is True
    assert binding.metadata["real_held_object_motion_default_enabled"] is False
    assert binding.metadata["new_physical_hardware_validation"] is False
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    for index, path in enumerate(expected_offline):
        ref = refs[f"real-held-own-wrist-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path and old_ref.artifact_kind == ref.artifact_kind
        assert (old_ref.sha256 != ref.sha256) is (path in binding.metadata["changed_previous_offline_inputs"])
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"real-held-own-wrist-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    reports = {
        "real-held-own-wrist-focused-report-" + tag: "07ed40079902c2a14198473c6e9954a92d7e3bc435558ec51d7f818ff7b93e9c",
        "real-held-own-wrist-focused-binding-" + tag: "db548e1f19e2b9447064c25b70d386b5740ea9c0ba5651e61313a1379d62a2c0",
        "real-held-own-wrist-point-miss-check-" + tag: "64602101928f395d4dd47c1cc51be91b08f8493d8226c1ed9715c40eabfa28ea",
        "real-held-own-wrist-offline-samples-report-" + tag: "616d098c7c36128325117ed934243b185d3a3b0d240524105b9edd6621e2fad5",
    }
    assert tuple(binding.metadata["offline_validation_report_artifact_ids"]) == tuple(reports)
    for artifact_id, digest in reports.items():
        assert refs[artifact_id].sha256 == digest and refs[artifact_id].artifact_kind == "offline_validation_report"
    assert binding.metadata["offline_validation_report_artifact_id"] == "real-held-own-wrist-focused-report-" + tag
    offline = next(e for e in binding.validation_evidence if e.validation_id == "real-held-own-wrist-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 2262.0, "skipped_cases": 1.0, "failed_cases": 0., "error_cases": 0.}
    integrity = next(e for e in binding.validation_evidence
                     if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert dict(integrity.metrics) == {
        "bound_runtime_input_file_count": 192.0, "bound_runtime_source_file_count": 188.0,
        "changed_runtime_source_file_count_from_previous_binding": 5.0, "new_runtime_source_file_count": 0.0,
        "new_runtime_configuration_file_count": 0., "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    assert binding.artifact_sha256 == "bce4be51399d0901083df2f1c9413259507ce43230360cb68fe212cea218a00b"
    assert ArtifactRef.from_file(REAL_HELD_OWN_WRIST_V1_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "0df22f253ed33e456485539738a825a8f4f1e487b0a0d555044c7ba521420ba1"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_mujoco_agent_place_v3_binding_remains_frozen_after_real_held_own_wrist() -> None:
    binding = Real2SimArtifactManifest.load(MUJOCO_AGENT_PLACE_V3_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(MUJOCO_AGENT_PLACE_V2_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261031-mujoco-agent-place-v3", "20261030-mujoco-agent-place-v2"
    source = "576aacb4137c3514a762e8c5e7d7e8044d8b0b79"
    source_tree = "81e7d55886e8c6851f103b041d374ac39b34a931"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert extras == ()
    assert len(current) == 192 and len(inherited) == 192 and set(current) == set(inherited) | set(extras)
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"mujoco-agent-place-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == "runtime_source"
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {
        "agent_gateway/profiles/measured_support.py",
        "agent_gateway/profiles/mujoco/support_release_query.py",
        "agent_gateway/skills/perception/support_release_query.py",
        "robot_motion/object_transport.py",
        "sim/mujoco/deploy/planned_tcp_motion.py",
    }
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == ()
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["mainline_input_commit"] == "6deeefe32f22739917e02f51c9038505cbd7f9fc"
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        MUJOCO_AGENT_PLACE_V2_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    parent = refs["rm65-arm-runtime-binding-mujoco-agent-place-20261030-v2"]
    assert parent.locator == str(MUJOCO_AGENT_PLACE_V2_BINDING_PATH)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    # Every non-runtime predecessor reference is inherited unchanged.
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == (
        "default_off_mujoco_held_object_motion_start_speed_endpoint_settle_and_mujoco_place_queries")
    assert binding.metadata["migration"] == "namespaced_mujoco_agent_place_binding_v3"
    # Batch 2d, quarter speed and 2e.
    assert binding.metadata["mujoco_held_object_speed_scale"] == 0.25
    assert binding.metadata["support_release_nearest_to_held_candidate"] is True
    assert binding.metadata["support_release_nearest_inset_m"] == 1e-06
    assert binding.metadata["support_release_lean_rad"] == 0.10471975511965978
    assert tuple(binding.metadata["support_release_lean_yaw_offsets_rad"]) == (0.0,)
    assert binding.metadata["support_release_lean_toward"] == "motion_arm_base"
    assert binding.metadata["support_release_lean_order"] == "after_the_held_orientation_candidates_of_each_region"
    assert binding.metadata["support_release_lean_maximum_tilt_from_support_normal_rad"] == 0.17453292519943295
    assert binding.metadata["support_release_refusals"] == "typed_no_release_candidates"
    assert binding.metadata["support_release_independent_review"] == "approve_with_findings_resolved_in_eb229e6af"
    assert binding.metadata["support_release_top_height"] == "certified_plane_highest_corner_plus_2mm_residual"
    assert binding.metadata["support_release_measurement"] == "depth_only_certified_top_rectangle_v2"
    # Inherited from v2 unchanged.
    assert binding.metadata["mujoco_object_motion_start"] == "held_joint_command_when_within_tolerance_of_measured_arms"
    assert binding.metadata["mujoco_object_motion_held_command_start_tolerance_rad"] == 0.005
    assert binding.metadata["mujoco_object_motion_goal_hold_sim_sec"] == 0.5
    assert (binding.metadata["mujoco_held_object_endpoint_sag_margin_m"],
            binding.metadata["mujoco_held_object_endpoint_sag_margin_rad"]) == (0.001, 0.008726646259971648)
    assert binding.metadata["mujoco_object_motion_tcp_settle_fail_fast"] is True
    assert binding.metadata["mujoco_object_motion_terminal_grace_sim_sec"] == 1.5
    assert binding.metadata["mujoco_grasped_object_capture_reuse"] == "identical_native_state_only"
    assert binding.metadata["support_release_max_regions"] == 3
    assert tuple(binding.metadata["grasp_approach_query_candidates"]) == (
        "top_down", "top_down_turned_left", "top_down_turned_right", "side", "top_down_half_turn")
    assert binding.metadata["grasp_approach_query_pitched_face_grasps"] is False
    assert binding.metadata["grasp_approach_query_path_search_budget_sec"] == 12.0
    assert tuple(binding.metadata["mujoco_place_scene_suite_cases_added"]) == ("mat_disturbance",)
    # Gates stay as they were; the real-place claims are inherited unchanged.
    assert binding.metadata["mujoco_object_motion_slip_gate_reference"] == "frozen_lift_relation_3mm_2deg_unchanged"
    assert (binding.metadata["hard_penetration_gate_changed"], binding.metadata["observation_gap_gate_changed"],
            binding.metadata["release_window_changed"], binding.metadata["noncontact_clearance_gate_changed"]) == (
        False, False, False, False)
    assert binding.metadata["mujoco_agent_place_skills_default_enabled"] is False
    assert binding.metadata["real_gateway_held_object_motion"] is True
    assert binding.metadata["real_held_object_motion_default_enabled"] is False
    assert binding.metadata["new_physical_hardware_validation"] is False
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    for index, path in enumerate(expected_offline):
        ref = refs[f"mujoco-agent-place-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path and old_ref.artifact_kind == ref.artifact_kind
        assert (old_ref.sha256 != ref.sha256) is (path in binding.metadata["changed_previous_offline_inputs"])
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"mujoco-agent-place-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    reports = {
        "mujoco-agent-place-focused-report-" + tag: "1e8522e219a7d98f31f4573fb839837bfc249f2a00c689f762f9566b4afa1dc0",
        "mujoco-agent-place-focused-binding-" + tag: "18db0a69414fbaf2ca4054e3d91351c302faee3cd997c22d8948245f2b6dbeb0",
        "mujoco-agent-place-agent-cohort-report-" + tag: "2d9a059c88e140af80b72e916b140f2f1b5ec15a4886ea67f9a020ab569a44e3",
    }
    assert tuple(binding.metadata["offline_validation_report_artifact_ids"]) == tuple(reports)
    for artifact_id, digest in reports.items():
        assert refs[artifact_id].sha256 == digest and refs[artifact_id].artifact_kind == "offline_validation_report"
    assert binding.metadata["offline_validation_report_artifact_id"] == "mujoco-agent-place-focused-report-" + tag
    offline = next(e for e in binding.validation_evidence if e.validation_id == "mujoco-agent-place-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 875.0, "skipped_cases": 0.0, "failed_cases": 0., "error_cases": 0.}
    integrity = next(e for e in binding.validation_evidence
                     if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert dict(integrity.metrics) == {
        "bound_runtime_input_file_count": 192.0, "bound_runtime_source_file_count": 188.0,
        "changed_runtime_source_file_count_from_previous_binding": 5.0, "new_runtime_source_file_count": 0.0,
        "new_runtime_configuration_file_count": 0., "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    assert binding.artifact_sha256 == "1f458f87f1657f82a13ef241292853ac10760334a9df6b593797f0388af5bf2c"
    assert ArtifactRef.from_file(MUJOCO_AGENT_PLACE_V3_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "5e6fafc11072471deb9d70d5bb1f7e9d4128126fdaa2e63f615691253fa0449d"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_mujoco_agent_place_v2_binding_remains_frozen_after_v3() -> None:
    binding = Real2SimArtifactManifest.load(MUJOCO_AGENT_PLACE_V2_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(REAL_PLACE_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261030-mujoco-agent-place-v2", "20261030-real-place-v1"
    source = "eb44be38aa6017fc850012e26ad01823bc123c28"
    source_tree = "b742f0c3f09a24551f94a49433c440c0592616c1"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert extras == ()
    assert len(current) == 192 and len(inherited) == 192 and set(current) == set(inherited) | set(extras)
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"mujoco-agent-place-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == "runtime_source"
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {
        "agent_gateway/profiles/measured_support.py",
        "agent_gateway/profiles/mujoco/grasp_approach_query.py",
        "agent_gateway/profiles/mujoco/motion_deployment.py",
        "agent_gateway/profiles/mujoco/object_motion.py",
        "agent_gateway/profiles/mujoco/support_release_query.py",
        "agent_gateway/skills/perception/grasp_approach_query.py",
        "agent_gateway/skills/perception/support_release_query.py",
        "sim/mujoco/deploy/grasped_object.py",
        "sim/mujoco/deploy/planned_tcp_motion.py",
    }
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == ()
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["mainline_input_commit"] == "a0121721f6470c8cc2407e05bd3e41396f7724dc"
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        REAL_PLACE_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    parent = refs["rm65-arm-runtime-binding-real-place-20261030-v1"]
    assert parent.locator == str(REAL_PLACE_BINDING_PATH)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    # Every non-runtime predecessor reference is inherited unchanged.
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == (
        "default_off_mujoco_held_object_motion_start_speed_endpoint_settle_and_mujoco_place_queries")
    assert binding.metadata["mujoco_object_motion_start"] == "held_joint_command_when_within_tolerance_of_measured_arms"
    assert binding.metadata["mujoco_object_motion_held_command_start_tolerance_rad"] == 0.005
    assert binding.metadata["mujoco_held_object_speed_scale"] == 0.5
    assert binding.metadata["mujoco_object_motion_goal_hold_sim_sec"] == 0.5
    assert (binding.metadata["mujoco_held_object_endpoint_sag_margin_m"],
            binding.metadata["mujoco_held_object_endpoint_sag_margin_rad"]) == (0.001, 0.008726646259971648)
    assert binding.metadata["mujoco_object_motion_tcp_settle_fail_fast"] is True
    assert binding.metadata["mujoco_object_motion_terminal_grace_sim_sec"] == 1.5
    assert binding.metadata["mujoco_grasped_object_capture_reuse"] == "identical_native_state_only"
    assert binding.metadata["support_release_max_regions"] == 3
    assert tuple(binding.metadata["grasp_approach_query_candidates"]) == (
        "top_down", "top_down_turned_left", "top_down_turned_right", "side", "top_down_half_turn")
    assert binding.metadata["grasp_approach_query_pitched_face_grasps"] is False
    assert binding.metadata["grasp_approach_query_path_search_budget_sec"] == 12.0
    assert tuple(binding.metadata["mujoco_place_scene_suite_cases_added"]) == ("mat_disturbance",)
    # Gates stay as they were; the real-place claims are inherited unchanged.
    assert binding.metadata["mujoco_object_motion_slip_gate_reference"] == "frozen_lift_relation_3mm_2deg_unchanged"
    assert (binding.metadata["hard_penetration_gate_changed"], binding.metadata["observation_gap_gate_changed"],
            binding.metadata["release_window_changed"]) == (False, False, False)
    assert binding.metadata["mujoco_agent_place_skills_default_enabled"] is False
    assert binding.metadata["real_gateway_held_object_motion"] is True
    assert binding.metadata["real_held_object_motion_default_enabled"] is False
    assert binding.metadata["new_physical_hardware_validation"] is False
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    for index, path in enumerate(expected_offline):
        ref = refs[f"mujoco-agent-place-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path and old_ref.artifact_kind == ref.artifact_kind
        assert (old_ref.sha256 != ref.sha256) is (path in binding.metadata["changed_previous_offline_inputs"])
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"mujoco-agent-place-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    reports = {
        "mujoco-agent-place-focused-report-" + tag: "e1586262f81f1540b7f83aec08a80aaae0f63b7b4e1fff9115df5dbcc8f628b8",
        "mujoco-agent-place-focused-binding-" + tag: "d339b81adb4196f98abffd9d21e3d0b088d209a5acf0db6d1038dfc772cde828",
        "mujoco-agent-place-agent-cohort-report-" + tag: "773af23f34203e9367f13cf125e83ba0026bb241a5494e15a895703384c0a0e2",
    }
    assert tuple(binding.metadata["offline_validation_report_artifact_ids"]) == tuple(reports)
    for artifact_id, digest in reports.items():
        assert refs[artifact_id].sha256 == digest and refs[artifact_id].artifact_kind == "offline_validation_report"
    assert binding.metadata["offline_validation_report_artifact_id"] == "mujoco-agent-place-focused-report-" + tag
    offline = next(e for e in binding.validation_evidence if e.validation_id == "mujoco-agent-place-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 772.0, "skipped_cases": 0.0, "failed_cases": 0., "error_cases": 0.}
    integrity = next(e for e in binding.validation_evidence
                     if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert dict(integrity.metrics) == {
        "bound_runtime_input_file_count": 192.0, "bound_runtime_source_file_count": 188.0,
        "changed_runtime_source_file_count_from_previous_binding": 9.0, "new_runtime_source_file_count": 0.0,
        "new_runtime_configuration_file_count": 0., "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    assert binding.artifact_sha256 == "7b6e8ffbdd9fe93f54686ff6cde35a75304aa03ca774e6c126a40d343f99c38d"
    assert ArtifactRef.from_file(MUJOCO_AGENT_PLACE_V2_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "bfaced6fc44df348e3095350b31af071bc35f9388c4ae265132a49d371d5511f"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_real_place_binding_remains_frozen_after_mujoco_agent_place_v2() -> None:
    binding = Real2SimArtifactManifest.load(REAL_PLACE_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(MUJOCO_AGENT_PLACE_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261030-real-place-v1", "20261030-mujoco-agent-place-v1"
    source = "bba13f9d4d4a100ce95e910549223536d84bd25e"
    source_tree = "a2822d1a4b409cff109b6a33ecb52f53010ecb17"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert extras == (
        "robot_motion/held_object_measurement.py",
        "agent_gateway/profiles/real/held_object.py",
        "agent_gateway/profiles/real/object_motion.py",
    )
    assert len(current) == 192 and len(inherited) == 189 and set(current) == set(inherited) | set(extras)
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"real-place-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == "runtime_source"
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {
        "agent_gateway/profiles/mujoco/gateway_cli.py",
        "agent_gateway/profiles/mujoco/motion_deployment.py",
        "agent_gateway/profiles/mujoco/motion_factory.py",
        "agent_gateway/profiles/mujoco/motion_observed_scene.py",
        "agent_gateway/profiles/real/exclusive_factory.py",
        "agent_gateway/profiles/real/gateway_cli.py",
        "agent_gateway/profiles/real/gripper_skill.py",
        "agent_gateway/profiles/real/motion_gateway_session.py",
        "agent_gateway/profiles/real/motion_primitive_profiles.py",
        "agent_gateway/profiles/real/motion_primitive_skill.py",
        "agent_gateway/profiles/real/rm65_composition.py",
        "agent_gateway/profiles/real/session_authorization.py",
        "agent_gateway/profiles/real/tool_entity.py",
        "agent_gateway/profiles/tool_entity.py",
        "agent_gateway/skills/perception/spatial_goal.py",
        "agent_gateway/skills/perception/tool_entity.py",
        "robot_motion/observed_obstacles.py",
        "robot_motion/rm65/obstacle_collision.py",
        "robot_motion/tool_entity_scene.py",
        "teleop_runtime/skill_deploy/rm65_grasp_trajectory.py",
    }
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == (
        "robot_motion/observed_obstacles.py",
        "robot_motion/tool_entity_goals.py",
    )
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["mainline_input_commit"] == "81d1fe75f0716cec2801ecd428b4efd65b8ce66b"
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        MUJOCO_AGENT_PLACE_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    parent = refs["rm65-arm-runtime-binding-mujoco-agent-place-20261030-v1"]
    assert parent.locator == str(MUJOCO_AGENT_PLACE_BINDING_PATH)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    # Every non-runtime predecessor reference is inherited unchanged.
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == (
        "default_off_real_held_object_motion_and_real_carried_object_planning")
    assert binding.metadata["real_gateway_held_object_motion"] is True
    assert binding.metadata["real_held_object_motion_default_enabled"] is False
    assert tuple(binding.metadata["real_place_gateway_skill_ids"]) == ("motion.move_grasped_object_to_pose@1",)
    assert binding.metadata["real_place_enable_action"] == "move_grasped_object_to_pose"
    assert binding.metadata["real_carried_object_planning_default_enabled"] is False
    assert binding.metadata["real_held_object_measurement"] == "stationary_rgbd_closing_region_point_prompt_plus_dmtac_closure"
    assert tuple(binding.metadata["real_place_support_gap_m"]) == (0.0, 0.02)
    assert binding.metadata["new_physical_hardware_validation"] is False
    # MuJoCo place-chain and batch-2 claims remain inherited unchanged.
    assert binding.metadata["mujoco_agent_place_skills_default_enabled"] is False
    assert binding.metadata["mujoco_object_motion_slip_gate_reference"] == "frozen_lift_relation_3mm_2deg_unchanged"
    assert (binding.metadata["hard_penetration_gate_changed"], binding.metadata["observation_gap_gate_changed"],
            binding.metadata["release_window_changed"]) == (False, False, False)
    assert binding.metadata["inherited_start_clearance_default"] is True
    assert binding.metadata["tool_entity_gateway_skills_default_enabled"] is False
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    for index, path in enumerate(expected_offline):
        ref = refs[f"real-place-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path and old_ref.artifact_kind == ref.artifact_kind
        assert (old_ref.sha256 != ref.sha256) is (path in binding.metadata["changed_previous_offline_inputs"])
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"real-place-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    reports = {
        "real-place-focused-report-" + tag: "82bdd2888aadff02ee559bd47767c2267d9359c1977c054df9ad4eb18c5abba4",
        "real-place-focused-binding-" + tag: "4264ec58076e7d5c892f6444a87888e8c0d743de0419b508fd86e6d25546ffc8",
        "real-place-offline-carry-report-" + tag: "b25fcdaa7d87accecebdfae7adb9e3e38009d5448af533f63ff968aa9a197bee",
    }
    assert tuple(binding.metadata["offline_validation_report_artifact_ids"]) == tuple(reports)
    for artifact_id, digest in reports.items():
        assert refs[artifact_id].sha256 == digest and refs[artifact_id].artifact_kind == "offline_validation_report"
    assert binding.metadata["offline_validation_report_artifact_id"] == "real-place-focused-report-" + tag
    offline = next(e for e in binding.validation_evidence if e.validation_id == "real-place-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 2166.0, "skipped_cases": 0.0, "failed_cases": 0., "error_cases": 0.}
    integrity = next(e for e in binding.validation_evidence
                     if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert dict(integrity.metrics) == {
        "bound_runtime_input_file_count": 192.0, "bound_runtime_source_file_count": 188.0,
        "changed_runtime_source_file_count_from_previous_binding": 20.0, "new_runtime_source_file_count": 3.0,
        "new_runtime_configuration_file_count": 0., "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    assert binding.artifact_sha256 == "bd41cd81d212ae13a62b0d59004d2882e0ddd5fe63e0b2bada12f02bf54613e5"
    assert ArtifactRef.from_file(REAL_PLACE_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "609278175ab9906fae62a2eede288f165f1f9fcc9296ddf99fe71684a0ab04df"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_mujoco_agent_place_binding_remains_frozen_after_real_place() -> None:
    binding = Real2SimArtifactManifest.load(MUJOCO_AGENT_PLACE_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(GATEWAY_UNIFY_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261030-mujoco-agent-place-v1", "20261029-skill-gateway-unify-v1"
    source = "c8d22fd5109cb7ec62fa76ecc6915552a5c256a2"
    source_tree = "5d53ffe0af40da4fa2da177b40cc471ec48e0ea2"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert extras == (
        "agent_gateway/profiles/measured_support.py",
        "agent_gateway/skills/perception/support_release_query.py",
        "agent_gateway/profiles/mujoco/support_release_query.py",
        "agent_gateway/skills/perception/grasp_approach_query.py",
        "agent_gateway/profiles/mujoco/grasp_approach_query.py",
        "agent_gateway/skills/motion/schemas.py",
    )
    assert len(current) == 189 and len(inherited) == 183 and set(current) == set(inherited) | set(extras)
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"mujoco-agent-place-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == "runtime_source"
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {
        "agent_gateway/profiles/mujoco/gateway_cli.py",
        "agent_gateway/profiles/mujoco/gripper_skill.py",
        "agent_gateway/profiles/mujoco/motion_deployment.py",
        "agent_gateway/profiles/mujoco/motion_factory.py",
        "agent_gateway/profiles/mujoco/motion_gateway_session.py",
        "agent_gateway/profiles/mujoco/object_motion.py",
        "agent_gateway/skills/gripper/runtime.py",
        "agent_gateway/skills/motion/grasped_object.py",
        "agent_gateway/skills/motion/path_constraint.py",
        "robot_motion/object_transport.py",
        "sim/mujoco/deploy/grasped_object.py",
    }
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == ()
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["mainline_input_commit"] == "4df7e8db17eea453bda309d27a4fc779e6eac007"
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        GATEWAY_UNIFY_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    parent = refs["rm65-arm-runtime-binding-skill-gateway-unify-20261029-v1"]
    assert parent.locator == str(GATEWAY_UNIFY_BINDING_PATH)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    # Every non-runtime predecessor reference is inherited unchanged.
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == (
        "default_off_mujoco_held_object_queries_and_mujoco_object_motion_lifecycle")
    assert tuple(binding.metadata["mujoco_agent_place_gateway_skill_ids"]) == (
        "perception.query_support_release_pose@1", "perception.query_grasp_approach@1")
    assert binding.metadata["mujoco_agent_place_skills_default_enabled"] is False
    assert binding.metadata["mujoco_agent_place_skills_read_only"] is True
    assert binding.metadata["query_and_move_composite_skill"] is False
    assert binding.metadata["mujoco_object_motion_lift_corridor_removed"] is True
    assert binding.metadata["mujoco_object_motion_goal_relation"] == "measured_in_the_same_correlated_cut"
    assert binding.metadata["mujoco_object_motion_slip_gate_reference"] == "frozen_lift_relation_3mm_2deg_unchanged"
    assert (binding.metadata["hard_penetration_gate_changed"], binding.metadata["observation_gap_gate_changed"],
            binding.metadata["release_window_changed"]) == (False, False, False)
    assert binding.metadata["real_gateway_held_object_motion"] is False
    assert binding.metadata["rigid_pose_validation_tolerances_changed"] is False
    assert binding.metadata["new_physical_hardware_validation"] is False
    # Batch-2 claims remain inherited unchanged.
    assert binding.metadata["inherited_start_clearance_default"] is True
    assert binding.metadata["tool_entity_gateway_skills_default_enabled"] is False
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    for index, path in enumerate(expected_offline):
        ref = refs[f"mujoco-agent-place-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path and old_ref.artifact_kind == ref.artifact_kind and old_ref.sha256 == ref.sha256
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"mujoco-agent-place-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    reports = {
        "mujoco-agent-place-focused-report-" + tag: "5a755457cc791dd77546fce9242adeb41b1dbd98bf6298d59bd2dfcdcd38069e",
        "mujoco-agent-place-focused-binding-" + tag: "6241a80defcd25a35801e17582c62330af75f6b7e8cb91eefe7b8b654ac4424f",
        "mujoco-agent-place-agent-cohort-report-" + tag: "cb61b7ebd56e1dcc28eb0c561a5421c50684bf6a1436ba5ac1c07b7786bfe661",
    }
    assert tuple(binding.metadata["offline_validation_report_artifact_ids"]) == tuple(reports)
    for artifact_id, digest in reports.items():
        assert refs[artifact_id].sha256 == digest and refs[artifact_id].artifact_kind == "offline_validation_report"
    assert binding.metadata["offline_validation_report_artifact_id"] == "mujoco-agent-place-focused-report-" + tag
    offline = next(e for e in binding.validation_evidence if e.validation_id == "mujoco-agent-place-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 659.0, "skipped_cases": 0.0, "failed_cases": 0., "error_cases": 0.}
    integrity = next(e for e in binding.validation_evidence
                     if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert dict(integrity.metrics) == {
        "bound_runtime_input_file_count": 189.0, "bound_runtime_source_file_count": 185.0,
        "changed_runtime_source_file_count_from_previous_binding": 11.0, "new_runtime_source_file_count": 6.0,
        "new_runtime_configuration_file_count": 0., "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    assert binding.artifact_sha256 == "772ed05939829d8e9f8328180e89e3ec926eac47f194b917da02b6a472c624ef"
    assert ArtifactRef.from_file(MUJOCO_AGENT_PLACE_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "19ce010aa88190b13b3c067837bc894d3e97cc6d177774f9e7b0fd24a38dc76e"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_skill_gateway_unify_binding_remains_frozen_after_mujoco_agent_place() -> None:
    binding = Real2SimArtifactManifest.load(GATEWAY_UNIFY_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(SKILL_RELATIONS_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261029-skill-gateway-unify-v1", "20261029-skill-relations-v1"
    source = "9b5c06ce9523a9d8b28bdb82acafd0b0394157d2"
    source_tree = "2a231ba6ec942165a45886d0d0926f4a79d51e20"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert extras == (
        "robot_motion/tool_entity_scene.py", "deployments/perception/dmtac_tool_entity.py",
        "agent_gateway/skills/perception/tool_entity.py", "agent_gateway/profiles/tool_entity.py",
        "agent_gateway/profiles/real/tool_entity.py", "agent_gateway/profiles/mujoco/tool_entity.py")
    assert len(current) == 183 and len(inherited) == 177 and set(current) == set(inherited) | set(extras)
    assert sum(path.endswith(".py") for path in current) == 179
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"tool-entity-gateway-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == "runtime_source"
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {
        "agent_gateway/profiles/mujoco/gateway_cli.py", "agent_gateway/profiles/mujoco/motion_deployment.py",
        "agent_gateway/profiles/mujoco/motion_factory.py", "agent_gateway/profiles/mujoco/motion_gateway_session.py",
        "agent_gateway/profiles/mujoco/motion_observed_scene.py", "agent_gateway/profiles/mujoco/observed_point_motion.py",
        "agent_gateway/profiles/real/exclusive_factory.py", "agent_gateway/profiles/real/gateway_cli.py",
        "agent_gateway/profiles/real/motion_gateway_session.py", "agent_gateway/profiles/real/motion_primitive_profiles.py",
        "agent_gateway/profiles/real/rm65_composition.py", "sim/mujoco/deploy/planned_tcp_motion.py"}
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == ("scripts/probes/query_safe_move_snapshot.py",)
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["mainline_input_commit"] == "ed18f3952c039990f06790dd7597cd0dd151ef1a"
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        SKILL_RELATIONS_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    parent = refs["rm65-arm-runtime-binding-skill-relations-20261029-v1"]
    assert parent.locator == str(SKILL_RELATIONS_BINDING_PATH)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    # Every non-runtime predecessor reference is inherited unchanged.
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["ordinary_clearance_m"] == .02
    assert binding.metadata["inherited_start_clearance_default"] is True
    assert binding.metadata["inherited_start_clearance_default_scope"] == (
        "saved_scene_query_real_gateway_and_mujoco_gateway_cli_observed_scene")
    assert binding.metadata["mujoco_observed_start_clearance_config_default"] == "strict"
    assert binding.metadata["mujoco_gateway_cli_observed_start_clearance_default"] == "inherited-positive-v1"
    assert binding.metadata["direct_planner_start_clearance_default_changed"] is False
    assert binding.metadata["tool_object_relation_semantics"] == "agenticrobot.relation.tool_object.v1"
    assert binding.metadata["tool_hover_semantics"] == "agenticrobot.relation.tool_entity_hover.v1"
    assert binding.metadata["tool_object_relation_gateway_route"] is True
    assert tuple(binding.metadata["tool_entity_gateway_skill_ids"]) == (
        "perception.describe_tool_object_relations@1", "perception.query_tool_hover_paths@1")
    assert binding.metadata["tool_entity_gateway_skills_default_enabled"] is False
    assert binding.metadata["tool_entity_gateway_skills_read_only"] is True
    assert binding.metadata["query_and_move_composite_skill"] is False
    assert (binding.metadata["hover_region_budget_sec"], binding.metadata["hover_candidate_budget_sec"],
            binding.metadata["tcp_region_candidate_budget_sec"]) == (20., 5., 3.)
    assert binding.metadata["hover_requires_vertical_public_frame"] is True
    assert binding.metadata["hover_certified_pose_field"] == "planned_endpoint_pose_xyz_quat_xyzw"
    assert binding.metadata["mujoco_twin_hover_collision_configuration_changed"] is False
    assert binding.metadata["hover_tool_axis_max_tilt_default_rad"] == 0.
    assert binding.metadata["new_physical_hardware_validation"] is False
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    changed_offline = []
    for index, path in enumerate(expected_offline):
        ref = refs[f"gateway-unify-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path and old_ref.artifact_kind == ref.artifact_kind
        if old_ref.sha256 != ref.sha256:
            changed_offline.append(path)
    assert tuple(sorted(changed_offline)) == tuple(binding.metadata["changed_previous_offline_inputs"])
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"gateway-unify-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    reports = {
        "gateway-unify-focused-report-" + tag: "ae1fede89d51aa43e6bde7418cfdf14ac9095167b472ae9b99088b338d7f37a2",
        "gateway-unify-focused-binding-" + tag: "bd9a9fc41a8235b2f6d159b9189efd3064e2070c693b0408cb529595507e7fce",
        "gateway-unify-real-samples-report-" + tag: "a8467d695ff2dd79b69415948e650b7f8f3c206f05f9a387a02158f7e65669b0",
        "gateway-unify-mujoco-hover-report-" + tag: "701ea4927f91552a2e4cf651989bc83ee87cf10007c66c92c6b497070dfba68d",
    }
    assert tuple(binding.metadata["offline_validation_report_artifact_ids"]) == tuple(reports)
    for artifact_id, digest in reports.items():
        assert refs[artifact_id].sha256 == digest and refs[artifact_id].artifact_kind == "offline_validation_report"
    assert binding.metadata["offline_validation_report_artifact_id"] == "gateway-unify-focused-report-" + tag
    offline = next(e for e in binding.validation_evidence if e.validation_id == "tool-entity-gateway-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 2492., "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    integrity = next(e for e in binding.validation_evidence
                     if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert dict(integrity.metrics) == {
        "bound_runtime_input_file_count": 183., "bound_runtime_source_file_count": 179.,
        "changed_runtime_source_file_count_from_previous_binding": 12., "new_runtime_source_file_count": 6.,
        "new_runtime_configuration_file_count": 0., "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    assert binding.artifact_sha256 == "bd099ec3b57cc5427023cadd9f3955bfcb8d918ca218e6da96c0e5da9f83b919"
    assert ArtifactRef.from_file(GATEWAY_UNIFY_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "d2a5227263fe0515456b80794b98c3254755effb338ef0bec6dbaf049527eb47"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_skill_relations_binding_remains_frozen_after_gateway_unify() -> None:
    binding = Real2SimArtifactManifest.load(SKILL_RELATIONS_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(OBSERVED_CACHE_ORDER_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261029-skill-relations-v1", "20261029-observed-cache-order-v1"
    source = "d20dd86bb0e1493c19357b1053e639b0750dcf0e"
    source_tree = "ddedae194c106d43c0d7c62c6ec109f0fc61e457"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    assert tuple(binding.metadata["new_runtime_sources"]) == ()
    assert len(current) == len(inherited) == 177 and set(current) == set(inherited)
    assert sum(path.endswith(".py") for path in current) == 173
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
        assert ref.artifact_kind == inherited[path].artifact_kind
        assert ref.sha256 == inherited[path].sha256
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == ()
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == (
        "robot_motion/tool_entity_goals.py", "scripts/probes/query_safe_move_snapshot.py")
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["mainline_input_commit"] == "3efadf50b68583dd0ae4e476db716b9a573f3e43"
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        OBSERVED_CACHE_ORDER_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    parent = refs["rm65-arm-runtime-binding-observed-cache-order-20261029-v1"]
    assert parent.locator == str(OBSERVED_CACHE_ORDER_BINDING_PATH)
    assert parent.sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    # Every non-runtime predecessor reference is inherited unchanged.
    for ref in previous.dependency_artifacts:
        if "-runtime-" + old_tag not in ref.artifact_id:
            assert refs[ref.artifact_id] == ref
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is False
    assert binding.metadata["inherited_start_clearance_default"] is True
    assert binding.metadata["inherited_start_clearance_default_scope"] == "saved_scene_query_and_real_gateway_observed_scene"
    assert binding.metadata["direct_planner_start_clearance_default_changed"] is False
    assert binding.metadata["ordinary_clearance_m"] == .02
    assert binding.metadata["observed_distance_cache_default"] is True
    assert binding.metadata["saved_scene_endpoint_ordering_default"] == "joint_distance"
    assert binding.metadata["tool_object_relation_semantics"] == "agenticrobot.relation.tool_object.v1"
    assert binding.metadata["tool_object_relation_gateway_route"] is False
    assert binding.metadata["tool_object_relation_contact_solids"] == "urdf_dmtac_sensor_collision_boxes_open_v1"
    assert binding.metadata["tool_object_relation_gripper_pose_source"] == "frame_capture_qpos"
    assert binding.metadata["tool_object_relation_planning_self_filter_used"] is False
    assert binding.metadata["hover_tool_axis_max_tilt_default_rad"] == 0.
    assert binding.metadata["hover_achieved_tilt_checked"] is True
    assert binding.metadata["new_physical_hardware_validation"] is False
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    changed_offline = []
    for index, path in enumerate(expected_offline):
        ref = refs[f"relation-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path and old_ref.artifact_kind == ref.artifact_kind
        if old_ref.sha256 != ref.sha256:
            changed_offline.append(path)
    assert tuple(sorted(changed_offline)) == tuple(binding.metadata["changed_previous_offline_inputs"])
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"relation-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    reports = {
        "relation-focused-report-" + tag: "4e57d7e46c32569429d438d9f8841813f41fadbbd32611075e3ba4915070c02c",
        "relation-focused-binding-" + tag: "83121638167cec24fa82533ce8106184110ebe4eac9671e60eb4431b45c31e79",
        "relation-real-samples-report-" + tag: "54de033312d9a3ccc51a93f2c1ce86c9acfd4a73d7837f3f00a96a6b25ee5b85",
        "relation-mujoco-report-" + tag: "f175ae499a8ebf76d011a79e1c80186b4c3a5be54df5d7441bbc576c32be90f3",
    }
    assert tuple(binding.metadata["offline_validation_report_artifact_ids"]) == tuple(reports)
    for artifact_id, digest in reports.items():
        assert refs[artifact_id].sha256 == digest and refs[artifact_id].artifact_kind == "offline_validation_report"
    assert binding.metadata["offline_validation_report_artifact_id"] == "relation-focused-report-" + tag
    offline = next(e for e in binding.validation_evidence if e.validation_id == "relation-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 1630., "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    assert binding.artifact_sha256 == "0569e69f7e77d69a83b5ad93dc76fe3c76b4fd1260d3bc784977c563c7461ff0"
    assert ArtifactRef.from_file(SKILL_RELATIONS_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "ea11d919ae694c5568014953117e6ec317db6ef4ffcadc81ab8b7ea15a1211f4"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_observed_cache_order_binding_remains_frozen_after_skill_relations() -> None:
    binding = Real2SimArtifactManifest.load(OBSERVED_CACHE_ORDER_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(HOVER_VISUAL_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261029-observed-cache-order-v1", "20261029-skill-hover-visual-v1"
    source = "b2b77e61e09b374d51c29cb72fb5887037ade551"
    source_tree = "31e3dc9d40f05dc37184c8c2c7a9f2eaf93b1722"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert extras == ("robot_motion/observed_obstacles.py",)
    assert len(current) == 177 and len(inherited) == 176 and set(current) == set(inherited) | set(extras)
    assert sum(path.endswith(".py") for path in current) == 173
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"observed-cache-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == "runtime_source"
    assert all(current[path].sha256 == inherited[path].sha256 for path in inherited)
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == ()
    assert tuple(binding.metadata["changed_previous_offline_inputs"]) == (
        "robot_motion/observed_obstacles.py", "scripts/probes/query_safe_move_snapshot.py")
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        HOVER_VISUAL_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["inherited_start_clearance_default"] is True
    assert binding.metadata["inherited_start_clearance_default_scope"] == "saved_scene_query_and_real_gateway_observed_scene"
    assert binding.metadata["direct_planner_start_clearance_default_changed"] is False
    assert binding.metadata["inherited_start_clearance_manifest_bound"] is True
    assert binding.metadata["inherited_start_query_failure_preserved"] is True
    assert binding.metadata["ordinary_clearance_m"] == .02
    assert binding.metadata["observed_distance_cache_default"] is True
    assert binding.metadata["saved_scene_endpoint_ordering_default"] == "joint_distance"
    assert tuple(binding.metadata["saved_scene_endpoint_ordering_alternatives"]) == (
        "estimated_execution_time", "tcp_arc_estimated_execution_time")
    assert binding.metadata["new_physical_hardware_validation"] is False
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    previous_refs = {ref.artifact_id: ref for ref in previous.dependency_artifacts}
    changed_offline = []
    for index, path in enumerate(expected_offline):
        ref = refs[f"observed-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        old_ref = previous_refs[previous.metadata["offline_query_input_artifact_ids"][index]]
        assert old_ref.locator == path
        if old_ref.sha256 != ref.sha256:
            changed_offline.append(path)
    assert tuple(sorted(changed_offline)) == tuple(binding.metadata["changed_previous_offline_inputs"])
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"observed-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    assert refs[binding.metadata["offline_validation_report_artifact_id"]].sha256 == "21fd2272868f3f099c0d2b9a2a50cdcb541504e114f669f23113b3d65b6fd18f"
    offline = next(e for e in binding.validation_evidence if e.validation_id == "observed-cache-offline-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 978., "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    assert binding.artifact_sha256 == "ef5bc3ae629ec628d03b001201acfb7b99039b51e2e3a66d5e5372a84de39baa"
    assert ArtifactRef.from_file(OBSERVED_CACHE_ORDER_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "ff265d915da9226371af42185d351587ad146683822255a54304a45d975abca5"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_skill_hover_visual_binding_remains_frozen_after_observed_cache() -> None:
    binding = Real2SimArtifactManifest.load(HOVER_VISUAL_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(SKILL_REGION_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261029-skill-hover-visual-v1", "20261028-skill-region-checked-v2"
    source = "1e2ab619c2f3c30ba4bf28eba6344589014ae1c1"
    source_tree = "cc53ad4a5ceb42838717747aa45e6af9cf563af1"
    current = {ref.locator: ref for ref in binding.dependency_artifacts if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert extras == (
        "sim/mujoco/actuation/__init__.py", "sim/mujoco/actuation/gripper_profile.py",
        "sim/mujoco/mjcf.py", "sim/mujoco/modeling/geometry.py",
        "sim/mujoco/modeling/gripper_actuators.py", "sim/mujoco/modeling/objects.py")
    assert len(current) == 176 and len(inherited) == 170 and set(current) == set(inherited) | set(extras)
    assert sum(path.endswith(".py") for path in current) == 172
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"combined-runtime-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == "runtime_source"
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {"agent_gateway/profiles/real/gateway_cli.py", "sim/mujoco/deploy/grasp_contact.py",
                       "sim/mujoco/deploy/payload_motion.py", "sim/mujoco/scene.py"}
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == source
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == source_tree
    assert subprocess.check_output(["git", "rev-parse", source + "^{tree}"], text=True).strip() == source_tree
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        SKILL_REGION_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["inherited_start_clearance_default"] is True
    assert binding.metadata["inherited_start_clearance_default_scope"] == "saved_scene_query_and_real_gateway_observed_scene"
    assert binding.metadata["direct_planner_start_clearance_default_changed"] is False
    assert binding.metadata["inherited_start_clearance_manifest_bound"] is True
    assert binding.metadata["inherited_start_query_failure_preserved"] is True
    assert binding.metadata["ordinary_clearance_m"] == .02
    assert binding.metadata["new_physical_hardware_validation"] is False
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    expected_offline = (
        "scripts/probes/query_safe_move_snapshot.py", "robot_motion/tool_entity_goals.py",
        "robot_motion/observed_obstacles.py", "robot_motion/grasp_visible_scene.py",
        "deployments/perception/dmtac_grasp_overlay.py",
        "deployments/perception/profiles/dmtac_graspgenx_alignment_v5_q0.json")
    for index, path in enumerate(expected_offline):
        ref = refs[f"combined-query-input-{index}-{tag}"]
        assert ref.locator == path
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(["git", "show", source + ":" + path])).hexdigest()
    assert tuple(binding.metadata["offline_query_input_artifact_ids"]) == tuple(
        f"combined-query-input-{index}-{tag}" for index in range(len(expected_offline)))
    assert refs[binding.metadata["offline_validation_report_artifact_id"]].sha256 == "343381c2b13e8a8a5baff1a43c94bc150d354a2d2c4e5b5c30ce637182f98a77"
    offline = next(e for e in binding.validation_evidence if e.validation_id == "combined-offline-source-consumers-" + tag)
    assert offline.status == "passed" and dict(offline.metrics) == {
        "passed_cases": 1297., "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    assert binding.artifact_sha256 == "5ac0927c6c5799ab423b4f6b394cfff928732073d8fe67d7d0a3dc999d4c2f87"
    assert ArtifactRef.from_file(HOVER_VISUAL_BINDING_PATH, artifact_id="binding-file-check",
        artifact_kind="real2sim_runtime_binding").sha256 == "2be3606c9b558ccfe1cd925062f481c148a5df2658e9be5456217d5f15da0c99"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_skill_region_binding_remains_frozen_after_skill_hover() -> None:
    binding = Real2SimArtifactManifest.load(SKILL_REGION_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(EXPLICIT_CLEARANCE_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261028-skill-region-checked-v2", "20261028-explicit-clearance-v1"
    current = {ref.locator: ref for ref in binding.dependency_artifacts
               if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts
                 if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert len(current) == 170 and len(inherited) == 168 and len(extras) == 2
    assert extras == ('robot_motion/path_planning/motion_quality.py', 'robot_motion/path_planning/sampling.py')
    assert set(current) == set(inherited) | set(extras)
    assert sum(path.endswith(".py") for path in current) == 166
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(
            ["git", "show", "b3b03e8a6212c94f7cfd6efce0abd377fe0f311a:" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"skill-region-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == ("runtime_source" if path.endswith(".py") else "runtime_configuration")
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == set(['robot_motion/spatial_goal_regions.py', 'teleop_runtime/skill_deploy/rm65_grasp_trajectory.py'])
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["fit_simulation_contract_sha256"] == fit.simulation_contract.sha256
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        EXPLICIT_CLEARANCE_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    assert binding.metadata["arm_profile_refit"] is False and binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "safe_pose_endpoint_quality_lower_bound_and_offline_region_query"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == "b3b03e8a6212c94f7cfd6efce0abd377fe0f311a"
    assert binding.metadata["offline_validation_execution_head"] == "8c321a452bbb0896c79472adb42ce1930932d7dd"
    assert binding.metadata["offline_validation_mode"] == "fixed_staged_tree"
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == "70bdae8f4d3e16e5aafffd5ee0890684545c48c4"
    assert binding.artifact_sha256 == "d8996bea68e10117c62a937473a3da99c26bc95e0df0a2f30a8c0ed30f50743a"
    assert ArtifactRef.from_file(SKILL_REGION_BINDING_PATH,
        artifact_id="binding-file-check", artifact_kind="real2sim_runtime_binding").sha256 == "56d0e4562d8abec99bfc5a543949fea5b128ae6378056ff243d729acc44ec31f"
    assert binding.fidelity_claims[0].status == "provisional"
    content = next(e for e in binding.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert content.status == "passed" and content.split == "validation"
    assert dict(content.metrics) == {'bound_runtime_input_file_count': 170.0, 'bound_runtime_source_file_count': 166.0, 'changed_runtime_source_file_count_from_previous_binding': 2.0, 'new_runtime_source_file_count': 2.0, 'new_runtime_configuration_file_count': 0.0, 'arm_profile_refit_count': 0.0, 'mutated_historical_artifact_count': 0.0}
    offline = next(e for e in binding.validation_evidence if e.validation_id == "skill-region-source-consumers-" + tag)
    assert offline.status == "passed" and offline.split == "validation"
    assert dict(offline.metrics) == {"core_passed_cases": 805.0, "query_passed_cases": 252.0, "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    assert refs[binding.metadata["offline_validation_report_artifact_id"]].sha256 == "112ea68158843001c3fcff48d5dc9cb1073c0b20ce5260739e1df60f47f27785"
    assert binding.metadata["gateway_prunes_expired_proposals"] is True
    assert binding.metadata["lower_factory_prune_default"] is False
    assert binding.metadata["ttl_and_hold_changed"] is False
    assert binding.metadata["gripper_max_operation_duration_ms"] == 15000
    assert binding.metadata["support_footprint_maximum_samples"] == 100000
    assert binding.metadata["named_release_v3_default"] is False
    assert binding.metadata["named_release_v3_sim_only"] is True
    assert binding.metadata["unchanged_gripper_v1_v2_defaults"] is True
    assert binding.metadata["real_observed_point_default"] is False
    assert binding.metadata["real_surface_anchor_v2_default"] is False
    assert binding.metadata["real_anchor_motion_v2_default"] is False
    assert binding.metadata["capture_self_filter_default_changed"] is False
    assert binding.metadata["operator_authorization_required"] is True
    assert binding.metadata["inherited_start_clearance_default"] is False
    assert binding.metadata["inherited_start_clearance_manifest_bound"] is True
    assert binding.metadata["inherited_start_query_failure_preserved"] is True
    assert binding.metadata["ordinary_clearance_m"] == 0.02
    assert binding.metadata["saved_scene_region_candidate_policy_default"] == "proximity"
    assert binding.metadata["saved_scene_region_halton_explicit"] is True
    assert binding.metadata["source_native_comparison_inherited_clearance_enabled"] is True
    assert all(binding.metadata[f"source_native_comparison_formal_0{n}_positive_start_count"] == 12 for n in (2, 3, 4))
    assert binding.metadata["source_native_comparison_scenes_per_run"] == 12
    assert binding.metadata["strict_start_native_comparison_status"] == "not_run"
    query_ref = refs[binding.metadata["offline_query_source_artifact_id"]]
    assert query_ref.locator == "scripts/probes/query_safe_move_snapshot.py"
    assert query_ref.artifact_kind == "offline_query_source"
    assert query_ref.sha256 == hashlib.sha256(subprocess.check_output(
        ["git", "show", "b3b03e8a6212c94f7cfd6efce0abd377fe0f311a:" + query_ref.locator])).hexdigest()
    assert query_ref.sha256 == 'c602f2037315d8c8b002319f5d778f2980dd899c2d760d0006c8db5eddaddfc1'
    assert refs[binding.metadata["offline_query_validation_report_artifact_id"]].sha256 == '4335d7e8f5ce58ce0bd1550ecb9435fea1d1a3bdddf6caf76c07b61613f30d91'
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_unqualified_region_binding_remains_frozen_as_evidence() -> None:
    binding = Real2SimArtifactManifest.load(UNQUALIFIED_REGION_BINDING_PATH)
    assert binding.code_revision == "6b3a21148f901c3760353c3a54cf4afbce8b99b0"
    assert binding.artifact_sha256 == "caf361bdb7dacb8bf7f9ff408229be4c8132d293c974e8f30c114f0ca9864af9"
    assert ArtifactRef.from_file(UNQUALIFIED_REGION_BINDING_PATH,
        artifact_id="unqualified-region-evidence", artifact_kind="real2sim_runtime_binding").sha256 == (
        "f522ba859a7194398fb92b135fa40b1e5b96890471f23b4882022c338ca662be")
    query_ref = next(ref for ref in binding.dependency_artifacts if ref.artifact_kind == "offline_query_source")
    historical_source = subprocess.check_output([
        "git", "show", binding.code_revision + ":" + query_ref.locator])
    assert hashlib.sha256(historical_source).hexdigest() == query_ref.sha256


def test_explicit_clearance_binding_remains_frozen_after_skill_region() -> None:
    binding = Real2SimArtifactManifest.load(EXPLICIT_CLEARANCE_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(REAL_OBSERVED_MOTION_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261028-explicit-clearance-v1", "20261028-real-observed-motion-v1"
    current = {ref.locator: ref for ref in binding.dependency_artifacts
               if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts
                 if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert len(current) == 168 and len(inherited) == 168 and len(extras) == 0
    assert extras == ()
    assert set(current) == set(inherited) | set(extras)
    assert sum(path.endswith(".py") for path in current) == 164
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(
            ["git", "show", "e9245d2fdd878dba7f4434a8aa0096a37bc4c4d0:" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"explicit-clearance-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == ("runtime_source" if path.endswith(".py") else "runtime_configuration")
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == set(['agent_gateway/profiles/real/gateway_cli.py', 'agent_gateway/profiles/real/rm65_composition.py', 'robot_motion/rm65/obstacle_collision.py', 'teleop_runtime/skill_deploy/rm65_grasp_trajectory.py'])
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["fit_simulation_contract_sha256"] == fit.simulation_contract.sha256
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        REAL_OBSERVED_MOTION_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    assert binding.metadata["arm_profile_refit"] is False and binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "explicit_inherited_start_clearance_on_existing_safe_motion_owners"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == "e9245d2fdd878dba7f4434a8aa0096a37bc4c4d0"
    assert binding.metadata["offline_validation_execution_head"] == binding.code_revision
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == "4c126ac23dd781bc746f5629c3bd59ecd74f8506"
    assert binding.artifact_sha256 == "d848da50f936e8f997c73975d540df36b66dcf66d9480e24bb8e6451f3a49569"
    assert ArtifactRef.from_file(EXPLICIT_CLEARANCE_BINDING_PATH,
        artifact_id="binding-file-check", artifact_kind="real2sim_runtime_binding").sha256 == "7b55fba29f586b194529fa3c86ced30a44f6fff4f62243f1c2595baf28e698a7"
    assert binding.fidelity_claims[0].status == "provisional"
    content = next(e for e in binding.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert content.status == "passed" and content.split == "validation"
    assert dict(content.metrics) == {'bound_runtime_input_file_count': 168.0, 'bound_runtime_source_file_count': 164.0, 'changed_runtime_source_file_count_from_previous_binding': 4.0, 'new_runtime_source_file_count': 0.0, 'new_runtime_configuration_file_count': 0.0, 'arm_profile_refit_count': 0.0, 'mutated_historical_artifact_count': 0.0}
    offline = next(e for e in binding.validation_evidence if e.validation_id == "explicit-clearance-source-consumers-" + tag)
    assert offline.status == "passed" and offline.split == "validation"
    assert dict(offline.metrics) == {"passed_cases": 977.0, "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    assert refs[binding.metadata["offline_validation_report_artifact_id"]].sha256 == "ef7282a6019d176030ccbd7f425383f595c54d81047714a89a411a0afce8a3c1"
    assert binding.metadata["gateway_prunes_expired_proposals"] is True
    assert binding.metadata["lower_factory_prune_default"] is False
    assert binding.metadata["ttl_and_hold_changed"] is False
    assert binding.metadata["gripper_max_operation_duration_ms"] == 15000
    assert binding.metadata["support_footprint_maximum_samples"] == 100000
    assert binding.metadata["named_release_v3_default"] is False
    assert binding.metadata["named_release_v3_sim_only"] is True
    assert binding.metadata["unchanged_gripper_v1_v2_defaults"] is True
    assert binding.metadata["real_observed_point_default"] is False
    assert binding.metadata["real_surface_anchor_v2_default"] is False
    assert binding.metadata["real_anchor_motion_v2_default"] is False
    assert binding.metadata["capture_self_filter_default_changed"] is False
    assert binding.metadata["operator_authorization_required"] is True
    assert binding.metadata["inherited_start_clearance_default"] is False
    assert binding.metadata["inherited_start_clearance_manifest_bound"] is True
    assert binding.metadata["inherited_start_query_failure_preserved"] is True
    assert binding.metadata["ordinary_clearance_m"] == 0.02
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_real_observed_motion_binding_remains_frozen_after_explicit_clearance() -> None:
    binding = Real2SimArtifactManifest.load(REAL_OBSERVED_MOTION_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(NAMED_RELEASE_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261028-real-observed-motion-v1", "20261028-named-release-v1"
    current = {ref.locator: ref for ref in binding.dependency_artifacts
               if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts
                 if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert len(current) == 168 and len(inherited) == 164 and len(extras) == 4
    assert extras == ('agent_gateway/profiles/real/exclusive_factory.py', 'agent_gateway/profiles/real/motion_primitive_skill.py', 'agent_gateway/profiles/real/observed_point_motion.py', 'teleop_runtime/configs/skill_deploy/rm65_dmtac_capture_self_filter_cad_v1.json')
    assert set(current) == set(inherited) | set(extras)
    assert sum(path.endswith(".py") for path in current) == 164
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(
            ["git", "show", "c0334f16467f230c730f83001e6e01751aa8bd4f:" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"real-observed-motion-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == ("runtime_source" if path.endswith(".py") else "runtime_configuration")
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == set(['agent_gateway/profiles/real/gateway_cli.py', 'agent_gateway/profiles/real/motion_gateway_session.py', 'agent_gateway/profiles/real/motion_primitive_profiles.py', 'agent_gateway/profiles/real/rm65_composition.py', 'agent_gateway/profiles/real/session_authorization.py', 'agent_gateway/profiles/surface_anchor.py'])
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["fit_simulation_contract_sha256"] == fit.simulation_contract.sha256
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        NAMED_RELEASE_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    assert binding.metadata["arm_profile_refit"] is False and binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "optional_real_observed_point_and_anchor_routes_on_existing_safe_motion_owners"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == "c0334f16467f230c730f83001e6e01751aa8bd4f"
    assert binding.metadata["offline_validation_execution_head"] == binding.code_revision
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == "47828ee7028e0542f2a61bf253518c1bdc3b82d6"
    assert binding.artifact_sha256 == "5209a6d8f2dcf494f3118c4afec4aa88832a1a6a18725caeffb2f613a0ab2b38"
    assert ArtifactRef.from_file(REAL_OBSERVED_MOTION_BINDING_PATH,
        artifact_id="binding-file-check", artifact_kind="real2sim_runtime_binding").sha256 == "a795f95e7d99d952c2277a7c55e66ed1ad95ca3509a77843968ff2131b9098df"
    assert binding.fidelity_claims[0].status == "provisional"
    content = next(e for e in binding.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert content.status == "passed" and content.split == "validation"
    assert dict(content.metrics) == {'bound_runtime_input_file_count': 168.0, 'bound_runtime_source_file_count': 164.0, 'changed_runtime_source_file_count_from_previous_binding': 6.0, 'new_runtime_source_file_count': 3.0, 'new_runtime_configuration_file_count': 1.0, 'arm_profile_refit_count': 0.0, 'mutated_historical_artifact_count': 0.0}
    offline = next(e for e in binding.validation_evidence if e.validation_id == "real-observed-motion-source-consumers-" + tag)
    assert offline.status == "passed" and offline.split == "validation"
    assert dict(offline.metrics) == {"passed_cases": 632.0, "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    assert refs[binding.metadata["offline_validation_report_artifact_id"]].sha256 == "252dda8154ce050fc2de9d05859342f035ca585b1bc04f68229f364bdc9218cd"
    assert binding.metadata["gateway_prunes_expired_proposals"] is True
    assert binding.metadata["lower_factory_prune_default"] is False
    assert binding.metadata["ttl_and_hold_changed"] is False
    assert binding.metadata["gripper_max_operation_duration_ms"] == 15000
    assert binding.metadata["support_footprint_maximum_samples"] == 100000
    assert binding.metadata["named_release_v3_default"] is False
    assert binding.metadata["named_release_v3_sim_only"] is True
    assert binding.metadata["unchanged_gripper_v1_v2_defaults"] is True
    assert binding.metadata["real_observed_point_default"] is False
    assert binding.metadata["real_surface_anchor_v2_default"] is False
    assert binding.metadata["real_anchor_motion_v2_default"] is False
    assert binding.metadata["capture_self_filter_default_changed"] is False
    assert binding.metadata["operator_authorization_required"] is True
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_named_release_binding_remains_frozen_after_real_observed_motion() -> None:
    binding = Real2SimArtifactManifest.load(NAMED_RELEASE_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(AGENT_PROPOSAL_CUT_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261028-named-release-v1", "20261028-agent-proposal-cut-v1"
    current = {ref.locator: ref for ref in binding.dependency_artifacts
               if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts
                 if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert len(current) == 164 and len(inherited) == 163 and len(extras) == 1
    assert extras == ("agent_gateway/skills/gripper/definition.py",)
    assert set(current) == set(inherited) | set(extras)
    assert sum(path.endswith(".py") for path in current) == 161
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(
            ["git", "show", "88db4d1db3c3009e46dbde5a34f6072c590c55d7:" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"named-release-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == "runtime_source"
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == set(['agent_gateway/profiles/mujoco/gateway_cli.py', 'agent_gateway/profiles/mujoco/gripper_skill.py', 'agent_gateway/profiles/mujoco/motion_application.py', 'agent_gateway/profiles/mujoco/motion_deployment.py', 'agent_gateway/profiles/mujoco/motion_factory.py', 'agent_gateway/profiles/mujoco/motion_gateway_session.py', 'agent_gateway/skills/gripper/deployment.py', 'agent_gateway/skills/gripper/runtime.py', 'sim/mujoco/actuation/gripper_executor.py', 'sim/mujoco/adapters/robot_motion.py', 'sim/mujoco/deploy/grasped_object.py'])
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["fit_simulation_contract_sha256"] == fit.simulation_contract.sha256
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        AGENT_PROPOSAL_CUT_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    assert binding.metadata["arm_profile_refit"] is False and binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "optional_named_support_release_on_existing_atomic_gripper_and_observer_owners"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == "88db4d1db3c3009e46dbde5a34f6072c590c55d7"
    assert binding.metadata["offline_validation_execution_head"] == binding.code_revision
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == "c7fa6329c2644d7e45575becc5dbfb29625a3b1e"
    assert binding.artifact_sha256 == "603a75aecdd45a871a57f8ec102e183e256f2e696e246b94b6b0a85c89795c7c"
    assert ArtifactRef.from_file(NAMED_RELEASE_BINDING_PATH,
        artifact_id="binding-file-check", artifact_kind="real2sim_runtime_binding").sha256 == "76c5d691fbf2eee6ba6a5c26e9be3b469aa1763f38d043e97da11e9cb8bccd66"
    assert binding.fidelity_claims[0].status == "provisional"
    content = next(e for e in binding.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert content.status == "passed" and content.split == "validation"
    assert dict(content.metrics) == {"bound_runtime_input_file_count": 164., "bound_runtime_source_file_count": 161., "changed_runtime_source_file_count_from_previous_binding": 11., "new_runtime_source_file_count": 1., "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    offline = next(e for e in binding.validation_evidence if e.validation_id == "named-release-source-consumers-" + tag)
    assert offline.status == "passed" and offline.split == "validation"
    assert dict(offline.metrics) == {"passed_cases": 479.0, "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    assert refs[binding.metadata["offline_validation_report_artifact_id"]].sha256 == "62a2613926e21b72e6d668d975b785e1859297f5c8fc6d7aec4fa72fc3f57cb4"
    assert binding.metadata["gateway_prunes_expired_proposals"] is True
    assert binding.metadata["lower_factory_prune_default"] is False
    assert binding.metadata["ttl_and_hold_changed"] is False
    assert binding.metadata["gripper_max_operation_duration_ms"] == 15000
    assert binding.metadata["support_footprint_maximum_samples"] == 100000
    assert binding.metadata["named_release_v3_default"] is False
    assert binding.metadata["named_release_v3_sim_only"] is True
    assert binding.metadata["unchanged_gripper_v1_v2_defaults"] is True
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_agent_proposal_cut_binding_remains_frozen_after_named_release() -> None:
    binding = Real2SimArtifactManifest.load(AGENT_PROPOSAL_CUT_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(SAM3_PLACEMENT_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261028-agent-proposal-cut-v1", "20261028-sam3-placement-v1"
    current = {ref.locator: ref for ref in binding.dependency_artifacts
               if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts
                 if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert len(current) == 163 and len(inherited) == 163 and len(extras) == 0
    assert set(current) == set(inherited) | set(extras)
    assert sum(path.endswith(".py") for path in current) == 160
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(
            ["git", "show", "4de2591b3c48ac9e8787acd7c28f8c14cb551b45:" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"placement-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == "runtime_source"
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == set(['agent_gateway/profiles/mujoco/motion_factory.py', 'sim/mujoco/deploy/agent_skill_target.py', 'sim/mujoco/deploy/payload_motion.py'])
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["fit_simulation_contract_sha256"] == fit.simulation_contract.sha256
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        SAM3_PLACEMENT_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    assert binding.metadata["arm_profile_refit"] is False and binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "agent_mujoco_expired_proposal_pruning_and_single_measured_private_cut"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == "4de2591b3c48ac9e8787acd7c28f8c14cb551b45"
    assert binding.metadata["offline_validation_execution_head"] == binding.code_revision
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == "f9807beb1462e3ddbdd02f9ac4b71777fe5a1832"
    assert binding.artifact_sha256 == "c3f7f618bc2d23e12e2412a600475dace617ede5d3ad5a0bca8d07e0ecc3b370"
    assert ArtifactRef.from_file(AGENT_PROPOSAL_CUT_BINDING_PATH,
        artifact_id="binding-file-check", artifact_kind="real2sim_runtime_binding").sha256 == "ef4e1473204ed99a2b17208b3e53279aa10ca7d8f9cfd8be2e21cfd05044b394"
    assert binding.fidelity_claims[0].status == "provisional"
    content = next(e for e in binding.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert content.status == "passed" and content.split == "validation"
    assert dict(content.metrics) == {"bound_runtime_input_file_count": 163., "bound_runtime_source_file_count": 160., "changed_runtime_source_file_count_from_previous_binding": 3., "new_runtime_source_file_count": 0., "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    offline = next(e for e in binding.validation_evidence if e.validation_id == "agent-proposal-cut-source-consumers-" + tag)
    assert offline.status == "passed" and offline.split == "validation"
    assert dict(offline.metrics) == {"passed_cases": 373., "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    assert refs[binding.metadata["offline_validation_report_artifact_id"]].sha256 == "da77476444daa176b5411d7d1ce636a45d8d37a502e9edec1e3fc304ad3daf5f"
    assert binding.metadata["gateway_prunes_expired_proposals"] is True
    assert binding.metadata["lower_factory_prune_default"] is False
    assert binding.metadata["ttl_and_hold_changed"] is False
    assert binding.metadata["gripper_max_operation_duration_ms"] == 15000
    assert binding.metadata["support_footprint_maximum_samples"] == 100000
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_sam3_placement_binding_remains_frozen_after_proposal_cut() -> None:
    binding = Real2SimArtifactManifest.load(SAM3_PLACEMENT_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(SAM3_CANDIDATES_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag, old_tag = "20261028-sam3-placement-v1", "20261028-sam3-candidates-v1"
    current = {ref.locator: ref for ref in binding.dependency_artifacts
               if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts
                 if "-runtime-" + old_tag in ref.artifact_id}
    extras = tuple(binding.metadata["new_runtime_sources"])
    assert len(current) == 163 and len(inherited) == 154 and len(extras) == 9
    assert set(current) == set(inherited) | set(extras)
    assert sum(path.endswith(".py") for path in current) == 160
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(
            ["git", "show", "b584813411ccfffa0853f12b052e4f1d2c821050:" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == f"placement-owner-{extras.index(path)}-runtime-" + tag
            assert ref.artifact_kind == "runtime_source"
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert len(changed) == 10
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["fit_simulation_contract_sha256"] == fit.simulation_contract.sha256
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        SAM3_CANDIDATES_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    assert binding.metadata["arm_profile_refit"] is False and binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "bounded_mujoco_payload_support_placement_and_real_sim_gripper_15s"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == "b584813411ccfffa0853f12b052e4f1d2c821050"
    assert binding.metadata["offline_validation_execution_head"] == binding.code_revision
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == "8d6662180e8e010dd978babb67fae8ad4448a8f4"
    assert binding.artifact_sha256 == "ff9ee9a590b0df3b219041465362956bc165cdb7dfe64d5c2fae1422b9ec4067"
    assert ArtifactRef.from_file(SAM3_PLACEMENT_BINDING_PATH,
        artifact_id="binding-file-check", artifact_kind="real2sim_runtime_binding").sha256 == "fa5b47717090f98cd963d23dafe220625b4a94e70d79d0e99cc94bc891335049"
    assert binding.fidelity_claims[0].status == "provisional"
    content = next(e for e in binding.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert content.status == "passed" and content.split == "validation"
    assert dict(content.metrics) == {"bound_runtime_input_file_count": 163., "bound_runtime_source_file_count": 160., "changed_runtime_source_file_count_from_previous_binding": 10., "new_runtime_source_file_count": 9., "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    offline = next(e for e in binding.validation_evidence if e.validation_id == "sam3-placement-source-consumers-" + tag)
    assert offline.status == "passed" and offline.split == "validation"
    assert dict(offline.metrics) == {"passed_cases": 652., "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    assert refs[binding.metadata["offline_validation_report_artifact_id"]].sha256 == "e7c2f19f90da42da568ce5667d4c4f75554637e9c83dd63958a24aca71acc1bd"
    assert binding.metadata["gripper_max_operation_duration_ms"] == 15000
    assert binding.metadata["support_footprint_maximum_samples"] == 100000
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_sam3_candidates_binding_remains_frozen_after_placement() -> None:
    binding = Real2SimArtifactManifest.load(SAM3_CANDIDATES_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(MOTION_GRIPPER_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag = "20261028-sam3-candidates-v1"
    old_tag = "20261027-motion-gripper-v1"
    current = {ref.locator: ref for ref in binding.dependency_artifacts
               if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts
                 if "-runtime-" + old_tag in ref.artifact_id}
    assert len(current) == len(inherited) == 154 and set(current) == set(inherited)
    assert sum(path.endswith(".py") for path in current) == 151
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(
            ["git", "show", "66765c65e736fc6dcdf80d0f0e96878c513ed3cb:" + path])).hexdigest()
        assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
        assert ref.artifact_kind == inherited[path].artifact_kind
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {"agent_gateway/profiles/mujoco/gateway_cli.py",
        "agent_gateway/profiles/real/gateway_cli.py", "deployments/perception/sam3_service.py"}
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["new_runtime_sources"]) == ()
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["fit_simulation_contract_sha256"] == fit.simulation_contract.sha256
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        MOTION_GRIPPER_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    assert binding.metadata["arm_profile_refit"] is False and binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "explicit_sam3_multi_instance_localization_default_binding_unchanged"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == "66765c65e736fc6dcdf80d0f0e96878c513ed3cb"
    assert binding.metadata["offline_validation_execution_head"] == binding.code_revision
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == "ecdbc7053b04cdc7c9e61bb07856ca6bcaaa88c3"
    assert binding.artifact_sha256 == "085a6b6e92fd0a606953ccfeca3dc577142fe621f577e413ef0ca208fe5a0e82"
    assert ArtifactRef.from_file(SAM3_CANDIDATES_BINDING_PATH,
        artifact_id="binding-file-check", artifact_kind="real2sim_runtime_binding").sha256 == "63b0ea27ad8614fd3dde48964c1b574150e52b73fdf7fe416db87e896704366d"
    assert binding.fidelity_claims[0].status == "provisional"
    content = next(e for e in binding.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert content.status == "passed" and content.split == "validation"
    assert dict(content.metrics) == {"bound_runtime_input_file_count": 154., "bound_runtime_source_file_count": 151., "changed_runtime_source_file_count_from_previous_binding": 3., "new_runtime_source_file_count": 0., "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    offline = next(e for e in binding.validation_evidence if e.validation_id == "sam3-candidates-source-consumers-" + tag)
    assert offline.status == "passed" and offline.split == "validation"
    assert dict(offline.metrics) == {"passed_cases": 674., "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    assert refs[binding.metadata["offline_validation_report_artifact_id"]].sha256 == "422b228d067f107d6b0d77d69b9bcf30af3e409616e10a9451a5143daa1db4b8"
    assert binding.metadata["external_candidate_rpc_dependency_commit"] == "24eb2ff00cc70ac782dbfa82c59020a228164a47"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_motion_gripper_binding_remains_frozen_after_sam3_candidates() -> None:
    binding = Real2SimArtifactManifest.load(MOTION_GRIPPER_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(SAM3_DEFAULT_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag = "20261027-motion-gripper-v1"
    old_tag = "20261027-sam3-default-v1"
    current = {ref.locator: ref for ref in binding.dependency_artifacts
               if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts
                 if "-runtime-" + old_tag in ref.artifact_id}
    assert len(current) == 154 and len(inherited) == 153
    assert set(current) == set(inherited) | {"agent_gateway/profiles/real/gripper_skill.py"}
    assert sum(path.endswith(".py") for path in current) == 151
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output(
            ["git", "show", "6c8f9289d4ad672ac6fda7e1d266d58dfa297f3f:" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
        else:
            assert ref.artifact_id == "real-gripper-resources-runtime-" + tag
            assert ref.artifact_kind == "runtime_source"
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {
        "agent_gateway/profiles/mujoco/gateway_cli.py",
        "agent_gateway/profiles/mujoco/motion_deployment.py",
        "agent_gateway/profiles/mujoco/motion_factory.py",
        "agent_gateway/profiles/mujoco/motion_gateway_session.py",
        "agent_gateway/profiles/mujoco/observed_point_motion.py",
        "agent_gateway/profiles/real/gateway_cli.py",
        "agent_gateway/profiles/real/rm65_composition.py",
        "agent_gateway/skills/motion/observed_point.py",
        "teleop_runtime/skill_deploy/rm65_grasp_trajectory.py",
    }
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["new_runtime_sources"]) == ("agent_gateway/profiles/real/gripper_skill.py",)
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["fit_simulation_contract_sha256"] == fit.simulation_contract.sha256
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        SAM3_DEFAULT_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    assert binding.metadata["arm_profile_refit"] is False and binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "anchored_motion_versions_pregrasp_comparison_and_gripper_lifecycle"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == "6c8f9289d4ad672ac6fda7e1d266d58dfa297f3f"
    assert binding.metadata["offline_validation_execution_head"] == binding.code_revision
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == "f2c3445c0cd96e536dc4c731a4cad61dcf0b5ed3"
    assert binding.artifact_sha256 == "91eacc6abaf737a9c0d7a76f759d19c281733161a8df695ef339d2d0ebd4db85"
    assert ArtifactRef.from_file(MOTION_GRIPPER_BINDING_PATH,
        artifact_id="binding-file-check", artifact_kind="real2sim_runtime_binding").sha256 == "780e43c74196bb8e44a6a23fd98a0f0d6400e9057a414d0e02ca518841be7966"
    assert binding.fidelity_claims[0].status == "provisional"
    content = next(e for e in binding.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert content.status == "passed" and content.split == "validation"
    assert dict(content.metrics) == {"bound_runtime_input_file_count": 154., "bound_runtime_source_file_count": 151., "changed_runtime_source_file_count_from_previous_binding": 9., "new_runtime_source_file_count": 1., "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    offline = next(e for e in binding.validation_evidence if e.validation_id == "motion-gripper-source-consumers-" + tag)
    assert offline.status == "passed" and offline.split == "validation"
    assert dict(offline.metrics) == {"passed_cases": 572., "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    assert refs[binding.metadata["offline_validation_report_artifact_id"]].sha256 == "94fc1ebbd943550b85c5b5d05da8b05f43e8fa84123d94395bb3e6ad82db153f"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"



def test_sam3_default_binding_remains_frozen_after_motion_gripper() -> None:
    binding = Real2SimArtifactManifest.load(SAM3_DEFAULT_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(READONLY_DEFAULTS_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag = "20261027-sam3-default-v1"
    old_tag = "20261027-readonly-defaults-v1"
    current = {ref.locator: ref for ref in binding.dependency_artifacts
               if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts
                 if "-runtime-" + old_tag in ref.artifact_id}
    assert len(current) == len(inherited) == 153 and set(current) == set(inherited)
    assert sum(path.endswith(".py") for path in current) == 150
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output([
            "git", "show", "8bdf790dde02c44cfcff5188fa019fa6af6859fd:" + path])).hexdigest()
        assert ref.artifact_id == inherited[path].artifact_id.replace(old_tag, tag)
        assert ref.artifact_kind == inherited[path].artifact_kind
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {
        "agent_gateway/profiles/localization_deployment.py",
        "agent_gateway/profiles/mujoco/gateway_cli.py",
        "agent_gateway/profiles/mujoco/motion_deployment.py",
        "agent_gateway/profiles/mujoco/motion_factory.py",
        "agent_gateway/profiles/real/gateway_cli.py",
        "agent_gateway/profiles/real/rm65_composition.py",
        "deployments/perception/sam3_service.py",
    }
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["new_runtime_sources"]) == ()
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["fit_simulation_contract_sha256"] == fit.simulation_contract.sha256
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        READONLY_DEFAULTS_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    assert binding.metadata["arm_profile_refit"] is False and binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "sam3_default_and_versioned_external_localization_binding"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == "8bdf790dde02c44cfcff5188fa019fa6af6859fd"
    assert binding.metadata["offline_validation_execution_head"] == binding.code_revision
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == "015dee6f18f11e97060fb278ca1c068bca4223ec"
    assert binding.artifact_sha256 == "2df6d70730907919e61af8c682ff8c5a9c76890efa6d361977391ae9b7c2e209"
    assert ArtifactRef.from_file(SAM3_DEFAULT_BINDING_PATH,
        artifact_id="binding-file-check", artifact_kind="real2sim_runtime_binding").sha256 == "1fcd40b52c7cba40e62f5ad1c3d3f9442376b0ffc720769d6ea3b7e00475580e"
    assert binding.fidelity_claims[0].status == "provisional"
    content = next(e for e in binding.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert content.status == "passed" and content.split == "validation"
    assert dict(content.metrics) == {"bound_runtime_input_file_count": 153., "bound_runtime_source_file_count": 150., "changed_runtime_source_file_count_from_previous_binding": 7., "new_runtime_source_file_count": 0., "arm_profile_refit_count": 0., "mutated_historical_artifact_count": 0.}
    offline = next(e for e in binding.validation_evidence if e.validation_id == "sam3-default-consumers-" + tag)
    assert offline.status == "passed" and offline.split == "validation"
    assert dict(offline.metrics) == {"passed_cases": 590., "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    assert refs[binding.metadata["offline_validation_report_artifact_id"]].sha256 == "d4851c0b1870ceace39f06b37ac5c924efff0ccf359ab29ddaee7edccc58b02c"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_readonly_defaults_binding_remains_frozen_after_sam3_default() -> None:
    binding = Real2SimArtifactManifest.load(READONLY_DEFAULTS_BINDING_PATH)
    previous = Real2SimArtifactManifest.load(CHECKPOINT_INTEGRATION_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    tag = "20261027-readonly-defaults-v1"
    current = {ref.locator: ref for ref in binding.dependency_artifacts
               if "-runtime-" + tag in ref.artifact_id}
    inherited = {ref.locator: ref for ref in previous.dependency_artifacts
                 if "-runtime-20261026-checkpoint-integration-v1" in ref.artifact_id}
    assert len(current) == 153 and len(inherited) == 152
    assert sum(path.endswith(".py") for path in current) == 150
    assert set(current) == set(inherited) | {"teleop_runtime/skill_deploy/rm65_grasp_trajectory.py"}
    for path, ref in current.items():
        assert ref.sha256 == hashlib.sha256(subprocess.check_output([
            "git", "show", "b19fd84165d710bc01aa18a94c50fc50c1b12d8a:" + path])).hexdigest()
        if path in inherited:
            assert ref.artifact_id == inherited[path].artifact_id.replace("20261026-checkpoint-integration-v1", tag)
            assert ref.artifact_kind == inherited[path].artifact_kind
    changed = {path for path in inherited if current[path].sha256 != inherited[path].sha256}
    assert changed == {"agent_gateway/profiles/mujoco/gateway_cli.py", "agent_gateway/profiles/real/gateway_cli.py"}
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed))
    assert tuple(binding.metadata["new_runtime_sources"]) == ("teleop_runtime/skill_deploy/rm65_grasp_trajectory.py",)
    assert binding.resolved_timing == previous.resolved_timing == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    assert binding.simulation_contract.sha256 == previous.simulation_contract.sha256
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["fit_simulation_contract_sha256"] == fit.simulation_contract.sha256
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == previous.artifact_sha256
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        CHECKPOINT_INTEGRATION_BINDING_PATH, artifact_id="parent-check", artifact_kind="real2sim_runtime_binding").sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is True
    assert binding.metadata["runtime_behavior_change_scope"] == "readonly_cli_catalog_defaults_and_optional_bounded_tcp_arc_ordering"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"] == 'b19fd84165d710bc01aa18a94c50fc50c1b12d8a'
    assert binding.metadata["offline_validation_execution_head"] == 'b19fd84165d710bc01aa18a94c50fc50c1b12d8a'
    assert binding.environment_fingerprint["source_tree"] == binding.metadata["offline_validation_source_tree"] == 'bb3a8b9339c7d03f4aafc0a26179de959d0b732b'
    assert binding.artifact_sha256 == 'e7fdb5d558bf413535436851b9aab66ffa7b90048c3bc1345bd85ac99ddac240'
    assert ArtifactRef.from_file(READONLY_DEFAULTS_BINDING_PATH,
        artifact_id="binding-file-check", artifact_kind="real2sim_runtime_binding").sha256 == '2e050d85c5f2f34fa72e655dbfa2cf4ec39b4710d2ab14004a205680e32cd523'
    assert binding.fidelity_claims[0].status == "provisional"
    content = next(e for e in binding.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-" + tag)
    assert content.status == "passed" and content.split == "validation"
    assert dict(content.metrics) == {'bound_runtime_input_file_count': 153.0, 'bound_runtime_source_file_count': 150.0, 'changed_runtime_source_file_count_from_previous_binding': 2.0, 'new_runtime_source_file_count': 1.0, 'arm_profile_refit_count': 0.0, 'mutated_historical_artifact_count': 0.0}
    offline = next(e for e in binding.validation_evidence if e.validation_id == "readonly-defaults-consumers-" + tag)
    assert offline.status == "passed" and offline.split == "validation"
    assert dict(offline.metrics) == {"passed_cases": 601., "skipped_cases": 0., "failed_cases": 0., "error_cases": 0.}
    refs = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    assert refs[binding.metadata["offline_validation_report_artifact_id"]].sha256 == "2466618ed98732d4be629477617835a4b18375ffe9a178e8ef59a7512e564848"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_checkpoint_integration_binding_remains_frozen_after_readonly_defaults() -> None:
    binding = Real2SimArtifactManifest.load(CHECKPOINT_INTEGRATION_BINDING_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    parents = [
        Real2SimArtifactManifest.load(path)
        for path in (ARM_RUNTIME_BINDING_V25_PATH, CHECKPOINT_OBSERVER_BINDING_PATH)
    ]
    artifacts = {ref.artifact_id: ref for ref in binding.dependency_artifacts}
    tag = "20261026-checkpoint-integration-v1"
    current = {
        ref.locator: ref
        for ref in binding.dependency_artifacts
        if "-runtime-" + tag in ref.artifact_id
    }
    previous = [
        {
            ref.locator: ref
            for ref in parent.dependency_artifacts
            if "-runtime-20261026-v25" in ref.artifact_id
        }
        for parent in parents
    ]
    assert [len(part) for part in previous] == [92, 147]
    assert len(current) == 152
    assert sum(path.endswith(".py") for path in current) == 149
    assert set(current) == set(previous[0]) | set(previous[1]) | {
        "robot_motion/target_safety/checkpoint.py"
    }
    # Frozen predecessor manifests fix the full inherited scope. Neither
    # side's current-input list can disappear in this combined candidate.
    for parent_inputs in previous:
        for path, inherited in parent_inputs.items():
            ref = current[path]
            assert ref.artifact_id == inherited.artifact_id.replace("20261026-v25", tag)
            assert ref.artifact_kind == inherited.artifact_kind
    for ref in current.values():
        assert (
            ref.sha256
            == hashlib.sha256(subprocess.check_output(
                ["git", "show", f"{binding.code_revision}:{ref.locator}"],
                cwd=Path(__file__).resolve().parents[1],
            )).hexdigest()
        )
    expected_changes = [
        {
            "sim/mujoco/deploy/agent_skill_target.py",
            "sim/mujoco/deploy/runtime.py",
            "sim/rm65/actuation/core.py",
        },
        {
            "agent_gateway/profiles/mujoco/gateway_cli.py",
            "agent_gateway/profiles/mujoco/object_goal_query.py",
            "sim/mujoco/deploy/agent_skill_target.py",
            "sim/mujoco/deploy/planned_tcp_motion.py",
            "sim/mujoco/deploy/target_checkpoint.py",
        },
    ]
    for inherited, expected in zip(previous, expected_changes):
        assert {
            path
            for path, ref in inherited.items()
            if current[path].sha256 != ref.sha256
        } == expected
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(
        sorted(expected_changes[0])
    )
    assert tuple(
        binding.metadata["changed_checkpoint_observer_runtime_sources"]
    ) == tuple(sorted(expected_changes[1]))
    assert tuple(binding.metadata["new_runtime_sources"]) == tuple(
        sorted(set(current) - set(previous[0]))
    )
    assert len(binding.metadata["new_runtime_sources"]) == 60
    assert (
        binding.simulation_contract.sha256
        == ArtifactRef.from_file(
            CONTRACT_PATH,
            artifact_id="contract-check",
            artifact_kind="simulation_contract",
        ).sha256
    )
    assert (
        binding.resolved_timing
        == Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    )
    assert (
        artifacts["rm65-arm-free-motion-fit-20260721-v4"].sha256
        == ArtifactRef.from_file(
            ARM_BASELINE_PATH,
            artifact_id="fit-check",
            artifact_kind="real2sim_fit_artifact",
        ).sha256
    )
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert (
        binding.metadata["fit_simulation_contract_sha256"]
        == fit.simulation_contract.sha256
    )
    for path, parent, identity, artifact_key, file_key in (
        (
            ARM_RUNTIME_BINDING_V25_PATH,
            parents[0],
            "rm65-arm-runtime-binding-20261026-v25",
            "previous_runtime_binding_artifact_sha256",
            "previous_runtime_binding_file_sha256",
        ),
        (
            CHECKPOINT_OBSERVER_BINDING_PATH,
            parents[1],
            "rm65-arm-runtime-binding-checkpoint-observer-20261026-v1",
            "checkpoint_observer_predecessor_artifact_sha256",
            "checkpoint_observer_predecessor_file_sha256",
        ),
    ):
        digest = ArtifactRef.from_file(
            path,
            artifact_id="parent-file-check",
            artifact_kind="real2sim_runtime_binding",
        ).sha256
        assert binding.metadata[artifact_key] == parent.artifact_sha256
        assert binding.metadata[file_key] == digest == artifacts[identity].sha256
        assert artifacts[identity].locator == str(path)
    assert (
        binding.artifact_sha256
        == "99f072d45d47f1a7ee366a8a6bba83ad259d7a56ea6555a474fa3ec26622a0c4"
    )
    assert (
        ArtifactRef.from_file(
            CHECKPOINT_INTEGRATION_BINDING_PATH,
            artifact_id="integration-file-check",
            artifact_kind="real2sim_runtime_binding",
        ).sha256
        == "8e30199e20433a163ef1b4ec8d337ec6daa5c7324cd26752842f8472ed1d5032"
    )
    assert (
        binding.code_revision
        == binding.environment_fingerprint["source_commit"]
        == (binding.metadata["runtime_source_commit"])
        == "a47d7e3922ef2156ab04dc184222494ea368af5c"
    )
    assert (
        binding.metadata["offline_validation_execution_head"] == binding.code_revision
    )
    assert (
        binding.metadata["offline_validation_source_tree"]
        == binding.environment_fingerprint["source_tree"]
        == ("cef6019e5851c6382221043c569d4bc318d49c11")
    )
    assert (
        binding.metadata["mainline_input_commit"]
        == "6ecb2307b49cbaeeaa1ee28e2ccc7852af6b6773"
    )
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is True
    assert (
        binding.metadata["runtime_behavior_change_scope"]
        == "private_checkpoint_continuation_with_inactive_tcp_envelope_and_mainline_model_graph"
    )
    assert binding.fidelity_claims[0].status == "provisional"
    assert (
        next(e for e in binding.validation_evidence if e.split == "held_out").status
        == "not_run"
    )
    content = next(
        e
        for e in binding.validation_evidence
        if e.validation_id
        == "runtime-source-content-binding-integrity-checkpoint-integration-v1"
    )
    assert content.status == "passed" and content.split == "validation"
    assert dict(content.metrics) == {
        "bound_runtime_input_file_count": 152.0,
        "bound_runtime_source_file_count": 149.0,
        "changed_runtime_source_file_count_from_previous_binding": 3.0,
        "new_runtime_source_file_count": 60.0,
        "arm_profile_refit_count": 0.0,
        "mutated_historical_artifact_count": 0.0,
    }
    offline = next(
        e
        for e in binding.validation_evidence
        if e.validation_id == "checkpoint-integration-consumers-v1"
    )
    assert offline.status == "passed" and offline.split == "validation"
    assert dict(offline.metrics) == {
        "passed_cases": 218.0,
        "skipped_cases": 0.0,
        "failed_cases": 0.0,
        "error_cases": 0.0,
    }
    for key, digest in (
        (
            "offline_validation_report_artifact_id",
            "9d36a79ed11c82d65e55a329249b5b7f87dd3574911694bdba3329ff3b64a80d",
        ),
        (
            "offline_validation_binding_artifact_id",
            "4866f303f483d5100624795bbc2204f62fd029b03818b3a68e0370100d77d0b4",
        ),
    ):
        ref = artifacts[binding.metadata[key]]
        assert ref.sha256 == digest
        assert ref.locator == "artifact://" + ref.artifact_id


def test_mainline_v25_remains_frozen_after_checkpoint_integration() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V25_PATH)
    previous = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V24_PATH)
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    assert (
        binding.artifact_sha256
        == "ee264b0fc454cbd6f3ae04a0d0cb2e5f38ed6cc05f1051f26ce21365506cd8ed"
    )
    assert (
        ArtifactRef.from_file(
            ARM_RUNTIME_BINDING_V25_PATH,
            artifact_id="frozen-mainline-v25",
            artifact_kind="real2sim_runtime_binding",
        ).sha256
        == "6b9eae23fb2e84c1b4c9d58c6fb14a7ec4f8c07035ff4f7066766894f1340c1d"
    )
    assert (
        binding.code_revision
        == binding.environment_fingerprint["source_commit"]
        == (binding.metadata["runtime_source_commit"])
        == "83581f2acbdd614f2f75a32cb6996fe4fb9abddc"
    )
    assert (
        binding.metadata["previous_runtime_binding_artifact_sha256"]
        == previous.artifact_sha256
    )
    assert (
        binding.metadata["previous_runtime_binding_file_sha256"]
        == ArtifactRef.from_file(
            ARM_RUNTIME_BINDING_V24_PATH,
            artifact_id="v24-file-check",
            artifact_kind="real2sim_runtime_binding",
        ).sha256
    )
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.fidelity_claims[0].status == "provisional"
    assert (
        next(e for e in binding.validation_evidence if e.split == "held_out").status
        == "not_run"
    )


def test_checkpoint_observer_binding_remains_frozen_after_model_graph_merge() -> None:
    path = CHECKPOINT_OBSERVER_BINDING_PATH
    binding = Real2SimArtifactManifest.load(path)
    assert binding.artifact_sha256 == (
        "76fe4c1986f052cd438ff1c30461ab035c937db685e7ad05706c03efefd6d302"
    )
    assert ArtifactRef.from_file(
        path, artifact_id="frozen-checkpoint-observer-binding",
        artifact_kind="real2sim_runtime_binding",
    ).sha256 == "ed75ba721879cc066138002595c6c754141abfcf16a7ccaf6ee293a6084a67c4"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == (
        binding.metadata["runtime_source_commit"]
    ) == "db09fe45844420d70ee2635a26f85b30d89fbf07"
    assert binding.metadata["offline_validation_source_tree"] == (
        "86cd3aa93b73adc06907df98c4665fece201e86c"
    )
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(
        e for e in binding.validation_evidence if e.split == "held_out"
    ).status == "not_run"


def test_v9_arm_runtime_binding_remains_frozen_after_native_payload_extension() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V9_PATH)
    artifacts = {artifact.artifact_id: artifact for artifact in binding.dependency_artifacts}
    historical_sources = {
        artifact.locator: artifact.sha256 for artifact in binding.dependency_artifacts
        if "-runtime-20261022-v9" in artifact.artifact_id
    }
    assert binding.artifact_sha256 == "016de1e5b798f4858caba54fcf568423a62a3f3e9d25f8842076158563d1f591"
    assert ArtifactRef.from_file(
        ARM_RUNTIME_BINDING_V9_PATH, artifact_id="frozen-v9-check",
        artifact_kind="real2sim_runtime_binding",
    ).sha256 == "092b48c008aba04d647ff2fa8da5677f2dde18529a6db26dc4330053c8f71f43"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == (
        binding.metadata["runtime_source_commit"]
    ) == "e4ab4e5d0580dd4cae3d2ce8aa783979025e5c0c"
    assert artifacts["rm65-arm-mujoco-executor-runtime-20261022-v9"].sha256 == (
        "313aac56f4fcdfc17176bfc433af75776aa34f40e577caa3e8f3a8b1eb53928e"
    )
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == (
        Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V8_PATH).artifact_sha256
    )
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["fit_artifact_sha256"] == Real2SimArtifactManifest.load(ARM_BASELINE_PATH).artifact_sha256
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"
    assert tuple(binding.metadata["new_runtime_sources"]) == ("sim/mujoco/deploy/agent_skill_target.py",)
    content = next(e for e in binding.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-v9")
    assert content.status == "passed"
    assert content.metrics["bound_runtime_input_file_count"] == len(historical_sources) == 14
    assert content.metrics["bound_runtime_source_file_count"] == 13
    assert content.metrics["arm_profile_refit_count"] == content.metrics["mutated_historical_artifact_count"] == 0
    offline = next(e for e in binding.validation_evidence if e.validation_id == "offline-fixed-arm-mujoco-skills-v9")
    assert offline.status == "passed" and offline.split == "validation"
    assert dict(offline.metrics) == {"passed_cases": 1893.0, "skipped_cases": 6.0, "failed_cases": 0.0, "error_cases": 0.0}
    assert artifacts[binding.metadata["offline_validation_report_artifact_id"]].sha256 == (
        "b7e54c0751ccc76cf9a993317196611a75fcaa8b7958d1ae943caa73ef7a884e"
    )
    assert binding.metadata["offline_validation_source_tree"] == "84cc94c8134b1c66a6b4b555eaa214f00ded8505"

    assert binding.metadata["offline_validation_execution_head"] == "11c34162e819356da9e86dbfad6424f3ac3e3ddb"
    assert binding.metadata["offline_validation_execution_merge_head"] == "71faf8b1d7a1d6c4d463695aa98a00c8f1e5c9f3"


def test_v7_arm_runtime_binding_remains_frozen_after_scoped_contact_guard() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V7_PATH)
    assert binding.artifact_sha256 == "bfb52780d549c5221201fde7f33dd95079c33dafe94c40ab4cf5a7931d0fbac6"
    assert ArtifactRef.from_file(
        ARM_RUNTIME_BINDING_V7_PATH, artifact_id="frozen-v7-check",
        artifact_kind="real2sim_runtime_binding",
    ).sha256 == "1ce52c5c3cbd0a95e999efead5fa00ca2c8098b75eb76bfc896ad503b26dbbe7"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_v5_arm_runtime_binding_remains_frozen_after_final_command_admission() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V5_PATH)
    assert binding.artifact_sha256 == "9b5bc0cd6e45b8f907a921f3fee811b062efc71db93a4cec5a7ed2bd61724f76"
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(x for x in binding.validation_evidence if x.split == "held_out").status == "not_run"


def test_v4_runtime_binding_remains_recoverable() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V4_PATH)

    assert binding.artifact_sha256 == (
        "1933f1a764b95ee7eb06a6feca16ed198a72529ce788d98235da8aecd2f71346"
    )


def test_v3_runtime_binding_remains_recoverable() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V3_PATH)

    assert binding.artifact_sha256 == (
        "c8781f8b9f206133adb58eb0393f4932e9f4442bbc533206deb9479a13b693c1"
    )


def test_v1_runtime_binding_camera_dependency_remains_recoverable() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V1_PATH)
    artifacts = {
        artifact.artifact_id: artifact
        for artifact in binding.dependency_artifacts
    }

    assert artifacts["rm65-camera-registry-runtime-20260724-v1"].sha256 == (
        ArtifactRef.from_file(
            CAMERA_REGISTRY_V1_ARCHIVE_PATH,
            artifact_id="camera-registry-v1-archive-check",
            artifact_kind="camera_registry",
        ).sha256
    )
    assert binding.artifact_sha256 == (
        "ee37ac1f0128e6254e2ea6b557adda098aad0afa189c84b454fc3c16451c8820"
    )


def test_v2_runtime_binding_remains_recoverable() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V2_PATH)

    assert binding.artifact_sha256 == (
        "ca0d2e58c6799557f9ed1f781da9a4fc7e9f5b9c28ccff0caad99b8dd613a60b"
    )


def test_current_runtime_binding_records_exact_scene_contract_migration() -> None:
    current = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    archived = json.loads(CONTRACT_V2_ARCHIVE_PATH.read_text(encoding="utf-8"))

    assert current["scene"]["urdf_path"] == _CAD_SCENE_URDF
    assert archived["scene"]["urdf_path"].endswith(
        "rm65_dual_arm_wrist_cameras_v3_head_camera.urdf"
    )
    assert archived["scene"]["collision"]["excluded_body_pairs"] == []
    assert len(current["scene"]["collision"]["excluded_body_pairs"]) == 24

    current["scene"]["urdf_path"] = archived["scene"]["urdf_path"]
    current["scene"]["collision"]["excluded_body_pairs"] = []
    assert current == archived


def test_current_camera_registry_change_is_exactly_documented_v3_migration() -> None:
    archived_registry = json.loads(
        CAMERA_REGISTRY_V1_ARCHIVE_PATH.read_text(encoding="utf-8")
    )
    current_registry = json.loads(CAMERA_REGISTRY_PATH.read_text(encoding="utf-8"))

    archived_left_serial = archived_registry["cameras"]["left_wrist"]["serial"]
    archived_right_serial = archived_registry["cameras"]["right_wrist"]["serial"]
    assert current_registry["cameras"]["left_wrist"]["serial"] == (
        archived_right_serial
    )
    assert current_registry["cameras"]["right_wrist"]["serial"] == (
        archived_left_serial
    )

    archived_registry["cameras"]["left_wrist"]["serial"] = archived_right_serial
    archived_registry["cameras"]["right_wrist"]["serial"] = archived_left_serial

    for role, (frame_id, parent_frame) in _CAMERA_FRAME_BY_ROLE.items():
        associations = archived_registry["cameras"][role]["extrinsics"][
            "frames"
        ][frame_id]["placements"]
        associations.append(
            {
                "urdf": _CAD_SCENE_URDF,
                "frame": frame_id,
                "parent_frame": parent_frame,
            }
        )
    assert current_registry == archived_registry


def test_historical_v3_baseline_preserves_failed_held_out_result() -> None:
    manifest = Real2SimArtifactManifest.load(ARM_V3_BASELINE_PATH)
    held_out = next(
        evidence
        for evidence in manifest.validation_evidence
        if evidence.split == "held_out"
    )

    assert held_out.status == "failed"
    assert held_out.metrics["session_count"] == 4.0
    assert held_out.metrics["max_effective_delay_error_sec"] == 0.008


def test_historical_v2_baseline_preserves_failed_held_out_result() -> None:
    manifest = Real2SimArtifactManifest.load(ARM_V2_BASELINE_PATH)
    held_out = next(
        evidence
        for evidence in manifest.validation_evidence
        if evidence.split == "held_out"
    )

    assert held_out.status == "failed"
    assert held_out.metrics["max_effective_delay_error_sec"] == 0.008


def test_historical_v1_baseline_preserves_failed_held_out_result() -> None:
    manifest = Real2SimArtifactManifest.load(ARM_V1_BASELINE_PATH)
    held_out = next(
        evidence
        for evidence in manifest.validation_evidence
        if evidence.split == "held_out"
    )

    assert held_out.status == "failed"
    assert held_out.metrics["max_effective_delay_error_sec"] == 0.009


def test_v8_arm_runtime_binding_remains_frozen_after_single_arm_skill_extension() -> None:
    # Historical contact evidence binds its own source revision, not mutable current files.
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V8_PATH)
    artifacts = {artifact.artifact_id: artifact for artifact in binding.dependency_artifacts}
    assert binding.artifact_sha256 == "246bd5a0273cac92e637b307513e25810f5bda22bfc3879ca3086824df381737"
    assert ArtifactRef.from_file(
        ARM_RUNTIME_BINDING_V8_PATH, artifact_id="frozen-v8-check",
        artifact_kind="real2sim_runtime_binding",
    ).sha256 == "48ee5ff7dc5b54c7d98e3ae2cb7b7707237a40dab742e10a473d771b8b7fea68"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == (
        binding.metadata["runtime_source_commit"]
    ) == "910c6b76636c894ef08fbd6af762e8587126ba71"
    assert binding.metadata["runtime_behavior_change_scope"] == "offline_mujoco_target_bound_grasp_contact_guard"
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["fit_artifact_sha256"] == Real2SimArtifactManifest.load(ARM_BASELINE_PATH).artifact_sha256
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == (
        Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V7_PATH).artifact_sha256
    )
    assert artifacts["rm65-planned-tcp-motion-runtime-20261018-v8"].sha256 == (
        "ebcdf78dcae391bdfd3d008378279f9f112d58072020899ad6b9e921c95f1e81"
    )
    assert artifacts["mujoco-scoped-grasp-contact-offline-validation-20261018-v8"].sha256 == (
        "4c75c7682d9f4d0bd87542af50279a26f53b6f8db3444c4b5d5cdf048384dde0"
    )
    assert binding.fidelity_claims[0].status == "provisional"
    assert next(e for e in binding.validation_evidence if e.split == "held_out").status == "not_run"


def test_v10_binding_remains_frozen_after_native_chain_extension() -> None:
    previous = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V10_PATH)
    assert previous.artifact_sha256 == "c341a48710ce7743941a0980b820c0627ec98dba8a27bb7df574d5418736a880"
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_V10_PATH,
        artifact_id="frozen-v10-check", artifact_kind="real2sim_runtime_binding").sha256 == "5538d158d25ee13bb7e81e515e1aff4e709c2534c38c79fe94dde81338e4841c"
    assert previous.metadata["arm_profile_refit"] is False
    assert previous.code_revision == previous.metadata["runtime_source_commit"] == "7d66cb491d44c891ef41678f066b80d072f87315"
    assert previous.fidelity_claims[0].status == "provisional"
    content = next(e for e in previous.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-v10")
    assert dict(content.metrics) == {"bound_runtime_source_file_count":20., "bound_runtime_input_file_count":21.,
        "changed_runtime_source_file_count_from_previous_binding":3., "new_runtime_source_file_count":7.,
        "arm_profile_refit_count":0., "mutated_historical_artifact_count":0.}
    offline = next(e for e in previous.validation_evidence if e.validation_id == "offline-object-follow-payload-consumers-v10")
    assert dict(offline.metrics) == {"passed_cases":115., "skipped_cases":0., "failed_cases":0., "error_cases":0.}
    artifacts = {a.artifact_id:a for a in previous.dependency_artifacts}
    assert artifacts[previous.metadata["offline_validation_report_artifact_id"]].sha256 == "e1ef7b83610f7b4e8f5055287f7784a7ed182274494215470e7824e4ad82c2e7"
    assert previous.environment_fingerprint["source_tree"] == previous.metadata["offline_validation_source_tree"] == "c5b5fa16e7cdd202e30990e4b9c3c6dd086d9166"
    assert next(e for e in previous.validation_evidence if e.split == "held_out").status == "not_run"


def test_v12_binding_remains_frozen_after_collision_query_extension() -> None:
    previous = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V12_PATH)
    assert previous.artifact_sha256 == "ed87161262be60cb983fc14c3a583242f267930ae54577949763e6f511d2f381"
    assert ArtifactRef.from_file(
        ARM_RUNTIME_BINDING_V12_PATH, artifact_id="frozen-v12-check",
        artifact_kind="real2sim_runtime_binding").sha256 == 'eb186b84b763e95aa9007bec06938f743f3c9d96f299635aa8a1b9deb25749cf'
    assert previous.code_revision == "73b40a4b177ffe59026b40ff193b8e4ef93b7b7d"
    content = next(e for e in previous.validation_evidence
        if e.validation_id == "runtime-source-content-binding-integrity-v12")
    assert content.metrics["bound_runtime_input_file_count"] == 22
    offline = next(e for e in previous.validation_evidence
        if e.validation_id == "offline-native-chain-consumers-v12")
    assert dict(offline.metrics) == {"passed_cases": 304.0, "skipped_cases": 0.0, "failed_cases": 0.0, "error_cases": 0.0}
    artifacts = {item.artifact_id: item for item in previous.dependency_artifacts}
    assert artifacts[previous.metadata["offline_validation_report_artifact_id"]].sha256 == "99dd8b4b6cf21d72113a660d7076014097c4ae7648dce3e92fa38c7cca8298df"


def test_v10_binding_remains_frozen_after_read_only_collision_query() -> None:
    previous = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V10_PATH)
    assert previous.artifact_sha256 == "c341a48710ce7743941a0980b820c0627ec98dba8a27bb7df574d5418736a880"
    assert previous.metadata["arm_profile_refit"] is False
    content = next(e for e in previous.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-v10")
    assert content.metrics["bound_runtime_source_file_count"] == 20
    assert content.metrics["bound_runtime_input_file_count"] == 21
    assert content.metrics["changed_runtime_source_file_count_from_previous_binding"] == 3
    assert content.metrics["new_runtime_source_file_count"] == 7
    offline = next(e for e in previous.validation_evidence if e.validation_id == "offline-object-follow-payload-consumers-v10")
    assert dict(offline.metrics) == {"passed_cases":115.,"skipped_cases":0.,"failed_cases":0.,"error_cases":0.}
    assert next(e for e in previous.validation_evidence if e.split == "held_out").status == "not_run"


def test_v13_binding_remains_frozen_after_read_only_inventory_extension() -> None:
    previous = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V13_PATH)
    assert previous.artifact_sha256 == "f2dcb2d05ae30eb0848acbd4d43316838a364e6c7e88cf774d21eae917a0cc99"
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_V13_PATH,
        artifact_id="frozen-v13-check", artifact_kind="real2sim_runtime_binding").sha256 == '17823a6795c15a7ebf72ec214fcfa3bddd90fb8e16fc74d3d89e5bb80df49195'
    assert previous.code_revision == "b74fd69c52ab4bd107fd3860dd8c253f68cfc3b3"
    assert previous.metadata["runtime_behavior_change"] is True
    content = next(e for e in previous.validation_evidence
        if e.validation_id == "runtime-source-content-binding-integrity-v13")
    assert content.metrics["bound_runtime_input_file_count"] == 22
    offline = next(e for e in previous.validation_evidence
        if e.validation_id == "offline-frozen-grasp-consumers-v13")
    assert dict(offline.metrics) == {"passed_cases": 285.0, "skipped_cases": 0.0, "failed_cases": 0.0, "error_cases": 0.0}
    artifacts = {item.artifact_id: item for item in previous.dependency_artifacts}
    assert artifacts[previous.metadata["offline_validation_report_artifact_id"]].sha256 == "3e73983f3bf1b06409239a540dd69b9a26ff7e3acd9884fbbfcfabf17a9b084c"


def test_v14_binding_remains_frozen_after_independent_skill_extension() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V14_PATH)
    assert binding.artifact_sha256 == '17a6ee44bd1753832d7e06a47574891e3d36db76e496b770fa779b778f2ce659'
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_V14_PATH,
        artifact_id="frozen-v14-check", artifact_kind="real2sim_runtime_binding").sha256 == 'edbe370d312a8a9ddb468c709e9fdab385e3dd632210e601c5bb60164ae7bc67'
    assert binding.metadata["runtime_source_commit"] == 'b978c73d386bab608a0febb5ba1ba51e9cafded6'
    assert binding.metadata["arm_profile_refit"] is False


def test_v15_binding_remains_frozen_after_observed_skill_extension() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V15_PATH)
    assert binding.artifact_sha256 == '64caf100e77b66b42d5db14c1349bdad0e00d6be5cde858736e178a534cdae5c'
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_V15_PATH,
        artifact_id="frozen-v15-check", artifact_kind="real2sim_runtime_binding").sha256 == 'd1bdc45d9b97cae9b751c178c43d0479614dc11c4a1ce00ef8e6bc93a19583f5'
    assert binding.metadata["runtime_source_commit"] == '88c9cb924166246e89b9f7f5922291f2ea39702d'
    assert binding.metadata["arm_profile_refit"] is False


def test_v16_binding_remains_frozen_after_paired_rgbd_extension() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V16_PATH)
    assert binding.artifact_sha256 == 'c87940fc9d557d55aa06586f9219684184dcdd668788a015da7366c145e7bb29'
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_V16_PATH,
        artifact_id="frozen-v16-check", artifact_kind="real2sim_runtime_binding").sha256 == 'bdce9be991355646e0044f8ba1044c1f5c2d26a479bfa187d2f884ae65f3b8d1'
    assert binding.metadata["runtime_source_commit"] == 'addd6248ac87d0402686b4481671cb875df2d15d'
    assert binding.metadata["arm_profile_refit"] is False


def test_v17_binding_remains_frozen_after_segmentation_extension() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V17_PATH)
    assert binding.artifact_sha256 == '9a74714f536772aefb3d26e41f3760a34b03e2ee26f126cb7aca9af88b9c4761'
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_V17_PATH,
        artifact_id="frozen-v17-check", artifact_kind="real2sim_runtime_binding").sha256 == '28bb9dcccf33fe31240e97357202f21592da80046dcec4840bd29c183b1a8697'
    assert binding.metadata["runtime_source_commit"] == 'e82e3d6c860ca722597d8304ed849881c2501f89'
    assert binding.metadata["arm_profile_refit"] is False


def test_v18_binding_remains_frozen_after_frozen_query_extension() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V18_PATH)
    assert binding.artifact_sha256 == '7f43883e5ebdd7e7bee5db2519c90809274630ffc41ef66c456a9032f3464bcc'
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_V18_PATH,
        artifact_id="frozen-v18-check", artifact_kind="real2sim_runtime_binding").sha256 == '486f5e8ddf1ea043919a1d260003b872b13b3f23fa5d93188aede09ea244381b'
    assert binding.metadata["runtime_source_commit"] == '31aa9fc29028644d79db0fc137dbf40cd6232017'
    assert binding.metadata["arm_profile_refit"] is False


def test_v19_binding_remains_frozen_after_query_anchor_extension() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V19_PATH)
    assert binding.artifact_sha256 == '37f9ced7a3904f0cd0aeb2e60c0aae004125f9d8362d7bd3914cca65a144cd4c'
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_V19_PATH,
        artifact_id="frozen-v19-check", artifact_kind="real2sim_runtime_binding").sha256 == '2006daae911cd81a64b073fa0b766c8daf92feced8088d2a4b9a0db96c4e718d'
    assert binding.metadata["runtime_source_commit"] == '9bd4a748431dcda9163faa4fbc827769faf16dca'
    assert binding.metadata["arm_profile_refit"] is False


def test_v20_binding_remains_frozen_after_query_anchor_extension() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V20_PATH)
    assert binding.artifact_sha256 == 'bbb5b8f88d8c3bdc6cf375ee9e7b3c951bba3bd210f129a9de4edc9bdb6ede86'
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_V20_PATH,
        artifact_id="frozen-v20-check", artifact_kind="real2sim_runtime_binding").sha256 == '1eb876c2a58425e23f49cd727a4d40bcf51cfd09d21138a3133ba4d3396f9417'
    assert binding.metadata["runtime_source_commit"] == '78b09d63ff9e1efc42b9ca7738a00d76107d8c68'
    assert binding.metadata["arm_profile_refit"] is False


def test_v21_binding_remains_frozen_after_localization_extension() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V21_PATH)
    assert binding.artifact_sha256 == 'c87318aef94c78659a5543a0432ae82766ec04df30a0719c5e3cfd0531f3ff7a'
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_V21_PATH,
        artifact_id="frozen-v21-check", artifact_kind="real2sim_runtime_binding").sha256 == 'b08732bb1948a3f5592e20fe9353a78851850c51fd987373ad3252495252785b'
    assert binding.metadata["runtime_source_commit"] == 'd1a54844dcc27a9f24f48e7dd77788bbb11c7319'
    assert binding.metadata["arm_profile_refit"] is False


def test_v22_binding_remains_frozen_after_continuation_and_archival() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V22_PATH)
    assert binding.artifact_sha256 == '240e8b54284048ae3de66066cf7a1b27c138bba12b554a19a014392991b2d5bb'
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_V22_PATH,
        artifact_id="frozen-v22-check", artifact_kind="real2sim_runtime_binding").sha256 == 'dfb07cb6f0159796e677ea03fa2783e6d372a54490502c75612112cc5c249fd4'
    assert binding.metadata["runtime_source_commit"] == 'ab5de19b3dcda8cc66bae9a85f4916d0d7f6a28a'
    assert binding.metadata["arm_profile_refit"] is False


def test_v23_binding_remains_frozen_after_controlstation_migration() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V23_PATH)
    assert binding.artifact_sha256 == '170e1e96802d342b1b52d98a2be0e6181fac11a6b44a50d5e2067fa5b1a16236'
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_V23_PATH,
        artifact_id="frozen-v23-check", artifact_kind="real2sim_runtime_binding").sha256 == '3044ac2565ee102e5f594c91c90213847c64aa4298dc297739f32ea7d83faff1'
    assert binding.metadata["runtime_source_commit"] == '6b36561dcefa55fe7f5c8e7dd31ea8948a215388'
    assert binding.metadata["arm_profile_refit"] is False


def test_v24_binding_remains_frozen_after_model_graph_integration() -> None:
    binding = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V24_PATH)
    assert binding.artifact_sha256 == '55dd077ac16e164be790d7e35656c05650b3af2c8077e9a844bf8495d03e795e'
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_V24_PATH,
        artifact_id="frozen-v24-check", artifact_kind="real2sim_runtime_binding").sha256 == '475e7e045be44493b9ca6d35793ee1240b5b47a249a33181b77c6ff694475101'
    assert binding.metadata["runtime_source_commit"] == '9156ae01d7992c042666101304ea8b83f7787395'
    assert binding.metadata["arm_profile_refit"] is False
