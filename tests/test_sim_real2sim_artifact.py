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
        assert ref.sha256 == ArtifactRef.from_file(path,
            artifact_id="current-input-check", artifact_kind=ref.artifact_kind).sha256
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
    assert ArtifactRef.from_file(ARM_RUNTIME_BINDING_PATH,
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
