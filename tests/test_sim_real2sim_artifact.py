from __future__ import annotations

import json
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
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    previous = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V3_PATH)
    artifacts = {
        artifact.artifact_id: artifact
        for artifact in binding.dependency_artifacts
    }

    assert binding.simulation_contract.sha256 == ArtifactRef.from_file(
        CONTRACT_PATH,
        artifact_id="contract-check",
        artifact_kind="simulation_contract",
    ).sha256
    assert artifacts["rm65-arm-free-motion-fit-20260721-v4"].sha256 == (
        ArtifactRef.from_file(
            ARM_BASELINE_PATH,
            artifact_id="fit-check",
            artifact_kind="real2sim_fit_artifact",
        ).sha256
    )
    assert artifacts["rm65-arm-profile-runtime-20260829-v4"].sha256 == (
        ArtifactRef.from_file(
            ARM_PROFILE_PATH,
            artifact_id="profile-check",
            artifact_kind="actuation_profile",
        ).sha256
    )
    assert artifacts["rm65-arm-actuator-core-runtime-20260829-v4"].sha256 == (
        ArtifactRef.from_file(
            ARM_CORE_PATH,
            artifact_id="core-check",
            artifact_kind="sim_actuator_core_source",
        ).sha256
    )
    assert artifacts["rm65-arm-mujoco-executor-runtime-20260829-v4"].sha256 == (
        ArtifactRef.from_file(
            ARM_MUJOCO_ADAPTER_PATH,
            artifact_id="executor-check",
            artifact_kind="sim_backend_adapter_source",
        ).sha256
    )
    assert artifacts["rm65-camera-registry-runtime-20260829-v4"].sha256 == (
        ArtifactRef.from_file(
            CAMERA_REGISTRY_PATH,
            artifact_id="camera-registry-check",
            artifact_kind="camera_registry",
        ).sha256
    )
    assert artifacts["rm65-arm-runtime-binding-20260822-v3"].sha256 == (
        ArtifactRef.from_file(
            ARM_RUNTIME_BINDING_V3_PATH,
            artifact_id="runtime-binding-v3-check",
            artifact_kind="real2sim_runtime_binding",
        ).sha256
    )
    assert binding.resolved_timing == (
        Real2SimArtifactManifest.resolved_timing_from_contract_file(CONTRACT_PATH)
    )
    assert binding.metadata["fit_artifact_sha256"] == fit.artifact_sha256
    assert binding.metadata["previous_runtime_binding_artifact_sha256"] == (
        previous.artifact_sha256
    )
    assert (
        binding.metadata["fit_simulation_contract_sha256"]
        == fit.simulation_contract.sha256
    )
    assert binding.metadata["arm_profile_refit"] is False
    assert binding.metadata["runtime_behavior_change"] is False
    assert binding.metadata["runtime_behavior_change_scope"] == "none_evidence_only"
    assert binding.metadata["runtime_source_commit"] == (
        "4c56650070a05f8a7f3c97b9fe6851886fd9b4c4"
    )
    assert binding.fidelity_claims[0].status == "provisional"
    held_out = next(
        evidence
        for evidence in binding.validation_evidence
        if evidence.split == "held_out"
    )
    assert held_out.status == "not_run"


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
