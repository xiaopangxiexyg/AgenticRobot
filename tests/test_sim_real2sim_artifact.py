from __future__ import annotations

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
ARM_SOURCE_INDEX_PATH = Path(
    "sim/rm65/real2sim/artifacts/rm65_arm_all_valid_fit_20260721_v4_sources.json"
)
ARM_BASELINE_PATH = Path(
    "sim/rm65/real2sim/artifacts/rm65_arm_free_motion_20260721_v4.json"
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


def test_current_arm_baseline_is_bound_to_exact_sim_inputs_and_stays_provisional() -> None:
    manifest = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    artifacts = {
        artifact.artifact_id: artifact
        for artifact in manifest.dependency_artifacts
    }

    assert manifest.simulation_contract.sha256 == ArtifactRef.from_file(
        CONTRACT_PATH,
        artifact_id="contract-check",
        artifact_kind="simulation_contract",
    ).sha256
    assert artifacts["rm65-arm-profile-source-20260721-v4"].sha256 == (
        ArtifactRef.from_file(
            ARM_PROFILE_PATH,
            artifact_id="profile-check",
            artifact_kind="actuation_profile",
        ).sha256
    )
    assert artifacts["rm65-arm-actuator-core-source-20260721-v4"].sha256 == (
        ArtifactRef.from_file(
            ARM_CORE_PATH,
            artifact_id="core-check",
            artifact_kind="sim_actuator_core_source",
        ).sha256
    )
    assert artifacts["rm65-arm-mujoco-adapter-source-20260721-v4"].sha256 == (
        ArtifactRef.from_file(
            ARM_MUJOCO_ADAPTER_PATH,
            artifact_id="adapter-check",
            artifact_kind="sim_backend_adapter_source",
        ).sha256
    )
    assert artifacts[
        "rm65-arm-all-valid-fit-source-index-20260721-v4"
    ].sha256 == ArtifactRef.from_file(
        ARM_SOURCE_INDEX_PATH,
        artifact_id="source-index-check",
        artifact_kind="fit_source_index",
    ).sha256
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


def test_old_real2sim_import_path_is_a_compatibility_alias() -> None:
    from sim.real2sim import Real2SimArtifactManifest as CompatibilityManifest

    assert CompatibilityManifest is Real2SimArtifactManifest
