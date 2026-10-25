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
    "rm65_arm_free_motion_20260721_v4_runtime_binding_20261025_v19.json"
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
    fit = Real2SimArtifactManifest.load(ARM_BASELINE_PATH)
    previous = Real2SimArtifactManifest.load(ARM_RUNTIME_BINDING_V18_PATH)
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
    assert artifacts["rm65-arm-profile-runtime-20261025-v19"].sha256 == (
        ArtifactRef.from_file(
            ARM_PROFILE_PATH,
            artifact_id="profile-check",
            artifact_kind="actuation_profile",
        ).sha256
    )
    assert artifacts["rm65-arm-actuator-core-runtime-20261025-v19"].sha256 == (
        ArtifactRef.from_file(
            ARM_CORE_PATH,
            artifact_id="core-check",
            artifact_kind="sim_actuator_core_source",
        ).sha256
    )
    assert artifacts["rm65-arm-mujoco-executor-runtime-20261025-v19"].sha256 == (
        ArtifactRef.from_file(
            ARM_MUJOCO_ADAPTER_PATH,
            artifact_id="executor-check",
            artifact_kind="sim_backend_adapter_source",
        ).sha256
    )
    assert artifacts["rm65-camera-registry-runtime-20261025-v19"].sha256 == (
        ArtifactRef.from_file(
            CAMERA_REGISTRY_PATH,
            artifact_id="camera-registry-check",
            artifact_kind="camera_registry",
        ).sha256
    )
    assert artifacts["rm65-arm-runtime-binding-20260922-v5"].sha256 == (
        ArtifactRef.from_file(
            ARM_RUNTIME_BINDING_V5_PATH,
            artifact_id="runtime-binding-v5-check",
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
    assert binding.metadata["runtime_behavior_change"] is True
    assert (
        binding.metadata["runtime_behavior_change_scope"]
        == "bounded_frozen_native_queries_and_calibrated_tcp_ik"
    )
    assert binding.metadata["runtime_source_commit"] == (
        "9bd4a748431dcda9163faa4fbc827769faf16dca"
    )
    assert binding.fidelity_claims[0].status == "provisional"
    held_out = next(
        evidence
        for evidence in binding.validation_evidence
        if evidence.split == "held_out"
    )
    assert held_out.status == "not_run"

    assert artifacts["rm65-arm-runtime-binding-20261017-v7"].sha256 == (
        ArtifactRef.from_file(
            ARM_RUNTIME_BINDING_V7_PATH, artifact_id="previous-binding-check",
            artifact_kind="real2sim_runtime_binding",
        ).sha256
    )
    for label in (
        "command-guard", "envelope-adapter", "position-rig", "position-assembly",
        "collision-oracle", "collision-projection", "collision-result",
        "grasp-contact", "planned-tcp-motion", "agent-skill-target",
        "gripper-executor", "collision-policy", "contact-timing",
        "grasped-object-observer", "payload-motion", "object-release", "object-transport",
    ):
        artifact = artifacts[f"rm65-{label}-runtime-20261025-v19"]
        assert artifact.sha256 == ArtifactRef.from_file(
            artifact.locator, artifact_id="command-source-check",
            artifact_kind="sim_command_admission_source",
        ).sha256

    prepared = artifacts["rm65-native-prepared-skill-runtime-20261025-v19"]
    assert prepared.locator == "agent_gateway/profiles/mujoco/motion_skill.py"
    assert prepared.sha256 == ArtifactRef.from_file(
        prepared.locator, artifact_id="prepared-consumer-check",
        artifact_kind="sim_native_skill_consumer_source",
    ).sha256

    assert binding.artifact_sha256 == "37f9ced7a3904f0cd0aeb2e60c0aae004125f9d8362d7bd3914cca65a144cd4c"
    assert binding.code_revision == binding.environment_fingerprint["source_commit"] == binding.metadata["runtime_source_commit"]
    assert binding.metadata["previous_runtime_binding_file_sha256"] == ArtifactRef.from_file(
        ARM_RUNTIME_BINDING_V18_PATH, artifact_id="previous-v18-file-check",
        artifact_kind="real2sim_runtime_binding",
    ).sha256
    assert artifacts["rm65-arm-runtime-binding-20261025-v18"].sha256 == binding.metadata["previous_runtime_binding_file_sha256"]
    previous_sources = {
        artifact.locator: artifact.sha256 for artifact in previous.dependency_artifacts
        if "-runtime-20261025-v18" in artifact.artifact_id
    }
    current_sources = {
        artifact.locator: artifact.sha256 for artifact in binding.dependency_artifacts
        if "-runtime-20261025-v19" in artifact.artifact_id
    }
    new_sources = {'robot_motion/position_control/offline_waypoint_trajectory.py', 'robot_motion/target_safety/runtime.py'}
    assert set(current_sources) - set(previous_sources) == new_sources
    assert set(previous_sources) - set(current_sources) == set()
    changed_sources = {'robot_motion/rm65/single_arm_ik.py', 'sim/mujoco/deploy/planned_tcp_motion.py'}
    assert {path for path, digest in previous_sources.items() if current_sources[path] != digest} == changed_sources
    assert tuple(binding.metadata["changed_previous_runtime_sources"]) == tuple(sorted(changed_sources))
    assert tuple(binding.metadata["new_runtime_sources"]) == tuple(sorted(new_sources))
    content = next(e for e in binding.validation_evidence if e.validation_id == "runtime-source-content-binding-integrity-v19")
    assert content.status == "passed"
    assert content.metrics["bound_runtime_input_file_count"] == len(current_sources) == 46
    assert content.metrics["bound_runtime_source_file_count"] == 45
    assert content.metrics["changed_runtime_source_file_count_from_previous_binding"] == len(changed_sources) == 2
    assert content.metrics["new_runtime_source_file_count"] == len(new_sources) == 2
    assert content.metrics["arm_profile_refit_count"] == content.metrics["mutated_historical_artifact_count"] == 0
    offline = next(e for e in binding.validation_evidence if e.validation_id == "offline-frozen-query-consumers-v19")
    assert offline.status == "passed" and offline.split == "validation"
    assert dict(offline.metrics) == {'passed_cases': 320.0, 'skipped_cases': 0.0, 'failed_cases': 0.0, 'error_cases': 0.0}
    assert artifacts[binding.metadata["offline_validation_report_artifact_id"]].sha256 == (
        "812ce0517296456c5d98770836fb1180e85340a60d1c32616d50129826901dc8"
    )
    assert binding.metadata["offline_validation_source_tree"] == "36644ac0d3a73135e5d8b66762219e699fea1c16"
    assert binding.metadata["offline_validation_execution_head"] == binding.metadata["runtime_source_commit"]
    for label, path in {'single-arm-ik': 'robot_motion/rm65/single_arm_ik.py', 'tcp-pose-envelope-core': 'robot_motion/target_safety/tcp_pose_envelope.py', 'motion-application': 'agent_gateway/profiles/mujoco/motion_application.py', 'deploy-runtime-core': 'sim/mujoco/deploy/runtime.py'}.items():
        artifact = artifacts[f"rm65-{label}-runtime-20261025-v19"]
        assert artifact.locator == path
        assert artifact.sha256 == ArtifactRef.from_file(
            path, artifact_id="new-core-source-check", artifact_kind=artifact.artifact_kind,
        ).sha256
    for label, path in {'observed-joint-input': 'robot_motion/observed_planning.py', 'observed-collision-model': 'robot_motion/rm65/obstacle_collision.py', 'observed-scene': 'agent_gateway/profiles/mujoco/motion_observed_scene.py', 'retained-observed-point': 'agent_gateway/profiles/observed_target.py', 'observed-point-consumer': 'agent_gateway/profiles/mujoco/observed_point_motion.py', 'observed-point-contract': 'agent_gateway/skills/motion/observed_point.py', 'rgbd-source': 'sim/mujoco/depth_snapshot.py', 'depth-capture-conversion': 'agent_gateway/profiles/mujoco/back_projection.py'}.items():
        artifact = artifacts[f"rm65-{label}-runtime-20261025-v19"]
        assert artifact.locator == path
        assert artifact.sha256 == ArtifactRef.from_file(
            path, artifact_id="observed-source-check", artifact_kind=artifact.artifact_kind,
        ).sha256
    for label, path in {'correlated-rgbd-contract': 'agent_gateway/profiles/back_projection.py', 'paired-rgbd-codec': 'agent_gateway/profiles/paired_rgbd.py', 'paired-rgbd-capture': 'agent_gateway/profiles/rgbd_capture.py', 'paired-rgbd-skill-contract': 'agent_gateway/skills/perception/rgbd_schemas.py', 'paired-rgbd-artifact-contract': 'dataset_contracts/perception_artifacts.py'}.items():
        artifact = artifacts[f"rm65-{label}-runtime-20261025-v19"]
        assert artifact.locator == path
        assert artifact.sha256 == ArtifactRef.from_file(
            path, artifact_id="paired-source-check", artifact_kind=artifact.artifact_kind,
        ).sha256
    for label, path in {'segmentation-provider': 'agent_gateway/profiles/segmentation.py', 'segmentation-batch': 'agent_gateway/profiles/segmentation_batch.py', 'segmentation-definitions': 'agent_gateway/skills/perception/definition.py', 'segmentation-schemas': 'agent_gateway/skills/perception/segmentation_schemas.py', 'segmentation-session': 'agent_gateway/profiles/mujoco/motion_gateway_session.py'}.items():
        artifact = artifacts[f"rm65-{label}-runtime-20261025-v19"]
        assert artifact.locator == path
        assert artifact.sha256 == ArtifactRef.from_file(
            path, artifact_id="paired-source-check", artifact_kind=artifact.artifact_kind,
        ).sha256
    for label, path in {'waypoint-cancellation': 'robot_motion/position_control/offline_waypoint_trajectory.py', 'tcp-envelope-query': 'robot_motion/target_safety/runtime.py'}.items():
        artifact = artifacts[f"rm65-{label}-runtime-20261025-v19"]
        assert artifact.locator == path
        assert artifact.sha256 == ArtifactRef.from_file(
            path, artifact_id="paired-source-check", artifact_kind=artifact.artifact_kind,
        ).sha256







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
