from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import pytest

from control_station.composition import (
    CS3B_OPENPI_RTC_SOURCE_PREFLIGHT_MANIFEST_ID_V1,
    assess_openpi_rtc_passive_source,
    build_cs3b_openpi_rtc_source_preflight_manifest,
)
from dataset_contracts.policy_artifact import (
    PolicyArtifactContract,
    PolicyCameraInputContract,
)
from dataset_contracts.v22 import PolicyIOLayout
from deployments.openpi.client.preflight import preflight_policy_artifact
from deployments.openpi.client.rtc import (
    OpenPIFinalChunkRtcBackend,
    OpenPIFinalChunkRtcConfig,
)
from deployments.openpi.client.policy import OPENPI_LOCAL_PROTOCOL_VERSION
from teleop_runtime.deploy.rtc_step_fanout import RtcFrontendStepSourceBinding


class _MetadataOnlyTransport:
    def __init__(self, metadata: Mapping[str, Any]) -> None:
        self._metadata = dict(metadata)
        self.metadata_read_count = 0
        self.infer_count = 0

    @property
    def metadata(self) -> dict[str, Any]:
        self.metadata_read_count += 1
        return dict(self._metadata)

    def infer(self, _observation: dict[str, Any]) -> Mapping[str, Any]:
        self.infer_count += 1
        raise AssertionError("CS3b-a preflight must not perform inference")

    def reset(self) -> None:
        raise AssertionError("CS3b-a preflight must not reset transport")


def _artifact() -> PolicyArtifactContract:
    return PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id="cs3b-openpi-rtc-unit",
        checkpoint_step=123,
        state_key="robot_state",
        state_discrete_key="robot_state_tokens",
        action_key="robot_actions",
        action_discrete_key="robot_action_tokens",
        camera_input=PolicyCameraInputContract(inputs=(), required=False),
        metadata={"fixture": "cs3b-openpi-rtc-source-preflight"},
    )


def _server_metadata(artifact: PolicyArtifactContract) -> dict[str, Any]:
    layout = artifact.policy_io_layout
    return {
        "protocol_version": OPENPI_LOCAL_PROTOCOL_VERSION,
        "deploy_schema_version": artifact.deploy_schema_version,
        "model_id": artifact.model_id,
        "deployment": {"checkpoint_step": artifact.checkpoint_step},
        "model": {
            "action_horizon": 4,
            "action_discrete_dim": layout.action_discrete_dim,
            "policy_io_layout_hash": layout.layout_hash,
        },
        "local_robot": {
            "input_state_key": artifact.state_key,
            "input_state_dim": artifact.state_input_dim,
            "input_state_mode": artifact.state_input_mode,
            "input_state_discrete_key": artifact.state_discrete_key,
            "input_state_discrete_dim": layout.state_discrete_dim,
            "output_action_dim": layout.action_dim,
            "output_action_discrete_dim": layout.action_discrete_dim,
            "active_sides": ["left", "right"],
        },
        "dataset_adapter": {
            "policy_io_layout_hash": layout.layout_hash,
            "camera_slots": [],
        },
    }


def _backend() -> tuple[
    PolicyArtifactContract,
    OpenPIFinalChunkRtcBackend,
    _MetadataOnlyTransport,
]:
    artifact = _artifact()
    transport = _MetadataOnlyTransport(_server_metadata(artifact))
    backend = OpenPIFinalChunkRtcBackend(
        transport=transport,
        config=OpenPIFinalChunkRtcConfig(
            artifact=artifact,
            frame_contract_hash="frame:cs3b-openpi-unit-v1",
            normalization_hash="normalization:cs3b-openpi-unit-v1",
            minimum_execution_horizon_steps=2,
        ),
    )
    return artifact, backend, transport


def _binding(backend: OpenPIFinalChunkRtcBackend) -> RtcFrontendStepSourceBinding:
    return RtcFrontendStepSourceBinding(
        source_slot_id="rtc-policy-source:cs3b-openpi-unit",
        runtime_session_id="runtime:cs3b-openpi-unit",
        control_domain_id="deploy-control:cs3b-openpi-unit",
        producer_id="policy:cs3b-openpi-unit",
        producer_epoch=0,
        policy_node_id="single_policy",
        policy_clock_domain_id="control-monotonic",
        policy_runtime_config_hash="runtime-config:cs3b-openpi-unit",
        action_adapter_contract_hash="adapter:cs3b-openpi-unit",
        rtc_frontend_config_hash="rtc-frontend:cs3b-openpi-unit",
        rtc_profile_hash="rtc-profile:cs3b-openpi-unit",
        artifact_hash=backend.config.artifact_hash,
        action_codec_hash=backend.config.action_codec_hash,
        frame_contract_hash=backend.config.frame_contract_hash,
        layout_hash=backend.config.artifact.layout_hash,
        normalization_hash=backend.config.normalization_hash,
    )


def test_cs3b_source_preflight_is_pure_dimensioned_and_non_authoritative() -> None:
    artifact, backend, transport = _backend()
    binding = _binding(backend)
    report = preflight_policy_artifact(artifact)
    metadata_reads_before = transport.metadata_read_count

    assessment = assess_openpi_rtc_passive_source(
        source_binding=binding,
        artifact_preflight=report,
        backend=backend,
    )

    assert assessment.accepted
    assert assessment.artifact_contract_status == "verified"
    assert assessment.rtc_protocol_status == "verified"
    assert assessment.source_binding_status == "verified"
    assert (
        assessment.server_metadata_contract_status
        == "validated_at_backend_construction"
    )
    assert assessment.live_transport_status == "not_assessed"
    assert assessment.real_observation_status == "not_assessed"
    assert assessment.target_binding_status == "not_assessed"
    assert assessment.execution_authorization_status == "denied"
    assert assessment.permitted_uses == ("record_only", "shadow")
    assert assessment.execution_candidate_allowed is False
    assert assessment.motion_authorized is False
    assert transport.metadata_read_count == metadata_reads_before
    assert transport.infer_count == 0

    repeated = assess_openpi_rtc_passive_source(
        source_binding=binding,
        artifact_preflight=report,
        backend=backend,
    )
    assert repeated.semantic_content_hash == assessment.semantic_content_hash


def test_cs3b_manifest_binds_assessment_but_stays_null_shadow_only() -> None:
    artifact, backend, _transport = _backend()
    binding = _binding(backend)
    assessment = assess_openpi_rtc_passive_source(
        source_binding=binding,
        artifact_preflight=preflight_policy_artifact(artifact),
        backend=backend,
    )

    manifest = build_cs3b_openpi_rtc_source_preflight_manifest(
        binding,
        assessment,
    )

    assert manifest.manifest_id == CS3B_OPENPI_RTC_SOURCE_PREFLIGHT_MANIFEST_ID_V1
    assert manifest.execution_binding.target_kind == "null"
    assert manifest.source_slots[0].supported_uses == ("shadow",)
    assert assessment.semantic_content_hash in manifest.source_slots[0].required_binding_refs
    assert "execution_candidate" in manifest.explicit_exclusions
    assert "live_transport_readiness" in manifest.explicit_exclusions
    assert "motion" in manifest.explicit_exclusions
    assert "request_candidate" not in manifest.allowed_intent_kinds
    assert "request_motion_authority" not in manifest.allowed_intent_kinds


def test_cs3b_known_source_binding_mismatch_is_rejected_without_transport_io() -> None:
    artifact, backend, transport = _backend()
    binding = replace(
        _binding(backend),
        normalization_hash="normalization:wrong",
    )
    metadata_reads_before = transport.metadata_read_count

    assessment = assess_openpi_rtc_passive_source(
        source_binding=binding,
        artifact_preflight=preflight_policy_artifact(artifact),
        backend=backend,
    )

    assert not assessment.accepted
    assert assessment.source_binding_status == "rejected"
    assert assessment.permitted_uses == ("record_only",)
    assert "source_binding_normalization_hash_mismatch" in assessment.rejection_reasons
    assert transport.metadata_read_count == metadata_reads_before
    assert transport.infer_count == 0
    with pytest.raises(ValueError, match="Rejected RTC source assessment"):
        build_cs3b_openpi_rtc_source_preflight_manifest(binding, assessment)


def test_cs3b_assessment_cannot_be_promoted_to_transport_or_motion_evidence() -> None:
    artifact, backend, _transport = _backend()
    assessment = assess_openpi_rtc_passive_source(
        source_binding=_binding(backend),
        artifact_preflight=preflight_policy_artifact(artifact),
        backend=backend,
    )

    with pytest.raises(ValueError, match="cannot verify live_transport_status"):
        replace(assessment, live_transport_status="verified")
    with pytest.raises(ValueError, match="cannot allow an execution candidate or motion"):
        replace(assessment, motion_authorized=True)
