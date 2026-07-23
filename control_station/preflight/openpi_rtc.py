"""Pure CS3b-a projection for an already-constructed OpenPI RTC source.

This module is intentionally downstream of both deployment and RTC packages.
It never constructs a websocket transport, loads an artifact, prepares an
observation, starts a policy worker, or performs inference.  It only compares
immutable evidence that a lower-owned composition has already produced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from dataset_contracts.canonical_encoding import canonical_content_hash
from deployments.openpi.client.preflight import OpenPIArtifactPreflightReport
from deployments.openpi.client.rtc import (
    OpenPIFinalChunkRtcBackend,
    openpi_rtc_artifact_hash,
)
from robot_motion.deploy.rtc_protocol import negotiate_rtc_capability
from teleop_runtime.deploy.rtc_step_fanout import RtcFrontendStepSourceBinding

from ..contracts import ControlStationRunManifest, SourceSlotSpec
from ..composition.rtc_policy_shadow import (
    build_cs3_rtc_policy_passive_shadow_manifest,
)


CS3B_OPENPI_RTC_SOURCE_PREFLIGHT_MANIFEST_ID_V1 = (
    "control_station_cs3b_openpi_rtc_source_preflight_v1"
)
CS3B_OPENPI_RTC_SOURCE_PREFLIGHT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs3b_openpi_rtc_source_preflight.v1"
)
CS3B_OPENPI_ARTIFACT_PREFLIGHT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs3b_openpi_artifact_preflight.v1"
)

RTC_SOURCE_STATUS_VERIFIED = "verified"
RTC_SOURCE_STATUS_REJECTED = "rejected"
RTC_SOURCE_STATUS_NOT_ASSESSED = "not_assessed"
RTC_SOURCE_STATUS_DENIED = "denied"
RTC_SERVER_METADATA_STATUS_VALIDATED_AT_BACKEND_CONSTRUCTION = (
    "validated_at_backend_construction"
)


@dataclass(frozen=True)
class OpenPIRtcSourcePreflightAssessment:
    """Dimensioned source evidence that cannot authorize execution.

    ``server_metadata_contract_status`` only records what the already-created
    backend validated during construction.  It is deliberately separate from
    current websocket/session health.
    """

    source_binding_hash: str
    artifact_preflight_hash: str
    artifact_hash: str
    rtc_capability_hash: str
    rtc_request_spec_hash: str
    rtc_negotiation_receipt_hash: str
    selected_continuation_hash: str | None
    server_metadata_hash: str
    artifact_contract_status: str
    rtc_protocol_status: str
    source_binding_status: str
    server_metadata_contract_status: str
    live_transport_status: str
    real_observation_status: str
    target_binding_status: str
    execution_authorization_status: str
    permitted_uses: tuple[str, ...]
    execution_candidate_allowed: bool
    motion_authorized: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "source_binding_hash",
            "artifact_preflight_hash",
            "artifact_hash",
            "rtc_capability_hash",
            "rtc_request_spec_hash",
            "rtc_negotiation_receipt_hash",
            "server_metadata_hash",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty.")
        verification_statuses = {
            RTC_SOURCE_STATUS_VERIFIED,
            RTC_SOURCE_STATUS_REJECTED,
        }
        for field_name in (
            "artifact_contract_status",
            "rtc_protocol_status",
            "source_binding_status",
        ):
            if getattr(self, field_name) not in verification_statuses:
                raise ValueError(f"Unsupported {field_name}.")
        if (
            self.server_metadata_contract_status
            != RTC_SERVER_METADATA_STATUS_VALIDATED_AT_BACKEND_CONSTRUCTION
        ):
            raise ValueError(
                "CS3b-a only accepts cached backend-construction metadata evidence."
            )
        for field_name in (
            "live_transport_status",
            "real_observation_status",
            "target_binding_status",
        ):
            if getattr(self, field_name) != RTC_SOURCE_STATUS_NOT_ASSESSED:
                raise ValueError(f"CS3b-a cannot verify {field_name}.")
        if self.execution_authorization_status != RTC_SOURCE_STATUS_DENIED:
            raise ValueError("CS3b-a execution authorization must remain denied.")
        if self.execution_candidate_allowed or self.motion_authorized:
            raise ValueError("CS3b-a cannot allow an execution candidate or motion.")
        reasons = tuple(str(value).strip() for value in self.rejection_reasons)
        if any(not value for value in reasons) or len(set(reasons)) != len(reasons):
            raise ValueError("rejection_reasons must be unique non-empty values.")
        object.__setattr__(self, "rejection_reasons", reasons)
        all_verified = all(
            getattr(self, field_name) == RTC_SOURCE_STATUS_VERIFIED
            for field_name in (
                "artifact_contract_status",
                "rtc_protocol_status",
                "source_binding_status",
            )
        )
        expected_uses = (
            ("record_only", "shadow")
            if all_verified and not reasons
            else ("record_only",)
        )
        if self.permitted_uses != expected_uses:
            raise ValueError("permitted_uses disagree with CS3b-a verification status.")
        if all_verified != (not reasons):
            raise ValueError("Verification status and rejection reasons disagree.")
        if all_verified and self.selected_continuation_hash is None:
            raise ValueError("Verified RTC protocol requires a selected continuation.")

    @property
    def accepted(self) -> bool:
        return (
            self.artifact_contract_status == RTC_SOURCE_STATUS_VERIFIED
            and self.rtc_protocol_status == RTC_SOURCE_STATUS_VERIFIED
            and self.source_binding_status == RTC_SOURCE_STATUS_VERIFIED
            and not self.rejection_reasons
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "source_binding_hash": self.source_binding_hash,
                "artifact_preflight_hash": self.artifact_preflight_hash,
                "artifact_hash": self.artifact_hash,
                "rtc_capability_hash": self.rtc_capability_hash,
                "rtc_request_spec_hash": self.rtc_request_spec_hash,
                "rtc_negotiation_receipt_hash": (
                    self.rtc_negotiation_receipt_hash
                ),
                "selected_continuation_hash": self.selected_continuation_hash,
                "server_metadata_hash": self.server_metadata_hash,
                "artifact_contract_status": self.artifact_contract_status,
                "rtc_protocol_status": self.rtc_protocol_status,
                "source_binding_status": self.source_binding_status,
                "server_metadata_contract_status": (
                    self.server_metadata_contract_status
                ),
                "live_transport_status": self.live_transport_status,
                "real_observation_status": self.real_observation_status,
                "target_binding_status": self.target_binding_status,
                "execution_authorization_status": (
                    self.execution_authorization_status
                ),
                "permitted_uses": self.permitted_uses,
                "execution_candidate_allowed": self.execution_candidate_allowed,
                "motion_authorized": self.motion_authorized,
                "rejection_reasons": self.rejection_reasons,
            },
            domain=CS3B_OPENPI_RTC_SOURCE_PREFLIGHT_HASH_DOMAIN_V1,
        )


def assess_openpi_rtc_passive_source(
    *,
    source_binding: RtcFrontendStepSourceBinding,
    artifact_preflight: OpenPIArtifactPreflightReport,
    backend: OpenPIFinalChunkRtcBackend,
) -> OpenPIRtcSourcePreflightAssessment:
    """Compare cached typed facts without touching transport or inference."""

    if not isinstance(source_binding, RtcFrontendStepSourceBinding):
        raise TypeError("source_binding must be an RtcFrontendStepSourceBinding.")
    if not isinstance(artifact_preflight, OpenPIArtifactPreflightReport):
        raise TypeError("artifact_preflight must be an OpenPIArtifactPreflightReport.")
    if not isinstance(backend, OpenPIFinalChunkRtcBackend):
        raise TypeError(
            "backend must be an already-constructed OpenPIFinalChunkRtcBackend."
        )

    capability = backend.rtc_capability
    request_spec = backend.request_spec
    negotiation = negotiate_rtc_capability(request_spec, capability)
    backend_artifact_hash = backend.config.artifact_hash
    preflight_artifact_hash = openpi_rtc_artifact_hash(artifact_preflight.artifact)
    artifact_preflight_hash = canonical_content_hash(
        artifact_preflight.to_dict(),
        domain=CS3B_OPENPI_ARTIFACT_PREFLIGHT_HASH_DOMAIN_V1,
    )

    artifact_reasons: list[str] = []
    if preflight_artifact_hash != backend_artifact_hash:
        artifact_reasons.append("artifact_preflight_backend_hash_mismatch")
    if artifact_preflight.action_pose_mode != backend.config.action_pose_mode:
        artifact_reasons.append("artifact_preflight_action_pose_mode_mismatch")
    preflight_sides = tuple(
        str(value)
        for value in artifact_preflight.session_contract.get("active_sides", ())
    )
    if preflight_sides != backend.config.active_sides:
        artifact_reasons.append("artifact_preflight_active_sides_mismatch")

    protocol_reasons = list(negotiation.rejection_reasons)
    if request_spec.artifact_hash != backend_artifact_hash:
        protocol_reasons.append("request_spec_artifact_hash_mismatch")
    if capability.artifact_hash != backend_artifact_hash:
        protocol_reasons.append("capability_artifact_hash_mismatch")

    binding_reasons: list[str] = []
    expected_binding_values = {
        "artifact_hash": backend_artifact_hash,
        "action_codec_hash": backend.config.action_codec_hash,
        "frame_contract_hash": backend.config.frame_contract_hash,
        "layout_hash": backend.config.artifact.layout_hash,
        "normalization_hash": backend.config.normalization_hash,
    }
    for field_name, expected in expected_binding_values.items():
        if getattr(source_binding, field_name) != expected:
            binding_reasons.append(f"source_binding_{field_name}_mismatch")

    rejection_reasons = tuple(
        dict.fromkeys((*artifact_reasons, *protocol_reasons, *binding_reasons))
    )
    accepted = not rejection_reasons
    return OpenPIRtcSourcePreflightAssessment(
        source_binding_hash=source_binding.semantic_content_hash,
        artifact_preflight_hash=artifact_preflight_hash,
        artifact_hash=backend_artifact_hash,
        rtc_capability_hash=capability.semantic_content_hash,
        rtc_request_spec_hash=request_spec.semantic_content_hash,
        rtc_negotiation_receipt_hash=negotiation.semantic_content_hash,
        selected_continuation_hash=negotiation.selected_continuation_hash,
        server_metadata_hash=backend.server_metadata_hash,
        artifact_contract_status=(
            RTC_SOURCE_STATUS_VERIFIED
            if not artifact_reasons
            else RTC_SOURCE_STATUS_REJECTED
        ),
        rtc_protocol_status=(
            RTC_SOURCE_STATUS_VERIFIED
            if not protocol_reasons
            else RTC_SOURCE_STATUS_REJECTED
        ),
        source_binding_status=(
            RTC_SOURCE_STATUS_VERIFIED
            if not binding_reasons
            else RTC_SOURCE_STATUS_REJECTED
        ),
        server_metadata_contract_status=(
            RTC_SERVER_METADATA_STATUS_VALIDATED_AT_BACKEND_CONSTRUCTION
        ),
        live_transport_status=RTC_SOURCE_STATUS_NOT_ASSESSED,
        real_observation_status=RTC_SOURCE_STATUS_NOT_ASSESSED,
        target_binding_status=RTC_SOURCE_STATUS_NOT_ASSESSED,
        execution_authorization_status=RTC_SOURCE_STATUS_DENIED,
        permitted_uses=("record_only", "shadow") if accepted else ("record_only",),
        execution_candidate_allowed=False,
        motion_authorized=False,
        rejection_reasons=rejection_reasons,
    )


def build_cs3b_openpi_rtc_source_preflight_manifest(
    source_binding: RtcFrontendStepSourceBinding,
    assessment: OpenPIRtcSourcePreflightAssessment,
) -> ControlStationRunManifest:
    """Bind an accepted source assessment to a still-null Station manifest."""

    if not isinstance(source_binding, RtcFrontendStepSourceBinding):
        raise TypeError("source_binding must be an RtcFrontendStepSourceBinding.")
    if not isinstance(assessment, OpenPIRtcSourcePreflightAssessment):
        raise TypeError("assessment must be an OpenPIRtcSourcePreflightAssessment.")
    if assessment.source_binding_hash != source_binding.semantic_content_hash:
        raise ValueError("RTC source assessment belongs to another source binding.")
    if not assessment.accepted:
        raise ValueError("Rejected RTC source assessment cannot bind a Station manifest.")

    base = build_cs3_rtc_policy_passive_shadow_manifest(source_binding)
    source = base.source_slots[0]
    bound_source = replace(
        source,
        required_binding_refs=(
            *source.required_binding_refs,
            assessment.semantic_content_hash,
            assessment.artifact_preflight_hash,
            assessment.rtc_capability_hash,
            assessment.rtc_negotiation_receipt_hash,
            assessment.server_metadata_hash,
        ),
        explicit_exclusions=(
            *source.explicit_exclusions,
            "execution_candidate",
            "live_transport_readiness",
            "real_observation_readiness",
            "target_binding",
        ),
    )
    assert isinstance(bound_source, SourceSlotSpec)
    return replace(
        base,
        manifest_id=CS3B_OPENPI_RTC_SOURCE_PREFLIGHT_MANIFEST_ID_V1,
        source_slots=(bound_source,),
        explicit_exclusions=(
            *base.explicit_exclusions,
            "execution_candidate",
            "live_transport_readiness",
            "real_observation_readiness",
            "target_binding",
        ),
    )


__all__ = [
    "CS3B_OPENPI_RTC_SOURCE_PREFLIGHT_MANIFEST_ID_V1",
    "OpenPIRtcSourcePreflightAssessment",
    "assess_openpi_rtc_passive_source",
    "build_cs3b_openpi_rtc_source_preflight_manifest",
]
