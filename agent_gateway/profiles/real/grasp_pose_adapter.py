"""Resolve one camera-relative grasp proposal into canonical Station tool TCP.

This is a pure proposal adapter.  It owns neither perception capture nor motion
admission: the input must identify one completed real Depth Skill Artifact, and
the output remains an unprivileged ``canonical_station`` tool-TCP target.  The
real motion Route, Grant, safety envelope, IK, collision checks, dispatch, and
HOLD remain authoritative downstream.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.skills.perception.definition import CAPTURE_DEPTH_SKILL_ID_V1
from agent_gateway.skills.perception.schemas import (
    DEPTH_FRAME_COMPRESSION_V1,
    DEPTH_FRAME_ENCODING_V1,
    DEPTH_FRAME_MEDIA_TYPE_V1,
    DEPTH_FRAME_SCHEMA_ID_V1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.relative_action import (
    pose_matrix_to_xyzquat,
    xyzquat_to_pose_matrix,
)

from .perception_deployment import (
    REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1,
    REAL_CAPTURE_DEPTH_ROUTE_ID_V2,
)

CAMERA_RELATIVE_GRASP_PROPOSAL_SCHEMA_V1 = (
    "dexteleop.real.camera_relative_grasp_proposal.v1"
)
_MAX_PROPOSAL_BYTES = 256 * 1024
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "runtime_session_id",
        "skill_route_manifest_hash",
        "source_operation_id",
        "source_operation_hash",
        "perception_route_id",
        "perception_route_hash",
        "perception_deployment_binding_hash",
        "component_id",
        "station_revision",
        "reference_frame_id",
        "reference_frame_spec_hash",
        "frame_graph_hash",
        "camera_pose_reference_xyz_quat_xyzw",
        "tool_tcp_pose_camera_xyz_quat_xyzw",
        "depth_artifact_binding",
    }
)
_OPTIONAL_FIELDS = frozenset({"capture_group_id"})


def _station_depth_route_entry(
    route_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(route_manifest, Mapping):
        raise TypeError("route_manifest must be a mapping.")
    entries = route_manifest.get("skill_routes")
    if not isinstance(entries, (tuple, list)):
        raise TypeError("Gateway Route manifest has no skill_routes list.")
    matching_entries = tuple(
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and isinstance(entry.get("route"), Mapping)
        and entry["route"].get("skill_id") == CAPTURE_DEPTH_SKILL_ID_V1
        and entry["route"].get("skill_version") == "2"
        and entry["route"].get("route_id") == REAL_CAPTURE_DEPTH_ROUTE_ID_V2
    )
    if len(matching_entries) != 1:
        raise RuntimeError("Gateway does not advertise one Station-aware Depth Route.")
    return matching_entries[0]


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _pose(value: object, *, field_name: str) -> tuple[float, ...]:
    matrix = xyzquat_to_pose_matrix(value, name=field_name)
    return tuple(
        float(item)
        for item in pose_matrix_to_xyzquat(matrix, name=field_name)
    )


def _capture_group_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _LOWER_SHA256.fullmatch(value) is None:
        raise ValueError("capture_group_id must be a lowercase SHA-256 digest.")
    return value


@dataclass(frozen=True)
class CameraRelativeGraspProposal:
    """One grasp generator result bound to a real Depth Skill capture."""

    runtime_session_id: str
    skill_route_manifest_hash: str
    source_operation_id: str
    source_operation_hash: str
    perception_route_id: str
    perception_route_hash: str
    perception_deployment_binding_hash: str
    component_id: str
    station_revision: int
    reference_frame_id: str
    reference_frame_spec_hash: str
    frame_graph_hash: str
    camera_pose_reference_xyz_quat_xyzw: tuple[float, ...]
    tool_tcp_pose_camera_xyz_quat_xyzw: tuple[float, ...]
    depth_artifact_binding: AgentArtifactBinding
    capture_group_id: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_session_id",
            "skill_route_manifest_hash",
            "source_operation_id",
            "source_operation_hash",
            "perception_route_id",
            "perception_route_hash",
            "perception_deployment_binding_hash",
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "frame_graph_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        if self.perception_route_id != REAL_CAPTURE_DEPTH_ROUTE_ID_V2:
            raise ValueError("Proposal must identify the Station-aware Depth v2 Route.")
        if self.reference_frame_id != "canonical_station":
            raise ValueError("Proposal reference_frame_id must be canonical_station.")
        if self.component_id not in {"left_arm", "right_arm"}:
            raise ValueError("Proposal component_id must select one RM65 arm.")
        object.__setattr__(
            self,
            "station_revision",
            _nonnegative_integer(
                self.station_revision,
                field_name="station_revision",
            ),
        )
        object.__setattr__(
            self,
            "camera_pose_reference_xyz_quat_xyzw",
            _pose(
                self.camera_pose_reference_xyz_quat_xyzw,
                field_name="camera_pose_reference_xyz_quat_xyzw",
            ),
        )
        object.__setattr__(
            self,
            "tool_tcp_pose_camera_xyz_quat_xyzw",
            _pose(
                self.tool_tcp_pose_camera_xyz_quat_xyzw,
                field_name="tool_tcp_pose_camera_xyz_quat_xyzw",
            ),
        )
        if not isinstance(self.depth_artifact_binding, AgentArtifactBinding):
            raise TypeError("depth_artifact_binding must be typed.")
        binding = self.depth_artifact_binding
        if binding.slot_id != REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1 or (
            binding.ordinal != 0
        ):
            raise ValueError("Proposal requires the unique depth_frame Artifact binding.")
        if not binding.production_receipt_ref or not (
            binding.production_receipt_ref.startswith("real-depth-capture:")
        ):
            raise ValueError("Depth Artifact requires its real capture receipt.")
        artifact = binding.artifact
        expected_artifact_contract = (
            DEPTH_FRAME_MEDIA_TYPE_V1,
            DEPTH_FRAME_SCHEMA_ID_V1,
            DEPTH_FRAME_ENCODING_V1,
            DEPTH_FRAME_COMPRESSION_V1,
        )
        actual_artifact_contract = (
            artifact.media_type,
            artifact.schema_id,
            artifact.encoding,
            artifact.compression,
        )
        if actual_artifact_contract != expected_artifact_contract:
            raise ValueError("Proposal source is not the reviewed metric Depth Artifact.")
        object.__setattr__(
            self,
            "capture_group_id",
            _capture_group_id(self.capture_group_id),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "schema": CAMERA_RELATIVE_GRASP_PROPOSAL_SCHEMA_V1,
                    "runtime_session_id": self.runtime_session_id,
                    "skill_route_manifest_hash": self.skill_route_manifest_hash,
                    "source_operation_id": self.source_operation_id,
                    "source_operation_hash": self.source_operation_hash,
                    "perception_route_id": self.perception_route_id,
                    "perception_route_hash": self.perception_route_hash,
                    "perception_deployment_binding_hash": (
                        self.perception_deployment_binding_hash
                    ),
                    "component_id": self.component_id,
                    "station_revision": self.station_revision,
                    "reference_frame_id": self.reference_frame_id,
                    "reference_frame_spec_hash": self.reference_frame_spec_hash,
                    "frame_graph_hash": self.frame_graph_hash,
                    "camera_pose_reference_xyz_quat_xyzw": (
                        self.camera_pose_reference_xyz_quat_xyzw
                    ),
                    "tool_tcp_pose_camera_xyz_quat_xyzw": (
                        self.tool_tcp_pose_camera_xyz_quat_xyzw
                    ),
                    "depth_artifact_binding_hash": binding.semantic_content_hash,
                    "capture_group_id": self.capture_group_id,
                },
                domain="dexteleop.real.camera_relative_grasp_proposal.v1",
            ),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CameraRelativeGraspProposal:
        if not isinstance(payload, Mapping):
            raise TypeError("Camera grasp proposal must be a JSON object.")
        fields = frozenset(payload)
        missing = sorted(_REQUIRED_FIELDS - fields)
        unknown = sorted(fields - _REQUIRED_FIELDS - _OPTIONAL_FIELDS)
        if missing:
            raise ValueError(f"Camera grasp proposal is missing fields: {missing}.")
        if unknown:
            raise ValueError(f"Camera grasp proposal has unknown fields: {unknown}.")
        if payload["schema"] != CAMERA_RELATIVE_GRASP_PROPOSAL_SCHEMA_V1:
            raise ValueError("Unsupported camera grasp proposal schema.")
        return cls(
            runtime_session_id=payload["runtime_session_id"],
            skill_route_manifest_hash=payload["skill_route_manifest_hash"],
            source_operation_id=payload["source_operation_id"],
            source_operation_hash=payload["source_operation_hash"],
            perception_route_id=payload["perception_route_id"],
            perception_route_hash=payload["perception_route_hash"],
            perception_deployment_binding_hash=payload[
                "perception_deployment_binding_hash"
            ],
            component_id=payload["component_id"],
            station_revision=payload["station_revision"],
            reference_frame_id=payload["reference_frame_id"],
            reference_frame_spec_hash=payload["reference_frame_spec_hash"],
            frame_graph_hash=payload["frame_graph_hash"],
            camera_pose_reference_xyz_quat_xyzw=payload[
                "camera_pose_reference_xyz_quat_xyzw"
            ],
            tool_tcp_pose_camera_xyz_quat_xyzw=payload[
                "tool_tcp_pose_camera_xyz_quat_xyzw"
            ],
            depth_artifact_binding=AgentArtifactBinding.from_wire(
                payload["depth_artifact_binding"]
            ),
            capture_group_id=payload.get("capture_group_id"),
        )

    @classmethod
    def from_depth_operation_wire(
        cls,
        *,
        route_manifest: Mapping[str, Any],
        depth_operation: Mapping[str, Any],
        component_id: str,
        tool_tcp_pose_camera_xyz_quat_xyzw: object,
        capture_group_id: str | None = None,
    ) -> CameraRelativeGraspProposal:
        """Build a proposal without manually copying Depth operation identity."""

        entry = _station_depth_route_entry(route_manifest)
        route = entry["route"]
        deployment_binding = entry.get("deployment_binding")
        if not isinstance(deployment_binding, Mapping):
            raise TypeError("Depth Route has no deployment binding.")
        if not isinstance(depth_operation, Mapping):
            raise TypeError("depth_operation must be a wire mapping.")
        if (
            depth_operation.get("status") != "completed"
            or depth_operation.get("execution_outcome") != "completed"
            or depth_operation.get("terminal") is not True
        ):
            raise ValueError("Depth operation must be terminal and completed.")
        runtime_session_id = _required_text(
            route_manifest.get("runtime_session_id"),
            field_name="route_manifest.runtime_session_id",
        )
        if depth_operation.get("runtime_session_id") != runtime_session_id:
            raise ValueError("Depth operation belongs to another Gateway session.")
        result = depth_operation.get("execution_result")
        if not isinstance(result, Mapping):
            raise TypeError("Completed Depth operation has no execution_result.")
        required_result_fields = {
            "station_revision",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "frame_graph_hash",
            "camera_pose_reference_xyz_quat_xyzw",
        }
        if not required_result_fields.issubset(result):
            raise ValueError("Depth v2 execution_result lacks Station pose identity.")
        if depth_operation.get("station_revision_at_admission") != result.get(
            "station_revision"
        ):
            raise ValueError("Depth result Station revision differs from admission.")
        bindings = depth_operation.get("output_artifact_bindings")
        if not isinstance(bindings, (tuple, list)) or len(bindings) != 1:
            raise ValueError("Depth operation must return exactly one Artifact binding.")
        return cls(
            runtime_session_id=runtime_session_id,
            skill_route_manifest_hash=_required_text(
                route_manifest.get("route_manifest_hash"),
                field_name="route_manifest.route_manifest_hash",
            ),
            source_operation_id=_required_text(
                depth_operation.get("operation_id"),
                field_name="depth_operation.operation_id",
            ),
            source_operation_hash=_required_text(
                depth_operation.get("operation_hash"),
                field_name="depth_operation.operation_hash",
            ),
            perception_route_id=_required_text(
                route.get("route_id"),
                field_name="depth_route.route_id",
            ),
            perception_route_hash=_required_text(
                route.get("route_hash"),
                field_name="depth_route.route_hash",
            ),
            perception_deployment_binding_hash=_required_text(
                deployment_binding.get("deployment_binding_hash"),
                field_name="depth_route.deployment_binding_hash",
            ),
            component_id=component_id,
            station_revision=result["station_revision"],
            reference_frame_id=result["reference_frame_id"],
            reference_frame_spec_hash=result["reference_frame_spec_hash"],
            frame_graph_hash=result["frame_graph_hash"],
            camera_pose_reference_xyz_quat_xyzw=result[
                "camera_pose_reference_xyz_quat_xyzw"
            ],
            tool_tcp_pose_camera_xyz_quat_xyzw=(
                tool_tcp_pose_camera_xyz_quat_xyzw
            ),
            depth_artifact_binding=AgentArtifactBinding.from_wire(bindings[0]),
            capture_group_id=capture_group_id,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the exact bounded JSON contract consumed by the real client."""

        payload: dict[str, Any] = {
            "schema": CAMERA_RELATIVE_GRASP_PROPOSAL_SCHEMA_V1,
            "runtime_session_id": self.runtime_session_id,
            "skill_route_manifest_hash": self.skill_route_manifest_hash,
            "source_operation_id": self.source_operation_id,
            "source_operation_hash": self.source_operation_hash,
            "perception_route_id": self.perception_route_id,
            "perception_route_hash": self.perception_route_hash,
            "perception_deployment_binding_hash": (
                self.perception_deployment_binding_hash
            ),
            "component_id": self.component_id,
            "station_revision": self.station_revision,
            "reference_frame_id": self.reference_frame_id,
            "reference_frame_spec_hash": self.reference_frame_spec_hash,
            "frame_graph_hash": self.frame_graph_hash,
            "camera_pose_reference_xyz_quat_xyzw": list(
                self.camera_pose_reference_xyz_quat_xyzw
            ),
            "tool_tcp_pose_camera_xyz_quat_xyzw": list(
                self.tool_tcp_pose_camera_xyz_quat_xyzw
            ),
            "depth_artifact_binding": self.depth_artifact_binding.to_wire(),
        }
        if self.capture_group_id is not None:
            payload["capture_group_id"] = self.capture_group_id
        return payload


@dataclass(frozen=True)
class ResolvedCameraGraspTarget:
    """Canonical Station actual tool-TCP target plus immutable provenance."""

    component_id: str
    station_revision: int
    target_pose_reference_xyz_quat_xyzw: tuple[float, ...]
    proposal_hash: str
    depth_artifact_binding_hash: str
    capture_group_id: str | None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "component_id": self.component_id,
                    "station_revision": self.station_revision,
                    "target_pose_reference_xyz_quat_xyzw": (
                        self.target_pose_reference_xyz_quat_xyzw
                    ),
                    "proposal_hash": self.proposal_hash,
                    "depth_artifact_binding_hash": self.depth_artifact_binding_hash,
                    "capture_group_id": self.capture_group_id,
                    "target_semantics": "T_canonical_station_actual_tool_tcp",
                },
                domain="dexteleop.real.resolved_camera_grasp_target.v1",
            ),
        )


def load_camera_relative_grasp_proposal(
    path: str | Path,
) -> CameraRelativeGraspProposal:
    proposal_path = Path(path)
    if not proposal_path.is_absolute():
        raise ValueError("--camera-grasp-proposal must be an absolute path.")
    try:
        size = proposal_path.stat().st_size
    except OSError as exc:
        raise ValueError("Camera grasp proposal is not readable.") from exc
    if size <= 0 or size > _MAX_PROPOSAL_BYTES:
        raise ValueError("Camera grasp proposal size is outside the reviewed bound.")
    try:
        encoded = proposal_path.read_bytes()
        if not encoded or len(encoded) > _MAX_PROPOSAL_BYTES:
            raise ValueError(
                "Camera grasp proposal size changed outside the reviewed bound."
            )
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Camera grasp proposal is not valid UTF-8 JSON.") from exc
    return CameraRelativeGraspProposal.from_mapping(payload)


def resolve_camera_relative_grasp_target(
    *,
    proposal: CameraRelativeGraspProposal,
    route_manifest: Mapping[str, Any],
    motion_snapshot: Mapping[str, Any],
    expected_component_id: str,
) -> ResolvedCameraGraspTarget:
    """Compose ``T_station_camera @ T_camera_tool_tcp`` after identity checks."""

    if not isinstance(proposal, CameraRelativeGraspProposal):
        raise TypeError("proposal must be CameraRelativeGraspProposal.")
    if proposal.component_id != expected_component_id:
        raise ValueError("Camera grasp proposal selects a different arm.")
    perception_entry = _station_depth_route_entry(route_manifest)
    if route_manifest.get("runtime_session_id") != proposal.runtime_session_id or (
        route_manifest.get("route_manifest_hash")
        != proposal.skill_route_manifest_hash
    ):
        raise RuntimeError("Camera grasp proposal belongs to another Gateway session.")
    route = perception_entry["route"]
    binding = perception_entry.get("deployment_binding")
    if route.get("route_hash") != proposal.perception_route_hash or not isinstance(
        binding, Mapping
    ) or binding.get("deployment_binding_hash") != (
        proposal.perception_deployment_binding_hash
    ):
        raise RuntimeError("Camera grasp proposal perception deployment changed.")
    if not isinstance(motion_snapshot, Mapping):
        raise TypeError("motion_snapshot must be a mapping.")
    if motion_snapshot.get("station_revision") != proposal.station_revision:
        raise RuntimeError("Camera grasp proposal Station revision is stale.")
    if motion_snapshot.get("frame_graph_hash") != proposal.frame_graph_hash:
        raise RuntimeError("Camera grasp proposal FrameGraph changed.")
    summary = motion_snapshot.get("measurement_summary")
    if not isinstance(summary, Mapping):
        raise TypeError("Motion observation has no measurement_summary.")
    expected_reference = {
        "reference_frame_id": proposal.reference_frame_id,
        "reference_frame_spec_hash": proposal.reference_frame_spec_hash,
    }
    if any(summary.get(key) != value for key, value in expected_reference.items()):
        raise RuntimeError("Camera grasp proposal Station frame identity changed.")

    station_from_camera = xyzquat_to_pose_matrix(
        proposal.camera_pose_reference_xyz_quat_xyzw,
        name="camera_pose_reference_xyz_quat_xyzw",
    )
    camera_from_tool_tcp = xyzquat_to_pose_matrix(
        proposal.tool_tcp_pose_camera_xyz_quat_xyzw,
        name="tool_tcp_pose_camera_xyz_quat_xyzw",
    )
    station_from_tool_tcp = station_from_camera @ camera_from_tool_tcp
    target = tuple(
        float(item)
        for item in pose_matrix_to_xyzquat(
            station_from_tool_tcp,
            name="target_pose_reference_xyz_quat_xyzw",
        )
    )
    return ResolvedCameraGraspTarget(
        component_id=proposal.component_id,
        station_revision=proposal.station_revision,
        target_pose_reference_xyz_quat_xyzw=target,
        proposal_hash=proposal.semantic_content_hash,
        depth_artifact_binding_hash=(
            proposal.depth_artifact_binding.semantic_content_hash
        ),
        capture_group_id=proposal.capture_group_id,
    )


__all__ = [
    "CAMERA_RELATIVE_GRASP_PROPOSAL_SCHEMA_V1",
    "CameraRelativeGraspProposal",
    "ResolvedCameraGraspTarget",
    "load_camera_relative_grasp_proposal",
    "resolve_camera_relative_grasp_target",
]
