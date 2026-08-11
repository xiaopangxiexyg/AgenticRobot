"""Read-only OpenPI policy-artifact deployment preflight.

This module deliberately validates the same typed contracts used by the real
deployment runtime without opening a websocket, cameras, or robot backends.
It does not repair or infer missing policy provenance: the checkpoint sidecar
remains the authority for model-facing pose semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_contracts.policy_artifact import PolicyArtifactContract
from dataset_contracts.pose_frames import PoseFrameContract
from dataset_contracts.pose_training import RUNTIME_POSE_BINDING_EXPLICIT_CANONICAL_DESK
from deployments.client.artifact import deploy_contract_from_policy_artifact
from deployments.policy_artifact import load_policy_artifact
from robot_motion.deploy.session_contract import (
    resolve_deploy_session_contract,
    resolve_policy_action_pose_mode_from_artifact,
)
from robot_motion.pose_frame_profiles import default_rm65_workspace_near_boundary_runtime_pose_frame_contract


@dataclass(frozen=True)
class OpenPIArtifactPreflightReport:
    """Serializable result of a policy-artifact-only deployment check."""

    artifact: PolicyArtifactContract
    action_pose_mode: str
    runtime_pose_frame_contract_source: str
    session_contract: Mapping[str, Any]
    server_contract: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        pose_training_data = self.artifact.pose_training_data
        return {
            "ok": True,
            "model_id": self.artifact.model_id,
            "checkpoint_step": self.artifact.checkpoint_step,
            "dataset_schema": self.artifact.dataset_schema,
            "action_pose_mode": self.action_pose_mode,
            "pose_training_origin": (
                None if pose_training_data is None else pose_training_data.origin
            ),
            "runtime_pose_frame_contract_source": self.runtime_pose_frame_contract_source,
            "state_input_mode": self.artifact.state_input_mode,
            "state_dim": self.artifact.state_input_dim,
            "action_dim": self.artifact.policy_io_layout.action_dim,
            "camera_slots": (
                []
                if self.artifact.camera_input is None
                else list(self.artifact.camera_input.server_camera_slots)
            ),
            "session_contract": dict(self.session_contract),
            "server_contract": dict(self.server_contract),
        }


def preflight_policy_artifact(
    artifact: PolicyArtifactContract,
    *,
    active_sides: Sequence[str] = ("left", "right"),
    runtime_pose_frame_contract: PoseFrameContract | None = None,
) -> OpenPIArtifactPreflightReport:
    """Validate one OpenPI artifact against the real deployment contracts.

    Native-v2.3 policy state uses the checked-in RM65 workspace-near-boundary runtime
    profile by default, matching ``teleop_runtime.deploy``.  A model trained
    through the explicit v2.2 raw-identity compatibility view deliberately
    leaves the runtime contract unset here; the shared session resolver then
    constructs and audits the same-source identity binding.
    """

    if not isinstance(artifact, PolicyArtifactContract):
        raise TypeError("artifact must be a PolicyArtifactContract.")
    active = tuple(str(side).strip().lower() for side in active_sides)
    resolved_runtime_contract = runtime_pose_frame_contract
    runtime_contract_source = "explicit"
    pose_training_data = artifact.pose_training_data
    if resolved_runtime_contract is None:
        if (
            pose_training_data is not None
            and pose_training_data.runtime_pose_binding
            == RUNTIME_POSE_BINDING_EXPLICIT_CANONICAL_DESK
        ):
            resolved_runtime_contract = default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
            runtime_contract_source = "default_rm65_dual_arm_workspace_near_boundary_v1"
        else:
            runtime_contract_source = "v22_identity_compatibility"

    action_pose_mode = resolve_policy_action_pose_mode_from_artifact(
        artifact,
        configured_mode="absolute",
        explicit=False,
    )
    session = resolve_deploy_session_contract(
        active_sides=active,
        policy_artifact=artifact,
        runtime_pose_frame_contract=resolved_runtime_contract,
        hardware_output=False,
        allow_missing_end_effectors=True,
    )
    server = deploy_contract_from_policy_artifact(
        artifact,
        active_sides=active,
    )
    return OpenPIArtifactPreflightReport(
        artifact=artifact,
        action_pose_mode=action_pose_mode,
        runtime_pose_frame_contract_source=runtime_contract_source,
        session_contract=session.to_dict(),
        server_contract=asdict(server),
    )


def preflight_policy_artifact_path(
    path: str | Path,
    *,
    active_sides: Sequence[str] = ("left", "right"),
    runtime_pose_frame_contract: PoseFrameContract | None = None,
) -> OpenPIArtifactPreflightReport:
    """Load and preflight a checkpoint sidecar without external I/O."""

    return preflight_policy_artifact(
        load_policy_artifact(path),
        active_sides=active_sides,
        runtime_pose_frame_contract=runtime_pose_frame_contract,
    )


__all__ = [
    "OpenPIArtifactPreflightReport",
    "preflight_policy_artifact",
    "preflight_policy_artifact_path",
]
