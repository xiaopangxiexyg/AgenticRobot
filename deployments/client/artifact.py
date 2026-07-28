"""Filesystem boundary for versioned policy artifact sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.policy_artifact import PolicyArtifactContract

from .contracts import DeployPolicyContract


def load_policy_artifact(path: str | Path) -> PolicyArtifactContract:
    """Load and validate a checkpoint/server policy artifact JSON sidecar."""

    artifact_path = Path(path).expanduser()
    with artifact_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Policy artifact {artifact_path} must contain a JSON object.")
    return PolicyArtifactContract.from_dict(payload)


def deploy_contract_from_policy_artifact(
    artifact: PolicyArtifactContract,
    *,
    protocol_version: str | None = None,
    active_sides: tuple[str, ...] = ("left", "right"),
    required_camera_slots: tuple[str, ...] | None = None,
    strict_identity: bool = False,
) -> DeployPolicyContract:
    """Derive a server protocol contract from one semantic authority.

    Model identity is provenance, not wire compatibility.  By default the
    returned contract deliberately omits the artifact's model id and
    checkpoint step so a server can switch to another model instance that
    preserves the same protocol.  Callers performing an explicit provenance
    audit may set ``strict_identity=True``.
    """

    artifact_slots = (
        () if artifact.camera_input is None else artifact.camera_input.server_camera_slots
    )
    if required_camera_slots is not None:
        requested_slots = tuple(str(slot) for slot in required_camera_slots)
        if artifact.camera_input is not None and requested_slots != artifact_slots:
            raise ValueError(
                "required_camera_slots disagrees with policy_artifact camera_input."
            )
        camera_slots = requested_slots
    else:
        camera_slots = artifact_slots
    return DeployPolicyContract(
        protocol_version=protocol_version,
        deploy_schema_version=artifact.deploy_schema_version,
        model_id=artifact.model_id if strict_identity else None,
        checkpoint_step=artifact.checkpoint_step if strict_identity else None,
        state_key=artifact.state_key,
        state_dim=artifact.state_input_dim,
        state_input_mode=artifact.state_input_mode,
        state_discrete_key=artifact.state_discrete_key,
        state_discrete_dim=artifact.policy_io_layout.state_discrete_dim,
        action_key=artifact.action_key,
        action_dim=artifact.policy_io_layout.action_dim,
        action_discrete_dim=artifact.policy_io_layout.action_discrete_dim,
        policy_io_layout_hash=artifact.layout_hash,
        active_sides=active_sides,
        required_camera_slots=camera_slots,
        hand_layout=None,
    )


__all__ = ["deploy_contract_from_policy_artifact", "load_policy_artifact"]
