"""Application-side policy artifact and observation-contract binding.

The lower ``robot_motion.deploy`` package defines the contracts.  This module
binds them to the real deployment application's CLI namespace, loaded policy,
artifact, and model-I/O layout.  It does not load models, construct resources,
advance policy queues, or own action/safety semantics.
"""

from __future__ import annotations

import argparse
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.deploy.observation_view import (
    PolicyObservationContract,
    default_deploy_policy_observation_contract,
)
from robot_motion.deploy.schema import ALL_SIDES
from robot_motion.deploy.session_contract import (
    resolve_policy_action_pose_mode_from_artifact,
)


def bind_policy_pose_mode_from_artifact(
    args: argparse.Namespace,
    policy_artifact: Any,
    *,
    raw_argv: list[str],
) -> None:
    """Bind CLI pose mode to the shared artifact contract."""

    explicit = any(
        token == "--policy-action-pose-mode"
        or token.startswith("--policy-action-pose-mode=")
        for token in raw_argv
    )
    args.policy_action_pose_mode = resolve_policy_action_pose_mode_from_artifact(
        policy_artifact,
        configured_mode=str(args.policy_action_pose_mode),
        explicit=explicit,
    )


def build_policy_observation_contract(
    *,
    policy: Any,
    policy_artifact: Any,
    deploy_layout: Any,
    pose_mode: str,
) -> PolicyObservationContract:
    """Bind the strict actor view to the artifact/layout and adapter needs."""

    effective_layout = deploy_layout
    if effective_layout is None:
        wrapped_policy = getattr(policy, "policy", None)
        policy_config = getattr(wrapped_policy, "config", None)
        effective_layout = getattr(policy_config, "policy_io_layout", None)
    hand_state_keys: set[str] = set()
    if effective_layout is not None:
        for side in ALL_SIDES:
            spec = effective_layout.end_effector(side)
            if spec.present:
                hand_state_keys.add(str(spec.state_source_field))
            channel = effective_layout.channel("state", side)
            for token in channel.tokens:
                if token.source_metadata_key:
                    hand_state_keys.add(str(token.source_metadata_key))

    if policy_artifact is not None:
        artifact_hash = canonical_content_hash(
            policy_artifact.to_dict(),
            domain="dexteleop.policy_observation_artifact_binding.v1",
        )
        binding_id = f"artifact={artifact_hash}"
    elif effective_layout is not None:
        binding_id = f"layout={effective_layout.layout_hash};model=unversioned"
    else:
        binding_id = f"policy={getattr(policy, 'name', 'unknown')}"
    return default_deploy_policy_observation_contract(
        binding_id=binding_id,
        last_action_pose_mode=str(pose_mode),
        compatibility_hand_state_keys=tuple(sorted(hand_state_keys)),
    )


__all__ = [
    "bind_policy_pose_mode_from_artifact",
    "build_policy_observation_contract",
]
