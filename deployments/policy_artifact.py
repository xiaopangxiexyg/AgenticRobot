"""Lightweight filesystem boundary for policy artifact sidecars.

This module intentionally depends only on pure dataset contracts.  Server,
training, and evidence tooling can therefore inspect a checkpoint without
importing client runtimes, robot motion, hardware, or control dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.policy_artifact import PolicyArtifactContract


def load_policy_artifact_payload(path: str | Path) -> dict[str, Any]:
    """Load one JSON-object sidecar without discarding additive metadata."""

    artifact_path = Path(path).expanduser()
    with artifact_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Policy artifact {artifact_path} must contain a JSON object."
        )
    return {str(key): value for key, value in payload.items()}


def load_policy_artifact(path: str | Path) -> PolicyArtifactContract:
    """Load and validate one checkpoint/server policy artifact sidecar."""

    return PolicyArtifactContract.from_dict(load_policy_artifact_payload(path))


__all__ = ["load_policy_artifact", "load_policy_artifact_payload"]
