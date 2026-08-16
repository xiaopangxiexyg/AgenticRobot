from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lab_control.command_registry import (
    load_command_registry,
    resolve_trusted_command,
)
from lab_control.remote_request import load_remote_profile
from lab_control.workspace import LabError


@pytest.fixture
def protected_inputs():
    repository = Path(__file__).resolve().parents[1]
    profile = load_remote_profile(
        repository / "integration" / "remote_profiles",
        "agenticrobot-openpi-gpu",
    )
    registry = load_command_registry(
        repository
        / "integration"
        / "command_registries"
        / "agenticrobot-openpi-gpu.json"
    )
    return profile, registry


def test_registry_resolves_only_profile_authorized_static_adapter(
    protected_inputs,
) -> None:
    profile, registry = protected_inputs

    spec = resolve_trusted_command(
        registry,
        profile=profile,
        command_id="agenticrobot-openpi-gpu-smoke",
    )

    assert spec.adapter_id == "agenticrobot-openpi-gpu-smoke-v1"
    assert spec.network_egress == "loopback-only"
    assert spec.downloads_permitted is False
    assert spec.device_access is False
    assert len(registry.sha256) == 64


def test_registry_rejects_raw_execution_fields(
    protected_inputs,
    tmp_path: Path,
) -> None:
    _, registry = protected_inputs
    payload = json.loads(registry.path.read_text(encoding="utf-8"))
    payload["commands"]["agenticrobot-openpi-gpu-smoke"]["argv"] = [
        "python",
        "anything.py",
    ]
    path = tmp_path / "unsafe-registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabError, match="invalid command"):
        load_command_registry(path)


def test_registry_rejects_network_or_profile_expansion(
    protected_inputs,
    tmp_path: Path,
) -> None:
    profile, registry = protected_inputs
    payload = json.loads(registry.path.read_text(encoding="utf-8"))
    payload["commands"]["agenticrobot-openpi-gpu-smoke"][
        "downloads_permitted"
    ] = True
    path = tmp_path / "download-registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabError, match="unsafe policy"):
        load_command_registry(path)

    unlisted = replace(profile, command_ids=("some-other-command",))
    with pytest.raises(LabError, match="not authorized"):
        resolve_trusted_command(
            registry,
            profile=unlisted,
            command_id="agenticrobot-openpi-gpu-smoke",
        )
