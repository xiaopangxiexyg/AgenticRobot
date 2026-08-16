from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lab_control.preparation_profile import load_preparation_profile
from lab_control.remote_preflight import load_host_profile
from lab_control.transport import load_transport_profile
from lab_control.workspace import LabError


def _paths() -> tuple[Path, Path]:
    repository = Path(__file__).resolve().parents[1]
    return (
        repository / "integration" / "host_profiles",
        repository / "integration" / "preparation_profiles",
    )


def test_protected_preparation_profile_matches_host_paths_and_policies() -> None:
    host_root, preparation_root = _paths()
    host = load_host_profile(host_root, "remote-gpu-hardware")

    profile = load_preparation_profile(
        preparation_root,
        "remote-gpu-hardware-bootstrap",
        host_profile=host,
    )

    assert profile.host_id == host.host_id
    assert profile.stable_sync["AgenticRobot"].path == Path(
        "/home/user/zc/AgenticRobot"
    )
    anchor = profile.source_anchors["openpi"]
    assert anchor.source_path == Path("/home/user/zc/openpi")
    assert anchor.target_path == host.repositories["openpi"].path
    assert set(anchor.submodule_sources) == {
        "third_party/aloha",
        "third_party/libero",
    }
    assert profile.control_plane.install_root == Path(
        "/home/user/zc/robot-stack/control-plane"
    )
    assert profile.control_plane.current_link == Path(
        "/home/user/zc/robot-stack/control-plane/current"
    )
    assert (
        profile.control_plane.current_link / "projectctl"
        == host.control_plane_projectctl
    )
    repository = Path(__file__).resolve().parents[1]
    transport = load_transport_profile(
        repository / "integration" / "transport_profiles",
        "remote-gpu-hardware-ssh",
    )
    assert transport.remote_projectctl == host.control_plane_projectctl
    assert profile.control_plane.max_versions == 3
    assert profile.control_plane.environment_id == "agenticrobot-py310"
    assert len(profile.sha256) == 64


def test_profile_rejects_target_escape_and_network_expansion(tmp_path: Path) -> None:
    host_root, preparation_root = _paths()
    host = load_host_profile(host_root, "remote-gpu-hardware")
    source = json.loads(
        (preparation_root / "remote-gpu-hardware-bootstrap.json").read_text(
            encoding="utf-8"
        )
    )
    escaped = copy.deepcopy(source)
    escaped["profile_id"] = "escaped"
    escaped["source_anchors"]["openpi"]["target_path"] = "/tmp/openpi"
    path = tmp_path / "escaped.json"
    path.write_text(json.dumps(escaped), encoding="utf-8")
    with pytest.raises(LabError, match="differs from host Profile"):
        load_preparation_profile(tmp_path, "escaped", host_profile=host)

    expanded = copy.deepcopy(source)
    expanded["profile_id"] = "expanded"
    expanded["network_policy"]["large_file_downloads"] = True
    path = tmp_path / "expanded.json"
    path.write_text(json.dumps(expanded), encoding="utf-8")
    with pytest.raises(LabError, match="unsafe network policy"):
        load_preparation_profile(tmp_path, "expanded", host_profile=host)

    unsafe_control = copy.deepcopy(source)
    unsafe_control["profile_id"] = "unsafe-control"
    unsafe_control["control_plane"]["current_link"] = "/tmp/current"
    path = tmp_path / "unsafe-control.json"
    path.write_text(json.dumps(unsafe_control), encoding="utf-8")
    with pytest.raises(LabError, match="unsafe control-plane placement"):
        load_preparation_profile(tmp_path, "unsafe-control", host_profile=host)
