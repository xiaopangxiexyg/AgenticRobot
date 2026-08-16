from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lab_control.project_cli import main as project_main
from lab_control.remote_request import (
    build_remote_request,
    load_remote_profile,
    validate_remote_request,
)
from lab_control.workspace import LabError

CONTROL_VERSION = "0.3.0-m3-bootstrap"
ASSET_LOCK = {
    "lock_id": "test-assets-v1",
    "workload_id": "test-workload-v1",
    "sha256": "d" * 64,
}


@pytest.fixture
def profile():
    repository = Path(__file__).resolve().parents[1]
    return load_remote_profile(
        repository / "integration" / "remote_profiles",
        "agenticrobot-openpi-gpu",
    )


@pytest.fixture
def repositories() -> dict[str, dict[str, object]]:
    return {
        "AgenticRobot": {
            "commit": "1" * 40,
            "tree_oid": "2" * 40,
            "dirty": False,
            "submodules": {},
        },
        "openpi": {
            "commit": "3" * 40,
            "tree_oid": "4" * 40,
            "dirty": False,
            "submodules": {
                "third_party/aloha": "5" * 40,
                "third_party/libero": "6" * 40,
            },
        },
    }


@pytest.fixture
def assets() -> dict[str, dict[str, str]]:
    return {
        "openpi-checkpoint": {"asset_id": "checkpoint-v1", "sha256": "a" * 64},
        "openpi-config": {"asset_id": "config-v1", "sha256": "b" * 64},
        "openpi-norm-stats": {"asset_id": "norm-stats-v1", "sha256": "c" * 64},
    }


def test_remote_request_is_content_addressed_and_has_no_raw_command(
    profile,
    repositories,
    assets,
) -> None:
    first = build_remote_request(
        task_id="AR-0301",
        snapshot_id="snapshot-0301",
        repositories=repositories,
        assets=assets,
        asset_lock=ASSET_LOCK,
        profile=profile,
        control_plane_version=CONTROL_VERSION,
    )
    second = build_remote_request(
        task_id="AR-0301",
        snapshot_id="snapshot-0301",
        repositories=repositories,
        assets=assets,
        asset_lock=ASSET_LOCK,
        profile=profile,
        control_plane_version=CONTROL_VERSION,
    )

    assert first == second
    assert first["request_id"].startswith("Q-")
    assert len(first["idempotency_key"]) == 64
    assert first["asset_lock"] == ASSET_LOCK
    assert first["resources"] == {
        "gpu_count": 1,
        "device_access": False,
        "network_egress": "loopback-only",
        "downloads_permitted": False,
    }
    assert first["service"] == {
        "service_name": "openpi-policy",
        "repository": "openpi",
        "environment_id": "openpi-py311-cuda",
        "protocol": "openpi-policy-service.v1",
        "launch_mode": "ephemeral",
        "launch_command_id": "openpi-policy-service",
        "endpoint_scheme": "ws",
    }
    assert "argv" not in str(first)
    assert (
        validate_remote_request(
            first,
            profile=profile,
            control_plane_version=CONTROL_VERSION,
        )
        == first
    )


def test_remote_request_allows_project_without_workload_identity(
    profile,
    repositories,
    assets,
) -> None:
    request = build_remote_request(
        task_id="GENERIC-0301",
        snapshot_id="snapshot-generic-0301",
        repositories=repositories,
        assets=assets,
        asset_lock={
            "lock_id": "generic-assets-v1",
            "workload_id": None,
            "sha256": "e" * 64,
        },
        profile=profile,
        control_plane_version=CONTROL_VERSION,
    )

    assert request["asset_lock"]["workload_id"] is None
    assert (
        validate_remote_request(
            request,
            profile=profile,
            control_plane_version=CONTROL_VERSION,
        )
        == request
    )


def test_remote_request_binds_repository_links(
    profile,
    repositories,
    assets,
) -> None:
    linked_repositories = copy.deepcopy(repositories)
    child_commit = linked_repositories["openpi"]["commit"]
    linked_repositories["AgenticRobot"]["submodules"] = {
        "components/openpi": child_commit
    }
    links = [
        {
            "parent_repository": "AgenticRobot",
            "submodule_path": "components/openpi",
            "child_repository": "openpi",
            "commit": child_commit,
        }
    ]

    request = build_remote_request(
        task_id="GENERIC-0302",
        snapshot_id="snapshot-generic-0302",
        repositories=linked_repositories,
        repository_links=links,
        assets=assets,
        asset_lock=ASSET_LOCK,
        profile=profile,
        control_plane_version=CONTROL_VERSION,
    )

    assert request["schema"] == "lab.remote-request.v2"
    assert request["repository_links"] == links
    assert (
        validate_remote_request(
            request,
            profile=profile,
            control_plane_version=CONTROL_VERSION,
        )
        == request
    )

    mismatched = copy.deepcopy(linked_repositories)
    mismatched["openpi"]["commit"] = "7" * 40
    with pytest.raises(LabError, match="differs from repository identities"):
        build_remote_request(
            task_id="GENERIC-0302",
            snapshot_id="snapshot-generic-0302",
            repositories=mismatched,
            repository_links=links,
            assets=assets,
            asset_lock=ASSET_LOCK,
            profile=profile,
            control_plane_version=CONTROL_VERSION,
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("resources", "device_access"), True),
        (("resources", "gpu_count"), 2),
        (("resources", "network_egress"), "internet"),
        (("resources", "downloads_permitted"), True),
        (("asset_lock", "sha256"), "e" * 64),
        (("profile", "target_host"), "some-other-host"),
        (("repositories", "openpi", "dirty"), True),
        (("command_ids",), ["arbitrary-shell"]),
        (("service", "launch_mode"), "attach-existing"),
    ],
)
def test_remote_request_tampering_fails_closed(
    profile,
    repositories,
    assets,
    path,
    replacement,
) -> None:
    request = build_remote_request(
        task_id="AR-0302",
        snapshot_id="snapshot-0302",
        repositories=repositories,
        assets=assets,
        asset_lock=ASSET_LOCK,
        profile=profile,
        control_plane_version=CONTROL_VERSION,
    )
    tampered = copy.deepcopy(request)
    target = tampered
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement

    with pytest.raises(LabError, match="idempotency hash mismatch"):
        validate_remote_request(
            tampered,
            profile=profile,
            control_plane_version=CONTROL_VERSION,
        )


def test_request_rejects_missing_asset_and_unpinned_submodule(
    profile,
    repositories,
    assets,
) -> None:
    missing_asset = dict(assets)
    missing_asset.pop("openpi-norm-stats")
    with pytest.raises(LabError, match="assets do not match"):
        build_remote_request(
            task_id="AR-0303",
            snapshot_id="snapshot-0303",
            repositories=repositories,
            assets=missing_asset,
            asset_lock=ASSET_LOCK,
            profile=profile,
            control_plane_version=CONTROL_VERSION,
        )

    invalid_submodule = copy.deepcopy(repositories)
    invalid_submodule["openpi"]["submodules"]["third_party/aloha"] = "unpinned"
    with pytest.raises(LabError, match="invalid submodule identity"):
        build_remote_request(
            task_id="AR-0303",
            snapshot_id="snapshot-0303",
            repositories=invalid_submodule,
            assets=assets,
            asset_lock=ASSET_LOCK,
            profile=profile,
            control_plane_version=CONTROL_VERSION,
        )


def test_request_rejects_management_paths_and_extra_resources(
    profile,
    repositories,
    assets,
) -> None:
    with_management_path = copy.deepcopy(repositories)
    with_management_path["AgenticRobot"]["management_path"] = "/local/path"
    request = build_remote_request(
        task_id="AR-0304",
        snapshot_id="snapshot-0304",
        repositories=with_management_path,
        assets=assets,
        asset_lock=ASSET_LOCK,
        profile=profile,
        control_plane_version=CONTROL_VERSION,
    )
    assert "management_path" not in request["repositories"]["AgenticRobot"]

    profile_path = (
        Path(__file__).resolve().parents[1] / "integration" / "remote_profiles"
    )
    assert load_remote_profile(profile_path, profile.profile_id).device_access is False
    assert (
        load_remote_profile(profile_path, profile.profile_id).network_egress
        == "loopback-only"
    )
    assert (
        load_remote_profile(profile_path, profile.profile_id).downloads_permitted
        is False
    )

    with pytest.raises(LabError, match="exact asset lock identity"):
        build_remote_request(
            task_id="AR-0304",
            snapshot_id="snapshot-0304",
            repositories=repositories,
            assets=assets,
            asset_lock={"lock_id": "test-assets-v1"},
            profile=profile,
            control_plane_version=CONTROL_VERSION,
        )


def test_request_validator_rejects_non_object_payloads(
    profile,
    repositories,
    assets,
) -> None:
    request = build_remote_request(
        task_id="AR-0305",
        snapshot_id="snapshot-0305",
        repositories=repositories,
        assets=assets,
        asset_lock=ASSET_LOCK,
        profile=profile,
        control_plane_version=CONTROL_VERSION,
    )
    malformed = copy.deepcopy(request)
    malformed["repositories"] = []
    content = {
        key: value
        for key, value in malformed.items()
        if key not in {"request_id", "idempotency_key"}
    }
    key = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    malformed["idempotency_key"] = key
    malformed["request_id"] = f"Q-{key[:24]}"

    with pytest.raises(LabError, match="must be objects"):
        validate_remote_request(
            malformed,
            profile=profile,
            control_plane_version=CONTROL_VERSION,
        )


def test_projectctl_builds_and_validates_request_without_execution(
    repositories,
    assets,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    assets_path = tmp_path / "assets.json"
    request_path = tmp_path / "request.json"
    snapshot_path.write_text(
        json.dumps({"repositories": repositories}), encoding="utf-8"
    )
    assets_path.write_text(
        json.dumps({**ASSET_LOCK, "assets": assets}), encoding="utf-8"
    )

    result = project_main(
        [
            "remote-request",
            "build",
            "--task-id",
            "AR-0306",
            "--snapshot-id",
            "snapshot-0306",
            "--snapshot",
            str(snapshot_path),
            "--assets",
            str(assets_path),
            "--profile",
            "agenticrobot-openpi-gpu",
            "--control-version",
            CONTROL_VERSION,
        ]
    )
    assert result == 0
    request = json.loads(capsys.readouterr().out)
    request_path.write_text(json.dumps(request), encoding="utf-8")

    result = project_main(
        [
            "remote-request",
            "validate",
            str(request_path),
            "--profile",
            "agenticrobot-openpi-gpu",
            "--control-version",
            CONTROL_VERSION,
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == request
