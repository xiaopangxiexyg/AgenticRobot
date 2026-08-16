from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from lab_control.asset_lock import load_asset_lock
from lab_control.asset_observation import load_asset_observations
from lab_control.asset_selection import (
    build_runtime_asset_verification,
    build_selected_asset_lock,
    collect_artifact_manifest,
    load_workload_selection,
    validate_artifact_manifest,
)
from lab_control.project_cli import (
    _require_file_under_roots,
    _require_protected_file,
    _write_new_object_under_roots,
)
from lab_control.workspace import LabError


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "component"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Lab Test")
    _git(repository, "config", "user.email", "lab-test@example.invalid")
    _git(repository, "remote", "add", "origin", "git@example.invalid:Example/Component.git")
    (repository / ".gitignore").write_text(
        "artifacts/\nassets/\n",
        encoding="utf-8",
    )
    registry = repository / "configs" / "dexteleop_policy_registry.json"
    registry.parent.mkdir()
    registry.write_text('{"workloads": {}}\n', encoding="utf-8")
    _git(repository, "add", ".gitignore", "configs/dexteleop_policy_registry.json")
    _git(repository, "commit", "-m", "source identity")

    artifact = repository / "artifacts" / "workload" / "100"
    (artifact / "payload").mkdir(parents=True)
    (artifact / "payload" / "part-0").write_bytes(b"artifact-0")
    (artifact / "metadata.json").write_text('{"version": 100}\n', encoding="utf-8")
    companion = repository / "assets" / "workload" / "metadata.json"
    companion.parent.mkdir(parents=True)
    companion.write_text('{"value": [0]}\n', encoding="utf-8")

    observations_path = tmp_path / "observations.json"
    observations_path.write_text(
        json.dumps(
            {
                "schema": "lab.asset-observations.v2",
                "observation_id": "component-assets-test",
                "host_id": "remote-gpu-hardware",
                "repository": {
                    "canonical_identity": "Example/Component",
                    "commit": _git(repository, "rev-parse", "HEAD"),
                    "tree_oid": _git(repository, "rev-parse", "HEAD^{tree}"),
                },
                "registry": {
                    "asset_role": "runtime-config",
                    "relative_path": "configs/dexteleop_policy_registry.json",
                    "sha256": _sha256(registry),
                },
                "candidates": {
                    "workload-latest": {
                        "candidate_id": "workload-latest",
                        "content": {
                            "asset_role": "runtime-content",
                            "relative_path": "artifacts/workload/100",
                            "exists": True,
                            "content_manifest_sha256": None,
                        },
                        "companion_assets": {
                            "runtime-config": {
                                "asset_id": "runtime-registry-test",
                                "relative_path": "configs/dexteleop_policy_registry.json",
                                "sha256": _sha256(registry),
                            },
                            "runtime-metadata": {
                                "asset_id": "workload-metadata-test",
                                "relative_path": "assets/workload/metadata.json",
                                "sha256": _sha256(companion),
                            },
                        },
                        "selection_authorized": False,
                        "activation_ready": False,
                        "blockers": [
                            "content-manifest",
                            "explicit-selection",
                        ],
                    }
                },
                "observed_at": "2026-08-14T16:00:00+08:00",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return repository, observations_path


def _selection(
    tmp_path: Path,
    *,
    observations_path: Path,
    manifest: dict[str, object],
) -> Path:
    observations = load_asset_observations(observations_path)
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "schema": "lab.workload-selection.v1",
                "selection_id": "gpu-smoke-workload-latest",
                "observation_id": observations.observation_id,
                "observation_sha256": observations.sha256,
                "artifact_manifest_receipt_sha256": manifest["receipt_sha256"],
                "workload_id": "workload-latest",
                "purpose": "gpu-only-validation",
                "selected_at": "2026-08-14T16:01:00+08:00",
                "device_access": False,
                "network_egress": "loopback-only",
                "downloads_permitted": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_artifact_manifest_and_explicit_selection_build_a_resolved_lock(
    tmp_path: Path,
) -> None:
    repository, observations_path = _fixture(tmp_path)
    observations = load_asset_observations(observations_path)

    manifest = collect_artifact_manifest(
        observations,
        workload_id="workload-latest",
        source_root=repository,
        expected_canonical_identity="Example/Component",
    )

    assert manifest["file_count"] == 2
    assert manifest["total_bytes"] > 0
    assert manifest["network_used"] is False
    assert manifest["large_file_downloaded"] is False
    assert manifest["selection_authorized"] is False
    selection = load_workload_selection(
        _selection(
            tmp_path,
            observations_path=observations_path,
            manifest=manifest,
        )
    )
    lock = build_selected_asset_lock(
        observations,
        artifact_manifest=manifest,
        selection=selection,
    )
    assert lock["workload_id"] == "workload-latest"
    assert lock["unresolved"] == []
    assert lock["assets"]["runtime-content"]["sha256"] == manifest[
        "content_sha256"
    ]
    lock_path = tmp_path / "remote-assets.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    active_lock = load_asset_lock(lock_path)
    assert active_lock.workload_id == "workload-latest"
    runtime = build_runtime_asset_verification(
        observations,
        artifact_manifest=manifest,
        selection=selection,
        asset_lock=active_lock,
        source_root=repository,
        expected_canonical_identity="Example/Component",
    )
    assert runtime["content_rehashed"] is True
    assert runtime["assets"]["runtime-content"]["path"].endswith(
        "/artifacts/workload/100"
    )


def test_resolved_asset_lock_does_not_require_a_workload_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "generic-assets.json"
    path.write_text(
        json.dumps(
            {
                "schema": "lab.asset-lock.v1",
                "lock_id": "generic-assets-v1",
                "workload_id": None,
                "assets": {
                    "runtime-config": {
                        "asset_id": "runtime-config-v1",
                        "sha256": "a" * 64,
                    }
                },
                "unresolved": [],
            }
        ),
        encoding="utf-8",
    )

    lock = load_asset_lock(path)

    assert lock.workload_id is None
    assert tuple(lock.assets) == ("runtime-config",)


def test_artifact_manifest_rejects_tampering_and_symlinks(tmp_path: Path) -> None:
    repository, observations_path = _fixture(tmp_path)
    observations = load_asset_observations(observations_path)
    manifest = collect_artifact_manifest(
        observations,
        workload_id="workload-latest",
        source_root=repository,
        expected_canonical_identity="Example/Component",
    )
    tampered = copy.deepcopy(manifest)
    tampered["files"]["metadata.json"]["sha256"] = "f" * 64
    with pytest.raises(LabError, match="content identity mismatch"):
        validate_artifact_manifest(
            tampered,
            observations=observations,
            workload_id="workload-latest",
        )

    (repository / "artifacts" / "workload" / "100" / "escape").symlink_to(
        repository / ".gitignore"
    )
    with pytest.raises(LabError, match="symlink"):
        collect_artifact_manifest(
            observations,
            workload_id="workload-latest",
            source_root=repository,
            expected_canonical_identity="Example/Component",
        )


def test_selection_is_bound_to_observation_and_artifact_receipt(tmp_path: Path) -> None:
    repository, observations_path = _fixture(tmp_path)
    observations = load_asset_observations(observations_path)
    manifest = collect_artifact_manifest(
        observations,
        workload_id="workload-latest",
        source_root=repository,
        expected_canonical_identity="Example/Component",
    )
    selection_path = _selection(
        tmp_path,
        observations_path=observations_path,
        manifest=manifest,
    )
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    payload["artifact_manifest_receipt_sha256"] = "0" * 64
    selection_path.write_text(json.dumps(payload), encoding="utf-8")
    selection = load_workload_selection(selection_path)

    with pytest.raises(LabError, match="does not match artifact manifest"):
        build_selected_asset_lock(
            observations,
            artifact_manifest=manifest,
            selection=selection,
        )


def test_asset_control_paths_reject_aliases_and_overwrite(tmp_path: Path) -> None:
    asset_root = (tmp_path / "assets").resolve()
    asset_root.mkdir()
    manifest = asset_root / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")

    assert _require_file_under_roots(
        manifest,
        roots=(asset_root,),
        label="artifact manifest",
    ) == manifest

    alias = asset_root / "alias"
    alias.symlink_to(asset_root, target_is_directory=True)
    with pytest.raises(LabError, match="outside protected roots"):
        _require_file_under_roots(
            alias / "manifest.json",
            roots=(asset_root,),
            label="artifact manifest",
        )

    protected_root = (tmp_path / "protected").resolve()
    protected_root.mkdir()
    protected_directory = protected_root / "reviewed"
    protected_directory.mkdir()
    selection = protected_directory / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")
    protected_alias = protected_root / "selection-alias"
    protected_alias.symlink_to(protected_directory, target_is_directory=True)
    with pytest.raises(LabError, match="not a protected control-plane file"):
        _require_protected_file(
            protected_alias / "selection.json",
            root=protected_root,
            label="workload selection",
        )

    output = asset_root / "new-manifest.json"
    assert _write_new_object_under_roots(
        output,
        {"schema": "test"},
        allowed_roots=(asset_root,),
    ) == output
    with pytest.raises(LabError, match="output is unsafe"):
        _write_new_object_under_roots(
            output,
            {"schema": "replacement"},
            allowed_roots=(asset_root,),
        )
