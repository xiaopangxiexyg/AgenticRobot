from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from lab_control.control_plane_activation import (
    activate_control_plane_version,
    rollback_control_plane_version,
)
from lab_control.control_plane_package import (
    build_control_plane_package,
    verify_control_plane_package,
)
from lab_control.control_plane_stage import stage_control_plane_package
from lab_control.preparation_profile import (
    ControlPlanePlacement,
    PreparationProfile,
    StableSyncPlacement,
)
from lab_control.project_cli import main as project_main
from lab_control.remote_preflight import (
    EnvironmentPlacement,
    HostNetworkPolicy,
    HostProfile,
    RepositoryPlacement,
)
from lab_control.workspace import LabError


def _run(*arguments: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    repository = (tmp_path / "AgenticRobot").resolve()
    _run("git", "init", "-b", "main", str(repository))
    _run("git", "config", "user.name", "Package Test", cwd=repository)
    _run(
        "git",
        "config",
        "user.email",
        "package@example.invalid",
        cwd=repository,
    )
    files = {
        "projectctl": "#!/usr/bin/env python3\n",
        "lab_control/project_cli.py": "VALUE = 1\n",
        "lab_control/managed_service.py": "VALUE = 2\n",
        "lab_control/service_adapters.py": "VALUE = 3\n",
        "lab_control/service_launch_plan.py": "VALUE = 4\n",
        "lab_control/worker.py": "VALUE = 5\n",
        "integration/command_registries/gpu.json": "{}\n",
        "integration/project.json": "{}\n",
        "integration/host_profiles/host.json": "{}\n",
        "integration/preparation_profiles/bootstrap.json": "{}\n",
        "integration/remote_profiles/gpu.json": "{}\n",
        "integration/service_adapters/service.json": "{}\n",
        "stack/remote-assets.json": "{}\n",
        "stack/remote-asset-observations.json": "{}\n",
        "stack/workload-selections/README.md": "# Reviewed selections\n",
        "tests/should-not-ship.py": "SECRET = False\n",
        "models/should-not-ship.bin": "not a model\n",
    }
    for relative_path, content in files.items():
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (repository / "projectctl").chmod(0o755)
    _run("git", "add", ".", cwd=repository)
    _run("git", "commit", "-m", "protected", cwd=repository)
    _run(
        "git",
        "remote",
        "add",
        "origin",
        "git@example.invalid:TestOrg/AgenticRobot.git",
        cwd=repository,
    )
    _run(
        "git",
        "update-ref",
        "refs/remotes/origin/main",
        "HEAD",
        cwd=repository,
    )
    return repository


def _host_profile(
    tmp_path: Path,
    profile: PreparationProfile,
) -> HostProfile:
    environment_root = Path(sys.prefix).resolve()
    return HostProfile(
        host_id=profile.host_id,
        hostname="test-host",
        capabilities=("control-plane",),
        repositories={
            "AgenticRobot": RepositoryPlacement(
                name="AgenticRobot",
                canonical_identity="TestOrg/AgenticRobot",
                path=tmp_path / "remote-stable",
            )
        },
        environments={
            "test-control-env": EnvironmentPlacement(
                environment_id="test-control-env",
                path=environment_root,
                python_major_minor=f"{sys.version_info.major}.{sys.version_info.minor}",
                repository="AgenticRobot",
                import_modules=("lab_control",),
                source_roots=(".",),
            )
        },
        asset_roots=(tmp_path / "assets",),
        slots=("test-slot",),
        slot_root=tmp_path / "slots",
        control_plane_projectctl=profile.control_plane.current_link / "projectctl",
        device_access=False,
        network_policy=HostNetworkPolicy(
            task_payload_egress="loopback-only",
            trusted_prepare_egress="explicit-only",
            implicit_downloads=False,
            large_file_downloads=False,
        ),
        sha256="b" * 64,
    )


def test_package_contains_only_protected_small_control_plane(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    output = (tmp_path / "control-plane.tar").resolve()

    receipt = build_control_plane_package(
        repository,
        expected_identity="TestOrg/AgenticRobot",
        output_path=output,
    )

    assert receipt["network_used"] is False
    assert receipt["working_tree_content_used"] is False
    assert receipt["large_files_included"] is False
    assert receipt["execution_authorized"] is False
    assert receipt["device_access"] is False
    verification = verify_control_plane_package(
        output,
        receipt,
        expected_identity="TestOrg/AgenticRobot",
        expected_commit=str(receipt["commit"]),
    )
    assert verification["verified"] is True
    assert verification["extracted"] is False
    assert verification["executed"] is False
    with tarfile.open(output, "r:") as archive:
        paths = set(archive.getnames())
    assert "projectctl" in paths
    assert "lab_control/project_cli.py" in paths
    assert "lab_control/managed_service.py" in paths
    assert "lab_control/service_adapters.py" in paths
    assert "lab_control/service_launch_plan.py" in paths
    assert "integration/project.json" in paths
    assert "integration/service_adapters/service.json" in paths
    assert "stack/workload-selections/README.md" in paths
    assert "tests/should-not-ship.py" not in paths
    assert "models/should-not-ship.bin" not in paths
    assert {entry["path"] for entry in receipt["entries"]} <= paths


def test_package_verification_rejects_tampering(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    output = (tmp_path / "control-plane.tar").resolve()
    receipt = build_control_plane_package(
        repository,
        expected_identity="TestOrg/AgenticRobot",
        output_path=output,
    )
    with output.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(LabError, match="archive identity mismatch"):
        verify_control_plane_package(
            output,
            receipt,
            expected_identity="TestOrg/AgenticRobot",
            expected_commit=str(receipt["commit"]),
        )


def test_package_refuses_dirty_or_unpublished_control_plane(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(LabError, match="clean main"):
        build_control_plane_package(
            repository,
            expected_identity="TestOrg/AgenticRobot",
            output_path=(tmp_path / "dirty.tar").resolve(),
        )
    (repository / "dirty.txt").unlink()
    (repository / "projectctl").write_text("changed\n", encoding="utf-8")
    _run("git", "commit", "-am", "unpublished", cwd=repository)
    with pytest.raises(LabError, match="not exact protected"):
        build_control_plane_package(
            repository,
            expected_identity="TestOrg/AgenticRobot",
            output_path=(tmp_path / "unpublished.tar").resolve(),
        )


def test_package_refuses_large_control_file(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    large = repository / "lab_control" / "large.py"
    large.write_bytes(b"x" * (1024 * 1024 + 1))
    _run("git", "add", "lab_control/large.py", cwd=repository)
    _run("git", "commit", "-m", "large", cwd=repository)
    _run(
        "git",
        "update-ref",
        "refs/remotes/origin/main",
        "HEAD",
        cwd=repository,
    )

    with pytest.raises(LabError, match="exceeds 1 MiB"):
        build_control_plane_package(
            repository,
            expected_identity="TestOrg/AgenticRobot",
            output_path=(tmp_path / "large.tar").resolve(),
        )


def test_projectctl_package_identity_comes_from_protected_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    placement = StableSyncPlacement(
        repository="AgenticRobot",
        canonical_identity="TestOrg/AgenticRobot",
        path=tmp_path / "remote-stable",
        branch="main",
        remote_ref="refs/remotes/origin/main",
    )
    profile = PreparationProfile(
        profile_id="bootstrap-v1",
        host_id="remote-host",
        stable_sync={"AgenticRobot": placement},
        source_anchors={},
        control_plane=ControlPlanePlacement(
            repository="AgenticRobot",
            install_root=tmp_path / "control-plane",
            current_link=tmp_path / "control-plane" / "current",
            environment_id="test-control-env",
            max_versions=3,
        ),
        allowed_target_roots=(tmp_path,),
        path=tmp_path / "profile.json",
        sha256="a" * 64,
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "lab_control.project_cli.load_host_profile",
        lambda root, profile_id: object(),
    )
    monkeypatch.setattr(
        "lab_control.project_cli.load_preparation_profile",
        lambda root, profile_id, host_profile: profile,
    )

    def build(
        repository: Path,
        *,
        expected_identity: str,
        output_path: Path,
    ) -> dict[str, object]:
        observed["repository"] = repository
        observed["expected_identity"] = expected_identity
        observed["output_path"] = output_path
        return {"schema": "test.package", "execution_authorized": False}

    monkeypatch.setattr("lab_control.project_cli.build_control_plane_package", build)
    output = (tmp_path / "package.tar").resolve()
    assert (
        project_main(
            [
                "control-plane",
                "package",
                "--preparation-profile",
                "bootstrap-v1",
                "--host-profile",
                "remote-host",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["execution_authorized"] is False
    assert observed["expected_identity"] == "TestOrg/AgenticRobot"
    assert observed["output_path"] == output


def test_verified_package_stages_version_without_activation(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    archive = (tmp_path / "control-plane.tar").resolve()
    package = build_control_plane_package(
        repository,
        expected_identity="TestOrg/AgenticRobot",
        output_path=archive,
    )
    install_root = (tmp_path / "install").resolve()
    install_root.mkdir()
    profile = PreparationProfile(
        profile_id="bootstrap-v1",
        host_id="remote-host",
        stable_sync={
            "AgenticRobot": StableSyncPlacement(
                repository="AgenticRobot",
                canonical_identity="TestOrg/AgenticRobot",
                path=tmp_path / "remote-stable",
                branch="main",
                remote_ref="refs/remotes/origin/main",
            )
        },
        source_anchors={},
        control_plane=ControlPlanePlacement(
            repository="AgenticRobot",
            install_root=install_root,
            current_link=install_root / "current",
            environment_id="test-control-env",
            max_versions=3,
        ),
        allowed_target_roots=(install_root,),
        path=tmp_path / "profile.json",
        sha256="a" * 64,
    )

    staged = stage_control_plane_package(
        profile,
        archive_path=archive,
        package_receipt=package,
    )

    version = Path(str(staged["version_path"]))
    assert version == install_root / "versions" / str(package["commit"])
    assert (version / "projectctl").is_file()
    assert (version / ".lab-stage-receipt.json").is_file()
    assert not (version / "tests").exists()
    assert staged["active"] is False
    assert staged["current_link_updated"] is False
    assert staged["executed"] is False
    assert staged["device_access"] is False
    assert not (install_root / "current").exists()

    with pytest.raises(LabError, match="already exists"):
        stage_control_plane_package(
            profile,
            archive_path=archive,
            package_receipt=package,
        )


def test_projectctl_stages_package_without_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _repository(tmp_path)
    archive = (tmp_path / "control-plane.tar").resolve()
    package = build_control_plane_package(
        repository,
        expected_identity="TestOrg/AgenticRobot",
        output_path=archive,
    )
    receipt_path = tmp_path / "package-receipt.json"
    receipt_path.write_text(json.dumps(package), encoding="utf-8")
    install_root = (tmp_path / "install").resolve()
    install_root.mkdir()
    profile = PreparationProfile(
        profile_id="bootstrap-v1",
        host_id="remote-host",
        stable_sync={
            "AgenticRobot": StableSyncPlacement(
                repository="AgenticRobot",
                canonical_identity="TestOrg/AgenticRobot",
                path=tmp_path / "remote-stable",
                branch="main",
                remote_ref="refs/remotes/origin/main",
            )
        },
        source_anchors={},
        control_plane=ControlPlanePlacement(
            repository="AgenticRobot",
            install_root=install_root,
            current_link=install_root / "current",
            environment_id="test-control-env",
            max_versions=3,
        ),
        allowed_target_roots=(install_root,),
        path=tmp_path / "profile.json",
        sha256="a" * 64,
    )
    monkeypatch.setattr(
        "lab_control.project_cli.load_host_profile",
        lambda root, profile_id: object(),
    )
    monkeypatch.setattr(
        "lab_control.project_cli.load_preparation_profile",
        lambda root, profile_id, host_profile: profile,
    )

    assert (
        project_main(
            [
                "control-plane",
                "stage",
                "--preparation-profile",
                "bootstrap-v1",
                "--host-profile",
                "remote-host",
                "--archive",
                str(archive),
                "--receipt",
                str(receipt_path),
            ]
        )
        == 0
    )
    staged = json.loads(capsys.readouterr().out)
    assert staged["active"] is False
    assert staged["current_link_updated"] is False
    assert not (install_root / "current").exists()


def test_staged_package_health_checks_and_activates_atomically(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    archive = (tmp_path / "control-plane.tar").resolve()
    package = build_control_plane_package(
        repository,
        expected_identity="TestOrg/AgenticRobot",
        output_path=archive,
    )
    install_root = (tmp_path / "install").resolve()
    install_root.mkdir()
    profile = PreparationProfile(
        profile_id="bootstrap-v1",
        host_id="remote-host",
        stable_sync={
            "AgenticRobot": StableSyncPlacement(
                repository="AgenticRobot",
                canonical_identity="TestOrg/AgenticRobot",
                path=tmp_path / "remote-stable",
                branch="main",
                remote_ref="refs/remotes/origin/main",
            )
        },
        source_anchors={},
        control_plane=ControlPlanePlacement(
            repository="AgenticRobot",
            install_root=install_root,
            current_link=install_root / "current",
            environment_id="test-control-env",
            max_versions=3,
        ),
        allowed_target_roots=(install_root,),
        path=tmp_path / "profile.json",
        sha256="a" * 64,
    )
    staged = stage_control_plane_package(
        profile,
        archive_path=archive,
        package_receipt=package,
    )

    activated = activate_control_plane_version(
        profile,
        _host_profile(tmp_path, profile),
        stage_receipt=staged,
    )

    assert activated["active"] is True
    assert activated["current_link_updated"] is True
    assert activated["control_plane_health_check_executed"] is True
    assert activated["task_payload_executed"] is False
    assert activated["network_downloads_permitted"] is False
    assert activated["device_access"] is False
    assert profile.control_plane.current_link.is_symlink()
    assert profile.control_plane.current_link.resolve() == Path(
        str(staged["version_path"])
    )


def test_activation_refuses_tampered_staged_file(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    archive = (tmp_path / "control-plane.tar").resolve()
    package = build_control_plane_package(
        repository,
        expected_identity="TestOrg/AgenticRobot",
        output_path=archive,
    )
    install_root = (tmp_path / "install").resolve()
    install_root.mkdir()
    profile = PreparationProfile(
        profile_id="bootstrap-v1",
        host_id="remote-host",
        stable_sync={
            "AgenticRobot": StableSyncPlacement(
                repository="AgenticRobot",
                canonical_identity="TestOrg/AgenticRobot",
                path=tmp_path / "remote-stable",
                branch="main",
                remote_ref="refs/remotes/origin/main",
            )
        },
        source_anchors={},
        control_plane=ControlPlanePlacement(
            repository="AgenticRobot",
            install_root=install_root,
            current_link=install_root / "current",
            environment_id="test-control-env",
            max_versions=3,
        ),
        allowed_target_roots=(install_root,),
        path=tmp_path / "profile.json",
        sha256="a" * 64,
    )
    staged = stage_control_plane_package(
        profile,
        archive_path=archive,
        package_receipt=package,
    )
    (Path(str(staged["version_path"])) / "lab_control/worker.py").write_text(
        "tampered = True\n",
        encoding="utf-8",
    )

    with pytest.raises(LabError, match="file mismatch"):
        activate_control_plane_version(
            profile,
            _host_profile(tmp_path, profile),
            stage_receipt=staged,
        )
    assert not profile.control_plane.current_link.exists()


def test_activation_retains_previous_version_for_health_checked_rollback(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    install_root = (tmp_path / "install").resolve()
    install_root.mkdir()
    profile = PreparationProfile(
        profile_id="bootstrap-v1",
        host_id="remote-host",
        stable_sync={
            "AgenticRobot": StableSyncPlacement(
                repository="AgenticRobot",
                canonical_identity="TestOrg/AgenticRobot",
                path=tmp_path / "remote-stable",
                branch="main",
                remote_ref="refs/remotes/origin/main",
            )
        },
        source_anchors={},
        control_plane=ControlPlanePlacement(
            repository="AgenticRobot",
            install_root=install_root,
            current_link=install_root / "current",
            environment_id="test-control-env",
            max_versions=3,
        ),
        allowed_target_roots=(install_root,),
        path=tmp_path / "profile.json",
        sha256="a" * 64,
    )
    host = _host_profile(tmp_path, profile)

    archive_v1 = (tmp_path / "control-plane-v1.tar").resolve()
    package_v1 = build_control_plane_package(
        repository,
        expected_identity="TestOrg/AgenticRobot",
        output_path=archive_v1,
    )
    stage_v1 = stage_control_plane_package(
        profile,
        archive_path=archive_v1,
        package_receipt=package_v1,
    )
    activate_control_plane_version(profile, host, stage_receipt=stage_v1)

    (repository / "projectctl").write_text(
        "#!/usr/bin/env python3\nVALUE = 2\n",
        encoding="utf-8",
    )
    _run("git", "commit", "-am", "protected v2", cwd=repository)
    _run(
        "git",
        "update-ref",
        "refs/remotes/origin/main",
        "HEAD",
        cwd=repository,
    )
    archive_v2 = (tmp_path / "control-plane-v2.tar").resolve()
    package_v2 = build_control_plane_package(
        repository,
        expected_identity="TestOrg/AgenticRobot",
        output_path=archive_v2,
    )
    stage_v2 = stage_control_plane_package(
        profile,
        archive_path=archive_v2,
        package_receipt=package_v2,
    )
    active_v2 = activate_control_plane_version(profile, host, stage_receipt=stage_v2)
    assert active_v2["previous_commit"] == package_v1["commit"]
    assert active_v2["rollback_available"] is True

    rollback = rollback_control_plane_version(
        profile,
        host,
        activation_receipt=active_v2,
    )

    assert rollback["rolled_back"] is True
    assert rollback["from_commit"] == package_v2["commit"]
    assert rollback["restored_commit"] == package_v1["commit"]
    assert rollback["health_checked"] is True
    assert rollback["old_versions_deleted"] is False
    assert profile.control_plane.current_link.resolve() == Path(
        str(stage_v1["version_path"])
    )
    assert Path(str(stage_v2["version_path"])).is_dir()
