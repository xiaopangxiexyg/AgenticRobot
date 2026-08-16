from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from lab_control import draft_cleanup
from lab_control.cli import main
from lab_control.draft_bundle import MAX_BUNDLE_MANIFEST_BYTES, verify_draft_bundle_set
from lab_control.project_cli import main as project_main
from lab_control.slot_materializer import materialize_draft_bundle_set
from lab_control.workspace import LabError


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip()


def _repository(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Draft Test")
    _git(path, "config", "user.email", "draft-test@example.invalid")
    (path / "README.md").write_text("initial\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    return path.resolve()


@pytest.fixture
def draft_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    repositories = tmp_path / "repositories"
    roots = {
        "AgenticRobot": _repository(repositories / "AgenticRobot"),
        "RPent": _repository(repositories / "RPent"),
        "openpi": _repository(repositories / "openpi"),
    }
    monkeypatch.setenv("LAB_REPOSITORY_AGENTICROBOT", str(roots["AgenticRobot"]))
    monkeypatch.setenv("LAB_REPOSITORY_RPENT", str(roots["RPent"]))
    monkeypatch.setenv("LAB_REPOSITORY_OPENPI", str(roots["openpi"]))
    monkeypatch.setenv(
        "LAB_PROJECT_MANIFEST",
        str(Path(__file__).resolve().parents[1] / "integration" / "project.json"),
    )
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "lab-home"))
    return roots


def _create_task(
    task_id: str,
    capsys: pytest.CaptureFixture[str],
    *,
    actor: str = "agent-draft-test",
    claim_writer: bool = True,
) -> Path:
    assert main(["task", "create", task_id, "--repos", "AgenticRobot", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    if claim_writer:
        assert (
            main(
                [
                    "task",
                    "writer",
                    "claim",
                    task_id,
                    "--repo",
                    "AgenticRobot",
                    "--actor",
                    actor,
                ]
            )
            == 0
        )
        capsys.readouterr()
    return Path(payload["workspace_root"]) / "AgenticRobot"


def _claim_writer(
    task_id: str,
    repository: str,
    actor: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "task",
                "writer",
                "claim",
                task_id,
                "--repo",
                repository,
                "--actor",
                actor,
            ]
        )
        == 0
    )
    capsys.readouterr()


def test_synthetic_snapshot_preserves_branch_head_index_and_worktree(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    task = _create_task("AR-DRAFT-01", capsys)
    (task / "README.md").write_text("staged\n", encoding="utf-8")
    _git(task, "add", "README.md")
    (task / "README.md").write_text("working tree wins\n", encoding="utf-8")
    before = {
        "branch": _git(task, "branch", "--show-current"),
        "head": _git(task, "rev-parse", "HEAD"),
        "status": _git(task, "status", "--short"),
        "cached": _git(task, "diff", "--cached", "--binary"),
        "working": _git(task, "diff", "--binary"),
    }

    assert main(["snapshot", "create", "AR-DRAFT-01", "--json"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    repository = snapshot["repositories"]["AgenticRobot"]

    assert snapshot["snapshot_type"] == "synthetic-draft-git"
    assert snapshot["remote_eligible"] is False
    assert snapshot["verify_eligible"] is False
    assert snapshot["device_eligible"] is False
    assert repository["synthetic"] is True
    assert repository["base_commit"] == before["head"]
    assert repository["commit"] != before["head"]
    assert (
        _git(task, "show", f"{repository['commit']}:README.md") == "working tree wins"
    )
    assert _git(task, "rev-parse", repository["ref"]) == repository["commit"]
    assert _git(task, "branch", "--show-current") == before["branch"]
    assert _git(task, "rev-parse", "HEAD") == before["head"]
    assert _git(task, "status", "--short") == before["status"]
    assert _git(task, "diff", "--cached", "--binary") == before["cached"]
    assert _git(task, "diff", "--binary") == before["working"]


def test_untracked_files_require_allowlist_and_secret_paths_fail_closed(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    task = _create_task("AR-DRAFT-02", capsys)
    (task / "new_source.py").write_text("VALUE = 1\n", encoding="utf-8")
    (task / ".env").write_text("TOKEN=secret\n", encoding="utf-8")

    assert main(["snapshot", "create", "AR-DRAFT-02"]) == 2
    assert "no captured changes" in capsys.readouterr().err

    assert (
        main(
            [
                "snapshot",
                "create",
                "AR-DRAFT-02",
                "--include",
                "*.py",
                "--json",
            ]
        )
        == 0
    )
    snapshot = json.loads(capsys.readouterr().out)
    repository = snapshot["repositories"]["AgenticRobot"]
    assert repository["included_untracked"] == ["new_source.py"]
    assert repository["excluded_untracked"] == [".env"]
    assert _git(task, "show", f"{repository['commit']}:new_source.py") == "VALUE = 1"
    missing = subprocess.run(
        ["git", "-C", str(task), "cat-file", "-e", f"{repository['commit']}:.env"],
        check=False,
        capture_output=True,
    )
    assert missing.returncode != 0

    assert main(["snapshot", "create", "AR-DRAFT-02", "--include", "*"]) == 2
    assert "denylist rejects untracked path: .env" in capsys.readouterr().err


def test_oversized_untracked_file_is_rejected_without_moving_user_state(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    task = _create_task("AR-DRAFT-03", capsys)
    payload = task / "payload.bin"
    payload.write_bytes(b"x" * 32)
    before_head = _git(task, "rev-parse", "HEAD")
    before_status = _git(task, "status", "--short")

    assert (
        main(
            [
                "snapshot",
                "create",
                "AR-DRAFT-03",
                "--include",
                "*.bin",
                "--max-file-bytes",
                "16",
            ]
        )
        == 2
    )

    assert "refuses untracked file over 16 bytes" in capsys.readouterr().err
    assert _git(task, "rev-parse", "HEAD") == before_head
    assert _git(task, "status", "--short") == before_status
    assert _git(task, "for-each-ref", "--format=%(refname)", "refs/lab") == ""


def test_repeated_identical_snapshot_is_content_idempotent(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    task = _create_task("AR-DRAFT-04", capsys)
    (task / "README.md").write_text("draft\n", encoding="utf-8")

    assert main(["snapshot", "create", "AR-DRAFT-04", "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(["snapshot", "create", "AR-DRAFT-04", "--json"]) == 0
    second = json.loads(capsys.readouterr().out)

    assert first == second


def test_snapshot_identity_records_the_writer_barrier(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    task = _create_task(
        "AR-DRAFT-WRITER",
        capsys,
        actor="agent-draft-01",
    )
    (task / "README.md").write_text("writer barrier\n", encoding="utf-8")

    assert main(["snapshot", "create", "AR-DRAFT-WRITER", "--json"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    writer = snapshot["writer_leases"]["AgenticRobot"]
    assert writer["state"] == "ACTIVE"
    assert writer["actor"] == "agent-draft-01"
    assert writer["lease_id"].startswith("WL-")
    assert len(writer["lease_receipt_sha256"]) == 64


def test_dirty_snapshot_requires_an_active_writer_lease(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    task = _create_task(
        "AR-DRAFT-NO-WRITER",
        capsys,
        claim_writer=False,
    )
    (task / "README.md").write_text("unclaimed writer\n", encoding="utf-8")

    assert main(["snapshot", "create", "AR-DRAFT-NO-WRITER"]) == 2
    assert "requires an ACTIVE writer lease" in capsys.readouterr().err


def test_incremental_bundle_restores_exact_draft_in_base_only_clone(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    task = _create_task("AR-DRAFT-05", capsys)
    (task / "README.md").write_text("transported draft\n", encoding="utf-8")
    assert main(["snapshot", "create", "AR-DRAFT-05", "--json"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    snapshot_id = snapshot["snapshot_id"]
    synthetic_commit = snapshot["repositories"]["AgenticRobot"]["commit"]
    export_root = tmp_path / "exports"

    assert (
        main(
            [
                "snapshot",
                "export",
                "AR-DRAFT-05",
                snapshot_id,
                "--output",
                str(export_root),
                "--json",
            ]
        )
        == 0
    )
    exported = json.loads(capsys.readouterr().out)
    bundle_root = Path(exported["path"])
    bundle_manifest = bundle_root / "bundle.json"
    assert exported["bundle"]["transport_eligible"] is True
    assert exported["bundle"]["remote_run_eligible"] is False
    assert exported["verification"]["target_repository_mutated"] is False

    with pytest.raises(LabError, match="target already contains synthetic commit"):
        verify_draft_bundle_set(
            bundle_manifest,
            repository_paths={"AgenticRobot": task},
            require_synthetic_absent=True,
        )

    base_only = tmp_path / "base-only"
    _git(tmp_path, "clone", "--no-local", str(task), str(base_only))
    absent_before = subprocess.run(
        ["git", "-C", str(base_only), "cat-file", "-e", synthetic_commit],
        check=False,
        capture_output=True,
    )
    assert absent_before.returncode != 0

    receipt = verify_draft_bundle_set(
        bundle_manifest,
        repository_paths={"AgenticRobot": base_only},
        require_synthetic_absent=True,
    )

    assert receipt["source_verified"] is True
    assert receipt["materialized_in_temporary_repository"] is True
    assert receipt["repositories"]["AgenticRobot"]["commit"] == synthetic_commit
    assert receipt["repositories"]["AgenticRobot"]["synthetic_preexisting"] is False
    absent_after = subprocess.run(
        ["git", "-C", str(base_only), "cat-file", "-e", synthetic_commit],
        check=False,
        capture_output=True,
    )
    assert absent_after.returncode != 0

    assert (
        project_main(
            [
                "draft-bundle",
                "verify",
                str(bundle_manifest),
                "--repository",
                f"AgenticRobot={base_only}",
            ]
        )
        == 0
    )
    assert (
        json.loads(capsys.readouterr().out)["receipt_sha256"]
        == receipt["receipt_sha256"]
    )

    assert (
        main(
            [
                "snapshot",
                "export",
                "AR-DRAFT-05",
                snapshot_id,
                "--output",
                str(export_root),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == exported


def test_bundle_tampering_is_rejected_before_temporary_materialization(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    task = _create_task("AR-DRAFT-06", capsys)
    (task / "README.md").write_text("tamper target\n", encoding="utf-8")
    assert main(["snapshot", "create", "AR-DRAFT-06", "--json"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    export_root = tmp_path / "exports"
    assert (
        main(
            [
                "snapshot",
                "export",
                "AR-DRAFT-06",
                snapshot["snapshot_id"],
                "--output",
                str(export_root),
                "--json",
            ]
        )
        == 0
    )
    exported = json.loads(capsys.readouterr().out)
    bundle_root = Path(exported["path"])
    record = exported["bundle"]["repositories"]["AgenticRobot"]
    bundle_path = bundle_root / record["bundle_file"]
    original = bundle_path.read_bytes()
    bundle_path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))

    with pytest.raises(LabError, match="bundle file identity mismatch"):
        verify_draft_bundle_set(
            bundle_root / "bundle.json",
            repository_paths={"AgenticRobot": task},
            require_synthetic_absent=False,
        )


def test_multi_repository_bundle_omits_clean_repository_payload(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    assert (
        main(
            [
                "task",
                "create",
                "AR-DRAFT-07",
                "--repos",
                "AgenticRobot,RPent",
                "--json",
            ]
        )
        == 0
    )
    task_payload = json.loads(capsys.readouterr().out)
    task_root = Path(task_payload["workspace_root"])
    agenticrobot = task_root / "AgenticRobot"
    rpent = task_root / "RPent"
    _claim_writer("AR-DRAFT-07", "AgenticRobot", "agent-main-07", capsys)
    (agenticrobot / "README.md").write_text("multi-repo draft\n", encoding="utf-8")

    assert main(["snapshot", "create", "AR-DRAFT-07", "--json"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    assert snapshot["repositories"]["AgenticRobot"]["synthetic"] is True
    assert snapshot["repositories"]["RPent"]["synthetic"] is False

    export_root = tmp_path / "exports"
    assert (
        main(
            [
                "snapshot",
                "export",
                "AR-DRAFT-07",
                snapshot["snapshot_id"],
                "--output",
                str(export_root),
                "--json",
            ]
        )
        == 0
    )
    exported = json.loads(capsys.readouterr().out)
    rpent_record = exported["bundle"]["repositories"]["RPent"]
    assert rpent_record["bundle_file"] is None
    assert rpent_record["bundle_size"] == 0

    base_agenticrobot = tmp_path / "base-agenticrobot"
    base_rpent = tmp_path / "base-rpent"
    _git(tmp_path, "clone", "--no-local", str(agenticrobot), str(base_agenticrobot))
    _git(tmp_path, "clone", "--no-local", str(rpent), str(base_rpent))
    receipt = verify_draft_bundle_set(
        Path(exported["path"]) / "bundle.json",
        repository_paths={
            "AgenticRobot": base_agenticrobot,
            "RPent": base_rpent,
        },
        require_synthetic_absent=True,
    )
    assert receipt["repositories"]["AgenticRobot"]["synthetic"] is True
    assert receipt["repositories"]["RPent"]["synthetic"] is False
    assert receipt["repositories"]["RPent"]["verified"] is True


def test_bundle_verifier_rejects_oversized_manifest_before_parsing(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "bundle.json"
    manifest.write_text(
        json.dumps({"padding": "x" * MAX_BUNDLE_MANIFEST_BYTES}),
        encoding="utf-8",
    )

    with pytest.raises(LabError, match="protected byte limit"):
        verify_draft_bundle_set(
            manifest,
            repository_paths={},
            require_synthetic_absent=True,
        )


def test_verified_bundle_materializes_as_inert_detached_slot_source(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    task = _create_task("AR-DRAFT-08", capsys)
    (task / "README.md").write_text("materialized draft\n", encoding="utf-8")
    assert main(["snapshot", "create", "AR-DRAFT-08", "--json"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    export_root = tmp_path / "exports"
    assert (
        main(
            [
                "snapshot",
                "export",
                "AR-DRAFT-08",
                snapshot["snapshot_id"],
                "--output",
                str(export_root),
                "--json",
            ]
        )
        == 0
    )
    exported = json.loads(capsys.readouterr().out)
    manifest = Path(exported["path"]) / "bundle.json"
    synthetic_commit = snapshot["repositories"]["AgenticRobot"]["commit"]
    anchor = tmp_path / "clean-anchor"
    _git(tmp_path, "clone", "--no-local", str(task), str(anchor))
    anchor_before = {
        "head": _git(anchor, "rev-parse", "HEAD"),
        "status": _git(anchor, "status", "--short"),
    }

    receipt = materialize_draft_bundle_set(
        manifest,
        repository_paths={"AgenticRobot": anchor},
        slot_root=(tmp_path / "slots").resolve(),
        slot_id="gpu-slot-01",
    )
    checkout = Path(receipt["repositories"]["AgenticRobot"]["path"])

    assert receipt["source_verified"] is True
    assert receipt["source_anchor_mutated"] is False
    assert receipt["lfs_smudge_disabled"] is True
    assert receipt["execution_eligible"] is False
    assert receipt["remote_run_eligible"] is False
    assert _git(checkout, "branch", "--show-current") == ""
    assert _git(checkout, "rev-parse", "HEAD") == synthetic_commit
    assert _git(checkout, "status", "--short") == ""
    assert (checkout / "README.md").read_text(encoding="utf-8") == (
        "materialized draft\n"
    )
    assert _git(checkout, "config", "--get", "lfs.fetchexclude") == "*"
    assert _git(anchor, "rev-parse", "HEAD") == anchor_before["head"]
    assert _git(anchor, "status", "--short") == anchor_before["status"]
    assert (
        subprocess.run(
            ["git", "-C", str(anchor), "cat-file", "-e", synthetic_commit],
            check=False,
            capture_output=True,
        ).returncode
        != 0
    )

    repeated = materialize_draft_bundle_set(
        manifest,
        repository_paths={"AgenticRobot": anchor},
        slot_root=(tmp_path / "slots").resolve(),
        slot_id="gpu-slot-01",
    )
    assert repeated == receipt
    assert (
        project_main(
            [
                "draft-bundle",
                "materialize",
                str(manifest),
                "--repository",
                f"AgenticRobot={anchor}",
                "--slot-root",
                str((tmp_path / "slots").resolve()),
                "--slot",
                "gpu-slot-01",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == receipt

    (checkout / "README.md").write_text("drifted\n", encoding="utf-8")
    with pytest.raises(LabError, match="materialized slot repository drifted"):
        materialize_draft_bundle_set(
            manifest,
            repository_paths={"AgenticRobot": anchor},
            slot_root=(tmp_path / "slots").resolve(),
            slot_id="gpu-slot-01",
        )


def test_materializer_preserves_exact_submodule_identity_without_lfs_fetch(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    dependency = _repository(tmp_path / "dependency")
    management = draft_environment["AgenticRobot"]
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(management),
            "submodule",
            "add",
            str(dependency),
            "third_party/dependency",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(management, "commit", "-am", "add test submodule")
    task = _create_task("AR-DRAFT-09", capsys)
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(task),
            "submodule",
            "update",
            "--init",
            "--checkout",
            "--no-fetch",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (task / "README.md").write_text("draft with submodule\n", encoding="utf-8")
    assert main(["snapshot", "create", "AR-DRAFT-09", "--json"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    expected_submodules = snapshot["repositories"]["AgenticRobot"]["submodules"]
    assert expected_submodules == {
        "third_party/dependency": _git(dependency, "rev-parse", "HEAD")
    }
    export_root = tmp_path / "exports"
    assert (
        main(
            [
                "snapshot",
                "export",
                "AR-DRAFT-09",
                snapshot["snapshot_id"],
                "--output",
                str(export_root),
                "--json",
            ]
        )
        == 0
    )
    exported = json.loads(capsys.readouterr().out)
    assert (
        exported["bundle"]["repositories"]["AgenticRobot"]["submodules"]
        == expected_submodules
    )
    anchor = tmp_path / "submodule-anchor"
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "clone",
            "--no-local",
            "--recurse-submodules",
            str(task),
            str(anchor),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    receipt = materialize_draft_bundle_set(
        Path(exported["path"]) / "bundle.json",
        repository_paths={"AgenticRobot": anchor},
        slot_root=(tmp_path / "slots-submodule").resolve(),
        slot_id="gpu-slot-01",
    )
    checkout = Path(receipt["repositories"]["AgenticRobot"]["path"])

    assert receipt["repositories"]["AgenticRobot"]["submodules"] == (
        expected_submodules
    )
    assert _git(checkout, "submodule", "status", "--recursive").startswith(" ")
    assert (
        _git(checkout / "third_party/dependency", "rev-parse", "HEAD")
        == expected_submodules["third_party/dependency"]
    )
    assert (
        _git(
            checkout / "third_party/dependency",
            "config",
            "--get",
            "lfs.fetchexclude",
        )
        == "*"
    )


def test_expired_draft_prune_is_dry_run_by_default_and_apply_is_exact(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    task = _create_task("AR-DRAFT-10", capsys)
    (task / "README.md").write_text("expiring draft\n", encoding="utf-8")
    before = {
        "branch": _git(task, "branch", "--show-current"),
        "head": _git(task, "rev-parse", "HEAD"),
        "status": _git(task, "status", "--short"),
    }
    assert main(["snapshot", "create", "AR-DRAFT-10", "--json"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    snapshot_id = snapshot["snapshot_id"]
    reference = snapshot["repositories"]["AgenticRobot"]["ref"]
    assert main(["snapshot", "export", "AR-DRAFT-10", snapshot_id]) == 0
    capsys.readouterr()
    state_root = task.parents[2] / "State"
    snapshot_root = state_root / "drafts" / "AR-DRAFT-10" / snapshot_id
    export_root = state_root / "draft-exports" / snapshot_id
    manifest_path = snapshot_root / "snapshot.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expires_at"] = "2000-01-01T00:00:00+00:00"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert main(["snapshot", "prune", "--task", "AR-DRAFT-10", "--json"]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["applied"] is False
    assert planned["expired_count"] == 1
    assert snapshot_root.is_dir()
    assert export_root.is_dir()
    assert (
        _git(task, "rev-parse", reference)
        == snapshot["repositories"]["AgenticRobot"]["commit"]
    )

    assert (
        main(
            [
                "snapshot",
                "prune",
                "--task",
                "AR-DRAFT-10",
                "--apply",
                "--json",
            ]
        )
        == 0
    )
    applied = json.loads(capsys.readouterr().out)
    assert applied["applied"] is True
    assert applied["retired"] == [snapshot_id]
    assert not snapshot_root.exists()
    assert not export_root.exists()
    assert (
        subprocess.run(
            ["git", "-C", str(task), "show-ref", "--verify", "--quiet", reference],
            check=False,
            capture_output=True,
        ).returncode
        == 1
    )
    assert _git(task, "branch", "--show-current") == before["branch"]
    assert _git(task, "rev-parse", "HEAD") == before["head"]
    assert _git(task, "status", "--short") == before["status"]


def test_draft_prune_refuses_ref_identity_drift(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    task = _create_task("AR-DRAFT-11", capsys)
    (task / "README.md").write_text("drifted ref\n", encoding="utf-8")
    assert main(["snapshot", "create", "AR-DRAFT-11", "--json"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    repository = snapshot["repositories"]["AgenticRobot"]
    _git(task, "update-ref", repository["ref"], repository["base_commit"])

    assert main(["snapshot", "prune", "--task", "AR-DRAFT-11"]) == 2
    assert "draft ref/commit mismatch" in capsys.readouterr().err


def test_draft_prune_rolls_back_refs_and_state_on_multi_repo_failure(
    draft_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        main(
            [
                "task",
                "create",
                "AR-DRAFT-12",
                "--repos",
                "AgenticRobot,RPent",
                "--json",
            ]
        )
        == 0
    )
    task_payload = json.loads(capsys.readouterr().out)
    task_root = Path(task_payload["workspace_root"])
    for name in ("AgenticRobot", "RPent"):
        _claim_writer("AR-DRAFT-12", name, f"agent-{name.lower()}-12", capsys)
        (task_root / name / "README.md").write_text(
            f"{name} expiring\n", encoding="utf-8"
        )
    assert main(["snapshot", "create", "AR-DRAFT-12", "--json"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    snapshot_root = (
        task_root.parents[1]
        / "State"
        / "drafts"
        / "AR-DRAFT-12"
        / snapshot["snapshot_id"]
    )
    manifest_path = snapshot_root / "snapshot.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expires_at"] = "2000-01-01T00:00:00+00:00"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    original_run_git = draft_cleanup.git.run_git
    rpent_path = (task_root / "RPent").resolve()

    def fail_second_delete(repository, arguments, *, check=True):
        arguments = list(arguments)
        if repository.resolve() == rpent_path and arguments[:2] == ["update-ref", "-d"]:
            raise draft_cleanup.git.GitError("injected second ref deletion failure")
        return original_run_git(repository, arguments, check=check)

    monkeypatch.setattr(draft_cleanup.git, "run_git", fail_second_delete)
    assert (
        main(
            [
                "snapshot",
                "prune",
                "--task",
                "AR-DRAFT-12",
                "--apply",
            ]
        )
        == 2
    )
    assert "injected second ref deletion failure" in capsys.readouterr().err
    assert manifest_path.is_file()
    for name in ("AgenticRobot", "RPent"):
        repository = snapshot["repositories"][name]
        assert (
            _git(task_root / name, "rev-parse", repository["ref"])
            == repository["commit"]
        )
