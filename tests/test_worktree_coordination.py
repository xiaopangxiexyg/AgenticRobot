from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def coordination(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "coordination_test", SCRIPTS / "worktree_coordination.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True, capture_output=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.delenv("DEXTELEOP_ASSET_ROOT", raising=False)
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "user.email", "test@example.invalid")
    (path / "tracked.txt").write_text("base\n")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-m", "base")
    return path


def _claim(repo, target, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "claim_worktree.py"),
            "--repo-root",
            str(repo),
            "--path",
            str(target),
            "--branch",
            "codex/test",
            "--base",
            "main",
            "--owner",
            "test",
            "--write-scope",
            "tracked.txt",
            "--write-scope",
            ".github/**",
            "--semantic-scope",
            "test workflow",
            "--validation",
            "offline pytest",
            "--json",
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


@pytest.mark.integration
def test_claim_without_assets_round_trips_through_inventory(repo, tmp_path):
    target = tmp_path / "worktree"
    result = _claim(repo, target)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["asset_root"] is None
    assert payload["asset_warning"]
    assert not (target / "assets").exists()
    (target / "tracked.txt").write_text("changed\n")
    workflow = target / ".github" / "workflows" / "test.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: fixture\n")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "worktree_inventory.py"),
            str(repo),
            "--branch",
            "codex/test",
            "--json",
            "--strict",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["selection"] == ["codex/test"]
    (record,) = report["worktrees"]
    assert record["owner"] == "test"
    assert record["base"] == _git(repo, "rev-parse", "main")
    assert record["changed_paths"] == [".github/workflows/test.yml", "tracked.txt"]
    assert record["out_of_scope_paths"] == []


@pytest.mark.integration
@pytest.mark.parametrize("extra", [("--asset-root", "missing"), ("--require-assets",)])
def test_required_assets_fail_before_creating_a_branch(repo, tmp_path, extra):
    target = tmp_path / "worktree"
    result = _claim(repo, target, *extra)
    assert result.returncode == 2
    assert not target.exists()
    assert _git(repo, "branch", "--list", "codex/test") == ""


@pytest.mark.integration
def test_claim_preserves_existing_directory(repo, tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    (target / "peer.txt").write_text("peer work\n")
    result = _claim(repo, target)
    assert result.returncode == 2
    assert (target / "peer.txt").read_text() == "peer work\n"
    assert _git(repo, "branch", "--list", "codex/test") == ""


def test_legacy_record_is_readable_but_not_pinned(coordination):
    fields, errors = coordination.parse_description(
        "owner=test; state=active; base=abc; write_scope=scripts/**; "
        "semantic_scope=workflow; excludes=none; dependencies=none; validation=pytest"
    )
    assert errors == []
    assert fields["write-scope"] == ["scripts/**"]
    assert fields["depends-on"] == ["none"]
    assert fields["tip"] == ["pending"]


@pytest.mark.parametrize(
    "record, expected",
    [
        ("owner: first\nowner: second", "must appear once"),
        ("state: ready", "unknown state"),
        ("unknown: x", "unknown field"),
        ("owner:", "empty key or value"),
        ("tip: abc\ncandidate: def", "disagree"),
        ("owner: test", "missing description fields"),
    ],
)
def test_description_faults_remain_visible(coordination, record, expected):
    _, errors = coordination.parse_description(record)
    assert any(expected in error for error in errors)


@pytest.mark.integration
def test_ready_record_requires_clean_exact_tip(repo, tmp_path, coordination):
    target = tmp_path / "worktree"
    assert _claim(repo, target).returncode == 0
    key = "branch.codex/test.description"
    fields, errors = coordination.parse_description(_git(repo, "config", "--get", key))
    assert errors == []
    fields["state"] = ["ready-for-integration"]
    fields["tip"] = [_git(target, "rev-parse", "HEAD")]
    _git(repo, "config", key, coordination.serialize_description(fields))
    (target / "tracked.txt").write_text("not ready\n")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "worktree_inventory.py"),
            str(repo),
            "--branch",
            "codex/test",
            "--json",
            "--strict",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "immutable-state branch has working-tree changes" in result.stdout


@pytest.mark.integration
def test_unknown_selected_branch_is_an_error(repo):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "worktree_inventory.py"),
            str(repo),
            "--branch",
            "codex/typo",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "no worktree" in result.stderr


@pytest.mark.integration
def test_installed_skill_cli_delegates_to_canonical_inventory(repo):
    shim = (
        SCRIPTS.parent
        / ".agents/skills/integrate-shared-worktrees/scripts/worktree_inventory.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(shim),
            str(repo),
            "--branch",
            "main",
            "--json",
            "--strict",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["worktrees"][0]["state"] == "mainline"


@pytest.mark.integration
def test_changed_ready_tip_and_peer_overlap_are_reported(repo, tmp_path, coordination):
    target = tmp_path / "worktree"
    assert _claim(repo, target).returncode == 0
    key = "branch.codex/test.description"
    fields, _ = coordination.parse_description(_git(repo, "config", "--get", key))
    base = _git(target, "rev-parse", "HEAD")
    fields["state"] = ["ready-for-integration"]
    fields["tip"] = [base]
    _git(repo, "config", key, coordination.serialize_description(fields))
    (target / "tracked.txt").write_text("candidate changed\n")
    _git(target, "commit", "-am", "candidate change")
    peer = tmp_path / "peer"
    _git(repo, "worktree", "add", "-b", "codex/peer", str(peer), base)
    fields["owner"] = ["peer"]
    fields["state"] = ["active"]
    fields["tip"] = ["pending"]
    _git(
        repo,
        "config",
        "branch.codex/peer.description",
        coordination.serialize_description(fields),
    )
    (peer / "tracked.txt").write_text("peer changed\n")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "worktree_inventory.py"),
            str(repo),
            "--branch",
            "codex/test",
            "--branch",
            "codex/peer",
            "--json",
            "--strict",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    candidate = next(
        item for item in report["worktrees"] if item["branch"] == "codex/test"
    )
    assert "pinned tip no longer matches worktree HEAD" in candidate["warnings"]
    assert set(report["overlaps"]["changed_paths"]["tracked.txt"]) == {
        "codex/test",
        "codex/peer",
    }
    assert set(report["overlaps"]["semantic_scope"]["test workflow"]) == {
        "codex/test",
        "codex/peer",
    }
    assert (peer / "tracked.txt").read_text() == "peer changed\n"
