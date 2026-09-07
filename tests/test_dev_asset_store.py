from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_TOOL_PATH = REPO_ROOT / "scripts" / "dev_asset_store.py"
CLAIM_TOOL_PATH = REPO_ROOT / "scripts" / "claim_worktree.py"


def _load_asset_tool():
    spec = importlib.util.spec_from_file_location(
        "dev_asset_store_test_module", ASSET_TOOL_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_required_store(root: Path, asset_tool) -> None:
    for requirement in asset_tool.load_requirements():
        if not requirement.required_by_default:
            continue
        for relative in requirement.required_paths:
            path = root / requirement.relative_path / relative
            if Path(relative).suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            else:
                path.mkdir(parents=True, exist_ok=True)


def _run_git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_repository(repo: Path) -> str:
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "asset-test@example.invalid")
    _run_git(repo, "config", "user.name", "Asset Test")
    (repo / "tracked.txt").write_text("fixture\n", encoding="utf-8")
    _run_git(repo, "add", "tracked.txt")
    _run_git(repo, "commit", "-m", "fixture")
    return _run_git(repo, "rev-parse", "HEAD")


def test_store_status_requires_only_default_bundles(tmp_path: Path) -> None:
    asset_tool = _load_asset_tool()
    store = tmp_path / "store"
    _write_required_store(store, asset_tool)

    status = asset_tool.inspect_asset_store(store, source="test")

    assert status.available is True
    assert all(item.available for item in status.bundles if item.required_by_default)
    assert any(
        not item.available for item in status.bundles if not item.required_by_default
    )
    localization = next(
        item
        for item in status.bundles
        if item.bundle_id == "perception.sam2_grounding_dino"
    )
    assert localization.relative_path == "perception/sam2_grounding_dino"
    assert localization.required_by_default is False
    point_grasp = next(
        item
        for item in status.bundles
        if item.bundle_id == "perception.sam2_vgn_point_grasp"
    )
    assert point_grasp.relative_path == "perception/sam2_vgn_point_grasp"
    assert point_grasp.required_by_default is False


def test_resolution_precedence_is_explicit_environment_git_then_worktree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset_tool = _load_asset_tool()
    repo = tmp_path / "repo"
    repo.mkdir()
    local = repo / "assets"
    local.mkdir()
    configured = tmp_path / "configured"
    environment = tmp_path / "environment"
    explicit = tmp_path / "explicit"
    for path in (configured, environment, explicit):
        path.mkdir()
    monkeypatch.setattr(asset_tool, "_git_config_get", lambda *_: str(configured))

    assert asset_tool.resolve_asset_root(repo, environ={}) == (
        configured.resolve(),
        f"git:{asset_tool.ASSET_ROOT_GIT_CONFIG}",
    )
    assert asset_tool.resolve_asset_root(
        repo,
        environ={asset_tool.ASSET_ROOT_ENV: str(environment)},
    ) == (environment.resolve(), f"env:{asset_tool.ASSET_ROOT_ENV}")
    assert asset_tool.resolve_asset_root(
        repo,
        explicit_root=explicit,
        environ={asset_tool.ASSET_ROOT_ENV: str(environment)},
    ) == (explicit.resolve(), "explicit")


def test_claim_worktree_records_owner_and_links_shared_assets(tmp_path: Path) -> None:
    asset_tool = _load_asset_tool()
    repo = tmp_path / "repo"
    base = _init_repository(repo)
    store = tmp_path / "asset-store"
    _write_required_store(store, asset_tool)
    worktree = tmp_path / "claimed-worktree"
    branch = "task/test/shared-assets-20260818"

    completed = subprocess.run(
        [
            sys.executable,
            str(CLAIM_TOOL_PATH),
            "--repo-root",
            str(repo),
            "--branch",
            branch,
            "--path",
            str(worktree),
            "--base",
            base,
            "--owner",
            "test",
            "--write-scope",
            "scripts/**",
            "--semantic-scope",
            "development paths only",
            "--validation",
            "focused offline tests",
            "--asset-root",
            str(store),
            "--json",
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert (worktree / "assets").is_symlink()
    assert (worktree / "assets").resolve() == store.resolve()
    assert _run_git(repo, "config", "--local", "--get", "dexteleop.assetRoot") == str(
        store.resolve()
    )
    description = _run_git(
        repo,
        "config",
        "--local",
        "--get",
        f"branch.{branch}.description",
    )
    spec = importlib.util.spec_from_file_location(
        "worktree_coordination", REPO_ROOT / "scripts" / "worktree_coordination.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fields, errors = module.parse_description(description)
    assert errors == []
    assert fields["base"] == [base]
    assert fields["owner"] == ["test"]
    assert fields["state"] == ["active"]
    assert fields["tip"] == ["pending"]


def test_link_refuses_to_replace_a_different_existing_asset_path(
    tmp_path: Path,
) -> None:
    asset_tool = _load_asset_tool()
    repo = tmp_path / "repo"
    base = _init_repository(repo)
    worktree = tmp_path / "worktree"
    _run_git(repo, "worktree", "add", "--detach", str(worktree), base)
    existing = worktree / "assets"
    existing.mkdir()
    store = tmp_path / "store"
    _write_required_store(store, asset_tool)

    try:
        asset_tool.ensure_worktree_asset_link(repo, worktree, store)
    except asset_tool.AssetStoreError as exc:
        assert "refusing to replace existing worktree asset path" in str(exc)
    else:
        raise AssertionError("a different existing asset path must not be replaced")
    assert existing.is_dir()
    assert not existing.is_symlink()
