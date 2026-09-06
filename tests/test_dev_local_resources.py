from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "scripts" / "dev_local_resources.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "dev_local_resources_test_module", TOOL_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def test_repository_local_resource_policy_matches_gitignore_and_tracking() -> None:
    tool = _load_tool()

    payload = tool.audit_resources(REPO_ROOT)

    assert payload["ok"] is True
    assert payload["issue_count"] == 0
    assert (
        payload["gitignore_inventory"]["rule_count"]
        == payload["gitignore_inventory"]["registered_rule_count"]
    )
    policies = {item["resource_id"]: item for item in payload["resources"]}
    assert {
        "development-assets",
        "quest-development-certificates",
        "offline-datasets",
        "runtime-and-collection-evidence",
        "models-and-local-vendors",
        "machine-local-configuration",
        "generated-and-rebuildable-state",
        "developer-private-state",
    } <= policies.keys()
    assert policies["development-assets"]["migration"] == "asset-store-link"
    assert policies["quest-development-certificates"]["migration"] == "recreate"

    locations = {
        location["path"]: location
        for policy in payload["resources"]
        for location in policy["locations"]
        if location["path"] is not None
    }
    for mixed_root in (
        ".agents",
        ".agents/skills",
        "teleop_records",
        "model",
        "hardware/VIVE/triad_openvr",
        "hardware/quest_dev/assets/certs",
    ):
        assert locations[mixed_root]["tracked_policy"] == "mixed"
        assert locations[mixed_root]["tracked_count"] > 0

    quest_certs = locations["hardware/quest_dev/assets/certs"]
    assert quest_certs["ignore_source"] == "hardware/quest_dev/.gitignore"
    assert quest_certs["matched_rule"]["source"] == "hardware/quest_dev/.gitignore"

    mujoco_log = locations["MUJOCO_LOG.TXT"]
    assert mujoco_log["tracked_policy"] == "none"
    assert mujoco_log["matched_rule"]["rule"] == "/MUJOCO_LOG.TXT"

    legacy = locations["legacy"]
    assert legacy["tracked_policy"] == "none"
    assert legacy["tracked_count"] == 0
    assert legacy["matched_rule"]["rule"] == "/legacy/"

    assert locations[".agents"]["matched_rule"]["rule"] == ".agents/*"
    assert locations[".agents/skills"]["matched_rule"]["rule"] == ".agents/skills/*"


def test_only_development_assets_are_automatically_linkable(tmp_path: Path) -> None:
    tool = _load_tool()
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    for relative in (
        "assets",
        "datasets",
        "runtime_artifacts",
        "teleop_records",
        "model",
        "hardware/VIVE/triad_openvr",
    ):
        (source / relative).mkdir(parents=True)
    (target / "runtime_artifacts").mkdir()

    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    payload = tool.build_migration_plan(source, target)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    assert payload["read_only"] is True
    assert before == after
    items = {
        item["location"]: item for item in payload["items"] if item["source_present"]
    }
    assert items["assets"]["automatic"] is True
    assert items["assets"]["migration"] == "asset-store-link"
    assert items["datasets"]["automatic"] is False
    assert items["runtime_artifacts"]["migration"] == "verified-copy"
    assert items["runtime_artifacts"]["collision_review_required"] is True
    assert items["teleop_records"]["automatic"] is False
    assert items["teleop_records"]["migration"] == "verified-copy"
    for mixed_root in ("model", "hardware/VIVE/triad_openvr"):
        assert items[mixed_root]["automatic"] is False
        assert items[mixed_root]["migration"] == "manual-review"


def test_audit_reports_gitignore_rule_drift(tmp_path: Path) -> None:
    tool = _load_tool()
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    (repo / ".gitignore").write_text("assets/\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": tool.SCHEMA_VERSION,
                "resources": [
                    {
                        "resource_id": "development-assets",
                        "classification": "shared-read-mostly",
                        "migration": "asset-store-link",
                        "summary": "fixture",
                        "locations": [
                            {
                                "label": "assets",
                                "path": "assets",
                                "probe": "assets/.probe",
                                "expected_rule": "wrong-assets-rule",
                                "tracked_policy": "none",
                            }
                        ],
                    }
                ],
                "ignore_exceptions": [
                    {
                        "label": "tracked fixture",
                        "probe": "tracked.txt",
                        "expected_rule": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = tool.audit_resources(repo, manifest_path=manifest)

    assert payload["ok"] is False
    assert payload["issue_count"] == 3
    assert set(payload["gitignore_inventory"]["issues"]) == {
        "unregistered .gitignore rule: assets/",
        "manifest rule missing from .gitignore: wrong-assets-rule",
    }
    issues = payload["resources"][0]["locations"][0]["issues"]
    assert issues == ["ignore rule drift: expected 'wrong-assets-rule', got 'assets/'"]


def test_cli_emits_machine_readable_audit() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "audit",
            "--repo-root",
            str(REPO_ROOT),
            "--json",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema"] == "dexteleop.development_local_resource_audit.v1"
    assert payload["ok"] is True


def test_perception_sources_are_registered_gitlink_boundaries() -> None:
    gitmodules = (REPO_ROOT / ".gitmodules").read_text(encoding="utf-8")
    expected = {
        "model/perception/sam2": "https://github.com/facebookresearch/sam2.git",
        "model/perception/GroundingDINO": "https://github.com/IDEA-Research/GroundingDINO.git",
        "model/perception/vgn": "https://github.com/ethz-asl/vgn.git",
        "model/perception/graspnet-baseline": (
            "https://github.com/graspnet/graspnet-baseline.git"
        ),
    }
    for path, url in expected.items():
        assert f'[submodule "{path}"]' in gitmodules
        assert f"path = {path}" in gitmodules
        assert f"url = {url}" in gitmodules
        completed = subprocess.run(
            ["git", "ls-files", "--stage", "--", path],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.startswith("160000 "), completed.stdout
