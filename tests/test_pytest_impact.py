from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE_ROOT = Path(__file__).resolve().parents[1]
SELECTOR = SOURCE_ROOT / "scripts/pytest_impact.py"


@pytest.fixture
def impact():
    spec = importlib.util.spec_from_file_location("pytest_impact_under_test", SELECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(repository: Path, path: str, text: str) -> None:
    destination = repository / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)


def _checkpoint(repository: Path) -> str:
    _git(repository, "add", "-A")
    _git(repository, "-c", "commit.gpgsign=false", "commit", "-qm", "test checkpoint")
    return _git(repository, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path, impact):
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Impact test")
    _git(repository, "config", "user.email", "impact-test@example.invalid")
    for path in impact.B601_CONTROL_MODULES:
        _write(
            repository,
            path,
            "from robot_motion import shared\ndef test_control(): pass\n",
        )
    _write(repository, "robot_motion/shared.py", "VALUE = 1\n")
    _write(repository, "tests/conftest.py", "# shared fixtures\n")
    _write(
        repository,
        "tests/test_b601_workspace_collision.py",
        "def test_safety(): pass\n",
    )
    _write(repository, "tests/test_mujoco_runtime.py", "def test_native(): pass\n")
    _write(repository, "agent_gateway/feature.py", "VALUE = 1\n")
    _write(repository, ".gitignore", "__pycache__/\n*.pyc\nignored/\n")
    for script in ("pytest_sharding.py", "pytest_impact.py"):
        _write(
            repository,
            "scripts/" + script,
            (SOURCE_ROOT / "scripts" / script).read_text(),
        )
    return repository, _checkpoint(repository)


@pytest.mark.parametrize(
    "path",
    [
        "agent_gateway/feature.py",
        "deployments/perception/provider.py",
        "docs/agent_gateway/design.md",
        "tests/test_agent_gateway_new.py",
        "tests/test_sam3_service.py",
    ],
)
def test_independent_change_omits_only_reviewed_b601_controls(repository, impact, path):
    repo, base = repository
    _write(repo, path, "# independent change\n")
    head = _checkpoint(repo)
    report = impact.selection(repo, base)
    assert report["mode"] == "affected"
    assert report["head_commit"] == head
    assert set(report["excluded_modules"]) == impact.B601_CONTROL_MODULES
    assert "tests/test_b601_workspace_collision.py" in report["selected_modules"]
    assert "tests/test_mujoco_runtime.py" in report["selected_modules"]
    if path.startswith("tests/"):
        assert path in report["selected_modules"]
    assert set(report["selected_modules"]).isdisjoint(report["excluded_modules"])
    assert "robot_motion/shared.py" in report["protected_dependencies"]


@pytest.mark.parametrize(
    "path",
    [
        "robot_motion/shared.py",
        "hardware/driver.py",
        "control_station/runtime.py",
        "sim/new.py",
        "scripts/run_pytest.sh",
        "tests/conftest.py",
        "tests/test_new.py",
        "pyproject.toml",
        "uv.lock",
        "assets/model.xml",
        "agent_gateway/__init__.py",
        "agent_gateway/native.so",
        "unknown.json",
        "tests/test_agent_gateway_unreviewed/test_extra.py",
    ],
)
def test_shared_or_unknown_changes_restore_every_test(repository, impact, path):
    repo, base = repository
    _write(repo, path, "# shared or unknown\n")
    _checkpoint(repo)
    report = impact.selection(repo, base)
    assert report["mode"] == "full"
    assert not report["excluded_modules"]
    assert impact.B601_CONTROL_MODULES <= set(report["selected_modules"])
    assert any(path in reason for reason in report["fallback_reasons"])


@pytest.mark.parametrize(
    "change", ["delete", "rename", "symlink", "executable", "binary", "b601"]
)
def test_unexplained_file_or_model_changes_restore_full(repository, impact, change):
    repo, base = repository
    path = repo / "agent_gateway/feature.py"
    if change in {"delete", "symlink"}:
        path.unlink()
        if change == "symlink":
            path.symlink_to("../robot_motion/shared.py")
    elif change == "rename":
        path.rename(path.with_name("renamed.py"))
    elif change == "executable":
        path.chmod(0o755)
    elif change == "binary":
        path.write_bytes(b"\xff\xfe")
    else:
        path.write_text("# change affects B601\n")
    _checkpoint(repo)
    assert impact.selection(repo, base)["mode"] == "full"


@pytest.mark.parametrize(
    "reference",
    [
        "import agent_gateway.feature\n",
        "from agent_gateway import feature\n",
        "from importlib import import_module\nx = import_module('agent_gateway.feature')\n",
        "RESOURCE = 'agent_gateway/feature.py'\n",
    ],
)
def test_import_and_resource_consumers_prevent_omission(repository, impact, reference):
    repo, _ = repository
    _write(repo, "robot_motion/shared.py", reference)
    base = _checkpoint(repo)
    _write(repo, "agent_gateway/feature.py", "VALUE = 2\n")
    _checkpoint(repo)
    report = impact.selection(repo, base)
    assert report["mode"] == "full"
    assert any("dependency" in reason for reason in report["fallback_reasons"])


def test_newly_resolved_import_is_protected_in_the_head_tree(repository, impact):
    repo, _ = repository
    _write(repo, "robot_motion/shared.py", "import agent_gateway.new_module\n")
    base = _checkpoint(repo)
    _write(repo, "agent_gateway/new_module.py", "VALUE = 1\n")
    _checkpoint(repo)
    report = impact.selection(repo, base)
    assert report["mode"] == "full"
    assert "agent_gateway/new_module.py" in report["protected_dependencies"]


def test_protected_import_through_a_symlink_is_not_silently_lost(repository, impact):
    repo, _ = repository
    _write(repo, "robot_motion/shared.py", "import fixture_link\n")
    (repo / "tests/fixture_link.py").symlink_to("../agent_gateway/feature.py")
    base = _checkpoint(repo)
    _write(repo, "agent_gateway/feature.py", "VALUE = 2\n")
    _checkpoint(repo)
    report = impact.selection(repo, base)
    assert report["mode"] == "full"
    assert any(
        "non-regular dependency" in reason for reason in report["fallback_reasons"]
    )


def test_colliding_test_root_helper_keeps_every_import_candidate(repository, impact):
    repo, _ = repository
    _write(repo, "robot_motion/shared.py", "import zhelper\n")
    _write(repo, "zhelper.py", "VALUE = 1\n")
    _write(repo, "tests/zhelper.py", "import agent_gateway.feature\n")
    base = _checkpoint(repo)
    _write(repo, "agent_gateway/feature.py", "VALUE = 2\n")
    _checkpoint(repo)
    report = impact.selection(repo, base)
    assert report["mode"] == "full"
    assert {"zhelper.py", "tests/zhelper.py"} <= set(report["protected_dependencies"])


def test_shared_pytest_plugin_dependency_forces_full_without_an_import_edge(
    repository, impact
):
    repo, _ = repository
    _write(
        repo, "tests/conftest.py", "pytest_plugins = ['agent_gateway.fixture_plugin']\n"
    )
    _write(repo, "agent_gateway/fixture_plugin.py", "settings = 1\n")
    base = _checkpoint(repo)
    _write(repo, "agent_gateway/fixture_plugin.py", "settings = 2\n")
    _checkpoint(repo)
    report = impact.selection(repo, base)
    assert report["mode"] == "full"
    assert any("pytest_plugins" in reason for reason in report["fallback_reasons"])


def test_new_test_cannot_introduce_a_global_plugin_and_still_omit_controls(
    repository, impact
):
    repo, base = repository
    _write(
        repo,
        "tests/test_agent_gateway_plugin.py",
        "pytest_plugins = ['shared_fixture']\n",
    )
    _checkpoint(repo)
    report = impact.selection(repo, base)
    assert report["mode"] == "full"
    assert any("pytest_plugins" in reason for reason in report["fallback_reasons"])


@pytest.mark.parametrize(
    "source",
    [
        "from importlib import import_module\nx = import_module(variable)\n",
        "from importlib import import_module\nx = import_module('.feature', PACKAGE)\n",
        "from importlib import import_module as load\nloader = load\n",
        "import importlib\n",
        "from importlib.machinery import SourceFileLoader\n",
        "from importlib.metadata import entry_points\n",
        "import runpy\n",
        "exec(variable)\n",
        "def broken syntax\n",
    ],
)
def test_unknown_loading_or_syntax_fails_closed(repository, impact, source):
    repo, _ = repository
    _write(repo, "robot_motion/shared.py", source)
    base = _checkpoint(repo)
    _write(repo, "agent_gateway/feature.py", "VALUE = 2\n")
    _checkpoint(repo)
    report = impact.selection(repo, base)
    assert report["mode"] == "full"
    assert any(
        "cannot establish isolation" in reason for reason in report["fallback_reasons"]
    )


@pytest.mark.parametrize(
    "resource",
    [
        "ROOT / 'deployments' / 'perception' / 'settings.json'",
        "'deployments.perception.settings'",
    ],
)
def test_segmented_or_dotted_resource_owners_are_protected(
    repository, impact, resource
):
    repo, _ = repository
    _write(repo, "robot_motion/shared.py", "RESOURCE = " + resource + "\n")
    _write(repo, "deployments/perception/settings.json", "{}\n")
    base = _checkpoint(repo)
    _write(repo, "deployments/perception/settings.json", '{"updated": true}\n')
    _checkpoint(repo)
    report = impact.selection(repo, base)
    assert report["mode"] == "full"
    assert "deployments/perception/settings.json" in report["protected_dependencies"]


def test_relative_builtin_import_with_explicit_level_is_not_guessed(repository, impact):
    repo, _ = repository
    _write(repo, "robot_motion/shared.py", "import agent_gateway.helper\n")
    _write(
        repo,
        "agent_gateway/helper.py",
        "module = __import__('feature', globals(), locals(), ['VALUE'], 1)\n",
    )
    base = _checkpoint(repo)
    _write(repo, "agent_gateway/feature.py", "VALUE = 2\n")
    _checkpoint(repo)
    report = impact.selection(repo, base)
    assert report["mode"] == "full"
    assert any(
        "unaudited dynamic dependency" in reason
        for reason in report["fallback_reasons"]
    )


def test_audited_lazy_exports_are_traversed_and_content_drift_fails_closed(
    repository, impact
):
    repo, _ = repository
    source = (
        "from importlib import import_module\n"
        "_LAZY_EXPORTS = {'VALUE': ('agent_gateway.feature', 'VALUE')}\n"
        "def __getattr__(name): return getattr(import_module(_LAZY_EXPORTS[name][0]), name)\n"
    )
    path = "robot_motion/__init__.py"
    _write(repo, path, source)
    impact.DYNAMIC_IMPORT_AUDITS[path] = hashlib.sha256(source.encode()).hexdigest()
    base = _checkpoint(repo)
    _write(repo, "agent_gateway/feature.py", "VALUE = 2\n")
    _checkpoint(repo)
    report = impact.selection(repo, base)
    assert report["mode"] == "full"
    assert "agent_gateway/feature.py" in report["protected_dependencies"]
    _write(repo, path, source + "# unreviewed drift\n")
    base = _checkpoint(repo)
    _write(repo, "agent_gateway/feature.py", "VALUE = 3\n")
    _checkpoint(repo)
    assert any(
        "cannot establish isolation" in reason
        for reason in impact.selection(repo, base)["fallback_reasons"]
    )


def test_no_change_never_reuses_an_unproven_result(repository, impact):
    repo, base = repository
    report = impact.selection(repo, base)
    assert report["mode"] == "full"
    assert report["fallback_reasons"]


def test_invalid_or_nonancestor_base_is_rejected(repository, impact):
    repo, base = repository
    with pytest.raises(ValueError, match="git rev-parse"):
        impact.selection(repo, "missing-ref")
    _write(repo, "agent_gateway/feature.py", "VALUE = 2\n")
    future = _checkpoint(repo)
    with pytest.raises(ValueError, match="ancestor"):
        impact.selection(repo, future, base)


def _cli(repo, *arguments):
    if arguments[0] == "verify":
        arguments = (*arguments, "--shards", "2")
    return subprocess.run(
        [sys.executable, str(SELECTOR), *arguments, "--repository", str(repo)],
        capture_output=True,
        text=True,
        check=False,
    )


def _plan(repository, tmp_path):
    repo, base = repository
    _write(repo, "agent_gateway/feature.py", "VALUE = 2\n")
    _checkpoint(repo)
    output = tmp_path / "output"
    output.mkdir()
    result = _cli(
        repo, "plan", "--base", base, "--output-dir", str(output), "--shards", "2"
    )
    assert result.returncode == 0, result.stderr
    return repo, output, output / "selection.json"


def test_plan_preserves_every_selected_module_and_binds_its_targets(
    repository, tmp_path
):
    repo, output, selection_path = _plan(repository, tmp_path)
    report = json.loads(selection_path.read_text())
    plan = json.loads((output / "shard-plan.json").read_text())
    targets = [target for shard in plan["shards"] for target in shard["test_targets"]]
    assert set(targets) == set(report["selected_modules"])
    assert len(targets) == len(set(targets))
    assert _cli(repo, "verify", "--selection", str(selection_path)).returncode == 0
    (output / "shard-0-paths.txt").write_text("tests/other.py\n")
    result = _cli(repo, "verify", "--selection", str(selection_path))
    assert result.returncode != 0 and "targets changed" in result.stderr


def test_self_consistent_but_incomplete_plan_cannot_pass_coverage_verification(
    repository, tmp_path
):
    repo, output, selection_path = _plan(repository, tmp_path)
    report = json.loads(selection_path.read_text())
    plan_path = output / "shard-plan.json"
    plan = json.loads(plan_path.read_text())
    shard = plan["shards"][0]
    removed = shard["test_targets"].pop()
    shard["test_modules"].remove(removed)
    shard["file_count"] -= 1
    shard["target_count"] -= 1
    plan["file_count"] -= 1
    plan["target_count"] -= 1
    paths = output / shard["paths_file"]
    paths.write_text(
        "".join(str(repo / target) + "\n" for target in shard["test_targets"])
    )
    plan_path.write_text(json.dumps(plan))
    report["shard_plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    report["paths_sha256"][paths.name] = hashlib.sha256(paths.read_bytes()).hexdigest()
    selection_path.write_text(json.dumps(report))
    result = _cli(repo, "verify", "--selection", str(selection_path))
    assert result.returncode != 0
    assert "coverage differs" in result.stderr


@pytest.mark.parametrize(
    "mutation", ["tracked", "untracked", "head", "selection", "plan", "ignored-test"]
)
def test_evidence_rejects_source_discovery_and_manifest_drift(
    repository, tmp_path, mutation
):
    repo, output, selection_path = _plan(repository, tmp_path)
    if mutation in {"tracked", "head"}:
        _write(repo, "agent_gateway/feature.py", "VALUE = 3\n")
        if mutation == "head":
            _checkpoint(repo)
    elif mutation == "untracked":
        _write(repo, "unexpected.txt", "changed\n")
    elif mutation == "ignored-test":
        _write(repo, "tests/ignored/test_hidden.py", "def test_hidden(): pass\n")
    elif mutation == "selection":
        report = json.loads(selection_path.read_text())
        report["excluded_modules"] = []
        selection_path.write_text(json.dumps(report))
    else:
        (output / "shard-plan.json").write_text("{}\n")
    result = _cli(repo, "verify", "--selection", str(selection_path))
    assert result.returncode != 0, result.stdout


def test_plan_refuses_dirty_source_before_generating_evidence(repository, tmp_path):
    repo, base = repository
    _write(repo, "agent_gateway/feature.py", "VALUE = 2\n")
    output = tmp_path / "output"
    output.mkdir()
    result = _cli(
        repo, "plan", "--base", base, "--output-dir", str(output), "--shards", "2"
    )
    assert result.returncode != 0
    assert not (output / "selection.json").exists()
    assert not (output / "shard-plan.json").exists()
