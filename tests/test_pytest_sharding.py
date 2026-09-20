from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SHARDING = Path(__file__).resolve().parents[1] / "scripts/pytest_sharding.py"


def _load_sharding_module():
    spec = importlib.util.spec_from_file_location("pytest_sharding", SHARDING)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SHARDING), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_plan_is_deterministic_complete_and_weighted(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    tests_root = repository / "tests"
    tests_root.mkdir(parents=True)
    sizes = {
        "test_alpha.py": 120_000,
        "test_beta.py": 70_000,
        "test_gamma.py": 20_000,
        "test_delta.py": 10_000,
        "test_epsilon.py": 5_000,
    }
    for name, size in sizes.items():
        (tests_root / name).write_text("#" * size, encoding="utf-8")

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    common = (
        "plan",
        "--repository",
        str(repository),
        "--tests-root",
        str(tests_root),
        "--shards",
        "2",
        "--profile",
        "fast",
    )
    first_result = _run(*common, "--output-dir", str(first))
    second_result = _run(*common, "--output-dir", str(second))

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    first_manifest = json.loads((first / "shard-plan.json").read_text())
    second_manifest = json.loads((second / "shard-plan.json").read_text())
    assert first_manifest == second_manifest
    assert first_manifest["schema"] == "dexteleop.pytest-shard-plan.v2"
    assert first_manifest["file_count"] == len(sizes)
    assert first_manifest["target_count"] == len(sizes)
    planned = [
        module for shard in first_manifest["shards"] for module in shard["test_modules"]
    ]
    assert len(planned) == len(set(planned)) == len(sizes)
    assert set(planned) == {f"tests/{name}" for name in sizes}
    loads = [shard["estimated_seconds"] for shard in first_manifest["shards"]]
    assert max(loads) - min(loads) <= 0.3


def test_full_profile_tracks_recent_dominant_modules() -> None:
    module = _load_sharding_module()

    assert module.FULL_WEIGHTS["tests/test_mujoco_temp_pipeline.py"] == pytest.approx(
        134.914
    )
    assert module.FULL_WEIGHTS[
        "tests/test_replay_position_admittance_active_pipeline.py"
    ] == pytest.approx(108.657)
    assert module.FULL_WEIGHTS["tests/test_mujoco_fast_runtime.py"] == pytest.approx(
        45.701
    )
    assert module.FULL_WEIGHTS["tests/test_rm65_grasp_trajectory.py"] == pytest.approx(
        91.046
    )
    assert module.FULL_WEIGHTS[
        "tests/test_mujoco_formal_compound_authority.py"
    ] == pytest.approx(6.681)
    assert (
        module.FULL_WEIGHTS["tests/test_mujoco_temp_pipeline.py"]
        > module.FULL_WEIGHTS[
            "tests/test_replay_position_admittance_active_pipeline.py"
        ]
    )
    assert module.FULL_WEIGHTS[
        "tests/test_sim_point_grasp_experiment.py"
    ] == pytest.approx(16.729)
    assert module.FULL_WEIGHTS["tests/test_mujoco_cube_lift.py"] == pytest.approx(
        11.965
    )
    assert module.FULL_WEIGHTS["tests/test_rebot_b601_real2sim.py"] == pytest.approx(
        23.742
    )
    assert module.FULL_WEIGHTS["tests/test_runtime_artifact_migration.py"] > 26.0
    assert module.FULL_WEIGHTS["tests/test_modality_storage.py"] > 7.0


def test_fast_integration_profile_tracks_its_unsharded_critical_path() -> None:
    module = _load_sharding_module()

    assert module.FAST_INTEGRATION_WEIGHTS[
        "tests/test_rm65_grasp_trajectory.py"
    ] == pytest.approx(74.288)
    assert module.FAST_INTEGRATION_WEIGHTS[
        "tests/test_runtime_artifact_migration.py"
    ] == pytest.approx(25.525)
    assert module.FAST_INTEGRATION_WEIGHTS[
        "tests/test_collection_process_writer.py"
    ] == pytest.approx(14.611)
    assert "fast-integration" in module.WEIGHT_PROFILES


@pytest.mark.parametrize(
    "relative_path",
    (
        "tests/test_mujoco_temp_pipeline.py",
        "tests/test_replay_position_admittance_active_pipeline.py",
        "tests/test_mujoco_fast_runtime.py",
    ),
)
def test_full_plan_splits_registered_large_module_without_losing_tests(
    tmp_path: Path,
    relative_path: str,
) -> None:
    module = _load_sharding_module()
    repository = tmp_path / "checkout"
    tests_root = repository / "tests"
    tests_root.mkdir(parents=True)
    split_path = repository / relative_path
    registered = module.FULL_SPLIT_NODE_WEIGHTS[relative_path]
    test_names = sorted(registered) + ["test_new_profile_is_still_collected"]
    split_path.write_text(
        "\n\n".join(f"def {name}():\n    pass" for name in test_names) + "\n",
        encoding="utf-8",
    )
    (tests_root / "test_small.py").write_text(
        "def test_small():\n    pass\n", encoding="utf-8"
    )

    shards = module.plan_shards(repository, tests_root, 2, "full")
    targets = [target for shard in shards for target, _path, _weight in shard]

    assert len(targets) == len(set(targets)) == len(test_names) + 1
    assert {
        target.split("::", 1)[1]
        for target in targets
        if target.startswith(f"{relative_path}::")
    } == set(test_names)
    assert "tests/test_small.py" in targets
    assert all(
        any(target.startswith(f"{relative_path}::") for target, _, _ in shard)
        for shard in shards
    )
    for affinity_group in module.FULL_SPLIT_NODE_GROUPS.get(relative_path, ()):
        group_targets = {
            f"{relative_path}::{test_name}" for test_name in affinity_group
        }
        group_shards = {
            index
            for index, shard in enumerate(shards)
            if group_targets
            & {target for target, _path, _weight in shard}
        }
        assert len(group_shards) == 1
    assert module.plan_shards(repository, tests_root, 2, "full") == shards
    fast_targets = [
        target
        for shard in module.plan_shards(repository, tests_root, 2, "fast")
        for target, _, _ in shard
    ]
    assert set(fast_targets) == {relative_path, "tests/test_small.py"}


def test_full_plan_rejects_missing_split_affinity_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_sharding_module()
    relative_path = "tests/test_mujoco_fast_runtime.py"
    repository = tmp_path / "checkout"
    split_path = repository / relative_path
    split_path.parent.mkdir(parents=True)
    registered = module.FULL_SPLIT_NODE_WEIGHTS[relative_path]
    split_path.write_text(
        "\n\n".join(f"def {name}():\n    pass" for name in registered) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        module.FULL_SPLIT_NODE_GROUPS,
        relative_path,
        (("test_missing_fixture_consumer",),),
    )

    with pytest.raises(ValueError, match="affinity group names missing tests"):
        module.plan_shards(repository, split_path.parent, 1, "full")


@pytest.mark.parametrize(
    "unsupported_source",
    (
        "class TestGenerated:\n    def test_member(self):\n        pass\n",
        """def pytest_generate_tests(metafunc):
    pass

def test_static():
    pass
""",
    ),
    ids=("test-class", "collection-hook"),
)
def test_full_plan_rejects_unsupported_split_module_collection(
    tmp_path: Path,
    unsupported_source: str,
) -> None:
    module = _load_sharding_module()
    repository = tmp_path / "checkout"
    tests_root = repository / "tests"
    tests_root.mkdir(parents=True)
    (tests_root / "test_mujoco_temp_pipeline.py").write_text(
        unsupported_source, encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unsupported collection"):
        module.plan_shards(repository, tests_root, 1, "full")


def test_socket_preflight_reports_the_blocked_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_sharding_module()

    def blocked_socket(*_args, **_kwargs):
        raise PermissionError(1, "blocked for test")

    monkeypatch.setattr(socket, "socket", blocked_socket)

    with pytest.raises(OSError, match="unavailable for AF_UNIX"):
        module.check_required_local_sockets()


def test_socket_preflight_accepts_both_local_transports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_sharding_module()
    families: list[int] = []

    class FakeConnection:
        def close(self) -> None:
            return None

    class FakeSocket:
        def __init__(self, family: int, _kind: int) -> None:
            self.family = family
            self.address = None
            families.append(family)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def settimeout(self, _timeout: float) -> None:
            return None

        def bind(self, address) -> None:
            self.address = address

        def listen(self, _backlog: int) -> None:
            return None

        def getsockname(self):
            return self.address

        def connect(self, _address) -> None:
            return None

        def accept(self):
            return FakeConnection(), None

    monkeypatch.setattr(socket, "socket", FakeSocket)

    module.check_required_local_sockets()

    assert families == [socket.AF_UNIX, socket.AF_UNIX, socket.AF_INET, socket.AF_INET]


def _write_junit(path: Path, *, name: str, failures: int = 0) -> None:
    path.write_text(
        "<testsuites><testsuite tests='1' errors='0' failures='"
        f"{failures}' skipped='0' time='1.25'><testcase classname='tests.fake' "
        f"name='{name}' time='1.25'/></testsuite></testsuites>",
        encoding="utf-8",
    )


def test_merge_junit_sums_counts_and_rejects_duplicate_cases(tmp_path: Path) -> None:
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    merged = tmp_path / "merged.xml"
    _write_junit(first, name="one")
    _write_junit(second, name="two", failures=1)

    completed = _run("merge-junit", "--output", str(merged), str(first), str(second))

    assert completed.returncode == 0, completed.stderr
    root = ET.parse(merged).getroot()
    assert root.attrib["tests"] == "2"
    assert root.attrib["failures"] == "1"
    assert root.attrib["time"] == "2.500000"

    duplicate = _run(
        "merge-junit",
        "--output",
        str(tmp_path / "duplicate.xml"),
        str(first),
        str(first),
    )
    assert duplicate.returncode == 2
    assert "duplicate testcase" in duplicate.stderr
