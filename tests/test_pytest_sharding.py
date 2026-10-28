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


def test_selected_module_plan_preserves_all_requested_targets(tmp_path: Path) -> None:
    module = _load_sharding_module()
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    for name in ("first", "second", "omitted"):
        (tests_root / f"test_{name}.py").write_text("def test_case(): pass\n")
    selected = ["tests/test_first.py", "tests/test_second.py"]
    shards = module.plan_shards(tmp_path, tests_root, 2, "full", test_modules=selected)
    assert {target for shard in shards for target, _, _ in shard} == set(selected)
    complete = module.plan_shards(tmp_path, tests_root, 2, "full")
    assert shards == [
        [item for item in shard if item[0] in selected] for shard in complete
    ]
    for invalid in ([selected[0], selected[0]], ["tests/missing.py"], []):
        with pytest.raises(ValueError):
            module.plan_shards(tmp_path, tests_root, 1, "full", test_modules=invalid)


def test_selected_split_families_keep_their_complete_plan_batches(tmp_path):
    module = _load_sharding_module()
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    relative = "tests/test_mujoco_temp_pipeline.py"
    functions = sorted(
        set(module.FULL_SPLIT_NODE_WEIGHTS[relative]) | {"test_new_family"}
    )
    (tmp_path / relative).write_text(
        "\n\n".join(f"def {name}():\n    pass" for name in functions) + "\n"
    )
    (tests_root / "test_omitted.py").write_text("def test_other(): pass\n")
    full = module.plan_shards(tmp_path, tests_root, 2, "full")
    selected = module.plan_shards(
        tmp_path, tests_root, 2, "full", test_modules=[relative]
    )
    assert selected == [
        [item for item in shard if item[0].startswith(relative + "::")]
        for shard in full
    ]
    assert {item[0].split("::")[1] for shard in selected for item in shard} == set(
        functions
    )


def test_full_profile_tracks_recent_dominant_modules() -> None:
    module = _load_sharding_module()

    assert module.FULL_WEIGHTS["tests/test_mujoco_temp_pipeline.py"] == pytest.approx(
        118.275
    )
    assert module.FULL_WEIGHTS[
        "tests/test_replay_position_admittance_active_pipeline.py"
    ] == pytest.approx(101.928)
    assert module.FULL_WEIGHTS["tests/test_mujoco_fast_runtime.py"] == pytest.approx(
        27.069
    )
    assert module.FULL_WEIGHTS["tests/test_rm65_grasp_trajectory.py"] == pytest.approx(
        85.626
    )
    assert module.FULL_WEIGHTS[
        "tests/test_mujoco_formal_compound_authority.py"
    ] == pytest.approx(4.29)
    assert (
        module.FULL_WEIGHTS["tests/test_mujoco_temp_pipeline.py"]
        > module.FULL_WEIGHTS[
            "tests/test_replay_position_admittance_active_pipeline.py"
        ]
    )
    assert module.FULL_WEIGHTS[
        "tests/test_sim_point_grasp_experiment.py"
    ] == pytest.approx(14.281)
    assert module.FULL_WEIGHTS["tests/test_mujoco_cube_lift.py"] == pytest.approx(
        8.851
    )
    assert module.FULL_WEIGHTS["tests/test_rebot_b601_real2sim.py"] == pytest.approx(
        21.849
    )
    assert module.FULL_WEIGHTS["tests/test_runtime_artifact_migration.py"] == pytest.approx(
        7.655
    )
    assert module.FULL_WEIGHTS["tests/test_runtime_artifact_resolution.py"] == pytest.approx(
        1.579
    )
    assert module.FULL_WEIGHTS["tests/test_modality_storage.py"] == pytest.approx(5.531)


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
        "tests/test_agent_gateway_mujoco_fast_grasp.py",
        "tests/test_agent_gateway_mujoco_safe_tcp_pose.py",
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
    test_names = sorted(set(registered) | {
        "test_new_profile_is_still_collected", "test_unmeasured_peer_is_collected",
    })
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
    assert module.plan_shards(repository, tests_root, 2, "full") == shards
    fast_targets = [
        target
        for shard in module.plan_shards(repository, tests_root, 2, "fast")
        for target, _, _ in shard
    ]
    assert set(fast_targets) == {relative_path, "tests/test_small.py"}


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
