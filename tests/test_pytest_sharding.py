from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SHARDING = Path(__file__).resolve().parents[1] / "scripts/pytest_sharding.py"


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
    planned = [
        module for shard in first_manifest["shards"] for module in shard["test_modules"]
    ]
    assert len(planned) == len(set(planned)) == len(sizes)
    assert set(planned) == {f"tests/{name}" for name in sizes}
    loads = [shard["estimated_seconds"] for shard in first_manifest["shards"]]
    assert max(loads) - min(loads) <= 0.3


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
