from __future__ import annotations

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

SOURCE_WRAPPER = Path(__file__).resolve().parents[1] / "scripts/run_fast_feedback.sh"
SOURCE_SHARDING = Path(__file__).resolve().parents[1] / "scripts/pytest_sharding.py"
UNIT_MARKER = "unit and not optional and not hardware and not slow_sim"
INTEGRATION_MARKER = (
    "integration and not optional and not hardware and not slow_sim"
)
DEFAULT_UNIT_SHARDS = 2
DEFAULT_INTEGRATION_SHARDS = 2


def _wrapper_checkout(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkout = tmp_path / "checkout"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    tests = checkout / "tests"
    tests.mkdir()
    for name in ("test_alpha.py", "test_beta.py", "test_gamma.py", "test_delta.py"):
        (tests / name).write_text("", encoding="utf-8")
    wrapper = scripts / "run_fast_feedback.sh"
    shutil.copy2(SOURCE_WRAPPER, wrapper)
    shutil.copy2(SOURCE_SHARDING, scripts / "pytest_sharding.py")

    fake_sync_root = tmp_path / "fake-sync"
    fake_sync_root.mkdir()
    fake_runner = scripts / "run_pytest.sh"
    fake_runner.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

marker=""
junit=""
arguments_file=""
all_arguments=("$@")
while (($#)); do
  case "$1" in
    -m)
      marker="$2"
      shift 2
      ;;
    --junitxml=*)
      junit="${1#--junitxml=}"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

case "$marker" in
  unit*) tier="unit"; other="integration" ;;
  integration*) tier="integration"; other="unit" ;;
  *) echo "unexpected marker: $marker" >&2; exit 90 ;;
esac

report_name="${junit##*/}"
report_name="${report_name%.xml}"
if [[ "$report_name" == unit-* || "$report_name" == integration-* ]]; then
  name="$report_name"
else
  name="$tier"
fi
arguments_file="$FAKE_SYNC_ROOT/$name.args"
printf '%s\\n' "${all_arguments[@]}" >"$arguments_file"
printf '%s\\n' "$PYTHONPYCACHEPREFIX" >"$FAKE_SYNC_ROOT/$name.python-cache"
printf '%s\\n' "$SIM_MUJOCO_CACHE_DIR" >"$FAKE_SYNC_ROOT/$name.mujoco-cache"
umask >"$FAKE_SYNC_ROOT/$name.umask"
: >"$FAKE_SYNC_ROOT/$tier.started"
for _ in {1..250}; do
  [[ -e "$FAKE_SYNC_ROOT/$other.started" ]] && break
  sleep 0.02
done
[[ -e "$FAKE_SYNC_ROOT/$other.started" ]] || {
  echo "$other tier did not start concurrently" >&2
  exit 91
}

mkdir -p -- "$(dirname -- "$junit")"
printf '<testsuites name="pytest tests"><testsuite name="%s" tests="1" failures="0"/></testsuites>\\n' "$name" >"$junit"
echo "fake tier: $name"
if [[ "${FAKE_FAIL_TIER:-}" == "$name" ]]; then
  echo "requested failure: $name" >&2
  exit 7
fi
""",
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)
    return checkout, wrapper, fake_sync_root


def _run_wrapper(
    wrapper: Path,
    fake_sync_root: Path,
    *,
    run_root: str,
    asset_root: Path,
    fail_tier: str = "",
    extra_arguments: tuple[str, ...] = (),
    unit_shards: str | None = None,
    integration_shards: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    # A wrapper contract test must not inherit the shard count of an outer
    # fast-feedback run.  Tests that exercise a non-default count opt in via
    # the explicit unit_shards argument below.
    environment.pop("DEXTELEOP_FAST_FEEDBACK_UNIT_SHARDS", None)
    environment.pop("DEXTELEOP_FAST_FEEDBACK_INTEGRATION_SHARDS", None)
    environment.update(
        {
            "DEXTELEOP_ASSET_ROOT": str(asset_root.resolve()),
            "DEXTELEOP_FAST_FEEDBACK_RUN_ROOT": run_root,
            "FAKE_FAIL_TIER": fail_tier,
            "FAKE_SYNC_ROOT": str(fake_sync_root),
        }
    )
    if unit_shards is not None:
        environment["DEXTELEOP_FAST_FEEDBACK_UNIT_SHARDS"] = unit_shards
    if integration_shards is not None:
        environment["DEXTELEOP_FAST_FEEDBACK_INTEGRATION_SHARDS"] = (
            integration_shards
        )
    return subprocess.run(
        [str(wrapper), *extra_arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _recorded_arguments(fake_sync_root: Path, tier: str) -> list[str]:
    return (fake_sync_root / f"{tier}.args").read_text(encoding="utf-8").splitlines()


def test_wrapper_writes_parallel_tier_evidence_to_declared_run_root(
    tmp_path: Path,
) -> None:
    checkout, wrapper, fake_sync_root = _wrapper_checkout(tmp_path)
    asset_root = tmp_path / "logical-assets"
    asset_root.mkdir()
    run_root = tmp_path / "harness-run" / "fast-feedback"
    run_root.parent.mkdir()

    completed = _run_wrapper(
        wrapper,
        fake_sync_root,
        run_root=str(run_root),
        asset_root=asset_root,
        extra_arguments=("--durations=2",),
    )

    assert completed.returncode == 0, completed.stderr
    assert "fast unit (status=0)" in completed.stdout
    assert "fast integration (status=0)" in completed.stdout
    assert f"artifacts: {run_root}" in completed.stdout
    unit_log = (run_root / "unit.log").read_text(encoding="utf-8")
    integration_log = (run_root / "integration.log").read_text(encoding="utf-8")
    unit_shards = DEFAULT_UNIT_SHARDS
    integration_shards = DEFAULT_INTEGRATION_SHARDS
    for index in range(unit_shards):
        assert f"===== unit shard {index} =====" in unit_log
        assert f"fake tier: unit-{index}" in unit_log
    for index in range(integration_shards):
        assert f"===== integration shard {index} =====" in integration_log
        assert f"fake tier: integration-{index}" in integration_log
    assert (run_root / "unit.xml").is_file()
    assert (run_root / "integration.xml").is_file()
    assert (run_root / "shard-plan.json").is_file()
    assert (run_root / "integration-shard-plan.json").is_file()

    unit_arguments = [
        _recorded_arguments(fake_sync_root, f"unit-{index}")
        for index in range(unit_shards)
    ]
    integration_arguments = [
        _recorded_arguments(fake_sync_root, f"integration-{index}")
        for index in range(integration_shards)
    ]
    for index, arguments in enumerate(unit_arguments):
        assert UNIT_MARKER in arguments
        assert f"cache_dir={run_root / f'unit-{index}-pytest-cache'}" in arguments
        assert f"--junitxml={run_root / f'unit-{index}.xml'}" in arguments
        assert (fake_sync_root / f"unit-{index}.python-cache").read_text(
            encoding="utf-8"
        ).strip() == str(run_root / f"unit-{index}-python-bytecode")
        assert (fake_sync_root / f"unit-{index}.mujoco-cache").read_text(
            encoding="utf-8"
        ).strip() == str(run_root / f"unit-{index}-mujoco-cache")
        assert (fake_sync_root / f"unit-{index}.umask").read_text().strip() == "0022"
    for index, arguments in enumerate(integration_arguments):
        assert INTEGRATION_MARKER in arguments
        assert (
            f"cache_dir={run_root / f'integration-{index}-pytest-cache'}"
            in arguments
        )
        assert f"--junitxml={run_root / f'integration-{index}.xml'}" in arguments
        assert (fake_sync_root / f"integration-{index}.python-cache").read_text(
            encoding="utf-8"
        ).strip() == str(run_root / f"integration-{index}-python-bytecode")
        assert (fake_sync_root / f"integration-{index}.mujoco-cache").read_text(
            encoding="utf-8"
        ).strip() == str(run_root / f"integration-{index}-mujoco-cache")
        assert (
            fake_sync_root / f"integration-{index}.umask"
        ).read_text().strip() == "0022"
    assert all("--durations=2" in arguments for arguments in unit_arguments)
    assert all("--durations=2" in arguments for arguments in integration_arguments)
    shard_paths = [
        set((run_root / f"unit-{index}-paths.txt").read_text().splitlines())
        for index in range(unit_shards)
    ]
    expected_paths = {
        str(path) for path in (checkout / "tests").glob("test_*.py")
    }
    assert sum(len(paths) for paths in shard_paths) == len(expected_paths)
    assert len(set().union(*shard_paths)) == len(expected_paths)
    assert set().union(*shard_paths) == expected_paths
    for index, paths in enumerate(shard_paths):
        assert all(
            paths.isdisjoint(other)
            for other_index, other in enumerate(shard_paths)
            if index != other_index
        )
    integration_paths = [
        set((run_root / f"integration-{index}-paths.txt").read_text().splitlines())
        for index in range(integration_shards)
    ]
    assert sum(len(paths) for paths in integration_paths) == len(expected_paths)
    assert set().union(*integration_paths) == expected_paths
    for index, paths in enumerate(integration_paths):
        assert all(
            paths.isdisjoint(other)
            for other_index, other in enumerate(integration_paths)
            if index != other_index
        )
    merged = ET.parse(run_root / "unit.xml").getroot()
    assert merged.attrib["tests"] == str(unit_shards)
    assert merged.attrib["failures"] == "0"
    merged_integration = ET.parse(run_root / "integration.xml").getroot()
    assert merged_integration.attrib["tests"] == str(integration_shards)
    assert merged_integration.attrib["failures"] == "0"
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_rejects_relative_declared_run_root_before_starting_tiers(
    tmp_path: Path,
) -> None:
    checkout, wrapper, fake_sync_root = _wrapper_checkout(tmp_path)
    asset_root = tmp_path / "logical-assets"
    asset_root.mkdir()

    completed = _run_wrapper(
        wrapper,
        fake_sync_root,
        run_root="relative-output",
        asset_root=asset_root,
    )

    assert completed.returncode == 2
    assert "DEXTELEOP_FAST_FEEDBACK_RUN_ROOT must be an absolute path" in completed.stderr
    assert not list(fake_sync_root.glob("*.started"))
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_refuses_to_reuse_an_existing_declared_run_root(
    tmp_path: Path,
) -> None:
    checkout, wrapper, fake_sync_root = _wrapper_checkout(tmp_path)
    asset_root = tmp_path / "logical-assets"
    asset_root.mkdir()
    run_root = tmp_path / "existing-evidence"
    run_root.mkdir()
    sentinel = run_root / "preserve.txt"
    sentinel.write_text("owned by caller", encoding="utf-8")

    completed = _run_wrapper(
        wrapper,
        fake_sync_root,
        run_root=str(run_root),
        asset_root=asset_root,
    )

    assert completed.returncode == 2
    assert "DEXTELEOP_FAST_FEEDBACK_RUN_ROOT must not already exist" in completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "owned by caller"
    assert not list(fake_sync_root.glob("*.started"))
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_waits_for_both_tiers_and_propagates_a_tier_failure(
    tmp_path: Path,
) -> None:
    checkout, wrapper, fake_sync_root = _wrapper_checkout(tmp_path)
    asset_root = tmp_path / "logical-assets"
    asset_root.mkdir()
    run_root = tmp_path / "failed-run"

    completed = _run_wrapper(
        wrapper,
        fake_sync_root,
        run_root=str(run_root),
        asset_root=asset_root,
        fail_tier="integration-1",
    )

    assert completed.returncode == 1
    assert "fast unit (status=0)" in completed.stdout
    assert "fast integration (status=1)" in completed.stdout
    assert (run_root / "unit.xml").is_file()
    assert (run_root / "integration.xml").is_file()
    assert "requested failure: integration-1" in (
        run_root / "integration.log"
    ).read_text(encoding="utf-8")
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_propagates_a_unit_shard_failure_and_keeps_aggregate_evidence(
    tmp_path: Path,
) -> None:
    checkout, wrapper, fake_sync_root = _wrapper_checkout(tmp_path)
    asset_root = tmp_path / "logical-assets"
    asset_root.mkdir()
    run_root = tmp_path / "failed-unit-run"

    completed = _run_wrapper(
        wrapper,
        fake_sync_root,
        run_root=str(run_root),
        asset_root=asset_root,
        fail_tier="unit-1",
    )

    assert completed.returncode == 1
    assert "fast unit (status=1)" in completed.stdout
    assert "fast integration (status=0)" in completed.stdout
    assert "requested failure: unit-1" in (
        run_root / "unit-1.log"
    ).read_text(encoding="utf-8")
    assert (run_root / "unit.xml").is_file()
    merged = ET.parse(run_root / "unit.xml").getroot()
    assert merged.attrib["tests"] == str(DEFAULT_UNIT_SHARDS)
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_supports_single_unit_shard_opt_out(
    tmp_path: Path,
) -> None:
    checkout, wrapper, fake_sync_root = _wrapper_checkout(tmp_path)
    asset_root = tmp_path / "logical-assets"
    asset_root.mkdir()
    run_root = tmp_path / "single-unit-run"

    completed = _run_wrapper(
        wrapper,
        fake_sync_root,
        run_root=str(run_root),
        asset_root=asset_root,
        unit_shards="1",
    )

    assert completed.returncode == 0, completed.stderr
    assert (fake_sync_root / "unit-0.args").is_file()
    assert not (fake_sync_root / "unit-1.args").exists()
    assert (run_root / "unit-0-paths.txt").read_text().splitlines()
    assert ET.parse(run_root / "unit.xml").getroot().attrib["tests"] == "1"
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_falls_back_to_one_unit_child_for_a_targeted_test(
    tmp_path: Path,
) -> None:
    checkout, wrapper, fake_sync_root = _wrapper_checkout(tmp_path)
    asset_root = tmp_path / "logical-assets"
    asset_root.mkdir()
    run_root = tmp_path / "targeted-run"

    completed = _run_wrapper(
        wrapper,
        fake_sync_root,
        run_root=str(run_root),
        asset_root=asset_root,
        extra_arguments=("tests/test_alpha.py",),
    )

    assert completed.returncode == 0, completed.stderr
    assert "unit sharding disabled for explicit test target" in completed.stderr
    assert "integration sharding disabled for explicit test target" in completed.stderr
    assert (fake_sync_root / "unit.args").is_file()
    assert not (fake_sync_root / "unit-0.args").exists()
    assert (fake_sync_root / "integration.args").is_file()
    assert not (fake_sync_root / "integration-0.args").exists()
    assert _recorded_arguments(fake_sync_root, "unit").count("tests/test_alpha.py") == 1
    assert (
        _recorded_arguments(fake_sync_root, "integration").count(
            "tests/test_alpha.py"
        )
        == 1
    )
    assert (run_root / "unit.xml").is_file()
    assert (run_root / "integration.xml").is_file()
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_rejects_an_invalid_unit_shard_count_before_starting_tiers(
    tmp_path: Path,
) -> None:
    checkout, wrapper, fake_sync_root = _wrapper_checkout(tmp_path)
    asset_root = tmp_path / "logical-assets"
    asset_root.mkdir()

    completed = _run_wrapper(
        wrapper,
        fake_sync_root,
        run_root=str(tmp_path / "invalid-run"),
        asset_root=asset_root,
        unit_shards="5",
    )

    assert completed.returncode == 2
    assert "DEXTELEOP_FAST_FEEDBACK_UNIT_SHARDS" in completed.stderr
    assert not list(fake_sync_root.glob("*.started"))
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_rejects_an_invalid_integration_shard_count_before_starting_tiers(
    tmp_path: Path,
) -> None:
    checkout, wrapper, fake_sync_root = _wrapper_checkout(tmp_path)
    asset_root = tmp_path / "logical-assets"
    asset_root.mkdir()

    completed = _run_wrapper(
        wrapper,
        fake_sync_root,
        run_root=str(tmp_path / "invalid-integration-run"),
        asset_root=asset_root,
        integration_shards="5",
    )

    assert completed.returncode == 2
    assert "DEXTELEOP_FAST_FEEDBACK_INTEGRATION_SHARDS" in completed.stderr
    assert not list(fake_sync_root.glob("*.started"))
    assert not os.path.lexists(checkout / "assets")
