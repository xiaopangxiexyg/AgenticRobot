from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

SOURCE_WRAPPER = Path(__file__).resolve().parents[1] / "scripts/run_fast_feedback.sh"
UNIT_MARKER = "unit and not optional and not hardware and not slow_sim"
INTEGRATION_MARKER = (
    "integration and not optional and not hardware and not slow_sim"
)


def _wrapper_checkout(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkout = tmp_path / "checkout"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    wrapper = scripts / "run_fast_feedback.sh"
    shutil.copy2(SOURCE_WRAPPER, wrapper)

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
  unit*) name="unit"; other="integration" ;;
  integration*) name="integration"; other="unit" ;;
  *) echo "unexpected marker: $marker" >&2; exit 90 ;;
esac

arguments_file="$FAKE_SYNC_ROOT/$name.args"
printf '%s\\n' "${all_arguments[@]}" >"$arguments_file"
printf '%s\\n' "$PYTHONPYCACHEPREFIX" >"$FAKE_SYNC_ROOT/$name.python-cache"
printf '%s\\n' "$SIM_MUJOCO_CACHE_DIR" >"$FAKE_SYNC_ROOT/$name.mujoco-cache"
: >"$FAKE_SYNC_ROOT/$name.started"
for _ in {1..250}; do
  [[ -e "$FAKE_SYNC_ROOT/$other.started" ]] && break
  sleep 0.02
done
[[ -e "$FAKE_SYNC_ROOT/$other.started" ]] || {
  echo "$other tier did not start concurrently" >&2
  exit 91
}

mkdir -p -- "$(dirname -- "$junit")"
printf '<testsuite name="%s" tests="1" failures="0"/>\\n' "$name" >"$junit"
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
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(
        {
            "DEXTELEOP_ASSET_ROOT": str(asset_root.resolve()),
            "DEXTELEOP_FAST_FEEDBACK_RUN_ROOT": run_root,
            "FAKE_FAIL_TIER": fail_tier,
            "FAKE_SYNC_ROOT": str(fake_sync_root),
        }
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
    assert (run_root / "unit.log").read_text(encoding="utf-8").strip() == (
        "fake tier: unit"
    )
    assert (run_root / "integration.log").read_text(encoding="utf-8").strip() == (
        "fake tier: integration"
    )
    assert (run_root / "unit.xml").is_file()
    assert (run_root / "integration.xml").is_file()

    unit_arguments = _recorded_arguments(fake_sync_root, "unit")
    integration_arguments = _recorded_arguments(fake_sync_root, "integration")
    assert UNIT_MARKER in unit_arguments
    assert INTEGRATION_MARKER in integration_arguments
    assert f"cache_dir={run_root / 'unit-pytest-cache'}" in unit_arguments
    assert f"cache_dir={run_root / 'integration-pytest-cache'}" in integration_arguments
    assert f"--junitxml={run_root / 'unit.xml'}" in unit_arguments
    assert f"--junitxml={run_root / 'integration.xml'}" in integration_arguments
    assert "--durations=2" in unit_arguments
    assert "--durations=2" in integration_arguments
    assert (fake_sync_root / "unit.python-cache").read_text(
        encoding="utf-8"
    ).strip() == str(run_root / "unit-python-bytecode")
    assert (fake_sync_root / "integration.python-cache").read_text(
        encoding="utf-8"
    ).strip() == str(run_root / "integration-python-bytecode")
    assert (fake_sync_root / "unit.mujoco-cache").read_text(
        encoding="utf-8"
    ).strip() == str(run_root / "unit-mujoco-cache")
    assert (fake_sync_root / "integration.mujoco-cache").read_text(
        encoding="utf-8"
    ).strip() == str(run_root / "integration-mujoco-cache")
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
        fail_tier="integration",
    )

    assert completed.returncode == 1
    assert "fast unit (status=0)" in completed.stdout
    assert "fast integration (status=7)" in completed.stdout
    assert (run_root / "unit.xml").is_file()
    assert (run_root / "integration.xml").is_file()
    assert "requested failure: integration" in (
        run_root / "integration.log"
    ).read_text(encoding="utf-8")
    assert not os.path.lexists(checkout / "assets")
