from __future__ import annotations

import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WRAPPER = SOURCE_ROOT / "scripts/run_full_acceptance.sh"
SOURCE_SHARDING = SOURCE_ROOT / "scripts/pytest_sharding.py"


def _wrapper_checkout(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkout = tmp_path / "checkout"
    scripts = checkout / "scripts"
    tests = checkout / "tests"
    scripts.mkdir(parents=True)
    tests.mkdir()
    for index in range(9):
        (tests / f"test_case_{index}.py").write_text(
            "#" * (1_000 + index * 200), encoding="utf-8"
        )
    wrapper = scripts / "run_full_acceptance.sh"
    shutil.copy2(SOURCE_WRAPPER, wrapper)
    shutil.copy2(SOURCE_SHARDING, scripts / "pytest_sharding.py")

    sync_root = tmp_path / "sync"
    sync_root.mkdir()
    runner = scripts / "run_pytest.sh"
    runner.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
junit=""
all_arguments=("$@")
while (($#)); do
  case "$1" in
    --junitxml=*) junit="${1#--junitxml=}" ;;
  esac
  shift
done
name="${junit##*/}"
name="${name%.xml}"
printf '%s\\n' "${all_arguments[@]}" >"$FAKE_SYNC_ROOT/$name.args"
printf '%s\\n' "$TMPDIR" >"$FAKE_SYNC_ROOT/$name.tmpdir"
printf '%s\\n' "$XDG_CACHE_HOME" >"$FAKE_SYNC_ROOT/$name.xdg-cache"
printf '%s\\n' "$MUJOCO_GL" >"$FAKE_SYNC_ROOT/$name.mujoco-gl"
umask >"$FAKE_SYNC_ROOT/$name.umask"
: >"$FAKE_SYNC_ROOT/$name.started"
for _ in {1..250}; do
  started=$(find "$FAKE_SYNC_ROOT" -name '*.started' | wc -l)
  [[ "$started" -eq "$FAKE_SHARD_COUNT" ]] && break
  sleep 0.02
done
[[ "$started" -eq "$FAKE_SHARD_COUNT" ]] || exit 91
mkdir -p -- "$(dirname -- "$junit")"
printf '<testsuite tests="1" errors="0" failures="0" skipped="0" time="0.1"><testcase classname="tests.fake" name="%s" time="0.1"/></testsuite>\\n' "$name" >"$junit"
echo "fake shard: $name"
[[ "${FAKE_FAIL_SHARD:-}" != "$name" ]] || exit 7
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    return checkout, wrapper, sync_root


def _run_wrapper(
    wrapper: Path,
    sync_root: Path,
    asset_root: Path,
    run_root: Path,
    *,
    shard_count: int = 4,
    fail_shard: str = "",
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(
        {
            "DEXTELEOP_ASSET_ROOT": str(asset_root),
            "DEXTELEOP_FULL_ACCEPTANCE_RUN_ROOT": str(run_root),
            "DEXTELEOP_FULL_ACCEPTANCE_SHARDS": str(shard_count),
            "FAKE_FAIL_SHARD": fail_shard,
            "FAKE_SHARD_COUNT": str(shard_count),
            "FAKE_SYNC_ROOT": str(sync_root),
        }
    )
    return subprocess.run(
        [str(wrapper), "--durations=3"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_wrapper_runs_complete_isolated_shards_and_merges_evidence(
    tmp_path: Path,
) -> None:
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    run_root = tmp_path / "full"

    completed = _run_wrapper(wrapper, sync_root, assets, run_root)

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((run_root / "shard-plan.json").read_text())
    assert manifest["shard_count"] == 4
    planned = [
        module for shard in manifest["shards"] for module in shard["test_modules"]
    ]
    assert len(planned) == len(set(planned)) == 9
    tmpdirs = {
        (sync_root / f"shard-{index}.tmpdir").read_text().strip() for index in range(4)
    }
    xdg_caches = {
        (sync_root / f"shard-{index}.xdg-cache").read_text().strip()
        for index in range(4)
    }
    assert len(tmpdirs) == len(xdg_caches) == 4
    for index in range(4):
        arguments = (sync_root / f"shard-{index}.args").read_text().splitlines()
        assert "not hardware" in arguments
        assert "--durations=3" in arguments
        assert f"--junitxml={run_root / f'shard-{index}.xml'}" in arguments
        assert (sync_root / f"shard-{index}.mujoco-gl").read_text().strip() == "egl"
        assert (sync_root / f"shard-{index}.umask").read_text().strip() == "0022"
    aggregate = ET.parse(run_root / "full-acceptance.xml").getroot()
    assert aggregate.attrib["tests"] == "4"
    assert aggregate.attrib["failures"] == "0"
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_propagates_failure_after_preserving_other_shards(
    tmp_path: Path,
) -> None:
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    run_root = tmp_path / "failed"

    completed = _run_wrapper(wrapper, sync_root, assets, run_root, fail_shard="shard-2")

    assert completed.returncode == 1
    assert "full acceptance shard 2 (status=7)" in completed.stdout
    assert all((run_root / f"shard-{index}.xml").is_file() for index in range(4))
    assert (run_root / "full-acceptance.xml").is_file()
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_rejects_existing_run_root_before_starting(tmp_path: Path) -> None:
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    run_root = tmp_path / "existing"
    run_root.mkdir()
    sentinel = run_root / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    completed = _run_wrapper(wrapper, sync_root, assets, run_root)

    assert completed.returncode == 2
    assert "must not already exist" in completed.stderr
    assert sentinel.read_text() == "preserve"
    assert not list(sync_root.glob("*.started"))
    assert not os.path.lexists(checkout / "assets")
