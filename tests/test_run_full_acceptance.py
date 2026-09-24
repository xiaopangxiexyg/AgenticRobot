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
REBOT_MODEL = Path("hardware/reBotArm_control_py/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf")


def _wrapper_checkout(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkout = tmp_path / "checkout"
    scripts = checkout / "scripts"
    tests = checkout / "tests"
    scripts.mkdir(parents=True)
    tests.mkdir()
    model = checkout / REBOT_MODEL
    model.parent.mkdir(parents=True)
    model.write_text("<robot name='preflight-fixture'/>", encoding="utf-8")
    model.chmod(0o444)
    for index in range(9):
        (tests / f"test_case_{index}.py").write_text(
            "#" * (1_000 + index * 200), encoding="utf-8"
        )
    wrapper = scripts / "run_full_acceptance.sh"
    shutil.copy2(SOURCE_WRAPPER, wrapper)
    sharding_real = scripts / "pytest_sharding_real.py"
    shutil.copy2(SOURCE_SHARDING, sharding_real)
    (scripts / "pytest_sharding.py").write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

if sys.argv[1:2] == ["check-sockets"]:
    status = int(os.environ.get("FAKE_SOCKET_PREFLIGHT_STATUS", "0"))
    if status:
        print("fake socket preflight rejected", file=sys.stderr)
    raise SystemExit(status)

real = Path(__file__).with_name("pytest_sharding_real.py")
raise SystemExit(subprocess.run([sys.executable, str(real), *sys.argv[1:]]).returncode)
""",
        encoding="utf-8",
    )

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
printf '%s\\n' "$PYTHONPYCACHEPREFIX" >"$FAKE_SYNC_ROOT/$name.python-cache"
printf '%s\\n' "$SIM_MUJOCO_CACHE_DIR" >"$FAKE_SYNC_ROOT/$name.mujoco-cache"
printf '%s\\n' "$MPLCONFIGDIR" >"$FAKE_SYNC_ROOT/$name.matplotlib-cache"
printf '%s\\n' "$NUMBA_CACHE_DIR" >"$FAKE_SYNC_ROOT/$name.numba-cache"
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
    shard_count: int | str | None = None,
    fail_shard: str = "",
    socket_preflight_status: int = 0,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.pop("DEXTELEOP_FULL_ACCEPTANCE_SHARDS", None)
    environment.update(
        {
            "DEXTELEOP_ASSET_ROOT": str(asset_root),
            "DEXTELEOP_FULL_ACCEPTANCE_RUN_ROOT": str(run_root),
            "FAKE_FAIL_SHARD": fail_shard,
            "FAKE_SHARD_COUNT": str(4 if shard_count is None else shard_count),
            "FAKE_SOCKET_PREFLIGHT_STATUS": str(socket_preflight_status),
            "FAKE_SYNC_ROOT": str(sync_root),
        }
    )
    if shard_count is not None:
        environment["DEXTELEOP_FULL_ACCEPTANCE_SHARDS"] = str(shard_count)
    return subprocess.run(
        [str(wrapper), "--durations=3"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _assert_complete_run(
    sync_root: Path, run_root: Path, *, shard_count: int
) -> None:
    manifest = json.loads((run_root / "shard-plan.json").read_text())
    assert manifest["shard_count"] == shard_count
    planned = [
        module for shard in manifest["shards"] for module in shard["test_modules"]
    ]
    assert len(planned) == len(set(planned)) == 9
    isolated_paths = {
        "tmpdir": "tmp",
        "xdg-cache": "xdg-cache",
        "python-cache": "python-bytecode",
        "matplotlib-cache": "matplotlib-cache",
        "numba-cache": "numba-cache",
    }
    for evidence_suffix, run_suffix in isolated_paths.items():
        observed = {
            (sync_root / f"shard-{index}.{evidence_suffix}").read_text().strip()
            for index in range(shard_count)
        }
        expected = {
            str(run_root / f"shard-{index}-{run_suffix}")
            for index in range(shard_count)
        }
        assert observed == expected
    assert {
        (sync_root / f"shard-{index}.mujoco-cache").read_text().strip()
        for index in range(shard_count)
    } == {str(run_root / "mujoco-cache")}
    assert (run_root / "mujoco-cache").is_dir()
    for index in range(shard_count):
        arguments = (sync_root / f"shard-{index}.args").read_text().splitlines()
        assert "not hardware" in arguments
        assert "--durations=3" in arguments
        assert f"cache_dir={run_root / f'shard-{index}-pytest-cache'}" in arguments
        assert f"--junitxml={run_root / f'shard-{index}.xml'}" in arguments
        assert (sync_root / f"shard-{index}.mujoco-gl").read_text().strip() == "egl"
        assert (sync_root / f"shard-{index}.umask").read_text().strip() == "0022"
    aggregate = ET.parse(run_root / "full-acceptance.xml").getroot()
    cases = aggregate.findall(".//testcase")
    assert aggregate.attrib["tests"] == str(shard_count)
    assert aggregate.attrib["failures"] == "0"
    assert {case.attrib["name"] for case in cases} == {
        f"shard-{index}" for index in range(shard_count)
    }


def test_wrapper_defaults_to_four_complete_shards(
    tmp_path: Path,
) -> None:
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    run_root = tmp_path / "full"

    completed = _run_wrapper(wrapper, sync_root, assets, run_root)

    assert completed.returncode == 0, completed.stderr
    _assert_complete_run(sync_root, run_root, shard_count=4)
    assert not os.path.lexists(checkout / "assets")


@pytest.mark.parametrize("shard_count", [5, 6])
def test_wrapper_supports_explicit_additional_shards(
    tmp_path: Path, shard_count: int
) -> None:
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    run_root = tmp_path / f"full-{shard_count}"

    completed = _run_wrapper(
        wrapper, sync_root, assets, run_root, shard_count=shard_count
    )

    assert completed.returncode == 0, completed.stderr
    _assert_complete_run(sync_root, run_root, shard_count=shard_count)
    assert not os.path.lexists(checkout / "assets")


@pytest.mark.parametrize("shard_count", ["7", "auto"])
def test_wrapper_rejects_unsupported_shard_counts(
    tmp_path: Path, shard_count: str
) -> None:
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    run_root = tmp_path / f"invalid-{shard_count}"

    completed = _run_wrapper(
        wrapper, sync_root, assets, run_root, shard_count=shard_count
    )

    assert completed.returncode == 2
    assert "must be an integer from 1 to 6" in completed.stderr
    assert not list(sync_root.glob("*.started"))
    assert not os.path.lexists(checkout / "assets")


@pytest.mark.parametrize("model_state", ["absent", "directory", "dangling-link"])
def test_wrapper_rejects_missing_model_before_artifacts_or_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_state: str,
) -> None:
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    model = checkout / REBOT_MODEL
    model.unlink()
    if model_state == "directory":
        model.mkdir()
    elif model_state == "dangling-link":
        model.symlink_to(tmp_path / "missing.urdf")
    assets = tmp_path / "assets"
    assets.mkdir()
    run_root = tmp_path / "model-blocked"
    # Some tests accept an external input; it cannot hide the missing checkout input.
    monkeypatch.setenv("DEXTELEOP_REBOT_SOURCE_ROOT", str(assets))

    completed = _run_wrapper(wrapper, sync_root, assets, run_root)

    assert completed.returncode == 2
    assert "required B601 URDF is missing or unreadable" in completed.stderr
    assert str(model) in completed.stderr
    assert "recorded gitlink" in completed.stderr
    assert not os.path.lexists(run_root)
    assert not list(sync_root.glob("*.started"))
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_rejects_missing_socket_capability_before_artifact_root(
    tmp_path: Path,
) -> None:
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    run_root = tmp_path / "socket-blocked"

    completed = _run_wrapper(
        wrapper,
        sync_root,
        assets,
        run_root,
        socket_preflight_status=2,
    )

    assert completed.returncode == 2
    assert "fake socket preflight rejected" in completed.stderr
    assert "complete acceptance requires working loopback TCP" in completed.stderr
    assert not os.path.lexists(run_root)
    assert not list(sync_root.glob("*.started"))
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_propagates_failure_after_preserving_other_shards(
    tmp_path: Path,
) -> None:
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    run_root = tmp_path / "failed"

    completed = _run_wrapper(
        wrapper,
        sync_root,
        assets,
        run_root,
        shard_count=6,
        fail_shard="shard-5",
    )

    assert completed.returncode == 1
    assert "full acceptance shard 5 (status=7)" in completed.stdout
    assert all((run_root / f"shard-{index}.xml").is_file() for index in range(6))
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
