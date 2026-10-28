from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WRAPPER = SOURCE_ROOT / "scripts/run_full_acceptance.sh"
SOURCE_SHARDING = SOURCE_ROOT / "scripts/pytest_sharding.py"
SOURCE_IMPACT = SOURCE_ROOT / "scripts/pytest_impact.py"
REBOT_MODEL = Path("hardware/reBotArm_control_py/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf")
FIXTURE_MODULE_COUNT = 12


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
    for index in range(FIXTURE_MODULE_COUNT):
        (tests / f"test_case_{index}.py").write_text(
            "#" * (1_000 + index * 200), encoding="utf-8"
        )
    wrapper = scripts / "run_full_acceptance.sh"
    shutil.copy2(SOURCE_WRAPPER, wrapper)
    sharding_real = scripts / "pytest_sharding_real.py"
    shutil.copy2(SOURCE_SHARDING, sharding_real)
    shutil.copy2(SOURCE_IMPACT, scripts / "pytest_impact.py")
    node = scripts / "node"
    node.write_text('#!/bin/sh\nexit "${FAKE_NODE_PREFLIGHT_STATUS:-0}"\n', encoding="utf-8")
    node.chmod(0o755)
    (scripts / "pytest_sharding.py").write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

from pytest_sharding_real import plan_shards, write_plan

if __name__ == "__main__":
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
printf '%s\\n' "${DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS-unset}" >"$FAKE_SYNC_ROOT/$name.max-jobs"
printf 'start %s\\n' "$name" >>"$FAKE_SYNC_ROOT/events"
: >"$FAKE_SYNC_ROOT/$name.started"
for _ in {1..250}; do
  started=$(find "$FAKE_SYNC_ROOT" -name '*.started' | wc -l)
  [[ "$started" -ge "$FAKE_START_BARRIER" ]] && break
  sleep 0.02
done
[[ "$started" -ge "$FAKE_START_BARRIER" ]] || exit 91
if [[ "$FAKE_CHECK_FORKSERVER" == 1 ]]; then
  python3 -c 'from multiprocessing import forkserver; forkserver.ensure_running()'
fi
mkdir -p -- "$(dirname -- "$junit")"
printf '<testsuite tests="1" errors="0" failures="0" skipped="0" time="0.1"><testcase classname="tests.fake" name="%s" time="0.1"/></testsuite>\\n' "$name" >"$junit"
echo "fake shard: $name"
if [[ "$name" == shard-0 && -n "${FAKE_MUTATE_SOURCE:-}" ]]; then
  printf '# source drift\\n' >>"$FAKE_MUTATE_SOURCE"
fi
printf 'end %s\\n' "$name" >>"$FAKE_SYNC_ROOT/events"
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
    max_jobs: int | str | None = None,
    check_forkserver: bool = False,
    fail_shard: str = "",
    socket_preflight_status: int = 0,
    node_preflight_status: int = 0,
    extra_arguments: tuple[str, ...] = (),
    mutate_source: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.pop("DEXTELEOP_FULL_ACCEPTANCE_SHARDS", None)
    environment.pop("DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS", None)
    effective_shards = 4 if shard_count is None else shard_count
    environment.update(
        {
            "DEXTELEOP_ASSET_ROOT": str(asset_root),
            "DEXTELEOP_FULL_ACCEPTANCE_RUN_ROOT": str(run_root),
            "FAKE_FAIL_SHARD": fail_shard,
            "FAKE_START_BARRIER": str(effective_shards if max_jobs is None else max_jobs),
            "FAKE_CHECK_FORKSERVER": "1" if check_forkserver else "0",
            "FAKE_SOCKET_PREFLIGHT_STATUS": str(socket_preflight_status),
            "FAKE_NODE_PREFLIGHT_STATUS": str(node_preflight_status),
            "FAKE_MUTATE_SOURCE": str(mutate_source) if mutate_source else "",
            "FAKE_SYNC_ROOT": str(sync_root),
            "PATH": str(wrapper.parent) + os.pathsep + environment.get("PATH", ""),
        }
    )
    if shard_count is not None:
        environment["DEXTELEOP_FULL_ACCEPTANCE_SHARDS"] = str(shard_count)
    if max_jobs is not None:
        environment["DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS"] = str(max_jobs)
    return subprocess.run(
        [str(wrapper), "--durations=3", *extra_arguments],
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
    assert len(planned) == len(set(planned)) == FIXTURE_MODULE_COUNT
    isolated_paths = {
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
    temporary_paths = {
        Path((sync_root / f"shard-{index}.tmpdir").read_text().strip())
        for index in range(shard_count)
    }
    assert len(temporary_paths) == shard_count
    assert len({path.parent for path in temporary_paths}) == 1
    for path in temporary_paths:
        assert str(path).startswith("/tmp/dt-full-tmp-")
        assert len(os.fsencode(path)) <= 48
        assert not path.parent.exists()
    assert {
        (sync_root / f"shard-{index}.mujoco-cache").read_text().strip()
        for index in range(shard_count)
    } == {str(run_root / "mujoco-cache")}
    assert (run_root / "mujoco-cache").is_dir()
    for index in range(shard_count):
        arguments = (sync_root / f"shard-{index}.args").read_text().splitlines()
        assert arguments.count("scripts.pytest_native_memory") == 1
        assert arguments[arguments.index("scripts.pytest_native_memory") - 1] == "-p"
        assert "not hardware" in arguments
        assert "--durations=3" in arguments
        assert f"cache_dir={run_root / f'shard-{index}-pytest-cache'}" in arguments
        assert f"--junitxml={run_root / f'shard-{index}.xml'}" in arguments
        assert (sync_root / f"shard-{index}.mujoco-gl").read_text().strip() == "egl"
        assert (sync_root / f"shard-{index}.umask").read_text().strip() == "0022"
        assert (sync_root / f"shard-{index}.max-jobs").read_text().strip() == "unset"
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


@pytest.mark.parametrize(
    "shard_count,max_jobs",
    [(6, 1), (6, 2), (6, 3), (5, 2), (6, 5), (7, 2), (12, 1), (12, 2), (12, 5), (12, 6)],
)
def test_wrapper_bounds_live_jobs_without_reducing_the_plan(
    tmp_path: Path, shard_count: int, max_jobs: int,
) -> None:
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    run_root = tmp_path / "bounded"

    completed = _run_wrapper(
        wrapper, sync_root, assets, run_root, shard_count=shard_count, max_jobs=max_jobs
    )

    assert completed.returncode == 0, completed.stderr
    _assert_complete_run(sync_root, run_root, shard_count=shard_count)
    active: set[str] = set()
    peak = 0
    for line in (sync_root / "events").read_text().splitlines():
        event, name = line.split()
        if event == "start":
            assert name not in active
            active.add(name)
            peak = max(peak, len(active))
        else:
            assert event == "end" and name in active
            active.remove(name)
    assert not active
    assert peak == max_jobs
    assert not os.path.lexists(checkout / "assets")


@pytest.mark.parametrize(
    "shard_count,max_jobs",
    [(None, 0), (None, 5), (None, 7), (None, "auto"), (12, None), (12, 7), (12, "auto")],
)
def test_wrapper_rejects_invalid_job_limits_before_artifacts(
    tmp_path: Path, shard_count: int | None, max_jobs: int | str | None,
) -> None:
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    run_root = tmp_path / "invalid-jobs"

    completed = _run_wrapper(
        wrapper, sync_root, assets, run_root, shard_count=shard_count, max_jobs=max_jobs
    )

    assert completed.returncode == 2
    assert "DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS" in completed.stderr
    assert not run_root.exists()
    assert not list(sync_root.glob("*.started"))
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_keeps_forkserver_working_with_a_long_evidence_root(
    tmp_path: Path, short_unix_socket_path_factory,
) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(str(short_unix_socket_path_factory("full")))
        except OSError as exc:
            pytest.skip(f"AF_UNIX unavailable: {exc}")
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    run_root = tmp_path / ("evidence-" + "x" * 120)
    assert len(os.fsencode(run_root)) > 107

    completed = _run_wrapper(
        wrapper, sync_root, assets, run_root, shard_count=2, check_forkserver=True
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    _assert_complete_run(sync_root, run_root, shard_count=2)
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


@pytest.mark.parametrize("shard_count", ["0", "13", "auto"])
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
    assert "must be an integer from 1 to 12" in completed.stderr
    assert not run_root.exists()
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


def test_wrapper_rejects_incompatible_node_before_artifacts_or_shards(
    tmp_path: Path,
) -> None:
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    run_root = tmp_path / "node-blocked"

    completed = _run_wrapper(
        wrapper, sync_root, assets, run_root, node_preflight_status=17
    )

    assert completed.returncode == 2
    assert "Node.js on PATH cannot run the annotation browser contracts" in completed.stderr
    assert not run_root.exists()
    assert not list(sync_root.glob("*.started"))
    assert not os.path.lexists(checkout / "assets")


@pytest.mark.parametrize("shard_count,max_jobs", [(6, None), (6, 1), (6, 2), (12, 2), (12, 5)])
def test_wrapper_propagates_failure_after_preserving_other_shards(
    tmp_path: Path, shard_count: int, max_jobs: int | None,
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
        shard_count=shard_count,
        max_jobs=max_jobs,
        fail_shard="shard-1",
    )

    assert completed.returncode == 1
    assert "full acceptance shard 1 (status=7)" in completed.stdout
    assert all((run_root / f"shard-{index}.xml").is_file() for index in range(shard_count))
    assert (run_root / "full-acceptance.xml").is_file()
    _assert_complete_run(sync_root, run_root, shard_count=shard_count)
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


def _changed_wrapper_checkout(tmp_path):
    checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    modules = {
        f"tests/test_{name}.py"
        for name in (
            "b601_automatic_calibration",
            "b601_calibration_cycle",
            "b601_coordinated_holding",
            "b601_cruise_velocity",
            "b601_workspace_calibration",
            "b601_workspace_identification",
            "rebot_b601_staged_assist",
        )
    }
    for path in modules:
        (checkout / path).write_text("def test_control(): pass\n")
    (checkout / "tests/conftest.py").write_text("# shared fixtures\n")
    (checkout / ".gitignore").write_text("assets\n__pycache__/\n*.pyc\n")
    (checkout / "agent_gateway").mkdir()
    feature = checkout / "agent_gateway/feature.py"
    feature.write_text("VALUE = 1\n")

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=checkout, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Wrapper test")
    git("config", "user.email", "wrapper-test@example.invalid")
    git("add", "-A")
    git("-c", "commit.gpgsign=false", "commit", "-qm", "baseline")
    base = git("rev-parse", "HEAD")
    feature.write_text("VALUE = 2\n")
    git("add", "-A")
    git("-c", "commit.gpgsign=false", "commit", "-qm", "independent feature")
    return checkout, wrapper, sync_root, base, modules


def test_change_aware_wrapper_omits_only_b601_and_labels_its_evidence(tmp_path):
    checkout, wrapper, sync_root, base, excluded = _changed_wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "changed"
    result = _run_wrapper(
        wrapper,
        sync_root,
        assets,
        output,
        shard_count=2,
        extra_arguments=("--changed-since", base),
    )
    assert result.returncode == 0, result.stderr + result.stdout
    selection = json.loads((output / "selection.json").read_text())
    assert selection["mode"] == "affected"
    assert set(selection["excluded_modules"]) == excluded
    plan = json.loads((output / "shard-plan.json").read_text())
    planned = [p for shard in plan["shards"] for p in shard["test_modules"]]
    assert set(planned) == {
        f"tests/test_case_{i}.py" for i in range(FIXTURE_MODULE_COUNT)
    }
    for index in range(2):
        arguments = (sync_root / f"shard-{index}.args").read_text().splitlines()
        assert not any("changed-since" in arg or arg == base for arg in arguments)
        assert not any(str(checkout / p) in arguments for p in excluded)
    assert not (output / "full-acceptance.xml").exists()
    root = ET.parse(output / "changed-acceptance.xml").getroot()
    assert root.attrib["name"] == "changed-acceptance"
    suites = list(root.iter("testsuite"))
    assert suites
    for suite in suites:
        properties = {
            p.attrib["name"]: p.attrib["value"]
            for p in suite.findall("properties/property")
        }
        assert properties["acceptance.kind"] == "change-aware"
        assert properties["acceptance.mode"] == "affected"
        assert properties["acceptance.head"] == selection["head_commit"]


def test_change_aware_wrapper_shared_change_falls_back_to_every_module(tmp_path):
    checkout, wrapper, sync_root, base, excluded = _changed_wrapper_checkout(tmp_path)
    (checkout / "shared.py").write_text("VALUE = 1\n")
    subprocess.run(
        ["git", "add", "shared.py"], cwd=checkout, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "shared change"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "changed"
    result = _run_wrapper(
        wrapper,
        sync_root,
        assets,
        output,
        shard_count=2,
        extra_arguments=(f"--changed-since={base}",),
    )
    assert result.returncode == 0, result.stderr + result.stdout
    selection = json.loads((output / "selection.json").read_text())
    assert selection["mode"] == "full" and selection["excluded_modules"] == []
    assert excluded <= set(selection["selected_modules"])
    assert len(selection["selected_modules"]) == FIXTURE_MODULE_COUNT + len(excluded)


@pytest.mark.parametrize(
    "arguments",
    [
        ("-k", "happy"),
        ("--deselect=tests/test_case_0.py",),
        ("--collect-only",),
        ("--help",),
        ("-p", "custom_plugin"),
        ("--durations=nan",),
    ],
)
def test_change_aware_wrapper_rejects_narrowing_or_unverified_options(
    tmp_path, arguments
):
    _checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "changed"
    result = _run_wrapper(
        wrapper,
        sync_root,
        assets,
        output,
        extra_arguments=("--changed-since", "HEAD", *arguments),
    )
    assert result.returncode == 2
    assert not output.exists()
    assert not list(sync_root.glob("*.started"))


@pytest.mark.parametrize("name", ["PYTEST_ADDOPTS", "PYTEST_PLUGINS"])
def test_change_aware_wrapper_rejects_ambient_pytest_selection(
    tmp_path, monkeypatch, name
):
    _checkout, wrapper, sync_root = _wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "changed"
    monkeypatch.setenv(name, "unexpected")
    result = _run_wrapper(
        wrapper, sync_root, assets, output, extra_arguments=("--changed-since", "HEAD")
    )
    assert result.returncode == 2
    assert name in result.stderr
    assert not output.exists()
    assert not list(sync_root.glob("*.started"))


@pytest.mark.parametrize(
    "problem", ["dirty-before", "invalid-base", "drift-during-run"]
)
def test_change_aware_wrapper_refuses_invalid_source_evidence(tmp_path, problem):
    checkout, wrapper, sync_root, base, _excluded = _changed_wrapper_checkout(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "changed"
    feature = checkout / "agent_gateway/feature.py"
    if problem == "dirty-before":
        feature.write_text("VALUE = 3\n")
    if problem == "invalid-base":
        base = "missing-reference"
    result = _run_wrapper(
        wrapper,
        sync_root,
        assets,
        output,
        shard_count=2,
        extra_arguments=("--changed-since", base),
        mutate_source=feature if problem == "drift-during-run" else None,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    if problem != "drift-during-run":
        assert not list(sync_root.glob("*.started"))
    else:
        assert len(list(sync_root.glob("*.started"))) == 2
        assert not (output / "changed-acceptance.xml").exists()
    assert not os.path.lexists(checkout / "assets")
