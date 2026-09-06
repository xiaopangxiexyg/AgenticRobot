from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE_WRAPPER = Path(__file__).resolve().parents[1] / "scripts/run_pytest.sh"
SOURCE_PYTHON_RESOLVER = (
    Path(__file__).resolve().parents[1] / "scripts/resolve_dev_python.sh"
)


def _wrapper_checkout(tmp_path: Path) -> tuple[Path, Path]:
    checkout = tmp_path / "checkout"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    wrapper = scripts / "run_pytest.sh"
    shutil.copy2(SOURCE_WRAPPER, wrapper)
    shutil.copy2(SOURCE_PYTHON_RESOLVER, scripts / "resolve_dev_python.sh")
    return checkout, wrapper


def _run_wrapper(
    wrapper: Path,
    probe: Path,
    *,
    asset_root: str,
    expected_root: Path,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(
        {
            "DEXTELEOP_ASSET_ROOT": asset_root,
            "DEXTELEOP_PYTHON": sys.executable,
            "EXPECTED_ASSET_ROOT": str(expected_root),
        }
    )
    return subprocess.run(
        [str(wrapper), "-p", "no:cacheprovider", "-q", str(probe)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _write_probe(path: Path, *, passes: bool = True) -> None:
    assertion = "True" if passes else "False"
    path.write_text(
        "from pathlib import Path\n"
        "import os\n\n"
        "def test_asset_view():\n"
        "    link = Path.cwd() / 'assets'\n"
        "    assert link.is_symlink()\n"
        "    assert link.resolve() == Path(os.environ['EXPECTED_ASSET_ROOT']).resolve()\n"
        f"    assert {assertion}\n",
        encoding="utf-8",
    )


def _environment(python: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["DEXTELEOP_PYTHON"] = str(python)
    return environment


def test_wrapper_mounts_declared_asset_root_and_removes_temporary_link(tmp_path: Path) -> None:
    checkout, wrapper = _wrapper_checkout(tmp_path)
    asset_root = tmp_path / "logical-input"
    asset_root.mkdir()
    probe = tmp_path / "test_probe.py"
    _write_probe(probe)

    completed = _run_wrapper(
        wrapper,
        probe,
        asset_root=str(asset_root.resolve()),
        expected_root=asset_root,
    )

    assert completed.returncode == 0, completed.stderr
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_removes_temporary_asset_link_after_pytest_failure(tmp_path: Path) -> None:
    checkout, wrapper = _wrapper_checkout(tmp_path)
    asset_root = tmp_path / "logical-input"
    asset_root.mkdir()
    probe = tmp_path / "test_probe.py"
    _write_probe(probe, passes=False)

    completed = _run_wrapper(
        wrapper,
        probe,
        asset_root=str(asset_root.resolve()),
        expected_root=asset_root,
    )

    assert completed.returncode != 0
    assert not os.path.lexists(checkout / "assets")


def test_wrapper_preserves_an_existing_matching_asset_link(tmp_path: Path) -> None:
    checkout, wrapper = _wrapper_checkout(tmp_path)
    asset_root = tmp_path / "logical-input"
    asset_root.mkdir()
    existing = checkout / "assets"
    existing.symlink_to(asset_root, target_is_directory=True)
    probe = tmp_path / "test_probe.py"
    _write_probe(probe)

    completed = _run_wrapper(
        wrapper,
        probe,
        asset_root=str(asset_root.resolve()),
        expected_root=asset_root,
    )

    assert completed.returncode == 0, completed.stderr
    assert existing.is_symlink()
    assert existing.resolve() == asset_root.resolve()


def test_wrapper_rejects_a_mismatched_existing_asset_path(tmp_path: Path) -> None:
    checkout, wrapper = _wrapper_checkout(tmp_path)
    asset_root = tmp_path / "logical-input"
    asset_root.mkdir()
    existing = checkout / "assets"
    existing.mkdir()
    probe = tmp_path / "test_probe.py"
    _write_probe(probe)

    completed = _run_wrapper(
        wrapper,
        probe,
        asset_root=str(asset_root.resolve()),
        expected_root=asset_root,
    )

    assert completed.returncode != 0
    assert "existing assets path does not match DEXTELEOP_ASSET_ROOT" in completed.stderr
    assert existing.is_dir()


def test_wrapper_rejects_a_relative_asset_root_without_creating_a_link(tmp_path: Path) -> None:
    checkout, wrapper = _wrapper_checkout(tmp_path)
    probe = tmp_path / "test_probe.py"
    _write_probe(probe)

    completed = _run_wrapper(
        wrapper,
        probe,
        asset_root="relative-assets",
        expected_root=tmp_path,
    )

    assert completed.returncode != 0
    assert "DEXTELEOP_ASSET_ROOT must be an absolute directory" in completed.stderr
    assert not os.path.lexists(checkout / "assets")


@pytest.mark.parametrize(
    ("overrides", "expected_timeout", "expected_faulthandler"),
    [
        ({}, "120", "60"),
        (
            {
                "DEXTELEOP_PYTEST_TIMEOUT_SECONDS": "7.5",
                "DEXTELEOP_PYTEST_FAULTHANDLER_TIMEOUT_SECONDS": ".25",
            },
            "7.5",
            ".25",
        ),
    ],
)
def test_wrapper_injects_owned_watchdog_options(
    tmp_path: Path,
    overrides: dict[str, str],
    expected_timeout: str,
    expected_faulthandler: str,
) -> None:
    _checkout, wrapper = _wrapper_checkout(tmp_path)
    invocation_log = tmp_path / "pytest-arguments.txt"
    fake_python = tmp_path / "python3.10"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ ${1:-} == -m && ${2:-} == pytest ]]; then\n"
        "  shift 2\n"
        '  printf \'%s\\n\' "$@" >"$PYTEST_INVOCATION_LOG"\n'
        "fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = _environment(fake_python)
    environment["PYTEST_INVOCATION_LOG"] = str(invocation_log)
    environment.update(overrides)

    completed = subprocess.run(
        [str(wrapper), "-q", "tests/test_example.py"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = invocation_log.read_text(encoding="utf-8").splitlines()
    assert arguments[:6] == [
        "-p",
        "pytest_timeout",
        f"--timeout={expected_timeout}",
        "--timeout-method=signal",
        "-o",
        f"faulthandler_timeout={expected_faulthandler}",
    ]
    assert arguments[6].startswith("--basetemp=/tmp/dt-pytest-")
    assert arguments[-2:] == ["-q", "tests/test_example.py"]


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("DEXTELEOP_PYTEST_TIMEOUT_SECONDS", "0"),
        ("DEXTELEOP_PYTEST_FAULTHANDLER_TIMEOUT_SECONDS", "not-a-number"),
    ],
)
def test_wrapper_rejects_invalid_watchdog_values(
    tmp_path: Path,
    variable: str,
    value: str,
) -> None:
    _checkout, wrapper = _wrapper_checkout(tmp_path)
    fake_python = tmp_path / "python3.10"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    environment = _environment(fake_python)
    environment[variable] = value

    completed = subprocess.run(
        [str(wrapper), "-q"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert f"{variable} must be a positive number" in completed.stderr


def test_wrapper_terminates_a_hung_test_with_diagnostics(tmp_path: Path) -> None:
    pytest.importorskip("pytest_timeout")
    _checkout, wrapper = _wrapper_checkout(tmp_path)
    probe = tmp_path / "test_hung_probe.py"
    probe.write_text(
        "import time\n\ndef test_hung_probe():\n    time.sleep(30)\n",
        encoding="utf-8",
    )
    environment = _environment(Path(sys.executable))
    environment.update(
        {
            "DEXTELEOP_PYTEST_TIMEOUT_SECONDS": "0.4",
            "DEXTELEOP_PYTEST_FAULTHANDLER_TIMEOUT_SECONDS": "0.2",
        }
    )

    completed = subprocess.run(
        [str(wrapper), "-p", "no:cacheprovider", "-q", str(probe)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "timeout" in output.lower()
    assert "test_hung_probe" in output
