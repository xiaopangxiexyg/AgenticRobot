from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
import subprocess

import pytest
from setuptools import find_packages

from tests.support.repository_imports import absolute_import_modules


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE_PATH = "hardware/reBotArm_control_py"
SUBMODULE_URL = "https://github.com/Seeed-Projects/reBotArm_control_py.git"
SUBMODULE_COMMIT = "5ba28acef46237eb6a7560658bbc43b06cf8a259"

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _submodule_section() -> tuple[ConfigParser, str]:
    parser = ConfigParser()
    parser.read(REPO_ROOT / ".gitmodules", encoding="utf-8")
    matches = [
        section
        for section in parser.sections()
        if parser.get(section, "path", fallback="") == SUBMODULE_PATH
    ]
    assert len(matches) == 1
    return parser, matches[0]


def _parent_python_files() -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        REPO_ROOT / relative
        for relative in result.stdout.split("\0")
        if relative.strip()
    )


def test_rebot_source_boundary_is_one_exact_non_floating_gitlink() -> None:
    parser, section = _submodule_section()

    assert parser.get(section, "url") == SUBMODULE_URL
    assert not parser.has_option(section, "branch")

    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", SUBMODULE_PATH],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata, path = result.stdout.strip().split("\t", 1)
    mode, object_id, stage = metadata.split()
    assert (mode, object_id, stage, path) == (
        "160000",
        SUBMODULE_COMMIT,
        "0",
        SUBMODULE_PATH,
    )


def test_rebot_vendor_tree_is_not_part_of_parent_package_discovery() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package_find_section = pyproject.split(
        "[tool.setuptools.packages.find]", maxsplit=1
    )[1].split("\n[", maxsplit=1)[0]

    assert '"hardware.reBotArm_control_py*"' in package_find_section
    assert not any(
        package == "hardware.reBotArm_control_py"
        or package.startswith("hardware.reBotArm_control_py.")
        for package in find_packages(where=str(REPO_ROOT))
    )


@pytest.mark.slow_sim
def test_m0_adds_no_parent_runtime_import_of_rebot_vendor_code() -> None:
    forbidden = {"reBotArm_control_py", "hardware.reBotArm_control_py"}
    violations: dict[str, list[str]] = {}

    for path in _parent_python_files():
        # A forbidden import must contain one of these exact module names in
        # its source. Keep the AST helper authoritative for candidates, while
        # avoiding a full parse of thousands of unrelated parent files.
        source = path.read_text(encoding="utf-8")
        if not any(root in source for root in forbidden):
            continue
        imports = absolute_import_modules(path)
        matched = sorted(
            module
            for module in imports
            if any(
                module == root or module.startswith(f"{root}.")
                for root in forbidden
            )
        )
        if matched:
            violations[path.relative_to(REPO_ROOT).as_posix()] = matched

    assert violations == {}
