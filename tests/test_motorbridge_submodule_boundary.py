from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE_PATH = "hardware/MotorBridge"
SUBMODULE_URL = "git@github.com:CYBER-QAQ/motorbridge.git"
SUBMODULE_COMMIT = "1e479d280f29dd88d6a7e4270aeab4342d682bde"

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_motorbridge_is_one_exact_non_floating_gitlink() -> None:
    parser = ConfigParser()
    parser.read(REPO_ROOT / ".gitmodules", encoding="utf-8")
    sections = [
        section
        for section in parser.sections()
        if parser.get(section, "path", fallback="") == SUBMODULE_PATH
    ]

    assert len(sections) == 1
    section = sections[0]
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
