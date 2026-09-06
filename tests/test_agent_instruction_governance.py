from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT_AGENTS = REPOSITORY_ROOT / "AGENTS.md"
SCOPED_AGENTS = (
    REPOSITORY_ROOT / "agent_gateway" / "AGENTS.md",
    REPOSITORY_ROOT / "codex_remote" / "AGENTS.md",
)
REPOSITORY_SKILLS = (
    "change-agent-gateway",
    "debug-teleop-ik-motion",
    "manage-project-local-resources",
    "remote-access",
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_agent_instruction_chains_stay_well_below_the_default_limit() -> None:
    root_size = len(ROOT_AGENTS.read_bytes())
    assert root_size <= 6 * 1024

    for scoped_file in SCOPED_AGENTS:
        combined_size = root_size + len(scoped_file.read_bytes()) + 2
        assert combined_size <= 10 * 1024, scoped_file


def test_repository_skills_are_named_bounded_and_complete() -> None:
    skills_root = REPOSITORY_ROOT / ".agents" / "skills"

    for name in REPOSITORY_SKILLS:
        skill_root = skills_root / name
        skill_text = (skill_root / "SKILL.md").read_text(encoding="utf-8")
        interface_text = (skill_root / "agents" / "openai.yaml").read_text(
            encoding="utf-8"
        )

        assert skill_text.startswith(f"---\nname: {name}\n")
        assert len(skill_text.encode("utf-8")) <= 8 * 1024
        assert "[TODO" not in skill_text
        assert ("$" + name) in interface_text

        local_links = re.findall(r"\[[^]]+\]\(([^)]+)\)", skill_text)
        broken = [
            link
            for link in local_links
            if not link.startswith(("http://", "https://", "#"))
            and not (skill_root / link).resolve().exists()
        ]
        assert broken == [], name

        for reference in (skill_root / "references").glob("*.md"):
            assert len(reference.read_bytes()) <= 8 * 1024
            assert "[TODO" not in reference.read_text(encoding="utf-8")

    tracked = subprocess.run(
        ["git", "ls-files", "--", ".agents/skills/*/SKILL.md"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tracked_names = {Path(path).parent.name for path in tracked}
    assert tracked_names == set(REPOSITORY_SKILLS)


def test_instruction_routes_resolve_to_versioned_sources() -> None:
    expected_paths = (
        "README_ENV.md",
        "docs/architecture/runtime_domain_boundaries.md",
        "docs/setup/LOCAL_RESOURCES.md",
        "docs/setup/local_resources_manifest_v1.json",
        "agent_gateway/README.md",
        "codex_remote/README.md",
        "scripts/dev_local_resources.py",
        "scripts/run_pytest.sh",
    )

    missing = [path for path in expected_paths if not (REPOSITORY_ROOT / path).exists()]
    assert missing == []
