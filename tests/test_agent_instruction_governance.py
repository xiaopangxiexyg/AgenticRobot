from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT_AGENTS = REPOSITORY_ROOT / "AGENTS.md"
GATEWAY_AGENTS = REPOSITORY_ROOT / "agent_gateway" / "AGENTS.md"
REMOTE_AGENTS = REPOSITORY_ROOT / "codex_remote" / "AGENTS.md"
SCOPED_AGENTS = (GATEWAY_AGENTS, REMOTE_AGENTS)
SKILLS_ROOT = REPOSITORY_ROOT / ".agents" / "skills"

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _tracked_skill_names() -> tuple[str, ...]:
    tracked = subprocess.run(
        ["git", "ls-files", "--", ".agents/skills/*/SKILL.md"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return tuple(sorted(Path(path).parent.name for path in tracked))


def test_agent_instruction_chains_stay_well_below_the_default_limit() -> None:
    root_size = len(ROOT_AGENTS.read_bytes())
    assert root_size <= 6 * 1024

    for scoped_file in SCOPED_AGENTS:
        combined_size = root_size + len(scoped_file.read_bytes()) + 2
        assert combined_size <= 10 * 1024, scoped_file

    effective_chains = {
        "gateway-change": (
            ROOT_AGENTS,
            GATEWAY_AGENTS,
            SKILLS_ROOT / "change-agent-gateway" / "SKILL.md",
        ),
        "teleop-motion-diagnosis": (
            ROOT_AGENTS,
            SKILLS_ROOT / "debug-teleop-ik-motion" / "SKILL.md",
            SKILLS_ROOT
            / "debug-teleop-ik-motion"
            / "references"
            / "motion-contracts.md",
            SKILLS_ROOT
            / "debug-teleop-ik-motion"
            / "references"
            / "evidence-and-metrics.md",
        ),
        "local-resource-change": (
            ROOT_AGENTS,
            SKILLS_ROOT / "manage-project-local-resources" / "SKILL.md",
        ),
        "remote-operation": (
            ROOT_AGENTS,
            REMOTE_AGENTS,
            SKILLS_ROOT / "remote-access" / "SKILL.md",
            SKILLS_ROOT / "remote-access" / "references" / "wrapper-catalog.md",
        ),
    }
    for label, files in effective_chains.items():
        size = sum(len(path.read_bytes()) + 2 for path in files)
        assert size <= 16 * 1024, (label, size)


def test_repository_skills_are_named_bounded_and_complete() -> None:
    skill_names = _tracked_skill_names()
    assert skill_names

    ignore_rules = set((REPOSITORY_ROOT / ".gitignore").read_text().splitlines())
    manifest = json.loads(
        (REPOSITORY_ROOT / "docs/setup/local_resources_manifest_v1.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_exceptions = {
        (item["probe"], item["expected_rule"]) for item in manifest["ignore_exceptions"]
    }
    instruction_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT_AGENTS, *SCOPED_AGENTS)
    )

    for name in skill_names:
        skill_root = SKILLS_ROOT / name
        skill_text = (skill_root / "SKILL.md").read_text(encoding="utf-8")
        interface_text = (skill_root / "agents" / "openai.yaml").read_text(
            encoding="utf-8"
        )

        assert len(name) < 64
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
        assert skill_text.startswith(f"---\nname: {name}\n")
        assert len(skill_text.encode("utf-8")) <= 8 * 1024
        assert "[TODO" not in skill_text
        interface_match = re.fullmatch(
            r'interface:\n  display_name: "([^"\n]+)"\n'
            r'  short_description: "([^"\n]+)"\n'
            r'  default_prompt: "([^"\n]+)"\n?',
            interface_text,
        )
        assert interface_match is not None, name
        _, short_description, default_prompt = interface_match.groups()
        assert 25 <= len(short_description) <= 64, name
        assert ("$" + name) in default_prompt

        root_rule = f"!.agents/skills/{name}/"
        content_rule = f"!.agents/skills/{name}/**"
        assert root_rule in ignore_rules, name
        assert content_rule in ignore_rules, name
        assert (f".agents/skills/{name}", root_rule) in manifest_exceptions, name
        assert any(
            probe.startswith(f".agents/skills/{name}/") and rule == content_rule
            for probe, rule in manifest_exceptions
        ), name
        assert ("$" + name) in instruction_text, name

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


def test_remote_wrapper_catalog_covers_public_entrypoints() -> None:
    catalog = (
        SKILLS_ROOT / "remote-access" / "references" / "wrapper-catalog.md"
    ).read_text(encoding="utf-8")
    documented = set(re.findall(r"`(remote_[a-z0-9_]+\.sh)`", catalog))
    wrappers = {
        path.name for path in (REPOSITORY_ROOT / "codex_remote").glob("remote_*.sh")
    }
    uncovered = {
        name
        for name in wrappers
        if name != "remote_common.sh"
        and not name.startswith("remote_dataset_")
        and name not in documented
    }
    assert uncovered == set()
