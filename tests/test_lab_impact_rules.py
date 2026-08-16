from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_control.impact_rules import load_impact_rules, recommend_profiles
from lab_control.workspace import LabError


def _write_rules(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "lab.impact-rules.v1",
                "rules": [
                    {
                        "rule_id": "control-plane",
                        "paths": {"AgenticRobot": ["lab_control/**"]},
                        "profiles": ["lab-control-unit"],
                    }
                ],
                "fallback_profiles": {
                    "AgenticRobot": ["broad-regression"],
                },
            }
        ),
        encoding="utf-8",
    )


def test_path_rules_and_unknown_path_fallback_are_both_applied(
    tmp_path: Path,
) -> None:
    path = tmp_path / "impact-rules.json"
    _write_rules(path)
    rules = load_impact_rules(path)

    recommendation = recommend_profiles(
        rules,
        {
            "AgenticRobot": (
                "lab_control/inspection.py",
                "unclassified/runtime.py",
            )
        },
        known_profiles={"lab-control-unit", "broad-regression"},
    )

    assert recommendation.profiles == ("broad-regression", "lab-control-unit")
    assert recommendation.matched_rules == ("control-plane",)
    assert recommendation.fallback_repositories == ("AgenticRobot",)
    assert recommendation.uncovered_paths == {}


def test_impact_rules_report_unknown_profiles_without_authorizing_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "impact-rules.json"
    _write_rules(path)
    rules = load_impact_rules(path)

    recommendation = recommend_profiles(
        rules,
        {"AgenticRobot": ("lab_control/inspection.py",)},
        known_profiles=set(),
    )

    assert recommendation.profiles == ()
    assert recommendation.unknown_profiles == ("lab-control-unit",)


def test_impact_rules_reject_traversal_patterns(tmp_path: Path) -> None:
    path = tmp_path / "impact-rules.json"
    _write_rules(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rules"][0]["paths"]["AgenticRobot"] = ["../outside/**"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabError, match="unsafe write scope"):
        load_impact_rules(path)


def test_checked_in_impact_rules_reference_existing_profiles() -> None:
    repository = Path(__file__).resolve().parents[1]
    rules = load_impact_rules(repository / "stack" / "impact-rules.json")
    profiles = {
        path.stem for path in (repository / "integration" / "profiles").glob("*.json")
    }
    referenced = {
        profile
        for rule in rules.rules
        for profile in rule.profiles
    } | {
        profile
        for fallback in rules.fallback_profiles.values()
        for profile in fallback
    }

    assert referenced <= profiles
    lab_change = recommend_profiles(
        rules,
        {"AgenticRobot": ("lab_control/inspection.py",)},
        known_profiles=profiles,
    )
    assert lab_change.profiles == ("lab-control-unit",)
    assert lab_change.matched_rules == ("lab-control",)

    unknown_change = recommend_profiles(
        rules,
        {"AgenticRobot": ("README.md",)},
        known_profiles=profiles,
    )
    assert unknown_change.profiles == (
        "agenticrobot-rpent-contract",
        "lab-control-unit",
    )
    assert unknown_change.fallback_repositories == ("AgenticRobot",)
