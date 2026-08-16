from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .workspace import (
    LabError,
    scope_matches,
    validate_repository_name,
    validate_scope,
    validate_task_id,
)

IMPACT_RULES_SCHEMA_ID = "lab.impact-rules.v1"


@dataclass(frozen=True)
class ImpactRule:
    rule_id: str
    paths: dict[str, tuple[str, ...]]
    profiles: tuple[str, ...]


@dataclass(frozen=True)
class ImpactRules:
    rules: tuple[ImpactRule, ...]
    fallback_profiles: dict[str, tuple[str, ...]]
    path: Path
    sha256: str


@dataclass(frozen=True)
class ImpactRecommendation:
    profiles: tuple[str, ...]
    matched_rules: tuple[str, ...]
    fallback_repositories: tuple[str, ...]
    uncovered_paths: dict[str, tuple[str, ...]]
    unknown_profiles: tuple[str, ...]


def _profile_ids(value: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) for item in value)
        or len(value) != len(set(value))
    ):
        raise LabError(f"impact rules require unique Profiles for {label}")
    return tuple(validate_task_id(item) for item in value)


def load_impact_rules(path: Path) -> ImpactRules:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read impact rules {resolved}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "rules",
        "fallback_profiles",
    }:
        raise LabError(f"impact rules have an unexpected shape: {resolved}")
    if payload.get("schema") != IMPACT_RULES_SCHEMA_ID:
        raise LabError(f"unsupported impact rules: {resolved}")
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise LabError(f"impact rules require at least one rule: {resolved}")
    rules: list[ImpactRule] = []
    rule_ids: set[str] = set()
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict) or set(raw_rule) != {
            "rule_id",
            "paths",
            "profiles",
        }:
            raise LabError(f"impact rule has an unexpected shape: {resolved}")
        rule_id = raw_rule.get("rule_id")
        if not isinstance(rule_id, str):
            raise LabError("impact rule has an invalid ID")
        validate_task_id(rule_id)
        if rule_id in rule_ids:
            raise LabError(f"impact rules repeat rule ID: {rule_id}")
        rule_ids.add(rule_id)
        raw_paths = raw_rule.get("paths")
        if not isinstance(raw_paths, dict) or not raw_paths:
            raise LabError(f"impact rule has no paths: {rule_id}")
        paths: dict[str, tuple[str, ...]] = {}
        for repository, patterns in raw_paths.items():
            if not isinstance(repository, str):
                raise LabError(f"impact rule has invalid repository: {rule_id}")
            validate_repository_name(repository)
            if (
                not isinstance(patterns, list)
                or not patterns
                or not all(isinstance(pattern, str) for pattern in patterns)
                or len(patterns) != len(set(patterns))
            ):
                raise LabError(f"impact rule has invalid paths: {rule_id}/{repository}")
            paths[repository] = tuple(validate_scope(pattern) for pattern in patterns)
        rules.append(
            ImpactRule(
                rule_id=rule_id,
                paths=dict(sorted(paths.items())),
                profiles=_profile_ids(
                    raw_rule.get("profiles"),
                    label=f"rule {rule_id}",
                ),
            )
        )
    raw_fallback = payload.get("fallback_profiles")
    if not isinstance(raw_fallback, dict):
        raise LabError("impact rules fallback Profiles must be an object")
    fallback: dict[str, tuple[str, ...]] = {}
    for repository, profiles in raw_fallback.items():
        if not isinstance(repository, str):
            raise LabError("impact rules fallback has an invalid repository")
        validate_repository_name(repository)
        fallback[repository] = _profile_ids(
            profiles,
            label=f"fallback {repository}",
        )
    return ImpactRules(
        rules=tuple(rules),
        fallback_profiles=dict(sorted(fallback.items())),
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def recommend_profiles(
    rules: ImpactRules,
    changed_paths: dict[str, tuple[str, ...]],
    *,
    known_profiles: set[str],
) -> ImpactRecommendation:
    selected: set[str] = set()
    matched_rules: set[str] = set()
    fallback_repositories: set[str] = set()
    uncovered: dict[str, list[str]] = {}
    for repository, paths in sorted(changed_paths.items()):
        validate_repository_name(repository)
        for path in paths:
            validate_scope(path)
            matched = False
            for rule in rules.rules:
                patterns = rule.paths.get(repository)
                if patterns is not None and scope_matches(path, patterns):
                    matched = True
                    matched_rules.add(rule.rule_id)
                    selected.update(rule.profiles)
            if matched:
                continue
            fallback = rules.fallback_profiles.get(repository)
            if fallback is None:
                uncovered.setdefault(repository, []).append(path)
            else:
                fallback_repositories.add(repository)
                selected.update(fallback)
    unknown = tuple(sorted(selected - known_profiles))
    return ImpactRecommendation(
        profiles=tuple(sorted(selected & known_profiles)),
        matched_rules=tuple(sorted(matched_rules)),
        fallback_repositories=tuple(sorted(fallback_repositories)),
        uncovered_paths={
            repository: tuple(sorted(paths))
            for repository, paths in sorted(uncovered.items())
        },
        unknown_profiles=unknown,
    )
