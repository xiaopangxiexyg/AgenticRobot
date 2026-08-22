#!/usr/bin/env python3
"""Audit and plan worktree-local resource handling without moving any data.

The versioned manifest classifies ignored datasets, records, model artifacts,
secrets, generated outputs, caches, and developer-local state.  This tool only
inspects path metadata and Git ignore/tracking contracts.  It never opens or
traverses resource directory contents and never copies, links, edits, or
deletes local content; top-level symlink targets are resolved only for status.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "docs" / "setup" / "local_resources_manifest_v1.json"
SCHEMA_VERSION = "dexteleop.development_local_resources.v1"
CLASSIFICATIONS = {
    "shared-read-mostly",
    "external-input",
    "mutable-evidence",
    "local-secret",
    "generated",
    "developer-private",
    "local-vendor",
    "local-archive",
}
MIGRATION_MODES = {
    "asset-store-link",
    "external-binding",
    "verified-copy",
    "recreate",
    "rebuild",
    "do-not-migrate",
    "manual-review",
}
TRACKED_POLICIES = {"none", "mixed"}


class LocalResourceError(RuntimeError):
    """Raised when the local-resource policy or repository contract is invalid."""


@dataclass(frozen=True)
class IgnoreMatch:
    source: str
    line: int
    rule: str

    @property
    def ignored(self) -> bool:
        return not self.rule.startswith("!")


@dataclass(frozen=True)
class ResourceLocation:
    label: str
    path: str | None
    probe: str
    expected_rule: str
    ignore_source: str
    tracked_policy: str | None


@dataclass(frozen=True)
class ResourcePolicy:
    resource_id: str
    classification: str
    migration: str
    summary: str
    locations: tuple[ResourceLocation, ...]


@dataclass(frozen=True)
class IgnoreException:
    label: str
    probe: str
    expected_rule: str | None


def _safe_relative(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise LocalResourceError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\x00" in value:
        raise LocalResourceError(f"{field} must stay inside the repository: {value!r}")
    return value


def _load_location(raw: Any, *, resource_id: str) -> ResourceLocation:
    if not isinstance(raw, dict):
        raise LocalResourceError(f"resource {resource_id!r} location must be an object")
    label = raw.get("label")
    path = raw.get("path")
    probe = raw.get("probe")
    expected_rule = raw.get("expected_rule")
    ignore_source = raw.get("ignore_source", ".gitignore")
    tracked_policy = raw.get("tracked_policy")
    if not isinstance(label, str) or not label.strip():
        raise LocalResourceError(f"resource {resource_id!r} location label is invalid")
    if path is not None:
        path = _safe_relative(path, field=f"resource {resource_id!r} path")
        if tracked_policy not in TRACKED_POLICIES:
            raise LocalResourceError(
                f"resource {resource_id!r} path {path!r} needs tracked_policy "
                f"in {sorted(TRACKED_POLICIES)}"
            )
    elif tracked_policy is not None:
        raise LocalResourceError(
            f"resource {resource_id!r} pattern-only location cannot set tracked_policy"
        )
    probe = _safe_relative(probe, field=f"resource {resource_id!r} probe")
    if not isinstance(expected_rule, str) or not expected_rule:
        raise LocalResourceError(
            f"resource {resource_id!r} location {label!r} needs expected_rule"
        )
    if expected_rule.startswith("!"):
        raise LocalResourceError(
            f"resource {resource_id!r} location {label!r} must be ignored, not exempted"
        )
    ignore_source = _safe_relative(
        ignore_source,
        field=f"resource {resource_id!r} ignore_source",
    )
    if PurePosixPath(ignore_source).name != ".gitignore":
        raise LocalResourceError(
            f"resource {resource_id!r} ignore_source must name a .gitignore file"
        )
    return ResourceLocation(
        label=label,
        path=path,
        probe=probe,
        expected_rule=expected_rule,
        ignore_source=ignore_source,
        tracked_policy=tracked_policy,
    )


def load_manifest(
    path: Path = MANIFEST_PATH,
) -> tuple[tuple[ResourcePolicy, ...], tuple[IgnoreException, ...]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalResourceError(f"cannot load local-resource manifest: {exc}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise LocalResourceError("unsupported local-resource manifest schema")
    raw_resources = payload.get("resources")
    if not isinstance(raw_resources, list) or not raw_resources:
        raise LocalResourceError("local-resource manifest contains no resources")

    resources: list[ResourcePolicy] = []
    resource_ids: set[str] = set()
    probes: set[str] = set()
    for raw in raw_resources:
        if not isinstance(raw, dict):
            raise LocalResourceError("local-resource entry must be an object")
        resource_id = raw.get("resource_id")
        classification = raw.get("classification")
        migration = raw.get("migration")
        summary = raw.get("summary")
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise LocalResourceError("resource_id must be non-empty")
        if resource_id in resource_ids:
            raise LocalResourceError(f"duplicate resource_id: {resource_id}")
        if classification not in CLASSIFICATIONS:
            raise LocalResourceError(
                f"resource {resource_id!r} has invalid classification {classification!r}"
            )
        if migration not in MIGRATION_MODES:
            raise LocalResourceError(
                f"resource {resource_id!r} has invalid migration mode {migration!r}"
            )
        if not isinstance(summary, str) or not summary.strip():
            raise LocalResourceError(f"resource {resource_id!r} summary is invalid")
        raw_locations = raw.get("locations")
        if not isinstance(raw_locations, list) or not raw_locations:
            raise LocalResourceError(f"resource {resource_id!r} has no locations")
        locations = tuple(
            _load_location(item, resource_id=resource_id) for item in raw_locations
        )
        for location in locations:
            if location.probe in probes:
                raise LocalResourceError(
                    f"duplicate local-resource ignore probe: {location.probe}"
                )
            probes.add(location.probe)
        resources.append(
            ResourcePolicy(
                resource_id=resource_id,
                classification=classification,
                migration=migration,
                summary=summary,
                locations=locations,
            )
        )
        resource_ids.add(resource_id)

    raw_exceptions = payload.get("ignore_exceptions")
    if not isinstance(raw_exceptions, list) or not raw_exceptions:
        raise LocalResourceError(
            "local-resource manifest contains no ignore exceptions"
        )
    exceptions: list[IgnoreException] = []
    for raw in raw_exceptions:
        if not isinstance(raw, dict):
            raise LocalResourceError("ignore exception must be an object")
        label = raw.get("label")
        probe = raw.get("probe")
        expected_rule = raw.get("expected_rule")
        if not isinstance(label, str) or not label.strip():
            raise LocalResourceError("ignore exception label is invalid")
        probe = _safe_relative(probe, field=f"ignore exception {label!r} probe")
        if expected_rule is not None and (
            not isinstance(expected_rule, str) or not expected_rule.startswith("!")
        ):
            raise LocalResourceError(
                f"ignore exception {label!r} expected_rule must be null or a negation"
            )
        if probe in probes:
            raise LocalResourceError(f"duplicate local-resource ignore probe: {probe}")
        probes.add(probe)
        exceptions.append(
            IgnoreException(label=label, probe=probe, expected_rule=expected_rule)
        )
    return tuple(resources), tuple(exceptions)


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _require_repository(repo_root: Path) -> Path:
    candidate = repo_root.expanduser().resolve()
    completed = _run_git(candidate, "rev-parse", "--show-toplevel")
    if completed.returncode != 0:
        raise LocalResourceError(
            completed.stderr.strip() or f"not a Git worktree: {candidate}"
        )
    top_level = Path(completed.stdout.strip()).resolve()
    if top_level != candidate:
        raise LocalResourceError(
            f"--repo-root must be the worktree root: expected {top_level}, got {candidate}"
        )
    return candidate


def _ignore_match(repo_root: Path, probe: str) -> IgnoreMatch | None:
    completed = _run_git(repo_root, "check-ignore", "-v", "--no-index", "--", probe)
    if completed.returncode == 1:
        return None
    if completed.returncode != 0:
        raise LocalResourceError(
            completed.stderr.strip() or f"cannot check ignore rule for {probe!r}"
        )
    output = completed.stdout.rstrip("\n")
    metadata, separator, _matched_path = output.rpartition("\t")
    if not separator:
        raise LocalResourceError(f"unexpected git check-ignore output for {probe!r}")
    source, line_text, rule = metadata.rsplit(":", 2)
    try:
        line = int(line_text)
    except ValueError as exc:
        raise LocalResourceError(
            f"unexpected git check-ignore line for {probe!r}: {line_text!r}"
        ) from exc
    return IgnoreMatch(source=source, line=line, rule=rule)


def _tracked_paths(repo_root: Path, relative_path: str) -> tuple[str, ...]:
    completed = _run_git(repo_root, "ls-files", "-z", "--", relative_path)
    if completed.returncode != 0:
        raise LocalResourceError(
            completed.stderr.strip() or f"cannot inspect tracked path {relative_path!r}"
        )
    return tuple(item for item in completed.stdout.split("\x00") if item)


def _path_status(root: Path, relative_path: str) -> tuple[bool, str, str | None]:
    path = root / relative_path
    if not os.path.lexists(path):
        return False, "missing", None
    if path.is_symlink():
        try:
            target = str(path.resolve(strict=True))
            return True, "symlink", target
        except OSError as exc:
            return True, "broken-symlink", str(exc)
    if path.is_dir():
        return True, "directory", None
    if path.is_file():
        return True, "file", None
    return True, "special", None


def _gitignore_rules(
    repo_root: Path,
    sources: set[str],
) -> tuple[tuple[str, str], ...]:
    rules: list[tuple[str, str]] = []
    for source in sorted(sources):
        path = repo_root / source
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise LocalResourceError(f"cannot read {path}: {exc}") from exc
        rules.extend(
            (source, line.strip())
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        )
    return tuple(rules)


def _rule_label(source: str, rule: str) -> str:
    return rule if source == ".gitignore" else f"{source}:{rule}"


def audit_resources(
    repo_root: Path,
    *,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    root = _require_repository(repo_root)
    resources, exceptions = load_manifest(manifest_path)
    manifest_rules = {
        (location.ignore_source, location.expected_rule)
        for resource in resources
        for location in resource.locations
    } | {
        (".gitignore", exception.expected_rule)
        for exception in exceptions
        if exception.expected_rule is not None
    }
    gitignore_rules = _gitignore_rules(root, {source for source, _ in manifest_rules})
    rule_counts = Counter(gitignore_rules)
    duplicate_rules = sorted(rule for rule, count in rule_counts.items() if count > 1)
    unregistered_rules = sorted(set(gitignore_rules) - manifest_rules)
    missing_rules = sorted(manifest_rules - set(gitignore_rules))
    inventory_issues = [
        *(
            f"unregistered .gitignore rule: {_rule_label(source, rule)}"
            for source, rule in unregistered_rules
        ),
        *(
            f"manifest rule missing from .gitignore: {_rule_label(source, rule)}"
            for source, rule in missing_rules
        ),
        *(
            f"duplicate .gitignore rule: {_rule_label(source, rule)}"
            for source, rule in duplicate_rules
        ),
    ]
    reports: list[dict[str, Any]] = []
    issue_count = len(inventory_issues)
    for resource in resources:
        location_reports: list[dict[str, Any]] = []
        for location in resource.locations:
            issues: list[str] = []
            match = _ignore_match(root, location.probe)
            if match is None:
                issues.append("ignore probe is not matched")
            elif match.source != location.ignore_source:
                issues.append(
                    f"ignore probe is owned by {match.source!r}, expected "
                    f"{location.ignore_source!r}"
                )
            elif not match.ignored:
                issues.append(f"ignore probe is exempted by {match.rule!r}")
            elif match.rule != location.expected_rule:
                issues.append(
                    f"ignore rule drift: expected {location.expected_rule!r}, got {match.rule!r}"
                )

            present = None
            path_kind = None
            symlink_target = None
            tracked_paths: tuple[str, ...] = ()
            if location.path is not None:
                present, path_kind, symlink_target = _path_status(root, location.path)
                tracked_paths = _tracked_paths(root, location.path)
                if location.tracked_policy == "none" and tracked_paths:
                    issues.append(
                        f"expected no tracked content under {location.path!r}, found "
                        f"{len(tracked_paths)} path(s)"
                    )
                if location.tracked_policy == "mixed" and not tracked_paths:
                    issues.append(
                        f"expected tracked exceptions under mixed root {location.path!r}"
                    )
            issue_count += len(issues)
            location_reports.append(
                {
                    "label": location.label,
                    "path": location.path,
                    "probe": location.probe,
                    "expected_rule": location.expected_rule,
                    "ignore_source": location.ignore_source,
                    "matched_rule": asdict(match) if match is not None else None,
                    "present": present,
                    "path_kind": path_kind,
                    "symlink_target": symlink_target,
                    "tracked_policy": location.tracked_policy,
                    "tracked_count": len(tracked_paths),
                    "issues": issues,
                }
            )
        reports.append(
            {
                "resource_id": resource.resource_id,
                "classification": resource.classification,
                "migration": resource.migration,
                "summary": resource.summary,
                "locations": location_reports,
            }
        )

    exception_reports: list[dict[str, Any]] = []
    for exception in exceptions:
        issues: list[str] = []
        match = _ignore_match(root, exception.probe)
        if exception.expected_rule is None:
            if match is not None and match.ignored:
                issues.append(f"path is unexpectedly ignored by {match.rule!r}")
        elif match is None:
            issues.append("expected ignore exemption is not matched")
        elif match.source != ".gitignore":
            issues.append(
                f"ignore exemption is owned by {match.source!r}, expected root .gitignore"
            )
        elif match.rule != exception.expected_rule:
            issues.append(
                f"ignore exemption drift: expected {exception.expected_rule!r}, "
                f"got {match.rule!r}"
            )
        elif match.ignored:
            issues.append(f"expected an exemption, got ignore rule {match.rule!r}")
        issue_count += len(issues)
        exception_reports.append(
            {
                "label": exception.label,
                "probe": exception.probe,
                "expected_rule": exception.expected_rule,
                "matched_rule": asdict(match) if match is not None else None,
                "issues": issues,
            }
        )

    return {
        "schema": "dexteleop.development_local_resource_audit.v1",
        "ok": issue_count == 0,
        "repo_root": str(root),
        "manifest_path": str(manifest_path.resolve()),
        "issue_count": issue_count,
        "gitignore_inventory": {
            "rule_count": len(gitignore_rules),
            "registered_rule_count": len(manifest_rules),
            "issues": inventory_issues,
        },
        "resources": reports,
        "ignore_exceptions": exception_reports,
    }


def _migration_recommendation(resource: ResourcePolicy) -> str:
    return {
        "asset-store-link": (
            "Use dev_asset_store.py; its ignored assets link is the only automatic "
            "worktree symlink allowed by this policy."
        ),
        "external-binding": (
            "Keep the canonical data outside worktrees and bind it through local_paths.json, "
            "an environment variable, or the owning runtime profile."
        ),
        "verified-copy": (
            "Stop writers, inventory the source, copy to a new destination, verify hashes and "
            "counts, then switch the explicit binding. Never share a live writable root."
        ),
        "recreate": (
            "Recreate from a tracked example or machine-specific secret manager; do not copy "
            "credentials into a worktree or migration receipt."
        ),
        "rebuild": "Rebuild in the destination; do not migrate this generated state.",
        "do-not-migrate": "Keep this developer-local state local and recreate it if needed.",
        "manual-review": (
            "Review tracked exceptions and provenance before moving individual ignored children; "
            "never link or replace the whole mixed root."
        ),
    }[resource.migration]


def build_migration_plan(
    source_worktree: Path,
    target_worktree: Path,
    *,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    source = source_worktree.expanduser().resolve()
    target = target_worktree.expanduser().resolve()
    if source == target:
        raise LocalResourceError("source and target worktrees must differ")
    resources, _exceptions = load_manifest(manifest_path)
    items: list[dict[str, Any]] = []
    for resource in resources:
        for location in resource.locations:
            if location.path is None:
                continue
            source_present, source_kind, source_link = _path_status(
                source, location.path
            )
            target_present, target_kind, target_link = _path_status(
                target, location.path
            )
            items.append(
                {
                    "resource_id": resource.resource_id,
                    "location": location.path,
                    "classification": resource.classification,
                    "migration": resource.migration,
                    "automatic": resource.migration == "asset-store-link",
                    "source_present": source_present,
                    "source_kind": source_kind,
                    "source_symlink_target": source_link,
                    "target_present": target_present,
                    "target_kind": target_kind,
                    "target_symlink_target": target_link,
                    "collision_review_required": bool(
                        source_present and target_present
                    ),
                    "recommendation": _migration_recommendation(resource),
                }
            )
    return {
        "schema": "dexteleop.development_local_resource_migration_plan.v1",
        "read_only": True,
        "source_worktree": str(source),
        "target_worktree": str(target),
        "items": items,
    }


def _print_audit(payload: dict[str, Any]) -> None:
    print(f"Local-resource policy: {'OK' if payload['ok'] else 'FAILED'}")
    print(f"Repository: {payload['repo_root']}")
    print(f"Manifest: {payload['manifest_path']}")
    for issue in payload["gitignore_inventory"]["issues"]:
        print(f"- .gitignore inventory: {issue}")
    for resource in payload["resources"]:
        present = sum(item["present"] is True for item in resource["locations"])
        issues = sum(len(item["issues"]) for item in resource["locations"])
        print(
            f"- {resource['resource_id']}: class={resource['classification']} "
            f"migration={resource['migration']} present={present}/"
            f"{len(resource['locations'])} issues={issues}"
        )
        for location in resource["locations"]:
            for issue in location["issues"]:
                print(f"    {location['label']}: {issue}")
    for exception in payload["ignore_exceptions"]:
        for issue in exception["issues"]:
            print(f"- ignore exception {exception['label']}: {issue}")


def _print_plan(payload: dict[str, Any]) -> None:
    print("Read-only local-resource migration plan")
    print(f"Source: {payload['source_worktree']}")
    print(f"Target: {payload['target_worktree']}")
    for item in payload["items"]:
        if not item["source_present"]:
            continue
        collision = " collision-review" if item["collision_review_required"] else ""
        print(
            f"- {item['location']}: {item['migration']}"
            f" ({item['source_kind']} -> {item['target_kind']}){collision}"
        )
        print(f"    {item['recommendation']}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit", help="validate ignore policy and inspect path metadata"
    )
    audit.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    audit.add_argument("--json", action="store_true")

    plan = subparsers.add_parser(
        "plan", help="produce a read-only source-to-target migration plan"
    )
    plan.add_argument("--source-worktree", type=Path, required=True)
    plan.add_argument("--target-worktree", type=Path, required=True)
    plan.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            payload = audit_resources(args.repo_root, manifest_path=args.manifest)
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_audit(payload)
            return 0 if payload["ok"] else 1
        payload = build_migration_plan(
            args.source_worktree,
            args.target_worktree,
            manifest_path=args.manifest,
        )
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_plan(payload)
        return 0
    except LocalResourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
