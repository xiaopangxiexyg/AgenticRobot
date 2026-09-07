"""Report concurrent Git worktrees and coordination metadata without writes."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any

from worktree_coordination import ACTIVE_STATES, IMMUTABLE_STATES, parse_description


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def parse_worktrees(output: str) -> list[dict[str, str]]:
    worktrees: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in output.splitlines():
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    if current:
        worktrees.append(current)
    return worktrees


def status_paths(path: Path) -> tuple[list[str], list[str]]:
    result = git(path, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    records = [record for record in result.stdout.split("\0") if record]
    lines: list[str] = []
    paths: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        code = record[:2]
        value = record[3:]
        lines.append(f"{code} {value}")
        paths.append(value)
        if "R" in code or "C" in code:
            index += 1
            if index < len(records):
                paths.append(records[index])
        index += 1
    return lines, paths


def resolve_commit(repo: Path, value: str) -> str | None:
    result = git(
        repo,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{value}^{{commit}}",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def committed_paths(repo: Path, base: str, head: str) -> list[str]:
    result = git(repo, "diff", "--name-only", f"{base}...{head}", "--")
    return [line for line in result.stdout.splitlines() if line]


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = git(
        repo,
        "merge-base",
        "--is-ancestor",
        ancestor,
        descendant,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.returncode == 0


def path_matches_scope(path: str, pattern: str) -> bool:
    normalized = pattern.strip().removeprefix("./")
    if normalized.lower() in {"none", "n/a"}:
        return False
    prefix = normalized.rstrip("/")
    if not any(character in normalized for character in "*?["):
        return path == prefix or path.startswith(prefix + "/")

    path_parts = tuple(path.split("/"))
    pattern_parts = tuple(normalized.split("/"))

    @cache
    def matches(pattern_index: int, path_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        component = pattern_parts[pattern_index]
        if component == "**":
            return matches(pattern_index + 1, path_index) or (
                path_index < len(path_parts) and matches(pattern_index, path_index + 1)
            )
        return (
            path_index < len(path_parts)
            and fnmatch.fnmatchcase(path_parts[path_index], component)
            and matches(pattern_index + 1, path_index + 1)
        )

    return matches(0, 0)


def inspect(
    repo: Path,
    *,
    branches: list[str] | None = None,
    mainlines: tuple[str, ...] = ("main", "master"),
) -> dict[str, Any]:
    root = Path(git(repo, "rev-parse", "--show-toplevel").stdout.strip())
    current_main_sha = resolve_commit(root, "main")
    target_refs_before: dict[str, str | None] = {}
    entries = parse_worktrees(git(root, "worktree", "list", "--porcelain").stdout)
    inspected: list[dict[str, Any]] = []
    selected = set(branches or [])
    present = {entry.get("branch", "").removeprefix("refs/heads/") for entry in entries}
    missing = selected - present
    if missing:
        raise ValueError(
            "selected branches have no worktree: " + ", ".join(sorted(missing))
        )

    for entry in entries:
        path = Path(entry["worktree"])
        branch_ref = entry.get("branch")
        branch = branch_ref.removeprefix("refs/heads/") if branch_ref else None
        if selected and branch not in selected:
            continue
        warnings: list[str] = []
        status: list[str] = []
        dirty_paths: list[str] = []
        head = entry["HEAD"]
        if path.exists():
            try:
                status, dirty_paths = status_paths(path)
                observed_head = git(
                    path, "rev-parse", "--verify", "HEAD"
                ).stdout.strip()
                if observed_head != head:
                    warnings.append("worktree HEAD moved during inventory")
                    head = observed_head
            except subprocess.CalledProcessError as error:
                warnings.append("cannot inspect status: " + error.stderr.strip())
        else:
            warnings.append("worktree path is missing")

        description = ""
        is_mainline = branch in mainlines
        if branch and not is_mainline:
            result = git(
                root,
                "config",
                "--get",
                f"branch.{branch}.description",
                check=False,
            )
            description = result.stdout.rstrip("\n")
        fields, description_errors = parse_description(description)
        if is_mainline:
            fields = {"owner": ["mainline"], "state": ["mainline"]}
            description_errors = []
            if status:
                warnings.append("mainline worktree has changes")
        elif not branch:
            description_errors = ["detached worktree has no branch coordination record"]
        warnings.extend(description_errors)
        state = fields.get("state", [None])[0]
        is_integration_branch = bool(branch and branch.startswith("integration/"))

        base_sha: str | None = None
        target_ref: str | None = None
        target_sha: str | None = None
        changed: list[str] = []
        base_values = fields.get("base", [])
        if base_values:
            base_value = base_values[0]
            base_sha = resolve_commit(root, base_values[0])
            if not base_sha:
                warnings.append(f"base does not resolve: {base_values[0]}")
            else:
                if not re.fullmatch(r"[0-9a-fA-F]{40,64}", base_value):
                    warnings.append("base must be recorded as a full commit SHA")
                try:
                    changed = committed_paths(root, base_sha, head)
                except subprocess.CalledProcessError as error:
                    warnings.append("cannot diff base to head: " + error.stderr.strip())
        if is_integration_branch:
            declared_target_refs = fields.get("target-ref", [])
            if declared_target_refs:
                target_ref = declared_target_refs[0]
            elif (
                base_sha
                and current_main_sha
                and is_ancestor(root, base_sha, current_main_sha)
            ):
                # Keep legacy integration records usable when their base was
                # pinned from main. Alternate lines must declare target-ref.
                target_ref = "main"
            elif base_sha:
                warnings.append(
                    "integration target ref is not declared; cannot verify "
                    "target movement"
                )
            if target_ref:
                target_sha = resolve_commit(root, target_ref)
                if not target_sha:
                    warnings.append(
                        f"integration target ref does not resolve: {target_ref}"
                    )
                else:
                    target_refs_before.setdefault(target_ref, target_sha)
            if (
                is_integration_branch
                and base_sha
                and target_sha
                and target_sha != base_sha
            ):
                if target_ref == "main":
                    warnings.append("main moved after the integration base was pinned")
                elif target_ref:
                    warnings.append(
                        "integration target moved after the integration base was "
                        f"pinned: {target_ref}"
                    )

        all_changed = sorted(set(changed) | set(dirty_paths))
        scopes = fields.get("write-scope", [])
        out_of_scope = [
            path_value
            for path_value in all_changed
            if scopes
            and not any(path_matches_scope(path_value, scope) for scope in scopes)
        ]
        if out_of_scope:
            warnings.append(f"{len(out_of_scope)} changed path(s) outside write-scope")

        used_compatibility_tip = "tip" not in fields and "candidate" in fields
        tip_values = fields.get("tip", fields.get("candidate", []))
        tip_sha: str | None = None
        if tip_values and tip_values[0].casefold() != "pending":
            tip_value = tip_values[0]
            tip_sha = resolve_commit(root, tip_value)
            if not tip_sha:
                warnings.append(f"tip does not resolve: {tip_value}")
            elif not re.fullmatch(r"[0-9a-fA-F]{40,64}", tip_value):
                warnings.append("tip must be recorded as a full commit SHA")
        if state in IMMUTABLE_STATES:
            if not tip_sha:
                warnings.append("immutable state lacks a pinned tip SHA")
            elif tip_sha != head:
                warnings.append("pinned tip no longer matches worktree HEAD")
            if base_sha and tip_sha and not is_ancestor(root, base_sha, tip_sha):
                warnings.append("pinned base is not an ancestor of the immutable tip")
            if status:
                warnings.append("immutable-state branch has working-tree changes")

        inputs: list[dict[str, str | None]] = []
        for input_value in fields.get("input", []):
            parts = input_value.rsplit(maxsplit=1)
            if len(parts) != 2:
                warnings.append("input must be '<branch-or-ref> <full-commit-sha>'")
                continue
            input_ref, input_value_sha = parts
            input_sha = resolve_commit(root, input_value_sha)
            current_input_sha = resolve_commit(root, input_ref)
            if not input_sha:
                warnings.append(f"input SHA does not resolve: {input_value_sha}")
            elif not re.fullmatch(r"[0-9a-fA-F]{40,64}", input_value_sha):
                warnings.append("input must use a full commit SHA")
            if not current_input_sha:
                warnings.append(f"input ref does not resolve: {input_ref}")
            elif input_sha and current_input_sha != input_sha:
                warnings.append(f"input ref moved after pin: {input_ref}")
            inputs.append(
                {"ref": input_ref, "pinned": input_sha, "current": current_input_sha}
            )
        if (
            state in {"integrating", "validated-for-mainline"}
            and not inputs
            and not used_compatibility_tip
        ):
            warnings.append("integration state lacks ordered input records")
        if state == "validated-for-mainline" and tip_sha:
            for input_record in inputs:
                input_sha = input_record["pinned"]
                if input_sha and not is_ancestor(root, input_sha, tip_sha):
                    warnings.append(
                        f"validated tip does not contain input: {input_record['ref']}"
                    )

        inspected.append(
            {
                "path": str(path),
                "branch": branch,
                "head": head,
                "state": state,
                "owner": fields.get("owner", [None])[0],
                "base": base_sha,
                "target_ref": target_ref,
                "target": target_sha,
                "tip": tip_sha,
                "inputs": inputs,
                "dirty_entries": len(status),
                "changed_paths": all_changed,
                "out_of_scope_paths": out_of_scope,
                "write_scope": fields.get("write-scope", []),
                "semantic_scope": fields.get("semantic-scope", []),
                "warnings": warnings,
            }
        )

    file_owners: dict[str, list[str]] = defaultdict(list)
    scope_owners: dict[str, list[str]] = defaultdict(list)
    semantic_owners: dict[str, list[str]] = defaultdict(list)
    declared_inputs: dict[str, set[str]] = defaultdict(set)
    for item in inspected:
        if item["state"] not in ACTIVE_STATES:
            continue
        label = item["branch"] or item["path"]
        for input_record in item["inputs"]:
            declared_inputs[label].add(input_record["ref"].removeprefix("refs/heads/"))
        for changed_path in item["changed_paths"]:
            if label not in file_owners[changed_path]:
                file_owners[changed_path].append(label)
        for scope in item["write_scope"]:
            normalized = scope.strip().removeprefix("./")
            if (
                normalized.casefold() not in {"", "none", "n/a"}
                and label not in scope_owners[normalized]
            ):
                scope_owners[normalized].append(label)
        for semantic in item["semantic_scope"]:
            normalized = semantic.strip().casefold()
            if (
                normalized not in {"", "none", "n/a"}
                and label not in semantic_owners[normalized]
            ):
                semantic_owners[normalized].append(label)

    def has_unordered_pair(owners: list[str]) -> bool:
        for index, first in enumerate(owners):
            for second in owners[index + 1 :]:
                if second in declared_inputs[first] or first in declared_inputs[second]:
                    continue
                return True
        return False

    def unresolved(values: dict[str, list[str]]) -> dict[str, list[str]]:
        return {
            key: owners
            for key, owners in sorted(values.items())
            if len(owners) > 1 and has_unordered_pair(owners)
        }

    overlaps = {
        "changed_paths": unresolved(file_owners),
        "write_scope": unresolved(scope_owners),
        "semantic_scope": unresolved(semantic_owners),
    }
    snapshot_warnings: list[str] = []
    final_main_sha = resolve_commit(root, "main")
    if final_main_sha != current_main_sha:
        snapshot_warnings.append("main moved during inventory")
    for target_ref, target_sha in sorted(target_refs_before.items()):
        if target_ref == "main":
            continue
        final_target_sha = resolve_commit(root, target_ref)
        if final_target_sha != target_sha:
            snapshot_warnings.append(
                f"integration target moved during inventory: {target_ref}"
            )
    return {
        "repository": str(root),
        "selection": sorted(selected) if selected else "all worktrees",
        "main_before": current_main_sha,
        "main_after": final_main_sha,
        "worktrees": inspected,
        "overlaps": overlaps,
        "snapshot_warnings": snapshot_warnings,
    }


def print_text(report: dict[str, Any]) -> None:
    print(f"repository: {report['repository']}")
    print(f"selection: {report['selection']}")
    print("note: this is a best-effort snapshot; recheck pins before promotion or push")
    for warning in report["snapshot_warnings"]:
        print(f"warning: {warning}")
    for item in report["worktrees"]:
        print(f"\nworktree: {item['path']}")
        print(f"  branch: {item['branch'] or '(detached)'}")
        print(f"  head: {item['head']}")
        print(
            f"  owner/state: {item['owner'] or '(unknown)'}/{item['state'] or '(unknown)'}"
        )
        if item["target_ref"]:
            print(f"  integration target: {item['target_ref']}")
        print(f"  pinned tip: {item['tip'] or '(pending/unknown)'}")
        print(f"  ordered inputs: {len(item['inputs'])}")
        print(f"  dirty entries: {item['dirty_entries']}")
        print(
            f"  changed paths (base diff plus worktree): {len(item['changed_paths'])}"
        )
        for warning in item["warnings"]:
            print(f"  warning: {warning}")

    overlaps = report["overlaps"]
    print("\nautomatic exact-overlap findings (manual review still required):")
    if not any(overlaps.values()):
        print("  none among branches with declared active states")
    for path, branches in overlaps["changed_paths"].items():
        print(f"  changed path {path}: {', '.join(branches)}")
    for scope, branches in overlaps["write_scope"].items():
        print(f"  declared write scope {scope}: {', '.join(branches)}")
    for semantic, branches in overlaps["semantic_scope"].items():
        print(f"  semantic scope {semantic}: {', '.join(branches)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", nargs="?", default=".")
    parser.add_argument(
        "--branch",
        action="append",
        help="inspect a relevant branch only; repeat for peers (default: all)",
    )
    parser.add_argument(
        "--mainline",
        action="append",
        help="shared branch with no task description; repeat (default: main, master)",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return status 1 when warnings or overlaps are present",
    )
    args = parser.parse_args()

    try:
        report = inspect(
            Path(args.repository).resolve(),
            branches=args.branch,
            mainlines=tuple(args.mainline or ("main", "master")),
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"inventory failed: {error}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)

    has_warnings = bool(report["snapshot_warnings"]) or any(
        item["warnings"] for item in report["worktrees"]
    )
    has_overlaps = any(report["overlaps"].values())
    return 1 if args.strict and (has_warnings or has_overlaps) else 0


if __name__ == "__main__":
    raise SystemExit(main())
