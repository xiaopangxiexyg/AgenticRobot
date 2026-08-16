from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .workspace import LabError, TaskPlan, WorkspaceManager
from .writer_lease import WriterLeaseManager


def _control_repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _repository_list(value: str) -> list[str]:
    repositories = [item.strip() for item in value.split(",") if item.strip()]
    if not repositories:
        raise argparse.ArgumentTypeError("repository list cannot be empty")
    return repositories


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab",
        description="Task-oriented multi-repository workspace control plane.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    change = subcommands.add_parser("change", help="manage multi-Task Change groups")
    change_commands = change.add_subparsers(dest="change_command", required=True)
    change_create = change_commands.add_parser("create")
    change_create.add_argument("change_id")
    change_create.add_argument("--title", default="")
    change_create.add_argument("--json", action="store_true")
    change_inspect = change_commands.add_parser("inspect")
    change_inspect.add_argument("change_id")
    change_inspect.add_argument("--json", action="store_true")

    task = subcommands.add_parser("task", help="manage development Tasks")
    task_commands = task.add_subparsers(dest="task_command", required=True)
    create = task_commands.add_parser(
        "create",
        help="create an isolated multi-repository Task workspace",
    )
    create.add_argument("task_id")
    create.add_argument("--repos", required=True, type=_repository_list)
    create.add_argument("--title", default="")
    create.add_argument("--change-id")
    create.add_argument("--dev-host")
    create.add_argument(
        "--base",
        action="append",
        default=[],
        metavar="REPOSITORY=REF",
    )
    create.add_argument(
        "--scope",
        action="append",
        default=[],
        metavar="REPOSITORY=PATH_OR_GLOB",
    )
    create.add_argument("--dry-run", action="store_true")
    create.add_argument("--json", action="store_true")

    writer = task_commands.add_parser(
        "writer",
        help="coordinate the active Agent writing each Task repository",
    )
    writer_commands = writer.add_subparsers(dest="writer_command", required=True)
    writer_status = writer_commands.add_parser("status")
    writer_status.add_argument("task_id")
    writer_status.add_argument("--repo")
    writer_status.add_argument("--json", action="store_true")
    writer_claim = writer_commands.add_parser("claim")
    writer_claim.add_argument("task_id")
    writer_claim.add_argument("--repo", required=True)
    writer_claim.add_argument("--actor", required=True)
    writer_claim.add_argument("--ttl-seconds", type=int, default=3600)
    writer_claim.add_argument("--replace-expired")
    writer_claim.add_argument("--json", action="store_true")
    writer_renew = writer_commands.add_parser("renew")
    writer_renew.add_argument("task_id")
    writer_renew.add_argument("--repo", required=True)
    writer_renew.add_argument("--actor", required=True)
    writer_renew.add_argument("--lease-id", required=True)
    writer_renew.add_argument("--ttl-seconds", type=int, default=3600)
    writer_renew.add_argument("--json", action="store_true")
    writer_release = writer_commands.add_parser("release")
    writer_release.add_argument("task_id")
    writer_release.add_argument("--repo", required=True)
    writer_release.add_argument("--actor", required=True)
    writer_release.add_argument("--lease-id", required=True)
    writer_release.add_argument("--json", action="store_true")

    open_parser = subcommands.add_parser("open", help="open a Task workspace")
    open_parser.add_argument("task_id")
    open_parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="print the workspace path without launching an editor",
    )
    open_parser.add_argument(
        "--editor",
        help="editor command; defaults to LAB_EDITOR or 'code -n'",
    )

    inspect = subcommands.add_parser(
        "inspect",
        help="inspect a Task or Change without modifying it",
    )
    inspect.add_argument("identifier")
    inspect.add_argument("--json", action="store_true")
    return parser


def _plan_document(plan: TaskPlan) -> dict[str, object]:
    return {
        "task_id": plan.task_id,
        "title": plan.title,
        "change_id": plan.change_id,
        "development_host": plan.development_host,
        "workspace_root": str(plan.task_root),
        "repositories": {
            repository.name: {
                "management_path": str(repository.management_path),
                "worktree_path": str(repository.worktree_path),
                "branch": repository.branch,
                "base_ref": repository.base_ref,
                "base_commit": repository.base_commit,
                "write_scope": list(repository.scopes),
                "management_dirty": repository.management_dirty,
            }
            for repository in plan.repositories
        },
    }


def _print_task(payload: dict[str, object]) -> None:
    print(f"Task {payload.get('task_id')}: {payload.get('title') or '(untitled)'}")
    print(f"Development host: {payload.get('development_host')}")
    print(f"Status: {payload.get('status')}")
    print(f"Workspace: {payload.get('workspace_root')}")
    repositories = payload.get("repositories")
    if isinstance(repositories, dict):
        for name, raw_repository in repositories.items():
            if not isinstance(raw_repository, dict):
                continue
            state = "dirty" if raw_repository.get("dirty") else "clean"
            branch_marker = (
                "" if raw_repository.get("branch_matches") else " [BRANCH DRIFT]"
            )
            print(
                f"  {name}: {raw_repository.get('branch')} "
                f"@ {str(raw_repository.get('commit', ''))[:12]} "
                f"{state}{branch_marker}"
            )
            violations = raw_repository.get("scope_violations")
            if isinstance(violations, list) and violations:
                print(f"    scope violations: {', '.join(violations)}")
            writer = raw_repository.get("writer_lease")
            if isinstance(writer, dict):
                lease = writer.get("lease")
                actor = lease.get("actor") if isinstance(lease, dict) else None
                actor_suffix = f" ({actor})" if actor else ""
                print(f"    writer: {writer.get('state')}{actor_suffix}")
    print(f"Next action: {payload.get('next_action')}")


def _print_change(payload: dict[str, object]) -> None:
    print(f"Change {payload.get('change_id')}: {payload.get('title') or '(untitled)'}")
    print(f"Status: {payload.get('status')}")
    print(f"Tasks: {payload.get('task_count', 0)}")
    tasks = payload.get("tasks")
    if isinstance(tasks, dict):
        for task_id, raw_task in tasks.items():
            if isinstance(raw_task, dict):
                print(
                    f"  {task_id}: {raw_task.get('development_host')} "
                    f"{raw_task.get('status')}"
                )
    print(f"Next action: {payload.get('next_action')}")


def _print_writer(payload: dict[str, object]) -> None:
    repositories = payload.get("repositories")
    if isinstance(repositories, dict):
        print(f"Task: {payload.get('task_id')}")
        print(f"Writer coordination: {payload.get('status')}")
        for name, raw_status in repositories.items():
            if not isinstance(raw_status, dict):
                continue
            lease = raw_status.get("lease")
            actor = lease.get("actor") if isinstance(lease, dict) else None
            suffix = f" actor={actor}" if actor else ""
            print(f"  {name}: {raw_status.get('state')}{suffix}")
        return
    if payload.get("schema") == "lab.writer-lease-status.v1":
        authority = payload.get("authority")
        lease = payload.get("lease")
        repository = (
            authority.get("repository") if isinstance(authority, dict) else None
        )
        print(f"Repository: {repository}")
        print(f"Writer state: {payload.get('state')}")
        if isinstance(lease, dict):
            print(f"Actor: {lease.get('actor')}")
            print(f"Lease: {lease.get('lease_id')}")
            print(f"Expires: {lease.get('expires_at')}")
        return
    print(f"Writer lease: {payload.get('status')}")
    print(f"Task/repository: {payload.get('task_id')}/{payload.get('repository')}")
    print(f"Actor: {payload.get('actor')}")
    print(f"Lease: {payload.get('lease_id')}")


def _emit(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        manager = WorkspaceManager(control_repository=_control_repository())
        if arguments.command == "change":
            if arguments.change_command == "create":
                payload = manager.create_change(arguments.change_id, title=arguments.title)
                if arguments.json:
                    _emit(payload, as_json=True)
                else:
                    print(f"Created Change {payload['change_id']}")
                return 0
            payload = manager.inspect_change(arguments.change_id)
            if arguments.json:
                _emit(payload, as_json=True)
            else:
                _print_change(payload)
            return 0

        if arguments.command == "task" and arguments.task_command == "create":
            plan = manager.plan_task(
                task_id=arguments.task_id,
                repository_names=arguments.repos,
                title=arguments.title,
                change_id=arguments.change_id,
                development_host=arguments.dev_host,
                base_assignments=arguments.base,
                scope_assignments=arguments.scope,
            )
            payload = _plan_document(plan) if arguments.dry_run else manager.create_task(plan)
            if arguments.json:
                _emit(payload, as_json=True)
            else:
                verb = "Would create" if arguments.dry_run else "Created"
                print(f"{verb} Task {payload['task_id']} at {payload['workspace_root']}")
            return 0

        if arguments.command == "task" and arguments.task_command == "writer":
            writer = WriterLeaseManager(manager)
            if arguments.writer_command == "status":
                payload = (
                    writer.inspect_repository(arguments.task_id, arguments.repo)
                    if arguments.repo
                    else writer.inspect_task(arguments.task_id)
                )
            elif arguments.writer_command == "claim":
                payload = writer.claim(
                    arguments.task_id,
                    arguments.repo,
                    actor=arguments.actor,
                    ttl_seconds=arguments.ttl_seconds,
                    replace_expired=arguments.replace_expired,
                )
            elif arguments.writer_command == "renew":
                payload = writer.renew(
                    arguments.task_id,
                    arguments.repo,
                    actor=arguments.actor,
                    lease_id=arguments.lease_id,
                    ttl_seconds=arguments.ttl_seconds,
                )
            else:
                payload = writer.release(
                    arguments.task_id,
                    arguments.repo,
                    actor=arguments.actor,
                    lease_id=arguments.lease_id,
                )
            if arguments.json:
                _emit(payload, as_json=True)
            else:
                _print_writer(payload)
            return 0

        if arguments.command == "open":
            workspace = manager.workspace_path(arguments.task_id)
            if arguments.print_only:
                print(workspace)
                return 0
            configured_editor = arguments.editor or os.environ.get("LAB_EDITOR") or "code -n"
            command = shlex.split(configured_editor)
            if not command:
                raise LabError("editor command is empty")
            completed = subprocess.run([*command, str(workspace)], check=False)
            if completed.returncode != 0:
                raise LabError(
                    f"editor exited with status {completed.returncode}: {configured_editor}"
                )
            return 0

        if arguments.command == "inspect":
            if manager.task_manifest_path(arguments.identifier).is_file():
                payload = manager.inspect_task(arguments.identifier)
                is_change = False
            else:
                payload = manager.inspect_change(arguments.identifier)
                is_change = True
            if arguments.json:
                _emit(payload, as_json=True)
            elif is_change:
                _print_change(payload)
            else:
                _print_task(payload)
            return 0
    except LabError as exc:
        print(f"lab: error: {exc}", file=sys.stderr)
        return 2

    parser.error("unhandled command")
    return 2
