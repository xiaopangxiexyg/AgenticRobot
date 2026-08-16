from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .baseline import load_baseline_lock
from .control_install import (
    inspect_control_plane_install,
    install_control_plane,
    plan_control_plane_install,
)
from .environments import inspect_environments
from .inspection import (
    attach_run_comparison,
    enrich_change_inspection,
    enrich_task_inspection,
)
from .linked_sync import apply_linked_project_checkout, plan_linked_project_checkout
from .project_maintenance import apply_guarded_project_checkout
from .project_sync import fetch_project_sync, plan_project_sync
from .runs import CANDIDATE_EVIDENCE_TIER, DEVELOPMENT_EVIDENCE_TIER, RunManager
from .workspace import LabError, TaskPlan, WorkspaceManager
from .writer_lease import WriterLeaseManager


def _control_repository() -> Path:
    configured = os.environ.get("LAB_CONTROL_REPOSITORY_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[1]
    )


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

    control = subcommands.add_parser(
        "control",
        help="manage the external protected lab control-plane installation",
    )
    control_commands = control.add_subparsers(dest="control_command", required=True)
    for command_name in ("plan-install", "status"):
        command = control_commands.add_parser(command_name)
        command.add_argument("--install-root")
        command.add_argument("--json", action="store_true")
    control_install = control_commands.add_parser("install")
    control_install.add_argument("--install-root")
    control_install.add_argument("--confirm-plan-sha256", required=True)
    control_install.add_argument("--json", action="store_true")

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

    sync = subcommands.add_parser(
        "sync",
        help="inspect or fetch Git metadata without moving any checkout",
    )
    sync_commands = sync.add_subparsers(dest="sync_command", required=True)
    for command_name, help_text in (
        ("inspect", "inspect stable refs, gitlinks and active Task writers"),
        ("fetch", "fetch stable refs without checkout, LFS or submodules"),
    ):
        command = sync_commands.add_parser(command_name, help=help_text)
        command.add_argument("--repos", type=_repository_list)
        if command_name == "fetch":
            command.add_argument("--timeout-seconds", type=int, default=120)
        command.add_argument("--json", action="store_true")
    sync_apply = sync_commands.add_parser(
        "apply",
        help="guard one unlinked stable checkout and apply an exact fast-forward",
    )
    sync_apply.add_argument("--repo", required=True)
    sync_apply.add_argument("--confirm-plan-sha256", required=True)
    sync_apply.add_argument("--reservation-id", required=True)
    sync_apply.add_argument("--json", action="store_true")
    sync_linked_plan = sync_commands.add_parser(
        "plan-linked",
        help="plan one stable parent and its direct pinned children",
    )
    sync_linked_plan.add_argument("--repo", required=True)
    sync_linked_plan.add_argument("--json", action="store_true")
    sync_linked_apply = sync_commands.add_parser(
        "apply-linked",
        help="apply one confirmed parent/gitlink-child checkout plan",
    )
    sync_linked_apply.add_argument("--repo", required=True)
    sync_linked_apply.add_argument("--confirm-plan-sha256", required=True)
    sync_linked_apply.add_argument("--reservation-id", required=True)
    sync_linked_apply.add_argument("--json", action="store_true")

    open_parser = subcommands.add_parser(
        "open", help="open a Task workspace or an exact Run snapshot"
    )
    open_parser.add_argument("task_id", nargs="?")
    open_parser.add_argument(
        "--run",
        dest="run_id",
        help="open the detached multi-repository snapshot tested by a Run",
    )
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
    inspect.add_argument(
        "--against-run",
        dest="against_run_id",
        help="compare a current Task with one exact recorded Run",
    )
    inspect.add_argument("--json", action="store_true")

    doctor = subcommands.add_parser(
        "doctor",
        help="verify repository environments and first-party import provenance",
    )
    doctor.add_argument(
        "--environment",
        action="append",
        dest="environment_ids",
        help="environment ID; repeatable, defaults to all local Profiles",
    )
    doctor.add_argument("--json", action="store_true")

    for command_name, help_text in (
        ("test", "run Development evidence on a clean detached snapshot"),
        ("verify", "run Candidate evidence using a protected control plane"),
    ):
        command = subcommands.add_parser(command_name, help=help_text)
        command.add_argument("task_id")
        command.add_argument("profile_id", nargs="?")
        command.add_argument(
            "--recommended",
            action="store_true",
            help="run protected impact-rule recommendations",
        )
        command.add_argument("--json", action="store_true")

    runs = subcommands.add_parser("runs", help="list recorded local Runs")
    runs.add_argument("task_id", nargs="?")
    runs.add_argument("--json", action="store_true")
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
    readiness = payload.get("readiness")
    if isinstance(readiness, dict):
        print(f"Test readiness: {readiness.get('status')}")
        recommended = readiness.get("recommended_profiles")
        if isinstance(recommended, list):
            print(
                "Recommended Profiles: "
                + (", ".join(str(item) for item in recommended) or "none")
            )
        profiles = readiness.get("profiles")
        if isinstance(profiles, dict):
            for profile_id, raw_profile in profiles.items():
                if isinstance(raw_profile, dict):
                    print(
                        f"  {profile_id}: "
                        f"{raw_profile.get('evidence_freshness')}"
                    )
    comparison = payload.get("against_run")
    if isinstance(comparison, dict):
        print(
            f"Against Run: {comparison.get('run_id')} "
            f"({comparison.get('classification')})"
        )
        repositories = comparison.get("repositories")
        if isinstance(repositories, dict):
            for name, raw_repository in repositories.items():
                if isinstance(raw_repository, dict):
                    print(
                        f"  {name}: "
                        f"{len(raw_repository.get('changed_paths', []))} changed paths"
                    )
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
    readiness = payload.get("readiness")
    if isinstance(readiness, dict):
        print(f"Test readiness: {readiness.get('status')}")
        recommended = readiness.get("recommended")
        if isinstance(recommended, list):
            for item in recommended:
                if isinstance(item, dict):
                    print(
                        f"  recommended: {item.get('task_id')} / "
                        f"{item.get('profile_id')}"
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


def _print_sync(payload: dict[str, object]) -> None:
    post_fetch = payload.get("post_fetch_plan")
    plan = post_fetch if isinstance(post_fetch, dict) else payload
    print(f"Project: {plan.get('project_id')}")
    if payload.get("schema") == "lab.project-sync-fetch.v1":
        print(f"Fetch: {payload.get('status')}")
    print(f"Metadata readiness: {plan.get('fetch_status')}")
    print(f"Stable checkout status: {plan.get('checkout_sync_status')}")
    print("Fetch order: " + ", ".join(plan.get("fetch_order", [])))
    repositories = plan.get("repositories")
    if isinstance(repositories, dict):
        for name in plan.get("fetch_order", []):
            repository = repositories.get(name)
            if not isinstance(repository, dict):
                continue
            writers = repository.get("active_writers")
            writer_count = len(writers) if isinstance(writers, list) else 0
            print(
                f"  {name}: fetch={repository.get('fetch_action')} "
                f"checkout={repository.get('checkout_sync_action')} "
                f"writers={writer_count}"
            )
            blocker = repository.get("blocker")
            if blocker:
                print(f"    BLOCKED: {blocker}")
    failures = payload.get("failures")
    if isinstance(failures, dict):
        for name, failure in failures.items():
            print(f"  FETCH FAILED {name}: {failure}")
    print("Checkout/worktree updates: none")


def _print_guarded_sync(payload: dict[str, object]) -> None:
    guarded = payload.get("guarded_sync")
    stable = guarded.get("stable_sync_receipt") if isinstance(guarded, dict) else None
    print(f"Project: {payload.get('project_id')}")
    print(f"Repository: {payload.get('repository')}")
    print(f"Status: {payload.get('status')}")
    if isinstance(stable, dict):
        print(f"Before: {stable.get('before_commit')}")
        print(f"After:  {stable.get('after_commit')}")
    print(f"Receipt: {payload.get('receipt_path')}")
    print("Network/LFS/submodule/task-worktree updates: none")


def _print_linked_sync(payload: dict[str, object]) -> None:
    print(f"Project: {payload.get('project_id')}")
    print(f"Root repository: {payload.get('root_repository')}")
    print(f"Action/status: {payload.get('action', payload.get('status'))}")
    print(f"Plan: {payload.get('plan_sha256', payload.get('linked_plan_sha256'))}")
    children = payload.get("children")
    if isinstance(children, dict):
        for name, raw in children.items():
            if isinstance(raw, dict):
                print(
                    f"  {name}: {str(raw.get('current_commit', ''))[:12]} -> "
                    f"{str(raw.get('target_commit', ''))[:12]}"
                )
    if payload.get("receipt_path"):
        print(f"Receipt: {payload.get('receipt_path')}")
    if payload.get("error"):
        print(f"QUARANTINED: {payload.get('error')}")
    print("Network/fetch/LFS/reset/clean/task-worktree updates: none")


def _print_control_install(payload: dict[str, object]) -> None:
    print(f"Status/action: {payload.get('status', payload.get('action'))}")
    print(f"Commit: {payload.get('current_commit', payload.get('commit'))}")
    print(f"Install root: {payload.get('install_root')}")
    if payload.get("plan_sha256"):
        print(f"Plan: {payload.get('plan_sha256')}")
    if payload.get("wrapper_path"):
        print(f"Wrapper: {payload.get('wrapper_path')}")
    print("Network/task payload/LFS downloads: none")


def _print_run(payload: dict[str, object]) -> None:
    outcome = payload.get("outcome")
    classification = (
        outcome.get("classification") if isinstance(outcome, dict) else "UNKNOWN"
    )
    profile = payload.get("profile")
    profile_id = profile.get("profile_id") if isinstance(profile, dict) else "?"
    print(f"Run: {payload.get('run_id')}")
    print(f"Task: {payload.get('task_id')}")
    print(f"Profile: {profile_id}")
    print(f"Evidence: {payload.get('evidence_tier')}")
    print(f"Result: {classification}")
    print(f"Snapshot: {payload.get('snapshot_id')}")


def _print_run_list(payloads: list[dict[str, object]]) -> None:
    if not payloads:
        print("No Runs found.")
        return
    for payload in payloads:
        outcome = payload.get("outcome")
        classification = (
            outcome.get("classification") if isinstance(outcome, dict) else "RUNNING"
        )
        profile = payload.get("profile")
        profile_id = profile.get("profile_id") if isinstance(profile, dict) else "?"
        print(
            f"{payload.get('run_id')}  {payload.get('task_id')}  "
            f"{profile_id}  {classification}"
        )


def _recommended_profiles(
    manager: WorkspaceManager,
    task_id: str,
) -> list[str]:
    inspection = enrich_task_inspection(manager, manager.inspect_task(task_id))
    if inspection.get("status") != "ACTIVE":
        raise LabError("recommended tests require an ACTIVE Task without drift")
    repositories = inspection.get("repositories")
    if not isinstance(repositories, dict):
        raise LabError("Task inspection lacks repositories")
    dirty = sorted(
        name
        for name, raw_repository in repositories.items()
        if isinstance(raw_repository, dict) and raw_repository.get("dirty") is True
    )
    if dirty:
        raise LabError(
            "recommended clean tests require checkpointed repositories: "
            + ", ".join(dirty)
        )
    readiness = inspection.get("readiness")
    if not isinstance(readiness, dict) or readiness.get("status") != "READY":
        errors = readiness.get("profile_errors") if isinstance(readiness, dict) else None
        detail = (
            "; ".join(str(error) for error in errors)
            if isinstance(errors, list) and errors
            else "required Profiles or baseline inputs are unavailable"
        )
        raise LabError(f"recommended test readiness requires attention: {detail}")
    profiles = readiness.get("recommended_profiles")
    if not isinstance(profiles, list) or not all(
        isinstance(profile_id, str) for profile_id in profiles
    ):
        raise LabError("Task inspection has invalid recommended Profiles")
    return list(profiles)


def _run_recommended_profiles(
    manager: WorkspaceManager,
    task_id: str,
    *,
    evidence_tier: str,
) -> dict[str, object]:
    profile_ids = _recommended_profiles(manager, task_id)
    run_manager = RunManager(manager)
    run_manager.validate_evidence_tier(evidence_tier)
    runs = [
        run_manager.run_clean_profile(
            task_id,
            profile_id,
            evidence_tier=evidence_tier,
        )
        for profile_id in profile_ids
    ]
    classifications = []
    for run in runs:
        outcome = run.get("outcome")
        classifications.append(
            outcome.get("classification") if isinstance(outcome, dict) else None
        )
    return {
        "schema": "lab.recommended-test-batch.v1",
        "task_id": task_id,
        "evidence_tier": evidence_tier,
        "profiles": profile_ids,
        "runs": runs,
        "all_passed": all(result == "PASSED" for result in classifications),
        "no_tests_required": not profile_ids,
    }


def _print_recommended_batch(payload: dict[str, object]) -> None:
    print(f"Task: {payload.get('task_id')}")
    print(f"Evidence: {payload.get('evidence_tier')}")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        print("Recommended Profiles: none")
        print("No tests required for the current Task diff.")
        return
    print("Recommended Profiles: " + ", ".join(str(item) for item in profiles))
    runs = payload.get("runs")
    if isinstance(runs, list):
        for run in runs:
            if isinstance(run, dict):
                _print_run(run)
    print(f"All passed: {str(payload.get('all_passed')).lower()}")


def _emit(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        manager = WorkspaceManager(control_repository=_control_repository())
        if arguments.command == "control":
            install_root = (
                Path(arguments.install_root).expanduser().resolve()
                if arguments.install_root
                else Path(
                    manager.environ.get(
                        "LAB_CONTROL_PLANE_ROOT",
                        manager.state_root / "control-plane",
                    )
                )
                .expanduser()
                .resolve()
            )
            if arguments.control_command == "status":
                payload = inspect_control_plane_install(install_root)
            else:
                baseline_path = Path(
                    manager.environ.get(
                        "LAB_BASELINE_LOCK",
                        manager.control_repository / "stack" / "baseline.json",
                    )
                ).expanduser().resolve()
                baseline = load_baseline_lock(baseline_path)
                control_name = manager.project.control_repository
                pin = baseline.repositories.get(control_name)
                if pin is None:
                    raise LabError(
                        f"baseline lacks control repository: {control_name}"
                    )
                control_repository = manager.catalog[control_name]
                control_plan = plan_control_plane_install(
                    control_repository,
                    managed_control_repository=control_repository,
                    expected_identity=pin.canonical_identity,
                    protected_branch=manager.project.stable_branches[control_name],
                    install_root=install_root,
                )
                payload = (
                    install_control_plane(
                        control_plan,
                        confirmed_plan_sha256=arguments.confirm_plan_sha256,
                    )
                    if arguments.control_command == "install"
                    else control_plan
                )
            if arguments.json:
                _emit(payload, as_json=True)
            else:
                _print_control_install(payload)
            return 0
        if arguments.command == "change":
            if arguments.change_command == "create":
                payload = manager.create_change(arguments.change_id, title=arguments.title)
                if arguments.json:
                    _emit(payload, as_json=True)
                else:
                    print(f"Created Change {payload['change_id']}")
                return 0
            payload = enrich_change_inspection(
                manager,
                manager.inspect_change(arguments.change_id),
            )
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

        if arguments.command == "sync":
            baseline_path = Path(
                manager.environ.get(
                    "LAB_BASELINE_LOCK",
                    manager.control_repository / "stack" / "baseline.json",
                )
            ).expanduser().resolve()
            if arguments.sync_command in {"plan-linked", "apply-linked"}:
                raw_name = arguments.repo
                name = manager.repository_aliases.get(raw_name.casefold(), raw_name)
                project_plan = plan_project_sync(
                    manager,
                    baseline_path=baseline_path,
                    repository_names=(name,),
                )
                linked_plan = plan_linked_project_checkout(
                    manager,
                    project_plan,
                    root_repository=name,
                )
                if arguments.sync_command == "apply-linked":
                    sync_payload = apply_linked_project_checkout(
                        manager,
                        linked_plan,
                        confirmed_plan_sha256=arguments.confirm_plan_sha256,
                        reservation_id=arguments.reservation_id,
                    )
                else:
                    sync_payload = linked_plan
                if arguments.json:
                    _emit(sync_payload, as_json=True)
                else:
                    _print_linked_sync(sync_payload)
                return (
                    0
                    if sync_payload.get("status")
                    in {None, "COMPLETE", "UP_TO_DATE"}
                    else 1
                )
            if arguments.sync_command == "apply":
                raw_name = arguments.repo
                name = manager.repository_aliases.get(raw_name.casefold(), raw_name)
                sync_plan = plan_project_sync(
                    manager,
                    baseline_path=baseline_path,
                    repository_names=(name,),
                )
                sync_payload = apply_guarded_project_checkout(
                    manager,
                    sync_plan,
                    repository=name,
                    confirmed_plan_sha256=arguments.confirm_plan_sha256,
                    reservation_id=arguments.reservation_id,
                )
                if arguments.json:
                    _emit(sync_payload, as_json=True)
                else:
                    _print_guarded_sync(sync_payload)
                return 0
            sync_plan = plan_project_sync(
                manager,
                baseline_path=baseline_path,
                repository_names=arguments.repos or (),
            )
            sync_payload = (
                fetch_project_sync(
                    manager,
                    sync_plan,
                    timeout_seconds=arguments.timeout_seconds,
                )
                if arguments.sync_command == "fetch"
                else sync_plan
            )
            if arguments.json:
                _emit(sync_payload, as_json=True)
            else:
                _print_sync(sync_payload)
            if arguments.sync_command == "fetch":
                return 0 if sync_payload.get("status") == "COMPLETE" else 1
            return 0 if sync_payload.get("fetch_status") == "READY" else 1

        if arguments.command == "open":
            if bool(arguments.task_id) == bool(arguments.run_id):
                raise LabError("open requires exactly one Task ID or --run RUN_ID")
            if arguments.run_id:
                run = RunManager(manager).find_run(arguments.run_id)
                workspace = Path(str(run.get("snapshot_workspace", "")))
                if not workspace.is_file():
                    raise LabError(f"Run snapshot workspace is missing: {workspace}")
            else:
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
                payload = enrich_task_inspection(
                    manager,
                    manager.inspect_task(arguments.identifier),
                )
                if arguments.against_run_id:
                    payload = attach_run_comparison(
                        payload,
                        RunManager(manager).find_run(arguments.against_run_id),
                    )
                is_change = False
            else:
                if arguments.against_run_id:
                    raise LabError("--against-run requires a Task, not a Change")
                payload = enrich_change_inspection(
                    manager,
                    manager.inspect_change(arguments.identifier),
                )
                is_change = True
            if arguments.json:
                _emit(payload, as_json=True)
            elif is_change:
                _print_change(payload)
            else:
                _print_task(payload)
            return 0

        if arguments.command == "doctor":
            environment_root = (
                Path(manager.environ["LAB_ENVIRONMENT_PROFILE_ROOT"])
                .expanduser()
                .resolve()
                if manager.environ.get("LAB_ENVIRONMENT_PROFILE_ROOT")
                else manager.control_repository / "integration" / "environments"
            )
            environment_ids = arguments.environment_ids or sorted(
                path.stem for path in environment_root.glob("*.json") if path.is_file()
            )
            if not environment_ids:
                raise LabError(f"no environment Profiles found: {environment_root}")
            payload = inspect_environments(
                environment_root,
                environment_ids,
                repository_paths=manager.catalog,
                environ=manager.environ,
            )
            if arguments.json:
                _emit(payload, as_json=True)
            else:
                print(f"Status: {payload['status']}")
                overlay = payload.get("source_overlay")
                if isinstance(overlay, list):
                    print("Source overlay:")
                    for path in overlay:
                        print(f"  {path}")
                receipts = payload.get("environments")
                if isinstance(receipts, dict):
                    for environment_id, receipt in receipts.items():
                        if isinstance(receipt, dict):
                            print(
                                f"{environment_id}: "
                                f"{receipt.get('python_executable')} "
                                f"Python {receipt.get('python_major_minor')}"
                            )
                errors = payload.get("errors")
                if isinstance(errors, list):
                    for error in errors:
                        print(f"ERROR: {error}")
            return 0 if payload.get("status") == "READY" else 1

        if arguments.command in {"test", "verify"}:
            if bool(arguments.profile_id) == bool(arguments.recommended):
                raise LabError(
                    f"{arguments.command} requires exactly one Profile ID or "
                    "--recommended"
                )
            evidence_tier = (
                CANDIDATE_EVIDENCE_TIER
                if arguments.command == "verify"
                else DEVELOPMENT_EVIDENCE_TIER
            )
            if arguments.recommended:
                payload = _run_recommended_profiles(
                    manager,
                    arguments.task_id,
                    evidence_tier=evidence_tier,
                )
                if arguments.json:
                    _emit(payload, as_json=True)
                else:
                    _print_recommended_batch(payload)
                return 0 if payload.get("all_passed") is True else 1
            payload = RunManager(manager).run_clean_profile(
                arguments.task_id,
                arguments.profile_id,
                evidence_tier=evidence_tier,
            )
            if arguments.json:
                _emit(payload, as_json=True)
            else:
                _print_run(payload)
            outcome = payload.get("outcome")
            return (
                0
                if isinstance(outcome, dict)
                and outcome.get("classification") == "PASSED"
                else 1
            )

        if arguments.command == "runs":
            payloads = RunManager(manager).list_runs(arguments.task_id)
            if arguments.json:
                print(json.dumps(payloads, indent=2, sort_keys=True))
            else:
                _print_run_list(payloads)
            return 0
    except LabError as exc:
        print(f"lab: error: {exc}", file=sys.stderr)
        return 2

    parser.error("unhandled command")
    return 2
