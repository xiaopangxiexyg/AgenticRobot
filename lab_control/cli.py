from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .draft_bundle import export_draft_bundle
from .draft_cleanup import DraftSnapshotPruner
from .drafts import DEFAULT_MAX_FILE_BYTES, DraftSnapshotManager
from .environments import inspect_environments
from .inspection import (
    attach_run_comparison,
    enrich_change_inspection,
    enrich_task_inspection,
)
from .project_sync import fetch_project_sync, plan_project_sync
from .remote_dispatch import RemoteDispatchPlanner
from .runs import CANDIDATE_EVIDENCE_TIER, DEVELOPMENT_EVIDENCE_TIER, RunManager
from .ssh_transport import (
    build_ssh_dispatch_argv,
    control_remote_request,
    dispatch_remote_outbox,
    install_pinned_known_host,
)
from .transport import load_transport_profile
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
        description="Task-oriented local workspace control plane.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    change = subcommands.add_parser("change", help="manage multi-Task Change groups")
    change_subcommands = change.add_subparsers(dest="change_command", required=True)
    change_create = change_subcommands.add_parser("create")
    change_create.add_argument("change_id")
    change_create.add_argument("--title", default="")
    change_create.add_argument("--json", action="store_true")
    change_inspect = change_subcommands.add_parser("inspect")
    change_inspect.add_argument("change_id")
    change_inspect.add_argument("--json", action="store_true")

    task = subcommands.add_parser("task", help="manage development Tasks")
    task_subcommands = task.add_subparsers(dest="task_command", required=True)
    create = task_subcommands.add_parser(
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

    writer = task_subcommands.add_parser(
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
        help="inspect or fetch project Git metadata without moving worktrees",
    )
    sync_commands = sync.add_subparsers(dest="sync_command", required=True)
    for command_name, help_text in (
        ("inspect", "inspect multi-repository refs, gitlinks and active writers"),
        ("fetch", "fetch protected stable refs without checkout, LFS or submodules"),
    ):
        sync_command = sync_commands.add_parser(command_name, help=help_text)
        sync_command.add_argument("--repos", type=_repository_list)
        if command_name == "fetch":
            sync_command.add_argument("--timeout-seconds", type=int, default=120)
        sync_command.add_argument("--json", action="store_true")

    open_parser = subcommands.add_parser("open", help="open a Task workspace")
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

    inspect_parser = subcommands.add_parser(
        "inspect",
        help="inspect branch, revision, dirty state and write scope",
    )
    inspect_parser.add_argument("task_id")
    inspect_parser.add_argument(
        "--against-run",
        dest="against_run_id",
        help="compare current Task sources with one exact recorded Run",
    )
    inspect_parser.add_argument("--json", action="store_true")

    doctor_parser = subcommands.add_parser(
        "doctor",
        help="verify repository-specific local environments and import provenance",
    )
    doctor_parser.add_argument(
        "--environment",
        action="append",
        dest="environment_ids",
        help="environment ID; repeatable, defaults to all protected local Profiles",
    )
    doctor_parser.add_argument("--json", action="store_true")

    test_parser = subcommands.add_parser(
        "test",
        help="run a trusted local R0/R1 Profile on a clean detached snapshot",
    )
    test_parser.add_argument("task_id")
    test_parser.add_argument("profile_id", nargs="?")
    test_parser.add_argument(
        "--recommended",
        action="store_true",
        help="preflight and run protected impact-rule recommendations",
    )
    test_parser.add_argument("--json", action="store_true")

    verify_parser = subcommands.add_parser(
        "verify",
        help="run clean Candidate evidence using the protected control plane",
    )
    verify_parser.add_argument("task_id")
    verify_parser.add_argument("profile_id", nargs="?")
    verify_parser.add_argument(
        "--recommended",
        action="store_true",
        help="preflight and verify protected impact-rule recommendations",
    )
    verify_parser.add_argument("--json", action="store_true")

    snapshot_parser = subcommands.add_parser(
        "snapshot",
        help="create or export a local synthetic draft Git snapshot",
    )
    snapshot_commands = snapshot_parser.add_subparsers(
        dest="snapshot_command", required=True
    )
    snapshot_create = snapshot_commands.add_parser(
        "create", help="capture working content without moving branch/index"
    )
    snapshot_create.add_argument("task_id")
    snapshot_create.add_argument(
        "--include",
        action="append",
        default=[],
        dest="include_untracked",
        help="allowlist untracked glob; repeatable, defaults to none",
    )
    snapshot_create.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
    )
    snapshot_create.add_argument("--json", action="store_true")
    snapshot_export = snapshot_commands.add_parser(
        "export", help="create a content-addressed incremental Git bundle set"
    )
    snapshot_export.add_argument("task_id")
    snapshot_export.add_argument("snapshot_id")
    snapshot_export.add_argument("--output", type=Path)
    snapshot_export.add_argument("--json", action="store_true")
    snapshot_prune = snapshot_commands.add_parser(
        "prune", help="inventory expired synthetic snapshots; apply only when requested"
    )
    snapshot_prune.add_argument("--task", dest="task_id")
    snapshot_prune.add_argument("--apply", action="store_true")
    snapshot_prune.add_argument("--json", action="store_true")

    runs_parser = subcommands.add_parser("runs", help="list recorded Runs")
    runs_parser.add_argument("task_id", nargs="?")
    runs_parser.add_argument("--json", action="store_true")

    remote_parser = subcommands.add_parser(
        "remote",
        help="plan and inspect exact remote requests without implicit dispatch",
    )
    remote_commands = remote_parser.add_subparsers(
        dest="remote_command",
        required=True,
    )
    remote_plan = remote_commands.add_parser(
        "plan",
        help="resolve a clean Task into a PENDING_DISPATCH outbox envelope",
    )
    remote_plan.add_argument("task_id")
    remote_plan.add_argument("profile_id")
    remote_plan.add_argument("--json", action="store_true")
    remote_status = remote_commands.add_parser("status")
    remote_status.add_argument("request_id")
    remote_status.add_argument("--json", action="store_true")
    remote_list = remote_commands.add_parser("list")
    remote_list.add_argument("task_id", nargs="?")
    remote_list.add_argument("--json", action="store_true")
    remote_doctor = remote_commands.add_parser(
        "doctor",
        help="report protected asset/transport readiness without network access",
    )
    remote_doctor.add_argument("profile_id")
    remote_doctor.add_argument("--json", action="store_true")
    remote_trust = remote_commands.add_parser(
        "trust",
        help="install an already-pinned host key into Lab-owned known_hosts",
    )
    remote_trust.add_argument("--transport", required=True)
    remote_trust.add_argument("--source-known-hosts", type=Path)
    remote_trust.add_argument("--json", action="store_true")
    remote_dispatch = remote_commands.add_parser(
        "dispatch",
        help="send an outbox envelope over pinned SSH and record its acknowledgement",
    )
    remote_dispatch.add_argument("request_id")
    remote_dispatch.add_argument("--transport", required=True)
    remote_dispatch.add_argument("--timeout-seconds", type=int, default=30)
    remote_dispatch.add_argument("--dry-run", action="store_true")
    remote_dispatch.add_argument("--json", action="store_true")
    for command_name, help_text in (
        ("refresh", "fetch and persist the authenticated remote request state"),
        ("cancel", "request cancellation without claiming executor teardown"),
    ):
        control = remote_commands.add_parser(command_name, help=help_text)
        control.add_argument("request_id")
        control.add_argument("--transport", required=True)
        control.add_argument("--timeout-seconds", type=int, default=30)
        control.add_argument("--json", action="store_true")
    remote_watch = remote_commands.add_parser(
        "watch",
        help="wait for authenticated remote state to advance and persist it",
    )
    remote_watch.add_argument("request_id")
    remote_watch.add_argument("--transport", required=True)
    remote_watch.add_argument("--after-sequence", type=int)
    remote_watch.add_argument("--timeout-seconds", type=float, default=30.0)
    remote_watch.add_argument("--poll-interval-seconds", type=float, default=0.25)
    remote_watch.add_argument("--json", action="store_true")
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


def _print_create_result(payload: dict[str, object], *, dry_run: bool) -> None:
    prefix = "Would create" if dry_run else "Created"
    print(f"{prefix} Task {payload['task_id']} at {payload['workspace_root']}")
    repositories = payload.get("repositories", {})
    if isinstance(repositories, dict):
        for name, raw_repository in repositories.items():
            if not isinstance(raw_repository, dict):
                continue
            print(
                f"  {name}: {raw_repository.get('branch')} "
                f"@ {str(raw_repository.get('base_commit', ''))[:12]}"
            )


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
    expires_at = payload.get("expires_at")
    if expires_at:
        print(f"Expires: {expires_at}")


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


def _print_inspection(payload: dict[str, object]) -> None:
    print(f"Task {payload['task_id']}: {payload.get('title') or '(untitled)'}")
    print(f"Development host: {payload.get('development_host')}")
    print(f"Status: {payload.get('status')}")
    print(f"Workspace: {payload.get('workspace_root')}")
    print()
    repositories = payload.get("repositories", {})
    if isinstance(repositories, dict):
        for name, raw_repository in repositories.items():
            if not isinstance(raw_repository, dict):
                continue
            state = "dirty" if raw_repository.get("dirty") else "clean"
            branch_marker = (
                "" if raw_repository.get("branch_matches") else " [BRANCH DRIFT]"
            )
            violations = raw_repository.get("scope_violations", [])
            print(
                f"{name}: {raw_repository.get('branch')} "
                f"@ {str(raw_repository.get('commit', ''))[:12]} {state}{branch_marker}"
            )
            print(f"  path: {raw_repository.get('path')}")
            print(f"  scope: {', '.join(raw_repository.get('write_scope', []))}")
            print(f"  changed files: {len(raw_repository.get('changed_paths', []))}")
            writer_lease = raw_repository.get("writer_lease")
            if isinstance(writer_lease, dict):
                lease = writer_lease.get("lease")
                actor = lease.get("actor") if isinstance(lease, dict) else None
                actor_suffix = f" ({actor})" if actor else ""
                print(f"  writer: {writer_lease.get('state')}{actor_suffix}")
            if violations:
                print(f"  scope violations: {', '.join(violations)}")
    print()
    readiness = payload.get("readiness")
    if isinstance(readiness, dict):
        print(f"Test readiness: {readiness.get('status')}")
        recommended = readiness.get("recommended_profiles")
        if isinstance(recommended, list):
            print(
                "Recommended profiles: "
                + (", ".join(str(value) for value in recommended) or "none")
            )
        profiles = readiness.get("profiles")
        if isinstance(profiles, dict):
            for profile_id, raw_profile in profiles.items():
                if isinstance(raw_profile, dict):
                    required = (
                        "required"
                        if raw_profile.get("required_after_changes")
                        else "optional"
                    )
                    print(
                        f"  {profile_id}: {raw_profile.get('evidence_freshness')} "
                        f"({required})"
                    )
        print()
    comparison = payload.get("against_run")
    if isinstance(comparison, dict):
        print(
            f"Against Run: {comparison.get('run_id')} "
            f"({comparison.get('classification')})"
        )
        repositories = comparison.get("repositories")
        if isinstance(repositories, dict):
            for name, raw_repository in repositories.items():
                if not isinstance(raw_repository, dict):
                    continue
                if raw_repository.get("tested") is not True:
                    print(f"  {name}: not tested")
                    continue
                print(
                    f"  {name}: {len(raw_repository.get('changed_paths', []))} "
                    "changed paths"
                )
        print()
    print(f"Next action: {payload.get('next_action')}")


def _print_change(payload: dict[str, object]) -> None:
    print(f"Change {payload.get('change_id')}: {payload.get('title') or '(untitled)'}")
    print(f"Status: {payload.get('status')}")
    print(f"Tasks: {payload.get('task_count', 0)}")
    tasks = payload.get("tasks")
    if isinstance(tasks, dict):
        for task_id, task in tasks.items():
            if isinstance(task, dict):
                print(
                    f"  {task_id}: {task.get('development_host')} {task.get('status')}"
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


def _print_run(payload: dict[str, object]) -> None:
    outcome = payload.get("outcome")
    classification = (
        outcome.get("classification") if isinstance(outcome, dict) else "UNKNOWN"
    )
    print(f"Run: {payload.get('run_id')}")
    print(f"Task: {payload.get('task_id')}")
    profile = payload.get("profile")
    if isinstance(profile, dict):
        print(f"Profile: {profile.get('profile_id')}")
    print(f"Evidence: {payload.get('evidence_tier')}")
    print(f"Result: {classification}")
    print(f"Snapshot: {payload.get('snapshot_id')}")
    run_id = payload.get("run_id")
    task_id = payload.get("task_id")
    if isinstance(run_id, str) and isinstance(task_id, str):
        print(f"Manifest: Runtime/tasks/{task_id}/{run_id}/run.yaml")


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
) -> tuple[list[str], dict[str, object]]:
    inspection = enrich_task_inspection(manager, manager.inspect_task(task_id))
    if inspection.get("status") != "ACTIVE":
        raise LabError("recommended tests require an ACTIVE Task without drift")
    repositories = inspection.get("repositories")
    if not isinstance(repositories, dict):
        raise LabError("Task inspection lacks repositories")
    dirty_repositories = sorted(
        name
        for name, raw_repository in repositories.items()
        if isinstance(raw_repository, dict) and raw_repository.get("dirty") is True
    )
    if dirty_repositories:
        raise LabError(
            "recommended clean tests require checkpointed/committed repositories: "
            + ", ".join(dirty_repositories)
        )
    readiness = inspection.get("readiness")
    if not isinstance(readiness, dict):
        raise LabError("Task inspection lacks test readiness")
    if readiness.get("status") != "READY":
        errors = readiness.get("profile_errors")
        detail = (
            "; ".join(str(error) for error in errors)
            if isinstance(errors, list) and errors
            else "required Profiles or baseline inputs are unavailable"
        )
        raise LabError(f"recommended test readiness requires attention: {detail}")
    recommended = readiness.get("recommended_profiles")
    if not isinstance(recommended, list) or not all(
        isinstance(profile_id, str) for profile_id in recommended
    ):
        raise LabError("Task inspection has invalid recommended Profiles")
    return list(recommended), inspection


def _run_recommended_profiles(
    manager: WorkspaceManager,
    task_id: str,
    *,
    evidence_tier: str = DEVELOPMENT_EVIDENCE_TIER,
) -> dict[str, object]:
    profile_ids, inspection = _recommended_profiles(manager, task_id)
    readiness = inspection.get("readiness")
    impact = readiness.get("impact") if isinstance(readiness, dict) else None
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
        "impact": impact,
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
    print("Recommended Profiles: " + ", ".join(str(value) for value in profiles))
    runs = payload.get("runs")
    if isinstance(runs, list):
        for run in runs:
            if isinstance(run, dict):
                _print_run(run)
    print(f"All passed: {str(payload.get('all_passed')).lower()}")


def _print_remote_outbox(payload: dict[str, object]) -> None:
    dispatch = payload.get("dispatch")
    status = dispatch.get("status") if isinstance(dispatch, dict) else "UNKNOWN"
    profile = payload.get("profile")
    profile_id = profile.get("profile_id") if isinstance(profile, dict) else "?"
    print(f"Request: {payload.get('request_id')}")
    print(f"Task: {payload.get('task_id')}")
    print(f"Profile: {profile_id}")
    print(f"Snapshot: {payload.get('snapshot_id')}")
    print(f"Status: {status}")
    print("Remote acknowledged: false")
    print("Execution eligible: false")


def _print_remote_outbox_list(payloads: list[dict[str, object]]) -> None:
    if not payloads:
        print("No remote outbox requests found.")
        return
    for payload in payloads:
        print(
            f"{payload.get('request_id')}  {payload.get('task_id')}  "
            f"{payload.get('status')}"
        )


def _print_remote_status(payload: dict[str, object]) -> None:
    print(f"Request: {payload.get('request_id')}")
    print(f"Task: {payload.get('task_id')}")
    print(f"Snapshot: {payload.get('snapshot_id')}")
    print(f"Status: {payload.get('status')}")
    acknowledged = str(payload.get("remote_acknowledged")).lower()
    print(f"Remote acknowledged: {acknowledged}")
    observation = payload.get("remote_observation")
    if isinstance(observation, dict):
        remote_state = observation.get("remote_state")
        if isinstance(remote_state, dict):
            print(
                "Remote state: "
                f"{remote_state.get('state')} "
                f"(sequence {observation.get('remote_state_sequence')})"
            )
    print("Execution eligible: false")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        manager = WorkspaceManager(control_repository=_control_repository())
        if arguments.command == "change":
            if arguments.change_command == "create":
                payload = manager.create_change(
                    arguments.change_id,
                    title=arguments.title,
                )
                if arguments.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print(f"Created Change {payload['change_id']}")
                return 0
            payload = manager.inspect_change(arguments.change_id)
            payload = enrich_change_inspection(manager, payload)
            if arguments.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
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
            payload = _plan_document(plan)
            if not arguments.dry_run:
                payload = manager.create_task(plan)
            if arguments.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_create_result(payload, dry_run=arguments.dry_run)
            return 0

        if arguments.command == "task" and arguments.task_command == "writer":
            writer_manager = WriterLeaseManager(manager)
            if arguments.writer_command == "status":
                payload = (
                    writer_manager.inspect_repository(
                        arguments.task_id,
                        arguments.repo,
                    )
                    if arguments.repo
                    else writer_manager.inspect_task(arguments.task_id)
                )
            elif arguments.writer_command == "claim":
                payload = writer_manager.claim(
                    arguments.task_id,
                    arguments.repo,
                    actor=arguments.actor,
                    ttl_seconds=arguments.ttl_seconds,
                    replace_expired=arguments.replace_expired,
                )
            elif arguments.writer_command == "renew":
                payload = writer_manager.renew(
                    arguments.task_id,
                    arguments.repo,
                    actor=arguments.actor,
                    lease_id=arguments.lease_id,
                    ttl_seconds=arguments.ttl_seconds,
                )
            else:
                payload = writer_manager.release(
                    arguments.task_id,
                    arguments.repo,
                    actor=arguments.actor,
                    lease_id=arguments.lease_id,
                )
            if arguments.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
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
                print(json.dumps(sync_payload, indent=2, sort_keys=True))
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
            configured_editor = (
                arguments.editor
                or os.environ.get("LAB_EDITOR")
                or "code -n"
            )
            editor_command = shlex.split(configured_editor)
            if not editor_command:
                raise LabError("editor command is empty")
            completed = subprocess.run([*editor_command, str(workspace)], check=False)
            if completed.returncode != 0:
                raise LabError(
                    f"editor exited with status {completed.returncode}: {configured_editor}"
                )
            return 0

        if arguments.command == "inspect":
            if manager.task_manifest_path(arguments.task_id).is_file():
                payload = manager.inspect_task(arguments.task_id)
                payload = enrich_task_inspection(manager, payload)
                if arguments.against_run_id:
                    payload = attach_run_comparison(
                        payload,
                        RunManager(manager).find_run(arguments.against_run_id),
                    )
                is_change = False
            else:
                if arguments.against_run_id:
                    raise LabError("--against-run requires a Task, not a Change")
                payload = manager.inspect_change(arguments.task_id)
                payload = enrich_change_inspection(manager, payload)
                is_change = True
            if arguments.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            elif is_change:
                _print_change(payload)
            else:
                _print_inspection(payload)
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
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"Status: {payload['status']}")
                print("Source overlay:")
                overlay = payload.get("source_overlay")
                if isinstance(overlay, list):
                    for path in overlay:
                        print(f"  {path}")
                receipts = payload.get("environments", {})
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
            return 0 if payload["status"] == "READY" else 1

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
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    _print_recommended_batch(payload)
                return 0 if payload.get("all_passed") is True else 1
            payload = RunManager(manager).run_clean_profile(
                arguments.task_id,
                arguments.profile_id,
                evidence_tier=evidence_tier,
            )
            if arguments.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_run(payload)
            outcome = payload.get("outcome")
            return (
                0
                if isinstance(outcome, dict)
                and outcome.get("classification") == "PASSED"
                else 1
            )

        if arguments.command == "snapshot":
            if arguments.snapshot_command == "prune":
                pruner = DraftSnapshotPruner(manager)
                payload = (
                    pruner.prune(task_id=arguments.task_id)
                    if arguments.apply
                    else pruner.inventory(task_id=arguments.task_id)
                )
            elif arguments.snapshot_command == "create":
                payload = DraftSnapshotManager(manager).create(
                    arguments.task_id,
                    include_untracked=arguments.include_untracked,
                    max_file_bytes=arguments.max_file_bytes,
                )
            else:
                output_root = (
                    arguments.output.expanduser().resolve()
                    if arguments.output is not None
                    else manager.state_root / "draft-exports"
                )
                payload = export_draft_bundle(
                    manager,
                    arguments.task_id,
                    arguments.snapshot_id,
                    output_root,
                )
            if arguments.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            elif arguments.snapshot_command == "export":
                bundle = payload.get("bundle")
                bundle_id = (
                    bundle.get("bundle_set_id") if isinstance(bundle, dict) else None
                )
                print(f"Bundle: {bundle_id}")
                print(f"Snapshot: {arguments.snapshot_id}")
                print(f"Path: {payload.get('path')}")
                print("Remote-run/verify/device eligible: false")
            elif arguments.snapshot_command == "create":
                print(f"Snapshot: {payload.get('snapshot_id')}")
                print(f"Task: {payload.get('task_id')}")
                print("Type: synthetic-draft-git")
                print("Remote/verify/device eligible: false")
            else:
                action = "Retired" if payload.get("applied") else "Would retire"
                print(f"{action}: {payload.get('expired_count')} expired snapshots")
                print("External export directories are not managed by this command.")
            return 0

        if arguments.command == "runs":
            payloads = RunManager(manager).list_runs(arguments.task_id)
            if arguments.json:
                print(json.dumps(payloads, indent=2, sort_keys=True))
            else:
                _print_run_list(payloads)
            return 0

        if arguments.command == "remote":
            planner = RemoteDispatchPlanner(manager)
            if arguments.remote_command == "trust":
                transport_profile = load_transport_profile(
                    planner.transport_profile_root,
                    arguments.transport,
                )
                source = (
                    arguments.source_known_hosts
                    if arguments.source_known_hosts is not None
                    else Path.home() / ".ssh" / "known_hosts"
                )
                target = Path(
                    manager.environ.get(
                        "LAB_KNOWN_HOSTS_FILE",
                        manager.state_root / "trust" / "known_hosts",
                    )
                ).expanduser()
                payload = install_pinned_known_host(
                    transport_profile,
                    source_known_hosts=source,
                    target_known_hosts=target,
                )
                if arguments.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print(f"Transport: {transport_profile.transport_id}")
                    print(f"Fingerprint: {payload['host_key_fingerprint']}")
                    print(f"Known hosts: {payload['target_known_hosts']}")
                    print(f"Changed: {str(payload['changed']).lower()}")
                    print("Network used: false")
                return 0
            if arguments.remote_command == "doctor":
                payload = planner.doctor(arguments.profile_id)
                if arguments.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print(f"Status: {payload.get('status')}")
                    print(f"Planning ready: {payload.get('planning_ready')}")
                    print(f"Dispatch ready: {payload.get('dispatch_ready')}")
                    blockers = payload.get("blockers")
                    if isinstance(blockers, list):
                        for blocker in blockers:
                            print(f"BLOCKED: {blocker}")
                return 0 if payload.get("status") == "READY" else 1
            if arguments.remote_command == "dispatch":
                envelope = planner.load(arguments.request_id)
                transport_profile = load_transport_profile(
                    planner.transport_profile_root,
                    arguments.transport,
                )
                known_hosts = Path(
                    manager.environ.get(
                        "LAB_KNOWN_HOSTS_FILE",
                        manager.state_root / "trust" / "known_hosts",
                    )
                ).expanduser()
                ssh_binary = Path(
                    manager.environ.get("LAB_SSH_BIN", "/usr/bin/ssh")
                ).expanduser()
                if arguments.dry_run:
                    argv, fingerprint = build_ssh_dispatch_argv(
                        envelope=envelope,
                        profile=transport_profile,
                        known_hosts_path=known_hosts,
                        ssh_binary=ssh_binary,
                    )
                    payload = {
                        "schema": "lab.ssh-dispatch-plan.v1",
                        "request_id": arguments.request_id,
                        "transport_id": transport_profile.transport_id,
                        "host_key_fingerprint": fingerprint,
                        "argv": argv,
                        "stdin": "outbox-envelope-json",
                        "network_used": False,
                        "execution_eligible": False,
                    }
                else:
                    acceptance, transport_receipt = dispatch_remote_outbox(
                        envelope=envelope,
                        profile=transport_profile,
                        known_hosts_path=known_hosts,
                        timeout_seconds=arguments.timeout_seconds,
                        ssh_binary=ssh_binary,
                    )
                    payload = planner.attach_acknowledgement(
                        arguments.request_id,
                        acceptance=acceptance,
                        transport_receipt=transport_receipt,
                    )
                if arguments.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                elif arguments.dry_run:
                    print(f"Request: {arguments.request_id}")
                    print(f"Transport: {transport_profile.transport_id}")
                    print("Dry run: true")
                    print("Network used: false")
                else:
                    _print_remote_status(payload)
                return 0
            if arguments.remote_command in {"refresh", "watch", "cancel"}:
                local_status = planner.status(arguments.request_id)
                if local_status.get("remote_acknowledged") is not True:
                    raise LabError(
                        "remote control requires an authenticated dispatch acknowledgement"
                    )
                envelope = planner.load(arguments.request_id)
                transport_profile = load_transport_profile(
                    planner.transport_profile_root,
                    arguments.transport,
                )
                known_hosts = Path(
                    manager.environ.get(
                        "LAB_KNOWN_HOSTS_FILE",
                        manager.state_root / "trust" / "known_hosts",
                    )
                ).expanduser()
                ssh_binary = Path(
                    manager.environ.get("LAB_SSH_BIN", "/usr/bin/ssh")
                ).expanduser()
                action = (
                    "status"
                    if arguments.remote_command == "refresh"
                    else arguments.remote_command
                )
                after_sequence = -1
                watch_timeout = 0.0
                poll_interval = 0.25
                ssh_timeout = arguments.timeout_seconds
                if arguments.remote_command == "watch":
                    observation = local_status.get("remote_observation")
                    latest_sequence = (
                        observation.get("remote_state_sequence")
                        if isinstance(observation, dict)
                        else -1
                    )
                    if not isinstance(latest_sequence, int):
                        latest_sequence = -1
                    after_sequence = (
                        arguments.after_sequence
                        if arguments.after_sequence is not None
                        else latest_sequence
                    )
                    watch_timeout = arguments.timeout_seconds
                    poll_interval = arguments.poll_interval_seconds
                    ssh_timeout = int(watch_timeout) + 6
                remote_state, fingerprint = control_remote_request(
                    envelope=envelope,
                    profile=transport_profile,
                    known_hosts_path=known_hosts,
                    action=action,
                    after_sequence=after_sequence,
                    watch_timeout_seconds=watch_timeout,
                    poll_interval_seconds=poll_interval,
                    ssh_timeout_seconds=ssh_timeout,
                    ssh_binary=ssh_binary,
                )
                payload = planner.attach_remote_state(
                    arguments.request_id,
                    remote_state=remote_state,
                    transport_id=transport_profile.transport_id,
                    observed_host_key_fingerprint=fingerprint,
                )
                if arguments.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    _print_remote_status(payload)
                    if arguments.remote_command == "cancel":
                        print(
                            "Cancellation is only a remote request; hardware safety "
                            "is not claimed."
                        )
                return 0
            if arguments.remote_command == "plan":
                payload = planner.plan(arguments.task_id, arguments.profile_id)
                if arguments.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    _print_remote_outbox(payload)
                return 0
            if arguments.remote_command == "status":
                payload = planner.status(arguments.request_id)
                if arguments.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    _print_remote_status(payload)
                return 0
            payloads = planner.list_status(arguments.task_id)
            if arguments.json:
                print(json.dumps(payloads, indent=2, sort_keys=True))
            else:
                _print_remote_outbox_list(payloads)
            return 0
    except LabError as exc:
        print(f"lab: error: {exc}", file=sys.stderr)
        return 2

    parser.error("unhandled command")
    return 2
