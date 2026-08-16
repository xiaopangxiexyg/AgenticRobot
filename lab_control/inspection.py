from __future__ import annotations

from pathlib import Path

from . import git
from .baseline import BaselineLock, load_baseline_lock
from .impact_rules import load_impact_rules, recommend_profiles
from .profiles import TestProfile, load_profile
from .runs import RunManager
from .workspace import LabError, WorkspaceManager


def _load_profiles(
    profile_root: Path,
    *,
    excluded_paths: tuple[Path, ...] = (),
) -> tuple[list[TestProfile], list[str]]:
    profiles: list[TestProfile] = []
    errors: list[str] = []
    if not profile_root.is_dir():
        return profiles, errors
    excluded = {path.resolve() for path in excluded_paths}
    for path in sorted(profile_root.glob("*.json")):
        if path.resolve() in excluded:
            continue
        try:
            profiles.append(load_profile(profile_root, path.stem))
        except LabError as exc:
            errors.append(str(exc))
    return profiles, errors


def _load_optional_baseline(path: Path) -> tuple[BaselineLock | None, str | None]:
    if not path.is_file():
        return None, f"baseline lock is missing: {path}"
    try:
        return load_baseline_lock(path), None
    except LabError as exc:
        return None, str(exc)


def _latest_runs_by_profile(
    runs: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for run in runs:
        raw_profile = run.get("profile")
        profile_id = (
            raw_profile.get("profile_id") if isinstance(raw_profile, dict) else None
        )
        if isinstance(profile_id, str) and profile_id not in latest:
            latest[profile_id] = run
    return latest


def _profile_availability(
    profile: TestProfile,
    *,
    workspace_manager: WorkspaceManager,
    task_repositories: dict[str, object],
    baseline: BaselineLock | None,
    baseline_error: str | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for repository in profile.repositories:
        if repository not in workspace_manager.catalog:
            reasons.append(f"repository is not configured locally: {repository}")
        if repository in task_repositories:
            continue
        if baseline is None:
            reasons.append(baseline_error or "baseline lock is unavailable")
            continue
        pin = baseline.repositories.get(repository)
        if pin is None:
            reasons.append(
                f"baseline {baseline.baseline_id} does not pin {repository}"
            )
            continue
        configured = workspace_manager.catalog.get(repository)
        if configured is None:
            continue
        try:
            root = git.repository_root(configured)
            identity = git.origin_identity(root)
            commit = git.resolve_commit(root, pin.commit)
            tree_oid = git.tree_oid(root, commit)
        except git.GitError as exc:
            reasons.append(f"baseline {repository} is unavailable locally: {exc}")
            continue
        if identity != pin.canonical_identity:
            reasons.append(
                f"baseline {repository} identity mismatch: "
                f"{identity} != {pin.canonical_identity}"
            )
        if tree_oid != pin.tree_oid:
            reasons.append(f"baseline {repository} tree does not match its lock")
    return not reasons, reasons


def _evidence_freshness(
    profile: TestProfile,
    latest_run: dict[str, object] | None,
    *,
    task_repositories: dict[str, object],
    baseline: BaselineLock | None,
) -> tuple[str, list[str], dict[str, object] | None]:
    if latest_run is None:
        return "NOT_RUN", ["profile has no recorded Run"], None
    reasons: list[str] = []
    raw_run_profile = latest_run.get("profile")
    if (
        not isinstance(raw_run_profile, dict)
        or raw_run_profile.get("sha256") != profile.sha256
    ):
        reasons.append("Profile changed after the Run")
    outcome = latest_run.get("outcome")
    classification = (
        outcome.get("classification") if isinstance(outcome, dict) else None
    )
    if classification != "PASSED":
        reasons.append(f"latest Run result is {classification or 'INCOMPLETE'}")
    run_repositories = latest_run.get("repositories")
    if not isinstance(run_repositories, dict):
        reasons.append("latest Run lacks repository identities")
        run_repositories = {}
    for repository in profile.repositories:
        current = task_repositories.get(repository)
        tested = run_repositories.get(repository)
        if isinstance(current, dict):
            if current.get("dirty") is True:
                reasons.append(f"{repository} has uncommitted changes after the Run")
            if not isinstance(tested, dict) or any(
                tested.get(field) != current.get(field)
                for field in ("commit", "tree_oid")
            ):
                reasons.append(f"{repository} revision differs from the Run")
    run_baseline = latest_run.get("baseline")
    if isinstance(run_baseline, dict) and (
        baseline is None or run_baseline.get("sha256") != baseline.sha256
    ):
        reasons.append("baseline lock changed or is unavailable")
    freshness = "CURRENT" if not reasons else "STALE"
    if classification not in {None, "PASSED"}:
        freshness = "FAILED"
    summary = {
        "run_id": latest_run.get("run_id"),
        "snapshot_id": latest_run.get("snapshot_id"),
        "classification": classification,
        "started_at": latest_run.get("started_at"),
        "evidence_tier": latest_run.get("evidence_tier"),
    }
    return freshness, reasons, summary


def enrich_task_inspection(
    workspace_manager: WorkspaceManager,
    inspection: dict[str, object],
) -> dict[str, object]:
    """Add Profile readiness and Run freshness without running tests or environments."""

    task_id = str(inspection.get("task_id", ""))
    raw_repositories = inspection.get("repositories")
    if not isinstance(raw_repositories, dict):
        raise LabError(f"task inspection has no repositories: {task_id}")
    task_repositories = raw_repositories
    run_manager = RunManager(workspace_manager)
    configured_impact_path = workspace_manager.environ.get(
        "LAB_IMPACT_RULES"
    )
    impact_path = (
        Path(configured_impact_path).expanduser().resolve()
        if configured_impact_path
        else (
            run_manager.profile_root / "impact-rules.json"
            if workspace_manager.environ.get("LAB_PROFILE_ROOT")
            else workspace_manager.control_repository / "stack" / "impact-rules.json"
        )
    )
    profiles, profile_errors = _load_profiles(
        run_manager.profile_root,
        excluded_paths=(impact_path,),
    )
    baseline, baseline_error = _load_optional_baseline(run_manager.baseline_path)
    latest_runs = _latest_runs_by_profile(run_manager.list_runs(task_id))
    changed_repositories = sorted(
        name
        for name, raw_repository in task_repositories.items()
        if isinstance(raw_repository, dict)
        and bool(raw_repository.get("changed_paths"))
    )
    changed_paths = {
        name: tuple(str(path) for path in raw_repository.get("changed_paths", []))
        for name, raw_repository in task_repositories.items()
        if isinstance(name, str)
        and isinstance(raw_repository, dict)
        and raw_repository.get("changed_paths")
    }
    known_profile_ids = {profile.profile_id for profile in profiles}
    required_profiles: set[str]
    impact_summary: dict[str, object]
    if impact_path.is_file():
        try:
            impact_rules = load_impact_rules(impact_path)
            impact = recommend_profiles(
                impact_rules,
                changed_paths,
                known_profiles=known_profile_ids,
            )
            required_profiles = set(impact.profiles)
            if impact.unknown_profiles:
                profile_errors.append(
                    "impact rules reference unknown Profiles: "
                    + ", ".join(impact.unknown_profiles)
                )
            if impact.uncovered_paths:
                profile_errors.append(
                    "impact rules have uncovered changed paths: "
                    + ", ".join(
                        f"{repository}:{path}"
                        for repository, paths in impact.uncovered_paths.items()
                        for path in paths
                    )
                )
            impact_summary = {
                "mode": "protected-path-rules",
                "path": str(impact_rules.path),
                "sha256": impact_rules.sha256,
                "matched_rules": list(impact.matched_rules),
                "fallback_repositories": list(impact.fallback_repositories),
                "uncovered_paths": {
                    repository: list(paths)
                    for repository, paths in impact.uncovered_paths.items()
                },
                "unknown_profiles": list(impact.unknown_profiles),
            }
        except LabError as exc:
            profile_errors.append(str(exc))
            required_profiles = {
                profile.profile_id
                for profile in profiles
                if set(profile.repositories).intersection(changed_repositories)
            }
            impact_summary = {
                "mode": "repository-fallback-after-invalid-rules",
                "path": str(impact_path),
                "error": str(exc),
            }
    else:
        required_profiles = {
            profile.profile_id
            for profile in profiles
            if set(profile.repositories).intersection(changed_repositories)
        }
        impact_summary = {
            "mode": "repository-fallback",
            "path": str(impact_path),
        }
    profile_status: dict[str, object] = {}
    recommended: list[str] = []
    for profile in profiles:
        available, availability_reasons = _profile_availability(
            profile,
            workspace_manager=workspace_manager,
            task_repositories=task_repositories,
            baseline=baseline,
            baseline_error=baseline_error,
        )
        impacted = profile.profile_id in required_profiles
        freshness, evidence_reasons, latest_run = _evidence_freshness(
            profile,
            latest_runs.get(profile.profile_id),
            task_repositories=task_repositories,
            baseline=baseline,
        )
        if impacted and available:
            recommended.append(profile.profile_id)
        profile_status[profile.profile_id] = {
            "repositories": list(profile.repositories),
            "risk_level": profile.risk_level,
            "available": available,
            "availability_reasons": availability_reasons,
            "required_after_changes": impacted,
            "evidence_freshness": freshness,
            "evidence_reasons": evidence_reasons,
            "latest_run": latest_run,
            "command": f"lab test {task_id} {profile.profile_id}",
        }

    readiness_status = "READY"
    if profile_errors or any(
        isinstance(value, dict)
        and value.get("required_after_changes") is True
        and value.get("available") is not True
        for value in profile_status.values()
    ):
        readiness_status = "ATTENTION"
    enriched = dict(inspection)
    enriched["readiness"] = {
        "status": readiness_status,
        "environment": {
            "status": "DEFERRED",
            "command": "lab doctor --json",
        },
        "changed_repositories": changed_repositories,
        "recommended_profiles": recommended,
        "profiles": profile_status,
        "profile_errors": profile_errors,
        "impact": impact_summary,
    }
    if inspection.get("status") == "DRIFTED":
        return enriched
    if recommended:
        dirty = any(
            isinstance(raw_repository, dict)
            and raw_repository.get("dirty") is True
            for raw_repository in task_repositories.values()
        )
        prefix = "Checkpoint/commit Task changes, then run" if dirty else "Run"
        enriched["next_action"] = f"{prefix}: {', '.join(recommended)}."
    elif changed_repositories:
        enriched["next_action"] = (
            "No registered local Profile covers the changed repositories; "
            "review impact rules before verification."
        )
    else:
        enriched["next_action"] = (
            "No Task changes detected; run lab doctor before starting development."
        )
    return enriched


def attach_run_comparison(
    inspection: dict[str, object],
    run: dict[str, object],
) -> dict[str, object]:
    """Compare current Task worktrees with the exact source identities in one Run."""

    task_id = inspection.get("task_id")
    if run.get("task_id") != task_id:
        raise LabError(
            f"Run {run.get('run_id')} belongs to {run.get('task_id')}, not {task_id}"
        )
    task_repositories = inspection.get("repositories")
    run_repositories = run.get("repositories")
    if not isinstance(task_repositories, dict) or not isinstance(
        run_repositories, dict
    ):
        raise LabError("Task or Run lacks repository identities for comparison")
    comparisons: dict[str, object] = {}
    for name, raw_current in task_repositories.items():
        if not isinstance(name, str) or not isinstance(raw_current, dict):
            raise LabError("Task inspection has an invalid repository entry")
        raw_tested = run_repositories.get(name)
        if not isinstance(raw_tested, dict):
            comparisons[name] = {
                "tested": False,
                "reason": "repository was not part of the Run",
            }
            continue
        tested_commit = raw_tested.get("commit")
        if not isinstance(tested_commit, str):
            raise LabError(f"Run repository lacks tested commit: {name}")
        repository_path = Path(str(raw_current.get("path", "")))
        try:
            resolved_tested = git.resolve_commit(repository_path, tested_commit)
            changed_paths = git.changed_paths(repository_path, resolved_tested)
        except git.GitError as exc:
            raise LabError(f"cannot compare {name} with Run: {exc}") from exc
        comparisons[name] = {
            "tested": True,
            "tested_commit": resolved_tested,
            "tested_tree_oid": raw_tested.get("tree_oid"),
            "current_commit": raw_current.get("commit"),
            "current_tree_oid": raw_current.get("tree_oid"),
            "current_dirty": raw_current.get("dirty"),
            "changed_paths": list(changed_paths),
        }
    enriched = dict(inspection)
    raw_profile = run.get("profile")
    raw_outcome = run.get("outcome")
    enriched["against_run"] = {
        "run_id": run.get("run_id"),
        "snapshot_id": run.get("snapshot_id"),
        "profile_id": (
            raw_profile.get("profile_id") if isinstance(raw_profile, dict) else None
        ),
        "classification": (
            raw_outcome.get("classification")
            if isinstance(raw_outcome, dict)
            else None
        ),
        "repositories": comparisons,
    }
    return enriched


def enrich_change_inspection(
    workspace_manager: WorkspaceManager,
    inspection: dict[str, object],
) -> dict[str, object]:
    """Aggregate child Task readiness without merging their writer identities."""

    raw_tasks = inspection.get("tasks")
    if not isinstance(raw_tasks, dict):
        raise LabError("Change inspection has no child Tasks")
    tasks: dict[str, object] = {}
    recommendations: list[dict[str, str]] = []
    readiness_status = "READY"
    for task_id, raw_task in raw_tasks.items():
        if not isinstance(task_id, str) or not isinstance(raw_task, dict):
            raise LabError("Change inspection has an invalid child Task")
        enriched_task = enrich_task_inspection(workspace_manager, raw_task)
        tasks[task_id] = enriched_task
        readiness = enriched_task.get("readiness")
        if enriched_task.get("status") != "ACTIVE" or (
            isinstance(readiness, dict) and readiness.get("status") != "READY"
        ):
            readiness_status = "ATTENTION"
        if isinstance(readiness, dict):
            profiles = readiness.get("recommended_profiles")
            if isinstance(profiles, list):
                recommendations.extend(
                    {"task_id": task_id, "profile_id": profile_id}
                    for profile_id in profiles
                    if isinstance(profile_id, str)
                )
    enriched = dict(inspection)
    enriched["tasks"] = tasks
    enriched["readiness"] = {
        "status": readiness_status,
        "recommended": recommendations,
    }
    if inspection.get("status") == "ACTIVE" and recommendations:
        enriched["next_action"] = "Run the recommended child Task Profiles."
    return enriched
