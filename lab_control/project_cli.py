from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .asset_lock import load_asset_lock
from .asset_observation import AssetObservationSet, load_asset_observations
from .asset_selection import (
    build_runtime_asset_verification,
    build_selected_asset_lock,
    collect_artifact_manifest,
    load_workload_selection,
    validate_artifact_manifest,
)
from .control_plane_activation import (
    activate_control_plane_version,
    rollback_control_plane_version,
)
from .control_plane_package import build_control_plane_package
from .control_plane_stage import stage_control_plane_package
from .draft_bundle import verify_draft_bundle_set
from .execution_sandbox import (
    build_service_sandbox_receipt,
    collect_sandbox_backend_probe,
)
from .gpu_lease import (
    GpuLeaseStore,
    collect_nvidia_compute_process_inventory,
    collect_nvidia_gpu_inventory,
    select_single_gpu,
)
from .host_observation import (
    build_runtime_receipts,
    collect_host_observation,
    load_runtime_receipts,
)
from .managed_service import build_attested_service_receipt
from .preparation_profile import (
    PreparationProfile,
    SourceAnchorPlacement,
    load_preparation_profile,
)
from .remote_inbox import RemoteInbox
from .remote_preflight import load_host_profile, preflight_remote_request
from .remote_request import (
    build_remote_request,
    load_remote_profile,
    validate_remote_request,
)
from .remote_runner import (
    prepare_remote_request,
    stage_remote_request_readiness,
    stage_remote_request_sources,
    stage_remote_service_launch_plan,
    stage_remote_service_launch_spec,
    stage_remote_service_receipt,
)
from .resource_reservation import validate_gpu_resource_reservation
from .runner_state import RequestStateStore, validate_request_state_identity
from .runs import CONTROL_PLANE_VERSION
from .runtime_inventory import collect_checkout_runtime_inventory
from .service_identity import (
    build_service_launch_spec,
    verify_attach_existing_service,
)
from .slot_materializer import materialize_draft_bundle_set
from .stable_sync import plan_stable_checkout_sync
from .trusted_preparation import (
    perform_guarded_stable_sync,
    prepare_source_anchor,
    synchronize_stable_checkout,
)
from .workspace import LabError, _atomic_write_text, validate_repository_name


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    if path == Path("-"):
        try:
            payload = json.load(sys.stdin)
        except (OSError, json.JSONDecodeError) as exc:
            raise LabError(f"cannot read {label} from stdin: {exc}") from exc
        if not isinstance(payload, dict):
            raise LabError(f"{label} from stdin must contain an object")
        return payload
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabError(f"{label} must contain an object: {path}")
    return payload


def _source_anchor_for_observations(
    preparation_profile: PreparationProfile,
    observations: AssetObservationSet,
) -> SourceAnchorPlacement:
    anchors = preparation_profile.source_anchors
    identity = observations.canonical_identity
    matches = [
        placement
        for placement in anchors.values()
        if placement.canonical_identity == identity
    ]
    if len(matches) != 1:
        raise LabError("asset observations do not identify one protected placement")
    return matches[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="projectctl",
        description="Internal/admin utilities for the protected control plane.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    assets = commands.add_parser(
        "assets",
        help="inventory existing artifact content or resolve a reviewed selection",
    )
    asset_commands = assets.add_subparsers(dest="asset_command", required=True)
    artifact_manifest = asset_commands.add_parser(
        "artifact-manifest",
        help="hash one observed host-local artifact without network or activation",
    )
    artifact_manifest.add_argument("--observations", required=True, type=Path)
    artifact_manifest.add_argument("--workload", required=True)
    artifact_manifest.add_argument("--preparation-profile", required=True)
    artifact_manifest.add_argument("--host-profile", required=True)
    artifact_manifest.add_argument("--output", required=True, type=Path)
    select_assets = asset_commands.add_parser(
        "select",
        help="build, but do not install, an asset lock from a reviewed selection",
    )
    select_assets.add_argument("--observations", required=True, type=Path)
    select_assets.add_argument("--artifact-manifest", required=True, type=Path)
    select_assets.add_argument("--selection", required=True, type=Path)
    runtime_assets = asset_commands.add_parser(
        "runtime-receipts",
        help="rehash selected assets and bind them to one reserved request/GPU",
    )
    runtime_assets.add_argument("--observations", required=True, type=Path)
    runtime_assets.add_argument("--artifact-manifest", required=True, type=Path)
    runtime_assets.add_argument("--selection", required=True, type=Path)
    runtime_assets.add_argument("--asset-lock", required=True, type=Path)
    runtime_assets.add_argument("--request", required=True, type=Path)
    runtime_assets.add_argument("--profile", required=True)
    runtime_assets.add_argument("--preparation-profile", required=True)
    runtime_assets.add_argument("--host-profile", required=True)
    runtime_assets.add_argument("--state-root", required=True, type=Path)
    runtime_assets.add_argument("--output", required=True, type=Path)
    runtime_assets.add_argument("--control-version", default=CONTROL_PLANE_VERSION)
    request = commands.add_parser(
        "remote-request",
        help="build or validate an M3 GPU-only request without executing it",
    )
    request_commands = request.add_subparsers(dest="request_command", required=True)
    build = request_commands.add_parser("build")
    build.add_argument("--task-id", required=True)
    build.add_argument("--snapshot-id", required=True)
    build.add_argument("--snapshot", required=True, type=Path)
    build.add_argument("--assets", required=True, type=Path)
    build.add_argument("--profile", required=True)
    build.add_argument("--control-version", default=CONTROL_PLANE_VERSION)
    validate = request_commands.add_parser("validate")
    validate.add_argument("request", type=Path)
    validate.add_argument("--profile", required=True)
    validate.add_argument("--control-version", default=CONTROL_PLANE_VERSION)

    preflight = commands.add_parser(
        "remote-preflight",
        help="validate an M3 request against a protected host observation",
    )
    preflight.add_argument("--request", required=True, type=Path)
    preflight.add_argument("--observation", required=True, type=Path)
    preflight.add_argument("--profile", required=True)
    preflight.add_argument("--host-profile", required=True)
    preflight.add_argument("--control-version", default=CONTROL_PLANE_VERSION)

    observe = commands.add_parser(
        "host-observe",
        help="collect read-only repository/environment facts on a configured host",
    )
    observe.add_argument("--host-profile", required=True)
    observe.add_argument("--runtime-receipts", required=True, type=Path)

    sandbox_probe = commands.add_parser(
        "sandbox-probe",
        help=(
            "run the protected fixed bubblewrap capability probe without task code"
        ),
    )
    sandbox_probe.add_argument("--host-profile", required=True)

    gpu_lease = commands.add_parser(
        "gpu-lease",
        help="inventory and reserve GPU identity without starting a payload",
    )
    gpu_commands = gpu_lease.add_subparsers(dest="gpu_command", required=True)
    gpu_inventory = gpu_commands.add_parser("inventory")
    gpu_inventory.add_argument("--host-profile", required=True)
    gpu_reserve = gpu_commands.add_parser("reserve")
    gpu_reserve.add_argument("--request", required=True, type=Path)
    gpu_reserve.add_argument("--profile", required=True)
    gpu_reserve.add_argument("--host-profile", required=True)
    gpu_reserve.add_argument("--slot", required=True)
    gpu_reserve.add_argument("--state-root", required=True, type=Path)
    gpu_release = gpu_commands.add_parser("release")
    gpu_release.add_argument("--lease", required=True, type=Path)
    gpu_release.add_argument("--cleanup", required=True, type=Path)
    gpu_release.add_argument("--host-profile", required=True)
    gpu_reconcile = gpu_commands.add_parser("reconcile-nonstarted")
    gpu_reconcile.add_argument("--request", required=True, type=Path)
    gpu_reconcile.add_argument("--state-root", required=True, type=Path)
    gpu_reconcile.add_argument("--host-profile", required=True)

    inbox = commands.add_parser(
        "remote-inbox",
        help="accept a validated outbox envelope into SUBMITTED state without execution",
    )
    inbox_commands = inbox.add_subparsers(dest="inbox_command", required=True)
    inbox_accept = inbox_commands.add_parser("accept")
    inbox_accept.add_argument("--envelope", required=True, type=Path)
    inbox_accept.add_argument("--profile", required=True)
    inbox_accept.add_argument("--inbox-root", required=True, type=Path)
    inbox_accept.add_argument("--state-root", required=True, type=Path)
    inbox_accept.add_argument("--control-version", default=CONTROL_PLANE_VERSION)

    runner = commands.add_parser(
        "remote-runner",
        help="prepare, inspect, or cancel persistent M3 request state; never execute",
    )
    runner_commands = runner.add_subparsers(dest="runner_command", required=True)
    prepare = runner_commands.add_parser("prepare")
    prepare.add_argument("--request", required=True, type=Path)
    prepare.add_argument("--observation", required=True, type=Path)
    prepare.add_argument("--profile", required=True)
    prepare.add_argument("--host-profile", required=True)
    prepare.add_argument("--slot", required=True)
    prepare.add_argument("--state-root", required=True, type=Path)
    prepare.add_argument("--control-version", default=CONTROL_PLANE_VERSION)
    stage = runner_commands.add_parser(
        "stage-source",
        help="materialize exact clean source for a PREPARING request without execution",
    )
    stage.add_argument("--request", required=True, type=Path)
    stage.add_argument("--profile", required=True)
    stage.add_argument("--host-profile", required=True)
    stage.add_argument("--state-root", required=True, type=Path)
    stage.add_argument("--control-version", default=CONTROL_PLANE_VERSION)
    readiness = runner_commands.add_parser(
        "stage-readiness",
        help="probe staged imports and assets without starting services or execution",
    )
    readiness.add_argument("--request", required=True, type=Path)
    readiness.add_argument("--observation", required=True, type=Path)
    readiness.add_argument("--profile", required=True)
    readiness.add_argument("--host-profile", required=True)
    readiness.add_argument("--state-root", required=True, type=Path)
    readiness.add_argument("--control-version", default=CONTROL_PLANE_VERSION)
    service_spec = runner_commands.add_parser(
        "stage-service-spec",
        help="bind an inert managed-service launch identity without execution",
    )
    service_spec.add_argument("--request", required=True, type=Path)
    service_spec.add_argument("--profile", required=True)
    service_spec.add_argument("--state-root", required=True, type=Path)
    service_spec.add_argument("--launcher-nonce", required=True)
    service_spec.add_argument("--control-version", default=CONTROL_PLANE_VERSION)
    service_plan = runner_commands.add_parser(
        "stage-service-plan",
        help="compile and persist an inert protected service plan without launch",
    )
    service_plan.add_argument("--request", required=True, type=Path)
    service_plan.add_argument("--backend-probe", required=True, type=Path)
    service_plan.add_argument("--profile", required=True)
    service_plan.add_argument("--host-profile", required=True)
    service_plan.add_argument("--runtime-root", required=True, type=Path)
    service_plan.add_argument("--endpoint-port", required=True, type=int)
    service_plan.add_argument("--state-root", required=True, type=Path)
    service_plan.add_argument("--control-version", default=CONTROL_PLANE_VERSION)
    service_receipt = runner_commands.add_parser(
        "stage-service-receipt",
        help="bind a verified managed-service receipt without executing a command",
    )
    service_receipt.add_argument("--request", required=True, type=Path)
    service_receipt.add_argument("--receipt", required=True, type=Path)
    service_receipt.add_argument("--profile", required=True)
    service_receipt.add_argument("--state-root", required=True, type=Path)
    service_receipt.add_argument("--control-version", default=CONTROL_PLANE_VERSION)
    status = runner_commands.add_parser("status")
    status.add_argument("request_id")
    status.add_argument("--state-root", required=True, type=Path)
    watch = runner_commands.add_parser(
        "watch",
        help="wait for request history to advance without mutating it",
    )
    watch.add_argument("request_id")
    watch.add_argument("--state-root", required=True, type=Path)
    watch.add_argument("--after-sequence", type=int, default=-1)
    watch.add_argument("--timeout-seconds", type=float, default=30.0)
    watch.add_argument("--poll-interval-seconds", type=float, default=0.25)
    cancel = runner_commands.add_parser("cancel")
    cancel.add_argument("request_id")
    cancel.add_argument("--state-root", required=True, type=Path)
    acknowledge_cancel = runner_commands.add_parser(
        "ack-cancel",
        help="complete active cancellation with a trusted teardown receipt",
    )
    acknowledge_cancel.add_argument("request_id")
    acknowledge_cancel.add_argument("--request", required=True, type=Path)
    acknowledge_cancel.add_argument("--receipt", required=True, type=Path)
    acknowledge_cancel.add_argument("--state-root", required=True, type=Path)

    service = commands.add_parser(
        "service-identity",
        help="build or verify protected service identities without launching a process",
    )
    service_commands = service.add_subparsers(
        dest="service_command",
        required=True,
    )
    launch_spec = service_commands.add_parser("launch-spec")
    launch_spec.add_argument("--request", required=True, type=Path)
    launch_spec.add_argument("--readiness", required=True, type=Path)
    launch_spec.add_argument("--profile", required=True)
    launch_spec.add_argument("--control-version", default=CONTROL_PLANE_VERSION)
    launch_spec.add_argument("--slot", required=True)
    launch_spec.add_argument("--launcher-nonce", required=True)
    service_sandbox = service_commands.add_parser(
        "sandbox-receipt",
        help="bind exact service inputs to a live protected sandbox probe",
    )
    service_sandbox.add_argument("--launch-spec", required=True, type=Path)
    service_sandbox.add_argument("--readiness", required=True, type=Path)
    service_sandbox.add_argument("--backend-probe", required=True, type=Path)
    service_sandbox.add_argument("--host-profile", required=True)
    service_sandbox.add_argument("--runtime-root", required=True, type=Path)
    service_sandbox.add_argument(
        "--control-version",
        default=CONTROL_PLANE_VERSION,
    )
    verify_health = service_commands.add_parser("verify-health")
    verify_health.add_argument("--launch-spec", required=True, type=Path)
    verify_health.add_argument("--service-sandbox", required=True, type=Path)
    verify_health.add_argument("--process-receipt", required=True, type=Path)
    verify_health.add_argument("--health", required=True, type=Path)
    verify_health.add_argument("--host-profile", required=True)
    verify_existing = service_commands.add_parser("verify-existing")
    verify_existing.add_argument("--launch-spec", required=True, type=Path)
    verify_existing.add_argument("--receipt", required=True, type=Path)

    bundle = commands.add_parser(
        "draft-bundle",
        help="verify an incremental draft bundle without mutating target repositories",
    )
    bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
    bundle_verify = bundle_commands.add_parser("verify")
    bundle_verify.add_argument("manifest", type=Path)
    bundle_verify.add_argument(
        "--repository",
        action="append",
        required=True,
        dest="repository_paths",
        help="NAME=PATH target repository with the exact prerequisite base; repeatable",
    )
    bundle_materialize = bundle_commands.add_parser(
        "materialize",
        help="prepare an inert detached source tree in a trusted slot",
    )
    bundle_materialize.add_argument("manifest", type=Path)
    bundle_materialize.add_argument(
        "--repository",
        action="append",
        required=True,
        dest="repository_paths",
        help="NAME=PATH clean source anchor with the exact prerequisite base; repeatable",
    )
    bundle_materialize.add_argument("--slot-root", required=True, type=Path)
    bundle_materialize.add_argument("--slot", required=True)

    trusted_prepare = commands.add_parser(
        "trusted-prepare",
        help=(
            "inspect or prepare protected source placements; paths and authority "
            "come only from reviewed Profiles"
        ),
    )
    trusted_prepare_commands = trusted_prepare.add_subparsers(
        dest="trusted_prepare_command",
        required=True,
    )
    trusted_status = trusted_prepare_commands.add_parser(
        "status",
        help="inspect one stable checkout against an already-fetched protected ref",
    )
    trusted_status.add_argument("--preparation-profile", required=True)
    trusted_status.add_argument("--host-profile", required=True)
    trusted_status.add_argument("--repository", required=True)
    trusted_sync = trusted_prepare_commands.add_parser(
        "sync",
        help=(
            "explicitly fetch one protected ref and apply ff-only only with a "
            "fresh maintenance receipt"
        ),
    )
    trusted_sync.add_argument("--preparation-profile", required=True)
    trusted_sync.add_argument("--host-profile", required=True)
    trusted_sync.add_argument("--repository", required=True)
    trusted_sync.add_argument("--maintenance-receipt", type=Path)
    trusted_anchor = trusted_prepare_commands.add_parser(
        "source-anchor",
        help="materialize an exact local-only source anchor without downloads",
    )
    trusted_anchor.add_argument("--preparation-profile", required=True)
    trusted_anchor.add_argument("--host-profile", required=True)
    trusted_anchor.add_argument("--repository", required=True)
    trusted_anchor.add_argument("--commit", required=True)
    trusted_inventory = trusted_prepare_commands.add_parser(
        "runtime-inventory",
        help="collect a read-only active-process inventory for one stable checkout",
    )
    trusted_inventory.add_argument("--preparation-profile", required=True)
    trusted_inventory.add_argument("--host-profile", required=True)
    trusted_inventory.add_argument("--repository", required=True)
    trusted_auto_sync = trusted_prepare_commands.add_parser(
        "auto-sync",
        help="reserve runtime starts, inventory, fetch, guarded ff-only, and release",
    )
    trusted_auto_sync.add_argument("--preparation-profile", required=True)
    trusted_auto_sync.add_argument("--host-profile", required=True)
    trusted_auto_sync.add_argument("--repository", required=True)
    trusted_auto_sync.add_argument("--reservation-id", required=True)

    control_plane = commands.add_parser(
        "control-plane",
        help="build protected, inert control-plane deployment material",
    )
    control_plane_commands = control_plane.add_subparsers(
        dest="control_plane_command",
        required=True,
    )
    control_plane_package = control_plane_commands.add_parser(
        "package",
        help="archive the exact clean origin/main control plane without large files",
    )
    control_plane_package.add_argument("--preparation-profile", required=True)
    control_plane_package.add_argument("--host-profile", required=True)
    control_plane_package.add_argument("--output", required=True, type=Path)
    control_plane_stage = control_plane_commands.add_parser(
        "stage",
        help="verify and stage a version without updating current or executing it",
    )
    control_plane_stage.add_argument("--preparation-profile", required=True)
    control_plane_stage.add_argument("--host-profile", required=True)
    control_plane_stage.add_argument("--archive", required=True, type=Path)
    control_plane_stage.add_argument("--receipt", required=True, type=Path)
    control_plane_activate = control_plane_commands.add_parser(
        "activate",
        help="health-check a staged protected version and atomically update current",
    )
    control_plane_activate.add_argument("--preparation-profile", required=True)
    control_plane_activate.add_argument("--host-profile", required=True)
    control_plane_activate.add_argument("--stage-receipt", required=True, type=Path)
    control_plane_rollback = control_plane_commands.add_parser(
        "rollback",
        help="health-check and restore the previous staged control-plane version",
    )
    control_plane_rollback.add_argument("--preparation-profile", required=True)
    control_plane_rollback.add_argument("--host-profile", required=True)
    control_plane_rollback.add_argument(
        "--activation-receipt",
        required=True,
        type=Path,
    )
    return parser


def _repository_assignments(values: Sequence[str]) -> dict[str, Path]:
    repositories: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise LabError(f"repository assignment must use NAME=PATH: {value!r}")
        name, raw_path = value.split("=", 1)
        validate_repository_name(name)
        if name in repositories or not raw_path:
            raise LabError(f"invalid or duplicate repository assignment: {value!r}")
        repositories[name] = Path(raw_path).expanduser().resolve()
    return repositories


def _write_new_object_under_roots(
    output_path: Path,
    payload: dict[str, object],
    *,
    allowed_roots: Sequence[Path],
) -> Path:
    requested = output_path.expanduser()
    output = requested.resolve()
    parent = output.parent
    roots = tuple(allowed_roots)
    if (
        not roots
        or any(
            root.is_symlink() or root.resolve() != root or not root.is_dir()
            for root in roots
        )
        or not requested.is_absolute()
        or requested.is_symlink()
        or requested.absolute() != output
        or output.exists()
        or parent.is_symlink()
        or not parent.is_dir()
        or not any(output.is_relative_to(root) for root in roots)
    ):
        raise LabError(f"asset manifest output is unsafe: {output_path}")
    _atomic_write_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def _require_protected_file(path: Path, *, root: Path, label: str) -> Path:
    requested = path.expanduser()
    resolved = requested.resolve()
    protected_root = root.resolve()
    if (
        requested.absolute() != resolved
        or requested.is_symlink()
        or not resolved.is_file()
        or resolved.parent.is_symlink()
        or not resolved.is_relative_to(protected_root)
    ):
        raise LabError(f"{label} is not a protected control-plane file: {path}")
    return resolved


def _require_file_under_roots(
    path: Path,
    *,
    roots: Sequence[Path],
    label: str,
) -> Path:
    requested = path.expanduser()
    resolved = requested.resolve()
    safe_roots = tuple(roots)
    if (
        not safe_roots
        or any(
            root.is_symlink() or root.resolve() != root or not root.is_dir()
            for root in safe_roots
        )
        or requested.is_symlink()
        or requested.absolute() != resolved
        or not resolved.is_file()
        or resolved.parent.is_symlink()
        or not any(resolved.is_relative_to(root) for root in safe_roots)
    ):
        raise LabError(f"{label} is outside protected roots: {path}")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    remote_profile_root = (
        Path(
            os.environ.get(
                "LAB_REMOTE_PROFILE_ROOT",
                repository_root / "integration" / "remote_profiles",
            )
        )
        .expanduser()
        .resolve()
    )
    try:
        if arguments.command == "assets":
            protected_observations = _require_protected_file(
                arguments.observations,
                root=repository_root / "stack",
                label="asset observations",
            )
            if (
                protected_observations
                != (repository_root / "stack" / "remote-asset-observations.json").resolve()
            ):
                raise LabError("asset observations are not the active protected record")
            observations = load_asset_observations(protected_observations)
            if arguments.asset_command == "artifact-manifest":
                host_profile = load_host_profile(
                    repository_root / "integration" / "host_profiles",
                    arguments.host_profile,
                )
                preparation_profile = load_preparation_profile(
                    repository_root / "integration" / "preparation_profiles",
                    arguments.preparation_profile,
                    host_profile=host_profile,
                )
                asset_source_placement = _source_anchor_for_observations(
                    preparation_profile, observations
                )
                if (
                    observations.host_id != host_profile.host_id
                ):
                    raise LabError(
                        "artifact observations do not match the protected placement"
                    )
                manifest = collect_artifact_manifest(
                    observations,
                    workload_id=arguments.workload,
                    source_root=asset_source_placement.source_path,
                    expected_canonical_identity=(
                        asset_source_placement.canonical_identity
                    ),
                )
                output = _write_new_object_under_roots(
                    arguments.output,
                    manifest,
                    allowed_roots=host_profile.asset_roots,
                )
                payload = {
                    "schema": "lab.artifact-manifest-write.v1",
                    "path": str(output),
                    "workload_id": manifest["workload_id"],
                    "content_sha256": manifest["content_sha256"],
                    "receipt_sha256": manifest["receipt_sha256"],
                    "network_used": False,
                    "large_file_downloaded": False,
                    "selection_authorized": False,
                    "activation_ready": False,
                    "device_access": False,
                }
            elif arguments.asset_command == "select":
                selection = load_workload_selection(
                    _require_protected_file(
                        arguments.selection,
                        root=repository_root / "stack" / "workload-selections",
                        label="workload selection",
                    )
                )
                manifest = validate_artifact_manifest(
                    _read_object(
                        arguments.artifact_manifest,
                        label="artifact manifest",
                    ),
                    observations=observations,
                    workload_id=selection.workload_id,
                )
                payload = build_selected_asset_lock(
                    observations,
                    artifact_manifest=manifest,
                    selection=selection,
                )
            else:
                host_profile = load_host_profile(
                    repository_root / "integration" / "host_profiles",
                    arguments.host_profile,
                )
                preparation_profile = load_preparation_profile(
                    repository_root / "integration" / "preparation_profiles",
                    arguments.preparation_profile,
                    host_profile=host_profile,
                )
                asset_source_placement = _source_anchor_for_observations(
                    preparation_profile, observations
                )
                if (
                    observations.host_id != host_profile.host_id
                ):
                    raise LabError(
                        "runtime asset observations do not match the protected placement"
                    )
                selection = load_workload_selection(
                    _require_protected_file(
                        arguments.selection,
                        root=repository_root / "stack" / "workload-selections",
                        label="workload selection",
                    )
                )
                manifest = validate_artifact_manifest(
                    _read_object(
                        _require_file_under_roots(
                            arguments.artifact_manifest,
                            roots=host_profile.asset_roots,
                            label="artifact manifest",
                        ),
                        label="artifact manifest",
                    ),
                    observations=observations,
                    workload_id=selection.workload_id,
                )
                protected_asset_lock = _require_protected_file(
                    arguments.asset_lock,
                    root=repository_root / "stack",
                    label="active asset lock",
                )
                if protected_asset_lock != (
                    repository_root / "stack" / "remote-assets.json"
                ).resolve():
                    raise LabError("asset lock is not the active protected lock")
                active_asset_lock = load_asset_lock(protected_asset_lock)
                profile = load_remote_profile(remote_profile_root, arguments.profile)
                request = validate_remote_request(
                    _read_object(arguments.request, label="remote request"),
                    profile=profile,
                    control_plane_version=arguments.control_version,
                )
                if request.get("asset_lock") != {
                    "lock_id": active_asset_lock.lock_id,
                    "workload_id": active_asset_lock.workload_id,
                    "sha256": active_asset_lock.sha256,
                }:
                    raise LabError("remote request does not use the active asset lock")
                requested_state_root = arguments.state_root.expanduser()
                state_root = requested_state_root.resolve()
                if (
                    not requested_state_root.is_absolute()
                    or requested_state_root.is_symlink()
                    or requested_state_root.absolute() != state_root
                    or not state_root.is_relative_to(host_profile.slot_root)
                    or not state_root.is_dir()
                ):
                    raise LabError("runtime receipt state root is unsafe")
                state_store = RequestStateStore(state_root)
                state = validate_request_state_identity(
                    state_store.load(str(request["request_id"])),
                    request=request,
                )
                reservation = state.get("resource_reservation_receipt")
                slot_id = state.get("slot_id")
                if (
                    state.get("state") != "RESERVED"
                    or not isinstance(reservation, dict)
                    or not isinstance(slot_id, str)
                    or slot_id not in host_profile.slots
                ):
                    raise LabError(
                        "runtime receipts require a reserved, not-yet-prepared request"
                    )
                validated_reservation = validate_gpu_resource_reservation(
                    reservation,
                    request=request,
                    slot_id=slot_id,
                    host_id=host_profile.host_id,
                    host_profile_sha256=host_profile.sha256,
                    remote_profile_sha256=profile.sha256,
                    require_active=True,
                )
                lease = validated_reservation.get("gpu_lease_receipt")
                if not isinstance(lease, dict):
                    raise LabError("runtime receipt reservation lacks a GPU lease")
                verification = build_runtime_asset_verification(
                    observations,
                    artifact_manifest=manifest,
                    selection=selection,
                    asset_lock=active_asset_lock,
                    source_root=asset_source_placement.source_path,
                    expected_canonical_identity=(
                        asset_source_placement.canonical_identity
                    ),
                )
                verified_assets = verification.get("assets")
                if not isinstance(verified_assets, dict):
                    raise LabError("runtime asset verification lacks assets")
                request_assets = {
                    role: {
                        "asset_id": identity.get("asset_id"),
                        "sha256": identity.get("sha256"),
                    }
                    for role, identity in verified_assets.items()
                    if isinstance(role, str) and isinstance(identity, dict)
                }
                if request_assets != request.get("assets"):
                    raise LabError("verified runtime assets differ from the request")
                runtime_receipt = build_runtime_receipts(
                    host_profile,
                    gpu={
                        "reserved_count": 1,
                        "lease_id": lease.get("lease_id"),
                        "gpu_id": lease.get("gpu_id"),
                        "gpu_uuid": lease.get("gpu_uuid"),
                        "lease_receipt_sha256": lease.get("receipt_sha256"),
                        "lease_receipt": lease,
                    },
                    assets=verified_assets,
                    asset_verification=verification,
                )
                output = _write_new_object_under_roots(
                    arguments.output,
                    runtime_receipt,
                    allowed_roots=(host_profile.slot_root,),
                )
                payload = {
                    "schema": "lab.runtime-receipts-write.v1",
                    "request_id": request["request_id"],
                    "path": str(output),
                    "runtime_receipt_sha256": runtime_receipt["receipt_sha256"],
                    "asset_verification_receipt_sha256": verification[
                        "receipt_sha256"
                    ],
                    "network_used": False,
                    "large_file_downloaded": False,
                    "execution_eligible": False,
                    "device_access": False,
                }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if arguments.command == "control-plane":
            host_profile = load_host_profile(
                repository_root / "integration" / "host_profiles",
                arguments.host_profile,
            )
            preparation_profile = load_preparation_profile(
                repository_root / "integration" / "preparation_profiles",
                arguments.preparation_profile,
                host_profile=host_profile,
            )
            if arguments.control_plane_command == "package":
                placement = preparation_profile.stable_sync.get(
                    preparation_profile.control_plane.repository
                )
                if placement is None:
                    raise LabError(
                        "control-plane package requires its protected repository identity"
                    )
                payload = build_control_plane_package(
                    repository_root,
                    expected_identity=placement.canonical_identity,
                    output_path=arguments.output,
                )
            elif arguments.control_plane_command == "stage":
                payload = stage_control_plane_package(
                    preparation_profile,
                    archive_path=arguments.archive,
                    package_receipt=_read_object(
                        arguments.receipt,
                        label="control-plane package receipt",
                    ),
                )
            elif arguments.control_plane_command == "activate":
                payload = activate_control_plane_version(
                    preparation_profile,
                    host_profile,
                    stage_receipt=_read_object(
                        arguments.stage_receipt,
                        label="control-plane stage receipt",
                    ),
                )
            else:
                payload = rollback_control_plane_version(
                    preparation_profile,
                    host_profile,
                    activation_receipt=_read_object(
                        arguments.activation_receipt,
                        label="control-plane activation receipt",
                    ),
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if arguments.command == "trusted-prepare":
            host_profile = load_host_profile(
                repository_root / "integration" / "host_profiles",
                arguments.host_profile,
            )
            preparation_profile = load_preparation_profile(
                repository_root / "integration" / "preparation_profiles",
                arguments.preparation_profile,
                host_profile=host_profile,
            )
            repository = validate_repository_name(arguments.repository)
            if arguments.trusted_prepare_command == "auto-sync":
                payload = perform_guarded_stable_sync(
                    preparation_profile,
                    repository=repository,
                    reservation_id=arguments.reservation_id,
                )
            elif arguments.trusted_prepare_command == "status":
                placement = preparation_profile.stable_sync.get(repository)
                if placement is None:
                    raise LabError(
                        f"repository has no stable sync policy: {repository}"
                    )
                payload = plan_stable_checkout_sync(
                    placement.path,
                    expected_identity=placement.canonical_identity,
                    branch=placement.branch,
                    remote_ref=placement.remote_ref,
                )
            elif arguments.trusted_prepare_command == "sync":
                maintenance_receipt = (
                    _read_object(
                        arguments.maintenance_receipt,
                        label="maintenance receipt",
                    )
                    if arguments.maintenance_receipt is not None
                    else None
                )
                payload = synchronize_stable_checkout(
                    preparation_profile,
                    repository=repository,
                    maintenance_receipt=maintenance_receipt,
                )
            elif arguments.trusted_prepare_command == "source-anchor":
                payload = prepare_source_anchor(
                    preparation_profile,
                    repository=repository,
                    commit=arguments.commit,
                )
            else:
                placement = preparation_profile.stable_sync.get(repository)
                if placement is None:
                    raise LabError(
                        f"repository has no stable sync policy: {repository}"
                    )
                payload = collect_checkout_runtime_inventory(placement.path)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if arguments.command == "host-observe":
            host_profile = load_host_profile(
                repository_root / "integration" / "host_profiles",
                arguments.host_profile,
            )
            payload = collect_host_observation(
                host_profile,
                load_runtime_receipts(
                    arguments.runtime_receipts,
                    host_profile=host_profile,
                ),
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if arguments.command == "sandbox-probe":
            host_profile = load_host_profile(
                repository_root / "integration" / "host_profiles",
                arguments.host_profile,
            )
            payload = collect_sandbox_backend_probe(host_profile)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if arguments.command == "gpu-lease":
            host_profile = load_host_profile(
                repository_root / "integration" / "host_profiles",
                arguments.host_profile,
            )
            if arguments.gpu_command == "inventory":
                gpu_payload = collect_nvidia_gpu_inventory()
            else:
                store = GpuLeaseStore(
                    host_profile.slot_root / ".gpu-leases",
                    host_id=host_profile.host_id,
                    host_profile_sha256=host_profile.sha256,
                )
                if arguments.gpu_command == "reserve":
                    profile = load_remote_profile(
                        remote_profile_root,
                        arguments.profile,
                    )
                    request = validate_remote_request(
                        _read_object(arguments.request, label="remote request"),
                        profile=profile,
                        control_plane_version=CONTROL_PLANE_VERSION,
                    )
                    if (
                        profile.target_host != host_profile.host_id
                        or profile.gpu_count != 1
                        or arguments.slot not in host_profile.slots
                    ):
                        raise LabError("GPU reservation Profile/host/slot mismatch")
                    inventory = collect_nvidia_gpu_inventory()
                    process_inventory = collect_nvidia_compute_process_inventory()
                    select_single_gpu(inventory)
                    state = RequestStateStore(arguments.state_root).reserve_gpu(
                        request,
                        slot_id=arguments.slot,
                        allowed_slots=host_profile.slots,
                        host_id=host_profile.host_id,
                        host_profile_sha256=host_profile.sha256,
                        remote_profile_sha256=profile.sha256,
                        inventory_receipt=inventory,
                        process_inventory_receipt=process_inventory,
                        gpu_store=store,
                    )
                    reservation = state.get("resource_reservation_receipt")
                    if not isinstance(reservation, dict):
                        raise LabError("GPU resource reservation was not persisted")
                    gpu_payload = {
                        "schema": "lab.gpu-reservation.v2",
                        "request_state": state,
                        "resource_reservation": reservation,
                        "service_started": False,
                        "execution_eligible": False,
                        "device_access": False,
                    }
                elif arguments.gpu_command == "release":
                    lease = _read_object(arguments.lease, label="GPU lease receipt")
                    release_gpu_uuid = lease.get("gpu_uuid")
                    if not isinstance(release_gpu_uuid, str):
                        raise LabError("GPU lease receipt lacks GPU UUID")
                    gpu_payload = store.release(
                        lease,
                        gpu_uuid=release_gpu_uuid,
                        cleanup_receipt=_read_object(
                            arguments.cleanup,
                            label="GPU cleanup receipt",
                        ),
                    )
                else:
                    inventory = collect_nvidia_gpu_inventory()
                    _, gpu_uuid = select_single_gpu(inventory)
                    process_inventory = collect_nvidia_compute_process_inventory()
                    request = _read_object(arguments.request, label="remote request")
                    state = RequestStateStore(
                        arguments.state_root
                    ).reconcile_nonstarted_gpu(
                        request,
                        gpu_uuid=gpu_uuid,
                        process_inventory_receipt=process_inventory,
                        gpu_store=store,
                    )
                    release = state.get("resource_release_receipt")
                    if not isinstance(release, dict):
                        raise LabError("GPU reconciliation did not persist a release")
                    gpu_payload = {
                        "schema": "lab.gpu-reconciliation.v1",
                        "request_state": state,
                        "resource_release": release,
                        "service_started": False,
                        "execution_eligible": False,
                        "device_access": False,
                    }
            print(json.dumps(gpu_payload, indent=2, sort_keys=True))
            return 0
        if arguments.command == "remote-inbox":
            profile = load_remote_profile(remote_profile_root, arguments.profile)
            payload = RemoteInbox(
                arguments.inbox_root,
                RequestStateStore(arguments.state_root),
            ).accept(
                _read_object(arguments.envelope, label="remote outbox envelope"),
                remote_profile=profile,
                control_plane_version=arguments.control_version,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if arguments.command == "draft-bundle":
            bundle_repositories = _repository_assignments(arguments.repository_paths)
            if arguments.bundle_command == "materialize":
                payload = materialize_draft_bundle_set(
                    arguments.manifest,
                    repository_paths=bundle_repositories,
                    slot_root=arguments.slot_root,
                    slot_id=arguments.slot,
                )
            else:
                payload = verify_draft_bundle_set(
                    arguments.manifest,
                    repository_paths=bundle_repositories,
                    require_synthetic_absent=True,
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if arguments.command == "remote-runner":
            state_store = RequestStateStore(arguments.state_root)
            if arguments.runner_command == "status":
                payload = state_store.load(arguments.request_id)
            elif arguments.runner_command == "watch":
                payload = state_store.watch(
                    arguments.request_id,
                    after_sequence=arguments.after_sequence,
                    timeout_seconds=arguments.timeout_seconds,
                    poll_interval_seconds=arguments.poll_interval_seconds,
                )
            elif arguments.runner_command == "cancel":
                payload = state_store.request_cancel(arguments.request_id)
            elif arguments.runner_command == "ack-cancel":
                cancel_request = _read_object(
                    arguments.request, label="remote request"
                )
                if cancel_request.get("request_id") != arguments.request_id:
                    raise LabError("ack-cancel request path has a different request ID")
                payload = state_store.acknowledge_cancel(
                    cancel_request,
                    _read_object(
                        arguments.receipt,
                        label="cancel teardown receipt",
                    ),
                )
            elif arguments.runner_command == "stage-source":
                profile = load_remote_profile(remote_profile_root, arguments.profile)
                host_profile = load_host_profile(
                    repository_root / "integration" / "host_profiles",
                    arguments.host_profile,
                )
                payload = stage_remote_request_sources(
                    _read_object(arguments.request, label="remote request"),
                    remote_profile=profile,
                    host_profile=host_profile,
                    control_plane_version=arguments.control_version,
                    state_store=state_store,
                )
            elif arguments.runner_command == "stage-readiness":
                profile = load_remote_profile(remote_profile_root, arguments.profile)
                host_profile = load_host_profile(
                    repository_root / "integration" / "host_profiles",
                    arguments.host_profile,
                )
                payload = stage_remote_request_readiness(
                    _read_object(arguments.request, label="remote request"),
                    remote_profile=profile,
                    host_profile=host_profile,
                    observation=_read_object(
                        arguments.observation,
                        label="host observation",
                    ),
                    control_plane_version=arguments.control_version,
                    state_store=state_store,
                )
            elif arguments.runner_command == "stage-service-spec":
                profile = load_remote_profile(remote_profile_root, arguments.profile)
                payload = stage_remote_service_launch_spec(
                    _read_object(arguments.request, label="remote request"),
                    remote_profile=profile,
                    control_plane_version=arguments.control_version,
                    state_store=state_store,
                    launcher_nonce=arguments.launcher_nonce,
                )
            elif arguments.runner_command == "stage-service-plan":
                profile = load_remote_profile(remote_profile_root, arguments.profile)
                host_profile = load_host_profile(
                    repository_root / "integration" / "host_profiles",
                    arguments.host_profile,
                )
                payload = stage_remote_service_launch_plan(
                    _read_object(arguments.request, label="remote request"),
                    remote_profile=profile,
                    host_profile=host_profile,
                    backend_probe_receipt=_read_object(
                        arguments.backend_probe,
                        label="sandbox backend probe",
                    ),
                    runtime_root=arguments.runtime_root,
                    adapter_root=(
                        repository_root / "integration" / "service_adapters"
                    ),
                    endpoint_port=arguments.endpoint_port,
                    control_plane_version=arguments.control_version,
                    state_store=state_store,
                )
            elif arguments.runner_command == "stage-service-receipt":
                profile = load_remote_profile(remote_profile_root, arguments.profile)
                payload = stage_remote_service_receipt(
                    _read_object(arguments.request, label="remote request"),
                    remote_profile=profile,
                    control_plane_version=arguments.control_version,
                    state_store=state_store,
                    service_receipt=_read_object(
                        arguments.receipt,
                        label="service receipt",
                    ),
                )
            else:
                profile = load_remote_profile(remote_profile_root, arguments.profile)
                host_profile = load_host_profile(
                    repository_root / "integration" / "host_profiles",
                    arguments.host_profile,
                )
                payload = prepare_remote_request(
                    _read_object(arguments.request, label="remote request"),
                    remote_profile=profile,
                    host_profile=host_profile,
                    observation=_read_object(
                        arguments.observation,
                        label="host observation",
                    ),
                    control_plane_version=arguments.control_version,
                    state_store=state_store,
                    slot_id=arguments.slot,
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if arguments.command == "service-identity":
            if arguments.service_command == "launch-spec":
                profile = load_remote_profile(remote_profile_root, arguments.profile)
                payload = build_service_launch_spec(
                    _read_object(arguments.request, label="remote request"),
                    remote_profile=profile,
                    readiness_receipt=_read_object(
                        arguments.readiness,
                        label="slot readiness receipt",
                    ),
                    control_plane_version=arguments.control_version,
                    slot_id=arguments.slot,
                    launcher_nonce=arguments.launcher_nonce,
                )
            elif arguments.service_command == "sandbox-receipt":
                host_profile = load_host_profile(
                    repository_root / "integration" / "host_profiles",
                    arguments.host_profile,
                )
                payload = build_service_sandbox_receipt(
                    launch_spec=_read_object(
                        arguments.launch_spec,
                        label="service launch spec",
                    ),
                    readiness_receipt=_read_object(
                        arguments.readiness,
                        label="slot readiness receipt",
                    ),
                    backend_probe_receipt=_read_object(
                        arguments.backend_probe,
                        label="sandbox backend probe",
                    ),
                    host_profile=host_profile,
                    control_plane_version=arguments.control_version,
                    runtime_root=arguments.runtime_root,
                )
            elif arguments.service_command == "verify-health":
                host_profile = load_host_profile(
                    repository_root / "integration" / "host_profiles",
                    arguments.host_profile,
                )
                payload = build_attested_service_receipt(
                    _read_object(arguments.health, label="service health identity"),
                    launch_spec=_read_object(
                        arguments.launch_spec,
                        label="service launch spec",
                    ),
                    service_sandbox_receipt=_read_object(
                        arguments.service_sandbox,
                        label="service sandbox receipt",
                    ),
                    managed_process_receipt=_read_object(
                        arguments.process_receipt,
                        label="managed service process receipt",
                    ),
                    host_profile=host_profile,
                )
            else:
                payload = verify_attach_existing_service(
                    _read_object(
                        arguments.launch_spec,
                        label="service launch spec",
                    ),
                    _read_object(arguments.receipt, label="service receipt"),
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        profile = load_remote_profile(remote_profile_root, arguments.profile)
        if arguments.command == "remote-preflight":
            host_profile = load_host_profile(
                repository_root / "integration" / "host_profiles",
                arguments.host_profile,
            )
            payload = preflight_remote_request(
                _read_object(arguments.request, label="remote request"),
                remote_profile=profile,
                host_profile=host_profile,
                observation=_read_object(
                    arguments.observation,
                    label="host observation",
                ),
                control_plane_version=arguments.control_version,
            )
        elif arguments.request_command == "build":
            snapshot = _read_object(arguments.snapshot, label="snapshot identity")
            repositories = snapshot.get("repositories")
            if not isinstance(repositories, dict):
                raise LabError("snapshot identity has no repositories object")
            assets_payload = _read_object(arguments.assets, label="asset identity")
            assets = assets_payload.get("assets")
            if not isinstance(assets, dict):
                raise LabError("asset identity has no assets object")
            asset_lock_id = assets_payload.get("lock_id")
            asset_lock_workload_id = assets_payload.get("workload_id")
            asset_lock_sha256 = assets_payload.get("sha256")
            if (
                not isinstance(asset_lock_id, str)
                or (
                    asset_lock_workload_id is not None
                    and not isinstance(asset_lock_workload_id, str)
                )
                or not isinstance(asset_lock_sha256, str)
            ):
                raise LabError("asset identity has no exact lock/SHA-256 identity")
            asset_lock = {
                "lock_id": asset_lock_id,
                "workload_id": asset_lock_workload_id,
                "sha256": asset_lock_sha256,
            }
            payload = build_remote_request(
                task_id=arguments.task_id,
                snapshot_id=arguments.snapshot_id,
                repositories=repositories,
                repository_links=snapshot.get("repository_links", []),
                assets=assets,
                asset_lock=asset_lock,
                profile=profile,
                control_plane_version=arguments.control_version,
            )
        else:
            payload = validate_remote_request(
                _read_object(arguments.request, label="remote request"),
                profile=profile,
                control_plane_version=arguments.control_version,
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except LabError as exc:
        print(f"projectctl: error: {exc}", file=sys.stderr)
        return 2
