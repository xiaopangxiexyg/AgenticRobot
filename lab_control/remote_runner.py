from __future__ import annotations

import shutil
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from .clean_slot import materialize_clean_request_sources
from .command_registry import TrustedCommandRegistry, resolve_trusted_command
from .endpoint_lease import EndpointLeaseStore
from .execution_sandbox import (
    build_service_sandbox_receipt,
    validate_execution_sandbox_for_launch,
)
from .network_namespace import (
    NetworkNamespaceProbePolicy,
    build_network_namespace_plan,
    collect_network_namespace_anchor_observation,
    collect_network_namespace_probe,
    validate_network_namespace_probe_for_use,
)
from .remote_execution import (
    TrustedCommandAdapter,
    build_attempt_result,
    build_attempt_start,
    collect_attempt_artifacts,
)
from .remote_preflight import HostProfile, preflight_remote_request
from .remote_request import RemoteProfile, validate_remote_request
from .runner_state import (
    TERMINAL_STATES,
    RequestStateStore,
    validate_request_state_identity,
)
from .service_adapters import load_trusted_service_adapter
from .service_identity import build_service_launch_spec, validate_service_receipt
from .service_launch_plan import build_service_launch_plan
from .slot_readiness import prepare_slot_readiness
from .workspace import LabError


def prepare_remote_request(
    request: object,
    *,
    remote_profile: RemoteProfile,
    host_profile: HostProfile,
    observation: Mapping[str, object],
    control_plane_version: str,
    state_store: RequestStateStore,
    slot_id: str,
) -> dict[str, object]:
    """Preflight an already resource-reserved request; never execute its payload."""

    validated_request = validate_remote_request(
        request,
        profile=remote_profile,
        control_plane_version=control_plane_version,
    )
    request_id = str(validated_request["request_id"])
    current = state_store.submit(validated_request)
    state = str(current.get("state"))
    if state in TERMINAL_STATES or state in {
        "PREPARING",
        "RUNNING",
        "COLLECTING",
        "CANCEL_REQUESTED",
    }:
        return current
    if state != "RESERVED" or not isinstance(
        current.get("resource_reservation_receipt"),
        dict,
    ):
        raise LabError(
            "remote prepare requires an atomic slot/GPU resource reservation"
        )
    try:
        if current.get("slot_id") != slot_id:
            raise LabError("remote prepare slot does not match resource reservation")
        receipt = preflight_remote_request(
            validated_request,
            remote_profile=remote_profile,
            host_profile=host_profile,
            observation=observation,
            control_plane_version=control_plane_version,
        )
        return state_store.attach_preflight(request_id, receipt)
    except LabError as exc:
        current = state_store.load(request_id)
        if current.get("state") == "RESERVED":
            state_store.fail(
                request_id,
                classification="BLOCKED_PRECONDITION",
                detail=str(exc),
            )
        raise


def stage_remote_request_sources(
    request: object,
    *,
    remote_profile: RemoteProfile,
    host_profile: HostProfile,
    control_plane_version: str,
    state_store: RequestStateStore,
) -> dict[str, object]:
    """Materialize exact clean source and attach its inert receipt; never execute it."""

    validated_request = validate_remote_request(
        request,
        profile=remote_profile,
        control_plane_version=control_plane_version,
    )
    request_id = str(validated_request["request_id"])
    current = state_store.load(request_id)
    if current.get("state") != "PREPARING":
        raise LabError(f"request is not ready for source staging: {request_id}")
    slot_id = str(current.get("slot_id", ""))
    receipt = materialize_clean_request_sources(
        validated_request,
        remote_profile=remote_profile,
        host_profile=host_profile,
        control_plane_version=control_plane_version,
        slot_id=slot_id,
    )
    return state_store.attach_source_preparation(request_id, receipt)


def stage_remote_request_readiness(
    request: object,
    *,
    remote_profile: RemoteProfile,
    host_profile: HostProfile,
    observation: Mapping[str, object],
    control_plane_version: str,
    state_store: RequestStateStore,
) -> dict[str, object]:
    """Validate staged imports/assets while keeping service/execution disabled."""

    validated_request = validate_remote_request(
        request,
        profile=remote_profile,
        control_plane_version=control_plane_version,
    )
    request_id = str(validated_request["request_id"])
    current = state_store.load(request_id)
    if current.get("state") != "PREPARING":
        raise LabError(f"request is not ready for readiness staging: {request_id}")
    source = current.get("source_receipt")
    if not isinstance(source, dict):
        raise LabError("request source has not been staged")
    receipt = prepare_slot_readiness(
        validated_request,
        remote_profile=remote_profile,
        host_profile=host_profile,
        observation=observation,
        source_receipt=source,
        control_plane_version=control_plane_version,
        slot_id=str(current.get("slot_id", "")),
    )
    return state_store.attach_readiness(request_id, receipt)


def stage_remote_service_launch_spec(
    request: object,
    *,
    remote_profile: RemoteProfile,
    control_plane_version: str,
    state_store: RequestStateStore,
    launcher_nonce: str,
) -> dict[str, object]:
    """Bind an inert service launch identity; never launch or probe a process."""

    validated_request = validate_remote_request(
        request,
        profile=remote_profile,
        control_plane_version=control_plane_version,
    )
    request_id = str(validated_request["request_id"])
    current = state_store.load(request_id)
    if current.get("state") != "PREPARING":
        raise LabError(f"request is not ready for a service launch spec: {request_id}")
    readiness = current.get("readiness_receipt")
    if not isinstance(readiness, dict):
        raise LabError("request readiness has not been staged")
    spec = build_service_launch_spec(
        validated_request,
        remote_profile=remote_profile,
        readiness_receipt=readiness,
        control_plane_version=control_plane_version,
        slot_id=str(current.get("slot_id", "")),
        launcher_nonce=launcher_nonce,
    )
    return state_store.attach_service_launch_spec(request_id, spec)


def stage_remote_service_receipt(
    request: object,
    *,
    remote_profile: RemoteProfile,
    control_plane_version: str,
    state_store: RequestStateStore,
    service_receipt: object,
) -> dict[str, object]:
    """Attach launcher-produced identity evidence without executing a command."""

    validated_request = validate_remote_request(
        request,
        profile=remote_profile,
        control_plane_version=control_plane_version,
    )
    request_id = str(validated_request["request_id"])
    receipt = validate_service_receipt(service_receipt)
    return state_store.attach_service_receipt(request_id, receipt)


def stage_remote_service_launch_plan(
    request: object,
    *,
    remote_profile: RemoteProfile,
    host_profile: HostProfile,
    backend_probe_receipt: object,
    runtime_root: Path,
    adapter_root: Path,
    endpoint_port: int,
    control_plane_version: str,
    state_store: RequestStateStore,
) -> dict[str, object]:
    """Compile and persist an inert protected service plan; never launch it.

    The Task request cannot provide an adapter path, argv, environment, mounts,
    or endpoint host. The installed control plane selects the adapter by the
    protected Profile command ID. The endpoint receives a framework-local
    endpoint lease and a shared-network topology plan, but creates no namespace,
    binds no socket, and grants no launch authority.
    """

    validated_request = validate_remote_request(
        request,
        profile=remote_profile,
        control_plane_version=control_plane_version,
    )
    request_id = str(validated_request["request_id"])
    current = state_store.load(request_id)
    if current.get("state") != "PREPARING":
        raise LabError(f"request is not ready for a service launch plan: {request_id}")
    spec = current.get("service_launch_spec")
    readiness = current.get("readiness_receipt")
    if not isinstance(spec, dict) or not isinstance(readiness, dict):
        raise LabError("service launch plan requires readiness and launch spec")
    policy = remote_profile.service_policy
    if spec.get("launch_command_id") != policy.launch_command_id:
        raise LabError("service launch spec differs from protected Profile adapter")
    adapter = load_trusted_service_adapter(
        adapter_root,
        policy.launch_command_id,
    )
    endpoint_policy = spec.get("endpoint_policy")
    if not isinstance(endpoint_policy, dict):
        raise LabError("service launch spec lacks an endpoint policy")
    endpoint_store = EndpointLeaseStore(
        state_store.root / "endpoint-leases",
        host_id=host_profile.host_id,
        host_profile_sha256=host_profile.sha256,
    )

    existing_endpoint = current.get("endpoint_lease_receipt")
    existing_network = current.get("network_namespace_plan")
    existing_sandbox = current.get("service_sandbox_receipt")
    existing_plan = current.get("service_launch_plan")
    if (
        existing_endpoint is not None
        or existing_network is not None
        or existing_sandbox is not None
        or existing_plan is not None
    ):
        if (
            not isinstance(existing_endpoint, dict)
            or not isinstance(existing_network, dict)
            or not isinstance(existing_sandbox, dict)
            or not isinstance(existing_plan, dict)
        ):
            raise LabError("request has a partial service launch plan")
        endpoint = existing_plan.get("endpoint")
        raw_adapter = existing_plan.get("adapter")
        if (
            not isinstance(endpoint, dict)
            or endpoint.get("port") != endpoint_port
            or not isinstance(raw_adapter, dict)
            or raw_adapter.get("manifest_sha256") != adapter.manifest_sha256
            or raw_adapter.get("adapter_id") != adapter.adapter_id
            or existing_endpoint.get("port") != endpoint_port
            or existing_network.get("endpoint_lease_receipt_sha256")
            != existing_endpoint.get("receipt_sha256")
        ):
            raise LabError("request already has a different service launch plan")
        active_endpoint = endpoint_store.active_lease(port=endpoint_port)
        if active_endpoint != existing_endpoint:
            raise LabError("request service endpoint lease is no longer active")
        return validate_request_state_identity(
            current,
            request=validated_request,
        )

    endpoint_lease = endpoint_store.acquire(
        request_id=request_id,
        slot_id=str(current.get("slot_id", "")),
        scheme=str(endpoint_policy.get("scheme", "")),
        port=endpoint_port,
    )
    try:
        network_plan = build_network_namespace_plan(
            endpoint_lease,
            host_profile=host_profile,
        )
        sandbox = build_service_sandbox_receipt(
            launch_spec=spec,
            readiness_receipt=readiness,
            backend_probe_receipt=backend_probe_receipt,
            host_profile=host_profile,
            control_plane_version=control_plane_version,
            runtime_root=runtime_root,
        )
        plan = build_service_launch_plan(
            launch_spec=spec,
            service_sandbox_receipt=sandbox,
            host_profile=host_profile,
            adapter=adapter,
            endpoint_lease_receipt=endpoint_lease,
            network_namespace_plan=network_plan,
        )
        return state_store.attach_service_launch_plan(
            request_id,
            service_sandbox_receipt=sandbox,
            launch_plan=plan,
            endpoint_lease_receipt=endpoint_lease,
            network_namespace_plan=network_plan,
            host_profile=host_profile,
            adapter=adapter,
        )
    except Exception as exc:
        try:
            endpoint_store.release_nonstarted(endpoint_lease)
        except LabError as cleanup_error:
            raise LabError(
                f"service launch plan failed: {exc}; endpoint cleanup failed: "
                f"{cleanup_error}"
            ) from exc
        raise


def stage_remote_network_namespace_evidence(
    request: object,
    *,
    remote_profile: RemoteProfile,
    host_profile: HostProfile,
    control_plane_version: str,
    state_store: RequestStateStore,
    anchor_pid: int,
    policy: NetworkNamespaceProbePolicy,
    proc_root: Path = Path("/proc"),
    collector_pid: int | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Observe and persist one live anchor/probe pair; never create an anchor.

    A trusted launcher must have created the direct-child anchor beforehand.
    Replays revalidate the stored probe against live executables and `/proc`
    rather than generating a new observation epoch.
    """

    validated_request = validate_remote_request(
        request,
        profile=remote_profile,
        control_plane_version=control_plane_version,
    )
    request_id = str(validated_request["request_id"])
    current = state_store.load(request_id)
    if current.get("state") != "PREPARING":
        raise LabError(
            f"request is not ready for network namespace evidence: {request_id}"
        )
    current = validate_request_state_identity(current, request=validated_request)
    plan = current.get("network_namespace_plan")
    if not isinstance(plan, dict) or not isinstance(
        current.get("service_launch_plan"), dict
    ):
        raise LabError("network namespace evidence requires a service launch plan")
    existing_anchor = current.get("network_namespace_anchor_observation")
    existing_probe = current.get("network_namespace_probe_receipt")
    if existing_anchor is not None or existing_probe is not None:
        if not isinstance(existing_anchor, dict) or not isinstance(
            existing_probe, dict
        ):
            raise LabError("request has partial network namespace evidence")
        if existing_anchor.get("pid") != anchor_pid:
            raise LabError("request already uses a different namespace anchor")
        validate_network_namespace_probe_for_use(
            existing_probe,
            network_namespace_plan=plan,
            anchor_observation=existing_anchor,
            host_profile=host_profile,
            policy=policy,
            proc_root=proc_root,
            now=now,
        )
        return current
    anchor = collect_network_namespace_anchor_observation(
        plan,
        host_profile=host_profile,
        pid=anchor_pid,
        proc_root=proc_root,
        collector_pid=collector_pid,
        now=now,
    )
    probe = collect_network_namespace_probe(
        plan,
        anchor_observation=anchor,
        host_profile=host_profile,
        policy=policy,
        proc_root=proc_root,
        now=now,
    )
    attached = state_store.attach_network_namespace_evidence(
        request_id,
        anchor_observation=anchor,
        probe_receipt=probe,
        host_profile=host_profile,
        policy=policy,
    )
    return validate_request_state_identity(attached, request=validated_request)


def execute_remote_request_attempt(
    request: object,
    *,
    remote_profile: RemoteProfile,
    control_plane_version: str,
    state_store: RequestStateStore,
    command_registry: TrustedCommandRegistry,
    command_id: str,
    executor_nonce: str,
    execution_sandbox_receipt: object,
    runtime_root: Path,
    adapter: TrustedCommandAdapter,
) -> dict[str, object]:
    """Run one protected adapter after every source/service identity is bound.

    The adapter is trusted control-plane code.  No argv, cwd, environment, or
    artifact path is accepted from the Task request.  The adapter receives a
    receipt that authorizes loopback-only networking and forbids downloads;
    the eventual process launcher must enforce that envelope.  This initial
    R1 edge is intentionally single-attempt and rejects all device-enabled
    Profiles.
    """

    validated_request = validate_remote_request(
        request,
        profile=remote_profile,
        control_plane_version=control_plane_version,
    )
    if remote_profile.device_access:
        raise LabError("trusted GPU attempt refuses device-enabled Profiles")
    if (
        remote_profile.network_egress != "loopback-only"
        or remote_profile.downloads_permitted
    ):
        raise LabError("trusted GPU attempt requires an offline payload envelope")
    command = resolve_trusted_command(
        command_registry,
        profile=remote_profile,
        command_id=command_id,
    )
    request_id = str(validated_request["request_id"])
    current = state_store.load(request_id)
    if current.get("state") != "PREPARING":
        raise LabError(f"request is not ready for execution: {request_id}")
    launch_spec = current.get("service_launch_spec")
    service_receipt = current.get("service_receipt")
    if not isinstance(launch_spec, dict) or not isinstance(service_receipt, dict):
        raise LabError("trusted attempt requires launch spec and service receipt")
    attempts = current.get("attempts")
    if not isinstance(attempts, list):
        raise LabError("request attempts state is invalid")
    sandbox = validate_execution_sandbox_for_launch(execution_sandbox_receipt)
    current = state_store.attach_execution_sandbox(
        request_id,
        sandbox,
    )
    start = build_attempt_start(
        validated_request,
        remote_profile=remote_profile,
        control_plane_version=control_plane_version,
        launch_spec=launch_spec,
        service_receipt=service_receipt,
        execution_sandbox_receipt=sandbox,
        command_id=command_id,
        command_adapter_id=command.adapter_id,
        command_registry_sha256=command_registry.sha256,
        executor_nonce=executor_nonce,
        attempt_number=len(attempts) + 1,
    )
    requested_root = runtime_root.expanduser()
    root = requested_root.resolve()
    if (
        not requested_root.is_absolute()
        or requested_root.is_symlink()
        or requested_root.absolute() != root
    ):
        raise LabError(f"attempt runtime root is unsafe: {runtime_root}")
    root.mkdir(parents=True, exist_ok=True)
    request_root = root / request_id
    if request_root.is_symlink():
        raise LabError(f"attempt request runtime path is a symlink: {request_root}")
    request_root.mkdir(exist_ok=True)
    if request_root.resolve() != request_root.absolute():
        raise LabError(f"attempt request runtime path escaped: {request_root}")
    target = request_root / str(start["attempt_id"])
    if target.exists() or target.is_symlink():
        raise LabError(f"attempt runtime path already exists: {target}")
    state_store.begin_attempt(request_id, start)
    try:
        target.mkdir(parents=True)
        outcome = adapter(start, target)
        artifacts = collect_attempt_artifacts(target, outcome.artifact_paths)
        result = build_attempt_result(
            start,
            exit_code=outcome.exit_code,
            assertions_passed=outcome.assertions_passed,
            task_success=outcome.task_success,
            service_acknowledged=outcome.service_acknowledged,
            artifacts=artifacts,
        )
        return state_store.finish_attempt(request_id, result)
    except Exception as exc:
        active = state_store.load(request_id)
        if active.get("state") == "RUNNING":
            state_store.fail(
                request_id,
                classification="INFRASTRUCTURE_FAILURE",
                detail=str(exc) or type(exc).__name__,
            )
        if (
            not target.is_symlink()
            and target.is_dir()
            and target.resolve() == target.absolute()
            and not any(target.iterdir())
        ):
            shutil.rmtree(target)
        raise
