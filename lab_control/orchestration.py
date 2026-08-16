from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

from .queue_profiles import load_queue_profile
from .run_requests import RunRequestStore
from .runner_protocol import RunnerProfile, load_runner_profile
from .workspace import (
    LabError,
    validate_repository_name,
    validate_task_id,
)

DISTRIBUTED_PROFILE_SCHEMA_ID = "lab.distributed-profile.v1"
ORCHESTRATION_PLAN_SCHEMA_ID = "lab.orchestration-plan.v1"
COMPOSITE_RESERVATION_SCHEMA_ID = "lab.composite-reservation.v1"
RESERVATION_RELEASE_SCHEMA_ID = "lab.composite-reservation-release.v1"
SERVICE_RECEIPT_SCHEMA_ID = "lab.service-receipt.v1"
READY_BARRIER_SCHEMA_ID = "lab.ready-barrier.v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_PROFILE_BYTES = 512 * 1024
_MAX_STATE_BYTES = 2 * 1024 * 1024
_MAX_NODES = 32
_MAX_SERVICES = 64
_ALLOWED_ENDPOINT_SCHEMES = frozenset({"http", "https", "tcp", "unix"})


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _observed(value: datetime | None) -> datetime:
    observed = value or _utc_now()
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise LabError("orchestration time must include a UTC offset")
    return observed


def _timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise LabError(f"invalid {label}: {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise LabError(f"invalid {label}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LabError(f"{label} must include a UTC offset")
    return parsed


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise LabError(f"invalid {label}: {value!r}")
    return value


def _ids(value: object, *, label: str, allow_empty: bool = False) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        raise LabError(f"{label} must be a unique {qualifier} string list")
    return tuple(validate_task_id(item) for item in value)


def _repositories(value: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        raise LabError(f"{label} must be a unique non-empty string list")
    return tuple(validate_repository_name(item) for item in value)


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise LabError(f"orchestration evidence is missing or unsafe: {path}")
    try:
        if path.stat().st_size > _MAX_STATE_BYTES:
            raise LabError(f"orchestration evidence exceeds its size limit: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read orchestration evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabError(f"orchestration evidence must be an object: {path}")
    return payload


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if len(encoded.encode()) > _MAX_STATE_BYTES:
        raise LabError("orchestration evidence exceeds its size limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise LabError(f"orchestration evidence already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class ServiceDefinition:
    service_id: str
    contract_sha256: str

    def public_identity(self) -> dict[str, str]:
        return {
            "service_id": self.service_id,
            "contract_sha256": self.contract_sha256,
        }


@dataclass(frozen=True)
class DistributedNode:
    node_id: str
    target_id: str
    repositories: tuple[str, ...]
    command_ids: tuple[str, ...]
    capabilities: tuple[str, ...]
    resources: tuple[str, ...]
    produces_services: tuple[ServiceDefinition, ...]

    def public_identity(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "target_id": self.target_id,
            "repositories": list(self.repositories),
            "command_ids": list(self.command_ids),
            "capabilities": list(self.capabilities),
            "resources": list(self.resources),
            "produces_services": [
                service.public_identity() for service in self.produces_services
            ],
        }


@dataclass(frozen=True)
class ServiceDependency:
    producer_node: str
    consumer_node: str
    service_id: str

    def public_identity(self) -> dict[str, str]:
        return {
            "producer_node": self.producer_node,
            "consumer_node": self.consumer_node,
            "service_id": self.service_id,
        }


@dataclass(frozen=True)
class DistributedProfile:
    profile_id: str
    description: str
    queue_profile_id: str
    target_id: str
    nodes: tuple[DistributedNode, ...]
    dependencies: tuple[ServiceDependency, ...]
    path: Path
    sha256: str

    def node(self, node_id: str) -> DistributedNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise LabError(f"distributed Profile has no node: {node_id}")

    def service(self, service_id: str) -> tuple[DistributedNode, ServiceDefinition]:
        for node in self.nodes:
            for service in node.produces_services:
                if service.service_id == service_id:
                    return node, service
        raise LabError(f"distributed Profile has no service: {service_id}")


def load_distributed_profile(profile_root: Path, profile_id: str) -> DistributedProfile:
    validate_task_id(profile_id)
    root = profile_root.expanduser().resolve()
    candidate = root / f"{profile_id}.json"
    if candidate.parent != root or candidate.is_symlink() or not candidate.is_file():
        raise LabError(f"unknown or unsafe distributed Profile: {profile_id}")
    path = candidate.resolve()
    if path.parent != root:
        raise LabError(f"distributed Profile path escapes trusted root: {profile_id}")
    try:
        if path.stat().st_size > _MAX_PROFILE_BYTES:
            raise LabError(f"distributed Profile exceeds its size limit: {path}")
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read distributed Profile {path}: {exc}") from exc
    expected = {
        "schema",
        "profile_id",
        "description",
        "queue_profile_id",
        "target_id",
        "nodes",
        "dependencies",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LabError(f"distributed Profile has an unexpected shape: {path}")
    if payload.get("schema") != DISTRIBUTED_PROFILE_SCHEMA_ID:
        raise LabError(f"unsupported distributed Profile schema: {path}")
    if payload.get("profile_id") != profile_id:
        raise LabError(f"distributed Profile identity mismatch: {path}")
    description = payload.get("description")
    if not isinstance(description, str) or not description or len(description) > 512:
        raise LabError(f"distributed Profile has an invalid description: {path}")
    queue_profile_id = validate_task_id(str(payload.get("queue_profile_id", "")))
    target_id = validate_task_id(str(payload.get("target_id", "")))
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or not 2 <= len(raw_nodes) <= _MAX_NODES:
        raise LabError("distributed Profile requires between 2 and 32 nodes")
    nodes: list[DistributedNode] = []
    node_ids: set[str] = set()
    service_ids: set[str] = set()
    resource_ids: set[str] = set()
    command_ids: set[str] = set()
    for raw_node in raw_nodes:
        node_fields = {
            "node_id",
            "target_id",
            "repositories",
            "command_ids",
            "capabilities",
            "resources",
            "produces_services",
        }
        if not isinstance(raw_node, dict) or set(raw_node) != node_fields:
            raise LabError("distributed Profile node has an unexpected shape")
        node_id = validate_task_id(str(raw_node.get("node_id", "")))
        if node_id in node_ids:
            raise LabError(f"duplicate distributed node: {node_id}")
        node_ids.add(node_id)
        node_commands = _ids(raw_node.get("command_ids"), label="node commands")
        overlap = command_ids & set(node_commands)
        if overlap:
            raise LabError("commands belong to multiple nodes: " + ", ".join(sorted(overlap)))
        command_ids.update(node_commands)
        node_resources = _ids(
            raw_node.get("resources"), label="node resources", allow_empty=True
        )
        overlap = resource_ids & set(node_resources)
        if overlap:
            raise LabError(
                "exclusive resources belong to multiple nodes: "
                + ", ".join(sorted(overlap))
            )
        resource_ids.update(node_resources)
        raw_services = raw_node.get("produces_services")
        if not isinstance(raw_services, list):
            raise LabError("node produced services must be a list")
        services: list[ServiceDefinition] = []
        for raw_service in raw_services:
            if not isinstance(raw_service, dict) or set(raw_service) != {
                "service_id",
                "contract_sha256",
            }:
                raise LabError("service definition has an unexpected shape")
            service_id = validate_task_id(str(raw_service.get("service_id", "")))
            if service_id in service_ids or len(service_ids) >= _MAX_SERVICES:
                raise LabError(f"duplicate or excessive service definition: {service_id}")
            service_ids.add(service_id)
            services.append(
                ServiceDefinition(
                    service_id=service_id,
                    contract_sha256=_sha256(
                        raw_service.get("contract_sha256"),
                        label=f"service contract {service_id}",
                    ),
                )
            )
        nodes.append(
            DistributedNode(
                node_id=node_id,
                target_id=validate_task_id(str(raw_node.get("target_id", ""))),
                repositories=_repositories(
                    raw_node.get("repositories"), label="node repositories"
                ),
                command_ids=node_commands,
                capabilities=_ids(
                    raw_node.get("capabilities"), label="node capabilities"
                ),
                resources=node_resources,
                produces_services=tuple(services),
            )
        )
    raw_dependencies = payload.get("dependencies")
    if not isinstance(raw_dependencies, list):
        raise LabError("distributed Profile dependencies must be a list")
    dependencies: list[ServiceDependency] = []
    dependency_ids: set[tuple[str, str, str]] = set()
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    service_producers = {
        service.service_id: node.node_id
        for node in nodes
        for service in node.produces_services
    }
    for raw_dependency in raw_dependencies:
        if not isinstance(raw_dependency, dict) or set(raw_dependency) != {
            "producer_node",
            "consumer_node",
            "service_id",
        }:
            raise LabError("distributed dependency has an unexpected shape")
        producer = validate_task_id(str(raw_dependency.get("producer_node", "")))
        consumer = validate_task_id(str(raw_dependency.get("consumer_node", "")))
        service_id = validate_task_id(str(raw_dependency.get("service_id", "")))
        identity = (producer, consumer, service_id)
        if (
            producer not in node_ids
            or consumer not in node_ids
            or producer == consumer
            or identity in dependency_ids
            or service_producers.get(service_id) != producer
        ):
            raise LabError(f"invalid distributed dependency: {identity}")
        dependency_ids.add(identity)
        dependencies.append(
            ServiceDependency(
                producer_node=producer,
                consumer_node=consumer,
                service_id=service_id,
            )
        )
        adjacency[producer].add(consumer)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise LabError("distributed Profile dependencies contain a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for child in sorted(adjacency[node_id]):
            visit(child)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in sorted(node_ids):
        visit(node_id)
    return DistributedProfile(
        profile_id=profile_id,
        description=description,
        queue_profile_id=queue_profile_id,
        target_id=target_id,
        nodes=tuple(sorted(nodes, key=lambda node: node.node_id)),
        dependencies=tuple(
            sorted(
                dependencies,
                key=lambda item: (
                    item.producer_node,
                    item.consumer_node,
                    item.service_id,
                ),
            )
        ),
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _endpoint(value: str) -> str:
    if not value or len(value) > 2048 or any(character.isspace() for character in value):
        raise LabError(f"invalid service endpoint: {value!r}")
    parsed = urlparse(value)
    if (
        parsed.scheme not in _ALLOWED_ENDPOINT_SCHEMES
        or parsed.fragment
        or parsed.query
    ):
        raise LabError(f"invalid service endpoint: {value!r}")
    if parsed.username is not None or parsed.password is not None:
        raise LabError("service endpoint must not embed credentials")
    try:
        if parsed.scheme == "unix":
            if parsed.netloc or not parsed.path.startswith("/"):
                raise LabError(f"invalid unix service endpoint: {value!r}")
        elif not parsed.hostname or parsed.port is None:
            raise LabError(
                f"network service endpoint requires host and port: {value!r}"
            )
    except ValueError as exc:
        raise LabError(f"invalid service endpoint: {value!r}") from exc
    return value


class OrchestrationManager:
    """Resolve and reserve a distributed Run without launching its payload."""

    def __init__(
        self,
        store: RunRequestStore,
        *,
        distributed_profile_root: Path,
        queue_profile_root: Path,
        runner_profile_root: Path,
        state_root: Path | None = None,
    ) -> None:
        self.store = store
        self.distributed_profile_root = distributed_profile_root.expanduser().resolve()
        self.queue_profile_root = queue_profile_root.expanduser().resolve()
        self.runner_profile_root = runner_profile_root.expanduser().resolve()
        self.root = (
            state_root.expanduser().resolve()
            if state_root is not None
            else store.root / "orchestration"
        )
        self.lock_path = self.root / "orchestration.lock"

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise LabError(f"orchestration root must not be a symlink: {self.root}")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.lock_path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _runner_profiles(self) -> dict[str, RunnerProfile]:
        profiles: dict[str, RunnerProfile] = {}
        resource_owners: dict[str, str] = {}
        for path in sorted(self.runner_profile_root.glob("*.json")):
            runner = load_runner_profile(self.runner_profile_root, path.stem)
            profiles[runner.runner_id] = runner
            for resource in runner.resources:
                previous = resource_owners.setdefault(resource, runner.runner_id)
                if previous != runner.runner_id:
                    raise LabError(
                        f"resource {resource} belongs to multiple runners: "
                        f"{previous}, {runner.runner_id}"
                    )
        if not profiles:
            raise LabError(f"no trusted runner Profiles: {self.runner_profile_root}")
        return profiles

    def _bind(
        self, request_id: str, distributed_profile_id: str
    ) -> tuple[dict[str, object], DistributedProfile]:
        request = self.store.load_request(request_id)
        state = self.store.load(request_id)
        if state.get("state") != "SUBMITTED":
            raise LabError("composite reservation requires a SUBMITTED request")
        profile = load_distributed_profile(
            self.distributed_profile_root, distributed_profile_id
        )
        raw_request_profile = request.get("profile")
        raw_snapshot = request.get("snapshot")
        if not isinstance(raw_request_profile, dict) or not isinstance(raw_snapshot, dict):
            raise LabError("Run request lost its Profile or snapshot identity")
        if raw_request_profile.get("profile_id") != profile.queue_profile_id:
            raise LabError("distributed Profile references a different Queue Profile")
        queue_profile = load_queue_profile(
            self.queue_profile_root, profile.queue_profile_id
        )
        if (
            raw_request_profile.get("sha256") != queue_profile.sha256
            or raw_request_profile.get("execution_topology") != "distributed"
            or queue_profile.execution_topology != "distributed"
            or request.get("target_id") != queue_profile.target_id
            or profile.target_id != queue_profile.target_id
        ):
            raise LabError("distributed/Queue Profile identity drift")
        repository_union = {
            repository for node in profile.nodes for repository in node.repositories
        }
        command_union = {command for node in profile.nodes for command in node.command_ids}
        capability_union = {
            capability for node in profile.nodes for capability in node.capabilities
        }
        raw_repositories = raw_snapshot.get("repositories")
        if (
            not isinstance(raw_repositories, dict)
            or repository_union != set(queue_profile.repositories)
            or repository_union != set(raw_repositories)
            or command_union != set(queue_profile.command_ids)
            or capability_union != set(queue_profile.capabilities)
        ):
            raise LabError("distributed Profile does not exactly cover the Run request")
        return request, profile

    def _reservation_path(self, reservation_id: str) -> Path:
        root = self.root / "reservations"
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise LabError(f"reservation root is unsafe: {root}")
        return root / f"{validate_task_id(reservation_id)}.json"

    def _release_path(self, reservation_id: str) -> Path:
        root = self.root / "releases"
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise LabError(f"reservation release root is unsafe: {root}")
        return root / f"{validate_task_id(reservation_id)}.json"

    def _service_path(self, reservation_id: str, service_id: str) -> Path:
        services_root = self.root / "services"
        if services_root.is_symlink() or (
            services_root.exists() and not services_root.is_dir()
        ):
            raise LabError(f"service receipt root is unsafe: {services_root}")
        reservation_root = services_root / validate_task_id(reservation_id)
        if reservation_root.is_symlink() or (
            reservation_root.exists() and not reservation_root.is_dir()
        ):
            raise LabError(f"service reservation root is unsafe: {reservation_root}")
        return reservation_root / f"{validate_task_id(service_id)}.json"

    def _barrier_path(self, reservation_id: str) -> Path:
        root = self.root / "barriers"
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise LabError(f"ready barrier root is unsafe: {root}")
        return root / f"{validate_task_id(reservation_id)}.json"

    def _validate_reservation(self, value: object) -> dict[str, object]:
        expected = {
            "schema",
            "reservation_id",
            "request_id",
            "idempotency_key",
            "orchestrator_id",
            "distributed_profile",
            "queue_profile",
            "assignments",
            "dependencies",
            "plan_sha256",
            "reserved_at",
            "expires_at",
            "receipt_sha256",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise LabError("composite reservation has an unexpected shape")
        if value.get("schema") != COMPOSITE_RESERVATION_SCHEMA_ID:
            raise LabError("unsupported composite reservation schema")
        validate_task_id(str(value.get("reservation_id", "")))
        validate_task_id(str(value.get("request_id", "")))
        validate_task_id(str(value.get("orchestrator_id", "")))
        _sha256(value.get("idempotency_key"), label="reservation idempotency key")
        _sha256(value.get("plan_sha256"), label="reservation plan SHA-256")
        reserved_at = _timestamp(value.get("reserved_at"), label="reservation time")
        expires_at = _timestamp(value.get("expires_at"), label="reservation expiry")
        if expires_at <= reserved_at:
            raise LabError("composite reservation expiry is not after reservation")
        assignments = value.get("assignments")
        if not isinstance(assignments, list) or not assignments:
            raise LabError("composite reservation has no assignments")
        seen_nodes: set[str] = set()
        seen_slots: set[tuple[str, str]] = set()
        seen_resources: set[str] = set()
        for assignment in assignments:
            fields = {
                "node_id",
                "runner_id",
                "runner_profile_sha256",
                "host_id",
                "slot_id",
                "resources",
            }
            if not isinstance(assignment, dict) or set(assignment) != fields:
                raise LabError("composite assignment has an unexpected shape")
            node_id = validate_task_id(str(assignment.get("node_id", "")))
            runner_id = validate_task_id(str(assignment.get("runner_id", "")))
            slot_id = validate_task_id(str(assignment.get("slot_id", "")))
            validate_task_id(str(assignment.get("host_id", "")))
            _sha256(
                assignment.get("runner_profile_sha256"),
                label=f"runner Profile {runner_id}",
            )
            resources = _ids(
                assignment.get("resources"),
                label="assignment resources",
                allow_empty=True,
            )
            if (
                node_id in seen_nodes
                or (runner_id, slot_id) in seen_slots
                or seen_resources & set(resources)
            ):
                raise LabError("composite reservation contains overlapping assignments")
            seen_nodes.add(node_id)
            seen_slots.add((runner_id, slot_id))
            seen_resources.update(resources)
        raw_profile = value.get("distributed_profile")
        raw_queue = value.get("queue_profile")
        if not isinstance(raw_profile, dict) or set(raw_profile) != {
            "profile_id",
            "sha256",
        }:
            raise LabError("reservation distributed Profile identity is invalid")
        if not isinstance(raw_queue, dict) or set(raw_queue) != {
            "profile_id",
            "sha256",
        }:
            raise LabError("reservation Queue Profile identity is invalid")
        validate_task_id(str(raw_profile.get("profile_id", "")))
        validate_task_id(str(raw_queue.get("profile_id", "")))
        _sha256(raw_profile.get("sha256"), label="distributed Profile SHA-256")
        _sha256(raw_queue.get("sha256"), label="Queue Profile SHA-256")
        dependencies = value.get("dependencies")
        if not isinstance(dependencies, list):
            raise LabError("reservation dependencies must be a list")
        for dependency in dependencies:
            if not isinstance(dependency, dict) or set(dependency) != {
                "producer_node",
                "consumer_node",
                "service_id",
            }:
                raise LabError("reservation dependency has an unexpected shape")
            validate_task_id(str(dependency.get("producer_node", "")))
            validate_task_id(str(dependency.get("consumer_node", "")))
            validate_task_id(str(dependency.get("service_id", "")))
        plan_content = {
            "schema": ORCHESTRATION_PLAN_SCHEMA_ID,
            "request_id": value["request_id"],
            "idempotency_key": value["idempotency_key"],
            "distributed_profile": value["distributed_profile"],
            "queue_profile": value["queue_profile"],
            "assignments": value["assignments"],
            "dependencies": value["dependencies"],
        }
        if value.get("plan_sha256") != _sha256_json(plan_content):
            raise LabError("composite reservation plan hash mismatch")
        content = {
            key: item
            for key, item in value.items()
            if key not in {"reservation_id", "receipt_sha256"}
        }
        if value.get("receipt_sha256") != _sha256_json(content):
            raise LabError("composite reservation hash mismatch")
        return value

    def _release_receipt(
        self, reservation: Mapping[str, object]
    ) -> dict[str, object] | None:
        reservation_id = str(reservation.get("reservation_id", ""))
        path = self._release_path(reservation_id)
        if not path.exists():
            return None
        receipt = _read_json(path)
        expected = {
            "schema",
            "request_id",
            "reservation_id",
            "reservation_receipt_sha256",
            "actor",
            "reason",
            "released_at",
            "receipt_sha256",
        }
        if set(receipt) != expected:
            raise LabError("reservation release receipt has an unexpected shape")
        if (
            receipt.get("schema") != RESERVATION_RELEASE_SCHEMA_ID
            or receipt.get("request_id") != reservation.get("request_id")
            or receipt.get("reservation_id") != reservation_id
            or receipt.get("reservation_receipt_sha256")
            != reservation.get("receipt_sha256")
        ):
            raise LabError("reservation release receipt identity mismatch")
        validate_task_id(str(receipt.get("actor", "")))
        reason = receipt.get("reason")
        if not isinstance(reason, str) or not reason or len(reason) > 512:
            raise LabError("reservation release receipt has an invalid reason")
        _timestamp(receipt.get("released_at"), label="reservation release time")
        content = {
            key: item for key, item in receipt.items() if key != "receipt_sha256"
        }
        if receipt.get("receipt_sha256") != _sha256_json(content):
            raise LabError("reservation release receipt hash mismatch")
        return receipt

    def _active_reservations(self, now: datetime) -> list[dict[str, object]]:
        active: list[dict[str, object]] = []
        root = self.root / "reservations"
        if root.is_symlink():
            raise LabError(f"reservation root must not be a symlink: {root}")
        if not root.exists():
            return active
        for path in sorted(root.glob("*.json")):
            reservation = self._validate_reservation(_read_json(path))
            if self._release_receipt(reservation) is None and _timestamp(
                reservation["expires_at"], label="reservation expiry"
            ) > now:
                active.append(reservation)
        return active

    def _plan_locked(
        self,
        request_id: str,
        distributed_profile_id: str,
        *,
        now: datetime,
    ) -> dict[str, object]:
        request, profile = self._bind(request_id, distributed_profile_id)
        runners = self._runner_profiles()
        occupied_slots: set[tuple[str, str]] = set()
        occupied_resources: set[str] = set()
        for reservation in self._active_reservations(now):
            raw_assignments = reservation["assignments"]
            assert isinstance(raw_assignments, list)
            for assignment in raw_assignments:
                assert isinstance(assignment, dict)
                occupied_slots.add(
                    (str(assignment["runner_id"]), str(assignment["slot_id"]))
                )
                resources = assignment["resources"]
                assert isinstance(resources, list)
                occupied_resources.update(str(resource) for resource in resources)
        raw_request_profile = request["profile"]
        assert isinstance(raw_request_profile, dict)
        evidence_tier = str(request["evidence_tier"])
        risk_level = str(raw_request_profile["risk_level"])
        candidates: dict[str, list[tuple[str, str]]] = {}
        for node in profile.nodes:
            if occupied_resources & set(node.resources):
                candidates[node.node_id] = []
                continue
            node_candidates: list[tuple[str, str]] = []
            for runner in runners.values():
                if (
                    node.target_id not in runner.target_ids
                    or not set(node.capabilities).issubset(runner.capabilities)
                    or not set(node.resources).issubset(runner.resources)
                    or risk_level not in runner.accepted_risk_levels
                    or evidence_tier not in runner.accepted_evidence_tiers
                ):
                    continue
                for slot in runner.slots:
                    if (runner.runner_id, slot) not in occupied_slots:
                        node_candidates.append((runner.runner_id, slot))
            candidates[node.node_id] = sorted(node_candidates)
        assignment_slots: dict[str, tuple[str, str]] = {}
        used_slots = set(occupied_slots)
        node_order = sorted(profile.nodes, key=lambda node: (len(candidates[node.node_id]), node.node_id))

        def assign(index: int) -> bool:
            if index == len(node_order):
                return True
            node = node_order[index]
            for runner_id, slot_id in candidates[node.node_id]:
                identity = (runner_id, slot_id)
                if identity in used_slots:
                    continue
                used_slots.add(identity)
                assignment_slots[node.node_id] = identity
                if assign(index + 1):
                    return True
                assignment_slots.pop(node.node_id, None)
                used_slots.remove(identity)
            return False

        if not assign(0):
            unavailable = [
                node.node_id for node in profile.nodes if not candidates[node.node_id]
            ]
            detail = ", ".join(unavailable or sorted(candidates))
            raise LabError(f"cannot atomically reserve all distributed nodes: {detail}")
        assignments: list[dict[str, object]] = []
        for node in profile.nodes:
            runner_id, slot_id = assignment_slots[node.node_id]
            runner = runners[runner_id]
            assignments.append(
                {
                    "node_id": node.node_id,
                    "runner_id": runner_id,
                    "runner_profile_sha256": runner.sha256,
                    "host_id": runner.host_id,
                    "slot_id": slot_id,
                    "resources": list(node.resources),
                }
            )
        queue_profile = load_queue_profile(
            self.queue_profile_root, profile.queue_profile_id
        )
        content: dict[str, object] = {
            "schema": ORCHESTRATION_PLAN_SCHEMA_ID,
            "request_id": request_id,
            "idempotency_key": request["idempotency_key"],
            "distributed_profile": {
                "profile_id": profile.profile_id,
                "sha256": profile.sha256,
            },
            "queue_profile": {
                "profile_id": queue_profile.profile_id,
                "sha256": queue_profile.sha256,
            },
            "assignments": assignments,
            "dependencies": [
                dependency.public_identity() for dependency in profile.dependencies
            ],
        }
        return {**content, "plan_sha256": _sha256_json(content)}

    def plan(
        self,
        request_id: str,
        distributed_profile_id: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, object]:
        observed = _observed(now)
        with self._lock():
            return self._plan_locked(
                request_id, distributed_profile_id, now=observed
            )

    def reserve(
        self,
        request_id: str,
        distributed_profile_id: str,
        *,
        orchestrator_id: str,
        ttl_seconds: int = 3600,
        now: datetime | None = None,
    ) -> dict[str, object]:
        orchestrator_id = validate_task_id(orchestrator_id)
        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool) or not 60 <= ttl_seconds <= 86400:
            raise LabError("composite reservation TTL must be between 60 and 86400 seconds")
        observed = _observed(now)
        with self._lock():
            for existing in self._active_reservations(observed):
                if existing.get("request_id") == request_id:
                    raw_profile = existing.get("distributed_profile")
                    if (
                        isinstance(raw_profile, dict)
                        and raw_profile.get("profile_id") == distributed_profile_id
                        and existing.get("orchestrator_id") == orchestrator_id
                    ):
                        return existing
                    raise LabError("Run request already has a different active reservation")
            plan = self._plan_locked(
                request_id, distributed_profile_id, now=observed
            )
            reserved_at = observed.isoformat()
            expires_at = (observed + timedelta(seconds=ttl_seconds)).isoformat()
            content = {
                "schema": COMPOSITE_RESERVATION_SCHEMA_ID,
                "request_id": plan["request_id"],
                "idempotency_key": plan["idempotency_key"],
                "orchestrator_id": orchestrator_id,
                "distributed_profile": plan["distributed_profile"],
                "queue_profile": plan["queue_profile"],
                "assignments": plan["assignments"],
                "dependencies": plan["dependencies"],
                "plan_sha256": plan["plan_sha256"],
                "reserved_at": reserved_at,
                "expires_at": expires_at,
            }
            receipt_sha256 = _sha256_json(content)
            reservation = {
                **content,
                "reservation_id": f"RSV-{receipt_sha256[:24]}",
                "receipt_sha256": receipt_sha256,
            }
            _write_new_json(
                self._reservation_path(str(reservation["reservation_id"])),
                reservation,
            )
            return self._validate_reservation(reservation)

    def _load_reservation(
        self, request_id: str, reservation_id: str, *, now: datetime
    ) -> dict[str, object]:
        reservation = self._validate_reservation(
            _read_json(self._reservation_path(reservation_id))
        )
        if reservation.get("request_id") != request_id:
            raise LabError("reservation belongs to a different Run request")
        if self._release_receipt(reservation) is not None:
            raise LabError("composite reservation has been released")
        if _timestamp(reservation["expires_at"], label="reservation expiry") <= now:
            raise LabError("composite reservation has expired")
        state = self.store.load(request_id)
        if state.get("state") != "SUBMITTED":
            raise LabError(
                f"Run request is no longer active for orchestration: "
                f"{state.get('state')}"
            )
        raw_queue = reservation.get("queue_profile")
        if not isinstance(raw_queue, dict):
            raise LabError("reservation lost its Queue Profile identity")
        queue_profile = load_queue_profile(
            self.queue_profile_root, str(raw_queue.get("profile_id", ""))
        )
        if queue_profile.sha256 != raw_queue.get("sha256"):
            raise LabError("Queue Profile changed after reservation")
        return reservation

    def release(
        self,
        request_id: str,
        reservation_id: str,
        *,
        actor: str,
        reason: str,
        now: datetime | None = None,
    ) -> dict[str, object]:
        actor = validate_task_id(actor)
        if not reason or len(reason) > 512:
            raise LabError("reservation release requires a bounded reason")
        observed = _observed(now)
        with self._lock():
            reservation = self._validate_reservation(
                _read_json(self._reservation_path(reservation_id))
            )
            if reservation.get("request_id") != request_id:
                raise LabError("reservation belongs to a different Run request")
            path = self._release_path(reservation_id)
            if path.exists():
                existing = self._release_receipt(reservation)
                assert existing is not None
                return existing
            content: dict[str, object] = {
                "schema": RESERVATION_RELEASE_SCHEMA_ID,
                "request_id": request_id,
                "reservation_id": reservation_id,
                "reservation_receipt_sha256": reservation["receipt_sha256"],
                "actor": actor,
                "reason": reason,
                "released_at": observed.isoformat(),
            }
            release = {**content, "receipt_sha256": _sha256_json(content)}
            _write_new_json(path, release)
            return release

    def _current_profile_and_assignment(
        self,
        reservation: Mapping[str, object],
        service_id: str,
    ) -> tuple[DistributedNode, ServiceDefinition, dict[str, object]]:
        raw_profile = reservation.get("distributed_profile")
        if not isinstance(raw_profile, dict):
            raise LabError("reservation lost its distributed Profile identity")
        profile = load_distributed_profile(
            self.distributed_profile_root, str(raw_profile.get("profile_id", ""))
        )
        if profile.sha256 != raw_profile.get("sha256"):
            raise LabError("distributed Profile changed after reservation")
        node, service = profile.service(service_id)
        raw_assignments = reservation.get("assignments")
        if not isinstance(raw_assignments, list):
            raise LabError("reservation lost its assignments")
        assignment = next(
            (
                item
                for item in raw_assignments
                if isinstance(item, dict) and item.get("node_id") == node.node_id
            ),
            None,
        )
        if not isinstance(assignment, dict):
            raise LabError("service producer has no reserved assignment")
        runner = load_runner_profile(
            self.runner_profile_root, str(assignment.get("runner_id", ""))
        )
        if runner.sha256 != assignment.get("runner_profile_sha256"):
            raise LabError("runner Profile changed after reservation")
        return node, service, assignment

    def record_service_ready(
        self,
        request_id: str,
        reservation_id: str,
        service_id: str,
        *,
        runner_id: str,
        endpoint: str,
        process_id: int,
        process_start_token: str,
        environment_sha256: str,
        configuration_sha256: str,
        ttl_seconds: int = 300,
        now: datetime | None = None,
    ) -> dict[str, object]:
        service_id = validate_task_id(service_id)
        runner_id = validate_task_id(runner_id)
        process_start_token = validate_task_id(process_start_token)
        if not isinstance(process_id, int) or isinstance(process_id, bool) or process_id <= 0:
            raise LabError("service process ID must be a positive integer")
        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool) or not 10 <= ttl_seconds <= 3600:
            raise LabError("service receipt TTL must be between 10 and 3600 seconds")
        observed = _observed(now)
        with self._lock():
            reservation = self._load_reservation(
                request_id, reservation_id, now=observed
            )
            node, service, assignment = self._current_profile_and_assignment(
                reservation, service_id
            )
            if assignment.get("runner_id") != runner_id:
                raise LabError("service receipt runner differs from its reservation")
            request = self.store.load_request(request_id)
            raw_snapshot = request.get("snapshot")
            if not isinstance(raw_snapshot, dict):
                raise LabError("Run request lost its snapshot")
            raw_repositories = raw_snapshot.get("repositories")
            if not isinstance(raw_repositories, dict):
                raise LabError("Run request snapshot lost its repositories")
            source: dict[str, dict[str, str]] = {}
            for repository in node.repositories:
                raw_repository = raw_repositories.get(repository)
                if not isinstance(raw_repository, dict):
                    raise LabError(f"service source repository is absent: {repository}")
                source[repository] = {
                    "commit": str(raw_repository.get("commit", "")),
                    "tree_oid": str(raw_repository.get("tree_oid", "")),
                }
            ready_at = observed.isoformat()
            expires_at = min(
                observed + timedelta(seconds=ttl_seconds),
                _timestamp(reservation["expires_at"], label="reservation expiry"),
            ).isoformat()
            content: dict[str, object] = {
                "schema": SERVICE_RECEIPT_SCHEMA_ID,
                "request_id": request_id,
                "idempotency_key": request["idempotency_key"],
                "reservation_id": reservation_id,
                "reservation_receipt_sha256": reservation["receipt_sha256"],
                "node_id": node.node_id,
                "service_id": service.service_id,
                "contract_sha256": service.contract_sha256,
                "runner_id": runner_id,
                "runner_profile_sha256": assignment["runner_profile_sha256"],
                "host_id": assignment["host_id"],
                "slot_id": assignment["slot_id"],
                "source": source,
                "environment_sha256": _sha256(
                    environment_sha256, label="service environment SHA-256"
                ),
                "configuration_sha256": _sha256(
                    configuration_sha256, label="service configuration SHA-256"
                ),
                "endpoint": _endpoint(endpoint),
                "process_id": process_id,
                "process_start_token": process_start_token,
                "status": "READY",
                "ready_at": ready_at,
                "expires_at": expires_at,
            }
            receipt_sha256 = _sha256_json(content)
            receipt = {
                **content,
                "service_instance_id": f"SVC-{receipt_sha256[:24]}",
                "receipt_sha256": receipt_sha256,
            }
            path = self._service_path(reservation_id, service_id)
            if path.exists():
                existing = _read_json(path)
                comparable = {
                    key: value
                    for key, value in existing.items()
                    if key not in {
                        "ready_at",
                        "expires_at",
                        "service_instance_id",
                        "receipt_sha256",
                    }
                }
                requested = {
                    key: value
                    for key, value in receipt.items()
                    if key not in {
                        "ready_at",
                        "expires_at",
                        "service_instance_id",
                        "receipt_sha256",
                    }
                }
                if comparable != requested:
                    raise LabError("repeated ServiceReceipt changed service identity")
                self._validate_service_receipt(
                    existing, reservation=reservation, now=observed
                )
                return existing
            _write_new_json(path, receipt)
            return self._validate_service_receipt(
                receipt, reservation=reservation, now=observed
            )

    def _validate_service_receipt(
        self,
        value: object,
        *,
        reservation: Mapping[str, object],
        now: datetime,
    ) -> dict[str, object]:
        expected = {
            "schema",
            "request_id",
            "idempotency_key",
            "reservation_id",
            "reservation_receipt_sha256",
            "node_id",
            "service_id",
            "contract_sha256",
            "runner_id",
            "runner_profile_sha256",
            "host_id",
            "slot_id",
            "source",
            "environment_sha256",
            "configuration_sha256",
            "endpoint",
            "process_id",
            "process_start_token",
            "status",
            "ready_at",
            "expires_at",
            "service_instance_id",
            "receipt_sha256",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise LabError("ServiceReceipt has an unexpected shape")
        if (
            value.get("schema") != SERVICE_RECEIPT_SCHEMA_ID
            or value.get("request_id") != reservation.get("request_id")
            or value.get("idempotency_key") != reservation.get("idempotency_key")
            or value.get("reservation_id") != reservation.get("reservation_id")
            or value.get("reservation_receipt_sha256")
            != reservation.get("receipt_sha256")
            or value.get("status") != "READY"
        ):
            raise LabError("ServiceReceipt identity mismatch")
        service_id = validate_task_id(str(value.get("service_id", "")))
        node, service, assignment = self._current_profile_and_assignment(
            reservation, service_id
        )
        if (
            value.get("node_id") != node.node_id
            or value.get("contract_sha256") != service.contract_sha256
            or value.get("runner_id") != assignment.get("runner_id")
            or value.get("runner_profile_sha256")
            != assignment.get("runner_profile_sha256")
            or value.get("host_id") != assignment.get("host_id")
            or value.get("slot_id") != assignment.get("slot_id")
        ):
            raise LabError("ServiceReceipt producer identity mismatch")
        _endpoint(str(value.get("endpoint", "")))
        process_id = value.get("process_id")
        if (
            not isinstance(process_id, int)
            or isinstance(process_id, bool)
            or process_id <= 0
        ):
            raise LabError("ServiceReceipt process ID is invalid")
        validate_task_id(str(value.get("process_start_token", "")))
        _sha256(value.get("environment_sha256"), label="service environment SHA-256")
        _sha256(
            value.get("configuration_sha256"),
            label="service configuration SHA-256",
        )
        ready_at = _timestamp(value.get("ready_at"), label="service ready time")
        expires_at = _timestamp(value.get("expires_at"), label="service expiry")
        if expires_at <= ready_at or expires_at <= now:
            raise LabError("ServiceReceipt is expired or has an invalid lifetime")
        content = {
            key: item
            for key, item in value.items()
            if key not in {"service_instance_id", "receipt_sha256"}
        }
        receipt_sha256 = _sha256_json(content)
        if (
            value.get("receipt_sha256") != receipt_sha256
            or value.get("service_instance_id") != f"SVC-{receipt_sha256[:24]}"
        ):
            raise LabError("ServiceReceipt hash mismatch")
        request = self.store.load_request(str(reservation["request_id"]))
        raw_snapshot = request.get("snapshot")
        raw_repositories = (
            raw_snapshot.get("repositories") if isinstance(raw_snapshot, dict) else None
        )
        if not isinstance(raw_repositories, dict):
            raise LabError("Run request snapshot lost its repositories")
        expected_source = {
            repository: {
                "commit": raw_repositories[repository]["commit"],
                "tree_oid": raw_repositories[repository]["tree_oid"],
            }
            for repository in node.repositories
        }
        if value.get("source") != expected_source:
            raise LabError("ServiceReceipt source differs from the Run snapshot")
        return value

    def _validate_ready_barrier(
        self,
        value: object,
        *,
        reservation: Mapping[str, object],
        services: list[dict[str, str]],
    ) -> dict[str, object]:
        expected = {
            "schema",
            "request_id",
            "reservation_id",
            "reservation_receipt_sha256",
            "services",
            "ready_at",
            "barrier_id",
            "receipt_sha256",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise LabError("ready barrier has an unexpected shape")
        if (
            value.get("schema") != READY_BARRIER_SCHEMA_ID
            or value.get("request_id") != reservation.get("request_id")
            or value.get("reservation_id") != reservation.get("reservation_id")
            or value.get("reservation_receipt_sha256")
            != reservation.get("receipt_sha256")
            or value.get("services") != services
        ):
            raise LabError("ready barrier identity mismatch")
        _timestamp(value.get("ready_at"), label="ready barrier time")
        content = {
            key: item
            for key, item in value.items()
            if key not in {"barrier_id", "receipt_sha256"}
        }
        receipt_sha256 = _sha256_json(content)
        if (
            value.get("receipt_sha256") != receipt_sha256
            or value.get("barrier_id") != f"BAR-{receipt_sha256[:24]}"
        ):
            raise LabError("ready barrier hash mismatch")
        return value

    def ready_barrier(
        self,
        request_id: str,
        reservation_id: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, object]:
        observed = _observed(now)
        with self._lock():
            reservation = self._load_reservation(
                request_id, reservation_id, now=observed
            )
            raw_dependencies = reservation.get("dependencies")
            if not isinstance(raw_dependencies, list):
                raise LabError("reservation lost its dependencies")
            service_ids = sorted(
                {
                    str(dependency["service_id"])
                    for dependency in raw_dependencies
                    if isinstance(dependency, dict)
                }
            )
            receipts: list[dict[str, str]] = []
            missing: list[str] = []
            for service_id in service_ids:
                path = self._service_path(reservation_id, service_id)
                if not path.is_file() or path.is_symlink():
                    missing.append(service_id)
                    continue
                receipt = self._validate_service_receipt(
                    _read_json(path), reservation=reservation, now=observed
                )
                receipts.append(
                    {
                        "service_id": service_id,
                        "service_instance_id": str(receipt["service_instance_id"]),
                        "receipt_sha256": str(receipt["receipt_sha256"]),
                    }
                )
            if missing:
                raise LabError("ready barrier is missing services: " + ", ".join(missing))
            content: dict[str, object] = {
                "schema": READY_BARRIER_SCHEMA_ID,
                "request_id": request_id,
                "reservation_id": reservation_id,
                "reservation_receipt_sha256": reservation["receipt_sha256"],
                "services": receipts,
                "ready_at": observed.isoformat(),
            }
            receipt_sha256 = _sha256_json(content)
            barrier = {
                **content,
                "barrier_id": f"BAR-{receipt_sha256[:24]}",
                "receipt_sha256": receipt_sha256,
            }
            path = self._barrier_path(reservation_id)
            if path.exists():
                existing = _read_json(path)
                return self._validate_ready_barrier(
                    existing,
                    reservation=reservation,
                    services=receipts,
                )
            _write_new_json(path, barrier)
            return self._validate_ready_barrier(
                barrier,
                reservation=reservation,
                services=receipts,
            )

    def inspect(self, request_id: str, *, now: datetime | None = None) -> dict[str, object]:
        self.store.load(request_id)
        observed = _observed(now)
        with self._lock():
            reservations = [
                reservation
                for reservation in self._active_reservations(observed)
                if reservation.get("request_id") == request_id
            ]
            services: list[dict[str, object]] = []
            barrier: dict[str, object] | None = None
            if reservations:
                reservation = reservations[0]
                reservation_id = str(reservation["reservation_id"])
                service_root = self.root / "services" / reservation_id
                if service_root.is_symlink():
                    raise LabError(f"service reservation root is unsafe: {service_root}")
                if service_root.is_dir():
                    services = [
                        self._validate_service_receipt(
                            _read_json(path),
                            reservation=reservation,
                            now=observed,
                        )
                        for path in sorted(service_root.glob("*.json"))
                    ]
                barrier_path = self._barrier_path(reservation_id)
                if barrier_path.is_file() and not barrier_path.is_symlink():
                    raw_barrier = _read_json(barrier_path)
                    raw_services = raw_barrier.get("services")
                    if not isinstance(raw_services, list) or not all(
                        isinstance(item, dict) for item in raw_services
                    ):
                        raise LabError("ready barrier services are invalid")
                    barrier = self._validate_ready_barrier(
                        raw_barrier,
                        reservation=reservation,
                        services=[dict(item) for item in raw_services],
                    )
            return {
                "schema": "lab.orchestration-inspection.v1",
                "request_id": request_id,
                "active_reservations": reservations,
                "services": services,
                "ready_barrier": barrier,
            }
