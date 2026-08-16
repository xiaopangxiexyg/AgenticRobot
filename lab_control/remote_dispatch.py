from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import git
from .asset_lock import load_asset_lock, resolve_asset_identities
from .asset_observation import load_asset_observations
from .remote_request import (
    build_remote_request,
    load_remote_profile,
    validate_remote_request_identity,
)
from .runs import CONTROL_PLANE_VERSION
from .snapshot_resolution import resolve_clean_snapshot_identity
from .workspace import LabError, WorkspaceManager, _atomic_write_text, validate_task_id

REMOTE_OUTBOX_SCHEMA_ID = "lab.remote-outbox.v2"


def _sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_remote_outbox(envelope: object) -> dict[str, object]:
    if not isinstance(envelope, dict):
        raise LabError("remote outbox envelope must be an object")
    expected_keys = {
        "schema",
        "outbox_id",
        "request_id",
        "task_id",
        "snapshot_id",
        "profile",
        "effective_workspace",
        "asset_lock",
        "request",
        "dispatch",
        "created_at",
        "execution_eligible",
        "envelope_sha256",
    }
    if set(envelope) != expected_keys:
        raise LabError("remote outbox envelope has an unexpected shape")
    if envelope.get("schema") != REMOTE_OUTBOX_SCHEMA_ID:
        raise LabError("unsupported remote outbox envelope schema")
    request = validate_remote_request_identity(envelope.get("request"))
    request_id = str(request.get("request_id", ""))
    if (
        envelope.get("request_id") != request_id
        or envelope.get("outbox_id") != f"O-{request_id.removeprefix('Q-')}"
        or envelope.get("task_id") != request.get("task_id")
        or envelope.get("snapshot_id") != request.get("snapshot_id")
    ):
        raise LabError("remote outbox/request identity mismatch")
    profile = envelope.get("profile")
    if not isinstance(profile, dict) or profile != request.get("profile"):
        raise LabError("remote outbox Profile identity mismatch")
    workspace = envelope.get("effective_workspace")
    if not isinstance(workspace, dict) or set(workspace) != {
        "baseline",
        "repositories",
        "repository_links",
    }:
        raise LabError("remote outbox effective workspace is invalid")
    repository_records = workspace.get("repositories")
    request_repositories = request.get("repositories")
    if (
        not isinstance(repository_records, dict)
        or not isinstance(request_repositories, dict)
        or set(repository_records) != set(request_repositories)
    ):
        raise LabError("remote outbox repository mapping mismatch")
    if workspace.get("repository_links") != request.get("repository_links"):
        raise LabError("remote outbox repository links differ from request")
    for name, record in repository_records.items():
        if not isinstance(record, dict) or set(record) != {"source", "published_refs"}:
            raise LabError(f"remote outbox repository record is invalid: {name}")
        refs = record.get("published_refs")
        if (
            not isinstance(refs, list)
            or not refs
            or not all(
                isinstance(ref, str) and ref.startswith("refs/remotes/") for ref in refs
            )
        ):
            raise LabError(f"remote outbox repository is not published: {name}")
    asset_lock = envelope.get("asset_lock")
    if not isinstance(asset_lock, dict) or set(asset_lock) != {
        "lock_id",
        "workload_id",
        "sha256",
    }:
        raise LabError("remote outbox asset lock identity is invalid")
    if not isinstance(asset_lock.get("lock_id"), str):
        raise LabError("remote outbox asset lock ID is invalid")
    validate_task_id(asset_lock["lock_id"])
    workload_id = asset_lock.get("workload_id")
    if workload_id is not None:
        if not isinstance(workload_id, str):
            raise LabError("remote outbox asset lock workload ID is invalid")
        validate_task_id(workload_id)
    asset_lock_sha = asset_lock.get("sha256")
    if (
        not isinstance(asset_lock_sha, str)
        or len(asset_lock_sha) != 64
        or any(character not in "0123456789abcdef" for character in asset_lock_sha)
    ):
        raise LabError("remote outbox asset lock hash is invalid")
    if request.get("asset_lock") != asset_lock:
        raise LabError("remote outbox/request asset lock mismatch")
    dispatch = envelope.get("dispatch")
    if not isinstance(dispatch, dict) or dispatch != {
        "status": "PENDING_DISPATCH",
        "attempts": [],
        "remote_ack": None,
    }:
        raise LabError("remote outbox must remain PENDING_DISPATCH before transport")
    if envelope.get("execution_eligible") is not False:
        raise LabError("remote outbox cannot grant execution")
    created_at = envelope.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise LabError("remote outbox created_at is invalid")
    try:
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError("remote outbox created_at is invalid") from exc
    if timestamp.tzinfo is None:
        raise LabError("remote outbox created_at requires a timezone")
    digest = envelope.get("envelope_sha256")
    content = {
        key: value for key, value in envelope.items() if key != "envelope_sha256"
    }
    if not isinstance(digest, str) or digest != _sha256(content):
        raise LabError("remote outbox envelope hash mismatch")
    return envelope


class RemoteDispatchPlanner:
    """Resolve exact remote requests into a local outbox without network effects."""

    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        self.workspace_manager = workspace_manager
        control = workspace_manager.control_repository
        environment = workspace_manager.environ
        self.remote_profile_root = (
            Path(
                environment.get(
                    "LAB_REMOTE_PROFILE_ROOT",
                    control / "integration" / "remote_profiles",
                )
            )
            .expanduser()
            .resolve()
        )
        self.baseline_path = (
            Path(
                environment.get(
                    "LAB_BASELINE_LOCK",
                    control / "stack" / "baseline.json",
                )
            )
            .expanduser()
            .resolve()
        )
        self.asset_lock_path = (
            Path(
                environment.get(
                    "LAB_REMOTE_ASSET_LOCK",
                    control / "stack" / "remote-assets.json",
                )
            )
            .expanduser()
            .resolve()
        )
        self.asset_observation_path = (
            Path(
                environment.get(
                    "LAB_REMOTE_ASSET_OBSERVATIONS",
                    control / "stack" / "remote-asset-observations.json",
                )
            )
            .expanduser()
            .resolve()
        )
        self.outbox_root = (
            Path(
                environment.get(
                    "LAB_REMOTE_OUTBOX_ROOT",
                    workspace_manager.state_root / "remote-outbox",
                )
            )
            .expanduser()
            .resolve()
        )
        self.transport_profile_root = (
            Path(
                environment.get(
                    "LAB_TRANSPORT_PROFILE_ROOT",
                    control / "integration" / "transport_profiles",
                )
            )
            .expanduser()
            .resolve()
        )

    def doctor(self, profile_id: str) -> dict[str, object]:
        from .transport import load_transport_profile

        profile = load_remote_profile(self.remote_profile_root, profile_id)
        blockers: list[str] = []
        asset_record: dict[str, object]
        try:
            asset_lock = load_asset_lock(self.asset_lock_path)
            resolve_asset_identities(asset_lock, profile.required_assets)
            asset_record = {
                "status": "READY",
                "lock_id": asset_lock.lock_id,
                "workload_id": asset_lock.workload_id,
                "sha256": asset_lock.sha256,
                "required_roles": list(profile.required_assets),
                "unresolved": [],
            }
        except LabError as exc:
            asset_record = {
                "status": "BLOCKED",
                "lock_id": None,
                "workload_id": None,
                "sha256": None,
                "required_roles": list(profile.required_assets),
                "unresolved": list(profile.required_assets),
                "detail": str(exc),
            }
            blockers.append(f"assets: {exc}")

        if self.asset_observation_path.is_file():
            try:
                observations = load_asset_observations(self.asset_observation_path)
                candidate_record: dict[str, object] = {
                    "status": "OBSERVED_NOT_AUTHORIZED",
                    "observation_id": observations.observation_id,
                    "host_id": observations.host_id,
                    "repository_commit": observations.commit,
                    "registry_sha256": observations.registry_sha256,
                    "candidates": [
                        {
                            "candidate_id": candidate.candidate_id,
                            "blockers": list(candidate.blockers),
                        }
                        for candidate in observations.candidates.values()
                    ],
                    "sha256": observations.sha256,
                    "activation_authorized": False,
                }
            except LabError as exc:
                candidate_record = {
                    "status": "INVALID",
                    "detail": str(exc),
                    "activation_authorized": False,
                }
        else:
            candidate_record = {
                "status": "NOT_RECORDED",
                "activation_authorized": False,
            }

        matching_transports: list[dict[str, object]] = []
        if self.transport_profile_root.is_dir():
            for path in sorted(self.transport_profile_root.glob("*.json")):
                try:
                    transport = load_transport_profile(
                        self.transport_profile_root,
                        path.stem,
                    )
                except LabError as exc:
                    blockers.append(f"transport {path.name}: {exc}")
                    continue
                if transport.target_host != profile.target_host:
                    continue
                matching_transports.append(
                    {
                        "transport_id": transport.transport_id,
                        "status": transport.status,
                        "ssh_alias": transport.ssh_alias,
                        "host_key_pin_count": len(transport.host_key_fingerprints),
                        "sha256": transport.sha256,
                    }
                )
        active: list[dict[str, object]] = []
        for transport_record in matching_transports:
            pin_count = transport_record.get("host_key_pin_count")
            if (
                transport_record.get("status") == "ACTIVE"
                and isinstance(pin_count, int)
                and pin_count > 0
            ):
                active.append(transport_record)
        if len(active) != 1:
            blockers.append(
                "transport: expected exactly one ACTIVE pinned Profile for "
                f"{profile.target_host}, found {len(active)}"
            )
        return {
            "schema": "lab.remote-doctor.v1",
            "profile": {
                "profile_id": profile.profile_id,
                "sha256": profile.sha256,
                "target_host": profile.target_host,
            },
            "assets": asset_record,
            "asset_observations": candidate_record,
            "transports": matching_transports,
            "planning_ready": asset_record["status"] == "READY",
            "dispatch_ready": asset_record["status"] == "READY" and len(active) == 1,
            "status": "READY" if not blockers else "BLOCKED",
            "blockers": blockers,
            "network_checked": False,
            "execution_eligible": False,
        }

    def plan(self, task_id: str, profile_id: str) -> dict[str, object]:
        profile = load_remote_profile(self.remote_profile_root, profile_id)
        asset_lock = load_asset_lock(self.asset_lock_path)
        snapshot_id, snapshot = resolve_clean_snapshot_identity(
            self.workspace_manager,
            task_id=task_id,
            profile_id=profile.profile_id,
            profile_sha256=profile.sha256,
            repositories=profile.repositories,
            baseline_path=self.baseline_path,
        )
        raw_repositories = snapshot.get("repositories")
        if not isinstance(raw_repositories, dict):
            raise LabError("resolved remote snapshot has no repositories")
        request_repositories: dict[str, dict[str, object]] = {}
        published: dict[str, dict[str, object]] = {}
        for name in profile.repositories:
            identity = raw_repositories.get(name)
            if not isinstance(identity, dict):
                raise LabError(f"resolved remote snapshot lacks repository: {name}")
            management_path = Path(str(identity.get("management_path", ""))).resolve()
            commit = str(identity.get("commit", ""))
            try:
                refs = git.remote_tracking_refs_containing(management_path, commit)
            except git.GitError as exc:
                raise LabError(str(exc)) from exc
            if not refs:
                raise LabError(
                    f"remote Profile requires published commit for {name}: {commit}; "
                    "checkpoint and push it before planning dispatch"
                )
            request_repositories[name] = {
                "commit": commit,
                "tree_oid": identity.get("tree_oid"),
                "dirty": False,
                "submodules": identity.get("submodules", {}),
            }
            published[name] = {
                "source": identity.get("source"),
                "published_refs": list(refs),
            }
        raw_repository_links = snapshot.get("repository_links", [])
        if not isinstance(raw_repository_links, list):
            raise LabError("resolved remote snapshot has invalid repository links")
        request_repository_links = [
            link
            for link in raw_repository_links
            if isinstance(link, dict)
            and link.get("parent_repository") in request_repositories
            and link.get("child_repository") in request_repositories
        ]
        assets = resolve_asset_identities(asset_lock, profile.required_assets)
        request = build_remote_request(
            task_id=task_id,
            snapshot_id=snapshot_id,
            repositories=request_repositories,
            repository_links=request_repository_links,
            assets=assets,
            asset_lock={
                "lock_id": asset_lock.lock_id,
                "workload_id": asset_lock.workload_id,
                "sha256": asset_lock.sha256,
            },
            profile=profile,
            control_plane_version=CONTROL_PLANE_VERSION,
        )
        content: dict[str, object] = {
            "schema": REMOTE_OUTBOX_SCHEMA_ID,
            "outbox_id": f"O-{str(request['request_id']).removeprefix('Q-')}",
            "request_id": request["request_id"],
            "task_id": task_id,
            "snapshot_id": snapshot_id,
            "profile": request["profile"],
            "effective_workspace": {
                "baseline": snapshot.get("baseline"),
                "repositories": published,
                "repository_links": request.get("repository_links", []),
            },
            "asset_lock": {
                "lock_id": asset_lock.lock_id,
                "workload_id": asset_lock.workload_id,
                "sha256": asset_lock.sha256,
            },
            "request": request,
            "dispatch": {
                "status": "PENDING_DISPATCH",
                "attempts": [],
                "remote_ack": None,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_eligible": False,
        }
        envelope = {**content, "envelope_sha256": _sha256(content)}
        validate_remote_outbox(envelope)
        request_id = str(request["request_id"])
        target = self.outbox_root / request_id
        path = target / "envelope.json"
        with self.workspace_manager.advisory_locks("remote-outbox", [request_id]):
            if self.outbox_root.is_symlink():
                raise LabError(f"remote outbox root is a symlink: {self.outbox_root}")
            if path.is_file():
                existing = self.load(request_id)
                if existing.get("request") != request:
                    raise LabError(f"remote outbox request collision: {request_id}")
                return existing
            if target.exists() or target.is_symlink():
                raise LabError(f"remote outbox target is unsafe: {target}")
            target.mkdir(parents=True)
            _atomic_write_text(
                path, json.dumps(envelope, indent=2, sort_keys=True) + "\n"
            )
        return envelope

    def load(self, request_id: str) -> dict[str, object]:
        validate_task_id(request_id)
        if self.outbox_root.is_symlink():
            raise LabError(f"remote outbox root is a symlink: {self.outbox_root}")
        path = self.outbox_root / request_id / "envelope.json"
        if path.is_symlink() or not path.is_file():
            raise LabError(f"remote outbox request does not exist: {request_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LabError(f"cannot read remote outbox envelope {path}: {exc}") from exc
        return validate_remote_outbox(payload)

    def attach_acknowledgement(
        self,
        request_id: str,
        *,
        acceptance: object,
        transport_receipt: object,
    ) -> dict[str, object]:
        from .remote_inbox import validate_remote_acceptance
        from .transport import load_transport_profile, validate_transport_receipt

        envelope = self.load(request_id)
        request = envelope["request"]
        if not isinstance(request, dict):
            raise LabError("remote outbox lacks request")
        request_profile = request.get("profile")
        if not isinstance(request_profile, dict):
            raise LabError("remote outbox request lacks Profile")
        target_host = str(request_profile.get("target_host", ""))
        validated_acceptance = validate_remote_acceptance(
            acceptance,
            request_id=request_id,
            idempotency_key=str(request["idempotency_key"]),
            target_host=target_host,
        )
        if not isinstance(transport_receipt, dict):
            raise LabError("authenticated transport receipt must be an object")
        transport_profile = load_transport_profile(
            self.transport_profile_root,
            str(transport_receipt.get("transport_id", "")),
        )
        if transport_profile.target_host != target_host:
            raise LabError("transport Profile targets a different host")
        validated_transport = validate_transport_receipt(
            transport_receipt,
            envelope=envelope,
            acceptance=validated_acceptance,
            profile=transport_profile,
        )
        target = self.outbox_root / request_id
        acceptance_path = target / "acceptance.json"
        transport_path = target / "transport.json"
        with self.workspace_manager.advisory_locks("remote-outbox", [request_id]):
            for path, payload, label in (
                (acceptance_path, validated_acceptance, "acceptance"),
                (transport_path, validated_transport, "transport"),
            ):
                if path.is_symlink():
                    raise LabError(f"remote outbox {label} path is a symlink: {path}")
                if path.is_file():
                    try:
                        existing = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise LabError(
                            f"cannot read remote outbox {label}: {path}"
                        ) from exc
                    if existing != payload:
                        raise LabError(
                            f"remote outbox already has different {label}: {request_id}"
                        )
                else:
                    _atomic_write_text(
                        path,
                        json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    )
        return self.status(request_id)

    def attach_remote_state(
        self,
        request_id: str,
        *,
        remote_state: object,
        transport_id: str,
        observed_host_key_fingerprint: str,
    ) -> dict[str, object]:
        """Persist one authenticated state observation without allowing forks."""

        from .remote_observation import build_remote_state_observation
        from .runner_state import validate_request_state_identity
        from .transport import load_transport_profile

        envelope = self.load(request_id)
        request = envelope.get("request")
        if not isinstance(request, dict):
            raise LabError("remote outbox lacks request")
        state = validate_request_state_identity(remote_state, request=request)
        history = state.get("history")
        if (
            not isinstance(history, list)
            or not history
            or not isinstance(history[-1], dict)
        ):
            raise LabError("remote request state has invalid history")
        sequence = history[-1].get("sequence")
        if not isinstance(sequence, int):
            raise LabError("remote request state has invalid sequence")
        profile = load_transport_profile(self.transport_profile_root, transport_id)
        target = self.outbox_root / request_id / "remote-observations"
        path = target / f"{sequence:08d}.json"
        with self.workspace_manager.advisory_locks("remote-outbox", [request_id]):
            if target.is_symlink():
                raise LabError(f"remote observation root is a symlink: {target}")
            existing_paths = sorted(target.glob("*.json")) if target.is_dir() else []
            if existing_paths:
                latest_sequence = int(existing_paths[-1].stem)
                if sequence < latest_sequence:
                    raise LabError(
                        "remote request state regression: "
                        f"observed={sequence} latest={latest_sequence}"
                    )
            if path.is_symlink():
                raise LabError(f"remote observation path is a symlink: {path}")
            if path.is_file():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise LabError(f"cannot read remote observation: {path}") from exc
                existing_state = (
                    existing.get("remote_state") if isinstance(existing, dict) else None
                )
                if existing_state != state:
                    raise LabError(
                        f"remote request state fork at sequence {sequence}: {request_id}"
                    )
                return self.status(request_id)
            receipt = build_remote_state_observation(
                envelope=envelope,
                remote_state=state,
                profile=profile,
                observed_host_key_fingerprint=observed_host_key_fingerprint,
            )
            target.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(
                path,
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            )
        return self.status(request_id)

    def status(self, request_id: str) -> dict[str, object]:
        from .remote_inbox import validate_remote_acceptance
        from .remote_observation import validate_remote_state_observation
        from .transport import load_transport_profile, validate_transport_receipt

        envelope = self.load(request_id)
        request = envelope["request"]
        if not isinstance(request, dict):
            raise LabError("remote outbox lacks request")
        request_profile = request.get("profile")
        if not isinstance(request_profile, dict):
            raise LabError("remote outbox request lacks Profile")
        target_host = str(request_profile.get("target_host", ""))
        target = self.outbox_root / request_id
        acceptance_path = target / "acceptance.json"
        transport_path = target / "transport.json"
        acceptance: dict[str, object] | None = None
        transport: dict[str, object] | None = None
        remote_observation: dict[str, object] | None = None
        if acceptance_path.is_file():
            try:
                raw_acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LabError(
                    f"cannot read remote outbox acceptance: {acceptance_path}"
                ) from exc
            acceptance = validate_remote_acceptance(
                raw_acceptance,
                request_id=request_id,
                idempotency_key=str(request["idempotency_key"]),
                target_host=target_host,
            )
        if transport_path.is_file():
            if acceptance is None:
                raise LabError(
                    f"remote outbox transport exists without acceptance: {request_id}"
                )
            try:
                raw_transport = json.loads(transport_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LabError(
                    f"cannot read remote outbox transport: {transport_path}"
                ) from exc
            if not isinstance(raw_transport, dict):
                raise LabError("remote outbox transport receipt must be an object")
            transport_profile = load_transport_profile(
                self.transport_profile_root,
                str(raw_transport.get("transport_id", "")),
            )
            transport = validate_transport_receipt(
                raw_transport,
                envelope=envelope,
                acceptance=acceptance,
                profile=transport_profile,
            )
        observations_root = target / "remote-observations"
        if observations_root.is_symlink():
            raise LabError(f"remote observation root is a symlink: {observations_root}")
        if observations_root.is_dir():
            previous_sequence = -1
            for observation_path in sorted(observations_root.glob("*.json")):
                if observation_path.is_symlink():
                    raise LabError(
                        f"remote observation path is a symlink: {observation_path}"
                    )
                try:
                    raw_observation = json.loads(
                        observation_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise LabError(
                        f"cannot read remote observation: {observation_path}"
                    ) from exc
                if not isinstance(raw_observation, dict):
                    raise LabError("remote observation must be an object")
                observation_profile = load_transport_profile(
                    self.transport_profile_root,
                    str(raw_observation.get("transport_id", "")),
                )
                validated_observation = validate_remote_state_observation(
                    raw_observation,
                    envelope=envelope,
                    profile=observation_profile,
                )
                sequence = validated_observation.get("remote_state_sequence")
                if (
                    not isinstance(sequence, int)
                    or sequence <= previous_sequence
                    or observation_path.stem != f"{sequence:08d}"
                ):
                    raise LabError("remote observations are not strictly append-only")
                previous_sequence = sequence
                remote_observation = validated_observation
        if transport is not None:
            status = "ACKNOWLEDGED"
        elif acceptance is not None:
            status = "ACK_RECEIVED_UNVERIFIED"
        else:
            status = "PENDING_DISPATCH"
        return {
            "schema": "lab.remote-dispatch-status.v1",
            "request_id": request_id,
            "task_id": envelope["task_id"],
            "snapshot_id": envelope["snapshot_id"],
            "status": status,
            "remote_acknowledged": status == "ACKNOWLEDGED",
            "execution_eligible": False,
            "envelope": envelope,
            "acceptance": acceptance,
            "transport": transport,
            "remote_observation": remote_observation,
        }

    def list_status(self, task_id: str | None = None) -> list[dict[str, object]]:
        if task_id is not None:
            validate_task_id(task_id)
        if not self.outbox_root.is_dir():
            return []
        statuses = [
            self.status(path.parent.name)
            for path in sorted(self.outbox_root.glob("*/envelope.json"))
        ]
        return [
            status
            for status in statuses
            if task_id is None or status.get("task_id") == task_id
        ]
