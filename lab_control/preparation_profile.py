from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .remote_preflight import HostProfile
from .workspace import LabError, validate_repository_name, validate_task_id

PREPARATION_PROFILE_SCHEMA_ID = "lab.preparation-profile.v1"


@dataclass(frozen=True)
class StableSyncPlacement:
    repository: str
    canonical_identity: str
    path: Path
    branch: str
    remote_ref: str


@dataclass(frozen=True)
class SourceAnchorPlacement:
    repository: str
    canonical_identity: str
    source_path: Path
    target_path: Path
    submodule_sources: dict[str, Path]
    max_checkout_blob_bytes: int


@dataclass(frozen=True)
class ControlPlanePlacement:
    repository: str
    install_root: Path
    current_link: Path
    environment_id: str
    max_versions: int


@dataclass(frozen=True)
class PreparationProfile:
    profile_id: str
    host_id: str
    stable_sync: dict[str, StableSyncPlacement]
    source_anchors: dict[str, SourceAnchorPlacement]
    control_plane: ControlPlanePlacement
    allowed_target_roots: tuple[Path, ...]
    path: Path
    sha256: str


def _absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise LabError(f"preparation Profile {label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise LabError(f"preparation Profile {label} is unsafe")
    return path


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise LabError(f"preparation Profile {label} must be a relative path")
    pure = PurePosixPath(value)
    if value.startswith(("/", "~")) or "\\" in value or ".." in pure.parts:
        raise LabError(f"preparation Profile {label} is unsafe")
    return value


def load_preparation_profile(
    profile_root: Path,
    profile_id: str,
    *,
    host_profile: HostProfile,
) -> PreparationProfile:
    validate_task_id(profile_id)
    root = profile_root.expanduser().resolve()
    path = (root / f"{profile_id}.json").resolve()
    if path.parent != root or not path.is_file():
        raise LabError(f"unknown preparation Profile: {profile_id} ({path})")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read preparation Profile {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "profile_id",
        "host_id",
        "network_policy",
        "allowed_target_roots",
        "stable_sync",
        "source_anchors",
        "control_plane",
    }:
        raise LabError(f"preparation Profile has an unexpected shape: {path}")
    if (
        payload.get("schema") != PREPARATION_PROFILE_SCHEMA_ID
        or payload.get("profile_id") != profile_id
        or payload.get("host_id") != host_profile.host_id
    ):
        raise LabError(f"preparation Profile identity mismatch: {path}")
    network = payload.get("network_policy")
    if not isinstance(network, dict) or network != {
        "trusted_prepare_egress": "explicit-only",
        "task_payload_egress": "loopback-only",
        "implicit_downloads": False,
        "large_file_downloads": False,
        "lfs_smudge": False,
        "submodule_network_fetch": False,
    }:
        raise LabError(f"preparation Profile has an unsafe network policy: {path}")
    if (
        host_profile.network_policy.trusted_prepare_egress != "explicit-only"
        or host_profile.network_policy.task_payload_egress != "loopback-only"
        or host_profile.network_policy.implicit_downloads
        or host_profile.network_policy.large_file_downloads
    ):
        raise LabError("preparation and host network policies do not match")

    raw_roots = payload.get("allowed_target_roots")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise LabError("preparation Profile requires allowed target roots")
    allowed_roots = tuple(
        _absolute_path(value, label="allowed target root") for value in raw_roots
    )
    if len(allowed_roots) != len(set(allowed_roots)):
        raise LabError("preparation Profile repeats an allowed target root")

    raw_control_plane = payload.get("control_plane")
    if not isinstance(raw_control_plane, dict) or set(raw_control_plane) != {
        "repository",
        "install_root",
        "current_link",
        "environment_id",
        "max_versions",
    }:
        raise LabError("preparation Profile requires a control-plane placement")
    install_root = _absolute_path(
        raw_control_plane.get("install_root"),
        label="control-plane install root",
    )
    current_link = _absolute_path(
        raw_control_plane.get("current_link"),
        label="control-plane current link",
    )
    max_versions = raw_control_plane.get("max_versions")
    control_repository = raw_control_plane.get("repository")
    if not isinstance(control_repository, str):
        raise LabError("preparation Profile requires a control repository")
    validate_repository_name(control_repository)
    environment_id = raw_control_plane.get("environment_id")
    environment = (
        host_profile.environments.get(environment_id)
        if isinstance(environment_id, str)
        else None
    )
    if (
        install_root not in allowed_roots
        or current_link != install_root / "current"
        or current_link / "projectctl" != host_profile.control_plane_projectctl
        or environment is None
        or environment.repository != control_repository
        or control_repository not in host_profile.repositories
        or not isinstance(max_versions, int)
        or isinstance(max_versions, bool)
        or not 1 <= max_versions <= 5
    ):
        raise LabError("preparation Profile has an unsafe control-plane placement")
    control_plane = ControlPlanePlacement(
        repository=control_repository,
        install_root=install_root,
        current_link=current_link,
        environment_id=str(environment_id),
        max_versions=max_versions,
    )

    raw_sync = payload.get("stable_sync")
    if not isinstance(raw_sync, dict) or not raw_sync:
        raise LabError("preparation Profile requires stable sync placements")
    stable_sync: dict[str, StableSyncPlacement] = {}
    for repository, raw_placement in raw_sync.items():
        validate_repository_name(repository)
        if not isinstance(raw_placement, dict) or set(raw_placement) != {
            "canonical_identity",
            "path",
            "branch",
            "remote_ref",
        }:
            raise LabError(f"invalid stable sync placement: {repository}")
        host_repository = host_profile.repositories.get(repository)
        canonical_identity = raw_placement.get("canonical_identity")
        placement_path = _absolute_path(
            raw_placement.get("path"), label=f"{repository} stable path"
        )
        branch = raw_placement.get("branch")
        remote_ref = raw_placement.get("remote_ref")
        if (
            host_repository is None
            or canonical_identity != host_repository.canonical_identity
            or placement_path != host_repository.path
            or not isinstance(branch, str)
            or not branch
            or remote_ref != f"refs/remotes/origin/{branch}"
        ):
            raise LabError(f"stable sync differs from host Profile: {repository}")
        stable_sync[repository] = StableSyncPlacement(
            repository=repository,
            canonical_identity=str(canonical_identity),
            path=placement_path,
            branch=branch,
            remote_ref=str(remote_ref),
        )

    raw_anchors = payload.get("source_anchors")
    if not isinstance(raw_anchors, dict) or not raw_anchors:
        raise LabError("preparation Profile requires source anchor placements")
    source_anchors: dict[str, SourceAnchorPlacement] = {}
    for repository, raw_placement in raw_anchors.items():
        validate_repository_name(repository)
        if not isinstance(raw_placement, dict) or set(raw_placement) != {
            "canonical_identity",
            "source_path",
            "target_path",
            "submodule_sources",
            "max_checkout_blob_bytes",
        }:
            raise LabError(f"invalid source anchor placement: {repository}")
        host_repository = host_profile.repositories.get(repository)
        canonical_identity = raw_placement.get("canonical_identity")
        source_path = _absolute_path(
            raw_placement.get("source_path"), label=f"{repository} source path"
        )
        target_path = _absolute_path(
            raw_placement.get("target_path"), label=f"{repository} target path"
        )
        max_blob = raw_placement.get("max_checkout_blob_bytes")
        if (
            host_repository is None
            or canonical_identity != host_repository.canonical_identity
            or target_path != host_repository.path
            or source_path == target_path
            or not isinstance(max_blob, int)
            or isinstance(max_blob, bool)
            or not 1 <= max_blob <= 16 * 1024 * 1024
            or not any(target_path.is_relative_to(root) for root in allowed_roots)
        ):
            raise LabError(f"source anchor differs from host Profile: {repository}")
        raw_submodules = raw_placement.get("submodule_sources")
        if not isinstance(raw_submodules, dict):
            raise LabError(f"source anchor has invalid submodules: {repository}")
        submodules: dict[str, Path] = {}
        for relative_path, source in raw_submodules.items():
            safe_path = _relative_path(
                relative_path,
                label=f"{repository} submodule path",
            )
            submodules[safe_path] = _absolute_path(
                source,
                label=f"{repository}/{safe_path} source",
            )
        source_anchors[repository] = SourceAnchorPlacement(
            repository=repository,
            canonical_identity=str(canonical_identity),
            source_path=source_path,
            target_path=target_path,
            submodule_sources=dict(sorted(submodules.items())),
            max_checkout_blob_bytes=max_blob,
        )
    return PreparationProfile(
        profile_id=profile_id,
        host_id=host_profile.host_id,
        stable_sync=dict(sorted(stable_sync.items())),
        source_anchors=dict(sorted(source_anchors.items())),
        control_plane=control_plane,
        allowed_target_roots=allowed_roots,
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
