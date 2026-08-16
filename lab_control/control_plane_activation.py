from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import subprocess
from pathlib import Path, PurePosixPath

from .control_plane_package import validate_control_plane_package_receipt
from .control_plane_stage import validate_control_plane_stage_receipt
from .preparation_profile import PreparationProfile
from .remote_preflight import HostProfile
from .remote_request import OID_PATTERN, SHA256_PATTERN
from .workspace import LabError, _atomic_write_text

_METADATA_FILES = {
    ".lab-package-receipt.json",
    ".lab-verification-receipt.json",
    ".lab-stage-receipt.json",
    ".lab-activation-receipt.json",
}


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabError(f"{label} must contain an object: {path}")
    return payload


def validate_control_plane_activation_receipt(
    receipt: object,
    *,
    profile: PreparationProfile,
    host_profile: HostProfile,
) -> dict[str, object]:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "preparation_profile_id",
        "preparation_profile_sha256",
        "host_profile_sha256",
        "host_id",
        "commit",
        "tree_oid",
        "version_path",
        "current_link",
        "previous_commit",
        "stage_receipt_sha256",
        "tree_verification_receipt_sha256",
        "python",
        "health_command",
        "health_exit_code",
        "health_stdout_sha256",
        "health_stderr_sha256",
        "control_plane_health_check_executed",
        "task_payload_executed",
        "network_downloads_permitted",
        "device_access",
        "active",
        "current_link_updated",
        "rollback_available",
        "receipt_sha256",
    }:
        raise LabError("control-plane activation receipt has an unexpected shape")
    commit = receipt.get("commit")
    previous = receipt.get("previous_commit")
    if (
        receipt.get("schema") != "lab.control-plane-activation.v1"
        or receipt.get("preparation_profile_id") != profile.profile_id
        or receipt.get("preparation_profile_sha256") != profile.sha256
        or receipt.get("host_profile_sha256") != host_profile.sha256
        or receipt.get("host_id") != profile.host_id
        or not OID_PATTERN.fullmatch(str(commit))
        or receipt.get("version_path")
        != str(profile.control_plane.install_root / "versions" / str(commit))
        or receipt.get("current_link") != str(profile.control_plane.current_link)
        or (previous is not None and not OID_PATTERN.fullmatch(str(previous)))
        or receipt.get("health_command") != "projectctl-help"
        or receipt.get("health_exit_code") != 0
    ):
        raise LabError("control-plane activation receipt identity mismatch")
    for field in (
        "tree_oid",
        "stage_receipt_sha256",
        "tree_verification_receipt_sha256",
        "health_stdout_sha256",
        "health_stderr_sha256",
        "receipt_sha256",
    ):
        pattern = OID_PATTERN if field == "tree_oid" else SHA256_PATTERN
        if not pattern.fullmatch(str(receipt.get(field, ""))):
            raise LabError(f"control-plane activation receipt has invalid {field}")
    expected_rollback = previous is not None and previous != commit
    if (
        not isinstance(receipt.get("python"), str)
        or not Path(str(receipt["python"])).is_absolute()
        or receipt.get("control_plane_health_check_executed") is not True
        or receipt.get("task_payload_executed") is not False
        or receipt.get("network_downloads_permitted") is not False
        or receipt.get("device_access") is not False
        or receipt.get("active") is not True
        or not isinstance(receipt.get("current_link_updated"), bool)
        or receipt.get("rollback_available") is not expected_rollback
    ):
        raise LabError("control-plane activation receipt has an unsafe authority envelope")
    digest = receipt.get("receipt_sha256")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if digest != hashlib.sha256(_canonical_json(content)).hexdigest():
        raise LabError("control-plane activation receipt hash mismatch")
    return receipt


def verify_staged_control_plane_version(
    profile: PreparationProfile,
    stage_receipt: object,
) -> dict[str, object]:
    staged = validate_control_plane_stage_receipt(stage_receipt, profile=profile)
    version = Path(str(staged["version_path"]))
    if version.is_symlink() or not version.is_dir() or version.resolve() != version:
        raise LabError(f"staged control-plane version is unsafe: {version}")
    package_receipt = _read_object(
        version / ".lab-package-receipt.json",
        label="staged package receipt",
    )
    validated_package = validate_control_plane_package_receipt(
        package_receipt,
        expected_identity=str(staged["canonical_identity"]),
        expected_commit=str(staged["commit"]),
    )
    if (
        validated_package.get("receipt_sha256")
        != staged.get("package_receipt_sha256")
        or not SHA256_PATTERN.fullmatch(
            str(staged.get("verification_receipt_sha256", ""))
        )
    ):
        raise LabError("staged control-plane package binding mismatch")
    raw_entries = validated_package.get("entries")
    if not isinstance(raw_entries, list):
        raise LabError("staged control-plane package lacks entries")
    expected = {
        str(entry["path"]): entry for entry in raw_entries if isinstance(entry, dict)
    }
    expected_directories: set[str] = set()
    for relative_path in expected:
        parent = PurePosixPath(relative_path).parent
        while str(parent) != ".":
            expected_directories.add(str(parent))
            parent = parent.parent
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in version.rglob("*"):
        if path.is_symlink():
            raise LabError(f"staged control-plane contains a symlink: {path}")
        relative = path.relative_to(version).as_posix()
        if path.is_dir():
            observed_directories.add(relative)
            continue
        if not path.is_file():
            raise LabError(f"staged control-plane contains an unsafe node: {path}")
        if relative in _METADATA_FILES:
            continue
        entry = expected.get(relative)
        if entry is None or relative in observed_files:
            raise LabError(f"staged control-plane contains an extra file: {relative}")
        expected_mode = 0o755 if entry["mode"] == "100755" else 0o644
        file_content = path.read_bytes()
        if (
            len(file_content) != entry["size_bytes"]
            or stat.S_IMODE(path.stat().st_mode) != expected_mode
            or hashlib.sha256(file_content).hexdigest() != entry["sha256"]
        ):
            raise LabError(f"staged control-plane file mismatch: {relative}")
        observed_files.add(relative)
    if observed_files != set(expected) or observed_directories != expected_directories:
        raise LabError("staged control-plane tree is incomplete or has extra directories")
    content: dict[str, object] = {
        "schema": "lab.control-plane-staged-verification.v1",
        "preparation_profile_sha256": profile.sha256,
        "stage_receipt_sha256": staged["receipt_sha256"],
        "package_receipt_sha256": staged["package_receipt_sha256"],
        "commit": staged["commit"],
        "tree_oid": staged["tree_oid"],
        "version_path": str(version),
        "entry_count": len(observed_files),
        "verified": True,
        "executed": False,
        "device_access": False,
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }


def activate_control_plane_version(
    profile: PreparationProfile,
    host_profile: HostProfile,
    *,
    stage_receipt: object,
    timeout_seconds: int = 10,
) -> dict[str, object]:
    """Health-check a staged trusted version and atomically update current."""

    if host_profile.host_id != profile.host_id:
        raise LabError("control-plane activation host/Profile mismatch")
    if host_profile.control_plane_projectctl != profile.control_plane.current_link / "projectctl":
        raise LabError("control-plane activation projectctl path mismatch")
    if not 1 <= timeout_seconds <= 30:
        raise LabError("control-plane activation timeout must be between 1 and 30 seconds")
    staged = validate_control_plane_stage_receipt(stage_receipt, profile=profile)
    tree_verification = verify_staged_control_plane_version(profile, staged)
    version = Path(str(staged["version_path"]))
    projectctl = version / "projectctl"
    environment = host_profile.environments.get(profile.control_plane.environment_id)
    if (
        environment is None
        or environment.repository != profile.control_plane.repository
    ):
        raise LabError("control-plane activation environment is not protected")
    python = environment.path / "bin" / "python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise LabError(f"control-plane activation Python is unavailable: {python}")
    try:
        completed = subprocess.run(
            [str(python), "-I", str(projectctl), "--help"],
            cwd=version,
            env={
                "HOME": str(profile.control_plane.install_root / ".isolated-home"),
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LabError(f"control-plane activation health check failed: {exc}") from exc
    if completed.returncode != 0:
        detail = " ".join((completed.stderr or completed.stdout).split())[:1000]
        raise LabError(
            f"control-plane activation health check exited {completed.returncode}: "
            f"{detail or 'no diagnostic'}"
        )
    current = profile.control_plane.current_link
    previous_commit: str | None = None
    if os.path.lexists(current):
        if not current.is_symlink():
            raise LabError(f"control-plane current path is not a symlink: {current}")
        raw_previous = os.readlink(current)
        previous = PurePosixPath(raw_previous)
        if (
            previous.is_absolute()
            or len(previous.parts) != 2
            or previous.parts[0] != "versions"
            or len(previous.parts[1]) != 40
            or not (profile.control_plane.install_root / raw_previous).is_dir()
        ):
            raise LabError("control-plane current symlink has an unsafe target")
        previous_commit = previous.parts[1]
    target_relative = f"versions/{staged['commit']}"
    changed = os.readlink(current) != target_relative if current.is_symlink() else True
    content: dict[str, object] = {
        "schema": "lab.control-plane-activation.v1",
        "preparation_profile_id": profile.profile_id,
        "preparation_profile_sha256": profile.sha256,
        "host_profile_sha256": host_profile.sha256,
        "host_id": profile.host_id,
        "commit": staged["commit"],
        "tree_oid": staged["tree_oid"],
        "version_path": str(version),
        "current_link": str(current),
        "previous_commit": previous_commit,
        "stage_receipt_sha256": staged["receipt_sha256"],
        "tree_verification_receipt_sha256": tree_verification["receipt_sha256"],
        "python": str(python.resolve()),
        "health_command": "projectctl-help",
        "health_exit_code": completed.returncode,
        "health_stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "health_stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "control_plane_health_check_executed": True,
        "task_payload_executed": False,
        "network_downloads_permitted": False,
        "device_access": False,
        "active": True,
        "current_link_updated": changed,
        "rollback_available": previous_commit is not None and previous_commit != staged["commit"],
    }
    activation: dict[str, object] = {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }
    _atomic_write_text(
        version / ".lab-activation-receipt.json",
        json.dumps(activation, indent=2, sort_keys=True) + "\n",
    )
    if changed:
        temporary = current.parent / f".current-{os.getpid()}-{secrets.token_hex(8)}"
        try:
            os.symlink(target_relative, temporary)
            os.replace(temporary, current)
        finally:
            temporary.unlink(missing_ok=True)
    if not current.is_symlink() or current.resolve() != version:
        raise LabError("control-plane activation failed current-link verification")
    return validate_control_plane_activation_receipt(
        activation,
        profile=profile,
        host_profile=host_profile,
    )


def rollback_control_plane_version(
    profile: PreparationProfile,
    host_profile: HostProfile,
    *,
    activation_receipt: object,
    timeout_seconds: int = 10,
) -> dict[str, object]:
    """Restore the previous staged version through the same health-checked path."""

    active = validate_control_plane_activation_receipt(
        activation_receipt,
        profile=profile,
        host_profile=host_profile,
    )
    current = profile.control_plane.current_link
    if (
        not current.is_symlink()
        or current.resolve() != Path(str(active["version_path"]))
    ):
        raise LabError("control-plane rollback receipt is not the current version")
    previous = active.get("previous_commit")
    if not isinstance(previous, str) or previous == active.get("commit"):
        raise LabError("control-plane rollback has no previous version")
    previous_version = profile.control_plane.install_root / "versions" / previous
    previous_stage = _read_object(
        previous_version / ".lab-stage-receipt.json",
        label="previous control-plane stage receipt",
    )
    restored = activate_control_plane_version(
        profile,
        host_profile,
        stage_receipt=previous_stage,
        timeout_seconds=timeout_seconds,
    )
    content: dict[str, object] = {
        "schema": "lab.control-plane-rollback.v1",
        "preparation_profile_sha256": profile.sha256,
        "host_profile_sha256": host_profile.sha256,
        "from_commit": active["commit"],
        "restored_commit": previous,
        "source_activation_receipt_sha256": active["receipt_sha256"],
        "restored_activation_receipt_sha256": restored["receipt_sha256"],
        "current_link": str(current),
        "health_checked": True,
        "old_versions_deleted": False,
        "task_payload_executed": False,
        "network_downloads_permitted": False,
        "device_access": False,
        "rolled_back": True,
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }
