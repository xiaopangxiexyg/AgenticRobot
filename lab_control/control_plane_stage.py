from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .control_plane_package import (
    MAX_ARCHIVE_BYTES,
    MAX_FILE_BYTES,
    validate_control_plane_package_receipt,
    verify_control_plane_package,
)
from .preparation_profile import PreparationProfile
from .remote_request import OID_PATTERN, SHA256_PATTERN
from .workspace import LabError, _atomic_write_text

CONTROL_PLANE_STAGE_SCHEMA_ID = "lab.control-plane-stage.v1"


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def validate_control_plane_stage_receipt(
    receipt: object,
    *,
    profile: PreparationProfile,
) -> dict[str, object]:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "preparation_profile_id",
        "preparation_profile_sha256",
        "host_id",
        "canonical_identity",
        "commit",
        "tree_oid",
        "install_root",
        "version_path",
        "package_receipt_sha256",
        "package_archive_sha256",
        "verification_receipt_sha256",
        "staged_at",
        "active",
        "current_link_updated",
        "executed",
        "network_used",
        "device_access",
        "receipt_sha256",
    }:
        raise LabError("control-plane stage receipt has an unexpected shape")
    stable = profile.stable_sync.get(profile.control_plane.repository)
    commit = receipt.get("commit")
    expected_version = profile.control_plane.install_root / "versions" / str(commit)
    if (
        stable is None
        or receipt.get("schema") != CONTROL_PLANE_STAGE_SCHEMA_ID
        or receipt.get("preparation_profile_id") != profile.profile_id
        or receipt.get("preparation_profile_sha256") != profile.sha256
        or receipt.get("host_id") != profile.host_id
        or receipt.get("canonical_identity") != stable.canonical_identity
        or not OID_PATTERN.fullmatch(str(commit))
        or not OID_PATTERN.fullmatch(str(receipt.get("tree_oid", "")))
        or receipt.get("install_root") != str(profile.control_plane.install_root)
        or receipt.get("version_path") != str(expected_version)
    ):
        raise LabError("control-plane stage receipt identity mismatch")
    for field in (
        "package_receipt_sha256",
        "package_archive_sha256",
        "verification_receipt_sha256",
        "receipt_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(receipt.get(field, ""))):
            raise LabError(f"control-plane stage receipt has invalid {field}")
    if (
        receipt.get("active") is not False
        or receipt.get("current_link_updated") is not False
        or receipt.get("executed") is not False
        or receipt.get("network_used") is not False
        or receipt.get("device_access") is not False
    ):
        raise LabError("control-plane stage receipt has an unsafe authority envelope")
    staged_at = receipt.get("staged_at")
    if not isinstance(staged_at, str):
        raise LabError("control-plane stage receipt has invalid timestamp")
    try:
        timestamp = datetime.fromisoformat(staged_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError("control-plane stage receipt has invalid timestamp") from exc
    if timestamp.tzinfo is None:
        raise LabError("control-plane stage receipt timestamp requires a timezone")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt["receipt_sha256"] != hashlib.sha256(_canonical_json(content)).hexdigest():
        raise LabError("control-plane stage receipt hash mismatch")
    return receipt


def stage_control_plane_package(
    profile: PreparationProfile,
    *,
    archive_path: Path,
    package_receipt: object,
) -> dict[str, object]:
    """Verify and stage one version; never update current or execute it."""

    stable = profile.stable_sync.get(profile.control_plane.repository)
    if stable is None:
        raise LabError("control-plane staging requires its repository identity policy")
    if not isinstance(package_receipt, dict):
        raise LabError("control-plane staging requires a package receipt")
    commit = package_receipt.get("commit")
    if not OID_PATTERN.fullmatch(str(commit)):
        raise LabError("control-plane staging package has invalid commit")
    validated_package = validate_control_plane_package_receipt(
        package_receipt,
        expected_identity=stable.canonical_identity,
        expected_commit=str(commit),
    )
    archive_record = validated_package.get("archive")
    if not isinstance(archive_record, dict):
        raise LabError("control-plane staging package lacks archive identity")
    install_root = profile.control_plane.install_root
    if (
        not install_root.is_absolute()
        or install_root.is_symlink()
        or not install_root.is_dir()
        or install_root.resolve() != install_root
    ):
        raise LabError(f"control-plane install root is unsafe: {install_root}")
    versions = install_root / "versions"
    if versions.exists() or versions.is_symlink():
        if versions.is_symlink() or not versions.is_dir():
            raise LabError(f"control-plane versions root is unsafe: {versions}")
    else:
        versions.mkdir(mode=0o755)
    target = versions / str(commit)
    if target.exists() or target.is_symlink():
        raise LabError(f"control-plane version already exists: {target}")

    temporary = Path(
        tempfile.mkdtemp(
            dir=versions,
            prefix=f".{commit}.staging-",
        )
    )
    try:
        source = archive_path.expanduser()
        if (
            not source.is_absolute()
            or source.is_symlink()
            or source.absolute() != source.resolve()
            or source.name != archive_record.get("name")
        ):
            raise LabError("control-plane staging archive path is unsafe")
        snapshot = temporary / source.name
        source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            source_descriptor = os.open(source, source_flags)
        except OSError as exc:
            raise LabError(f"cannot open control-plane staging archive: {exc}") from exc
        copied = 0
        with os.fdopen(source_descriptor, "rb") as source_stream:
            source_metadata = os.fstat(source_stream.fileno())
            if (
                not stat.S_ISREG(source_metadata.st_mode)
                or not 1 <= source_metadata.st_size <= MAX_ARCHIVE_BYTES
            ):
                raise LabError("control-plane staging archive is not a bounded file")
            destination_descriptor = os.open(
                snapshot,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(destination_descriptor, "wb") as snapshot_stream:
                while True:
                    chunk = source_stream.read(1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > MAX_ARCHIVE_BYTES:
                        raise LabError("control-plane staging archive exceeds size limit")
                    snapshot_stream.write(chunk)
                snapshot_stream.flush()
                os.fsync(snapshot_stream.fileno())
        if copied != source_metadata.st_size:
            raise LabError("control-plane staging archive changed during snapshot")
        verification = verify_control_plane_package(
            snapshot,
            validated_package,
            expected_identity=stable.canonical_identity,
            expected_commit=str(commit),
        )
        raw_entries = validated_package.get("entries")
        if not isinstance(raw_entries, list):
            raise LabError("control-plane staging package lacks entries")
        expected = {
            str(entry["path"]): entry
            for entry in raw_entries
            if isinstance(entry, dict)
        }
        expected_directories: set[str] = set()
        for relative_path in expected:
            parent = PurePosixPath(relative_path).parent
            while str(parent) != ".":
                expected_directories.add(str(parent))
                parent = parent.parent
        observed_files: set[str] = set()
        with tarfile.open(snapshot, "r:") as package:
            for member in package.getmembers():
                pure = PurePosixPath(member.name)
                if (
                    member.name.startswith(("/", "~"))
                    or "\\" in member.name
                    or any(part in {"", ".", ".."} for part in pure.parts)
                ):
                    raise LabError(
                        f"control-plane staging has unsafe member: {member.name}"
                    )
                if member.isdir():
                    if member.name.rstrip("/") not in expected_directories:
                        raise LabError(
                            f"control-plane staging has extra directory: {member.name}"
                        )
                    continue
                entry = expected.get(member.name)
                if (
                    not member.isfile()
                    or entry is None
                    or member.name in observed_files
                    or member.size != entry["size_bytes"]
                    or member.mode
                    != (0o755 if entry["mode"] == "100755" else 0o644)
                ):
                    raise LabError(
                        f"control-plane staging member identity mismatch: {member.name}"
                    )
                stream = package.extractfile(member)
                if stream is None:
                    raise LabError(
                        f"control-plane staging cannot read member: {member.name}"
                    )
                member_content = stream.read(MAX_FILE_BYTES + 1)
                if (
                    len(member_content) != member.size
                    or hashlib.sha256(member_content).hexdigest()
                    != entry["sha256"]
                ):
                    raise LabError(
                        f"control-plane staging member changed: {member.name}"
                    )
                destination = temporary / member.name
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                if destination.exists() or destination.is_symlink():
                    raise LabError(
                        f"control-plane staging destination repeats: {member.name}"
                    )
                file_descriptor = os.open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    member.mode,
                )
                try:
                    with os.fdopen(file_descriptor, "wb") as stream_out:
                        stream_out.write(member_content)
                        stream_out.flush()
                        os.fsync(stream_out.fileno())
                except BaseException:
                    try:
                        os.close(file_descriptor)
                    except OSError:
                        pass
                    raise
                os.chmod(destination, member.mode)
                observed_files.add(member.name)
        if observed_files != set(expected):
            raise LabError("control-plane staging archive is incomplete")
        snapshot.unlink()
        package_digest = package_receipt.get("receipt_sha256")
        verification_digest = verification.get("receipt_sha256")
        package_archive = package_receipt.get("archive")
        if (
            not SHA256_PATTERN.fullmatch(str(package_digest))
            or not SHA256_PATTERN.fullmatch(str(verification_digest))
            or not isinstance(package_archive, dict)
            or not SHA256_PATTERN.fullmatch(str(package_archive.get("sha256", "")))
        ):
            raise LabError("control-plane staging lacks verified package identities")
        content: dict[str, object] = {
            "schema": CONTROL_PLANE_STAGE_SCHEMA_ID,
            "preparation_profile_id": profile.profile_id,
            "preparation_profile_sha256": profile.sha256,
            "host_id": profile.host_id,
            "canonical_identity": stable.canonical_identity,
            "commit": str(commit),
            "tree_oid": package_receipt["tree_oid"],
            "install_root": str(install_root),
            "version_path": str(target),
            "package_receipt_sha256": package_digest,
            "package_archive_sha256": package_archive["sha256"],
            "verification_receipt_sha256": verification_digest,
            "staged_at": datetime.now(timezone.utc).isoformat(),
            "active": False,
            "current_link_updated": False,
            "executed": False,
            "network_used": False,
            "device_access": False,
        }
        stage_receipt: dict[str, object] = {
            **content,
            "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
        }
        _atomic_write_text(
            temporary / ".lab-package-receipt.json",
            json.dumps(package_receipt, indent=2, sort_keys=True) + "\n",
        )
        _atomic_write_text(
            temporary / ".lab-verification-receipt.json",
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
        )
        _atomic_write_text(
            temporary / ".lab-stage-receipt.json",
            json.dumps(stage_receipt, indent=2, sort_keys=True) + "\n",
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_control_plane_stage_receipt(stage_receipt, profile=profile)
