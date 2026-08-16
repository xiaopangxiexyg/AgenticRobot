from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .remote_dispatch import validate_remote_outbox
from .remote_inbox import validate_remote_acceptance
from .remote_observation import validate_remote_watch_result
from .runner_state import validate_request_state_identity
from .transport import TransportProfile, build_transport_receipt
from .workspace import LabError, _atomic_write_text

SSHRunner = Callable[[list[str], str, int], subprocess.CompletedProcess[str]]
MAX_SSH_JSON_BYTES = 4 * 1024 * 1024
MAX_KNOWN_HOSTS_BYTES = 1024 * 1024


def _ssh_keygen_binary() -> Path:
    candidate = shutil.which("ssh-keygen")
    if candidate is None:
        raise LabError("trusted SSH transport requires ssh-keygen")
    return Path(candidate).resolve()


def _key_fingerprint(encoded_key: str) -> str:
    try:
        blob = base64.b64decode(encoded_key.encode(), validate=True)
    except (ValueError, TypeError) as exc:
        raise LabError("known_hosts contains an invalid public key") from exc
    digest = base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
    return f"SHA256:{digest}"


def verify_pinned_known_host(
    profile: TransportProfile,
    known_hosts_path: Path,
) -> str:
    """Verify the dedicated known_hosts entry without contacting the host."""

    if profile.status != "ACTIVE" or len(profile.host_key_fingerprints) != 1:
        raise LabError("SSH dispatch requires an ACTIVE single-pin transport Profile")
    expanded = known_hosts_path.expanduser()
    if expanded.is_symlink():
        raise LabError(f"trusted known_hosts file is a symlink: {expanded}")
    path = expanded.resolve()
    if not path.is_file() or not os.access(path, os.R_OK):
        raise LabError(f"trusted known_hosts file is missing or unreadable: {path}")
    if path.stat().st_size > MAX_KNOWN_HOSTS_BYTES:
        raise LabError("trusted known_hosts file exceeds the protected byte limit")
    lookup = (
        profile.ssh_host
        if profile.ssh_port == 22
        else f"[{profile.ssh_host}]:{profile.ssh_port}"
    )
    completed = subprocess.run(
        [str(_ssh_keygen_binary()), "-F", lookup, "-f", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        detail = completed.stderr.strip() or "ssh-keygen lookup failed"
        raise LabError(f"cannot inspect trusted known_hosts: {detail}")
    fingerprints: set[str] = set()
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        key_index = 3 if fields[0].startswith("@") else 2
        if len(fields) <= key_index:
            raise LabError("cannot parse trusted known_hosts entry")
        fingerprints.add(_key_fingerprint(fields[key_index]))
    expected = set(profile.host_key_fingerprints)
    if fingerprints != expected:
        raise LabError(
            "trusted known_hosts does not exactly match transport host-key pin: "
            f"expected={sorted(expected)} observed={sorted(fingerprints)}"
        )
    return profile.host_key_fingerprints[0]


def install_pinned_known_host(
    profile: TransportProfile,
    *,
    source_known_hosts: Path,
    target_known_hosts: Path,
) -> dict[str, object]:
    """Install one already-pinned host entry into Lab-owned trust state.

    This is an offline trust materialization step.  It never contacts the host,
    scans keys, or changes the protected Profile.
    """

    fingerprint = verify_pinned_known_host(profile, source_known_hosts)
    source = source_known_hosts.expanduser().resolve()
    lookup = (
        profile.ssh_host
        if profile.ssh_port == 22
        else f"[{profile.ssh_host}]:{profile.ssh_port}"
    )
    completed = subprocess.run(
        [str(_ssh_keygen_binary()), "-F", lookup, "-f", str(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise LabError("cannot extract pinned known_hosts entry")
    entries = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(entries) != 1:
        raise LabError("pinned host must resolve to exactly one known_hosts entry")
    fields = entries[0].split()
    key_index = 3 if fields[0].startswith("@") else 2
    if len(fields) <= key_index or _key_fingerprint(fields[key_index]) != fingerprint:
        raise LabError("extracted known_hosts entry does not match the protected pin")

    requested_target = target_known_hosts.expanduser()
    if not requested_target.is_absolute() or requested_target.is_symlink():
        raise LabError(f"Lab known_hosts target is unsafe: {requested_target}")
    parent = requested_target.parent
    if parent.is_symlink():
        raise LabError(f"Lab known_hosts parent is a symlink: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.resolve() != parent.absolute():
        raise LabError(f"Lab known_hosts parent escaped: {parent}")
    target = requested_target.absolute()
    content = entries[0] + "\n"
    changed = True
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise LabError(f"Lab known_hosts target is unsafe: {target}")
        try:
            existing = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise LabError(f"cannot read Lab known_hosts target: {target}") from exc
        if existing != content:
            raise LabError(
                "Lab known_hosts already exists with different trust material"
            )
        changed = stat.S_IMODE(target.stat().st_mode) != 0o600
    else:
        _atomic_write_text(target, content)
    target.chmod(0o600)
    installed = verify_pinned_known_host(profile, target)
    target_sha256 = hashlib.sha256(content.encode()).hexdigest()
    receipt_content: dict[str, object] = {
        "schema": "lab.known-host-install.v1",
        "transport_id": profile.transport_id,
        "transport_profile_sha256": profile.sha256,
        "target_host": profile.target_host,
        "ssh_host": profile.ssh_host,
        "host_key_fingerprint": installed,
        "target_known_hosts": str(target),
        "target_sha256": target_sha256,
        "changed": changed,
        "network_used": False,
        "accept_new_used": False,
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    return {
        **receipt_content,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                receipt_content,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }


def _trusted_ssh_prefix(
    *,
    profile: TransportProfile,
    known_hosts_path: Path,
    ssh_binary: Path = Path("/usr/bin/ssh"),
) -> tuple[list[str], str]:
    fingerprint = verify_pinned_known_host(profile, known_hosts_path)
    expanded_ssh = ssh_binary.expanduser()
    if expanded_ssh.is_symlink():
        raise LabError(f"trusted SSH executable is a symlink: {expanded_ssh}")
    ssh = expanded_ssh.resolve()
    if not ssh.is_file() or not os.access(ssh, os.X_OK):
        raise LabError(f"trusted SSH executable is missing: {ssh}")
    known_hosts = known_hosts_path.expanduser().resolve()
    return [
        str(ssh),
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-p",
        str(profile.ssh_port),
        f"{profile.ssh_user}@{profile.ssh_host}",
        "--",
    ], fingerprint


def build_ssh_dispatch_argv(
    *,
    envelope: object,
    profile: TransportProfile,
    known_hosts_path: Path,
    ssh_binary: Path = Path("/usr/bin/ssh"),
) -> tuple[list[str], str]:
    validated = validate_remote_outbox(envelope)
    request = validated["request"]
    if not isinstance(request, dict):
        raise LabError("remote outbox lacks request")
    request_profile = request.get("profile")
    control_plane = request.get("control_plane")
    if not isinstance(request_profile, dict) or not isinstance(control_plane, dict):
        raise LabError("remote outbox lacks Profile/control-plane identity")
    if request_profile.get("target_host") != profile.target_host:
        raise LabError("transport Profile targets a different request host")
    prefix, fingerprint = _trusted_ssh_prefix(
        profile=profile,
        known_hosts_path=known_hosts_path,
        ssh_binary=ssh_binary,
    )
    argv = [
        *prefix,
        str(profile.remote_projectctl),
        "remote-inbox",
        "accept",
        "--envelope",
        "-",
        "--profile",
        str(request_profile["profile_id"]),
        "--inbox-root",
        str(profile.remote_inbox_root),
        "--state-root",
        str(profile.remote_state_root),
        "--control-version",
        str(control_plane["minimum_version"]),
    ]
    return argv, fingerprint


def build_ssh_control_argv(
    *,
    envelope: object,
    profile: TransportProfile,
    known_hosts_path: Path,
    action: str,
    after_sequence: int = -1,
    watch_timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.25,
    ssh_binary: Path = Path("/usr/bin/ssh"),
) -> tuple[list[str], str]:
    validated = validate_remote_outbox(envelope)
    request = validated.get("request")
    if not isinstance(request, dict):
        raise LabError("remote outbox lacks request")
    request_profile = request.get("profile")
    if (
        not isinstance(request_profile, dict)
        or request_profile.get("target_host") != profile.target_host
    ):
        raise LabError("transport Profile targets a different request host")
    if action not in {"status", "watch", "cancel"}:
        raise LabError(f"unsupported trusted remote control action: {action}")
    if after_sequence < -1:
        raise LabError("watch after_sequence must be at least -1")
    if not 0 <= watch_timeout_seconds <= 60:
        raise LabError("watch timeout must be between 0 and 60 seconds")
    if not 0.01 <= poll_interval_seconds <= 5:
        raise LabError("watch poll interval must be between 0.01 and 5 seconds")
    prefix, fingerprint = _trusted_ssh_prefix(
        profile=profile,
        known_hosts_path=known_hosts_path,
        ssh_binary=ssh_binary,
    )
    argv = [
        *prefix,
        str(profile.remote_projectctl),
        "remote-runner",
        action,
        str(request["request_id"]),
        "--state-root",
        str(profile.remote_state_root),
    ]
    if action == "watch":
        argv.extend(
            [
                "--after-sequence",
                str(after_sequence),
                "--timeout-seconds",
                str(watch_timeout_seconds),
                "--poll-interval-seconds",
                str(poll_interval_seconds),
            ]
        )
    return argv, fingerprint


def _default_ssh_runner(
    argv: list[str],
    stdin_payload: str,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        input=stdin_payload,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _run_json_ssh(
    *,
    argv: list[str],
    stdin_payload: str,
    timeout_seconds: int,
    ssh_runner: SSHRunner,
    operation: str,
) -> object:
    try:
        completed = ssh_runner(argv, stdin_payload, timeout_seconds)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LabError(f"authenticated SSH {operation} failed: {exc}") from exc
    if completed.returncode != 0:
        detail = " ".join(completed.stderr.split())[:1000]
        raise LabError(
            f"authenticated SSH {operation} exited {completed.returncode}: "
            f"{detail or 'no diagnostic'}"
        )
    if len(completed.stdout.encode()) > MAX_SSH_JSON_BYTES:
        raise LabError(f"authenticated SSH {operation} returned oversized JSON")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LabError(f"authenticated SSH {operation} returned invalid JSON") from exc


def dispatch_remote_outbox(
    *,
    envelope: object,
    profile: TransportProfile,
    known_hosts_path: Path,
    timeout_seconds: int = 30,
    ssh_binary: Path = Path("/usr/bin/ssh"),
    ssh_runner: SSHRunner = _default_ssh_runner,
) -> tuple[dict[str, object], dict[str, object]]:
    if not 1 <= timeout_seconds <= 300:
        raise LabError("SSH dispatch timeout must be between 1 and 300 seconds")
    validated = validate_remote_outbox(envelope)
    request = validated["request"]
    if not isinstance(request, dict):
        raise LabError("remote outbox lacks request")
    argv, fingerprint = build_ssh_dispatch_argv(
        envelope=validated,
        profile=profile,
        known_hosts_path=known_hosts_path,
        ssh_binary=ssh_binary,
    )
    raw_acceptance = _run_json_ssh(
        argv=argv,
        stdin_payload=json.dumps(validated, sort_keys=True) + "\n",
        timeout_seconds=timeout_seconds,
        ssh_runner=ssh_runner,
        operation="dispatch",
    )
    acceptance = validate_remote_acceptance(
        raw_acceptance,
        request_id=str(request["request_id"]),
        idempotency_key=str(request["idempotency_key"]),
        target_host=profile.target_host,
    )
    transport_receipt = build_transport_receipt(
        envelope=validated,
        acceptance=acceptance,
        profile=profile,
        observed_host_key_fingerprint=fingerprint,
    )
    return acceptance, transport_receipt


def control_remote_request(
    *,
    envelope: object,
    profile: TransportProfile,
    known_hosts_path: Path,
    action: str,
    after_sequence: int = -1,
    watch_timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.25,
    ssh_timeout_seconds: int = 35,
    ssh_binary: Path = Path("/usr/bin/ssh"),
    ssh_runner: SSHRunner = _default_ssh_runner,
) -> tuple[dict[str, object], str]:
    """Query or cancel one accepted request through a fixed trusted command."""

    if not 1 <= ssh_timeout_seconds <= 300:
        raise LabError("SSH control timeout must be between 1 and 300 seconds")
    if action == "watch" and ssh_timeout_seconds <= watch_timeout_seconds:
        raise LabError("SSH timeout must exceed the remote watch timeout")
    validated = validate_remote_outbox(envelope)
    request = validated.get("request")
    if not isinstance(request, dict):
        raise LabError("remote outbox lacks request")
    argv, fingerprint = build_ssh_control_argv(
        envelope=validated,
        profile=profile,
        known_hosts_path=known_hosts_path,
        action=action,
        after_sequence=after_sequence,
        watch_timeout_seconds=watch_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        ssh_binary=ssh_binary,
    )
    raw = _run_json_ssh(
        argv=argv,
        stdin_payload="",
        timeout_seconds=ssh_timeout_seconds,
        ssh_runner=ssh_runner,
        operation=action,
    )
    if action == "watch":
        watched = validate_remote_watch_result(
            raw,
            request=request,
            after_sequence=after_sequence,
        )
        state = watched.get("request")
        if not isinstance(state, dict):
            raise LabError("remote watch result lacks request state")
        return state, fingerprint
    return validate_request_state_identity(raw, request=request), fingerprint
