from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab_control import network_namespace
from lab_control.endpoint_lease import EndpointLeaseStore
from lab_control.network_namespace import (
    NetworkNamespaceProbePolicy,
    build_network_namespace_plan,
    collect_network_namespace_anchor_observation,
    collect_network_namespace_member_observation,
    collect_network_namespace_probe,
    validate_network_namespace_anchor_observation,
    validate_network_namespace_member_observation,
    validate_network_namespace_plan,
    validate_network_namespace_probe_for_use,
    validate_network_namespace_probe_receipt,
)
from lab_control.remote_preflight import HostNetworkPolicy, HostProfile
from lab_control.remote_request import _sha256
from lab_control.workspace import LabError


def _host(tmp_path: Path, *, host_id: str = "network-test-host") -> HostProfile:
    return HostProfile(
        host_id=host_id,
        hostname=host_id,
        capabilities=(),
        repositories={},
        environments={},
        asset_roots=(),
        slots=("slot-01",),
        slot_root=(tmp_path / "slots").resolve(),
        control_plane_projectctl=(tmp_path / "projectctl").resolve(),
        device_access=False,
        network_policy=HostNetworkPolicy(
            task_payload_egress="loopback-only",
            trusted_prepare_egress="explicit-only",
            implicit_downloads=False,
            large_file_downloads=False,
        ),
        sha256="a" * 64,
    )


def _lease(
    tmp_path: Path,
    host: HostProfile,
    *,
    request_id: str = "REQ-network-0001",
    port: int = 18427,
) -> dict[str, object]:
    return EndpointLeaseStore(
        (tmp_path / request_id / "leases").resolve(),
        host_id=host.host_id,
        host_profile_sha256=host.sha256,
    ).acquire(
        request_id=request_id,
        slot_id="slot-01",
        scheme="http",
        port=port,
        now=datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc),
    )


def _rehash(plan: dict[str, object]) -> None:
    identity_content = {
        key: value
        for key, value in plan.items()
        if key not in {"namespace_id", "receipt_sha256"}
    }
    plan["namespace_id"] = f"NET-{_sha256(identity_content)[:24]}"
    content = {key: value for key, value in plan.items() if key != "receipt_sha256"}
    plan["receipt_sha256"] = _sha256(content)


def _rehash_receipt(receipt: dict[str, object]) -> None:
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = _sha256(content)


def _write_process(
    proc_root: Path,
    *,
    pid: int,
    parent_pid: int,
    process_group_id: int,
    session_id: int,
    start_ticks: int,
    no_new_privileges: bool = True,
) -> None:
    process = proc_root / str(pid)
    (process / "ns").mkdir(parents=True)
    fields = [
        "S",
        str(parent_pid),
        str(process_group_id),
        str(session_id),
        *(["0"] * 15),
        str(start_ticks),
    ]
    (process / "stat").write_text(
        f"{pid} (network process) {' '.join(fields)}\n",
        encoding="utf-8",
    )
    (process / "status").write_text(
        f"Name:\tnetwork-process\nNoNewPrivs:\t{int(no_new_privileges)}\n",
        encoding="utf-8",
    )


def _proc_fixture(tmp_path: Path) -> tuple[Path, int, int, int]:
    root = (tmp_path / "proc").resolve()
    collector_pid = 4100
    anchor_pid = 4200
    member_pid = 4300
    _write_process(
        root,
        pid=collector_pid,
        parent_pid=1,
        process_group_id=collector_pid,
        session_id=collector_pid,
        start_ticks=100,
    )
    _write_process(
        root,
        pid=anchor_pid,
        parent_pid=collector_pid,
        process_group_id=anchor_pid,
        session_id=anchor_pid,
        start_ticks=200,
    )
    _write_process(
        root,
        pid=member_pid,
        parent_pid=anchor_pid,
        process_group_id=member_pid,
        session_id=anchor_pid,
        start_ticks=300,
    )
    (root / str(collector_pid) / "ns" / "net").write_text(
        "collector-net\n",
        encoding="utf-8",
    )
    anchor_net = root / str(anchor_pid) / "ns" / "net"
    anchor_net.write_text("anchor-net\n", encoding="utf-8")
    (root / str(member_pid) / "ns" / "net").hardlink_to(anchor_net)
    return root, collector_pid, anchor_pid, member_pid


def _probe_policy(
    tmp_path: Path,
    *,
    external_routes_absent: bool = True,
) -> NetworkNamespaceProbePolicy:
    enter = (tmp_path / "protected" / "nsenter").resolve()
    python = (tmp_path / "protected" / "python").resolve()
    enter.parent.mkdir(parents=True, exist_ok=True)
    enter.write_text(
        "#!/usr/bin/python3\n"
        "import json, sys\n"
        "assert sys.argv[1].startswith('--net=')\n"
        "assert sys.argv[2] == '--'\n"
        "assert sys.argv[4:6] == ['-I', '-c']\n"
        "inode = int(sys.argv[-1])\n"
        "print(json.dumps({\n"
        f"  'external_routes_absent': {external_routes_absent!r},\n"
        "  'loopback_available': True,\n"
        "  'network_namespace_inode': inode,\n"
        "  'only_loopback_interfaces': True,\n"
        "  'requested_namespace_matched': True,\n"
        "}, sort_keys=True))\n",
        encoding="utf-8",
    )
    enter.chmod(0o700)
    python.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    python.chmod(0o700)
    return NetworkNamespaceProbePolicy(
        policy_id="fixed-netns-probe-v1",
        enter_executable=enter,
        python_executable=python,
        max_probe_age_seconds=300,
    )


def test_shared_network_plan_is_deterministic_and_non_executable(
    tmp_path: Path,
) -> None:
    host = _host(tmp_path)
    lease = _lease(tmp_path, host)
    observed = datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc)

    first = build_network_namespace_plan(lease, host_profile=host, now=observed)
    second = build_network_namespace_plan(lease, host_profile=host, now=observed)

    assert first == second
    assert (
        validate_network_namespace_plan(
            first,
            endpoint_lease_receipt=lease,
            host_profile=host,
            now=observed,
        )
        == first
    )
    assert first["topology"] == "shared-isolated-loopback"
    assert first["required_members"] == ["attempt", "service"]
    assert first["namespace_created"] is False
    assert first["service_joined"] is False
    assert first["attempt_joined"] is False
    assert first["launch_authorized"] is False
    assert (
        validate_network_namespace_plan(
            first,
            endpoint_lease_receipt=lease,
            host_profile=host,
            require_active=False,
            now=datetime(2026, 8, 14, 12, 11, tzinfo=timezone.utc),
        )
        == first
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("namespace_created", True),
        ("loopback_ready", True),
        ("service_joined", True),
        ("attempt_joined", True),
        ("launch_authorized", True),
        ("external_network_permitted", True),
        ("device_access", True),
    ],
)
def test_network_plan_rejects_rehashed_runtime_or_authority_claims(
    tmp_path: Path,
    field: str,
    value: bool,
) -> None:
    host = _host(tmp_path)
    lease = _lease(tmp_path, host)
    plan = build_network_namespace_plan(
        lease,
        host_profile=host,
        now=datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc),
    )
    plan[field] = value
    _rehash(plan)

    with pytest.raises(LabError, match="unsafe authority envelope"):
        validate_network_namespace_plan(plan)


def test_network_plan_rejects_cross_request_or_host_replay(tmp_path: Path) -> None:
    host = _host(tmp_path)
    first_lease = _lease(tmp_path, host)
    plan = build_network_namespace_plan(
        first_lease,
        host_profile=host,
        now=datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc),
    )
    other_lease = _lease(
        tmp_path,
        host,
        request_id="REQ-network-0002",
        port=18428,
    )
    with pytest.raises(LabError, match="identity mismatch"):
        validate_network_namespace_plan(
            plan,
            endpoint_lease_receipt=other_lease,
            host_profile=host,
            now=datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc),
        )

    other_host = replace(host, host_id="other-network-host", sha256="b" * 64)
    with pytest.raises(LabError, match="host Profile mismatch"):
        validate_network_namespace_plan(
            plan,
            host_profile=other_host,
            now=datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc),
        )


def test_anchor_and_member_observations_bind_one_shared_namespace(
    tmp_path: Path,
) -> None:
    host = _host(tmp_path)
    lease = _lease(tmp_path, host)
    plan = build_network_namespace_plan(
        lease,
        host_profile=host,
        now=datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc),
    )
    proc_root, collector_pid, anchor_pid, member_pid = _proc_fixture(tmp_path)
    observed = datetime(2026, 8, 14, 12, 2, tzinfo=timezone.utc)

    anchor = collect_network_namespace_anchor_observation(
        plan,
        host_profile=host,
        pid=anchor_pid,
        proc_root=proc_root,
        collector_pid=collector_pid,
        now=observed,
    )
    member = collect_network_namespace_member_observation(
        plan,
        anchor_observation=anchor,
        role="service",
        pid=member_pid,
        proc_root=proc_root,
        now=observed,
    )

    assert (
        validate_network_namespace_anchor_observation(
            anchor,
            network_namespace_plan=plan,
            host_profile=host,
        )
        == anchor
    )
    assert (
        validate_network_namespace_member_observation(
            member,
            network_namespace_plan=plan,
            anchor_observation=anchor,
            expected_role="service",
        )
        == member
    )
    assert member["network_namespace_inode"] == anchor[
        "anchor_network_namespace_inode"
    ]
    assert anchor["loopback_verified"] is False
    assert member["connectivity_authorized"] is False
    assert member["launch_authorized"] is False


def test_member_observation_rejects_a_different_network_namespace(
    tmp_path: Path,
) -> None:
    host = _host(tmp_path)
    lease = _lease(tmp_path, host)
    plan = build_network_namespace_plan(
        lease,
        host_profile=host,
        now=datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc),
    )
    proc_root, collector_pid, anchor_pid, member_pid = _proc_fixture(tmp_path)
    anchor = collect_network_namespace_anchor_observation(
        plan,
        host_profile=host,
        pid=anchor_pid,
        proc_root=proc_root,
        collector_pid=collector_pid,
    )
    member_net = proc_root / str(member_pid) / "ns" / "net"
    member_net.unlink()
    member_net.write_text("different-net\n", encoding="utf-8")

    with pytest.raises(LabError, match="did not join the shared namespace"):
        collect_network_namespace_member_observation(
            plan,
            anchor_observation=anchor,
            role="service",
            pid=member_pid,
            proc_root=proc_root,
        )


def test_anchor_observation_rejects_pid_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(tmp_path)
    lease = _lease(tmp_path, host)
    plan = build_network_namespace_plan(
        lease,
        host_profile=host,
        now=datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc),
    )
    proc_root, collector_pid, anchor_pid, _member_pid = _proc_fixture(tmp_path)
    original = network_namespace._process_stat
    calls = 0

    def changing_stat(path: Path, *, expected_pid: int) -> dict[str, int]:
        nonlocal calls
        result = original(path, expected_pid=expected_pid)
        calls += 1
        if calls == 2:
            result["proc_start_time_ticks"] += 1
        return result

    monkeypatch.setattr(network_namespace, "_process_stat", changing_stat)
    with pytest.raises(LabError, match="identity changed"):
        collect_network_namespace_anchor_observation(
            plan,
            host_profile=host,
            pid=anchor_pid,
            proc_root=proc_root,
            collector_pid=collector_pid,
        )


def test_observations_reject_rehashed_authority_claims(tmp_path: Path) -> None:
    host = _host(tmp_path)
    lease = _lease(tmp_path, host)
    plan = build_network_namespace_plan(
        lease,
        host_profile=host,
        now=datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc),
    )
    proc_root, collector_pid, anchor_pid, member_pid = _proc_fixture(tmp_path)
    anchor = collect_network_namespace_anchor_observation(
        plan,
        host_profile=host,
        pid=anchor_pid,
        proc_root=proc_root,
        collector_pid=collector_pid,
    )
    member = collect_network_namespace_member_observation(
        plan,
        anchor_observation=anchor,
        role="attempt",
        pid=member_pid,
        proc_root=proc_root,
    )

    anchor["membership_eligible"] = True
    _rehash_receipt(anchor)
    with pytest.raises(LabError, match="unsafe authority envelope"):
        validate_network_namespace_anchor_observation(anchor)

    member["connectivity_authorized"] = True
    _rehash_receipt(member)
    with pytest.raises(LabError, match="unsafe authority envelope"):
        validate_network_namespace_member_observation(member)


def test_observations_reject_cross_request_and_role_replay(tmp_path: Path) -> None:
    host = _host(tmp_path)
    first_lease = _lease(tmp_path, host)
    first_plan = build_network_namespace_plan(
        first_lease,
        host_profile=host,
        now=datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc),
    )
    second_lease = _lease(
        tmp_path,
        host,
        request_id="REQ-network-0002",
        port=18428,
    )
    second_plan = build_network_namespace_plan(
        second_lease,
        host_profile=host,
        now=datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc),
    )
    proc_root, collector_pid, anchor_pid, member_pid = _proc_fixture(tmp_path)
    anchor = collect_network_namespace_anchor_observation(
        first_plan,
        host_profile=host,
        pid=anchor_pid,
        proc_root=proc_root,
        collector_pid=collector_pid,
    )
    member = collect_network_namespace_member_observation(
        first_plan,
        anchor_observation=anchor,
        role="service",
        pid=member_pid,
        proc_root=proc_root,
    )

    with pytest.raises(LabError, match="anchor binding mismatch"):
        validate_network_namespace_anchor_observation(
            anchor,
            network_namespace_plan=second_plan,
            host_profile=host,
        )
    with pytest.raises(LabError, match="invalid role"):
        validate_network_namespace_member_observation(
            member,
            network_namespace_plan=first_plan,
            anchor_observation=anchor,
            expected_role="attempt",
        )


def _probe_fixture(
    tmp_path: Path,
    *,
    external_routes_absent: bool = True,
) -> tuple[
    HostProfile,
    dict[str, object],
    dict[str, object],
    Path,
    NetworkNamespaceProbePolicy,
]:
    host = _host(tmp_path)
    lease = _lease(tmp_path, host)
    plan = build_network_namespace_plan(
        lease,
        host_profile=host,
        now=datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc),
    )
    proc_root, collector_pid, anchor_pid, _member_pid = _proc_fixture(tmp_path)
    anchor = collect_network_namespace_anchor_observation(
        plan,
        host_profile=host,
        pid=anchor_pid,
        proc_root=proc_root,
        collector_pid=collector_pid,
        now=datetime(2026, 8, 14, 12, 2, tzinfo=timezone.utc),
    )
    return (
        host,
        plan,
        anchor,
        proc_root,
        _probe_policy(
            tmp_path,
            external_routes_absent=external_routes_absent,
        ),
    )


def test_fixed_namespace_probe_binds_policy_and_live_anchor(tmp_path: Path) -> None:
    host, plan, anchor, proc_root, policy = _probe_fixture(tmp_path)
    observed = datetime(2026, 8, 14, 12, 3, tzinfo=timezone.utc)

    receipt = collect_network_namespace_probe(
        plan,
        anchor_observation=anchor,
        host_profile=host,
        policy=policy,
        proc_root=proc_root,
        now=observed,
    )

    assert (
        validate_network_namespace_probe_receipt(
            receipt,
            network_namespace_plan=plan,
            anchor_observation=anchor,
            policy=policy,
        )
        == receipt
    )
    assert (
        validate_network_namespace_probe_for_use(
            receipt,
            network_namespace_plan=plan,
            anchor_observation=anchor,
            host_profile=host,
            policy=policy,
            proc_root=proc_root,
            now=datetime(2026, 8, 14, 12, 4, tzinfo=timezone.utc),
        )
        == receipt
    )
    assert receipt["loopback_verified"] is True
    assert receipt["external_network_absence_verified"] is True
    assert receipt["network_ready"] is True
    assert receipt["launch_authorized"] is False


def test_namespace_probe_payload_is_fixed_and_does_not_connect_externally() -> None:
    compile(network_namespace._PROBE_SCRIPT, "<network-probe>", "exec")
    assert "connect(" not in network_namespace._PROBE_SCRIPT
    assert "127.0.0.1" in network_namespace._PROBE_SCRIPT


def test_namespace_probe_rejects_external_route_or_policy_symlink(
    tmp_path: Path,
) -> None:
    host, plan, anchor, proc_root, policy = _probe_fixture(
        tmp_path,
        external_routes_absent=False,
    )
    with pytest.raises(LabError, match="did not prove loopback isolation"):
        collect_network_namespace_probe(
            plan,
            anchor_observation=anchor,
            host_profile=host,
            policy=policy,
            proc_root=proc_root,
            now=datetime(2026, 8, 14, 12, 3, tzinfo=timezone.utc),
        )

    real_enter = policy.enter_executable
    symlink = real_enter.parent / "nsenter-link"
    symlink.symlink_to(real_enter)
    unsafe_policy = replace(policy, enter_executable=symlink)
    with pytest.raises(LabError, match="path is unsafe"):
        collect_network_namespace_probe(
            plan,
            anchor_observation=anchor,
            host_profile=host,
            policy=unsafe_policy,
            proc_root=proc_root,
            now=datetime(2026, 8, 14, 12, 3, tzinfo=timezone.utc),
        )


def test_namespace_probe_rejects_stale_or_changed_backend(tmp_path: Path) -> None:
    host, plan, anchor, proc_root, policy = _probe_fixture(tmp_path)
    receipt = collect_network_namespace_probe(
        plan,
        anchor_observation=anchor,
        host_profile=host,
        policy=policy,
        proc_root=proc_root,
        now=datetime(2026, 8, 14, 12, 3, tzinfo=timezone.utc),
    )
    with pytest.raises(LabError, match="stale or from the future"):
        validate_network_namespace_probe_for_use(
            receipt,
            network_namespace_plan=plan,
            anchor_observation=anchor,
            host_profile=host,
            policy=policy,
            proc_root=proc_root,
            now=datetime(2026, 8, 14, 12, 9, tzinfo=timezone.utc),
        )

    policy.python_executable.write_text(
        "#!/bin/sh\nexit 98\n",
        encoding="utf-8",
    )
    policy.python_executable.chmod(0o700)
    with pytest.raises(LabError, match="changed after network namespace probe"):
        validate_network_namespace_probe_for_use(
            receipt,
            network_namespace_plan=plan,
            anchor_observation=anchor,
            host_profile=host,
            policy=policy,
            proc_root=proc_root,
            now=datetime(2026, 8, 14, 12, 4, tzinfo=timezone.utc),
        )


def test_namespace_probe_rejects_rehashed_launch_authority(tmp_path: Path) -> None:
    host, plan, anchor, proc_root, policy = _probe_fixture(tmp_path)
    receipt = collect_network_namespace_probe(
        plan,
        anchor_observation=anchor,
        host_profile=host,
        policy=policy,
        proc_root=proc_root,
        now=datetime(2026, 8, 14, 12, 3, tzinfo=timezone.utc),
    )
    receipt["launch_authorized"] = True
    _rehash_receipt(receipt)

    with pytest.raises(LabError, match="unsafe authority envelope"):
        validate_network_namespace_probe_receipt(receipt)
