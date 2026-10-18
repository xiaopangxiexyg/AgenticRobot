"""Change only the left UDP receiver IP, then verify ten seconds of arrival."""

from __future__ import annotations

import argparse
import copy
import ipaddress
import json
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path


def update_destination(
    arm_ips, from_ip, to_ip, output, *, clock=time.monotonic, wait=time.sleep
):
    from hardware.rm65.utils.scripts import rm65_udp_actuator_trace as trace

    output.mkdir(parents=True, exist_ok=False)
    journal_path = output / "rollback.json"
    guardian = guardian_pipe = None
    write_attempted = False
    journal = None
    result = {
        "side": "left",
        "scope": "configuration_and_arrival_only_not_motion_validation",
    }

    def acknowledge_guardian():
        nonlocal guardian_pipe
        pipe = guardian_pipe
        guardian_pipe = None
        try:
            trace._notify_guardian_restored(guardian, pipe)
        finally:
            if pipe is not None and not pipe.closed:
                pipe.close()

    try:
        with ExitStack() as resources:
            writer = trace.UdpTraceWriter(output / "left_udp.jsonl")
            receiver = trace.UdpReceiver(
                trace.ArmSpec("left", arm_ips["left"], to_ip, 8090),
                writer,
                trace_format="full",
            )
            resources.callback(receiver.socket.close)
            client = trace.Api2PushConfigClient(arm_ips["left"])
            resources.callback(client.close)
            original = client.get_config()
            if original["ip"] != from_ip:
                raise RuntimeError(
                    f"Left destination changed since inspection: {original['ip']}"
                )
            if not original["enable"] or original["port"] != 8090:
                raise RuntimeError(
                    "Existing left UDP push must already be enabled on port 8090"
                )
            target = copy.deepcopy(original)
            target["ip"] = to_ip
            trace._atomic_write_json(output / "original_left.json", original)
            journal = {
                "schema": trace.JOURNAL_SCHEMA,
                "status": "prepared",
                "arms": [
                    {
                        "side": "left",
                        "arm_ip": arm_ips["left"],
                        "original_config": original,
                        "target_config": target,
                    }
                ],
            }
            trace._atomic_write_json(journal_path, journal)
            guardian, guardian_pipe = trace._start_guardian(journal_path)
            writer.start()
            resources.callback(writer.close)
            receiver.start()
            resources.callback(receiver.close)
            # A failed setter may still have changed the controller.
            write_attempted = True
            client.set_config(target)
            observed = client.get_config()
            if not trace.push_configs_equal(observed, target):
                raise RuntimeError(
                    f"Left UDP readback differs from the IP-only target: {observed}"
                )
            deadline = clock() + 10.0
            while (remaining := deadline - clock()) > 0.0:
                if writer.error is not None:
                    raise RuntimeError(f"UDP evidence writer failed: {writer.error}")
                if receiver.error is not None:
                    raise RuntimeError(f"UDP receiver failed: {receiver.error}")
                wait(min(0.05, remaining))
            observed = client.get_config()
            if not trace.push_configs_equal(observed, target):
                raise RuntimeError(
                    "Left UDP configuration changed during arrival verification"
                )
        if not receiver.sequence or not receiver.joint_position_count:
            raise RuntimeError(
                "No left-arm joint-position UDP feedback arrived during verification"
            )
        if receiver.unexpected_source_count or writer.dropped:
            raise RuntimeError(
                "UDP verification has unexpected sources or dropped evidence"
            )
        journal["status"] = "committed"
        trace._atomic_write_json(journal_path, journal)
        result.update(
            status="passed",
            configuration_persisted=True,
            controller_config=observed,
            packet_count=receiver.sequence,
            json_packet_count=receiver.json_payload_count,
            joint_position_packet_count=receiver.joint_position_count,
            cleanup_complete=True,
            right_controller_connected=False,
        )
        trace._atomic_write_json(output / "result.json", result)
        # Acknowledge the committed configuration only after final evidence.
        acknowledge_guardian()
        return result
    except BaseException as error:
        result.pop("cleanup_complete", None)
        result.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            configuration_persisted=None,
            configuration_write_attempted=write_attempted,
        )
        record_errors = []

        def record_failure():
            try:
                trace._atomic_write_json(output / "result.json", result)
            except OSError as record_error:
                record_errors.append(f"{type(record_error).__name__}: {record_error}")

        # Failure evidence must not prevent restoring a possibly applied write.
        record_failure()
        restored = False
        if write_attempted:
            try:
                trace.restore_journal(
                    journal_path, actor="left_destination_update_failure"
                )
                restored = True
            except Exception as restore_error:  # noqa: BLE001 - preserve recovery failure evidence.
                result["restore_error"] = (
                    f"{type(restore_error).__name__}: {restore_error}"
                )
        if guardian_pipe is not None:
            if restored or not write_attempted:
                acknowledge_guardian()
            else:
                guardian_pipe.close()
                guardian.wait(timeout=10.0)
                restored = trace._load_json(journal_path).get("status") == "restored"
            guardian_pipe = None
        result["original_configuration_restored"] = restored
        if restored or not write_attempted:
            result["configuration_persisted"] = False
        record_failure()
        if record_errors:
            detail = "; ".join(record_errors)
            raise error from trace.RestoreError(
                f"Failure evidence could not be saved: {detail}; "
                f"original_configuration_restored={restored}; "
                f"restore_error={result.get('restore_error')}"
            )
        raise
    finally:
        if guardian_pipe is not None:
            # EOF activates the detached original-config restorer after an
            # unexpected parent failure; its journal contains only the left arm.
            guardian_pipe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--arm-ip-assignment", type=Path, required=True)
    parser.add_argument("--left-controller-ip", required=True)
    parser.add_argument("--from-ip", required=True)
    parser.add_argument("--to-ip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    output = args.output.resolve()
    if output == Path("/") or output.is_relative_to(repo) or output.exists():
        raise ValueError(
            "Output must be a new explicit directory outside the source checkout"
        )
    ips = json.loads(args.arm_ip_assignment.read_text())["arm_ip_by_side"]
    if set(ips) != {"left", "right"} or len(set(ips.values())) != 2:
        raise ValueError("Expected distinct explicit left/right controller addresses")
    if ips["left"] != args.left_controller_ip:
        raise RuntimeError(
            "Left controller assignment differs from the explicitly authorized address"
        )
    for value in (*ips.values(), args.from_ip, args.to_ip):
        ipaddress.IPv4Address(value)
    if args.from_ip == args.to_ip:
        raise ValueError(
            "The receiver-IP correction must name distinct old and new addresses"
        )
    route = json.loads(
        subprocess.check_output(["ip", "-j", "route", "get", ips["left"]], text=True)
    )
    if len(route) != 1 or route[0].get("prefsrc") != args.to_ip:
        raise RuntimeError(
            "New destination does not match this host's left-arm route source"
        )
    listeners = subprocess.check_output(["ss", "-lunH"], text=True)
    if any(
        field.endswith(":8090")
        for line in listeners.splitlines()
        for field in line.split()
    ):
        raise RuntimeError("UDP 8090 is occupied; no process will be interrupted")
    sys.path.insert(0, str(repo))
    from scripts.probes.capture_rm65_grasp_snapshot import assert_arm_connections_idle

    assert_arm_connections_idle({"left": ips["left"]})
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    sdk_paths = [
        "hardware/rm65/utils/scripts/rm65_udp_actuator_trace.py",
        "hardware/rm65/Robotic_Arm/rm_robot_interface.py",
        "hardware/rm65/Robotic_Arm/rm_ctypes_wrap.py",
        "scripts/probes/capture_rm65_grasp_snapshot.py",
    ]
    subprocess.run(
        ["git", "-C", str(repo), "diff", "HEAD", "--quiet", "--", *sdk_paths],
        check=True,
    )
    print(
        json.dumps(
            {
                "preflight": "passed",
                "repo_commit": head,
                "left_controller": ips["left"],
                "from_ip": args.from_ip,
                "to_ip": args.to_ip,
                "output": str(output),
                "apply": args.apply,
            }
        ),
        flush=True,
    )
    if not args.apply:
        return 0

    def interrupted(signum, _frame):
        raise InterruptedError(f"Destination update interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    print(
        json.dumps(
            update_destination(ips, args.from_ip, args.to_ip, output),
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
