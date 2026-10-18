"""Set only both RM65 UDP force coordinates to tool=2; no actuator commands."""

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
from dataclasses import asdict
from pathlib import Path


def update_force_coordinates(arm_ips, local_ips, output, *, clock=time.monotonic, wait=time.sleep):
    from hardware.rm65.realtime_feedback import RM65UDPFeedbackReceiver
    from hardware.rm65.utils.scripts import rm65_udp_actuator_trace as trace

    output.mkdir(parents=True, exist_ok=False)
    journal_path = output / "rollback.json"
    guardian = guardian_pipe = None
    write_attempted = False
    result = {"scope": "UDP_force_coordinate_only_no_motion", "status": "running"}

    def acknowledge_guardian():
        nonlocal guardian_pipe
        pipe, guardian_pipe = guardian_pipe, None
        try:
            trace._notify_guardian_restored(guardian, pipe)
        finally:
            if pipe is not None and not pipe.closed:
                pipe.close()

    try:
        with ExitStack() as resources:
            receiver = RM65UDPFeedbackReceiver(arm_ip_by_side=arm_ips, port=8090)
            resources.callback(receiver.close)
            # Claim the passive port before opening any controller connection.
            receiver.start()
            clients, originals, targets = {}, {}, {}
            for side in ("left", "right"):
                client = trace.Api2PushConfigClient(arm_ips[side])
                resources.callback(client.close)
                clients[side] = client
                original = client.get_config()
                if not original["enable"] or original["port"] != 8090:
                    raise RuntimeError(f"{side}: existing UDP must be enabled on port 8090")
                if original["ip"] != local_ips[side]:
                    raise RuntimeError(f"{side}: existing destination does not match the host route")
                originals[side] = original
                targets[side] = copy.deepcopy(original)
                targets[side]["force_coordinate"] = 2
                trace._atomic_write_json(output / f"original_{side}.json", original)
            changed = [side for side in ("left", "right") if originals[side] != targets[side]]
            journal = {"schema": trace.JOURNAL_SCHEMA, "status": "prepared", "arms": [
                {"side": side, "arm_ip": arm_ips[side], "original_config": originals[side],
                 "target_config": targets[side]} for side in changed
            ]}
            trace._atomic_write_json(journal_path, journal)
            if changed:
                guardian, guardian_pipe = trace._start_guardian(journal_path)
            for side in changed:
                # A failed setter may already have taken effect.
                write_attempted = True
                clients[side].set_config(targets[side])
                if not trace.push_configs_equal(clients[side].get_config(), targets[side]):
                    raise RuntimeError(f"{side}: readback differs from the coordinate-only target")
            receiver.wait_ready(timeout_sec=2., max_age_sec=.2)
            counts = dict.fromkeys(arm_ips, 0)
            last_sequences = dict.fromkeys(arm_ips, 0)
            deadline = clock() + 10.
            with (output / "observed_feedback.jsonl").open("w", encoding="utf-8", buffering=1) as evidence:
                while clock() < deadline:
                    for side in ("left", "right"):
                        sample = receiver.snapshot(side)
                        age = time.monotonic() - sample.monotonic_time_sec
                        if sample.force_coordinate != 2 or not 0 <= age <= .2:
                            raise RuntimeError(f"{side}: passive tool-frame feedback stale or coordinate changed")
                        if sample.sequence != last_sequences[side]:
                            evidence.write(json.dumps(asdict(sample), separators=(",", ":")) + "\n")
                            counts[side] += 1
                            last_sequences[side] = sample.sequence
                    wait(min(.02, max(0., deadline - clock())))
            observed = {side: clients[side].get_config() for side in arm_ips}
            if any(not trace.push_configs_equal(observed[side], targets[side]) for side in arm_ips):
                raise RuntimeError("UDP configuration changed during passive verification")
            diagnostic = receiver.diagnostics()
            if (any(diagnostic["invalid_packet_count_by_side"].values())
                    or diagnostic["unknown_source_count"] or diagnostic["fault"]
                    or not all(counts.values())):
                raise RuntimeError(f"Invalid passive UDP evidence: {diagnostic}")
        journal["status"] = "committed"
        trace._atomic_write_json(journal_path, journal)
        result.update(status="passed", configuration_persisted=True, changed_sides=changed,
                      controller_config_by_side=observed, sampled_packet_count_by_side=counts,
                      udp_diagnostic=diagnostic, cleanup_complete=True)
        trace._atomic_write_json(output / "result.json", result)
        if guardian_pipe is not None:
            # Acknowledge the committed configuration only after final evidence.
            acknowledge_guardian()
        return result
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      configuration_write_attempted=write_attempted, configuration_persisted=None)
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
                trace.restore_journal(journal_path, actor="force_coordinate_update_failure")
                restored = True
            except Exception as restore_error:  # noqa: BLE001 - retain rollback failure evidence.
                result["restore_error"] = f"{type(restore_error).__name__}: {restore_error}"
        if guardian_pipe is not None:
            if restored or not write_attempted:
                acknowledge_guardian()
            else:
                guardian_pipe.close()
                guardian_pipe = None
                guardian.wait(timeout=10.)
                restored = trace._load_json(journal_path).get("status") == "restored"
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
            guardian_pipe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--arm-ip-assignment", type=Path, required=True)
    parser.add_argument("--left-controller-ip", required=True)
    parser.add_argument("--right-controller-ip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    output = args.output.resolve()
    if output == Path("/") or output.is_relative_to(repo) or output.exists():
        raise ValueError("Output must be a new explicit directory outside the source checkout")
    ips = json.loads(args.arm_ip_assignment.read_text())["arm_ip_by_side"]
    authorized = {"left": args.left_controller_ip, "right": args.right_controller_ip}
    if ips != authorized or len(set(ips.values())) != 2:
        raise RuntimeError("Controller assignment differs from the authorized left/right addresses")
    local_ips = {}
    for side, ip in ips.items():
        ipaddress.IPv4Address(ip)
        route = json.loads(subprocess.check_output(["ip", "-j", "route", "get", ip], text=True))
        if len(route) != 1 or not route[0].get("prefsrc"):
            raise RuntimeError(f"{side}: cannot resolve a unique local route source")
        local_ips[side] = str(ipaddress.IPv4Address(route[0]["prefsrc"]))
    listeners = subprocess.check_output(["ss", "-lunH"], text=True)
    if any(field.endswith(":8090") for line in listeners.splitlines() for field in line.split()):
        raise RuntimeError("UDP 8090 is occupied; no process will be interrupted")
    sys.path.insert(0, str(repo))
    from scripts.probes.capture_rm65_grasp_snapshot import assert_arm_connections_idle

    assert_arm_connections_idle(ips)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(repo), "diff", "HEAD", "--quiet", "--",
                    "hardware/rm65/realtime_feedback.py", "hardware/rm65/utils/scripts/rm65_udp_actuator_trace.py",
                    "hardware/rm65/Robotic_Arm", "scripts/probes/capture_rm65_grasp_snapshot.py"], check=True)
    print(json.dumps({"preflight": "passed", "repo_commit": head, "controllers": ips,
                      "route_sources": local_ips, "output": str(output), "apply": args.apply}), flush=True)
    if not args.apply:
        return 0

    def interrupted(signum, _frame):
        raise InterruptedError(f"UDP coordinate update interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    print(json.dumps(update_force_coordinates(ips, local_ips, output)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
