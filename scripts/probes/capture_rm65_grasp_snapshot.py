"""Read-only device snapshot. No runtime backend, HOLD, STOP or camera reset.

Run through remote_rm65_grasp_snapshot.sh after explicit device-read approval.
SDK construction reads robot info; cleanup deletes only this process's handles.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROLES = {"front": "main", "left_wrist": "wrist_left", "right_wrist": "wrist_right"}


def readonly_camera_sources(config_path):
    from teleop_runtime.real_teleop.camera_runtime import apply_real_teleop_camera_startup_defaults
    from teleop_runtime.sources.camera import RealSenseRGBCameraConfig, RealSenseRGBSource, load_realsense_camera_config_map

    configs = load_realsense_camera_config_map(Path(config_path))
    for key in ROLES:
        config = configs[key]
        if not isinstance(config, RealSenseRGBCameraConfig):
            raise ValueError(f"{key}: read-only snapshot requires a RealSense config")
        if not config.enable_depth or not config.align_depth_to_color:
            raise ValueError(f"{key}: snapshot requires aligned RGB-D; RGB-only fallback is not allowed")
        # Reuse formal collection defaults without its optional device writes.
        # Preserve the configured streams, serials and startup priorities.
        apply_real_teleop_camera_startup_defaults({key: config}, color_format=config.color_format)
        config.startup_hardware_reset = False
        config.startup_hardware_reset_on_failure = False
        config.startup_hardware_reset_on_warmup_failure = False
        config.auto_exposure_priority = None
    return {key: RealSenseRGBSource(config) for key, config in configs.items() if key in ROLES}


class CameraProgress:
    """Persist canonical startup progress before any potentially blocked call."""

    def __init__(self, stage):
        self.stage = stage

    def begin(self, key, **details):
        self.stage("camera_start_begin", camera=key, **details)

    def update(self, key, **details):
        self.stage("camera_start_update", camera=key, **details)

    def succeed(self, key, **details):
        self.stage("camera_start_ready", camera=key, **details)

    def fail(self, key, error, **details):
        self.stage("camera_start_failed", camera=key, error=error, **details)


def next_frames(sources, previous, *, after, timeout_sec=1.5):
    """Read immutable correlated frames, never drain or poll native pipelines."""
    from teleop_runtime.real_teleop.camera_runtime import CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC

    deadline = time.monotonic() + timeout_sec
    while True:
        frames = {key: source.get_latest_frame() for key, source in sources.items()}
        now = time.time()
        missing = []
        for key, frame in frames.items():
            if frame is None:
                missing.append(key)
                continue
            received = float(frame.source_timestamp)
            if not np.isfinite(received) or received > now:
                raise RuntimeError(f"{key}: invalid camera receive timestamp")
            identity = (frame.metadata["source_generation"], frame.frame_index)
            if key in previous and identity[0] != previous[key][0]:
                raise RuntimeError(f"{key}: camera generation changed during snapshot")
            if (received < after or identity == previous.get(key)
                    or now - received > CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC):
                missing.append(key)
        if not missing:
            return frames
        if time.monotonic() >= deadline:
            raise TimeoutError(f"No new fresh RGB-D frame within {timeout_sec:.3f}s: {missing}")
        time.sleep(.01)


def frame_payload(source, frame, *, index):
    info = source.device_info()  # Cached source state; no native SDK query.
    rgb = np.asarray(frame.image)
    space = frame.metadata["color_space"]
    if space == "bgr":
        rgb = rgb[..., ::-1]
    elif space != "rgb":
        raise ValueError(f"Unsupported frame color space: {space}")
    depth = frame.depth_image
    scale = frame.depth_metadata.get("depth_scale_m_per_unit")
    if (depth is None or np.shape(depth) != rgb.shape[:2]
            or frame.depth_metadata.get("aligned_to") != "color"
            or scale is None or not np.isfinite(scale) or scale <= 0):
        raise RuntimeError(f"{source.camera_key}: missing aligned metric depth")
    intr = info["intrinsics"]
    return rgb.copy(), np.asarray(depth, dtype=np.float32) * scale, {
        # Keep receiver time, availability and device time distinct. In
        # particular, reading the cache now does not make an old frame fresh.
        "received_at": float(frame.source_timestamp),
        "available_at": frame.available_timestamp,
        "device_timestamp_ms": frame.metadata["camera_timestamp_sec"] * 1000,
        "timestamp_domain": frame.metadata["timestamp_domain"],
        "frame_number": frame.metadata["color_frame_number"],
        "source_frame_index": frame.frame_index,
        "source_generation": frame.metadata["source_generation"],
        "depth_metadata": dict(frame.depth_metadata),
        "intrinsics": {"fx": intr["fx"], "fy": intr["fy"], "cx": intr["ppx"], "cy": intr["ppy"]},
        "index": index,
    }


def assert_arm_connections_idle(arm_ips: dict[str, str]) -> None:
    connections = subprocess.run(
        ["ss", "-tnH", "state", "established"], capture_output=True, text=True, check=True,
    ).stdout
    for line in connections.splitlines():
        if any(f"{ip}:8080" in line for ip in arm_ips.values()):
            raise RuntimeError(f"An existing RM65 connection is present: {line}")


def assert_devices_idle(arm_ips: dict[str, str]) -> None:
    assert_arm_connections_idle(arm_ips)
    devices = sorted(str(path) for path in Path("/dev").glob("video*"))
    if devices:
        holders = subprocess.run(["fuser", *devices], capture_output=True, text=True)
        if holders.returncode == 0 or holders.stdout.strip():
            raise RuntimeError(f"Camera devices are occupied: {holders.stdout.strip()}")
        if holders.returncode != 1:
            raise RuntimeError(f"Cannot establish camera ownership: {holders.stderr.strip()}")


def read_arm(arm) -> dict:
    started = time.time()
    status, joints_deg, pose, arm_error, system_error = arm.Get_Current_Arm_State(retry=0)
    finished = time.time()
    qpos = np.deg2rad(joints_deg)
    if status or arm_error or system_error or qpos.shape != (6,) or not np.isfinite(qpos).all():
        raise RuntimeError(f"Invalid read-only arm state: status={status}, arm_error={arm_error}, system_error={system_error}")
    return {"started_at": started, "finished_at": finished,
            "qpos_rad": qpos.tolist(), "sdk_pose": list(pose)}


def stationarity(samples: list[dict], started: float, finished: float) -> dict:
    result = {}
    for side in ("left", "right"):
        states = [row[side] for row in samples]
        if len(states) < 3 or states[0]["finished_at"] > started or states[-1]["started_at"] < finished:
            raise RuntimeError(f"{side}: qpos history does not bracket the capture")
        times = np.array([s["started_at"] for s in states])
        gaps = np.diff(times)
        spans = np.ptp([s["qpos_rad"] for s in states], axis=0)
        max_read = max(s["finished_at"] - s["started_at"] for s in states)
        if gaps.max() > .1 or max_read > .1 or spans.max() > .002:
            raise RuntimeError(f"{side}: stationary evidence failed; gap={gaps.max():.3f}s read={max_read:.3f}s joint_span={spans.max():.6f}rad")
        result[side] = {"max_gap_sec": float(gaps.max()), "max_read_sec": max_read,
                        "joint_span_rad": spans.tolist(), "sample_count": len(states)}
    return result


def capture(args, *, arm_factory) -> dict:
    from PIL import Image
    from teleop_runtime.real_teleop import camera_runtime
    from teleop_runtime.deploy.camera_runtime import run_camera_startup_self_check

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    ips = json.loads(Path(args.arm_ip_assignment).read_text())["arm_ip_by_side"]
    report = {"schema": "agenticrobot.readonly_rgbd_qpos.v1", "status": "failed",
              "motion_commands": [], "camera_reset": False, "arm_ips": ips,
              "pid": os.getpid(), "camera_lifecycle": "real_teleop",
              "cleanup_complete": False, "data_captured": False,
              "views": {}, "qpos_samples": [], "cleanup_errors": [], "persistence_errors": []}
    arms, sources, frames, errors = {}, {}, {}, []
    stopped = threading.Event()
    sampler = None

    def save_report():
        temporary = output / "snapshot.json.pending"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "snapshot.json")

    def stage(name: str, **details) -> None:
        # Flush before native calls: Python's finally/timeout cannot complete
        # while a kernel USB call remains in uninterruptible sleep.
        event = {"stage": name, "wall_time_sec": time.time(), **details}
        with (output / "capture_stages.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps(event, allow_nan=False), flush=True)

    def sample() -> None:
        while not stopped.is_set():
            cycle = time.monotonic()
            try:
                report["qpos_samples"].append({side: read_arm(arm) for side, arm in arms.items()})
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                return
            stopped.wait(max(0, .05 - (time.monotonic() - cycle)))

    try:
        save_report()
        stage("check_device_ownership")
        assert_devices_idle(ips)
        sources = readonly_camera_sources(args.camera_config)
        for key, source in sources.items():
            report["views"][ROLES[key]] = {"serial": source.config.serial, "status": "unavailable", "frames": []}
        save_report()
        stage("start_cameras", lifecycle="real_teleop")
        camera_errors = camera_runtime._start_camera_sources_best_effort(
            sources, startup_progress=CameraProgress(stage),
        )
        for key, source in sources.items():
            record = report["views"][ROLES[key]]
            record["device_info"] = source.device_info()
            if key in camera_errors:
                record["error"] = camera_errors[key]
            else:
                record["status"] = "ready"
        save_report()
        # The canonical batch reports exception types in its error strings.
        # Honor its terminal TimeoutError even if the native call returned and
        # the source has already transitioned back to stopped.
        startup_timed_out = any("TimeoutError:" in detail for detail in camera_errors.values())
        source_faulted = any(record["device_info"]["lifecycle_state"] == "faulted" for record in report["views"].values())
        if startup_timed_out or source_faulted:
            raise RuntimeError(f"Camera lifecycle fault or startup timeout; no arm connection attempted: {camera_errors}")
        ready = {key: source for key, source in sources.items() if key not in camera_errors}
        if not ready:
            raise RuntimeError(f"No usable camera; no reset attempted: {camera_errors}")
        stage("camera_freshness_check", cameras=list(ready))
        run_camera_startup_self_check(ready)
        # Only admit arm reads after cameras are healthy; a native camera
        # startup stall must not retain RM65 connections. Recheck arm ownership
        # because camera negotiation may have taken several seconds.
        assert_arm_connections_idle(ips)
        for side in ("left", "right"):
            stage("connect_arm_readonly", side=side)
            arms[side] = arm_factory(ips[side])
            stage("read_initial_qpos", side=side)
            initial = read_arm(arms[side])
            stage("initial_qpos_ready", side=side, **initial)
        frames = {ROLES[key]: [] for key in ready}
        sampler = threading.Thread(target=sample, name="readonly-qpos", daemon=True)
        sampler.start()
        time.sleep(.2)
        capture_start = time.time()
        previous = {}
        for index in range(5):
            if errors:
                raise RuntimeError(errors[0])
            batch = next_frames(ready, previous, after=capture_start)
            for key, frame in batch.items():
                frames[ROLES[key]].append(frame_payload(ready[key], frame, index=index))
                previous[key] = (frame.metadata["source_generation"], frame.frame_index)
            time.sleep(.07)
        capture_end = time.time()
        time.sleep(.15)
        stopped.set()
        sampler.join(timeout=2)
        if sampler.is_alive() or errors:
            raise RuntimeError(f"Read-only qpos sampler did not complete: {errors}")
        report["stationarity"] = stationarity(report["qpos_samples"], capture_start, capture_end)
        report["capture_started_at"], report["capture_finished_at"] = capture_start, capture_end
        report["data_captured"] = True
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        stopped.set()
        if sampler is not None:
            sampler.join(timeout=2)
        # Preserve data and the primary error before SDK teardown. A blocked
        # close or an outer SIGKILL must not erase the acquisition evidence.
        try:
            for role, observations in frames.items():
                for rgb, depth, metadata in observations:
                    stem = f"{role}_{metadata['index']:02d}"
                    np.savez_compressed(output / f"{stem}.npz", rgb=rgb, depth_m=depth)
                    Image.fromarray(rgb).save(output / f"{stem}.jpg")
                    report["views"][role]["frames"].append({**metadata, "file": f"{stem}.npz"})
            report["missing_roles"] = [role for role in ROLES.values() if not report["views"].get(role, {}).get("frames")]
            report["all_views_captured"] = all(len(report["views"].get(role, {}).get("frames", [])) == 5 for role in ROLES.values())
            save_report()
        except Exception as exc:
            # Artifact failure is fatal, but must not bypass device teardown.
            report["persistence_errors"].append(f"{type(exc).__name__}: {exc}")
        # Never close an SDK handle concurrently with an inflight read. The
        # bounded outer process owns and releases these sockets on exit.
        if sampler is None or not sampler.is_alive():
            for side, arm in arms.items():
                try:
                    try:
                        stage("close_arm_readonly", side=side)
                    finally:
                        code = arm.Arm_Socket_Close()
                    if code:
                        report["cleanup_errors"].append(f"{side}: close returned {code}")
                except Exception as exc:
                    report["cleanup_errors"].append(f"{side}: {exc}")
        else:
            report["cleanup_errors"].append("SDK read still inflight; process exit releases sockets")
        try:
            try:
                stage("close_cameras", lifecycle="real_teleop")
            finally:
                camera_runtime._close_camera_sources(sources)
        except Exception as exc:
            report["cleanup_errors"].append(f"cameras: {type(exc).__name__}: {exc}")
        report["cleanup_complete"] = not report["cleanup_errors"]
        if report["data_captured"] and report["cleanup_complete"] and not report["persistence_errors"] and "error" not in report:
            report["status"] = "captured"
        save_report()
    print(json.dumps({"status": report["status"], "output": str(output), "error": report.get("error"),
                      "roles": list(frames), "qpos_samples": len(report["qpos_samples"]),
                      "cleanup_errors": report["cleanup_errors"],
                      "persistence_errors": report["persistence_errors"]}), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--camera-config", required=True)
    parser.add_argument("--arm-ip-assignment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    from hardware.rm65.rm_api2_adapter import Arm, RM65

    def interrupted(signum, frame):
        # Not a recoverable camera startup error: the serial lifecycle must
        # unwind immediately instead of treating cancellation as a retry.
        raise KeyboardInterrupt(f"Read-only acquisition interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = capture(args, arm_factory=lambda ip: Arm(RM65, ip))
    if result.get("error", {}).get("type") == "KeyboardInterrupt":
        return 130
    return 0 if result["status"] == "captured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
