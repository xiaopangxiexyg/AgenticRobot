"""Read-only device snapshot. No runtime backend, HOLD, STOP or camera reset.

Run through remote_rm65_grasp_snapshot.sh after explicit device-read approval.
SDK construction reads robot info; cleanup deletes only this process's handles.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROLES = {"front": "main", "left_wrist": "wrist_left", "right_wrist": "wrist_right"}
SCENE_OBJECTS = {
    "1": ("blue_cube", "蓝色方块"),
    "2": ("blue_small_bottle", "蓝色小瓶子"),
    "3": ("red_cube", "红色方块"),
    "4": ("black_cup", "黑色水杯"),
}
CONTEXT_FILES = ("camera_config", "arm_ip_assignment", "camera_registry",
                 "camera_mount_urdf", "left_tool_profile", "right_tool_profile")


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
        config.retain_unaligned_depth = True
        # This capture owns one long-lived pipeline per device. Preserve fault
        # diagnostics and admission checks, but never recover by reopening it.
        config.startup_reset = False
        config.startup_software_retry_attempts = 0
        config.restart_on_stale_timeout = False
        config.restart_on_runtime_error = False
        config.restart_on_black_frames = False
        config.restart_on_incomplete_frameset = False
        config.metadata["perception_single_start"] = True
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



def diagnostic_frame_payload(source, frame, *, index):
    """Persist native Z16 beside the unchanged aligned metric input."""
    rgb, depth, metadata = frame_payload(source, frame, index=index)
    raw = frame.unaligned_depth_image
    raw_meta = frame.unaligned_depth_metadata
    if (raw is None or raw.dtype != np.uint16 or raw.ndim != 2
            or raw_meta.get("aligned_to") != "none"
            or raw_meta.get("frame_number") != frame.depth_metadata.get("frame_number")
            or raw_meta.get("camera_timestamp_sec") != frame.depth_metadata.get("camera_timestamp_sec")
            or raw_meta.get("depth_scale_m_per_unit") != frame.depth_metadata.get("depth_scale_m_per_unit")):
        raise RuntimeError(f"{source.camera_key}: missing or mismatched unaligned depth evidence")
    intr = raw_meta.get("intrinsics", {})
    if (raw.shape != (intr.get("height"), intr.get("width"))
            or not all(key in raw_meta for key in ("color_intrinsics", "depth_to_color_extrinsics"))):
        raise RuntimeError(f"{source.camera_key}: unaligned depth calibration is incomplete")
    metadata["unaligned_depth_metadata"] = copy.deepcopy(raw_meta)
    return {"rgb": rgb, "depth_m": depth, "unaligned_depth_z16": raw}, metadata


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


def save_json(path: Path, value: dict) -> None:
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


class ReadOnlySnapshotSession:
    """One canonical camera/SDK owner, shared by one-shot and interactive capture."""

    def __init__(self, args, *, arm_factory):
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.camera_config = args.camera_config
        self.ips = json.loads(Path(args.arm_ip_assignment).read_text())["arm_ip_by_side"]
        self.arm_factory = arm_factory
        self.arms, self.sources, self.ready = {}, {}, {}
        self.samples = deque(maxlen=256)
        self.sample_lock = threading.Lock()
        self.errors = []
        self.stopped = threading.Event()
        self.sampler = None
        self.closed = False
        self.info = {"status": "starting", "cleanup_complete": False,
            "motion_commands": [], "camera_reset": False, "views": {},
            "cleanup_errors": [], "pid": os.getpid(), "camera_lifecycle": "real_teleop"}
        self.save()
        # Acquire one watchdog before opening any device; teardown reuses it.
        self._operation_condition = threading.Condition()
        self._operation = None
        self._watchdog_stopped = False
        self._watchdog = threading.Thread(
            target=self._watch_operations, name="readonly-operation-watchdog", daemon=True)
        self._watchdog.start()

    def save(self):
        save_json(self.output / "lifecycle.json", self.info)

    def stage(self, name, **details):
        row = {"stage": name, "wall_time_sec": time.time(), **details}
        with (self.output / "capture_stages.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)

    def _watch_operations(self):
        with self._operation_condition:
            while not self._watchdog_stopped:
                if self._operation is None:
                    self._operation_condition.wait()
                    continue
                _token, name, timeout_sec, deadline, abort = self._operation
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self._operation_condition.wait(timeout=remaining)
                    continue
                self._operation = None
                try:
                    self.info.update(status="operation_timeout", cleanup_complete=False,
                        error={"type": "TimeoutError", "message": f"{name} exceeded {timeout_sec:.1f}s"})
                    self.stage("operation_timeout", operation=name, timeout_sec=timeout_sec)
                    self.save()
                finally:
                    abort(124)

    def _finish_watchdog_when_idle(self):
        with self._operation_condition:
            if self._operation is not None:
                return
            self._watchdog_stopped = True
            self._operation_condition.notify_all()
        self._watchdog.join(timeout=.5)

    @contextmanager
    def bounded_operation(self, name, *, timeout_sec=60., abort=os._exit):
        # Human input has no deadline; every native operation retains its budget.
        token = object()
        with self._operation_condition:
            if self._watchdog_stopped or not self._watchdog.is_alive():
                raise RuntimeError("Read-only operation watchdog is unavailable")
            if self._operation is not None:
                raise RuntimeError("Read-only native operations cannot overlap")
            self._operation = (token, name, timeout_sec, time.monotonic() + timeout_sec, abort)
            self._operation_condition.notify_all()
        try:
            yield
        finally:
            with self._operation_condition:
                if self._operation is not None and self._operation[0] is token:
                    self._operation = None
                    self._operation_condition.notify_all()
            if self.closed:
                self._finish_watchdog_when_idle()
    def start(self, *, cameras_only=False, arms_only=False):
        if cameras_only and arms_only:
            raise ValueError("Camera-only and arm-only checks are mutually exclusive")
        if arms_only:
            self.stage("check_arm_ownership", cameras_enabled=False)
            self._start_arm_readers()
            return
        from teleop_runtime.deploy.camera_runtime import run_camera_startup_self_check
        from teleop_runtime.real_teleop import camera_runtime

        self.stage("check_device_ownership")
        assert_devices_idle(self.ips)
        self.sources = readonly_camera_sources(self.camera_config)
        for key, source in self.sources.items():
            self.info["views"][ROLES[key]] = {"serial": source.config.serial, "status": "unavailable", "frames": []}
        self.save()
        self.stage("start_cameras", lifecycle="real_teleop")
        camera_errors = camera_runtime._start_camera_sources_best_effort(
            self.sources, startup_progress=CameraProgress(self.stage))
        for key, source in self.sources.items():
            record = self.info["views"][ROLES[key]]
            record["device_info"] = source.device_info()
            if key in camera_errors:
                record["error"] = camera_errors[key]
            else:
                record["status"] = "ready"
        self.save()
        startup_timed_out = any("TimeoutError:" in value for value in camera_errors.values())
        source_faulted = any(record["device_info"]["lifecycle_state"] == "faulted"
            for record in self.info["views"].values())
        if startup_timed_out or source_faulted:
            raise RuntimeError(f"Camera lifecycle fault or startup timeout; no arm connection attempted: {camera_errors}")
        self.ready = {key: source for key, source in self.sources.items() if key not in camera_errors}
        if not self.ready:
            raise RuntimeError(f"No usable camera; no reset attempted: {camera_errors}")
        self.stage("camera_freshness_check", cameras=list(self.ready))
        run_camera_startup_self_check(self.ready)
        if cameras_only:
            self.info["status"] = "ready"
            self.save()
            self.stage("camera_check_ready", cameras=list(self.ready))
            return
        self._start_arm_readers()

    def _start_arm_readers(self):
        assert_arm_connections_idle(self.ips)
        for side in ("left", "right"):
            self.stage("connect_arm_readonly", side=side, controller_ip=self.ips[side])
            started = time.monotonic()
            arm = self.arm_factory(self.ips[side])
            elapsed = time.monotonic() - started
            self.arms[side] = arm
            state = arm.Arm_Socket_State()
            diagnostic = {"side": side, "controller_ip": self.ips[side],
                "port": getattr(arm, "port", 8080), "socket_state": state,
                "sdk_handle_id": getattr(arm, "nSocket", None),
                "sdk_version": arm.API_Version(),
                "connect_elapsed_sec": elapsed}
            self.info.setdefault("arm_connections", {})[side] = diagnostic
            self.stage("arm_connection_result", **diagnostic)
            if state != 0:
                raise RuntimeError(f"Read-only RM65 connection failed before qpos: {diagnostic}")
            self.stage("read_initial_qpos", side=side)
            initial = read_arm(arm)
            self.info.setdefault("initial_qpos", {})[side] = initial
            self.stage("initial_qpos_ready", side=side, **initial)
        self.sampler = threading.Thread(target=self._sample, name="readonly-qpos", daemon=True)
        self.sampler.start()
        time.sleep(.2)
        self.assert_healthy()
        self.info["status"] = "ready"
        self.save()
        self.stage("session_ready", cameras=list(self.ready),
            missing_roles=[role for role, view in self.info["views"].items() if view["status"] != "ready"])

    def _sample(self):
        while not self.stopped.is_set():
            cycle = time.monotonic()
            try:
                row = {side: read_arm(arm) for side, arm in self.arms.items()}
                with self.sample_lock:
                    self.samples.append(row)
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
                self.stage("qpos_sampler_failed", error=self.errors[-1])
                return
            self.stopped.wait(max(0, .05 - (time.monotonic() - cycle)))

    def history(self):
        with self.sample_lock:
            return list(self.samples)

    def assert_healthy(self):
        if self.closed or self.errors or self.sampler is None or not self.sampler.is_alive():
            raise RuntimeError(f"Read-only qpos sampler is unavailable: {self.errors}")
        rows = self.history()
        if not rows:
            raise RuntimeError("Read-only sampler has no measured state")
        age = time.time() - min(row["finished_at"] for row in rows[-1].values())
        if not 0 <= age <= .2:
            raise RuntimeError(f"Read-only measured qpos is stale or has invalid time: {age:.3f}s")
        for key, source in self.ready.items():
            info = source.device_info()
            if not info["running"] or info["lifecycle_state"] != "running":
                raise RuntimeError(f"{key}: camera is no longer running: {info}")

    def capture_state(self, args, *, create_output=True):
        from PIL import Image

        output = Path(args.output)
        if create_output:
            output.mkdir(parents=True, exist_ok=False)
        held = create_output
        report = {"schema": "agenticrobot.readonly_rgbd_qpos.v2" if held else "agenticrobot.readonly_rgbd_qpos.v1",
            "status": "failed", "motion_commands": [], "camera_reset": False, "arm_ips": self.ips,
            "pid": os.getpid(), "camera_lifecycle": "real_teleop", "cleanup_complete": False,
            "data_captured": False, "views": copy.deepcopy(self.info["views"]),
            "qpos_samples": [], "cleanup_errors": [], "persistence_errors": []}
        if held:
            report.update(resource_lifecycle="session",
                session_lifecycle=os.path.relpath(self.output / "lifecycle.json", output))
        if getattr(args, "scene_object", None):
            report["scene_objects"] = [{"id": key, "name": name, "source": "operator_declared"}
                for key in dict.fromkeys(args.scene_object)
                for known, name in SCENE_OBJECTS.values() if known == key]
            report["scene_note"] = getattr(args, "scene_note", "")
        save_json(output / "snapshot.json", report)
        self.stage("capture_state_begin", output=str(output))
        started = time.time()
        finished = started
        try:
            self.assert_healthy()
            # Pre-bracket the acquisition from the continuous, bounded history.
            time.sleep(.05)
            started = time.time()
            previous = {}
            for index in range(5):
                self.assert_healthy()
                frames = next_frames(self.ready, previous, after=started)
                for key, frame in frames.items():
                    arrays, metadata = diagnostic_frame_payload(self.ready[key], frame, index=index)
                    stem = f"{ROLES[key]}_{index:02d}"
                    # Persist each observed frame before the next native wait.
                    try:
                        np.savez_compressed(output / f"{stem}.npz", **arrays)
                        Image.fromarray(arrays["rgb"]).save(output / f"{stem}.jpg")
                    except OSError as exc:
                        report["persistence_errors"].append(f"{type(exc).__name__}: {exc}")
                        raise
                    report["views"][ROLES[key]]["frames"].append({**metadata, "file": f"{stem}.npz"})
                    previous[key] = (frame.metadata["source_generation"], frame.frame_index)
                finished = time.time()
                save_json(output / "snapshot.json", report)
                time.sleep(.07)
            finished = time.time()
            time.sleep(.15)
            self.assert_healthy()
            report["qpos_samples"] = self._capture_history(started, finished)
            report["stationarity"] = stationarity(report["qpos_samples"], started, finished)
            report["data_captured"] = True
            if held:
                report["status"] = "captured"
        except (Exception, KeyboardInterrupt) as exc:
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            report["capture_started_at"], report["capture_finished_at"] = started, finished
            if not report["qpos_samples"]:
                report["qpos_samples"] = self._capture_history(started, finished)
            report["missing_roles"] = [role for role in ROLES.values() if not report["views"].get(role, {}).get("frames")]
            report["all_views_captured"] = all(len(report["views"].get(role, {}).get("frames", [])) == 5 for role in ROLES.values())
            save_json(output / "snapshot.json", report)
            self.stage("capture_state_saved", output=str(output), status=report["status"],
                data_captured=report["data_captured"], all_views_captured=report["all_views_captured"],
                error=report.get("error"))
        return report

    def _capture_history(self, started, finished):
        return [row for row in self.history()
            if max(item["finished_at"] for item in row.values()) >= started - .2
            and min(item["started_at"] for item in row.values()) <= finished + .15]

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stopped.set()
        self.info["status"] = "closing"
        deferred_error = None
        try:
            try:
                self.save()
            except BaseException as exc:
                # Re-raise after independent owned cleanup, including cancellation.
                deferred_error = exc
                self.info["cleanup_errors"].append(f"lifecycle record: {type(exc).__name__}: {exc}")
            sampler_finished = self.sampler is None
            try:
                if self.sampler is not None:
                    self.sampler.join(timeout=2)
                    sampler_finished = not self.sampler.is_alive()
            except BaseException as exc:
                if deferred_error is None or (
                    not isinstance(exc, Exception) and isinstance(deferred_error, Exception)
                ):
                    deferred_error = exc
                self.info["cleanup_errors"].append(f"sampler join: {type(exc).__name__}: {exc}")
            if sampler_finished:
                for side, arm in self.arms.items():
                    try:
                        try:
                            self.stage("close_arm_readonly", side=side)
                        finally:
                            code = arm.Arm_Socket_Close()
                        if code:
                            self.info["cleanup_errors"].append(f"{side}: close returned {code}")
                    except BaseException as exc:
                        self.info["cleanup_errors"].append(f"{side}: {type(exc).__name__}: {exc}")
                        if not isinstance(exc, Exception) and (
                            deferred_error is None or isinstance(deferred_error, Exception)
                        ):
                            deferred_error = exc
            else:
                self.info["cleanup_errors"].append("SDK read still inflight or unconfirmed; process exit releases sockets")
            try:
                if self.sources:
                    from teleop_runtime.real_teleop import camera_runtime

                    try:
                        self.stage("close_cameras", lifecycle="real_teleop")
                    finally:
                        camera_runtime._close_camera_sources(self.sources)
            except BaseException as exc:
                self.info["cleanup_errors"].append(f"cameras: {type(exc).__name__}: {exc}")
                if not isinstance(exc, Exception) and (
                    deferred_error is None or isinstance(deferred_error, Exception)
                ):
                    deferred_error = exc
            self.info["cleanup_complete"] = not self.info["cleanup_errors"]
            self.info["status"] = "closed" if self.info["cleanup_complete"] else "cleanup_failed"
            try:
                self.save()
            except BaseException as exc:
                self.info.update(cleanup_complete=False, status="cleanup_failed")
                if deferred_error is None or (
                    not isinstance(exc, Exception) and isinstance(deferred_error, Exception)
                ):
                    deferred_error = exc
            if deferred_error is not None:
                raise deferred_error
        finally:
            # An active close phase owns the deadline until its context exits.
            self._finish_watchdog_when_idle()

def capture(args, *, arm_factory) -> dict:
    session = ReadOnlySnapshotSession(args, arm_factory=arm_factory)
    report = {"schema": "agenticrobot.readonly_rgbd_qpos.v1", "status": "failed",
        "data_captured": False, "cleanup_complete": False, "motion_commands": [], "camera_reset": False,
        "views": {}, "qpos_samples": [], "persistence_errors": []}
    try:
        with session.bounded_operation("startup"):
            session.start()
        with session.bounded_operation("capture"):
            report = session.capture_state(args, create_output=False)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(error={"type": type(exc).__name__, "message": str(exc)},
            views=copy.deepcopy(session.info["views"]))
    finally:
        # Keep the original one-shot evidence-before-teardown contract.
        try:
            save_json(session.output / "snapshot.json", report)
        except OSError as exc:
            report["persistence_errors"].append(f"{type(exc).__name__}: {exc}")
        finally:
            with session.bounded_operation("close"):
                session.close()
        report["cleanup_errors"] = session.info["cleanup_errors"]
        report["cleanup_complete"] = session.info["cleanup_complete"]
        if report["data_captured"] and report["cleanup_complete"] and not report["persistence_errors"] and "error" not in report:
            report["status"] = "captured"
        save_json(session.output / "snapshot.json", report)
    print(json.dumps({"status": report["status"], "output": str(session.output),
        "error": report.get("error"), "cleanup_errors": report["cleanup_errors"]}), flush=True)
    return report


def check_arm_readonly(args, *, arm_factory) -> dict:
    """Use the existing reader owner without constructing camera sources."""
    session = ReadOnlySnapshotSession(args, arm_factory=arm_factory)
    report = {"check": "arm_readonly", "status": "failed", "cameras_opened": False,
        "motion_commands": [], "camera_reset": False, "qpos_samples": []}
    try:
        with session.bounded_operation("arm_startup"):
            session.start(arms_only=True)
        with session.sample_lock:
            report["qpos_samples"] = copy.deepcopy(list(session.samples))
        report["status"] = "ready"
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        session.stage("arm_read_check_failed", **report["error"])
    finally:
        with session.bounded_operation("close"):
            session.close()
        report.update(arm_connections=session.info.get("arm_connections", {}),
            initial_qpos=session.info.get("initial_qpos", {}),
            reader_errors=list(session.errors),
            cleanup_complete=session.info["cleanup_complete"],
            cleanup_errors=session.info["cleanup_errors"])
        if report["reader_errors"]:
            report["status"] = "failed"
            report.setdefault("error", {"type": "RuntimeError",
                "message": "Read-only arm sampler failed: " + "; ".join(report["reader_errors"])})
        if not report["cleanup_complete"]:
            report["status"] = "failed"
        save_json(session.output / "arm_read_check.json", report)
    print(json.dumps({"status": report["status"], "output": str(session.output),
        "error": report.get("error"), "cleanup_errors": report["cleanup_errors"]}), flush=True)
    return report


def camera_sensor_diagnostics(serials) -> dict:
    """Read stream capabilities and current settings; never open or set sensors."""
    import pyrealsense2 as rs

    result = {}
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        if serial not in serials:
            continue
        record = {"name": device.get_info(rs.camera_info.name), "sensors": []}
        for sensor in device.query_sensors():
            item = {"name": sensor.get_info(rs.camera_info.name), "options": {}, "profiles": []}
            for name in ("inter_cam_sync_mode", "enable_auto_exposure", "exposure",
                         "auto_exposure_priority", "global_time_enabled", "frames_queue_size"):
                option = getattr(rs.option, name, None)
                if option is not None and sensor.supports(option):
                    item["options"][name] = sensor.get_option(option)
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() not in (rs.stream.color, rs.stream.depth):
                    continue
                video = profile.as_video_stream_profile()
                item["profiles"].append({"stream": str(profile.stream_type()),
                    "width": video.width(), "height": video.height(), "fps": profile.fps(),
                    "format": str(profile.format())})
            record["sensors"].append(item)
        result[serial] = record
    return result


def check_camera_startup(args) -> dict:
    """Exercise the canonical lifecycle without importing or connecting an arm SDK."""
    from PIL import Image
    from teleop_runtime.sources.camera import load_realsense_camera_config_map

    def forbidden_arm_factory(ip):
        raise AssertionError("Camera startup check must not connect an arm")

    session = ReadOnlySnapshotSession(args, arm_factory=forbidden_arm_factory)
    keys = set(ROLES)
    report = {"schema": "agenticrobot.readonly_camera_startup.v1", "status": "failed",
        "arm_connected": False, "camera_reset": False, "views": {},
        "required_roles": list(ROLES.values()),
        "all_views_ready": False}
    try:
        with session.bounded_operation("startup"):
            sensors_only = getattr(args, "camera_startup_check", "streams") == "sensors"
            if sensors_only:
                assert_devices_idle(session.ips)
                session.stage("camera_sensor_diagnostics")
                configs = load_realsense_camera_config_map(Path(args.camera_config))
                report["sensors"] = camera_sensor_diagnostics({configs[key].serial for key in keys})
                save_json(session.output / "camera_sensors.json", report["sensors"])
                report["status"] = "observed"
            else:
                session.start(cameras_only=True)
        if not sensors_only:
            with session.bounded_operation("capture"):
                previous = {}
                for index in range(2):
                    frames = next_frames(session.ready, previous, after=time.time())
                    for key, frame in frames.items():
                        arrays, metadata = diagnostic_frame_payload(session.ready[key], frame, index=index)
                        role = ROLES[key]
                        np.savez_compressed(session.output / f"{role}_{index:02d}.npz", **arrays)
                        Image.fromarray(arrays["rgb"]).save(session.output / f"{role}_{index:02d}.jpg")
                        session.info["views"][role]["frames"].append(metadata)
                        previous[key] = (frame.metadata["source_generation"], frame.frame_index)
                    session.save()
                report["status"] = "ready" if set(session.ready) == set(keys) else "partial"
                report["all_views_ready"] = set(session.ready) == set(ROLES)
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        with session.bounded_operation("close"):
            session.close()
        report.update(views=session.info["views"], cleanup_complete=session.info["cleanup_complete"],
            cleanup_errors=session.info["cleanup_errors"])
        if not report["cleanup_complete"]:
            report["status"] = "failed"
            report["all_views_ready"] = False
        save_json(session.output / "camera_startup.json", report)
    print(json.dumps({"status": report["status"], "output": str(session.output),
        "error": report.get("error"), "cleanup_complete": report["cleanup_complete"]}, ensure_ascii=False), flush=True)
    return report


def check_inputs(args) -> dict:
    """Validate source/configuration/dependencies without opening any device."""
    repo = Path(args.repo).resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("Capture source has tracked changes; deploy a clean Git revision first")
    sys.path.insert(0, str(repo))
    required = CONTEXT_FILES if args.interactive else CONTEXT_FILES[:2]
    resolved = {}
    for name in required:
        value = getattr(args, name, None)
        if not value:
            raise ValueError(f"--{name.replace('_', '-')} is required for interactive collection")
        path = Path(value).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"{name} must be a file")
        resolved[name] = str(path)
    for name in ("PIL", "pyrealsense2"):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(f"Missing dependency: {name}; no devices opened")
    for name in ("ss", "fuser", "timeout"):
        if shutil.which(name) is None:
            raise RuntimeError(f"Missing executable in PATH: {name}; no devices opened")
    ips = json.loads(Path(resolved["arm_ip_assignment"]).read_text())["arm_ip_by_side"]
    if any(not isinstance(ips.get(side), str) or not ips[side] for side in ("left", "right")):
        raise ValueError("Arm assignment must name both controller IPs")
    # Construction holds only cached Python state. Do not call start()/query_devices().
    sources = readonly_camera_sources(resolved["camera_config"])
    from hardware.rm65.rm_api2_adapter import RM65, Arm
    del Arm, RM65  # Verify the SDK can load, without constructing a connection.
    return {"source_commit": commit, "python": sys.executable, "files": resolved,
            "cameras": {ROLES[key]: {"serial": source.config.serial,
                "width": source.config.width, "height": source.config.height,
                "fps": source.config.fps} for key, source in sources.items()},
            "hardware_connected": False}


def parse_scene_selection(answer: str, previous):
    if not answer.strip():
        if previous is None:
            raise ValueError("首次采集请至少输入一个物体编号，例如 1 或 1 4")
        return previous
    numbers, _, note = answer.partition("|")
    tokens = numbers.replace(",", " ").replace("，", " ").split()
    if not tokens or any(token not in SCENE_OBJECTS for token in tokens):
        raise ValueError("物体编号只允许 1、2、3、4；多个编号用空格分隔")
    return tuple(dict.fromkeys(SCENE_OBJECTS[token][0] for token in tokens)), note.strip()


def interactive_capture(args, *, context, arm_factory, input_fn=input) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config_dir = output / "context"
    config_dir.mkdir()
    frozen_files = {}
    for name, path in context["files"].items():
        target = config_dir / (name + Path(path).suffix)
        shutil.copyfile(path, target)
        frozen_files[name] = target
    metadata = {**context, "schema": "agenticrobot.readonly_grasp_collection.v2",
        "context_copies": {name: str(path.relative_to(output)) for name, path in frozen_files.items()},
        "object_catalog": [{"id": key, "name": name} for key, name in SCENE_OBJECTS.values()],
        "motion_commands": [], "operation_timeout_sec": 60, "resource_lifecycle": "session",
        "annotation_source": "operator_declared", "device_lifecycle": "devices/lifecycle.json",
        "status": "starting", "hardware_connected": False}
    save_json(output / "session.json", metadata)
    device_args = SimpleNamespace(output=output / "devices",
        camera_config=frozen_files["camera_config"], arm_ip_assignment=frozen_files["arm_ip_assignment"])
    session = ReadOnlySnapshotSession(device_args, arm_factory=arm_factory)
    previous, ordinal, code = None, 0, 0
    try:
        print(f"只读采集目录：{output}；正在启动一次设备会话。", flush=True)
        with session.bounded_operation("startup"):
            session.start()
        metadata.update(status="ready", hardware_connected=True)
        save_json(output / "session.json", metadata)
        print("不发送运动、HOLD、STOP、夹爪或相机 reset 指令。设备保持连接直到退出。", flush=True)
        missing = [role for role, view in session.info["views"].items() if view["status"] != "ready"]
        if missing:
            print(f"注意：缺少视角 {missing}；本会话只能采集部分视角，不是完整三视角数据。", flush=True)
        print("物体编号：1 蓝色方块；2 蓝色小瓶子；3 红色方块；4 黑色水杯。", flush=True)
        print("输入编号并回车采集，例如 1 4；可加 | 场景说明/实际尺寸。", flush=True)
        print("之后直接回车重复上一组标签；物体变化时重新输入编号；q 统一关闭设备并结束。", flush=True)
        while True:
            answer = input_fn("物体编号 / 回车采集 / q 结束: ").strip()
            if answer.lower() == "q":
                break
            try:
                objects, note = parse_scene_selection(answer, previous)
            except ValueError as exc:
                print(str(exc), flush=True)
                continue
            previous = objects, note
            ordinal += 1
            sample_dir = output / f"state_{ordinal:04d}"
            sample_dir.mkdir()
            save_json(sample_dir / "scene.json", {"scene_objects": objects,
                "scene_note": note, "source": "operator_declared", "triggered_at": time.time()})
            capture_args = SimpleNamespace(output=sample_dir / "capture",
                scene_object=objects, scene_note=note)
            print(f"保存新状态，不重启设备：{sample_dir}；请保持场景静止。", flush=True)
            with session.bounded_operation("capture"):
                report = session.capture_state(capture_args)
            # Explicitly link a held-resource capture to its eventual teardown,
            # rather than pretending each state disconnected the devices.
            record = {"state": sample_dir.name, "scene_objects": objects, "scene_note": note,
                "status": report["status"], "resource_lifecycle": "session",
                "all_views_captured": report["all_views_captured"],
                "missing_roles": report["missing_roles"],
                "snapshot": str((capture_args.output / "snapshot.json").relative_to(output))}
            with (output / "samples.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
            if report.get("error", {}).get("type") == "KeyboardInterrupt":
                code = 130
                break
            if report["status"] != "captured":
                print(f"本状态失败，诊断已保存：{report.get('error')}。", flush=True)
                # A moving scene can be re-sampled. A dead/stale device cannot
                # be restored by relabeling or by silently restarting it.
                session.assert_healthy()
                if report["persistence_errors"]:
                    code = 1
                    break
            else:
                print(f"已保存 {sample_dir.name}；三视角完整={record['all_views_captured']}。", flush=True)
    except (EOFError, KeyboardInterrupt):
        code = 130
        print("会话取消，正在统一关闭设备；已保存的数据不删除。", flush=True)
    except Exception as exc:
        code = 1
        session.info["error"] = {"type": type(exc).__name__, "message": str(exc)}
        session.save()
        session.stage("session_failed", **session.info["error"])
    finally:
        with session.bounded_operation("close"):
            session.close()
        metadata.update(cleanup_complete=session.info["cleanup_complete"],
            status="closed" if session.info["cleanup_complete"] else "cleanup_failed",
            sample_count=ordinal, exit_code=code,
            hardware_connected=False if session.info["cleanup_complete"] else None)
        if not session.info["cleanup_complete"]:
            code = 1
            metadata["exit_code"] = code
        save_json(output / "session.json", metadata)
        print(f"采集结束；状态数={ordinal}；设备清理完成={session.info['cleanup_complete']}；记录保存在：{output}", flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--camera-config", required=True)
    parser.add_argument("--arm-ip-assignment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interactive", action="store_true", help="Operator-labeled repeated read-only snapshots")
    parser.add_argument("--check-only", action="store_true", help="Validate paths/imports without opening devices")
    parser.add_argument("--arm-read-check", action="store_true",
        help="Read arm connection/state using the existing owner, without starting cameras")
    parser.add_argument("--arm-sdk-log-level", type=int, choices=(0, 1, 2, 3), default=3,
        help="Existing SDK verbosity: 0 debug, 1 info, 2 warning, 3 error (default)")
    parser.add_argument("--camera-startup-check", nargs="?", const="streams", choices=("streams", "sensors"),
        help="Bounded RGB-D check, or sensors to read settings only; no arm SDK import or connection")
    for name in CONTEXT_FILES[2:]:
        parser.add_argument("--" + name.replace("_", "-"))
    parser.add_argument("--scene-object", action="append", choices=[key for key, _ in SCENE_OBJECTS.values()])
    parser.add_argument("--scene-note", default="")
    args = parser.parse_args()
    if args.arm_read_check and (args.camera_startup_check or args.check_only or args.interactive):
        parser.error("--arm-read-check cannot be combined with camera startup, check-only or interactive modes")
    if args.camera_startup_check and (args.check_only or args.interactive):
        parser.error("--camera-startup-check cannot be combined with --check-only or --interactive")
    if args.check_only:
        print(json.dumps(check_inputs(args), ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.interactive:
        if not sys.stdin.isatty():
            parser.error("Interactive collection needs terminal input; do not pipe a heredoc into Python")
        context = check_inputs(args)
    sys.path.insert(0, args.repo)
    def interrupted(signum, frame):
        # Not a recoverable camera startup error: the serial lifecycle must
        # unwind immediately instead of treating cancellation as a retry.
        raise KeyboardInterrupt(f"Read-only acquisition interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.camera_startup_check:
        result = check_camera_startup(args)
        return 0 if result["status"] in ("ready", "observed") else 1
    from hardware.rm65.rm_api2_adapter import Arm, RM65
    arm_factory = lambda ip: Arm(RM65, ip, log_level=args.arm_sdk_log_level)
    if args.arm_read_check:
        result = check_arm_readonly(args, arm_factory=arm_factory)
        if result.get("error", {}).get("type") == "KeyboardInterrupt":
            return 130
        return 0 if result["status"] == "ready" else 1
    if args.interactive:
        return interactive_capture(args, context=context, arm_factory=arm_factory)
    result = capture(args, arm_factory=arm_factory)
    if result.get("error", {}).get("type") == "KeyboardInterrupt":
        return 130
    return 0 if result["status"] == "captured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
