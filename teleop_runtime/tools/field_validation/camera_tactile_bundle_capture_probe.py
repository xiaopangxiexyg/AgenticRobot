"""No-motion live camera/tactile capture probe for native bundle storage.

This field tool deliberately does not construct an arm backend, a gripper
transport, a teleoperation session, or any command dispatcher.  It opens the
same RealSense and DMTAC tactile sources used by real teleoperation, captures a
short legacy-v2.1-style diagnostic episode, and writes it through the
production ``LeRobotV21Writer`` with native tactile separation enabled.

The output is diagnostic evidence for the storage protocol.  It is not a
formal robot episode and must not be mixed into a training dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import uuid

import numpy as np


def _repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate the DexTeleop repository root")


PROJECT_ROOT = _repository_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collection_runtime.tactile_bundle_materialize import (
    plan_tactile_bundle_materialization,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_interface.tactile_bundle import attach_tactile_bundle
from teleop_runtime.backends.grippers import (
    DMTacGripperBackend,
    DMTacGripperBackendConfig,
)
from teleop_runtime.collection.sample_assembler import (
    CameraFrameMatch,
    CollectorSample,
)
from teleop_runtime.collection.writer_v21 import (
    LeRobotV21Writer,
    LeRobotV21WriterConfig,
)
from teleop_runtime.sources.camera.realsense_rgb import (
    RealSenseRGBSource,
    load_realsense_camera_config_map,
)
from teleop_runtime.sources.camera.startup import start_real_sense_source_with_containment
from teleop_runtime.tools.field_validation.camera_tactile_health_probe import (
    _fresh_camera,
    _fresh_tactile,
    _tactile_ids_by_side,
    _wait_until,
)


def _validate_roots(core_root: Path, tactile_root: Path) -> None:
    if core_root.exists():
        raise ValueError(f"core output must not already exist: {core_root}")
    if not tactile_root.is_dir():
        raise ValueError(
            "tactile storage root must be a pre-existing directory: "
            f"{tactile_root}"
        )
    if tactile_root.is_symlink():
        raise ValueError(f"tactile storage root must not be a symlink: {tactile_root}")
    core_parent = core_root.parent.resolve()
    tactile_resolved = tactile_root.resolve()
    if int(core_parent.stat().st_dev) == int(tactile_resolved.stat().st_dev):
        raise ValueError(
            "core output parent and tactile storage root must be on distinct "
            "filesystem devices"
        )
    if not os.access(core_parent, os.W_OK):
        raise ValueError(f"core output parent is not writable: {core_parent}")
    if not os.access(tactile_resolved, os.W_OK):
        raise ValueError(f"tactile storage root is not writable: {tactile_root}")


def _camera_sources(
    camera_config: Path,
    *,
    startup_grace_sec: float,
) -> dict[str, RealSenseRGBSource]:
    configs = load_realsense_camera_config_map(camera_config)
    for config in configs.values():
        # A storage protocol probe must not reset shared USB hardware.
        config.startup_reset = False
        config.startup_hardware_reset = False
        config.startup_hardware_reset_on_failure = False
        config.startup_hardware_reset_on_warmup_failure = False
        config.startup_hardware_reset_retry_attempts = 0
        config.startup_warmup_frames = 10
        # A slow first frame is not a reason to reset USB hardware.  Keep the
        # SDK polling cadence short while allowing one longer, still-bounded
        # no-reset admission window for this diagnostic probe.
        config.startup_frame_grace_sec = float(startup_grace_sec)
        config.startup_wait_timeout_ms = max(
            int(config.startup_wait_timeout_ms),
            int(round(float(startup_grace_sec) * 1000.0)),
        )
    return {key: RealSenseRGBSource(config) for key, config in configs.items()}


def _tactile_sources() -> dict[str, DMTacGripperBackend]:
    return {
        side: DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side=side,
                connect_hardware=False,
                connect_tactile=True,
                enable_telemetry=True,
                tactile_ids=ids,
                tactile_fields=("raw",),
                tactile_compute_profile="requested",
                tactile_calibration_policy="required",
                tactile_read_hz=15.0,
                tactile_max_fps=30,
                tactile_process_read=True,
                tactile_reset_on_start=False,
                tactile_warmup_frames=0,
                tactile_store_flat_vector=False,
            )
        )
        for side, ids in _tactile_ids_by_side().items()
        if ids
    }


def _start_camera_no_reset(
    camera_key: str,
    source: RealSenseRGBSource,
    *,
    attempts: int,
    retry_delay_sec: float,
    startup_timeout_sec: float,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, int(attempts) + 1):
        print(
            f"[NoMotionBundleProbe] starting camera {camera_key} "
            f"attempt={attempt}/{attempts}",
            flush=True,
        )
        try:
            start_real_sense_source_with_containment(
                source,
                timeout_sec=startup_timeout_sec,
                software_retry_attempts=0,
            )
            return
        except TimeoutError:
            # The native worker may still own the SDK lifecycle lock.
            # Do not close or retry this source in the same process.
            raise
        except Exception as exc:
            last_error = exc
            source.close()
            if attempt < int(attempts) and float(retry_delay_sec) > 0.0:
                time.sleep(float(retry_delay_sec))
    assert last_error is not None
    raise RuntimeError(
        f"camera {camera_key!r} failed after {attempts} no-reset startup attempts"
    ) from last_error


def _capture_sample(
    *,
    sample_index: int,
    timestamp: float,
    cameras: dict[str, RealSenseRGBSource],
    tactile: dict[str, DMTacGripperBackend],
) -> CollectorSample:
    camera_matches: dict[str, CameraFrameMatch] = {}
    for key, source in cameras.items():
        frame = source.get_latest_frame()
        if frame is None:
            raise RuntimeError(f"camera {key!r} has no frame during capture")
        camera_matches[key] = CameraFrameMatch(
            camera_key=key,
            frame=frame,
            selected_timestamp=float(frame.source_timestamp),
            age_sec=float(timestamp - float(frame.source_timestamp)),
        )

    hand_states: dict[str, dict[str, Any]] = {}
    for side, backend in tactile.items():
        state = dict(
            backend.get_state(
                include_tactile_arrays=True,
                include_tactile_vector=False,
                copy_tactile_arrays=True,
            )
        )
        if not state.get("tactile_dict"):
            raise RuntimeError(f"tactile side {side!r} has no arrays during capture")
        version = int(state.get("tactile_version") or 0)
        if version <= 0:
            raise RuntimeError(f"tactile side {side!r} has no valid snapshot version")
        # Match the production collector's immutable snapshot de-duplication:
        # multiple 15 Hz samples may reference one producer snapshot, while
        # the large tactile arrays are written only once.
        state["tactile_snapshot_sequence"] = version
        state["tactile_snapshot_source_timestamp"] = state.get(
            "tactile_timestamp"
        )
        state["tactile_snapshot_available_timestamp"] = timestamp
        state["tactile_snapshot_source_clock"] = "local_wall_time"
        state["tactile_snapshot_available_clock"] = "local_wall_time"
        hand_states[side] = state

    return CollectorSample(
        index=sample_index,
        episode_index=0,
        frame_index=sample_index,
        timestamp=timestamp,
        session_state="running",
        control_hz=15.0,
        action=np.zeros(26, dtype=np.float32),
        observation_state=np.zeros(38, dtype=np.float32),
        observation_last_action=np.zeros(26, dtype=np.float32),
        arm_state_source="no_arm_online_storage_probe",
        active_sides=tuple(sorted(tactile)),
        arm_states={},
        hand_states=hand_states,
        camera_frames=camera_matches,
        metadata={
            "diagnostic_only": True,
            "no_arm_backend_created": True,
            "no_gripper_transport_created": True,
            "probe": "camera_tactile_bundle_capture_probe",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--tactile-root", type=Path, required=True)
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=Path("teleop_runtime/configs/devices/realsense.json"),
    )
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--sample-hz", type=float, default=15.0)
    parser.add_argument("--ready-timeout-sec", type=float, default=35.0)
    parser.add_argument("--camera-startup-grace-sec", type=float, default=15.0)
    parser.add_argument("--camera-start-stagger-sec", type=float, default=0.5)
    parser.add_argument("--camera-start-attempts", type=int, default=3)
    parser.add_argument("--camera-start-retry-delay-sec", type=float, default=1.0)
    parser.add_argument("--tactile-compression", choices=("zstd", "lz4", "none"), default="zstd")
    args = parser.parse_args()

    core_root = args.core_root.expanduser().resolve()
    tactile_root = args.tactile_root.expanduser().resolve()
    camera_config = args.camera_config.expanduser().resolve()
    if float(args.duration_sec) <= 0.0 or float(args.sample_hz) <= 0.0:
        parser.error("--duration-sec and --sample-hz must be positive")
    if not camera_config.is_file():
        parser.error(f"camera config does not exist: {camera_config}")
    _validate_roots(core_root, tactile_root)

    if float(args.camera_startup_grace_sec) <= 0.0:
        parser.error("--camera-startup-grace-sec must be positive")
    if float(args.camera_start_stagger_sec) < 0.0:
        parser.error("--camera-start-stagger-sec must be non-negative")
    if int(args.camera_start_attempts) <= 0:
        parser.error("--camera-start-attempts must be positive")
    if float(args.camera_start_retry_delay_sec) < 0.0:
        parser.error("--camera-start-retry-delay-sec must be non-negative")
    cameras = _camera_sources(
        camera_config,
        startup_grace_sec=float(args.camera_startup_grace_sec),
    )
    tactile = _tactile_sources()
    if set(cameras) != {"front", "left_wrist", "right_wrist"}:
        raise RuntimeError(f"expected three RGB cameras, found {sorted(cameras)}")
    if set(tactile) != {"left", "right"}:
        raise RuntimeError(f"expected two tactile sides, found {sorted(tactile)}")

    writer: LeRobotV21Writer | None = None
    timed_out_camera: str | None = None
    started_at = time.time()
    try:
        # Preserve the proven serial RealSense startup contract.  The probe
        # changes no startup concurrency or USB recovery behavior.
        camera_items = list(cameras.items())
        for camera_index, (camera_key, source) in enumerate(camera_items):
            try:
                _start_camera_no_reset(
                    camera_key,
                    source,
                    attempts=int(args.camera_start_attempts),
                    retry_delay_sec=float(args.camera_start_retry_delay_sec),
                    startup_timeout_sec=float(args.camera_startup_grace_sec) + 5.0,
                )
            except TimeoutError:
                timed_out_camera = camera_key
                raise
            if camera_index + 1 < len(camera_items):
                stagger_sec = float(args.camera_start_stagger_sec)
                if stagger_sec > 0.0:
                    time.sleep(stagger_sec)
        for backend in tactile.values():
            backend.start()
        ready = _wait_until(
            lambda: all(_fresh_camera(source) for source in cameras.values())
            and all(_fresh_tactile(backend) for backend in tactile.values()),
            timeout_sec=float(args.ready_timeout_sec),
        )
        if not ready:
            raise RuntimeError("camera/tactile sources did not become fresh before timeout")

        writer = LeRobotV21Writer(
            LeRobotV21WriterConfig(
                root_dir=core_root,
                nominal_fps=float(args.sample_hz),
                task_id="native-tactile-bundle-no-motion-online-probe",
                task_description=(
                    "diagnostic no-arm live RGB and DMTAC capture for native "
                    "tactile separation verification"
                ),
                dataset_name=core_root.name,
                write_parquet=True,
                write_video=True,
                tactile_storage="external",
                tactile_compression=str(args.tactile_compression),
                tactile_artifact_integrity=True,
                tactile_bundle_root=tactile_root,
                tactile_bundle_require_distinct_device=True,
                dataset_metadata={
                    "diagnostic_only": True,
                    "no_arm_backend_created": True,
                    "no_gripper_transport_created": True,
                    "capture_probe_schema": "dexteleop.no_motion_sensor_bundle_probe.v1",
                },
            )
        )
        identity = EpisodeCaptureIdentity(
            episode_uuid=str(uuid.uuid4()),
            legacy_episode_index=0,
            lifecycle_owner_id="field_validation.camera_tactile_bundle_capture_probe",
            prepared_at=time.time(),
            prepared_clock_domain_id="local_wall_time",
        )
        writer.start_episode(
            metadata={
                "diagnostic_only": True,
                "no_robot_motion": True,
                "device_source": "live",
            },
            episode_identity=identity,
        )

        sample_period = 1.0 / float(args.sample_hz)
        sample_count = max(
            1,
            int(round(float(args.duration_sec) * float(args.sample_hz))),
        )
        next_sample_at = time.monotonic()
        for sample_index in range(sample_count):
            remaining = next_sample_at - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            timestamp = time.time()
            sample = _capture_sample(
                sample_index=sample_index,
                timestamp=timestamp,
                cameras=cameras,
                tactile=tactile,
            )
            for job in writer.plan_sample_jobs(sample):
                writer.handle_job(job)
            next_sample_at += sample_period
        writer.finalize_episode(
            0,
            summary={
                "diagnostic_only": True,
                "no_robot_motion": True,
                "requested_sample_count": sample_count,
            },
        )

        bundle_store = writer.tactile_bundle_store
        if bundle_store is None:
            raise RuntimeError("native tactile bundle store was not created")
        attachment = attach_tactile_bundle(
            core_root,
            tactile_root,
            verify_payload_hashes=True,
        )
        materialization_plan = plan_tactile_bundle_materialization(
            core_root,
            tactile_root,
        )
        report = {
            "schema": "dexteleop.no_motion_sensor_bundle_probe.report.v1",
            "ok": True,
            "core_root": str(core_root),
            "tactile_storage_root": str(tactile_root),
            "tactile_bundle_dataset_root": str(bundle_store.bundle_dataset_root),
            "dataset_uuid": bundle_store.dataset_identity.dataset_uuid,
            "episode_uuid": identity.episode_uuid,
            "sample_count": sample_count,
            "elapsed_sec": time.time() - started_at,
            "core_device": int(core_root.stat().st_dev),
            "tactile_device": int(tactile_root.stat().st_dev),
            "camera_frame_counts": {
                key: int(source.device_info().get("frame_count") or 0)
                for key, source in cameras.items()
            },
            "tactile_versions": {
                side: int(
                    backend.get_state(include_tactile_arrays=False).get(
                        "tactile_version"
                    )
                    or 0
                )
                for side, backend in tactile.items()
            },
            "arm_backend_count": 0,
            "gripper_transport_count": 0,
            "attachment": attachment.report.to_dict(),
            "materialization_plan": materialization_plan.to_dict(),
        }
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except BaseException:
        if writer is not None and 0 in writer._episodes:
            writer.discard_episode(0)
        raise
    finally:
        for backend in tactile.values():
            backend.close()
        for camera_key, source in cameras.items():
            if camera_key != timed_out_camera:
                source.close()


if __name__ == "__main__":
    raise SystemExit(main())
