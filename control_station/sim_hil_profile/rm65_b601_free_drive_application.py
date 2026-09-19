"""Read-only RealMan RM65 free-drive to B601 MuJoCo application.

The command in this module owns only application assembly and lifecycle.  The
RM65 backend remains the sole owner of SDK reads, the free-drive source admits
only fresh observation-only state, and the B601 bridge owns simulation
retargeting and position-driver safety.  In ``--connect-read-only`` mode no
RM65 command or enable/disable operation is issued by this application.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

from control_station.sim_hil_profile.rm65_b601_free_drive import (
    RM65ToB601FreeDriveSimConfig,
    build_rm65_to_b601_free_drive_bridge,
    load_rm65_to_b601_mapping_profile,
)
from control_station.sources.rm65_free_drive import (
    RM65FreeDriveSource,
    RM65FreeDriveSourceSpec,
)
from teleop_runtime.backends.arms.rm65 import (
    RM65ArmBackend,
    RM65ArmBackendConfig,
)

READ_ONLY_CONFIRMATION = "RM65_B601_READ_ONLY_FREE_DRIVE"


def _positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def _non_negative(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return result


def _gain_triplet(values: list[str]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("translation gain requires exactly 3 values")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise argparse.ArgumentTypeError("translation gains must be positive and finite")
    return result  # type: ignore[return-value]


def build_observation_only_rm65_backend(
    *,
    arm_ip: str,
    arm_urdf_path: Path,
    connect_hardware: bool,
) -> RM65ArmBackend:
    """Build the existing RM65 backend with the read-only contract fixed on.

    Keeping this construction in one application-owned helper makes the field
    command auditable and prevents a caller from accidentally reusing a normal
    command-capable RM65 profile for this reverse simulation path.
    """

    normalized_ip = str(arm_ip).strip()
    if not normalized_ip:
        raise ValueError("arm_ip must be non-empty")
    resolved_urdf = Path(arm_urdf_path).expanduser().resolve()
    if not resolved_urdf.is_file():
        raise FileNotFoundError(f"RM65 arm URDF does not exist: {resolved_urdf}")
    return RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip=normalized_ip,
            side="right",
            arm_urdf_path=resolved_urdf,
            connect_hardware=bool(connect_hardware),
            observation_only=True,
            background_streaming_enabled=False,
            force_position_control=False,
            canfd_follow=False,
            auto_load_tool_profile=False,
        )
    )


def _record_factory(log, *, flush_interval_sec: float = 0.25):
    last_flush = time.monotonic()

    def record(event: object) -> None:
        nonlocal last_flush
        log.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        now = time.monotonic()
        event_name = event.get("event") if isinstance(event, dict) else None
        if (
            event_name in {"command", "error", "result", "cleanup_error"}
            or now - last_flush >= float(flush_interval_sec)
        ):
            log.flush()
            last_flush = now

    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the bridge/model/profile without opening RM65 or a GUI",
    )
    mode.add_argument(
        "--connect-read-only",
        action="store_true",
        help="read the existing right RM65 state; never send RM65 commands",
    )
    parser.add_argument("--confirmation")
    parser.add_argument("--arm-ip", required=True)
    parser.add_argument("--arm-urdf-path", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--mapping-profile", type=Path, required=True)
    parser.add_argument("--rebot-source-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate-hz", type=_positive, default=1000.0)
    parser.add_argument(
        "--viewer-sync-hz",
        type=_positive,
        default=60.0,
        help="passive viewer refresh rate; physics remains at the B601 plant rate",
    )
    parser.add_argument("--source-poll-hz", type=_positive, default=20.0)
    parser.add_argument("--maximum-source-age-sec", type=_positive, default=0.25)
    parser.add_argument("--maximum-receive-age-sec", type=_positive, default=0.25)
    parser.add_argument("--maximum-future-skew-sec", type=_non_negative, default=0.05)
    parser.add_argument(
        "--translation-gain-target-reference-xyz",
        nargs=3,
        default=("1.0", "1.0", "1.0"),
        metavar=("GX", "GY", "GZ"),
        help="positive translation gains applied after clutch anchoring",
    )
    parser.add_argument("--rotation-gain", type=_positive, default=1.0)
    args = parser.parse_args(argv)
    if args.connect_read_only and args.confirmation != READ_ONLY_CONFIRMATION:
        parser.error(
            "read-only RM65 connection requires "
            f"--confirmation {READ_ONLY_CONFIRMATION}"
        )
    gains = _gain_triplet(list(args.translation_gain_target_reference_xyz))
    values = list(sys.argv[1:] if argv is None else argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("x", encoding="utf-8") as output,
        args.log.open("a", encoding="utf-8") as log,
    ):
        record = _record_factory(log)
        result: dict[str, object] = {
            "status": "ERROR",
            "execution_status": "NOT_STARTED",
            "rm65_commands_sent": False,
            "rm65_observation_only": True,
            "b601_hardware_commands_sent": False,
            "mujoco_target_cleanup_completed": False,
            "backend_start_completed": False,
            "bridge_state": "not_started",
            "argv": [sys.executable, "-m", "control_station", "rm65-b601-free-drive", *values],
            "mapping_profile": str(args.mapping_profile),
            "arm_ip": str(args.arm_ip),
            "arm_urdf_path": str(args.arm_urdf_path),
            "translation_gain_target_reference_xyz": list(gains),
            "rotation_gain": float(args.rotation_gain),
            "rate_hz": float(args.rate_hz),
            "viewer_sync_hz": float(args.viewer_sync_hz),
            "source_poll_hz": float(args.source_poll_hz),
            "source_poll_clock": "monotonic_wall_time",
        }

        def emit(message: str) -> None:
            print(message, flush=True)
            record({"event": "terminal", "message": message})

        record({"event": "command", "argv": result["argv"]})
        backend: RM65ArmBackend | None = None
        bridge = None
        try:
            mapping = load_rm65_to_b601_mapping_profile(args.mapping_profile)
            backend = build_observation_only_rm65_backend(
                arm_ip=args.arm_ip,
                arm_urdf_path=args.arm_urdf_path,
                connect_hardware=bool(args.connect_read_only),
            )
            source = RM65FreeDriveSource(
                backend,
                spec=RM65FreeDriveSourceSpec(
                    maximum_measurement_age_sec=args.maximum_source_age_sec,
                    maximum_receive_age_sec=args.maximum_receive_age_sec,
                    maximum_future_skew_sec=args.maximum_future_skew_sec,
                ),
            )
            bridge = build_rm65_to_b601_free_drive_bridge(
                args.rebot_source_root,
                source,
                mapping,
                config=RM65ToB601FreeDriveSimConfig(
                    translation_gain_target_reference_xyz=gains,
                    rotation_gain=args.rotation_gain,
                    source_poll_hz=args.source_poll_hz,
                    maximum_source_age_sec=args.maximum_source_age_sec,
                    maximum_receive_age_sec=args.maximum_receive_age_sec,
                    maximum_future_skew_sec=args.maximum_future_skew_sec,
                ),
                source_poll_clock=time.monotonic,
            )
            result["execution_status"] = "VALIDATED"
            result["bridge_state"] = bridge.state
            if args.validate_only:
                result["status"] = "VALIDATED"
                output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                record({"event": "result", **result, "exit_code": 0})
                return 0

            assert backend is not None
            from control_station.sim_hil_profile.rm65_b601_free_drive_foreground import (
                run_rm65_to_b601_foreground,
            )
            from sim.mujoco.viewer import MujocoPassiveViewerSession

            emit(
                "[RM65→B601] RECEIVE ONLY: RM65 observation-only; no RM65 "
                "enable, disable, position, torque, or stop command is issued."
            )
            emit(
                "[RM65→B601] MuJoCo B601 target is simulation-only; "
                "Ctrl-C/EOF closes the viewer and leaves RM65 unchanged."
            )
            backend.start()
            result["backend_start_completed"] = True
            viewer = MujocoPassiveViewerSession(bridge.driver.scene)
            with ExitStack() as resources:
                resources.callback(viewer.close)
                resources.callback(backend.close)
                viewer.start()
                run_rm65_to_b601_foreground(
                    bridge,
                    viewer=viewer,
                    rate_hz=args.rate_hz,
                    viewer_sync_hz=args.viewer_sync_hz,
                    emit=emit,
                    record=record,
                )
                result["bridge_state"] = bridge.state
                result["mujoco_target_cleanup_completed"] = True
            result["execution_status"] = "COMPLETED"
            result["status"] = "COMPLETED"
            output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            record({"event": "result", **result, "exit_code": 0})
            return 0
        except KeyboardInterrupt:
            result["status"] = "INTERRUPTED"
            result["execution_status"] = "INTERRUPTED"
            emit("[RM65→B601] Interrupted; MuJoCo target is withdrawn; RM65 remains unchanged.")
            output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            record({"event": "result", **result, "exit_code": 130})
            return 130
        except Exception as error:  # noqa: BLE001 - CLI boundary records evidence.
            result["error"] = f"{type(error).__name__}: {error}"
            record({"event": "error", "traceback": traceback.format_exc()})
            emit(f"[RM65→B601] ERROR: {result['error']}")
            output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            record({"event": "result", **result, "exit_code": 1})
            return 1
        finally:
            # The ExitStack owns normal live cleanup.  If construction or start
            # failed before it was entered, still close the observation backend
            # without issuing any RM65 motion command.
            if backend is not None and not bool(result.get("backend_start_completed")):
                try:
                    backend.close()
                except Exception as cleanup_error:  # noqa: BLE001
                    record(
                        {
                            "event": "cleanup_error",
                            "error": f"{type(cleanup_error).__name__}: {cleanup_error}",
                        }
                    )


__all__ = [
    "READ_ONLY_CONFIRMATION",
    "build_observation_only_rm65_backend",
    "main",
]
