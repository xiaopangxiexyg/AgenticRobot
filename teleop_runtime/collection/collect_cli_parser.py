"""Argument parser construction for the headless collection compatibility path."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence

import numpy as np

from dataset_contracts.pose_frames import load_pose_frame_contract
from project_paths import calibration_profiles_dir, default_dataset_root

from ..backends.grippers import DMTAC_TACTILE_FIELDS, normalize_dmtac_tactile_fields
from ..backends.hands import DEFAULT_INSPIRE_HAND_MAX_FORCE
from ..configs.camera_paths import (
    DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH,
    DEFAULT_REALSENSE_DEVICE_CONFIG_PATH,
)
from ..configs.dmtac_hardware import add_dmtac_assignment_arguments
from ..defaults import (
    DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
    DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    DEFAULT_TELEOP_SAMPLE_HZ,
)
from ..hand_mapping import DEFAULT_INSPIRE_HAND_JOINT_ORDER
from ..hand_tcp_alignment import add_vive_tracker_height_correction_args
from ..sources.pose import DEFAULT_VIVE_CLOCK_SYNC_PORT
from .topology import parse_active_sides as _topology_parse_active_sides

DEFAULT_CAMERA_CONFIG_PATH = DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
DEFAULT_CAMERA_REGISTRY_PATH = DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH
DEFAULT_DATASET_ROOT = default_dataset_root()
DEFAULT_CALIBRATION_PROFILE_DIR = calibration_profiles_dir()
HAND_BACKEND_INSPIRE = "inspire"
HAND_BACKEND_DMTAC_GRIPPER = "dmtac-gripper"
HAND_BACKEND_NONE = "none"
HAND_BACKEND_CHOICES = (
    HAND_BACKEND_INSPIRE,
    HAND_BACKEND_DMTAC_GRIPPER,
    HAND_BACKEND_NONE,
)


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def _alpha_float(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("alpha must be in (0, 1]")
    return value


def _non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _dmtac_worker_cpu_affinity(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in re.split(r"[,\s]+", str(text)) if part.strip()]
    if not parts:
        return ()
    cpus: list[int] = []
    for part in parts:
        try:
            cpu = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "CPU affinity must be a comma-separated integer list"
            ) from exc
        if cpu < 0:
            raise argparse.ArgumentTypeError("CPU affinity indices must be >= 0")
        cpus.append(cpu)
    return tuple(dict.fromkeys(cpus))


def _dmtac_worker_nice(text: str) -> int:
    value = int(text)
    if not 0 <= value <= 19:
        raise argparse.ArgumentTypeError("worker nice must be in [0, 19]")
    return value


def _parse_dmtac_tactile_fields(text: str) -> tuple[str, ...]:
    try:
        return normalize_dmtac_tactile_fields(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_joint_order(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "joint order must contain 6 comma-separated indices"
        )
    try:
        order = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid joint order: {text}") from exc
    if sorted(order) != [0, 1, 2, 3, 4, 5]:
        raise argparse.ArgumentTypeError(
            "joint order must be a permutation of 0,1,2,3,4,5"
        )
    return order


def _parse_arm_max_angvel(text: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("arm max angvel must contain 1 or 6 floats")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid arm max angvel: {text}") from exc
    if len(values) == 1:
        if values[0] <= 0.0:
            raise argparse.ArgumentTypeError("arm max angvel must be positive")
        return (float(values[0]),) * 6
    if len(values) != 6:
        raise argparse.ArgumentTypeError("arm max angvel must contain 1 or 6 floats")
    if any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("arm max angvel must be positive")
    return tuple(float(value) for value in values)


def _parse_qpos6(text: str) -> tuple[float, ...]:
    cleaned = str(text).strip().strip("[]()")
    parts = [part for part in re.split(r"[\s,]+", cleaned) if part]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("qpos must contain exactly 6 floats")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid qpos: {text}") from exc
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise argparse.ArgumentTypeError("qpos values must be finite")
    return values


def _looks_like_single_token_qpos6(text: str) -> bool:
    token = str(text).strip()
    if not token or token.startswith("--"):
        return False
    parts = [part for part in token.split(",") if part.strip()]
    if len(parts) != 6:
        return False
    try:
        values = np.asarray([float(part) for part in parts], dtype=float)
    except ValueError:
        return False
    return bool(np.all(np.isfinite(values)))


def normalize_qpos_option_argv(argv: Sequence[str]) -> list[str]:
    """Protect comma-separated negative qpos vectors from option-token parsing."""

    qpos_flags = {
        "--return-left-qpos",
        "--return-right-qpos",
        "--return-left-qpos-deg",
        "--return-right-qpos-deg",
    }
    normalized: list[str] = []
    index = 0
    tokens = list(argv)
    while index < len(tokens):
        token = str(tokens[index])
        if token in qpos_flags and (index + 1) < len(tokens):
            next_token = str(tokens[index + 1])
            if _looks_like_single_token_qpos6(next_token):
                normalized.append(f"{token}={next_token}")
                index += 2
                continue
        normalized.append(token)
        index += 1
    return normalized


def _parse_active_sides(text: str) -> tuple[str, ...]:
    try:
        return _topology_parse_active_sides(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_collection_parser() -> argparse.ArgumentParser:
    """Build the collection parser without reading argv or initializing runtime resources."""

    parser = argparse.ArgumentParser(
        description="Collect LeRobot-style teleop episodes from real RM65/Inspire hardware."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help=(
            "Base output directory. Actual dataset path is "
            f"<dataset-root>/<YYYY-MM-DD>_<task-name>. Default: {DEFAULT_DATASET_ROOT}."
        ),
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="",
        help="Task name used for dataset folder naming and metadata. Defaults to --task, then 'default'.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Legacy task id/name. Used as task id and as task name when --task-name is omitted.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="",
        help="Optional task id; defaults to a slug from task name.",
    )
    parser.add_argument(
        "--task-description", type=str, default="", help="Optional task description."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Optional dataset name written into meta/info.json.",
    )
    parser.add_argument(
        "--active-sides",
        type=_parse_active_sides,
        default=("left", "right"),
        help="One of: left, right, both.",
    )
    parser.add_argument(
        "--load-profile",
        type=str,
        default="",
        help=(
            "Path to a TeleopCalibration JSON profile. If omitted, the newest "
            f"profile in {DEFAULT_CALIBRATION_PROFILE_DIR} is loaded when available."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="UDP bind host for the VIVE receiver.",
    )
    parser.add_argument(
        "--port", type=int, default=9999, help="UDP bind port for the VIVE receiver."
    )
    parser.add_argument(
        "--vive-clock-sync-port",
        type=int,
        default=DEFAULT_VIVE_CLOCK_SYNC_PORT,
        help=(
            "Separate UDP clock-control bind port for the default new protocol. "
            "Set 0 to select the legacy v1-only VIVE receiver."
        ),
    )
    parser.add_argument(
        "--vive-clock-sync-bind-host",
        type=str,
        default="",
        help="Optional bind host for the VIVE clock-control port; defaults to --host.",
    )
    parser.add_argument(
        "--vive-clock-sync-expected-sender-host",
        type=str,
        default="",
        help="Optional VIVE sender IP allowed to submit clock-control results.",
    )
    parser.add_argument(
        "--vive-clock-sync-min-samples",
        type=int,
        default=4,
        help="Accepted four-timestamp samples required before sync is ready.",
    )
    parser.add_argument(
        "--vive-clock-sync-max-rtt-ms",
        type=float,
        default=25.0,
        help="Reject VIVE clock samples above this RTT; <=0 disables the RTT cap.",
    )
    parser.add_argument(
        "--vive-clock-sync-stale-after-sec",
        type=float,
        default=15.0,
        help="Mark mapped remote capture time stale after this many seconds.",
    )
    parser.add_argument(
        "--vive-clock-sync-required",
        action="store_true",
        help="Fail startup if optional VIVE clock control does not become ready.",
    )
    parser.add_argument(
        "--vive-clock-sync-startup-timeout-sec",
        type=float,
        default=5.0,
        help="Maximum pre-recording wait for required VIVE clock calibration.",
    )
    parser.add_argument(
        "--left-tracker-name",
        type=str,
        default="tracker_1",
        help="Left wrist tracker name.",
    )
    parser.add_argument(
        "--right-tracker-name",
        type=str,
        default="tracker_2",
        help="Right wrist tracker name.",
    )
    parser.add_argument("--left-arm-ip", type=str, default="", help="Left RM65 IP.")
    parser.add_argument("--right-arm-ip", type=str, default="", help="Right RM65 IP.")
    parser.add_argument(
        "--left-tool-profile",
        type=str,
        default="",
        help=(
            "Optional explicit left RM65 tool profile name or JSON path. "
            "Use position_endpoint for endpoint-TCP position control without payload/impedance metadata."
        ),
    )
    parser.add_argument(
        "--right-tool-profile",
        type=str,
        default="",
        help=(
            "Optional explicit right RM65 tool profile name or JSON path. "
            "Use position_endpoint for endpoint-TCP position control without payload/impedance metadata."
        ),
    )
    parser.add_argument(
        "--left-arm-fixed",
        action="store_true",
        help=(
            "Keep the left arm fixed at the hardware position captured by p; "
            "right arm and both hands keep normal teleop, and bimanual dataset fields remain populated."
        ),
    )
    parser.add_argument(
        "--left-arm-jitter",
        action="store_true",
        help=(
            "Keep the left arm near the hardware position captured by p, but feed a smooth random "
            "small cartesian jitter target into the dual-arm IK retargeter."
        ),
    )
    parser.add_argument(
        "--left-arm-jitter-position-m",
        type=float,
        default=0.01,
        help="Maximum left TCP jitter translation radius in desk-frame meters.",
    )
    parser.add_argument(
        "--left-arm-jitter-rotation-deg",
        type=float,
        default=3.0,
        help="Maximum left TCP jitter rotation angle in degrees.",
    )
    parser.add_argument(
        "--left-arm-jitter-refresh-sec",
        type=_positive_float,
        default=1.2,
        help="Seconds between new random left-arm jitter targets.",
    )
    parser.add_argument(
        "--left-arm-jitter-smoothing-sec",
        type=float,
        default=0.45,
        help="First-order smoothing time constant for left-arm jitter; 0 jumps directly to each random target.",
    )
    parser.add_argument(
        "--left-arm-jitter-seed",
        type=int,
        default=-1,
        help="Random seed for left-arm jitter; negative uses OS entropy.",
    )
    parser.add_argument(
        "--left-hand-ip", type=str, default="", help="Optional left InspireTac IP."
    )
    parser.add_argument(
        "--right-hand-ip", type=str, default="", help="Optional right InspireTac IP."
    )
    parser.add_argument(
        "--left-hand-port", type=str, default="", help="Optional left q6 serial port."
    )
    parser.add_argument(
        "--right-hand-port", type=str, default="", help="Optional right q6 serial port."
    )
    parser.add_argument(
        "--hand-backend",
        choices=HAND_BACKEND_CHOICES,
        default=HAND_BACKEND_INSPIRE,
        help=(
            "Hand/gripper backend type. Use dmtac-gripper to map Q6 to a generic "
            "1D gripper controller through the DMTAC gripper backend."
        ),
    )
    parser.add_argument(
        "--left-hand-backend",
        choices=HAND_BACKEND_CHOICES,
        default=None,
        help="Optional left-side override for --hand-backend.",
    )
    parser.add_argument(
        "--right-hand-backend",
        choices=HAND_BACKEND_CHOICES,
        default=None,
        help="Optional right-side override for --hand-backend.",
    )
    parser.add_argument(
        "--q6-config",
        type=str,
        default="latest",
        help=(
            "Hand Q6 left/right assignment config. Explicit --left/right-hand-port "
            "wins; missing ports are loaded from this config when available."
        ),
    )
    parser.add_argument(
        "--no-q6-default-config",
        dest="q6_default_config",
        action="store_false",
        default=True,
        help="Do not auto-fill missing hand Q6 ports from the latest saved assignment.",
    )
    parser.add_argument(
        "--q6-config-required",
        action="store_true",
        help="Fail if --q6-config cannot be loaded while an active side is missing a hand port.",
    )
    add_dmtac_assignment_arguments(parser, assignment_includes_tactile=True)
    parser.add_argument(
        "--dmtac-tactile-backend",
        type=str,
        default="cpu",
        help="DM-Tac tactile backend passed to portable_dmtac_gripper.",
    )
    parser.add_argument(
        "--dmtac-tactile-timeout-ms",
        type=_positive_int,
        default=None,
        help="Per-sensor DM-Tac read timeout. Runtime telemetry does not wait for new frames by default.",
    )
    parser.add_argument(
        "--dmtac-tactile-max-fps",
        type=_positive_int,
        default=None,
        help="DM-Tac camera max_fps. Lower values reduce USB/CPU pressure; default follows backend config.",
    )
    parser.add_argument(
        "--dmtac-tactile-warmup-frames",
        type=_non_negative_int,
        default=None,
        help="DM-Tac tactile filter warmup frames. Runtime default is 0 like the assignment GUI.",
    )
    parser.add_argument(
        "--dmtac-tactile-wait-for-new",
        action="store_true",
        default=None,
        help="Wait for a new DM-Tac frame on each telemetry read. Higher latency; off by default.",
    )
    parser.add_argument(
        "--dmtac-tactile-read-hz",
        type=_non_negative_float,
        default=None,
        help=(
            "Max DM-Tac tactile read rate per gripper backend. Default is backend config "
            "1 Hz; use 0 to read every gripper telemetry cycle."
        ),
    )
    parser.add_argument(
        "--dmtac-tactile-process-read",
        dest="dmtac_tactile_process_read",
        action="store_true",
        default=None,
        help="Read DM-Tac tactile frames in a separate low-priority process. Runtime default is on.",
    )
    parser.add_argument(
        "--dmtac-tactile-in-process-read",
        dest="dmtac_tactile_process_read",
        action="store_false",
        help="Read DM-Tac tactile frames in the main collection process.",
    )
    parser.add_argument(
        "--dmtac-tactile-worker-cpus",
        type=_dmtac_worker_cpu_affinity,
        default=None,
        help=(
            "Comma/space-separated CPU cores for low-priority DM-Tac tactile worker processes, "
            "for example '6,7'. Unset lets the OS schedule normally."
        ),
    )
    parser.add_argument(
        "--dmtac-tactile-worker-nice",
        type=_dmtac_worker_nice,
        default=None,
        help="Lower scheduling priority for DM-Tac tactile worker processes; 0 is normal, 19 is lowest.",
    )
    parser.add_argument(
        "--dmtac-tactile-worker-queue-size",
        type=_positive_int,
        default=None,
        help="Latest-frame queue size between each DM-Tac tactile worker process and the main process.",
    )
    parser.add_argument(
        "--dmtac-tactile-parallel-read",
        dest="dmtac_tactile_parallel_read",
        action="store_true",
        default=None,
        help="Read sensors in each DM-Tac gripper backend concurrently. Higher USB/CPU pressure.",
    )
    parser.add_argument(
        "--dmtac-tactile-sequential-read",
        dest="dmtac_tactile_parallel_read",
        action="store_false",
        help="Read sensors in each DM-Tac gripper backend sequentially. This is the runtime default.",
    )
    parser.add_argument(
        "--dmtac-tactile-global-read-lock",
        dest="dmtac_tactile_global_read_lock",
        action="store_true",
        default=None,
        help="Serialize DMTAC tactile reads across left/right gripper backends. Runtime default is on.",
    )
    parser.add_argument(
        "--dmtac-tactile-no-global-read-lock",
        dest="dmtac_tactile_global_read_lock",
        action="store_false",
        help="Allow left/right DMTAC tactile reads to overlap.",
    )
    parser.add_argument(
        "--dmtac-tactile-fields",
        type=_parse_dmtac_tactile_fields,
        default=None,
        metavar="FIELDS",
        help=(
            "DM-Tac fields to read/store. Default is raw-image-only collection; pass "
            "'distributed_force' for real-time force output. Valid fields: "
            + ", ".join(DMTAC_TACTILE_FIELDS)
        ),
    )
    parser.add_argument(
        "--dmtac-tactile-compute-profile",
        choices=("dist-only", "legacy", "requested", "shear-depth"),
        default=None,
        help=(
            "SDK compute channels for DMTAC telemetry. Default follows backend config; "
            "use dist-only with distributed_force for the real-time force path."
        ),
    )
    parser.add_argument(
        "--dmtac-tactile-calibrate",
        dest="dmtac_tactile_calibrate",
        action="store_true",
        default=None,
        help=(
            "Explicitly capture a DM-Tac no-contact base frame before reads. "
            "Disabled by default for raw images and solved tactile fields."
        ),
    )
    parser.add_argument(
        "--dmtac-tactile-no-calibrate",
        dest="dmtac_tactile_calibrate",
        action="store_false",
        help="Do not capture a DM-Tac base frame at startup (the default).",
    )
    parser.add_argument(
        "--dmtac-tactile-denoise",
        dest="dmtac_tactile_denoise",
        action="store_true",
        default=None,
        help="Enable runtime distributed_force denoise. This is heavier than the assignment GUI default.",
    )
    parser.add_argument(
        "--dmtac-tactile-no-denoise",
        dest="dmtac_tactile_denoise",
        action="store_false",
        help="Disable runtime distributed_force denoise.",
    )
    parser.add_argument(
        "--dmtac-tactile-flat-vector",
        dest="dmtac_tactile_flat_vector",
        action="store_true",
        default=None,
        help="Keep building the legacy flattened tactile vector in telemetry.",
    )
    parser.add_argument(
        "--dmtac-tactile-no-flat-vector",
        dest="dmtac_tactile_flat_vector",
        action="store_false",
        help="Do not build the legacy flattened tactile vector in telemetry.",
    )
    parser.add_argument(
        "--dmtac-with-tactile", dest="dmtac_tactile", action="store_true", default=None
    )
    parser.add_argument(
        "--dmtac-no-tactile", dest="dmtac_tactile", action="store_false"
    )
    parser.add_argument(
        "--gripper-task-profile",
        type=str,
        default="",
        help=(
            "Optional JSON profile with top-level gripper defaults and/or tasks.<name>.gripper "
            "overrides. It can set speed, force/torque, max_action, send_threshold, command_hz, "
            "continuous_grasp, source_dim, output_dim, invert, alpha, input_min, input_max."
        ),
    )
    parser.add_argument(
        "--gripper-task",
        type=str,
        default="",
        help="Explicit gripper task key; overrides --task-id/--task-name for --gripper-task-profile.",
    )
    parser.add_argument(
        "--gripper-source-dim",
        type=int,
        default=None,
        help="Q6 source dimension for 1D gripper.",
    )
    parser.add_argument(
        "--gripper-output-dim",
        type=int,
        default=None,
        help="Output slot in the 6-wide hand vector.",
    )
    parser.add_argument(
        "--gripper-invert", dest="gripper_invert", action="store_true", default=None
    )
    parser.add_argument(
        "--gripper-no-invert", dest="gripper_invert", action="store_false"
    )
    parser.add_argument(
        "--gripper-alpha",
        type=_alpha_float,
        default=None,
        help="EMA alpha for 1D gripper command.",
    )
    parser.add_argument(
        "--gripper-input-min",
        type=float,
        default=None,
        help="Q6 value mapped to fully open gripper before optional inversion.",
    )
    parser.add_argument(
        "--gripper-input-max",
        type=float,
        default=None,
        help="Q6 value mapped to fully closed gripper before optional inversion.",
    )
    parser.add_argument(
        "--gripper-speed",
        type=float,
        default=None,
        help="Gripper move speed for this task.",
    )
    parser.add_argument(
        "--gripper-torque",
        "--gripper-force",
        dest="gripper_torque",
        type=float,
        default=None,
        help="Gripper torque/force setting for this task.",
    )
    parser.add_argument(
        "--gripper-max-action",
        type=float,
        default=None,
        help="Max normalized close command.",
    )
    parser.add_argument(
        "--gripper-send-threshold",
        type=float,
        default=None,
        help="Minimum normalized delta before sending another gripper command.",
    )
    parser.add_argument(
        "--gripper-continuous",
        dest="gripper_continuous_grasp",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--gripper-no-continuous", dest="gripper_continuous_grasp", action="store_false"
    )
    parser.add_argument("--gripper-command-hz", type=_positive_float, default=None)
    parser.add_argument("--gripper-telemetry-hz", type=_positive_float, default=None)
    parser.add_argument("--gripper-telemetry-command-grace-s", type=float, default=None)
    parser.add_argument(
        "--hand-reader-baudrate", type=int, default=115200, help="Hand q6 baudrate."
    )
    parser.add_argument(
        "--hand-reader-timeout",
        type=float,
        default=0.05,
        help="Hand q6 serial timeout in seconds.",
    )
    parser.add_argument(
        "--exo-hand-calibration",
        action="store_true",
        help="Re-run exoskeleton hand zero-pose calibration and update the fixed calibration file.",
    )
    parser.add_argument(
        "--exo-hand-calibration-path",
        type=str,
        default="",
        help="Optional fixed exoskeleton hand zero-pose calibration JSON path.",
    )
    parser.add_argument(
        "--show-hand-raw-lines", action="store_true", help="Print raw hand q6 lines."
    )
    parser.add_argument(
        "--hand-joint-order",
        type=_parse_joint_order,
        default=DEFAULT_INSPIRE_HAND_JOINT_ORDER,
        help="Permutation applied to the incoming q6 hand order.",
    )
    parser.add_argument("--hand-invert", dest="hand_invert", action="store_true")
    parser.add_argument("--hand-no-invert", dest="hand_invert", action="store_false")
    parser.set_defaults(hand_invert=True)
    parser.add_argument(
        "--hand-command-alpha",
        type=_alpha_float,
        default=1.0,
        help="EMA for hand command mapping.",
    )
    parser.add_argument(
        "--hand-send-threshold",
        type=float,
        default=0.002,
        help="Only send hand commands when max delta exceeds this threshold.",
    )
    parser.add_argument(
        "--hand-telemetry-hz",
        type=_positive_float,
        default=30.0,
        help="Hand telemetry polling rate.",
    )
    parser.add_argument(
        "--hand-tactile-regions",
        type=str,
        default="all",
        help="Comma-separated tactile regions or 'all'. Defaults to all; pass an empty string to disable tactile reads.",
    )
    parser.add_argument("--with-tactile", dest="with_tactile", action="store_true")
    parser.add_argument("--without-tactile", dest="with_tactile", action="store_false")
    parser.set_defaults(with_tactile=True)
    parser.add_argument(
        "--visualize-tactile",
        action="store_true",
        help="Show the InspireTac tactile UV OpenCV window while collecting.",
    )
    parser.add_argument(
        "--tactile-vis-value-scale",
        type=float,
        default=0.05,
        help="Scale used by the tactile visualizer response curve.",
    )
    parser.add_argument(
        "--tactile-vis-size",
        type=_positive_int,
        default=640,
        help="Per-hand tactile visualizer panel size in pixels.",
    )
    parser.add_argument(
        "--tactile-vis-hz",
        type=_positive_float,
        default=15.0,
        help="Maximum tactile visualization update rate. Frames above this rate are skipped.",
    )
    parser.add_argument(
        "--inspire-hand-max-force",
        "--max-force",
        "--max-fingertip-force",
        dest="max_fingertip_force",
        type=_non_negative_int,
        default=DEFAULT_INSPIRE_HAND_MAX_FORCE,
        help="Max force value written to every InspireTac finger on startup.",
    )
    parser.add_argument(
        "--control-mode",
        choices=["absolute"],
        default="absolute",
        help="Arm retargeting mode. The active collection runtime uses absolute pose mapping.",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="Desk-frame translation scale.",
    )
    add_vive_tracker_height_correction_args(parser)
    parser.add_argument(
        "--hz",
        type=_positive_float,
        default=DEFAULT_TELEOP_SAMPLE_HZ,
        help="Dataset sampling rate. Defaults to 15 Hz for single-arm and bimanual collection.",
    )
    parser.add_argument(
        "--position-reference-hz",
        type=_positive_float,
        default=DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
        help=(
            "Teleop IK/safety/qpos-reference rate for position control. This clock is "
            "independent of --hz and defaults to 60 Hz."
        ),
    )
    parser.add_argument(
        "--position-reference-max-age-sec",
        type=_positive_float,
        default=0.2,
        help=(
            "Maximum age of the latest causally published qpos reference used for a "
            "15 Hz action label. Older/missing references drop the training row."
        ),
    )
    parser.add_argument(
        "--arm-control-hz",
        type=_positive_float,
        default=DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
        help="RM65 position command streaming rate beneath each qpos reference. Defaults to 120 Hz.",
    )
    parser.add_argument(
        "--dual-arm-ik-hz",
        type=_positive_float,
        default=DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
        help="Target dual-arm MultiTask IK update frequency used for solver dt and default time budget.",
    )
    parser.add_argument(
        "--arm-max-angvel",
        type=_parse_arm_max_angvel,
        default=(0.35, 0.35, 0.35, 0.5, 0.5, 0.5),
        help="Scalar or 6 comma-separated arm velocity limits in rad/s.",
    )
    parser.add_argument(
        "--return-left-qpos",
        type=_parse_qpos6,
        default=None,
        help="Optional fixed left-arm 6D joint target in radians. Press R to return to it.",
    )
    parser.add_argument(
        "--return-right-qpos",
        type=_parse_qpos6,
        default=None,
        help="Optional fixed right-arm 6D joint target in radians. Press R to return to it.",
    )
    parser.add_argument(
        "--return-left-qpos-deg",
        type=_parse_qpos6,
        default=None,
        help="Optional fixed left-arm 6D joint target in degrees. Converted to radians internally.",
    )
    parser.add_argument(
        "--return-right-qpos-deg",
        type=_parse_qpos6,
        default=None,
        help="Optional fixed right-arm 6D joint target in degrees. Converted to radians internally.",
    )
    parser.add_argument(
        "--return-home-angvel",
        type=_parse_arm_max_angvel,
        default=(0.18, 0.18, 0.18, 0.25, 0.25, 0.25),
        help="Scalar or 6 comma-separated joint velocity limits in rad/s used by R return-home.",
    )
    parser.add_argument(
        "--return-home-tolerance",
        type=_positive_float,
        default=0.01,
        help="Per-joint radian tolerance for completing R return-home.",
    )
    parser.add_argument(
        "--return-home-timeout",
        type=_positive_float,
        default=30.0,
        help="Maximum seconds allowed for an R return-home move.",
    )
    parser.add_argument(
        "--minimum-control-hz",
        type=_positive_float,
        default=15.0,
        help="Freeze when control rate drops below this threshold.",
    )
    parser.add_argument(
        "--invalid-frame-limit",
        type=_positive_int,
        default=3,
        help="Freeze after this many consecutive invalid critical tracker frames.",
    )
    parser.add_argument(
        "--recovery-healthy-frames",
        type=_positive_int,
        default=10,
        help="Number of consecutive fault-free control frames required before auto recovery starts.",
    )
    parser.add_argument(
        "--tracker-auto-hold-sec",
        type=_positive_float,
        default=0.10,
        help="Automatically hold arm motion after a tracker target is unavailable for this long.",
    )
    parser.add_argument(
        "--tracker-manual-hold-sec",
        type=_positive_float,
        default=1.0,
        help=(
            "Escalate tracker loss to a latched manual hold after this long; "
            "a new runtime home capture is then required. The default leaves a 0.1 s "
            "automatic-hold/recovery window before escalating at 1.0 s."
        ),
    )
    parser.add_argument(
        "--recovery-ramp-sec",
        type=_positive_float,
        default=1.0,
        help="Absolute-control recovery ramp duration.",
    )
    parser.add_argument(
        "--disable-dual-arm-collision-avoidance",
        action="store_true",
        help="Disable the joint-space dual-arm collision-aware retargeter.",
    )
    parser.add_argument(
        "--collision-avoidance-mode",
        choices=["boundary_slide", "freeze_on_contact"],
        default="boundary_slide",
        help="Dual-arm collision response mode.",
    )
    parser.add_argument(
        "--collision-scene-profile",
        choices=[
            "dual_arm",
            "dual_arm_hand_proxy_open",
            "dual_arm_hand_proxy_dmtac_strict",
            "dual_arm_hand",
            "dual_arm_cylinder_table",
        ],
        default="dual_arm_hand_proxy_open",
        help=(
            "Dual-arm collision scene profile. "
            "dual_arm_hand_proxy_dmtac_strict is an explicit DMTAC opt-in."
        ),
    )
    parser.add_argument(
        "--rm65-ik-framework",
        choices=["robot_motion"],
        default="robot_motion",
        help=(
            "RM65 IK implementation for arm backends and dual-arm collision IK. "
            "robot_motion routes all shared RM65 IK construction and solving through robot_motion.rm65."
        ),
    )
    parser.add_argument(
        "--camera-config",
        type=str,
        default=str(DEFAULT_CAMERA_CONFIG_PATH),
        help="JSON config for RealSense RGB cameras.",
    )
    parser.add_argument(
        "--camera-registry",
        type=str,
        default=str(DEFAULT_CAMERA_REGISTRY_PATH),
        help=(
            "Project-level camera registry with profile intrinsics and URDF frame placements. "
            "Use an empty string to disable registry snapshots."
        ),
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Disable camera startup and video export.",
    )
    parser.add_argument(
        "--vis_cam",
        "--vis-cam",
        dest="vis_cam",
        action="store_true",
        help="Show live RealSense RGB frames at the same resolution used for collection.",
    )
    parser.add_argument(
        "--vis-cam-key",
        "--vis-camera-key",
        type=str,
        default="",
        help=(
            "Camera key to visualize when --vis_cam is set. "
            "Defaults to 'front' when present, otherwise the first configured camera."
        ),
    )
    parser.add_argument(
        "--vis-cam-hz",
        type=_positive_float,
        default=15.0,
        help="Maximum camera preview update rate. Frames above this rate are skipped.",
    )
    parser.add_argument("--write-video", dest="write_video", action="store_true")
    parser.add_argument("--no-video", dest="write_video", action="store_false")
    parser.set_defaults(write_video=True)
    parser.add_argument("--write-parquet", dest="write_parquet", action="store_true")
    parser.add_argument("--no-parquet", dest="write_parquet", action="store_false")
    parser.set_defaults(write_parquet=True)
    parser.add_argument(
        "--policy-io-layout",
        type=str,
        default="",
        help=(
            "Optional DexTeleop v2.2 policy-I/O layout JSON, or literal 'auto' to resolve a "
            "strict layout from the instantiated backend/retargeter pairs. When supplied, "
            "collection keeps LeRobot-v2.1-style storage but writes typed v2.2 fields."
        ),
    )
    pose_frame_group = parser.add_mutually_exclusive_group()
    pose_frame_group.add_argument(
        "--pose-frame-contract",
        type=load_pose_frame_contract,
        default=None,
        metavar="JSON",
        help=(
            "Explicit v2.3 pose-frame contract JSON. Requires "
            "--policy-io-layout auto or a layout JSON. raw_absolute preserves pose values; "
            "desk_absolute converts only the detached writer row. Dual-RM65 collection "
            "defaults to the calibrated URDF workspace-near-boundary desk contract."
        ),
    )
    pose_frame_group.add_argument(
        "--no-pose-frame-contract",
        action="store_true",
        help=(
            "Disable the default dual-RM65 v2.3 desk contract and retain the legacy "
            "v2.2/raw collection path."
        ),
    )
    parser.add_argument(
        "--nominal-fps",
        type=_positive_float,
        default=None,
        help="Dataset nominal fps metadata. Defaults to --hz so recorded metadata matches the collection rate.",
    )
    parser.add_argument(
        "--record-actual-wrist-camera-pose",
        "--record-actual-wrist-pose",
        dest="record_actual_wrist_camera_pose",
        action="store_true",
        default=True,
        help=(
            "Annotate each dataset frame with the latest background-polled RM65 feedback "
            "TCP pose as observation.wrist_camera.* without changing training state/action fields. "
            "Enabled by default; pass --no-record-actual-wrist-camera-pose to disable."
        ),
    )
    parser.add_argument(
        "--no-record-actual-wrist-camera-pose",
        "--no-record-actual-wrist-pose",
        dest="record_actual_wrist_camera_pose",
        action="store_false",
    )
    parser.add_argument(
        "--actual-wrist-camera-pose-hz",
        "--actual-wrist-pose-hz",
        type=_positive_float,
        default=30.0,
        help="Background feedback polling rate for actual wrist camera pose annotation.",
    )
    parser.add_argument(
        "--actual-wrist-camera-pose-max-age-sec",
        "--actual-wrist-pose-max-age-sec",
        type=_positive_float,
        default=0.2,
        help="Mark actual wrist camera pose annotations stale when their sample-time age exceeds this value.",
    )
    parser.add_argument(
        "--actual-wrist-camera-pose-warmup-timeout-sec",
        "--actual-wrist-pose-warmup-timeout-sec",
        type=_non_negative_float,
        default=2.0,
        help=(
            "Before an episode starts, wait up to this duration for each active arm to have "
            "a fresh causal actual TCP feedback sample. Set to 0 to only check once."
        ),
    )
    parser.add_argument(
        "--replay-dataset",
        type=str,
        default="",
        help=(
            "Enable trajectory replay collection from a source dataset root. "
            "Press replay group keys to start automatic episodes."
        ),
    )
    parser.add_argument(
        "--replay-parquet",
        type=str,
        default="",
        help="Enable trajectory replay collection from a single source parquet file.",
    )
    parser.add_argument(
        "--replay-episode",
        type=int,
        default=0,
        help=(
            "Source episode index used with --replay-dataset when balanced replay is disabled, "
            "and with --replay-parquet metadata."
        ),
    )
    parser.add_argument(
        "--replay-all-episodes",
        dest="replay_all_episodes",
        action="store_true",
        help=(
            "With --replay-dataset, build each replay key from all dataset episodes and "
            "cycle through them evenly. Default."
        ),
    )
    parser.add_argument(
        "--replay-single-episode",
        dest="replay_all_episodes",
        action="store_false",
        help="Use only --replay-episode, preserving the older fixed-source behavior.",
    )
    parser.set_defaults(replay_all_episodes=True)
    parser.add_argument(
        "--replay-action-column",
        type=str,
        default="action",
        help="Source parquet full action column name; falls back to component action columns.",
    )
    parser.add_argument(
        "--replay-group",
        action="append",
        default=[],
        help=(
            "Replay key/source-episode mapping 'key:episode_range'. "
            "Episode ranges support 'all', '0-49', '2,4,8'. "
            "Each selected source episode is replayed from start to end. Defaults to 'a:all'."
        ),
    )
    parser.add_argument(
        "--replay-init-time",
        "--default-init-time",
        dest="replay_init_time",
        type=float,
        default=1.0,
        help="Seconds used to smoothly move to the first replay pose; this phase is recorded.",
    )
    parser.add_argument(
        "--replay-max-arm-xyz-step-m",
        type=float,
        default=0.0,
        help="Optional replay arm Cartesian per-step limiter; 0 disables extra limiting.",
    )
    parser.add_argument(
        "--replay-max-arm-rot-step-deg",
        type=float,
        default=0.0,
        help="Optional replay arm rotation per-step limiter; 0 disables extra limiting.",
    )
    parser.add_argument(
        "--replay-arm-max-angvel",
        type=_parse_arm_max_angvel,
        default=(3.0, 3.0, 3.0, 4.0, 4.0, 4.0),
        help=(
            "Temporary RM65 joint velocity limits in rad/s used while executing replay/init "
            "fixed trajectories. Higher values let hardware follow the source action Hz."
        ),
    )
    parser.add_argument(
        "--replay-randomize",
        action="store_true",
        help="Enable bounded smooth random offsets for replay arm poses.",
    )
    parser.add_argument(
        "--replay-random-seed",
        type=int,
        default=-1,
        help="Seed for replay randomization; negative uses OS entropy.",
    )
    parser.add_argument(
        "--replay-random-mode",
        choices=["episode_offset", "smooth_noise", "both"],
        default="both",
        help="Replay randomization mode.",
    )
    parser.add_argument(
        "--replay-random-xyz-m",
        type=_non_negative_float,
        default=0.01,
        help="Max absolute replay arm position offset in meters.",
    )
    parser.add_argument(
        "--replay-random-rot-deg",
        type=_non_negative_float,
        default=2.0,
        help="Max replay arm rotation offset in degrees.",
    )
    parser.add_argument(
        "--replay-random-noise-xyz-m",
        type=float,
        default=-1.0,
        help=(
            "Max smooth-noise xyz offset in meters; negative uses half of --replay-random-xyz-m."
        ),
    )
    parser.add_argument(
        "--replay-random-noise-rot-deg",
        type=float,
        default=-1.0,
        help=(
            "Max smooth-noise rotation offset in degrees; negative uses half of --replay-random-rot-deg."
        ),
    )
    parser.add_argument(
        "--replay-random-refresh-sec",
        type=_positive_float,
        default=0.8,
        help="Seconds between smooth random replay offset targets.",
    )
    parser.add_argument(
        "--replay-random-smoothing-sec",
        type=_non_negative_float,
        default=0.3,
        help="Seconds used to interpolate toward each smooth random offset target.",
    )
    parser.add_argument(
        "--replay-random-independent-sides",
        action="store_true",
        help="Use independent random offsets per arm side instead of one shared task-space offset.",
    )
    parser.add_argument(
        "--replay-init-episode",
        "--init-episode",
        dest="replay_init_episode",
        type=str,
        default="",
        help=(
            "Optional unrecorded replay pre-initialization trajectory T. "
            "Format 'episode[:start:end]' with start/end/mid tokens, for example '0' or '3:0:end'. "
            "In replay mode, press H once to move safely to T[0], then H again to execute T without recording."
        ),
    )
    parser.add_argument(
        "--replay-init-dataset",
        type=str,
        default="",
        help=(
            "Optional dataset root used only for --replay-init-episode. "
            "Defaults to --replay-dataset when omitted."
        ),
    )
    parser.add_argument(
        "--replay-init-parquet",
        type=str,
        default="",
        help=(
            "Optional single parquet used only for --replay-init-episode. "
            "Use only one of --replay-init-dataset or --replay-init-parquet."
        ),
    )
    parser.add_argument(
        "--replay-init-episode-move-time",
        type=_positive_float,
        default=3.0,
        help=(
            "Legacy first-H move-time value. Current first H uses strict qpos targets from "
            "action.replay.<side>_arm_joint_target instead of end-effector interpolation; not recorded."
        ),
    )
    parser.add_argument(
        "--replay-init-episode-to-record-time",
        type=_non_negative_float,
        default=1.0,
        help=(
            "Seconds used to move from init-episode T end pose to the selected A/D replay start pose; "
            "not recorded in init-episode mode."
        ),
    )
    parser.add_argument(
        "--replay-init-episode-require-ready",
        dest="replay_init_episode_require_ready",
        action="store_true",
        help="Require the H init-episode sequence to complete before replay group keys can start recording. Default.",
    )
    parser.add_argument(
        "--no-replay-init-episode-require-ready",
        dest="replay_init_episode_require_ready",
        action="store_false",
        help="Allow replay group keys to start recording even if the H init-episode sequence has not completed.",
    )
    parser.set_defaults(replay_init_episode_require_ready=True)
    parser.add_argument(
        "--writer-workers", type=int, default=2, help="Async writer worker count."
    )
    parser.add_argument(
        "--writer-queue-size", type=int, default=256, help="Async writer queue size."
    )
    parser.add_argument(
        "--storage-queue-size",
        type=_non_negative_int,
        default=64,
        help=(
            "Per-camera video storage queue size before the async writer pool. "
            "Use 0 for direct submit."
        ),
    )
    parser.add_argument(
        "--storage-queue-backpressure-policy",
        choices=("block", "drop_oldest"),
        default="block",
        help=(
            "Queue-full behavior. block preserves required-camera row alignment; "
            "drop_oldest is legacy best-effort storage and may create frame gaps."
        ),
    )
    parser.add_argument(
        "--tactile-storage",
        choices=("parquet", "external"),
        default="external",
        help=(
            "How high-dimensional tactile frames are stored. external is the default and writes "
            "binary per-sensor streams; parquet keeps the legacy inline vector."
        ),
    )
    parser.add_argument(
        "--external-tactile",
        dest="tactile_storage",
        action="store_const",
        const="external",
        help="Shortcut for --tactile-storage external.",
    )
    parser.add_argument(
        "--tactile-compression",
        choices=("zstd", "lz4", "none"),
        default="none",
        help=(
            "Compression for external tactile streams. none is the default and writes "
            "raw .bin streams; use lz4 or zstd for lossless chunk compression."
        ),
    )
    parser.add_argument(
        "--calibration-timeout",
        type=_positive_float,
        default=10.0,
        help="Timeout for runtime home calibration after startup.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=_positive_int,
        default=20,
        help="Number of valid wrist frames to average per moving arm side during runtime home calibration.",
    )
    parser.add_argument(
        "--max-steps",
        type=_non_negative_int,
        default=0,
        help="Stop after N control steps, 0 means infinite.",
    )
    parser.add_argument(
        "--status-sec",
        type=_positive_float,
        default=2.0,
        help="Status print interval in seconds.",
    )
    parser.add_argument(
        "--debug-frozen",
        "--debug_frozen",
        dest="debug_frozen",
        action="store_true",
        default=False,
        help=(
            "Enable red stall diagnostics. When active loop/save work exceeds "
            "--debug-frozen-threshold-sec, print phase timings and device state snapshots."
        ),
    )
    parser.add_argument(
        "--debug-frozen-threshold-sec",
        "--debug_frozen_threshold_sec",
        dest="debug_frozen_threshold_sec",
        type=_positive_float,
        default=0.1,
        help="Stall threshold in seconds for --debug-frozen diagnostics.",
    )
    parser.add_argument(
        "--debug-frozen-report-interval-sec",
        "--debug_frozen_report_interval_sec",
        dest="debug_frozen_report_interval_sec",
        type=_non_negative_float,
        default=1.0,
        help="Minimum seconds between repeated --debug-frozen reports; use 0 to print every stall.",
    )
    parser.add_argument(
        "--silent",
        "--quiet",
        dest="silent",
        action="store_true",
        default=True,
        help="Suppress periodic status lines; key events and warnings are still printed. Default.",
    )
    parser.add_argument(
        "--no-silent",
        "--no-quiet",
        "--verbose-status",
        dest="silent",
        action="store_false",
        help="Print periodic status lines every --status-sec seconds.",
    )

    return parser
