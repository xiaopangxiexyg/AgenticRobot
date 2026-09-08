"""CLI argument groups for real hand, gripper, and tactile teleoperation.

The functions in this module only register argparse actions.  They do not
parse arguments, probe devices, load assignments, or construct backends.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..dmtac_tactile_fields import DMTAC_TACTILE_FIELDS
from ..cli_value_parsers import (
    _alpha_float,
    _non_negative_int,
    _parse_joint_order,
    _positive_float,
)
from ..configs.dmtac_hardware import add_dmtac_assignment_arguments
from ..hand_mapping import (
    DEFAULT_INSPIRE_HAND_JOINT_ORDER,
    TARGET_INSPIRE_HAND_Q6_ORDER,
)
from .cli_value_parsers import (
    _dmtac_worker_cpu_affinity,
    _dmtac_worker_nice,
    _non_negative_float,
    _parse_dmtac_tactile_fields,
    _positive_int,
)
from .hand_config import HAND_BACKEND_CHOICES, HAND_BACKEND_INSPIRE


DEFAULT_TASKS_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "tasks"
    / "vive_rm65_real_teleop_tasks.json"
)


class _GripperInputSourceAction(argparse.Action):
    """Accept the new source selector while keeping the legacy flag shape."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | None,
        option_string: str | None = None,
    ) -> None:
        option = str(option_string or "")
        if option == "--gripper-input-source" and values is None:
            parser.error(
                "--gripper-input-source requires q6, angle_trigger, or none"
            )
        source = "angle_trigger" if values is None else str(values)
        previous = getattr(namespace, "gripper_input_source", None)
        if previous not in (None, source):
            parser.error(
                "--gripper-angle-trigger and --gripper-input-source select "
                "conflicting gripper inputs"
            )
        setattr(namespace, "gripper_input_source", source)
        # Preserve the boolean attribute consumed by older callers that build
        # an argparse namespace before the source is resolved.
        setattr(
            namespace,
            "gripper_angle_trigger",
            bool(option == "--gripper-angle-trigger" and values is None),
        )


def add_hand_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hand-backend",
        choices=HAND_BACKEND_CHOICES,
        default=HAND_BACKEND_INSPIRE,
        help=(
            "Hand/gripper backend type. Use dmtac-gripper to expose the DMTAC "
            "1D gripper backend and browser GUI control/status."
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
            "Q6 left/right assignment config used by --hand-backend dmtac-gripper. "
            "Explicit --left/right-hand-port wins."
        ),
    )
    parser.add_argument(
        "--no-q6-default-config",
        dest="q6_default_config",
        action="store_false",
        default=True,
        help="Do not auto-fill missing Q6 serial ports from the latest saved assignment.",
    )
    parser.add_argument(
        "--q6-config-required",
        action="store_true",
        help="Fail if --q6-config cannot be loaded while a DMTAC gripper side is missing a Q6 port.",
    )
    parser.add_argument(
        "--gripper-angle-trigger",
        "--gripper-input-source",
        dest="gripper_input_source",
        nargs="?",
        choices=("q6", "angle_trigger", "none"),
        metavar="{q6,angle_trigger,none}",
        default=None,
        action=_GripperInputSourceAction,
        help=(
            "Select the DMTAC gripper input explicitly: q6, angle_trigger, or "
            "none. The none mode keeps configured gripper state/tactile backends "
            "but creates no human gripper input source or retargeter. "
            "The legacy bare --gripper-angle-trigger selects angle_trigger; "
            "omit both options to preserve automatic Q6/AngleTrigger discovery."
        ),
    )
    # Keep the legacy boolean namespace field present when neither selector is
    # supplied; older callers read it before the resolver runs.
    parser.set_defaults(gripper_angle_trigger=False)
    parser.add_argument(
        "--gripper-angle-trigger-config-root",
        type=Path,
        default=None,
        help="Optional angle-trigger assignment/angle-calibration directory.",
    )
    parser.add_argument(
        "--gripper-angle-trigger-rate-hz",
        type=_positive_float,
        default=30.0,
        help="Angle-trigger reader polling rate.",
    )
    parser.add_argument(
        "--q6-hold-recalibration-gesture",
        action="store_true",
        help=(
            "Enable dual-arm Q6 gesture control: close both hands' q6 dims 4/5/6 for "
            "1s to hold, release them for 1s while holding, then close again for 1s "
            "to re-capture runtime home and resume teleop."
        ),
    )
    parser.add_argument(
        "--q6-gesture-hold-sec",
        type=_positive_float,
        default=1.0,
        help="Required continuous Q6 gesture duration for hold/release/restart.",
    )
    parser.add_argument(
        "--q6-gesture-close-threshold",
        type=float,
        default=0.95,
        help="Q6 dims 4/5/6 must be above this value on both hands to count as closed.",
    )
    parser.add_argument(
        "--q6-gesture-open-threshold",
        type=float,
        default=0.85,
        help=(
            "Q6 dims 4/5/6 must be below this value on both hands to count as released "
            "between hold/restart gestures."
        ),
    )

def add_dmtac_arguments(parser: argparse.ArgumentParser) -> None:
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
            "Max DM-Tac tactile read rate per gripper backend. Formal collection defaults "
            "to --hz (capped by max_fps); non-collection tools retain the 1 Hz backend "
            "default. Use 0 to read every gripper telemetry cycle."
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
        help="Read DM-Tac tactile frames in the main teleop process.",
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
        "--dmtac-tactile-calibration-policy",
        choices=("auto", "required", "off"),
        default=None,
        help=(
            "Fixed per-sensor base-frame policy. GUI distributed-force solving uses "
            "'required' so an uncalibrated sensor cannot be presented as calibrated."
        ),
    )
    parser.add_argument(
        "--dmtac-tactile-calibration-dir",
        default="",
        help="Optional directory containing fixed per-sensor DM-Tac base-frame calibrations.",
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
        "--dmtac-with-tactile",
        dest="dmtac_tactile",
        action="store_true",
        default=None,
        help=(
            "Opt in to DM-Tac tactile sensors during real teleop. Off by default so gripper/Q6 "
            "control does not depend on the DM-Tac SDK camera threads."
        ),
    )
    parser.add_argument("--dmtac-no-tactile", dest="dmtac_tactile", action="store_false")
    parser.add_argument(
        "--no-device-preflight",
        dest="device_preflight",
        action="store_false",
        default=True,
        help="Skip GUI startup USB/camera/Q6/DMTAC assignment validation.",
    )
    parser.add_argument(
        "--device-fix-permissions",
        action="store_true",
        help="Before GUI startup preflight, try setfacl on serial/video nodes for the current user.",
    )
    parser.add_argument(
        "--no-device-auto-configure",
        dest="device_auto_configure",
        action="store_false",
        default=True,
        help="Only validate devices; do not enter Q6/DMTAC/RealSense assignment flows.",
    )
    parser.add_argument(
        "--device-preflight-yes",
        action="store_true",
        help="Use default left/right assignment choices during GUI device preflight.",
    )
    parser.add_argument(
        "--device-preflight-probe-all-serial",
        action="store_true",
        help=(
            "Probe all current USB serial ports during GUI device preflight. Useful after USB "
            "disconnect/reconnect when descriptors or by-id aliases changed."
        ),
    )
    parser.add_argument(
        "--device-preflight-force-reconfigure",
        action="store_true",
        help=(
            "Ignore existing valid Q6/DMTAC assignments during GUI preflight and re-run "
            "their selection flows, saving refreshed configs. RealSense and arm IP configs stay unchanged."
        ),
    )
    parser.add_argument(
        "--device-preflight-no-tactile",
        dest="device_preflight_require_tactile",
        action="store_false",
        default=True,
        help="Do not require DMTAC tactile sensors during GUI device preflight.",
    )

def add_gripper_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gripper-task-profile",
        type=str,
        default="",
        help=(
            "Optional JSON profile with a top-level gripper object and/or tasks.<name>.gripper "
            "overrides. It can set speed, force/torque, max_action, send_threshold, command_hz, "
            "continuous_grasp, source_dim, output_dim, invert, alpha, input_min, input_max."
        ),
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default=str(DEFAULT_TASKS_CONFIG_PATH),
        help=(
            "JSON task registry for the GUI. Each task defines key, english_name, "
            "chinese_name, and gripper_task."
        ),
    )
    parser.add_argument(
        "--gripper-task",
        "--task-id",
        "--task-name",
        dest="gripper_task",
        type=str,
        default="",
        help="Task key used to select tasks.<key>.gripper from --gripper-task-profile.",
    )
    parser.add_argument("--gripper-source-dim", type=int, default=None, help="Q6 source dimension for 1D gripper.")
    parser.add_argument("--gripper-output-dim", type=int, default=None, help="Output slot in the 6-wide hand vector.")
    parser.add_argument("--gripper-invert", dest="gripper_invert", action="store_true", default=None)
    parser.add_argument("--gripper-no-invert", dest="gripper_invert", action="store_false")
    parser.add_argument("--gripper-alpha", type=_alpha_float, default=None, help="EMA alpha for 1D gripper command.")
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
    parser.add_argument("--gripper-speed", type=float, default=None, help="Gripper move speed for this task.")
    parser.add_argument(
        "--gripper-torque",
        "--gripper-force",
        dest="gripper_torque",
        type=float,
        default=None,
        help="Gripper torque/force setting for this task.",
    )
    parser.add_argument("--gripper-max-action", type=float, default=None, help="Max normalized close command.")
    parser.add_argument(
        "--gripper-send-threshold",
        type=float,
        default=None,
        help="Minimum normalized delta before sending another gripper command.",
    )
    parser.add_argument("--gripper-continuous", dest="gripper_continuous_grasp", action="store_true", default=None)
    parser.add_argument("--gripper-no-continuous", dest="gripper_continuous_grasp", action="store_false")
    parser.add_argument("--gripper-command-hz", type=_positive_float, default=None)
    parser.add_argument("--gripper-telemetry-hz", type=_positive_float, default=None)
    parser.add_argument(
        "--gripper-telemetry-command-grace-s", type=float, default=None
    )
    parser.add_argument(
        "--gripper-warmup-probe",
        dest="gripper_warmup_probe",
        action="store_true",
        default=True,
        help=(
            "During recording warmup, command DMTAC grippers open->half-close->open "
            "and reconnect any gripper whose measured state does not follow."
        ),
    )
    parser.add_argument(
        "--no-gripper-warmup-probe",
        dest="gripper_warmup_probe",
        action="store_false",
        help="Disable the DMTAC gripper motion probe during recording warmup.",
    )
    parser.add_argument(
        "--gripper-warmup-probe-tolerance",
        type=float,
        default=0.18,
        help="Allowed normalized error when validating DMTAC gripper warmup motion.",
    )
    parser.add_argument(
        "--gripper-warmup-probe-step-timeout-sec",
        type=_positive_float,
        default=2.0,
        help="Timeout per open/half/open gripper warmup step.",
    )
    parser.add_argument(
        "--gripper-warmup-probe-settle-sec",
        type=float,
        default=0.20,
        help="Settling delay after each gripper warmup command before reading state.",
    )
    parser.add_argument(
        "--gripper-warmup-probe-poll-sec",
        type=_positive_float,
        default=0.05,
        help="Polling interval while waiting for DMTAC gripper warmup state.",
    )

def add_inspire_hand_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hand-reader-baudrate", type=int, default=115200, help="Hand teleop serial baudrate")
    parser.add_argument("--hand-reader-timeout", type=float, default=0.05, help="Hand teleop serial timeout in seconds")
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
    parser.add_argument("--show-hand-raw-lines", action="store_true", help="Print every raw line from hand teleop serial devices")
    parser.add_argument(
        "--hand-joint-order",
        type=_parse_joint_order,
        default=DEFAULT_INSPIRE_HAND_JOINT_ORDER,
        help=(
            "Permutation applied to hand q6 before mapping to InspireTac q6. "
            f"Default maps into InspireTac order {list(TARGET_INSPIRE_HAND_Q6_ORDER)}."
        ),
    )
    parser.add_argument(
        "--hand-invert",
        dest="hand_invert",
        action="store_true",
        help="Invert normalized q6 before sending to the InspireTac hand. Default is enabled.",
    )
    parser.add_argument(
        "--hand-no-invert",
        dest="hand_invert",
        action="store_false",
        help="Disable q6 inversion if the hand already moves in the correct direction.",
    )
    parser.set_defaults(hand_invert=True)
    parser.add_argument(
        "--hand-command-alpha",
        type=_alpha_float,
        default=1.0,
        help="EMA smoothing factor for hand q6 mapping.",
    )
    parser.add_argument(
        "--hand-send-threshold",
        type=float,
        default=0.002,
        help="Only send/update hand q6 when max delta exceeds this threshold.",
    )
    parser.add_argument(
        "--hand-telemetry-hz",
        type=_positive_float,
        default=30.0,
        help="Telemetry polling rate for each real InspireTac hand backend.",
    )
    parser.add_argument(
        "--hand-tactile-regions",
        type=str,
        default="all",
        help=(
            "Comma-separated tactile regions to poll, for example "
            "'thumb_1,index_1,palm' or 'all'. Defaults to all; pass an empty string to skip tactile reads."
        ),
    )
    parser.add_argument(
        "--print-tactile-sum-only",
        action="store_true",
        help=(
            "Suppress normal stdout logs and print only real-time left/right hand tactile sums. "
            "Teleop behavior is unchanged. Disabled by default."
        ),
    )
    parser.add_argument(
        "--visualize-tactile",
        action="store_true",
        help=(
            "Show an OpenCV tactile window during real teleop. InspireTac uses the UV layout; "
            "DMTAC gripper uses 20x30x3 force grids: color=max(Fy,0), arrows=(Fx,Fz)."
        ),
    )
    parser.add_argument(
        "--tactile-vis-value-scale",
        type=float,
        default=0.05,
        help="Scale used by the tactile visualizer response curve.",
    )
    parser.add_argument(
        "--tactile-vis-size",
        type=int,
        default=640,
        help="Per-hand tactile visualizer panel size in pixels.",
    )
    parser.add_argument(
        "--tactile-vis-hz",
        type=_positive_float,
        default=5.0,
        help="Maximum tactile visualization update rate.",
    )
    parser.add_argument(
        "--dmtac-tactile-vis-fixed-ref",
        type=_positive_float,
        default=1.0,
        help=(
            "Fixed common DM-TAC force visualization scale used for every sensor "
            "and frame (default: 1.0). This keeps the color map and displayed "
            "reference stable instead of amplifying per-frame startup noise."
        ),
    )

__all__ = [
    "DEFAULT_TASKS_CONFIG_PATH",
    "add_dmtac_arguments",
    "add_gripper_policy_arguments",
    "add_hand_backend_arguments",
    "add_inspire_hand_arguments",
]
