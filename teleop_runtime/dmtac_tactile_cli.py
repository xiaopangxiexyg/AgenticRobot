"""Side-effect-free DM-Tac tactile CLI options shared by application entrypoints."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable

from .dmtac_tactile_fields import DMTAC_TACTILE_FIELDS


def _dmtac_worker_cpu_affinity(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in re.split(r"[,\s]+", str(text)) if part.strip()]
    if not parts:
        return ()
    cpus: list[int] = []
    for part in parts:
        try:
            cpu = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("CPU affinity must be a comma-separated integer list") from exc
        if cpu < 0:
            raise argparse.ArgumentTypeError("CPU affinity indices must be >= 0")
        cpus.append(cpu)
    return tuple(dict.fromkeys(cpus))


def _dmtac_worker_nice(text: str) -> int:
    value = int(text)
    if not 0 <= value <= 19:
        raise argparse.ArgumentTypeError("worker nice must be in [0, 19]")
    return value


def add_dmtac_tactile_arguments(
    parser: argparse.ArgumentParser,
    *,
    positive_int: Callable[[str], int],
    non_negative_int: Callable[[str], int],
    non_negative_float: Callable[[str], float],
    worker_cpu_affinity: Callable[[str], tuple[int, ...]],
    worker_nice: Callable[[str], int],
    parse_tactile_fields: Callable[[str], tuple[str, ...]],
    read_hz_help: str,
    in_process_help: str,
    enable_help: str | None = None,
    fixed_calibration: bool = False,
) -> None:
    """Register common tactile options without reading devices or changing defaults."""

    parser.add_argument(
        "--dmtac-tactile-backend",
        type=str,
        default="cpu",
        help="DM-Tac tactile backend passed to portable_dmtac_gripper.",
    )
    parser.add_argument(
        "--dmtac-tactile-timeout-ms",
        type=positive_int,
        default=None,
        help="Per-sensor DM-Tac read timeout. Runtime telemetry does not wait for new frames by default.",
    )
    parser.add_argument(
        "--dmtac-tactile-max-fps",
        type=positive_int,
        default=None,
        help="DM-Tac camera max_fps. Lower values reduce USB/CPU pressure; default follows backend config.",
    )
    parser.add_argument(
        "--dmtac-tactile-warmup-frames",
        type=non_negative_int,
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
        type=non_negative_float,
        default=None,
        help=read_hz_help,
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
        help=in_process_help,
    )
    parser.add_argument(
        "--dmtac-tactile-worker-cpus",
        type=worker_cpu_affinity,
        default=None,
        help=(
            "Comma/space-separated CPU cores for low-priority DM-Tac tactile worker processes, "
            "for example '6,7'. Unset lets the OS schedule normally."
        ),
    )
    parser.add_argument(
        "--dmtac-tactile-worker-nice",
        type=worker_nice,
        default=None,
        help="Lower scheduling priority for DM-Tac tactile worker processes; 0 is normal, 19 is lowest.",
    )
    parser.add_argument(
        "--dmtac-tactile-worker-queue-size",
        type=positive_int,
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
        type=parse_tactile_fields,
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
    if fixed_calibration:
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
        help=enable_help,
    )
    parser.add_argument(
        "--dmtac-no-tactile", dest="dmtac_tactile", action="store_false"
    )


__all__ = ["add_dmtac_tactile_arguments"]
