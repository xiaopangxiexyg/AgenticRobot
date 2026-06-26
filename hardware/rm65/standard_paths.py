from __future__ import annotations

from pathlib import Path


RM65_DIR = Path(__file__).resolve().parent
UTILS_DIR = RM65_DIR / "utils"
DATA_DIR = UTILS_DIR / "data"
POSE_DIR = DATA_DIR / "poses"
MEASUREMENTS_DIR = DATA_DIR / "measurements"
TRAJECTORY_DIR = DATA_DIR / "trajectories"
RESULTS_DIR = UTILS_DIR / "results"


def force_sensor_calib4_pose_sequence_path(arm: str) -> Path:
    return POSE_DIR / f"rm65_{arm}_force_sensor_calib4_pose_sequence.json"


def force_sensor_calib4_measurements_path(arm: str, tool: str) -> Path:
    return MEASUREMENTS_DIR / f"rm65_{arm}_{tool}_force_sensor_calib4_measurements.json"


def force_measurements_path(arm: str, tool: str) -> Path:
    return MEASUREMENTS_DIR / f"rm65_{arm}_{tool}_force_measurements.json"


def force_sensor_calib4_fit_path(arm: str, tool: str) -> Path:
    return RESULTS_DIR / f"rm65_{arm}_{tool}_force_sensor_calib4_measurements_calibration_fit.json"


def force_measurements_fit_path(arm: str, tool: str) -> Path:
    return RESULTS_DIR / f"rm65_{arm}_{tool}_force_measurements_calibration_fit.json"


def official_force_sensor_report_path(arm: str, tool: str) -> Path:
    return RESULTS_DIR / f"rm65_{arm}_{tool}_official_force_sensor_calibration_report.json"


def tool_payload_validation_path(arm: str, tool: str) -> Path:
    return RESULTS_DIR / f"rm65_{arm}_{tool}_tool_payload_validation.json"


def teach_trajectory_path(arm: str) -> Path:
    return TRAJECTORY_DIR / f"rm65_{arm}_teach_trajectory.json"


def fixed_point_impedance_tune_path(arm: str, tool: str, mode: str) -> Path:
    return RESULTS_DIR / f"rm65_{arm}_{tool}_fixed_{mode}_impedance_tune.json"


def trajectory_impedance_replay_path(arm: str, mode: str) -> Path:
    return RESULTS_DIR / f"rm65_{arm}_teach_trajectory_{mode}_impedance_replay.json"


def candidate_fit_paths(arm: str, tool: str | None) -> list[Path]:
    if not tool:
        return []
    return [
        force_sensor_calib4_fit_path(arm, tool),
        force_measurements_fit_path(arm, tool),
    ]


def resolve_fit_path(arm: str, tool: str | None, *, require_existing: bool = True) -> Path:
    for path in candidate_fit_paths(arm, tool):
        if path.exists():
            return path
    if tool:
        matches = sorted(
            RESULTS_DIR.glob(f"rm65_{arm}_{tool}_*_calibration_fit.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]
        if not require_existing:
            return force_sensor_calib4_fit_path(arm, tool)
    matches = sorted(
        RESULTS_DIR.glob(f"rm65_{arm}_*_calibration_fit.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    if tool and not require_existing:
        return force_sensor_calib4_fit_path(arm, tool)
    raise RuntimeError(f"no calibration fit found for arm={arm}, tool={tool or '(any)'}")
