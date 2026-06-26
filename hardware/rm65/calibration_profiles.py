from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable


RM65_DIR = Path(__file__).resolve().parent
PROFILE_DIR = RM65_DIR / "utils" / "configs" / "calibrations"
ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HAND_BASE_TCP_POSE_FLANGE_M_RAD = [0.0, 0.0, 0.009, 0.0, 0.0, 0.0]
DEFAULT_MAX_TCP_COM_DISTANCE_M = 0.20
DEFAULT_TCP_SAFETY_MAX_ENDPOINT_DISTANCE_M = 0.20
DEFAULT_TCP_SAFETY_MAX_ROTATION_DEG = 90.0


def normalize_profile_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    value = value.strip("._-")
    if not value:
        raise ValueError("calibration profile name must not be empty")
    return value


def finite_float_list(values: Iterable[float], *, expected_len: int, label: str) -> list[float]:
    out = [float(value) for value in values]
    if len(out) != expected_len:
        raise ValueError(f"{label} must contain {expected_len} values, got {len(out)}")
    if not all(math.isfinite(value) for value in out):
        raise ValueError(f"{label} contains non-finite values: {out}")
    return out


def vector_norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _rotation_angle_deg_from_euler_xyz(rx: float, ry: float, rz: float) -> float:
    rot = _rotation_matrix_from_euler_xyz(rx, ry, rz)
    trace = rot[0][0] + rot[1][1] + rot[2][2]
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(cos_angle))


def tcp_endpoint_delta(
    pose_flange_m_rad: Iterable[float],
) -> dict[str, float]:
    pose = finite_float_list(
        pose_flange_m_rad,
        expected_len=6,
        label="tcp.pose_flange_m_rad",
    )
    return {
        "translation_m": vector_norm(pose[:3]),
        "rotation_deg": _rotation_angle_deg_from_euler_xyz(*pose[3:6]),
    }


def tcp_safety_warnings(
    pose_flange_m_rad: Iterable[float],
    *,
    max_endpoint_distance_m: float = DEFAULT_TCP_SAFETY_MAX_ENDPOINT_DISTANCE_M,
    max_rotation_deg: float = DEFAULT_TCP_SAFETY_MAX_ROTATION_DEG,
    label: str = "tool TCP",
) -> list[str]:
    metrics = tcp_endpoint_delta(pose_flange_m_rad)
    warnings: list[str] = []
    if metrics["translation_m"] > float(max_endpoint_distance_m):
        warnings.append(
            f"{label} is {metrics['translation_m']:.6f}m from arm_endpoint, "
            f"exceeding {float(max_endpoint_distance_m):.6f}m"
        )
    if metrics["rotation_deg"] > float(max_rotation_deg):
        warnings.append(
            f"{label} rotation differs from arm_endpoint by {metrics['rotation_deg']:.3f}deg, "
            f"exceeding {float(max_rotation_deg):.3f}deg"
        )
    return warnings


def _rotation_matrix_from_euler_xyz(rx: float, ry: float, rz: float) -> list[list[float]]:
    sx, cx = math.sin(float(rx)), math.cos(float(rx))
    sy, cy = math.sin(float(ry)), math.cos(float(ry))
    sz, cz = math.sin(float(rz)), math.cos(float(rz))
    rx_mat = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
    ry_mat = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    rz_mat = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
    return _matmul3(_matmul3(rz_mat, ry_mat), rx_mat)


def _matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]


def _matvec3(a: list[list[float]], b: list[float]) -> list[float]:
    return [sum(a[row][k] * b[k] for k in range(3)) for row in range(3)]


def tcp_pose_from_tool_pose_and_offset(
    tool_pose_flange_m_rad: Iterable[float],
    offset_tool_m: Iterable[float],
) -> list[float]:
    """Return flange->TCP pose when TCP is offset from the current tool origin.

    RM65 tool poses use meters and XYZ Euler angles in radians.  The output
    preserves the input orientation and only changes translation.
    """

    pose = finite_float_list(tool_pose_flange_m_rad, expected_len=6, label="tool_pose_flange_m_rad")
    offset = finite_float_list(offset_tool_m, expected_len=3, label="offset_tool_m")
    rot = _rotation_matrix_from_euler_xyz(*pose[3:6])
    shifted = _matvec3(rot, offset)
    return [pose[index] + shifted[index] for index in range(3)] + pose[3:6]


def hand_base_tcp() -> dict[str, Any]:
    now = time.time()
    return {
        "schema_version": 1,
        "frame_name": "hand_base",
        "pose_flange_m_rad": list(HAND_BASE_TCP_POSE_FLANGE_M_RAD),
        "position_flange_m": list(HAND_BASE_TCP_POSE_FLANGE_M_RAD[:3]),
        "rotation_flange_rpy_rad": list(HAND_BASE_TCP_POSE_FLANGE_M_RAD[3:6]),
        "position_source": "hand_base_default",
        "rotation_policy": "preserve_existing_tool_rotation",
        "confirmed": False,
        "updated_at": now,
    }


def arm_endpoint_tcp() -> dict[str, Any]:
    now = time.time()
    return {
        "schema_version": 1,
        "frame_name": "tool_tcp",
        "pose_flange_m_rad": list(ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD),
        "position_flange_m": list(ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD[:3]),
        "rotation_flange_rpy_rad": list(ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD[3:6]),
        "position_source": "arm_endpoint_default",
        "rotation_policy": "arm_endpoint_identity",
        "confirmed": False,
        "updated_at": now,
    }


def make_payload(
    *,
    mass_kg: float,
    com_tool_m: Iterable[float],
    source_fit_json: str = "",
    fit_source: str = "",
    fit_mode: str = "",
) -> dict[str, Any]:
    mass = float(mass_kg)
    if not math.isfinite(mass) or mass < 0.0:
        raise ValueError(f"invalid payload mass_kg: {mass}")
    return {
        "schema_version": 1,
        "mass_kg": mass,
        "com_tool_m": finite_float_list(com_tool_m, expected_len=3, label="com_tool_m"),
        "source_fit_json": str(source_fit_json),
        "fit_source": str(fit_source),
        "fit_mode": str(fit_mode),
        "updated_at": time.time(),
    }


def make_tcp_from_fit(
    *,
    tool_name: str,
    tool_pose_flange_m_rad: Iterable[float],
    com_tool_m: Iterable[float],
    source_fit_json: str = "",
    seed_source: str = "auto",
    max_com_distance_m: float = DEFAULT_MAX_TCP_COM_DISTANCE_M,
) -> dict[str, Any]:
    """Build a TCP profile entry from a payload fit and current tool pose.

    ``com_tool_m`` is a vector from the current tool origin to the fitted
    payload CoM.  A COM-seeded TCP therefore preserves the current tool
    orientation and shifts the current tool pose by that vector.
    """

    if seed_source not in ("auto", "com", "hand-base", "arm-endpoint"):
        raise ValueError("seed_source must be auto, com, hand-base, or arm-endpoint")
    pose = finite_float_list(tool_pose_flange_m_rad, expected_len=6, label="tool_pose_flange_m_rad")
    com = finite_float_list(com_tool_m, expected_len=3, label="com_tool_m")
    com_distance = vector_norm(com)
    max_distance = float(max_com_distance_m)
    if not math.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("max_com_distance_m must be positive")

    normalized_tool = normalize_profile_name(tool_name).lower()
    warnings: list[str] = []
    if com_distance > max_distance:
        default_source = "arm-endpoint"
        warnings.append(
            f"fitted CoM distance {com_distance:.6f}m exceeds {max_distance:.6f}m; "
            "using arm_endpoint as the default TCP"
        )
    elif seed_source == "auto" and normalized_tool == "inspire":
        default_source = "hand-base"
        warnings.append("tool inspire defaults to hand_base TCP; use GUI adjustment to confirm a task TCP")
    elif seed_source == "hand-base":
        default_source = "hand-base"
    elif seed_source == "arm-endpoint":
        default_source = "arm-endpoint"
    else:
        default_source = "com"

    if default_source == "com":
        tcp_pose = tcp_pose_from_tool_pose_and_offset(pose, com)
        position_source = "gravity_fit_com_tool_m"
    else:
        if default_source == "hand-base":
            tcp_pose = list(HAND_BASE_TCP_POSE_FLANGE_M_RAD)
            position_source = "hand_base_default"
        else:
            tcp_pose = list(ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD)
            position_source = "arm_endpoint_default"

    now = time.time()
    return {
        "schema_version": 1,
        "frame_name": "tool_tcp",
        "pose_flange_m_rad": tcp_pose,
        "position_flange_m": tcp_pose[:3],
        "rotation_flange_rpy_rad": tcp_pose[3:6],
        "position_source": position_source,
        "rotation_policy": "preserve_existing_tool_rotation",
        "confirmed": False,
        "seed_source": default_source,
        "initial_pose_flange_m_rad": tcp_pose,
        "base_tool_pose_flange_m_rad": pose,
        "com_tool_m": com,
        "com_distance_m": com_distance,
        "max_auto_com_distance_m": max_distance,
        "source_fit_json": str(source_fit_json),
        "warnings": warnings,
        "updated_at": now,
    }


def update_payload(profile: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    profile["payload"] = dict(payload)
    return profile


def update_tcp(profile: dict[str, Any], tcp: dict[str, Any]) -> dict[str, Any]:
    pose = finite_float_list(tcp.get("pose_flange_m_rad", []), expected_len=6, label="tcp.pose_flange_m_rad")
    current = dict(tcp)
    current["pose_flange_m_rad"] = pose
    current["position_flange_m"] = pose[:3]
    current["rotation_flange_rpy_rad"] = pose[3:6]
    current["safety_warnings"] = tcp_safety_warnings(pose, label="profile TCP")
    current["updated_at"] = time.time()
    profile["tcp"] = current
    return profile


def default_impedance(mode: str = "pose") -> dict[str, Any]:
    if mode not in ("pose", "joint"):
        raise ValueError("mode must be pose or joint")
    return {
        "mode": mode,
        "force_source": "work",
        "axes": "all",
        "force_sign": 1.0,
        "angular_force_sign": -1.0,
        "angular_axis_signs": [1.0, 1.0, -1.0],
        "control_hz": 40.0,
        "action_hz": 20.0,
        "lowpass_alpha": 0.85,
        "force_deadband_n": 0.30,
        "torque_deadband_nm": 0.03,
        "linear_gain_m_s_per_n": 0.0,
        "angular_gain_rad_s_per_nm": 0.0,
        "linear_damping_n_s_per_m": 0.0,
        "angular_damping_nm_s_per_rad": 0.0,
        "linear_compliance_m_per_n": 0.004,
        "angular_compliance_rad_per_nm": 0.04,
        "max_linear_velocity_m_s": 0.04,
        "max_angular_velocity_rad_s": 0.15,
        "max_linear_offset_m": 0.05,
        "max_angular_offset_rad": 0.4,
        "return_rate_s": 0.35,
        "trajectory_mode": 0,
        "radio": 0,
        "hard_stop_force_n": 50.0,
        "hard_stop_torque_nm": 10.0,
        "max_command_joint_step_deg": 20.0,
    }


def profile_path(name: str, arm: str | None = None) -> Path:
    safe_name = normalize_profile_name(name)
    if arm:
        return PROFILE_DIR / f"rm65_{arm}_{safe_name}.json"
    matches = sorted(PROFILE_DIR.glob(f"rm65_*_{safe_name}.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return PROFILE_DIR / f"{safe_name}.json"
    raise ValueError(f"profile name '{name}' is ambiguous; pass arm")


def make_profile(
    *,
    name: str,
    arm: str,
    tool_name: str,
    pose_sequence_json: str = "",
    force_measurements_json: str = "",
    fit_json: str = "",
    official_report_json: str = "",
    tool_payload_validation_json: str = "",
    notes: str = "",
) -> dict[str, Any]:
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    safe_name = normalize_profile_name(name)
    now = time.time()
    return {
        "schema_version": 1,
        "data_type": "rm65_calibration_profile",
        "name": safe_name,
        "arm": arm,
        "tool_name": str(tool_name),
        "created_at": now,
        "updated_at": now,
        "notes": notes,
        "paths": {
            "pose_sequence_json": pose_sequence_json,
            "force_measurements_json": force_measurements_json,
            "fit_json": fit_json,
            "official_report_json": official_report_json,
            "tool_payload_validation_json": tool_payload_validation_json,
        },
        "payload": {},
        "tcp": arm_endpoint_tcp(),
        "impedance": {
            "pose": default_impedance("pose"),
            "joint": default_impedance("joint"),
        },
    }


def load_profile(name_or_path: str | Path, *, arm: str | None = None) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.exists():
        path = profile_path(str(name_or_path), arm)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("data_type") != "rm65_calibration_profile":
        raise ValueError(f"{path} is not an rm65_calibration_profile")
    payload["_profile_path"] = str(path)
    return payload


def save_profile(profile: dict[str, Any], path: str | Path | None = None) -> Path:
    if profile.get("data_type") != "rm65_calibration_profile":
        raise ValueError("profile data_type must be rm65_calibration_profile")
    profile = dict(profile)
    profile.pop("_profile_path", None)
    profile["updated_at"] = time.time()
    output_path = Path(path) if path else profile_path(str(profile["name"]), str(profile["arm"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)
    return output_path


def update_impedance(profile: dict[str, Any], mode: str, values: dict[str, Any]) -> dict[str, Any]:
    if mode not in ("pose", "joint"):
        raise ValueError("mode must be pose or joint")
    profile.setdefault("impedance", {})
    current = dict(default_impedance(mode))
    current.update(profile["impedance"].get(mode, {}))
    current.update(values)
    current["mode"] = mode
    profile["impedance"][mode] = current
    return profile


def list_profiles() -> list[Path]:
    if not PROFILE_DIR.exists():
        return []
    return sorted(PROFILE_DIR.glob("*.json"))
