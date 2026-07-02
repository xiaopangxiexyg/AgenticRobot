from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.force_compensation import finite_list, norm  # noqa: E402
from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65  # noqa: E402


DATA_TYPE = "rm65_standard_pose"
POSE_UNITS = ["m", "m", "m", "rad", "rad", "rad"]
SIDES = ("left", "right")
EMPTY_TOOL_FRAME = "empty"
EMPTY_TOOL_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def default_pose_path() -> str:
    return "hardware/rm65/utils/init_arm/standard_pose.json"


def selected_sides(arm: str) -> list[str]:
    if arm == "both":
        return ["left", "right"]
    if arm in SIDES:
        return [arm]
    raise RuntimeError(f"unsupported arm selection '{arm}'")


def side_ip(side: str, args) -> str:
    return str(args.left_ip if side == "left" else args.right_ip)


def format_values(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def is_empty_tool_frame_name(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    stem = Path(text).stem
    normalized = stem if stem else text
    return normalized == EMPTY_TOOL_FRAME or normalized.endswith(f"_{EMPTY_TOOL_FRAME}")


def connect_robot(side: str, ip: str) -> Arm:
    robot = Arm(RM65, ip)
    if int(robot.Arm_Socket_State()) != 0:
        robot.Arm_Socket_Close()
        raise RuntimeError(f"{side}: failed to connect ip={ip}")
    print(f"{side}: connected ip={ip} api={robot.API_Version()}")
    return robot


def _endpoint_pose_from_joint(robot: Arm, side: str, joint_deg: list[float]) -> list[float]:
    try:
        pose = robot.Algo_Forward_Kinematics(joint_deg, flag=1)
    except Exception as exc:
        raise RuntimeError(f"{side}: Algo_Forward_Kinematics failed for empty endpoint pose: {exc}") from exc
    try:
        return normalize_pose(pose[:6])
    except ValueError as exc:
        raise RuntimeError(f"{side}: invalid FK endpoint pose: {exc}") from exc


def read_state(robot: Arm, side: str, *, tool_frame: str = "") -> dict[str, list[float]]:
    code, joint_deg, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{side} reports arm_err={arm_err}, sys_err={sys_err}")
    try:
        joint = finite_list(joint_deg, expected_len=6, label="joint_deg")
        state_pose = finite_list(pose, expected_len=6, label="pose")
        if is_empty_tool_frame_name(tool_frame):
            state_pose = _endpoint_pose_from_joint(robot, side, joint)
        return {
            "joint_deg": joint,
            "pose": state_pose,
        }
    except ValueError as exc:
        raise RuntimeError(f"{side}: invalid arm state: {exc}") from exc


def read_force(robot: Arm, side: str) -> dict[str, list[float]]:
    code, raw, zero, work, tool = robot.Get_Force_Data()
    require_zero(code, f"{side} Get_Force_Data")
    try:
        return {
            "raw": finite_list(raw, expected_len=6, label="raw"),
            "zero": finite_list(zero, expected_len=6, label="zero"),
            "work": finite_list(work, expected_len=6, label="work"),
            "tool": finite_list(tool, expected_len=6, label="tool"),
        }
    except ValueError as exc:
        raise RuntimeError(f"{side}: invalid force data: {exc}") from exc


def current_tool_name(frame_result: object) -> str:
    if isinstance(frame_result, tuple) and len(frame_result) >= 2:
        frame = frame_result[1]
    else:
        frame = frame_result
    if isinstance(frame, dict):
        value = frame.get("name") or frame.get("frame_name") or frame.get("tool_name")
        if value is not None:
            return str(value).strip("\x00")
    return str(frame_result).strip("\x00")


def use_tool_frame(robot: Arm, side: str, tool_frame: str) -> str:
    """Switch the controller to the requested tool frame, normally ``empty``."""
    frame_name = str(tool_frame or "").strip()
    if frame_name:
        if is_empty_tool_frame_name(frame_name):
            update_code = robot.Update_Tool_Frame(frame_name, EMPTY_TOOL_POSE, 0.0, 0.0, 0.0, 0.0)
            action = "updated"
            if int(update_code) != 0:
                require_zero(
                    robot.Manual_Set_Tool_Frame(frame_name, EMPTY_TOOL_POSE, 0.0, 0.0, 0.0, 0.0),
                    f"{side} Manual_Set_Tool_Frame({frame_name}=zero TCP)",
                )
                action = "created"
            print(f"{side}: {action} zero-TCP empty tool frame '{frame_name}'")
        require_zero(robot.Change_Tool_Frame(frame_name), f"{side} Change_Tool_Frame({frame_name})")
        time.sleep(0.05)
    code, frame = robot.Get_Current_Tool_Frame()
    require_zero(code, f"{side} Get_Current_Tool_Frame")
    current_name = current_tool_name((code, frame))
    print(f"{side}: active_tool_frame={current_name}")
    if frame_name and frame_name not in current_name:
        raise RuntimeError(
            f"{side}: requested tool frame '{frame_name}', got active frame '{current_name}'"
        )
    return current_name


def normalize_angle(value: float) -> float:
    out = float(value)
    while out > math.pi:
        out -= 2.0 * math.pi
    while out < -math.pi:
        out += 2.0 * math.pi
    return out


def angle_delta(current: float, previous: float) -> float:
    return normalize_angle(float(current) - float(previous))


def normalize_pose(pose: Iterable[float]) -> list[float]:
    values = finite_list(pose, expected_len=6, label="pose")
    return values[:3] + [normalize_angle(value) for value in values[3:]]


def _rotation_from_pose(pose: Iterable[float]) -> R:
    values = finite_list(pose, expected_len=6, label="pose")
    return R.from_euler("xyz", values[3:6], degrees=False)


def pose_delta(a: Iterable[float], b: Iterable[float]) -> tuple[float, float]:
    pose_a = finite_list(a, expected_len=6, label="pose_a")
    pose_b = finite_list(b, expected_len=6, label="pose_b")
    linear = norm(pose_a[index] - pose_b[index] for index in range(3))
    angular = float((_rotation_from_pose(pose_a).inv() * _rotation_from_pose(pose_b)).magnitude())
    return linear, angular


def joint_delta_deg(a: Iterable[float], b: Iterable[float]) -> float:
    joint_a = finite_list(a, expected_len=6, label="joint_a")
    joint_b = finite_list(b, expected_len=6, label="joint_b")
    return max(abs(x - y) for x, y in zip(joint_a, joint_b))


def smoothstep(alpha: float) -> float:
    bounded = max(0.0, min(1.0, float(alpha)))
    return bounded * bounded * (3.0 - 2.0 * bounded)


def interpolate_pose(start_pose: Iterable[float], target_pose: Iterable[float], alpha: float) -> list[float]:
    start = normalize_pose(start_pose)
    target = normalize_pose(target_pose)
    blend = smoothstep(alpha)
    out = [
        start[index] + (target[index] - start[index]) * blend
        for index in range(3)
    ]
    rotations = R.from_euler("xyz", np.asarray([start[3:6], target[3:6]], dtype=float), degrees=False)
    euler = Slerp([0.0, 1.0], rotations)([blend]).as_euler("xyz", degrees=False)[0]
    out.extend(normalize_angle(value) for value in euler)
    return out


def mean_pose(poses: list[list[float]]) -> list[float]:
    if not poses:
        raise RuntimeError("cannot average an empty pose list")
    base = normalize_pose(poses[0])
    unwrapped = []
    for pose in poses:
        values = normalize_pose(pose)
        unwrapped.append(
            values[:3]
            + [
                base[index] + angle_delta(values[index], base[index])
                for index in range(3, 6)
            ]
        )
    return normalize_pose(
        [
            sum(row[index] for row in unwrapped) / len(unwrapped)
            for index in range(6)
        ]
    )


def pose_spread(poses: list[list[float]], center: list[float]) -> tuple[float, float]:
    if not poses:
        return 0.0, 0.0
    linear = 0.0
    angular = 0.0
    for pose in poses:
        pose_linear, pose_angular = pose_delta(pose, center)
        linear = max(linear, pose_linear)
        angular = max(angular, pose_angular)
    return linear, angular


def load_pose_file(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"standard pose file {path} does not exist. Record it first, for example:\n"
            f"  python hardware/rm65/utils/init_arm/rm65_record_standard_pose.py "
            f"--arm both --tool-frame empty --overwrite --output-json {path}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"failed to read standard pose file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse standard pose file {path}: {exc}") from exc
    if data.get("data_type") != DATA_TYPE:
        raise RuntimeError(f"{path} is not an {DATA_TYPE} JSON")
    arms = data.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise RuntimeError(f"{path} does not contain any arm poses")
    return data


def target_pose_for_side(data: dict[str, object], side: str) -> list[float]:
    arms = data.get("arms")
    if not isinstance(arms, dict) or side not in arms:
        raise RuntimeError(f"standard pose file does not contain arm '{side}'")
    arm_data = arms[side]
    if not isinstance(arm_data, dict):
        raise RuntimeError(f"standard pose entry for '{side}' is invalid")
    try:
        return normalize_pose(arm_data.get("pose", []))
    except ValueError as exc:
        raise RuntimeError(f"standard pose entry for '{side}' is invalid: {exc}") from exc


def recorded_tool_frame_for_side(data: dict[str, object], side: str) -> str:
    arms = data.get("arms")
    if not isinstance(arms, dict):
        return ""
    arm_data = arms.get(side)
    if not isinstance(arm_data, dict):
        return ""
    return str(arm_data.get("tool_frame") or "")


def write_json(path_text: str, payload: dict[str, object]) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote JSON: {path}")


def timestamp() -> float:
    return time.time()
