#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, TYPE_CHECKING

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import redis
except Exception:
    redis = None

try:
    from serial.tools import list_ports
except Exception:
    list_ports = None

if TYPE_CHECKING:
    from drivers.motor_driver import DynamixelDriver


# Same exoskeleton mapping used by teleop.py / teleop_arm_only_to_redis.py.
CONFIG = {
    "joint_ids": (
        0, 1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12, 13,
    ),
    "default_motor_joints": (
        np.pi + np.pi / 6,
        np.pi - np.pi / 8,
        np.pi,
        np.pi,
        np.pi,
        np.pi,
        np.pi,
        np.pi - np.pi / 6,
        np.pi + np.pi / 8,
        np.pi,
        np.pi,
        np.pi,
        np.pi,
        np.pi,
    ),
    "default_robot_joints": (
        0.0, 0.0, 0.0, 0.5 * np.pi, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.5 * np.pi, 0.0, 0.0, 0.0,
    ),
    "joint_signs": (
        1, 1, -1, 1, 1, 1, 1,
        -1, 1, -1, -1, 1, -1, 1,
    ),
}

EXO_IDS = list(CONFIG["joint_ids"])
DEFAULT_EXO_PORT = (
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NOIE-if00-port0"
)
DEFAULT_EXO_BAUD = 57600

ARM_Q14_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def dynamixel_to_robot_arm(joint_angles: Sequence[float]) -> np.ndarray:
    q_m = np.asarray(joint_angles, dtype=np.float64)
    q_m0 = np.asarray(CONFIG["default_motor_joints"], dtype=np.float64)
    q_r0 = np.asarray(CONFIG["default_robot_joints"], dtype=np.float64)
    signs = np.asarray(CONFIG["joint_signs"], dtype=np.float64)
    gears = np.asarray(CONFIG.get("gear_ratios", (1,) * len(q_m)), dtype=np.float64)
    return gears * signs * (q_m - q_m0) + q_r0


def reset_motors(driver: "DynamixelDriver") -> None:
    reset_ids = [2, 4, 9, 11]
    angle_map = {2: np.pi, 4: np.pi, 9: np.pi, 11: np.pi}
    driver.set_torque_mode(True, ids=reset_ids)
    driver.set_profile(profile_velocity=50, profile_acceleration=10, ids=reset_ids)
    driver.set_joints([angle_map[i] for i in reset_ids], ids=reset_ids)
    time.sleep(3.0)
    driver.set_torque_mode(False, ids=reset_ids)


def read_exo_q14(driver: "DynamixelDriver") -> np.ndarray:
    dxl = driver.get_joints()
    return dynamixel_to_robot_arm(dxl)


def _rot_x(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def _rot_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def _rot_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def axis_angle_to_matrix(axis_xyz: Sequence[float], angle: float) -> np.ndarray:
    axis = np.asarray(axis_xyz, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    x, y, z = axis.tolist()
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    v = 1.0 - c
    return np.array(
        [
            [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
        ],
        dtype=np.float64,
    )


def make_transform(rotation: np.ndarray, translation_xyz: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return transform


def xyz_rpy_to_transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    return make_transform(rpy_to_matrix(roll, pitch, yaw), xyz)


def rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    sy = float(np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    singular = sy < 1e-8
    if not singular:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        roll = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def transform_to_xyz_rpy(transform: np.ndarray) -> np.ndarray:
    xyz = transform[:3, 3]
    rpy = rotation_matrix_to_rpy(transform[:3, :3])
    return np.concatenate([xyz, rpy], axis=0)


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: Tuple[float, float, float]
    origin_rpy: Tuple[float, float, float]
    axis_xyz: Tuple[float, float, float]

    def transform(self, q: float) -> np.ndarray:
        static_tf = xyz_rpy_to_transform(self.origin_xyz, self.origin_rpy)
        if self.joint_type in {"revolute", "continuous"}:
            motion_tf = make_transform(axis_angle_to_matrix(self.axis_xyz, q), (0.0, 0.0, 0.0))
            return static_tf @ motion_tf
        if self.joint_type in {"fixed"}:
            return static_tf
        raise NotImplementedError(f"Unsupported joint type: {self.joint_type}")


class URDFArmFK:
    def __init__(self, joints_by_child: Dict[str, JointSpec]):
        self.joints_by_child = joints_by_child

    @classmethod
    def from_file(cls, urdf_path: Path) -> "URDFArmFK":
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        joints_by_child: Dict[str, JointSpec] = {}

        for joint in root.findall("joint"):
            name = joint.attrib["name"]
            joint_type = joint.attrib["type"]

            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue

            origin = joint.find("origin")
            axis = joint.find("axis")

            origin_xyz = _parse_triplet(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0")
            origin_rpy = _parse_triplet(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0")
            axis_xyz = _parse_triplet(axis.attrib.get("xyz", "0 0 0") if axis is not None else "0 0 0")

            spec = JointSpec(
                name=name,
                joint_type=joint_type,
                parent_link=parent.attrib["link"],
                child_link=child.attrib["link"],
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis_xyz=axis_xyz,
            )
            joints_by_child[spec.child_link] = spec

        return cls(joints_by_child=joints_by_child)

    def chain(self, base_link: str, target_link: str) -> List[JointSpec]:
        current_link = target_link
        chain_rev: List[JointSpec] = []
        seen = set()
        while current_link != base_link:
            if current_link in seen:
                raise RuntimeError(f"Loop detected while tracing link chain to {target_link}")
            seen.add(current_link)

            joint = self.joints_by_child.get(current_link)
            if joint is None:
                raise KeyError(f"Could not trace from {target_link} back to base link {base_link}")

            chain_rev.append(joint)
            current_link = joint.parent_link

        chain_rev.reverse()
        return chain_rev

    def forward(
        self,
        base_link: str,
        target_link: str,
        joint_positions: Dict[str, float],
    ) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        for joint in self.chain(base_link, target_link):
            q = float(joint_positions.get(joint.name, 0.0))
            transform = transform @ joint.transform(q)
        return transform


def _parse_triplet(text: str) -> Tuple[float, float, float]:
    vals = [float(v) for v in text.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 floats, got: {text!r}")
    return (vals[0], vals[1], vals[2])


def q14_to_joint_dict(q14: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(q14, dtype=np.float64).reshape(-1)
    if arr.shape != (14,):
        raise ValueError(f"Expected q14 shape (14,), got {arr.shape}")
    return {name: float(arr[i]) for i, name in enumerate(ARM_Q14_NAMES)}


def pose_packet(
    q14: Sequence[float],
    fk: URDFArmFK,
    base_link: str,
    left_link: str,
    right_link: str,
) -> Dict[str, np.ndarray]:
    joint_positions = q14_to_joint_dict(q14)
    left_tf = fk.forward(base_link=base_link, target_link=left_link, joint_positions=joint_positions)
    right_tf = fk.forward(base_link=base_link, target_link=right_link, joint_positions=joint_positions)

    left_pose = transform_to_xyz_rpy(left_tf).astype(np.float32)
    right_pose = transform_to_xyz_rpy(right_tf).astype(np.float32)
    q14_arr = np.asarray(q14, dtype=np.float32).reshape(14)

    return {
        "timestamp": np.asarray([time.time()], dtype=np.float64),
        "q14": q14_arr,
        "left_xyz_rpy": left_pose,
        "right_xyz_rpy": right_pose,
        "pose6d": np.concatenate([left_pose, right_pose], axis=0),
        "base_link": base_link,
        "left_link": left_link,
        "right_link": right_link,
    }


def _format_pose(name: str, pose: Sequence[float]) -> str:
    x, y, z, roll, pitch, yaw = [float(v) for v in pose]
    return (
        f"{name}: "
        f"xyz=({x:+.4f}, {y:+.4f}, {z:+.4f}) m, "
        f"rpy=({roll:+.4f}, {pitch:+.4f}, {yaw:+.4f}) rad"
    )


def _print_packet(packet: Dict[str, np.ndarray], print_q14: bool) -> None:
    if print_q14:
        q14 = np.array2string(packet["q14"], precision=4, separator=", ")
        print(f"q14: {q14}")
    print(_format_pose("left ", packet["left_xyz_rpy"]))
    print(_format_pose("right", packet["right_xyz_rpy"]))


def list_serial_port_info() -> List[Dict[str, str]]:
    if list_ports is None:
        return []

    ports = []
    for port in list_ports.comports():
        ports.append(
            {
                "device": str(port.device),
                "description": str(port.description),
                "hwid": str(port.hwid),
                "manufacturer": str(getattr(port, "manufacturer", "") or ""),
                "product": str(getattr(port, "product", "") or ""),
                "serial_number": str(getattr(port, "serial_number", "") or ""),
                "location": str(getattr(port, "location", "") or ""),
            }
        )
    return ports


def print_serial_port_info() -> None:
    ports = list_serial_port_info()
    if list_ports is None:
        print("pyserial list_ports is unavailable in this environment")
        return
    if not ports:
        print("No serial ports found")
        return

    print("Detected serial ports:")
    for idx, port in enumerate(ports, start=1):
        print(f"[{idx}] device       : {port['device']}")
        print(f"    description  : {port['description']}")
        print(f"    hwid         : {port['hwid']}")
        if port["manufacturer"]:
            print(f"    manufacturer : {port['manufacturer']}")
        if port["product"]:
            print(f"    product      : {port['product']}")
        if port["serial_number"]:
            print(f"    serial_number: {port['serial_number']}")
        if port["location"]:
            print(f"    location     : {port['location']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read exoskeleton q14 and convert it to left/right 6-DoF end-effector "
            "poses [x, y, z, roll, pitch, yaw] using the local URDF."
        )
    )
    parser.add_argument("--port", type=str, default=DEFAULT_EXO_PORT, help="Dynamixel bus port")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_EXO_BAUD, help="Dynamixel baudrate")
    parser.add_argument("--hz", type=float, default=20.0, help="Read frequency")
    parser.add_argument("--frames", type=int, default=0, help="How many frames to read; 0 means infinite")
    parser.add_argument(
        "--urdf",
        type=str,
        default=str(ROOT / "assets" / "g1_o6" / "robot.urdf"),
        help="URDF used for forward kinematics",
    )
    parser.add_argument("--base-link", type=str, default="torso_link", help="Pose reference frame")
    parser.add_argument("--left-link", type=str, default="left_hand_base_link", help="Left target link")
    parser.add_argument("--right-link", type=str, default="right_hand_base_link", help="Right target link")
    parser.add_argument("--reset-exo", action="store_true", help="Run the same exo reset used by teleop.py before reading")
    parser.add_argument("--print-q14", action="store_true", help="Also print the 14 arm joint values")
    parser.add_argument("--dry-run", action="store_true", help="Skip hardware access and print the zero-q14 FK result once")
    parser.add_argument("--list-ports", action="store_true", help="List visible serial ports and exit")
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument(
        "--redis-key",
        type=str,
        default="",
        help="If set, publish the pose packet to Redis with pickle serialization",
    )
    return parser.parse_args()


def maybe_build_redis_client(args: argparse.Namespace):
    if not args.redis_key:
        return None
    if redis is None:
        raise RuntimeError("redis package is not available, but --redis-key was provided")
    return redis.Redis(host=args.redis_host, port=args.redis_port)


def run_dry(fk: URDFArmFK, args: argparse.Namespace) -> None:
    packet = pose_packet(
        q14=np.zeros(14, dtype=np.float64),
        fk=fk,
        base_link=args.base_link,
        left_link=args.left_link,
        right_link=args.right_link,
    )
    print("[dry-run] q14 is all zeros")
    _print_packet(packet, print_q14=args.print_q14)


def main() -> None:
    args = parse_args()
    fk = URDFArmFK.from_file(Path(args.urdf))

    if args.list_ports:
        print_serial_port_info()
        return

    if args.dry_run:
        run_dry(fk=fk, args=args)
        return

    publisher = maybe_build_redis_client(args)

    from drivers.motor_driver import DynamixelDriver

    try:
        exo = DynamixelDriver(EXO_IDS, port=args.port, baudrate=args.baudrate)
    except Exception as exc:
        raise RuntimeError(
            "Failed to open the exoskeleton serial port.\n"
            f"requested port: {args.port}\n"
            "Use `python minimal_control/teleop_6dof_reader.py --list-ports` or "
            "`python -m serial.tools.list_ports -v` to find the correct device, "
            "then rerun with `--port /dev/ttyUSB0` or a `/dev/serial/by-id/...` alias."
        ) from exc
    try:
        exo.get_joints()

        if args.reset_exo:
            print("[Stage 0] exo reset ...")
            exo.pause_reading()
            reset_motors(exo)
            exo.resume_reading()

        print(
            f"[Start] read exo q14 -> pose6d at {args.hz:.1f}Hz, "
            f"base={args.base_link}, left={args.left_link}, right={args.right_link}"
        )
        if publisher is not None:
            print(f"[Redis] publishing to key '{args.redis_key}'")

        dt = 1.0 / float(args.hz)
        frame_idx = 0
        while True:
            t0 = time.perf_counter()

            q14 = read_exo_q14(exo)
            packet = pose_packet(
                q14=q14,
                fk=fk,
                base_link=args.base_link,
                left_link=args.left_link,
                right_link=args.right_link,
            )

            _print_packet(packet, print_q14=args.print_q14)
            if publisher is not None:
                publisher.set(args.redis_key, pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL))

            frame_idx += 1
            if args.frames > 0 and frame_idx >= args.frames:
                break

            sleep_s = dt - (time.perf_counter() - t0)
            if sleep_s > 0.0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\n[Exit] stopped by user")
    finally:
        exo.close()


if __name__ == "__main__":
    main()
