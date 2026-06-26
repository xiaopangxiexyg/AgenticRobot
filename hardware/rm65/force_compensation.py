from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


AXES = ("x", "y", "z", "rx", "ry", "rz")


def finite_list(values: Iterable[float], *, expected_len: int = 6, label: str = "values") -> list[float]:
    out = [float(value) for value in values]
    if len(out) != expected_len:
        raise ValueError(f"{label} must contain {expected_len} values, got {len(out)}")
    if not all(math.isfinite(value) for value in out):
        raise ValueError(f"{label} contains non-finite values: {out}")
    return out


def norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def normalize_angle(value: float) -> float:
    value = float(value)
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def angle_delta(current: float, previous: float) -> float:
    return normalize_angle(float(current) - float(previous))


def normalize_pose(pose: Iterable[float]) -> list[float]:
    values = finite_list(pose, expected_len=6, label="pose")
    return values[:3] + [normalize_angle(value) for value in values[3:]]


def axis_index(axis: str) -> int:
    if axis not in AXES:
        raise ValueError(f"unsupported axis '{axis}', expected one of {AXES}")
    return AXES.index(axis)


def parse_signs(text: str, *, expected_len: int = 3, label: str = "signs") -> list[float]:
    normalized = text.lower().replace(",", " ").replace("+", " ")
    tokens = [token.strip() for token in normalized.split() if token.strip()]
    if len(tokens) != expected_len:
        raise ValueError(f"{label} must contain {expected_len} signs, got {len(tokens)}")
    signs = []
    for token in tokens:
        value = float(token)
        if value not in (-1.0, 1.0):
            raise ValueError(f"{label} values must be -1 or 1, got {value}")
        signs.append(value)
    return signs


def parse_axes(text: str, *, allow_rotation: bool = False) -> list[bool]:
    normalized = text.lower().replace(",", " ").replace("+", " ")
    tokens = [token.strip() for token in normalized.split() if token.strip()]
    if not tokens:
        raise ValueError("axis list must not be empty")
    if tokens == ["all"]:
        return [True] * (6 if allow_rotation else 3) + ([False] * 3 if not allow_rotation else [])
    enabled = [False] * 6
    for token in tokens:
        if token in ("xyz", "linear", "translation", "trans"):
            enabled[:3] = [True, True, True]
            continue
        if token in ("rxyz", "rpy", "rot", "rotation", "angular"):
            if not allow_rotation:
                raise ValueError("rotational admittance is disabled for this smoke test")
            enabled[3:] = [True, True, True]
            continue
        if token not in AXES:
            raise ValueError(f"unsupported axis token '{token}', expected axes such as x,y,z,rx,ry,rz")
        index = axis_index(token)
        if index >= 3 and not allow_rotation:
            raise ValueError("rotational admittance is disabled for this smoke test")
        enabled[index] = True
    if not any(enabled):
        raise ValueError("at least one axis must be enabled")
    return enabled


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def clamp_norm(values: list[float], max_norm: float) -> list[float]:
    current = norm(values)
    if current <= max_norm or current <= 1e-12:
        return list(values)
    scale = max_norm / current
    return [value * scale for value in values]


def deadband_value(value: float, deadband: float) -> float:
    magnitude = abs(value)
    if magnitude <= deadband:
        return 0.0
    return math.copysign(magnitude - deadband, value)


def mean_wrench(wrenches: Iterable[Iterable[float]]) -> list[float]:
    rows = [finite_list(wrench, expected_len=6, label="wrench") for wrench in wrenches]
    if not rows:
        raise ValueError("cannot average an empty wrench list")
    return [sum(row[index] for row in rows) / len(rows) for index in range(6)]


@dataclass
class WrenchProcessor:
    """Startup tare, low-pass filtering, and deadband for external wrench data."""

    tare: list[float] = field(default_factory=lambda: [0.0] * 6)
    lowpass_alpha: float = 0.25
    force_deadband_n: float = 0.30
    torque_deadband_nm: float = 0.03
    _filtered: list[float] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.tare = finite_list(self.tare, expected_len=6, label="tare")
        if not 0.0 < self.lowpass_alpha <= 1.0:
            raise ValueError("lowpass_alpha must be in (0, 1]")
        if self.force_deadband_n < 0.0 or self.torque_deadband_nm < 0.0:
            raise ValueError("deadbands must be non-negative")

    def process(self, wrench: Iterable[float]) -> dict[str, list[float]]:
        measured = finite_list(wrench, expected_len=6, label="wrench")
        tare_corrected = [measured[index] - self.tare[index] for index in range(6)]
        if self._filtered is None:
            self._filtered = list(tare_corrected)
        else:
            alpha = self.lowpass_alpha
            self._filtered = [
                alpha * tare_corrected[index] + (1.0 - alpha) * self._filtered[index]
                for index in range(6)
            ]
        deadbanded = [
            deadband_value(self._filtered[index], self.force_deadband_n if index < 3 else self.torque_deadband_nm)
            for index in range(6)
        ]
        return {
            "measured": measured,
            "tare_corrected": tare_corrected,
            "filtered": list(self._filtered),
            "deadbanded": deadbanded,
        }


@dataclass
class PoseAdmittance:
    """A conservative outer-loop Cartesian admittance state for pose targets."""

    enabled_axes: list[bool]
    force_sign: float = 1.0
    angular_force_sign: float = -1.0
    angular_axis_signs: list[float] = field(default_factory=lambda: [1.0, 1.0, -1.0])
    linear_gain_m_s_per_n: float = 0.006
    angular_gain_rad_s_per_nm: float = 0.0
    linear_compliance_m_per_n: float = 0.0
    angular_compliance_rad_per_nm: float = 0.0
    max_linear_velocity_m_s: float = 0.03
    max_angular_velocity_rad_s: float = 0.0
    max_linear_offset_m: float = 0.03
    max_angular_offset_rad: float = 0.0
    return_rate_s: float = 0.35
    offset: list[float] = field(default_factory=lambda: [0.0] * 6)

    def __post_init__(self) -> None:
        if len(self.enabled_axes) != 6:
            raise ValueError("enabled_axes must contain 6 booleans")
        self.enabled_axes = [bool(value) for value in self.enabled_axes]
        self.offset = finite_list(self.offset, expected_len=6, label="offset")
        for name in (
            "linear_gain_m_s_per_n",
            "angular_gain_rad_s_per_nm",
            "linear_compliance_m_per_n",
            "angular_compliance_rad_per_nm",
            "max_linear_velocity_m_s",
            "max_angular_velocity_rad_s",
            "max_linear_offset_m",
            "max_angular_offset_rad",
            "return_rate_s",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.force_sign == 0.0:
            raise ValueError("force_sign must be nonzero")
        if self.angular_force_sign == 0.0:
            raise ValueError("angular_force_sign must be nonzero")
        self.angular_axis_signs = finite_list(self.angular_axis_signs, expected_len=3, label="angular_axis_signs")
        for value in self.angular_axis_signs:
            if value not in (-1.0, 1.0):
                raise ValueError("angular_axis_signs values must be -1 or 1")

    def reset(self) -> None:
        self.offset = [0.0] * 6

    def update(self, wrench: Iterable[float], dt: float) -> dict[str, list[float] | float]:
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError("dt must be positive and finite")
        wrench_values = finite_list(wrench, expected_len=6, label="wrench")
        active_wrench = [
            wrench_values[index] if self.enabled_axes[index] else 0.0
            for index in range(6)
        ]
        velocity = [0.0] * 6
        for index in range(3):
            velocity[index] = self.force_sign * self.linear_gain_m_s_per_n * active_wrench[index]
        for index in range(3, 6):
            axis_sign = self.angular_force_sign * self.angular_axis_signs[index - 3]
            velocity[index] = axis_sign * self.angular_gain_rad_s_per_nm * active_wrench[index]

        target_offset = [0.0] * 6
        if self.linear_compliance_m_per_n > 0.0 or self.angular_compliance_rad_per_nm > 0.0:
            for index in range(3):
                target_offset[index] = self.force_sign * self.linear_compliance_m_per_n * active_wrench[index]
            for index in range(3, 6):
                axis_sign = self.angular_force_sign * self.angular_axis_signs[index - 3]
                target_offset[index] = axis_sign * self.angular_compliance_rad_per_nm * active_wrench[index]
            target_offset[:3] = clamp_norm(target_offset[:3], self.max_linear_offset_m)
            if self.max_angular_offset_rad > 0.0:
                target_offset[3:] = clamp_norm(target_offset[3:], self.max_angular_offset_rad)
            else:
                target_offset[3:] = [0.0, 0.0, 0.0]

            delta = [target_offset[index] - self.offset[index] for index in range(6)]
            step = [0.0] * 6
            step[:3] = clamp_norm(delta[:3], self.max_linear_velocity_m_s * dt)
            if self.max_angular_velocity_rad_s > 0.0:
                step[3:] = clamp_norm(delta[3:], self.max_angular_velocity_rad_s * dt)
            self.offset = [self.offset[index] + step[index] for index in range(6)]
            velocity = [step[index] / dt for index in range(6)]
        else:
            velocity[:3] = clamp_norm(velocity[:3], self.max_linear_velocity_m_s)
            if self.max_angular_velocity_rad_s > 0.0:
                velocity[3:] = clamp_norm(velocity[3:], self.max_angular_velocity_rad_s)
            else:
                velocity[3:] = [0.0, 0.0, 0.0]

            if norm(active_wrench[:3]) <= 1e-9 and norm(active_wrench[3:]) <= 1e-9 and self.return_rate_s > 0.0:
                decay = max(0.0, 1.0 - self.return_rate_s * dt)
                self.offset = [value * decay for value in self.offset]
            else:
                self.offset = [
                    self.offset[index] + velocity[index] * dt
                    for index in range(6)
                ]

        self.offset[:3] = clamp_norm(self.offset[:3], self.max_linear_offset_m)
        if self.max_angular_offset_rad > 0.0:
            self.offset[3:] = clamp_norm(self.offset[3:], self.max_angular_offset_rad)
        else:
            self.offset[3:] = [0.0, 0.0, 0.0]

        return {
            "active_wrench": active_wrench,
            "target_offset": target_offset,
            "velocity": velocity,
            "offset": list(self.offset),
            "linear_offset_norm_m": norm(self.offset[:3]),
            "angular_offset_norm_rad": norm(self.offset[3:]),
        }


def _euler_zyx_to_matrix(rx: float, ry: float, rz: float) -> list[list[float]]:
    """Convert ZYX intrinsic Euler angles (rx, ry, rz) to a 3x3 rotation matrix.

    The RealMan SDK uses intrinsic ZYX convention: R = Rz(rz) * Ry(ry) * Rx(rx).
    """
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy,     cy * sx,                 cy * cx],
    ]


def _matrix_to_euler_zyx(matrix: list[list[float]]) -> tuple[float, float, float]:
    """Extract ZYX intrinsic Euler angles (rx, ry, rz) from a 3x3 rotation matrix."""
    sy = -matrix[2][0]
    cy = math.sqrt(matrix[0][0] ** 2 + matrix[1][0] ** 2)
    if cy > 1e-9:
        rx = math.atan2(matrix[2][1], matrix[2][2])
        ry = math.atan2(sy, cy)
        rz = math.atan2(matrix[1][0], matrix[0][0])
    else:
        # Gimbal lock: ry ≈ ±π/2
        rx = math.atan2(-matrix[1][2], matrix[1][1])
        ry = math.atan2(sy, cy)
        rz = 0.0
    return rx, ry, rz


def _small_angle_to_matrix(wx: float, wy: float, wz: float) -> list[list[float]]:
    """Convert a small angular offset (in radians, base-frame axis-angle approximation)
    to a rotation matrix using Rodrigues' formula."""
    angle = math.sqrt(wx * wx + wy * wy + wz * wz)
    if angle < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    # Normalize axis
    ax, ay, az = wx / angle, wy / angle, wz / angle
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return [
        [t * ax * ax + c,      t * ax * ay - s * az, t * ax * az + s * ay],
        [t * ax * ay + s * az, t * ay * ay + c,      t * ay * az - s * ax],
        [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c],
    ]


def _matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply two 3x3 matrices."""
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def apply_pose_offset_work_rotvec(pose: Iterable[float], offset: Iterable[float]) -> list[float]:
    """Apply a 6D pose offset with work/base-frame rotation-vector semantics.

    The linear part (x, y, z) is added directly.
    The angular part is interpreted as a small rotation vector in the work/base frame
    and composed via pre-multiplication: R_new = R_offset * R_current.
    """
    pose_values = finite_list(pose, expected_len=6, label="pose")
    offset_values = finite_list(offset, expected_len=6, label="offset")

    # Linear part: direct addition
    x = pose_values[0] + offset_values[0]
    y = pose_values[1] + offset_values[1]
    z = pose_values[2] + offset_values[2]

    # Angular part: proper rotation composition
    wx, wy, wz = offset_values[3], offset_values[4], offset_values[5]
    if abs(wx) < 1e-15 and abs(wy) < 1e-15 and abs(wz) < 1e-15:
        # No angular offset, skip expensive math
        return [x, y, z, pose_values[3], pose_values[4], pose_values[5]]

    r_current = _euler_zyx_to_matrix(pose_values[3], pose_values[4], pose_values[5])
    r_offset = _small_angle_to_matrix(wx, wy, wz)
    r_new = _matmul3(r_offset, r_current)
    rx, ry, rz = _matrix_to_euler_zyx(r_new)

    return [x, y, z, rx, ry, rz]


def apply_pose_offset(pose: Iterable[float], offset: Iterable[float]) -> list[float]:
    """Apply a 6D offset to a controller pose.

    User-facing impedance axes are named x, y, z, rx, ry, rz, so the angular part
    is applied directly to the controller pose's Euler channels. This keeps single-axis
    tuning intuitive even when the current pose has a large yaw/pitch offset.

    For physical work/base-frame rotation-vector semantics, use
    ``apply_pose_offset_work_rotvec`` instead.
    """
    pose_values = finite_list(pose, expected_len=6, label="pose")
    offset_values = finite_list(offset, expected_len=6, label="offset")
    return normalize_pose(
        [
            pose_values[index] + offset_values[index]
            for index in range(6)
        ]
    )
