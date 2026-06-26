from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.calibration_profiles import (  # noqa: E402
    ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD,
    finite_float_list,
    load_profile,
    make_payload,
    make_tcp_from_fit,
    save_profile,
    update_payload,
    update_tcp,
)
from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65  # noqa: E402
from hardware.rm65.standard_paths import resolve_fit_path  # noqa: E402


CONFIRM_TEXT = "APPLY_RM65_TOOL_TCP"
LINK_FRAME_NAMES = ("base_link", "arm_link1", "arm_link2", "arm_link3", "arm_link4", "arm_link5", "arm_link6", "hand_base")


def _format(values: Iterable[float], *, precision: int = 5) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _rotation_matrix_from_euler_xyz(rx: float, ry: float, rz: float) -> list[list[float]]:
    sx, cx = math.sin(float(rx)), math.cos(float(rx))
    sy, cy = math.sin(float(ry)), math.cos(float(ry))
    sz, cz = math.sin(float(rz)), math.cos(float(rz))
    rx_mat = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
    ry_mat = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    rz_mat = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
    return _matmul3(_matmul3(rz_mat, ry_mat), rx_mat)


def _matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3)] for row in range(3)]


def _matvec3(a: list[list[float]], b: list[float]) -> list[float]:
    return [sum(a[row][k] * b[k] for k in range(3)) for row in range(3)]


def _pose_to_transform(pose: Iterable[float]) -> list[list[float]]:
    values = finite_float_list(pose, expected_len=6, label="pose")
    rot = _rotation_matrix_from_euler_xyz(*values[3:6])
    return [
        [rot[0][0], rot[0][1], rot[0][2], values[0]],
        [rot[1][0], rot[1][1], rot[1][2], values[1]],
        [rot[2][0], rot[2][1], rot[2][2], values[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _transform_point(transform: list[list[float]], point: Iterable[float]) -> list[float]:
    p = finite_float_list(point, expected_len=3, label="point")
    return [
        transform[row][0] * p[0] + transform[row][1] * p[1] + transform[row][2] * p[2] + transform[row][3]
        for row in range(3)
    ]


def _transform_matrix(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)] for row in range(4)]


def _dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _cross(a: Iterable[float], b: Iterable[float]) -> list[float]:
    ax, ay, az = [float(value) for value in a]
    bx, by, bz = [float(value) for value in b]
    return [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]


def _unit(values: Iterable[float]) -> list[float]:
    out = [float(value) for value in values]
    length = _norm(out)
    if length <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [value / length for value in out]


def _selected_ip(arm: str, args) -> tuple[str, str]:
    if arm == "left":
        return "left", args.left_ip
    if arm == "right":
        return "right", args.right_ip
    raise RuntimeError("profile arm must be left or right")


def _connect_robot(side: str, ip: str) -> Arm:
    robot = Arm(RM65, ip)
    if int(robot.Arm_Socket_State()) != 0:
        robot.Arm_Socket_Close()
        raise RuntimeError(f"{side}: failed to connect ip={ip}")
    print(f"{side}: connected ip={ip} api={robot.API_Version()}")
    return robot


def _pose_from_tool_frame(frame: object) -> list[float] | None:
    if isinstance(frame, tuple) and len(frame) >= 2:
        frame = frame[1]
    if isinstance(frame, dict):
        pose = frame.get("pose")
        if pose is not None:
            return finite_float_list(pose, expected_len=6, label="tool pose")
    return None


def _payload_from_tool_frame(frame: object) -> tuple[float, list[float]] | None:
    if isinstance(frame, tuple) and len(frame) >= 2:
        frame = frame[1]
    if not isinstance(frame, dict):
        return None
    try:
        payload = float(frame.get("payload"))
        com = [float(frame.get(key)) for key in ("x", "y", "z")]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in [payload, *com]):
        return None
    # RealMan API2 reads payload CoM back in millimeters on this setup.
    if _norm(com) > 1.0:
        com = [value * 0.001 for value in com]
    return payload, com


def _tool_name_from_frame(frame: object) -> str:
    if isinstance(frame, tuple) and len(frame) >= 2:
        frame = frame[1]
    if isinstance(frame, dict):
        value = frame.get("name") or frame.get("frame_name") or frame.get("tool_name")
        if value is not None:
            return str(value).strip("\x00")
    return ""


def _read_state(robot: Arm, side: str) -> dict[str, list[float]]:
    code, joint_deg, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{side} reports arm_err={arm_err}, sys_err={sys_err}")
    return {
        "joint_deg": finite_float_list(joint_deg, expected_len=6, label="joint_deg"),
        "pose": finite_float_list(pose, expected_len=6, label="pose"),
    }


def _load_fit(path: Path, *, allow_non_tool_source: bool = False) -> tuple[dict[str, object], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit = payload.get("fit")
    if not isinstance(fit, dict):
        raise RuntimeError(f"{path} does not contain a fit object")
    if fit.get("mode") != "tool_gravity_model":
        raise RuntimeError(f"{path} is not a tool gravity model fit")
    if fit.get("source") != "tool" and not allow_non_tool_source:
        raise RuntimeError(f"{path} source={fit.get('source')} is not tool; review frame semantics first")
    return payload, fit


def _resolve_fit(profile: dict[str, object], args) -> tuple[Path | None, dict[str, object] | None, dict[str, object] | None]:
    fit_text = args.fit_json
    if not fit_text:
        paths = profile.get("paths") if isinstance(profile.get("paths"), dict) else {}
        fit_text = str(paths.get("fit_json") or "")
    if fit_text:
        path = Path(fit_text)
        if path.exists():
            payload, fit = _load_fit(path, allow_non_tool_source=args.allow_non_tool_source)
            return path, payload, fit
    if args.expected_tool:
        path = resolve_fit_path(str(profile["arm"]), args.expected_tool, require_existing=False)
        if path.exists():
            payload, fit = _load_fit(path, allow_non_tool_source=args.allow_non_tool_source)
            return path, payload, fit
    return None, None, None


def _tcp_from_profile(profile: dict[str, object]) -> dict[str, object]:
    tcp = profile.get("tcp")
    if not isinstance(tcp, dict) or not tcp.get("pose_flange_m_rad"):
        return {
            "schema_version": 1,
            "frame_name": "tool_tcp",
            "pose_flange_m_rad": list(ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD),
            "position_source": "arm_endpoint_default",
            "confirmed": False,
        }
    return dict(tcp)


def _init_profile_from_fit_if_requested(
    profile: dict[str, object],
    *,
    fit_path: Path | None,
    fit_payload: dict[str, object] | None,
    fit: dict[str, object] | None,
    current_tool_pose: list[float],
    args,
) -> tuple[dict[str, object], bool]:
    if fit is None or fit_path is None:
        return profile, False
    updated = False
    mass = float(fit["mass_kg_abs"])
    com = finite_float_list(fit["com_tool_m"], expected_len=3, label="fit.com_tool_m")
    update_payload(
        profile,
        make_payload(
            mass_kg=mass,
            com_tool_m=com,
            source_fit_json=str(fit_path),
            fit_source=str(fit.get("source") or ""),
            fit_mode=str(fit.get("mode") or ""),
        ),
    )
    updated = True
    tcp = _tcp_from_profile(profile)
    if args.overwrite_tcp or not tcp.get("confirmed"):
        update_tcp(
            profile,
            make_tcp_from_fit(
                tool_name=str(fit_payload.get("tool_label") or profile.get("tool_name") or args.expected_tool),
                tool_pose_flange_m_rad=current_tool_pose,
                com_tool_m=com,
                source_fit_json=str(fit_path),
                seed_source=args.tcp_seed_source,
                max_com_distance_m=args.max_tcp_com_distance_m,
            ),
        )
        updated = True
    return profile, updated


def _compute_arm_points(joint_deg: list[float] | None, tcp_pose_flange: list[float]) -> dict[str, object]:
    if joint_deg is None:
        return {"error": "no robot joint state available"}
    try:
        import numpy as np
        from rm65_inspire_tac.ik import SingleArmIKSolver
    except Exception as exc:
        return {"error": f"Pinocchio/IK import failed: {exc}"}
    try:
        solver = SingleArmIKSolver(eef_name="hand_base")
        qpos = np.asarray(joint_deg, dtype=float).reshape(6) / 180.0 * math.pi
        solver.pin.forwardKinematics(solver.model, solver.data, qpos)
        solver.pin.updateFramePlacements(solver.model, solver.data)
        points: list[list[float]] = []
        named: dict[str, list[float]] = {}
        for name in LINK_FRAME_NAMES:
            frame_id = solver.model.getFrameId(name)
            if frame_id >= len(solver.model.frames):
                continue
            trans = solver.data.oMf[frame_id].translation
            point = [float(trans[0]), float(trans[1]), float(trans[2])]
            points.append(point)
            named[name] = point
        link6_id = solver.model.getFrameId("arm_link6")
        link6 = solver.data.oMf[link6_id]
        t_base_link6 = [
            [float(link6.rotation[row, col]) for col in range(3)] + [float(link6.translation[row])]
            for row in range(3)
        ] + [[0.0, 0.0, 0.0, 1.0]]
        t_tcp = _transform_matrix(t_base_link6, _pose_to_transform(tcp_pose_flange))
        tcp_point = [t_tcp[0][3], t_tcp[1][3], t_tcp[2][3]]
        return {
            "points": points,
            "named": named,
            "link6_transform": t_base_link6,
            "tcp_point": tcp_point,
            "tcp_axes": [
                _transform_point(t_tcp, [0.04, 0.0, 0.0]),
                _transform_point(t_tcp, [0.0, 0.04, 0.0]),
                _transform_point(t_tcp, [0.0, 0.0, 0.04]),
            ],
        }
    except Exception as exc:
        return {"error": f"arm geometry failed: {exc}"}


class TcpAdjustGui:
    def __init__(
        self,
        *,
        profile: dict[str, object],
        profile_ref: str,
        robot: Arm | None,
        joint_deg: list[float] | None,
        current_tool_frame: object | None,
        args,
    ):
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.profile = profile
        self.profile_ref = profile_ref
        self.robot = robot
        self.joint_deg = joint_deg
        self.current_tool_frame = current_tool_frame
        self.args = args
        self.tool_name = str(profile.get("tool_name") or args.expected_tool or profile.get("name"))
        self.root = tk.Tk()
        self.root.title(f"RM65 TCP Adjust - {profile.get('arm')} / {self.tool_name}")
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        tcp = _tcp_from_profile(profile)
        pose = finite_float_list(tcp.get("pose_flange_m_rad", ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD), expected_len=6, label="tcp pose")
        self.rotation = pose[3:6]
        self.vars = [tk.DoubleVar(value=pose[index] * 1000.0) for index in range(3)]
        self.drag_plane_var = tk.StringVar(value="xy")
        self.status_var = tk.StringVar(value=self._status_text())
        self.view_yaw = math.radians(-45.0)
        self.view_pitch = math.radians(24.0)
        self.view_zoom = 1.0
        self.view_pan = [0.0, 0.0]
        self._mouse_mode: str | None = None
        self._mouse_start_xy = (0.0, 0.0)
        self._drag_start_vars = [var.get() for var in self.vars]
        self._last_geometry: dict[str, object] | None = None
        self._last_view: dict[str, object] | None = None
        self._build()
        self._redraw()

    def _status_text(self) -> str:
        tcp = _tcp_from_profile(self.profile)
        payload = self.profile.get("payload") if isinstance(self.profile.get("payload"), dict) else {}
        profile_name = str(self.profile.get("name") or self.profile_ref)
        arm = str(self.profile.get("arm") or "")
        if profile_name == self.tool_name:
            identity = f"{arm} / {self.tool_name}"
        else:
            identity = f"{arm} / profile={profile_name} / tool={self.tool_name}"
        lines = [
            f"arm/profile/tool: {identity}",
            f"tcp source: {tcp.get('position_source')} confirmed={tcp.get('confirmed')}",
            f"payload mass={payload.get('mass_kg')} kg com={payload.get('com_tool_m')}",
            f"tcp rpy locked: {_format(self.rotation, precision=5)} rad",
        ]
        warnings = tcp.get("warnings", [])
        if warnings:
            lines.append("warnings: " + " | ".join(str(item) for item in warnings))
        if self.current_tool_frame is not None:
            current_tool = _tool_name_from_frame(self.current_tool_frame)
            if current_tool and current_tool != self.tool_name:
                lines.append(f"controller current tool: {current_tool}")
        return "\n".join(lines)

    def _build(self) -> None:
        tk = self.tk
        ttk = self.ttk
        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        ttk.Label(outer, textvariable=self.status_var, justify="left").grid(row=0, column=0, sticky="ew")
        self.canvas = tk.Canvas(outer, width=980, height=620, background="#f8f8f6", highlightthickness=1, highlightbackground="#c7c7c7")
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-3>", self._on_right_press)
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_left_release)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

        controls = ttk.Frame(outer)
        controls.grid(row=2, column=0, sticky="ew")
        labels = ("X mm", "Y mm", "Z mm")
        for index, label in enumerate(labels):
            frame = ttk.Frame(controls)
            frame.grid(row=0, column=index, sticky="ew", padx=6)
            controls.columnconfigure(index, weight=1)
            ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
            spin = ttk.Spinbox(
                frame,
                from_=-float(self.args.slider_range_mm),
                to=float(self.args.slider_range_mm),
                increment=float(self.args.step_mm),
                textvariable=self.vars[index],
                width=12,
                command=self._redraw,
            )
            spin.grid(row=1, column=0, sticky="ew")
            spin.bind("<KeyRelease>", lambda _event: self._redraw())
            scale = ttk.Scale(
                frame,
                from_=-float(self.args.slider_range_mm),
                to=float(self.args.slider_range_mm),
                variable=self.vars[index],
                command=lambda _value: self._redraw(),
            )
            scale.grid(row=2, column=0, sticky="ew")
        plane = ttk.LabelFrame(controls, text="3D drag plane")
        plane.grid(row=0, column=3, sticky="nsew", padx=6)
        controls.columnconfigure(3, weight=0)
        for col, (label, value) in enumerate((("XY", "xy"), ("XZ", "xz"), ("YZ", "yz"))):
            ttk.Radiobutton(
                plane,
                text=label,
                value=value,
                variable=self.drag_plane_var,
            ).grid(row=0, column=col, padx=4, pady=4)
        ttk.Label(
            plane,
            text="Drag orange TCP point. Drag empty canvas to orbit; wheel zooms.",
            wraplength=260,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=4)
        buttons = ttk.Frame(outer)
        buttons.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(buttons, text="Save Profile", command=self._save_profile).grid(row=0, column=0, padx=4)
        state = "disabled"
        ttk.Button(buttons, text="Controller Write Disabled", command=self._save_and_write_controller, state=state).grid(row=0, column=1, padx=4)
        ttk.Button(buttons, text="Reload Profile", command=self._reload_profile).grid(row=0, column=2, padx=4)
        ttk.Button(buttons, text="Close", command=self.root.destroy).grid(row=0, column=3, padx=4)
        ttk.Label(buttons, text="RealMan controller TCP write is disabled until SDK TCP semantics are reviewed.").grid(row=0, column=4, padx=10)

    def _current_pose(self) -> list[float]:
        return [float(var.get()) * 0.001 for var in self.vars] + list(self.rotation)

    def _update_profile_tcp(self, *, source: str, confirmed: bool) -> Path:
        tcp = dict(_tcp_from_profile(self.profile))
        tcp.update(
            {
                "schema_version": 1,
                "frame_name": "tool_tcp",
                "pose_flange_m_rad": self._current_pose(),
                "position_source": source,
                "rotation_policy": "preserve_existing_tool_rotation",
                "confirmed": bool(confirmed),
            }
        )
        update_tcp(self.profile, tcp)
        return save_profile(self.profile, self.profile.get("_profile_path"))

    def _save_profile(self) -> None:
        path = self._update_profile_tcp(source="gui_adjusted", confirmed=True)
        self.status_var.set(self._status_text())
        self.messagebox.showinfo("Saved", f"Saved TCP to profile:\n{path}")

    def _save_and_write_controller(self) -> None:
        if self.robot is None:
            self.messagebox.showerror("No Robot", "Robot is not connected.")
            return
        path = self._update_profile_tcp(source="gui_adjusted_controller_written", confirmed=True)
        payload = self.profile.get("payload") if isinstance(self.profile.get("payload"), dict) else {}
        frame_payload = _payload_from_tool_frame(self.current_tool_frame) if self.current_tool_frame is not None else None
        mass = float(payload.get("mass_kg", frame_payload[0] if frame_payload else 0.0))
        com = payload.get("com_tool_m", frame_payload[1] if frame_payload else [0.0, 0.0, 0.0])
        com = finite_float_list(com, expected_len=3, label="payload.com_tool_m")
        try:
            _require_zero(self.robot.Update_Tool_Frame(self.tool_name, self._current_pose(), mass, *com), "Update_Tool_Frame")
            _require_zero(self.robot.Change_Tool_Frame(self.tool_name), "Change_Tool_Frame")
        except RuntimeError as exc:
            self.messagebox.showerror("Controller Write Failed", str(exc))
            return
        self.status_var.set(self._status_text())
        self.messagebox.showinfo("Saved", f"Saved profile and wrote controller tool frame:\n{path}")

    def _reload_profile(self) -> None:
        self.profile = load_profile(self.profile_ref, arm=str(self.profile.get("arm") or "") or None)
        pose = finite_float_list(_tcp_from_profile(self.profile).get("pose_flange_m_rad"), expected_len=6, label="tcp pose")
        self.rotation = pose[3:6]
        for index in range(3):
            self.vars[index].set(pose[index] * 1000.0)
        self.status_var.set(self._status_text())
        self._redraw()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        geometry = _compute_arm_points(self.joint_deg, self._current_pose())
        self._last_geometry = geometry
        if "error" in geometry:
            self.canvas.create_text(24, 24, anchor="nw", text=str(geometry["error"]), fill="#9a4f00")
            self._draw_tcp_axes_only()
            return
        points = list(geometry.get("points") or [])
        tcp = geometry.get("tcp_point")
        axes = geometry.get("tcp_axes") or []
        self._draw_3d_scene(points, tcp if isinstance(tcp, list) else None, axes)
        pose = self._current_pose()
        self.canvas.create_text(
            24,
            self._canvas_height() - 46,
            anchor="nw",
            text=f"TCP pose flange [m,rad]: {_format(pose, precision=5)}",
            fill="#222222",
            font=("TkDefaultFont", 11),
        )
        self.canvas.create_text(
            self._canvas_width() - 24,
            self._canvas_height() - 46,
            anchor="ne",
            text=f"drag plane: {self.drag_plane_var.get().upper()}",
            fill="#555555",
            font=("TkDefaultFont", 11),
        )

    def _draw_tcp_axes_only(self) -> None:
        pose = self._current_pose()
        self.canvas.create_text(24, 60, anchor="nw", text=f"TCP pose flange [m,rad]: {_format(pose, precision=5)}")

    def _camera_basis(self) -> tuple[list[float], list[float], list[float]]:
        yaw = float(self.view_yaw)
        pitch = max(math.radians(-82.0), min(math.radians(82.0), float(self.view_pitch)))
        forward = [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ]
        right = _unit([-math.sin(yaw), math.cos(yaw), 0.0])
        up = _unit(_cross(forward, right))
        return right, up, _unit(forward)

    def _projection_setup(self, points: list[list[float]], tcp: list[float] | None, axes: object) -> dict[str, object]:
        all_points = list(points)
        if tcp is not None:
            all_points.append(tcp)
        if isinstance(axes, list):
            all_points.extend(axis for axis in axes if isinstance(axis, list))
        if not all_points:
            all_points = [[0.0, 0.0, 0.0]]
        mins = [min(point[index] for point in all_points) for index in range(3)]
        maxs = [max(point[index] for point in all_points) for index in range(3)]
        center = [(mins[index] + maxs[index]) * 0.5 for index in range(3)]
        span = max(maxs[index] - mins[index] for index in range(3))
        span = max(span, 0.35)
        width = self._canvas_width()
        height = self._canvas_height()
        scale = float(self.view_zoom) * 0.72 * min(width, height) / span
        right, up, forward = self._camera_basis()
        view = {
            "center": center,
            "scale": scale,
            "right": right,
            "up": up,
            "forward": forward,
            "width": width,
            "height": height,
        }
        self._last_view = view
        return view

    def _project3(self, point: Iterable[float], view: dict[str, object] | None = None) -> tuple[float, float, float]:
        if view is None:
            if self._last_view is None:
                raise RuntimeError("no active 3D view")
            view = self._last_view
        p = [float(value) for value in point]
        center = view["center"]
        right = view["right"]
        up = view["up"]
        forward = view["forward"]
        scale = float(view["scale"])
        width = float(view["width"])
        height = float(view["height"])
        rel = [p[index] - center[index] for index in range(3)]
        x = width * 0.5 + float(self.view_pan[0]) + _dot(rel, right) * scale
        y = height * 0.5 + float(self.view_pan[1]) - _dot(rel, up) * scale
        depth = _dot(rel, forward)
        return x, y, depth

    def _draw_3d_scene(self, points: list[list[float]], tcp: list[float] | None, axes: object) -> None:
        view = self._projection_setup(points, tcp, axes)
        width = int(view["width"])
        height = int(view["height"])
        self.canvas.create_rectangle(0, 0, width, height, outline="", fill="#f8f8f6")
        self._draw_floor_grid(view)
        self._draw_world_axes(view)
        if points:
            for p0, p1 in zip(points, points[1:]):
                x0, y0, _ = self._project3(p0, view)
                x1, y1, _ = self._project3(p1, view)
                self.canvas.create_line(x0, y0, x1, y1, fill="#24598f", width=4, capstyle="round")
            for index, point in enumerate(points):
                x, y, depth = self._project3(point, view)
                radius = max(3.0, min(6.0, 5.0 - depth * 0.25))
                fill = "#1f4f82" if index + 1 < len(points) else "#163b61"
                self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline="#12314f", fill=fill)
        if tcp is not None:
            tx, ty, _ = self._project3(tcp, view)
            axis_points = axes if isinstance(axes, list) else []
            colors = ("#d62728", "#2ca02c", "#1f77b4")
            labels = ("X", "Y", "Z")
            for index, axis_point in enumerate(axis_points[:3]):
                if not isinstance(axis_point, list):
                    continue
                ax, ay, _ = self._project3(axis_point, view)
                self.canvas.create_line(tx, ty, ax, ay, fill=colors[index], width=3, arrow="last")
                self.canvas.create_text(ax + 8, ay, text=labels[index], fill=colors[index], font=("TkDefaultFont", 11, "bold"))
            self.canvas.create_oval(tx - 9, ty - 9, tx + 9, ty + 9, outline="#a83b00", width=2, fill="#ff7a1a", tags=("tcp_point",))
            self.canvas.create_text(tx + 12, ty - 14, text="TCP", fill="#a83b00", anchor="sw", font=("TkDefaultFont", 11, "bold"))
        self.canvas.create_text(
            18,
            18,
            anchor="nw",
            text="3D view: drag empty space to orbit, wheel to zoom, drag TCP point to edit",
            fill="#333333",
            font=("TkDefaultFont", 11),
        )

    def _draw_floor_grid(self, view: dict[str, object]) -> None:
        center = view["center"]
        span = 0.7
        step = 0.1
        z = 0.0
        start_x = math.floor((center[0] - span) / step) * step
        end_x = math.ceil((center[0] + span) / step) * step
        start_y = math.floor((center[1] - span) / step) * step
        end_y = math.ceil((center[1] + span) / step) * step
        x = start_x
        while x <= end_x + 1e-9:
            p0 = [x, start_y, z]
            p1 = [x, end_y, z]
            x0, y0, _ = self._project3(p0, view)
            x1, y1, _ = self._project3(p1, view)
            self.canvas.create_line(x0, y0, x1, y1, fill="#e5e5e0")
            x += step
        y = start_y
        while y <= end_y + 1e-9:
            p0 = [start_x, y, z]
            p1 = [end_x, y, z]
            x0, y0, _ = self._project3(p0, view)
            x1, y1, _ = self._project3(p1, view)
            self.canvas.create_line(x0, y0, x1, y1, fill="#e5e5e0")
            y += step

    def _draw_world_axes(self, view: dict[str, object]) -> None:
        origin = [0.0, 0.0, 0.0]
        axes = [([0.12, 0.0, 0.0], "#d62728", "base X"), ([0.0, 0.12, 0.0], "#2ca02c", "base Y"), ([0.0, 0.0, 0.12], "#1f77b4", "base Z")]
        ox, oy, _ = self._project3(origin, view)
        for point, color, label in axes:
            ax, ay, _ = self._project3(point, view)
            self.canvas.create_line(ox, oy, ax, ay, fill=color, width=2, arrow="last")
            self.canvas.create_text(ax + 6, ay + 6, text=label, fill=color, anchor="nw", font=("TkDefaultFont", 9))

    def _hit_tcp(self, x: float, y: float) -> bool:
        geometry = self._last_geometry or {}
        tcp = geometry.get("tcp_point")
        if not isinstance(tcp, list):
            return False
        tx, ty, _ = self._project3(tcp)
        return math.hypot(float(x) - tx, float(y) - ty) <= 18.0

    def _on_left_press(self, event) -> None:
        self._mouse_start_xy = (float(event.x), float(event.y))
        self._drag_start_vars = [float(var.get()) for var in self.vars]
        self._mouse_mode = "tcp" if self._hit_tcp(event.x, event.y) else "orbit"

    def _on_left_drag(self, event) -> None:
        if self._mouse_mode == "tcp":
            self._drag_tcp_to(float(event.x), float(event.y))
            return
        if self._mouse_mode == "orbit":
            start_x, start_y = self._mouse_start_xy
            dx = float(event.x) - start_x
            dy = float(event.y) - start_y
            self.view_yaw += dx * 0.006
            self.view_pitch = max(math.radians(-82.0), min(math.radians(82.0), self.view_pitch - dy * 0.006))
            self._mouse_start_xy = (float(event.x), float(event.y))
            self._redraw()

    def _on_left_release(self, _event) -> None:
        self._mouse_mode = None

    def _on_right_press(self, event) -> None:
        self._mouse_mode = "pan"
        self._mouse_start_xy = (float(event.x), float(event.y))

    def _on_right_drag(self, event) -> None:
        if self._mouse_mode != "pan":
            return
        start_x, start_y = self._mouse_start_xy
        self.view_pan[0] += float(event.x) - start_x
        self.view_pan[1] += float(event.y) - start_y
        self._mouse_start_xy = (float(event.x), float(event.y))
        self._redraw()

    def _on_mouse_wheel(self, event) -> None:
        delta = getattr(event, "delta", 0)
        num = getattr(event, "num", 0)
        zoom_in = delta > 0 or num == 4
        self.view_zoom *= 1.12 if zoom_in else 1.0 / 1.12
        self.view_zoom = max(0.25, min(8.0, self.view_zoom))
        self._redraw()

    def _drag_tcp_to(self, x: float, y: float) -> None:
        geometry = self._last_geometry or {}
        view = self._last_view
        link6_transform = geometry.get("link6_transform")
        if view is None or not isinstance(link6_transform, list):
            return
        start_x, start_y = self._mouse_start_xy
        dx = float(x) - start_x
        dy = float(y) - start_y
        plane = self.drag_plane_var.get()
        axis_pairs = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        first, second = axis_pairs.get(plane, (0, 1))
        local_axes = []
        for axis in (first, second):
            local_axes.append([float(link6_transform[row][axis]) for row in range(3)])
        right = view["right"]
        up = view["up"]
        scale = float(view["scale"])
        screen_axes = [
            [_dot(axis, right) * scale, -_dot(axis, up) * scale]
            for axis in local_axes
        ]
        a00, a10 = screen_axes[0]
        a01, a11 = screen_axes[1]
        det = a00 * a11 - a01 * a10
        if abs(det) <= 1e-9:
            return
        delta_first = (dx * a11 - a01 * dy) / det
        delta_second = (a00 * dy - dx * a10) / det
        updates = {first: delta_first * 1000.0, second: delta_second * 1000.0}
        limit = float(self.args.slider_range_mm)
        for axis, delta_mm in updates.items():
            value = self._drag_start_vars[axis] + delta_mm
            value = max(-limit, min(limit, value))
            self.vars[axis].set(value)
        self._redraw()

    def _canvas_width(self) -> int:
        width = int(self.canvas.winfo_width() or 0)
        return width if width >= 100 else 980

    def _canvas_height(self) -> int:
        height = int(self.canvas.winfo_height() or 0)
        return height if height >= 100 else 620

    def run(self) -> None:
        self.root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize or adjust an RM65 arm/tool TCP saved in a calibration profile.")
    parser.add_argument("--profile", required=True, help="Calibration profile name or path.")
    parser.add_argument("--arm", choices=("left", "right"), default="")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--expected-tool", default="", help="Optional active tool consistency check; defaults to profile tool_name.")
    parser.add_argument("--fit-json", default="")
    parser.add_argument("--allow-non-tool-source", action="store_true")
    parser.add_argument("--tcp-seed-source", choices=("auto", "com", "hand-base"), default="auto")
    parser.add_argument("--max-tcp-com-distance-m", type=float, default=0.20)
    parser.add_argument("--overwrite-tcp", action="store_true")
    parser.add_argument("--no-connect", action="store_true", help="Do not connect to the robot; GUI will not show current arm geometry.")
    parser.add_argument("--no-gui", action="store_true", help="Initialize/save profile from fit/defaults without opening the GUI.")
    parser.add_argument("--allow-controller-write", action="store_true", help="Reserved; controller TCP writes are currently disabled.")
    parser.add_argument("--confirm", default="", help=f"Reserved; controller TCP writes are currently disabled.")
    parser.add_argument("--slider-range-mm", type=float, default=300.0)
    parser.add_argument("--step-mm", type=float, default=1.0)
    parser.add_argument("--output-json", default="")
    return parser


def _write_report(path_text: str, report: dict[str, object]) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote report: {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.allow_controller_write or args.confirm == CONFIRM_TEXT:
        print("WARNING: RealMan controller TCP writes are disabled pending SDK TCP semantics review.")
    robot = None
    try:
        profile = load_profile(args.profile, arm=args.arm or None)
        arm = str(args.arm or profile["arm"])
        side, ip = _selected_ip(arm, args)
        expected_tool = args.expected_tool or str(profile.get("tool_name") or profile.get("name") or "")
        joint_deg = None
        current_tool_frame = None
        current_tool_pose = list(ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD)
        if not args.no_connect:
            robot = _connect_robot(side, ip)
            code, frame = robot.Get_Current_Tool_Frame()
            _require_zero(code, "Get_Current_Tool_Frame")
            current_tool_frame = frame
            current_tool = _tool_name_from_frame(frame)
            if expected_tool and expected_tool not in current_tool:
                raise RuntimeError(f"expected active tool containing '{expected_tool}', got '{current_tool}'")
            pose = _pose_from_tool_frame(frame)
            if pose is not None:
                current_tool_pose = pose
            joint_deg = _read_state(robot, side)["joint_deg"]
        fit_path, fit_payload, fit = _resolve_fit(profile, args)
        profile, profile_updated = _init_profile_from_fit_if_requested(
            profile,
            fit_path=fit_path,
            fit_payload=fit_payload,
            fit=fit,
            current_tool_pose=current_tool_pose,
            args=args,
        )
        if profile_updated and args.no_gui:
            saved = save_profile(profile, profile.get("_profile_path"))
            print(f"saved profile: {saved}")
        tcp = _tcp_from_profile(profile)
        print(f"profile={profile.get('_profile_path') or args.profile}")
        print(f"arm={arm} tool={profile.get('tool_name')}")
        print(f"tcp_pose_flange_m_rad={_format(tcp.get('pose_flange_m_rad', []), precision=5)}")
        print(f"tcp_position_source={tcp.get('position_source')} confirmed={tcp.get('confirmed')}")
        for warning in tcp.get("warnings", []):
            print(f"WARNING: {warning}")
        report = {
            "schema_version": 1,
            "data_type": "rm65_tool_tcp_adjust",
            "arm": arm,
            "tool_name": profile.get("tool_name"),
            "profile": profile.get("_profile_path") or args.profile,
            "fit_json": None if fit_path is None else str(fit_path),
            "tcp": tcp,
        }
        _write_report(args.output_json, report)
        if args.no_gui:
            return 0
        gui = TcpAdjustGui(
            profile=profile,
            profile_ref=args.profile,
            robot=robot,
            joint_deg=joint_deg,
            current_tool_frame=current_tool_frame,
            args=args,
        )
        gui.run()
        return 0
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if robot is not None:
            robot.Arm_Socket_Close()


if __name__ == "__main__":
    raise SystemExit(main())
