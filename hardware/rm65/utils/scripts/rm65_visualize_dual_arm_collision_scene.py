from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_motion.rm65 import (  # noqa: E402
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
    build_dual_arm_scene_profile,
)


def _parse_joint_deg(text: str, *, default_rad: np.ndarray, label: str) -> np.ndarray:
    stripped = str(text).strip()
    if not stripped:
        return np.asarray(default_rad, dtype=float).reshape(6).copy()
    values = [float(item.strip()) for item in stripped.split(",") if item.strip()]
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain 6 finite comma-separated degree values")
    return np.deg2rad(np.asarray(values, dtype=float))


def _embed_arm_qpos(profile: Any, model_q: np.ndarray, left_q: np.ndarray, right_q: np.ndarray) -> np.ndarray:
    q = np.asarray(model_q, dtype=float).reshape(-1).copy()
    if profile.solution_layout == "dual_arm_hand" and q.shape[0] >= 24:
        q[0:6] = left_q
        q[18:24] = right_q
        return q
    if q.shape[0] == 14:
        q[1:7] = left_q
        q[8:14] = right_q
        return q
    if q.shape[0] == 12:
        q[0:6] = left_q
        q[6:12] = right_q
        return q
    raise ValueError(f"unsupported model q shape {q.shape}")


def _geometry_side(name: str) -> str:
    if name.startswith("right_"):
        return "right"
    if name.startswith("arm_") or name.startswith("left_"):
        return "left"
    return "fixed"


def _geometry_label(name: str) -> str:
    return name.removesuffix("_0")


def _distance_value(result: Any) -> float | None:
    value = getattr(result, "min_distance", None)
    if value is None:
        value = getattr(result, "minimum_distance", None)
    if value is None:
        return None
    distance = float(value)
    return distance if math.isfinite(distance) else None


def _project(point: np.ndarray, *, axes: tuple[int, int], bounds: tuple[float, float, float, float], size: int) -> tuple[float, float]:
    min_a, max_a, min_b, max_b = bounds
    span_a = max(max_a - min_a, 1e-6)
    span_b = max(max_b - min_b, 1e-6)
    margin = 36.0
    usable = float(size) - 2.0 * margin
    x = margin + (float(point[axes[0]]) - min_a) / span_a * usable
    y = margin + (max_b - float(point[axes[1]])) / span_b * usable
    return x, y


def _svg_projection(
    *,
    title: str,
    axes: tuple[int, int],
    axis_names: tuple[str, str],
    points: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    size: int = 720,
) -> str:
    values = np.asarray(list(points.values()), dtype=float)
    min_a = float(np.min(values[:, axes[0]]))
    max_a = float(np.max(values[:, axes[0]]))
    min_b = float(np.min(values[:, axes[1]]))
    max_b = float(np.max(values[:, axes[1]]))
    pad = max(max_a - min_a, max_b - min_b, 0.1) * 0.12
    bounds = (min_a - pad, max_a + pad, min_b - pad, max_b + pad)

    lines: list[str] = [
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img">',
        f'<rect x="0" y="0" width="{size}" height="{size}" fill="#fafafa"/>',
        f'<text x="20" y="28" class="title">{html.escape(title)} ({axis_names[0]}-{axis_names[1]})</text>',
    ]
    for pair_index, (first, second) in enumerate(pairs, start=1):
        if first not in points or second not in points:
            continue
        x1, y1 = _project(points[first], axes=axes, bounds=bounds, size=size)
        x2, y2 = _project(points[second], axes=axes, bounds=bounds, size=size)
        lines.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" class="pair-line"/>')
        mid_x = (x1 + x2) * 0.5
        mid_y = (y1 + y2) * 0.5
        lines.append(
            f'<circle cx="{mid_x:.2f}" cy="{mid_y:.2f}" r="8" class="pair-index-bg"/>'
        )
        lines.append(
            f'<text x="{mid_x:.2f}" y="{mid_y + 3.5:.2f}" class="pair-index">{pair_index}</text>'
        )
    for name, point in sorted(points.items()):
        x, y = _project(point, axes=axes, bounds=bounds, size=size)
        side = _geometry_side(name)
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" class="geom geom-{side}"/>')
        lines.append(
            f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" class="geom-label">{html.escape(_geometry_label(name))}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def _frame_point(robot: Any, frame_name: str) -> list[float] | None:
    frame_id = robot.model.getFrameId(frame_name)
    if frame_id >= len(robot.model.frames):
        return None
    return [float(value) for value in np.asarray(robot.data.oMf[frame_id].translation, dtype=float).reshape(3)]


def _link_chain(robot: Any, *, name: str, side: str, frame_names: list[str]) -> dict[str, object]:
    points: list[dict[str, object]] = []
    for frame_name in frame_names:
        xyz = _frame_point(robot, frame_name)
        if xyz is None:
            continue
        points.append({"name": frame_name, "xyz": xyz})
    return {"name": name, "side": side, "points": points}


def _box_vertices(transform: Any, size_xyz: tuple[float, float, float]) -> list[list[float]]:
    size = np.asarray(size_xyz, dtype=float).reshape(3)
    corners = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                corners.append(np.array([sx * size[0], sy * size[1], sz * size[2]], dtype=float))
    rotation = np.asarray(transform.rotation, dtype=float)
    translation = np.asarray(transform.translation, dtype=float).reshape(3)
    return [
        [float(value) for value in (translation + rotation @ corner).reshape(3)]
        for corner in corners
    ]


def _scene_box_records(profile: Any, robot: Any) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    box_specs = [*profile.hand_proxy_boxes, *profile.fixed_scene_boxes]
    edges = (
        (0, 1),
        (0, 2),
        (0, 4),
        (3, 1),
        (3, 2),
        (3, 7),
        (5, 1),
        (5, 4),
        (5, 7),
        (6, 2),
        (6, 4),
        (6, 7),
    )
    for spec in box_specs:
        geometry_id = robot.collision_model.getGeometryId(spec.geometry_name)
        if geometry_id >= len(robot.collision_model.geometryObjects):
            continue
        side = getattr(spec, "side", "fixed")
        records.append(
            {
                "name": spec.geometry_name,
                "side": side,
                "size": list(spec.box_size_xyz),
                "vertices": _box_vertices(robot.collision_data.oMg[geometry_id], spec.box_size_xyz),
                "edges": [list(edge) for edge in edges],
            }
        )
    return records


def _scene_payload(
    *,
    profile: Any,
    robot: Any,
    geometry_points: dict[str, np.ndarray],
    pair_distances: list[tuple[int, str, str, float | None]],
) -> dict[str, object]:
    left_chain = _link_chain(
        robot,
        name="left_arm",
        side="left",
        frame_names=[
            "base_link",
            "arm_link1",
            "arm_link3",
            "arm_link4",
            "arm_link5",
            "arm_link6",
            "hand_base",
            "left_hand_proxy",
        ],
    )
    right_chain = _link_chain(
        robot,
        name="right_arm",
        side="right",
        frame_names=[
            "right_base_link",
            "right_arm_link1",
            "right_arm_link3",
            "right_arm_link4",
            "right_arm_link5",
            "right_arm_link6",
            "right_hand_base",
            "right_hand_proxy",
        ],
    )
    points = [
        {
            "name": name,
            "label": _geometry_label(name),
            "side": _geometry_side(name),
            "xyz": [float(value) for value in point.reshape(3)],
        }
        for name, point in sorted(geometry_points.items())
    ]
    pairs = [
        {
            "index": int(index) + 1,
            "a": first,
            "b": second,
            "distance": None if distance is None else float(distance),
        }
        for index, first, second, distance in pair_distances
    ]
    all_points: list[list[float]] = []
    for point in points:
        all_points.append(list(point["xyz"]))
    for chain in (left_chain, right_chain):
        for point in chain["points"]:
            all_points.append(list(point["xyz"]))
    center = np.mean(np.asarray(all_points, dtype=float), axis=0) if all_points else np.zeros(3)
    return {
        "profile": profile.name,
        "points": points,
        "pairs": pairs,
        "chains": [left_chain, right_chain],
        "boxes": _scene_box_records(profile, robot),
        "center": [float(value) for value in center.reshape(3)],
    }


def _build_html(
    *,
    profile: Any,
    q: np.ndarray,
    geometry_points: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    pair_distances: list[tuple[int, str, str, float | None]],
    scene_payload: dict[str, object],
) -> str:
    closest = sorted(
        [item for item in pair_distances if item[3] is not None],
        key=lambda item: float(item[3]),
    )[:12]
    pair_rows = "\n".join(
        "<tr>"
        f"<td>{index + 1}</td>"
        f"<td>{html.escape(first)}</td>"
        f"<td>{html.escape(second)}</td>"
        f"<td>{'' if distance is None else f'{distance:.5f}'}</td>"
        "</tr>"
        for index, first, second, distance in pair_distances
    )
    closest_rows = "\n".join(
        "<tr>"
        f"<td>{index + 1}</td>"
        f"<td>{html.escape(first)}</td>"
        f"<td>{html.escape(second)}</td>"
        f"<td>{distance:.5f}</td>"
        "</tr>"
        for index, first, second, distance in closest
    )
    top_svg = _svg_projection(
        title="Top view",
        axes=(0, 1),
        axis_names=("X", "Y"),
        points=geometry_points,
        pairs=pairs,
    )
    side_svg = _svg_projection(
        title="Front view",
        axes=(1, 2),
        axis_names=("Y", "Z"),
        points=geometry_points,
        pairs=pairs,
    )
    q_deg = ", ".join(f"{value:.2f}" for value in np.rad2deg(q).tolist())
    hand_proxy = ", ".join(
        f"{spec.geometry_name}@{spec.parent_link_name} size={spec.box_size_xyz}"
        for spec in profile.hand_proxy_boxes
    ) or "none"
    fixed_boxes = ", ".join(
        f"{spec.geometry_name}@{spec.parent_link_name} size={spec.box_size_xyz}"
        for spec in profile.fixed_scene_boxes
    ) or "none"
    scene_json = json.dumps(scene_payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RM65 Dual-Arm Collision Scene</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #202124; }}
.summary {{ display: grid; grid-template-columns: max-content 1fr; gap: 6px 14px; margin-bottom: 18px; }}
.viewer-wrap {{ border: 1px solid #bbb; background: #f8fafc; margin: 16px 0 20px; }}
#scene3d {{ display: block; width: 100%; height: 720px; cursor: grab; }}
#scene3d.dragging {{ cursor: grabbing; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center; padding: 10px 12px; border-bottom: 1px solid #d6d6d6; background: #fff; }}
.toolbar label {{ white-space: nowrap; font-size: 14px; }}
.hint {{ color: #5f6368; font-size: 13px; }}
.viz {{ display: flex; flex-wrap: wrap; gap: 18px; margin-top: 14px; }}
svg {{ border: 1px solid #ccc; background: #fafafa; }}
.title {{ font-size: 18px; font-weight: 700; }}
    .pair-line {{ stroke: #374151; stroke-opacity: 0.55; stroke-width: 2; }}
    .pair-index-bg {{ fill: rgba(255,255,255,0.88); stroke: #374151; stroke-width: 1; }}
    .pair-index {{ font-size: 9px; fill: #111827; text-anchor: middle; font-weight: 700; }}
.geom {{ stroke: #222; stroke-width: 1; }}
.geom-left {{ fill: #3b82f6; }}
.geom-right {{ fill: #f97316; }}
.geom-fixed {{ fill: #ef4444; }}
.geom-label {{ font-size: 10px; fill: #111; }}
table {{ border-collapse: collapse; margin-top: 14px; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 5px 8px; font-size: 13px; }}
th {{ background: #f3f4f6; text-align: left; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>RM65 Dual-Arm Collision Scene</h1>
<div class="summary">
<strong>profile</strong><span><code>{html.escape(profile.name)}</code></span>
<strong>urdf</strong><span><code>{html.escape(str(profile.urdf_path))}</code></span>
<strong>collision pairs</strong><span>{len(pairs)}</span>
<strong>q deg</strong><span>{html.escape(q_deg)}</span>
<strong>hand proxy</strong><span>{html.escape(hand_proxy)}</span>
<strong>fixed boxes</strong><span>{html.escape(fixed_boxes)}</span>
</div>
<p>Blue is left arm, orange is right arm, red is fixed scene geometry. Thick colored lines are link chains, translucent boxes are hand/camera proxies, and gray lines are the configured collision pairs checked by Pinocchio/HPP-FCL.</p>
<div class="viewer-wrap">
  <div class="toolbar">
    <label><input id="togglePairs" type="checkbox" checked> collision pairs</label>
    <label><input id="toggleLabels" type="checkbox" checked> labels</label>
    <label><input id="togglePoints" type="checkbox" checked> collision points</label>
    <label><input id="toggleBoxes" type="checkbox" checked> proxy boxes</label>
    <span class="hint">Drag to rotate, wheel to zoom, double-click to reset.</span>
  </div>
  <canvas id="scene3d"></canvas>
</div>
<details>
<summary>Static top/front fallback views</summary>
<div class="viz">{top_svg}{side_svg}</div>
</details>
<h2>Closest Pair Distances</h2>
<table><thead><tr><th>#</th><th>Geometry A</th><th>Geometry B</th><th>distance m</th></tr></thead><tbody>{closest_rows}</tbody></table>
<h2>All Collision Pairs</h2>
<table><thead><tr><th>#</th><th>Geometry A</th><th>Geometry B</th><th>distance m</th></tr></thead><tbody>{pair_rows}</tbody></table>
<script id="scene-data" type="application/json">{scene_json}</script>
<script>
(() => {{
  const scene = JSON.parse(document.getElementById("scene-data").textContent);
  const canvas = document.getElementById("scene3d");
  const ctx = canvas.getContext("2d");
  const toggles = {{
    pairs: document.getElementById("togglePairs"),
    labels: document.getElementById("toggleLabels"),
    points: document.getElementById("togglePoints"),
    boxes: document.getElementById("toggleBoxes"),
  }};
  const color = {{
    left: "#2563eb",
    right: "#ea580c",
    fixed: "#dc2626",
    pair: "rgba(31,41,55,0.42)",
    grid: "rgba(90,90,90,0.14)",
    text: "#111827",
  }};
  let state = {{ yaw: -0.75, pitch: 0.48, scale: 520, panX: 0, panY: 20 }};
  let dragging = false;
  let last = {{ x: 0, y: 0 }};
  const center = scene.center || [0, 0, 0];
  const pointsByName = new Map(scene.points.map((point) => [point.name, point]));

  function resize() {{
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    draw();
  }}

  function rotate(point) {{
    const x = point[0] - center[0];
    const y = point[1] - center[1];
    const z = point[2] - center[2];
    const cy = Math.cos(state.yaw);
    const sy = Math.sin(state.yaw);
    const cp = Math.cos(state.pitch);
    const sp = Math.sin(state.pitch);
    const x1 = cy * x - sy * y;
    const y1 = sy * x + cy * y;
    const z1 = z;
    const y2 = cp * y1 - sp * z1;
    const z2 = sp * y1 + cp * z1;
    return [x1, y2, z2];
  }}

  function project(point) {{
    const rotated = rotate(point);
    const rect = canvas.getBoundingClientRect();
    return {{
      x: rect.width / 2 + state.panX + rotated[0] * state.scale,
      y: rect.height / 2 + state.panY - rotated[2] * state.scale,
      depth: rotated[1],
    }};
  }}

  function line3(a, b, stroke, width = 1, dash = []) {{
    const pa = project(a);
    const pb = project(b);
    ctx.save();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
    ctx.restore();
  }}

  function pairIndex3(a, b, text, close) {{
    const mid = [
      (a[0] + b[0]) * 0.5,
      (a[1] + b[1]) * 0.5,
      (a[2] + b[2]) * 0.5,
    ];
    const p = project(mid);
    ctx.save();
    ctx.fillStyle = close ? "rgba(254,242,242,0.94)" : "rgba(255,255,255,0.9)";
    ctx.strokeStyle = close ? "#dc2626" : "#374151";
    ctx.lineWidth = close ? 2 : 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, close ? 10 : 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = close ? "#991b1b" : "#111827";
    ctx.font = "bold 10px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(text), p.x, p.y + 0.5);
    ctx.restore();
  }}

  function circle3(point, radius, fill, stroke = "#111827") {{
    const p = project(point);
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
    return p;
  }}

  function text3(point, text, dx = 8, dy = -8) {{
    const p = project(point);
    ctx.fillStyle = color.text;
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText(text, p.x + dx, p.y + dy);
  }}

  function drawGrid() {{
    const z = 0;
    for (let v = -0.8; v <= 0.8; v += 0.1) {{
      line3([-0.55, v, z], [0.55, v, z], color.grid, 1);
      line3([v, -0.55, z], [v, 0.55, z], color.grid, 1);
    }}
    line3([0, 0, 0], [0.18, 0, 0], "#ef4444", 2);
    line3([0, 0, 0], [0, 0.18, 0], "#22c55e", 2);
    line3([0, 0, 0], [0, 0, 0.18], "#3b82f6", 2);
    text3([0.19, 0, 0], "X", 2, 4);
    text3([0, 0.19, 0], "Y", 2, 4);
    text3([0, 0, 0.19], "Z", 2, 4);
  }}

  function drawBoxes() {{
    if (!toggles.boxes.checked) return;
    for (const box of scene.boxes) {{
      const sideColor = color[box.side] || color.fixed;
      for (const edge of box.edges) {{
        line3(box.vertices[edge[0]], box.vertices[edge[1]], sideColor, 2);
      }}
      if (toggles.labels.checked && box.vertices.length) {{
        text3(box.vertices[0], box.name);
      }}
    }}
  }}

  function drawChains() {{
    for (const chain of scene.chains) {{
      const chainColor = color[chain.side] || "#111827";
      const points = chain.points || [];
      for (let i = 0; i < points.length - 1; i += 1) {{
        line3(points[i].xyz, points[i + 1].xyz, chainColor, 5);
      }}
      for (const point of points) {{
        circle3(point.xyz, 4, chainColor);
        if (toggles.labels.checked) text3(point.xyz, point.name);
      }}
    }}
  }}

  function drawPairs() {{
    if (!toggles.pairs.checked) return;
    for (const pair of scene.pairs) {{
      const a = pointsByName.get(pair.a);
      const b = pointsByName.get(pair.b);
      if (!a || !b) continue;
      const distance = pair.distance == null ? 1 : pair.distance;
      const close = distance < 0.08;
      line3(a.xyz, b.xyz, close ? "rgba(220,38,38,0.8)" : color.pair, close ? 3 : 2);
      if (toggles.labels.checked) pairIndex3(a.xyz, b.xyz, pair.index, close);
    }}
  }}

  function drawPoints() {{
    if (!toggles.points.checked) return;
    const ordered = [...scene.points].sort((a, b) => project(a.xyz).depth - project(b.xyz).depth);
    for (const point of ordered) {{
      const sideColor = color[point.side] || color.fixed;
      circle3(point.xyz, 5, sideColor);
      if (toggles.labels.checked) text3(point.xyz, point.label || point.name);
    }}
  }}

  function draw() {{
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(0, 0, rect.width, rect.height);
    drawGrid();
    drawPairs();
    drawBoxes();
    drawChains();
    drawPoints();
    ctx.fillStyle = "rgba(255,255,255,0.86)";
    ctx.fillRect(12, rect.height - 76, 310, 56);
    ctx.fillStyle = "#111827";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("left: blue  right: orange  fixed: red", 24, rect.height - 50);
    ctx.fillText("close pairs (<0.08m) are red", 24, rect.height - 28);
  }}

  canvas.addEventListener("mousedown", (event) => {{
    dragging = true;
    last = {{ x: event.clientX, y: event.clientY }};
    canvas.classList.add("dragging");
  }});
  window.addEventListener("mouseup", () => {{
    dragging = false;
    canvas.classList.remove("dragging");
  }});
  window.addEventListener("mousemove", (event) => {{
    if (!dragging) return;
    const dx = event.clientX - last.x;
    const dy = event.clientY - last.y;
    last = {{ x: event.clientX, y: event.clientY }};
    state.yaw += dx * 0.008;
    state.pitch = Math.max(-1.35, Math.min(1.35, state.pitch + dy * 0.008));
    draw();
  }});
  canvas.addEventListener("wheel", (event) => {{
    event.preventDefault();
    const factor = Math.exp(-event.deltaY * 0.001);
    state.scale = Math.max(120, Math.min(1600, state.scale * factor));
    draw();
  }}, {{ passive: false }});
  canvas.addEventListener("dblclick", () => {{
    state = {{ yaw: -0.75, pitch: 0.48, scale: 520, panX: 0, panY: 20 }};
    draw();
  }});
  Object.values(toggles).forEach((input) => input.addEventListener("change", draw));
  window.addEventListener("resize", resize);
  resize();
}})();
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize and list RM65 dual-arm URDF collision pairs.")
    parser.add_argument("--scene-profile", default="dual_arm_hand_proxy_open")
    parser.add_argument("--left-joint-deg", default="", help="Optional 6D left q in degrees. Empty uses default q.")
    parser.add_argument("--right-joint-deg", default="", help="Optional 6D right q in degrees. Empty uses default q.")
    parser.add_argument(
        "--output-html",
        default="hardware/rm65/utils/results/rm65_dual_arm_collision_scene.html",
    )
    parser.add_argument("--print-pairs", action="store_true", help="Print the configured collision pairs and distances.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import pinocchio as pin
    except ModuleNotFoundError as exc:
        raise RuntimeError("This utility requires pinocchio in the active environment.") from exc

    profile = build_dual_arm_scene_profile(args.scene_profile)
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=str(profile.urdf_path),
        package_dirs=[str(path) for path in profile.package_dirs],
        root_joint=None,
    )
    for first, second in profile.collision_pairs:
        robot.collision_model.addCollisionPair(
            pin.CollisionPair(
                robot.collision_model.getGeometryId(first),
                robot.collision_model.getGeometryId(second),
            )
        )
    robot.collision_data = pin.GeometryData(robot.collision_model)

    left_q = _parse_joint_deg(args.left_joint_deg, default_rad=DEFAULT_LEFT_ARM_QPOS, label="--left-joint-deg")
    right_q = _parse_joint_deg(args.right_joint_deg, default_rad=DEFAULT_RIGHT_ARM_QPOS, label="--right-joint-deg")
    q = _embed_arm_qpos(profile, np.zeros(robot.model.nq, dtype=float), left_q, right_q)
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q)
    pin.computeDistances(robot.model, robot.data, robot.collision_model, robot.collision_data, q)

    active_geometry_names = {
        geometry_name
        for pair in profile.collision_pairs
        for geometry_name in pair
    }
    geometry_points = {
        geometry.name: np.asarray(robot.collision_data.oMg[index].translation, dtype=float).reshape(3)
        for index, geometry in enumerate(robot.collision_model.geometryObjects)
        if geometry.name in active_geometry_names
    }
    pair_distances: list[tuple[int, str, str, float | None]] = []
    for index, pair in enumerate(robot.collision_model.collisionPairs):
        first = robot.collision_model.geometryObjects[pair.first].name
        second = robot.collision_model.geometryObjects[pair.second].name
        pair_distances.append((index, first, second, _distance_value(robot.collision_data.distanceResults[index])))

    if args.print_pairs:
        print(f"profile={profile.name} urdf={profile.urdf_path} pairs={len(profile.collision_pairs)}")
        for index, first, second, distance in pair_distances:
            distance_text = "n/a" if distance is None else f"{distance:.5f}m"
            print(f"{index + 1:02d}: {first} <-> {second} distance={distance_text}")

    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _build_html(
            profile=profile,
            q=q,
            geometry_points=geometry_points,
            pairs=list(profile.collision_pairs),
            pair_distances=pair_distances,
            scene_payload=_scene_payload(
                profile=profile,
                robot=robot,
                geometry_points=geometry_points,
                pair_distances=pair_distances,
            ),
        ),
        encoding="utf-8",
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
