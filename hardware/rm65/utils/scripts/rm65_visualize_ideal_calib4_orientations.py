from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_DIR = SCRIPT_DIR.parent
DATA_DIR = UTILS_DIR / "data"
POSE_DIR = DATA_DIR / "poses"
RESULTS_DIR = UTILS_DIR / "results"

GRAVITY_BASE = [0.0, 0.0, -1.0]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _normalize(values: Iterable[float]) -> list[float]:
    vector = [float(value) for value in values]
    scale = _norm(vector)
    if scale <= 1e-12:
        return vector
    return [value / scale for value in vector]


def _mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    b_t = list(zip(*b))
    return [[_dot(row, list(col)) for col in b_t] for row in a]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def _rot_x(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _rot_y(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rot_z(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _parse_float_list(text: str, *, expected_len: int, label: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != expected_len:
        raise RuntimeError(f"{label} must contain {expected_len} comma-separated values")
    return values


def _parse_label_list(text: str, *, expected_len: int) -> list[str]:
    labels = [item.strip() for item in text.split(",") if item.strip()]
    if len(labels) != expected_len:
        raise RuntimeError("--labels must contain 4 comma-separated values")
    return labels


def _round_vector(values: Iterable[float]) -> list[float]:
    return [round(float(value), 6) for value in values]


def _rotation_from_rpy(pose: list[float], order: str) -> list[list[float]]:
    angles = {"x": pose[3], "y": pose[4], "z": pose[5]}
    rotations = {"x": _rot_x, "y": _rot_y, "z": _rot_z}
    matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for axis in order:
        matrix = _mat_mul(matrix, rotations[axis](angles[axis]))
    return matrix


def _angle_deg(a: list[float], b: list[float]) -> float:
    clamped = max(-1.0, min(1.0, _dot(_normalize(a), _normalize(b))))
    return math.degrees(math.acos(clamped))


def _spread_stats(items: list[dict[str, object]]) -> dict[str, object]:
    vectors = [item["gravity_tool"] for item in items]
    angles = []
    for i, left in enumerate(vectors):
        for j, right in enumerate(vectors):
            if j <= i:
                continue
            angles.append(_angle_deg(left, right))
    return {
        "min_angle_deg": round(min(angles), 3) if angles else 0.0,
        "max_angle_deg": round(max(angles), 3) if angles else 0.0,
        "angles_deg": [round(value, 3) for value in angles],
    }


def _orientation_payload(
    label: str,
    gravity_tool: list[float],
    *,
    kind: str,
    pitch_deg: float | None = None,
    roll_deg: float | None = None,
    pre_tilt_deg: float | None = None,
    joint_deg: list[float] | None = None,
    pose: list[float] | None = None,
) -> dict[str, object]:
    payload = {
        "label": label,
        "kind": kind,
        "gravity_tool": _round_vector(_normalize(gravity_tool)),
    }
    if pitch_deg is not None:
        payload["pitch_deg"] = round(float(pitch_deg), 6)
    if roll_deg is not None:
        payload["roll_deg"] = round(float(roll_deg), 6)
    if pre_tilt_deg is not None:
        payload["pre_tilt_deg"] = round(float(pre_tilt_deg), 6)
    if joint_deg is not None:
        payload["joint_deg"] = _round_vector(joint_deg)
    if pose is not None:
        payload["pose"] = _round_vector(pose)
    return payload


def _ideal_orientation(
    label: str,
    *,
    pre_tilt_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> dict[str, object]:
    # The documented RealMan RM65 calibration points keep an upstream joint at
    # about -60 deg and then sweep wrist pitch/roll. For a calibration-quality
    # visualization we only need where gravity lands in tool coordinates.
    rotation = _mat_mul(
        _mat_mul(_rot_y(math.radians(pre_tilt_deg)), _rot_y(math.radians(pitch_deg))),
        _rot_z(math.radians(roll_deg)),
    )
    gravity_tool = _mat_vec(_transpose(rotation), GRAVITY_BASE)
    return _orientation_payload(
        label,
        gravity_tool,
        kind="ideal",
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        pre_tilt_deg=pre_tilt_deg,
    )


def _load_recorded_orientations(path: Path, order: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    poses = data.get("poses", data.get("records", []))
    out = []
    for index, item in enumerate(poses[:4], start=1):
        pose = item.get("pose", item.get("pose_mean"))
        joint_deg = item.get("joint_deg", item.get("joint_deg_mean"))
        if pose is None:
            continue
        pose = [float(value) for value in pose]
        rotation = _rotation_from_rpy(pose, order)
        gravity_tool = _mat_vec(_transpose(rotation), GRAVITY_BASE)
        out.append(
            _orientation_payload(
                f"R{index}",
                gravity_tool,
                kind="recorded",
                joint_deg=[float(value) for value in joint_deg] if joint_deg else None,
                pose=pose,
            )
        )
    return out


def _build_html(
    *,
    ideal: list[dict[str, object]],
    recorded: list[dict[str, object]],
    pose_json: str,
    rpy_order: str,
    pre_tilt_deg: float,
    pitch_deg: list[float],
    roll_deg: list[float],
) -> str:
    payload = json.dumps(
        {
            "pose_json": pose_json,
            "rpy_order": rpy_order,
            "pre_tilt_deg": pre_tilt_deg,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "ideal": ideal,
            "recorded": recorded,
            "ideal_stats": _spread_stats(ideal),
            "recorded_stats": _spread_stats(recorded),
        },
        indent=2,
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RM65 4-Point Gravity Coverage</title>
<style>
  :root {{
    color-scheme: light;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f3f5f7;
    color: #1b2430;
  }}
  body {{
    margin: 0;
    padding: 20px;
  }}
  main {{
    max-width: 1320px;
    margin: 0 auto;
  }}
  header {{
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 16px;
    align-items: end;
    margin-bottom: 12px;
  }}
  h1 {{
    margin: 0 0 6px;
    font-size: 24px;
    letter-spacing: 0;
  }}
  p {{
    margin: 0;
    color: #536171;
    line-height: 1.45;
  }}
  .metrics {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
  }}
  .metric {{
    min-width: 132px;
    padding: 8px 10px;
    border: 1px solid #d5dde7;
    border-radius: 8px;
    background: #ffffff;
  }}
  .metric span {{
    display: block;
    font-size: 12px;
    color: #617080;
  }}
  .metric strong {{
    display: block;
    margin-top: 2px;
    font-size: 17px;
  }}
  .toolbar {{
    display: grid;
    grid-template-columns: repeat(5, minmax(120px, 1fr));
    gap: 10px;
    padding: 12px;
    margin-bottom: 10px;
    background: #ffffff;
    border: 1px solid #d5dde7;
    border-radius: 8px;
  }}
  label {{
    display: grid;
    gap: 4px;
    font-size: 13px;
    color: #46525f;
  }}
  input[type="range"] {{
    width: 100%;
  }}
  .checks {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    font-size: 13px;
  }}
  .checks label {{
    display: inline-flex;
    grid-template-columns: none;
    align-items: center;
    gap: 6px;
  }}
  canvas {{
    width: 100%;
    height: auto;
    display: block;
    background: #ffffff;
    border: 1px solid #d5dde7;
    border-radius: 8px;
    touch-action: none;
  }}
  .hint {{
    margin-top: 8px;
    color: #617080;
    font-size: 13px;
  }}
  @media (max-width: 920px) {{
    header {{
      grid-template-columns: 1fr;
    }}
    .metrics {{
      justify-content: flex-start;
    }}
    .toolbar {{
      grid-template-columns: 1fr 1fr;
    }}
  }}
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>RM65 4-Point Gravity Coverage</h1>
      <p>
        The useful calibration signal is where base gravity lands in the tool frame.
        This page shows those four unit vectors, not the robot workspace.
      </p>
    </div>
    <div class="metrics">
      <div class="metric"><span>ideal min angle</span><strong id="idealMin"></strong></div>
      <div class="metric"><span>recorded min angle</span><strong id="recordedMin"></strong></div>
      <div class="metric"><span>recorded poses</span><strong id="recordedCount"></strong></div>
    </div>
  </header>
  <section class="toolbar">
    <label>pre-tilt deg <span id="preTiltValue"></span>
      <input id="preTilt" type="range" min="-140" max="80" step="1">
    </label>
    <label>yaw <span id="yawValue"></span>
      <input id="yaw" type="range" min="-180" max="180" step="1">
    </label>
    <label>pitch <span id="pitchValue"></span>
      <input id="camPitch" type="range" min="-80" max="80" step="1">
    </label>
    <label>zoom <span id="zoomValue"></span>
      <input id="zoom" type="range" min="70" max="135" step="1">
    </label>
    <div class="checks">
      <label><input id="showIdeal" type="checkbox" checked> ideal</label>
      <label><input id="showRecorded" type="checkbox" checked> recorded</label>
      <label><input id="showMatrix" type="checkbox" checked> matrix</label>
    </div>
  </section>
  <canvas id="view" width="1320" height="820"></canvas>
  <div class="hint">
    Drag the large sphere to rotate it. The three right-hand disks are XY, XZ, and YZ
    projections, so overlap cannot hide coverage quality.
  </div>
</main>
<script>
const initialData = {payload};
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const controls = {{
  preTilt: document.getElementById("preTilt"),
  yaw: document.getElementById("yaw"),
  camPitch: document.getElementById("camPitch"),
  zoom: document.getElementById("zoom"),
  showIdeal: document.getElementById("showIdeal"),
  showRecorded: document.getElementById("showRecorded"),
  showMatrix: document.getElementById("showMatrix")
}};
const labels = {{
  preTilt: document.getElementById("preTiltValue"),
  yaw: document.getElementById("yawValue"),
  pitch: document.getElementById("pitchValue"),
  zoom: document.getElementById("zoomValue"),
  idealMin: document.getElementById("idealMin"),
  recordedMin: document.getElementById("recordedMin"),
  recordedCount: document.getElementById("recordedCount")
}};
const colors = {{
  bg: "#ffffff",
  panel: "#f8fafc",
  border: "#d5dde7",
  text: "#1b2430",
  muted: "#617080",
  grid: "#e1e7ef",
  ideal: "#2563eb",
  idealSoft: "rgba(37, 99, 235, 0.12)",
  recorded: "#f59e0b",
  recordedSoft: "rgba(245, 158, 11, 0.16)",
  axisX: "#d3342f",
  axisY: "#269653",
  axisZ: "#2563eb",
  good: "#15803d",
  warn: "#b45309",
  bad: "#b91c1c"
}};
let state = {{
  preTilt: initialData.pre_tilt_deg,
  yaw: -35,
  pitch: 22,
  zoom: 100,
  dragging: false,
  lastX: 0,
  lastY: 0
}};
controls.preTilt.value = state.preTilt;
controls.yaw.value = state.yaw;
controls.camPitch.value = state.pitch;
controls.zoom.value = state.zoom;

function degToRad(value) {{
  return value * Math.PI / 180;
}}

function dot(a, b) {{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}}

function norm(v) {{
  return Math.hypot(v[0], v[1], v[2]);
}}

function normalize(v) {{
  const n = norm(v);
  return n > 1e-12 ? [v[0] / n, v[1] / n, v[2] / n] : v;
}}

function angleDeg(a, b) {{
  const c = Math.max(-1, Math.min(1, dot(normalize(a), normalize(b))));
  return Math.acos(c) * 180 / Math.PI;
}}

function rotY(angle) {{
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
}}

function rotX(angle) {{
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [[1, 0, 0], [0, c, -s], [0, s, c]];
}}

function rotZ(angle) {{
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
}}

function matMul(a, b) {{
  return a.map(row => [0, 1, 2].map(col => row[0] * b[0][col] + row[1] * b[1][col] + row[2] * b[2][col]));
}}

function matVec(m, v) {{
  return [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
}}

function transpose(m) {{
  return [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]];
}}

function idealSet() {{
  return initialData.pitch_deg.map((pitch, index) => {{
    const roll = initialData.roll_deg[index];
    const rotation = matMul(matMul(rotY(degToRad(state.preTilt)), rotY(degToRad(pitch))), rotZ(degToRad(roll)));
    return {{
      label: `P${{index + 1}}`,
      kind: "ideal",
      gravity_tool: normalize(matVec(transpose(rotation), [0, 0, -1])),
      pitch_deg: pitch,
      roll_deg: roll
    }};
  }});
}}

function currentData() {{
  const ideal = idealSet();
  const recorded = initialData.recorded.map(item => ({{
    ...item,
    gravity_tool: normalize(item.gravity_tool)
  }}));
  return {{ideal, recorded}};
}}

function minAngle(items) {{
  let best = Infinity;
  for (let i = 0; i < items.length; i++) {{
    for (let j = i + 1; j < items.length; j++) {{
      best = Math.min(best, angleDeg(items[i].gravity_tool, items[j].gravity_tool));
    }}
  }}
  return Number.isFinite(best) ? best : 0;
}}

function scoreColor(angle) {{
  if (angle >= 75) return colors.good;
  if (angle >= 45) return colors.warn;
  return colors.bad;
}}

function cameraMatrix() {{
  return matMul(rotX(degToRad(state.pitch)), rotZ(degToRad(state.yaw)));
}}

function projectSphere(v, center, radius) {{
  const p = matVec(cameraMatrix(), v);
  return {{
    x: center.x + p[0] * radius,
    y: center.y - p[1] * radius,
    depth: p[2],
    raw: p
  }};
}}

function drawPanel(x, y, w, h, title) {{
  ctx.save();
  ctx.fillStyle = colors.bg;
  ctx.strokeStyle = colors.border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = colors.text;
  ctx.font = "700 16px system-ui, sans-serif";
  ctx.fillText(title, x + 16, y + 26);
  ctx.restore();
}}

function drawSphere(items, x, y, w, h) {{
  drawPanel(x, y, w, h, "g_tool on unit sphere");
  const center = {{x: x + w * 0.52, y: y + h * 0.54}};
  const radius = Math.min(w, h) * 0.34 * state.zoom / 100;
  ctx.save();
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  for (let k = -2; k <= 2; k++) {{
    ctx.beginPath();
    ctx.ellipse(center.x, center.y, radius, radius * Math.cos(k * Math.PI / 12), 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(center.x, center.y, radius * Math.cos(k * Math.PI / 12), radius, 0, 0, Math.PI * 2);
    ctx.stroke();
  }}
  ctx.strokeStyle = "#9aa7b6";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  drawSphereAxis([1, 0, 0], "X", colors.axisX, center, radius);
  drawSphereAxis([0, 1, 0], "Y", colors.axisY, center, radius);
  drawSphereAxis([0, 0, 1], "Z", colors.axisZ, center, radius);

  const sorted = items
    .map(item => ({{...item, p: projectSphere(item.gravity_tool, center, radius)}}))
    .sort((a, b) => a.p.depth - b.p.depth);
  for (const item of sorted) {{
    const color = item.kind === "ideal" ? colors.ideal : colors.recorded;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = item.p.depth < 0 ? 0.32 : 0.9;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(center.x, center.y);
    ctx.lineTo(item.p.x, item.p.y);
    ctx.stroke();
    ctx.fillStyle = item.kind === "ideal" ? colors.idealSoft : colors.recordedSoft;
    ctx.beginPath();
    ctx.arc(item.p.x, item.p.y, item.kind === "ideal" ? 13 : 15, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(item.p.x, item.p.y, item.kind === "ideal" ? 6 : 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "700 15px system-ui, sans-serif";
    ctx.fillText(item.label, item.p.x + 10, item.p.y - 8);
    ctx.restore();
  }}
}}

function drawSphereAxis(v, label, color, center, radius) {{
  const p = projectSphere(v, center, radius * 1.08);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(center.x, center.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  ctx.font = "700 14px system-ui, sans-serif";
  ctx.fillText(label, p.x + 5, p.y - 5);
  ctx.restore();
}}

function drawProjection(items, x, y, w, h, title, ax, ay) {{
  drawPanel(x, y, w, h, title);
  const cx = x + w * 0.5;
  const cy = y + h * 0.57;
  const r = Math.min(w, h) * 0.30;
  ctx.save();
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx - r, cy);
  ctx.lineTo(cx + r, cy);
  ctx.moveTo(cx, cy - r);
  ctx.lineTo(cx, cy + r);
  ctx.stroke();
  ctx.fillStyle = colors.muted;
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText(`+${{ax.toUpperCase()}}`, cx + r + 4, cy + 4);
  ctx.fillText(`+${{ay.toUpperCase()}}`, cx + 4, cy - r - 4);
  for (const item of items) {{
    const v = item.gravity_tool;
    const px = cx + v[axisIndex(ax)] * r;
    const py = cy - v[axisIndex(ay)] * r;
    const color = item.kind === "ideal" ? colors.ideal : colors.recorded;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px, py, item.kind === "ideal" ? 5 : 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "700 12px system-ui, sans-serif";
    ctx.fillText(item.label, px + 7, py - 5);
  }}
  ctx.restore();
}}

function axisIndex(axis) {{
  return axis === "x" ? 0 : axis === "y" ? 1 : 2;
}}

function drawMatrix(items, x, y, w, h, title) {{
  drawPanel(x, y, w, h, title);
  if (!controls.showMatrix.checked) return;
  const n = items.length;
  if (!n) return;
  const marginLeft = x + 60;
  const top = y + 54;
  const cell = Math.min(52, (w - 86) / n, (h - 80) / n);
  ctx.save();
  ctx.font = "700 12px system-ui, sans-serif";
  for (let i = 0; i < n; i++) {{
    ctx.fillStyle = items[i].kind === "ideal" ? colors.ideal : colors.recorded;
    ctx.fillText(items[i].label, marginLeft + i * cell + 12, y + 44);
    ctx.fillText(items[i].label, x + 18, top + i * cell + 31);
  }}
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  for (let r = 0; r < n; r++) {{
    for (let c = 0; c < n; c++) {{
      const left = marginLeft + c * cell;
      const cellY = top + r * cell;
      const angle = r === c ? 0 : angleDeg(items[r].gravity_tool, items[c].gravity_tool);
      ctx.fillStyle = r === c ? "#eef2f7" : `${{scoreColor(angle)}}22`;
      ctx.strokeStyle = colors.border;
      ctx.fillRect(left, cellY, cell, cell);
      ctx.strokeRect(left, cellY, cell, cell);
      ctx.fillStyle = r === c ? colors.muted : scoreColor(angle);
      ctx.fillText(r === c ? "-" : Math.round(angle).toString(), left + 14, cellY + 31);
    }}
  }}
  ctx.fillStyle = colors.muted;
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText("degrees between g_tool vectors; avoid very small off-diagonal angles", x + 16, y + h - 16);
  ctx.restore();
}}

function drawVectorList(items, x, y, w, h, title) {{
  drawPanel(x, y, w, h, title);
  ctx.save();
  ctx.font = "13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  let yy = y + 54;
  for (const item of items) {{
    ctx.fillStyle = item.kind === "ideal" ? colors.ideal : colors.recorded;
    ctx.fillText(`${{item.label}}  [${{item.gravity_tool.map(v => v.toFixed(3)).join(", ")}}]`, x + 18, yy);
    yy += 24;
  }}
  ctx.fillStyle = colors.muted;
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText(`pose_json: ${{initialData.pose_json || "(none)"}}`, x + 18, y + h - 36);
  ctx.fillText(`RPY order: ${{initialData.rpy_order}}`, x + 18, y + h - 18);
  ctx.restore();
}}

function draw() {{
  state.preTilt = Number(controls.preTilt.value);
  state.yaw = Number(controls.yaw.value);
  state.pitch = Number(controls.camPitch.value);
  state.zoom = Number(controls.zoom.value);
  labels.preTilt.textContent = state.preTilt;
  labels.yaw.textContent = state.yaw;
  labels.pitch.textContent = state.pitch;
  labels.zoom.textContent = state.zoom;

  const data = currentData();
  const visible = [];
  if (controls.showIdeal.checked) visible.push(...data.ideal);
  if (controls.showRecorded.checked) visible.push(...data.recorded);

  const idealMin = minAngle(data.ideal);
  const recordedMin = minAngle(data.recorded);
  labels.idealMin.textContent = `${{idealMin.toFixed(1)}} deg`;
  labels.idealMin.style.color = scoreColor(idealMin);
  labels.recordedMin.textContent = data.recorded.length ? `${{recordedMin.toFixed(1)}} deg` : "n/a";
  labels.recordedMin.style.color = data.recorded.length ? scoreColor(recordedMin) : colors.muted;
  labels.recordedCount.textContent = data.recorded.length.toString();

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawSphere(visible, 16, 16, 690, 528);
  drawProjection(visible, 724, 16, 278, 252, "XY projection", "x", "y");
  drawProjection(visible, 1020, 16, 278, 252, "XZ projection", "x", "z");
  drawProjection(visible, 724, 292, 278, 252, "YZ projection", "y", "z");
  drawMatrix(visible, 1020, 292, 278, 252, "angle matrix");
  drawVectorList(data.ideal, 16, 566, 405, 232, "ideal g_tool");
  drawVectorList(data.recorded, 441, 566, 405, 232, "recorded g_tool");
  drawMatrix(data.recorded.length ? data.recorded : data.ideal, 866, 566, 432, 232, "recorded-only matrix");
}}

Object.values(controls).forEach(control => control.addEventListener("input", draw));
canvas.addEventListener("pointerdown", event => {{
  state.dragging = true;
  state.lastX = event.clientX;
  state.lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
}});
canvas.addEventListener("pointermove", event => {{
  if (!state.dragging) return;
  const dx = event.clientX - state.lastX;
  const dy = event.clientY - state.lastY;
  state.lastX = event.clientX;
  state.lastY = event.clientY;
  controls.yaw.value = Math.max(-180, Math.min(180, Number(controls.yaw.value) + dx * 0.42));
  controls.camPitch.value = Math.max(-80, Math.min(80, Number(controls.camPitch.value) - dy * 0.34));
  draw();
}});
canvas.addEventListener("pointerup", event => {{
  state.dragging = false;
  canvas.releasePointerCapture(event.pointerId);
}});
canvas.addEventListener("wheel", event => {{
  event.preventDefault();
  controls.zoom.value = Math.max(70, Math.min(135, Number(controls.zoom.value) - event.deltaY * 0.035));
  draw();
}}, {{passive: false}});
if (!CanvasRenderingContext2D.prototype.roundRect) {{
  CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {{
    this.moveTo(x + r, y);
    this.arcTo(x + w, y, x + w, y + h, r);
    this.arcTo(x + w, y + h, x, y + h, r);
    this.arcTo(x, y + h, x, y, r);
    this.arcTo(x, y, x + w, y, r);
    return this;
  }};
}}
draw();
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an offline interactive HTML dashboard for RM65 4-point "
            "force-sensor calibration gravity coverage."
        )
    )
    parser.add_argument(
        "--output-html",
        default=str(RESULTS_DIR / "rm65_ideal_force_sensor_calib4_tool_directions.html"),
    )
    parser.add_argument(
        "--pose-json",
        default=str(POSE_DIR / "rm65_left_force_sensor_calib4_pose_sequence.json"),
        help="Optional recorded pose sequence to overlay as R1..R4.",
    )
    parser.add_argument(
        "--rpy-order",
        choices=("zyx", "xyz", "xzy", "yxz", "yzx", "zxy"),
        default="zyx",
        help="Euler order for recorded RM65 pose [x,y,z,rx,ry,rz] visualization.",
    )
    parser.add_argument(
        "--pre-tilt-deg",
        type=float,
        default=-60.0,
        help="Conceptual upstream arm tilt before wrist pitch/roll.",
    )
    parser.add_argument(
        "--pitch-deg",
        default="60,-30,-30,-120",
        help="Four conceptual wrist-pitch angles in degrees.",
    )
    parser.add_argument(
        "--roll-deg",
        default="0,0,180,0",
        help="Four conceptual wrist-roll angles in degrees.",
    )
    parser.add_argument("--labels", default="P1,P2,P3,P4")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pitch_deg = _parse_float_list(args.pitch_deg, expected_len=4, label="--pitch-deg")
        roll_deg = _parse_float_list(args.roll_deg, expected_len=4, label="--roll-deg")
        labels = _parse_label_list(args.labels, expected_len=4)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    ideal = [
        _ideal_orientation(
            label,
            pre_tilt_deg=args.pre_tilt_deg,
            pitch_deg=pitch,
            roll_deg=roll,
        )
        for label, pitch, roll in zip(labels, pitch_deg, roll_deg)
    ]
    pose_path = Path(args.pose_json) if args.pose_json else Path()
    recorded = _load_recorded_orientations(pose_path, args.rpy_order) if args.pose_json else []

    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _build_html(
            ideal=ideal,
            recorded=recorded,
            pose_json=str(pose_path) if args.pose_json else "",
            rpy_order=args.rpy_order,
            pre_tilt_deg=args.pre_tilt_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        ),
        encoding="utf-8",
    )
    print(f"wrote {html.escape(str(output_path))}")
    print(f"ideal spread: {_spread_stats(ideal)}")
    print(f"recorded spread: {_spread_stats(recorded)}")
    for item in recorded:
        print(f"{item['label']}: g_tool={item['gravity_tool']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
