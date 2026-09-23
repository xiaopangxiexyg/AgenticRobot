#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import tempfile
import warnings
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Keep the documented direct-script invocation usable outside an installed package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.portable_vis.shared_episode_pose import _quat_xyzw_to_matrix

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import cgi


SIDES = ("left", "right")
SIDE_COLORS = {
    "left": "#2563eb",
    "right": "#dc2626",
}


@dataclass(frozen=True)
class PoseSample:
    xyz: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    timestamp: float | None
    frame_index: int | None
    episode_index: int | None
    source: str


@dataclass
class SideTrajectory:
    side: str
    samples: list[PoseSample]
    source: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize single-arm or bimanual end-effector trajectories in the robot base frame. "
            "Input can be a LeRobot-style session/dataset root, a data/ directory, one parquet file, "
            "or an EpisodeRecorder directory containing frames.jsonl."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "session",
        nargs="?",
        default="",
        type=str,
        help=(
            "Session/dataset directory, episode directory, frames.jsonl, or parquet file. "
            "When omitted, open a browser picker first."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output HTML path. Defaults to <input>/visualizations/eef_trajectory.html.",
    )
    parser.add_argument(
        "-e",
        "--episode",
        dest="episodes",
        nargs="+",
        default=None,
        help="Optional episode index filter, e.g. --episode 0 3 or --episode 0,3.",
    )
    parser.add_argument(
        "--pose-source",
        choices=("auto", "actual", "current", "observation"),
        default="auto",
        help=(
            "Pose source priority. auto prefers recorded actual wrist camera pose, then current pose, "
            "then observation.state."
        ),
    )
    parser.add_argument(
        "--max-points-per-side",
        type=int,
        default=20000,
        help="Uniformly downsample each side to at most this many points for the HTML. 0 keeps all points.",
    )
    parser.add_argument(
        "--orientation-samples",
        type=int,
        default=80,
        help="Maximum number of historical orientation triads to draw per side. 0 disables historical triads.",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=("current", "history", "off"),
        default="current",
        help="Initial orientation display mode. Default only shows the current-frame TCP axes.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.0,
        help="Orientation triad axis length in meters. <=0 chooses an automatic length from trajectory size.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional title shown in the generated HTML.",
    )
    parser.add_argument(
        "--no-open-hint",
        action="store_true",
        help="Do not print the browser-open hint after writing the HTML.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the generated HTML or picker page in the default browser.",
    )
    parser.add_argument(
        "--picker-host",
        type=str,
        default="127.0.0.1",
        help="Host for the no-argument browser picker server.",
    )
    parser.add_argument(
        "--picker-port",
        type=int,
        default=0,
        help="Port for the no-argument browser picker server. 0 selects a free local port.",
    )
    return parser.parse_args()


def _parse_episode_filter(values: Sequence[str] | None) -> set[int] | None:
    if not values:
        return None
    out: set[int] = set()
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            out.add(int(part))
    return out


def _episode_index_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        if part.startswith("episode_"):
            suffix = part.removeprefix("episode_")
            if suffix.isdigit():
                return int(suffix)
    return None


def _resolve_output_path(input_path: Path, output: str) -> Path:
    if output:
        path = Path(output).expanduser()
        if path.suffix.lower() != ".html":
            path = path / "eef_trajectory.html"
        return path
    if input_path.is_file():
        root = input_path.parent
    else:
        root = input_path
    return root / "visualizations" / "eef_trajectory.html"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


def _as_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _as_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _pose_sample_from_value(
    value: Any,
    *,
    timestamp: float | None,
    frame_index: int | None,
    episode_index: int | None,
    source: str,
) -> PoseSample | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.shape == (4, 4):
        if not np.all(np.isfinite(arr)):
            return None
        xyz = arr[:3, 3].astype(float)
        rotation = arr[:3, :3].astype(float)
    else:
        flat = arr.reshape(-1)
        if flat.shape != (7,) or not np.all(np.isfinite(flat)):
            return None
        xyz = flat[:3].astype(float)
        rotation = _quat_xyzw_to_matrix(flat[3:7])
        if rotation is None:
            return None
    return PoseSample(
        xyz=tuple(float(v) for v in xyz.tolist()),
        rotation=tuple(tuple(float(v) for v in row) for row in rotation.tolist()),
        timestamp=timestamp,
        frame_index=frame_index,
        episode_index=episode_index,
        source=source,
    )


def _pose_from_observation_state(
    state: Any,
    *,
    side: str,
    timestamp: float | None,
    frame_index: int | None,
    episode_index: int | None,
) -> PoseSample | None:
    try:
        vector = np.asarray(state, dtype=float).reshape(-1)
    except Exception:
        return None
    if vector.shape[0] < 32:
        return None
    pose = vector[6:13] if side == "left" else vector[25:32]
    return _pose_sample_from_value(
        pose,
        timestamp=timestamp,
        frame_index=frame_index,
        episode_index=episode_index,
        source="observation.state",
    )


def _candidate_columns(side: str, pose_source: str) -> list[str]:
    actual = [
        f"observation.wrist_camera.{side}.pose_base",
        f"observation.arm.{side}.actual_tcp_pose",
    ]
    current = [
        f"observation.arm.{side}.current_tcp_pose",
        f"observation.{side}_arm.pose",
    ]
    observation = ["observation.state"]
    if pose_source == "actual":
        return actual
    if pose_source == "current":
        return current
    if pose_source == "observation":
        return observation
    return actual + current + observation


def _find_parquet_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".parquet":
        return [root]
    candidates: list[Path] = []
    data_dir = root / "data"
    if data_dir.exists():
        candidates.extend(data_dir.rglob("*.parquet"))
    candidates.extend(path for path in root.rglob("*.parquet") if path not in candidates)
    return sorted(candidates)


def _find_frames_jsonl(root: Path) -> list[Path]:
    if root.is_file() and root.name == "frames.jsonl":
        return [root]
    if root.is_dir() and (root / "frames.jsonl").exists():
        return [root / "frames.jsonl"]
    if not root.is_dir():
        return []
    return sorted(root.rglob("frames.jsonl"))


def _read_parquet_trajectories(
    files: Sequence[Path],
    *,
    pose_source: str,
    episode_filter: set[int] | None,
) -> dict[str, SideTrajectory]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading LeRobot parquet requires pyarrow. Install pyarrow or pass an EpisodeRecorder frames.jsonl."
        ) from exc

    trajectories = {
        side: SideTrajectory(side=side, samples=[], source="")
        for side in SIDES
    }
    for path in files:
        schema = pq.read_schema(path)
        names = set(schema.names)
        columns = {
            "timestamp",
            "frame_index",
            "episode_index",
            "observation.state",
        }
        for side in SIDES:
            columns.update(_candidate_columns(side, pose_source))
        selected = [name for name in columns if name in names]
        if not selected:
            continue
        table = pq.read_table(path, columns=selected)
        for row in table.to_pylist():
            row_episode = _as_int_or_none(row.get("episode_index"))
            if row_episode is None:
                row_episode = _episode_index_from_path(path)
            if episode_filter is not None and row_episode not in episode_filter:
                continue
            timestamp = _as_finite_float(row.get("timestamp"))
            frame_index = _as_int_or_none(row.get("frame_index"))
            for side in SIDES:
                sample = None
                source = ""
                for column in _candidate_columns(side, pose_source):
                    if column == "observation.state":
                        sample = _pose_from_observation_state(
                            row.get(column),
                            side=side,
                            timestamp=timestamp,
                            frame_index=frame_index,
                            episode_index=row_episode,
                        )
                    else:
                        sample = _pose_sample_from_value(
                            row.get(column),
                            timestamp=timestamp,
                            frame_index=frame_index,
                            episode_index=row_episode,
                            source=column,
                        )
                    if sample is not None:
                        source = sample.source
                        break
                if sample is not None:
                    traj = trajectories[side]
                    if not traj.source:
                        traj.source = source
                    traj.samples.append(sample)
    return {
        side: traj
        for side, traj in trajectories.items()
        if traj.samples
    }


def _read_frames_trajectories(
    files: Sequence[Path],
    *,
    episode_filter: set[int] | None,
) -> dict[str, SideTrajectory]:
    trajectories = {
        side: SideTrajectory(side=side, samples=[], source="frames.jsonl:arms.*.tcp_pose_base")
        for side in SIDES
    }
    for path in files:
        path_episode = _episode_index_from_path(path)
        if episode_filter is not None and path_episode not in episode_filter:
            continue
        for row in _iter_jsonl(path):
            timestamp = _as_finite_float(row.get("timestamp"))
            frame_index = _as_int_or_none(row.get("step_index"))
            row_episode = path_episode
            arms = row.get("arms")
            if not isinstance(arms, Mapping):
                continue
            for side in SIDES:
                payload = arms.get(side)
                if not isinstance(payload, Mapping):
                    continue
                sample = _pose_sample_from_value(
                    payload.get("tcp_pose_base"),
                    timestamp=timestamp,
                    frame_index=frame_index,
                    episode_index=row_episode,
                    source=f"frames.jsonl:arms.{side}.tcp_pose_base",
                )
                if sample is not None:
                    trajectories[side].samples.append(sample)
    return {
        side: traj
        for side, traj in trajectories.items()
        if traj.samples
    }


def _merge_trajectories(*groups: Mapping[str, SideTrajectory]) -> dict[str, SideTrajectory]:
    merged: dict[str, SideTrajectory] = {}
    for group in groups:
        for side, traj in group.items():
            if side not in merged:
                merged[side] = SideTrajectory(side=side, samples=[], source=traj.source)
            merged[side].samples.extend(traj.samples)
            if traj.source and traj.source not in merged[side].source:
                merged[side].source = (
                    traj.source if not merged[side].source else f"{merged[side].source}, {traj.source}"
                )
    for traj in merged.values():
        traj.samples.sort(
            key=lambda sample: (
                -1 if sample.episode_index is None else sample.episode_index,
                -1 if sample.frame_index is None else sample.frame_index,
                -math.inf if sample.timestamp is None else sample.timestamp,
            )
        )
    return merged


def _uniform_downsample(samples: Sequence[PoseSample], max_points: int) -> list[PoseSample]:
    if max_points <= 0 or len(samples) <= max_points:
        return list(samples)
    indices = np.linspace(0, len(samples) - 1, num=max_points, dtype=int)
    return [samples[int(index)] for index in indices.tolist()]


def _orientation_indices(count: int, limit: int) -> list[int]:
    if limit <= 0 or count <= 0:
        return []
    if count <= limit:
        return list(range(count))
    return sorted(set(int(i) for i in np.linspace(0, count - 1, num=limit, dtype=int).tolist()))


def _bounds_for_trajectories(trajectories: Mapping[str, SideTrajectory]) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float, float]] = []
    for traj in trajectories.values():
        points.extend(sample.xyz for sample in traj.samples)
    if not points:
        raise ValueError("No trajectory points available.")
    arr = np.asarray(points, dtype=float)
    return np.min(arr, axis=0), np.max(arr, axis=0)


def _auto_axis_length(
    trajectories: Mapping[str, SideTrajectory],
    requested: float,
) -> float:
    if requested > 0.0:
        return float(requested)
    mins, maxs = _bounds_for_trajectories(trajectories)
    diag = float(np.linalg.norm(maxs - mins))
    return max(0.025, min(0.12, diag * 0.08 if diag > 1e-9 else 0.05))


def _trajectory_payload(
    trajectories: Mapping[str, SideTrajectory],
    *,
    max_points_per_side: int,
    orientation_samples: int,
    axis_length: float,
    orientation_mode: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sides": {},
        "bounds": {},
        "axis_length": float(axis_length),
        "orientation_mode": str(orientation_mode),
    }
    downsampled: dict[str, SideTrajectory] = {}
    for side, traj in trajectories.items():
        samples = _uniform_downsample(traj.samples, max_points_per_side)
        downsampled[side] = SideTrajectory(side=side, samples=samples, source=traj.source)
    mins, maxs = _bounds_for_trajectories(downsampled)
    payload["bounds"] = {
        "min": [float(v) for v in mins.tolist()],
        "max": [float(v) for v in maxs.tolist()],
    }
    for side, traj in downsampled.items():
        triads: list[dict[str, Any]] = []
        for index in _orientation_indices(len(traj.samples), orientation_samples):
            sample = traj.samples[index]
            origin = np.asarray(sample.xyz, dtype=float)
            rotation = np.asarray(sample.rotation, dtype=float)
            triads.append(
                {
                    "index": int(index),
                    "origin": [float(v) for v in origin.tolist()],
                    "x": [float(v) for v in (origin + rotation[:, 0] * axis_length).tolist()],
                    "y": [float(v) for v in (origin + rotation[:, 1] * axis_length).tolist()],
                    "z": [float(v) for v in (origin + rotation[:, 2] * axis_length).tolist()],
                }
            )
        payload["sides"][side] = {
            "color": SIDE_COLORS.get(side, "#111827"),
            "source": traj.source,
            "original_count": len(trajectories[side].samples),
            "count": len(traj.samples),
            "points": [list(sample.xyz) for sample in traj.samples],
            "rotations": [
                [float(value) for row in sample.rotation for value in row]
                for sample in traj.samples
            ],
            "frames": [sample.frame_index for sample in traj.samples],
            "episodes": [sample.episode_index for sample in traj.samples],
            "timestamps": [sample.timestamp for sample in traj.samples],
            "triads": triads,
        }
    return payload


def _load_trajectories(
    input_path: Path,
    *,
    pose_source: str,
    episode_filter: set[int] | None,
) -> tuple[dict[str, SideTrajectory], list[Path], list[Path]]:
    parquet_files = _find_parquet_files(input_path)
    frame_files = _find_frames_jsonl(input_path)

    parquet_trajectories: dict[str, SideTrajectory] = {}
    if parquet_files:
        parquet_trajectories = _read_parquet_trajectories(
            parquet_files,
            pose_source=pose_source,
            episode_filter=episode_filter,
        )
    frame_trajectories: dict[str, SideTrajectory] = {}
    if not parquet_trajectories and frame_files:
        frame_trajectories = _read_frames_trajectories(
            frame_files,
            episode_filter=episode_filter,
        )
    trajectories = _merge_trajectories(parquet_trajectories, frame_trajectories)
    return trajectories, parquet_files, frame_files


def _html_document(title: str, data: Mapping[str, Any]) -> str:
    json_payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
  :root {{
    color-scheme: light;
    --bg: #f8fafc;
    --panel: #ffffff;
    --text: #111827;
    --muted: #6b7280;
    --line: #d1d5db;
  }}
  html, body {{
    margin: 0;
    height: 100%;
    overflow: hidden;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #app {{
    display: grid;
    grid-template-rows: auto 1fr auto;
    height: 100%;
  }}
  header {{
    display: flex;
    gap: 16px;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-bottom: 1px solid var(--line);
    background: var(--panel);
  }}
  h1 {{
    margin: 0;
    font-size: 16px;
    font-weight: 700;
  }}
  .meta {{
    margin-top: 3px;
    font-size: 12px;
    color: var(--muted);
  }}
  .controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    justify-content: flex-end;
    font-size: 13px;
  }}
  label {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
  }}
  button {{
    border: 1px solid var(--line);
    background: #fff;
    color: var(--text);
    padding: 5px 9px;
    border-radius: 6px;
    cursor: pointer;
  }}
  button:hover {{ background: #f3f4f6; }}
  input[type="range"] {{ width: min(42vw, 520px); }}
  main {{ position: relative; min-height: 0; }}
  canvas {{
    width: 100%;
    height: 100%;
    display: block;
    cursor: grab;
    background: linear-gradient(#ffffff, #eef2f7);
  }}
  canvas.dragging {{ cursor: grabbing; }}
  .legend {{
    position: absolute;
    left: 12px;
    bottom: 12px;
    padding: 9px 10px;
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(209,213,219,0.9);
    border-radius: 8px;
    font-size: 12px;
    line-height: 1.45;
    max-width: min(560px, calc(100vw - 24px));
  }}
  .legend-row {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .swatch {{
    width: 22px;
    height: 3px;
    border-radius: 999px;
    display: inline-block;
  }}
  footer {{
    padding: 7px 12px;
    color: var(--muted);
    background: var(--panel);
    border-top: 1px solid var(--line);
    font-size: 12px;
  }}
</style>
</head>
<body>
<div id="app">
  <header>
    <div>
      <h1>{escaped_title}</h1>
      <div class="meta" id="meta"></div>
    </div>
    <div class="controls">
      <label><input id="toggle-left" type="checkbox" checked>left</label>
      <label><input id="toggle-right" type="checkbox" checked>right</label>
      <label><input id="toggle-triads" type="checkbox" checked>orientation</label>
      <label><input id="orientation-current-only" type="checkbox" checked>current orientation only</label>
      <button id="reset">Reset view</button>
      <button id="play">Play</button>
      <input id="frame" type="range" min="0" max="0" value="0">
      <span id="frame-label"></span>
    </div>
  </header>
  <main>
    <canvas id="view"></canvas>
    <div class="legend" id="legend"></div>
  </main>
  <footer>Drag to rotate, mouse wheel to zoom, Shift+drag to pan. Coordinates are in robot base frame; red/green/blue triads are local TCP x/y/z axes.</footer>
</div>
<script id="trajectory-data" type="application/json">{json_payload}</script>
<script>
const DATA = JSON.parse(document.getElementById('trajectory-data').textContent);
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const state = {{
  yaw: -0.75,
  pitch: 0.62,
  zoom: 1.0,
  panX: 0,
  panY: 0,
  dragging: false,
  lastX: 0,
  lastY: 0,
  shift: false,
  playing: false,
  timer: null,
}};
const toggles = {{
  left: document.getElementById('toggle-left'),
  right: document.getElementById('toggle-right'),
  triads: document.getElementById('toggle-triads'),
  currentOnly: document.getElementById('orientation-current-only'),
}};
const frameSlider = document.getElementById('frame');
const frameLabel = document.getElementById('frame-label');
const playButton = document.getElementById('play');
const initialOrientationMode = DATA.orientation_mode || 'current';
toggles.triads.checked = initialOrientationMode !== 'off';
toggles.currentOnly.checked = initialOrientationMode !== 'history';

function vecSub(a, b) {{ return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }}
function vecAdd(a, b) {{ return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }}
function vecScale(a, s) {{ return [a[0]*s, a[1]*s, a[2]*s]; }}

const bmin = DATA.bounds.min;
const bmax = DATA.bounds.max;
const center = vecScale(vecAdd(bmin, bmax), 0.5);
const extent = Math.max(0.05, Math.hypot(bmax[0]-bmin[0], bmax[1]-bmin[1], bmax[2]-bmin[2]));
let baseScale = 1;
let maxCount = 0;
for (const side of Object.keys(DATA.sides)) {{
  maxCount = Math.max(maxCount, DATA.sides[side].points.length);
}}
frameSlider.max = Math.max(0, maxCount - 1);

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  baseScale = Math.min(rect.width, rect.height) * 0.68 / extent;
  draw();
}}

function project(point) {{
  const p = vecSub(point, center);
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  const x1 = cy * p[0] - sy * p[1];
  const y1 = sy * p[0] + cy * p[1];
  const z1 = p[2];
  const y2 = cp * y1 - sp * z1;
  const z2 = sp * y1 + cp * z1;
  const rect = canvas.getBoundingClientRect();
  return {{
    x: rect.width / 2 + state.panX + x1 * baseScale * state.zoom,
    y: rect.height / 2 + state.panY - y2 * baseScale * state.zoom,
    z: z2,
  }};
}}

function line3(a, b, color, width=1, alpha=1) {{
  const pa = project(a), pb = project(b);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.stroke();
  ctx.restore();
}}

function drawGrid() {{
  const size = Math.max(extent * 0.65, 0.2);
  const step = size / 5;
  ctx.save();
  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth = 1;
  for (let i = -5; i <= 5; i++) {{
    line3([center[0] + i * step, center[1] - size, 0], [center[0] + i * step, center[1] + size, 0], '#e5e7eb', 1, 0.75);
    line3([center[0] - size, center[1] + i * step, 0], [center[0] + size, center[1] + i * step, 0], '#e5e7eb', 1, 0.75);
  }}
  const axis = Math.max(0.12, extent * 0.28);
  line3([0,0,0], [axis,0,0], '#ef4444', 3, 1);
  line3([0,0,0], [0,axis,0], '#22c55e', 3, 1);
  line3([0,0,0], [0,0,axis], '#3b82f6', 3, 1);
  ctx.restore();
}}

function drawTrajectory(side, traj, frameLimit) {{
  const pts = traj.points;
  if (!pts.length) return;
  const color = traj.color;
  const visibleCount = Math.min(pts.length, frameLimit + 1);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.25;
  ctx.globalAlpha = 0.88;
  ctx.beginPath();
  for (let i = 0; i < visibleCount; i++) {{
    const p = project(pts[i]);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }}
  ctx.stroke();
  ctx.globalAlpha = 0.25;
  if (visibleCount < pts.length) {{
    ctx.beginPath();
    for (let i = visibleCount - 1; i < pts.length; i++) {{
      const p = project(pts[Math.max(0, i)]);
      if (i === visibleCount - 1) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }}
    ctx.stroke();
  }}
  const start = project(pts[0]);
  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(start.x, start.y, 4, 0, Math.PI * 2);
  ctx.fill();
  const currentIndex = Math.min(pts.length - 1, frameLimit);
  const current = project(pts[currentIndex]);
  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(current.x, current.y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}}

function drawTriads(traj, frameLimit) {{
  if (!toggles.triads.checked) return;
  if (toggles.currentOnly.checked) {{
    const index = Math.max(0, Math.min(traj.points.length - 1, frameLimit));
    drawCurrentTriad(traj, index);
    return;
  }}
  for (const triad of traj.triads) {{
    if (triad.index > frameLimit) continue;
    line3(triad.origin, triad.x, '#ef4444', 1.6, 0.75);
    line3(triad.origin, triad.y, '#22c55e', 1.6, 0.75);
    line3(triad.origin, triad.z, '#3b82f6', 1.6, 0.75);
  }}
}}

function drawCurrentTriad(traj, index) {{
  const origin = traj.points[index];
  const rotation = traj.rotations[index];
  if (!origin || !rotation || rotation.length < 9) return;
  const axis = DATA.axis_length || Math.max(0.025, extent * 0.08);
  const x = [
    origin[0] + rotation[0] * axis,
    origin[1] + rotation[3] * axis,
    origin[2] + rotation[6] * axis,
  ];
  const y = [
    origin[0] + rotation[1] * axis,
    origin[1] + rotation[4] * axis,
    origin[2] + rotation[7] * axis,
  ];
  const z = [
    origin[0] + rotation[2] * axis,
    origin[1] + rotation[5] * axis,
    origin[2] + rotation[8] * axis,
  ];
  line3(origin, x, '#ef4444', 2.2, 0.95);
  line3(origin, y, '#22c55e', 2.2, 0.95);
  line3(origin, z, '#3b82f6', 2.2, 0.95);
}}

function updateLabels() {{
  const frame = Number(frameSlider.value);
  frameLabel.textContent = `${{frame + 1}} / ${{Math.max(1, maxCount)}}`;
  const parts = [];
  for (const side of Object.keys(DATA.sides)) {{
    const traj = DATA.sides[side];
    parts.push(`${{side}}: ${{traj.count}} pts${{traj.original_count !== traj.count ? ` (from ${{traj.original_count}})` : ''}}`);
  }}
  document.getElementById('meta').textContent = parts.join(' | ');
}}

function drawLegend() {{
  const legend = document.getElementById('legend');
  const rows = [];
  for (const side of Object.keys(DATA.sides)) {{
    const traj = DATA.sides[side];
    rows.push(`<div class="legend-row"><span class="swatch" style="background:${{traj.color}}"></span><b>${{side}}</b><span>${{traj.source}}</span></div>`);
  }}
  rows.push('<div style="margin-top:6px;color:#6b7280">Base axes: X red, Y green, Z blue. Start is filled; current frame is ring. Orientation defaults to current frame only.</div>');
  legend.innerHTML = rows.join('');
}}

function draw() {{
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  drawGrid();
  const frameLimit = Number(frameSlider.value);
  for (const side of Object.keys(DATA.sides)) {{
    if (!toggles[side]?.checked) continue;
    drawTriads(DATA.sides[side], frameLimit);
  }}
  for (const side of Object.keys(DATA.sides)) {{
    if (!toggles[side]?.checked) continue;
    drawTrajectory(side, DATA.sides[side], frameLimit);
  }}
  updateLabels();
}}

function stopPlay() {{
  state.playing = false;
  playButton.textContent = 'Play';
  if (state.timer !== null) {{
    clearInterval(state.timer);
    state.timer = null;
  }}
}}

canvas.addEventListener('mousedown', (event) => {{
  state.dragging = true;
  state.shift = event.shiftKey;
  state.lastX = event.clientX;
  state.lastY = event.clientY;
  canvas.classList.add('dragging');
}});
window.addEventListener('mouseup', () => {{
  state.dragging = false;
  canvas.classList.remove('dragging');
}});
window.addEventListener('mousemove', (event) => {{
  if (!state.dragging) return;
  const dx = event.clientX - state.lastX;
  const dy = event.clientY - state.lastY;
  state.lastX = event.clientX;
  state.lastY = event.clientY;
  if (state.shift) {{
    state.panX += dx;
    state.panY += dy;
  }} else {{
    state.yaw += dx * 0.008;
    state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch + dy * 0.008));
  }}
  draw();
}});
canvas.addEventListener('wheel', (event) => {{
  event.preventDefault();
  const factor = Math.exp(-event.deltaY * 0.001);
  state.zoom = Math.max(0.08, Math.min(80, state.zoom * factor));
  draw();
}}, {{ passive: false }});

document.getElementById('reset').addEventListener('click', () => {{
  state.yaw = -0.75;
  state.pitch = 0.62;
  state.zoom = 1.0;
  state.panX = 0;
  state.panY = 0;
  draw();
}});
playButton.addEventListener('click', () => {{
  if (state.playing) {{
    stopPlay();
    return;
  }}
  state.playing = true;
  playButton.textContent = 'Pause';
  state.timer = setInterval(() => {{
    let value = Number(frameSlider.value);
    value = value >= Number(frameSlider.max) ? 0 : value + 1;
    frameSlider.value = String(value);
    draw();
  }}, 45);
}});
for (const input of [toggles.left, toggles.right, toggles.triads, toggles.currentOnly, frameSlider]) {{
  if (input) input.addEventListener('input', draw);
}}
window.addEventListener('resize', resize);
drawLegend();
resize();
</script>
</body>
</html>
"""


def _picker_document() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DexTeleop EEF Trajectory Picker</title>
<style>
  html, body {
    margin: 0;
    min-height: 100%;
    background: #f8fafc;
    color: #111827;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  main {
    max-width: 860px;
    margin: 0 auto;
    padding: 32px 18px 48px;
  }
  h1 { margin: 0 0 8px; font-size: 24px; }
  p { color: #4b5563; line-height: 1.55; }
  section {
    margin-top: 18px;
    padding: 16px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    background: #ffffff;
  }
  label {
    display: block;
    margin: 12px 0 6px;
    font-weight: 650;
  }
  input, select {
    box-sizing: border-box;
    width: 100%;
    max-width: 520px;
    padding: 8px;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    background: #fff;
  }
  .row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px 18px;
  }
  button {
    margin-top: 16px;
    padding: 9px 14px;
    border: 1px solid #1d4ed8;
    border-radius: 6px;
    background: #2563eb;
    color: white;
    font-weight: 700;
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.6;
    cursor: wait;
  }
  .hint {
    font-size: 13px;
    color: #64748b;
  }
  #status {
    margin-top: 12px;
    color: #334155;
    white-space: pre-wrap;
  }
</style>
</head>
<body>
<main>
  <h1>DexTeleop EEF Trajectory Picker</h1>
  <p>Select a recorded dataset/session directory, or choose one or more parquet / frames.jsonl files.
  Only <code>*.parquet</code> and <code>frames.jsonl</code> files are uploaded to this local Python process.</p>

  <section>
    <label for="directory">Dataset / session directory</label>
    <input id="directory" type="file" webkitdirectory directory multiple>
    <div class="hint">Works best in Chrome/Edge. Select the dataset root, data/ directory, trajectories/ directory, or one episode directory.</div>

    <label for="files">Or choose files directly</label>
    <input id="files" type="file" multiple accept=".parquet,.jsonl">
    <div class="hint">Direct files can be episode_*.parquet or frames.jsonl.</div>

    <div class="row">
      <div>
        <label for="pose_source">Pose source</label>
        <select id="pose_source">
          <option value="auto" selected>auto</option>
          <option value="actual">actual wrist camera pose</option>
          <option value="current">current TCP pose</option>
          <option value="observation">observation.state</option>
        </select>
      </div>
      <div>
        <label for="orientation_mode">Orientation</label>
        <select id="orientation_mode">
          <option value="current" selected>current frame only</option>
          <option value="history">history samples</option>
          <option value="off">off</option>
        </select>
      </div>
      <div>
        <label for="max_points">Max points per side</label>
        <input id="max_points" type="number" min="0" step="1000" value="20000">
      </div>
      <div>
        <label for="orientation_samples">History orientation samples</label>
        <input id="orientation_samples" type="number" min="0" step="10" value="80">
      </div>
    </div>
    <button id="render">Render trajectory</button>
    <div id="status"></div>
  </section>
</main>
<script>
const directoryInput = document.getElementById('directory');
const filesInput = document.getElementById('files');
const button = document.getElementById('render');
const statusBox = document.getElementById('status');

function relevant(file) {
  const name = (file.webkitRelativePath || file.name || '').toLowerCase();
  return name.endsWith('.parquet') || name.endsWith('/frames.jsonl') || name === 'frames.jsonl';
}

function selectedFiles() {
  const files = [...directoryInput.files, ...filesInput.files].filter(relevant);
  const unique = new Map();
  for (const file of files) {
    const key = file.webkitRelativePath || file.name;
    unique.set(key, file);
  }
  return [...unique.values()];
}

button.addEventListener('click', async () => {
  const files = selectedFiles();
  if (!files.length) {
    statusBox.textContent = 'No parquet or frames.jsonl files selected.';
    return;
  }
  const form = new FormData();
  for (const file of files) {
    form.append('files', file, file.webkitRelativePath || file.name);
  }
  form.append('pose_source', document.getElementById('pose_source').value);
  form.append('orientation_mode', document.getElementById('orientation_mode').value);
  form.append('max_points_per_side', document.getElementById('max_points').value);
  form.append('orientation_samples', document.getElementById('orientation_samples').value);
  button.disabled = true;
  statusBox.textContent = `Uploading ${files.length} trajectory file(s) to local parser...`;
  try {
    const response = await fetch('/render', { method: 'POST', body: form });
    const text = await response.text();
    if (!response.ok) {
      statusBox.textContent = text;
      return;
    }
    document.open();
    document.write(text);
    document.close();
  } catch (error) {
    statusBox.textContent = String(error);
  } finally {
    button.disabled = false;
  }
});
</script>
</body>
</html>
"""


def _error_document(message: str) -> str:
    escaped = html.escape(message)
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>EEF trajectory error</title></head>
<body style="font-family:system-ui,sans-serif;background:#f8fafc;color:#111827;padding:24px">
<h1>Could not render trajectory</h1>
<pre style="white-space:pre-wrap;background:#fff;border:1px solid #d1d5db;border-radius:8px;padding:12px">{escaped}</pre>
<button onclick="history.back()">Back</button>
</body>
</html>
"""


def _open_uri(uri: str) -> None:
    try:
        webbrowser.open(uri, new=2)
    except Exception as exc:
        print(f"WARNING: failed to open browser: {type(exc).__name__}: {exc}")


def _safe_upload_path(root: Path, filename: str, fallback: str) -> Path:
    raw_parts = Path(str(filename).replace("\\", "/")).parts
    parts = [
        re_part
        for part in raw_parts
        for re_part in [part.strip().replace("\x00", "")]
        if re_part and re_part not in {".", ".."} and not Path(re_part).is_absolute()
    ]
    if not parts:
        parts = [fallback]
    return root.joinpath(*parts)


def _field_text(form: cgi.FieldStorage, name: str, default: str) -> str:
    value = form.getfirst(name, default)
    if value is None:
        return default
    return str(value)


def _field_int(form: cgi.FieldStorage, name: str, default: int) -> int:
    try:
        return int(_field_text(form, name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _uploaded_file_items(form: cgi.FieldStorage) -> list[cgi.FieldStorage]:
    if "files" not in form:
        return []
    value = form["files"]
    items = value if isinstance(value, list) else [value]
    return [
        item
        for item in items
        if getattr(item, "filename", None) and getattr(item, "file", None) is not None
    ]


def _write_uploaded_files(form: cgi.FieldStorage, root: Path) -> tuple[list[Path], list[Path]]:
    parquet_files: list[Path] = []
    frame_files: list[Path] = []
    for index, item in enumerate(_uploaded_file_items(form)):
        filename = str(item.filename)
        lower = filename.lower()
        if not (lower.endswith(".parquet") or lower.endswith("/frames.jsonl") or lower.endswith("frames.jsonl")):
            continue
        path = _safe_upload_path(root, filename, f"upload_{index}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            while True:
                chunk = item.file.read(1024 * 1024)
                if not chunk:
                    break
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                handle.write(chunk)
        if path.suffix.lower() == ".parquet":
            parquet_files.append(path)
        elif path.name == "frames.jsonl":
            frame_files.append(path)
    return parquet_files, frame_files


def _render_trajectories_from_files(
    parquet_files: Sequence[Path],
    frame_files: Sequence[Path],
    *,
    pose_source: str,
    max_points_per_side: int,
    orientation_samples: int,
    orientation_mode: str,
    title: str,
) -> str:
    parquet_trajectories: dict[str, SideTrajectory] = {}
    if parquet_files:
        parquet_trajectories = _read_parquet_trajectories(
            parquet_files,
            pose_source=pose_source,
            episode_filter=None,
        )
    frame_trajectories: dict[str, SideTrajectory] = {}
    if not parquet_trajectories and frame_files:
        frame_trajectories = _read_frames_trajectories(
            frame_files,
            episode_filter=None,
        )
    trajectories = _merge_trajectories(parquet_trajectories, frame_trajectories)
    if not trajectories:
        raise RuntimeError("No valid base-frame end-effector poses found in the selected files.")
    axis_length = _auto_axis_length(trajectories, 0.0)
    payload = _trajectory_payload(
        trajectories,
        max_points_per_side=max_points_per_side,
        orientation_samples=orientation_samples,
        axis_length=axis_length,
        orientation_mode=orientation_mode,
    )
    payload["input"] = "browser-selected files"
    payload["pose_source"] = pose_source
    return _html_document(title, payload)


def _serve_picker(args: argparse.Namespace) -> int:
    class PickerHandler(BaseHTTPRequestHandler):
        def _send_html(self, status: int, text: str) -> None:
            encoded = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path not in {"/", "/index.html"}:
                self._send_html(404, _error_document(f"Unknown picker path: {self.path}"))
                return
            self._send_html(200, _picker_document())

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/render":
                self._send_html(404, _error_document(f"Unknown picker path: {self.path}"))
                return
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                        "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                    },
                )
                with tempfile.TemporaryDirectory(prefix="dexteleop_eef_vis_") as temp_dir:
                    parquet_files, frame_files = _write_uploaded_files(form, Path(temp_dir))
                    if not parquet_files and not frame_files:
                        raise RuntimeError("No parquet or frames.jsonl files were uploaded.")
                    document = _render_trajectories_from_files(
                        parquet_files,
                        frame_files,
                        pose_source=_field_text(form, "pose_source", "auto"),
                        max_points_per_side=_field_int(form, "max_points_per_side", 20000),
                        orientation_samples=_field_int(form, "orientation_samples", 80),
                        orientation_mode=_field_text(form, "orientation_mode", "current"),
                        title="EEF trajectory - browser selection",
                    )
                self._send_html(200, document)
            except Exception as exc:  # noqa: BLE001 - return the error in the browser picker.
                self._send_html(500, _error_document(f"{type(exc).__name__}: {exc}"))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(
        (str(args.picker_host), int(args.picker_port)),
        PickerHandler,
    )
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(f"Picker running at {url}")
    if not bool(args.no_open):
        _open_uri(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPicker stopped.")
    finally:
        server.server_close()
    return 0


def main() -> int:
    args = _parse_args()
    if not str(args.session).strip():
        return _serve_picker(args)
    input_path = Path(args.session).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    episode_filter = _parse_episode_filter(args.episodes)
    trajectories, parquet_files, frame_files = _load_trajectories(
        input_path,
        pose_source=str(args.pose_source),
        episode_filter=episode_filter,
    )
    if not trajectories:
        searched = [*(str(path) for path in parquet_files), *(str(path) for path in frame_files)]
        hint = "\n".join(searched[:20]) if searched else "no parquet or frames.jsonl files found"
        raise RuntimeError(
            "No valid end-effector base-frame poses found. Checked:\n"
            f"{hint}\n"
            "Expected columns include observation.wrist_camera.{side}.pose_base, "
            "observation.arm.{side}.actual_tcp_pose/current_tcp_pose, observation.state, "
            "or EpisodeRecorder arms.{side}.tcp_pose_base."
        )
    axis_length = _auto_axis_length(trajectories, float(args.axis_length))
    payload = _trajectory_payload(
        trajectories,
        max_points_per_side=int(args.max_points_per_side),
        orientation_samples=int(args.orientation_samples),
        axis_length=axis_length,
        orientation_mode=str(args.orientation_mode),
    )
    payload["input"] = str(input_path)
    payload["pose_source"] = str(args.pose_source)
    payload["axis_length"] = axis_length
    title = args.title or f"EEF trajectory - {input_path.name}"
    output_path = _resolve_output_path(input_path, str(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html_document(title, payload), encoding="utf-8")

    sides_text = ", ".join(
        f"{side}:{len(traj.samples)}"
        for side, traj in trajectories.items()
    )
    print(f"Wrote {output_path}")
    print(f"Loaded trajectories: {sides_text}")
    if not bool(args.no_open):
        _open_uri(output_path.resolve().as_uri())
    if not bool(args.no_open_hint):
        print(f"Open in a browser: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
