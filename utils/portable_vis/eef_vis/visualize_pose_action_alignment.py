#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import math
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SIDES = ("left", "right")
POSE_DIM = 7
ACTION_DIM = 26
STATE_DIM = 38
ACTION_ARM_POSE_SLICES = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}
STATE_ARM_POSE_SLICES = {
    "left": slice(6, 13),
    "right": slice(25, 32),
}
SIDE_STYLES = {
    "left": {
        "state": "#2563eb",
        "action": "#0f766e",
        "next": "#7c3aed",
    },
    "right": {
        "state": "#dc2626",
        "action": "#ea580c",
        "next": "#0891b2",
    },
}


@dataclass(frozen=True)
class EpisodeSource:
    dataset_root: Path
    parquet_path: Path
    episode_index: int | None
    episode_id: str


@dataclass(frozen=True)
class PoseValue:
    xyz: tuple[float, float, float]
    rotation: tuple[float, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize action pose alignment against same-frame state and next-frame state "
            "for one DexTeleop dataset episode."
        )
    )
    parser.add_argument(
        "dataset_or_parquet",
        type=str,
        help="Dataset root, data directory, or one episode_*.parquet file.",
    )
    parser.add_argument(
        "-e",
        "--episode",
        type=str,
        default="",
        help="Episode index or id, e.g. 10 or episode_000010. Defaults to the first episode.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output HTML path. Defaults to <dataset>/visualizations/pose_action_alignment_<episode>.html.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5000,
        help="Uniformly downsample displayed frames to at most this many. 0 keeps all frames.",
    )
    parser.add_argument(
        "--residual-stride",
        type=int,
        default=8,
        help="Draw one historical residual segment every N displayed frames. 0 disables history residuals.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.0,
        help="Current-pose orientation triad axis length in meters. <=0 uses an automatic value.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional HTML title.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the generated HTML in a browser.",
    )
    return parser.parse_args()


def _require_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading episode parquet requires pyarrow. Activate the DexTeleop "
            "environment with `conda activate dexteleop310`, then run "
            "`python utils/portable_vis/eef_vis/visualize_pose_action_alignment.py ...`."
        ) from exc
    return pq


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _episode_index_from_path(path: Path) -> int | None:
    stem = path.stem
    prefix = "episode_"
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _episode_id_from_index(index: int | None, fallback: str) -> str:
    if index is None:
        return fallback
    return f"episode_{int(index):06d}"


def _episode_matches(record: Mapping[str, Any], token: str, path: Path) -> bool:
    normalized = str(token).strip()
    if not normalized:
        return False
    episode_id = str(record.get("episode_id") or path.stem)
    if normalized == episode_id or normalized == path.stem:
        return True
    try:
        requested_index = int(normalized)
    except ValueError:
        return False
    record_index = record.get("episode_index")
    if record_index is None:
        record_index = _episode_index_from_path(path)
    try:
        return int(record_index) == requested_index
    except (TypeError, ValueError):
        return False


def _find_dataset_root(path: Path) -> Path:
    if path.is_file():
        current = path.parent
    else:
        current = path
    for candidate in (current, *current.parents):
        if (candidate / "meta" / "episodes.jsonl").exists() or (candidate / "data").exists():
            return candidate
    return current


def _parquet_candidates(root: Path) -> list[tuple[dict[str, Any], Path]]:
    dataset_root = _find_dataset_root(root)
    records = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    candidates: list[tuple[dict[str, Any], Path]] = []
    for record in records:
        rel = record.get("parquet_path")
        if not isinstance(rel, str) or not rel:
            continue
        path = dataset_root / rel
        if path.exists():
            candidates.append((record, path))
    if candidates:
        return candidates
    search_root = root if root.is_dir() else root.parent
    return [
        (
            {
                "episode_index": _episode_index_from_path(path),
                "episode_id": path.stem,
                "parquet_path": str(path.relative_to(dataset_root))
                if path.is_relative_to(dataset_root)
                else str(path),
            },
            path,
        )
        for path in sorted(search_root.rglob("episode_*.parquet"))
    ]


def _resolve_episode_source(path_text: str, episode_token: str) -> EpisodeSource:
    input_path = Path(path_text).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    dataset_root = _find_dataset_root(input_path).resolve()
    if input_path.is_file() and input_path.suffix.lower() == ".parquet":
        episode_index = _episode_index_from_path(input_path)
        return EpisodeSource(
            dataset_root=dataset_root,
            parquet_path=input_path,
            episode_index=episode_index,
            episode_id=_episode_id_from_index(episode_index, input_path.stem),
        )

    candidates = _parquet_candidates(input_path)
    if not candidates:
        raise FileNotFoundError(f"No episode parquet files found under {input_path}")
    chosen: tuple[dict[str, Any], Path]
    token = str(episode_token).strip()
    if token:
        matches = [(record, path) for record, path in candidates if _episode_matches(record, token, path)]
        if not matches:
            available = ", ".join(
                str(record.get("episode_index", path.stem)) for record, path in candidates[:20]
            )
            raise FileNotFoundError(f"Episode {token!r} not found. First available episodes: {available}")
        chosen = matches[0]
    else:
        chosen = sorted(
            candidates,
            key=lambda item: (
                10**12
                if item[0].get("episode_index") is None
                else int(item[0].get("episode_index")),
                str(item[1]),
            ),
        )[0]
    record, parquet_path = chosen
    episode_index = record.get("episode_index")
    try:
        episode_index = int(episode_index) if episode_index is not None else _episode_index_from_path(parquet_path)
    except (TypeError, ValueError):
        episode_index = _episode_index_from_path(parquet_path)
    return EpisodeSource(
        dataset_root=dataset_root,
        parquet_path=parquet_path.resolve(),
        episode_index=episode_index,
        episode_id=str(record.get("episode_id") or _episode_id_from_index(episode_index, parquet_path.stem)),
    )


def _selected_columns(names: Iterable[str]) -> list[str]:
    available = set(names)
    requested = {
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "action",
        "observation.state",
        "action.left_arm.pose",
        "action.right_arm.pose",
        "observation.left_arm.pose",
        "observation.right_arm.pose",
    }
    selected = sorted(requested & available)
    missing = {"action", "observation.state"} - set(selected)
    if missing:
        raise ValueError(f"Episode parquet is missing required column(s): {sorted(missing)}")
    return selected


def _read_episode_rows(parquet_path: Path) -> list[dict[str, Any]]:
    pq = _require_pyarrow_parquet()
    parquet = pq.ParquetFile(parquet_path)
    rows = pq.read_table(parquet_path, columns=_selected_columns(parquet.schema_arrow.names)).to_pylist()
    rows.sort(key=lambda row: int(row.get("frame_index", row.get("index", 0)) or 0))
    return rows


def _vector_or_none(value: Any, *, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.shape != (int(length),) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _quat_xyzw_to_matrix(quat: Sequence[float]) -> np.ndarray | None:
    q = np.asarray(quat, dtype=float).reshape(-1)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return None
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return None
    x, y, z, w = (q / norm).tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _pose_value(vector: np.ndarray | None) -> PoseValue | None:
    if vector is None:
        return None
    pose = np.asarray(vector, dtype=float).reshape(-1)
    if pose.shape != (POSE_DIM,) or not np.all(np.isfinite(pose)):
        return None
    rotation = _quat_xyzw_to_matrix(pose[3:7])
    if rotation is None:
        return None
    return PoseValue(
        xyz=tuple(float(value) for value in pose[:3].tolist()),
        rotation=tuple(float(value) for value in rotation.reshape(-1).tolist()),
    )


def _action_pose(row: Mapping[str, Any], side: str) -> PoseValue | None:
    split_key = f"action.{side}_arm.pose"
    pose = _vector_or_none(row.get(split_key), length=POSE_DIM)
    if pose is None:
        action = _vector_or_none(row.get("action"), length=ACTION_DIM)
        if action is not None:
            pose = action[ACTION_ARM_POSE_SLICES[side]]
    return _pose_value(pose)


def _state_pose(row: Mapping[str, Any], side: str) -> PoseValue | None:
    split_key = f"observation.{side}_arm.pose"
    pose = _vector_or_none(row.get(split_key), length=POSE_DIM)
    if pose is None:
        state = _vector_or_none(row.get("observation.state"), length=STATE_DIM)
        if state is not None:
            pose = state[STATE_ARM_POSE_SLICES[side]]
    return _pose_value(pose)


def _pose_error(a: PoseValue | None, b: PoseValue | None) -> dict[str, float] | None:
    if a is None or b is None:
        return None
    av = np.asarray(a.xyz, dtype=float)
    bv = np.asarray(b.xyz, dtype=float)
    translation_m = float(np.linalg.norm(av - bv))
    qa = _rotation_matrix_to_quat_xyzw(np.asarray(a.rotation, dtype=float).reshape(3, 3))
    qb = _rotation_matrix_to_quat_xyzw(np.asarray(b.rotation, dtype=float).reshape(3, 3))
    if qa is None or qb is None:
        return None
    dot = float(np.dot(qa, qb))
    dot = min(1.0, max(-1.0, abs(dot)))
    rotation_deg = float(math.degrees(2.0 * math.acos(dot)))
    return {
        "translation_m": translation_m,
        "rotation_deg": rotation_deg,
    }


def _rotation_matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray | None:
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(m)):
        return None
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
            if s <= 1e-12:
                return None
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif axis == 1:
            s = math.sqrt(max(0.0, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
            if s <= 1e-12:
                return None
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(max(0.0, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
            if s <= 1e-12:
                return None
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=float)
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 1e-12 else None


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _pose_to_json(value: PoseValue | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "xyz": [float(v) for v in value.xyz],
        "rotation": [float(v) for v in value.rotation],
    }


def _downsample_indices(count: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or count <= max_frames:
        return list(range(count))
    return sorted(set(int(i) for i in np.linspace(0, count - 1, num=max_frames, dtype=int).tolist()))


def _build_payload(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: EpisodeSource,
    max_frames: int,
    residual_stride: int,
    axis_length: float,
    title: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Episode parquet contains no rows: {source.parquet_path}")
    all_side_values: dict[str, dict[str, list[PoseValue | None]]] = {
        side: {"action": [], "state": [], "next_state": []}
        for side in SIDES
    }
    all_side_errors: dict[str, dict[str, list[dict[str, float] | None]]] = {
        side: {"same_state": [], "next_state": []}
        for side in SIDES
    }
    frame_indices: list[int | None] = []
    timestamps: list[float | None] = []
    for index, row in enumerate(rows):
        next_row = rows[index + 1] if index + 1 < len(rows) else None
        frame_indices.append(_int_or_none(row.get("frame_index", row.get("index"))))
        timestamps.append(_float_or_none(row.get("timestamp")))
        for side in SIDES:
            action = _action_pose(row, side)
            state = _state_pose(row, side)
            next_state = _state_pose(next_row, side) if next_row is not None else None
            all_side_values[side]["action"].append(action)
            all_side_values[side]["state"].append(state)
            all_side_values[side]["next_state"].append(next_state)
            all_side_errors[side]["same_state"].append(_pose_error(action, state))
            all_side_errors[side]["next_state"].append(_pose_error(action, next_state))

    indices = _downsample_indices(len(rows), max_frames)
    bounds_points: list[tuple[float, float, float]] = []
    sides_payload: dict[str, Any] = {}
    stats_payload: dict[str, Any] = {}
    for side in SIDES:
        side_values = all_side_values[side]
        side_errors = all_side_errors[side]
        stats_payload[side] = {
            relation: _relation_stats(errors)
            for relation, errors in side_errors.items()
        }
        display = {
            name: [_pose_to_json(values[i]) for i in indices]
            for name, values in side_values.items()
        }
        errors_display = {
            name: [errors[i] for i in indices]
            for name, errors in side_errors.items()
        }
        for values in display.values():
            for item in values:
                if item is not None:
                    bounds_points.append(tuple(float(v) for v in item["xyz"]))
        sides_payload[side] = {
            "styles": SIDE_STYLES[side],
            "traces": display,
            "errors": errors_display,
            "stats": stats_payload[side],
        }

    if not bounds_points:
        raise RuntimeError("No finite action/state pose points were found in the selected episode.")
    bounds = _bounds(bounds_points)
    axis = float(axis_length) if axis_length > 0.0 else _auto_axis_length(bounds)
    return {
        "schema": "dexteleop_pose_action_alignment_vis_v1",
        "title": title,
        "dataset_root": str(source.dataset_root),
        "parquet_path": str(source.parquet_path),
        "episode_index": source.episode_index,
        "episode_id": source.episode_id,
        "row_count": int(len(rows)),
        "display_count": int(len(indices)),
        "display_source_indices": indices,
        "frame_indices": [frame_indices[i] for i in indices],
        "timestamps": [timestamps[i] for i in indices],
        "residual_stride": int(residual_stride),
        "axis_length": axis,
        "bounds": bounds,
        "sides": sides_payload,
        "stats": stats_payload,
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _relation_stats(errors: Sequence[Mapping[str, float] | None]) -> dict[str, Any]:
    translation = [
        float(item["translation_m"])
        for item in errors
        if item is not None and math.isfinite(float(item["translation_m"]))
    ]
    rotation = [
        float(item["rotation_deg"])
        for item in errors
        if item is not None and math.isfinite(float(item["rotation_deg"]))
    ]
    return {
        "translation_error_m": _numeric_summary(translation),
        "rotation_error_deg": _numeric_summary(rotation),
    }


def _bounds(points: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    arr = np.asarray(points, dtype=float)
    return {
        "min": [float(v) for v in np.min(arr, axis=0).tolist()],
        "max": [float(v) for v in np.max(arr, axis=0).tolist()],
    }


def _auto_axis_length(bounds: Mapping[str, Sequence[float]]) -> float:
    bmin = np.asarray(bounds["min"], dtype=float)
    bmax = np.asarray(bounds["max"], dtype=float)
    diag = float(np.linalg.norm(bmax - bmin))
    return max(0.025, min(0.10, diag * 0.08 if diag > 1e-9 else 0.05))


def _resolve_output_path(source: EpisodeSource, output: str) -> Path:
    if output:
        out = Path(output).expanduser()
        if out.suffix.lower() != ".html":
            out = out / f"pose_action_alignment_{source.episode_id}.html"
        return out
    return source.dataset_root / "visualizations" / f"pose_action_alignment_{source.episode_id}.html"


def _open_uri(uri: str) -> None:
    try:
        webbrowser.open(uri, new=2)
    except Exception as exc:
        print(f"WARNING: failed to open browser: {type(exc).__name__}: {exc}")


def _format_stat(stats: Mapping[str, Any]) -> str:
    t = stats.get("translation_error_m", {}).get("mean")
    r = stats.get("rotation_error_deg", {}).get("mean")
    if t is None or r is None:
        return "n/a"
    return f"{float(t) * 1000.0:.2f} mm / {float(r):.2f} deg"


def _html_document(payload: Mapping[str, Any]) -> str:
    title = html.escape(str(payload.get("title", "Pose/action alignment")))
    json_payload = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    color-scheme: light;
    --bg: #f8fafc;
    --panel: #ffffff;
    --text: #111827;
    --muted: #64748b;
    --line: #d1d5db;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0;
    height: 100%;
    overflow: hidden;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #app {{
    height: 100%;
    display: grid;
    grid-template-rows: auto 1fr auto;
  }}
  header {{
    display: grid;
    grid-template-columns: minmax(240px, 1fr) auto;
    gap: 12px;
    align-items: center;
    padding: 10px 14px;
    border-bottom: 1px solid var(--line);
    background: var(--panel);
  }}
  h1 {{
    margin: 0;
    font-size: 16px;
    font-weight: 750;
  }}
  .meta {{
    margin-top: 3px;
    color: var(--muted);
    font-size: 12px;
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
    gap: 5px;
    align-items: center;
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
  input[type="range"] {{ width: min(36vw, 520px); }}
  main {{
    min-height: 0;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--line);
  }}
  .panel {{
    position: relative;
    min-width: 0;
    min-height: 0;
    background: #ffffff;
  }}
  canvas {{
    display: block;
    width: 100%;
    height: 100%;
    background: linear-gradient(#ffffff, #eef2f7);
    cursor: grab;
  }}
  canvas.dragging {{ cursor: grabbing; }}
  .panel-title {{
    position: absolute;
    left: 12px;
    top: 10px;
    padding: 7px 9px;
    border: 1px solid rgba(209,213,219,0.9);
    border-radius: 8px;
    background: rgba(255,255,255,0.86);
    font-size: 12px;
    line-height: 1.45;
    max-width: min(520px, calc(100% - 24px));
  }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px 12px;
    align-items: center;
    font-size: 12px;
    line-height: 1.55;
  }}
  .legend-row {{
    display: flex;
    align-items: center;
    gap: 7px;
  }}
  .swatch {{
    width: 23px;
    height: 3px;
    border-radius: 999px;
    display: inline-block;
  }}
  footer {{
    display: grid;
    grid-template-columns: minmax(260px, 1fr) auto;
    gap: 12px;
    align-items: center;
    padding: 7px 12px;
    background: var(--panel);
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 12px;
  }}
  #current-errors {{
    color: #111827;
    font-variant-numeric: tabular-nums;
  }}
  @media (max-width: 860px) {{
    header {{ grid-template-columns: 1fr; }}
    main {{ grid-template-columns: 1fr; grid-template-rows: 1fr 1fr; }}
    input[type="range"] {{ width: min(72vw, 520px); }}
  }}
</style>
</head>
<body>
<div id="app">
  <header>
    <div>
      <h1>{title}</h1>
      <div class="meta" id="meta"></div>
    </div>
    <div class="controls">
      <label><input id="toggle-left" type="checkbox" checked>left</label>
      <label><input id="toggle-right" type="checkbox" checked>right</label>
      <label><input id="toggle-residuals" type="checkbox" checked>residuals</label>
      <label><input id="toggle-orientation" type="checkbox" checked>orientation</label>
      <button id="reset">Reset view</button>
      <button id="play">Play</button>
      <input id="frame" type="range" min="0" max="0" value="0">
      <span id="frame-label"></span>
    </div>
  </header>
  <main>
    <section class="panel">
      <canvas id="same"></canvas>
      <div class="panel-title" id="same-title"></div>
    </section>
    <section class="panel">
      <canvas id="next"></canvas>
      <div class="panel-title" id="next-title"></div>
    </section>
  </main>
  <footer>
    <div>
      <div class="legend" id="legend"></div>
      <div>Drag either panel to rotate both. Mouse wheel zooms; Shift+drag pans. Triads show current local TCP axes.</div>
    </div>
    <div id="current-errors"></div>
  </footer>
</div>
<script id="payload" type="application/json">{json_payload}</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const panels = {{
  same_state: {{ canvas: document.getElementById('same'), relation: 'same_state', target: 'state', label: 'state_t + action_t' }},
  next_state: {{ canvas: document.getElementById('next'), relation: 'next_state', target: 'next_state', label: 'next_state_{{t+1}} + action_t' }},
}};
for (const panel of Object.values(panels)) panel.ctx = panel.canvas.getContext('2d');
const toggles = {{
  left: document.getElementById('toggle-left'),
  right: document.getElementById('toggle-right'),
  residuals: document.getElementById('toggle-residuals'),
  orientation: document.getElementById('toggle-orientation'),
}};
const frameSlider = document.getElementById('frame');
const frameLabel = document.getElementById('frame-label');
const playButton = document.getElementById('play');
const view = {{
  yaw: -0.72,
  pitch: 0.60,
  zoom: 1.0,
  panX: 0,
  panY: 0,
  dragging: false,
  dragCanvas: null,
  lastX: 0,
  lastY: 0,
  shift: false,
  playing: false,
  timer: null,
}};
const bmin = DATA.bounds.min;
const bmax = DATA.bounds.max;
const center = [(bmin[0]+bmax[0])/2, (bmin[1]+bmax[1])/2, (bmin[2]+bmax[2])/2];
const extent = Math.max(0.05, Math.hypot(bmax[0]-bmin[0], bmax[1]-bmin[1], bmax[2]-bmin[2]));
const frameCount = DATA.display_count || 0;
frameSlider.max = String(Math.max(0, frameCount - 1));

function vecSub(a, b) {{ return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }}
function vecAdd(a, b) {{ return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }}
function vecScale(a, s) {{ return [a[0]*s, a[1]*s, a[2]*s]; }}
function posePoint(pose) {{ return pose ? pose.xyz : null; }}
function fmt(value, digits=2) {{ return Number.isFinite(value) ? value.toFixed(digits) : 'n/a'; }}

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  for (const panel of Object.values(panels)) {{
    const rect = panel.canvas.getBoundingClientRect();
    panel.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    panel.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    panel.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    panel.baseScale = Math.min(rect.width, rect.height) * 0.70 / extent;
  }}
  draw();
}}

function project(panel, point) {{
  const p = vecSub(point, center);
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
  const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  const x1 = cy * p[0] - sy * p[1];
  const y1 = sy * p[0] + cy * p[1];
  const z1 = p[2];
  const y2 = cp * y1 - sp * z1;
  const z2 = sp * y1 + cp * z1;
  const rect = panel.canvas.getBoundingClientRect();
  return {{
    x: rect.width / 2 + view.panX + x1 * panel.baseScale * view.zoom,
    y: rect.height / 2 + view.panY - y2 * panel.baseScale * view.zoom,
    z: z2,
  }};
}}

function line3(panel, a, b, color, width=1, alpha=1, dash=null) {{
  if (!a || !b) return;
  const pa = project(panel, a), pb = project(panel, b);
  const ctx = panel.ctx;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.stroke();
  ctx.restore();
}}

function point3(panel, point, color, radius, fill=true) {{
  if (!point) return;
  const p = project(panel, point);
  const ctx = panel.ctx;
  ctx.save();
  ctx.fillStyle = fill ? color : '#ffffff';
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}}

function drawGrid(panel) {{
  const size = Math.max(extent * 0.65, 0.2);
  const step = size / 5;
  for (let i = -5; i <= 5; i++) {{
    line3(panel, [center[0] + i * step, center[1] - size, 0], [center[0] + i * step, center[1] + size, 0], '#e5e7eb', 1, 0.75);
    line3(panel, [center[0] - size, center[1] + i * step, 0], [center[0] + size, center[1] + i * step, 0], '#e5e7eb', 1, 0.75);
  }}
  const axis = Math.max(0.12, extent * 0.28);
  line3(panel, [0,0,0], [axis,0,0], '#ef4444', 3, 1);
  line3(panel, [0,0,0], [0,axis,0], '#22c55e', 3, 1);
  line3(panel, [0,0,0], [0,0,axis], '#3b82f6', 3, 1);
}}

function drawTrace(panel, poses, color, frameLimit, dashed=false) {{
  const dash = dashed ? [7, 5] : null;
  let previous = null;
  for (let i = 0; i <= frameLimit && i < poses.length; i++) {{
    const current = posePoint(poses[i]);
    if (previous && current) line3(panel, previous, current, color, dashed ? 1.9 : 2.35, dashed ? 0.75 : 0.90, dash);
    previous = current;
  }}
  previous = null;
  for (let i = frameLimit + 1; i < poses.length; i++) {{
    const current = posePoint(poses[i]);
    if (previous && current) line3(panel, previous, current, color, 1.2, 0.18, dash);
    previous = current;
  }}
}}

function drawResiduals(panel, actionPoses, targetPoses, frameLimit, color) {{
  if (!toggles.residuals.checked) return;
  const stride = Math.max(0, Number(DATA.residual_stride || 0));
  if (stride > 0) {{
    for (let i = 0; i <= frameLimit && i < actionPoses.length; i += stride) {{
      line3(panel, posePoint(actionPoses[i]), posePoint(targetPoses[i]), color, 1, 0.22);
    }}
  }}
  line3(panel, posePoint(actionPoses[frameLimit]), posePoint(targetPoses[frameLimit]), color, 3, 0.85);
}}

function drawTriad(panel, pose, alpha) {{
  if (!toggles.orientation.checked || !pose || !pose.rotation || !pose.xyz) return;
  const r = pose.rotation;
  if (r.length < 9) return;
  const origin = pose.xyz;
  const axis = DATA.axis_length || Math.max(0.025, extent * 0.08);
  const x = [origin[0] + r[0] * axis, origin[1] + r[3] * axis, origin[2] + r[6] * axis];
  const y = [origin[0] + r[1] * axis, origin[1] + r[4] * axis, origin[2] + r[7] * axis];
  const z = [origin[0] + r[2] * axis, origin[1] + r[5] * axis, origin[2] + r[8] * axis];
  line3(panel, origin, x, '#ef4444', 2, alpha);
  line3(panel, origin, y, '#22c55e', 2, alpha);
  line3(panel, origin, z, '#3b82f6', 2, alpha);
}}

function drawSide(panel, side, frameLimit) {{
  if (!toggles[side].checked) return;
  const sideData = DATA.sides[side];
  const styles = sideData.styles;
  const action = sideData.traces.action;
  const target = sideData.traces[panel.target];
  const targetColor = panel.target === 'state' ? styles.state : styles.next;
  drawTrace(panel, target, targetColor, frameLimit, false);
  drawTrace(panel, action, styles.action, frameLimit, true);
  drawResiduals(panel, action, target, frameLimit, targetColor);
  const currentAction = action[Math.min(frameLimit, action.length - 1)];
  const currentTarget = target[Math.min(frameLimit, target.length - 1)];
  point3(panel, posePoint(currentTarget), targetColor, 6, true);
  point3(panel, posePoint(currentAction), styles.action, 7, false);
  drawTriad(panel, currentTarget, 0.75);
  drawTriad(panel, currentAction, 0.95);
}}

function drawPanel(panel) {{
  const rect = panel.canvas.getBoundingClientRect();
  panel.ctx.clearRect(0, 0, rect.width, rect.height);
  drawGrid(panel);
  const frameLimit = Number(frameSlider.value);
  drawSide(panel, 'left', frameLimit);
  drawSide(panel, 'right', frameLimit);
}}

function updateText() {{
  const displayIndex = Number(frameSlider.value);
  const sourceIndex = DATA.display_source_indices[displayIndex] ?? displayIndex;
  const frameIndex = DATA.frame_indices[displayIndex];
  const timestamp = DATA.timestamps[displayIndex];
  frameLabel.textContent = `${{displayIndex + 1}} / ${{Math.max(1, frameCount)}}`;
  document.getElementById('meta').textContent =
    `${{DATA.episode_id}} | rows=${{DATA.row_count}} | displayed=${{DATA.display_count}} | parquet=${{DATA.parquet_path}}`;
  document.getElementById('same-title').innerHTML = `<b>same-frame</b><br>action_t vs state_t<br>frame=${{frameIndex ?? sourceIndex}}, t=${{timestamp == null ? 'n/a' : timestamp.toFixed(3)}}`;
  document.getElementById('next-title').innerHTML = `<b>next-frame</b><br>action_t vs state_{{t+1}}<br>frame=${{frameIndex ?? sourceIndex}}, t=${{timestamp == null ? 'n/a' : timestamp.toFixed(3)}}`;
  const parts = [];
  for (const side of ['left', 'right']) {{
    const same = DATA.sides[side].errors.same_state[displayIndex];
    const next = DATA.sides[side].errors.next_state[displayIndex];
    const sText = same ? `${{fmt(same.translation_m * 1000)}}mm/${{fmt(same.rotation_deg)}}deg` : 'n/a';
    const nText = next ? `${{fmt(next.translation_m * 1000)}}mm/${{fmt(next.rotation_deg)}}deg` : 'n/a';
    parts.push(`${{side}} same=${{sText}} next=${{nText}}`);
  }}
  document.getElementById('current-errors').textContent = parts.join(' | ');
}}

function drawLegend() {{
  const rows = [];
  for (const side of ['left', 'right']) {{
    const styles = DATA.sides[side].styles;
    rows.push(`<div class="legend-row"><span class="swatch" style="background:${{styles.action}}"></span>${{side}} action_t (dashed/ring)</div>`);
    rows.push(`<div class="legend-row"><span class="swatch" style="background:${{styles.state}}"></span>${{side}} state_t</div>`);
    rows.push(`<div class="legend-row"><span class="swatch" style="background:${{styles.next}}"></span>${{side}} next_state</div>`);
  }}
  rows.push('<div style="margin-top:6px;color:#64748b">Base axes: X red, Y green, Z blue. Residual lines connect action to state/next_state.</div>');
  document.getElementById('legend').innerHTML = rows.join('');
}}

function draw() {{
  for (const panel of Object.values(panels)) drawPanel(panel);
  updateText();
}}

function stopPlay() {{
  view.playing = false;
  playButton.textContent = 'Play';
  if (view.timer !== null) {{
    clearInterval(view.timer);
    view.timer = null;
  }}
}}

for (const panel of Object.values(panels)) {{
  panel.canvas.addEventListener('mousedown', (event) => {{
    view.dragging = true;
    view.dragCanvas = panel.canvas;
    view.shift = event.shiftKey;
    view.lastX = event.clientX;
    view.lastY = event.clientY;
    panel.canvas.classList.add('dragging');
  }});
  panel.canvas.addEventListener('wheel', (event) => {{
    event.preventDefault();
    const factor = Math.exp(-event.deltaY * 0.001);
    view.zoom = Math.max(0.08, Math.min(80, view.zoom * factor));
    draw();
  }}, {{ passive: false }});
}}
window.addEventListener('mouseup', () => {{
  view.dragging = false;
  for (const panel of Object.values(panels)) panel.canvas.classList.remove('dragging');
}});
window.addEventListener('mousemove', (event) => {{
  if (!view.dragging) return;
  const dx = event.clientX - view.lastX;
  const dy = event.clientY - view.lastY;
  view.lastX = event.clientX;
  view.lastY = event.clientY;
  if (view.shift) {{
    view.panX += dx;
    view.panY += dy;
  }} else {{
    view.yaw += dx * 0.008;
    view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + dy * 0.008));
  }}
  draw();
}});
document.getElementById('reset').addEventListener('click', () => {{
  view.yaw = -0.72;
  view.pitch = 0.60;
  view.zoom = 1.0;
  view.panX = 0;
  view.panY = 0;
  draw();
}});
playButton.addEventListener('click', () => {{
  if (view.playing) {{
    stopPlay();
    return;
  }}
  view.playing = true;
  playButton.textContent = 'Pause';
  view.timer = setInterval(() => {{
    let value = Number(frameSlider.value);
    value = value >= Number(frameSlider.max) ? 0 : value + 1;
    frameSlider.value = String(value);
    draw();
  }}, 55);
}});
for (const input of [toggles.left, toggles.right, toggles.residuals, toggles.orientation, frameSlider]) {{
  input.addEventListener('input', draw);
}}
window.addEventListener('resize', resize);
drawLegend();
resize();
</script>
</body>
</html>
"""


def main() -> int:
    args = _parse_args()
    source = _resolve_episode_source(args.dataset_or_parquet, str(args.episode))
    rows = _read_episode_rows(source.parquet_path)
    title = args.title or f"Pose/action alignment - {source.episode_id}"
    payload = _build_payload(
        rows,
        source=source,
        max_frames=int(args.max_frames),
        residual_stride=int(args.residual_stride),
        axis_length=float(args.axis_length),
        title=title,
    )
    output_path = _resolve_output_path(source, str(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html_document(payload), encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Episode: {source.episode_id} rows={len(rows)} displayed={payload['display_count']}")
    for side in SIDES:
        same = payload["stats"][side]["same_state"]
        next_ = payload["stats"][side]["next_state"]
        print(
            f"{side}: same mean={_format_stat(same)}; "
            f"next mean={_format_stat(next_)}"
        )
    if not bool(args.no_open):
        _open_uri(output_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
