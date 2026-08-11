from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .device import DMTacGripperDevice, TactileConfig
from .force_visualizer import (
    DistributedForceXYVisualizer,
    ForceVizConfig,
    _HELD_DEVICES,
    require_cv2,
)
from .tactile_filter import TactileDenoiseConfig
from .usb_assignment import SIDES, TACTILE_SLOTS, iso_timestamp, repo_root, resolve_tactile_token, timestamp_for_filename


TACTILE_SLOT_KEYMAP: dict[int, tuple[str, str]] = {
    ord("a"): ("left", "inner"),
    ord("s"): ("left", "outer"),
    ord("k"): ("right", "inner"),
    ord("l"): ("right", "outer"),
}
TACTILE_SLOT_LABELS: dict[tuple[str, str], str] = {
    ("left", "inner"): "left inner",
    ("left", "outer"): "left outer",
    ("right", "inner"): "right inner",
    ("right", "outer"): "right outer",
}
LEGACY_TACTILE_DEBUG_LAYOUT = "legacy"
CANONICAL_TACTILE_DEBUG_LAYOUT = "runtime-artifacts-v1"


@dataclass(frozen=True)
class TactileDebugRootResolution:
    path: Path
    source: str
    layout_name: str


@dataclass
class TactileAssignmentGuiConfig:
    backend: str = "cpu"
    mode: str = "standard"
    max_fps: int = 120
    compute_profile: str = "dist-only"
    timeout_ms: int = 100
    wait_for_new: bool = False
    parallel_read: bool = True
    max_hz: float = 30.0
    max_devices: int = 4
    force_decimation: int = 2
    render_mode: str = "normal-grid"
    denoise_distributed_force: bool = False
    normal_grid_rows: int = 20
    normal_grid_cols: int = 30
    normal_grid_vector_stride: int = 2
    normal_sign: float = 1.0
    normal_pressure_deadband: float = 5e-6
    normal_pressure_ref: float = 5e-5
    normal_vector_deadband: float = 2e-5
    normal_vector_ref: float = 5e-5
    record_readings: bool = True
    record_dir: str | Path | None = None
    artifact_layout: str = LEGACY_TACTILE_DEBUG_LAYOUT
    record_every_n: int = 1
    record_grid: bool = False
    window: str = "DM-Tac tactile assignment"
    reset_tactile: bool = False
    warmup_frames: int = 0
    close_devices: bool = False
    viz_config: ForceVizConfig = field(
        default_factory=lambda: ForceVizConfig(
            stride=16,
            scale=1,
            arrow_scale=14.0,
            arrow_pool=16,
            mode="delta",
            baseline_frames=8,
            fixed_ref=None,
        )
    )


@dataclass
class TactileAssignmentGuiResult:
    slots: dict[str, dict[str, str | None]]
    cancelled: bool = False
    record_dir: str | None = None


def default_tactile_debug_record_root() -> Path:
    try:
        import sys

        root = repo_root()
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from project_paths import teleop_records_root

        return teleop_records_root() / "dmtac_tactile_debug"
    except Exception:
        return repo_root() / "teleop_records" / "dmtac_tactile_debug"


def canonical_tactile_debug_record_root(*, now: float | None = None) -> Path:
    import sys

    root = repo_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from runtime_artifact_paths import RuntimeArtifactLayout

    timestamp = time.time() if now is None else float(now)
    date_partition = time.strftime("%Y-%m-%d", time.localtime(timestamp))
    return RuntimeArtifactLayout.configured().run_partition_dir(
        kind="dmtac_tactile_gui",
        date_partition=date_partition,
    )


def resolve_tactile_debug_record_root(
    record_dir: str | Path | None,
    *,
    artifact_layout: str = LEGACY_TACTILE_DEBUG_LAYOUT,
    now: float | None = None,
) -> TactileDebugRootResolution:
    # Keep the portable package's established explicit and legacy paths usable
    # without importing repository-level helpers.  Only the opt-in canonical
    # layout requires the surrounding DexTeleop checkout.
    if record_dir not in (None, ""):
        return TactileDebugRootResolution(
            path=Path(record_dir).expanduser(),
            source="explicit",
            layout_name="explicit",
        )

    selected_layout = str(
        artifact_layout or LEGACY_TACTILE_DEBUG_LAYOUT
    ).strip()
    if selected_layout == LEGACY_TACTILE_DEBUG_LAYOUT:
        return TactileDebugRootResolution(
            path=default_tactile_debug_record_root().expanduser(),
            source="legacy",
            layout_name=LEGACY_TACTILE_DEBUG_LAYOUT,
        )
    if selected_layout == CANONICAL_TACTILE_DEBUG_LAYOUT:
        return TactileDebugRootResolution(
            path=canonical_tactile_debug_record_root(now=now).expanduser(),
            source="canonical",
            layout_name=CANONICAL_TACTILE_DEBUG_LAYOUT,
        )
    raise ValueError(f"unsupported artifact layout: {selected_layout!r}")


class TactileGuiDebugRecorder:
    def __init__(
        self,
        record_dir: str | Path | None,
        *,
        metadata: Mapping[str, Any],
        artifact_layout: str = LEGACY_TACTILE_DEBUG_LAYOUT,
    ) -> None:
        resolution = resolve_tactile_debug_record_root(
            record_dir,
            artifact_layout=artifact_layout,
        )
        root = resolution.path.expanduser()
        if root.suffix:
            self.record_dir = root.parent / root.stem
        else:
            self.record_dir = root / f"{timestamp_for_filename()}_tactile_gui"
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.record_dir / "metadata.json"
        self.frames_path = self.record_dir / "frames.jsonl"
        self._fh = self.frames_path.open("a", encoding="utf-8", buffering=1)
        self.write_metadata(
            {
                **dict(metadata),
                "artifact_storage": {
                    "requested_layout": str(artifact_layout),
                    "selection_source": resolution.source,
                    "resolved_layout": resolution.layout_name,
                    "resolved_family_root": str(root.resolve()),
                    "semantics": "dmtac_tactile_assignment_gui_debug_run",
                },
            }
        )

    def write_metadata(self, metadata: Mapping[str, Any]) -> None:
        payload = {
            "schema_version": "dmtac_tactile_gui_debug.v1",
            "created_at": iso_timestamp(),
            **dict(metadata),
        }
        self.metadata_path.write_text(
            json.dumps(_jsonable(payload), indent=2, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )

    def write_frame(self, payload: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(_jsonable(dict(payload)), ensure_ascii=True, sort_keys=True) + "\n")

    def close(self) -> None:
        self._fh.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def empty_tactile_slot_map() -> dict[str, dict[str, str | None]]:
    return {side: {slot: None for slot in TACTILE_SLOTS} for side in SIDES}


def assigned_slot_for_device(
    slots: Mapping[str, Mapping[str, str | None]],
    dev_id: str,
) -> tuple[str, str] | None:
    target = str(dev_id)
    for side in SIDES:
        for slot in TACTILE_SLOTS:
            if str((slots.get(side) or {}).get(slot) or "") == target:
                return side, slot
    return None


def assign_device_to_tactile_slot(
    slots: dict[str, dict[str, str | None]],
    dev_id: str,
    *,
    side: str,
    slot: str,
) -> dict[str, dict[str, str | None]]:
    if side not in SIDES:
        raise ValueError(f"Unsupported side {side!r}; expected left/right.")
    if slot not in TACTILE_SLOTS:
        raise ValueError(f"Unsupported tactile slot {slot!r}; expected inner/outer.")
    normalized = str(dev_id)
    for side_name in SIDES:
        for slot_name in TACTILE_SLOTS:
            if slots[side_name][slot_name] == normalized:
                slots[side_name][slot_name] = None
    slots[side][slot] = normalized
    return slots


def clear_tactile_device_assignment(
    slots: dict[str, dict[str, str | None]],
    dev_id: str,
) -> dict[str, dict[str, str | None]]:
    normalized = str(dev_id)
    for side in SIDES:
        for slot in TACTILE_SLOTS:
            if slots[side][slot] == normalized:
                slots[side][slot] = None
    return slots


def tactile_slot_tokens_from_assignment(
    slots: Mapping[str, Mapping[str, str | None]],
) -> dict[str, str | None]:
    return {
        f"{side}_{slot}": (
            None
            if (slots.get(side) or {}).get(slot) in (None, "")
            else str((slots.get(side) or {}).get(slot))
        )
        for side in SIDES
        for slot in TACTILE_SLOTS
    }


def tactile_items_for_gui(
    scan: Mapping[str, Any],
    *,
    tokens: Sequence[str | int] | None = None,
    max_devices: int = 4,
) -> list[dict[str, Any]]:
    if int(max_devices) <= 0:
        raise ValueError("max_devices must be positive.")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    if tokens:
        candidates = [resolve_tactile_token(scan, token) for token in tokens]
    else:
        candidates = [dict(item) for item in scan.get("tactile") or []]
    for item in candidates:
        if item is None:
            continue
        dev_id = str(item.get("dev_id") or item.get("serial") or item.get("index"))
        if not dev_id or dev_id in seen:
            continue
        item = dict(item)
        item["dev_id"] = dev_id
        selected.append(item)
        seen.add(dev_id)
        if len(selected) >= int(max_devices):
            break
    return selected


def decimate_force_for_gui(distributed_force: Any, decimation: int = 1) -> np.ndarray:
    force = np.asarray(distributed_force, dtype=np.float32)
    if force.ndim != 3 or force.shape[2] < 2:
        raise ValueError("distributed_force must have shape (H, W, C) with at least two channels")
    step = max(1, int(decimation))
    if step == 1:
        return force
    return force[::step, ::step, :]


def coarse_force_grid(
    distributed_force: Any,
    *,
    rows: int = 20,
    cols: int = 30,
) -> np.ndarray:
    force = np.asarray(distributed_force, dtype=np.float32)
    if force.ndim != 3 or force.shape[2] < 3:
        raise ValueError("distributed_force must have shape (H, W, C) with at least three channels")
    row_count = max(1, int(rows))
    col_count = max(1, int(cols))
    if force.shape[0] < row_count or force.shape[1] < col_count:
        raise ValueError(
            f"distributed_force shape {force.shape[:2]} is smaller than grid {row_count}x{col_count}"
        )
    try:
        cv = require_cv2()
    except RuntimeError:
        cv = None
    if cv is not None:
        source = np.ascontiguousarray(
            np.nan_to_num(force[..., :3], nan=0.0, posinf=0.0, neginf=0.0)
        )
        resized = cv.resize(
            source,
            (col_count, row_count),
            interpolation=cv.INTER_AREA,
        )
        return np.nan_to_num(np.asarray(resized, dtype=np.float32), copy=False)

    row_edges = np.linspace(0, force.shape[0], row_count + 1, dtype=np.int32)
    col_edges = np.linspace(0, force.shape[1], col_count + 1, dtype=np.int32)
    grid = np.zeros((row_count, col_count, 3), dtype=np.float32)
    for row in range(row_count):
        row_start, row_end = int(row_edges[row]), int(row_edges[row + 1])
        for col in range(col_count):
            col_start, col_end = int(col_edges[col]), int(col_edges[col + 1])
            patch = force[row_start:row_end, col_start:col_end, :3]
            grid[row, col, :] = np.nanmean(patch, axis=(0, 1))
    return np.nan_to_num(grid, copy=False)


def positive_normal_pressure_grid(force_grid: Any, *, normal_sign: float = 1.0) -> np.ndarray:
    grid = np.asarray(force_grid, dtype=np.float32)
    if grid.ndim != 3 or grid.shape[2] < 3:
        raise ValueError("force_grid must have shape (H, W, C) with at least three channels")
    return np.maximum(float(normal_sign) * grid[..., 1], 0.0)


def xz_plane_vector_grid(force_grid: Any) -> np.ndarray:
    grid = np.asarray(force_grid, dtype=np.float32)
    if grid.ndim != 3 or grid.shape[2] < 3:
        raise ValueError("force_grid must have shape (H, W, C) with at least three channels")
    return grid[..., (0, 2)]


def normal_grid_display_components(
    force_grid: Any,
    *,
    normal_sign: float = 1.0,
    pressure_deadband: float = 0.0,
    vector_deadband: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    pressure = positive_normal_pressure_grid(force_grid, normal_sign=float(normal_sign))
    pressure_floor = max(0.0, float(pressure_deadband))
    if pressure_floor > 0.0:
        pressure = np.where(pressure >= pressure_floor, pressure, 0.0).astype(np.float32, copy=False)

    xz_vectors = xz_plane_vector_grid(force_grid)
    vector_floor = max(0.0, float(vector_deadband))
    if vector_floor > 0.0:
        vector_norm = np.linalg.norm(xz_vectors, axis=2)
        xz_vectors = np.where(vector_norm[..., None] >= vector_floor, xz_vectors, 0.0).astype(
            np.float32,
            copy=False,
        )
    return pressure, xz_vectors


def run_tactile_assignment_gui(
    scan: Mapping[str, Any],
    *,
    device_tokens: Sequence[str | int] | None = None,
    config: TactileAssignmentGuiConfig | None = None,
) -> TactileAssignmentGuiResult:
    cfg = config or TactileAssignmentGuiConfig()
    items = tactile_items_for_gui(scan, tokens=device_tokens, max_devices=int(cfg.max_devices))
    if not items:
        raise RuntimeError("No DM-Tac tactile sensors were discovered for GUI assignment.")

    cv = require_cv2()
    dev_ids = [str(item["dev_id"]) for item in items]
    tactile_config = TactileConfig(
        dev_ids=dev_ids,
        backend=str(cfg.backend),
        mode=str(cfg.mode),
        max_fps=int(cfg.max_fps),
        fields=("distributed_force",),
        compute_profile=str(cfg.compute_profile),
        parallel_read=bool(cfg.parallel_read),
        denoise_distributed_force=bool(cfg.denoise_distributed_force),
        denoise_config=TactileDenoiseConfig(
            enabled=True,
            mode=str(cfg.viz_config.mode),
            baseline_frames=int(cfg.viz_config.baseline_frames),
            deadband_abs=float(cfg.viz_config.deadband_abs),
            temporal_alpha=float(cfg.viz_config.temporal_alpha),
            spatial_blur_ksize=int(cfg.viz_config.spatial_blur_ksize),
        ),
    )
    visualizer = DistributedForceXYVisualizer.display_only(cfg.viz_config)
    slots = empty_tactile_slot_map()
    selected_index = 0
    message = "Select 1-4, assign with A/S/K/L, Enter saves."
    frame_count = 0
    fps_value = 0.0
    fps_start = time.perf_counter()
    read_ms = 0.0
    render_ms = 0.0
    display_ms = 0.0
    loop_ms = 0.0
    last_fids: dict[str, Any] = {}
    last_fid_change_s: dict[str, float] = {}
    recorder: TactileGuiDebugRecorder | None = None
    record_dir_text: str | None = None
    loop_index = 0

    dev = DMTacGripperDevice(
        tactile=tactile_config,
        connect_gripper=False,
        connect_tactile=True,
    ).connect()
    try:
        print("tactile ready:", dev.wait_tactile_ready(timeout_s=10))
        if cfg.reset_tactile:
            print("reset tactile: keep all sensor surfaces unloaded")
            dev.reset_tactile()
            print("tactile ready after reset:", dev.wait_tactile_ready(timeout_s=10))
        if int(cfg.warmup_frames) > 0:
            print(f"warmup tactile assignment frames={int(cfg.warmup_frames)}")
            dev.warmup_tactile_filter(frames=int(cfg.warmup_frames), timeout_ms=int(cfg.timeout_ms))
        if bool(cfg.record_readings):
            recorder = TactileGuiDebugRecorder(
                cfg.record_dir,
                artifact_layout=str(cfg.artifact_layout),
                metadata={
                    "dev_ids": dev_ids,
                    "selected_items": items,
                    "backend": str(cfg.backend),
                    "mode": str(cfg.mode),
                    "max_fps": int(cfg.max_fps),
                    "compute_profile": str(cfg.compute_profile),
                    "timeout_ms": int(cfg.timeout_ms),
                    "wait_for_new": bool(cfg.wait_for_new),
                    "parallel_read": bool(cfg.parallel_read),
                    "max_hz": float(cfg.max_hz),
                    "render_mode": str(cfg.render_mode),
                    "grid_rows": int(cfg.normal_grid_rows),
                    "grid_cols": int(cfg.normal_grid_cols),
                    "vector_stride": int(cfg.normal_grid_vector_stride),
                    "normal_sign": float(cfg.normal_sign),
                    "normal_pressure_deadband": float(cfg.normal_pressure_deadband),
                    "normal_pressure_ref": float(cfg.normal_pressure_ref),
                    "normal_vector_deadband": float(cfg.normal_vector_deadband),
                    "normal_vector_ref": float(cfg.normal_vector_ref),
                    "record_grid": bool(cfg.record_grid),
                    "record_every_n": max(1, int(cfg.record_every_n)),
                },
            )
            record_dir_text = str(recorder.record_dir)
            print(f"tactile GUI debug record: {record_dir_text}")

        while True:
            loop_index += 1
            loop_start = time.perf_counter()
            read_started_at = time.perf_counter()
            frames = dev.read_tactile(
                timeout_ms=int(cfg.timeout_ms),
                wait=bool(cfg.wait_for_new),
                require_ready=True,
            )
            read_ms = (time.perf_counter() - read_started_at) * 1000.0
            frames_by_id = {str(dev_id): frame for dev_id, frame in frames.items()}
            fid_age_by_id = _update_fid_age(frames_by_id, last_fids, last_fid_change_s)
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 0.5:
                fps_value = frame_count / elapsed
                frame_count = 0
                fps_start = time.perf_counter()

            render_started_at = time.perf_counter()
            image = _render_assignment_view(
                visualizer=visualizer,
                items=items,
                frames_by_id=frames_by_id,
                fid_age_by_id=fid_age_by_id,
                selected_index=selected_index,
                slots=slots,
                message=message,
                fps=fps_value,
                read_ms=read_ms,
                render_ms=render_ms,
                display_ms=display_ms,
                loop_ms=loop_ms,
                force_decimation=int(cfg.force_decimation),
                render_mode=str(cfg.render_mode),
                normal_grid_rows=int(cfg.normal_grid_rows),
                normal_grid_cols=int(cfg.normal_grid_cols),
                normal_grid_vector_stride=int(cfg.normal_grid_vector_stride),
                normal_sign=float(cfg.normal_sign),
                normal_pressure_deadband=float(cfg.normal_pressure_deadband),
                normal_pressure_ref=float(cfg.normal_pressure_ref),
                normal_vector_deadband=float(cfg.normal_vector_deadband),
                normal_vector_ref=float(cfg.normal_vector_ref),
                compute_profile=str(cfg.compute_profile),
            )
            render_ms = (time.perf_counter() - render_started_at) * 1000.0
            display_started_at = time.perf_counter()
            cv.imshow(str(cfg.window), image)
            key = cv.waitKey(1) & 0xFF
            display_ms = (time.perf_counter() - display_started_at) * 1000.0
            loop_ms = (time.perf_counter() - loop_start) * 1000.0
            if recorder is not None and loop_index % max(1, int(cfg.record_every_n)) == 0:
                recorder.write_frame(
                    _tactile_gui_record_payload(
                        loop_index=loop_index,
                        items=items,
                        frames_by_id=frames_by_id,
                        fid_age_by_id=fid_age_by_id,
                        slots=slots,
                        fps=fps_value,
                        read_ms=read_ms,
                        render_ms=render_ms,
                        display_ms=display_ms,
                        loop_ms=loop_ms,
                        force_grid_rows=int(cfg.normal_grid_rows),
                        force_grid_cols=int(cfg.normal_grid_cols),
                        normal_sign=float(cfg.normal_sign),
                        include_grid=bool(cfg.record_grid),
                    )
                )
            if key in range(ord("1"), ord("9") + 1):
                next_index = key - ord("1")
                if next_index < len(items):
                    selected_index = next_index
                    message = f"selected {selected_index + 1}: {items[selected_index]['dev_id']}"
            elif key in TACTILE_SLOT_KEYMAP:
                side, slot = TACTILE_SLOT_KEYMAP[key]
                dev_id = str(items[selected_index]["dev_id"])
                assign_device_to_tactile_slot(slots, dev_id, side=side, slot=slot)
                message = f"assigned {dev_id} -> {TACTILE_SLOT_LABELS[(side, slot)]}"
            elif key in (ord("u"), 8, 127):
                dev_id = str(items[selected_index]["dev_id"])
                clear_tactile_device_assignment(slots, dev_id)
                message = f"unassigned {dev_id}"
            elif key == ord("c"):
                slots = empty_tactile_slot_map()
                message = "cleared tactile assignments"
            elif key in (13, 10):
                return TactileAssignmentGuiResult(slots=slots, cancelled=False, record_dir=record_dir_text)
            elif key == 27:
                return TactileAssignmentGuiResult(slots=slots, cancelled=True, record_dir=record_dir_text)

            if float(cfg.max_hz) > 0:
                target_dt = 1.0 / float(cfg.max_hz)
                remain = target_dt - (time.perf_counter() - loop_start)
                if remain > 0:
                    time.sleep(remain)
    finally:
        try:
            cv.destroyWindow(str(cfg.window))
        except Exception:
            pass
        if recorder is not None:
            recorder.close()
        if cfg.close_devices:
            dev.close()
        else:
            _HELD_DEVICES.append(dev)


def _render_assignment_view(
    *,
    visualizer: DistributedForceXYVisualizer,
    items: Sequence[Mapping[str, Any]],
    frames_by_id: Mapping[str, Any],
    fid_age_by_id: Mapping[str, float],
    selected_index: int,
    slots: Mapping[str, Mapping[str, str | None]],
    message: str,
    fps: float,
    read_ms: float = 0.0,
    render_ms: float = 0.0,
    display_ms: float = 0.0,
    loop_ms: float = 0.0,
    force_decimation: int = 1,
    render_mode: str = "normal-grid",
    normal_grid_rows: int = 2,
    normal_grid_cols: int = 3,
    normal_grid_vector_stride: int = 2,
    normal_sign: float = 1.0,
    normal_pressure_deadband: float = 5e-6,
    normal_pressure_ref: float = 5e-5,
    normal_vector_deadband: float = 2e-5,
    normal_vector_ref: float = 5e-5,
    compute_profile: str = "dist-only",
) -> np.ndarray:
    cv = require_cv2()
    panels = []
    mode = str(render_mode).strip().lower().replace("_", "-")
    for index, item in enumerate(items):
        dev_id = str(item.get("dev_id"))
        frame = frames_by_id.get(dev_id)
        title = f"{index + 1}  dev {dev_id}"
        if frame is not None and getattr(frame, "fid", None) is not None:
            title += f"  fid={getattr(frame, 'fid')}"
        fid_age_s = fid_age_by_id.get(dev_id, 0.0)
        if frame is None or getattr(frame, "distributed_force", None) is None:
            if mode in {"xy", "force-xy", "distributed-force"}:
                panel = visualizer.render_blank(title=f"{title}: no force")
            else:
                panel = _render_normal_grid_panel(
                    title=f"{title}: no force",
                    force_grid=None,
                    rows=int(normal_grid_rows),
                    cols=int(normal_grid_cols),
                    vector_stride=int(normal_grid_vector_stride),
                    normal_sign=float(normal_sign),
                    pressure_deadband=float(normal_pressure_deadband),
                    pressure_ref=float(normal_pressure_ref),
                    vector_deadband=float(normal_vector_deadband),
                    vector_ref=float(normal_vector_ref),
                    fid_age_s=float(fid_age_s),
                    timing_text=_frame_timing_text(frame),
                )
        else:
            if mode in {"xy", "force-xy", "distributed-force"}:
                force = decimate_force_for_gui(
                    frame.distributed_force,
                    decimation=int(force_decimation),
                )
                panel = visualizer.render_force(force, key=dev_id, title=title)
            else:
                force_grid = coarse_force_grid(
                    frame.distributed_force,
                    rows=int(normal_grid_rows),
                    cols=int(normal_grid_cols),
                )
                panel = _render_normal_grid_panel(
                    title=title,
                    force_grid=force_grid,
                    rows=int(normal_grid_rows),
                    cols=int(normal_grid_cols),
                    vector_stride=int(normal_grid_vector_stride),
                    normal_sign=float(normal_sign),
                    pressure_deadband=float(normal_pressure_deadband),
                    pressure_ref=float(normal_pressure_ref),
                    vector_deadband=float(normal_vector_deadband),
                    vector_ref=float(normal_vector_ref),
                    fid_age_s=float(fid_age_s),
                    timing_text=_frame_timing_text(frame),
                )
        slot = assigned_slot_for_device(slots, dev_id)
        slot_text = "unassigned" if slot is None else TACTILE_SLOT_LABELS[slot]
        slot_org = (
            (8, 42)
            if mode in {"xy", "force-xy", "distributed-force"}
            else (max(8, panel.shape[1] - 150), 62)
        )
        _put_text(panel, slot_text, slot_org, 0.52)
        if index == int(selected_index):
            panel = cv.copyMakeBorder(panel, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=(0, 255, 0))
        else:
            panel = cv.copyMakeBorder(panel, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=(50, 50, 50))
        panels.append(panel)

    grid = _grid_panels(panels)
    footer_h = 82
    footer = np.zeros((footer_h, grid.shape[1], 3), dtype=np.uint8)
    slot_line = "  ".join(
        f"{label}: {((slots.get(side) or {}).get(slot) or '-')}"
        for (side, slot), label in TACTILE_SLOT_LABELS.items()
    )
    _put_text(
        footer,
        "1-4 select | A left-inner | S left-outer | K right-inner | L right-outer | U unassign | C clear | Enter save | Esc cancel",
        (8, 24),
        0.48,
    )
    _put_text(footer, slot_line, (8, 50), 0.48)
    _put_text(
        footer,
        (
            f"{message}  mode={mode} Y+=max({normal_sign:g}*Fy,0), arrows=XZ "
            f"profile={compute_profile} db={normal_pressure_deadband:g}/{normal_vector_deadband:g} "
            f"fps={fps:.1f} read={read_ms:.1f}ms render={render_ms:.1f}ms "
            f"display={display_ms:.1f}ms loop={loop_ms:.1f}ms"
        ),
        (8, 74),
        0.48,
    )
    return np.concatenate([grid, footer], axis=0)


def _update_fid_age(
    frames_by_id: Mapping[str, Any],
    last_fids: dict[str, Any],
    last_fid_change_s: dict[str, float],
) -> dict[str, float]:
    now = time.perf_counter()
    ages: dict[str, float] = {}
    for dev_id, frame in frames_by_id.items():
        fid = None if frame is None else getattr(frame, "fid", None)
        if dev_id not in last_fids or last_fids[dev_id] != fid:
            last_fids[dev_id] = fid
            last_fid_change_s[dev_id] = now
        ages[dev_id] = now - last_fid_change_s.get(dev_id, now)
    return ages


def _render_normal_grid_panel(
    *,
    title: str,
    force_grid: np.ndarray | None,
    rows: int,
    cols: int,
    vector_stride: int,
    normal_sign: float,
    pressure_deadband: float,
    pressure_ref: float,
    vector_deadband: float,
    vector_ref: float,
    fid_age_s: float,
    timing_text: str = "",
) -> np.ndarray:
    cv = require_cv2()
    row_count = max(1, int(rows))
    col_count = max(1, int(cols))
    dense = row_count * col_count > 64
    panel_w = 620 if dense else 372
    panel_h = 420 if dense else 236
    image = np.full((panel_h, panel_w, 3), 18, dtype=np.uint8)
    _put_text(image, title, (8, 22), 0.46)
    timing_suffix = f"  {timing_text}" if timing_text else ""
    _put_text(image, f"fid_age={fid_age_s * 1000.0:.0f}ms{timing_suffix}", (8, 62), 0.42)

    if force_grid is None:
        _put_text(image, "no distributed_force", (88, 132), 0.55)
        return image

    pressure, xz_vectors = normal_grid_display_components(
        force_grid,
        normal_sign=float(normal_sign),
        pressure_deadband=float(pressure_deadband),
        vector_deadband=float(vector_deadband),
    )
    positive_values = pressure[np.isfinite(pressure)]
    max_pressure = float(np.max(positive_values)) if positive_values.size else 0.0
    mean_pressure = float(np.mean(positive_values)) if positive_values.size else 0.0
    ref_pressure = max(max_pressure, max(1e-9, float(pressure_ref)))
    vector_norm = np.linalg.norm(xz_vectors, axis=2)
    vector_values = vector_norm[np.isfinite(vector_norm)]
    ref_vector = max(
        float(np.max(vector_values)) if vector_values.size else 0.0,
        max(1e-9, float(vector_ref)),
    )
    _put_text(
        image,
        f"Y+ mean={mean_pressure:.4g} max={max_pressure:.4g}  XZ max={float(np.max(vector_values)) if vector_values.size else 0.0:.4g}",
        (8, 42),
        0.42,
    )

    grid_x0 = 8
    grid_y0 = 70
    if dense:
        _render_dense_normal_grid(
            image,
            pressure=pressure,
            xz_vectors=xz_vectors,
            ref_pressure=ref_pressure,
            ref_vector=ref_vector,
            x0=grid_x0,
            y0=grid_y0,
            width=panel_w - 2 * grid_x0,
            height=panel_h - grid_y0 - 8,
            vector_stride=max(1, int(vector_stride)),
        )
        return image

    gap = 4
    cell_w = max(34, (panel_w - 2 * grid_x0 - (col_count - 1) * gap) // col_count)
    cell_h = max(48, (panel_h - grid_y0 - 8 - (row_count - 1) * gap) // row_count)
    for row in range(row_count):
        for col in range(col_count):
            x0 = grid_x0 + col * (cell_w + gap)
            y0 = grid_y0 + row * (cell_h + gap)
            x1 = min(panel_w - 8, x0 + cell_w)
            y1 = min(panel_h - 8, y0 + cell_h)
            normal_value = float(pressure[row, col])
            color = _positive_pressure_color(normal_value / ref_pressure)
            cv.rectangle(image, (x0, y0), (x1, y1), color, thickness=-1)
            cv.rectangle(image, (x0, y0), (x1, y1), (120, 120, 120), thickness=1)
            fx, fz = (float(v) for v in xz_vectors[row, col, :2])
            fy = float(force_grid[row, col, 1])
            _draw_xz_vector(
                image,
                cell=(x0, y0, x1, y1),
                fx=fx,
                fz=fz,
                ref=float(ref_vector),
            )
            _put_text(image, f"y+ {normal_value:.3g}", (x0 + 6, y0 + 20), 0.36)
            _put_text(image, f"x {fx:.2g} y {fy:.2g}", (x0 + 6, y1 - 28), 0.30)
            _put_text(image, f"z {fz:.2g}", (x0 + 6, y1 - 8), 0.32)
    return image


def _draw_xz_vector(
    image: np.ndarray,
    *,
    cell: tuple[int, int, int, int],
    fx: float,
    fz: float,
    ref: float,
    center_radius: int = 3,
    thickness: int = 2,
) -> None:
    cv = require_cv2()
    x0, y0, x1, y1 = cell
    cx = int((x0 + x1) * 0.5)
    cy = int((y0 + y1) * 0.5)
    half_span = 0.34 * float(min(max(1, x1 - x0), max(1, y1 - y0)))
    scale = half_span / max(float(ref), 1e-6)
    dx = float(fx) * scale
    dy = -float(fz) * scale
    max_len = max(1.0, half_span)
    length = math.hypot(dx, dy)
    if length > max_len:
        dx *= max_len / length
        dy *= max_len / length
    end = (int(round(cx + dx)), int(round(cy + dy)))
    cv.circle(
        image,
        (cx, cy),
        max(0, int(center_radius)),
        (245, 245, 245),
        thickness=-1,
        lineType=cv.LINE_AA,
    )
    if math.hypot(dx, dy) < 1.0:
        return
    cv.arrowedLine(
        image,
        (cx, cy),
        end,
        (255, 255, 255),
        max(1, int(thickness)),
        cv.LINE_AA,
        0,
        0.28,
    )


def _frame_timing_text(frame: Any) -> str:
    if frame is None:
        return ""
    timings = getattr(frame, "timings_ms", None) or {}
    parts = []
    if "distributed_force" in timings:
        parts.append(f"df={float(timings['distributed_force']):.1f}ms")
    if "force" in timings:
        parts.append(f"force={float(timings['force']):.1f}ms")
    if "wait_for_new" in timings:
        parts.append(f"wait={float(timings['wait_for_new']):.1f}ms")
    return " ".join(parts)


def _tactile_gui_record_payload(
    *,
    loop_index: int,
    items: Sequence[Mapping[str, Any]],
    frames_by_id: Mapping[str, Any],
    fid_age_by_id: Mapping[str, float],
    slots: Mapping[str, Mapping[str, str | None]],
    fps: float,
    read_ms: float,
    render_ms: float,
    display_ms: float,
    loop_ms: float,
    force_grid_rows: int,
    force_grid_cols: int,
    normal_sign: float,
    include_grid: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "loop_index": int(loop_index),
        "time_s": time.time(),
        "perf_counter_s": time.perf_counter(),
        "fps": float(fps),
        "read_ms": float(read_ms),
        "render_ms": float(render_ms),
        "display_ms": float(display_ms),
        "loop_ms": float(loop_ms),
        "slots": {
            side: {slot: ((slots.get(side) or {}).get(slot) or None) for slot in TACTILE_SLOTS}
            for side in SIDES
        },
        "devices": {},
    }
    devices: dict[str, Any] = {}
    for item in items:
        dev_id = str(item.get("dev_id"))
        frame = frames_by_id.get(dev_id)
        record: dict[str, Any] = {
            "fid_age_ms": float(fid_age_by_id.get(dev_id, 0.0)) * 1000.0,
            "assigned_slot": assigned_slot_for_device(slots, dev_id),
            "frame": None,
        }
        if frame is not None:
            record["frame"] = {
                "timestamp_ns": getattr(frame, "timestamp_ns", None),
                "status": getattr(frame, "status", None),
                "fid": getattr(frame, "fid", None),
                "fids": dict(getattr(frame, "fids", {}) or {}),
                "errors": dict(getattr(frame, "errors", {}) or {}),
                "timings_ms": dict(getattr(frame, "timings_ms", {}) or {}),
            }
            distributed_force = getattr(frame, "distributed_force", None)
            if distributed_force is not None:
                force = np.asarray(distributed_force, dtype=np.float32)
                record["distributed_force_shape"] = tuple(force.shape)
                grid = coarse_force_grid(
                    force,
                    rows=int(force_grid_rows),
                    cols=int(force_grid_cols),
                )
                pressure = positive_normal_pressure_grid(grid, normal_sign=float(normal_sign))
                xz = xz_plane_vector_grid(grid)
                record["grid_shape"] = tuple(grid.shape)
                record["pressure_y_positive"] = {
                    "mean": float(np.nanmean(pressure)) if pressure.size else 0.0,
                    "max": float(np.nanmax(pressure)) if pressure.size else 0.0,
                }
                xz_norm = np.linalg.norm(xz, axis=2)
                record["xz_vector"] = {
                    "mean": float(np.nanmean(xz_norm)) if xz_norm.size else 0.0,
                    "max": float(np.nanmax(xz_norm)) if xz_norm.size else 0.0,
                }
                if include_grid:
                    record["force_grid"] = np.round(grid.astype(np.float32, copy=False), 6)
                    record["pressure_y_positive_grid"] = np.round(pressure.astype(np.float32, copy=False), 6)
        devices[dev_id] = record
    payload["devices"] = devices
    return payload


def _render_dense_normal_grid(
    image: np.ndarray,
    *,
    pressure: np.ndarray,
    xz_vectors: np.ndarray,
    ref_pressure: float,
    ref_vector: float,
    x0: int,
    y0: int,
    width: int,
    height: int,
    vector_stride: int,
) -> None:
    cv = require_cv2()
    rows, cols = pressure.shape
    norm = np.clip(pressure / max(float(ref_pressure), 1e-6), 0.0, 1.0)
    heat = _positive_pressure_heatmap(norm)
    resized = cv.resize(heat, (int(width), int(height)), interpolation=cv.INTER_NEAREST)
    image[y0:y0 + height, x0:x0 + width] = resized

    cv.rectangle(image, (x0, y0), (x0 + width, y0 + height), (64, 64, 64), 1)

    stride = max(1, int(vector_stride))
    for row in range(0, rows, stride):
        for col in range(0, cols, stride):
            cell_x0 = int(round(x0 + col * width / cols))
            cell_y0 = int(round(y0 + row * height / rows))
            cell_x1 = int(round(x0 + (col + 1) * width / cols))
            cell_y1 = int(round(y0 + (row + 1) * height / rows))
            fx, fz = (float(v) for v in xz_vectors[row, col, :2])
            _draw_xz_vector_line(
                cv,
                image,
                cell=(cell_x0, cell_y0, cell_x1, cell_y1),
                fx=fx,
                fz=fz,
                ref=float(ref_vector),
            )


def _draw_xz_vector_line(
    cv,
    image: np.ndarray,
    *,
    cell: tuple[int, int, int, int],
    fx: float,
    fz: float,
    ref: float,
) -> None:
    x0, y0, x1, y1 = cell
    cx = int((x0 + x1) * 0.5)
    cy = int((y0 + y1) * 0.5)
    half_span = 0.42 * float(min(max(1, x1 - x0), max(1, y1 - y0)))
    scale = half_span / max(float(ref), 1e-6)
    dx = float(fx) * scale
    dy = -float(fz) * scale
    length = math.hypot(dx, dy)
    if length < 1.0:
        return
    if length > half_span:
        dx *= half_span / length
        dy *= half_span / length
    end = (int(round(cx + dx)), int(round(cy + dy)))
    cv.line(image, (cx, cy), end, (255, 255, 255), 1, cv.LINE_8)


def _positive_pressure_heatmap(norm: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(norm, dtype=np.float32), 0.0, 1.0)
    out = np.empty((*clipped.shape, 3), dtype=np.uint8)
    out[..., 0] = (28 + 22 * clipped).astype(np.uint8)
    out[..., 1] = (34 + 180 * clipped).astype(np.uint8)
    out[..., 2] = (42 + 210 * clipped).astype(np.uint8)
    return out


def _positive_pressure_color(value: float) -> tuple[int, int, int]:
    clipped = float(np.clip(value, 0.0, 1.0))
    return (
        int(28 + 22 * clipped),
        int(34 + 180 * clipped),
        int(42 + 210 * clipped),
    )


def _grid_panels(panels: Sequence[np.ndarray]) -> np.ndarray:
    if not panels:
        raise ValueError("at least one panel is required")
    cv = require_cv2()
    cols = 2 if len(panels) > 1 else 1
    rows = int(math.ceil(len(panels) / float(cols)))
    max_h = max(panel.shape[0] for panel in panels)
    max_w = max(panel.shape[1] for panel in panels)
    resized = [
        panel
        if panel.shape[:2] == (max_h, max_w)
        else cv.resize(panel, (max_w, max_h), interpolation=cv.INTER_LINEAR)
        for panel in panels
    ]
    blank = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    while len(resized) < rows * cols:
        resized.append(blank.copy())
    row_images = []
    for row in range(rows):
        start = row * cols
        row_images.append(np.concatenate(resized[start:start + cols], axis=1))
    return np.concatenate(row_images, axis=0)


def _put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.48) -> None:
    cv = require_cv2()
    cv.putText(img, str(text), org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, str(text), org, cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv.LINE_AA)
